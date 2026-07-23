from __future__ import annotations

from dataclasses import dataclass
from math import exp, lgamma, log
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import gammaln


REQUIRED_COLUMNS = {"date", "home_team", "away_team", "home_goals", "away_goals"}


@dataclass(frozen=True)
class Prediction:
    home_team: str
    away_team: str
    home_goal_expectation: float
    away_goal_expectation: float
    score_matrix: np.ndarray

    def result_probabilities(self) -> Dict[str, float]:
        home = float(np.tril(self.score_matrix, -1).sum())
        draw = float(np.trace(self.score_matrix))
        away = float(np.triu(self.score_matrix, 1).sum())
        return {"home_win": home, "draw": draw, "away_win": away}

    def over_under(self, line: float = 2.5) -> Dict[str, float]:
        over = 0.0
        under = 0.0
        for home_goals in range(self.score_matrix.shape[0]):
            for away_goals in range(self.score_matrix.shape[1]):
                prob = float(self.score_matrix[home_goals, away_goals])
                if home_goals + away_goals > line:
                    over += prob
                else:
                    under += prob
        return {"over": over, "under": under}

    def both_teams_to_score(self) -> Dict[str, float]:
        yes = float(self.score_matrix[1:, 1:].sum())
        return {"yes": yes, "no": 1.0 - yes}

    def top_scores(self, n: int = 10) -> List[Tuple[str, float]]:
        scores: List[Tuple[str, float]] = []
        for home_goals in range(self.score_matrix.shape[0]):
            for away_goals in range(self.score_matrix.shape[1]):
                scores.append((f"{home_goals}-{away_goals}", float(self.score_matrix[home_goals, away_goals])))
        return sorted(scores, key=lambda item: item[1], reverse=True)[:n]


class DixonColesModel:
    def __init__(
        self,
        half_life_days: float = 365.0,
        rho_bounds: Tuple[float, float] = (-0.2, 0.2),
        l2_penalty: float = 0.05,
    ):
        self.half_life_days = half_life_days
        self.rho_bounds = rho_bounds
        self.l2_penalty = l2_penalty
        self.teams_: List[str] = []
        self.attack_: Dict[str, float] = {}
        self.defense_: Dict[str, float] = {}
        self.home_advantage_: float = 0.0
        self.rho_: float = 0.0
        self.fitted_: bool = False
        self.fit_converged_: bool = False
        self.fit_message_: str = ""

    def fit(self, matches: pd.DataFrame) -> "DixonColesModel":
        matches = self._prepare_matches(matches)
        self.teams_ = sorted(set(matches["home_team"]) | set(matches["away_team"]))
        team_to_idx = {team: idx for idx, team in enumerate(self.teams_)}
        n_teams = len(self.teams_)

        initial = np.zeros(n_teams * 2 + 2)
        initial[n_teams * 2] = 0.20
        initial[n_teams * 2 + 1] = -0.05

        bounds = [(None, None)] * (n_teams * 2) + [(None, None), self.rho_bounds]

        home_idx = matches["home_team"].map(team_to_idx).to_numpy()
        away_idx = matches["away_team"].map(team_to_idx).to_numpy()
        home_goals = matches["home_goals"].astype(int).to_numpy()
        away_goals = matches["away_goals"].astype(int).to_numpy()
        neutral_site = matches["neutral_site"].astype(bool).to_numpy() if "neutral_site" in matches.columns else np.zeros(len(matches), dtype=bool)
        weights = self._time_weights(matches["date"])
        if "match_weight" in matches.columns:
            match_weights = pd.to_numeric(matches["match_weight"], errors="coerce").fillna(1.0).clip(lower=0.0).to_numpy()
            weights = weights * match_weights

        n_params = n_teams * 2 + 2
        max_function_evaluations = max(50000, min(350000, n_params * 3000))
        result = minimize(
            self._negative_log_likelihood,
            initial,
            args=(home_idx, away_idx, home_goals, away_goals, neutral_site, weights, n_teams, self.l2_penalty),
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 10000, "maxfun": max_function_evaluations},
        )
        if not result.success and not self._can_use_limit_result(result):
            raise RuntimeError(f"Model fitting failed: {result.message}")

        params = result.x
        attacks = params[:n_teams]
        defenses = params[n_teams : n_teams * 2]
        self.attack_ = dict(zip(self.teams_, attacks))
        self.defense_ = dict(zip(self.teams_, defenses))
        self.home_advantage_ = float(params[n_teams * 2])
        self.rho_ = float(params[n_teams * 2 + 1])
        self.fitted_ = True
        self.fit_converged_ = bool(result.success)
        self.fit_message_ = str(result.message)
        return self

    @staticmethod
    def _can_use_limit_result(result) -> bool:
        message = str(result.message).upper()
        reached_limit = "EVALUATIONS EXCEEDS LIMIT" in message or "ITERATIONS REACHED LIMIT" in message
        return reached_limit and np.isfinite(result.fun) and np.all(np.isfinite(result.x))

    def predict(self, home_team: str, away_team: str, max_goals: int = 8, neutral_site: bool = False) -> Prediction:
        if not self.fitted_:
            raise RuntimeError("Model is not fitted yet.")
        if home_team not in self.attack_:
            raise ValueError(f"Unknown home team: {home_team}")
        if away_team not in self.attack_:
            raise ValueError(f"Unknown away team: {away_team}")

        home_advantage = 0.0 if neutral_site else self.home_advantage_
        home_lambda = exp(home_advantage + self.attack_[home_team] + self.defense_[away_team])
        away_mu = exp(self.attack_[away_team] + self.defense_[home_team])
        return prediction_from_expectations(home_team, away_team, home_lambda, away_mu, self.rho_, max_goals)

    def parameters(self) -> pd.DataFrame:
        if not self.fitted_:
            raise RuntimeError("Model is not fitted yet.")
        rows = [
            {"team": team, "attack": self.attack_[team], "defense": self.defense_[team]}
            for team in self.teams_
        ]
        return pd.DataFrame(rows).sort_values("team").reset_index(drop=True)

    def _prepare_matches(self, matches: pd.DataFrame) -> pd.DataFrame:
        missing = REQUIRED_COLUMNS - set(matches.columns)
        if missing:
            raise ValueError(f"Missing required columns: {sorted(missing)}")

        prepared = matches.copy()
        prepared["date"] = pd.to_datetime(prepared["date"])
        prepared["home_goals"] = prepared["home_goals"].astype(int)
        prepared["away_goals"] = prepared["away_goals"].astype(int)
        if "neutral_site" in prepared.columns:
            prepared["neutral_site"] = prepared["neutral_site"].fillna(False).astype(bool)
        return prepared.sort_values("date").reset_index(drop=True)

    def _time_weights(self, dates: pd.Series) -> np.ndarray:
        if self.half_life_days <= 0:
            return np.ones(len(dates), dtype=float)
        latest = dates.max()
        age_days = (latest - dates).dt.days.to_numpy()
        return np.exp(-log(2.0) * age_days / self.half_life_days)

    @staticmethod
    def _negative_log_likelihood(
        params: np.ndarray,
        home_idx: np.ndarray,
        away_idx: np.ndarray,
        home_goals: np.ndarray,
        away_goals: np.ndarray,
        neutral_site: np.ndarray,
        weights: np.ndarray,
        n_teams: int,
        l2_penalty: float,
    ) -> float:
        attacks = params[:n_teams]
        defenses = params[n_teams : n_teams * 2]
        home_advantage = params[n_teams * 2]
        rho = params[n_teams * 2 + 1]

        penalty = 100.0 * (attacks.sum() ** 2 + defenses.sum() ** 2)
        penalty += l2_penalty * float(np.square(attacks).sum() + np.square(defenses).sum() + home_advantage**2 + rho**2)
        total = penalty

        match_home_advantage = np.where(neutral_site, 0.0, home_advantage)
        home_linear = match_home_advantage + attacks[home_idx] + defenses[away_idx]
        away_linear = attacks[away_idx] + defenses[home_idx]
        home_lambda = np.exp(np.clip(home_linear, -20.0, 20.0))
        away_mu = np.exp(np.clip(away_linear, -20.0, 20.0))

        tau = np.ones_like(home_lambda)
        mask_00 = (home_goals == 0) & (away_goals == 0)
        mask_01 = (home_goals == 0) & (away_goals == 1)
        mask_10 = (home_goals == 1) & (away_goals == 0)
        mask_11 = (home_goals == 1) & (away_goals == 1)
        tau[mask_00] = 1.0 - home_lambda[mask_00] * away_mu[mask_00] * rho
        tau[mask_01] = 1.0 + home_lambda[mask_01] * rho
        tau[mask_10] = 1.0 + away_mu[mask_10] * rho
        tau[mask_11] = 1.0 - rho
        if np.any(tau <= 0):
            return 1e12

        log_prob = (
            np.log(tau)
            + home_goals * np.log(home_lambda)
            - home_lambda
            - gammaln(home_goals + 1)
            + away_goals * np.log(away_mu)
            - away_mu
            - gammaln(away_goals + 1)
        )
        total -= float(np.sum(weights * log_prob))
        return float(total)

    @staticmethod
    def _tau(home_goals: int, away_goals: int, home_lambda: float, away_mu: float, rho: float) -> float:
        if home_goals == 0 and away_goals == 0:
            return 1.0 - home_lambda * away_mu * rho
        if home_goals == 0 and away_goals == 1:
            return 1.0 + home_lambda * rho
        if home_goals == 1 and away_goals == 0:
            return 1.0 + away_mu * rho
        if home_goals == 1 and away_goals == 1:
            return 1.0 - rho
        return 1.0

    @staticmethod
    def _poisson_log_probability(goals: int, rate: float) -> float:
        return goals * log(rate) - rate - lgamma(goals + 1)

    @staticmethod
    def _poisson_probability(goals: int, rate: float) -> float:
        return exp(DixonColesModel._poisson_log_probability(goals, rate))


def prediction_from_expectations(
    home_team: str,
    away_team: str,
    home_lambda: float,
    away_mu: float,
    rho: float,
    max_goals: int = 8,
) -> Prediction:
    matrix = np.zeros((max_goals + 1, max_goals + 1), dtype=float)

    for home_goals in range(max_goals + 1):
        for away_goals in range(max_goals + 1):
            base = DixonColesModel._poisson_probability(home_goals, home_lambda) * DixonColesModel._poisson_probability(away_goals, away_mu)
            matrix[home_goals, away_goals] = base * DixonColesModel._tau(home_goals, away_goals, home_lambda, away_mu, rho)

    matrix = matrix / matrix.sum()
    return Prediction(home_team, away_team, home_lambda, away_mu, matrix)
