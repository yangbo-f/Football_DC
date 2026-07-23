from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from .dixon_coles import Prediction
from .features import EloConfig, build_pre_match_features
from .markets import remove_vig, result_market


RESULT_SELECTIONS = ("Home", "Draw", "Away")


@dataclass(frozen=True)
class ModelProbabilities:
    model_name: str
    probabilities: dict[str, float]
    details: dict[str, float | str] = field(default_factory=dict)


@dataclass(frozen=True)
class PredictionResult:
    home_team: str
    away_team: str
    baseline: ModelProbabilities
    components: dict[str, ModelProbabilities]
    final: ModelProbabilities
    score_prediction: Prediction | None = None


def normalize_probability_dict(probabilities: Mapping[str, float]) -> dict[str, float]:
    cleaned = {selection: max(0.0, float(probabilities.get(selection, 0.0))) for selection in RESULT_SELECTIONS}
    total = sum(cleaned.values())
    if total <= 0:
        return {"Home": 1.0 / 3.0, "Draw": 1.0 / 3.0, "Away": 1.0 / 3.0}
    return {selection: value / total for selection, value in cleaned.items()}


class EloModel:
    def __init__(self, draw_base: float = 0.26, draw_sensitivity: float = 0.10):
        self.draw_base = draw_base
        self.draw_sensitivity = draw_sensitivity

    def predict_from_elos(self, home_elo: float, away_elo: float, neutral_site: bool = True, home_advantage: float = 60.0) -> ModelProbabilities:
        adjusted_home = home_elo + (0.0 if neutral_site else home_advantage)
        win_share = 1.0 / (1.0 + 10.0 ** ((away_elo - adjusted_home) / 400.0))
        diff = abs(adjusted_home - away_elo)
        draw = float(np.clip(self.draw_base * np.exp(-self.draw_sensitivity * diff / 100.0), 0.12, 0.32))
        remaining = 1.0 - draw
        probs = normalize_probability_dict({"Home": remaining * win_share, "Draw": draw, "Away": remaining * (1.0 - win_share)})
        return ModelProbabilities("Elo", probs, {"home_elo": home_elo, "away_elo": away_elo, "elo_diff": home_elo - away_elo})


class MarketModel:
    def predict(self, odds: Mapping[str, float]) -> ModelProbabilities | None:
        no_vig = remove_vig(dict(odds))
        if not no_vig:
            return None
        return ModelProbabilities("Market", normalize_probability_dict(no_vig))


class LogisticBaselineModel:
    def __init__(self, feature_columns: list[str] | None = None, l2_penalty: float = 0.1):
        self.feature_columns = feature_columns or [
            "elo_diff",
            "form_5_diff",
            "weighted_form_5_diff",
            "strength_of_schedule_diff",
        ]
        self.l2_penalty = l2_penalty
        self.coef_: np.ndarray | None = None
        self.mean_: np.ndarray | None = None
        self.scale_: np.ndarray | None = None
        self.fitted_ = False

    def fit(self, features: pd.DataFrame) -> "LogisticBaselineModel":
        training = features.dropna(subset=["home_goals", "away_goals"]).copy()
        if training.empty:
            raise ValueError("No completed matches available for LogisticBaselineModel.")
        x = self._feature_matrix(training, fit=True)
        y = training.apply(_result_class, axis=1).to_numpy(dtype=int)
        n_features = x.shape[1]
        initial = np.zeros((n_features + 1) * 3)
        result = minimize(
            _softmax_loss,
            initial,
            args=(x, y, self.l2_penalty),
            method="L-BFGS-B",
            options={"maxiter": 1000, "maxfun": 50000},
        )
        if not result.success:
            raise RuntimeError(f"Logistic baseline fitting failed: {result.message}")
        self.coef_ = result.x.reshape(n_features + 1, 3)
        self.fitted_ = True
        return self

    def predict_one(self, row: pd.Series | Mapping[str, float]) -> ModelProbabilities:
        if not self.fitted_ or self.coef_ is None:
            raise RuntimeError("LogisticBaselineModel is not fitted yet.")
        frame = pd.DataFrame([dict(row)])
        x = self._feature_matrix(frame, fit=False)
        logits = np.c_[np.ones(len(x)), x] @ self.coef_
        probs = _softmax(logits)[0]
        return ModelProbabilities("LogisticBaseline", dict(zip(RESULT_SELECTIONS, probs.astype(float))))

    def _feature_matrix(self, frame: pd.DataFrame, fit: bool) -> np.ndarray:
        matrix = frame.reindex(columns=self.feature_columns).copy()
        matrix = matrix.apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=float)
        if fit:
            self.mean_ = matrix.mean(axis=0)
            self.scale_ = matrix.std(axis=0)
            self.scale_[self.scale_ == 0] = 1.0
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("Feature scaler is not fitted.")
        return (matrix - self.mean_) / self.scale_


def build_prediction_result(
    score_prediction: Prediction,
    component_probabilities: Mapping[str, ModelProbabilities],
    weights: Mapping[str, float] | None = None,
) -> PredictionResult:
    baseline = ModelProbabilities("DixonColes", normalize_probability_dict(result_market(score_prediction)))
    components = {"DixonColes": baseline, **dict(component_probabilities)}
    final_probs = blend_model_probabilities(components, weights)
    return PredictionResult(
        home_team=score_prediction.home_team,
        away_team=score_prediction.away_team,
        baseline=baseline,
        components=components,
        final=ModelProbabilities("Ensemble", final_probs),
        score_prediction=score_prediction,
    )


def blend_model_probabilities(
    components: Mapping[str, ModelProbabilities],
    weights: Mapping[str, float] | None = None,
) -> dict[str, float]:
    if not components:
        return normalize_probability_dict({})
    raw_weights = weights or {"DixonColes": 0.55, "Elo": 0.2, "LogisticBaseline": 0.15, "Market": 0.1}
    usable = {name: max(0.0, float(raw_weights.get(name, 0.0))) for name in components}
    if sum(usable.values()) <= 0:
        usable = {name: 1.0 for name in components}
    blended = {selection: 0.0 for selection in RESULT_SELECTIONS}
    total_weight = sum(usable.values())
    for name, component in components.items():
        weight = usable[name] / total_weight
        probs = normalize_probability_dict(component.probabilities)
        for selection in RESULT_SELECTIONS:
            blended[selection] += weight * probs[selection]
    return normalize_probability_dict(blended)


def latest_feature_row_for_match(matches: pd.DataFrame, home_team: str, away_team: str, match_date, config: EloConfig | None = None) -> pd.Series:
    features = build_pre_match_features(matches, config=config)
    target_date = pd.to_datetime(match_date)
    candidates = features[
        (features["date"] == target_date)
        & (features["home_team"].astype(str) == home_team)
        & (features["away_team"].astype(str) == away_team)
    ]
    if candidates.empty:
        history = features[features["date"] < target_date].copy()
        if history.empty:
            elo = EloConfig().base_rating
            return pd.Series({"home_team": home_team, "away_team": away_team, "date": target_date, "home_elo": elo, "away_elo": elo, "elo_diff": 0.0})
        home_rows = history[(history["home_team"] == home_team) | (history["away_team"] == home_team)]
        away_rows = history[(history["home_team"] == away_team) | (history["away_team"] == away_team)]
        return pd.Series(
            {
                "home_team": home_team,
                "away_team": away_team,
                "date": target_date,
                "home_elo": _latest_team_elo(home_rows, home_team),
                "away_elo": _latest_team_elo(away_rows, away_team),
                "elo_diff": _latest_team_elo(home_rows, home_team) - _latest_team_elo(away_rows, away_team),
            }
        )
    return candidates.iloc[-1]


def _latest_team_elo(rows: pd.DataFrame, team: str, fallback: float = 1500.0) -> float:
    if rows.empty:
        return fallback
    last = rows.sort_values("date").iloc[-1]
    if last["home_team"] == team:
        return float(last["home_elo"])
    return float(last["away_elo"])


def _result_class(row: pd.Series) -> int:
    if int(row["home_goals"]) > int(row["away_goals"]):
        return 0
    if int(row["home_goals"]) == int(row["away_goals"]):
        return 1
    return 2


def _softmax_loss(params: np.ndarray, x: np.ndarray, y: np.ndarray, l2_penalty: float) -> float:
    coef = params.reshape(x.shape[1] + 1, 3)
    logits = np.c_[np.ones(len(x)), x] @ coef
    probs = _softmax(logits)
    likelihood = -np.log(probs[np.arange(len(y)), y] + 1e-12).mean()
    penalty = l2_penalty * float(np.square(coef[1:]).sum()) / max(1, len(y))
    return float(likelihood + penalty)


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp_values = np.exp(shifted)
    return exp_values / exp_values.sum(axis=1, keepdims=True)
