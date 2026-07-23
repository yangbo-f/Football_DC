from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict

import numpy as np

from .dixon_coles import Prediction


@dataclass(frozen=True)
class MarketRow:
    market: str
    selection: str
    line: float | None
    model_probability: float
    odds_decimal: float | None
    fair_odds: float | None
    implied_probability: float | None
    no_vig_probability: float | None
    edge: float | None
    expected_value: float | None

    def to_dict(self) -> dict:
        return asdict(self)


def fair_odds(probability: float) -> float | None:
    if probability <= 0:
        return None
    return 1.0 / probability


def implied_probability(odds_decimal: float | None) -> float | None:
    if odds_decimal is None or odds_decimal <= 1.0:
        return None
    return 1.0 / odds_decimal


def remove_vig(selection_to_odds: Dict[str, float]) -> Dict[str, float]:
    implied = {selection: implied_probability(odds) for selection, odds in selection_to_odds.items()}
    valid = {selection: prob for selection, prob in implied.items() if prob is not None}
    total = sum(valid.values())
    if total <= 0:
        return {}
    return {selection: prob / total for selection, prob in valid.items()}


def result_market(prediction: Prediction) -> Dict[str, float]:
    probs = prediction.result_probabilities()
    return {
        "Home": probs["home_win"],
        "Draw": probs["draw"],
        "Away": probs["away_win"],
    }


def totals_market(prediction: Prediction, line: float) -> Dict[str, float]:
    probs = prediction.over_under(line)
    return {"Over": probs["over"], "Under": probs["under"]}


def btts_market(prediction: Prediction) -> Dict[str, float]:
    probs = prediction.both_teams_to_score()
    return {"Yes": probs["yes"], "No": probs["no"]}


def asian_handicap_market(prediction: Prediction, handicap: float) -> Dict[str, float]:
    home = 0.0
    away = 0.0
    push = 0.0
    matrix = prediction.score_matrix
    for home_goals in range(matrix.shape[0]):
        for away_goals in range(matrix.shape[1]):
            prob = float(matrix[home_goals, away_goals])
            adjusted_margin = home_goals - away_goals + handicap
            if adjusted_margin > 0:
                home += prob
            elif adjusted_margin < 0:
                away += prob
            else:
                push += prob
    settled = home + away
    if settled <= 0:
        return {"Home": 0.0, "Away": 0.0, "Push": push}
    return {"Home": home / settled, "Away": away / settled, "Push": push}


def european_handicap_market(prediction: Prediction, handicap: int) -> Dict[str, float]:
    home = 0.0
    draw = 0.0
    away = 0.0
    matrix = prediction.score_matrix
    for home_goals in range(matrix.shape[0]):
        for away_goals in range(matrix.shape[1]):
            prob = float(matrix[home_goals, away_goals])
            adjusted_margin = home_goals + handicap - away_goals
            if adjusted_margin > 0:
                home += prob
            elif adjusted_margin == 0:
                draw += prob
            else:
                away += prob
    return {"Home": home, "Draw": draw, "Away": away}


def total_goals_distribution(prediction: Prediction) -> Dict[str, float]:
    buckets = {"0": 0.0, "1": 0.0, "2": 0.0, "3": 0.0, "4+": 0.0}
    matrix = prediction.score_matrix
    for home_goals in range(matrix.shape[0]):
        for away_goals in range(matrix.shape[1]):
            total = home_goals + away_goals
            key = str(total) if total <= 3 else "4+"
            buckets[key] += float(matrix[home_goals, away_goals])
    return buckets


def over_under_lines(prediction: Prediction, lines=(1.5, 2.5, 3.5)) -> Dict[str, float]:
    output: Dict[str, float] = {}
    for line in lines:
        probs = prediction.over_under(line)
        output[f"Over {line}"] = probs["over"]
        output[f"Under {line}"] = probs["under"]
    return output


def half_full_time_market(prediction: Prediction, first_half_share: float = 0.45, max_half_goals: int = 6) -> Dict[str, float]:
    home_first = prediction.home_goal_expectation * first_half_share
    away_first = prediction.away_goal_expectation * first_half_share
    home_second = prediction.home_goal_expectation * (1.0 - first_half_share)
    away_second = prediction.away_goal_expectation * (1.0 - first_half_share)
    labels = {"H": "胜", "D": "平", "A": "负"}
    result = {f"{first}{full}": 0.0 for first in labels.values() for full in labels.values()}

    first_home_probs = poisson_probs(home_first, max_half_goals)
    first_away_probs = poisson_probs(away_first, max_half_goals)
    second_home_probs = poisson_probs(home_second, max_half_goals)
    second_away_probs = poisson_probs(away_second, max_half_goals)

    for fh, p_fh in enumerate(first_home_probs):
        for fa, p_fa in enumerate(first_away_probs):
            first_label = result_label(fh, fa)
            first_prob = p_fh * p_fa
            for sh, p_sh in enumerate(second_home_probs):
                for sa, p_sa in enumerate(second_away_probs):
                    full_label = result_label(fh + sh, fa + sa)
                    result[f"{labels[first_label]}{labels[full_label]}"] += float(first_prob * p_sh * p_sa)
    total = sum(result.values())
    if total > 0:
        result = {key: value / total for key, value in result.items()}
    return result


def poisson_probs(rate: float, max_goals: int) -> np.ndarray:
    values = np.array([np.exp(goal * np.log(rate) - rate - sum(np.log(np.arange(1, goal + 1)))) if goal else np.exp(-rate) for goal in range(max_goals + 1)])
    return values / values.sum()


def result_label(home_goals: int, away_goals: int) -> str:
    if home_goals > away_goals:
        return "H"
    if home_goals == away_goals:
        return "D"
    return "A"


def blend_probabilities(model_probs: Dict[str, float], market_probs: Dict[str, float], market_weight: float) -> Dict[str, float]:
    if not market_probs:
        return model_probs.copy()
    weight = max(0.0, min(1.0, market_weight))
    blended = {}
    for selection, model_prob in model_probs.items():
        blended[selection] = (1.0 - weight) * model_prob + weight * market_probs.get(selection, model_prob)
    total = sum(blended.values())
    if total > 0:
        blended = {selection: prob / total for selection, prob in blended.items()}
    return blended


def value_rows(market: str, model_probs: Dict[str, float], odds: Dict[str, float], line: float | None = None) -> list[MarketRow]:
    no_vig = remove_vig(odds)
    rows: list[MarketRow] = []
    for selection, model_prob in model_probs.items():
        offered = odds.get(selection)
        implied = implied_probability(offered)
        no_vig_prob = no_vig.get(selection)
        rows.append(
            MarketRow(
                market=market,
                selection=selection,
                line=line,
                model_probability=model_prob,
                odds_decimal=offered,
                fair_odds=fair_odds(model_prob),
                implied_probability=implied,
                no_vig_probability=no_vig_prob,
                edge=None if no_vig_prob is None else model_prob - no_vig_prob,
                expected_value=None if offered is None else model_prob * offered - 1.0,
            )
        )
    return rows


def value_rows_with_blend(
    market: str,
    model_probs: Dict[str, float],
    odds: Dict[str, float],
    market_weight: float,
    line: float | None = None,
) -> list[dict]:
    no_vig = remove_vig(odds)
    blended = blend_probabilities(model_probs, no_vig, market_weight)
    rows = []
    for selection, model_prob in model_probs.items():
        offered = odds.get(selection)
        no_vig_prob = no_vig.get(selection)
        blended_prob = blended.get(selection, model_prob)
        rows.append(
            {
                "market": market,
                "selection": selection,
                "line": line,
                "model_probability": model_prob,
                "blended_probability": blended_prob,
                "odds_decimal": offered,
                "no_vig_probability": no_vig_prob,
                "model_ev": None if offered is None else model_prob * offered - 1.0,
                "blended_ev": None if offered is None else blended_prob * offered - 1.0,
            }
        )
    return rows
