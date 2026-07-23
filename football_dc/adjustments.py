from __future__ import annotations

from dataclasses import dataclass

from .dixon_coles import Prediction, prediction_from_expectations


@dataclass(frozen=True)
class ManualAdjustments:
    home_attack_pct: float = 0.0
    home_defense_pct: float = 0.0
    away_attack_pct: float = 0.0
    away_defense_pct: float = 0.0
    home_fitness_pct: float = 0.0
    away_fitness_pct: float = 0.0


def adjustment_factor(*values: float) -> float:
    factor = 1.0
    for value in values:
        factor *= 1.0 + value / 100.0
    return max(0.05, factor)


def apply_manual_adjustments(prediction: Prediction, rho: float, adjustments: ManualAdjustments, max_goals: int) -> Prediction:
    home_factor = adjustment_factor(adjustments.home_attack_pct, -adjustments.away_defense_pct, adjustments.home_fitness_pct)
    away_factor = adjustment_factor(adjustments.away_attack_pct, -adjustments.home_defense_pct, adjustments.away_fitness_pct)
    return prediction_from_expectations(
        prediction.home_team,
        prediction.away_team,
        prediction.home_goal_expectation * home_factor,
        prediction.away_goal_expectation * away_factor,
        rho,
        max_goals,
    )
