from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

from .data import filter_competition, filter_training_matches
from .markets import result_market
from .modeling import train_competition_model


SELECTIONS = ("Home", "Draw", "Away")


@dataclass(frozen=True)
class BacktestMetrics:
    model_name: str
    matches: int
    accuracy: float
    log_loss: float
    brier_score: float
    rps: float
    calibration_error: float

    def to_dict(self) -> dict[str, float | int | str]:
        return {
            "model_name": self.model_name,
            "matches": self.matches,
            "accuracy": self.accuracy,
            "log_loss": self.log_loss,
            "brier_score": self.brier_score,
            "rps": self.rps,
            "calibration_error": self.calibration_error,
        }


def walk_forward_dixon_coles(
    matches: pd.DataFrame,
    competition: str,
    min_train_matches: int = 20,
    half_life_days: float = 365.0,
) -> pd.DataFrame:
    rows = filter_training_matches(matches, competition).sort_values("date").reset_index(drop=True)
    predictions = []
    for idx in range(min_train_matches, len(rows)):
        train_rows = rows.iloc[:idx].copy()
        test_row = rows.iloc[idx]
        try:
            trained = train_competition_model(train_rows, competition, half_life_days=half_life_days)
            prediction = trained.model.predict(test_row["home_team"], test_row["away_team"], neutral_site=bool(test_row.get("neutral_site", False)))
        except Exception:
            continue
        probs = result_market(prediction)
        predictions.append(
            {
                "date": test_row["date"],
                "home_team": test_row["home_team"],
                "away_team": test_row["away_team"],
                "actual": actual_result_selection(test_row),
                "Home": probs["Home"],
                "Draw": probs["Draw"],
                "Away": probs["Away"],
                "model_name": "DixonColes",
            }
        )
    return pd.DataFrame(predictions)


def evaluate_predictions(predictions: pd.DataFrame, model_name: str = "model") -> BacktestMetrics:
    if predictions.empty:
        return BacktestMetrics(model_name, 0, 0.0, 0.0, 0.0, 0.0, 0.0)
    probs = predictions[list(SELECTIONS)].apply(pd.to_numeric, errors="coerce").fillna(1.0 / 3.0).to_numpy(dtype=float)
    probs = _normalize_rows(probs)
    actual = predictions["actual"].astype(str).to_numpy()
    y = np.array([SELECTIONS.index(value) for value in actual])
    clipped = np.clip(probs, 1e-12, 1.0)
    predicted = probs.argmax(axis=1)
    one_hot = np.zeros_like(probs)
    one_hot[np.arange(len(y)), y] = 1.0

    accuracy = float((predicted == y).mean())
    log_loss = float(-np.log(clipped[np.arange(len(y)), y]).mean())
    brier = float(np.square(probs - one_hot).sum(axis=1).mean())
    rps = ranked_probability_score(probs, one_hot)
    calibration = calibration_error(probs, y)
    return BacktestMetrics(model_name, len(predictions), accuracy, log_loss, brier, rps, calibration)


def compare_prediction_frames(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    return pd.DataFrame([evaluate_predictions(frame, name).to_dict() for name, frame in frames.items()])


def ranked_probability_score(probabilities: np.ndarray, actual_one_hot: np.ndarray) -> float:
    cumulative_prob = np.cumsum(probabilities, axis=1)
    cumulative_actual = np.cumsum(actual_one_hot, axis=1)
    return float(np.square(cumulative_prob[:, :-1] - cumulative_actual[:, :-1]).sum(axis=1).mean() / (probabilities.shape[1] - 1))


def calibration_error(probabilities: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    confidences = probabilities.max(axis=1)
    correct = probabilities.argmax(axis=1) == actual
    total = len(confidences)
    error = 0.0
    for lower in np.linspace(0.0, 1.0, bins, endpoint=False):
        upper = lower + 1.0 / bins
        mask = (confidences >= lower) & (confidences < upper if upper < 1.0 else confidences <= upper)
        if not mask.any():
            continue
        error += float(mask.mean()) * abs(float(confidences[mask].mean()) - float(correct[mask].mean()))
    return float(error) if total else 0.0


class TemperatureScaler:
    def __init__(self):
        self.temperature_: float = 1.0

    def fit(self, probabilities: pd.DataFrame, actual: pd.Series) -> "TemperatureScaler":
        probs = probabilities[list(SELECTIONS)].to_numpy(dtype=float)
        y = np.array([SELECTIONS.index(value) for value in actual.astype(str)])

        def objective(temp: float) -> float:
            scaled = apply_temperature(probs, temp)
            return float(-np.log(np.clip(scaled[np.arange(len(y)), y], 1e-12, 1.0)).mean())

        result = minimize_scalar(objective, bounds=(0.25, 5.0), method="bounded")
        self.temperature_ = float(result.x)
        return self

    def transform(self, probabilities: pd.DataFrame) -> pd.DataFrame:
        scaled = apply_temperature(probabilities[list(SELECTIONS)].to_numpy(dtype=float), self.temperature_)
        output = probabilities.copy()
        output.loc[:, list(SELECTIONS)] = scaled
        return output


class PlattScaler(TemperatureScaler):
    """Lightweight multinomial Platt-style scaler backed by temperature scaling."""


class IsotonicScaler:
    """Placeholder-compatible calibrator; returns probabilities unchanged until enough data exists."""

    def fit(self, _probabilities: pd.DataFrame, _actual: pd.Series) -> "IsotonicScaler":
        return self

    def transform(self, probabilities: pd.DataFrame) -> pd.DataFrame:
        return probabilities.copy()


def apply_temperature(probabilities: np.ndarray, temperature: float) -> np.ndarray:
    probs = _normalize_rows(np.clip(probabilities, 1e-12, 1.0))
    logits = np.log(probs) / max(temperature, 1e-6)
    logits -= logits.max(axis=1, keepdims=True)
    exp_values = np.exp(logits)
    return _normalize_rows(exp_values)


def actual_result_selection(row: pd.Series) -> str:
    if int(row["home_goals"]) > int(row["away_goals"]):
        return "Home"
    if int(row["home_goals"]) == int(row["away_goals"]):
        return "Draw"
    return "Away"


def write_backtest_report(metrics: pd.DataFrame, path: str | Path) -> Path:
    report_path = Path(path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(report_path, index=False)
    return report_path


def _normalize_rows(values: np.ndarray) -> np.ndarray:
    totals = values.sum(axis=1, keepdims=True)
    totals[totals <= 0] = 1.0
    return values / totals
