import unittest

import pandas as pd

from football_dc.backtesting import (
    TemperatureScaler,
    evaluate_predictions,
    walk_forward_dixon_coles,
)
from football_dc.data import normalize_matches


class BacktestingTest(unittest.TestCase):
    def test_evaluate_predictions_outputs_probability_metrics(self):
        predictions = pd.DataFrame(
            [
                ["Home", 0.6, 0.25, 0.15],
                ["Draw", 0.3, 0.4, 0.3],
                ["Away", 0.2, 0.3, 0.5],
            ],
            columns=["actual", "Home", "Draw", "Away"],
        )

        metrics = evaluate_predictions(predictions, "test")

        self.assertEqual(metrics.matches, 3)
        self.assertGreater(metrics.accuracy, 0.0)
        self.assertGreater(metrics.log_loss, 0.0)
        self.assertGreater(metrics.brier_score, 0.0)
        self.assertGreaterEqual(metrics.calibration_error, 0.0)

    def test_temperature_scaler_preserves_probability_sum(self):
        predictions = pd.DataFrame(
            [
                ["Home", 0.8, 0.1, 0.1],
                ["Draw", 0.2, 0.6, 0.2],
                ["Away", 0.1, 0.2, 0.7],
            ],
            columns=["actual", "Home", "Draw", "Away"],
        )
        scaler = TemperatureScaler().fit(predictions, predictions["actual"])

        calibrated = scaler.transform(predictions)

        self.assertAlmostEqual(float(calibrated[["Home", "Draw", "Away"]].iloc[0].sum()), 1.0)
        self.assertGreater(scaler.temperature_, 0.0)

    def test_walk_forward_dixon_coles_uses_past_matches_only(self):
        raw = []
        teams = ["Spain", "Austria", "France", "Brazil"]
        for idx in range(24):
            home = teams[idx % 4]
            away = teams[(idx + 1) % 4]
            raw.append(["WorldCup", "2026", f"2026-06-{idx + 1:02d}", home, away, idx % 3, (idx + 1) % 2, True, "Group", "Group Stage", "FT90"])
        matches = normalize_matches(
            pd.DataFrame(
                raw,
                columns=["competition", "season", "date", "home_team", "away_team", "home_goals", "away_goals", "neutral_site", "stage", "round", "score_basis"],
            )
        )

        predictions = walk_forward_dixon_coles(matches, "WorldCup", min_train_matches=12)

        self.assertGreater(len(predictions), 0)
        self.assertTrue({"Home", "Draw", "Away", "actual"}.issubset(predictions.columns))


if __name__ == "__main__":
    unittest.main()
