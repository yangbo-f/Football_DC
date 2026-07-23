import unittest

import pandas as pd

from football_dc.data import normalize_matches
from football_dc.dixon_coles import prediction_from_expectations
from football_dc.ensemble import (
    EloModel,
    LogisticBaselineModel,
    MarketModel,
    build_prediction_result,
    blend_model_probabilities,
)
from football_dc.features import build_pre_match_features


class EnsembleModelsTest(unittest.TestCase):
    def test_elo_model_outputs_normalized_1x2(self):
        probs = EloModel().predict_from_elos(1600, 1500).probabilities

        self.assertAlmostEqual(sum(probs.values()), 1.0)
        self.assertGreater(probs["Home"], probs["Away"])
        self.assertGreater(probs["Draw"], 0.0)

    def test_market_model_uses_no_vig_probabilities(self):
        market = MarketModel().predict({"Home": 2.0, "Draw": 4.0, "Away": 4.0})

        self.assertIsNotNone(market)
        self.assertAlmostEqual(sum(market.probabilities.values()), 1.0)
        self.assertAlmostEqual(market.probabilities["Home"], 0.5)

    def test_ensemble_weights_do_not_mutate_baseline(self):
        prediction = prediction_from_expectations("Spain", "Austria", 2.0, 0.8, rho=0.0)
        market = MarketModel().predict({"Home": 5.0, "Draw": 4.0, "Away": 1.8})

        result = build_prediction_result(prediction, {"Market": market}, {"DixonColes": 0.5, "Market": 0.5})

        self.assertNotEqual(result.baseline.probabilities["Home"], result.final.probabilities["Home"])
        self.assertAlmostEqual(sum(result.final.probabilities.values()), 1.0)

    def test_blend_model_probabilities_respects_weight_extremes(self):
        prediction = prediction_from_expectations("Spain", "Austria", 2.0, 0.8, rho=0.0)
        market = MarketModel().predict({"Home": 5.0, "Draw": 4.0, "Away": 1.8})
        baseline = build_prediction_result(prediction, {}, {}).baseline

        blended = blend_model_probabilities({"DixonColes": baseline, "Market": market}, {"DixonColes": 1.0, "Market": 0.0})

        self.assertAlmostEqual(blended["Home"], baseline.probabilities["Home"])

    def test_logistic_baseline_can_fit_feature_frame(self):
        raw = pd.DataFrame(
            [
                ["WorldCup", "2026", "2026-06-01", "Spain", "Austria", 2, 0, True, "Group", "Group Stage", "FT90"],
                ["WorldCup", "2026", "2026-06-02", "France", "Brazil", 0, 2, True, "Group", "Group Stage", "FT90"],
                ["WorldCup", "2026", "2026-06-03", "England", "Mexico", 1, 1, True, "Group", "Group Stage", "FT90"],
                ["WorldCup", "2026", "2026-06-10", "Spain", "France", 1, 0, True, "Group", "Group Stage", "FT90"],
                ["WorldCup", "2026", "2026-06-11", "Brazil", "Austria", 2, 1, True, "Group", "Group Stage", "FT90"],
                ["WorldCup", "2026", "2026-06-12", "Mexico", "England", 0, 0, True, "Group", "Group Stage", "FT90"],
            ],
            columns=["competition", "season", "date", "home_team", "away_team", "home_goals", "away_goals", "neutral_site", "stage", "round", "score_basis"],
        )
        features = build_pre_match_features(normalize_matches(raw))
        model = LogisticBaselineModel(l2_penalty=0.5).fit(features)

        probs = model.predict_one(features.iloc[-1]).probabilities

        self.assertAlmostEqual(sum(probs.values()), 1.0)
        self.assertEqual(set(probs), {"Home", "Draw", "Away"})


if __name__ == "__main__":
    unittest.main()
