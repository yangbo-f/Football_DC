import unittest

import pandas as pd

from football_dc.dixon_coles import prediction_from_expectations
from football_dc.forecast import assess_match_risk, data_completeness_score, market_disagreement, model_scope_summary


class ForecastLayerTest(unittest.TestCase):
    def test_data_completeness_increases_with_team_history(self):
        rows = []
        for idx in range(12):
            rows.append(["WorldCup", "2026", f"2026-06-{idx + 1:02d}", "Spain", "Austria" if idx % 2 else "France"])
            rows.append(["WorldCup", "2026", f"2026-06-{idx + 1:02d}", "Austria", "Brazil"])
        matches = pd.DataFrame(rows, columns=["competition", "season", "date", "home_team", "away_team"])

        score = data_completeness_score(matches, "Spain", "Austria")

        self.assertGreater(score, 0.7)

    def test_market_disagreement_uses_max_absolute_gap(self):
        gap = market_disagreement({"Home": 0.6, "Draw": 0.2, "Away": 0.2}, {"Home": 0.4, "Draw": 0.3, "Away": 0.3})

        self.assertAlmostEqual(gap, 0.2)

    def test_assess_match_risk_flags_uncertain_match(self):
        prediction = prediction_from_expectations("Spain", "Austria", 1.0, 1.0, rho=0.0)
        matches = pd.DataFrame([["WorldCup", "2026", "2026-06-01", "Spain", "France"]], columns=["competition", "season", "date", "home_team", "away_team"])

        risk = assess_match_risk(prediction, matches, "Spain", "Austria")

        self.assertIn(risk.level, {"中风险", "高风险"})
        self.assertLess(risk.data_completeness, 0.55)

    def test_model_scope_summary_mentions_missing_xg(self):
        scope = model_scope_summary("WorldCup", uses_market=True, uses_manual_adjustment=True, has_xg=False)

        self.assertTrue(any("90分钟" in item for item in scope))
        self.assertTrue(any("xG" in item for item in scope))


if __name__ == "__main__":
    unittest.main()
