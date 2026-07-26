import unittest
from pathlib import Path

import pandas as pd

from football_dc.adjustments import ManualAdjustments, apply_manual_adjustments
from football_dc.catalog import discover_data_sources, load_catalog_sources, source_display_label, source_group
from football_dc.data import COMPETITIONS, load_matches
from football_dc.markets import blend_probabilities, half_full_time_market, total_goals_distribution, value_rows_with_blend
from football_dc.modeling import train_competition_model
from football_dc.odds import match_odds, normalize_odds


ROOT = Path(__file__).resolve().parents[1]


class CatalogOddsAdjustmentsTest(unittest.TestCase):
    def test_catalog_recognizes_competition_directories(self):
        labels = {source.label for source in discover_data_sources()}
        competitions = {source.competition for source in discover_data_sources()}

        self.assertIn("世界杯 2026 正赛", labels)
        self.assertIn("世界杯 2026 预选赛周期", labels)
        self.assertIn("女足世界杯 2023 正赛", labels)
        self.assertIn("女足世界杯 2023 预选赛周期", labels)
        self.assertIn("女足世界杯 2027 预选赛周期", labels)
        self.assertIn("中超 2025", labels)
        self.assertIn("中超 2026", labels)
        self.assertIn("EPL", competitions)
        self.assertIn("CSL", competitions)
        self.assertIn("WorldCupQualifiers", competitions)
        self.assertIn("WomenWorldCup", competitions)
        self.assertIn("WomenWorldCupQualifiers", competitions)
        self.assertIn("CSL", COMPETITIONS)
        self.assertIn("WomenWorldCup", COMPETITIONS)

    def test_catalog_loads_worldcup_finals(self):
        matches, loaded = load_catalog_sources(["世界杯 2026 正赛"])
        expected_rows = len(pd.read_csv(ROOT / "data/worldcup/finals_2026.csv"))

        self.assertEqual(len(matches), expected_rows)
        self.assertEqual(matches["competition"].unique().tolist(), ["WorldCup"])
        self.assertEqual(matches["score_basis"].unique().tolist(), ["FT90"])
        self.assertIn("data/worldcup/finals_2026.csv", loaded)

    def test_catalog_source_display_is_clear(self):
        source = next(source for source in discover_data_sources() if source.key == "worldcup_finals_2026")
        expected_rows = len(pd.read_csv(ROOT / "data/worldcup/finals_2026.csv"))

        self.assertEqual(source_group(source), "世界杯")
        self.assertNotIn(".csv", source_display_label(source))
        self.assertIn(f"{expected_rows}场", source_display_label(source))

    def test_catalog_loads_worldcup_qualifier_cycle(self):
        matches, loaded = load_catalog_sources(["世界杯 2026 预选赛周期"])

        self.assertGreater(len(matches), 800)
        self.assertEqual(matches["competition"].unique().tolist(), ["WorldCupQualifiers"])
        self.assertIn("FT90", matches["score_basis"].unique().tolist())
        self.assertIn("data/worldcup/qualifiers_2026_cycle_all.csv", loaded)

    def test_catalog_loads_women_worldcup_sources(self):
        finals, finals_loaded = load_catalog_sources(["女足世界杯 2023 正赛"])
        previous_qualifiers, previous_qualifiers_loaded = load_catalog_sources(["女足世界杯 2023 预选赛周期"])
        qualifiers, qualifiers_loaded = load_catalog_sources(["女足世界杯 2027 预选赛周期"])

        self.assertEqual(len(finals), 64)
        self.assertEqual(finals["competition"].unique().tolist(), ["WomenWorldCup"])
        self.assertEqual(len(previous_qualifiers), 87)
        self.assertEqual(previous_qualifiers["competition"].unique().tolist(), ["WomenWorldCupQualifiers"])
        self.assertIn("VOID", previous_qualifiers["score_basis"].unique().tolist())
        self.assertEqual(len(qualifiers), 231)
        self.assertEqual(qualifiers["competition"].unique().tolist(), ["WomenWorldCupQualifiers"])
        self.assertEqual(qualifiers["date"].max().date().isoformat(), "2026-06-09")
        self.assertIn("data/women_worldcup/finals_2023.csv", finals_loaded)
        self.assertIn("data/women_worldcup/qualifiers_2023_cycle_all.csv", previous_qualifiers_loaded)
        self.assertIn("data/women_worldcup/qualifiers_2027_cycle_all.csv", qualifiers_loaded)

    def test_odds_csv_matches_fixture(self):
        odds = normalize_odds(
            pd.DataFrame(
                [
                    ["2026-07-03", "WorldCup", "Spain", "Austria", "1X2", "Home", None, 2.0, "Book", "2026-07-02T00:00:00", "csv"],
                    ["2026-07-03", "WorldCup", "Spain", "Austria", "1X2", "Draw", None, 3.5, "Book", "2026-07-02T00:00:00", "csv"],
                ],
                columns=["event_date", "competition", "home_team", "away_team", "market", "selection", "line", "odds_decimal", "bookmaker", "captured_at", "source"],
            )
        )

        matched = match_odds(odds, "WorldCup", "Spain", "Austria", "2026-07-03", "1X2")

        self.assertEqual(matched["Home"], 2.0)
        self.assertEqual(matched["Draw"], 3.5)

    def test_blending_weight_extremes(self):
        model = {"Home": 0.7, "Away": 0.3}
        market = {"Home": 0.4, "Away": 0.6}

        self.assertEqual(blend_probabilities(model, market, 0.0), model)
        self.assertEqual(blend_probabilities(model, market, 1.0), market)

    def test_value_rows_with_blend_changes_ev(self):
        rows = value_rows_with_blend("1X2", {"Home": 0.7, "Away": 0.3}, {"Home": 2.0, "Away": 2.0}, 0.5)

        self.assertNotEqual(rows[0]["model_ev"], rows[0]["blended_ev"])

    def test_manual_adjustment_changes_expectation(self):
        matches = load_matches(ROOT / "data/worldcup/finals_2026.csv")
        trained = train_competition_model(matches, "WorldCup")
        base = trained.model.predict("Spain", "Austria", neutral_site=True)
        adjusted = apply_manual_adjustments(base, trained.model.rho_, ManualAdjustments(home_attack_pct=10.0), 8)

        self.assertGreater(adjusted.home_goal_expectation, base.home_goal_expectation)
        self.assertAlmostEqual(float(adjusted.score_matrix.sum()), 1.0)

    def test_key_markets_sum_to_one(self):
        matches = load_matches(ROOT / "data/worldcup/finals_2026.csv")
        trained = train_competition_model(matches, "WorldCup")
        prediction = trained.model.predict("Spain", "Austria", neutral_site=True)

        self.assertAlmostEqual(sum(total_goals_distribution(prediction).values()), 1.0)
        self.assertAlmostEqual(sum(half_full_time_market(prediction).values()), 1.0)


if __name__ == "__main__":
    unittest.main()
