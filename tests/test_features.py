import unittest

import pandas as pd

from football_dc.data import normalize_matches
from football_dc.features import EloConfig, build_pre_match_features


class FeatureEngineeringTest(unittest.TestCase):
    def test_elo_features_are_pre_match_and_update_after_match(self):
        raw = pd.DataFrame(
            [
                ["WorldCup", "2026", "2026-06-01", "Spain", "Austria", 2, 0, True, "Group", "Group Stage", "FT90", 0.95],
                ["WorldCup", "2026", "2026-06-10", "Spain", "France", 1, 1, True, "Group", "Group Stage", "FT90", 0.95],
            ],
            columns=[
                "competition",
                "season",
                "date",
                "home_team",
                "away_team",
                "home_goals",
                "away_goals",
                "neutral_site",
                "stage",
                "round",
                "score_basis",
                "match_importance",
            ],
        )
        matches = normalize_matches(raw)

        features = build_pre_match_features(matches, EloConfig(base_rating=1500.0, k_factor=24.0))

        self.assertEqual(features.loc[0, "home_elo"], 1500.0)
        self.assertEqual(features.loc[0, "away_elo"], 1500.0)
        self.assertGreater(features.loc[1, "home_elo"], 1500.0)
        self.assertEqual(features.loc[1, "away_elo"], 1500.0)
        self.assertGreater(features.loc[1, "home_form_3"], 0.0)
        self.assertEqual(features.loc[0, "home_form_3"], 0.0)

    def test_strength_of_schedule_uses_previous_opponent_elo_only(self):
        raw = pd.DataFrame(
            [
                ["WorldCup", "2026", "2026-06-01", "Spain", "Austria", 2, 0, True, "Group", "Group Stage", "FT90"],
                ["WorldCup", "2026", "2026-06-02", "France", "Brazil", 3, 0, True, "Group", "Group Stage", "FT90"],
                ["WorldCup", "2026", "2026-06-10", "Spain", "France", 1, 1, True, "Group", "Group Stage", "FT90"],
            ],
            columns=["competition", "season", "date", "home_team", "away_team", "home_goals", "away_goals", "neutral_site", "stage", "round", "score_basis"],
        )
        matches = normalize_matches(raw)

        features = build_pre_match_features(matches)

        spain_second_match = features.loc[features["date"] == pd.Timestamp("2026-06-10")].iloc[0]
        self.assertEqual(spain_second_match["home_avg_opponent_elo_5"], 1500.0)
        self.assertEqual(spain_second_match["home_strength_of_schedule"], 0.0)
        self.assertNotEqual(spain_second_match["away_elo"], 1500.0)

    def test_xg_features_are_missing_when_xg_data_is_absent(self):
        raw = pd.DataFrame(
            [
                ["WorldCup", "2026", "2026-06-01", "Spain", "Austria", 2, 0, True, "Group", "Group Stage", "FT90"],
                ["WorldCup", "2026", "2026-06-10", "Spain", "France", 1, 1, True, "Group", "Group Stage", "FT90"],
            ],
            columns=["competition", "season", "date", "home_team", "away_team", "home_goals", "away_goals", "neutral_site", "stage", "round", "score_basis"],
        )
        matches = normalize_matches(raw)

        features = build_pre_match_features(matches)

        self.assertTrue(pd.isna(features.loc[1, "home_rolling_xG_5"]))
        self.assertTrue(pd.isna(features.loc[1, "rolling_xG_5_diff"]))

    def test_xg_features_use_previous_matches_only(self):
        raw = pd.DataFrame(
            [
                ["WorldCup", "2026", "2026-06-01", "Spain", "Austria", 2, 0, True, "Group", "Group Stage", "FT90", 1.8, 0.4],
                ["WorldCup", "2026", "2026-06-10", "Spain", "France", 1, 1, True, "Group", "Group Stage", "FT90", 0.7, 1.5],
            ],
            columns=[
                "competition",
                "season",
                "date",
                "home_team",
                "away_team",
                "home_goals",
                "away_goals",
                "neutral_site",
                "stage",
                "round",
                "score_basis",
                "home_xg",
                "away_xg",
            ],
        )
        matches = normalize_matches(raw)
        matches["home_xg"] = raw["home_xg"]
        matches["away_xg"] = raw["away_xg"]

        features = build_pre_match_features(matches)

        self.assertTrue(pd.isna(features.loc[0, "home_rolling_xG_5"]))
        self.assertEqual(features.loc[1, "home_rolling_xG_5"], 1.8)
        self.assertEqual(features.loc[1, "home_rolling_xGA_5"], 0.4)
        self.assertEqual(features.loc[1, "home_rolling_xGD_5"], 1.4)


if __name__ == "__main__":
    unittest.main()
