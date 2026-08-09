import unittest
from pathlib import Path

from football_dc.data import (
    STANDARD_MATCH_COLUMNS,
    filter_competition,
    filter_training_matches,
    load_matches,
    normalize_matches,
    normalize_team_name,
    teams_for_competition,
)
from football_dc.modeling import apply_training_match_weights, train_competition_model
from football_dc.quality import check_match_data
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


class DataModelingTest(unittest.TestCase):
    def test_competitions_are_trained_separately(self):
        matches = load_matches(ROOT / "data" / "combined_sample_matches.csv")
        epl = train_competition_model(matches, "EPL")
        worldcup = train_competition_model(matches, "WorldCup")

        self.assertEqual(epl.competition, "EPL")
        self.assertEqual(worldcup.competition, "WorldCup")
        self.assertIn("Arsenal", epl.model.teams_)
        self.assertNotIn("Arsenal", worldcup.model.teams_)
        self.assertIn("Mexico", worldcup.model.teams_)

    def test_worldcup_neutral_prediction_removes_home_advantage(self):
        matches = load_matches(ROOT / "data" / "worldcup_sample_matches.csv")
        trained = train_competition_model(matches, "WorldCup")
        neutral = trained.model.predict("Mexico", "Canada", neutral_site=True)
        non_neutral = trained.model.predict("Mexico", "Canada", neutral_site=False)

        self.assertNotEqual(round(neutral.home_goal_expectation, 6), round(non_neutral.home_goal_expectation, 6))

    def test_teams_for_competition_filters(self):
        matches = load_matches(ROOT / "data" / "combined_sample_matches.csv")
        teams = teams_for_competition(matches, "EPL")

        self.assertIn("Arsenal", teams)
        self.assertNotIn("Mexico", teams)

    def test_worldcup_finals_2026_loads_real_file(self):
        matches = load_matches(ROOT / "data" / "worldcup_finals_2026.csv")

        self.assertEqual(len(matches), 79)
        self.assertEqual(matches["competition"].unique().tolist(), ["WorldCup"])
        self.assertEqual(matches["date"].max().strftime("%Y-%m-%d"), "2026-06-30")

    def test_worldcup_finals_2026_stage_fields(self):
        matches = load_matches(ROOT / "data" / "worldcup" / "finals_2026.csv")

        group_rows = matches[matches["date"] <= pd.Timestamp("2026-06-27")]
        knockout_rows = matches[matches["date"] >= pd.Timestamp("2026-06-28")]
        round_of_32_rows = matches[
            (matches["date"] >= pd.Timestamp("2026-06-28"))
            & (matches["date"] <= pd.Timestamp("2026-07-03"))
        ]

        self.assertEqual(set(group_rows["stage"]), {"Group"})
        self.assertEqual(set(group_rows["round"]), {"Group Stage"})
        self.assertEqual(set(knockout_rows["stage"]), {"Knockout"})
        self.assertEqual(set(round_of_32_rows["round"]), {"Round of 32"})
        self.assertTrue({"Round of 16", "Quarterfinals", "Semifinals"}.issubset(set(knockout_rows["round"])))
        self.assertEqual(set(matches["score_basis"]), {"FT90"})

    def test_normalized_matches_include_standard_data_layer_columns(self):
        matches = load_matches(ROOT / "data" / "worldcup" / "finals_2026.csv")

        self.assertTrue(set(STANDARD_MATCH_COLUMNS).issubset(matches.columns))
        self.assertIn("match_importance", matches.columns)
        self.assertIn("prediction_available_at", matches.columns)
        self.assertEqual(matches["prediction_available_at"].max().strftime("%Y-%m-%d"), matches["date"].max().strftime("%Y-%m-%d"))

    def test_match_importance_initial_rules_are_inferred(self):
        raw = pd.DataFrame(
            [
                ["WorldCup", "2026", "2026-06-11", "Spain", "Austria", 2, 0, "true", "Group", "Group Stage", "FT90"],
                ["WorldCup", "2026", "2026-07-14", "Spain", "France", 2, 0, "true", "Knockout", "Semifinals", "FT90"],
                ["WorldCupQualifiers", "2026", "2025-01-01", "Spain", "Georgia", 1, 0, "false", "Qualification", "Qualification", "FT90"],
                ["EPL", "2025-2026", "2025-08-16", "Arsenal", "Chelsea", 2, 1, "false", "League", "Round 1", "FT90"],
            ],
            columns=["competition", "season", "date", "home_team", "away_team", "home_goals", "away_goals", "neutral_site", "stage", "round", "score_basis"],
        )
        matches = normalize_matches(raw)

        importance_by_date = {row["date"].strftime("%Y-%m-%d"): row["match_importance"] for _, row in matches.iterrows()}
        self.assertEqual(importance_by_date["2026-06-11"], 0.95)
        self.assertEqual(importance_by_date["2026-07-14"], 1.0)
        self.assertEqual(importance_by_date["2025-01-01"], 0.75)
        self.assertEqual(importance_by_date["2025-08-16"], 0.7)

    def test_optional_knockout_fields_are_preserved(self):
        raw = pd.DataFrame(
            [
                {
                    "competition": "WorldCup",
                    "season": "2026",
                    "date": "2026-06-29",
                    "home_team": "Germany",
                    "away_team": "Paraguay",
                    "home_goals": 1,
                    "away_goals": 1,
                    "neutral_site": True,
                    "stage": "Knockout",
                    "round": "Round of 32",
                    "score_basis": "FT90",
                    "decided_by_penalties": True,
                    "winner": "Germany",
                    "notes": "Germany advanced 5-4 on penalties",
                }
            ]
        )

        matches = normalize_matches(raw)

        self.assertTrue(bool(matches.loc[0, "decided_by_penalties"]))
        self.assertEqual(matches.loc[0, "winner"], "Germany")
        self.assertEqual(matches.loc[0, "score_basis"], "FT90")

    def test_non_ft90_matches_are_excluded_from_training(self):
        raw = pd.DataFrame(
            [
                ["WorldCup", "2026", "2026-06-11", "Mexico", "South Africa", 2, 0, True, "FT90"],
                ["WorldCup", "2026", "2026-06-12", "Canada", "Mexico", 1, 1, True, "FT90"],
                ["WorldCup", "2026", "2026-06-13", "South Africa", "Canada", 0, 0, True, "AET"],
            ],
            columns=["competition", "season", "date", "home_team", "away_team", "home_goals", "away_goals", "neutral_site", "score_basis"],
        )
        matches = normalize_matches(raw)

        trained = train_competition_model(matches, "WorldCup")

        self.assertEqual(trained.matches_used, 2)
        self.assertEqual(trained.excluded_non_ft90, 1)

    def test_worldcup_qualifiers_group_with_worldcup_and_full_time_is_ft90(self):
        raw = pd.DataFrame(
            [
                ["WorldCup", "2026", "2026-06-11", "Mexico", "South Africa", 2, 0, "true", "FT90"],
                ["WorldCupQualifiers", "2026", "2023-09-07", "Argentina", "Ecuador", 1, 0, "false", "full_time"],
            ],
            columns=["competition", "season", "date", "home_team", "away_team", "home_goals", "away_goals", "neutral_site", "score_basis"],
        )
        matches = normalize_matches(raw)

        grouped = filter_competition(matches, "WorldCup")
        training = filter_training_matches(matches, "WorldCup")

        self.assertEqual(len(grouped), 2)
        self.assertEqual(len(training), 2)
        qualifier = matches[matches["competition"] == "WorldCupQualifiers"].iloc[0]
        self.assertEqual(qualifier["score_basis"], "FT90")
        self.assertFalse(bool(qualifier["neutral_site"]))

    def test_worldcup_training_weights_reduce_qualifiers(self):
        raw = pd.DataFrame(
            [
                ["WorldCup", "2026", "2026-06-11", "Spain", "Austria", 2, 0, "true", "Group", "Group Stage", "FT90"],
                ["WorldCupQualifiers", "2026", "2025-01-01", "Spain", "Georgia", 1, 0, "false", "Qualification", "Qualification", "FT90"],
            ],
            columns=["competition", "season", "date", "home_team", "away_team", "home_goals", "away_goals", "neutral_site", "stage", "round", "score_basis"],
        )
        matches = normalize_matches(raw)

        weighted = apply_training_match_weights(matches, "WorldCup")

        self.assertEqual(weighted.loc[weighted["competition"] == "WorldCup", "match_weight"].iloc[0], 1.0)
        self.assertEqual(weighted.loc[weighted["competition"] == "WorldCupQualifiers", "match_weight"].iloc[0], 0.45)

    def test_women_worldcup_groups_finals_and_qualifiers(self):
        raw = pd.DataFrame(
            [
                ["WomenWorldCup", "2023", "2023-07-20", "New Zealand", "Norway", 1, 0, "true", "Group", "Group Stage", "FT90"],
                ["WomenWorldCupQualifiers", "2027", "2026-03-03", "Spain", "Iceland", 3, 0, "false", "Qualification", "UEFA league stage", "FT90"],
                ["WorldCup", "2026", "2026-06-11", "Spain", "Austria", 2, 0, "true", "Group", "Group Stage", "FT90"],
            ],
            columns=["competition", "season", "date", "home_team", "away_team", "home_goals", "away_goals", "neutral_site", "stage", "round", "score_basis"],
        )
        matches = normalize_matches(raw)

        grouped = filter_competition(matches, "WomenWorldCup")
        weighted = apply_training_match_weights(grouped, "WomenWorldCup")

        self.assertEqual(set(grouped["competition"]), {"WomenWorldCup", "WomenWorldCupQualifiers"})
        self.assertEqual(len(grouped), 2)
        self.assertEqual(weighted.loc[weighted["competition"] == "WomenWorldCup", "match_weight"].iloc[0], 1.0)
        self.assertEqual(weighted.loc[weighted["competition"] == "WomenWorldCupQualifiers", "match_weight"].iloc[0], 0.45)

    def test_champions_league_groups_main_and_qualifiers(self):
        raw = pd.DataFrame(
            [
                ["ChampionsLeague", "2025-2026", "2025-09-16", "PSV", "Union Saint-Gilloise", 1, 3, "false", "League", "League Phase", "FT90"],
                ["ChampionsLeagueQualifiers", "2025-2026", "2025-07-08", "KuPS Kuopio", "Milsami", 1, 0, "false", "Qualification", "First qualifying round", "FT90"],
                ["EPL", "2025-2026", "2025-08-16", "Arsenal", "Chelsea", 2, 1, "false", "League", "Round 1", "FT90"],
            ],
            columns=["competition", "season", "date", "home_team", "away_team", "home_goals", "away_goals", "neutral_site", "stage", "round", "score_basis"],
        )
        matches = normalize_matches(raw)

        grouped = filter_competition(matches, "ChampionsLeague")
        weighted = apply_training_match_weights(grouped, "ChampionsLeague")

        self.assertEqual(set(grouped["competition"]), {"ChampionsLeague", "ChampionsLeagueQualifiers"})
        self.assertEqual(len(grouped), 2)
        self.assertEqual(weighted.loc[weighted["competition"] == "ChampionsLeague", "match_weight"].iloc[0], 1.0)
        self.assertEqual(weighted.loc[weighted["competition"] == "ChampionsLeagueQualifiers", "match_weight"].iloc[0], 0.45)

    def test_team_aliases_are_normalized(self):
        raw = pd.DataFrame(
            [
                ["WorldCup", "2026", "2026-06-11", "Cape Verde", "Curacao", 1, 0, "true", "FT90"],
                ["WorldCup", "2026", "2026-06-12", "Czech Republic", "Spain", 1, 1, "true", "FT90"],
                ["ChampionsLeague", "2025-2026", "2025-09-16", "B. Dortmund", "Atleti", 4, 4, "false", "FT90"],
            ],
            columns=["competition", "season", "date", "home_team", "away_team", "home_goals", "away_goals", "neutral_site", "score_basis"],
        )

        matches = normalize_matches(raw)

        self.assertIn("Cabo Verde", matches["home_team"].tolist())
        self.assertIn("Curaçao", matches["away_team"].tolist())
        self.assertIn("Czechia", matches["home_team"].tolist())
        self.assertIn("Borussia Dortmund", matches["home_team"].tolist())
        self.assertIn("Atlético de Madrid", matches["away_team"].tolist())
        self.assertEqual(normalize_team_name("Cape Verde"), "Cabo Verde")

    def test_data_quality_report_flags_problem_rows(self):
        raw = pd.DataFrame(
            [
                ["WorldCup", "2026", "2026-06-11", "Spain", "Austria", 2, 0, "true", "Group", "Group Stage", "FT90", False, "", "", "", 0.95, "2026-06-11"],
                ["WorldCup", "2026", "2026-06-11", "Spain", "Austria", 2, 0, "true", "Group", "Group Stage", "FT90", False, "", "", "", 0.95, "2026-06-12"],
                ["WorldCup", "2026", "2026-06-12", "Spain", "Spain", -1, 0, "true", "Group", "Group Stage", "AET", False, "", "", "", "", "2026-06-12"],
            ],
            columns=STANDARD_MATCH_COLUMNS,
        )

        report = check_match_data(raw)

        self.assertTrue(report.has_errors)
        self.assertEqual(report.duplicate_rows, 2)
        self.assertEqual(report.invalid_goal_rows, 1)
        self.assertEqual(report.same_team_rows, 1)
        self.assertEqual(report.non_ft90_rows, 1)
        self.assertEqual(report.missing_match_importance_rows, 1)
        self.assertEqual(report.future_prediction_rows, 1)


if __name__ == "__main__":
    unittest.main()
