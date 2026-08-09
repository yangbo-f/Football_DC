import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

import app
from football_dc.dixon_coles import prediction_from_expectations


class AppDefaultsTest(unittest.TestCase):
    def test_worldcup_is_default_competition_when_available(self):
        self.assertEqual(app.default_competition_index(["EPL", "WorldCup"]), 1)

    def test_worldcup_default_fixture_prefers_spain_austria(self):
        teams = ["Argentina", "Austria", "Spain"]

        self.assertEqual(app.default_home_index(teams, "WorldCup"), 2)
        self.assertEqual(app.default_away_index(teams, "WorldCup"), 1)

    def test_team_select_options_start_empty(self):
        options = app.team_select_options(["Spain", "Austria"])

        self.assertEqual(options[0], app.NO_TEAM_SELECTION)
        self.assertEqual(app.team_select_display(app.NO_TEAM_SELECTION), "请选择球队")

    def test_all_team_candidates_exclude_aliases(self):
        from football_dc.team_names import all_team_names_zh

        names = all_team_names_zh()

        self.assertIn("Cabo Verde", names)
        self.assertIn("Curaçao", names)
        self.assertIn("Czechia", names)
        self.assertNotIn("Cape Verde", names)
        self.assertNotIn("Curacao", names)
        self.assertNotIn("Czech Republic", names)

    def test_history_html_escape(self):
        self.assertEqual(app.html_escape("<x&y>"), "&lt;x&amp;y&gt;")

    def test_top_scores_compact_html_styles_probability(self):
        html = app.top_scores_compact_html([("2-0", 0.173)])

        self.assertIn("top-score-score", html)
        self.assertIn("top-score-prob", html)
        self.assertIn("17.3%", html)

    def test_worldcup_qualifiers_are_grouped_under_worldcup(self):
        matches = pd.DataFrame({"competition": ["WorldCupQualifiers", "WomenWorldCupQualifiers", "ChampionsLeagueQualifiers"]})

        self.assertEqual(app.selectable_competitions(matches), ["ChampionsLeague", "WomenWorldCup", "WorldCup"])
        self.assertEqual(app.competition_display_name("WorldCup"), "世界杯（正赛 + 预选赛周期）")
        self.assertEqual(app.competition_display_name("WomenWorldCup"), "女足世界杯（正赛 + 预选赛周期）")
        self.assertEqual(app.competition_display_name("ChampionsLeague"), "欧冠（正赛 + 预选赛）")

    def test_stage_display_helpers_are_chinese(self):
        self.assertEqual(app.stage_display_name("Group"), "小组赛")
        self.assertEqual(app.round_display_name("Round of 32"), "32强淘汰赛")
        self.assertEqual(app.score_basis_display_name("FT90"), "90分钟比分（FT90）")

    def test_history_preview_uses_chinese_columns(self):
        matches = pd.DataFrame(
            [
                {
                    "competition": "WorldCup",
                    "season": "2026",
                    "date": pd.Timestamp("2026-06-28"),
                    "home_team": "Brazil",
                    "away_team": "Japan",
                    "home_goals": 2,
                    "away_goals": 1,
                    "neutral_site": True,
                    "stage": "Knockout",
                    "round": "Round of 32",
                    "score_basis": "FT90",
                    "winner": "",
                    "decided_by_penalties": False,
                    "notes": "",
                }
            ]
        )

        preview = app.history_preview(matches, "WorldCup")

        self.assertIn("阶段", preview.columns)
        self.assertIn("轮次", preview.columns)
        self.assertEqual(preview.iloc[0]["阶段"], "淘汰赛")

    def test_history_preview_filters_selected_teams(self):
        matches = pd.DataFrame(
            [
                {
                    "competition": "WorldCup",
                    "season": "2026",
                    "date": pd.Timestamp("2026-06-11"),
                    "home_team": "Spain",
                    "away_team": "Austria",
                    "home_goals": 2,
                    "away_goals": 0,
                    "neutral_site": True,
                    "stage": "Group",
                    "round": "Group Stage",
                    "score_basis": "FT90",
                    "winner": "",
                    "decided_by_penalties": False,
                    "notes": "",
                },
                {
                    "competition": "WorldCup",
                    "season": "2026",
                    "date": pd.Timestamp("2026-06-12"),
                    "home_team": "Brazil",
                    "away_team": "Japan",
                    "home_goals": 1,
                    "away_goals": 1,
                    "neutral_site": True,
                    "stage": "Group",
                    "round": "Group Stage",
                    "score_basis": "FT90",
                    "winner": "",
                    "decided_by_penalties": False,
                    "notes": "",
                },
            ]
        )

        preview = app.history_preview(matches, "WorldCup", "Spain", "Austria")

        self.assertEqual(len(preview), 1)
        self.assertIn("西班牙", preview.iloc[0]["主队"])

    def test_parse_decimal_input_allows_empty_odds(self):
        self.assertIsNone(app.parse_decimal_input(""))
        self.assertIsNone(app.parse_decimal_input("abc"))
        self.assertEqual(app.parse_decimal_input("2.15"), 2.15)

    def test_source_selection_local_state_round_trip(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "local_state.json"

            app.save_persisted_source_keys(["worldcup_finals_2026", "csl_2026"], path)
            keys, has_cache = app.load_persisted_source_keys({"worldcup_finals_2026", "epl_2025_2026"}, path)

            self.assertTrue(has_cache)
            self.assertEqual(keys, {"worldcup_finals_2026"})

            app.clear_persisted_source_keys(path)
            keys, has_cache = app.load_persisted_source_keys({"worldcup_finals_2026"}, path)

            self.assertFalse(has_cache)
            self.assertEqual(keys, set())

    def test_source_group_summary_from_keys(self):
        class Source:
            def __init__(self, key, competition):
                self.key = key
                self.competition = competition

        sources = [
            Source("worldcup_finals_2026", "WorldCup"),
            Source("women_worldcup_finals_2023", "WomenWorldCup"),
            Source("champions_league_main_2025_2026", "ChampionsLeague"),
            Source("csl_2026", "CSL"),
            Source("epl_2025_2026", "EPL"),
        ]

        self.assertEqual(app.source_group_summary_from_keys({"csl_2026"}, sources), "中超")
        self.assertEqual(app.source_group_summary_from_keys({"worldcup_finals_2026", "csl_2026"}, sources), "世界杯+中超")
        self.assertEqual(app.source_group_summary_from_keys({"women_worldcup_finals_2023"}, sources), "女足世界杯")
        self.assertEqual(app.source_group_summary_from_keys({"champions_league_main_2025_2026"}, sources), "欧冠")
        self.assertEqual(app.source_group_summary_from_keys(set(), sources), "未选数据")

    def test_match_entry_success_message_is_explicit(self):
        message = app.match_entry_success_message("Spain", "Austria", 3, 0, "世界杯 2026 正赛")

        self.assertIn("西班牙（Spain） vs 奥地利（Austria） 3:0 数据已保存", message)
        self.assertIn("世界杯 2026 正赛", message)

    def test_worldcup_qualifier_fast_mode_keeps_relevant_qualifiers(self):
        matches = pd.DataFrame(
            [
                {"competition": "WorldCup", "date": pd.Timestamp("2026-06-11"), "home_team": "Spain", "away_team": "Austria"},
                {"competition": "WorldCupQualifiers", "date": pd.Timestamp("2025-01-01"), "home_team": "Spain", "away_team": "Georgia"},
                {"competition": "WorldCupQualifiers", "date": pd.Timestamp("2025-01-02"), "home_team": "Brazil", "away_team": "Bolivia"},
            ]
        )

        optimized = app.optimize_worldcup_qualifiers(matches, "WorldCup", "Spain", "Austria", True)

        self.assertEqual(len(optimized), 2)
        self.assertIn("Georgia", optimized["away_team"].tolist())
        self.assertNotIn("Bolivia", optimized["away_team"].tolist())

    def test_league_fast_mode_keeps_recent_matches(self):
        matches = pd.DataFrame(
            [
                {
                    "competition": "EPL",
                    "date": pd.Timestamp("2020-01-01") + pd.Timedelta(days=index),
                    "home_team": f"Team{index % 20}",
                    "away_team": f"Team{(index + 1) % 20}",
                    "home_goals": 1,
                    "away_goals": 1,
                    "neutral_site": False,
                }
                for index in range(900)
            ]
        )

        optimized = app.optimize_training_matches(matches, "EPL", "Team1", "Team2", True)

        self.assertEqual(len(optimized), 800)
        self.assertEqual(optimized["date"].min(), pd.Timestamp("2020-04-10"))

    def test_adjustment_summary_shows_changes(self):
        base = prediction_from_expectations("A", "B", 1.0, 1.0, 0.0)
        adjusted = prediction_from_expectations("A", "B", 1.1, 0.9, 0.0)

        summary = app.adjustment_summary(base, adjusted)

        self.assertIn("人工修正后预期进球", summary)
        self.assertIn("主队 +0.10", summary)


if __name__ == "__main__":
    unittest.main()
