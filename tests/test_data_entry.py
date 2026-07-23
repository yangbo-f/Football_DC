import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from football_dc.catalog import DataSource
from football_dc.data_entry import MatchEntry, append_match_entry, default_entry_values, target_teams
from football_dc.team_names import save_custom_team_name, team_display_name


class DataEntryTest(unittest.TestCase):
    def test_append_match_entry_writes_selected_source_and_backup(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "finals_2026.csv"
            pd.DataFrame(
                [
                    {
                        "competition": "WorldCup",
                        "season": "2026",
                        "date": "2026-06-11",
                        "home_team": "Mexico",
                        "away_team": "South Africa",
                        "home_goals": 2,
                        "away_goals": 0,
                        "neutral_site": True,
                    }
                ]
            ).to_csv(path, index=False)
            source = DataSource("worldcup_finals_2026", "世界杯 2026 正赛", "WorldCup", path)
            entry = MatchEntry(
                competition="WorldCup",
                season="2026",
                date="2026-07-03",
                home_team="Spain",
                away_team="Austria",
                home_goals=1,
                away_goals=0,
                neutral_site=True,
                stage="Knockout",
                round="Round of 32",
            )

            backup = append_match_entry(source, entry)
            updated = pd.read_csv(path)

            self.assertEqual(len(updated), 2)
            self.assertEqual(updated.iloc[-1]["home_team"], "Spain")
            self.assertIn("stage", updated.columns)
            self.assertTrue(backup.exists())

    def test_append_match_entry_rejects_duplicate_match(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "csl_2026.csv"
            pd.DataFrame(
                [
                    {
                        "competition": "CSL",
                        "season": "2026",
                        "date": "2026-03-06",
                        "home_team": "Chengdu Rongcheng",
                        "away_team": "Shenzhen Peng City",
                        "home_goals": 5,
                        "away_goals": 1,
                        "neutral_site": False,
                    }
                ]
            ).to_csv(path, index=False)
            source = DataSource("csl_2026", "中超 2026", "CSL", path)
            entry = MatchEntry("CSL", "2026", "2026-03-06", "Chengdu Rongcheng", "Shenzhen Peng City", 5, 1, False)

            with self.assertRaisesRegex(ValueError, "重复比赛"):
                append_match_entry(source, entry)

    def test_target_defaults_and_teams_use_selected_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "epl_2025_2026.csv"
            pd.DataFrame(
                [
                    {
                        "competition": "EPL",
                        "season": "2025-2026",
                        "date": "2025-08-16",
                        "home_team": "Arsenal",
                        "away_team": "Chelsea",
                        "home_goals": 2,
                        "away_goals": 1,
                        "neutral_site": False,
                    }
                ]
            ).to_csv(path, index=False)
            source = DataSource("epl_2025_2026", "英超 2025-2026", "EPL", path)

            self.assertEqual(default_entry_values(source)["competition"], "EPL")
            self.assertEqual(target_teams(source), ["Arsenal", "Chelsea"])

    def test_custom_team_name_mapping_can_be_saved(self):
        with tempfile.TemporaryDirectory() as tmp:
            custom_path = Path(tmp) / "team_name_overrides.csv"
            with patch("football_dc.team_names.CUSTOM_TEAM_NAMES_PATH", custom_path):
                from football_dc import team_names

                team_names.custom_team_names.cache_clear()
                save_custom_team_name("Test United", "测试联")

                self.assertEqual(team_display_name("Test United"), "测试联（Test United）")
                self.assertTrue(custom_path.exists())


if __name__ == "__main__":
    unittest.main()
