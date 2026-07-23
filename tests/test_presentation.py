import inspect
import unittest
from pathlib import Path

import pandas as pd

from football_dc.presentation import matchup_display, selection_label, styled_probability_matrix
from football_dc.team_names import TEAM_NAME_ZH, team_display_name, team_name_zh


ROOT = Path(__file__).resolve().parents[1]


class PresentationTest(unittest.TestCase):
    def test_team_name_mapping_known_and_unknown(self):
        self.assertEqual(team_name_zh("Arsenal"), "阿森纳")
        self.assertEqual(team_display_name("Spain"), "西班牙（Spain）")
        self.assertEqual(team_display_name("Unknown FC"), "Unknown FC")

    def test_matchup_display_keeps_english_key_visible(self):
        self.assertEqual(matchup_display("Mexico", "Canada"), "墨西哥（Mexico） vs 加拿大（Canada）")

    def test_selection_labels_are_chinese(self):
        self.assertEqual(selection_label("Home"), "主胜")
        self.assertEqual(selection_label("Over"), "大")
        self.assertEqual(selection_label("No"), "否")

    def test_probability_matrix_style_does_not_use_matplotlib_gradient(self):
        source = inspect.getsource(styled_probability_matrix)
        self.assertNotIn("background_gradient", source)
        styled = styled_probability_matrix(pd.DataFrame([[0.10, 0.05], [0.03, 0.02]]))
        html = styled.to_html()
        self.assertIn("background-color", html)
        self.assertIn("10.0%", html)

    def test_current_csv_teams_have_chinese_names(self):
        teams = set()
        for path in (ROOT / "data").glob("**/*.csv"):
            frame = pd.read_csv(path)
            for column in ("home_team", "away_team"):
                if column in frame.columns:
                    teams.update(str(team).strip() for team in frame[column].dropna() if str(team).strip())

        missing = sorted(team for team in teams if team not in TEAM_NAME_ZH)

        self.assertEqual(missing, [])


if __name__ == "__main__":
    unittest.main()
