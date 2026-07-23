import unittest

import pandas as pd

from football_dc.data import normalize_matches


class FootballDataImportTest(unittest.TestCase):
    def test_football_data_columns_are_normalized(self):
        raw = pd.DataFrame(
            [
                {
                    "Div": "E0",
                    "Date": "16/08/25",
                    "HomeTeam": "Liverpool",
                    "AwayTeam": "Bournemouth",
                    "FTHG": 4,
                    "FTAG": 2,
                }
            ]
        )

        matches = normalize_matches(raw, season="2025-2026")

        self.assertEqual(matches.loc[0, "competition"], "EPL")
        self.assertEqual(matches.loc[0, "home_team"], "Liverpool")
        self.assertEqual(matches.loc[0, "date"].strftime("%Y-%m-%d"), "2025-08-16")
        self.assertFalse(bool(matches.loc[0, "neutral_site"]))


if __name__ == "__main__":
    unittest.main()
