import unittest

from football_dc.markets import implied_probability, remove_vig, value_rows


class MarketsTest(unittest.TestCase):
    def test_decimal_odds_to_implied_probability(self):
        self.assertAlmostEqual(implied_probability(2.0), 0.5)
        self.assertIsNone(implied_probability(1.0))

    def test_remove_vig_sums_to_one(self):
        no_vig = remove_vig({"Home": 2.0, "Draw": 4.0, "Away": 4.0})
        self.assertAlmostEqual(sum(no_vig.values()), 1.0)
        self.assertAlmostEqual(no_vig["Home"], 0.5)

    def test_value_rows_expected_value(self):
        rows = value_rows("胜平负", {"Home": 0.60, "Away": 0.40}, {"Home": 2.0, "Away": 2.0})
        self.assertAlmostEqual(rows[0].expected_value, 0.20)
        self.assertGreater(rows[0].edge, 0.0)


if __name__ == "__main__":
    unittest.main()
