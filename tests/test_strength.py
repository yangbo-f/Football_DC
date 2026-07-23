import unittest

import pandas as pd

from football_dc.dixon_coles import prediction_from_expectations
from football_dc.strength import (
    apply_cross_confederation_strength_correction,
    combine_strength_frames,
    default_strength_records,
    normalize_strength_frame,
    strength_lookup,
)


class StrengthCorrectionTest(unittest.TestCase):
    def test_default_strength_records_include_confederation(self):
        records = default_strength_records(["Egypt", "Austria"])

        lookup = strength_lookup(records)

        self.assertEqual(lookup["Egypt"].confederation, "CAF")
        self.assertEqual(lookup["Austria"].confederation, "UEFA")

    def test_uploaded_strength_overrides_default(self):
        defaults = default_strength_records(["Egypt"])
        override = normalize_strength_frame(pd.DataFrame([{"team": "Egypt", "confederation": "CAF", "rating": 1600}]))

        combined = combine_strength_frames(defaults, [override])

        self.assertEqual(strength_lookup(combined)["Egypt"].rating, 1600)

    def test_cross_confederation_correction_changes_expectations(self):
        records = default_strength_records(["Egypt", "Austria"])
        prediction = prediction_from_expectations("Egypt", "Austria", 1.0, 1.0, 0.0)

        corrected = apply_cross_confederation_strength_correction(prediction, strength_lookup(records), intensity=1.0)

        self.assertTrue(corrected.applied)
        self.assertLess(corrected.prediction.home_goal_expectation, prediction.home_goal_expectation)
        self.assertGreater(corrected.prediction.away_goal_expectation, prediction.away_goal_expectation)

    def test_same_confederation_not_corrected_by_default(self):
        records = default_strength_records(["Spain", "Austria"])
        prediction = prediction_from_expectations("Spain", "Austria", 1.0, 1.0, 0.0)

        corrected = apply_cross_confederation_strength_correction(prediction, strength_lookup(records), intensity=1.0)

        self.assertFalse(corrected.applied)
        self.assertEqual(corrected.reason, "same_confederation")


if __name__ == "__main__":
    unittest.main()
