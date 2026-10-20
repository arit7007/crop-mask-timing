"""Regression tests for validation completeness and failure handling.

Run: python -m unittest discover -s validation -p test_validation.py
"""

import unittest
from itertools import product

import pandas as pd

from check_units import add_range_check
from verify_masks import compare_saved
from verify_results import ARMS, MODELS, MONTHS, compare_results


class ValidationChecks(unittest.TestCase):
    def setUp(self):
        keys = ["crop", "cutoff_month", "model", "evaluation", "arm"]
        self.primary = pd.DataFrame(product(
            ("CORN",), MONTHS, MODELS,
            ("temporal_holdout", "county_heldout_cv"), ARMS,
        ), columns=keys).assign(rmse=2.0, q_vs_cropland=1.0)
        self.rebuilt = self.primary.rename(columns={
            "rmse": "verification_rmse",
            "q_vs_cropland": "verification_q_vs_cropland",
        })

    def test_complete_results_match(self):
        result = compare_results(self.rebuilt, self.primary)
        self.assertEqual(len(result), 24)
        self.assertTrue(result.passed.all())

    def test_missing_empty_and_duplicate_results_fail(self):
        for broken in (self.primary.iloc[:-1], self.primary.iloc[:0],
                       pd.concat([self.primary, self.primary.iloc[:1]])):
            with self.subTest(rows=len(broken)), self.assertRaises(ValueError):
                compare_results(self.rebuilt, broken)
        with self.assertRaises(ValueError):
            compare_results(self.rebuilt.iloc[:-1], self.primary)

    def test_changed_score_fails(self):
        self.primary.loc[0, "rmse"] = 3.0
        self.assertFalse(compare_results(self.rebuilt, self.primary).passed.all())

    def test_missing_mask_record_is_preserved(self):
        frame = pd.DataFrame({"county_fips": ["01001", "01003"],
                              "year": [2022, 2022], "month": [6, 6],
                              "audit_ndvi": [0.5, 0.6]})
        reference = frame.iloc[:1].rename(columns={"audit_ndvi": "ndvi_mean"})
        result = compare_saved(frame, reference)
        self.assertEqual(len(result), 2)
        self.assertEqual(result.saved_match.iloc[1], "left_only")
        with self.assertRaises(pd.errors.MergeError):
            compare_saved(frame, pd.concat([reference, reference]))

    def test_range_checks_reject_empty_missing_and_outliers(self):
        for values, expected in (([], False), ([None], False),
                                 ([0.5, 2.0], False), ([0.5], True)):
            rows = []
            add_range_check(rows, "test", pd.DataFrame({"value": values}),
                            "value", "unit", "unit", 0, 1, "")
            self.assertEqual(rows[0]["passed"], expected)


if __name__ == "__main__":
    unittest.main()
