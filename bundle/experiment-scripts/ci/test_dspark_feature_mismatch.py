import unittest

import torch

from dspark_feature_mismatch import FeatureMismatch


class FeatureMismatchTests(unittest.TestCase):
    def test_records_exact_first_difference_without_relaxing_tolerance(self):
        expected = torch.ones(1, 1, 2, 4, dtype=torch.bfloat16)
        actual = expected.clone()
        actual[0, 0, 1, 2] = 2
        error = FeatureMismatch(actual, expected, tap=5, chip=0, position=4096)
        self.assertEqual(error.evidence['mismatched_elements'], 1)
        self.assertEqual(error.evidence['first_coordinate'], [0, 0, 1, 2])
        self.assertEqual(error.evidence['max_abs'], '1.0')
        self.assertNotEqual(error.evidence['actual_sha256'], error.evidence['expected_sha256'])
