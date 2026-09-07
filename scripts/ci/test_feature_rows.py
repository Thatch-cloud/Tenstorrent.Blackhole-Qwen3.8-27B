import unittest

import torch

from feature_rows import compare_feature_rows


class FeatureRowsTests(unittest.TestCase):
    def fixture(self, rows):
        serial = [[[torch.full((1, 1, 1, 2560), row + layer + chip, dtype=torch.bfloat16)
                    for chip in (0, 1)] for layer in (5, 19)] for row in range(rows)]
        captured = [[torch.cat([step[index][chip] for step in serial], dim=2)
                     for chip in (0, 1)] for index in range(2)]
        return serial, captured

    def test_exact_and_shifted_feature_rows(self):
        for rows in (2, 16, 32):
            serial, captured = self.fixture(rows)
            self.assertEqual(len(compare_feature_rows(serial, captured, (5, 19))), 4)
            captured[1][1] = captured[1][1].roll(1, dims=2)
            with self.assertRaisesRegex(AssertionError, 'Stale or misaligned'):
                compare_feature_rows(serial, captured, (5, 19))

    def test_stale_features_missing_shards_and_wrong_axis_fail(self):
        serial, captured = self.fixture(16)
        changed = [[[value + 1 for value in parts] for parts in step] for step in serial]
        with self.assertRaises(AssertionError):
            compare_feature_rows(changed, captured, (5, 19))
        with self.assertRaises(ValueError):
            compare_feature_rows(serial, [captured[0][:1], captured[1]], (5, 19))
        captured[0][0] = captured[0][0].transpose(1, 2)
        with self.assertRaises(ValueError):
            compare_feature_rows(serial, captured, (5, 19))
