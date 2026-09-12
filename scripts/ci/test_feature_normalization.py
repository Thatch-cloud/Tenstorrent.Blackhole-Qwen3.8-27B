import unittest

import torch

from feature_normalization import bf16_ulp_distance, projection_row, rms_reference


class FeatureNormalizationTests(unittest.TestCase):
    def test_projection_reduction_preserves_all_channels(self):
        report = dict(passed=True, output_width=5120, checks=[dict(passed=True, chip=chip, rows=1,
            shape=[1, 1, 1, 5120], actual_first_row=[chip + 1.] * 5120) for chip in (1, 0)])
        self.assertTrue(torch.equal(projection_row(report), torch.full((1, 1, 1, 5120), 3., dtype=torch.bfloat16)))
        report['checks'][0]['rows'] = 8
        with self.assertRaises(ValueError):
            projection_row(report)
        with self.assertRaises(ValueError):
            projection_row(dict(passed=False, output_width=5120, checks=[]))

    def test_norm_applies_learned_weight_without_unit_offset(self):
        value = torch.full((1, 1, 1, 5120), 2., dtype=torch.bfloat16)
        weight = torch.full((5120,), .5, dtype=torch.bfloat16)
        self.assertTrue(torch.equal(rms_reference(value, weight), torch.full_like(value, .5)))
        self.assertEqual(rms_reference(torch.zeros_like(value), weight).count_nonzero(), 0)
        with self.assertRaises(ValueError):
            rms_reference(value.float(), weight)
        with self.assertRaises(ValueError):
            rms_reference(value, weight, epsilon=1e-5)

    def test_ulp_metric_handles_negative_values_and_signed_zero(self):
        values = torch.tensor([1., -1., 0., -0.], dtype=torch.bfloat16)
        adjacent = torch.tensor([1.0078125, -1.0078125, -0., 0.], dtype=torch.bfloat16)
        self.assertEqual(bf16_ulp_distance(values, adjacent).tolist(), [1, 1, 0, 0])
        with self.assertRaises(ValueError):
            bf16_ulp_distance(values.float(), adjacent)
