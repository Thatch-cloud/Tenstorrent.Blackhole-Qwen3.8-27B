import unittest

import torch

from projection_rounding import accumulate_fp32, grouped_projection_reference


class ProjectionRoundingTests(unittest.TestCase):
    def test_destination_accumulation_adds_and_cancels(self):
        initial = torch.tensor([2.0, -2.0, 2.0, 0.0], dtype=torch.float32)
        encoded = initial.view(torch.int32).to(torch.int64) & 0xffffffff
        exponent = torch.full((4,), 127, dtype=torch.int64)
        subtotal = torch.tensor([1024, -1024, -2048, 0], dtype=torch.int64)
        actual = accumulate_fp32(encoded, [(exponent, subtotal), (exponent, torch.zeros_like(subtotal))])
        self.assertTrue(torch.equal(actual.to(torch.int32).view(torch.float32), torch.tensor([3., -3., 0., 0.])))

    def test_destination_accumulation_overflow_and_underflow(self):
        initial = torch.zeros(2, dtype=torch.int64)
        exponent = torch.tensor([255, -1], dtype=torch.int64)
        subtotal = torch.full((2,), 1024, dtype=torch.int64)
        actual = accumulate_fp32(initial, [(exponent, subtotal)])
        self.assertEqual(actual.tolist(), [0x7f800000, 0])

    def test_full_accumulator_single_products_and_zero_groups(self):
        generator = torch.Generator().manual_seed(42)
        features = torch.randn(3, 32, generator=generator).bfloat16()
        features[:, 1:] = 0
        weight = torch.randn(32, 4, generator=generator).bfloat16()
        actual = grouped_projection_reference(features, weight, destination_rounding=True)
        self.assertTrue(torch.equal(actual, features.double() @ weight.double()))
        with self.assertRaises(ValueError):
            grouped_projection_reference(features[:, :8], weight[:8], destination_rounding=True)

    def test_single_products_are_exact_with_all_phases(self):
        generator = torch.Generator().manual_seed(42)
        features = torch.randn(3, 16, generator=generator).bfloat16()
        features[:, 1:] = 0
        weight = torch.randn(16, 4, generator=generator).bfloat16()
        for reverse in (False, True):
            actual = grouped_projection_reference(features, weight, reverse_sources=reverse)
            self.assertTrue(torch.equal(actual, features.double() @ weight.double()))

    def test_equal_exponent_products_are_exact(self):
        features = torch.tensor([[1, -1, 1, -1, 1, 1, 1, 1]], dtype=torch.bfloat16)
        weight = torch.tensor([[2, -2]] * 8, dtype=torch.bfloat16)
        self.assertTrue(torch.equal(grouped_projection_reference(features, weight), features.double() @ weight.double()))

    def test_rounding_boundary_preserves_sum_but_changes_grouped_result(self):
        generator = torch.Generator().manual_seed(1235)
        features = torch.randn(32, 16, generator=generator).bfloat16()
        features[:, 2:] = 0
        weight = torch.randn(16, 32, generator=generator).bfloat16()
        reference = features.double() @ weight.double()
        self.assertFalse(torch.equal(grouped_projection_reference(features, weight), reference))
        permutation = torch.arange(16)
        permutation[1], permutation[8] = 8, 1
        separated = grouped_projection_reference(features[:, permutation], weight[permutation])
        self.assertTrue(torch.equal(separated, reference))

    def test_zero_and_shapes(self):
        result = grouped_projection_reference(torch.zeros(1, 1, 3, 8, dtype=torch.bfloat16),
            torch.ones(8, 2, dtype=torch.bfloat16))
        self.assertEqual(result.shape, (1, 1, 3, 2))
        self.assertEqual(result.count_nonzero(), 0)

    def test_invalid_inputs(self):
        features = torch.ones(2, 8, dtype=torch.bfloat16)
        weight = torch.ones(8, 2, dtype=torch.bfloat16)
        for bad_features, bad_weight, phases in ((features.float(), weight, 4),
                (features[:, :7], weight[:7], 4), (features, weight, True),
                (features * float('inf'), weight, 4)):
            with self.assertRaises(ValueError):
                grouped_projection_reference(bad_features, bad_weight, phases=phases)
