import unittest

import torch

from draft_convolution import convolution_reference


class DraftConvolutionTests(unittest.TestCase):
    def test_causal_shift_and_group_boundary(self):
        hidden = torch.zeros((1, 1, 8, 5120), dtype=torch.bfloat16)
        hidden[..., 0, :] = 1
        dynamic = [torch.zeros((1, 1, 8, 320), dtype=torch.bfloat16) for offset in range(2)]
        base = [torch.zeros((1, 1, 1, 5120), dtype=torch.bfloat16) for offset in range(2)]
        dynamic[1][..., 1, 1] = 2
        actual = convolution_reference(hidden, dynamic, base)
        expected = torch.zeros_like(hidden)
        expected[..., 1, 16:32] = 2
        self.assertTrue(torch.equal(actual, expected))

    def test_one_row_does_not_wrap_last_value_into_first(self):
        hidden = torch.ones((1, 1, 1, 5120), dtype=torch.bfloat16)
        dynamic = [torch.ones((1, 1, 1, 320), dtype=torch.bfloat16) for offset in range(2)]
        base = [torch.ones_like(hidden) for offset in range(2)]
        self.assertTrue(torch.equal(convolution_reference(hidden, dynamic, base), hidden * 2))

    def test_rejects_wrong_group_count_and_nonfinite(self):
        hidden = torch.ones((1, 1, 8, 5120), dtype=torch.bfloat16)
        dynamic = [torch.ones((1, 1, 8, 320), dtype=torch.bfloat16) for offset in range(2)]
        base = [torch.ones((1, 1, 1, 5120), dtype=torch.bfloat16) for offset in range(2)]
        with self.assertRaises(ValueError):
            convolution_reference(hidden, dynamic[:1], base)
        hidden[..., 0, 0] = float('nan')
        with self.assertRaises(ValueError):
            convolution_reference(hidden, dynamic, base)
