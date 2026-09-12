"""Long-context replay refresh owns only the final native chunk."""

import unittest

import torch

from attention_head_fold import causal_mask, parallel_groups


class MaskContractTests(unittest.TestCase):
    def test_immutable_prefix_and_refresh_tail_cover_all_t16_positions(self):
        capacity = 4352
        for start in range(4096, capacity - 16 + 1):
            for bundle in parallel_groups(4096, 16):
                for group in bundle:
                    expected = causal_mask(group['rows'], start + group['offset'], capacity)
                    self.assertTrue(bool((expected[..., :capacity - 256] == 0).all()))
                    poisoned = torch.zeros_like(expected)
                    poisoned[..., capacity - 256:] = float('nan')
                    self.assertTrue(bool(torch.isnan(poisoned[..., capacity - 256:]).all()))
                    poisoned[..., capacity - 256:] = expected[..., capacity - 256:]
                    self.assertTrue(torch.equal(poisoned, expected))

    def test_poisoning_immutable_prefix_is_not_a_valid_refresh_test(self):
        expected = causal_mask(4, 4096, 4352)
        poisoned = torch.full_like(expected, float('nan'))
        poisoned[..., -256:] = expected[..., -256:]
        self.assertFalse(torch.equal(poisoned, expected))
        self.assertTrue(bool(torch.isnan(poisoned[..., :-256]).all()))


if __name__ == '__main__':
    unittest.main()
