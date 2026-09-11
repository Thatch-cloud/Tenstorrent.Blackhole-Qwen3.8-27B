from types import SimpleNamespace
import unittest

import torch

from dspark_inputs import MASK_TOKEN
from dspark_t32_inputs import proposal_inputs
from dspark_full_attention import geometry


class T32InputsTests(unittest.TestCase):
    def rotary(self):
        return SimpleNamespace(tables=lambda position, rows: (
            torch.full((1, 1, rows, 128), .5, dtype=torch.bfloat16),
            torch.full((1, 1, rows, 128), .25, dtype=torch.bfloat16)))

    def test_every_frontier_masks_uncommitted_gap_and_padding(self):
        for position, capacity in ((1, 32), (31, 32), (32, 32), (4096, 4384), (4384, 4384), (8192, 8192)):
            values = proposal_inputs(123, position, capacity, self.rotary())
            mask = values['mask'][0, 0]
            expected = torch.zeros_like(mask, dtype=torch.bool)
            expected[:31, :position] = True
            expected[:31, capacity:capacity + 31] = True
            expected[31, capacity] = True
            self.assertTrue(torch.equal(mask == 0, expected))
            self.assertTrue(torch.isneginf(mask[~expected]).all())
            self.assertTrue(torch.isfinite(mask).any(dim=-1).all())
            self.assertEqual(mask.shape[1] % 64, 0)

    def test_query_zero_and_inactive_row_are_preserved(self):
        values = proposal_inputs(123, 4096, 4384, self.rotary())
        self.assertEqual(values['identifiers'].tolist(), [[123] + [MASK_TOKEN] * 30])
        self.assertEqual(values['live'].sum(), 31)
        self.assertTrue(torch.all(values['cosine'][:, :, 31] == 1))
        self.assertTrue(torch.all(values['sine'][:, :, 31] == 0))
        self.assertTrue(torch.all(values['cosine'][:, :, :31] == .5))

    def test_invalid_inputs_reject(self):
        for anchor, position, capacity in ((True, 1, 32), (-1, 1, 32), (1, True, 32),
                (1, 0, 32), (1, 33, 32), (1, 1, 33), (1, 1, 8224)):
            with self.assertRaises(ValueError):
                proposal_inputs(anchor, position, capacity, self.rotary())
        with self.assertRaises(ValueError):
            proposal_inputs(1, 1, 32, SimpleNamespace(tables=lambda *args: (torch.zeros(31),) * 2))

    def test_existing_runtime_still_rejects_unqualified_width(self):
        with self.assertRaises(ValueError):
            geometry(4096, 31)


if __name__ == '__main__':
    unittest.main()
