import unittest

import torch

from dspark_t32_inputs import validate_fixed_mask
from test_dspark_native_fixed_probe import load


class T32DraftProbeTests(unittest.TestCase):
    def test_fixture_masks_all_31_queries_and_excludes_unused_row(self):
        probe = load('dspark-t32-draft-attention-probe')
        self.assertEqual(probe.PROPOSALS, 31)
        patterns = probe.fixtures()
        for position, values in zip(probe.POSITIONS, patterns, strict=True):
            validate_fixed_mask(values['mask'], position, probe.CAPACITY, 31)
            self.assertTrue(torch.all(values['mask'][:, :, :31, probe.CAPACITY + 30] == 0))
            self.assertTrue(torch.isneginf(values['mask'][:, :, :31, probe.CAPACITY + 31:]).all())
            corrupted = values['mask'].clone()
            corrupted[:, :, 30, position] = 0
            with self.assertRaises(ValueError):
                validate_fixed_mask(corrupted, position, probe.CAPACITY, 31)
        hashes = probe.source_hashes()
        self.assertIn('dspark_t32_inputs.py', hashes)
        self.assertIn('dspark_t32_attention.py', hashes)
        self.assertIn('dspark-t32-draft-attention-probe.py', hashes)


if __name__ == '__main__':
    unittest.main()
