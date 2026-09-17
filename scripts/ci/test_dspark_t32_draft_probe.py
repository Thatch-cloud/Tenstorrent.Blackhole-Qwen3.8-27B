import unittest

import torch

from dspark_t32_inputs import validate_fixed_mask
from test_dspark_native_fixed_probe import load


class T32DraftProbeTests(unittest.TestCase):
    def test_numerical_diagnostics_locate_live_and_padding_errors(self):
        probe = load('dspark-t32-draft-attention-probe')
        golden = torch.zeros(1, 16, 32, 128)
        actual = golden.bfloat16()
        actual[0, 2, 30, 3] = .25
        actual[0, 1, 31, 7] = .5
        result = probe.numerical_details(actual, golden)
        self.assertEqual(sum(result['failed_by_row']), 2)
        self.assertEqual(result['failed_by_row'][30:], [1, 1])
        self.assertEqual({tuple(entry['index']) for entry in result['failures']},
                         {(0, 2, 30, 3), (0, 1, 31, 7)})
        self.assertEqual(result['nonfinite'], 0)
        self.assertTrue(all(entry['absolute_error'] > entry['allowed_error'] for entry in result['failures']))

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
