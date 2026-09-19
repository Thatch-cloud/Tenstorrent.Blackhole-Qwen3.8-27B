from pathlib import Path
import unittest

import torch

from dflash_attention_mask import draft_attention_mask
from dflash_t32_native_attention import validate_mask
from dflash_t32_native_stage import payloads


class T32StageTests(unittest.TestCase):
    def test_distinct_gate_and_full_query_probe_preserve_t16(self):
        directory = Path(__file__).parent
        original = (directory / 'dflash-t16-native-attention-probe.py').read_bytes()
        sources = payloads(directory)
        probe = sources['dflash-t32-native-attention-probe.py']
        gate = sources['dflash_t32_native_attention_gate.py']
        self.assertIn('block_rows=32', probe)
        self.assertNotIn('block_rows=16', probe)
        self.assertIn("report['block_rows'] != 32", gate)
        self.assertIn("'dflash_t16_native_attention.py'", gate)
        self.assertIn("'dflash_t32_native_attention.py'", gate)
        self.assertIn('/experiment/results/dflash-t32-', sources['simulator-suite.sh'])
        self.assertEqual(original, (directory / 'dflash-t16-native-attention-probe.py').read_bytes())

    def test_both_contexts_have_masked_keys_and_visible_live_queries(self):
        for history in (31, 2048):
            mask = draft_attention_mask(history, block_rows=32)
            mask[..., :, :3] = float('-inf')
            validate_mask(mask)
            self.assertTrue(torch.isneginf(mask[0, 0]).all(dim=0).any())
            self.assertTrue((mask == 0).any(dim=-1).all())


if __name__ == '__main__':
    unittest.main()
