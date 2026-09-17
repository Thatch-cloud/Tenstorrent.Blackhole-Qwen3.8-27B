import hashlib
import json
from pathlib import Path
import unittest

import torch

from dflash_attention_mask import draft_attention_mask
from draft_attention import draft_attention_mask as original_mask


class DFlashMaskTests(unittest.TestCase):
    def test_t16_matches_independent_visibility_formula(self):
        for context in (0, 17, 31, 32, 170, 2048, 4096):
            for multiple in (32, 128):
                with self.subTest(context=context, multiple=multiple):
                    actual = draft_attention_mask(context, 16, key_multiple=multiple)
                    keys = ((context + 16 + multiple - 1) // multiple) * multiple
                    expected = torch.full((1, 1, 32, keys), float('-inf'), dtype=torch.bfloat16)
                    for row in range(16):
                        expected[0, 0, row, max(0, context + row - 2047):context] = 0
                        expected[0, 0, row, context:context + 16] = 0
                    expected[0, 0, 16:, context] = 0
                    self.assertTrue(torch.equal(actual, expected))

    def test_old_widths_and_invalid_inputs_keep_original_contract(self):
        for rows in (1, 8, 32):
            self.assertTrue(torch.equal(draft_attention_mask(170, rows), original_mask(170, rows)))
        for rows in (True, 16.0, 17):
            with self.assertRaises(ValueError):
                draft_attention_mask(170, rows)

    def test_shared_attention_matches_original_simulator_source_pin(self):
        directory = Path(__file__).parent
        report = json.loads((directory / 'dspark-layer-mesh-simulator.json').read_text())
        self.assertEqual(hashlib.sha256((directory / 'draft_attention.py').read_bytes()).hexdigest(),
                         report['sources']['draft_attention.py'])
