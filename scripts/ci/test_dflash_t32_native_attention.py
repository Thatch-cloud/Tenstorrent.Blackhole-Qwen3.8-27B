import os
import unittest
from unittest.mock import patch

import torch

from dflash_attention_mask import draft_attention_mask
from dflash_t32_native_attention import attention, numerical_difference, validate_mask


class T32NativeAttentionTests(unittest.TestCase):
    def test_every_live_row_is_checked(self):
        for history in (31, 2048):
            mask = draft_attention_mask(history, block_rows=32)
            validate_mask(mask)
            for row in (0, 15, 16, 31):
                invalid = mask.clone()
                invalid[..., row, :] = float('-inf')
                with self.assertRaises(ValueError):
                    validate_mask(invalid)

    def test_numerical_diagnostics_include_last_query(self):
        reference = torch.zeros((1, 16, 32, 128), dtype=torch.bfloat16)
        actual = reference.clone()
        actual[..., 31, 127] = 1
        result = numerical_difference(actual, reference)
        self.assertEqual(result['max_abs'], 1)
        self.assertFalse(result['legacy_close'])

    def test_hardware_rejected_before_native_call(self):
        for environment in ({}, dict(QWEN_SIM_ONLY='1', TT_METAL_SIMULATOR='sim', QWEN_CARDS_ALLOCATED='1')):
            with patch.dict(os.environ, environment, clear=True), \
                    patch('dflash_t32_native_attention.native_attention') as native:
                with self.assertRaises(ValueError):
                    attention(None, None, None, None, None)
                native.assert_not_called()


if __name__ == '__main__':
    unittest.main()
