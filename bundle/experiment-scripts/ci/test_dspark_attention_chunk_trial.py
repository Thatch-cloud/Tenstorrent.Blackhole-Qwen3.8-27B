"""Validate padding, ownership and dispatch without running a device."""

import unittest
from unittest.mock import Mock, patch

import torch

import dspark_attention_chunk_trial as candidate


class ChunkTrialTests(unittest.TestCase):
    def test_only_poisoned_masked_tail_is_added(self):
        query = torch.zeros(1, 16, 32, 128, dtype=torch.bfloat16)
        key = torch.full((1, 4, 8512, 128), 2., dtype=torch.bfloat16)
        value = torch.full_like(key, 3.)
        mask = torch.zeros(1, 1, 32, 8512, dtype=torch.bfloat16)
        operations = Mock()

        def pad(tensor, padding, fill):
            return torch.nn.functional.pad(tensor, tuple(number for pair in reversed(padding) for number in pair), value=fill)

        operations.pad.side_effect = pad
        owned = []
        with patch.object(candidate, 'validate_inputs') as validate:
            result = candidate.execute(operations, object(), query, key, value, mask,
                owned, context_rows=8448, proposals=15, mask_validated=True)
        validate.assert_called_once_with(operations, query, key, value, mask, 8448, 15, True)
        call = operations.transformer.scaled_dot_product_attention.call_args
        self.assertIs(call.args[0], query)
        for actual, original, poison in ((call.args[1], key, 8192.), (call.args[2], value, -8192.)):
            self.assertTrue(torch.equal(actual[:, :, :8512], original))
            self.assertEqual(actual.shape[2], 8704)
            self.assertTrue(bool((actual[:, :, 8512:] == poison).all()))
        padded_mask = call.kwargs['attn_mask']
        self.assertTrue(torch.equal(padded_mask[:, :, :, :8512], mask))
        self.assertTrue(bool(torch.isneginf(padded_mask[:, :, :, 8512:]).all()))
        self.assertEqual(operations.SDPAProgramConfig.call_args.kwargs['k_chunk_size'], 256)
        self.assertEqual(len(owned), 4)
        self.assertIs(owned[-1], result)

    def test_other_geometry_rejected_before_allocation(self):
        operations = Mock()
        with self.assertRaises(ValueError):
            candidate.execute(operations, object(), None, None, None, None, [],
                context_rows=4096, proposals=15, mask_validated=True)
        operations.pad.assert_not_called()


if __name__ == '__main__':
    unittest.main()
