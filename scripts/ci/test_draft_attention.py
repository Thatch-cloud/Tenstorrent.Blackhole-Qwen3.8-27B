import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import torch

from draft_attention import draft_attention_mask, draft_sdpa, composed_draft_attention


class DraftAttentionTests(unittest.TestCase):
    def operations_fixture(self):
        query = SimpleNamespace(shape=(1, 16, 32, 128), dtype='bf16')
        key = SimpleNamespace(shape=(1, 4, 64, 128), dtype='bf16')
        value = SimpleNamespace(shape=key.shape, dtype='bf16')
        mask = SimpleNamespace(shape=(1, 1, 32, 64), dtype='bf16')
        operations = SimpleNamespace(bfloat16='bf16', float32='fp32', DRAM_MEMORY_CONFIG='dram',
            MathFidelity=SimpleNamespace(HiFi4='hifi4'), WormholeComputeKernelConfig=Mock(), SDPAProgramConfig=Mock(),
            transformer=SimpleNamespace(scaled_dot_product_attention=Mock()), synchronize_device=Mock(), deallocate=Mock(),
            get_device_tensors=lambda tensor: [SimpleNamespace(buffer_address=lambda: id(tensor)),
                SimpleNamespace(buffer_address=lambda: id(tensor) + 1)])
        for name in ('repeat_interleave', 'transpose', 'matmul', 'multiply', 'typecast', 'add', 'softmax'):
            setattr(operations, name, Mock(side_effect=lambda *args, **kwargs: object()))
        return operations, query, key, value, mask

    def test_native_path_never_enables_causal_mask(self):
        operations, query, key, value, mask = self.operations_fixture()
        draft_sdpa(operations, query, key, value, mask)
        arguments = operations.transformer.scaled_dot_product_attention.call_args.kwargs
        self.assertIs(arguments['is_causal'], False)
        self.assertIs(arguments['attn_mask'], mask)

    def test_composed_path_keeps_inputs_and_output_owned_by_caller(self):
        operations, query, key, value, mask = self.operations_fixture()
        mesh = object()
        output = composed_draft_attention(operations, mesh, query, key, value, mask)
        freed = [call.args[0] for call in operations.deallocate.call_args_list]
        for protected in (query, key, value, mask, output):
            self.assertFalse(any(tensor is protected for tensor in freed))
        self.assertEqual(len(freed), 8)
        operations.synchronize_device.assert_called_once_with(mesh)
        self.assertEqual(operations.softmax.call_args.kwargs['numeric_stable'], True)
        self.assertEqual(operations.matmul.call_count, 2)

    def test_all_proposals_visible_but_padding_hidden(self):
        mask = draft_attention_mask(31)
        self.assertEqual(tuple(mask.shape), (1, 1, 32, 64))
        self.assertTrue((mask[..., :8, 31:39] == 0).all())
        self.assertTrue(torch.isneginf(mask[..., :8, 39:]).all())
        self.assertTrue((mask[..., 8:, :] == 0).sum(-1).eq(1).all())

    def test_sliding_boundary_advances_per_query(self):
        mask = draft_attention_mask(2048)[0, 0]
        for row in range(8):
            self.assertTrue(torch.isneginf(mask[row, :row + 1]).all())
            self.assertEqual(mask[row, row + 1].item(), 0)
            self.assertTrue((mask[row, 2048:2056] == 0).all())

    def test_empty_context_is_bidirectional_block(self):
        mask = draft_attention_mask(0)
        self.assertTrue((mask[..., :8, :8] == 0).all())
        self.assertTrue(torch.isneginf(mask[..., :8, 8:]).all())

    def test_invalid_geometry_is_rejected(self):
        for context in (-1, True, 262145):
            with self.assertRaises(ValueError):
                draft_attention_mask(context)
        with self.assertRaises(ValueError):
            draft_attention_mask(31, block_rows=2)
