import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

import torch

from dspark_attention import CONTEXTS, execute, fixtures, full_mask, numerical_difference, reference, validate_mask


class DSparkAttentionTests(unittest.TestCase):
    def test_accuracy_failure_is_recorded_without_relaxing_threshold(self):
        actual = torch.zeros(1, 16, 32, 128, dtype=torch.bfloat16)
        expected = actual.float()
        self.assertTrue(numerical_difference(actual, expected)['full_padded_close'])
        actual[0, 0, 31, 127] = .02
        result = numerical_difference(actual, expected)
        self.assertFalse(result['full_padded_close'])
        self.assertEqual(result['failed_elements'], 1)
        self.assertEqual(result['valid_max_abs'], 0)
        actual[0, 0, 0, 0] = float('nan')
        with self.assertRaises(ValueError):
            numerical_difference(actual, expected)

    def test_full_context_and_all_future_proposals_visible_with_finite_padding_rows(self):
        for context in (1, 31, 4096, 8192, 262137):
            mask = full_mask(context)
            validate_mask(mask, context)
            self.assertTrue((mask[..., :7, :context + 7] == 0).all())
            self.assertTrue(torch.isneginf(mask[..., :7, context + 7:]).all())
            self.assertTrue((torch.isfinite(mask[..., 7:, :]).sum(-1) == 1).all())
            self.assertTrue((mask[..., 7:, context] == 0).all())

    def test_sliding_causal_partial_and_invalid_masks_reject(self):
        for context in (True, 0, -1, 262138, 3.5):
            with self.assertRaises(ValueError):
                full_mask(context)
        for row, column, value in ((0, 0, float('-inf')), (0, 4102, float('-inf')),
                (0, 4103, 0), (7, 4096, float('-inf')), (7, 0, 0), (0, 0, float('nan'))):
            mask = full_mask(4096)
            mask[0, 0, row, column] = value
            with self.assertRaises(ValueError):
                validate_mask(mask, 4096)
        with self.assertRaises(ValueError):
            validate_mask(full_mask(31).float(), 31)

    def test_dispatch_retains_full_key_extent_and_is_noncausal(self):
        operations = SimpleNamespace(bfloat16='bf16', TILE_LAYOUT='tile', DRAM_MEMORY_CONFIG='dram',
            MathFidelity=SimpleNamespace(HiFi4='hifi4'), WormholeComputeKernelConfig=MagicMock(),
            SDPAProgramConfig=MagicMock(), transformer=SimpleNamespace(scaled_dot_product_attention=MagicMock()))
        tensors = [SimpleNamespace(shape=shape, dtype='bf16', layout='tile', memory_config=lambda: 'dram')
            for shape in ((1, 16, 32, 128), (1, 4, 4128, 128), (1, 4, 4128, 128), (1, 1, 32, 4128))]
        with self.assertRaises(ValueError):
            execute(operations, *tensors, context_rows=4096)
        result = execute(operations, *tensors, context_rows=4096, mask_validated=True)
        call = operations.transformer.scaled_dot_product_attention.call_args
        self.assertFalse(call.kwargs['is_causal'])
        self.assertIs(call.kwargs['attn_mask'], tensors[3])
        self.assertIs(result, operations.transformer.scaled_dot_product_attention.return_value)
        with self.assertRaises(ValueError):
            execute(operations, *tensors, context_rows=31, mask_validated=True)

    def test_chunk64_preserves_every_existing_operand_and_masks_only_extra_padding(self):
        for context in CONTEXTS:
            original = fixtures(context)
            candidate = fixtures(context, key_multiple=64)
            for pattern, (before, after) in enumerate(zip(original, candidate, strict=True)):
                old_extent = before[1].shape[-2]
                self.assertTrue(torch.equal(before[0], after[0]))
                for index in (1, 2):
                    self.assertTrue(torch.equal(before[index], after[index][..., :old_extent, :]))
                    self.assertTrue(torch.all(after[index][..., old_extent:, :] == (17 if pattern == 2 else 0)))
                self.assertTrue(torch.equal(before[3], after[3][..., :old_extent]))
                self.assertTrue(torch.isneginf(after[3][..., old_extent:]).all())
                validate_mask(after[3], context, key_multiple=64)
                for chip in range(2):
                    torch.testing.assert_close(reference(after, chip), reference(before, chip), rtol=1e-6, atol=1e-6)
        mask = full_mask(4096, key_multiple=64)
        self.assertEqual(mask.shape[-1], 4160)
        mask[0, 0, 0, -1] = 0
        with self.assertRaises(ValueError):
            validate_mask(mask, 4096, key_multiple=64)

    def test_chunk64_dispatch_preserves_precision_and_rejects_unaligned_operands(self):
        operations = SimpleNamespace(bfloat16='bf16', TILE_LAYOUT='tile', DRAM_MEMORY_CONFIG='dram',
            MathFidelity=SimpleNamespace(HiFi4='hifi4'), WormholeComputeKernelConfig=MagicMock(),
            SDPAProgramConfig=MagicMock(), transformer=SimpleNamespace(scaled_dot_product_attention=MagicMock()))
        tensors = [SimpleNamespace(shape=shape, dtype='bf16', layout='tile', memory_config=lambda: 'dram')
            for shape in ((1, 16, 32, 128), (1, 4, 4160, 128), (1, 4, 4160, 128), (1, 1, 32, 4160))]
        execute(operations, *tensors, context_rows=4096, mask_validated=True, key_chunk_size=64)
        program = operations.SDPAProgramConfig.call_args.kwargs
        compute = operations.WormholeComputeKernelConfig.call_args.kwargs
        self.assertEqual(program['k_chunk_size'], 64)
        self.assertFalse(program['exp_approx_mode'])
        self.assertEqual(compute['math_fidelity'], 'hifi4')
        self.assertTrue(compute['fp32_dest_acc_en'])
        self.assertFalse(operations.transformer.scaled_dot_product_attention.call_args.kwargs['is_causal'])
        with self.assertRaises(ValueError):
            execute(operations, *tensors, context_rows=4096, mask_validated=True, key_chunk_size=32)

    def test_undeclared_key_chunk_sizes_reject(self):
        for chunk in (True, 32., 16, 128):
            for operation in (full_mask, fixtures):
                with self.assertRaises(ValueError):
                    operation(31, key_multiple=chunk)
            with self.assertRaises(ValueError):
                execute(None, None, None, None, None, context_rows=31, mask_validated=True, key_chunk_size=chunk)

    def test_fixture_controls_detect_windowing_causality_and_masked_key_reads(self):
        for context in CONTEXTS:
            patterns = fixtures(context)
            for pattern in patterns:
                validate_mask(pattern[3], context)
            for chip in range(2):
                outputs = [reference(values, chip) for values in patterns]
                self.assertTrue(all(torch.isfinite(value).all() for value in outputs))
                self.assertTrue(torch.equal(outputs[0], outputs[2]))
                for changed in (1, 3, 4):
                    self.assertFalse(torch.equal(outputs[0][..., :1, :], outputs[changed][..., :1, :]))


if __name__ == '__main__':
    unittest.main()
