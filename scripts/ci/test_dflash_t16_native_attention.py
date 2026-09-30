from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import torch

from dflash_attention_mask import draft_attention_mask
from dflash_t16_native_attention import attention, numerical_difference, validate_mask
import dflash_t16_native_attention_gate as gate
import test_proposal_native_attention as legacy_tests


class T16NativeAttentionTests(unittest.TestCase):
    def report_fixture(self):
        report = legacy_tests.ProposalNativeAttentionGateTests().fixture()
        report.update(policy=gate.POLICY, block_rows=16, fixture_sha256=None,
            sources={name: 'a' * 64 for name in gate.SOURCES},
            runtime_binaries=dict.fromkeys(gate.BINARIES, gate.BINARY_SHA256),
            runtime_binaries_after=dict.fromkeys(gate.BINARIES, gate.BINARY_SHA256))
        for name in ('native_sources', 'native_sources_after'):
            report[name][gate.FACTORY] = gate.COMBINED_FACTORY
        return report

    def qualify(self, report):
        return gate.qualify(report, 31, report['sources'], report['native_sources'])

    def test_gate_rejects_t8_and_incomplete_evidence(self):
        report = self.report_fixture()
        result = self.qualify(report)
        self.assertFalse(result['accuracy_qualified'])
        self.assertFalse(result['target_integrated'])
        for name, value in (('block_rows', 8), ('block_rows', True),
                ('fixture_sha256', 'a' * 64), ('closed_cleanly', False), ('passed', False),
                ('target_integrated', True), ('accuracy_qualified', True)):
            report = self.report_fixture()
            report[name] = value
            with self.subTest(name=name, value=value), self.assertRaises(ValueError):
                self.qualify(report)
        for name in ('runtime_binaries', 'runtime_binaries_after'):
            report = self.report_fixture()
            report[name] = {}
            with self.subTest(name=name), self.assertRaises(ValueError):
                self.qualify(report)
        for name in ('eager_checks', 'replay_checks', 'input_checks', 'negative_controls', 'masked_input_checks'):
            report = self.report_fixture()
            report[name].pop()
            with self.subTest(name=name), self.assertRaises(ValueError):
                self.qualify(report)

    def test_all_live_rows_and_padding_are_checked(self):
        for context in (0, 17, 31, 256, 2048):
            mask = draft_attention_mask(context, block_rows=16)
            validate_mask(mask)
            for row in range(32):
                changed = mask.clone()
                changed[..., row, :] = float('-inf') if row < 16 else 0
                with self.subTest(context=context, row=row), self.assertRaises(ValueError):
                    validate_mask(changed)
        for invalid in (float('nan'), float('inf'), 1):
            mask = draft_attention_mask(31, block_rows=16)
            mask[..., 15, 0] = invalid
            with self.assertRaises(ValueError):
                validate_mask(mask)

    def test_diagnostic_includes_second_half_and_checks_padded_finiteness(self):
        reference = torch.zeros((1, 16, 32, 128))
        actual = reference.bfloat16()
        actual[..., 15, :] = 1
        report = numerical_difference(actual, reference)
        self.assertFalse(report['legacy_close'])
        self.assertEqual(report['max_abs'], 1)
        self.assertEqual(report['mean_abs'], 1 / 16)
        actual[..., 31, :] = float('nan')
        with self.assertRaises(ValueError):
            numerical_difference(actual, reference)

    def test_native_configuration_and_fail_closed_dispatch(self):
        operations = SimpleNamespace(bfloat16='bf16', TILE_LAYOUT='tile', DRAM_MEMORY_CONFIG='dram',
            MathFidelity=SimpleNamespace(HiFi4='hifi4'), WormholeComputeKernelConfig=lambda **args: args,
            SDPAProgramConfig=lambda **args: args,
            transformer=SimpleNamespace(scaled_dot_product_attention=Mock(return_value='output')))
        operands = [SimpleNamespace(shape=shape, dtype='bf16', layout='tile', memory_config=lambda: 'dram')
            for shape in ((1, 16, 32, 128), (1, 4, 2080, 128), (1, 4, 2080, 128), (1, 1, 32, 2080))]
        for validated in (False, None, 1):
            with self.assertRaises(ValueError):
                attention(operations, *operands, mask_validated=validated)
        operations.transformer.scaled_dot_product_attention.assert_not_called()
        self.assertEqual(attention(operations, *operands, mask_validated=True), 'output')
        arguments = operations.transformer.scaled_dot_product_attention.call_args.kwargs
        self.assertFalse(arguments['is_causal'])
        self.assertTrue(arguments['compute_kernel_config']['fp32_dest_acc_en'])
        self.assertFalse(arguments['program_config']['exp_approx_mode'])
        operands[0].layout = 'row_major'
        with self.assertRaises(ValueError):
            attention(operations, *operands, mask_validated=True)
        self.assertEqual(operations.transformer.scaled_dot_product_attention.call_count, 1)

    def test_masked_keys_cannot_influence_cpu_reference(self):
        generator = torch.Generator().manual_seed(3827)
        mask = draft_attention_mask(31, block_rows=16)
        mask[..., :16, :5] = float('-inf')
        validate_mask(mask)
        query = torch.randn((1, 16, 32, 128), generator=generator)
        key = torch.randn((1, 4, 64, 128), generator=generator)
        value = torch.randn((1, 4, 64, 128), generator=generator)

        def reference(keys, values):
            return torch.nn.functional.scaled_dot_product_attention(query,
                keys.repeat_interleave(4, dim=1), values.repeat_interleave(4, dim=1),
                attn_mask=mask.float(), is_causal=False)

        expected = reference(key, value)
        blocked = torch.isneginf(mask[0, 0]).all(dim=0)
        self.assertTrue(blocked.any())
        key[:, :, blocked, :] = 100
        value[:, :, blocked, :] = -100
        self.assertTrue(torch.equal(reference(key, value), expected))
        value[:, :, 31, :] += 10
        self.assertFalse(torch.equal(reference(key, value), expected))


if __name__ == '__main__':
    unittest.main()
