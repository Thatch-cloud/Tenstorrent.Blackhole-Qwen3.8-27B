from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch

from draft_attention import draft_attention_mask
from proposal_native_attention import POLICY, attention, numerical_difference, validate_mask
import proposal_native_attention_gate as gate


class ProposalNativeAttentionTests(unittest.TestCase):
    def fixture(self):
        operations = SimpleNamespace(bfloat16='bf16', TILE_LAYOUT='tile', DRAM_MEMORY_CONFIG='dram',
            MathFidelity=SimpleNamespace(HiFi4='hifi4'), WormholeComputeKernelConfig=lambda **arguments: arguments,
            SDPAProgramConfig=lambda **arguments: arguments,
            transformer=SimpleNamespace(scaled_dot_product_attention=Mock(return_value='output')))
        operands = [SimpleNamespace(shape=shape, dtype='bf16', layout='tile', memory_config=lambda: 'dram')
            for shape in ((1, 16, 32, 128), (1, 4, 64, 128), (1, 4, 64, 128), (1, 1, 32, 64))]
        return operations, operands

    def test_unmodified_native_configuration_and_explicit_mask_gate(self):
        operations, operands = self.fixture()
        for validated in (False, None, 1):
            with self.subTest(validated=validated), self.assertRaises(ValueError):
                attention(operations, *operands, mask_validated=validated)
        operations.transformer.scaled_dot_product_attention.assert_not_called()
        with patch('native_draft_sdpa.audit_active_kernel', side_effect=AssertionError('No global precision graft')):
            self.assertEqual(attention(operations, *operands, mask_validated=True), 'output')
        arguments = operations.transformer.scaled_dot_product_attention.call_args.kwargs
        self.assertIs(arguments['attn_mask'], operands[3])
        self.assertFalse(arguments['is_causal'])
        self.assertEqual(arguments['scale'], 128 ** -.5)
        self.assertEqual(arguments['program_config'], dict(compute_with_storage_grid_size=(8, 8),
            q_chunk_size=32, k_chunk_size=32, exp_approx_mode=False))
        self.assertEqual(arguments['compute_kernel_config'], dict(math_fidelity='hifi4',
            math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=False))

    def test_operand_layout_and_window_fail_before_dispatch(self):
        for field, value in (('layout', 'row_major'), ('dtype', 'fp32'), ('memory_config', lambda: 'l1'),
                             ('shape', (1, 4, 2112, 128))):
            operations, operands = self.fixture()
            setattr(operands[1], field, value)
            with self.subTest(field=field), self.assertRaises(ValueError):
                attention(operations, *operands, mask_validated=True)
            operations.transformer.scaled_dot_product_attention.assert_not_called()

    def test_mask_changes_are_validated_including_padding(self):
        for context in (31, 2048):
            mask = draft_attention_mask(context)
            validate_mask(mask)
            mask[..., :8, :5] = float('-inf')
            validate_mask(mask)
            for failure in ('nan', 'empty_live', 'padding', 'bias'):
                changed = mask.clone()
                if failure == 'nan':
                    changed[..., 0, 0] = float('nan')
                elif failure == 'empty_live':
                    changed[..., 0, :] = float('-inf')
                elif failure == 'padding':
                    changed[..., 8, :] = 0
                else:
                    changed[..., 0, 0] = 1
                with self.subTest(context=context, failure=failure), self.assertRaises(ValueError):
                    validate_mask(changed)

    def test_numerical_errors_are_reported_without_relabeling_accuracy(self):
        expected = torch.zeros((1, 16, 32, 128))
        current = torch.ones_like(expected, dtype=torch.bfloat16)
        difference = numerical_difference(current, expected)
        self.assertEqual(difference, dict(max_abs=1., mean_abs=1., rms=1., reference_max_abs=0., legacy_close=False))
        current[..., 8:, :] = float('nan')
        with self.assertRaises(ValueError):
            numerical_difference(current, expected)


class ProposalNativeAttentionGateTests(unittest.TestCase):
    def fixture(self, context=31):
        metrics = dict(max_abs=.05, mean_abs=.001, rms=.004, reference_max_abs=1., legacy_close=False)
        return dict(passed=True, closed_cleanly=True, backend='simulator', policy=POLICY, context=context, stage='complete',
            target_integrated=False, accuracy_qualified=False, packer_zero_graft=True,
            sources={name: 'a' * 64 for name in gate.SOURCES},
            native_sources={**{name: 'b' * 64 for name in gate.NATIVE_SOURCES}, **gate.ORIGINAL, gate.PACKER: gate.SIMULATOR_PACKER},
            native_sources_after={**{name: 'b' * 64 for name in gate.NATIVE_SOURCES}, **gate.ORIGINAL, gate.PACKER: gate.SIMULATOR_PACKER},
            fixture_sha256=gate.FIXTURE_SHA256 if context == 31 else None,
            eager_checks=[dict(pattern=pattern, chip=chip, finite_all_rows=True, numerical_difference=dict(metrics))
                for pattern in range(3) for chip in range(2)],
            replay_checks=[dict(repetition=repetition, pattern=pattern, chip=chip, exact_eager_all_rows=True,
                bindings_stable=True) for repetition, pattern in enumerate((0, 1, 2, 0)) for chip in range(2)],
            input_checks=[dict(phase=phase, pattern=pattern, tensor=tensor, chip=chip, unchanged=True)
                for phase, patterns in ((0, range(3)), (1, range(4))) for pattern in patterns
                for tensor in range(4) for chip in range(2)],
            negative_controls=[dict(chip=chip, stale_detected=True) for chip in range(2)],
            masked_input_checks=[dict(phase=phase, chip=chip, masked_changes_ignored=True)
                for phase in range(2) for chip in range(2)])

    def qualify(self, report):
        return gate.qualify(report, report['context'], report['sources'], report['native_sources'])

    def test_complete_gate_does_not_certify_accuracy_or_target_integration(self):
        for context in (31, 2048):
            report = self.fixture(context)
            result = self.qualify(report)
            self.assertTrue(result['passed'])
            self.assertFalse(result['accuracy_qualified'])
            self.assertFalse(result['target_integrated'])
            self.assertEqual(sum(len(report[name]) for name in ('eager_checks', 'replay_checks', 'input_checks',
                'negative_controls', 'masked_input_checks')), 76)
            restored = {**report['native_sources'], gate.PACKER: gate.ORIGINAL[gate.PACKER]}
            self.assertEqual(gate.qualify(report, context, report['sources'], restored), result)

    def test_every_check_matrix_is_required(self):
        for name in ('eager_checks', 'replay_checks', 'input_checks', 'negative_controls', 'masked_input_checks'):
            for failure in ('missing', 'duplicate', 'boolean_chip'):
                report = self.fixture()
                if failure == 'missing':
                    report[name].pop()
                elif failure == 'duplicate':
                    report[name].append(deepcopy(report[name][0]))
                else:
                    report[name][0]['chip'] = False
                with self.subTest(name=name, failure=failure), self.assertRaises(ValueError):
                    self.qualify(report)

    def test_failure_provenance_and_misleading_claims_are_rejected(self):
        for name, value in (('closed_cleanly', False), ('passed', False), ('error', 'failed'),
                ('backend', 'hardware'), ('policy', 'exact-target'), ('accuracy_qualified', True),
                ('target_integrated', True), ('context', True), ('fixture_sha256', None),
                ('native_sources_after', {}), ('stage', 'running'), ('packer_zero_graft', False)):
            report = self.fixture()
            report[name] = value
            with self.subTest(name=name), self.assertRaises(ValueError):
                self.qualify(report)
        report = self.fixture()
        report['native_sources'][gate.PACKER] = report['native_sources_after'][gate.PACKER] = 'c' * 64
        with self.assertRaises(ValueError):
            self.qualify(report)

    def test_numerical_differences_cannot_be_omitted_or_nonfinite(self):
        for name, value in (('max_abs', float('nan')), ('mean_abs', -1.), ('rms', True),
                ('reference_max_abs', float('inf')), ('legacy_close', 0), ('mean_abs', .9)):
            report = self.fixture()
            report['eager_checks'][0]['numerical_difference'][name] = value
            with self.subTest(name=name, value=value), self.assertRaises(ValueError):
                self.qualify(report)
        report['eager_checks'][0].pop('numerical_difference')
        with self.assertRaises(ValueError):
            self.qualify(report)

    def test_native_drift_and_graft_lock_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in gate.NATIVE_SOURCES:
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(name.encode())
            original = {name: hashlib.sha256(name.encode()).hexdigest() for name in gate.ORIGINAL}
            with patch.object(gate, 'ORIGINAL', original):
                self.assertEqual(set(gate.native_hashes(root)), set(gate.NATIVE_SOURCES))
                lock = root / gate.SDPA / 'device/kernels/compute/.qwen-precise-draft.lock'
                lock.touch()
                with self.assertRaises(ValueError):
                    gate.native_hashes(root)
                lock.unlink()
                (root / gate.PACKER).write_bytes(b'drift')
                with self.assertRaises(ValueError):
                    gate.native_hashes(root)

    def test_outer_exit_is_required_even_with_passing_json(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            report, status = root / 'report.json', root / 'exit-status'
            report.write_text(json.dumps(self.fixture()))
            status.write_text('124\n')
            with patch.object(sys, 'argv', ['gate', '--report', str(report), '--exit-status', str(status),
                    '--native-root', str(root), '--context', '31']), self.assertRaisesRegex(ValueError, 'outer'):
                gate.main()


if __name__ == '__main__':
    unittest.main()
