import copy
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from dspark_attention import CONTEXTS, POLICY
from dspark_attention_gate import qualify
from native_draft_sdpa import KERNEL_DIRECTORY, SIGNATURE, SOURCE_HASHES


def fixture():
    return dict(passed=True, closed_cleanly=True, stage='complete', backend='simulator', accuracy_policy=POLICY,
        target_integrated=False, eligible_for_hardware=False, contexts=list(CONTEXTS), packer_compat=False,
        precise_native=False, kernel_audit=None, sources={'probe': 'sha'}, sources_after={'probe': 'sha'},
        native_sources={'runtime': 'sha'}, native_sources_after={'runtime': 'sha'},
        eager_checks=[dict(context=context, pattern=pattern, chip=chip, full_padded_close=True, failed_elements=0, max_abs=.001, valid_max_abs=.001)
            for context in CONTEXTS for pattern in range(5) for chip in range(2)],
        replay_checks=[dict(context=context, ordinal=ordinal, pattern=pattern, chip=chip, exact=True, bindings_stable=True)
            for context in CONTEXTS for ordinal, pattern in enumerate((0, 1, 2, 3, 4, 0)) for chip in range(2)],
        input_checks=[dict(context=context, phase=phase, ordinal=ordinal, tensor=tensor, chip=chip, exact=True)
            for context in CONTEXTS for phase, count in (('eager', 5), ('replay', 6))
            for ordinal in range(count) for tensor in range(4) for chip in range(2)],
        dependency_controls=[dict(context=context, control=control, chip=chip, passed=True)
            for context in CONTEXTS for control in ('padding', 'oldest_history', 'future_proposal') for chip in range(2)],
        stale_controls=[dict(context=context, chip=chip, missing_update_detected=True) for context in CONTEXTS for chip in range(2)])


def check(report, **kwargs):
    return qualify(report, sources={'probe': 'sha'}, native=kwargs.pop('native', {'runtime': 'sha'}),
        exit_status=kwargs.pop('exit_status', '0'), **kwargs)


class DSparkAttentionGateTests(unittest.TestCase):
    def test_complete_gate_does_not_qualify_target_or_hardware(self):
        result = check(fixture())
        self.assertEqual(result['checks'], 236)
        self.assertFalse(result['eligible_for_hardware'])
        self.assertFalse(result['target_integrated'])

    def test_every_coordinate_and_control_is_required(self):
        for name in ('eager_checks', 'replay_checks', 'input_checks', 'dependency_controls', 'stale_controls'):
            for duplicate in (False, True):
                report = fixture()
                if duplicate:
                    report[name].append(copy.deepcopy(report[name][0]))
                else:
                    report[name].pop()
                with self.assertRaises(ValueError):
                    check(report)
        for field, value in (('full_padded_close', False), ('max_abs', float('nan')), ('valid_max_abs', .1), ('chip', True),
                ('failed_elements', 1), ('failed_elements', False)):
            report = fixture()
            report['eager_checks'][0][field] = value
            with self.assertRaises(ValueError):
                check(report)

    def test_key_chunk_policies_cannot_relabel_previous_evidence_or_accuracy_failure(self):
        report = fixture()
        with self.assertRaises(ValueError):
            check(report, key_chunk_size=64)
        report['key_chunk_size'] = 64
        self.assertEqual(check(report, key_chunk_size=64)['key_chunk_size'], 64)
        with self.assertRaises(ValueError):
            check(report)
        for chunk in (True, 32., 16, 128):
            report['key_chunk_size'] = chunk
            with self.assertRaises(ValueError):
                check(report, key_chunk_size=chunk)
        report['key_chunk_size'] = 64
        report['eager_checks'][0].update(full_padded_close=False, failed_elements=1)
        with self.assertRaises(ValueError):
            check(report, key_chunk_size=64)

    def test_lifecycle_and_packer_scope_cannot_be_relabelled(self):
        for key, value in (('closed_cleanly', False), ('passed', False), ('backend', 'hardware'), ('stage', 'capture'),
                ('sources_after', {}), ('native_sources_after', {}), ('accuracy_policy', 'approximate'),
                ('eligible_for_hardware', True), ('contexts', [31, 2048]), ('packer_compat', True), ('precise_native', True)):
            report = fixture()
            report[key] = value
            with self.assertRaises(ValueError):
                check(report)
        with self.assertRaises(ValueError):
            check(fixture(), exit_status='124')
        report = fixture()
        report['packer_compat'] = True
        self.assertTrue(check(report, packer_compat=True)['passed'])

    def test_precise_graft_requires_explicit_matching_source_and_signature_audit(self):
        report = fixture()
        native = {'runtime': 'sha', **{f'{KERNEL_DIRECTORY}/{name}': f'patched-{name}' for name in SOURCE_HASHES}}
        report.update(precise_native=True, native_sources=native, native_sources_after=native,
            kernel_audit=dict(original=SOURCE_HASHES, patched={name: f'patched-{name}' for name in SOURCE_HASHES},
                signature={str(index): value for index, value in SIGNATURE.items()}, runtime_sources={'runtime': 'sha'}))
        self.assertTrue(check(report, native=native, precise_native=True)['passed'])
        with self.assertRaises(ValueError):
            check(report, native=native)
        report['kernel_audit']['signature']['1'] = 32
        with self.assertRaises(ValueError):
            check(report, native=native, precise_native=True)

    def test_probe_refuses_hardware_before_runtime_or_graft(self):
        environment = {key: value for key, value in os.environ.items()
            if key not in ('TT_METAL_SIMULATOR', 'TT_METAL_SLOW_DISPATCH_MODE')}
        environment.update(QWEN_HARDWARE_TESTS='1', QWEN_CARDS_ALLOCATED='1')
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'unexpected.json'
            for flags in ([], ['--precise-native'], ['--precise-native', '--key-chunk-size', '64']):
                result = subprocess.run([sys.executable, '-B', str(Path(__file__).with_name('dspark-attention-probe.py')),
                    '--output', str(output), *flags], env=environment, text=True, capture_output=True, timeout=30)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn('Simulator required', result.stderr)
                self.assertFalse(output.exists())


if __name__ == '__main__':
    unittest.main()
