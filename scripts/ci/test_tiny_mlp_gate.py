from copy import deepcopy
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from tiny_mlp_gate import (HARDWARE_TP_COMMON, NATIVE_SOURCES, ORIGINAL_PACKER, PACKER, SIMULATOR_PACKER,
    SIMULATOR_TP_COMMON, SOURCES, TP_COMMON, qualify, qualify_hardware)


class TinyMlpGateTests(unittest.TestCase):
    def fixture(self):
        sources = dict.fromkeys(SOURCES, 'source-sha')
        native = dict.fromkeys(NATIVE_SOURCES, 'native-sha')
        native[PACKER] = ORIGINAL_PACKER
        components = ('gate', 'up', 'hidden', 'partial')
        report = dict(passed=True, stage='complete', rows=8, packer_zero_graft=True, packer_l1_acc=True,
            sources=dict(sources), native_sources=dict(native, **{PACKER: SIMULATOR_PACKER}),
            eager_checks=[dict(pattern=pattern, chip=chip, component=component, exact=True)
                for pattern in range(2) for chip in range(2) for component in components],
            trace_checks=[dict(pattern=pattern, arm=arm, chip=chip, component=component, exact=True)
                for pattern in range(2) for arm in range(2) for chip in range(2) for component in components],
            negative_controls=[dict(chip=chip, stale_input_detected=True) for chip in range(2)])
        return report, sources, native

    def test_complete_unchanged_gate_qualifies_only_for_hardware_testing(self):
        self.assertEqual(qualify(*self.fixture()), dict(passed=True, eager_checks=16, trace_checks=32, negative_controls=2))

    def test_only_exact_audited_prefill_source_pair_is_accepted(self):
        report, sources, native = self.fixture()
        report['native_sources'][TP_COMMON] = SIMULATOR_TP_COMMON
        native[TP_COMMON] = HARDWARE_TP_COMMON
        self.assertIn('T8 MLP decode only', qualify(report, sources, native)['native_equivalence']['scope'])
        for changed in ('simulator', 'hardware', 'unrelated'):
            modified, current = deepcopy(report), dict(native)
            if changed == 'simulator':
                modified['native_sources'][TP_COMMON] = 'changed'
            elif changed == 'hardware':
                current[TP_COMMON] = 'changed'
            else:
                current[NATIVE_SOURCES[-1]] = 'changed'
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                qualify(modified, sources, current)

    def test_incomplete_failure_and_ungrafted_reports_reject(self):
        for key, value in (('passed', False), ('stage', 'weights_ready'), ('rows', 1),
                ('packer_zero_graft', False), ('packer_l1_acc', False)):
            report, sources, native = self.fixture()
            report[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                qualify(report, sources, native)

    def test_changed_sources_or_hardware_packer_reject(self):
        for changed in ('local', 'native', 'packer', 'missing', 'simulation'):
            report, sources, native = self.fixture()
            if changed == 'local':
                sources[SOURCES[0]] = 'changed'
            elif changed == 'native':
                native[NATIVE_SOURCES[0]] = 'changed'
            elif changed == 'packer':
                native[PACKER] = SIMULATOR_PACKER
            elif changed == 'missing':
                del sources[SOURCES[0]]
            else:
                report['native_sources'][PACKER] = ORIGINAL_PACKER
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                qualify(report, sources, native)

    def test_missing_duplicate_and_failed_checks_reject(self):
        for group in ('eager_checks', 'trace_checks', 'negative_controls'):
            for mutation in ('missing', 'duplicate', 'failed'):
                report, sources, native = self.fixture()
                checks = report[group]
                if mutation == 'missing':
                    checks.pop()
                elif mutation == 'duplicate':
                    checks[0] = deepcopy(checks[1])
                else:
                    checks[0]['stale_input_detected' if group == 'negative_controls' else 'exact'] = False
                with self.subTest(group=group, mutation=mutation), self.assertRaises(ValueError):
                    qualify(report, sources, native)

    def hardware_fixture(self):
        return dict(passed=True, stage='complete', rows=8, streams=1, layer=0, collective_links=4,
            seeds=[1659, 2670, 3781], repeats_per_sample=50, control_ms=2.0, candidate_ms=1.0,
            eligible_for_full_model_gate=True,
            eager_checks=[dict(pattern=pattern, chip=chip, exact=True) for pattern in range(3) for chip in range(2)],
            trace_checks=[dict(pattern=pattern, arm=arm, chip=chip, exact=True)
                for pattern in range(3) for arm in range(2) for chip in range(2)],
            blocks=[dict(pattern=pattern, block=block, samples_ms=[2.0, 1.0, 1.0, 2.0],
                control_ms=2.0, candidate_ms=1.0, ratio=2.0) for pattern in range(3) for block in range(3)])

    def test_complete_hardware_matrix_recomputes_from_all_samples(self):
        report = self.hardware_fixture()
        self.assertTrue(qualify_hardware(report)['eligible_for_full_model_gate'])
        report['candidate_ms'] = 2.0
        report['eligible_for_full_model_gate'] = False
        for block in report['blocks']:
            block.update(samples_ms=[2.0] * 4, candidate_ms=2.0, ratio=1.0)
        self.assertFalse(qualify_hardware(report)['eligible_for_full_model_gate'])

    def test_incomplete_duplicate_or_failed_hardware_comparisons_reject(self):
        for group in ('eager_checks', 'trace_checks', 'blocks'):
            for mutation in ('missing', 'duplicate'):
                report = self.hardware_fixture()
                if mutation == 'missing':
                    report[group].pop()
                else:
                    report[group][0] = deepcopy(report[group][1])
                with self.subTest(group=group, mutation=mutation), self.assertRaises(ValueError):
                    qualify_hardware(report)
        report = self.hardware_fixture()
        report['eager_checks'][0]['exact'] = False
        with self.assertRaises(ValueError):
            qualify_hardware(report)

    def test_invalid_raw_samples_or_invented_summaries_reject(self):
        for mutation in ('nan', 'negative', 'missing_sample', 'block_summary', 'total', 'eligibility', 'failure'):
            report = self.hardware_fixture()
            if mutation == 'nan':
                report['blocks'][0]['samples_ms'][0] = float('nan')
            elif mutation == 'negative':
                report['blocks'][0]['samples_ms'][0] = -1
            elif mutation == 'missing_sample':
                report['blocks'][0]['samples_ms'].pop()
            elif mutation == 'block_summary':
                report['blocks'][0]['control_ms'] = 99
            elif mutation == 'total':
                report['control_ms'] = 99
            elif mutation == 'eligibility':
                report['eligible_for_full_model_gate'] = False
            else:
                report['passed'] = False
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                qualify_hardware(report)

    def test_missing_hardware_artifact_cannot_pass_the_cli(self):
        with tempfile.TemporaryDirectory() as directory:
            result = subprocess.run([sys.executable, '-B', str(Path(__file__).with_name('tiny_mlp_gate.py')),
                '--hardware-result', str(Path(directory) / 'missing.json')], capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('FileNotFoundError', result.stderr)
