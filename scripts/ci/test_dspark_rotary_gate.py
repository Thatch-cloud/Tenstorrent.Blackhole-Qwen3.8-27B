import copy
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from dspark_intake import FILES
from dspark_rotary_device import CASES, POLICY
from dspark_rotary_gate import qualify


def fixture():
    return dict(passed=True, closed_cleanly=True, stage='complete', backend='simulator', target_integrated=False,
        eligible_for_hardware=False, cases=[list(case) for case in CASES], accuracy_policy=POLICY,
        config_sha256=FILES['config.json'][1], cpu_report_sha256='cpu-comparison',
        sources={'probe': 'sha'}, sources_after={'probe': 'sha'}, native_sources={'binary': 'sha'}, native_sources_after={'binary': 'sha'},
        eager_checks=[dict(case=case, pattern=pattern, chip=chip, full_padded_close=True, cpu_bitwise_exact=False,
            max_abs=.001, valid_max_abs=.0005) for case in range(3) for pattern in range(4) for chip in range(2)],
        replay_checks=[dict(case=case, repetition=repetition, pattern=pattern, chip=chip, exact=True, bindings_stable=True)
            for case in range(3) for repetition, pattern in enumerate((0, 1, 2, 3, 0)) for chip in range(2)],
        input_checks=[dict(case=case, phase=phase, ordinal=ordinal, tensor=tensor, chip=chip, exact=True)
            for case in range(3) for phase, count in (('eager', 4), ('replay', 5))
            for ordinal in range(count) for tensor in range(3) for chip in range(2)],
        dependency_controls=[dict(case=case, control=control, chip=chip,
            **({'detected': True} if control == 'positions' else {'isolated': True}))
            for case in range(3) for control in ('positions', 'padding') for chip in range(2)],
        stale_controls=[dict(case=case, chip=chip, missing_update_detected=True) for case in range(3) for chip in range(2)])


def check(report, status='0'):
    return qualify(report, sources={'probe': 'sha'}, native={'binary': 'sha'}, cpu_report_sha256='cpu-comparison', exit_status=status)


class DSparkRotaryGateTests(unittest.TestCase):
    def test_complete_primitive_gate_does_not_certify_hardware_or_exact_cpu_arithmetic(self):
        result = check(fixture())
        self.assertEqual(result['checks'], 234)
        self.assertEqual(result['bitwise_cpu_checks'], 0)
        self.assertFalse(result['eligible_for_hardware'])
        self.assertFalse(result['target_integrated'])

    def test_every_coordinate_is_required_without_duplicates(self):
        for name in ('eager_checks', 'replay_checks', 'input_checks', 'dependency_controls', 'stale_controls'):
            for duplicate in (False, True):
                report = fixture()
                if duplicate:
                    report[name].append(copy.deepcopy(report[name][0]))
                else:
                    report[name].pop()
                with self.assertRaises(ValueError):
                    check(report)

    def test_report_lifecycle_policy_and_prerequisites_fail_closed(self):
        for key, value in (('closed_cleanly', False), ('passed', False), ('backend', 'hardware'), ('stage', 'capture'),
                ('sources_after', {}), ('native_sources_after', {}), ('accuracy_policy', 'exact'),
                ('eligible_for_hardware', True), ('cpu_report_sha256', 'different'), ('config_sha256', 'different'), ('cases', [])):
            report = fixture()
            report[key] = value
            with self.assertRaises(ValueError):
                check(report)
        with self.assertRaises(ValueError):
            check(fixture(), '124')

    def test_inconsistent_diagnostics_and_failed_negative_controls_reject(self):
        for field, value in (('max_abs', float('nan')), ('valid_max_abs', .1), ('cpu_bitwise_exact', True),
                ('chip', False), ('full_padded_close', False)):
            report = fixture()
            report['eager_checks'][0][field] = value
            with self.assertRaises(ValueError):
                check(report)
        for entry_index, field in ((0, 'detected'), (2, 'isolated')):
            report = fixture()
            report['dependency_controls'][entry_index][field] = False
            with self.assertRaises(ValueError):
                check(report)

    def test_probe_refuses_hardware_before_loading_config_or_runtime(self):
        environment = {key: value for key, value in os.environ.items()
            if key not in ('TT_METAL_SIMULATOR', 'TT_METAL_SLOW_DISPATCH_MODE')}
        environment.update(QWEN_HARDWARE_TESTS='1', QWEN_CARDS_ALLOCATED='1')
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'unexpected.json'
            result = subprocess.run([sys.executable, '-B', str(Path(__file__).with_name('dspark-rotary-probe.py')),
                '--config', str(Path(directory) / 'not-loaded'), '--output', str(output)],
                env=environment, text=True, capture_output=True, timeout=30)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('Simulator required', result.stderr)
            self.assertFalse(output.exists())


if __name__ == '__main__':
    unittest.main()
