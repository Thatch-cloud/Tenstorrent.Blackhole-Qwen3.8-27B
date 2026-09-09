from copy import deepcopy
import unittest

from tiny_mlp_gate import NATIVE_SOURCES, ORIGINAL_PACKER, PACKER, SIMULATOR_PACKER, SOURCES, qualify


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
