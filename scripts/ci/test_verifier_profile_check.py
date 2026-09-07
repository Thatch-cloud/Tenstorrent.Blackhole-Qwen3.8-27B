import importlib.util
from pathlib import Path
import unittest


spec = importlib.util.spec_from_file_location('verifier_check', Path(__file__).with_name('check-verifier-profile.py'))
checker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(checker)


class VerifierProfileCheckTests(unittest.TestCase):
    def fixture(self):
        timings, rows, console = [], [], []
        for context_index, length in enumerate((4095, 16383)):
            records = []
            for repeat in range(3):
                for arm_index, arm in enumerate(('serial', 'control', 'batch')):
                    trace_id = context_index * 3 + arm_index + 1
                    label = f'qwen_verifier_t8_ctx{length}_{arm}_{repeat}'
                    records.append(dict(repeat=repeat, arm=arm, trace_id=trace_id, label=label, exact=True))
                    console.extend(['QWEN_VERIFIER_PROFILE_BEGIN ' + label, 'QWEN_VERIFIER_PROFILE_END ' + label])
                    for chip in ('0', '1'):
                        rows.append({'METAL TRACE ID': str(trace_id), 'METAL TRACE REPLAY SESSION ID': str(repeat + 1),
                                     'DEVICE KERNEL DURATION [ns]': '100', 'DEVICE ID': chip, 'OP CODE': 'matmul'})
            timings.append(dict(length=length, rows=8, exact=True, blocks=[], device_profile=dict(records=records)))
        return dict(passed=True, instrumented_timing=True, timings=timings), rows, '\n'.join(console)

    def test_both_contexts_arms_chips_and_replays_required(self):
        result = checker.analyze(*self.fixture())
        self.assertEqual(len(result['traces']), 6)
        self.assertTrue(all(len(trace['devices']) == 2 for trace in result['traces']))
        self.assertIn('not critical-path', result['scope'])

    def test_missing_chip_or_replay_fails(self):
        report, rows, console = self.fixture()
        for selected in (rows[:-1], [row for row in rows if row['DEVICE ID'] == '0']):
            with self.assertRaises(AssertionError):
                checker.analyze(report, selected, console)

    def test_bad_markers_or_dropped_data_fail(self):
        report, rows, console = self.fixture()
        for changed in (console.replace('QWEN_VERIFIER_PROFILE_END', 'missing', 1),
                        console.replace('\n', '\nmarkers were dropped\n', 1)):
            with self.assertRaises(AssertionError):
                checker.analyze(report, rows, changed)

    def test_bad_correctness_or_scope_fails(self):
        for field, value in (('rows', 16), ('exact', False), ('blocks', [dict(batch_ms=1)])):
            report, rows, console = self.fixture()
            report['timings'][0][field] = value
            with self.assertRaises(AssertionError):
                checker.analyze(report, rows, console)

    def test_aliased_trace_identifiers_fail(self):
        report, rows, console = self.fixture()
        for record in report['timings'][1]['device_profile']['records']:
            record['trace_id'] -= 3
        with self.assertRaisesRegex(AssertionError, 'Distinct stable'):
            checker.analyze(report, rows, console)
