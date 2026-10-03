from contextlib import redirect_stdout
from copy import deepcopy
import io
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

from request_verifier_profile import PROFILER_FLAGS
from tensix_mlp_hardware_gate import qualify_hardware
from tensix_mlp_profile import MlpTraceProfile, require_profile_mode
from tensix_mlp_profile_report import attribute, validate_profile
import test_tensix_mlp_hardware_gate


def fixture():
    operations = SimpleNamespace(synchronize_device=Mock(), ReadDeviceProfiler=Mock(), execute_trace=Mock())
    observer = MlpTraceProfile(operations, object(), [23, 24], signpost=Mock())
    with redirect_stdout(io.StringIO()) as output:
        for pattern in range(3):
            for arm in range(2):
                observer.replay(arm, pattern, 'audit')
                if pattern == 0:
                    observer.replay(arm, pattern, 'stale')
            for sample, arm in enumerate((0, 1, 1, 0)):
                observer.replay(arm, pattern, 'measurement', sample)
    report = test_tensix_mlp_hardware_gate.TensixMlpHardwareGateTests().fixture()
    report.update(instrumented_timing=True, correctness_only=True, repeats_per_sample=1,
        all_samples_retained=False, eligible_for_full_model_gate=False, blocks=[], timed_checks=[],
        profile_checks=[dict(pattern=pattern, sample=sample, chip=chip, exact=True)
            for pattern in range(3) for sample in range(4) for chip in range(2)], profile=observer.summary())
    del report['control_ms'], report['candidate_ms']
    return operations, report, output.getvalue()


def device_rows(records):
    rows = []
    for record in records:
        for chip in ('0', '1'):
            for ordinal in range(2):
                start = 10000 + record['trace_ordinal'] * 1000 + ordinal * 100
                rows.append({'METAL TRACE ID': str(record['trace_id']),
                    'METAL TRACE REPLAY SESSION ID': str(record['trace_ordinal'] + 1), 'DEVICE ID': chip,
                    'GLOBAL CALL COUNT': str(record['trace_id'] * 10 + ordinal), 'OP NAME': f'op{ordinal}',
                    'CORE COUNT': '68', 'DEVICE KERNEL START CYCLE': str(start),
                    'DEVICE KERNEL END CYCLE': str(start + 50), 'DEVICE KERNEL DURATION [ns]': '50',
                    'DEVICE BRISC KERNEL DURATION [ns]': '45', 'DEVICE TRISC0 KERNEL DURATION [ns]': '42'})
    return rows


class MlpProfileTests(unittest.TestCase):
    def test_t16_records_all_eighteen_replays_without_changing_t8(self):
        operations = SimpleNamespace(synchronize_device=Mock(), ReadDeviceProfiler=Mock(), execute_trace=Mock())
        observer = MlpTraceProfile(operations, object(), [23, 24], signpost=Mock(), rows=16)
        with redirect_stdout(io.StringIO()):
            for pattern in range(3):
                for arm in range(2):
                    observer.replay(arm, pattern, 'audit')
                for sample, arm in enumerate((0, 1, 1, 0)):
                    observer.replay(arm, pattern, 'measurement', sample)
        report = observer.summary()
        self.assertEqual(report['trace_counts'], [9, 9])
        self.assertTrue(all(record['rows'] == 16 for record in report['records']))
        self.assertEqual(operations.ReadDeviceProfiler.call_count, 36)
        for rows in (True, 0, 32):
            with self.assertRaises(ValueError):
                MlpTraceProfile(operations, object(), [23, 24], signpost=Mock(), rows=rows)

    def test_profiler_configuration_cannot_contaminate_performance_mode(self):
        require_profile_mode({}, False)
        require_profile_mode({name: '0' for name in PROFILER_FLAGS}, False)
        require_profile_mode({name: '1' for name in PROFILER_FLAGS}, True)
        for name in PROFILER_FLAGS:
            with self.subTest(name=name), self.assertRaises(ValueError):
                require_profile_mode({name: '1'}, False)
            enabled = {flag: '1' for flag in PROFILER_FLAGS}
            del enabled[name]
            with self.assertRaises(ValueError):
                require_profile_mode(enabled, True)

    def test_all_twenty_real_replays_and_outer_dumps_are_recorded(self):
        operations, report, console = fixture()
        self.assertEqual(len(validate_profile(report, console)), 20)
        self.assertEqual(operations.execute_trace.call_count, 20)
        self.assertEqual(operations.ReadDeviceProfiler.call_count, 40)
        self.assertTrue(all(call.kwargs == dict(cq_id=0, blocking=True) for call in operations.execute_trace.call_args_list))
        with self.assertRaises(ValueError):
            qualify_hardware(report)

    def test_failed_trace_has_no_success_record_and_cannot_resume(self):
        operations = SimpleNamespace(synchronize_device=Mock(), ReadDeviceProfiler=Mock(),
            execute_trace=Mock(side_effect=RuntimeError('device failure')))
        observer = MlpTraceProfile(operations, object(), [23, 24], signpost=Mock())
        with redirect_stdout(io.StringIO()) as output:
            with self.assertRaises(RuntimeError):
                observer.replay(0, 0, 'audit')
        self.assertEqual(observer.records, [])
        self.assertIn('QWEN_MLP_PROFILE_END', output.getvalue())
        with self.assertRaises(ValueError):
            observer.replay(0, 0, 'audit')
        with self.assertRaises(ValueError):
            observer.summary()

    def test_benchmark_gate_rejects_instrumented_success(self):
        for field in ('instrumented_timing', 'correctness_only'):
            report = test_tensix_mlp_hardware_gate.TensixMlpHardwareGateTests().fixture()
            report[field] = True
            with self.subTest(field=field), self.assertRaises(ValueError):
                qualify_hardware(report)

    def test_missing_reordered_changed_or_promoted_profile_fails(self):
        unused_operations, original, console = fixture()
        for failure in ('missing', 'duplicate', 'order', 'ordinal', 'trace', 'promotion', 'timing', 'output', 'raw', 'native'):
            report = deepcopy(original)
            records = report['profile']['records']
            if failure == 'missing':
                records.pop()
            elif failure == 'duplicate':
                records[1] = records[0]
            elif failure == 'order':
                records[0], records[1] = records[1], records[0]
            elif failure == 'ordinal':
                records[0]['trace_ordinal'] = False
            elif failure == 'trace':
                report['profile']['trace_ids'] = [23, 23]
            elif failure == 'promotion':
                report['eligible_for_full_model_gate'] = True
            elif failure == 'timing':
                report['control_ms'] = 0.3
            elif failure == 'output':
                report['profile_checks'][0]['exact'] = False
            elif failure == 'raw':
                report['input_checks'].pop()
            else:
                report['native_weights_after'] = {}
            with self.subTest(failure=failure), self.assertRaises(ValueError):
                validate_profile(report, console)
        for changed in (console.replace('QWEN_MLP_PROFILE_END', 'missing', 1),
                console.replace('QWEN_MLP_PROFILE_END', 'markers were dropped\nQWEN_MLP_PROFILE_END', 1)):
            with self.assertRaises(ValueError):
                validate_profile(original, changed)

    def test_device_attribution_keeps_both_chips_and_only_measured_operations(self):
        unused_operations, report, console = fixture()
        records = validate_profile(report, console)
        result = attribute(records, device_rows(records))
        self.assertEqual({(entry['arm'], entry['device']) for entry in result},
            {(arm, chip) for arm in ('native', 'streamed') for chip in ('0', '1')})
        for entry in result:
            self.assertEqual(len(entry['measured_operations']), 2)
            self.assertEqual(entry['measurement_medians']['kernel_envelope_ns_estimate'], 150)
            self.assertTrue(all(operation['samples'] == 6 for operation in entry['measured_operations']))
            self.assertEqual(entry['measured_operations'][0]['risc_medians_ns']['BRISC'], 45)

    def test_missing_device_events_and_unstable_captured_operations_fail(self):
        unused_operations, report, console = fixture()
        records = validate_profile(report, console)
        for failure in ('missing', 'duplicate', 'chip', 'call', 'risc'):
            rows = device_rows(records)
            if failure == 'missing':
                rows.pop()
            elif failure == 'duplicate':
                rows.append(rows[-1])
            elif failure == 'chip':
                rows = [row for row in rows if row['DEVICE ID'] == '0']
            elif failure == 'call':
                rows[-1]['GLOBAL CALL COUNT'] = '99999'
            else:
                rows[-1]['DEVICE BRISC KERNEL DURATION [ns]'] = 'nan'
            with self.subTest(failure=failure), self.assertRaises(ValueError):
                attribute(records, rows)


if __name__ == '__main__':
    unittest.main()
