import copy
from contextlib import redirect_stdout
import io
import os
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from request_verifier_profile import PROFILER_FLAGS, RequestVerifierProfile
from request_verifier_profile_report import analyze_traces, intervals, validate_request


class RequestVerifierProfileTests(unittest.TestCase):
    def test_folded_observer_requires_matching_engine_and_records_path(self):
        engine = self.engine()
        engine.attention_replay = engine.target_attention_t16 = True
        ticket = SimpleNamespace(position=4096, tokens=list(range(16)))
        with patch.dict(os.environ, {name: '1' for name in PROFILER_FLAGS}), redirect_stdout(io.StringIO()):
            observer = RequestVerifierProfile(engine.operations, engine.mesh, signpost=Mock(),
                full_rows=16, target_attention_t16=True)
            for index in range(3):
                engine.phase, engine.pending = 'idle', None
                with observer(engine, ticket):
                    engine.phase, engine.pending = 'verified', ticket
                    engine.buckets[8]['first'] = False
            self.assertTrue(observer.summary()['target_attention_t16'])
            engine.phase, engine.pending = 'idle', None
            for replay, folded in ((False, True), (True, False), (False, False)):
                engine.attention_replay, engine.target_attention_t16 = replay, folded
                with self.assertRaises(ValueError):
                    with observer(engine, ticket):
                        self.fail('Mismatched attention path admitted')

    def test_folded_observer_rejects_implicit_or_wrong_width(self):
        for width, folded in ((8, True), (16, 1), (16, 'true')):
            with self.assertRaises(ValueError):
                RequestVerifierProfile(Mock(), object(), full_rows=width, target_attention_t16=folded)

    def test_explicit_t16_observer_requires_three_full_width_calls(self):
        engine = self.engine()
        engine.buckets = {16: engine.buckets[8]}
        engine.bucket_key = lambda ticket: 16
        with patch.dict(os.environ, {name: '1' for name in PROFILER_FLAGS}), redirect_stdout(io.StringIO()):
            observer = RequestVerifierProfile(engine.operations, engine.mesh, signpost=Mock(), full_rows=16)
            for index in range(3):
                ticket = SimpleNamespace(position=4096 + index * 16, tokens=list(range(16)))
                engine.phase, engine.pending = 'idle', None
                with observer(engine, ticket):
                    engine.phase, engine.pending = 'verified', ticket
                    engine.buckets[16]['first'] = False
            result = observer.summary()
        self.assertEqual(result['full_rows'], 16)
        self.assertEqual([record['rows'] for record in result['records']], [16, 16, 16])

    def test_invalid_profile_width_is_rejected(self):
        for width in (True, 7, 32, '16'):
            with self.assertRaises(ValueError):
                RequestVerifierProfile(Mock(), object(), full_rows=width)

    def engine(self):
        mesh = object()
        operations = SimpleNamespace(synchronize_device=Mock(), ReadDeviceProfiler=Mock())
        engine = SimpleNamespace(mesh=mesh, operations=operations, commit_only_gdn=True, norm_batch=True,
            native_sampling_rows=True, attention_replay=False, phase='idle', pending=None,
            bucket_key=lambda ticket: 8, buckets={8: {'trace': 23, 'first': True}})
        return engine

    def test_observer_records_real_calls_without_executing_or_publishing_them(self):
        engine = self.engine()
        with patch.dict(os.environ, {name: '1' for name in PROFILER_FLAGS}), redirect_stdout(io.StringIO()) as console:
            signpost = Mock()
            observer = RequestVerifierProfile(engine.operations, engine.mesh, signpost=signpost)
            for index in range(3):
                ticket = SimpleNamespace(position=4096 + index * 8, tokens=list(range(8)))
                engine.phase, engine.pending = 'idle', None
                with observer(engine, ticket):
                    self.assertEqual(engine.phase, 'idle')
                    engine.phase, engine.pending = 'verified', ticket
                    engine.buckets[8]['first'] = False
            result = observer.summary()
        self.assertEqual([record['trace_ordinal'] for record in result['records']], [0, 1, 2])
        self.assertEqual([record['first_replay'] for record in result['records']], [True, False, False])
        self.assertEqual(result['trace_counts'], {23: 3})
        self.assertTrue(result['host_calls'])
        self.assertEqual(engine.operations.ReadDeviceProfiler.call_count, 6)
        self.assertEqual(signpost.call_count, 6)
        self.assertEqual(console.getvalue().count('QWEN_REQUEST_VERIFY_BEGIN'), 3)

    def test_failed_verification_has_end_marker_but_no_success_record(self):
        engine = self.engine()
        ticket = SimpleNamespace(position=4096, tokens=list(range(8)))
        with patch.dict(os.environ, {name: '1' for name in PROFILER_FLAGS}), redirect_stdout(io.StringIO()) as console:
            observer = RequestVerifierProfile(engine.operations, engine.mesh, signpost=Mock())
            with self.assertRaisesRegex(RuntimeError, 'verification failed'):
                with observer(engine, ticket):
                    raise RuntimeError('verification failed')
        self.assertFalse(observer.active)
        self.assertEqual(observer.records, [])
        self.assertIn('QWEN_REQUEST_VERIFY_END', console.getvalue())
        with self.assertRaises(ValueError):
            observer.summary()

    def test_profiler_and_runtime_contracts_fail_before_observation(self):
        engine = self.engine()
        with patch.dict(os.environ, {name: '0' for name in PROFILER_FLAGS}):
            with self.assertRaises(ValueError):
                RequestVerifierProfile(engine.operations, engine.mesh, signpost=Mock())
        with patch.dict(os.environ, {name: '1' for name in PROFILER_FLAGS}):
            observer = RequestVerifierProfile(engine.operations, engine.mesh, signpost=Mock())
            engine.attention_replay = True
            with self.assertRaises(ValueError):
                with observer(engine, SimpleNamespace()):
                    self.fail('Observer yielded for another runtime')
        engine.operations.ReadDeviceProfiler.assert_not_called()

    def test_each_profiler_flag_including_incremental_dump_is_required(self):
        engine = self.engine()
        self.assertIn('TT_METAL_PROFILER_MID_RUN_DUMP', PROFILER_FLAGS)
        for missing in PROFILER_FLAGS:
            with self.subTest(missing=missing), patch.dict(os.environ, {name: '1' for name in PROFILER_FLAGS}):
                del os.environ[missing]
                with self.assertRaisesRegex(ValueError, 'incremental device-data dumps'):
                    RequestVerifierProfile(engine.operations, engine.mesh, signpost=Mock())
        engine.operations.ReadDeviceProfiler.assert_not_called()


class RequestProfileReportTests(unittest.TestCase):
    def test_t16_attribution_requires_explicit_width_and_both_chips(self):
        records, rows = self.fixture()
        for record in records:
            record['rows'] = 16
        with self.assertRaises(ValueError):
            analyze_traces(records, rows)
        result = analyze_traces(records, rows, full_rows=16)
        self.assertEqual({entry['device'] for entry in result}, {'0', '1'})
        self.assertTrue(all(entry['steady_replays'] == 2 for entry in result))
        with self.assertRaises(ValueError):
            analyze_traces(records, [row for row in rows if row['DEVICE ID'] == '0'], full_rows=16)

    def fixture(self):
        records, rows = [], []
        for repeat in range(3):
            record = dict(block=repeat, position=4096 + repeat * 8, rows=8, trace_id=23, trace_ordinal=repeat,
                first_replay=repeat == 0, instrumented_host_ms=1)
            record['label'] = f'qwen_request_verify_{repeat}_pos{record["position"]}_t8_trace23'
            records.append(record)
            for device in ('0', '1'):
                for name, cores, start, end in (('Matmul', '39', 100, 300), ('Generic', '96', 150, 450), ('AllGather', '10', 500, 600)):
                    base = 1000000 * (repeat + 1) + int(device) * 100000000
                    rows.append({'METAL TRACE ID': '23', 'METAL TRACE REPLAY SESSION ID': str(repeat + 10), 'DEVICE ID': device,
                        'OP NAME': name, 'CORE COUNT': cores, 'DEVICE KERNEL START CYCLE': str(base + start),
                        'DEVICE KERNEL END CYCLE': str(base + end), 'DEVICE KERNEL DURATION [ns]': str((end - start) * .75)})
        return records, rows

    def test_envelopes_and_unions_are_per_chip_not_summed_kernel_costs(self):
        records, rows = self.fixture()
        result = analyze_traces(records, rows)
        self.assertEqual([entry['device'] for entry in result], ['0', '1'])
        for device in result:
            self.assertEqual(device['steady_replays'], 2)
            self.assertEqual(device['steady_medians'], dict(kernel_envelope_ns_estimate=375,
                kernel_union_ns_estimate=337.5, uncovered_interval_ns_estimate=37.5))
            self.assertEqual(sum(group['median_summed_kernel_ns'] for group in device['operation_core_groups']), 450)

    def test_missing_extra_or_inconsistent_runtime_evidence_is_rejected(self):
        for mutation in ('chip', 'replay', 'extra', 'operation', 'clock', 'duration'):
            records, rows = self.fixture()
            if mutation == 'chip':
                rows = [row for row in rows if row['DEVICE ID'] == '0']
            elif mutation == 'replay':
                rows = [row for row in rows if row['METAL TRACE REPLAY SESSION ID'] != '11']
            elif mutation == 'extra':
                rows.extend([{**row, 'METAL TRACE REPLAY SESSION ID': '15'} for row in rows if row['METAL TRACE REPLAY SESSION ID'] == '10'])
            elif mutation == 'operation':
                rows.pop()
            else:
                rows[0]['DEVICE KERNEL DURATION [ns]'] = '600' if mutation == 'clock' else 'nan'
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                analyze_traces(records, rows)

    def test_request_markers_must_match_every_verified_block(self):
        records, unused = self.fixture()
        request = dict(commit_only_gdn=True, fused_convolution=True, cache_history=True, cache_projection_capture=False,
            dflash=dict(block_rows=8), blocks=[dict(position=entry['position'], rows=8) for entry in records],
            verifier_profile=dict(records=records, trace_counts={'23': 3}, host_calls=[dict(function='verify')]))
        report = dict(passed=True, instrumented_timing=True, correctness_only=True, request_checks=[request])
        console = '\n'.join(f'QWEN_REQUEST_VERIFY_BEGIN {entry["label"]}\nQWEN_REQUEST_VERIFY_END {entry["label"]}' for entry in records)
        with patch('request_verifier_profile_report.summarize_dflash_requests', return_value={'context': 4096}) as validate:
            self.assertEqual(validate_request(report, console), (request, records))
            validate.assert_called_once_with([request], audit_only=True)
            for mutation in ('dropped', 'marker', 'ordinal', 'frontier', 'instrumented', 'config'):
                changed = copy.deepcopy(report)
                text = console
                if mutation == 'dropped':
                    text = console.replace('QWEN_REQUEST_VERIFY_END', 'markers were dropped\nQWEN_REQUEST_VERIFY_END', 1)
                elif mutation == 'marker':
                    text = console.replace('QWEN_REQUEST_VERIFY_BEGIN', 'missing', 1)
                elif mutation == 'ordinal':
                    changed['request_checks'][0]['verifier_profile']['records'][1]['trace_ordinal'] = 0
                elif mutation == 'frontier':
                    changed['request_checks'][0]['blocks'][0]['position'] = 42
                elif mutation == 'instrumented':
                    changed['instrumented_timing'] = False
                else:
                    changed['request_checks'][0]['cache_projection_capture'] = True
                with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                    validate_request(changed, text)


if __name__ == '__main__':
    unittest.main()
