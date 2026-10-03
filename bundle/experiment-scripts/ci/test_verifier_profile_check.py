import importlib.util
import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


spec = importlib.util.spec_from_file_location('verifier_check', Path(__file__).with_name('check-verifier-profile.py'))
checker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(checker)


class VerifierProfileCheckTests(unittest.TestCase):
    def test_configuration_requires_exact_boolean_match(self):
        flags = (*checker.PROFILE_BASE_FLAGS, *checker.PROFILE_BEST_FLAGS)
        configuration = dict.fromkeys(flags, True)
        report = dict(profile_configuration=configuration)
        checker.validate_configuration(report, report)
        checker.validate_configuration({}, {})
        for changed in (None, {}, dict(configuration, norm_batch=False), dict(configuration, attention_tree=1)):
            with self.subTest(changed=changed):
                with self.assertRaisesRegex(AssertionError, 'configurations must match'):
                    checker.validate_configuration(report, dict(profile_configuration=changed))
        partial = dict(profile_configuration=dict(configuration, norm_batch=False))
        with self.assertRaisesRegex(AssertionError, 'configurations must match'):
            checker.validate_configuration(partial, partial)

    def test_isolated_processes_allow_reused_trace_ids_but_require_both_contexts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'correctness.json').write_text(json.dumps(self.correctness_fixture()))
            for context_index, context in enumerate((4095, 16383)):
                generation, rows, console = self.fixture()
                generation['timings'] = [generation['timings'][context_index]]
                for record in generation['timings'][0]['device_profile']['records']:
                    record['trace_id'] -= context_index * 3
                rows = [row for row in rows if context_index * 3 < int(row['METAL TRACE ID']) <= context_index * 3 + 3]
                for row in rows:
                    row['METAL TRACE ID'] = str(int(row['METAL TRACE ID']) - context_index * 3)
                    row['OP NAME'] = row.pop('OP CODE')
                selected = root / f'context-{context}'
                selected.mkdir()
                (selected / 'generation.json').write_text(json.dumps(generation))
                (selected / 'console.log').write_text('\n'.join(line for line in console.splitlines() if f'ctx{context}_' in line))
                with (selected / 'cpp_device_perf_report.csv').open('w', newline='') as stream:
                    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                    writer.writeheader()
                    writer.writerows(rows)
            with patch('builtins.print'):
                checker.main(root)
            result = json.loads((root / 'attribution.json').read_text())
            self.assertEqual([process['context'] for process in result['processes']], [4095, 16383])
            self.assertEqual([process['attribution']['traces'][0]['trace_id'] for process in result['processes']], [1, 1])
            (root / 'context-16383' / 'generation.json').unlink()
            with self.assertRaises(FileNotFoundError):
                checker.main(root)

    def correctness_fixture(self):
        report = dict(passed=True, correctness_only=True, instrumented_timing=False)
        for key, dimension, values in (('batch_checks', 'rows', (1, 2, 4, 8, 16, 32)),
                                       ('checks', 'prefix', (0, 1, 16, 32))):
            report[key] = [dict(length=length, trace=trace, **{dimension: value},
                logits_exact=True, all_gdn_states_exact=True, valid_kv_exact=True)
                for length in (4095, 16383) for trace in (False, True) for value in values]
        report['negative_controls'] = [dict(length=length, trace=trace,
            stale_gdn_detected=True, wrong_page_detected=True)
            for length in (4095, 16383) for trace in (False, True)]
        return report

    def test_separate_correctness_requires_complete_exact_matrix(self):
        checker.validate_correctness(self.correctness_fixture())
        for key in ('batch_checks', 'checks', 'negative_controls'):
            for change in ('missing', 'duplicate', 'inexact'):
                with self.subTest(key=key, change=change):
                    report = self.correctness_fixture()
                    if change == 'missing':
                        report[key].pop()
                    elif change == 'duplicate':
                        report[key][-1] = report[key][0]
                    else:
                        field = 'wrong_page_detected' if key == 'negative_controls' else 'valid_kv_exact'
                        report[key][0][field] = False
                    with self.assertRaises(AssertionError):
                        checker.validate_correctness(report)

    def test_separate_correctness_rejects_timing_or_failed_run(self):
        for field, value in (('passed', False), ('correctness_only', False),
                             ('instrumented_timing', True), ('timings', [{}])):
            report = self.correctness_fixture()
            report[field] = value
            with self.assertRaises(AssertionError):
                checker.validate_correctness(report)

    def test_runtime_cpp_report_is_accepted_without_expanding_host_metadata(self):
        generation, rows, console = self.fixture()
        for row in rows:
            row['OP NAME'] = row.pop('OP CODE')
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'correctness.json').write_text(json.dumps(self.correctness_fixture()))
            (root / 'generation.json').write_text(json.dumps(generation))
            (root / 'console.log').write_text(console, encoding='utf-8')
            with (root / 'cpp_device_perf_report.csv').open('w', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            with patch('builtins.print'):
                checker.main(root, root / 'generation.json')
            result = json.loads((root / 'attribution.json').read_text())
            self.assertTrue(result['passed'])
            self.assertEqual(result['source_format'], 'runtime C++ device operation report')

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
