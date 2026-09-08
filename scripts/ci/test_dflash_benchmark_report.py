import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from dflash_benchmark_report import markdown, report_rows
from full_dflash_request import summarize_dflash_convolution_requests, summarize_dflash_requests
import test_full_dflash_request as fixtures


class DFlashBenchmarkReportTests(unittest.TestCase):
    def report(self, *, abba=True):
        source = fixtures.FullDFlashRequestTests()
        requests = source.convolution_requests() if abba else source.requests()
        summarize = summarize_dflash_convolution_requests if abba else summarize_dflash_requests
        return dict(passed=True, context_lengths=[2], request_checks=requests, request_summary=summarize(requests))

    def test_abba_arms_are_separate_not_combined_into_fake_batch_throughput(self):
        rows = report_rows(self.report())
        self.assertEqual([row['arm'] for row in rows], ['control', 'candidate'])
        self.assertEqual([row['tg_tokens_per_second'] for row in rows], [40, 80])
        self.assertEqual([row['streams'] for row in rows], [1, 1])
        self.assertEqual([row['measured_requests'] for row in rows], [2, 2])
        self.assertNotIn('fused convolution', rows[0]['path'])
        self.assertIn('fused convolution', rows[1]['path'])

    def test_single_arm_historical_report_does_not_need_new_benchmark_fields(self):
        report = self.report(abba=False)
        del report['request_summary']['benchmark']
        rows = report_rows(report)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]['pp_tokens_per_second'], 200)
        self.assertEqual(rows[0]['tg_tokens_per_second'], 40)

    def test_failed_incomplete_or_inconsistent_artifacts_are_rejected(self):
        for mutation in ('failed', 'incomplete', 'context', 'recorded_context', 'streams', 'rate', 'audit', 'state'):
            report = self.report()
            if mutation == 'failed':
                report['passed'] = False
            elif mutation == 'incomplete':
                report['request_checks'].pop()
            elif mutation == 'context':
                report['context_lengths'] = [4096]
            elif mutation in ('recorded_context', 'streams', 'rate'):
                key = dict(recorded_context='context', streams='streams', rate='committed_tokens_per_second')[mutation]
                report['request_summary']['candidate'][key] = 4096
            elif mutation == 'audit':
                report['request_checks'][3]['instrumented_timing'] = True
            else:
                report['request_checks'][3]['state_exact'] = False
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                report_rows(report)

    def test_markdown_has_explicit_scope_and_does_not_edit_measurements(self):
        report = self.report()
        original = copy.deepcopy(report)
        rows = report_rows(report)
        for row in rows:
            row.update(run_id='123', run_url='https://example.test/run/123')
        rendered = markdown(rows)
        self.assertIn('PP tok/s | CTX tokens | TG tok/s', rendered)
        self.assertIn('| 200.00 | 2 | 80.00 | 1 | Up to 8 | 2, 2 | 0.09 s |', rendered)
        self.assertIn('not a serving or held-out quality benchmark', rendered)
        self.assertIn('Audits are excluded', rendered)
        self.assertEqual(report, original)

    def test_offline_cli_emits_raw_artifact_provenance_without_runtime_setup(self):
        payload = json.dumps(self.report()).encode()
        with tempfile.TemporaryDirectory() as temporary:
            report = Path(temporary) / 'report.json'
            report.write_bytes(payload)
            result = subprocess.run([sys.executable, '-B', str(Path(__file__).with_name('dflash_benchmark_report.py')),
                '--artifact', '123', str(report), '--format', 'json'], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        rows = json.loads(result.stdout)
        self.assertEqual(len(rows), 2)
        for row in rows:
            self.assertEqual(row['run_id'], '123')
            self.assertTrue(row['run_url'].endswith('/actions/runs/123'))
            self.assertEqual(row['artifact_sha256'], hashlib.sha256(payload).hexdigest())


if __name__ == '__main__':
    unittest.main()
