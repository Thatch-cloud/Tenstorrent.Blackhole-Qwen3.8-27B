import os
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

import test_full_dflash_request as fixtures
from dflash_benchmark_report import report_rows
from full_dflash_request import summarize_dflash_live_query_requests
from live_attention_request_gate import qualify_request


class LiveAttentionRequestTests(unittest.TestCase):
    def requests(self):
        requests = fixtures.FullDFlashRequestTests().projection_requests()
        for index, entry in enumerate(requests):
            candidate = index in (1, 3, 4)
            entry.update(cache_projection_capture=False, live_query_qk=candidate)
            entry['dflash'].update(cache_projection_replays=0, live_query_qk=candidate,
                validated_live_masks=len(entry['dflash']['proposal_contexts']) if candidate else 0)
        return requests

    def test_exact_abba_retains_both_caches_and_labels_the_actual_changed_arm(self):
        requests = self.requests()
        summary = summarize_dflash_live_query_requests(requests)
        report = dict(passed=True, context_lengths=[2], request_checks=requests, request_summary=summary)
        rows = report_rows(report)
        self.assertEqual([row['tg_tokens_per_second'] for row in rows], [40, 80])
        self.assertTrue(all('cached draft K/V' in row['path'] for row in rows))
        self.assertNotIn('live-query QK', rows[0]['path'])
        self.assertIn('live-query QK', rows[1]['path'])
        self.assertFalse(summary['control']['live_query_qk'])
        self.assertTrue(summary['candidate']['live_query_qk'])

    def test_missing_masks_changed_acceptance_or_other_optimizations_cannot_pass(self):
        for mutation in ('mask', 'boolean_mask', 'flag', 'projection', 'cache', 'acceptance', 'missing'):
            requests = self.requests()
            if mutation == 'mask':
                requests[3]['dflash']['validated_live_masks'] = 0
            elif mutation == 'boolean_mask':
                requests[3]['dflash']['validated_live_masks'] = True
            elif mutation == 'flag':
                requests[3]['live_query_qk'] = 1
            elif mutation == 'projection':
                requests[3]['cache_projection_capture'] = True
            elif mutation == 'cache':
                requests[2]['cache_history'] = False
            elif mutation == 'acceptance':
                requests[3]['blocks'][0]['accepted'] += 1
            else:
                requests.pop()
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                summarize_dflash_live_query_requests(requests)

    def test_hardware_gate_rejects_wrong_context_missing_evidence_and_changed_source(self):
        for mutation in ('context', 'reports', 'source', 'empty'):
            requests = self.requests()
            report = dict(passed=True, context_lengths=[4096], request_checks=requests,
                live_attention_simulator_reports={'31': 'short', '2048': 'long'})
            if mutation == 'context':
                report['context_lengths'] = [170]
            elif mutation == 'reports':
                report['live_attention_simulator_reports'].pop('2048')
            elif mutation == 'empty':
                report['request_checks'] = []
            with patch('live_attention_request_gate.simulator_digests', return_value={'31': 'short', '2048': 'long'}), \
                    patch('live_attention_request_gate.source_hashes', return_value={'integration': 'current'}), \
                    self.assertRaises(ValueError):
                qualify_request(report)

    def test_unisolated_live_query_selection_fails_before_device_access(self):
        base = dict(os.environ, QWEN_DFLASH_CONTEXT='4096', QWEN_DFLASH_CAPTURE='1', QWEN_DFLASH_DRAFTS='7',
            QWEN_DFLASH_LIVE_QUERY_ABBA='1', QWEN_DFLASH_PROJECTION_ABBA='0', QWEN_DFLASH_CACHE_ABBA='0',
            QWEN_DFLASH_CONVOLUTION_ABBA='0', QWEN_DFLASH_COMMIT_ABBA='0', QWEN_DFLASH_VERIFIER_PROFILE='0',
            QWEN_HARDWARE_TESTS='1', QWEN_CARDS_ALLOCATED='1')
        for key, value in (('QWEN_DFLASH_CONTEXT', '8192'), ('QWEN_DFLASH_CAPTURE', '0'), ('QWEN_DFLASH_DRAFTS', '31'),
                ('QWEN_DFLASH_PROJECTION_ABBA', '1'), ('QWEN_DFLASH_CACHE_ABBA', '1'),
                ('QWEN_DFLASH_VERIFIER_PROFILE', '1'), ('QWEN_DFLASH_LIVE_QUERY_ABBA', 'yes')):
            result = subprocess.run([sys.executable, '-B', str(Path(__file__).with_name('full-prefix.py')),
                '--request-pilot', '--norm-batch'], env={**base, key: value}, capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('Live-query ABBA requires', result.stderr)

    def test_failed_preflight_stops_before_rebuild_or_model(self):
        source = Path(__file__).with_name('baseline-suite.sh').read_text()
        start = source.index('if [ "${QWEN_DFLASH_LIVE_QUERY_ABBA:-0}" = 1 ]; then')
        end = source.index('if [ "${QWEN_LIVE_QK:-0}" = 1 ]; then', start)
        block = source[start:end].replace('> /experiment/results/live-attention-preflight.json', '> /dev/null')
        for status in (0, 19):
            stub = 'set -euo pipefail\npython3() { return "$FAILURE"; }\n'
            result = subprocess.run(['bash', '-c', stub + block + '\necho proceed_to_runtime'],
                env=dict(os.environ, QWEN_DFLASH_LIVE_QUERY_ABBA='1', QWEN_RUN_MODE='full-norm-engine',
                    QWEN_DFLASH_CONTEXT='4096', QWEN_DFLASH_CAPTURE='1', QWEN_DFLASH_DRAFTS='7', FAILURE=str(status)),
                capture_output=True, text=True)
            self.assertEqual(result.returncode, status, result.stderr)
            self.assertEqual('proceed_to_runtime' in result.stdout, status == 0)

    def test_missing_hardware_artifact_or_failed_validator_cannot_be_green(self):
        source = Path(__file__).with_name('run-baseline.sh').read_text()
        start = source.rindex('if [ "$dflash_live_query_abba" = 1 ]; then')
        block = source[start:source.index('if [ "$live_qk" = 1 ]; then', start)]
        for failure, status in (('', 0), ('copy', 17), ('validate', 18)):
            stub = '''set -euo pipefail
dflash_live_query_abba=1; test_id=container; output=results
docker() { [[ "$FAILURE" != copy ]] || return 17; }
python3() { [[ "$FAILURE" != validate ]] || return 18; }
'''
            result = subprocess.run(['bash', '-c', stub + block], env=dict(os.environ, FAILURE=failure),
                capture_output=True, text=True)
            self.assertEqual(result.returncode, status, result.stderr)
