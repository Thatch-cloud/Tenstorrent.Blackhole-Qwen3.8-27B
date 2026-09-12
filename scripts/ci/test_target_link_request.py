from copy import deepcopy
import os
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import Mock

from dflash_prefill_window import prefill_window, validate_prefill_chunks
from dflash_benchmark_report import report_rows
from full_dflash_request import TARGET_TAPS
from sampling_link_policy import SOURCES as FABRIC_SOURCES
from target_link_request import measure_target_links, summarize
import test_full_dflash_request as fixtures
from test_model_link_policy import model_fixture


class TargetLinkRequestTests(unittest.TestCase):
    def requests(self):
        requests = fixtures.FullDFlashRequestTests().projection_requests()
        chunks = [dict(chunk_start=start, valid_rows=2048, bucket=2048, retained_rows=0 if start == 0 else 2048)
            for start in (0, 2048)]
        window = prefill_window(4096)
        for index, entry in enumerate(requests):
            candidate, audited = index in (1, 3, 4), entry['instrumented_timing']
            entry.update(prompt_tokens=[10] * 4096, cache_projection_capture=False, live_query_qk=False,
                target_four_links=candidate, fabric_sources=FABRIC_SOURCES,
                target_link_audit=dict(requested_links=4 if candidate else 2, owners_validated=193,
                    original_requests=dict(default=2, axis0=2, axis1=2), restored=True,
                    effective_requests={name: 4 if candidate else 2 for name in ('default', 'axis0', 'axis1')},
                    calls=dict(default=1, axis0=128, axis1=1)),
                final_target_digests={name: ['a' * 64, 'b' * 64] for name in ('active_gdn', 'valid_kv', 'inactive')})
            entry['blocks'][0]['position'] = 4096
            for name in ('gdn_verify_checks',):
                for check in entry[name]:
                    check['position'] = 4096
            for check in entry['dflash']['convolution_checks']:
                check['position'] = 4096
            entry['dflash'].update(cache_projection_replays=0, prefill_window=dict(window), prefill_chunks=deepcopy([chunks, chunks]),
                prefill_checks=[dict(chip=chip, **piece, exact=True) for unused in range(2)
                    for piece in validate_prefill_chunks(4096, chunks) for tap in TARGET_TAPS for chip in range(2)] if audited else [],
                prefill_assembly_checks=[dict(tap=tap, chip=chip, **window, exact=True)
                    for unused in range(2) for tap in TARGET_TAPS for chip in range(2)] if audited else [],
                history_checks=[dict(position=position, rows=2048, layer=layer, head=head, chip=chip, exact=True)
                    for position in (4096, 4098) for layer in range(5) for head in ('k', 'v') for chip in range(2)] if audited else [])
        return requests

    def test_matched_abba_reports_real_context_and_retains_request_timing(self):
        result = summarize(self.requests())
        self.assertEqual(result['candidate_over_control'], 2)
        self.assertEqual(result['control']['benchmark']['ctx_tokens'], 4096)
        self.assertEqual(result['candidate']['benchmark']['ctx_tokens'], 4096)
        self.assertEqual(result['control']['committed_tokens_per_second'], 40)
        self.assertEqual(result['candidate']['committed_tokens_per_second'], 80)
        self.assertEqual(result['measured_order'], ['control', 'candidate', 'candidate', 'control'])
        rows = report_rows(dict(passed=True, context_lengths=[4096], request_checks=self.requests(), request_summary=result))
        self.assertIn('target CCL 2 links', rows[0]['path'])
        self.assertIn('target CCL 4 links', rows[1]['path'])
        self.assertEqual([row['tg_tokens_per_second'] for row in rows], [40, 80])

    def test_policy_state_and_workload_mismatches_are_rejected(self):
        for mutation in ('state', 'empty_state', 'false_digest', 'restored', 'calls', 'requested', 'boolean_count',
                'original', 'context', 'sampler', 'second_change', 'acceptance', 'order', 'missing'):
            requests = self.requests()
            entry = requests[3]
            if mutation == 'state':
                entry['final_target_digests']['valid_kv'][1] = 'c' * 64
            elif mutation == 'empty_state':
                entry['final_target_digests']['active_gdn'] = []
            elif mutation == 'false_digest':
                entry['final_target_digests']['inactive'] = [True]
            elif mutation == 'restored':
                entry['target_link_audit']['restored'] = False
            elif mutation == 'calls':
                entry['target_link_audit']['calls']['axis0'] = 0
            elif mutation == 'requested':
                entry['target_link_audit']['effective_requests']['axis0'] = 2
            elif mutation == 'boolean_count':
                entry['target_link_audit']['calls']['axis1'] = True
            elif mutation == 'original':
                entry['target_link_audit']['original_requests']['axis0'] = 4
            elif mutation == 'context':
                entry['prompt_tokens'].pop()
            elif mutation == 'sampler':
                entry['sampler_num_links'] = 2
            elif mutation == 'second_change':
                entry['live_query_qk'] = True
            elif mutation == 'acceptance':
                entry['blocks'][0]['accepted'] += 1
            elif mutation == 'order':
                requests[3], requests[5] = requests[5], requests[3]
            else:
                requests.pop()
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                summarize(requests)

    def test_native_measurement_is_wrapped_without_changing_sampler_or_arguments(self):
        model = model_fixture()
        operations, sampler = object(), object()
        def measure(found_operations, found_model, found_sampler, **options):
            self.assertIs(found_operations, operations)
            self.assertIs(found_model, model)
            self.assertIs(found_sampler, sampler)
            self.assertEqual(model.tt_ccl.get_num_links(0), 4)
            return dict(prompt_tokens=[1] * 4096, committed_decode_tokens=120)
        callbacks = dict(live_digest=Mock(return_value=['a' * 64]), kv_digest=Mock(return_value=['b' * 64]),
            inactive_digest=Mock(return_value=['c' * 64]))
        result = measure_target_links(measure, operations, model, sampler, candidate=True, **callbacks)
        self.assertTrue(result['target_link_audit']['restored'])
        self.assertEqual(model.tt_ccl.get_num_links(), 2)
        callbacks['kv_digest'].assert_called_once_with(4216)
        self.assertEqual(set(result['final_target_digests']), {'active_gdn', 'valid_kv', 'inactive'})

    def test_invalid_suite_combinations_fail_before_device_import(self):
        script = Path(__file__).with_name('full-prefix.py')
        base = dict(os.environ, QWEN_DFLASH_TARGET_LINKS_ABBA='1', QWEN_DFLASH_DRAFTS='7',
            QWEN_DFLASH_CAPTURE='1', QWEN_DFLASH_CONTEXT='4096', QWEN_HARDWARE_TESTS='1', QWEN_CARDS_ALLOCATED='1')
        for name, value in (('QWEN_DFLASH_CONTEXT', '8192'), ('QWEN_DFLASH_LIVE_QUERY_ABBA', '1'),
                ('QWEN_DFLASH_CAPTURE', '0'), ('QWEN_DFLASH_TARGET_LINKS_ABBA', 'true')):
            environment = {**base, name: value}
            result = subprocess.run([sys.executable, '-B', str(script), '--request-pilot', '--norm-batch'],
                env=environment, capture_output=True, text=True)
            with self.subTest(name=name):
                self.assertNotEqual(result.returncode, 0)
                self.assertIn('Target-link ABBA requires', result.stderr)
                self.assertNotIn('ModuleNotFoundError', result.stderr)

    def test_host_route_requires_the_full_request_artifact(self):
        source = Path(__file__).with_name('run-baseline.sh').read_text()
        start = source.index('if [ "$dflash_target_links_abba" = 1 ]; then')
        block = source[start:source.index('\nfi', start)]
        self.assertIn('docker cp "$test_id:/experiment/results/full-dflash-request.json"', block)
        self.assertIn('python3 scripts/ci/target_link_request.py --hardware-result', block)
        self.assertNotIn('|| true', block)

    def test_host_selects_only_the_cached_t8_target_link_experiment(self):
        source = Path(__file__).with_name('run-baseline.sh').read_text()
        prefix = source[:source.index('output=experiment-results')]
        environment = dict(os.environ, QWEN_CARDS_ALLOCATED='1', QWEN_RUN_MODE='full-dflash-target-links-request')
        result = subprocess.run(['bash', '-c', prefix + '''
printf '%s\\n' "$mode $dflash_drafts $dflash_capture $dflash_context $dflash_target_links_abba $dflash_live_query_abba $tensix_mlp $ccl_build $projection_links"
'''], env=environment, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), 'full-norm-engine 7 1 4096 1 0 0 1 4')


if __name__ == '__main__':
    unittest.main()
