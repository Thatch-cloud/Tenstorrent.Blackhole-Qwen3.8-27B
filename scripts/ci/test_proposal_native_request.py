from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys
from types import MethodType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch

from dflash_device import DFlashDevice
from dflash_proposal_inputs import proposal_inputs
from dflash_proposal_trace import PreparedDFlashProposal
from dflash_benchmark_report import report_rows
from full_dflash_request import summarize_dflash_abba_requests
from proposal_native_attention import validate_mask
from proposal_native_attention_gate import ORIGINAL, PACKER
from proposal_native_request import acceptance, input_evidence, qualify_request, summarize
import test_full_dflash_request as fixtures


class NativeProposalRequestTests(unittest.TestCase):
    def requests(self):
        requests = fixtures.FullDFlashRequestTests().projection_requests()
        for index, entry in enumerate(requests):
            candidate, audit = index in (1, 3, 4), entry['instrumented_timing']
            entry.update(native_proposal_attention=candidate, cache_projection_capture=False)
            if candidate:
                blocks = [dict(rows=2, source='dflash2', accepted=0, match_length=0,
                    position=2 + offset, input_tokens=[12 + offset, 99], committed=1) for offset in range(2)]
            else:
                blocks = [dict(rows=2, source='dflash2', accepted=1, match_length=0,
                    position=2, input_tokens=[12, 13], committed=2)]
            for block in blocks:
                block.update(draft_ms=1., input_ms=.1, verify_readback_ms=2., select_commit_ms=.5, cycle_ms=4.)
            entry.update(blocks=blocks, proposed=sum(block['rows'] - 1 for block in blocks),
                accepted=sum(block['accepted'] for block in blocks))
            entry['gdn_verify_checks'] = [dict(position=block['position'], rows=block['rows'], unchanged=True)
                for block in blocks] if audit else []
            draft = entry['dflash']
            draft.update(native_proposal_attention=candidate,
                validated_native_proposal_masks=len(draft['proposal_contexts']) if candidate else 0,
                attention='Native BF16 proposal-only attention; not an exact replacement' if candidate else 'control',
                cache_projection_replays=0, proposal_calls=len(blocks),
                convolution_checks=[dict(position=block['position'], layer=layer, chip=chip, rows=32, exact=True)
                    for block in blocks for layer in range(5) for convolution in range(4) for chip in range(2)] if audit else [],
                proposal_trace_checks=[dict(position=block['position'], context=256, tensors=2, exact=True)
                    for block in blocks] if audit else [])
            draft['cache_checks'] = deepcopy(draft['proposal_trace_checks'])
            draft['history_checks'] = [dict(position=position, rows=position, layer=layer, head=head, chip=chip, exact=True)
                for position in (2, *(block['position'] + block['committed'] for block in blocks))
                for layer in range(5) for head in ('k', 'v') for chip in range(2)] if audit else []
        return requests

    def test_different_proposals_preserve_exact_outputs_and_separate_accounting(self):
        requests = self.requests()
        summary = summarize(requests)
        self.assertTrue(summary['proposal_trajectories_may_differ'])
        self.assertEqual(summary['control']['committed_tokens_per_second'], 40)
        self.assertEqual(summary['candidate']['committed_tokens_per_second'], 80)
        self.assertEqual(summary['control']['acceptance']['acceptance_fraction'], 1)
        self.assertEqual(summary['candidate']['acceptance']['acceptance_fraction'], 0)
        self.assertEqual(summary['candidate']['acceptance']['zero_acceptance_blocks'], 4)
        self.assertEqual(summary['candidate']['mean_block_costs_ms']['draft_ms'], 1)
        report = dict(passed=True, context_lengths=[2], request_checks=requests, request_summary=summary)
        rows = report_rows(report)
        self.assertNotIn('native approximate', rows[0]['path'])
        self.assertIn('native approximate proposal attention', rows[1]['path'])
        with self.assertRaisesRegex(ValueError, 'separate policy'):
            summarize_dflash_abba_requests(requests, arm_key='cache_history')

    def test_changed_tokens_state_sources_or_mask_proof_cannot_pass(self):
        for failure in ('tokens', 'state', 'inactive', 'sources', 'policy', 'mask', 'boolean', 'cache', 'missing_audit'):
            requests = self.requests()
            if failure == 'tokens':
                requests[3]['emitted'][1] = 17
            elif failure in ('state', 'inactive'):
                requests[3]['state_exact' if failure == 'state' else 'inactive_exact'] = False
            elif failure == 'sources':
                requests[3]['sources'] = {'changed': 'code'}
            elif failure == 'policy':
                requests[3]['target_four_links'] = True
            elif failure == 'mask':
                requests[3]['dflash']['validated_native_proposal_masks'] = 0
            elif failure == 'boolean':
                requests[3]['native_proposal_attention'] = 1
            elif failure == 'cache':
                requests[1]['dflash']['history_checks'].pop()
            else:
                requests.pop(1)
            with self.subTest(failure=failure), self.assertRaises(ValueError):
                summarize(requests)

    def test_policy_must_reproduce_its_own_audited_proposal_tape(self):
        requests = self.requests()
        requests[3]['blocks'][0]['input_tokens'][1] = 98
        with self.assertRaisesRegex(ValueError, 'own audited'):
            summarize(requests)

    def test_proposal_counts_frontiers_and_matched_tokens_are_recomputed(self):
        for failure in ('proposed', 'accepted', 'position', 'prefix', 'count', 'time', 'boolean_rows'):
            entry = self.requests()[0]
            if failure in ('proposed', 'accepted'):
                entry[failure] += 1
            elif failure == 'position':
                entry['blocks'][0]['position'] += 1
            elif failure == 'prefix':
                entry['blocks'][0]['input_tokens'][1] = 99
            elif failure == 'count':
                entry['blocks'][0]['committed'] = 3
            elif failure == 'time':
                entry['blocks'][0]['draft_ms'] = float('nan')
            else:
                entry['blocks'][0]['rows'] = True
            with self.subTest(failure=failure), self.assertRaises(ValueError):
                acceptance(entry)

    def test_hardware_gate_requires_teardown_and_both_original_runtime_prerequisites(self):
        for field, value in (('closed_cleanly', False), ('error', 'teardown failed'),
                ('eligible_for_serving_gate', True), ('context_lengths', [31]), ('proposal_native_inputs', None)):
            report = dict(passed=True, closed_cleanly=True, eligible_for_serving_gate=False, context_lengths=[4096],
                proposal_native_inputs=None, request_checks=self.requests())
            report[field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                qualify_request(report)
        with self.assertRaisesRegex(ValueError, 'original packer'):
            input_evidence({})

    def test_checked_in_primitives_qualify_against_the_restored_native_packer(self):
        report = json.loads(Path(__file__).with_name('proposal-native-attention-simulator-31.json').read_text())
        result = input_evidence({**report['native_sources'], PACKER: ORIGINAL[PACKER]})
        self.assertEqual(set(result['reports']), {'31', '2048'})
        self.assertEqual(result['native_sources'][PACKER], ORIGINAL[PACKER])
        self.assertTrue(all(entry['gate']['passed'] for entry in result['reports'].values()))
        self.assertTrue(all(entry['gate']['accuracy_qualified'] is False for entry in result['reports'].values()))

    def test_cli_rejects_mixed_native_proposal_experiments_before_device_access(self):
        environment = dict(os.environ, QWEN_DFLASH_NATIVE_PROPOSAL_ABBA='1', QWEN_DFLASH_CONTEXT='4096',
            QWEN_DFLASH_CAPTURE='1', QWEN_DFLASH_DRAFTS='7', QWEN_HARDWARE_TESTS='1', QWEN_CARDS_ALLOCATED='1')
        for name, value in (('QWEN_DFLASH_CONTEXT', '8192'), ('QWEN_DFLASH_NATIVE_PROPOSAL_ABBA', 'yes'),
                ('QWEN_DFLASH_TARGET_LINKS_ABBA', '1'), ('QWEN_DFLASH_LIVE_QUERY_ABBA', '1'),
                ('QWEN_DFLASH_VERIFIER_PROFILE', '1'), ('QWEN_DFLASH_CAPTURE', '0')):
            result = subprocess.run([sys.executable, '-B', str(Path(__file__).with_name('full-prefix.py')),
                '--max-rows', '32', '--batch', '--coding-cost', '--serial-sdpa', '--request-pilot', '--norm-batch'],
                env={**environment, name: value}, capture_output=True, text=True)
            with self.subTest(name=name):
                self.assertEqual(result.returncode, 2)
                self.assertIn('Native proposal ABBA requires', result.stderr)


class NativeProposalMaskLifecycleTests(unittest.TestCase):
    def test_successful_upload_marks_mask_and_failed_update_revokes_it(self):
        operations = SimpleNamespace(ReplicateTensorToMesh=lambda mesh: mesh,
            from_torch=lambda value, **kwargs: value.clone(),
            copy_host_to_device_tensor=Mock(side_effect=lambda source, destination: destination.copy_(source)),
            slice=lambda value, start, end: value[tuple(slice(first, last) for first, last in zip(start, end, strict=True))],
            copy=Mock(side_effect=lambda source, destination: destination.copy_(source)), synchronize_device=Mock())
        active = [dict(k=torch.ones((1, 4, 256, 128), dtype=torch.bfloat16),
            v=torch.ones((1, 4, 256, 128), dtype=torch.bfloat16))]
        cache = SimpleNamespace(position=100, history_rows=100, active=active, pending=None, owned=list(active[0].values()))
        device = SimpleNamespace(operations=operations, mesh=object(), position=100, history_rows=100, block_rows=8,
            history=torch.zeros((1, 1, 2048, 5120), dtype=torch.bfloat16),
            spare_history=torch.zeros((1, 1, 2048, 5120), dtype=torch.bfloat16), progress=None,
            native_proposal_attention=True, validated_native_proposal_masks=set())
        device.temporaries = MethodType(DFlashDevice.temporaries, device)
        host = proposal_inputs(17, 100, 100, 8, 256)
        bucket = SimpleNamespace(context=256, identifiers=host['identifiers'].clone(), mask=host['mask'].clone(),
            rope={name: tuple(value.clone() for value in host['rope'][name]) for name in ('q', 'k')},
            history=torch.zeros((1, 1, 288, 5120), dtype=torch.bfloat16),
            cached_history=[{name: torch.zeros_like(value) for name, value in active[0].items()}])
        bucket.inputs = [bucket.identifiers, bucket.history, bucket.mask, *bucket.rope['q'], *bucket.rope['k'],
            *bucket.cached_history[0].values()]
        address = lambda operations, value: (value.untyped_storage().data_ptr(), value.untyped_storage().data_ptr() + 1)
        bucket.addresses = [address(operations, value) for value in bucket.inputs]
        prepared = PreparedDFlashProposal.__new__(PreparedDFlashProposal)
        prepared.operations, prepared.device, prepared.mesh = operations, device, device.mesh
        prepared.kv_history, prepared.owned = cache, bucket.inputs
        with patch('dflash_proposal_trace.addresses', side_effect=address), patch('dflash_device.addresses', side_effect=address), \
                patch('dflash_proposal_trace.release_owned'), patch('proposal_native_attention.validate_mask', wraps=validate_mask) as validate:
            prepared.update(bucket, 17)
            self.assertEqual(device.validated_native_proposal_masks, {address(operations, bucket.mask)})
            validate.assert_called_once()
            operations.copy_host_to_device_tensor.side_effect = RuntimeError('transfer failed')
            with self.assertRaisesRegex(RuntimeError, 'transfer failed'):
                prepared.update(bucket, 18)
            self.assertFalse(device.validated_native_proposal_masks)


class NativeProposalSuiteTests(unittest.TestCase):
    def test_simulator_preflight_fails_before_model_work(self):
        source = Path(__file__).with_name('baseline-suite.sh').read_text()
        start = source.index('if [ "${QWEN_DFLASH_NATIVE_PROPOSAL_ABBA:-0}" = 1 ]; then')
        end = source.index('if [ "${QWEN_DFLASH_LIVE_QUERY_ABBA:-0}" = 1 ]; then', start)
        body = source[start:end].replace('> /experiment/results/proposal-native-preflight.json', '')
        stub = 'set -euo pipefail\npython3() { echo "preflight $*"; return "$FAILURE"; }\n'
        for failure in ('0', '17'):
            result = subprocess.run(['bash', '-c', stub + body + '\necho model'], capture_output=True, text=True,
                env=dict(os.environ, QWEN_DFLASH_NATIVE_PROPOSAL_ABBA='1', QWEN_RUN_MODE='full-norm-engine',
                    QWEN_DFLASH_CONTEXT='4096', QWEN_DFLASH_CAPTURE='1', QWEN_DFLASH_DRAFTS='7', FAILURE=failure))
            self.assertEqual(result.returncode, int(failure), result.stderr)
            self.assertIn('proposal_native_request.py --metal-root /opt/tt-metal', result.stdout)
            self.assertEqual('model' in result.stdout.splitlines(), failure == '0')

    def test_ci_requires_complete_artifact_and_independent_policy_validation(self):
        source = Path(__file__).with_name('run-baseline.sh').read_text()
        start = source.index('if [ "$dflash_native_proposal_abba" = 1 ]; then')
        end = source.index('if [ "$dflash_target_links_abba" = 1 ]; then', start)
        body = source[start:end]
        stub = '''set -euo pipefail
dflash_native_proposal_abba=1
test_id=fixture
output=/unused
docker() { echo copy; [[ "$FAILURE" != copy ]] || return 17; }
python3() { echo "validate $*"; [[ "$FAILURE" != validation ]] || return 19; }
'''
        for failure, expected in (('', 0), ('copy', 17), ('validation', 19)):
            result = subprocess.run(['bash', '-c', stub + body], env=dict(os.environ, FAILURE=failure),
                capture_output=True, text=True)
            self.assertEqual(result.returncode, expected, result.stderr)
            self.assertEqual('proposal_native_request.py --hardware-result' in result.stdout, failure != 'copy')
        self.assertIn('-e "QWEN_DFLASH_NATIVE_PROPOSAL_ABBA=$dflash_native_proposal_abba"', source)

    def test_native_suite_routes_to_the_captured_four_k_request(self):
        source = Path(__file__).with_name('run-baseline.sh').read_text()
        end = source.index('if [ "$mode" = full-dflash-target-links-request ]; then')
        result = subprocess.run(['bash', '-c', source[:end] + '\nprintf "%s %s %s" "$mode" "$dflash_context" "$dflash_native_proposal_abba"'],
            env=dict(os.environ, QWEN_CARDS_ALLOCATED='1', QWEN_RUN_MODE='full-dflash-native-proposal-request'),
            capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout, 'full-dflash-trace-request 4096 1')


if __name__ == '__main__':
    unittest.main()
