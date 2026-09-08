import copy
import os
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'speculative-decoding' / 'harness'))
from draft_selector import greedy_selector_reference, select_active_candidates
from dflash_prefill_window import prefill_window
from full_dflash_request import (load_dflash_fixtures, summarize_dflash_requests,
    summarize_dflash_commit_requests, summarize_dflash_convolution_requests)


class FullDFlashRequestTests(unittest.TestCase):
    def requests(self):
        output = []
        for audit, duration in ((True, 100000), (False, 40), (False, 60)):
            output.append(dict(instrumented_timing=audit, exact=True, state_exact=True, inactive_exact=True,
                ended_with_eos=True, prompt_tokens=[10, 11], emitted=[12, 13, 14], max_new_tokens=513,
                eos_ids=[14], vocab_size=248320, committed_decode_tokens=2, selected_drafter='dflash2',
                sampler_num_links=4, fabric_sources={'audited': 'source'}, blocks=[{'committed': 2}],
                dflash=dict(committed_feature_rows=2, proposal_calls=1, feature_checks=[{'exact': True, 'rows': 2}] * 10 if audit else []),
                decode_ms=duration, prefill_ms=10, engine_setup_ms=20, feature_setup_ms=30,
                prefill_setup_decode_ms=duration + 60, setup_amortized=False, cross_request_trace_reuse=False,
                committed_tokens_per_second=None if audit else 2000 / duration))
        return output

    def test_only_complete_uninstrumented_requests_contribute_to_throughput(self):
        result = summarize_dflash_requests(self.requests())
        self.assertEqual(result['committed_tokens_per_second'], 40)
        self.assertEqual(result['committed_tokens'], 4)
        self.assertEqual(result['prefill_setup_decode_ms'], [100, 120])
        self.assertFalse(result['target_reached'])

    def test_long_context_requires_both_prefill_tail_audits_with_absolute_positions(self):
        for mutation in (None, 'missing', 'start', 'measured', 'window'):
            records = self.requests()
            window = prefill_window(4093)
            for record in records:
                record['prompt_tokens'] = [10] * 4093
                record['dflash']['prefill_window'] = dict(window)
                record['dflash']['prefill_checks'] = [dict(chip=chip, **window, exact=True)
                    for prefill in range(2) for tap in range(5) for chip in range(2)] if record['instrumented_timing'] else []
            if mutation == 'missing':
                records[0]['dflash']['prefill_checks'].pop()
            elif mutation == 'start':
                records[0]['dflash']['prefill_checks'][0]['start'] = 0
            elif mutation == 'measured':
                records[1]['dflash']['prefill_checks'] = records[0]['dflash']['prefill_checks']
            elif mutation == 'window':
                records[2]['dflash']['prefill_window']['rows'] = 4093
            if mutation is None:
                self.assertEqual(summarize_dflash_requests(records)['benchmark']['ctx_tokens'], 4093)
            else:
                with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                    summarize_dflash_requests(records)

    def test_pp_ctx_tg_use_actual_prompt_and_time_weighted_uninstrumented_samples(self):
        records = self.requests()
        records[0]['prefill_ms'] = 100000
        records[1]['prefill_ms'] = 20
        records[2]['prefill_ms'] = 60
        metrics = summarize_dflash_requests(records)['benchmark']
        self.assertEqual(metrics['pp_tokens_per_second'], 50)
        self.assertEqual(metrics['per_request_pp_tokens_per_second'], [100, 100 / 3])
        self.assertEqual(metrics['ctx_tokens'], 2)
        self.assertEqual(metrics['tg_tokens_per_second'], 40)
        self.assertEqual(metrics['streams'], 1)
        self.assertEqual(metrics['committed_decode_tokens_per_request'], [2, 2])
        self.assertEqual(metrics['mean_prefill_ms'], 40)
        self.assertEqual(metrics['mean_prefill_setup_decode_ms'], 110)

    def test_nonfinite_or_zero_benchmark_timings_are_rejected(self):
        for key in ('prefill_ms', 'decode_ms', 'prefill_setup_decode_ms'):
            for value in (0, -1, True, float('nan'), float('inf')):
                records = self.requests()
                records[1][key] = value
                with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                    summarize_dflash_requests(records)

    def commit_requests(self):
        control = self.requests()
        for record in control:
            record.update(commit_only_gdn=False, sources={'runtime': 'hash'}, gdn_verify_checks=[])
            record['blocks'][0].update(rows=8, source='dflash2', accepted=1, match_length=0,
                position=2, input_tokens=[12, 13, 15, 16, 17, 18, 19, 20])
            record['dflash'].update(checkpoints={'revision': 'pinned'}, block_rows=8, proposal_capture=True,
                proposal_contexts=[256, 512], target_taps=[5, 19, 33, 47, 61], policy='qualified',
                proposal_trace_checks=[dict(exact=True)] if record['instrumented_timing'] else [])
        candidate = copy.deepcopy(control)
        for record in candidate:
            record['commit_only_gdn'] = True
            if record['instrumented_timing']:
                record['gdn_verify_checks'] = [dict(position=2, rows=8, unchanged=True)]
            else:
                record['decode_ms'] /= 2
                record['prefill_setup_decode_ms'] = record['decode_ms'] + 60
                record['committed_tokens_per_second'] *= 2
        return [control[0], candidate[0], control[1], candidate[1], candidate[2], control[2]]

    def test_commit_only_abba_excludes_both_audits_and_preserves_setup_costs(self):
        result = summarize_dflash_commit_requests(self.commit_requests())
        self.assertEqual(result['control']['committed_tokens_per_second'], 40)
        self.assertEqual(result['candidate']['committed_tokens_per_second'], 80)
        self.assertEqual(result['candidate_over_control'], 2)
        self.assertEqual(result['candidate']['prefill_setup_decode_ms'], [80, 90])
        self.assertEqual(result['measured_order'], ['control', 'candidate', 'candidate', 'control'])
        self.assertFalse(result['target_reached'])

    def convolution_requests(self):
        records = self.commit_requests()
        for index, record in enumerate(records):
            record.update(commit_only_gdn=True, fused_convolution=index in (1, 3, 4))
            record['gdn_verify_checks'] = [dict(position=2, rows=8, unchanged=True)] if record['instrumented_timing'] else []
            record['dflash']['convolution_checks'] = [dict(position=2, layer=layer, chip=chip, rows=32, exact=True)
                for layer in range(5) for convolution in range(4) for chip in range(2)] if index == 1 else []
        return records

    def test_convolution_abba_retains_commit_only_and_audits_all_learned_calls(self):
        result = summarize_dflash_convolution_requests(self.convolution_requests())
        self.assertEqual(result['committed_tokens_per_second'], 80)
        self.assertEqual(result['candidate_over_control'], 2)
        self.assertTrue(result['candidate']['fused_convolution'])
        self.assertFalse(result['control']['fused_convolution'])
        self.assertTrue(result['control']['commit_only_gdn'])

    def test_convolution_abba_rejects_missing_wrong_or_instrumented_checks(self):
        for mutation in ('missing', 'layer', 'chip', 'exact', 'rows', 'instrumented', 'verifier'):
            records = self.convolution_requests()
            checks = records[1]['dflash']['convolution_checks']
            if mutation == 'missing':
                checks.pop()
            elif mutation in ('layer', 'chip', 'rows', 'exact'):
                checks[0][mutation] = False if mutation == 'exact' else 7
            elif mutation == 'instrumented':
                records[3]['dflash']['convolution_checks'] = copy.deepcopy(checks)
            else:
                records[4]['commit_only_gdn'] = False
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                summarize_dflash_convolution_requests(records)

    def test_commit_only_abba_rejects_a_second_optimization_in_candidate_arm(self):
        records = self.commit_requests()
        records[3]['fused_convolution'] = True
        with self.assertRaises(ValueError):
            summarize_dflash_commit_requests(records)

    def test_commit_only_abba_rejects_order_instrumentation_or_different_proposals(self):
        for index, key, value in ((3, 'commit_only_gdn', False), (4, 'instrumented_timing', True),
                (4, 'sources', {'runtime': 'other'}), (1, 'emitted', [12, 17, 14]),
                (3, 'committed_tokens_per_second', 200), (1, 'gdn_verify_checks', []),
                (3, 'gdn_verify_checks', [dict(position=2, rows=8, unchanged=True)])):
            records = self.commit_requests()
            records[index][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                summarize_dflash_commit_requests(records)
        for field, value in (('accepted', 0), ('input_tokens', [12] * 8), ('position', 3)):
            records = self.commit_requests()
            records[3]['blocks'][0][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                summarize_dflash_commit_requests(records)

    def test_commit_only_audit_rejects_changed_state_or_wrong_prefix_coverage(self):
        for check in (dict(position=2, rows=8, unchanged=False), dict(position=3, rows=8, unchanged=True),
                dict(position=2, rows=4, unchanged=True)):
            records = self.commit_requests()
            records[1]['gdn_verify_checks'] = [check]
            with self.subTest(check=check), self.assertRaises(ValueError):
                summarize_dflash_commit_requests(records)

    def test_rejects_missing_audit_or_incomplete_or_changed_request(self):
        for index, key, value in ((0, 'instrumented_timing', False), (1, 'ended_with_eos', False),
                (1, 'state_exact', False), (1, 'sampler_num_links', 1), (2, 'emitted', [12, 15, 14]),
                (2, 'decode_ms', float('nan')), (1, 'feature_setup_ms', 0), (0, 'committed_tokens_per_second', 200)):
            records = self.requests()
            records[index][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                summarize_dflash_requests(records)
        records = self.requests()
        records[1]['dflash']['committed_feature_rows'] = 1
        with self.assertRaises(ValueError):
            summarize_dflash_requests(records)

    def test_loads_all_five_pinned_layers_without_normalizing_global_keys(self):
        modules = ('attention', 'convolution', 'mlp', 'projection', 'selector', 'layer')
        with patch.multiple('full_dflash_request', **{f'load_{name}': unittest.mock.DEFAULT for name in modules}) as loaders:
            for name in ('attention', 'convolution', 'mlp', 'selector'):
                loaders[f'load_{name}'].return_value = ({'revision': 'pinned'}, {name: name})
            loaders['load_projection'].return_value = ({'revision': 'pinned'}, 'fc', 'norm')
            loaders['load_layer'].side_effect = lambda path, layer: ({'layer': layer},
                {f'layers.{layer}.q_proj.weight': layer, 'norm.weight': 'global'})
            manifests, layers, projection, selector = load_dflash_fixtures('/fixtures')
        self.assertEqual(len(layers), 5)
        self.assertEqual(len(manifests['layers']), 4)
        self.assertEqual(projection, {'fc.weight': 'fc', 'hidden_norm.weight': 'norm'})
        for layer in range(1, 5):
            self.assertEqual(layers[layer][0], {'layers.0.q_proj.weight': layer, 'norm.weight': 'global'})
        self.assertEqual(loaders['load_layer'].call_count, 4)

    def test_captured_request_requires_exact_proposer_audit_and_matching_contexts(self):
        records = self.requests()
        for record in records:
            record['dflash'].update(proposal_capture=True, proposal_contexts=[256, 512],
                proposal_trace_checks=[dict(exact=True)] if record['instrumented_timing'] else [])
        self.assertTrue(summarize_dflash_requests(records)['proposal_capture'])
        records[0]['dflash']['proposal_trace_checks'][0]['exact'] = False
        with self.assertRaises(ValueError):
            summarize_dflash_requests(records)
        records[0]['dflash']['proposal_trace_checks'][0]['exact'] = True
        records[2]['dflash']['proposal_contexts'] = [256]
        with self.assertRaises(ValueError):
            summarize_dflash_requests(records)

    def test_active_codebook_selection_preserves_full_fp64_result_and_ties(self):
        generator = torch.Generator().manual_seed(17)
        hidden = torch.randn((2, 7, 4), generator=generator)
        candidates = torch.stack([torch.randperm(80, generator=generator)[:16] for row in range(14)]).reshape(2, 7, 16)
        unary = torch.randn((2, 7, 16), generator=generator)
        predecessor = torch.randn((80, 4), generator=generator).double()
        successor = torch.randn((80, 4), generator=generator).double()
        anchors = torch.tensor([78, 79])
        for tied in (False, True):
            operands = (hidden, candidates, torch.zeros_like(unary) if tied else unary,
                torch.zeros_like(predecessor) if tied else predecessor, successor, anchors)
            actual = select_active_candidates(*operands)
            expected = greedy_selector_reference(*operands)
            self.assertTrue(all(torch.equal(left, right) for left, right in zip(actual, expected)))

    def test_invalid_dflash_suite_options_stop_before_fixture_or_device_access(self):
        for suite in ('full-dflash-4k-request', 'full-dflash-request', 'full-dflash-wide-request', 'full-dflash-trace-request',
                'full-dflash-wide-trace-request', 'full-dflash-commit-request', 'full-dflash-convolution-request'):
            environment = dict(os.environ, QWEN_RUN_MODE=suite, QWEN_CARDS_ALLOCATED='1',
                QWEN_LOOKUP_CAP_ABBA='1')
            result = subprocess.run(['bash', str(Path(__file__).with_name('run-baseline.sh'))],
                env=environment, capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertNotIn('docker', result.stdout + result.stderr)

    def test_long_context_request_cannot_enable_an_unqualified_runtime_combination(self):
        for context, capture, abba in (('8192', '1', '0'), ('4096', '0', '0'), ('4096', '1', '1')):
            environment = dict(os.environ, QWEN_DFLASH_CONTEXT=context, QWEN_DFLASH_CAPTURE=capture,
                QWEN_DFLASH_DRAFTS='7', QWEN_DFLASH_COMMIT_ABBA=abba,
                QWEN_HARDWARE_TESTS='1', QWEN_CARDS_ALLOCATED='1')
            result = subprocess.run([sys.executable, '-B', str(Path(__file__).with_name('full-prefix.py')),
                '--request-pilot', '--norm-batch'], env=environment, capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('4K qualification requires', result.stderr)

    def test_wide_selector_preserves_predecessors_across_oracle_chunk_boundaries(self):
        generator = torch.Generator().manual_seed(29)
        hidden = torch.randn((2, 31, 4), generator=generator).bfloat16()
        candidates = torch.stack([torch.randperm(80, generator=generator)[:16] for row in range(62)]).reshape(2, 31, 16)
        unary = torch.randn((2, 31, 16), generator=generator)
        predecessors = torch.randn((80, 4), generator=generator).double()
        successors = torch.randn((80, 4), generator=generator).double()
        anchors = torch.tensor([78, 79])
        selected, scores = select_active_candidates(hidden, candidates, unary, predecessors, successors, anchors)
        previous = anchors
        expected_tokens, expected_scores = [], []
        for position in range(31):
            edges = (predecessors[previous, None] * hidden.double()[:, position, None]
                * successors[candidates[:, position]]).sum(-1)
            row_scores = unary.double()[:, position] + edges
            previous = candidates[:, position].gather(1, row_scores.argmax(-1, keepdim=True)).squeeze(1)
            expected_tokens.append(previous)
            expected_scores.append(row_scores)
        self.assertTrue(torch.equal(selected, torch.stack(expected_tokens, dim=1)))
        self.assertTrue(torch.equal(scores, torch.stack(expected_scores, dim=1)))

    def test_selector_rejects_mismatched_or_unbounded_full_block(self):
        hidden = torch.ones((1, 31, 4))
        candidates = torch.zeros((1, 31, 1), dtype=torch.int64)
        unary = torch.zeros((1, 31, 1))
        codes = torch.ones((2, 4))
        anchors = torch.tensor([1])
        for projected, identifiers, scores in ((hidden, candidates[:, :30], unary[:, :30]),
                (hidden, candidates, unary[:, :30]), (hidden[:, :7], candidates, unary),
                (torch.ones((1, 32, 4)), candidates, unary)):
            with self.assertRaises(ValueError):
                select_active_candidates(projected, identifiers, scores, codes, codes, anchors)
