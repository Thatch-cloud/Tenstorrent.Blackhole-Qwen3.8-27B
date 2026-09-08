import os
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'speculative-decoding' / 'harness'))
from draft_selector import greedy_selector_reference, select_active_candidates
from full_dflash_request import load_dflash_fixtures, summarize_dflash_requests


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
        for suite in ('full-dflash-request', 'full-dflash-wide-request', 'full-dflash-trace-request', 'full-dflash-wide-trace-request'):
            environment = dict(os.environ, QWEN_RUN_MODE=suite, QWEN_CARDS_ALLOCATED='1',
                QWEN_LOOKUP_CAP_ABBA='1')
            result = subprocess.run(['bash', str(Path(__file__).with_name('run-baseline.sh'))],
                env=environment, capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertNotIn('docker', result.stdout + result.stderr)

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
