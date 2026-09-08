from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'speculative-decoding' / 'harness'))
from full_request import measure_request, terminal_ids


class RequestPilotTests(unittest.TestCase):
    def test_terminal_ids_come_from_generation_config(self):
        for content, expected in (('{"eos_token_id": 2}', (2,)), ('{"eos_token_id": [2, 3]}', (2, 3))):
            with patch('full_request.Path.read_text', return_value=content):
                self.assertEqual(terminal_ids('/frozen-weights', 100), expected)
        for content in ('{}', '{"eos_token_id": true}', '{"eos_token_id": [100]}', '{"eos_token_id": []}'):
            with patch('full_request.Path.read_text', return_value=content):
                with self.assertRaises(ValueError):
                    terminal_ids('/frozen-weights', 100)

    def run_fixture(self, *, seed=0, eos_ids=(), wrong=False, norm_batch=False,
                    attention_replay=False, attention_mask_once=False, replay_group_rows=4, lookup_max_rows=32):
        def decode(token, position, trace):
            logits = torch.zeros(1, 100)
            logits[0, (token + 1) % 3] = 1
            return logits

        def factory(model, session, pages, helpers, sampler, norm_batch, attention_replay=False,
                    attention_mask_once=False, replay_group_rows=4, max_verify_rows=32):
            engine = SimpleNamespace(setup_ms=12.0, phase='idle', close=Mock(), buckets={1: {}, 2: {}, 4: {}})
            if attention_replay:
                from attention_request_plan import capture_plan
                plan = capture_plan(session.position, pages.shape[1] * 64, 32, 32)
                engine.proposal_rows = lambda: plan.max_rows(session.position, session.max_new_tokens - len(session.emitted))

            def verify(ticket):
                engine.phase = 'verified'
                return [99 if wrong else (token + 1) % 3 for token in ticket.tokens], dict(input_ms=0, verify_readback_ms=0)

            def publish(prefix):
                self.assertEqual(session.phase, 'committing')
                engine.phase = 'idle'

            engine.verify, engine.publish = verify, publish
            return engine

        with patch('full_request.VerifierEngine', side_effect=factory) as constructor:
            result = measure_request(SimpleNamespace(args=SimpleNamespace(vocab_size=100)), None,
                [0, 1, 2] * (1365 if attention_replay else 12), SimpleNamespace(shape=(1, 1024)), [],
                prefill=lambda prompt: seed, decode=decode,
                live_digest=lambda: 'state', kv_digest=lambda position: position,
                inactive_digest=lambda: 'inactive', eos_ids=eos_ids, max_new_tokens=33, norm_batch=norm_batch,
                attention_replay=attention_replay, family_routing=attention_replay,
                attention_mask_once=attention_mask_once, replay_group_rows=replay_group_rows,
                lookup_max_rows=lookup_max_rows)
        return result, constructor

    def test_lookup_cap_bounds_proposals_and_capture_configuration(self):
        for width in (1, 2, 4, 8, 16):
            result, constructor = self.run_fixture(norm_batch=True, lookup_max_rows=width)
            self.assertEqual(result['committed_decode_tokens'], 32)
            self.assertEqual(result['lookup_max_rows'], width)
            self.assertEqual(constructor.call_args.kwargs['max_verify_rows'], width)
            self.assertTrue(all(block['rows'] <= width for block in result['blocks']))

    def test_lookup_cap_rejects_invalid_or_replay_options_before_prefill(self):
        for configuration in (dict(lookup_max_rows=True), dict(lookup_max_rows=3),
                dict(lookup_max_rows=8, family_routing=True)):
            prefill = Mock()
            with self.assertRaises(ValueError):
                measure_request(None, None, None, None, None, prefill=prefill, decode=None,
                    live_digest=None, kv_digest=None, inactive_digest=None, **configuration)
            prefill.assert_not_called()

    def test_norm_batch_selection_is_forwarded_and_recorded(self):
        for enabled in (False, True):
            result, constructor = self.run_fixture(norm_batch=enabled)
            self.assertIs(result['norm_batch'], enabled)
            self.assertIs(constructor.call_args.kwargs['norm_batch'], enabled)

    def test_replay_geometry_is_forwarded_and_recorded(self):
        with patch.dict('os.environ', QWEN_SDPA_TREE_SCRATCH_ROUNDS='1'):
            for width in (4, 8):
                for shared in (False, True):
                    result, constructor = self.run_fixture(norm_batch=True, attention_replay=True,
                        attention_mask_once=shared, replay_group_rows=width)
                    self.assertEqual(result['replay_group_rows'], width)
                    self.assertIs(result['attention_mask_once'], shared)
                    self.assertEqual(constructor.call_args.kwargs['replay_group_rows'], width)
                    self.assertIs(constructor.call_args.kwargs['attention_mask_once'], shared)
                    self.assertEqual(result['committed_decode_tokens'], 32)

    def test_invalid_replay_geometry_does_not_prefill(self):
        with patch.dict('os.environ', {}, clear=True):
            for options in (dict(attention_mask_once=True), dict(replay_group_rows=8),
                            dict(replay_group_rows=True), dict(replay_group_rows=16),
                            dict(norm_batch=True, attention_replay=True, family_routing=True, replay_group_rows=8)):
                prefill = Mock()
                with self.subTest(options=options), self.assertRaises(ValueError):
                    measure_request(None, None, None, None, None, prefill=prefill, decode=None,
                        live_digest=None, kv_digest=None, inactive_digest=None, **options)
                prefill.assert_not_called()

    def test_actual_lookup_accounting_and_setup_are_separate(self):
        result, constructor = self.run_fixture()
        self.assertEqual(result['committed_decode_tokens'], 32)
        self.assertEqual(sum(block['committed'] for block in result['blocks']), 32)
        self.assertGreater(result['accepted'], 0)
        self.assertEqual(result['engine_setup_ms'], 12)
        self.assertEqual(result['post_seed_including_setup_ms'], result['decode_ms'] + 12)
        self.assertEqual(result['emitted'], [index % 3 for index in range(33)])
        self.assertEqual(result['prompt_tokens'], [0, 1, 2] * 12)
        for block in result['blocks']:
            self.assertEqual(len(block['input_tokens']), block['rows'])
            self.assertGreaterEqual(block['position'], len(result['prompt_tokens']))
            self.assertEqual(block['match_length'] > 0, block['source'] == 'lookup')
        constructor.assert_called_once()

    def test_terminal_prefill_has_no_decode_throughput(self):
        result, constructor = self.run_fixture(seed=2, eos_ids=(2,))
        self.assertEqual(result['committed_decode_tokens'], 0)
        self.assertIsNone(result['committed_tokens_per_second'])
        self.assertEqual(result['engine_setup_ms'], 0)
        constructor.assert_not_called()

    def test_accepted_terminal_state_matches_native_position(self):
        result, constructor = self.run_fixture(eos_ids=(2,))
        self.assertEqual(result['emitted'], [0, 1, 2])
        self.assertEqual(result['committed_decode_tokens'], 2)

    def test_wrong_target_output_cannot_produce_successful_result(self):
        with self.assertRaisesRegex(AssertionError, 'differs from native'):
            self.run_fixture(wrong=True)
