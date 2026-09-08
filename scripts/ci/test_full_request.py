from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'speculative-decoding' / 'harness'))
from full_request import RequestMismatch, check_committed_prefix, measure_request, terminal_ids


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
                    attention_replay=False, attention_mask_once=False, replay_group_rows=4, lookup_max_rows=32,
                    neural=None, selected_drafter=None, lookup_enabled=True, mtp_runtime=None,
                    mtp_factory=None, prefill=None, progress=None, native_sampling_rows=False, short_context=False,
                    attention_audit=False, feature_factory=None, commit_only_gdn=False,
                    audit_commit_only_gdn=False, live_digest=None, verify_hook=None):
        def decode(token, position, trace):
            logits = torch.zeros(1, 100)
            logits[0, (token + 1) % 3] = 1
            return logits

        def factory(model, session, pages, helpers, sampler, norm_batch, attention_replay=False,
                    attention_mask_once=False, replay_group_rows=4, max_verify_rows=32, retain_mtp_hidden=False,
                    native_sampling_rows=False, short_context=False, attention_audit=False, retain_feature_taps=(),
                    commit_only_gdn=False):
            engine = SimpleNamespace(setup_ms=12.0, phase='idle', close=Mock(), buckets={1: {}, 2: {}, 4: {}})
            engine.retain_mtp_hidden = retain_mtp_hidden
            engine.retain_feature_taps, engine.session = retain_feature_taps, session
            engine.verified_features_for_publication = lambda ticket: ('features',)
            engine.verified_mtp_hidden_for_publication = lambda ticket: [[token] for token in ticket.tokens]
            if attention_replay:
                from attention_request_plan import capture_plan
                plan = capture_plan(session.position, pages.shape[1] * 64, 32, 32,
                    max_verify_rows=max_verify_rows, short_context=short_context)
                engine.proposal_rows = lambda: plan.max_rows(session.position, session.max_new_tokens - len(session.emitted))

            def verify(ticket):
                if verify_hook is not None:
                    verify_hook()
                engine.phase = 'verified'
                engine.pending = ticket
                return [99 if wrong else (token + 1) % 3 for token in ticket.tokens], dict(input_ms=0, verify_readback_ms=0)

            def publish(prefix):
                self.assertEqual(session.phase, 'committing')
                engine.phase = 'idle'

            engine.verify, engine.publish = verify, publish
            return engine

        with patch('full_request.VerifierEngine', side_effect=factory) as constructor:
            result = measure_request(SimpleNamespace(args=SimpleNamespace(vocab_size=100)), object() if native_sampling_rows else None,
                ([0, 1, 2] * 56 + [0, 1]) if short_context else [0, 1, 2] * (1365 if attention_replay else 12),
                SimpleNamespace(shape=(1, 1024)), [],
                prefill=prefill or (lambda prompt: seed), decode=decode,
                live_digest=live_digest or (lambda: 'state'), kv_digest=lambda position: position,
                inactive_digest=lambda: 'inactive', eos_ids=eos_ids, max_new_tokens=33, norm_batch=norm_batch,
                attention_replay=attention_replay, family_routing=attention_replay or short_context,
                attention_mask_once=attention_mask_once, replay_group_rows=replay_group_rows,
                lookup_max_rows=lookup_max_rows, neural=neural, selected_drafter=selected_drafter,
                lookup_enabled=lookup_enabled, mtp_runtime=mtp_runtime, mtp_factory=mtp_factory, progress=progress,
                native_sampling_rows=native_sampling_rows, short_context=short_context, attention_audit=attention_audit,
                feature_factory=feature_factory, commit_only_gdn=commit_only_gdn,
                audit_commit_only_gdn=audit_commit_only_gdn)
        return result, constructor

    def test_commit_only_audit_checks_every_multirow_before_decision_and_excludes_timing(self):
        result, constructor = self.run_fixture(norm_batch=True, commit_only_gdn=True, audit_commit_only_gdn=True)
        self.assertTrue(constructor.call_args.kwargs['commit_only_gdn'])
        expected = [dict(position=block['position'], rows=block['rows'], unchanged=True)
            for block in result['blocks'] if block['rows'] > 1]
        self.assertTrue(expected)
        self.assertEqual(result['gdn_verify_checks'], expected)
        self.assertIsNone(result['committed_tokens_per_second'])
        self.assertTrue(result['instrumented_timing'])

    def test_commit_only_measured_request_has_no_hidden_state_audit(self):
        result, constructor = self.run_fixture(norm_batch=True, commit_only_gdn=True)
        self.assertEqual(result['gdn_verify_checks'], [])
        self.assertFalse(result['instrumented_timing'])
        self.assertGreater(result['committed_tokens_per_second'], 0)

    def test_commit_only_rejects_premature_native_state_mutation(self):
        state = []
        with self.assertRaisesRegex(AssertionError, 'before the decision'):
            self.run_fixture(norm_batch=True, commit_only_gdn=True, audit_commit_only_gdn=True,
                live_digest=lambda: tuple(state), verify_hook=lambda: state.append('mutated'))

    def test_commit_only_audit_requires_explicit_candidate(self):
        with self.assertRaises(ValueError):
            self.run_fixture(audit_commit_only_gdn=True)

    def test_feature_drafter_owns_neural_route_and_prefix_publication(self):
        from dflash_request_runtime import DFlashRequestRuntime, TARGET_TAPS

        for rows in (8, 32):
            drafter = SimpleNamespace(position=36, max_drafts=rows - 1,
                propose=lambda seed, count: tuple((seed + offset + 1) % 3 for offset in range(count)),
                prepare_publication=lambda features, prefix, **kwargs: prefix,
                discard_publication=Mock())
            def commit(prefix):
                drafter.position += prefix
            drafter.commit_publication = commit
            runtime = DFlashRequestRuntime(drafter, position=36)
            factory = Mock(return_value=runtime)
            result, constructor = self.run_fixture(feature_factory=factory, lookup_max_rows=rows)
            self.assertEqual(result['selected_drafter'], 'dflash2')
            self.assertEqual(result['drafting_policy'], 'neural-with-target-fallback')
            self.assertEqual(constructor.call_args.kwargs['retain_feature_taps'], TARGET_TAPS)
            self.assertNotIn('retain_mtp_hidden', constructor.call_args.kwargs)
            self.assertEqual(runtime.committed_feature_rows, 32)
            self.assertEqual(drafter.position, 68)
            self.assertGreater(result['feature_setup_ms'], 0)
            self.assertEqual(result['mtp_setup_ms'], 0)
            self.assertEqual(max(block['rows'] for block in result['blocks']), rows)
            factory.assert_called_once()

    def test_instrumented_attention_cannot_claim_decode_throughput(self):
        result, constructor = self.run_fixture(norm_batch=True, native_sampling_rows=True,
            lookup_max_rows=8, short_context=True, attention_replay=True, attention_mask_once=True,
            attention_audit=True)
        self.assertIsNone(result['committed_tokens_per_second'])
        self.assertTrue(result['instrumented_timing'])
        self.assertTrue(constructor.call_args.kwargs['attention_audit'])
        with self.assertRaises(ValueError):
            self.run_fixture(attention_audit=True)

    def test_short_attention_compares_matched_native_row_t8_requests(self):
        records = []
        for enabled in (False, True):
            result, constructor = self.run_fixture(norm_batch=True, native_sampling_rows=True,
                lookup_max_rows=8, short_context=True, attention_replay=enabled, attention_mask_once=enabled)
            self.assertIs(constructor.call_args.kwargs['short_context'], True)
            self.assertIs(result['short_context'], True)
            self.assertIs(result['family_routing'], True)
            self.assertEqual(result['length'], 170)
            records.append(result)
        for key in ('emitted', 'accepted', 'proposed', 'committed_decode_tokens'):
            self.assertEqual(records[0][key], records[1][key])

    def test_native_sampling_rows_reaches_engine_without_changing_proposals(self):
        control, _ = self.run_fixture()
        candidate, constructor = self.run_fixture(native_sampling_rows=True)
        self.assertIs(constructor.call_args.kwargs['native_sampling_rows'], True)
        self.assertIs(candidate['native_sampling_rows'], True)
        for key in ('emitted', 'proposed', 'accepted', 'committed_decode_tokens'):
            self.assertEqual(candidate[key], control[key])

    def test_mtp_factory_runs_after_second_prefill_and_is_charged(self):
        from mtp_request_runtime import MTPRequestRuntime

        prefill = Mock(return_value=0)
        progress = Mock()
        def factory():
            self.assertEqual(prefill.call_count, 2)
            return MTPRequestRuntime(lambda token, hidden, position, select: ([token], (token + 1) % 3 if select else None),
                [0], copy_hidden=lambda source, destination: destination.__setitem__(0, source[0]),
                verified_row=lambda rows, index: rows[index], max_drafts=3)
        factory = Mock(side_effect=factory)
        result, constructor = self.run_fixture(mtp_factory=factory, prefill=prefill, progress=progress,
            norm_batch=True, lookup_max_rows=4)
        factory.assert_called_once_with()
        self.assertEqual(prefill.call_count, 2)
        self.assertEqual(progress.call_count, len(result['blocks']))
        self.assertGreater(result['mtp_setup_ms'], 0)
        self.assertEqual(result['post_seed_including_setup_ms'],
            result['mtp_setup_ms'] + result['engine_setup_ms'] + result['decode_ms'])
        self.assertEqual(result['committed_decode_tokens'], 32)
        self.assertTrue(constructor.call_args.kwargs['retain_mtp_hidden'])

    def test_invalid_mtp_factory_rejected_before_prefill(self):
        for options in (dict(mtp_factory=1), dict(mtp_factory=Mock(), mtp_runtime=Mock()),
                dict(mtp_factory=Mock(), selected_drafter='mtp'), dict(mtp_factory=Mock(), engine_factory=Mock()),
                dict(mtp_factory=Mock(), neural={'other': Mock()})):
            prefill = Mock()
            with self.assertRaises(ValueError):
                measure_request(None, None, None, None, None, prefill=prefill, decode=None,
                    live_digest=None, kv_digest=None, inactive_digest=None, **options)
            prefill.assert_not_called()

    def test_mtp_bridge_runs_proposals_catchup_and_complete_exact_request(self):
        from mtp_request_runtime import MTPRequestRuntime

        calls = []
        def step(token, hidden, position, *, select):
            calls.append((position, select))
            return [token], (token + 1) % 3 if select else None
        runtime = MTPRequestRuntime(step, [0], copy_hidden=lambda source, destination: destination.__setitem__(0, source[0]),
                                    verified_row=lambda rows, index: rows[index], max_drafts=3)
        result, constructor = self.run_fixture(mtp_runtime=runtime, norm_batch=True, lookup_max_rows=4)
        self.assertEqual(result['committed_decode_tokens'], 32)
        self.assertEqual(result['selected_drafter'], 'mtp')
        self.assertTrue(result['exact'] and result['state_exact'])
        self.assertEqual(sum(not select for _, select in calls), 32)
        self.assertEqual(runtime.position, 36 + 32)
        self.assertTrue(constructor.call_args.kwargs['retain_mtp_hidden'])
        self.assertTrue(all(block['source'] == 'mtp' for block in result['blocks']))

    def test_lookup_cap_bounds_proposals_and_capture_configuration(self):
        for width in (1, 2, 4, 8, 16):
            result, constructor = self.run_fixture(norm_batch=True, lookup_max_rows=width)
            self.assertEqual(result['committed_decode_tokens'], 32)
            self.assertEqual(result['lookup_max_rows'], width)
            self.assertEqual(constructor.call_args.kwargs['max_verify_rows'], width)
            self.assertTrue(all(block['rows'] <= width for block in result['blocks']))

    def test_mtp_reuse_keeps_native_outputs_through_accept_reject_and_eos(self):
        from mtp_request_runtime import MTPRequestRuntime

        for wrong_drafts, eos_ids in ((False, ()), (True, ()), (False, (2,)), (True, (2,))):
            for reuse in (False, True):
                calls = []
                def step(token, hidden, position, *, select):
                    calls.append((position, select))
                    increment = 2 if wrong_drafts and position % 3 == 0 else 1
                    return [token], (token + increment) % 3 if select else None
                runtime = MTPRequestRuntime(step, [0],
                    copy_hidden=lambda source, destination: destination.__setitem__(0, source[0]),
                    verified_row=lambda rows, index: rows[index], max_drafts=3, reuse_accepted_cache=reuse)
                result, _ = self.run_fixture(mtp_runtime=runtime, norm_batch=True, lookup_max_rows=4, eos_ids=eos_ids)
                self.assertTrue(result['exact'] and result['state_exact'] and result['inactive_exact'])
                accounting = runtime.cache_accounting
                self.assertEqual(sum(accounting.values()), result['committed_decode_tokens'])
                self.assertEqual(accounting['teacher_forced_rows'], sum(not select for _, select in calls))
                if reuse:
                    self.assertGreater(accounting['reused_rows'], 0)
                    self.assertLess(accounting['teacher_forced_rows'], result['committed_decode_tokens'])
                else:
                    self.assertEqual(accounting['reused_rows'], 0)

    def test_neural_adapter_runs_inside_verified_request_loop(self):
        def propose(request_id, history, count):
            self.assertEqual(request_id, 'lookup-pilot')
            return [(history[-1] + offset + 1) % 3 for offset in range(count)]
        adapter = Mock(side_effect=propose)
        result, constructor = self.run_fixture(neural={'fixture': adapter}, selected_drafter='fixture',
            norm_batch=True, lookup_max_rows=8, lookup_enabled=False)
        adapter.assert_called()
        self.assertEqual(result['committed_decode_tokens'], 32)
        self.assertTrue(result['exact'] and result['state_exact'] and result['inactive_exact'])
        self.assertTrue(any(block['source'] == 'fixture' for block in result['blocks']))
        self.assertEqual(result['selected_drafter'], 'fixture')
        self.assertEqual(result['drafting_policy'], 'neural-with-target-fallback')

    def test_invalid_neural_registration_fails_before_prefill(self):
        for options in (dict(selected_drafter='missing'), dict(neural={'fixture': Mock()}), dict(lookup_enabled=False),
                dict(neural={'fixture': None}, selected_drafter='fixture')):
            prefill = Mock()
            with self.assertRaises(ValueError):
                measure_request(None, None, None, None, None, prefill=prefill, decode=None,
                    live_digest=None, kv_digest=None, inactive_digest=None, **options)
            prefill.assert_not_called()

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
        progress = Mock()
        with self.assertRaisesRegex(AssertionError, 'differs from native'):
            self.run_fixture(wrong=True, progress=progress)
        progress.assert_not_called()

    def test_prefix_guard_keeps_first_bad_block_evidence(self):
        block = dict(position=307, input_tokens=[1, 2], predictions=[5, 6])
        check_committed_prefix([1, 2], [1, 2, 3], block)
        for actual, expected, index in (([1, 9], [1, 2, 3], 1), ([1, 2], [1], 1)):
            with self.assertRaises(RequestMismatch) as caught:
                check_committed_prefix(actual, expected, block)
            self.assertEqual(caught.exception.evidence['token_index'], index)
            self.assertEqual(caught.exception.evidence['block'], block)
            self.assertEqual(caught.exception.evidence['actual_prefix'], actual)
