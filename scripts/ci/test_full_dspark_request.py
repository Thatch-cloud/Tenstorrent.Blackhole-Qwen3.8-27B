from contextlib import ExitStack, contextmanager, nullcontext
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'speculative-decoding' / 'harness'))
import full_dspark_request as request
from dspark_prefill import FeatureChunk


class FullDSparkRequestTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.operations = SimpleNamespace(get_device_tensors=lambda value: [value, value], to_torch=lambda value: value,
            DRAM_MEMORY_CONFIG='dram', clone=lambda value, **kwargs: value.clone(), deallocate=Mock())
        self.captures, self.decode_captures = [], []
        self.stack.enter_context(patch.object(request, 'FullHistoryCapture', side_effect=self.capture))
        self.stack.enter_context(patch.object(request, 'LayerOutputCapture', side_effect=self.decode_capture))
        self.drafter = SimpleNamespace(position=32, max_drafts=15, propose=Mock(return_value=tuple(range(15))),
            history=SimpleNamespace(capacity=128),
            prepare_publication=Mock(), commit_publication=Mock(), discard_publication=Mock(), close=Mock(),
            prepare_trace=Mock(), prepared=SimpleNamespace(checks=[]))
        self.device = self.stack.enter_context(patch.object(request, 'DSparkDevice', return_value=self.drafter))
        self.traced_device = self.stack.enter_context(patch('dspark_prepared_proposal.TracedDSparkDevice', return_value=self.drafter))
        self.history_audit = self.stack.enter_context(patch('dspark_history_audit.AuditedHistoryDrafter',
            return_value=self.drafter))
        self.base_prefill = Mock(return_value=17)
        self.base_decode = Mock(return_value=torch.zeros(1, 100))
        self.events = []

    def capture(self, operations, model, position):
        values = tuple(torch.full((1, 1, 32, 2560), tap, dtype=torch.bfloat16) for tap in range(5))
        result = SimpleNamespace(capture=lambda: nullcontext(), outputs=lambda: (FeatureChunk(0, 32, values),), close=Mock())
        self.captures.append(result)
        return result

    def decode_capture(self, *args, **kwargs):
        position = 32 + len(self.decode_captures)
        values = tuple(torch.full((1, 1, 1, 2560), position + tap, dtype=torch.bfloat16) for tap in range(5))
        result = SimpleNamespace(capture=lambda: nullcontext(), outputs=lambda: values, close=Mock())
        self.decode_captures.append(result)
        return result

    def measure(self, *, audit=False, corrupt=False, wrong_accounting=False, fail_factory=False,
            proposal_trace=False, commit_only_gdn=False, **experiment_options):
        def native_request(model, sampler, prompt, pages, helpers, **options):
            self.events.append('native-control')
            options['prefill'](prompt)
            options['decode'](17, 32, True)
            options['decode'](18, 33, True)
            self.events.append('candidate-prefill')
            options['prefill'](prompt)
            if fail_factory:
                self.device.side_effect = RuntimeError('injected history preparation failure')
            self.events.append('factory')
            runtime = options['feature_factory']()
            if proposal_trace:
                self.drafter.prepare_trace.assert_not_called()
                self.drafter.propose.assert_not_called()
                options['verifier_before_capture'](SimpleNamespace(initial=[]))
            self.assertEqual(options['feature_drafter_name'], 'dspark')
            self.assertEqual(options['lookup_max_rows'], 16)
            self.assertTrue(options['norm_batch'] and options['native_sampling_rows'])
            self.assertIs(options['commit_only_gdn'], commit_only_gdn)
            self.assertIs(options['audit_commit_only_gdn'], audit and commit_only_gdn)
            if proposal_trace and audit:
                self.drafter.prepared.checks.extend([dict(position=32, tensors=6, exact=True)] * 2)
            if audit:
                values = [torch.cat([capture.outputs()[tap] for capture in self.decode_captures], dim=2) for tap in range(5)]
                if corrupt:
                    values[0] = values[0].flip(2)
                runtime.validate_features(values, 2, 32)
            else:
                self.assertIsNone(runtime.validate_features)
            self.drafter.position = 34
            runtime.committed_feature_rows = 1 if wrong_accounting else 2
            return dict(blocks=[dict(committed=2, rows=3)], committed_decode_tokens=2, committed_tokens_per_second=123.0)

        with patch('full_request.measure_request', side_effect=native_request):
            return request.measure_dspark_request(self.operations, object(), object(), list(range(32)), object(), [],
                collectives=object(), parameters={}, layer_weights=[], predecessor=object(), successor=object(), rotary=object(),
                prefill=self.base_prefill, decode=self.base_decode, live_digest=Mock(), kv_digest=Mock(), inactive_digest=Mock(),
                eos_ids=(99,), audit_features=audit, max_new_tokens=65,
                proposal_trace=proposal_trace, commit_only_gdn=commit_only_gdn, **experiment_options)

    def score_scope(self):
        for target in ('target_t16_attention_gate.qualify', 'target_t16_attention_gate.validate_request_option',
                'dspark_native_fixed_gate.qualify', 'native_draft_sdpa.audit_active_kernel'):
            self.stack.enter_context(patch(target))
        self.stack.enter_context(patch.dict('os.environ', {'TT_METAL_HOME': '/unused-unit-test-runtime'}))
        events = []

        @contextmanager
        def install():
            events.append('install')
            try:
                yield
            finally:
                events.append('restore')

        arm = SimpleNamespace(install=install, summary=Mock(return_value={'restored': True, 'calls': 2}))
        constructor = self.stack.enter_context(patch('dspark_score_layout_scope.ScoreLayoutArm', return_value=arm))
        self.drafter.prepare_trace.side_effect = lambda *args, **kwargs: events.append('capture')
        self.drafter.close.side_effect = lambda: events.append('close')
        return events, arm, constructor

    def test_score_scope_installs_before_capture_and_restores_before_device_close(self):
        events, arm, constructor = self.score_scope()
        result = self.measure(proposal_trace=True, commit_only_gdn=True, native_attention=True,
            target_attention_t16=True, score_layout=True)
        self.assertEqual(events, ['install', 'capture', 'restore', 'close'])
        constructor.assert_called_once_with(self.drafter)
        arm.summary.assert_called_once()
        self.assertEqual(result['score_layout'], {'restored': True, 'calls': 2})

    def test_score_scope_restores_on_request_validation_failure(self):
        events, arm, constructor = self.score_scope()
        with self.assertRaisesRegex(AssertionError, 'publication'):
            self.measure(proposal_trace=True, commit_only_gdn=True, native_attention=True,
                target_attention_t16=True, score_layout=True, wrong_accounting=True)
        self.assertEqual(events, ['install', 'capture', 'restore', 'close'])
        arm.summary.assert_not_called()

    def test_score_scope_requires_explicit_compatible_request_policy(self):
        with self.assertRaisesRegex(ValueError, 'Score layout requires'):
            self.measure(score_layout=True)
        self.base_prefill.assert_not_called()

    def test_captured_variant_prepares_before_target_and_reports_complete_replay_audit(self):
        events = []
        self.history_audit.side_effect = lambda *args: (events.append('snapshot_initial_history') or self.drafter)
        self.drafter.prepare_trace.side_effect = lambda *args, **kwargs: events.append('capture_proposal')
        self.drafter.propose.side_effect = lambda *args: (events.append('warm_replay') or tuple(range(15)))
        result = self.measure(audit=True, proposal_trace=True, commit_only_gdn=True)
        self.assertEqual(events, ['snapshot_initial_history', 'capture_proposal', 'warm_replay'])
        self.device.assert_not_called()
        self.traced_device.assert_called_once()
        self.drafter.prepare_trace.assert_called_once_with(17, audit=True)
        self.assertTrue(result['dspark']['proposal_trace'])
        self.assertEqual(len(result['dspark']['proposal_checks']), 2)
        self.assertIsNone(result['committed_tokens_per_second'])

    def test_timed_capture_keeps_expensive_replay_comparisons_out_of_tg(self):
        result = self.measure(proposal_trace=True)
        self.drafter.prepare_trace.assert_called_once_with(17, audit=False)
        self.assertEqual(result['dspark']['proposal_checks'], [])
        self.assertEqual(result['committed_tokens_per_second'], 123.0)

    def test_history_publication_slot_does_not_shadow_trace_owner(self):
        underlying = self.drafter
        class HistoryProxy:
            prepared = None

            def __getattr__(self, name):
                return getattr(underlying, name)

        self.history_audit.return_value = HistoryProxy()
        result = self.measure(audit=True, proposal_trace=True)
        self.assertEqual(len(result['dspark']['proposal_checks']), 2)

    def test_both_prefills_precede_history_setup_and_warm_proposal_is_charged_to_factory(self):
        result = self.measure()
        self.assertEqual(self.events, ['native-control', 'candidate-prefill', 'factory'])
        self.assertEqual(self.base_prefill.call_count, 2)
        self.drafter.propose.assert_called_once_with(17, 15)
        self.assertEqual(self.device.call_args.kwargs, dict(position=32, proposals=15, history_capacity=128))
        self.assertEqual(result['dspark']['committed_feature_rows'], 2)
        self.assertEqual(result['dspark']['final_position'], 34)
        self.assertEqual(len(result['dspark']['prefill_chunks']), 2)
        self.assertFalse(result['instrumented_timing'])
        self.assertEqual(result['committed_tokens_per_second'], 123.0)
        self.history_audit.assert_not_called()
        self.drafter.close.assert_called_once()
        self.assertTrue(all(capture.close.called for capture in self.captures))

    def test_instrumented_request_checks_all_taps_both_chips_and_cannot_claim_tg(self):
        result = self.measure(audit=True)
        self.assertTrue(result['instrumented_timing'])
        self.assertIsNone(result['committed_tokens_per_second'])
        self.assertEqual(len(result['dspark']['feature_checks']), 10)
        self.assertEqual(len(result['dspark']['prefill_hashes']), 10)
        self.history_audit.assert_called_once()
        self.assertTrue(all(call.args[-1] is False for call in self.base_decode.call_args_list))
        self.assertTrue(all(capture.close.called for capture in self.decode_captures))

    def test_reordered_actual_features_fail_before_success_and_release_request_resources(self):
        with self.assertRaisesRegex(AssertionError, 'committed feature mismatch'):
            self.measure(audit=True, corrupt=True)
        self.drafter.close.assert_called_once()
        self.assertTrue(all(capture.close.called for capture in self.captures))

    def test_missing_committed_feature_accounting_rejects_the_result(self):
        with self.assertRaisesRegex(AssertionError, 'publication'):
            self.measure(wrong_accounting=True)
        self.drafter.close.assert_called_once()

    def test_failed_history_setup_releases_prefill_but_not_an_unconstructed_drafter(self):
        with self.assertRaisesRegex(RuntimeError, 'history preparation'):
            self.measure(fail_factory=True)
        self.assertTrue(all(capture.close.called for capture in self.captures))
        self.drafter.close.assert_not_called()


if __name__ == '__main__':
    unittest.main()
