from types import SimpleNamespace
import unittest
from unittest.mock import Mock

from mtp_request_runtime import MTPRequestRuntime


class MTPRequestRuntimeTests(unittest.TestCase):
    def fixture(self, *, reuse_accepted_cache=False):
        anchor = [100]
        state = {}
        calls = []
        def step(token, hidden, position, *, select):
            state[position] = (token, hidden[0])
            calls.append((token, hidden[0], position, select))
            return [hidden[0] + 1000], token + 1 if select else None
        runtime = MTPRequestRuntime(step, anchor, copy_hidden=lambda source, destination: destination.__setitem__(0, source[0]),
                                    verified_row=lambda rows, index: rows[index], max_drafts=3,
                                    reuse_accepted_cache=reuse_accepted_cache)
        session = SimpleNamespace(request_id='request', phase='idle', position=10, seed=20, vocab_size=100, pending=None)
        engine = SimpleNamespace(retain_mtp_hidden=True, phase='idle', publish=Mock(),
                                 verified_mtp_hidden_for_publication=lambda ticket: [[101], [102], [103], [104]])
        runtime.bind(session, engine)
        session.phase = 'drafting'
        self.assertEqual(runtime('request', (20,), 3), (21, 22, 23))
        session.phase, engine.phase = 'committing', 'verified'
        session.pending = SimpleNamespace(position=10, tokens=(20, 21, 22, 23))
        return runtime, session, engine, state, calls, anchor

    def test_every_prefix_uses_target_hidden_and_rewinds_logical_frontier(self):
        for prefix in range(5):
            runtime, session, engine, state, calls, anchor = self.fixture()
            runtime.publish(prefix)
            self.assertEqual(runtime.position, 10 + prefix)
            self.assertEqual(anchor[0], 100 + prefix)
            self.assertEqual(len(calls), 3 + prefix)
            for offset in range(prefix):
                self.assertEqual(state[10 + offset], (20 + offset, 100 + offset))
            engine.publish.assert_called_once_with(prefix)
            self.assertEqual(runtime.phase, 'idle')

    def test_catchup_failure_does_not_publish_target_or_allow_retry(self):
        runtime, session, engine, state, calls, anchor = self.fixture()
        runtime.step = Mock(side_effect=RuntimeError('device failed'))
        with self.assertRaises(RuntimeError):
            runtime.publish(2)
        engine.publish.assert_not_called()
        self.assertEqual((runtime.phase, engine.phase), ('failed', 'failed'))
        self.assertEqual(anchor, [100])

    def test_prefix_zero_preserves_anchor_and_next_round_overwrites_rejected_kv(self):
        runtime, session, engine, state, calls, anchor = self.fixture()
        runtime.publish(0)
        session.phase = 'drafting'
        self.assertEqual(runtime('request', (20,), 1), (21,))
        self.assertEqual(state[10], (20, 100))
        self.assertEqual(anchor, [100])

    def test_reuse_keeps_accepted_draft_kv_and_only_catches_up_missing_input(self):
        for prefix in range(5):
            runtime, session, engine, state, calls, anchor = self.fixture(reuse_accepted_cache=True)
            original = dict(state)
            runtime.publish(prefix)
            reused = min(prefix, 3)
            self.assertEqual(runtime.position, 10 + prefix)
            self.assertEqual(anchor, [100 + prefix])
            self.assertEqual(len(calls), 3 + max(0, prefix - 3))
            for offset in range(reused):
                self.assertEqual(state[10 + offset], original[10 + offset])
            if prefix == 4:
                self.assertEqual(state[13], (23, 103))
            self.assertEqual(runtime.cache_accounting, dict(reused_rows=reused, teacher_forced_rows=prefix - reused))
            engine.publish.assert_called_once_with(prefix)
            self.assertEqual(runtime.drafted_inputs, ())

    def test_reuse_correction_overwrites_rejected_future_and_uses_target_anchor(self):
        runtime, session, engine, state, calls, anchor = self.fixture(reuse_accepted_cache=True)
        runtime.publish(2)
        session.position, session.seed, session.phase = 12, 80, 'drafting'
        self.assertEqual(runtime('request', (20, 21, 80), 1), (81,))
        self.assertEqual(state[11], (21, 1100))
        self.assertEqual(state[12], (80, 102))
        self.assertEqual(runtime.drafted_inputs, (80,))

    def test_reuse_rejects_a_different_ticket_before_publication(self):
        runtime, session, engine, state, calls, anchor = self.fixture(reuse_accepted_cache=True)
        session.pending.tokens = (20, 99, 22, 23)
        with self.assertRaisesRegex(ValueError, 'actual verified prefix'):
            runtime.publish(2)
        engine.publish.assert_not_called()
        self.assertEqual(anchor, [100])
        self.assertEqual((runtime.phase, engine.phase), ('failed', 'failed'))

    def test_reuse_tail_failure_poisoning_preserves_target_transaction(self):
        runtime, session, engine, state, calls, anchor = self.fixture(reuse_accepted_cache=True)
        runtime.step = Mock(side_effect=RuntimeError('tail failed'))
        with self.assertRaisesRegex(RuntimeError, 'tail failed'):
            runtime.publish(4)
        engine.publish.assert_not_called()
        self.assertEqual(anchor, [100])
        self.assertEqual(runtime.cache_accounting, dict(reused_rows=0, teacher_forced_rows=0))

    def test_reuse_target_only_fallback_still_initializes_missing_draft_kv(self):
        runtime, session, engine, state, calls, anchor = self.fixture(reuse_accepted_cache=True)
        runtime.publish(0)
        session.phase, engine.phase = 'committing', 'verified'
        session.pending.tokens = (20,)
        runtime.publish(1)
        self.assertEqual(state[10], (20, 100))
        self.assertEqual(anchor, [101])
        self.assertEqual(runtime.cache_accounting, dict(reused_rows=0, teacher_forced_rows=1))
