from types import SimpleNamespace
import unittest
from unittest.mock import Mock

from mtp_request_runtime import MTPRequestRuntime


class MTPRequestRuntimeTests(unittest.TestCase):
    def fixture(self):
        anchor = [100]
        state = {}
        calls = []
        def step(token, hidden, position, *, select):
            state[position] = (token, hidden[0])
            calls.append((token, hidden[0], position, select))
            return [hidden[0] + 1000], token + 1 if select else None
        runtime = MTPRequestRuntime(step, anchor, copy_hidden=lambda source, destination: destination.__setitem__(0, source[0]),
                                    verified_row=lambda rows, index: rows[index], max_drafts=3)
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
