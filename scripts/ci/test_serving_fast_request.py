from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'speculative-decoding/harness'))

from greedy_session import GreedySession
from dflash_request_runtime import DFlashRequestRuntime, TARGET_TAPS
from serving_fast_request import FastRequest


class FastServingTests(unittest.TestCase):
    def fixture(self, budget=32, eos_ids=()):
        events = []
        drafter = SimpleNamespace(position=2, max_drafts=15,
            propose=lambda seed, count: tuple(range(seed + 1, seed + count + 1)),
            discard_publication=Mock())
        drafter.prepare_publication = lambda features, prefix, **kwargs: prefix

        def commit_history(prefix):
            events.append(('history', prefix))
            drafter.position += prefix

        drafter.commit_publication = commit_history
        runtime = DFlashRequestRuntime(drafter, position=2)
        session = GreedySession('request', (1, 2), 10, vocab_size=100, max_new_tokens=budget,
            eos_ids=eos_ids, neural={'dflash2': runtime}, lookup_enabled=False)
        engine = SimpleNamespace(session=session, phase='idle', pending=None, retain_feature_taps=TARGET_TAPS)
        engine.proposal_rows = lambda: max(width for width in (1, 2, 4, 8, 16)
            if width <= budget - len(session.emitted))

        def verify(ticket):
            engine.pending, engine.phase = ticket, 'verified'
            events.append(('verify', len(ticket.tokens)))
            return tuple(token + 1 for token in ticket.tokens), {}

        def publish(prefix):
            self.assertEqual(session.phase, 'committing')
            events.append(('target', prefix))
            engine.phase, engine.pending = 'idle', None

        engine.verify = Mock(side_effect=verify)
        engine.publish = Mock(side_effect=publish)
        engine.verified_features_for_publication = Mock(return_value=('features',))
        engine.close = Mock(side_effect=lambda: events.append(('close_engine',)))
        release = Mock(side_effect=lambda: events.append(('close_drafter',)))
        return FastRequest(session, engine, runtime, release_drafter=release), events

    def test_returns_only_committed_block_after_both_publications(self):
        request, events = self.fixture()
        output = request.step('request', cancelled=lambda: False)
        self.assertEqual(output.token_ids, tuple(range(11, 27)))
        self.assertEqual(events, [('verify', 16), ('target', 16), ('history', 16)])
        self.assertEqual(request.runtime.position, output.position)
        self.assertFalse(output.finished)

    def test_generation_budget_and_eos_are_not_overrun(self):
        for budget, eos_ids, expected in ((3, (), (11, 12)), (32, (13,), (11, 12, 13))):
            request, _ = self.fixture(budget, eos_ids)
            output = request.step('request', cancelled=lambda: False)
            self.assertEqual(output.token_ids, expected)
            self.assertTrue(output.finished)

    def test_cancellation_after_verification_rolls_back_without_output(self):
        request, events = self.fixture()
        output = request.step('request', cancelled=Mock(side_effect=(False, True)))
        self.assertEqual(output.token_ids, ())
        self.assertTrue(output.cancelled)
        self.assertEqual(events, [('verify', 16), ('target', 0)])
        self.assertEqual(request.session.emitted, [10])
        self.assertEqual(request.runtime.position, 2)
        request.close('request')
        self.assertEqual(events[-2:], [('close_engine',), ('close_drafter',)])

    def test_cancel_before_drafting_and_close_are_idempotent(self):
        request, events = self.fixture()
        self.assertTrue(request.step('request', cancelled=lambda: True).cancelled)
        request.engine.verify.assert_not_called()
        request.close('request')
        request.close('request')
        self.assertEqual(events, [('close_engine',), ('close_drafter',)])

    def test_publication_failure_never_returns_tokens(self):
        request, _ = self.fixture()
        request.engine.publish.side_effect = RuntimeError('device publication failed')
        with self.assertRaises(RuntimeError):
            request.step('request', cancelled=lambda: False)
        self.assertEqual(request.session.emitted, [10])
        self.assertEqual(request.session.phase, 'failed')
        with self.assertRaises(ValueError):
            request.step('request', cancelled=lambda: False)

    def test_mismatched_owner_rejected_before_device_work(self):
        request, _ = self.fixture()
        with self.assertRaises(ValueError):
            request.step('another', cancelled=lambda: False)
        with self.assertRaises(ValueError):
            request.close('another')
        request.engine.verify.assert_not_called()
