from dataclasses import replace
import unittest
from unittest.mock import Mock

from greedy_session import GreedySession


class SessionTests(unittest.TestCase):
    def test_device_preparation_excludes_proposals_and_poisoned_retry(self):
        for fail in (False, True):
            session = self.fixture()
            session.begin_preparation('request')
            with self.assertRaises(ValueError):
                session.propose('request')
            with self.assertRaises(ValueError):
                session.begin_preparation('request')
            if fail:
                session.fail_preparation('request')
                with self.assertRaises(ValueError):
                    session.propose('request')
                with self.assertRaises(ValueError):
                    session.begin_preparation('request')
            else:
                session.finish_preparation('request')
                self.assertEqual(session.phase, 'idle')
            session.close('request')

    def fixture(self, budget=64, eos_ids=()):
        return GreedySession('request', [0, 1, 2] * 12, 0, vocab_size=100,
                             max_new_tokens=budget, eos_ids=eos_ids)

    def test_real_lookup_and_greedy_accounting_match_serial_sequence(self):
        session = self.fixture()
        published = []
        while not session.finished:
            ticket = session.propose('request')
            predictions = tuple((token + 1) % 3 for token in ticket.tokens)
            before = tuple(session.emitted)

            def publish(prefix):
                self.assertEqual(tuple(session.emitted), before)
                self.assertEqual(session.phase, 'committing')
                published.append(prefix)

            decision = session.commit('request', ticket, predictions, publish)
            self.assertEqual(len(decision.emitted), len(ticket.tokens))
        self.assertEqual(session.emitted, [index % 3 for index in range(64)])
        self.assertEqual(session.position, 36 + 63)
        self.assertEqual(session.committed_blocks, len(published))
        self.assertEqual(session.committed_decode_tokens, 63)
        self.assertGreater(session.accepted_proposals, 0)
        self.assertEqual(len(session.drafter.lookup.history), 36 + 64)

    def test_rejection_emits_correction_only_after_publication(self):
        session = self.fixture()
        ticket = session.propose('request')
        self.assertEqual(len(ticket.tokens), 4)
        publish = Mock()
        publish.return_value = None
        decision = session.commit('request', ticket, (99,) * len(ticket.tokens), publish)
        publish.assert_called_once_with(1)
        self.assertEqual(decision.emitted, (99,))
        self.assertEqual(session.emitted, [0, 99])
        self.assertEqual(session.position, 37)
        self.assertEqual(session.drafter.lookup.history[-2:], [0, 99])

    def test_generation_budget_selects_bucket_without_over_emission(self):
        for budget in range(2, 20):
            session = self.fixture(budget)
            while not session.finished:
                ticket = session.propose('request')
                self.assertLessEqual(len(ticket.tokens), budget - len(session.emitted))
                session.commit('request', ticket, tuple((token + 1) % 3 for token in ticket.tokens), lambda prefix: None)
            self.assertEqual(len(session.emitted), budget)

    def test_eos_finishes_without_later_proposals(self):
        session = self.fixture(eos_ids=(1,))
        ticket = session.propose('request')
        decision = session.commit('request', ticket, tuple((token + 1) % 3 for token in ticket.tokens), lambda prefix: None)
        self.assertEqual(decision.emitted, (1,))
        self.assertEqual(session.emitted, [0, 1])
        self.assertTrue(session.finished)
        with self.assertRaises(ValueError):
            session.propose('request')

    def test_singleton_fallback_without_lookup_match(self):
        session = GreedySession('request', [4, 5], 6, vocab_size=10, max_new_tokens=3)
        ticket = session.propose('request')
        self.assertEqual(ticket.tokens, (6,))
        self.assertEqual(ticket.source, 'target')
        session.commit('request', ticket, (7,), lambda prefix: None)
        self.assertEqual(session.emitted, [6, 7])

    def test_abort_preserves_history_and_rejects_stale_ticket(self):
        session = self.fixture()
        ticket = session.propose('request')
        history = list(session.drafter.lookup.history)
        restore = Mock(return_value=None)
        session.abort('request', ticket, restore)
        restore.assert_called_once_with(0)
        self.assertEqual(session.drafter.lookup.history, history)
        self.assertEqual(session.emitted, [0])
        self.assertEqual(session.committed_decode_tokens, 0)
        self.assertEqual(session.aborted_blocks, 1)
        replacement = session.propose('request')
        self.assertGreater(replacement.epoch, ticket.epoch)
        with self.assertRaises(ValueError):
            session.commit('request', ticket, (1,) * 16, Mock())

    def test_owner_ticket_identity_and_pending_guards(self):
        session = self.fixture()
        ticket = session.propose('request')
        for owner, submitted in (('other', ticket), ('request', replace(ticket))):
            publish = Mock()
            with self.assertRaises(ValueError):
                session.commit(owner, submitted, (1,) * 16, publish)
            publish.assert_not_called()
        with self.assertRaises(ValueError):
            session.propose('request')
        with self.assertRaises(ValueError):
            session.close('request')

    def test_publication_failure_poisoning_and_no_unverified_history(self):
        for failure in ('raise', 'false'):
            session = self.fixture()
            ticket = session.propose('request')
            publish = Mock(side_effect=RuntimeError('device')) if failure == 'raise' else Mock(return_value=False)
            with self.assertRaises(RuntimeError):
                session.commit('request', ticket, (99,) * len(ticket.tokens), publish)
            self.assertEqual(session.emitted, [0])
            self.assertEqual(session.phase, 'failed')
            with self.assertRaises(ValueError):
                session.commit('request', ticket, (99,) * 16, Mock())
            session.close('request')
            self.assertEqual(session.drafter.lookup.history, [])

    def test_invalid_predictions_do_not_publish_or_poison_device_state(self):
        session = self.fixture()
        ticket = session.propose('request')
        publish = Mock()
        with self.assertRaises(ValueError):
            session.commit('request', ticket, (100,) * len(ticket.tokens), publish)
        publish.assert_not_called()
        self.assertEqual(session.phase, 'pending')

    def test_reentrant_publication_cannot_start_another_block(self):
        session = self.fixture()
        ticket = session.propose('request')

        def publish(prefix):
            with self.assertRaises(ValueError):
                session.propose('request')
            with self.assertRaises(ValueError):
                session.commit('request', ticket, (99,) * 16, Mock())

        session.commit('request', ticket, (99,) * len(ticket.tokens), publish)

    def test_prefill_seed_is_not_counted_as_decode_work(self):
        session = self.fixture(budget=1)
        self.assertTrue(session.finished)
        self.assertEqual(session.emitted, [0])
        self.assertEqual(session.committed_decode_tokens, 0)
        session.close('request')
        self.assertEqual(session.drafter.lookup.history, [])

    def test_failed_abort_poisoning(self):
        session = self.fixture()
        ticket = session.propose('request')
        with self.assertRaises(RuntimeError):
            session.abort('request', ticket, Mock(side_effect=RuntimeError('device')))
        self.assertEqual(session.phase, 'failed')
        self.assertEqual(session.aborted_blocks, 0)
        self.assertEqual(session.committed_decode_tokens, 0)

    def test_drafter_cannot_reenter_or_close_live_session(self):
        def draft(request_id, history, count):
            with self.assertRaises(ValueError):
                session.propose(request_id)
            with self.assertRaises(ValueError):
                session.close(request_id)
            return [7, 8]

        session = GreedySession('request', [4, 5], 6, vocab_size=10, max_new_tokens=8, neural={'test': draft})
        ticket = session.propose('request', selected='test')
        self.assertEqual(ticket.tokens, (6, 7))
        self.assertEqual(ticket.source, 'test')

    def test_invalid_neural_proposals_fail_before_device_work(self):
        session = GreedySession('request', [4, 5], 6, vocab_size=10, max_new_tokens=8,
                                neural={'test': lambda *arguments: [10]})
        with self.assertRaises(ValueError):
            session.propose('request', selected='test')
        self.assertEqual(session.phase, 'failed')
        self.assertEqual(session.emitted, [6])
        session.close('request')


if __name__ == '__main__':
    unittest.main()
