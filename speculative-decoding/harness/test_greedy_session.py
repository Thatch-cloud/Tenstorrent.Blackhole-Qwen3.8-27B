from dataclasses import replace
import unittest
from unittest.mock import Mock

from greedy_session import GreedySession


class SessionTests(unittest.TestCase):
    def test_ticket_match_telemetry_does_not_leak_into_target_only_steps(self):
        session = self.fixture()
        ticket = session.propose('request')
        self.assertEqual(ticket.source, 'lookup')
        self.assertGreater(ticket.match_length, 0)
        session.abort('request', ticket, lambda prefix: None)
        target = session.propose('request', max_rows=1)
        self.assertEqual(target.source, 'target')
        self.assertEqual(target.match_length, 0)

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


def neural(request_id, history, count):
    """A feature drafter proposing `count` tokens that continue the history by one."""
    return tuple((history[-1] + 1 + index) % 100 for index in range(count))


def target(ticket):
    """The target's predictions for `ticket` when every proposal is right: the next token after each row."""
    return tuple((token + 1) % 100 for token in ticket.tokens)


class BudgetCapTests(unittest.TestCase):
    """QWEN_FAST_BUDGET_CAP: propose(full_width=True) keeps the ticket's width past the budget and
    commit cuts the emission at it."""

    def fixture(self, budget, emitted=1, eos_ids=()):
        session = GreedySession('request', [0, 1, 2] * 12, 0, vocab_size=100, max_new_tokens=budget,
                                eos_ids=eos_ids, neural={'dflash2': neural}, lookup_enabled=False)
        while len(session.emitted) < emitted:
            ticket = session.propose('request', max_rows=1, selected='dflash2')
            session.commit('request', ticket, target(ticket), lambda prefix: None)
        return session

    def propose(self, session, **options):
        return session.propose('request', max_rows=16, selected='dflash2', **options)

    def test_a_full_width_ticket_keeps_its_rows_past_the_budget_and_the_default_does_not(self):
        session = self.fixture(budget=6, emitted=3)
        self.assertEqual(len(self.propose(session, full_width=True).tokens), 16)
        session.abort('request', session.pending, lambda prefix: None)
        self.assertEqual(len(self.propose(session).tokens), 2, 'the default narrows to the widest bucket that fits 3 left')

    def test_an_all_accepting_commit_emits_exactly_the_remaining_tokens_and_finishes(self):
        session = self.fixture(budget=6, emitted=3)
        ticket = self.propose(session, full_width=True)
        position = session.position
        published = []
        decision = session.commit('request', ticket, target(ticket), published.append)
        self.assertEqual(len(decision.emitted), 3)
        self.assertEqual(published, [3])
        self.assertTrue(session.finished)
        self.assertEqual((len(session.emitted), session.position), (6, position + 3))

    def test_the_capped_emission_is_the_uncapped_emissions_first_remaining_tokens(self):
        for budget in range(2, 18):
            for accepted in range(0, 16):
                with self.subTest(budget=budget, accepted=accepted):
                    wide = self.fixture(budget=64, emitted=1)
                    capped = self.fixture(budget=budget, emitted=1)
                    ticket = self.propose(wide)
                    narrow = self.propose(capped, full_width=True)
                    self.assertEqual(ticket.tokens, narrow.tokens)
                    predictions = list(target(ticket))
                    for row in range(accepted, 16):
                        predictions[row] = 99  # the proposal after the accepted run is rejected
                    uncapped = wide.commit('request', ticket, predictions, lambda prefix: None)
                    cut = capped.commit('request', narrow, predictions, lambda prefix: None)
                    remaining = budget - 1
                    self.assertEqual(cut.emitted, uncapped.emitted[:remaining])
                    self.assertLessEqual(len(capped.emitted), budget)
                    self.assertEqual(capped.finished, len(capped.emitted) == budget or cut.finished)

    def test_an_eos_inside_the_cap_finishes_at_the_eos(self):
        session = self.fixture(budget=6, emitted=3, eos_ids=(5,))
        ticket = self.propose(session, full_width=True)  # tokens continue 3, 4, 5, ...
        decision = session.commit('request', ticket, target(ticket), lambda prefix: None)
        self.assertEqual(decision.emitted[-1], 5)
        self.assertLessEqual(len(decision.emitted), 3)
        self.assertTrue(session.finished)

    def test_a_full_width_ticket_narrowed_and_committed_without_max_rows_emits_at_most_the_remaining(self):
        session = self.fixture(budget=6, emitted=3)
        ticket = session.narrow('request', self.propose(session, full_width=True), 4)
        decision = session.commit('request', ticket, target(ticket), lambda prefix: None)
        self.assertEqual(len(decision.emitted), 3)
        self.assertTrue(session.finished)

    def test_max_rows_wider_than_the_remaining_budget_is_cut_to_it(self):
        session = self.fixture(budget=6, emitted=3)
        ticket = self.propose(session, full_width=True)
        decision = session.commit('request', ticket, target(ticket), lambda prefix: None, max_rows=8)
        self.assertEqual(len(decision.emitted), 3)

    def test_a_commit_cap_outside_the_ticket_is_still_refused(self):
        session = self.fixture(budget=6, emitted=3)
        ticket = self.propose(session, full_width=True)
        for rows in (0, 17, True, 2.0):
            with self.assertRaises(ValueError):
                session.commit('request', ticket, target(ticket), lambda prefix: None, max_rows=rows)

    def test_default_propose_and_commit_are_unchanged(self):
        session = self.fixture(budget=64)
        ticket = self.propose(session)
        self.assertEqual(len(ticket.tokens), 16)
        decision = session.commit('request', ticket, target(ticket), lambda prefix: None)
        self.assertEqual(len(decision.emitted), 16)
        self.assertFalse(session.finished)


if __name__ == '__main__':
    unittest.main()
