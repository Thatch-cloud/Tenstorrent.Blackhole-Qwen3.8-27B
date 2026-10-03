import unittest

from greedy_session import GreedySession
from greedy_verify import select_prefix
from hybrid_draft import HybridDraft
from lookup_draft import LookupDraft


class WideProposalTests(unittest.TestCase):
    def test_lookup_capacity_requires_opt_in_and_uses_only_known_tokens(self):
        history = [100, *range(40), 100]
        with self.assertRaises(ValueError):
            LookupDraft('request', history).propose('request', 31)
        draft = LookupDraft('request', history, max_proposals=31)
        self.assertEqual(draft.propose('request', 31), list(range(31)))
        hybrid = HybridDraft('request', history, vocab_size=200, max_proposals=31)
        self.assertEqual(hybrid.propose('request', 31, greedy=True, verifier_ready=True).tokens, tuple(range(31)))

    def test_every_wide_rejection_and_all_accepted_bonus(self):
        proposals = tuple(range(1, 32))
        for accepted in range(32):
            predictions = [*proposals, 32]
            predictions[accepted] = 99
            decision = select_prefix(proposals, predictions, vocab_size=100, max_proposals=31)
            self.assertEqual(decision.emitted, (*proposals[:accepted], 99))
            self.assertEqual((decision.accepted, decision.state_rows), (accepted, accepted + 1))
        with self.assertRaises(ValueError):
            select_prefix(proposals, [*proposals, 32], vocab_size=100)

    def test_wide_session_counts_committed_tokens_after_publication(self):
        session = GreedySession('request', [200, 201], 0, vocab_size=256, max_new_tokens=33,
            verifier_rows=32, neural={'fixture': lambda request, history, count: list(range(1, count + 1))})
        ticket = session.propose('request', max_rows=32, selected='fixture')
        self.assertEqual(ticket.tokens, tuple(range(32)))
        published = []
        def publish(prefix):
            self.assertEqual(session.committed_decode_tokens, 0)
            published.append(prefix)
        decision = session.commit('request', ticket, range(1, 33), publish)
        self.assertEqual((decision.accepted, session.committed_decode_tokens), (31, 32))
        self.assertEqual(published, [32])
        self.assertTrue(session.finished)
        self.assertEqual(session.emitted, list(range(33)))

    def test_wide_eos_and_generation_budget_do_not_over_emit(self):
        decision = select_prefix(range(1, 32), range(1, 33), vocab_size=100, eos_ids=(17,), max_proposals=31)
        self.assertEqual(decision.emitted, tuple(range(1, 18)))
        self.assertTrue(decision.finished)
        session = GreedySession('request', [200, 201], 0, vocab_size=256, max_new_tokens=32,
            verifier_rows=32, neural={'fixture': lambda request, history, count: list(range(1, count + 1))})
        self.assertEqual(len(session.propose('request', max_rows=32, selected='fixture').tokens), 16)

    def test_full_suffix_policy_reaches_t32_with_real_known_history(self):
        session = GreedySession('request', [0, 1, 2] * 100, 0, vocab_size=100,
            max_new_tokens=65, verifier_rows=32, prefer_full_suffix=True)
        while not session.finished:
            ticket = session.propose('request', max_rows=32)
            self.assertEqual(len(ticket.tokens), 32)
            self.assertEqual(ticket.source, 'lookup')
            predictions = [(token + 1) % 3 for token in ticket.tokens]
            session.commit('request', ticket, predictions, lambda prefix: None)
        self.assertEqual(session.committed_blocks, 2)
        self.assertEqual(session.committed_decode_tokens, 64)
        self.assertEqual(session.emitted, [index % 3 for index in range(65)])

    def test_defaults_remain_t16_and_invalid_capacities_are_rejected(self):
        session = GreedySession('request', [1, 2], 3, vocab_size=100, max_new_tokens=100)
        with self.assertRaises(ValueError):
            session.propose('request', max_rows=32)
        for capacity in (0, True, 16, 32, 31.0):
            with self.assertRaises(ValueError):
                LookupDraft('request', [1, 2], max_proposals=capacity)
