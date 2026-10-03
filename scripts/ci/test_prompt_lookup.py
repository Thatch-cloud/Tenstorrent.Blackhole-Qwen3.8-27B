"""QWEN_FAST_LOOKUP_DRAFT: the lookup, the policy, flag-off equality and where the proposal rows enter the verify."""
import random
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'speculative-decoding/harness'))

import prompt_lookup
from dflash_request_runtime import DFlashRequestRuntime, TARGET_TAPS
from greedy_session import GreedySession
from serving_fast_request import FastRequest

PROMPT = (1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2)       # ends 1, 2; the seed 3 makes the history end 1, 2, 3 (seen at its start)


def brute_force(history, n, count):
    """The reference: the most recent earlier occurrence of the last n tokens by scanning, then the backward extension."""
    if len(history) < n:
        return (), 0
    key = tuple(history[-n:])
    final = len(history) - 1
    for end in range(final - 1, n - 2, -1):
        if tuple(history[end - n + 1:end + 1]) == key:
            length = n
            while length < prompt_lookup.MAX_BACK and end - length >= 0 and history[end - length] == history[final - length]:
                length += 1
            tokens = tuple(history[end + 1:end + 1 + count])
            return (tokens, length) if tokens else ((), 0)
    return (), 0


class PolicyTests(unittest.TestCase):
    def test_parse(self):
        self.assertEqual(prompt_lookup.parse_policy('n3m12'), prompt_lookup.LookupPolicy(3, 12))
        self.assertEqual(prompt_lookup.parse_policy(' N2M2 '), prompt_lookup.LookupPolicy(2, 2))
        self.assertEqual(repr(prompt_lookup.parse_policy('n6m64')), 'n6m64')

    def test_off_values_are_none(self):
        for text in (None, '', '0', 'off', 'OFF', '  '):
            self.assertIsNone(prompt_lookup.parse_policy(text))

    def test_malformed_policies_are_refused_not_silently_off(self):
        for text in ('1', 'on', 'n3', 'm8', 'n3m', 'nxm8', 'n1m8', 'n7m8', 'n3m2', 'n3m65', 'n3m8x', 'n3m-1', 'n 3m8'):
            with self.subTest(text=text):
                with self.assertRaises(ValueError):
                    prompt_lookup.parse_policy(text)

    def test_environment(self):
        self.assertIsNone(prompt_lookup.policy_from_environment({}))
        self.assertEqual(prompt_lookup.policy_from_environment({prompt_lookup.LOOKUP_FLAG: 'n4m8'}).m, 8)


class LookupTests(unittest.TestCase):
    def test_proposes_what_followed_the_most_recent_earlier_occurrence(self):
        index = prompt_lookup.TokenLookup((9, 1, 2, 7, 7, 1, 2, 8, 8, 1), 2)
        index.extend((2,))                              # history ends 1, 2: earlier at 5-6 (then 8, 8, 1, 2) and 1-2
        tokens, match = index.propose(3)
        self.assertEqual(tokens, (8, 8, 1))
        self.assertEqual(match, 2 + 0)                  # the token before is 7 against 1: no extension

    def test_match_length_extends_backwards_and_caps(self):
        body = tuple(range(100, 200))
        index = prompt_lookup.TokenLookup(body + (5, 5) + body[:90], 3)
        tokens, match = index.propose(4)
        self.assertEqual(match, prompt_lookup.MAX_BACK)
        self.assertEqual(tokens, body[90:94])
        short = prompt_lookup.TokenLookup((3, 4, 5, 1, 9, 4, 5), 2)
        self.assertEqual(short.propose(2), ((1, 9), 2))

    def test_a_query_never_finds_itself_and_short_history_has_no_match(self):
        self.assertEqual(prompt_lookup.TokenLookup((1, 2, 3), 3).propose(4), ((), 0))
        self.assertEqual(prompt_lookup.TokenLookup((1,), 2).propose(4), ((), 0))
        self.assertEqual(prompt_lookup.TokenLookup((1, 2, 3, 4), 2).propose(4), ((), 0))

    def test_an_occurrence_at_the_end_of_the_history_has_nothing_to_propose_beyond_it(self):
        index = prompt_lookup.TokenLookup((5, 6, 5, 6), 2)    # earlier (5, 6) at 0-1 is followed by 5, 6
        self.assertEqual(index.propose(8), ((5, 6), 2))       # the proposal runs into the query's own tokens, all real

    def test_incremental_extension_equals_a_rebuild_and_the_brute_force_scan(self):
        rng = random.Random(7)
        for n in (2, 3, 4, 6):
            history = [rng.randrange(6) for _ in range(300)]
            incremental = prompt_lookup.TokenLookup(history[:40], n)
            for cut in range(40, 300, 7):
                incremental.extend(history[len(incremental):cut])
                self.assertEqual(incremental.propose(15), brute_force(history[:cut], n, 15), (n, cut))
                self.assertEqual(incremental.propose(15), prompt_lookup.TokenLookup(history[:cut], n).propose(15))

    def test_bad_inputs_fail_closed(self):
        with self.assertRaises(ValueError):
            prompt_lookup.TokenLookup((1, 2, 3), 1)
        with self.assertRaises(ValueError):
            prompt_lookup.TokenLookup((1, 2, 1 << 18), 2)
        with self.assertRaises(ValueError):
            prompt_lookup.TokenLookup((1, 2, -1), 2)
        with self.assertRaises(ValueError):
            prompt_lookup.TokenLookup((1, 2, 3), 2).propose(0)


class ChooseTests(unittest.TestCase):
    policy = prompt_lookup.LookupPolicy(3, 8)

    def test_the_gate(self):
        dflash = tuple(range(50, 65))
        self.assertEqual(prompt_lookup.choose(self.policy, (1, 2), 7, dflash), ('dflash2', dflash))
        self.assertEqual(prompt_lookup.choose(self.policy, (), 99, dflash), ('dflash2', dflash))
        self.assertEqual(prompt_lookup.choose(None, (1, 2), 99, dflash), ('dflash2', dflash))
        source, tokens = prompt_lookup.choose(self.policy, (1, 2), 8, dflash)
        self.assertEqual(source, 'lookup')
        self.assertEqual(tokens, (1, 2) + dflash[2:])

    def test_a_long_lookup_is_cut_to_the_ticket_width_and_a_full_one_replaces_every_row(self):
        dflash = tuple(range(50, 57))
        self.assertEqual(prompt_lookup.choose(self.policy, tuple(range(20)), 8, dflash), ('lookup', tuple(range(7))))


def session_fixture(test, prompt=PROMPT, seed=3, budget=64, environ=None):
    events = []
    drafter = SimpleNamespace(position=len(prompt), max_drafts=15,
        propose=lambda s, count: tuple(range(s + 1, s + count + 1)), discard_publication=Mock())
    drafter.prepare_publication = lambda features, prefix, **kwargs: prefix

    def commit_history(prefix):
        events.append(('history', prefix))
        drafter.position += prefix

    drafter.commit_publication = commit_history
    runtime = DFlashRequestRuntime(drafter, position=len(prompt))
    session = GreedySession('request', prompt, seed, vocab_size=1000, max_new_tokens=budget, neural={'dflash2': runtime},
                            lookup_enabled=False)
    engine = SimpleNamespace(session=session, phase='idle', pending=None, retain_feature_taps=TARGET_TAPS)
    engine.proposal_rows = lambda: 16
    seen = []

    def verify(ticket):
        seen.append(ticket)
        test.assertIs(session.pending, ticket)       # the verify stages the very ticket the session holds
        engine.pending, engine.phase = ticket, 'verified'
        events.append(('verify', ticket.tokens))
        return tuple(token + 1 for token in ticket.tokens), {}

    def publish(prefix):
        events.append(('target', prefix))
        engine.phase, engine.pending = 'idle', None

    engine.verify = Mock(side_effect=verify)
    engine.publish = Mock(side_effect=publish)
    engine.verified_features_for_publication = Mock(return_value=('features',))
    engine.close = Mock()
    request = FastRequest(session, engine, runtime, release_drafter=Mock())
    lines = []
    request.lookup = prompt_lookup.for_request(prompt, session, request_id='request', environ=environ or {},
                                               log=lambda text, **kwargs: lines.append(text))
    return request, events, seen, lines


class FlagOffTests(unittest.TestCase):
    def test_off_builds_nothing_and_the_ticket_is_the_sessions_own(self):
        request, events, seen, lines = session_fixture(self)
        self.assertIsNone(request.lookup)
        ticket = request.prepare('request')
        self.assertIs(request.session.pending, ticket)
        self.assertEqual(ticket.tokens, (3,) + tuple(range(4, 19)))
        self.assertEqual(ticket.source, 'dflash2')
        self.assertEqual(request.session.epoch, 1)
        self.assertEqual(lines, [])

    def test_off_equals_a_request_that_never_heard_of_the_lookup(self):
        """Two requests, one with the attribute untouched, one with the flag value 'off': the same events, outputs, epochs."""
        outcomes = []
        for environ in ({}, {prompt_lookup.LOOKUP_FLAG: 'off'}, {prompt_lookup.LOOKUP_FLAG: '0'}):
            request, events, seen, lines = session_fixture(self, environ=environ)
            outputs = [request.step('request', cancelled=lambda: False).token_ids for _ in range(3)]
            outcomes.append((outputs, events, request.session.epoch, request.session.emitted[:], lines))
            self.assertIsNone(request.lookup)
        self.assertEqual(outcomes[0], outcomes[1])
        self.assertEqual(outcomes[0], outcomes[2])

    def test_a_malformed_flag_value_is_refused_where_the_request_is_built(self):
        with self.assertRaises(ValueError):
            session_fixture(self, environ={prompt_lookup.LOOKUP_FLAG: 'yes'})


class EngagedTests(unittest.TestCase):
    ENVIRON = {prompt_lookup.LOOKUP_FLAG: 'n2m3'}

    def test_the_lookup_rows_replace_dflash2s_before_anything_reads_the_ticket(self):
        request, events, seen, lines = session_fixture(self, environ=self.ENVIRON)
        self.assertEqual(request.lookup.synced, 1)
        ticket = request.prepare('request')
        # the history ends 1, 2, 3 and occurred at its start, followed by 4..9, 1, 2, 3: nine real tokens, then DFlash2's
        # rows 9.. (the consecutive integers from 4, so 13..18) fill the remaining width of fifteen
        self.assertEqual(ticket.tokens, (3, 4, 5, 6, 7, 8, 9, 1, 2, 3, 13, 14, 15, 16, 17, 18))
        self.assertEqual(ticket.source, 'lookup')
        self.assertEqual(ticket.match_length, 3)
        self.assertIs(request.session.pending, ticket)
        self.assertEqual(request.session.epoch, 2)               # the session's proposal, then the replacement
        self.assertEqual(request.session.phase, 'pending')
        self.assertEqual(events, [])                              # nothing verified yet
        # the scheduler's draft ids and the verify's staging are both read from this ticket
        self.assertEqual(list(ticket.tokens[1:]), list(request.session.pending.tokens[1:]))

    def test_the_verify_sees_the_lookup_rows_and_commit_judges_them(self):
        request, events, seen, lines = session_fixture(self, environ=self.ENVIRON)
        output = request.step('request', cancelled=lambda: False)
        ticket, = seen
        self.assertEqual(ticket.source, 'lookup')
        self.assertEqual(events[0], ('verify', ticket.tokens))
        # predictions are token + 1 per row: the lookup rows 4, 5, ... are right while they count up and wrong after
        # the first break (9 then 1), so the commit is the seed's prediction plus the accepted run: 4..9 and the bonus 10
        self.assertEqual(output.token_ids, (4, 5, 6, 7, 8, 9, 10))
        self.assertEqual(request.session.seed, 10)
        self.assertEqual(request.session.committed_decode_tokens, 7)

    def test_a_round_the_gate_rejects_keeps_dflash2s_ticket_object_and_counts(self):
        request, events, seen, lines = session_fixture(self, environ={prompt_lookup.LOOKUP_FLAG: 'n2m4'})
        ticket = request.prepare('request')
        self.assertEqual(ticket.source, 'dflash2')              # the match is 3 < 4
        self.assertEqual(ticket.tokens, (3,) + tuple(range(4, 19)))
        self.assertEqual(request.session.epoch, 1)

    def test_every_round_is_logged_with_user_source_proposed_and_committed(self):
        request, events, seen, lines = session_fixture(self, environ=self.ENVIRON)
        request.step('request', cancelled=lambda: False)
        self.assertEqual(lines, [])                             # the commit is logged when the next round proposes
        request.step('request', cancelled=lambda: False)
        self.assertEqual(len(lines), 1)
        first = lines[0]
        self.assertTrue(first.startswith(prompt_lookup.ROUND_MARKER))
        for field in ('request=request', 'position=11', 'source=lookup', 'match=3', 'offered=9', 'proposed=9', 'committed=7'):
            self.assertIn(field, first.split())
        request.close('request')
        self.assertEqual(len(lines), 2)                         # the last round is logged at close
        self.assertEqual((request.lookup.rounds, request.lookup.lookup_rounds), (2, request.lookup.lookup_rounds))

    def test_a_ticket_discarded_before_its_verify_is_not_logged_as_a_round(self):
        request, events, seen, lines = session_fixture(self, environ=self.ENVIRON)
        request.prepare('request')
        session = request.session
        session.pending, session.phase = None, 'idle'            # serving_worker_hook.discard_stale_ticket's reset
        request.runtime.discard_proposal()
        request.prepare('request')
        request.step('request', cancelled=lambda: False)
        request.step('request', cancelled=lambda: False)
        self.assertEqual(len(lines), 1)
        self.assertIn('committed=7', lines[0])

    def test_the_index_follows_the_sessions_emitted_tokens_across_rounds(self):
        request, events, seen, lines = session_fixture(self, environ=self.ENVIRON)
        for _ in range(3):
            request.step('request', cancelled=lambda: False)
        request.prepare('request')
        history = list(PROMPT) + list(request.session.emitted)
        self.assertEqual(request.lookup.index.history, history)
        self.assertEqual(request.lookup.index.propose(15), prompt_lookup.TokenLookup(history, 2).propose(15))


class WriteOrderingTests(unittest.TestCase):
    """The tokens reach the card in stage_packed from each entry's ticket (PackedVerifier.segment_users), at verify time."""

    def test_segment_users_stages_the_replaced_ticket(self):
        try:
            import packed_verifier
        except ImportError as error:                              # no torch on this interpreter
            self.skipTest('packed_verifier needs its device imports: %s' % error)
        request, events, seen, lines = session_fixture(self, environ=EngagedTests.ENVIRON)
        ticket = request.prepare('request')
        block = SimpleNamespace(users=1)
        entry = dict(ticket=ticket, request=SimpleNamespace(engine=SimpleNamespace(pages='table')))
        import torch

        users = packed_verifier.PackedVerifierEngine.segment_users(block, [entry], (0,))
        users = [(tokens, start, torch.zeros((1, 8), dtype=torch.int32)) for tokens, start, _ in users]
        shape = SimpleNamespace(users=1, rows_per_user=16, page_width=8, capacity=1 << 20)
        staged_tokens = packed_verifier.packed_host_inputs(users, shape, 64, 1.0e6, 1000)[0]
        # the host tensor stage_packed writes into fixture.tokens is the replaced ticket's rows, in order
        self.assertEqual(staged_tokens.flatten().tolist(), list(ticket.tokens))
        users = packed_verifier.PackedVerifierEngine.segment_users(block, [entry], (0,))
        self.assertEqual(users[0], (ticket.tokens, ticket.position, 'table'))
        self.assertEqual(users[0][0][1:8], (4, 5, 6, 7, 8, 9, 1))
        self.assertEqual(users[0][0][10:], (13, 14, 15, 16, 17, 18))

    def test_nothing_in_the_lookup_touches_a_device(self):
        source = Path(prompt_lookup.__file__).read_text(encoding='utf-8')
        for word in ('ttnn', 'torch', 'operations', 'from_torch', 'copy_host_to_device'):
            self.assertNotIn(word, source)


if __name__ == '__main__':
    unittest.main()
