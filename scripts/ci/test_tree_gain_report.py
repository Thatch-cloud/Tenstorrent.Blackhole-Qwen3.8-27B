"""QWEN_FAST_LOOKUP_SHADOW and tree_gain_report.py: the SHADOW line, the replay of both candidates against the committed continuation, the two rates, the checks.

The lines are produced by the real RequestLookup on the real GreedySession and FastRequest (the fixture of test_prompt_lookup, with a drafter that is right
most of the time and a target that is a fixed stream of tokens), then read back by the report: the reading is exact when the replay of the rounds the run took
from DFlash2 gives the commits the run logged.
"""
import json
from pathlib import Path
import random
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / 'speculative-decoding/harness'))

import prompt_lookup  # noqa: E402
import tree_gain_report as report  # noqa: E402
from dflash_request_runtime import DFlashRequestRuntime, TARGET_TAPS  # noqa: E402
from greedy_session import GreedySession  # noqa: E402
from serving_fast_request import FastRequest  # noqa: E402

VOCAB = 1000


def make_stream(total, seed=5):
    """A token stream with repeated spans (what coding output does): half the time it copies an earlier stretch, else fresh tokens."""
    rng = random.Random(seed)
    out = []
    while len(out) < total:
        if len(out) > 60 and rng.random() < 0.5:
            start = rng.randrange(0, len(out) - 30)
            out.extend(out[start:start + rng.randrange(10, 30)])
        else:
            out.extend(rng.randrange(900) for _ in range(rng.randrange(3, 10)))
    return out[:total]


def run_request(environ, *, prompt_length=120, budget=150, right=0.7, seed=3, stream_seed=5, request_id='r'):
    """-> (request, lines, stream): a request whose target is `stream` (row i of a ticket at position p predicts stream[p + 1 + i]), whose drafter proposes the
    right token with probability `right` per row, driven to its end; `lines` is everything the lookup printed."""
    stream = make_stream(prompt_length + budget + 40, stream_seed)
    prompt, seed_token = tuple(stream[:prompt_length]), stream[prompt_length]
    rng = random.Random(seed)
    drafter = SimpleNamespace(position=prompt_length, max_drafts=15, discard_publication=Mock())

    def propose(seed_, count):
        return tuple(stream[drafter.position + 1 + i] if rng.random() < right else rng.randrange(900) for i in range(count))

    drafter.propose = propose
    drafter.prepare_publication = lambda features, prefix, **kwargs: prefix

    def commit_history(prefix):
        drafter.position += prefix

    drafter.commit_publication = commit_history
    runtime = DFlashRequestRuntime(drafter, position=prompt_length)
    session = GreedySession(request_id, prompt, seed_token, vocab_size=VOCAB, max_new_tokens=budget, neural={'dflash2': runtime}, lookup_enabled=False)
    engine = SimpleNamespace(session=session, phase='idle', pending=None, retain_feature_taps=TARGET_TAPS)
    engine.proposal_rows = lambda: 16

    def verify(ticket):
        engine.pending, engine.phase = ticket, 'verified'
        return tuple(stream[ticket.position + 1 + i] for i in range(len(ticket.tokens))), {}

    def publish(prefix):
        engine.phase, engine.pending = 'idle', None

    engine.verify = Mock(side_effect=verify)
    engine.publish = Mock(side_effect=publish)
    engine.verified_features_for_publication = Mock(return_value=('features',))
    engine.close = Mock()
    request = FastRequest(session, engine, runtime, release_drafter=Mock())
    lines = []
    request.lookup = prompt_lookup.for_request(prompt, session, request_id=request_id, environ=environ, log=lambda text, **kwargs: lines.append(text))
    while not session.finished:
        request.step(request_id, cancelled=lambda: False)
    request.final_emitted = list(session.emitted)         # session.close clears the list
    request.close(request_id)
    return request, lines, stream


SHADOW = {prompt_lookup.LOOKUP_FLAG: 'n3m12', prompt_lookup.SHADOW_FLAG: '1'}
LOOKUP_ONLY = {prompt_lookup.LOOKUP_FLAG: 'n3m12'}


class ShadowFlagTests(unittest.TestCase):
    def test_the_flag_is_zero_or_one(self):
        for value in (None, '', '0'):
            self.assertFalse(prompt_lookup.shadow_from_environment({} if value is None else {prompt_lookup.SHADOW_FLAG: value}))
        self.assertTrue(prompt_lookup.shadow_from_environment({prompt_lookup.SHADOW_FLAG: '1'}))
        for value in ('2', 'on', 'true', ' 1', 'yes'):
            with self.assertRaises(ValueError):
                prompt_lookup.shadow_from_environment({prompt_lookup.SHADOW_FLAG: value})

    def test_off_logs_no_shadow_line_and_is_the_lookup_that_was(self):
        _, plain, _ = run_request(LOOKUP_ONLY)
        _, zero, _ = run_request(dict(LOOKUP_ONLY, **{prompt_lookup.SHADOW_FLAG: '0'}))
        self.assertEqual(plain, zero)
        self.assertTrue(plain)
        self.assertFalse([line for line in plain if prompt_lookup.SHADOW_MARKER in line])
        self.assertTrue(all(line.startswith(prompt_lookup.ROUND_MARKER) for line in plain))

    def test_the_shadow_changes_nothing_but_adds_one_line_after_each_round_line(self):
        plain_request, plain, _ = run_request(LOOKUP_ONLY)
        shadow_request, shadow, _ = run_request(SHADOW)
        self.assertEqual([line for line in shadow if line.startswith(prompt_lookup.ROUND_MARKER)], plain)
        self.assertEqual(shadow_request.final_emitted, plain_request.final_emitted)
        self.assertGreater(len(shadow_request.final_emitted), 100)
        self.assertEqual(shadow_request.session.epoch, plain_request.session.epoch)
        self.assertEqual(len(shadow), 2 * len(plain))
        for index in range(0, len(shadow), 2):
            self.assertTrue(shadow[index].startswith(prompt_lookup.ROUND_MARKER))
            self.assertTrue(shadow[index + 1].startswith(prompt_lookup.SHADOW_MARKER))

    def test_a_shadow_line_round_trips_and_carries_both_candidates_and_the_commit(self):
        request, lines, stream = run_request(SHADOW)
        parsed = [prompt_lookup.parse_shadow_line(line) for line in lines if prompt_lookup.SHADOW_MARKER in line]
        self.assertTrue(all(parsed))
        first = parsed[0]
        self.assertEqual((first['request'], first['position']), ('r', 120))
        self.assertEqual(len(first['dflash']), 15)
        committed = [token for item in parsed for token in item['committed']]
        self.assertEqual(committed, request.final_emitted[1:], 'the committed tokens of the rounds are the session\'s emitted ones, in order')
        for item in parsed:
            self.assertEqual(list(item['committed']), stream[item['position'] + 1:item['position'] + 1 + len(item['committed'])])
        self.assertIsNone(prompt_lookup.parse_shadow_line('[LOOKUP-ROUND] request=r position=1'))
        self.assertIsNone(prompt_lookup.parse_shadow_line('[LOOKUP-SHADOW] request=r position=x'))
        self.assertEqual(prompt_lookup.parse_shadow_line(
            '[LOOKUP-SHADOW] request=a position=5 source=dflash2 match=0 dflash=1,2 lookup=- committed=3')['lookup'], ())


class AcceptedLengthTests(unittest.TestCase):
    known = {11: 5, 12: 6, 13: 7, 14: 8}         # the continuation after position 10

    def test_longest_common_prefix_with_the_continuation(self):
        self.assertEqual(report.accepted_length((5, 6, 9), self.known, 10), (2, False))
        self.assertEqual(report.accepted_length((9, 6, 7), self.known, 10), (0, False))
        self.assertEqual(report.accepted_length((5, 6, 7, 8), self.known, 10), (4, False))

    def test_a_chain_that_runs_off_the_known_continuation_is_censored(self):
        self.assertEqual(report.accepted_length((5, 6, 7, 8, 9), self.known, 10), (4, True))
        self.assertEqual(report.accepted_length((), self.known, 10), (0, False))

    def test_continuation_from_rounds_and_conflicts(self):
        rounds = [dict(position=10, committed=(5, 6)), dict(position=12, committed=(7,)), dict(position=13, committed=(8, 9))]
        known, conflicts = report.continuation(rounds)
        self.assertEqual((known, conflicts), ({11: 5, 12: 6, 13: 7, 14: 8, 15: 9}, 0))
        self.assertEqual(report.continuation(rounds + [dict(position=10, committed=(5, 4))])[1], 1)


class EvaluateTests(unittest.TestCase):
    def rounds(self, spec):
        """spec: [(position, source, match, dflash, lookup, committed)]."""
        return [dict(request='r', position=p, source=s, match=m, dflash=tuple(d), lookup=tuple(l), committed=tuple(c)) for p, s, m, d, l, c in spec]

    def test_the_better_of_two_candidates_with_ties_to_dflash(self):
        # continuation after 0: 1 2 3 4 5 6 7 8 | the first round's dflash is right for 1 row, the lookup for 4; the second round both for 2
        rounds = self.rounds([(0, 'dflash2', 3, (1, 9, 9, 9), (1, 2, 3, 4), (1, 2)),
                              (2, 'dflash2', 0, (3, 4, 9, 9), (), (3, 4, 5)),
                              (5, 'dflash2', 5, (6, 7, 9, 9), (6, 7, 8, 8), (6, 7, 8))])
        rows, conflicts = report.evaluate(rounds)
        self.assertEqual(conflicts, 0)
        # round 0: committed (1, 2) is dflash's 1 accepted + the correction; the lookup chain 1 2 3 4 then dflash's: accepted 4 → commit 5
        self.assertEqual((rows[0]['dflash'], rows[0]['lookup'], rows[0]['tree2'], rows[0]['b_wins']), (2, 5, 5, True))
        self.assertEqual((rows[1]['dflash'], rows[1]['lookup'], rows[1]['second']), (3, 3, False))
        self.assertEqual((rows[2]['dflash'], rows[2]['lookup'], rows[2]['b_wins']), (3, 4, True))
        self.assertFalse(any(row['replay_mismatch'] for row in rows))

    def test_a_mismatch_between_the_log_and_its_replay_is_counted(self):
        rounds = self.rounds([(0, 'dflash2', 0, (1, 2, 3), (), (1, 2, 3, 4)), (4, 'dflash2', 0, (5, 5, 5), (), (5, 6))])
        rows, _ = report.evaluate(rounds)
        self.assertEqual([row['replay_mismatch'] for row in rows], [False, False])      # dflash is right for all 3 rows: commit 4; then 1 row + correction: 2
        wrong = self.rounds([(0, 'dflash2', 0, (1, 9, 3), (), (1, 2, 3, 4)), (4, 'dflash2', 0, (5, 5, 5), (), (5, 6))])
        rows, _ = report.evaluate(wrong)
        self.assertTrue(rows[0]['replay_mismatch'], 'dflash2 round whose rows disagree with what it committed')

    def test_the_walk_of_the_runs_own_policy_is_the_run(self):
        rounds = self.rounds([(0, 'dflash2', 0, (1, 2, 9), (), (1, 2, 3)), (3, 'lookup', 12, (4, 9, 9), (4, 5, 6), (4, 5, 6, 7)), (7, 'dflash2', 0, (8, 9, 9), (), (8, 9))])
        rows, _ = report.evaluate(rounds)
        self.assertEqual(report.walk(rows, 'run'), (9, 3))
        self.assertEqual(sum(row['run'] for row in rows), 9)

    def test_the_walk_lands_between_logged_starts_and_counts_only_what_a_stale_chain_still_covers(self):
        # run: rounds at 0 (commit 2), 2 (commit 2), 4 (commit 2); a strategy that commits 5 at position 0 lands at 5, past the start at 4: the chain of the
        # round at 4 covers 4..6, so 1 new token beyond the landing, and the stream ends at 6
        rows = [dict(position=0, run=2, key=5), dict(position=2, run=2, key=1), dict(position=4, run=2, key=2)]
        self.assertEqual(report.walk(rows, 'key'), (6, 2))
        # nothing left in the stale chain: one token for one round
        rows = [dict(position=0, run=2, key=5), dict(position=2, run=2, key=1), dict(position=4, run=2, key=1), dict(position=6, run=2, key=2)]
        tokens, count = report.walk(rows, 'key')
        self.assertEqual(tokens, 8)
        self.assertGreaterEqual(count, 2)


class EndToEndTests(unittest.TestCase):
    def analysis(self, **kwargs):
        request, lines, stream = run_request(SHADOW, **kwargs)
        rounds, bad = report.load(lines)
        self.assertEqual(bad, 0)
        return request, rounds, report.analyze(rounds, bad), lines

    def test_the_replay_reproduces_every_logged_commit(self):
        for stream_seed in (5, 6, 7):
            request, rounds, summary, _ = self.analysis(stream_seed=stream_seed, seed=stream_seed)
            with self.subTest(stream_seed=stream_seed):
                self.assertEqual(report.problems(summary), [])
                self.assertEqual(summary['dflash_replay_mismatches'], 0)
                self.assertEqual(summary['continuation_conflicts'], 0)
                self.assertTrue(summary['walk_run_equals_run'])
                rows, _ = report.evaluate(rounds['r'])
                for row in rows:
                    if row['censored']:
                        continue
                    self.assertEqual(row['run'], row['dflash'] if row['source'] == 'dflash2' else row['lookup'], row)

    def test_the_tree_is_never_worse_than_either_candidate_nor_than_the_run(self):
        _, rounds, summary, _ = self.analysis(budget=400)
        rows, _ = report.evaluate(rounds['r'])
        self.assertGreater(len(rows), 20)
        for row in rows:
            self.assertGreaterEqual(row['tree2'], row['dflash'])
            self.assertGreaterEqual(row['tree2'], row['lookup'])
            if not row['censored']:
                self.assertGreaterEqual(row['tree2'], row['run'])
        recorded = summary['recorded_tokens_per_round']
        self.assertGreaterEqual(recorded['tree2'], recorded['run'])
        self.assertGreaterEqual(recorded['tree2'], recorded['dflash'])
        # the stream copies spans, so the lookup wins somewhere and the tree commits more than DFlash2 alone
        self.assertGreater(summary['second_candidate_wins'], 0)
        self.assertGreater(recorded['tree2'], recorded['dflash'])
        # the walk is the lower reading, the recorded mean the upper one
        self.assertLessEqual(summary['walk_tokens_per_round']['tree2'], recorded['tree2'] + 1e-9)

    def test_the_summary_holds_aggregates_only(self):
        request, rounds, summary, lines = self.analysis()
        text = json.dumps(summary, sort_keys=True)
        json.loads(text)
        self.assertNotIn('"r"', text, 'no request id')
        for token in request.final_emitted[1:6]:
            self.assertNotIn('[%d' % token, text)
        for line in report.summary_lines(summary):
            self.assertTrue(line.startswith('[TREEGAIN]'))
            self.assertNotIn('request=', line)
        self.assertEqual(set(summary['by_lookup_match']) - {'0-3', '4-7', '8-11', '12-23', '24-63', '64'}, set())

    def test_main_reads_logs_and_writes_the_json(self):
        import os
        import tempfile

        _, lines, _ = run_request(SHADOW, budget=300)
        directory = tempfile.mkdtemp()
        self.addCleanup(lambda: [os.remove(os.path.join(directory, name)) for name in os.listdir(directory)] and os.rmdir(directory))
        path, out = os.path.join(directory, 'server.log'), os.path.join(directory, 'out.json')
        with open(path, 'w') as handle:
            handle.write('noise\n' + '\n'.join(lines) + '\n[PACKED] noise\n')
        said = []
        self.assertEqual(report.main([path, '--json', out, '--min-rounds', '5'], say=said.append), 0)
        with open(out) as handle:
            written = json.load(handle)
        self.assertEqual(written['problems'], [])
        self.assertTrue(said[0].startswith('[TREEGAIN] requests=1'))
        self.assertEqual(report.main([path, '--min-rounds', '100000'], say=said.append), 1)
        self.assertTrue(any('PROBLEM' in line for line in said))

    def test_two_requests_do_not_share_a_continuation(self):
        _, first, _ = run_request(SHADOW, request_id='a', stream_seed=5)
        _, second, _ = run_request(SHADOW, request_id='b', stream_seed=8, seed=9)
        rounds, bad = report.load(first + second)
        self.assertEqual(sorted(rounds), ['a', 'b'])
        summary = report.analyze(rounds, bad)
        self.assertEqual(summary['requests'], 2)
        self.assertEqual(report.problems(summary), [])


if __name__ == '__main__':
    unittest.main()
