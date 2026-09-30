"""Stage E, E8: parked_judge - the rungs, the drafter-equivalence judge, the negative controls, the churn script and
the small judgements of the parked gates - on synthetic reports and logs."""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import parked_judge as judge  # noqa: E402
import real_text_prompts  # noqa: E402


def encoded(*rounds):
    """One user's path records as a report carries them: 'S<position>:<prefix>:<emitted>:<rows>:<cap>'."""
    return ' '.join('%s%s:%d:%d:16:-' % (path, position, prefix, emitted) for path, position, prefix, emitted in rounds)


def report(users, texts=None, paths=None, means=None, rounds=40):
    """A synthetic gate report: `users` streams, their texts, path records per user, and full-draft acceptance."""
    texts = texts or ['answer %d' % user for user in range(users)]
    paths = paths or dict((user, [('S', 100 * (user + 1), 3, 3), ('S', 100 * (user + 1) + 3, 5, 5),
                                  ('P', 100 * (user + 1) + 8, 2, 2)]) for user in range(users))
    means = means or [3.5] * users
    return dict(
        streams=[dict(text=text, finish_reason='length', completion_tokens=16) for text in texts],
        real_text=dict(users=[dict(user=i, prompt_sha256='%064x' % (i + 1), prompt_tokens=100) for i in range(users)]),
        comparisons=[dict(user=i, prompt_sha256='%064x' % (i + 1), max_tokens=16) for i in range(users)],
        max_tokens=16, qwen_configuration=dict(QWEN_FAST_SDPA_PF='1'),
        s2=dict(paths=dict(users=dict((str(user), dict(encoded=encoded(*rounds))) for user, rounds in paths.items()))),
        acceptance=dict(users=[dict(user=i, full_draft=dict(rounds=rounds, mean_emitted=mean))
                               for i, mean in enumerate(means)]))


class RungTests(unittest.TestCase):
    def test_the_chain_covers_the_budgets_and_leads_with_the_longest_prompt(self):
        lengths, budgets, ignore, permutation = judge.chain('descending')
        self.assertEqual(lengths[0], 123136)
        self.assertEqual(lengths, sorted(lengths, reverse=True))
        self.assertEqual(len(lengths), len(budgets))
        self.assertEqual(set(budgets), set(judge.BUDGET_SET))
        self.assertEqual(ignore, list(range(0, len(lengths), 2)))
        self.assertIsNone(permutation)
        for length, budget in zip(lengths, budgets):
            self.assertLessEqual(length + budget, 131328)
        self.assertTrue(all(length >= real_text_prompts.COMPACT_MIN_TARGET for length in lengths),
                        'every rung can be sent as real text')

    def test_the_three_orders_are_permutations_of_one_chain(self):
        count = len(judge.RUNGS)
        ascending = judge.chain('ascending')[3]
        shuffled = judge.chain('shuffled')[3]
        self.assertEqual(sorted(ascending), list(range(count)))
        self.assertEqual(sorted(shuffled), list(range(count)))
        self.assertEqual(ascending, list(reversed(range(count))))
        self.assertNotIn(shuffled, (ascending, list(range(count))))
        self.assertEqual(judge.chain('ascending')[:3], judge.chain('descending')[:3], 'the same prompts and budgets')
        with self.assertRaises(ValueError):
            judge.chain('sideways')

    def test_the_churn_script_is_deterministic_and_bounded(self):
        first = judge.churn_script(0, 50, 100000)
        self.assertEqual(first, judge.churn_script(0, 50, 100000))
        self.assertNotEqual(first, judge.churn_script(1, 50, 100000))
        lengths, budgets = first
        self.assertEqual((len(lengths), len(budgets)), (50, 50))
        self.assertLessEqual(max(lengths), 100000)
        self.assertTrue(set(budgets) <= {16, 64, 128, 256, 512})
        self.assertGreater(len(set(lengths)), 6, 'a spread of lengths')
        self.assertTrue(all(length >= real_text_prompts.COMPACT_MIN_TARGET for length in lengths))


class EquivalenceTests(unittest.TestCase):
    def test_identical_drafting_passes_and_a_changed_round_is_named(self):
        reference = report(3)
        self.assertEqual(judge.equivalence(reference, report(3), users=3)[:2], ([], []))
        changed = report(3)
        changed['s2']['paths']['users']['1']['encoded'] = encoded(('S', 200, 3, 3), ('S', 203, 4, 4), ('P', 207, 2, 2))
        problems, shortfalls, facts = judge.equivalence(reference, changed, users=3)
        self.assertEqual(len(problems), 1)
        self.assertIn('user 1 drafts differently', problems[0])
        self.assertIn('round 1', problems[0])
        self.assertFalse(facts['solo']['identical'])

    def test_a_missing_user_is_unseen_never_a_pass(self):
        other = report(3)
        del other['s2']['paths']['users']['2']
        problems, shortfalls, facts = judge.equivalence(report(3), other, users=3)
        self.assertEqual(problems, [])
        self.assertEqual(len(shortfalls), 1)
        self.assertFalse(facts['solo']['identical'])

    def test_a_shorter_sequence_differs_at_its_end(self):
        other = report(1, paths={0: [('S', 100, 3, 3)]})
        problems, _, _ = judge.equivalence(report(1), other, users=1)
        self.assertEqual(len(problems), 1)
        self.assertIn('round 1', problems[0])

    def test_acceptance_is_the_share_of_the_fifteen_proposals(self):
        value, rounds = judge.acceptance(report(2, means=[4.0, 4.0], rounds=30))
        self.assertAlmostEqual(value, 3.0 / 15)
        self.assertEqual(rounds, 60)
        self.assertEqual(judge.acceptance({}), (None, 0))

    def test_concurrent_acceptance_within_two_percent_absolute(self):
        base, close, far = report(2, means=[4.0, 4.0]), report(2, means=[4.2, 4.2]), report(2, means=[4.6, 4.6])
        self.assertEqual(judge.equivalence(report(2), report(2), base, close, users=2)[:2], ([], []))
        problems, _, facts = judge.equivalence(report(2), report(2), base, far, users=2)
        self.assertEqual(len(problems), 1)
        self.assertIn('concurrent acceptance', problems[0])
        self.assertAlmostEqual(facts['live']['delta'], 0.6 / 15, places=4)
        empty = dict(report(2), acceptance={})
        _, shortfalls, _ = judge.equivalence(report(2), report(2), base, empty, users=2)
        self.assertEqual(len(shortfalls), 1)


class NegativeControlTests(unittest.TestCase):
    def test_the_carry_control_must_change_a_token(self):
        reference = report(3)
        broken = report(3, texts=['answer 0', 'garbage', 'answer 2'])
        problems, facts = judge.negative_verdict('carry', reference, broken)
        self.assertEqual(problems, [])
        self.assertEqual(facts['users'], ['IDENTICAL', 'DIVERGED', 'IDENTICAL'])
        problems, _ = judge.negative_verdict('carry', reference, report(3))
        self.assertEqual(len(problems), 1)
        self.assertIn('kept every token', problems[0])

    def test_the_drafter_control_keeps_tokens_and_must_fail_the_equivalence_judge(self):
        reference = report(3)
        stale = report(3)
        stale['s2']['paths']['users']['0']['encoded'] = encoded(('S', 100, 1, 1), ('S', 101, 1, 1), ('P', 102, 1, 1))
        problems, facts = judge.negative_verdict('drafter', reference, stale)
        self.assertEqual(problems, [], 'same tokens and different drafting is exactly what the control must show')
        self.assertFalse(facts['equivalence']['solo']['identical'])
        # a control the judge cannot see is a problem: identical drafting means a broken rebind would pass
        problems, _ = judge.negative_verdict('drafter', reference, report(3))
        self.assertEqual(len(problems), 1)
        self.assertIn('invisible to the equivalence judge', problems[0])
        # a control that changed tokens is not the drafter-only break
        problems, _ = judge.negative_verdict('drafter', reference, report(3, texts=['x', 'answer 1', 'answer 2']))
        self.assertTrue(any('changed tokens' in text for text in problems))

    def test_an_unknown_control_is_refused(self):
        with self.assertRaises(ValueError):
            judge.negative_verdict('both', report(1), report(1))


class SmallJudgementTests(unittest.TestCase):
    def test_log_counts(self):
        text = '\n'.join(['[PACKED-PROPOSE] pair=[0, 1] fallback=dram_reserve', 'x',
                          'ValueError: Committed history exceeds prepared request contexts',
                          '[PINDIAG] quad draft disabled round=3', '[PINDIAG] verify t2 KV_SHARED slot=1',
                          '[PACKED-PROPOSE] pair=[0, 1] packed'])
        counts = judge.log_counts(text)
        self.assertEqual(counts, dict(pair_fallbacks=1, history_exceeded=1, quad_disabled=1, kv_shared=1))
        self.assertEqual(len(judge.count_problems(counts, 'arm')), 4)
        self.assertEqual(judge.count_problems(judge.log_counts(''), 'arm'), [])

    def test_the_idle_return_is_within_sixteen_megabytes(self):
        self.assertEqual(judge.idle_return({'0': 0.010, '1': -0.016}), ([], []))
        problems, _ = judge.idle_return({'0': 0.017, '1': 0.0})
        self.assertEqual(len(problems), 1)
        self.assertEqual(judge.idle_return(None)[0], [])
        self.assertEqual(len(judge.idle_return(None)[1]), 1)

    def test_the_trace_spread(self):
        self.assertEqual(judge.trace_spread(dict(readings=4, min_used_gb=0.2, max_used_gb=0.25)), ([], []))
        problems, _ = judge.trace_spread(dict(readings=4, min_used_gb=0.2, max_used_gb=0.3))
        self.assertEqual(len(problems), 1)
        self.assertEqual(len(judge.trace_spread(dict(readings=0))[1]), 1)
        self.assertEqual(len(judge.trace_spread(dict(readings=2, max_used_gb=0.3))[1]), 1)

    def test_the_ballast_leaves_the_transient_the_rebind_and_the_stranded_bytes(self):
        import serving_prefill_admission as admission

        free = 1_400_000_000
        need = admission.PREFILL_TRANSIENT_BYTES + admission.PARKED_REBIND_BYTES + admission.STRANDED_BYTES
        self.assertEqual(judge.ballast_advice(free), free - need)
        self.assertEqual(judge.ballast_advice(100), 0)
        self.assertIsNone(judge.ballast_advice(None))


if __name__ == '__main__':
    unittest.main()
