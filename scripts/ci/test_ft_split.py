"""ft_split: the guard refuses every overlap and every unauthorised own trace; the mix and the split are deterministic."""
import os
import sys
import unittest

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import ft_split as sp  # noqa: E402


def swe(repo, n=1):
    return [dict(source='swe', repo=repo, turn=i) for i in range(n)]


def own(conversation, project):
    return dict(source='own', conversation=conversation, project=project)


class MixTests(unittest.TestCase):
    def test_the_default_mix_is_the_d4_fallback_and_needs_no_own_trace(self):
        self.assertEqual(sp.check_mix(sp.DEFAULT_MIX), list(sp.DEFAULT_MIX))
        self.assertEqual(dict(sp.DEFAULT_MIX), dict(swe=0.5, chained=0.3, code=0.2))

    def test_refusals(self):
        for mix in ([('swe', 0.6), ('code', 0.3)], [('swe', 1.0), ('swe', 0.0)], [('web', 1.0)], [('swe', 1.2), ('code', -0.2)]):
            with self.assertRaises(sp.SplitError):
                sp.check_mix(mix)

    def test_own_needs_d4(self):
        mix = [('swe', 0.5), ('own', 0.5)]
        with self.assertRaises(sp.SplitError):
            sp.check_mix(mix)
        self.assertEqual(sp.check_mix(mix, d4_cleared=True), mix)

    def test_counts_sum_to_the_total(self):
        for total in (1, 7, 100, 1001):
            counts = sp.mix_counts(total, sp.DEFAULT_MIX)
            self.assertEqual(sum(counts.values()), total)
        self.assertEqual(sp.mix_counts(100, sp.DEFAULT_MIX), dict(swe=50, chained=30, code=20))


class SplitTests(unittest.TestCase):
    def test_a_value_lands_on_one_side_only_and_the_split_is_deterministic(self):
        records = [dict(repo='r%d' % (i % 40), i=i) for i in range(400)]
        train, held = sp.split_by_key(records, 'repo', 0.25, seed=3)
        self.assertEqual(len(train) + len(held), 400)
        self.assertFalse(sp.values_of(train, 'repo') & sp.values_of(held, 'repo'))
        again = sp.split_by_key(records, 'repo', 0.25, seed=3)
        self.assertEqual([r['i'] for r in train], [r['i'] for r in again[0]])
        other = sp.split_by_key(records, 'repo', 0.25, seed=4)
        self.assertNotEqual([r['i'] for r in train], [r['i'] for r in other[0]])
        self.assertTrue(10 < len(sp.values_of(held, 'repo')) < 30)               # about a quarter of 40 repositories

    def test_bad_fractions_and_missing_keys(self):
        with self.assertRaises(sp.SplitError):
            sp.split_by_key([dict(repo='a')], 'repo', 0.0, 1)
        with self.assertRaises(sp.SplitError):
            sp.split_by_key([dict(other='a')], 'repo', 0.5, 1)


class GuardTests(unittest.TestCase):
    def test_a_clean_set_passes(self):
        train = swe('train-a', 3) + swe('train-b') + [dict(source='chained', group='g1'), dict(source='code', group='c1')]
        self.assertTrue(sp.guard_training_set(train, a1=swe('held-a'), eval_tier=swe('eval-a') + [dict(group='g9')]))

    def test_a_repository_in_a1_is_refused(self):
        with self.assertRaises(sp.SplitError) as caught:
            sp.guard_training_set(swe('shared') + swe('x'), a1=swe('shared'))
        self.assertIn('repo', str(caught.exception))
        self.assertNotIn('shared', str(caught.exception))                          # the count and the key, never the value
        self.assertIn('1 repo', str(caught.exception))

    def test_a_repository_in_the_eval_tier_is_refused(self):
        with self.assertRaises(sp.SplitError):
            sp.guard_training_set(swe('e1'), eval_tier=swe('e1'))

    def test_a_group_in_the_eval_tier_is_refused(self):
        with self.assertRaises(sp.SplitError):
            sp.guard_training_set([dict(source='code', group='g')], eval_tier=[dict(group='g')])

    def test_own_traces_are_refused_without_d4_even_if_disjoint(self):
        with self.assertRaises(sp.SplitError):
            sp.guard_training_set([own('c1', 'p1')], a2=[own('c9', 'p9')])
        self.assertTrue(sp.guard_training_set([own('c1', 'p1')], a2=[own('c9', 'p9')], d4_cleared=True))

    def test_own_traces_must_be_disjoint_by_conversation_and_by_project(self):
        a2 = [own('c9', 'p9')]
        with self.assertRaises(sp.SplitError):
            sp.guard_training_set([own('c9', 'p1')], a2=a2, d4_cleared=True)       # same conversation
        with self.assertRaises(sp.SplitError):
            sp.guard_training_set([own('c1', 'p9')], a2=a2, d4_cleared=True)       # same project, another conversation

    def test_records_without_their_identifiers_are_refused(self):
        for record in (dict(source='swe'), dict(source='own', conversation='c'), dict(source='web'), dict()):
            with self.assertRaises(sp.SplitError):
                sp.guard_training_set([record], d4_cleared=True)


if __name__ == '__main__':
    unittest.main()
