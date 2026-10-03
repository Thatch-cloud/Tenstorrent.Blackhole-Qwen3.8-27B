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


def held(**override):
    """Non-vacuous held-out sets: A1 repositories, A2 sessions, an eval tier of repositories and groups."""
    sets = dict(a1=swe('held-a'), a2=[own('c9', 'p9')], eval_tier=swe('eval-a') + [dict(group='g9')])
    sets.update(override)
    return sets


class GuardTests(unittest.TestCase):
    def test_a_clean_set_passes(self):
        train = swe('train-a', 3) + swe('train-b') + [dict(source='chained', group='g1', repo='train-c'), dict(source='code', group='c1', repo_free=True)]
        self.assertTrue(sp.guard_training_set(train, **held()))

    def test_a_repository_in_a1_is_refused(self):
        with self.assertRaises(sp.SplitError) as caught:
            sp.guard_training_set(swe('shared') + swe('x'), **held(a1=swe('shared')))
        self.assertIn('repo', str(caught.exception))
        self.assertNotIn('shared', str(caught.exception))                          # the count and the key, never the value
        self.assertIn('1 repo', str(caught.exception))

    def test_a_repository_in_the_eval_tier_is_refused(self):
        with self.assertRaises(sp.SplitError):
            sp.guard_training_set(swe('e1'), **held(eval_tier=swe('e1')))

    def test_a_group_in_the_eval_tier_is_refused(self):
        with self.assertRaises(sp.SplitError):
            sp.guard_training_set([dict(source='code', group='g', repo_free=True)], **held(eval_tier=[dict(group='g')]))

    def test_own_traces_are_refused_without_d4_even_if_disjoint(self):
        with self.assertRaises(sp.SplitError):
            sp.guard_training_set([own('c1', 'p1')], **held())
        self.assertTrue(sp.guard_training_set([own('c1', 'p1')], d4_cleared=True, **held()))

    def test_own_traces_must_be_disjoint_by_conversation_and_by_project(self):
        with self.assertRaises(sp.SplitError):
            sp.guard_training_set([own('c9', 'p1')], d4_cleared=True, **held())       # same conversation
        with self.assertRaises(sp.SplitError):
            sp.guard_training_set([own('c1', 'p9')], d4_cleared=True, **held())       # same project, another conversation

    def test_records_without_their_identifiers_are_refused(self):
        for record in (dict(source='swe'), dict(source='own', conversation='c'), dict(source='web'), dict()):
            with self.assertRaises(sp.SplitError):
                sp.guard_training_set([record], d4_cleared=True, **held())


class VacuousGuardTests(unittest.TestCase):
    """The guard that checks nothing was the defect: every way to hand it nothing is now a refusal."""

    def test_the_held_out_sets_have_no_defaults(self):
        with self.assertRaises(TypeError):
            sp.guard_training_set(swe('a'))
        with self.assertRaises(TypeError):
            sp.guard_training_set(swe('a'), a1=swe('h'), a2=[own('c', 'p')])

    def test_an_empty_or_missing_held_out_set_is_refused(self):
        for name in ('a1', 'a2', 'eval_tier'):
            with self.assertRaises(sp.SplitError) as caught:
                sp.guard_training_set(swe('a'), **held(**{name: []}))
            self.assertIn('empty', str(caught.exception))
            with self.assertRaises(sp.SplitError):
                sp.guard_training_set(swe('a'), **held(**{name: None}))
        with self.assertRaises(sp.SplitError):
            sp.guard_training_set(swe('a'), **held(a1='held-a'))

    def test_a_held_out_record_without_its_key_is_refused_not_ignored(self):
        with self.assertRaises(sp.SplitError) as caught:
            sp.guard_training_set(swe('shared'), **held(a1=swe('x') + [dict(source='swe', turn=1)]))       # an A1 record with no repo
        self.assertIn('1 records', str(caught.exception))
        with self.assertRaises(sp.SplitError):
            sp.guard_training_set(swe('a'), **held(a2=[dict(conversation='c9')]))                          # no project
        with self.assertRaises(sp.SplitError):
            sp.guard_training_set(swe('a'), **held(eval_tier=[dict(turn=3)]))                              # neither repo nor group
        with self.assertRaises(sp.SplitError):
            sp.guard_training_set(swe('a'), **held(a1=['not a record']))

    def test_a_chained_or_code_record_with_a_repository_is_checked_against_a1(self):
        train = [dict(source='chained', group='g1', repo='shared')]
        with self.assertRaises(sp.SplitError):
            sp.guard_training_set(train, **held(a1=swe('shared')))
        with self.assertRaises(sp.SplitError):
            sp.guard_training_set([dict(source='code', group='c1', repo='e1')], **held(eval_tier=swe('e1')))

    def test_a_chained_or_code_record_must_name_a_repository_or_say_it_has_none(self):
        for source in ('chained', 'code'):
            with self.assertRaises(sp.SplitError) as caught:
                sp.guard_training_set([dict(source=source, group='g1')], **held())
            self.assertIn('repo_free', str(caught.exception))
            with self.assertRaises(sp.SplitError):
                sp.guard_training_set([dict(source=source, group='g1', repo_free='yes')], **held())              # only a real True counts
            self.assertTrue(sp.guard_training_set([dict(source=source, group='g1', repo_free=True)], **held()))

    def test_every_record_with_a_repo_is_checked_whatever_its_source(self):
        with self.assertRaises(sp.SplitError):
            sp.guard_training_set([dict(source='own', conversation='c1', project='p1', repo='shared')], d4_cleared=True, **held(a1=swe('shared')))


class LoadHeldOutTests(unittest.TestCase):
    def setUp(self):
        import tempfile
        self.root = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.root)

    def write(self, name, records, gz=False):
        import gzip
        import json
        path = os.path.join(self.root, name)
        opener = gzip.open if gz else open
        with opener(path, 'wt', encoding='utf-8') as handle:
            for record in records:
                handle.write(json.dumps(record) + chr(10))

    def full(self):
        self.write(sp.HELD_OUT_FILES['a1'], [dict(id='a', group='gg', repo='r1', turns=[dict(turn_id='t', secret='NOT-LOADED')])])
        self.write(sp.HELD_OUT_FILES['a2'], [dict(id='b', conversation='c1', project='p1')])
        self.write(sp.HELD_OUT_FILES['eval_tier'], [dict(repo='r2'), dict(group='g2')])

    def test_the_three_sets_are_read_and_only_identity_fields_are_kept(self):
        self.full()
        sets = sp.load_held_out(self.root)
        self.assertEqual(sorted(sets), ['a1', 'a2', 'eval_tier'])
        self.assertEqual(sets['a1'], [dict(group='gg', repo='r1')])
        self.assertNotIn('turns', sets['a1'][0])
        self.assertTrue(sp.guard_training_set(swe('fresh'), **sets))
        with self.assertRaises(sp.SplitError):
            sp.guard_training_set(swe('r1'), **sets)

    def test_a_missing_file_an_empty_file_or_a_record_without_its_key_is_refused(self):
        with self.assertRaises(sp.SplitError):
            sp.load_held_out(self.root)
        self.full()
        os.remove(os.path.join(self.root, sp.HELD_OUT_FILES['eval_tier']))
        with self.assertRaises(sp.SplitError):
            sp.load_held_out(self.root)
        self.write(sp.HELD_OUT_FILES['eval_tier'], [])
        with self.assertRaises(sp.SplitError):
            sp.load_held_out(self.root)
        self.write(sp.HELD_OUT_FILES['eval_tier'], [dict(note='x')])
        with self.assertRaises(sp.SplitError):
            sp.load_held_out(self.root)
        self.write(sp.HELD_OUT_FILES['eval_tier'], [dict(repo='r2')])
        self.write(sp.HELD_OUT_FILES['a1'], [dict(group='no-repo')])
        with self.assertRaises(sp.SplitError):
            sp.load_held_out(self.root)

    def test_a_line_that_is_not_json_is_refused(self):
        self.full()
        with open(os.path.join(self.root, sp.HELD_OUT_FILES['a2']), 'a') as handle:
            handle.write('not json' + chr(10))
        with self.assertRaises(sp.SplitError):
            sp.load_held_out(self.root)

    def test_the_messages_name_the_set_and_a_count_never_a_value(self):
        self.full()
        self.write(sp.HELD_OUT_FILES['a2'], [dict(conversation='SENTINEL-CONV')])
        with self.assertRaises(sp.SplitError) as caught:
            sp.load_held_out(self.root)
        self.assertNotIn('SENTINEL', str(caught.exception))


if __name__ == '__main__':
    unittest.main()
