"""Phase-1 quick wins: quickwin_markers, and the quickwin-ledger and quickwin-warm plans of c2_serving_gate (arms built
against the checkout's real profiles, judged on synthetic arm reports, the server logs read by the real producers' lines)."""
import copy
import json
import os
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import c2_serving_gate as gate  # noqa: E402
import c2_serving_job as job  # noqa: E402
import quickwin_markers as markers  # noqa: E402
import serving_fast_policy as policy  # noqa: E402
from test_parked_gate import Feed, synthetic  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(HERE, 'qwen_c2_profiles.json'), encoding='utf-8') as handle:
    PROFILE_SET = json.load(handle)
STICKY = 'c2-packed-prefix'
PLANS = job.QUICKWIN_GATE_PLANS
LEDGER_LINES = ['[MEMLEDGER] phase=P7 point=after_attach chip0 allocated=1.0GB', '[MEMLEDGER] before op=engine chip0 free=1']


def warm_line(run, skipped, request='r'):
    return '%srun=%s skipped=%s request=%s' % (policy.ENGINE_WARM_MARKER, run, skipped, request)


def arms_of(plan, profile=STICKY, **kwargs):
    return gate.plan_arms(plan, profile, PROFILE_SET, **kwargs)


def by_name(arms):
    return dict((arm[0], arm) for arm in arms)


class MarkerTests(unittest.TestCase):
    def test_the_marker_is_the_policy_s(self):
        self.assertEqual(markers.WARM_MARKER, policy.ENGINE_WARM_MARKER)

    def test_scan_reads_both_producers_lines(self):
        facts = markers.scan(LEDGER_LINES + [warm_line(3, 0, 'abc'), warm_line(0, 3, 'def'),
                                             policy.ENGINE_WARM_MARKER + 'run=0 skipped=3 slot=2',
                                             policy.ENGINE_WARM_MARKER + 'run=None skipped=None request=x',
                                             policy.ENGINE_WARM_MARKER + 'garbled'])
        self.assertEqual(facts['ledger'], 2)
        self.assertEqual(facts['warm_lines'], 5)
        self.assertEqual([(e['run'], e['skipped'], e['kind']) for e in facts['warm']],
                         [(3, 0, 'request'), (0, 3, 'request'), (0, 3, 'slot'), (None, None, 'request')])

    def test_ledger_judgement(self):
        ledger, quiet = markers.scan(LEDGER_LINES), markers.scan(['unrelated'])
        self.assertEqual(markers.problems(quiet, ledger_off=True), ([], []))
        self.assertEqual(len(markers.problems(ledger, ledger_off=True)[0]), 1)
        self.assertEqual(markers.problems(ledger, ledger_reference=True), ([], []))
        found, missing = markers.problems(quiet, ledger_reference=True)
        self.assertEqual((found, len(missing)), ([], 1))

    def test_warm_judgement(self):
        good = markers.scan([warm_line(3, 0), warm_line(0, 3)])
        self.assertEqual(markers.problems(good, warm_skip=True), ([], []))
        self.assertEqual(markers.problems(markers.scan([warm_line(3, 0), warm_line(3, 0)]), warm_skip=True)[1][0][:2], 'no')
        self.assertEqual(len(markers.problems(markers.scan([warm_line(0, 3)]), warm_skip=True)[0]), 1,
                         'a first build that skipped is wrong')
        found, missing = markers.problems(markers.scan([]), warm_skip=True)
        self.assertEqual((found, len(missing)), ([], 1))
        found, _ = markers.problems(markers.scan([warm_line(3, 0), 'x %sgarbled' % markers.WARM_MARKER]), warm_skip=True)
        self.assertTrue(any('do not parse' in text for text in found))
        found, _ = markers.problems(markers.scan([warm_line(3, 0)]), warm_skip=False)
        self.assertEqual(len(found), 1, 'a flag-off arm shows no warm line')
        self.assertEqual(markers.problems(markers.scan(LEDGER_LINES), warm_skip=False), ([], []))


class PlanTests(unittest.TestCase):
    def test_each_plan_is_four_arms_on_the_profile_with_only_the_flag_differing(self):
        for plan, (flag, value) in zip(PLANS, (('QWEN_FAST_MEMORY_LEDGER_OFF', '1'), ('QWEN_FAST_ENGINE_WARM_SKIP', '1'))):
            with self.subTest(plan=plan):
                arms = by_name(arms_of(plan))
                self.assertEqual(sorted(arms), sorted('%s-%s-%s' % (plan, state, label)
                                                      for state in ('off', 'on') for label in ('solo', 'live')))
                for label in ('solo', 'live'):
                    off, on = arms['%s-off-%s' % (plan, label)], arms['%s-on-%s' % (plan, label)]
                    self.assertEqual(off[1], on[1], 'the same prompts and budgets')
                    self.assertEqual(dict(on.env), dict(dict(off.env), **{flag: value}))
                    self.assertNotIn(flag, dict(off.env))
                    self.assertEqual((off.judged, on.judged), (False, True))
                    self.assertEqual((off.extra['quick']['on'], on.extra['quick']['on']), (False, True))
                    self.assertEqual(on.profile, STICKY)
                    self.assertEqual(on[1][on[1].index('--expect-profile') + 1], STICKY)
                for arm in arms.values():
                    for name, _ in arm.env:
                        self.assertIn(name, gate.ARM_ENV_NAMES)
                self.assertLess(gate.worst_case_seconds([plan], {plan: list(arms.values())}), 380 * 60 - 600)

    def test_the_solo_chain_varies_alignment_and_the_bucket_set_and_the_live_arm_is_four_users(self):
        arms = by_name(arms_of('quickwin-warm'))
        solo = gate.asked(arms['quickwin-warm-off-solo'][1])
        self.assertEqual(solo['lengths'], list(gate.QUICKWIN_LENGTHS))
        self.assertEqual(solo['streams'], len(gate.QUICKWIN_LENGTHS))
        self.assertEqual(min(solo['budgets']), 2, 'a one-bucket engine')
        self.assertGreaterEqual(max(solo['budgets']), 16, 'a three-bucket engine')
        self.assertTrue(any(length % 64 for length in solo['lengths']) and any(length % 64 == 0 for length in solo['lengths']))
        live = arms['quickwin-warm-on-live'][1]
        self.assertEqual(gate.asked(live)['streams'], 4)
        self.assertIn('--alive-check', live)
        self.assertIn('--user-ignore-eos', arms['quickwin-warm-on-solo'][1])

    def test_a_profile_without_the_extent_flag_is_refused_and_lengths_replace_the_chain(self):
        for profile in ('c2', 'general', 'exact'):
            with self.subTest(profile=profile), self.assertRaisesRegex(gate.PlanError, 'must have'):
                arms_of('quickwin-ledger', profile)
        notes = []
        arms = by_name(arms_of('quickwin-ledger', lengths=[100, 5000], notes=notes))
        self.assertEqual(gate.asked(arms['quickwin-ledger-on-solo'][1])['lengths'], [100, 5000])
        self.assertEqual(len(notes), 1)
        with self.assertRaises(gate.PlanError):
            arms_of('quickwin-ledger', lengths=[10])

    def test_the_parked_profile_is_served_too_and_the_plans_are_jobs(self):
        arms = arms_of('quickwin-warm', 'c2-packed-prefix-parked')
        self.assertEqual({arm.profile for arm in arms}, {'c2-packed-prefix-parked'})
        for plan in PLANS:
            self.assertIn(plan, job.ALL_GATE_PLANS)
        self.assertTrue(set(PLANS).isdisjoint(job.PARKED_GATE_PLANS + job.S2_GATE_PLANS + job.GATE_PLANS))
        self.assertTrue(job.QUICKWIN_GATE_PLANS == gate.QUICKWIN_PLANS)


class JudgeTests(unittest.TestCase):
    def run_plan(self, plan, table):
        arms = arms_of(plan)
        feed = Feed(table)
        with patch.object(gate, 'run_arm', feed):
            result = gate.run_quickwin_plan(plan, None, PROFILE_SET, arms)
        return result, feed, arms

    def table(self, plan, change=None, warm=None, ledger=None):
        table = {}
        for arm in arms_of(plan):
            name = arm[0]
            quick = dict(on=arm.extra['quick']['on'], kind=arm.extra['quick']['kind'],
                         ledger_lines=(ledger or {}).get(name, 0 if arm.extra['quick']['on'] else 5),
                         warm=(warm or {}).get(name, [dict(run=3, skipped=0), dict(run=0, skipped=3)]
                                               if arm.extra['quick']['on'] else []))
            table[name] = (lambda spec, name=name, quick=quick: synthetic(
                spec, text_of=(change or {}).get(name) or (lambda user: 'ref %d' % user), extra=dict(c2_gate_quickwin=quick)))
        return table

    def test_identical_texts_pass_solo_and_live(self):
        for plan in PLANS:
            with self.subTest(plan=plan):
                result, feed, arms = self.run_plan(plan, self.table(plan))
                self.assertEqual(result['verdict'], 'PASS', result['lines'])
                self.assertEqual(feed.asked, [arm[0] for arm in arms])
                self.assertEqual(sorted(result['facts']), sorted(['solo', 'live'] + [arm[0] for arm in arms]))

    def test_one_changed_token_fails_the_arm_it_is_in_and_only_the_on_arms_are_compared(self):
        for plan in PLANS:
            for label in ('solo', 'live'):
                name = '%s-on-%s' % (plan, label)
                with self.subTest(arm=name):
                    result, _, _ = self.run_plan(plan, self.table(plan, change={
                        name: lambda user: 'ref %d' % user if user != 2 else 'other'}))
                    self.assertEqual(result['verdict'], 'FAIL')
                    self.assertTrue(any(name in text for text in result['s2_problems']), result['s2_problems'])
                # a changed OFF arm is a different reference, and the on arm then differs from it
                off = '%s-off-%s' % (plan, label)
                result, _, _ = self.run_plan(plan, self.table(plan, change={off: lambda user: 'x %d' % user}))
                self.assertEqual(result['verdict'], 'FAIL')

    def test_the_arms_own_log_verdicts_reach_the_plan_through_the_runner(self):
        # Runner.run turns quick_arm_check's problems into report['c2_gate_problems'] and its shortfalls into unjudged;
        # here the check itself on the producers' lines.
        problems, missing, record = gate.quick_arm_check('\n'.join(LEDGER_LINES), dict(kind='ledger', on=True))
        self.assertEqual(len(problems), 1)
        self.assertEqual(record['ledger_lines'], 2)
        problems, missing, _ = gate.quick_arm_check('nothing', dict(kind='ledger', on=False))
        self.assertEqual((problems, len(missing)), ([], 1))
        problems, missing, record = gate.quick_arm_check('\n'.join([warm_line(3, 0), warm_line(0, 3)]),
                                                         dict(kind='warm', on=True))
        self.assertEqual((problems, missing), ([], []))
        self.assertEqual([entry['skipped'] for entry in record['warm']], [0, 3])
        problems, _, _ = gate.quick_arm_check(warm_line(3, 0), dict(kind='warm', on=False))
        self.assertEqual(len(problems), 1)
        self.assertEqual(gate.quick_arm_check(None, dict(kind='warm', on=True))[0][0][:2], 'no')

    def test_the_gate_holds_the_ledger_and_warm_flags_to_the_rules_the_comparison_needs(self):
        import real_text_compare

        self.assertIn('QWEN_FAST_MEMORY_LEDGER_OFF', real_text_compare.ARITHMETIC_NEUTRAL)
        self.assertIn('QWEN_FAST_ENGINE_WARM_SKIP', real_text_compare.S2_EXACT_CLAIMS)
        self.assertTrue(gate.counts_cache(list(PLANS), PROFILE_SET, 'general', 'auto'))


if __name__ == '__main__':
    unittest.main()
