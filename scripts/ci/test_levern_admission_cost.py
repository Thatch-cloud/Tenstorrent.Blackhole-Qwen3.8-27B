"""The deadline governor's admission cost under engine reuse (QWEN_FAST_LEVERN_BUILD_MS; levern_policy.AdmissionCost, effective_share, Alternator.govern).

Unset, the governor charges today's single fixed BUILD_MS = 2500 per governed queue, call for call; a number replaces that constant; `learned` charges EACH
pending prefill the cost of the slot it will take - a rebind while parked engines remain free, else a build - from an EWMA of what the bridge factory
measured (weight 0.2, clamp 100-6000 ms, seeds 2500 and 400). Held here: the flag parses strictly; the flag-off formula equals the base commit's over a grid;
the learned charge is per pending prefill and never lower than the fixed one when builds are charged; the EWMA's arithmetic; the registry the scheduler and
the bridge factory share; and the Alternator's share under each mode."""

import os
from pathlib import Path
import subprocess
import sys
from types import ModuleType
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import levern_policy as policy  # noqa: E402

BASE = '478e2736'


def base_policy():
    try:
        result = subprocess.run(['git', 'show', '%s:scripts/ci/levern_policy.py' % BASE], capture_output=True, cwd=str(ROOT), timeout=60)
    except (OSError, subprocess.SubprocessError):
        raise unittest.SkipTest('no git')
    if result.returncode != 0:
        raise unittest.SkipTest('no git history for %s' % BASE)
    module = ModuleType('levern_policy_base')
    module.__file__ = str(HERE / 'levern_policy.py')
    exec(compile(result.stdout.decode('utf-8'), 'levern_policy.py@%s' % BASE, 'exec'), module.__dict__)
    return module


class FlagTests(unittest.TestCase):
    def test_unset_learned_or_a_number_of_milliseconds(self):
        self.assertIsNone(policy.build_ms_mode({}))
        self.assertEqual(policy.build_ms_mode({policy.BUILD_MS_FLAG: 'learned'}), 'learned')
        self.assertEqual(policy.build_ms_mode({policy.BUILD_MS_FLAG: '400'}), 400.0)
        self.assertEqual(policy.build_ms_mode({policy.BUILD_MS_FLAG: '2500.5'}), 2500.5)
        for value in ('', 'now', '-1', '0', '60001', 'nan', 'inf', '2500ms'):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, policy.BUILD_MS_FLAG):
                policy.build_ms_mode({policy.BUILD_MS_FLAG: value})

    def test_the_flag_is_a_lever_n_sibling_and_a_typo_without_the_master_switch_is_refused(self):
        self.assertIn(policy.BUILD_MS_FLAG, policy.ALL_FLAGS)
        self.assertIn(policy.BUILD_MS_FLAG, policy.SIBLING_FLAGS)
        self.assertEqual(policy.config_problems({policy.FLAG: '1', policy.BUILD_MS_FLAG: 'learned'}), [])
        self.assertEqual(policy.config_problems({policy.FLAG: '1', policy.BUILD_MS_FLAG: 'whenever'}),
                         ['QWEN_FAST_LEVERN_BUILD_MS must be "learned" or a number of milliseconds in (0, 60000], got \'whenever\''])
        self.assertTrue(any('set without' in text for text in policy.config_problems({policy.BUILD_MS_FLAG: 'learned'})))


class FormulaTests(unittest.TestCase):
    PENDING = [(0.0, 20000.0), (5.0, 60000.0), (9.0, 90000.0)]

    def test_flag_off_is_the_base_commits_formula_over_a_grid(self):
        base = base_policy()
        for share in (0.2, 0.5, 0.9):
            for target in (0, 60, 180, 600):
                for pending in ([], self.PENDING[:1], self.PENDING, [(0.0, 5000.0)] * 4):
                    for now in (0.0, 10.0, 100.0, 400.0):
                        with self.subTest(share=share, target=target, pending=len(pending), now=now):
                            self.assertEqual(policy.effective_share(share, target, pending, now),
                                             base.effective_share(share, target, pending, now))
                            self.assertEqual(policy.effective_share(share, target, pending, now, build_ms=400.0),
                                             base.effective_share(share, target, pending, now, build_ms=400.0))

    def test_learned_charges_every_pending_prefill_where_the_fixed_formula_charges_one(self):
        pending, now, target = self.PENDING, 10.0, 300
        fixed = policy.effective_share(0.1, target, pending, now)
        learned_builds = policy.effective_share(0.1, target, pending, now, builds=[2500.0, 2500.0, 2500.0])
        self.assertGreater(learned_builds, fixed, 'three builds are charged, not one')
        # all rebinds: cheaper than today's single build only when the queue is short
        rebinds = policy.effective_share(0.1, target, pending, now, builds=[400.0, 400.0, 400.0])
        self.assertLess(rebinds, learned_builds)
        # by hand: the last prefill's need is (sum D + sum B) / slack
        cumulative = sum(remaining for _, remaining in pending)
        slack = target - (now - pending[2][0])
        self.assertAlmostEqual(policy.effective_share(0.01, target, pending, now, builds=[400.0] * 3), (cumulative + 1200.0) / (slack * 1000.0))

    def test_a_pending_past_its_deadline_gets_the_whole_device_and_the_governor_off_changes_nothing(self):
        self.assertEqual(policy.effective_share(0.5, 180, [(0.0, 1000.0)], 500.0, builds=[400.0]), 1.0)
        self.assertEqual(policy.effective_share(0.5, 0, self.PENDING, 10.0, builds=[400.0] * 3), 0.5)
        self.assertEqual(policy.effective_share(0.5, 180, [], 10.0, builds=[]), 0.5)

    def test_a_short_builds_list_repeats_its_last_charge(self):
        self.assertEqual(policy.effective_share(0.1, 300, self.PENDING, 10.0, builds=[400.0]),
                         policy.effective_share(0.1, 300, self.PENDING, 10.0, builds=[400.0] * 3))


class CostTests(unittest.TestCase):
    def test_the_ewma_moves_a_fifth_of_the_way_and_is_clamped(self):
        cost = policy.AdmissionCost()
        self.assertEqual((cost.predict('build'), cost.predict('rebind')), (2500.0, 400.0))
        self.assertAlmostEqual(cost.observe('rebind', 800.0), 480.0)
        self.assertAlmostEqual(cost.observe('rebind', 800.0), 544.0)
        for _ in range(200):
            cost.observe('build', 10 ** 6)
        self.assertEqual(cost.predict('build'), policy.COST_MAX_MS)
        for _ in range(200):
            cost.observe('build', 1.0)
        self.assertAlmostEqual(cost.predict('build'), policy.COST_MIN_MS, places=3)
        self.assertEqual(cost.observe('rebind', 0.0), cost.predict('rebind'), 'a non-positive reading changes nothing')
        self.assertEqual(cost.observed, dict(build=400, rebind=2))
        with self.assertRaises(ValueError):
            cost.observe('park', 5.0)

    def test_per_pending_charges_a_rebind_while_parked_engines_remain_free_then_a_build(self):
        cost = policy.AdmissionCost()
        self.assertEqual(cost.per_pending(3), [2500.0] * 3, 'no parked set: builds')
        cost.parked_free = lambda: 2
        self.assertEqual(cost.per_pending(4), [400.0, 400.0, 2500.0, 2500.0])
        cost.parked_free = lambda: 0
        self.assertEqual(cost.per_pending(2), [2500.0, 2500.0])
        cost.parked_free = lambda: 1 / 0
        self.assertEqual(cost.per_pending(1), [2500.0], 'a failing reader never fails a schedule')

    def test_the_first_pending_prefill_is_charged_by_the_slot_it_is_placed_on(self):
        cost = policy.AdmissionCost()
        cost.parked_free = lambda: 2
        cost.next_parked = lambda: False
        self.assertEqual(cost.per_pending(4), [2500.0, 400.0, 400.0, 2500.0], 'placement picked a free unparked slot ahead of the parked ones')
        cost.next_parked = lambda: True
        self.assertEqual(cost.per_pending(4), [400.0, 400.0, 2500.0, 2500.0])
        cost.next_parked = lambda: 1 / 0
        self.assertEqual(cost.per_pending(3), [400.0, 400.0, 2500.0], 'a failing reader falls back to the count')
        cost.parked_free = lambda: 0
        cost.next_parked = lambda: True
        self.assertEqual(cost.per_pending(2), [2500.0, 2500.0])

    def test_the_registry_is_one_object_per_process_and_observing_needs_the_learned_mode(self):
        with patch.dict(sys.modules):
            sys.modules.pop(policy.COST_KEY, None)
            self.assertIsNone(policy.admission_cost(create=False))
            with patch.dict(os.environ, {policy.BUILD_MS_FLAG: '400'}):
                self.assertIsNone(policy.observe_admission_cost('rebind', 300.0))
                self.assertIsNone(policy.admission_cost(create=False), 'nothing was created')
            lines = []
            with patch.dict(os.environ, {policy.BUILD_MS_FLAG: 'learned'}):
                value = policy.observe_admission_cost('rebind', 800.0, log=lambda template, *values: lines.append(template.format(*values)))
            self.assertAlmostEqual(value, 480.0)
            self.assertIs(policy.admission_cost(), policy.admission_cost(create=False))
            self.assertEqual(lines, ['[LEVERN] admission cost kind=rebind ms=800.0 ewma=480.0'])


class AlternatorTests(unittest.TestCase):
    def alternator(self, mode):
        environment = {} if mode is None else {policy.BUILD_MS_FLAG: mode}
        with patch.dict(os.environ, environment):
            cfg = policy.config({policy.FLAG: '1'})
            alternator = policy.Alternator(cfg, merged=policy.merged_config({}), wall=lambda: 10.0)
        alternator.pending_hint = [(0.0, 20000.0), (5.0, 60000.0)]
        return alternator

    def test_the_share_under_each_mode(self):
        base = self.alternator(None).govern()
        learned = self.alternator('learned')
        with patch.dict(sys.modules):
            sys.modules.pop(policy.COST_KEY, None)
            builds = learned.govern()
            policy.admission_cost().parked_free = lambda: 2
            rebinds = learned.govern()
        fixed = self.alternator('400').govern()
        self.assertGreaterEqual(builds, base, 'no parked engine free: each pending prefill is a build (two charged, one before)')
        self.assertLessEqual(rebinds, builds)
        self.assertLessEqual(fixed, base, '400 ms replaces 2,500')
        self.assertEqual(self.alternator(None).build_mode, None)
        self.assertEqual(self.alternator('400').build_mode, 400.0)


if __name__ == '__main__':
    unittest.main()
