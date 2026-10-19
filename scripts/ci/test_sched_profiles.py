"""tp4/sched-1: the six gate-only twins of the approved ship-4 profile (make_sched_profiles) and what the levers may and may not touch.

  -sch-lean       THE ADMISSION ARM    the approved profile with the gate marker in place of its owner waiver id, plus QWEN_FAST_ADMISSION_LEAN=1;
  -sch-spill      THE TIER ARM         ... plus QWEN_PREFIX_HOST_TIER_SPILL_MODE=complete;
  -sch-pace       THE PACING ARM       ... plus QWEN_FAST_LEVERN_OWED_CAP_MS / _SHORT_SHARE / _SHORT_WAIT_S;
  -sch-adaptive   THE FEW-DECODER ARM  ... plus QWEN_FAST_LEVERN_ADAPTIVE=1;
  -sch-all        the four together;
  -sch-all-audit  THE AUDITED ATTACH ARM: the ship-4 audited twin plus the four together.

The approved traffic profile and every profile that existed before this branch stay byte-identical (checked against the commit the branch was cut from); nothing here is hand-edited."""

import copy
import json
from pathlib import Path
import subprocess
import sys
import unittest

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import levern_policy  # noqa: E402
import make_sched_profiles as twins  # noqa: E402
import make_ship4_profiles as ship4  # noqa: E402
import profile_twins  # noqa: E402
import qwen_prefix_registry  # noqa: E402
import serving_c2_contract as contract  # noqa: E402
from test_octo3_contract import boot_problems  # noqa: E402

PROFILES_PATH = HERE / 'qwen_c2_profiles.json'
DATA = json.loads(PROFILES_PATH.read_text(encoding='utf-8'))
PROFILES = DATA['profiles']
SHIP, AUDIT_PARENT = twins.SHIP, twins.AUDIT_PARENT
PARENT = '27af6902'
FLAGS = {
    'lean': {'QWEN_FAST_ADMISSION_LEAN': '1'},
    'spill': {'QWEN_PREFIX_HOST_TIER_SPILL_MODE': 'complete'},
    'pace': {'QWEN_FAST_LEVERN_OWED_CAP_MS': '1500', 'QWEN_FAST_LEVERN_SHORT_SHARE': '0.65', 'QWEN_FAST_LEVERN_SHORT_WAIT_S': '5'},
    'adaptive': {'QWEN_FAST_LEVERN_ADAPTIVE': '1'},
}
FLAGS['all'] = FLAGS['all-audit'] = {name: value for arm in ('lean', 'spill', 'pace', 'adaptive') for name, value in FLAGS[arm].items()}
ALL_NAMES = sorted(FLAGS['all'])


def parent_profiles():
    result = subprocess.run(['git', 'show', '%s:scripts/ci/qwen_c2_profiles.json' % PARENT], capture_output=True, cwd=str(HERE), timeout=120)
    if result.returncode != 0:
        return None
    return json.loads(result.stdout.decode('utf-8'))


class NamesTests(unittest.TestCase):
    def test_the_names_fit_a_baked_default_and_are_registered_as_twins(self):
        self.assertEqual(twins.twin_names(), tuple(twins.NAMESPACE + suffix for suffix in ('lean', 'spill', 'pace', 'adaptive', 'all', 'all-audit')))
        for name in twins.twin_names():
            self.assertLessEqual(len(name), 64, name)
            self.assertIn(name, profile_twins.twin_names())
            self.assertIn(name, PROFILES)

    def test_the_levers_are_the_flags_the_modules_and_the_contract_know(self):
        self.assertEqual(sorted(twins.FLAGS), ALL_NAMES)
        for name in ALL_NAMES:
            known = name in levern_policy.ALL_FLAGS or name in contract.SCHED_NAMES or name in contract.PREFIX_TIER_ENV_NAMES or name in contract.PREFIX_POLICY_FLAGS
            self.assertTrue(known, name)
        self.assertEqual(twins.PACE, {levern_policy.OWED_CAP_FLAG: '1500', levern_policy.SHORT_SHARE_FLAG: '0.65', levern_policy.SHORT_WAIT_FLAG: '5'})
        self.assertEqual(twins.SPILL, {qwen_prefix_registry.ENV_TIER_SPILL_MODE: 'complete'})
        self.assertEqual(twins.LEAN, {contract.LEAN_SWITCH: '1'})
        self.assertEqual(twins.ADAPTIVE, {levern_policy.ADAPTIVE_FLAG: '1'})


class TwinTests(unittest.TestCase):
    def test_the_twins_are_gate_only_generated_and_boot_clean(self):
        for name in twins.twin_names():
            profile = PROFILES[name]
            self.assertIs(profile['gate_only'], True, name)
            self.assertNotIn(contract.TRAFFIC_WAIVER, profile)
            self.assertEqual(profile['env']['QWEN_C2_GATE_PROFILE'], '1')
            self.assertEqual(boot_problems(dict(profile, name=name)), [], name)
            self.assertEqual(contract.sched_problems(dict(profile, name=name)), [], name)
            self.assertEqual(contract.levern_problems(dict(profile, name=name)), [], name)
            self.assertEqual(contract.prefix_policy_problems(dict(profile, name=name)), [], name)
        self.assertEqual(twins.generate(DATA), DATA, 'the file is what the generator makes')
        self.assertEqual(twins.main(['--check']), 0)

    def test_each_timed_arm_is_the_approved_profile_with_the_gate_marker_for_the_waiver_id_and_exactly_its_flags(self):
        ship = PROFILES[SHIP]
        for suffix in ('lean', 'spill', 'pace', 'adaptive', 'all'):
            arm = PROFILES[twins.NAMESPACE + suffix]
            changed = {key: (ship['env'].get(key), arm['env'].get(key)) for key in set(ship['env']) | set(arm['env']) if ship['env'].get(key) != arm['env'].get(key)}
            expected = {contract.OCTO_WAIVER_ENV: (ship['env'][contract.OCTO_WAIVER_ENV], None), 'QWEN_C2_GATE_PROFILE': (None, '1')}
            expected.update({name: (None, value) for name, value in FLAGS[suffix].items()})
            self.assertEqual(changed, expected, suffix)
            self.assertEqual(arm['engine'], ship['engine'], 'the memory plan, the pool and the trace region are the parent\'s')
            for key in ship:
                if key not in ('env', 'description', 'gate_only', contract.TRAFFIC_WAIVER):
                    self.assertEqual(arm[key], ship[key], key)
            self.assertIn(SHIP, arm['description'])

    def test_the_audited_arm_is_the_ship4_audited_twin_plus_exactly_the_four_levers(self):
        parent, audit = PROFILES[AUDIT_PARENT], PROFILES[twins.NAMESPACE + 'all-audit']
        self.assertEqual({key: value for key, value in audit['env'].items() if parent['env'].get(key) != value}, FLAGS['all-audit'])
        self.assertEqual(sorted(set(parent['env']) - set(audit['env'])), [])
        for key in parent:
            if key not in ('env', 'description'):
                self.assertEqual(audit[key], parent[key], key)
        for name in ('QWEN_FAST_KV_PAGE_WRITER_AUDIT', 'QWEN_FAST_VERIFY_T2_AUDIT', 'QWEN_PREFIX_HOST_TIER_AUDIT', 'QWEN_FAST_TAP_WINDOW_ALIGNED_AUDIT'):
            self.assertIn(audit['env'].get(name), ('1', 'all'), name)
        self.assertEqual(audit['env']['QWEN_PREFIX_HOST_TIER_VERIFY'], 'all', 'every restored block is read back: the complete spills\' restores are checked byte for byte')

    def test_the_twins_sit_right_after_the_ship4_twins(self):
        names = list(PROFILES)
        last = max(names.index(name) for name in ship4.twin_names())
        self.assertEqual(names[last + 1:last + 1 + len(twins.NAMES)], list(twins.NAMES))

    def test_a_parent_that_cannot_be_twinned_is_refused(self):
        for mutate in (lambda data: data['profiles'][SHIP]['env'].__setitem__('QWEN_FAST_ADMISSION_LEAN', '1'),
                       lambda data: data['profiles'][AUDIT_PARENT]['env'].__setitem__('QWEN_PREFIX_HOST_TIER_SPILL_MODE', 'complete'),
                       lambda data: data['profiles'][SHIP]['env'].__setitem__('QWEN_C2_GATE_PROFILE', '1'),
                       lambda data: data['profiles'][SHIP]['env'].pop('QWEN_FAST_LEVERN_PARK'),
                       lambda data: data['profiles'][SHIP]['env'].pop('QWEN_PREFIX_HOST_TIER_GIB'),
                       lambda data: data['profiles'][AUDIT_PARENT].pop('gate_only'),
                       lambda data: data['profiles'].pop(SHIP)):
            data = copy.deepcopy(DATA)
            mutate(data)
            with self.assertRaises(ValueError):
                twins.generate(data)


class ProtectedTests(unittest.TestCase):
    def test_every_profile_that_existed_before_the_branch_is_byte_identical_and_the_default_is_untouched(self):
        before = parent_profiles()
        if before is None:
            self.skipTest('no git history for %s' % PARENT)
        self.assertEqual(DATA['default'], before['default'])
        for name, profile in before['profiles'].items():
            self.assertEqual(PROFILES[name], profile, name)
        self.assertEqual(sorted(set(PROFILES) - set(before['profiles'])), sorted(twins.NAMES))

    def test_the_approved_traffic_profile_never_names_a_lever_and_a_traffic_profile_that_did_would_be_refused(self):
        for name in ALL_NAMES:
            self.assertNotIn(name, PROFILES[SHIP]['env'], name)
            self.assertNotIn(name, PROFILES[AUDIT_PARENT]['env'], name)
        self.assertEqual(PROFILES[SHIP][contract.TRAFFIC_WAIVER]['decision'][:8], 'APPROVED')
        for name, value in sorted(FLAGS['all'].items()):
            stolen = copy.deepcopy(PROFILES[SHIP])
            stolen['env'][name] = value
            problems = (contract.levern_problems(dict(stolen, name='stolen')) + contract.sched_problems(dict(stolen, name='stolen'))
                        + contract.prefix_policy_problems(dict(stolen, name='stolen')))
            self.assertTrue([problem for problem in problems if name in problem], (name, problems))

    def test_no_other_profile_names_a_lever(self):
        for name, profile in PROFILES.items():
            if name not in twins.NAMES:
                self.assertFalse([flag for flag in ALL_NAMES if flag in profile['env']], name)


class ParsesTests(unittest.TestCase):
    """What the twins' environment parses to in each module that reads it: nothing is silently defaulted."""

    def env(self, suffix):
        return {key: str(value) for key, value in PROFILES[twins.NAMESPACE + suffix]['env'].items()}

    def test_the_levern_flags_of_every_arm_parse_clean(self):
        for suffix in ('lean', 'spill', 'pace', 'adaptive', 'all', 'all-audit'):
            env = self.env(suffix)
            self.assertEqual(levern_policy.config_problems(env), [], suffix)
        pace = self.env('pace')
        self.assertEqual(levern_policy.sched_config(pace), levern_policy.Sched(1500, 0.65, 5))
        self.assertIsNone(levern_policy.sched_config(self.env('lean')))
        self.assertEqual(levern_policy.adaptive_config(self.env('adaptive')), levern_policy.Adaptive(2, 0.9, 120))

    def test_the_tier_flags_of_every_arm_parse_to_a_plan_only_where_asked(self):
        for suffix in ('lean', 'pace', 'adaptive'):
            env = self.env(suffix)
            self.assertIsNone(qwen_prefix_registry.spill_plan(env, qwen_prefix_registry.tier_config(env, 8 << 30)), suffix)
        for suffix in ('spill', 'all', 'all-audit'):
            env = self.env(suffix)
            plan = qwen_prefix_registry.spill_plan(env, qwen_prefix_registry.tier_config(env, 8 << 30))
            self.assertEqual((plan.hard_max, plan.batch), (4096, 512), suffix)


if __name__ == '__main__':
    unittest.main()
