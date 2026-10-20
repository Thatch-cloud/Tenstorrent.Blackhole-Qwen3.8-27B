"""tp4/exact-end: the two gate-only twins of the approved ship-4 profile (make_xe_profiles) and what the exact-resume switches may and may not touch.

  -fx-s4-xe        THE TIMED ARM: the approved ship-4 profile with the gate marker in place of its owner waiver id, plus QWEN_PREFIX_STICKY_UNDROP=1 and QWEN_PREFIX_WINDOW_CARRY=1;
  -fx-s4-xe-audit  THE AUDITED ARM: the ship-4 audited twin plus those two and QWEN_PREFIX_WINDOW_CARRY_AUDIT=1.

The approved traffic profile and every profile that existed before this branch stay byte-identical (checked against the commit the branch merged); nothing here is hand-edited."""

import copy
import json
from pathlib import Path
import subprocess
import sys
import unittest

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import make_ship4_profiles as ship4  # noqa: E402
import make_xe_profiles as twins  # noqa: E402
import profile_twins  # noqa: E402
import serving_c2_contract as contract  # noqa: E402
from test_octo3_contract import boot_problems  # noqa: E402

PROFILES_PATH = HERE / 'qwen_c2_profiles.json'
DATA = json.loads(PROFILES_PATH.read_text(encoding='utf-8'))
PROFILES = DATA['profiles']
SHIP, AUDIT_PARENT, ARM, AUDIT = twins.SHIP, twins.AUDIT_PARENT, twins.ARM, twins.AUDIT
LEVER = dict(twins.LEVER_ENV)
AUDIT_ENV = dict(twins.AUDIT_ENV)
PARENT = '27af6902'         # tp4/swift, the commit this branch merged


def parent_profiles():
    result = subprocess.run(['git', 'show', '%s:scripts/ci/qwen_c2_profiles.json' % PARENT], capture_output=True, cwd=str(HERE), timeout=120)
    if result.returncode != 0:
        return None
    return json.loads(result.stdout.decode('utf-8'))


class NamesTests(unittest.TestCase):
    def test_the_names_fit_a_baked_default_and_sit_under_the_namespace(self):
        for name in (ARM, AUDIT):
            self.assertLessEqual(len(name), 64)
            self.assertTrue(name.startswith(ship4.NAMESPACE))
        self.assertEqual((ARM, AUDIT), (ship4.NAMESPACE + 's4-xe', ship4.NAMESPACE + 's4-xe-audit'))
        self.assertEqual(twins.twin_names(), (ARM, AUDIT))
        for name in twins.twin_names():
            self.assertIn(name, profile_twins.twin_names())

    def test_the_switches_are_the_contracts(self):
        self.assertEqual(sorted(list(LEVER) + list(AUDIT_ENV)), sorted(contract.EXACT_RESUME_NAMES))


class TwinTests(unittest.TestCase):
    def test_the_twins_are_gate_only_generated_and_boot_clean(self):
        for name in (ARM, AUDIT):
            self.assertIs(PROFILES[name]['gate_only'], True)
            self.assertNotIn(contract.TRAFFIC_WAIVER, PROFILES[name])
            for flag, value in LEVER.items():
                self.assertEqual(PROFILES[name]['env'][flag], value)
            self.assertEqual(PROFILES[name]['env']['QWEN_C2_GATE_PROFILE'], '1')
            self.assertEqual(boot_problems(dict(PROFILES[name], name=name)), [], name)
            self.assertEqual(contract.exact_resume_problems(dict(PROFILES[name], name=name)), [], name)
        self.assertEqual(twins.generate(DATA), DATA, 'the file is what the generator makes')
        self.assertEqual(twins.main(['--check']), 0)

    def test_the_arm_is_the_approved_profile_with_the_gate_marker_for_the_waiver_id_and_the_two_switches(self):
        ship, arm = PROFILES[SHIP], PROFILES[ARM]
        changed = {key: (ship['env'].get(key), arm['env'].get(key)) for key in set(ship['env']) | set(arm['env']) if ship['env'].get(key) != arm['env'].get(key)}
        expected = {contract.OCTO_WAIVER_ENV: (ship['env'][contract.OCTO_WAIVER_ENV], None), 'QWEN_C2_GATE_PROFILE': (None, '1')}
        expected.update((flag, (None, value)) for flag, value in LEVER.items())
        self.assertEqual(changed, expected)
        self.assertEqual(arm['engine'], ship['engine'], 'the memory plan, the pool and the trace region are the parent\'s')
        for key in ship:
            if key not in ('env', 'description', 'gate_only', contract.TRAFFIC_WAIVER):
                self.assertEqual(arm[key], ship[key], key)
        self.assertEqual(arm['env']['QWEN_FAST_TAP_WINDOW_ALIGNED'], '1', 'the carried rows join the tile-aligned window, which the candidate carries')
        self.assertIn(SHIP, arm['description'])

    def test_the_audit_twin_is_the_ship4_audit_twin_plus_exactly_the_three_switches(self):
        parent, audit = PROFILES[AUDIT_PARENT], PROFILES[AUDIT]
        expected = dict(LEVER)
        expected.update(AUDIT_ENV)
        self.assertEqual({key: value for key, value in audit['env'].items() if parent['env'].get(key) != value}, expected)
        self.assertEqual(sorted(set(parent['env']) - set(audit['env'])), [])
        for key in parent:
            if key not in ('env', 'description'):
                self.assertEqual(audit[key], parent[key], key)
        self.assertEqual(audit['env']['QWEN_FAST_TAP_WINDOW_ALIGNED_AUDIT'], '1', 'the aligned join of a carried turn is audited against today\'s composition')

    def test_the_twins_sit_right_after_the_ship4_twins(self):
        names = list(PROFILES)
        import make_np1_profiles
        import make_sched_profiles

        # (tp4/next-1: the lone-user round's twins sit right after the ship-4 twins, sched-1's after them, these after those, the prefill-2 and ship-5 twins after these: the generators anchor in that chain)
        last = max(names.index(name) for name in list(ship4.twin_names()) + list(make_np1_profiles.twin_names()) + list(make_sched_profiles.twin_names()))
        self.assertEqual(names[last + 1:last + 3], list(twins.NAMES))

    def test_a_parent_that_cannot_be_twinned_is_refused(self):
        for mutate in (lambda data: data['profiles'][SHIP]['env'].__setitem__('QWEN_PREFIX_WINDOW_CARRY', '1'),
                       lambda data: data['profiles'][AUDIT_PARENT]['env'].__setitem__('QWEN_PREFIX_STICKY_UNDROP', '1'),
                       lambda data: data['profiles'][AUDIT_PARENT]['env'].__setitem__('QWEN_PREFIX_WINDOW_CARRY_AUDIT', '1'),
                       lambda data: data['profiles'][SHIP]['env'].__setitem__('QWEN_C2_GATE_PROFILE', '1'),
                       lambda data: data['profiles'][SHIP]['env'].__setitem__('QWEN_FAST_STICKY_SESSIONS', '0'),
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
        added = sorted(set(PROFILES) - set(before['profiles']))
        self.assertTrue(set(twins.NAMES) <= set(added))
        import make_np1_profiles
        import make_pf2_profiles
        import make_s5_profiles
        import make_sched_profiles
        import make_pf3k_profiles

        # (tp4/next-1 integrated this branch with the lone-user round's, prefill-2's and sched-1's twins and the ship-5 candidate and its audited twins)
        added = [name for name in added if name not in make_np1_profiles.twin_names() and name not in make_s5_profiles.owned_names() and name not in make_pf2_profiles.twin_names()
                 and name not in make_sched_profiles.twin_names() and name not in make_pf3k_profiles.twin_names()]
        # (the other two are the ER package's own gate-only twins of the older combined arm, placed by make_fusion_profiles.py from fusion-wp/ER.json)
        self.assertEqual(sorted(name for name in added if name not in twins.NAMES), sorted(name for name in added if name.endswith(('-fx-all-er', '-fx-all-er-audit'))))
        self.assertEqual(len(added), 4)

    def test_the_approved_traffic_profile_never_names_the_switches_and_a_traffic_profile_that_did_would_be_refused(self):
        for flag in contract.EXACT_RESUME_NAMES:
            self.assertNotIn(flag, PROFILES[SHIP]['env'])
        self.assertEqual(contract.exact_resume_problems(dict(PROFILES[SHIP], name=SHIP)), [])
        stolen = copy.deepcopy(PROFILES[SHIP])
        stolen['env'].update(LEVER)
        problems = contract.exact_resume_problems(dict(stolen, name='stolen'))
        self.assertTrue(any('gate-only' in problem for problem in problems), problems)
        self.assertEqual(PROFILES[SHIP][contract.TRAFFIC_WAIVER]['decision'][:8], 'APPROVED')

    def test_an_inherited_switch_is_dropped_under_a_profile_that_never_named_it(self):
        for name in (SHIP, AUDIT_PARENT):
            environ = dict((flag, '1') for flag in contract.EXACT_RESUME_NAMES)
            contract.apply_environment(dict(PROFILES[name], name=name), environ)
            for flag in contract.EXACT_RESUME_NAMES:
                self.assertNotIn(flag, environ, name)
        environ = {}
        contract.apply_environment(dict(PROFILES[ARM], name=ARM), environ)
        self.assertEqual({flag: environ.get(flag) for flag in contract.EXACT_RESUME_NAMES}, {'QWEN_PREFIX_STICKY_UNDROP': '1', 'QWEN_PREFIX_WINDOW_CARRY': '1',
                                                                                           'QWEN_PREFIX_WINDOW_CARRY_AUDIT': None})


if __name__ == '__main__':
    unittest.main()
