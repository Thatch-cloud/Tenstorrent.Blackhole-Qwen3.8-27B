"""tp4/prefill-2: the three gate-only twins of the approved ship-4 profile (make_pf2_profiles) and what the hand-off flag may and may not touch.

  -pf2-s4-ctl     THE CONTROL OF THE TIMED PAIR: the approved ship-4 profile with the gate marker in place of its owner waiver id, plus QWEN_FAST_PREFILL_HANDOFF=0 (the host path, timed);
  -pf2-s4         THE TIMED ARM: the same with QWEN_FAST_PREFILL_HANDOFF=1;
  -pf2-s4-audit   THE AUDITED ATTACH ARM: the ship-4 audited twin plus the flag and its audit.

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

import levern_route  # noqa: E402
import make_pf2_profiles as twins  # noqa: E402
import make_ship4_profiles as ship4  # noqa: E402
import prefill_handoff  # noqa: E402
import profile_twins  # noqa: E402
import serving_c2_contract as contract  # noqa: E402
from test_octo3_contract import boot_problems  # noqa: E402

PROFILES_PATH = HERE / 'qwen_c2_profiles.json'
DATA = json.loads(PROFILES_PATH.read_text(encoding='utf-8'))
PROFILES = DATA['profiles']
SHIP, AUDIT_PARENT, CONTROL, ARM, AUDIT = twins.SHIP, twins.AUDIT_PARENT, twins.CONTROL, twins.ARM, twins.AUDIT
FLAG, AUDIT_FLAG = twins.FLAG, twins.AUDIT_FLAG
PARENT = '27af6902'


def parent_profiles():
    result = subprocess.run(['git', 'show', '%s:scripts/ci/qwen_c2_profiles.json' % PARENT], capture_output=True, cwd=str(HERE), timeout=120)
    if result.returncode != 0:
        return None
    return json.loads(result.stdout.decode('utf-8'))


def changed(parent, child):
    return {key: (parent.get(key), child.get(key)) for key in set(parent) | set(child) if parent.get(key) != child.get(key)}


class NamesTests(unittest.TestCase):
    def test_the_names_fit_a_profile_name_and_are_the_generators(self):
        for name in twins.NAMES:
            self.assertLessEqual(len(name), 64)
        self.assertEqual((CONTROL, ARM, AUDIT), tuple(twins.PREFIX + suffix for suffix in ('s4-ctl', 's4', 's4-audit')))
        self.assertEqual(twins.twin_names(), (CONTROL, ARM, AUDIT))
        for name in twins.twin_names():
            self.assertIn(name, profile_twins.twin_names())
            self.assertFalse(name.startswith(ship4.NAMESPACE), 'outside the fusion generator\'s namespace: make_fusion_profiles leaves them alone')

    def test_the_flags_are_the_modules_and_the_routes(self):
        self.assertEqual((FLAG, AUDIT_FLAG), (prefill_handoff.FLAG, prefill_handoff.AUDIT_FLAG))
        self.assertEqual(levern_route.HANDOFF_FLAGS, (FLAG, AUDIT_FLAG))


class TwinTests(unittest.TestCase):
    def test_the_twins_are_gate_only_generated_and_boot_clean(self):
        for name in twins.NAMES:
            self.assertIs(PROFILES[name]['gate_only'], True)
            self.assertNotIn(contract.TRAFFIC_WAIVER, PROFILES[name])
            self.assertEqual(PROFILES[name]['env']['QWEN_C2_GATE_PROFILE'] if name != AUDIT else PROFILES[name]['env'].get('QWEN_C2_GATE_PROFILE'), '1')
            self.assertEqual(boot_problems(dict(PROFILES[name], name=name)), [], name)
        self.assertEqual(twins.generate(DATA), DATA, 'the file is what the generator makes')
        self.assertEqual(twins.main(['--check']), 0)

    def test_the_hand_off_settings_the_flags_give_are_the_ones_the_names_say(self):
        self.assertEqual(prefill_handoff.settings(PROFILES[CONTROL]['env']), (True, False, False))
        self.assertEqual(prefill_handoff.settings(PROFILES[ARM]['env']), (True, True, False))
        self.assertEqual(prefill_handoff.settings(PROFILES[AUDIT]['env']), (True, True, True))
        self.assertEqual(levern_route.merged_requested(PROFILES[ARM]['env']), True, 'the twins run the merged route: Lever N beside prefix reuse and sticky sessions')

    def test_the_arm_is_the_approved_profile_with_the_gate_marker_for_the_waiver_id_and_the_flag(self):
        ship, arm = PROFILES[SHIP], PROFILES[ARM]
        self.assertEqual(changed(ship['env'], arm['env']), {contract.OCTO_WAIVER_ENV: (ship['env'][contract.OCTO_WAIVER_ENV], None), 'QWEN_C2_GATE_PROFILE': (None, '1'), FLAG: (None, '1')})
        self.assertEqual(arm['engine'], ship['engine'], 'the memory plan, the pool and the trace region are the parent\'s')
        for key in ship:
            if key not in ('env', 'description', 'gate_only', contract.TRAFFIC_WAIVER):
                self.assertEqual(arm[key], ship[key], key)
        self.assertIn(SHIP, arm['description'])

    def test_the_control_differs_from_the_arm_by_the_flag_value_alone(self):
        control, arm = PROFILES[CONTROL], PROFILES[ARM]
        self.assertEqual(changed(control['env'], arm['env']), {FLAG: ('0', '1')})
        for key in control:
            if key not in ('env', 'description'):
                self.assertEqual(control[key], arm[key], key)
        self.assertEqual(changed(PROFILES[SHIP]['env'], control['env']), {contract.OCTO_WAIVER_ENV: (PROFILES[SHIP]['env'][contract.OCTO_WAIVER_ENV], None), 'QWEN_C2_GATE_PROFILE': (None, '1'), FLAG: (None, '0')})

    def test_the_audit_twin_is_the_ship4_audit_twin_plus_exactly_the_two_flags(self):
        parent, audit = PROFILES[AUDIT_PARENT], PROFILES[AUDIT]
        self.assertEqual({key: value for key, value in audit['env'].items() if parent['env'].get(key) != value}, {FLAG: '1', AUDIT_FLAG: '1'})
        self.assertEqual(sorted(set(parent['env']) - set(audit['env'])), [])
        for key in parent:
            if key not in ('env', 'description'):
                self.assertEqual(audit[key], parent[key], key)
        self.assertEqual(audit['env']['QWEN_FAST_LEVER_N'], '1')

    def test_the_twins_sit_right_after_the_last_ship4_block_profile(self):
        names = list(PROFILES)
        block = [index for index, name in enumerate(names) if name.startswith(ship4.NAMESPACE + 's4-')]
        self.assertEqual(names[max(block) + 1:max(block) + 4], list(twins.NAMES))

    def test_a_parent_that_cannot_be_twinned_is_refused(self):
        for mutate in (lambda data: data['profiles'][SHIP]['env'].__setitem__(FLAG, '1'),
                       lambda data: data['profiles'][AUDIT_PARENT]['env'].__setitem__(AUDIT_FLAG, '1'),
                       lambda data: data['profiles'][SHIP]['env'].__setitem__('QWEN_C2_GATE_PROFILE', '1'),
                       lambda data: data['profiles'][SHIP]['env'].pop('QWEN_FAST_LEVER_N'),
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

    def test_the_approved_traffic_profile_never_names_either_flag(self):
        self.assertNotIn(FLAG, PROFILES[SHIP]['env'])
        self.assertNotIn(AUDIT_FLAG, PROFILES[SHIP]['env'])
        self.assertEqual(PROFILES[SHIP][contract.TRAFFIC_WAIVER]['decision'][:8], 'APPROVED')
        for name, profile in PROFILES.items():
            if name not in twins.NAMES:
                self.assertNotIn(FLAG, profile['env'], name)
                self.assertNotIn(AUDIT_FLAG, profile['env'], name)


if __name__ == '__main__':
    unittest.main()
