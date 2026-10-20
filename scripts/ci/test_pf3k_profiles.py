"""tp4/prefill-3: the three gate-only twins of the ship-5 candidate (make_pf3k_profiles), what the K-split flag may and may not touch, and the contract's refusal of the flag on a traffic profile.

  -pf3k-s5-ctl     THE CONTROL OF THE TIMED PAIR: the ship-5 candidate with the gate marker in place of its owner waiver id, plus QWEN_FAST_SDPA_PF_KSPLIT=0 (the oneq call, timed);
  -pf3k-s5         THE TIMED ARM: the same with QWEN_FAST_SDPA_PF_KSPLIT=1;
  -pf3k-s5-audit   THE AUDITED ATTACH ARM: the ship-5 audited twin plus the flag and its audit.

The approved traffic profile and every profile that existed before this branch stay byte-identical (checked against the commit the branch was cut from); nothing here is hand-edited.
The K-split is a numerics change: serving_c2_contract.ksplit_problems refuses the flag on a traffic profile unless the owner's traffic waiver names it."""

import copy
import json
from pathlib import Path
import subprocess
import sys
import unittest

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import make_pf3k_profiles as twins  # noqa: E402
import make_s5_profiles as s5  # noqa: E402
import profile_twins  # noqa: E402
import sdpa_pf_ksplit_tp as lever  # noqa: E402
import serving_c2_contract as contract  # noqa: E402
from test_octo3_contract import boot_problems  # noqa: E402

PROFILES_PATH = HERE / 'qwen_c2_profiles.json'
DATA = json.loads(PROFILES_PATH.read_text(encoding='utf-8'))
PROFILES = DATA['profiles']
SHIP, AUDIT_PARENT, CONTROL, ARM, AUDIT = twins.SHIP, twins.AUDIT_PARENT, twins.CONTROL, twins.ARM, twins.AUDIT
FLAG, AUDIT_FLAG = twins.FLAG, twins.AUDIT_FLAG
PARENT = '758264ac'
PROCESS = {'QWEN_FAST_SDPA_PF': '1', 'QWEN_FAST_TP': '4'}      # what the image environment sets for the process, beside the profile's env


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
        self.assertEqual((CONTROL, ARM, AUDIT), tuple(twins.PREFIX + suffix for suffix in ('s5-ctl', 's5', 's5-audit')))
        self.assertEqual(twins.twin_names(), (CONTROL, ARM, AUDIT))
        for name in twins.twin_names():
            self.assertIn(name, profile_twins.twin_names())
            self.assertFalse(name.startswith(s5.NAMESPACE), 'outside the fusion generator\'s namespace: make_fusion_profiles leaves them alone')

    def test_the_flags_are_the_modules_and_the_contracts(self):
        self.assertEqual((FLAG, AUDIT_FLAG), (lever.FLAG, lever.AUDIT_FLAG))
        self.assertEqual((FLAG, AUDIT_FLAG), (contract.KSPLIT_SWITCH, contract.KSPLIT_AUDIT))
        self.assertEqual(twins.ONEQ_FLAG, lever.ONEQ_FLAG)


class TwinTests(unittest.TestCase):
    def test_the_twins_are_gate_only_generated_and_boot_clean(self):
        for name in twins.NAMES:
            self.assertIs(PROFILES[name]['gate_only'], True)
            self.assertNotIn(contract.TRAFFIC_WAIVER, PROFILES[name])
            self.assertEqual(boot_problems(dict(PROFILES[name], name=name)), [], name)
            self.assertEqual(contract.ksplit_problems(dict(PROFILES[name], name=name)), [], name)
        self.assertEqual(twins.generate(DATA), DATA, 'the file is what the generator makes')
        self.assertEqual(twins.main(['--check']), 0)

    def test_the_settings_the_flags_give_are_the_ones_the_names_say(self):
        self.assertEqual(lever.settings(dict(PROFILES[CONTROL]['env'], **PROCESS)), (False, False))
        self.assertEqual(lever.settings(dict(PROFILES[ARM]['env'], **PROCESS)), (True, False))
        self.assertEqual(lever.settings(dict(PROFILES[AUDIT]['env'], **PROCESS)), (True, True))

    def test_the_arm_is_the_approved_profile_with_the_gate_marker_for_the_waiver_id_and_the_flag(self):
        ship, arm = PROFILES[SHIP], PROFILES[ARM]
        self.assertEqual(changed(ship['env'], arm['env']), {contract.OCTO_WAIVER_ENV: (ship['env'][contract.OCTO_WAIVER_ENV], None), 'QWEN_C2_GATE_PROFILE': (None, '1'), FLAG: (None, '1')})
        self.assertEqual(arm['engine'], ship['engine'], 'the memory plan, the pool and the trace region are the parent\'s')
        for key in ship:
            if key not in ('env', 'description', 'gate_only', contract.TRAFFIC_WAIVER):
                self.assertEqual(arm[key], ship[key], key)
        self.assertIn(SHIP, arm['description'])
        self.assertEqual(arm['env']['QWEN_FAST_SDPA_PF_ONEQ'], '1', 'the split extends the oneq call the candidate already carries')

    def test_the_control_differs_from_the_arm_by_the_flag_value_alone(self):
        control, arm = PROFILES[CONTROL], PROFILES[ARM]
        self.assertEqual(changed(control['env'], arm['env']), {FLAG: ('0', '1')})
        for key in control:
            if key not in ('env', 'description'):
                self.assertEqual(control[key], arm[key], key)
        self.assertEqual(changed(PROFILES[SHIP]['env'], control['env']), {contract.OCTO_WAIVER_ENV: (PROFILES[SHIP]['env'][contract.OCTO_WAIVER_ENV], None), 'QWEN_C2_GATE_PROFILE': (None, '1'), FLAG: (None, '0')})

    def test_the_audit_twin_is_the_ship5_audit_twin_plus_exactly_the_two_flags(self):
        parent, audit = PROFILES[AUDIT_PARENT], PROFILES[AUDIT]
        self.assertEqual({key: value for key, value in audit['env'].items() if parent['env'].get(key) != value}, {FLAG: '1', AUDIT_FLAG: '1'})
        self.assertEqual(sorted(set(parent['env']) - set(audit['env'])), [])
        for key in parent:
            if key not in ('env', 'description'):
                self.assertEqual(audit[key], parent[key], key)
        self.assertEqual(audit['env']['QWEN_FAST_SDPA_PF_ONEQ_AUDIT'], '1', 'the oneq audit stays: it judges every call the split does not take, bitwise')

    def test_the_twins_sit_right_after_the_last_ship5_twin(self):
        names = list(PROFILES)
        last = max(names.index(name) for name in s5.twin_names())
        self.assertEqual(names[last + 1:last + 4], list(twins.NAMES))

    def test_a_parent_that_cannot_be_twinned_is_refused(self):
        for mutate in (lambda data: data['profiles'][SHIP]['env'].__setitem__(FLAG, '1'),
                       lambda data: data['profiles'][AUDIT_PARENT]['env'].__setitem__(AUDIT_FLAG, '1'),
                       lambda data: data['profiles'][SHIP]['env'].__setitem__('QWEN_C2_GATE_PROFILE', '1'),
                       lambda data: data['profiles'][SHIP]['env'].pop('QWEN_FAST_SDPA_PF_ONEQ'),
                       lambda data: data['profiles'][AUDIT_PARENT].pop('gate_only'),
                       lambda data: data['profiles'].pop(SHIP)):
            data = copy.deepcopy(DATA)
            mutate(data)
            with self.assertRaises(ValueError):
                twins.generate(data)


class ContractTests(unittest.TestCase):
    def profile(self, **env):
        base = {'name': 'probe', 'env': {'QWEN_FAST_SDPA_PF_ONEQ': '1'}}
        base['env'].update(env)
        return base

    def test_a_profile_that_names_nothing_is_clean(self):
        self.assertEqual(contract.ksplit_problems({'name': 'x', 'env': {}}), [])
        self.assertEqual(contract.ksplit_problems({'name': 'x'}), [])

    def test_a_traffic_profile_carrying_the_flag_is_refused_for_want_of_the_numerics_waiver(self):
        problems = contract.ksplit_problems(self.profile(**{FLAG: '1'}))
        self.assertEqual(len(problems), 1)
        self.assertIn('numerics change', problems[0])
        self.assertIn('the numerics waiver is absent', problems[0])
        self.assertEqual(contract.ksplit_problems(self.profile(**{FLAG: '0'})), [], 'the control value is no lever')

    def test_the_flag_is_admitted_on_a_traffic_profile_only_under_a_waiver_that_names_it(self):
        waiver = {'id': 'ksplit-numerics-test', 'levers': {FLAG: '1'}, 'waives': 'the attention output of tail passes differs from the oneq call in the last bits', 'evidence': ['card M statistics'],
                  'decision': 'PENDING a test'}
        profile = dict(self.profile(**{FLAG: '1'}), **{contract.TRAFFIC_WAIVER: waiver})
        self.assertEqual(contract.traffic_waiver_problems(profile), [])
        self.assertEqual(contract.ksplit_problems(profile), [])
        self.assertEqual(contract.waived_levers(profile), {FLAG: '1'})
        stale = dict(self.profile(), **{contract.TRAFFIC_WAIVER: waiver})
        self.assertTrue(any('names %s=1 and the profile env has None' % FLAG in problem for problem in contract.traffic_waiver_problems(stale)))
        wrong_value = dict(profile, **{contract.TRAFFIC_WAIVER: dict(waiver, levers={FLAG: '2'})})
        self.assertTrue(contract.traffic_waiver_problems(wrong_value))

    def test_the_audit_is_a_gate_instrument_and_the_switch_is_strict(self):
        profile = self.profile(**{FLAG: '1', AUDIT_FLAG: '1'})
        self.assertTrue(any('gate instrument' in problem for problem in contract.ksplit_problems(profile)))
        self.assertEqual(contract.ksplit_problems(dict(profile, gate_only=True)), [])
        self.assertTrue(any('only 0 or 1' in problem for problem in contract.ksplit_problems(dict(self.profile(**{FLAG: 'yes'}), gate_only=True))))
        self.assertTrue(any('without %s=1' % FLAG in problem for problem in contract.ksplit_problems(dict(self.profile(**{AUDIT_FLAG: '1'}), gate_only=True))))
        self.assertTrue(any('needs QWEN_FAST_SDPA_PF_ONEQ=1' in problem for problem in contract.ksplit_problems({'name': 'x', 'gate_only': True, 'env': {FLAG: '1'}})))

    def test_boot_refuses_a_traffic_profile_with_the_flag_and_no_waiver(self):
        source = (HERE / 'serving_c2_contract.py').read_text(encoding='utf-8')
        self.assertIn("problems = ksplit_problems(profile)", source)
        self.assertIn("misuses the K-split switches", source)

    def test_no_traffic_profile_in_the_file_carries_the_flag_or_its_audit(self):
        for name, profile in PROFILES.items():
            if profile.get('gate_only') is not True:
                self.assertEqual(contract.ksplit_problems(dict(profile, name=name)), [], name)
                self.assertNotIn(FLAG, profile['env'], name)
                self.assertNotIn(AUDIT_FLAG, profile['env'], name)

    def test_the_waivable_table_names_the_flag_at_one_value(self):
        self.assertEqual(contract.NEXT_WAIVABLE_LEVERS[FLAG], '1')
        self.assertEqual(contract.KSPLIT_WAIVABLE_LEVERS, {FLAG: '1'})


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
