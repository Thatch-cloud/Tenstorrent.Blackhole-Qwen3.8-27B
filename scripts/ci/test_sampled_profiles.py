"""tp4/sampled-verify: the two gate-only profiles of sampled verification (make_sampled_profiles; docs/tp4-sampled-verify.md) and what they may and may not touch.

  -smp-s7        THE SAMPLED ARM     the ship-7 candidate (a traffic profile under its owner waiver) with the gate marker in place of the waiver id, gate_only, and QWEN_FAST_SAMPLED=1;
  -smp-s7-audit  THE AUDITED ARM     the ship-7 audited twin plus QWEN_FAST_SAMPLED=1 and QWEN_FAST_SAMPLED_AUDIT=1.

Neither replaces a shipped profile: the ship-7 candidate and every profile that existed before the branch stay byte-identical, and no other profile names the switch (test_sampled_contract)."""

import copy
import json
from pathlib import Path
import subprocess
import sys
import unittest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import make_s7_profiles as s7  # noqa: E402
import make_sampled_profiles as twins  # noqa: E402
import profile_twins  # noqa: E402
import serving_c2_contract as contract  # noqa: E402

PROFILES_PATH = HERE / 'qwen_c2_profiles.json'
DATA = json.loads(PROFILES_PATH.read_text(encoding='utf-8'))
PROFILES = DATA['profiles']
SHIP, AUDIT_PARENT = twins.SHIP, twins.AUDIT_PARENT
PARENT_COMMIT = 'f12a614e'                       # tp4/s7c: the tree this branch was cut from


def boot_problems(profile):
    from test_octo3_contract import boot_problems as every_check

    return (every_check(profile) + contract.prefix_policy_problems(profile) + contract.exact_resume_problems(profile) + contract.sched_problems(profile) + contract.sched2_problems(profile)
            + contract.ksplit_problems(profile) + contract.sampled_problems(profile))


def parent_profiles():
    result = subprocess.run(['git', 'show', '%s:scripts/ci/qwen_c2_profiles.json' % PARENT_COMMIT], capture_output=True, cwd=str(HERE), timeout=120)
    if result.returncode != 0:
        return None
    return json.loads(result.stdout.decode('utf-8'))['profiles']


class NamesTests(unittest.TestCase):
    def test_the_names_fit_a_baked_default_and_are_registered_as_twins(self):
        self.assertEqual(twins.twin_names(), (twins.BASE + 'smp-s7', twins.BASE + 'smp-s7-audit'))
        for name in twins.twin_names():
            self.assertLessEqual(len(name), 64, name)
            self.assertIn(name, profile_twins.twin_names())
            self.assertIn(name, PROFILES)

    def test_the_parents_are_the_ship_7_candidate_and_its_audited_twin(self):
        self.assertEqual(SHIP, s7.SHIP)
        self.assertEqual(AUDIT_PARENT, s7.AUDIT)
        self.assertEqual(SHIP, 'c2-packed-tp4-8x262k-ship-prefix-levern-w2-er-fx-traffic-s7')
        self.assertEqual(AUDIT_PARENT, 'c2-packed-tp4-8x262k-ship-prefix-levern-w2-er-fx-s7-audit')

    def test_the_names_are_outside_the_fusion_namespace_which_its_generator_owns(self):
        import make_fusion_profiles

        for name in twins.twin_names():
            self.assertFalse(name.startswith(make_fusion_profiles.NAMESPACE), name)

    def test_the_specs_name_a_parent_and_a_description_each(self):
        for name, parent, description in twins.specs():
            self.assertIn(parent, PROFILES)
            self.assertEqual(PROFILES[name]['description'], description)
            self.assertIn('tp4/sampled-verify', description)
            self.assertIn('GATE ONLY', description)
            self.assertIn(parent, description)

    def test_the_switches_are_the_contracts(self):
        self.assertEqual((twins.SWITCH, twins.AUDIT_SWITCH), (contract.SAMPLED_SWITCH, contract.SAMPLED_AUDIT_SWITCH))
        self.assertEqual(twins.TIMED_FLAGS, {contract.SAMPLED_SWITCH: '1'})
        self.assertEqual(twins.AUDIT_FLAGS, {contract.SAMPLED_SWITCH: '1', contract.SAMPLED_AUDIT_SWITCH: '1'})


class ProfileTests(unittest.TestCase):
    def test_both_are_gate_only_generated_and_boot_clean(self):
        for name in twins.twin_names():
            profile = dict(PROFILES[name], name=name)
            self.assertIs(profile['gate_only'], True, name)
            self.assertNotIn(contract.TRAFFIC_WAIVER, profile)
            self.assertEqual(profile['env']['QWEN_C2_GATE_PROFILE'], '1')
            self.assertEqual(profile['env'][contract.SAMPLED_SWITCH], '1')
            self.assertTrue(contract.sampled_on(profile))
            self.assertEqual(boot_problems(profile), [], name)
        self.assertEqual(twins.generate(DATA), DATA, 'the file is what the generator makes')
        self.assertEqual(twins.main(['--check']), 0)

    def test_the_sampled_arm_is_the_candidate_with_the_gate_marker_for_the_waiver_id_and_the_switch(self):
        ship, arm = PROFILES[SHIP], PROFILES[twins.TIMED]
        changed = {key: (ship['env'].get(key), arm['env'].get(key)) for key in set(ship['env']) | set(arm['env']) if ship['env'].get(key) != arm['env'].get(key)}
        self.assertEqual(changed, {contract.OCTO_WAIVER_ENV: (ship['env'][contract.OCTO_WAIVER_ENV], None), 'QWEN_C2_GATE_PROFILE': (None, '1'), contract.SAMPLED_SWITCH: (None, '1')})
        self.assertEqual(arm['engine'], ship['engine'], 'the memory plan, the pool and the trace region are the parent\'s')
        for key in ship:
            if key not in ('env', 'description', 'gate_only', contract.TRAFFIC_WAIVER):
                self.assertEqual(arm[key], ship[key], key)
        self.assertNotIn(contract.SAMPLED_AUDIT_SWITCH, arm['env'])
        self.assertIn(SHIP, arm['description'])

    def test_the_audited_arm_is_the_ship_7_audited_twin_plus_the_two_switches(self):
        parent, audit = PROFILES[AUDIT_PARENT], PROFILES[twins.AUDIT]
        self.assertEqual({key: value for key, value in audit['env'].items() if parent['env'].get(key) != value}, twins.AUDIT_FLAGS)
        self.assertEqual(sorted(set(parent['env']) - set(audit['env'])), [])
        for key in parent:
            if key not in ('env', 'description'):
                self.assertEqual(audit[key], parent[key], key)
        self.assertEqual(audit['env']['QWEN_FAST_AUDIT_FIRST'], parent['env']['QWEN_FAST_AUDIT_FIRST'], 'the audits keep their cadence')

    def test_the_profiles_sit_right_after_the_ship_7_audited_twin(self):
        names = list(PROFILES)
        start = names.index(AUDIT_PARENT)
        self.assertEqual(names[start + 1:start + 3], list(twins.NAMES))

    def test_the_sampled_profiles_serve_the_four_card_c2_any_contract_the_gate_needs(self):
        for name in twins.twin_names():
            profile = PROFILES[name]
            self.assertEqual(profile['mesh_device'], 'P150x4')
            self.assertEqual(profile['env']['QWEN_FAST_TP'], '4')
            self.assertEqual(profile['env']['QWEN_FAST_ANY_REQUEST'], '1')
            self.assertNotIn('generation-config', profile['engine'])
            self.assertNotIn('override-generation-config', profile['engine'])
            self.assertEqual(contract.sampled_launch_problems(['--model', 'm']), [])

    def test_a_parent_that_cannot_be_twinned_is_refused(self):
        for mutate in (lambda data: data['profiles'][SHIP]['env'].__setitem__(contract.SAMPLED_SWITCH, '1'),
                       lambda data: data['profiles'][AUDIT_PARENT]['env'].__setitem__(contract.SAMPLED_AUDIT_SWITCH, '1'),
                       lambda data: data['profiles'][SHIP]['env'].__setitem__('QWEN_C2_GATE_PROFILE', '1'),
                       lambda data: data['profiles'][SHIP]['env'].pop('QWEN_FAST_ANY_REQUEST'),
                       lambda data: data['profiles'][AUDIT_PARENT]['env'].pop('QWEN_FAST_TP'),
                       lambda data: data['profiles'][AUDIT_PARENT].pop('gate_only'),
                       lambda data: data['profiles'][SHIP].__setitem__('gate_only', True),
                       lambda data: data['profiles'][SHIP].pop(contract.TRAFFIC_WAIVER),
                       lambda data: data['profiles'].pop(SHIP),
                       lambda data: data['profiles'].pop(AUDIT_PARENT)):
            data = copy.deepcopy(DATA)
            mutate(data)
            with self.assertRaises(ValueError):
                twins.generate(data)


class ProtectedTests(unittest.TestCase):
    def test_every_profile_that_existed_before_the_branch_is_byte_identical_and_the_default_is_untouched(self):
        before = parent_profiles()
        if before is None:
            self.skipTest('no git history for %s' % PARENT_COMMIT)
        for name, body in before.items():
            self.assertEqual(PROFILES[name], body, name)
        added = [name for name in PROFILES if name not in before]
        self.assertEqual(sorted(added), sorted(twins.twin_names()))
        self.assertEqual(DATA['default'], 'c2-packed-tp4')
        self.assertEqual(list(PROFILES)[:list(PROFILES).index(AUDIT_PARENT) + 1], list(before)[:list(before).index(AUDIT_PARENT) + 1])

    def test_the_ship_7_candidate_still_carries_its_waiver_and_not_the_switch(self):
        ship = PROFILES[SHIP]
        self.assertIn(contract.TRAFFIC_WAIVER, ship)
        self.assertNotIn('gate_only', ship)
        self.assertNotIn(contract.SAMPLED_SWITCH, ship['env'])
        self.assertNotIn(contract.SAMPLED_SWITCH, PROFILES[AUDIT_PARENT]['env'])

    def test_the_other_generators_still_reproduce_the_file(self):
        import make_s6_profiles
        import make_sched2_profiles

        for generator in (s7, make_s6_profiles, make_sched2_profiles):
            self.assertEqual(generator.main(['--check']), 0, generator.__name__)


if __name__ == '__main__':
    unittest.main()
