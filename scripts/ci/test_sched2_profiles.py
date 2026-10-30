"""tp4/sched-2: the five gate-only twins of the ship-5 candidate (make_sched2_profiles) and what the levers may and may not touch.

  -sch-s5-seed    THE PIPELINING ARM  the candidate with the gate marker in place of its owner waiver id, plus QWEN_FAST_ADMISSION_SEED_FIRST=1;
  -sch-s5-hold    THE HOLD ARM        ... plus QWEN_FAST_ADMISSION_HOLD_REUSE=1;
  -sch-s5-ledger  THE LEDGER ARM      ... plus QWEN_FAST_ADMISSION_LEDGER_TRIM=1;
  -sch-s5-all     the three together;
  -sch-s5-audit   THE AUDITED ATTACH ARM: the ship-5 audited twin plus seed first and hold reuse.

The ship-5 candidate and every profile that existed before this branch stay byte-identical (checked against the commit the branch was cut from); nothing here is hand-edited."""

import copy
import json
from pathlib import Path
import subprocess
import sys
import unittest

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import admission_pipeline as pipeline  # noqa: E402
import make_s5_profiles as s5  # noqa: E402
import make_sched2_profiles as twins  # noqa: E402
import profile_twins  # noqa: E402
import serving_c2_contract as contract  # noqa: E402
from test_octo3_contract import boot_problems  # noqa: E402

PROFILES_PATH = HERE / 'qwen_c2_profiles.json'
DATA = json.loads(PROFILES_PATH.read_text(encoding='utf-8'))
PROFILES = DATA['profiles']
SHIP, AUDIT_PARENT = twins.SHIP, twins.AUDIT_PARENT
PARENT = '758264ac'                              # origin/tp4/next-1 when the branch was cut: the ship-5 tree
FLAGS = {
    'seed': {'QWEN_FAST_ADMISSION_SEED_FIRST': '1'},
    'hold': {'QWEN_FAST_ADMISSION_HOLD_REUSE': '1'},
    'ledger': {'QWEN_FAST_ADMISSION_LEDGER_TRIM': '1'},
}
FLAGS['all'] = {name: value for arm in ('seed', 'hold', 'ledger') for name, value in FLAGS[arm].items()}
FLAGS['audit'] = dict(FLAGS['seed'], **FLAGS['hold'])
ALL_NAMES = sorted(FLAGS['all'])


def parent_profiles():
    result = subprocess.run(['git', 'show', '%s:scripts/ci/qwen_c2_profiles.json' % PARENT], capture_output=True, cwd=str(HERE), timeout=120)
    if result.returncode != 0:
        return None
    return json.loads(result.stdout.decode('utf-8'))


class NamesTests(unittest.TestCase):
    def test_the_names_fit_a_baked_default_and_are_registered_as_twins(self):
        self.assertEqual(twins.twin_names(), tuple(twins.NAMESPACE + suffix for suffix in ('seed', 'hold', 'ledger', 'all', 'audit')))
        for name in twins.twin_names():
            self.assertLessEqual(len(name), 64, name)
            self.assertIn(name, profile_twins.twin_names())
            self.assertIn(name, PROFILES)

    def test_the_levers_are_the_flags_the_module_and_the_contract_know(self):
        self.assertEqual(sorted(twins.FLAGS), ALL_NAMES)
        self.assertEqual(ALL_NAMES, sorted(pipeline.SWITCHES))
        self.assertEqual(sorted(contract.SCHED2_SWITCHES), ALL_NAMES)
        self.assertEqual(twins.SEED, {pipeline.SEED_FLAG: '1'})
        self.assertEqual(twins.HOLD, {pipeline.HOLD_FLAG: '1'})
        self.assertEqual(twins.LEDGER, {pipeline.LEDGER_FLAG: '1'})

    def test_the_specs_name_a_parent_and_a_description_each(self):
        for name, parent, description in twins.specs():
            self.assertIn(name, PROFILES)
            self.assertIn(parent, PROFILES)
            self.assertIn('tp4/sched-2', description)
            self.assertEqual(PROFILES[name]['description'], description)


class TwinTests(unittest.TestCase):
    def test_the_twins_are_gate_only_generated_and_boot_clean(self):
        for name in twins.twin_names():
            profile = dict(PROFILES[name], name=name)
            self.assertIs(profile['gate_only'], True, name)
            self.assertNotIn(contract.TRAFFIC_WAIVER, profile)
            self.assertEqual(profile['env']['QWEN_C2_GATE_PROFILE'], '1')
            self.assertEqual(boot_problems(profile), [], name)
            self.assertEqual(contract.sched_problems(profile), [], name)
            self.assertEqual(contract.sched2_problems(profile), [], name)
            self.assertEqual(contract.levern_problems(profile), [], name)
            self.assertEqual(contract.prefix_policy_problems(profile), [], name)
            self.assertEqual(contract.exact_resume_problems(profile), [], name)
            self.assertEqual(pipeline.problems({key: str(value) for key, value in profile['env'].items()}), [], name)
        self.assertEqual(twins.generate(DATA), DATA, 'the file is what the generator makes')
        self.assertEqual(twins.main(['--check']), 0)

    def test_each_timed_arm_is_the_candidate_with_the_gate_marker_for_the_waiver_id_and_exactly_its_flags(self):
        ship = PROFILES[SHIP]
        for suffix in ('seed', 'hold', 'ledger', 'all'):
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

    def test_the_audited_arm_is_the_ship_5_audited_twin_plus_seed_first_and_hold_reuse_and_not_the_ledger_trim(self):
        parent, audit = PROFILES[AUDIT_PARENT], PROFILES[twins.NAMESPACE + 'audit']
        self.assertEqual({key: value for key, value in audit['env'].items() if parent['env'].get(key) != value}, FLAGS['audit'])
        self.assertEqual(sorted(set(parent['env']) - set(audit['env'])), [])
        for key in parent:
            if key not in ('env', 'description'):
                self.assertEqual(audit[key], parent[key], key)
        self.assertNotIn(pipeline.LEDGER_FLAG, audit['env'], 'the audit judge reads the ledger\'s points')

    def test_the_twins_sit_right_after_the_ship_5_twins(self):
        names = list(PROFILES)
        import make_s6_profiles

        last = max(names.index(name) for name in tuple(s5.twin_names()) + tuple(make_s6_profiles.twin_names()))      # (tp4/s7: the ship-6 audited twin sits between)
        self.assertEqual(names[last + 1:last + 1 + len(twins.NAMES)], list(twins.NAMES))

    def test_a_parent_that_cannot_be_twinned_is_refused(self):
        for mutate in (lambda data: data['profiles'][SHIP]['env'].__setitem__('QWEN_FAST_ADMISSION_SEED_FIRST', '1'),
                       lambda data: data['profiles'][AUDIT_PARENT]['env'].__setitem__('QWEN_FAST_ADMISSION_HOLD_REUSE', '1'),
                       lambda data: data['profiles'][SHIP]['env'].__setitem__('QWEN_C2_GATE_PROFILE', '1'),
                       lambda data: data['profiles'][SHIP]['env'].pop('QWEN_FAST_PARKED_ENGINES'),
                       lambda data: data['profiles'][SHIP]['env'].pop('QWEN_FAST_ADMISSION_LEAN'),
                       lambda data: data['profiles'][SHIP]['env'].pop('QWEN_FAST_EXTENT_REPLAY'),
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
        # (tp4/s7 merged other branches' generated twins and the s7 candidate beside these: they are not this branch's to protect or to count)
        self.assertEqual(sorted(set(PROFILES) - set(before['profiles']) - (set(profile_twins.generated_names()) - set(twins.NAMES))), sorted(twins.NAMES))

    def test_the_ship_5_candidate_never_names_a_lever_and_a_traffic_profile_that_did_would_be_refused(self):
        for name in ALL_NAMES:
            self.assertNotIn(name, PROFILES[SHIP]['env'], name)
            self.assertNotIn(name, PROFILES[AUDIT_PARENT]['env'], name)
        for name, value in sorted(FLAGS['all'].items()):
            stolen = copy.deepcopy(PROFILES[SHIP])
            stolen['env'][name] = value
            problems = contract.sched2_problems(dict(stolen, name='stolen'))
            self.assertTrue([problem for problem in problems if name in problem], (name, problems))

    def test_no_other_profile_names_a_lever(self):
        import make_nvme_profiles
        import make_s7_profiles

        for name, profile in PROFILES.items():
            if name not in twins.NAMES and name not in make_s7_profiles.owned_names() + make_nvme_profiles.twin_names():      # (tp4/s7c: the ship-7 candidate carries the ledger trim under its waiver, its audited twin none; test_s7_profiles holds that neither carries seed first)
                self.assertFalse([flag for flag in pipeline.NAMES if flag in profile['env']], name)

    def test_every_twin_names_exactly_its_levers_among_the_sched_2_names(self):
        for suffix, flags in FLAGS.items():
            env = PROFILES[twins.NAMESPACE + suffix]['env']
            self.assertEqual({flag: env[flag] for flag in pipeline.NAMES if flag in env}, flags, suffix)


if __name__ == '__main__':
    unittest.main()
