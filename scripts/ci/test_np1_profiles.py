"""tp4/np1-fastlane: the two gate-only twins of the approved ship-4 profile (make_np1_profiles) and what the flag may and may not touch.

  -fx-s4-np1        THE TIMED ARM: the approved ship-4 profile with the gate marker in place of its owner waiver id, plus QWEN_FAST_SOLO_PACKED=1;
  -fx-s4-np1-audit  THE AUDITED ATTACH ARM: the ship-4 audited twin plus QWEN_FAST_SOLO_PACKED=1.

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

import idle_scratch  # noqa: E402
import make_np1_profiles as twins  # noqa: E402
import make_ship4_profiles as ship4  # noqa: E402
import profile_twins  # noqa: E402
import serving_c2_contract as contract  # noqa: E402
import serving_octo as octo  # noqa: E402
from test_octo3_contract import boot_problems  # noqa: E402

PROFILES_PATH = HERE / 'qwen_c2_profiles.json'
DATA = json.loads(PROFILES_PATH.read_text(encoding='utf-8'))
PROFILES = DATA['profiles']
SHIP, AUDIT_PARENT, ARM, AUDIT = twins.SHIP, twins.AUDIT_PARENT, twins.ARM, twins.AUDIT
FLAG = idle_scratch.SOLO_PACKED_FLAG
PARENT = '27af6902'
# what the image's own ENV carries that the flag's admission reads (docker/qwen-c2-serving.Dockerfile)
IMAGE_ENV = {'QWEN_FAST_PACKED_STEP': '1', 'QWEN_FAST_PADDED_BLOCK': '1', 'QWEN_FAST_SINGLE_GATEUP': '1', 'QWEN_FAST_FOUR_AS_TWO': '0'}


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
        self.assertEqual((ARM, AUDIT), (ship4.NAMESPACE + 's4-np1', ship4.NAMESPACE + 's4-np1-audit'))
        self.assertEqual(twins.twin_names(), (ARM, AUDIT))
        for name in twins.twin_names():
            self.assertIn(name, profile_twins.twin_names())

    def test_the_flag_is_the_modules_and_the_contracts(self):
        self.assertEqual((twins.FLAG, FLAG), (octo.SOLO_PACKED_FLAG, octo.SOLO_PACKED_FLAG))
        self.assertIn(FLAG, contract.OCTO_GATE_ONLY)


class TwinTests(unittest.TestCase):
    def test_the_twins_are_gate_only_generated_and_boot_clean(self):
        for name in (ARM, AUDIT):
            self.assertIs(PROFILES[name]['gate_only'], True)
            self.assertNotIn(contract.TRAFFIC_WAIVER, PROFILES[name])
            self.assertEqual(PROFILES[name]['env'][FLAG], '1')
            self.assertEqual(PROFILES[name]['env']['QWEN_C2_GATE_PROFILE'], '1')
            self.assertEqual(boot_problems(dict(PROFILES[name], name=name)), [], name)
        self.assertEqual(twins.generate(DATA), DATA, 'the file is what the generator makes')
        self.assertEqual(twins.main(['--check']), 0)

    def test_the_arm_is_the_approved_profile_with_the_gate_marker_for_the_waiver_id_and_the_flag(self):
        ship, arm = PROFILES[SHIP], PROFILES[ARM]
        changed = {key: (ship['env'].get(key), arm['env'].get(key)) for key in set(ship['env']) | set(arm['env']) if ship['env'].get(key) != arm['env'].get(key)}
        self.assertEqual(changed, {contract.OCTO_WAIVER_ENV: (ship['env'][contract.OCTO_WAIVER_ENV], None), 'QWEN_C2_GATE_PROFILE': (None, '1'), FLAG: (None, '1')})
        self.assertEqual(arm['engine'], ship['engine'], 'the memory plan, the pool and the trace region are the parent\'s: the scratch page is one block the model\'s allocation adds')
        for key in ship:
            if key not in ('env', 'description', 'gate_only', contract.TRAFFIC_WAIVER):
                self.assertEqual(arm[key], ship[key], key)
        self.assertEqual((arm['engine']['num-gpu-blocks-override'], arm['env']['QWEN36_MAX_TOKENS_ALL_USERS']), (16500, '1055488'))
        self.assertIn(SHIP, arm['description'])
        self.assertIn('557,056', arm['description'])

    def test_the_audit_twin_is_the_ship4_audit_twin_plus_exactly_the_flag(self):
        parent, audit = PROFILES[AUDIT_PARENT], PROFILES[AUDIT]
        self.assertEqual({key: value for key, value in audit['env'].items() if parent['env'].get(key) != value}, {FLAG: '1'})
        self.assertEqual(sorted(set(parent['env']) - set(audit['env'])), [])
        for key in parent:
            if key not in ('env', 'description'):
                self.assertEqual(audit[key], parent[key], key)
        for name in ('QWEN_FAST_KV_PAGE_WRITER_AUDIT', 'QWEN_FAST_VERIFY_T2_AUDIT', 'QWEN_FAST_PRESTAGE_DIFF_AUDIT', 'QWEN_FAST_TP4_SDPA_AUDIT', 'QWEN_FAST_OCTO_ATTN_BUNDLE_AUDIT'):
            self.assertEqual(audit['env'][name], '1', 'the audits that read the idle segments\' K/V stay on')

    def test_the_twins_sit_right_after_the_ship4_twins(self):
        names = list(PROFILES)
        last = max(names.index(name) for name in ship4.twin_names())
        self.assertEqual(names[last + 1:last + 3], list(twins.NAMES))

    def test_a_parent_that_cannot_be_twinned_is_refused(self):
        for mutate in (lambda data: data['profiles'][SHIP]['env'].__setitem__(FLAG, '1'),
                       lambda data: data['profiles'][AUDIT_PARENT]['env'].__setitem__(FLAG, '1'),
                       lambda data: data['profiles'][SHIP]['env'].__setitem__('QWEN_C2_GATE_PROFILE', '1'),
                       lambda data: data['profiles'][SHIP]['env'].__setitem__('QWEN_FAST_PADDED_BLOCK', '0'),
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
        self.assertEqual(list(PROFILES)[:0], [])

    def test_the_approved_traffic_profile_never_names_the_flag_and_a_traffic_profile_that_did_would_be_refused(self):
        self.assertNotIn(FLAG, PROFILES[SHIP]['env'])
        self.assertEqual(contract.octo_problems(dict(PROFILES[SHIP], name=SHIP)), [])
        stolen = copy.deepcopy(PROFILES[SHIP])
        stolen['env'][FLAG] = '1'
        problems = contract.octo_problems(dict(stolen, name='stolen'))
        self.assertTrue(any(FLAG in problem and 'gate instrument' in problem for problem in problems), problems)
        self.assertEqual(PROFILES[SHIP][contract.TRAFFIC_WAIVER]['decision'][:8], 'APPROVED')


class AdmissionTests(unittest.TestCase):
    """The flag's admission over the twins' own environment: the image's ENV, the profile's env and the gate switch, with the scratch page recorded as the model's allocation would."""

    M3_TWO = (True, 'users=8 FOUR_AS_TWO=0 PACKED_STEP=1 M3_BLOCKS=2')

    def setUp(self):
        idle_scratch.forget()
        self.addCleanup(idle_scratch.forget)

    def environment(self, name):
        environ = dict(IMAGE_ENV)
        environ.update({key: str(value) for key, value in PROFILES[name]['env'].items()})
        environ['QWEN_C2_GATE'] = '1'
        return environ

    def test_the_arm_and_the_audit_arm_are_admitted_with_the_scratch_page_and_refused_without_it(self):
        for name in (ARM, AUDIT):
            environ = self.environment(name)
            with self.assertRaisesRegex(ValueError, 'KV scratch page'):
                octo.solo_packed_admission(self.M3_TWO, environ, log=lambda *args: None)
            idle_scratch.record(16500, 1)
            self.assertEqual(octo.solo_packed_admission(self.M3_TWO, environ, log=lambda *args: None), dict(min_users=1, scratch_pages=(16500,)), name)
            idle_scratch.forget()

    def test_the_approved_profile_asks_for_nothing_and_the_hook_wants_nothing_under_it(self):
        environ = self.environment(SHIP)
        self.assertIsNone(octo.solo_packed_admission(self.M3_TWO, environ))
        self.assertEqual(idle_scratch.wanted(environ), 0)
        self.assertEqual(idle_scratch.wanted(self.environment(ARM)), 1)

    def test_the_arm_keeps_the_octo_shapes_admission_and_the_audit_arm_does_too(self):
        for name in (ARM, AUDIT):
            environ = self.environment(name)
            self.assertEqual(environ[octo.OCTO_FLAG], 'live')
            self.assertEqual(environ[octo.MIN_LIVE_FLAG], '6', 'the octo block pads from six live seats: its capacity of two idle segments is untouched')
            self.assertTrue(octo.admitted_run(environ), name)
            self.assertFalse(octo.waiver_run(environ), 'the gate run admits the shape, not a waiver id')


if __name__ == '__main__':
    unittest.main()
