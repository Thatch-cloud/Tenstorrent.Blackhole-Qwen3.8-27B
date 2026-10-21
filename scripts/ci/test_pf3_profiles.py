"""tp4/prefill-3: the six gate-only twins of the ship-5 candidate (make_pf3_profiles) and what the traced-tail flags may and may not touch.

  -pf3-s5-ctl     THE CONTROL OF THE TIMED PAIR: the ship-5 candidate with the gate marker in place of its owner waiver id, plus QWEN_FAST_PREFILL_TAIL_TRACE=0 (the eager tail, timed);
  -pf3-s5         THE TIMED ARM: the same with QWEN_FAST_PREFILL_TAIL_TRACE=1 and the arena at the module's default;
  -pf3-s5-a160    the arena ladder, 160 MB;
  -pf3-s5-a80     the arena ladder, 80 MB;
  -pf3-s5-audit   THE AUDITED ATTACH ARM: the ship-5 audited twin without W3 WC plus the flag, the arena, the buckets 128 and 1024 and the audit;
  -pf3-s5-audit-mid  the same for the buckets 256 and 512.

The approved traffic profiles and every profile that existed before this branch stay byte-identical (checked against the commit the branch was cut from); nothing here is hand-edited."""

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
import make_pf3_profiles as twins  # noqa: E402
import make_s5_profiles as s5  # noqa: E402
import prefill_tail_trace  # noqa: E402
import profile_twins  # noqa: E402
import serving_c2_contract as contract  # noqa: E402
from test_octo3_contract import boot_problems  # noqa: E402

PROFILES_PATH = HERE / 'qwen_c2_profiles.json'
DATA = json.loads(PROFILES_PATH.read_text(encoding='utf-8'))
PROFILES = DATA['profiles']
SHIP, AUDIT_PARENT, CONTROL, ARM, AUDIT, AUDIT_MID = twins.SHIP, twins.AUDIT_PARENT, twins.CONTROL, twins.ARM, twins.AUDIT, twins.AUDIT_MID
FLAG, AUDIT_FLAG, ARENA_FLAG = twins.FLAG, twins.AUDIT_FLAG, twins.ARENA_FLAG
LADDER = [twins.PREFIX + 's5-' + rung for rung, _mb in twins.LADDER]
PARENT = '758264ac'


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
        self.assertEqual(twins.twin_names(), (CONTROL, ARM, LADDER[0], LADDER[1], AUDIT, AUDIT_MID))
        for name in twins.twin_names():
            self.assertIn(name, profile_twins.twin_names())
            self.assertFalse(name.startswith(s5.NAMESPACE), 'outside the fusion generator\'s namespace: make_fusion_profiles leaves them alone')

    def test_the_flags_are_the_modules_and_the_routes(self):
        self.assertEqual((FLAG, AUDIT_FLAG, ARENA_FLAG), (prefill_tail_trace.FLAG, prefill_tail_trace.AUDIT_FLAG, prefill_tail_trace.ARENA_FLAG))
        self.assertEqual(levern_route.TAILTRACE_FLAGS, prefill_tail_trace.FLAGS)
        self.assertEqual(twins.ARENA_MB, str(prefill_tail_trace.DEFAULT_ARENA_MB))


class TwinTests(unittest.TestCase):
    def test_the_twins_are_gate_only_generated_and_boot_clean(self):
        for name in twins.NAMES:
            self.assertIs(PROFILES[name]['gate_only'], True)
            self.assertNotIn(contract.TRAFFIC_WAIVER, PROFILES[name])
            self.assertEqual(PROFILES[name]['env'].get('QWEN_C2_GATE_PROFILE'), '1')
            self.assertEqual(boot_problems(dict(PROFILES[name], name=name)), [], name)
        self.assertEqual(twins.generate(DATA), DATA, 'the file is what the generator makes')
        self.assertEqual(twins.main(['--check']), 0)

    def test_the_settings_the_flags_give_are_the_ones_the_names_say(self):
        self.assertEqual(prefill_tail_trace.settings(PROFILES[CONTROL]['env'])[:3], (True, False, False))
        arm = prefill_tail_trace.settings(PROFILES[ARM]['env'])
        self.assertEqual((arm.present, arm.traced, arm.audit, arm.buckets, arm.arena_mb), (True, True, False, prefill_tail_trace.DEFAULT_BUCKETS, prefill_tail_trace.DEFAULT_ARENA_MB))
        self.assertEqual([prefill_tail_trace.settings(PROFILES[name]['env']).arena_mb for name in LADDER], [160, 80])
        audit = prefill_tail_trace.settings(PROFILES[AUDIT]['env'])
        self.assertEqual((audit.traced, audit.audit, audit.arena_mb, audit.buckets), (True, True, prefill_tail_trace.DEFAULT_ARENA_MB, (128, 1024)))
        self.assertEqual(prefill_tail_trace.settings(PROFILES[AUDIT_MID]['env']).buckets, (256, 512))
        self.assertEqual(levern_route.merged_requested(PROFILES[ARM]['env']), True, 'the twins run the merged route: Lever N beside prefix reuse and sticky sessions')

    def test_the_arm_is_the_candidate_with_the_gate_marker_for_the_waiver_id_and_the_flags(self):
        ship, arm = PROFILES[SHIP], PROFILES[ARM]
        self.assertEqual(changed(ship['env'], arm['env']), {contract.OCTO_WAIVER_ENV: (ship['env'][contract.OCTO_WAIVER_ENV], None), 'QWEN_C2_GATE_PROFILE': (None, '1'), FLAG: (None, '1'),
                                                             ARENA_FLAG: (None, twins.ARENA_MB)})
        self.assertEqual(arm['engine'], ship['engine'], 'the memory plan, the pool and the trace region are the parent\'s')
        for key in ship:
            if key not in ('env', 'description', 'gate_only', contract.TRAFFIC_WAIVER):
                self.assertEqual(arm[key], ship[key], key)
        self.assertIn(SHIP, arm['description'])

    def test_the_control_differs_from_the_arm_by_the_flag_value_and_the_arena_alone(self):
        control, arm = PROFILES[CONTROL], PROFILES[ARM]
        self.assertEqual(changed(control['env'], arm['env']), {FLAG: ('0', '1'), ARENA_FLAG: (None, twins.ARENA_MB)})
        for key in control:
            if key not in ('env', 'description'):
                self.assertEqual(control[key], arm[key], key)

    def test_the_ladder_differs_from_the_arm_by_the_arena_alone(self):
        for name, megabytes in zip(LADDER, ('160', '80')):
            self.assertEqual(changed(PROFILES[ARM]['env'], PROFILES[name]['env']), {ARENA_FLAG: (twins.ARENA_MB, megabytes)})
            for key in PROFILES[ARM]:
                if key not in ('env', 'description'):
                    self.assertEqual(PROFILES[name][key], PROFILES[ARM][key], key)

    def test_the_audit_twins_are_the_s5_audit_twin_without_wc_plus_exactly_the_flags(self):
        parent = PROFILES[AUDIT_PARENT]
        for name, buckets in ((AUDIT, '128,1024'), (AUDIT_MID, '256,512')):
            audit = PROFILES[name]
            self.assertEqual({key: value for key, value in audit['env'].items() if parent['env'].get(key) != value},
                             {FLAG: '1', ARENA_FLAG: twins.ARENA_MB, AUDIT_FLAG: '1', twins.BUCKETS_FLAG: buckets})
            self.assertEqual(sorted(set(parent['env']) - set(audit['env'])), [])
            for key in parent:
                if key not in ('env', 'description'):
                    self.assertEqual(audit[key], parent[key], key)
            self.assertNotIn('QWEN_FAST_TP4_GDN_WC', audit['env'], 'the traced tail does not ride on W3 WC\'s first compile')
            self.assertEqual(audit['env']['QWEN_FAST_SOLO_PACKED'], '1')

    def test_the_twins_sit_right_after_the_ship5_audited_twins(self):
        names = list(PROFILES)
        import make_s6_profiles
        import make_sched2_profiles

        last = max(names.index(name) for name in tuple(s5.twin_names()) + tuple(make_s6_profiles.twin_names()) + tuple(make_sched2_profiles.twin_names()))      # (tp4/s7: the ship-6 audited twin and the sched-2 twins sit between)
        self.assertEqual(names[last + 1:last + 1 + len(twins.NAMES)], list(twins.NAMES))

    def test_a_parent_that_cannot_be_twinned_is_refused(self):
        for mutate in (lambda data: data['profiles'][SHIP]['env'].__setitem__(FLAG, '1'),
                       lambda data: data['profiles'][AUDIT_PARENT]['env'].__setitem__(AUDIT_FLAG, '1'),
                       lambda data: data['profiles'][SHIP]['env'].__setitem__('QWEN_C2_GATE_PROFILE', '1'),
                       lambda data: data['profiles'][SHIP]['env'].pop('QWEN_FAST_LEVER_N'),
                       lambda data: data['profiles'][SHIP]['env'].pop('QWEN_FAST_SOLO_PACKED'),
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
        # (tp4/s7 merged other branches' generated twins and the s7 candidate beside these)
        self.assertEqual(sorted(set(PROFILES) - set(before['profiles']) - self.later_branches() - (set(profile_twins.generated_names()) - set(twins.NAMES))), sorted(twins.NAMES))

    @staticmethod
    def later_branches():
        """Twins another package generated after this branch was cut (the K-split twins of tp4/prefill-3's other lever, and any later ones): named by their generators."""
        names = set()
        for module in ('make_pf3k_profiles',):
            try:
                names |= set(__import__(module).twin_names())
            except ImportError:
                pass
        return names

    def test_no_approved_profile_names_a_flag(self):
        self.assertEqual(PROFILES[SHIP][contract.TRAFFIC_WAIVER]['decision'][:8], 'APPROVED', 'the ship-5 candidate\'s decision is the owner\'s; this branch did not touch it')
        self.assertNotIn(FLAG, PROFILES[SHIP]['env'])
        import make_s7_profiles

        for name, profile in PROFILES.items():
            if name not in twins.NAMES and name not in make_s7_profiles.owned_names():      # (tp4/s7: the ship-7 candidate carries the traced tails under its waiver, its audited twins with their audit)
                for flag in prefill_tail_trace.FLAGS:
                    self.assertNotIn(flag, profile['env'], '%s names %s' % (name, flag))


if __name__ == '__main__':
    unittest.main()
