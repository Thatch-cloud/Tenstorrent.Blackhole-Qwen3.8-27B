"""tp4/next-1: the ship-5 candidate (a traffic profile generated from fusion-wp/S5.json by make_fusion_profiles.py), its two gate-only audited twins (make_s5_profiles) and the contract's admission of what it carries.

  -fx-traffic-s5       the approved ship-4 profile + the lone-user padded round (NP1) + exact resume P0 (the undrop and the drafter-window carry) + the sliced host-tier restore + the decode verify
                       sub-switch of the 13x10 grids + the owner-approved host RAM budget (tier 48 GiB, checkpoint store 16 GiB), under ONE owner traffic waiver (decision PENDING or APPROVED: nothing here reads it);
  -fx-s5-audit         its audited twin: the ship-4 audited twin + every difference of the candidate + the carry's audit + W3 WC with its audit, less F1's audit;
  -fx-s5-audit-nowc    the fallback of the audited attach: the same without W3 WC.

The approved ship-4 profile, its waiver and every profile that existed before this branch stay byte-identical (checked against the commit the branch was cut from); nothing here is hand-edited. The tests
are decision-agnostic: the owner's approval at ship time changes one sentence of the manifest and these tests still pass (ApprovalTests)."""

import copy
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import gdn_wc  # noqa: E402
import make_fusion_profiles as fusion  # noqa: E402
import make_ship4_profiles as ship4  # noqa: E402
import make_s5_profiles as twins  # noqa: E402
import profile_twins  # noqa: E402
import serving_c2_contract as contract  # noqa: E402
import serving_octo as octo  # noqa: E402
import w3_wc_smoke  # noqa: E402

PROFILES_PATH = HERE / 'qwen_c2_profiles.json'
DATA = json.loads(PROFILES_PATH.read_text(encoding='utf-8'))
PROFILES = DATA['profiles']
NS = fusion.NAMESPACE
SHIP4, SHIP, AUDIT, NOWC = ship4.SHIP, twins.SHIP, twins.AUDIT, twins.AUDIT_NOWC
PLAN = fusion.normalise(fusion.read_manifests())
SPEC = next(item for item in PLAN['ship'] if item['name'] == 'traffic-s5')
PARENT_COMMIT = '27af6902'                      # tp4/swift: the production tree this branch was cut from
# what the candidate sets that ship 4 does not (or sets differently), the waiver id aside
CHANGES = {'QWEN_FAST_GRID13': '1', 'QWEN_FAST_SOLO_PACKED': '1', 'QWEN_PREFIX_STICKY_UNDROP': '1', 'QWEN_PREFIX_WINDOW_CARRY': '1', 'QWEN_PREFIX_TIER_RESTORE_SLICE': '512',
           'QWEN_PREFIX_TIER_RESTORE_SLICE_MS': '200', 'QWEN_PREFIX_HOST_TIER_GIB': '48', 'QWEN_PREFIX_STORE_GIB': '16'}
NEXT_LEVERS = {'QWEN_FAST_SOLO_PACKED': '1', 'QWEN_PREFIX_STICKY_UNDROP': '1', 'QWEN_PREFIX_WINDOW_CARRY': '1'}
# what the image's own ENV carries that the levers' admissions read (docker/qwen-c2-serving.Dockerfile)
IMAGE_ENV = {'QWEN_FAST_PACKED_STEP': '1', 'QWEN_FAST_PADDED_BLOCK': '1', 'QWEN_FAST_SINGLE_GATEUP': '1', 'QWEN_FAST_FOUR_AS_TWO': '0', 'QWEN_FAST_VERIFY_T2': '1'}


def boot_problems(profile):
    from test_octo3_contract import boot_problems as every_check

    return every_check(profile) + contract.prefix_policy_problems(profile) + contract.exact_resume_problems(profile)


def parent_profiles():
    result = subprocess.run(['git', 'show', '%s:scripts/ci/qwen_c2_profiles.json' % PARENT_COMMIT], capture_output=True, cwd=str(HERE), timeout=120)
    if result.returncode != 0:
        return None
    return json.loads(result.stdout.decode('utf-8'))['profiles']


class CandidateTests(unittest.TestCase):
    def test_the_candidate_is_a_generated_traffic_profile_and_no_gate_twin(self):
        self.assertEqual(SHIP, NS + 'traffic-s5')
        self.assertIn(SHIP, fusion.ship_names())
        self.assertNotIn(SHIP, fusion.twin_names())
        self.assertNotIn(SHIP, profile_twins.twin_names())
        self.assertIn(SHIP, profile_twins.generated_names())
        regenerated = fusion.generate_profiles(json.loads(PROFILES_PATH.read_text(encoding='utf-8')), PLAN)['profiles']
        self.assertEqual(regenerated[SHIP], PROFILES[SHIP])
        self.assertNotIn('gate_only', PROFILES[SHIP])
        self.assertEqual((SPEC['pack'], SPEC['image'], SPEC['control'], SPEC['bake_graft'], SPEC['memory_plan']), ('S5', 'tp4-next-1', 'traffic-s4', 'K64j-OQ', 'octo'))

    def test_the_candidate_is_the_approved_ship_4_profile_plus_exactly_the_named_differences(self):
        old, new = PROFILES[SHIP4], PROFILES[SHIP]
        changed = {key: new['env'].get(key) for key in set(old['env']) | set(new['env']) if old['env'].get(key) != new['env'].get(key)}
        self.assertEqual(changed, dict(CHANGES, **{contract.OCTO_WAIVER_ENV: PROFILES[SHIP][contract.TRAFFIC_WAIVER]['id']}))
        for key in old:
            if key not in ('env', 'description', contract.TRAFFIC_WAIVER):
                self.assertEqual(new[key], old[key], key)
        self.assertEqual((new['engine']['num-gpu-blocks-override'], new['env']['QWEN36_MAX_TOKENS_ALL_USERS']), (16500, '1055488'), 'the memory plan is the parent\'s')
        self.assertEqual(new['env']['QWEN_FAST_OUTPUT_BUDGET'], '16384', 'NP9 is a catalog key, not a profile line')

    def test_the_w3_levers_are_not_in_the_candidate(self):
        for key in PROFILES[SHIP]['env']:
            self.assertNotIn('GDN_WC', key)
        self.assertEqual(PROFILES[SHIP]['env']['QWEN_FAST_TP4_CONV_GATES_SPREAD'], '1', 'F1 stays the shipped launch')

    def test_the_approved_ship_4_profile_and_every_earlier_profile_are_untouched(self):
        parent = parent_profiles()
        if parent is None:
            self.skipTest('no git history for %s' % PARENT_COMMIT)
        for name, body in parent.items():
            self.assertEqual(PROFILES[name], body, name)
        added = [name for name in PROFILES if name not in parent]
        self.assertTrue({SHIP, AUDIT, NOWC} <= set(added), added)
        self.assertEqual(PROFILES[SHIP4][contract.TRAFFIC_WAIVER]['id'], 'ship4-fx-262k-2026-10-11')
        self.assertTrue(PROFILES[SHIP4][contract.TRAFFIC_WAIVER]['decision'].startswith('APPROVED'))
        self.assertEqual(PROFILES[SHIP4][contract.TRAFFIC_WAIVER]['levers']['QWEN_PREFIX_HOST_TIER_GIB'], '32')
        self.assertEqual(PROFILES[SHIP4]['env']['QWEN_FAST_GRID13'], 'prefill,drafter')

    def test_one_waiver_of_its_own_naming_the_added_levers_and_the_new_tier_size(self):
        waiver = PROFILES[SHIP][contract.TRAFFIC_WAIVER]
        profile = dict(PROFILES[SHIP], name=SHIP)
        self.assertEqual(contract.traffic_waiver_problems(profile), [])
        old = PROFILES[SHIP4][contract.TRAFFIC_WAIVER]['levers']
        self.assertEqual(waiver['levers'], dict(old, QWEN_PREFIX_HOST_TIER_GIB='48', **NEXT_LEVERS))
        self.assertTrue(waiver['decision'].startswith(('PENDING', 'APPROVED')), waiver['decision'])
        self.assertEqual(contract.traffic_waiver_pending(profile), waiver['decision'].startswith('PENDING'))
        self.assertEqual(waiver['id'], PROFILES[SHIP]['env'][contract.OCTO_WAIVER_ENV])
        self.assertNotEqual(waiver['id'], PROFILES[SHIP4][contract.TRAFFIC_WAIVER]['id'])
        self.assertEqual(waiver['decision'], SPEC['waiver']['decision'])
        self.assertEqual(sorted(waiver), ['decision', 'evidence', 'id', 'levers', 'waives'])
        ids = [body[contract.TRAFFIC_WAIVER]['id'] for body in PROFILES.values() if contract.TRAFFIC_WAIVER in body]
        self.assertEqual(len(ids), len(set(ids)), 'every waiver id is its own')

    def test_the_waiver_says_what_it_adds_changes_and_what_has_not_run(self):
        text = PROFILES[SHIP][contract.TRAFFIC_WAIVER]['waives']
        for phrase in ('ship4-fx-262k-2026-10-11', 'QWEN_FAST_SOLO_PACKED=1', 'QWEN_PREFIX_STICKY_UNDROP=1', 'QWEN_PREFIX_WINDOW_CARRY=1', 'QWEN_PREFIX_TIER_RESTORE_SLICE=512', 'QWEN_PREFIX_HOST_TIER_GIB=48',
                       'QWEN_PREFIX_STORE_GIB=16', 'QWEN_FAST_GRID13=1', 'output_budget', 'W3 levers', 'Nothing in this list has run on a card TOGETHER', 'switchable', 'idle_scratch.py'):
            self.assertIn(phrase, text)
        evidence = ' '.join(PROFILES[SHIP][contract.TRAFFIC_WAIVER]['evidence'])
        for phrase in ('tp4/np1-fastlane (CPU)', 'tp4/exact-resume (CPU)', 'tp4/tier-slice (CPU)', 'what has NOT run'):
            self.assertIn(phrase, evidence)

    def test_no_gate_instrument_is_in_the_candidate(self):
        env = PROFILES[SHIP]['env']
        self.assertNotIn('QWEN_C2_GATE_PROFILE', env)
        for key in contract.OCTO_GATE_ONLY:
            if key != 'QWEN_FAST_SOLO_PACKED':          # the one gate name the waiver admits (NEXT_WAIVABLE_LEVERS)
                self.assertNotIn(key, env, key)
        self.assertNotIn('QWEN_PREFIX_WINDOW_CARRY_AUDIT', env)
        for key, value in env.items():
            self.assertFalse(key.endswith('_AUDIT') and value == '1' and key not in PROFILES[fusion.PARENT]['env'], key)
        self.assertEqual((env['QWEN_PREFIX_HOST_TIER_AUDIT'], env['QWEN_PREFIX_CKPT_PRECONVERTED_AUDIT'], env['QWEN_PREFIX_HOST_TIER_VERIFY']), ('0', '0', 'sample'))

    def test_the_text_of_the_profile_is_public_safe(self):
        text = json.dumps([PROFILES[SHIP]['description'], PROFILES[SHIP][contract.TRAFFIC_WAIVER]])
        for word in fusion.SHIP_FORBIDDEN:
            self.assertNotIn(word, text)
        self.assertTrue(PROFILES[SHIP]['description'].startswith('A SHIP CANDIDATE'))

    def test_the_candidate_passes_every_boot_check(self):
        self.assertEqual(boot_problems(dict(PROFILES[SHIP], name=SHIP)), [])

    def test_the_tier_budget_is_a_whole_state_budget_with_a_store_inside_it(self):
        env = PROFILES[SHIP]['env']
        self.assertEqual((env['QWEN_PREFIX_HOST_TIER_GIB'], env['QWEN_PREFIX_STORE_GIB']), ('48', '16'))
        self.assertGreater(float(env['QWEN_PREFIX_HOST_TIER_GIB']), float(env['QWEN_PREFIX_STORE_GIB']), 'qwen_prefix_registry: the tier\'s GiB must exceed the store\'s')
        self.assertEqual(float(env['QWEN_PREFIX_HOST_TIER_GIB']) - float(env['QWEN_PREFIX_STORE_GIB']), 32.0, 'the KV share the owner approved')


class ContractTests(unittest.TestCase):
    """What the contract admits on a traffic profile now that a waiver may name the lone-user round, exact resume and the tier at 48 GiB."""

    def profile(self, **edits):
        profile = copy.deepcopy(dict(PROFILES[SHIP], name=SHIP))
        for key, value in edits.items():
            if value is None:
                profile['env'].pop(key, None)
            else:
                profile['env'][key] = value
        return profile

    def refused(self, profile):
        return boot_problems(profile)

    def test_each_added_lever_is_refused_on_a_traffic_profile_the_waiver_does_not_name(self):
        for flag in NEXT_LEVERS:
            profile = self.profile()
            del profile[contract.TRAFFIC_WAIVER]['levers'][flag]
            problems = self.refused(profile)
            self.assertTrue(any(flag in problem for problem in problems), (flag, problems))

    def test_a_waiver_naming_a_lever_the_profile_does_not_set_is_stale(self):
        for flag in NEXT_LEVERS:
            problems = contract.traffic_waiver_problems(self.profile(**{flag: None}))
            self.assertTrue(any(flag in problem for problem in problems), (flag, problems))

    def test_the_levers_are_waivable_at_their_value_only(self):
        for flag, bad in (('QWEN_FAST_SOLO_PACKED', '0'), ('QWEN_PREFIX_STICKY_UNDROP', '2'), ('QWEN_PREFIX_WINDOW_CARRY', 'on')):
            profile = self.profile()
            profile[contract.TRAFFIC_WAIVER]['levers'][flag] = bad
            self.assertTrue(any(flag in problem for problem in contract.traffic_waiver_problems(profile)), flag)

    def test_the_tier_is_waivable_at_32_and_48_and_at_no_other_size(self):
        self.assertEqual(contract.TIER_WAIVABLE_LEVERS['QWEN_PREFIX_HOST_TIER_GIB'], '32', 'ship 4\'s waiver and the tier candidate\'s are unchanged')
        for size, ok in (('32', True), ('48', True), ('40', False), ('64', False)):
            profile = self.profile(QWEN_PREFIX_HOST_TIER_GIB=size)
            profile[contract.TRAFFIC_WAIVER]['levers']['QWEN_PREFIX_HOST_TIER_GIB'] = size
            self.assertEqual(contract.traffic_waiver_problems(profile) == [], ok, size)

    def test_the_gate_names_stay_a_gates_own(self):
        for flag, value in (('QWEN_PREFIX_WINDOW_CARRY_AUDIT', '1'), ('QWEN_FAST_TP4_GDN_WC_AUDIT', '1')):
            problems = self.refused(self.profile(**{flag: value}))
            self.assertTrue(any(flag in problem for problem in problems) or flag.startswith('QWEN_FAST_TP4_GDN_WC'), (flag, problems))
        for flag in ('QWEN_FAST_AUDIT_FIRST', 'QWEN_FAST_OCTO_ATTN_BUNDLE_AUDIT'):
            self.assertTrue(any(flag in problem for problem in self.refused(self.profile(**{flag: '1'}))), flag)
        for flag in ('QWEN_PREFIX_TENANT_QUOTA_DEVICE_CAP_TOKENS',):
            self.assertTrue(self.refused(self.profile(**{flag: '65536'})), flag)

    def test_the_carry_needs_the_undrop_and_both_need_sticky_prefix_reuse(self):
        self.assertTrue(any('WINDOW_CARRY=1 without' in problem for problem in contract.exact_resume_problems(self.profile(QWEN_PREFIX_STICKY_UNDROP=None))))
        self.assertTrue(contract.exact_resume_problems(self.profile(QWEN_FAST_STICKY_SESSIONS=None)))

    def test_the_slice_needs_the_tier_in_the_waiver(self):
        profile = self.profile()
        profile[contract.TRAFFIC_WAIVER]['levers'].pop('QWEN_PREFIX_HOST_TIER_GIB')
        self.assertTrue(self.refused(profile))
        self.assertTrue(any('sliced restore' in problem for problem in contract.prefix_policy_problems(profile)))

    def test_the_lone_user_round_is_admitted_in_a_waiver_run_and_in_a_gate_run_and_nowhere_else(self):
        self.assertTrue(octo.admitted_run({octo.WAIVER_ENV: PROFILES[SHIP]['env'][contract.OCTO_WAIVER_ENV]}))
        self.assertTrue(octo.admitted_run({octo.GATE_ENV: '1', octo.GATE_PROFILE_ENV: '1'}))
        self.assertFalse(octo.admitted_run({}))
        lines = []
        env = dict(IMAGE_ENV, QWEN_FAST_SOLO_PACKED='1', QWEN_FAST_TP='4', QWEN_FAST_EXTENT_REPLAY='1', QWEN_FAST_PACKED_STEP='1', QWEN_FAST_M3_BLOCKS='2')
        with mock.patch.object(octo, 'scratch_pages', lambda: (16500,)):
            for extra, ok in (({octo.WAIVER_ENV: 'ship5-fx-262k-2026-10-19'}, True), ({octo.GATE_ENV: '1', octo.GATE_PROFILE_ENV: '1'}, True), ({}, False)):
                lines[:] = []
                try:
                    found = octo.solo_packed_admission((True, 'users=4'), dict(env, **extra), log=lambda template, *values: lines.append(template.format(*values)))
                    raised = False
                except ValueError:
                    found, raised = None, True
                self.assertEqual(not raised, ok, extra)
                if ok:
                    self.assertEqual(found['min_users'], 1)
                    self.assertTrue(lines[-1].endswith('(owner traffic waiver ship5-fx-262k-2026-10-19)' if octo.WAIVER_ENV in extra else '(gate only)'), lines)


class TwinTests(unittest.TestCase):
    def test_the_twins_are_gate_only_generated_and_boot_clean(self):
        for name in (AUDIT, NOWC):
            self.assertIs(PROFILES[name]['gate_only'], True)
            self.assertNotIn(contract.TRAFFIC_WAIVER, PROFILES[name])
            self.assertNotIn(contract.OCTO_WAIVER_ENV, PROFILES[name]['env'])
            self.assertEqual(PROFILES[name]['env']['QWEN_C2_GATE_PROFILE'], '1')
            self.assertEqual(boot_problems(dict(PROFILES[name], name=name)), [], name)
            self.assertIn(name, profile_twins.twin_names())
        self.assertEqual(twins.twin_names(), (AUDIT, NOWC))
        self.assertEqual(twins.generate(DATA), DATA, 'the file is what the generator makes')
        self.assertEqual(twins.main(['--check']), 0)

    def test_the_audited_twin_is_the_ship_4_audited_twin_plus_the_candidates_differences_the_carry_audit_and_wc_less_f1s_audit(self):
        parent, audit = PROFILES[ship4.AUDIT], PROFILES[AUDIT]
        changed = {key: audit['env'].get(key) for key in set(parent['env']) | set(audit['env']) if parent['env'].get(key) != audit['env'].get(key)}
        expected = {key: value for key, value in CHANGES.items() if parent['env'].get(key) != value}
        expected.update(twins.AUDIT_ADDS)
        expected.update({flag: None for flag in twins.AUDIT_REMOVES})
        self.assertEqual(changed, expected)
        self.assertEqual(expected['QWEN_PREFIX_WINDOW_CARRY_AUDIT'], '1')
        for key in parent:
            if key not in ('env', 'description'):
                self.assertEqual(audit[key], parent[key], key)
        self.assertEqual(audit['engine'], PROFILES[SHIP]['engine'], 'the audited twin has the candidate\'s memory plan')

    def test_the_fallback_is_the_audited_twin_without_wc_and_with_f1s_audit_back(self):
        audit, nowc = PROFILES[AUDIT]['env'], PROFILES[NOWC]['env']
        changed = {key: (audit.get(key), nowc.get(key)) for key in set(audit) | set(nowc) if audit.get(key) != nowc.get(key)}
        self.assertEqual(changed, {'QWEN_FAST_TP4_GDN_WC': ('1', None), 'QWEN_FAST_TP4_GDN_WC_AUDIT': ('1', None), 'QWEN_FAST_TP4_CONV_GATES_SPREAD_AUDIT': (None, '1')})
        self.assertEqual(nowc['QWEN_FAST_TP4_CONV_GATES_SPREAD_AUDIT'], PROFILES[ship4.AUDIT]['env']['QWEN_FAST_TP4_CONV_GATES_SPREAD_AUDIT'])

    def test_wc_and_its_prerequisites_hold_on_the_audited_twin_and_not_on_the_fallback(self):
        env = dict(IMAGE_ENV, **PROFILES[AUDIT]['env'])
        self.assertTrue(gdn_wc.enabled(env))
        self.assertTrue(gdn_wc.audit_enabled(env))
        self.assertFalse(gdn_wc.enabled(dict(IMAGE_ENV, **PROFILES[NOWC]['env'])))
        # the pair F1's audit refuses beside WC is the one the twin takes out; putting it back is refused by the module, by name
        with self.assertRaises(ValueError):
            gdn_wc.enabled(dict(env, QWEN_FAST_TP4_CONV_GATES_SPREAD_AUDIT='1'))

    def test_the_smoke_rule_accepts_the_octo_blocks_one_decline_on_the_audited_twin_only(self):
        decline = gdn_wc.FELL_BACK + ' reason=8 users (WC takes 2 or 4)'
        engaged = gdn_wc.ENGAGED + ' rows=64 conv_cores=80 gate_cores=2 z_cores=28 per_core=2 served_per_core=2 audit=1'
        audit = gdn_wc.AUDIT_MARKER + ' 3 exact=True layers=0,24 entries=64'
        text = '\n'.join([engaged, audit, decline])
        self.assertEqual(w3_wc_smoke.problems(PROFILES[AUDIT]['env'], text), [])
        self.assertEqual(len(w3_wc_smoke.problems(PROFILES[AUDIT]['env'], text + '\n' + gdn_wc.FELL_BACK + ' reason=3 users (WC takes 2 or 4)')), 1)
        self.assertEqual(len(w3_wc_smoke.problems(PROFILES[SHIP]['env'], decline)), 1, 'a profile that does not ask for WC logs none of its lines')

    def test_the_twins_need_no_owner_waiver_and_there_is_no_timed_twin(self):
        names = [name for name in PROFILES if name.startswith(NS) and 's5' in name]
        self.assertEqual(sorted(names), sorted([SHIP, AUDIT, NOWC]), 'the timed arm is the candidate itself: a PENDING traffic profile boots in a smoke')

    def test_a_twin_is_refused_when_its_parent_moves(self):
        data = copy.deepcopy(DATA)
        data['profiles'][ship4.AUDIT]['env']['QWEN_FAST_TP4_GDN_WC'] = '1'
        with self.assertRaises(ValueError):
            twins.generate(data)
        data = copy.deepcopy(DATA)
        del data['profiles'][ship4.AUDIT]['env']['QWEN_FAST_TP4_CONV_GATES_SPREAD_AUDIT']
        with self.assertRaises(ValueError):
            twins.generate(data)
        data = copy.deepcopy(DATA)
        data['profiles'][SHIP]['env'].pop(contract.OCTO_WAIVER_ENV)
        with self.assertRaises(ValueError):
            twins.generate(data)


class ApprovalTests(unittest.TestCase):
    def test_approving_the_candidate_changes_its_decision_sentence_and_nothing_else(self):
        directory = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, str(directory), True)
        for path in fusion.MANIFESTS.glob('*.json'):
            shutil.copy(str(path), str(directory / path.name))
        before = {path.name: path.read_text(encoding='utf-8') for path in directory.glob('*.json')}
        was = fusion.approve_ship('APPROVED by the owner (remote control): the ship-5 candidate', directory, SPEC['name'])
        self.assertEqual(was, SPEC['waiver']['decision'])
        after = {path.name: path.read_text(encoding='utf-8') for path in directory.glob('*.json')}
        self.assertEqual([name for name in after if after[name] != before[name]], ['S5.json'])
        old, new = json.loads(before['S5.json']), json.loads(after['S5.json'])
        new['ship'][0]['waiver']['decision'] = old['ship'][0]['waiver']['decision']
        self.assertEqual(new, old)
        plan = fusion.normalise(fusion.read_manifests(directory))
        generated = fusion.generate_profiles(json.loads(PROFILES_PATH.read_text(encoding='utf-8')), plan)['profiles']
        self.assertFalse(contract.traffic_waiver_pending(dict(generated[SHIP], name=SHIP)))
        self.assertEqual({key: value for key, value in generated[SHIP].items() if key != contract.TRAFFIC_WAIVER}, {key: value for key, value in PROFILES[SHIP].items() if key != contract.TRAFFIC_WAIVER})
        self.assertEqual(boot_problems(dict(generated[SHIP], name=SHIP)), [])
        for name in fusion.ship_names():
            if name != SHIP:
                self.assertEqual(generated[name], PROFILES[name], name)
        # the twins do not read the decision: regenerating them over the approved file changes nothing
        data = json.loads(PROFILES_PATH.read_text(encoding='utf-8'))
        data['profiles'] = generated
        self.assertEqual(twins.generate(data)['profiles'][AUDIT], PROFILES[AUDIT])


class ManifestTests(unittest.TestCase):
    def test_the_manifest_keeps_the_pack_folder_and_lists_its_tests(self):
        raw = json.loads((fusion.MANIFESTS / 'S5.json').read_text(encoding='utf-8'))
        self.assertEqual(raw['wp'], 'S5')
        self.assertEqual(raw['tests'], ['test_s5_profiles', 'test_s5_pack'])
        self.assertEqual(len(raw['ship']), 1)

    def test_the_w3_manifest_wires_the_wc_smoke_rule_and_nothing_else(self):
        raw = json.loads((fusion.MANIFESTS / 'W3.json').read_text(encoding='utf-8'))
        self.assertEqual(raw['smoke_rules'], ['w3_wc_smoke'])
        self.assertEqual(set(raw) - {'wp', 'branch', 'description', 'smoke_rules', 'docs'}, set())


if __name__ == '__main__':
    unittest.main()
