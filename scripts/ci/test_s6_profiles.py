"""tp4/s6: the ship-6 candidate (a traffic profile generated from fusion-wp/S6.json by make_fusion_profiles.py), its gate-only audited twin (make_s6_profiles) and the contract's admission of the governor's switch.

  -fx-traffic-s6   the ship-5 profile that serves production + W3 WC (QWEN_FAST_TP4_GDN_WC=1) + the adaptive Lever N governor (QWEN_FAST_LEVERN_ADAPTIVE=1, default knobs), under ONE owner traffic waiver
                   (decision PENDING or APPROVED: nothing here reads it);
  -fx-s6-audit     its audited twin: the ship-5 audited twin (WC and its audit already in it) + the governor's switch.

The ship-5 profile, its waiver and every profile that existed before this branch stay byte-identical (checked against the commit the branch was cut from); nothing here is hand-edited. The control of
every timed pair is the ship-5 profile itself. The tests are decision-agnostic: the owner's approval at ship time changes one sentence of the manifest and these tests still pass (ApprovalTests)."""

import copy
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import gdn_wc  # noqa: E402
import levern_policy  # noqa: E402
import make_fusion_profiles as fusion  # noqa: E402
import make_s5_profiles as s5  # noqa: E402
import make_s6_profiles as twins  # noqa: E402
import profile_twins  # noqa: E402
import serving_c2_contract as contract  # noqa: E402
import w3_wc_smoke  # noqa: E402

PROFILES_PATH = HERE / 'qwen_c2_profiles.json'
DATA = json.loads(PROFILES_PATH.read_text(encoding='utf-8'))
PROFILES = DATA['profiles']
NS = fusion.NAMESPACE
SHIP5, SHIP, AUDIT5, AUDIT = s5.SHIP, twins.SHIP, s5.AUDIT, twins.AUDIT
PLAN = fusion.normalise(fusion.read_manifests())
SPEC = next(item for item in PLAN['ship'] if item['name'] == 'traffic-s6')
PARENT_COMMIT = '758264ac'                      # tp4/next-1 at the 2026-10-19 window's end: the tree this branch was cut from
# what the candidate sets that ship 5 does not, the waiver id aside
CHANGES = {'QWEN_FAST_TP4_GDN_WC': '1', 'QWEN_FAST_LEVERN_ADAPTIVE': '1'}
# the sub-knobs and the sched-1 pacing knobs: a gate's own on every profile that serves traffic, with or without the waiver
GOVERNOR_KNOBS = (('QWEN_FAST_LEVERN_ADAPTIVE_DECODERS', '2'), ('QWEN_FAST_LEVERN_ADAPTIVE_SHARE', '0.9'), ('QWEN_FAST_LEVERN_ADAPTIVE_TTFT_S', '120'),
                  ('QWEN_FAST_LEVERN_OWED_CAP_MS', '1500'), ('QWEN_FAST_LEVERN_SHORT_SHARE', '0.65'), ('QWEN_FAST_LEVERN_SHORT_WAIT_S', '5'))
# what the image's own ENV carries that the levers' admissions read (docker/qwen-c2-serving.Dockerfile)
IMAGE_ENV = {'QWEN_FAST_PACKED_STEP': '1', 'QWEN_FAST_PADDED_BLOCK': '1', 'QWEN_FAST_SINGLE_GATEUP': '1', 'QWEN_FAST_FOUR_AS_TWO': '0', 'QWEN_FAST_VERIFY_T2': '1'}


def boot_problems(profile):
    from test_octo3_contract import boot_problems as every_check

    return every_check(profile) + contract.prefix_policy_problems(profile) + contract.exact_resume_problems(profile) + contract.sched_problems(profile)


def parent_profiles():
    result = subprocess.run(['git', 'show', '%s:scripts/ci/qwen_c2_profiles.json' % PARENT_COMMIT], capture_output=True, cwd=str(HERE), timeout=120)
    if result.returncode != 0:
        return None
    return json.loads(result.stdout.decode('utf-8'))['profiles']


class CandidateTests(unittest.TestCase):
    def test_the_candidate_is_a_generated_traffic_profile_and_no_gate_twin(self):
        self.assertEqual(SHIP, NS + 'traffic-s6')
        self.assertIn(SHIP, fusion.ship_names())
        self.assertNotIn(SHIP, fusion.twin_names())
        self.assertNotIn(SHIP, profile_twins.twin_names())
        self.assertIn(SHIP, profile_twins.generated_names())
        regenerated = fusion.generate_profiles(json.loads(PROFILES_PATH.read_text(encoding='utf-8')), PLAN)['profiles']
        self.assertEqual(regenerated[SHIP], PROFILES[SHIP])
        self.assertNotIn('gate_only', PROFILES[SHIP])
        self.assertEqual((SPEC['pack'], SPEC['image'], SPEC['control'], SPEC['bake_graft'], SPEC['memory_plan']), ('S6', 'tp4-s6-1', 'traffic-s5', 'K64j-OQ', 'octo'))

    def test_the_candidate_is_the_ship_5_profile_plus_exactly_wc_and_the_governor(self):
        old, new = PROFILES[SHIP5], PROFILES[SHIP]
        changed = {key: new['env'].get(key) for key in set(old['env']) | set(new['env']) if old['env'].get(key) != new['env'].get(key)}
        self.assertEqual(changed, dict(CHANGES, **{contract.OCTO_WAIVER_ENV: PROFILES[SHIP][contract.TRAFFIC_WAIVER]['id']}))
        for key in old:
            if key not in ('env', 'description', contract.TRAFFIC_WAIVER):
                self.assertEqual(new[key], old[key], key)
        self.assertEqual((new['engine']['num-gpu-blocks-override'], new['env']['QWEN36_MAX_TOKENS_ALL_USERS']), (16500, '1055488'), 'the memory plan is the parent\'s')
        self.assertEqual(candidate_env_shared_with_ship5(), [], 'no other line of ship 5 moved')

    def test_the_governor_runs_at_its_default_knobs_and_the_floor_stays_on(self):
        env = PROFILES[SHIP]['env']
        for flag, _value in GOVERNOR_KNOBS:
            self.assertNotIn(flag, env, flag)
        self.assertEqual(env['QWEN_FAST_LEVERN_MAX_DECODE_GAP_S'], '8', 'the policy REQUIRES the decode-gap floor')
        self.assertEqual((env['QWEN_FAST_LEVERN_PREFILL_SHARE'], env['QWEN_FAST_LEVERN_TTFT_TARGET_S']), ('0.5', '180'), 'today\'s share and target are what the policy raises and lowers')
        self.assertEqual(levern_policy.config_problems(env), [])
        policy = levern_policy.adaptive_config(env)
        self.assertEqual((policy.decoders, policy.share, policy.ttft_s), (2, 0.9, 120))
        self.assertIsNone(levern_policy.adaptive_config(PROFILES[SHIP5]['env']), 'the ship-5 profile runs today\'s pacing')

    def test_wc_and_its_prerequisites_hold_on_the_candidate(self):
        env = dict(IMAGE_ENV, **PROFILES[SHIP]['env'])
        self.assertTrue(gdn_wc.enabled(env))
        self.assertFalse(gdn_wc.audit_enabled(env), 'the audit is a gate\'s own')
        self.assertEqual(env['QWEN_FAST_TP4_CONV_GATES_SPREAD'], '1', 'F1 stays the launch for a layer WC declines (the octo block)')
        self.assertNotIn('QWEN_FAST_TP4_GDN_WC_AUDIT', PROFILES[SHIP]['env'])
        self.assertNotIn('QWEN_FAST_TP4_CONV_GATES_SPREAD_AUDIT', PROFILES[SHIP]['env'])

    def test_the_ship_5_profile_and_every_earlier_profile_are_untouched(self):
        parent = parent_profiles()
        if parent is None:
            self.skipTest('no git history for %s' % PARENT_COMMIT)
        for name, body in parent.items():
            self.assertEqual(PROFILES[name], body, name)
        added = [name for name in PROFILES if name not in parent]
        self.assertEqual(sorted(added), sorted([SHIP, AUDIT]))
        self.assertEqual(PROFILES[SHIP5][contract.TRAFFIC_WAIVER]['id'], 'ship5-fx-262k-2026-10-19')
        self.assertTrue(PROFILES[SHIP5][contract.TRAFFIC_WAIVER]['decision'].startswith('APPROVED'))
        self.assertNotIn('QWEN_FAST_LEVERN_ADAPTIVE', PROFILES[SHIP5]['env'])
        self.assertNotIn('QWEN_FAST_TP4_GDN_WC', PROFILES[SHIP5]['env'])

    def test_one_waiver_of_its_own_naming_the_governor_beside_ship_5s_levers(self):
        waiver = PROFILES[SHIP][contract.TRAFFIC_WAIVER]
        profile = dict(PROFILES[SHIP], name=SHIP)
        self.assertEqual(contract.traffic_waiver_problems(profile), [])
        old = PROFILES[SHIP5][contract.TRAFFIC_WAIVER]['levers']
        self.assertEqual(waiver['levers'], dict(old, QWEN_FAST_LEVERN_ADAPTIVE='1'))
        self.assertTrue(waiver['decision'].startswith(('PENDING', 'APPROVED')), waiver['decision'])
        self.assertEqual(contract.traffic_waiver_pending(profile), waiver['decision'].startswith('PENDING'))
        self.assertEqual(waiver['id'], PROFILES[SHIP]['env'][contract.OCTO_WAIVER_ENV])
        self.assertNotEqual(waiver['id'], PROFILES[SHIP5][contract.TRAFFIC_WAIVER]['id'])
        self.assertEqual(waiver['decision'], SPEC['waiver']['decision'])
        self.assertEqual(sorted(waiver), ['decision', 'evidence', 'id', 'levers', 'waives'])
        ids = [body[contract.TRAFFIC_WAIVER]['id'] for body in PROFILES.values() if contract.TRAFFIC_WAIVER in body]
        self.assertEqual(len(ids), len(set(ids)), 'every waiver id is its own')

    def test_the_waiver_carries_the_ship_5_text_verbatim_and_says_what_it_adds_and_what_has_not_run(self):
        text, old = PROFILES[SHIP][contract.TRAFFIC_WAIVER]['waives'], PROFILES[SHIP5][contract.TRAFFIC_WAIVER]['waives']
        self.assertTrue(text.startswith(old), 'the ship-5 waiver text is carried verbatim')
        added = text[len(old):]
        for phrase in ('ship5-fx-262k-2026-10-19', '(Q) W3 WC', 'QWEN_FAST_TP4_GDN_WC=1', '(R) The adaptive Lever N governor', 'QWEN_FAST_LEVERN_ADAPTIVE=1', 'ESTIMATE', 'switchable',
                       'NEVER run on a card', 'TRADES decoder speed for first-token time', 'octo', 'NEXT_WAIVABLE_LEVERS', 'IT SHIPS ONLY IF THE AUDITED RUN'):
            self.assertIn(phrase, added)
        evidence = ' '.join(PROFILES[SHIP][contract.TRAFFIC_WAIVER]['evidence'])
        for phrase in ('the ship-5 window', 'W3 WC on cards', 'tp4/prefill-1 and tp4/sched-1 (CPU)', 'what has NOT run'):
            self.assertIn(phrase, evidence)
        self.assertEqual(PROFILES[SHIP][contract.TRAFFIC_WAIVER]['evidence'][:len(PROFILES[SHIP5][contract.TRAFFIC_WAIVER]['evidence'])], PROFILES[SHIP5][contract.TRAFFIC_WAIVER]['evidence'])

    def test_no_gate_instrument_is_in_the_candidate(self):
        env = PROFILES[SHIP]['env']
        self.assertNotIn('QWEN_C2_GATE_PROFILE', env)
        for key in contract.OCTO_GATE_ONLY:
            if key != 'QWEN_FAST_SOLO_PACKED':          # the one gate name the waiver admits (NEXT_WAIVABLE_LEVERS)
                self.assertNotIn(key, env, key)
        for key, value in env.items():
            self.assertFalse(key.endswith('_AUDIT') and value == '1' and key not in PROFILES[fusion.PARENT]['env'], key)
        self.assertEqual((env['QWEN_PREFIX_HOST_TIER_AUDIT'], env['QWEN_PREFIX_CKPT_PRECONVERTED_AUDIT'], env['QWEN_PREFIX_HOST_TIER_VERIFY']), ('0', '0', 'sample'))

    def test_the_text_of_the_profile_is_public_safe(self):
        text = json.dumps([PROFILES[SHIP]['description'], PROFILES[SHIP][contract.TRAFFIC_WAIVER], PROFILES[AUDIT]['description']])
        for word in fusion.SHIP_FORBIDDEN:
            self.assertNotIn(word, text)
        self.assertTrue(PROFILES[SHIP]['description'].startswith('A SHIP CANDIDATE'))

    def test_the_candidate_passes_every_boot_check(self):
        self.assertEqual(boot_problems(dict(PROFILES[SHIP], name=SHIP)), [])


def candidate_env_shared_with_ship5():
    """Every line of the ship-5 profile the ship-6 profile does not carry at the same value (the three differences aside)."""
    old, new = PROFILES[SHIP5]['env'], PROFILES[SHIP]['env']
    return [key for key in sorted(old) if key != contract.OCTO_WAIVER_ENV and new.get(key) != old[key]]


class ContractTests(unittest.TestCase):
    """What the contract admits on a traffic profile now that a waiver may name the governor's switch."""

    def profile(self, **edits):
        profile = copy.deepcopy(dict(PROFILES[SHIP], name=SHIP))
        for key, value in edits.items():
            if value is None:
                profile['env'].pop(key, None)
            else:
                profile['env'][key] = value
        return profile

    def test_the_switch_is_waivable_at_one_and_nowhere_else_in_the_table(self):
        self.assertEqual(contract.NEXT_WAIVABLE_LEVERS['QWEN_FAST_LEVERN_ADAPTIVE'], '1')
        self.assertIn('QWEN_FAST_LEVERN_ADAPTIVE', contract.LEVERN_ADAPTIVE_FLAGS, 'it is still one of the governor\'s gate instruments')
        for flag, _value in GOVERNOR_KNOBS:
            self.assertNotIn(flag, contract.NEXT_WAIVABLE_LEVERS, flag)
        profile = self.profile()
        profile[contract.TRAFFIC_WAIVER]['levers']['QWEN_FAST_LEVERN_ADAPTIVE'] = '0'
        self.assertTrue(any('QWEN_FAST_LEVERN_ADAPTIVE' in problem for problem in contract.traffic_waiver_problems(profile)))

    def test_the_switch_is_refused_on_a_traffic_profile_the_waiver_does_not_name(self):
        profile = self.profile()
        del profile[contract.TRAFFIC_WAIVER]['levers']['QWEN_FAST_LEVERN_ADAPTIVE']
        problems = boot_problems(profile)
        self.assertTrue(any('QWEN_FAST_LEVERN_ADAPTIVE' in problem for problem in problems), problems)
        ship5 = copy.deepcopy(dict(PROFILES[SHIP5], name=SHIP5))
        ship5['env']['QWEN_FAST_LEVERN_ADAPTIVE'] = '1'
        problems = boot_problems(ship5)
        self.assertTrue(any('QWEN_FAST_LEVERN_ADAPTIVE' in problem for problem in problems), 'the ship-5 waiver does not name it: ' + repr(problems))

    def test_a_waiver_naming_the_switch_on_a_profile_that_does_not_set_it_is_stale(self):
        problems = contract.traffic_waiver_problems(self.profile(QWEN_FAST_LEVERN_ADAPTIVE=None))
        self.assertTrue(any('QWEN_FAST_LEVERN_ADAPTIVE' in problem for problem in problems), problems)

    def test_the_knobs_stay_a_gates_own_even_beside_the_waiver(self):
        for flag, value in GOVERNOR_KNOBS:
            problems = boot_problems(self.profile(**{flag: value}))
            self.assertTrue(any(flag in problem for problem in problems), (flag, problems))

    def test_the_switch_needs_lever_n_the_merged_route_and_the_floor(self):
        self.assertTrue(any('QWEN_FAST_LEVERN_ADAPTIVE' in problem for problem in boot_problems(self.profile(QWEN_FAST_LEVER_N=None))))
        self.assertTrue(any('QWEN_FAST_LEVERN_ADAPTIVE' in problem or 'GAP' in problem for problem in boot_problems(self.profile(QWEN_FAST_LEVERN_MAX_DECODE_GAP_S='0'))))
        self.assertTrue(boot_problems(self.profile(QWEN_PREFIX_REUSE=None)))
        self.assertTrue(boot_problems(self.profile(QWEN_FAST_LEVERN_ROUNDS='3')), 'static rounds override the share: nothing to adapt')

    def test_a_gate_only_profile_still_sets_the_whole_governor(self):
        gate = copy.deepcopy(dict(PROFILES[AUDIT], name=AUDIT))
        gate['env'].update(QWEN_FAST_LEVERN_ADAPTIVE_DECODERS='3', QWEN_FAST_LEVERN_ADAPTIVE_SHARE='0.8', QWEN_FAST_LEVERN_ADAPTIVE_TTFT_S='100')
        self.assertEqual(boot_problems(gate), [])

    def test_the_other_gate_names_stay_a_gates_own(self):
        for flag, value in (('QWEN_FAST_LEVERN_AUDIT', '1'), ('QWEN_FAST_LEVERN_FAULT', '1'), ('QWEN_PREFIX_WINDOW_CARRY_AUDIT', '1')):
            problems = boot_problems(self.profile(**{flag: value}))
            self.assertTrue(problems, (flag, problems))
        # WC's audit is the module's own refusal (gdn_wc.audit_enabled needs the V1 glue audit beside it), not the profile contract's: the candidate carries neither
        with self.assertRaises(ValueError):
            gdn_wc.audit_enabled(dict(IMAGE_ENV, **self.profile(QWEN_FAST_TP4_GDN_WC_AUDIT='1')['env']))


class TwinTests(unittest.TestCase):
    def test_the_twin_is_gate_only_generated_and_boots_clean(self):
        self.assertIs(PROFILES[AUDIT]['gate_only'], True)
        self.assertNotIn(contract.TRAFFIC_WAIVER, PROFILES[AUDIT])
        self.assertNotIn(contract.OCTO_WAIVER_ENV, PROFILES[AUDIT]['env'])
        self.assertEqual(PROFILES[AUDIT]['env']['QWEN_C2_GATE_PROFILE'], '1')
        self.assertEqual(boot_problems(dict(PROFILES[AUDIT], name=AUDIT)), [])
        self.assertIn(AUDIT, profile_twins.twin_names())
        self.assertEqual(twins.twin_names(), (AUDIT,))
        self.assertEqual(twins.owned_names(), (AUDIT, SHIP))
        self.assertEqual(twins.generate(DATA), DATA, 'the file is what the generator makes')
        self.assertEqual(twins.main(['--check']), 0)

    def test_the_twin_is_the_ship_5_audited_twin_plus_exactly_the_governor(self):
        parent, audit = PROFILES[AUDIT5], PROFILES[AUDIT]
        changed = {key: audit['env'].get(key) for key in set(parent['env']) | set(audit['env']) if parent['env'].get(key) != audit['env'].get(key)}
        self.assertEqual(changed, {'QWEN_FAST_LEVERN_ADAPTIVE': '1'}, 'WC and its audit are already in the parent')
        for key in parent:
            if key not in ('env', 'description'):
                self.assertEqual(audit[key], parent[key], key)
        self.assertEqual(audit['engine'], PROFILES[SHIP]['engine'], 'the audited twin has the candidate\'s memory plan')
        # every difference of the candidate from ship 5 is in the twin
        for flag in CHANGES:
            self.assertEqual(audit['env'][flag], '1')

    def test_wc_and_the_governor_hold_on_the_twin(self):
        env = dict(IMAGE_ENV, **PROFILES[AUDIT]['env'])
        self.assertTrue(gdn_wc.enabled(env))
        self.assertTrue(gdn_wc.audit_enabled(env))
        self.assertEqual(levern_policy.config_problems(env), [])
        policy = levern_policy.adaptive_config(env)
        self.assertEqual((policy.decoders, policy.share, policy.ttft_s), (2, 0.9, 120))

    def test_the_smoke_rule_accepts_the_octo_blocks_one_decline_on_both_profiles(self):
        decline = gdn_wc.FELL_BACK + ' reason=8 users (WC takes 2 or 4)'
        engaged = gdn_wc.ENGAGED + ' rows=64 conv_cores=80 gate_cores=2 z_cores=28 per_core=2 served_per_core=2 audit=1'
        audit = gdn_wc.AUDIT_MARKER + ' 3 exact=True layers=0,24 entries=64'
        self.assertEqual(w3_wc_smoke.problems(PROFILES[AUDIT]['env'], '\n'.join([engaged, audit, decline])), [])
        self.assertEqual(w3_wc_smoke.problems(PROFILES[SHIP]['env'], '\n'.join([engaged, decline])), [], 'the candidate asks for WC, logs it and declines the octo block once')
        self.assertEqual(len(w3_wc_smoke.problems(PROFILES[SHIP]['env'], decline)), 1, 'WC asked for and never engaged is a fault')
        self.assertEqual(len(w3_wc_smoke.problems(PROFILES[SHIP]['env'], '\n'.join([engaged, decline, gdn_wc.FELL_BACK + ' reason=3 users (WC takes 2 or 4)']))), 1)
        self.assertEqual(len(w3_wc_smoke.problems(PROFILES[SHIP5]['env'], engaged)), 1, 'the ship-5 control asks for no WC and logs none of its lines')

    def test_the_smoke_check_judges_the_governor_on_the_candidate_and_leaves_no_trace_on_the_control(self):
        import c2_smoke_check as check

        self.assertEqual(check.levern_adaptive_problems([], PROFILES[SHIP5]['env'], ''), [], 'the control logs nothing of it')
        self.assertTrue(check.levern_adaptive_problems([], PROFILES[SHIP5]['env'], check.LEVERN_ADAPTIVE_LINE + ' decoders<=2'), 'a line on a profile without the flag is a fault')
        problems = check.levern_adaptive_problems([], PROFILES[SHIP]['env'], '')
        self.assertTrue(any(check.LEVERN_ADAPTIVE_LINE.strip() in problem for problem in problems), 'the candidate asks for it and the log has no install line')

    def test_the_twins_need_no_owner_waiver_and_there_is_no_timed_twin(self):
        names = [name for name in PROFILES if name.startswith(NS) and 's6' in name]
        self.assertEqual(sorted(names), sorted([SHIP, AUDIT]), 'the timed arm is the candidate itself, its control the ship-5 profile')

    def test_a_twin_is_refused_when_its_parent_moves(self):
        data = copy.deepcopy(DATA)
        data['profiles'][AUDIT5]['env']['QWEN_FAST_LEVERN_ADAPTIVE'] = '1'
        with self.assertRaises(ValueError):
            twins.generate(data)
        data = copy.deepcopy(DATA)
        data['profiles'][SHIP]['env'].pop('QWEN_FAST_LEVERN_ADAPTIVE')
        with self.assertRaises(ValueError):
            twins.generate(data)
        data = copy.deepcopy(DATA)
        data['profiles'][SHIP]['env'].pop(contract.OCTO_WAIVER_ENV)
        with self.assertRaises(ValueError):
            twins.generate(data)
        data = copy.deepcopy(DATA)
        del data['profiles'][AUDIT5]
        with self.assertRaises(ValueError):
            twins.generate(data)


class ApprovalTests(unittest.TestCase):
    def test_approving_the_candidate_changes_its_decision_sentence_and_nothing_else(self):
        directory = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, str(directory), True)
        for path in fusion.MANIFESTS.glob('*.json'):
            shutil.copy(str(path), str(directory / path.name))
        before = {path.name: path.read_text(encoding='utf-8') for path in directory.glob('*.json')}
        was = fusion.approve_ship('APPROVED by the owner (remote control): the ship-6 candidate', directory, SPEC['name'])
        self.assertEqual(was, SPEC['waiver']['decision'])
        after = {path.name: path.read_text(encoding='utf-8') for path in directory.glob('*.json')}
        self.assertEqual([name for name in after if after[name] != before[name]], ['S6.json'])
        old, new = json.loads(before['S6.json']), json.loads(after['S6.json'])
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
        # the twin does not read the decision: regenerating it over the approved file changes nothing
        data = json.loads(PROFILES_PATH.read_text(encoding='utf-8'))
        data['profiles'] = generated
        self.assertEqual(twins.generate(data)['profiles'][AUDIT], PROFILES[AUDIT])


class ManifestTests(unittest.TestCase):
    def test_the_manifest_keeps_the_pack_folder_and_lists_its_tests(self):
        raw = json.loads((fusion.MANIFESTS / 'S6.json').read_text(encoding='utf-8'))
        self.assertEqual(raw['wp'], 'S6')
        self.assertEqual(raw['branch'], 'tp4/s6')
        self.assertEqual(len(raw['ship']), 1)
        self.assertEqual(set(raw) - {'wp', 'branch', 'description', 'tests', 'ship'}, set())
        self.assertIn('test_s6_profiles', raw['tests'])

    def test_the_manifest_is_in_the_canonical_form_approve_ship_rewrites(self):
        text = (fusion.MANIFESTS / 'S6.json').read_text(encoding='utf-8')
        self.assertEqual(json.dumps(json.loads(text), indent=1) + '\n', text)

    def test_the_ship_entry_is_the_ship_5_entry_plus_the_two_switches(self):
        old = json.loads((fusion.MANIFESTS / 'S5.json').read_text(encoding='utf-8'))['ship'][0]
        new = json.loads((fusion.MANIFESTS / 'S6.json').read_text(encoding='utf-8'))['ship'][0]
        self.assertEqual({key: value for key, value in new['env'].items() if old['env'].get(key) != value},
                         dict(CHANGES, QWEN_FAST_OCTO_WAIVER='ship6-fx-262k-2026-10-21'))
        self.assertEqual([key for key in old['env'] if key not in new['env']], [])
        for key in ('from', 'memory_plan', 'bake_graft'):
            self.assertEqual(new[key], old[key], key)
        self.assertEqual(new['control'], 'traffic-s5')

    def test_the_w3_manifest_is_unchanged_and_still_wires_only_the_wc_smoke_rule(self):
        raw = json.loads((fusion.MANIFESTS / 'W3.json').read_text(encoding='utf-8'))
        self.assertEqual(raw['smoke_rules'], ['w3_wc_smoke'])


if __name__ == '__main__':
    unittest.main()
