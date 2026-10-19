"""tp4/octo-3: the two octo SHIP CANDIDATES (traffic profiles generated from fusion-wp/O3.json by make_fusion_profiles.py) and their gate packs.

  -fx-traffic-o3        the approved fusion ship profile + octo-T8 and its three levers (non-alternate) + the tile-aligned drafter window + the third block's memory plan, under its own owner
                        traffic waiver (PENDING), naming every octo lever;
  -fx-traffic-o3-oneq   that plus QWEN_FAST_SDPA_PF_ONEQ=1 (its image bakes the K64j-OQ graft, whose packed-any records have landed).

The approved ship profile and both earlier ship waivers stay byte-identical; nothing here is hand-edited."""

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

import c2_serving_job as job  # noqa: E402
import make_fusion_jobs as jobs_gen  # noqa: E402
import make_fusion_profiles as fusion  # noqa: E402
import make_octo3_profiles as twins  # noqa: E402
import serving_c2_contract as contract  # noqa: E402

REPO = HERE.parent.parent
PROFILES_PATH = HERE / 'qwen_c2_profiles.json'
PROFILES = json.loads(PROFILES_PATH.read_text(encoding='utf-8'))['profiles']
NS = fusion.NAMESPACE
FUSION_SHIP = NS + 'traffic'
FUSION_ONEQ = NS + 'traffic-oneq'
SHIP = NS + 'traffic-o3'
ONEQ = NS + 'traffic-o3-oneq'
PLAN = fusion.normalise(fusion.read_manifests())
SPECS = {item['name']: item for item in PLAN['ship']}
OCTO_FLAGS = {'QWEN_FAST_OCTO': 'live', 'QWEN_FAST_OCTO_MIN_LIVE': '6', 'QWEN_FAST_OCTO_DRAFT': '1', 'QWEN_FAST_OCTO_ATTN_BUNDLE': '4', 'QWEN_FAST_OCTO_GLUE8': '1'}
FOUR = {'QWEN_FAST_TP4_SDPA': 'multi', 'QWEN_FAST_TP4_CONV_GATES_SPREAD': '1', 'QWEN_FAST_PARKED_ENGINES': '1', 'QWEN_FAST_PARKED_DRAFTS': '1'}
PACKS = {SHIP: HERE / 'references' / 'fusion-jobs' / 'ship-traffic-o3', ONEQ: HERE / 'references' / 'fusion-jobs' / 'ship-traffic-o3-oneq'}
IMAGES = {SHIP: 'tp4-octo3-ship-1', ONEQ: 'tp4-octo3-oneq-ship-1'}
NAMES = ('B0-ship-build', 'G1-ship-gate', 'GX0-exact-control', 'GX1-exact-ship', 'G2-prefix-hit', 'SR-platform-replay')


def pack_values(pack, name):
    return job.parse_env((pack / (name + '.env')).read_text(encoding='utf-8'))


class ProfileTests(unittest.TestCase):
    def test_the_candidates_are_generated_traffic_profiles_and_no_gate_twin(self):
        self.assertEqual(twins.ship_names(), (SHIP, ONEQ))
        for name in (SHIP, ONEQ):
            self.assertIn(name, fusion.ship_names())
            self.assertNotIn(name, fusion.twin_names())
            self.assertNotIn(name, twins.twin_names())
        regenerated = fusion.generate_profiles(json.loads(PROFILES_PATH.read_text(encoding='utf-8')), PLAN)['profiles']
        for name in (SHIP, ONEQ):
            self.assertEqual(regenerated[name], PROFILES[name])

    def test_the_names_fit_the_64_characters_a_baked_default_may_have(self):
        for name in (SHIP, ONEQ):
            self.assertLessEqual(len(name), 64, name)

    def test_the_first_is_the_approved_ship_profile_plus_exactly_the_octo_env_the_tap_lever_and_the_memory_plan(self):
        base, ship = PROFILES[FUSION_SHIP], PROFILES[SHIP]
        added = {key: value for key, value in ship['env'].items() if base['env'].get(key) != value}
        self.assertEqual(added, dict(OCTO_FLAGS, QWEN_FAST_OCTO_WAIVER=SPECS['traffic-o3']['waiver']['id'], QWEN_FAST_TAP_WINDOW_ALIGNED='1', QWEN36_MAX_TOKENS_ALL_USERS='1055488'))
        self.assertEqual(sorted(set(base['env']) - set(ship['env'])), [])
        for key in ship:
            if key not in ('env', 'engine', 'description', contract.TRAFFIC_WAIVER):
                self.assertEqual(ship[key], base[key], key)
        engine, base_engine = copy.deepcopy(ship['engine']), copy.deepcopy(base['engine'])
        for item in (engine, base_engine):
            item.pop('num-gpu-blocks-override')
            item['additional-config']['tt'].pop('trace_region_size')
        self.assertEqual(engine, base_engine)
        self.assertEqual((ship['engine']['num-gpu-blocks-override'], ship['engine']['additional-config']['tt']['trace_region_size']), (16500, 640 << 20))

    def test_the_second_is_the_first_plus_exactly_oneq(self):
        first, second = PROFILES[SHIP], PROFILES[ONEQ]
        self.assertEqual({key: value for key, value in second['env'].items() if first['env'].get(key) != value},
                         {'QWEN_FAST_SDPA_PF_ONEQ': '1', contract.OCTO_WAIVER_ENV: SPECS['traffic-o3-oneq']['waiver']['id']})
        self.assertEqual(second['engine'], first['engine'])
        self.assertEqual(SPECS['traffic-o3-oneq']['bake_graft'], 'K64j-OQ')
        self.assertIsNone(SPECS['traffic-o3'].get('bake_graft'))

    def test_the_approved_fusion_ship_profiles_and_their_waivers_are_untouched(self):
        for name in (FUSION_SHIP, FUSION_ONEQ):
            profile = PROFILES[name]
            self.assertEqual(set(profile[contract.TRAFFIC_WAIVER]['levers']), set(FOUR), name)
            self.assertNotIn('QWEN_FAST_OCTO', profile['env'])
            self.assertEqual(profile['engine']['num-gpu-blocks-override'], 19968)
        self.assertTrue(PROFILES[FUSION_SHIP][contract.TRAFFIC_WAIVER]['decision'].startswith('APPROVED'))
        self.assertEqual(PROFILES[FUSION_SHIP][contract.TRAFFIC_WAIVER]['id'], 'w2-er-fx-262k-2026-10-11')
        self.assertEqual(PROFILES[FUSION_ONEQ][contract.TRAFFIC_WAIVER]['id'], 'oneq-fx-262k-2026-10-11')
        self.assertEqual(PROFILES[fusion.PARENT][contract.TRAFFIC_WAIVER]['id'], 'w2-er-262k-2026-10-10')

    def test_each_has_its_own_pending_waiver_naming_the_four_and_every_octo_lever(self):
        for name, spec in ((SHIP, SPECS['traffic-o3']), (ONEQ, SPECS['traffic-o3-oneq'])):
            waiver = PROFILES[name][contract.TRAFFIC_WAIVER]
            with self.subTest(profile=name):
                self.assertEqual(contract.traffic_waiver_problems(dict(PROFILES[name], name=name)), [])
                self.assertEqual(waiver['levers'], dict(FOUR, **OCTO_FLAGS))
                self.assertTrue(waiver['decision'].startswith(('PENDING', 'APPROVED')), waiver['decision'])
                self.assertEqual(contract.traffic_waiver_pending(dict(PROFILES[name], name=name)), waiver['decision'].startswith('PENDING'))
                self.assertEqual(waiver['decision'], spec['waiver']['decision'])
                self.assertEqual(waiver['id'], PROFILES[name]['env'][contract.OCTO_WAIVER_ENV])
                self.assertNotEqual(waiver['id'], PROFILES[FUSION_SHIP][contract.TRAFFIC_WAIVER]['id'])
                self.assertGreaterEqual(len(waiver['evidence']), 5)
        self.assertEqual(sorted(name for name, body in PROFILES.items() if contract.TRAFFIC_WAIVER in body), sorted(list(fusion.ship_names()) + [fusion.PARENT] + list(__import__('make_prefix_tier_profiles').ship_names())))

    def test_the_waiver_says_what_the_levers_are_outside_of_and_what_would_record_them_instead(self):
        text = PROFILES[SHIP][contract.TRAFFIC_WAIVER]['waives']
        for phrase in ('QWEN_C2_GATE_PROFILE', 'packed_any_evidence_tp4.json', 'G8B1', 'B > 1 entries', 'batched GDN', 'quarter-tile', 'QWEN_FAST_OCTO_WAIVER', 'Q1w', 'Q1a', 'Q1b', 'Q2',
                       'not built', 'QWEN_FAST_OCTO_MIN_LIVE', 'tau'):
            self.assertIn(phrase, text)
        evidence = ' '.join(PROFILES[SHIP][contract.TRAFFIC_WAIVER]['evidence'])
        for phrase in ('+46.6%', '38101791976', '24 of 24', 'did not complete', 'have not run'):
            self.assertIn(phrase, evidence)
        self.assertIn('K64j-OQ', ' '.join(PROFILES[ONEQ][contract.TRAFFIC_WAIVER]['evidence']))

    def test_no_gate_instrument_is_in_a_candidate(self):
        for name in (SHIP, ONEQ):
            env = PROFILES[name]['env']
            self.assertNotIn('QWEN_C2_GATE_PROFILE', env)
            self.assertNotIn('gate_only', PROFILES[name])
            for key in contract.OCTO_GATE_ONLY:
                self.assertNotIn(key, env, key)
            for key, value in env.items():
                self.assertFalse(key.endswith('_AUDIT') and value == '1' and key not in PROFILES[fusion.PARENT]['env'], key)

    def test_the_text_of_the_profiles_is_public_safe(self):
        for name in (SHIP, ONEQ):
            text = json.dumps([PROFILES[name]['description'], PROFILES[name][contract.TRAFFIC_WAIVER]])
            for word in fusion.SHIP_FORBIDDEN:
                self.assertNotIn(word, text, name)
            self.assertTrue(PROFILES[name]['description'].startswith('A SHIP CANDIDATE'))

    def test_the_candidates_pass_every_boot_check(self):
        from test_octo3_contract import boot_problems

        for name in (SHIP, ONEQ):
            self.assertEqual(boot_problems(dict(PROFILES[name], name=name)), [], name)


class ApprovalTests(unittest.TestCase):
    def copy_manifests(self):
        directory = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, str(directory), True)
        for path in fusion.MANIFESTS.glob('*.json'):
            shutil.copy(str(path), str(directory / path.name))
        return directory

    def test_approving_one_candidate_changes_its_decision_sentence_and_nothing_else(self):
        directory = self.copy_manifests()
        before = {path.name: path.read_text(encoding='utf-8') for path in directory.glob('*.json')}
        was = fusion.approve_ship('APPROVED by the owner 2026-10-11 (remote control): octo everywhere', directory, 'traffic-o3')
        self.assertEqual(was, SPECS['traffic-o3']['waiver']['decision'])
        after = {path.name: path.read_text(encoding='utf-8') for path in directory.glob('*.json')}
        self.assertEqual([name for name in after if after[name] != before[name]], ['O3.json'])
        old, new = json.loads(before['O3.json']), json.loads(after['O3.json'])
        self.assertTrue(new['ship'][0]['waiver']['decision'].startswith('APPROVED'))
        self.assertEqual(new['ship'][1], old['ship'][1], 'the other candidate is not decided')
        new['ship'][0]['waiver']['decision'] = old['ship'][0]['waiver']['decision']
        self.assertEqual(new, old)
        plan = fusion.normalise(fusion.read_manifests(directory))
        generated = fusion.generate_profiles(json.loads(PROFILES_PATH.read_text(encoding='utf-8')), plan)['profiles']
        self.assertFalse(contract.traffic_waiver_pending(dict(generated[SHIP], name=SHIP)))
        self.assertTrue(contract.traffic_waiver_pending(dict(generated[ONEQ], name=ONEQ)))
        self.assertEqual({key: value for key, value in generated[SHIP].items() if key != contract.TRAFFIC_WAIVER}, {key: value for key, value in PROFILES[SHIP].items() if key != contract.TRAFFIC_WAIVER})
        for name in (FUSION_SHIP, FUSION_ONEQ, fusion.PARENT):
            self.assertEqual(generated[name], PROFILES[name], name)

    def test_the_decision_is_ambiguous_without_a_name_now_that_there_are_four_candidates(self):
        with self.assertRaises(fusion.ManifestError):
            fusion.approve_ship('APPROVED by the owner: ok', fusion.MANIFESTS, None)


class PackTests(unittest.TestCase):
    def test_each_pack_is_what_the_generator_makes(self):
        wanted = jobs_gen.generate(PLAN)
        for name, pack in PACKS.items():
            found = {path.name: path.read_text(encoding='utf-8') for path in pack.glob('*') if path.is_file()}
            prefix = pack.name + '/'
            self.assertEqual(found, {key.split('/', 1)[1]: text for key, text in wanted.items() if key.startswith(prefix)}, name)
            self.assertEqual(sorted(key[:-4] for key in found if key.endswith('.env')), sorted(NAMES))

    def test_the_gates_name_the_candidate_and_the_exactness_control_is_the_fusion_ship_on_the_same_image(self):
        for name, pack in PACKS.items():
            with self.subTest(profile=name):
                self.assertEqual(pack_values(pack, 'B0-ship-build')['C2_BAKE_DEFAULT_PROFILE'], name)
                self.assertEqual(pack_values(pack, 'G1-ship-gate')['C2_PROFILE'], name)
                self.assertEqual(pack_values(pack, 'GX1-exact-ship')['C2_PROFILE'], name)
                self.assertEqual(pack_values(pack, 'GX0-exact-control')['C2_PROFILE'], FUSION_SHIP, 'the approved fusion ship profile is the control, on this pack\'s image')
                self.assertEqual(pack_values(pack, 'G2-prefix-hit')['C2_PREFIX_PROFILE'], name)
                self.assertEqual(pack_values(pack, 'SR-platform-replay')['C2_REPLAY_PROFILE'], name)
                for job_name in NAMES:
                    self.assertEqual(pack_values(pack, job_name)['C2_IMAGE_TAG'], IMAGES[name], job_name)
        self.assertEqual(pack_values(PACKS[ONEQ], 'B0-ship-build').get('C2_BAKE_GRAFT'), 'K64j-OQ')
        self.assertNotIn('C2_BAKE_GRAFT', pack_values(PACKS[SHIP], 'B0-ship-build'))

    def test_every_gate_parses_and_the_build_is_refused_exactly_while_the_decision_is_pending(self):
        for name, pack in PACKS.items():
            pending = PROFILES[name][contract.TRAFFIC_WAIVER]['decision'].startswith('PENDING')
            for job_name in NAMES:
                with self.subTest(profile=name, job=job_name):
                    result = subprocess.run([sys.executable, '-B', '-s', str(HERE / 'c2_serving_job.py'), str(pack / (job_name + '.env')), str(PROFILES_PATH)],
                                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=str(REPO))
                    text = (result.stdout + result.stderr).decode('utf-8', 'replace')
                    if job_name == 'B0-ship-build' and pending:
                        self.assertIn('not APPROVED', text)
                    else:
                        self.assertEqual(result.returncode, 0, text)
                        self.assertNotIn('refused', text)

    def test_the_octo_read_rules_are_in_the_gates(self):
        for pack in PACKS.values():
            for job_name in ('G1-ship-gate', 'GX1-exact-ship', 'SR-platform-replay'):
                text = (pack / (job_name + '.env')).read_text(encoding='utf-8').replace('\n# ', ' ')
                for phrase in ('OCTO READ', 'owner traffic waiver <id>', 'UNQUALIFIED', 'octo 8-row served path', 'concurrent8_steady is the test that supplies the octo rounds'):
                    self.assertIn(phrase, text, '%s %s' % (pack.name, job_name))

    def test_the_orders_say_how_to_decide_and_which_candidate(self):
        for name, pack in PACKS.items():
            text = (pack / 'ORDER.txt').read_text(encoding='utf-8')
            self.assertIn('--ship %s' % name[len(NS):], text)
            self.assertIn(PROFILES[name][contract.TRAFFIC_WAIVER]['id'], text)
            rows = [line.split() for line in text.splitlines() if line and not line.startswith('#')]
            self.assertEqual([row[0] for row in rows], list(NAMES))
            self.assertEqual({row[2] for row in rows}, {IMAGES[name]})

    def test_the_candidates_carry_the_gates_tests_the_octo_shape_needs_to_run(self):
        tests = pack_values(PACKS[SHIP], 'G1-ship-gate')['C2_SMOKE_TESTS'].split(',')
        self.assertIn('concurrent8_steady', tests)
        self.assertTrue(set(jobs_gen.G1_TESTS.split(',')) <= set(tests))


if __name__ == '__main__':
    unittest.main()
