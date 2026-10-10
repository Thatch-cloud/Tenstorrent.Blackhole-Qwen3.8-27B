"""The op-fusion SHIP candidate: the traffic profile `...-w2-er-fx-traffic`, its pending owner waiver and the ship gate pack (references/fusion-jobs/ship).

Held here, on the CPU (nothing ran on a card for the ship profile itself; the combined arm it copies did: docs/tp4-fusion.md):

  profile   the production profile plus EXACTLY the env of the combined gate arm -fx-all (no audit flag, no telemetry key), a traffic profile (not gate_only), every other field the
            production profile's; its own owner_traffic_waiver carries the four waivable levers of the production waiver and a decision that is PENDING;
  contract  a PENDING profile BOOTS (every boot check of serving_c2_contract is clean: gates and smokes select it by name) and is refused only by the bake (c2_serving_job.read_bake,
            B0), which accepts it the moment the decision starts APPROVED; the decision is the one field that flips (make_fusion_profiles.approve_ship edits the manifest, the source of
            the profile, and nothing else);
  pack      B0 bakes the ship profile (rc 1 while pending), G1 / GX0 / GX1 run the smoke with C2_TELEMETRY=1 and G2 / SR / B0 carry none (their actions have no sidecar), GX0 and GX1 run the
            same prompts on one image (the production profile against the ship profile), SR boots the baked default on the thin layer;
  reading   parked_compare judges the ship arm's engines with the lever band.
"""

import copy
import json
import os
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

import c2_serving_job as job  # noqa: E402
import c2_smoke_check  # noqa: E402
import make_fusion_jobs as jobs_gen  # noqa: E402
import make_fusion_profiles as fusion  # noqa: E402
import parked_compare  # noqa: E402
import parked_judge  # noqa: E402
import serving_c2_contract as contract  # noqa: E402

REPO = HERE.parent.parent
PROFILES_PATH = HERE / 'qwen_c2_profiles.json'
PROFILES = json.loads(PROFILES_PATH.read_text(encoding='utf-8'))['profiles']
PARENT = fusion.PARENT
SHIP = fusion.NAMESPACE + 'traffic'
COMBINED = fusion.NAMESPACE + 'all'
COMBINED_AUDIT = fusion.NAMESPACE + 'all-audit'
PACK = HERE / 'references' / 'fusion-jobs' / 'ship'
PLAN = fusion.normalise(fusion.read_manifests())
SPEC = next(item for item in PLAN['ship'] if item['name'] == 'traffic')


def env_of(name):
    return dict(PROFILES[name]['env'])


def pack_values(name):
    values = {}
    for line in (PACK / (name + '.env')).read_text(encoding='utf-8').splitlines():
        if line and not line.startswith('#'):
            key, _sep, value = line.partition('=')
            values[key] = value
    return values


class ProfileTests(unittest.TestCase):
    def test_it_is_the_production_profile_plus_exactly_the_combined_arms_env(self):
        parent, ship, combined = PROFILES[PARENT], PROFILES[SHIP], PROFILES[COMBINED]
        additions = {key: value for key, value in ship['env'].items() if parent['env'].get(key) != value}
        self.assertEqual(additions, {key: value for key, value in combined['env'].items() if parent['env'].get(key) != value})
        self.assertEqual({key: parent['env'][key] for key in parent['env'] if key not in additions}, {key: ship['env'][key] for key in ship['env'] if key not in additions})
        for key in ('engine', 'snapshots', 'mesh_device', 'mesh_graph_descriptor', 'eos_ids', 'max_prompt_tokens', 'min_answer_tokens', 'default_max_tokens', 'drafter_headroom_tokens',
                    'parser_rechunk'):
            self.assertEqual(ship[key], parent[key], key)
        self.assertEqual(sorted(set(ship) - set(parent)), [])

    def test_it_is_a_traffic_profile_without_an_audit_a_telemetry_key_or_a_gate_marker(self):
        ship = PROFILES[SHIP]
        self.assertNotIn('gate_only', ship)
        self.assertTrue(ship['description'].startswith('The SHIP CANDIDATE'))
        for name in ship['env']:
            self.assertFalse(name.endswith('_AUDIT') and ship['env'][name] == '1' and name not in PROFILES[PARENT]['env'], name)
            self.assertNotIn('TELEMETRY', name)
        for flag in c2_smoke_check.fusion_arm_flags(ship['env'])[0]:
            self.assertEqual(ship['env'][flag], PROFILES[COMBINED]['env'][flag])
        self.assertEqual(c2_smoke_check.fusion_arm_flags(ship['env'])[1], [])
        self.assertEqual(c2_smoke_check.fusion_arm_flags(PROFILES[COMBINED_AUDIT]['env'])[0], c2_smoke_check.fusion_arm_flags(ship['env'])[0])

    def test_it_is_a_generated_traffic_profile_and_no_gate_twin(self):
        self.assertIn(SHIP, fusion.ship_names())
        self.assertNotIn(SHIP, fusion.twin_names())
        self.assertEqual(fusion.generate_profiles(json.loads(PROFILES_PATH.read_text(encoding='utf-8')), PLAN)['profiles'][SHIP], PROFILES[SHIP])

    def test_the_waiver_is_well_formed_names_the_four_levers_and_is_pending_or_approved(self):
        waiver = PROFILES[SHIP][contract.TRAFFIC_WAIVER]
        self.assertEqual(contract.traffic_waiver_problems(dict(PROFILES[SHIP], name=SHIP)), [])
        self.assertEqual(waiver['levers'], contract.WAIVABLE_LEVERS)
        self.assertEqual(waiver['levers'], PROFILES[PARENT][contract.TRAFFIC_WAIVER]['levers'])
        self.assertNotEqual(waiver['id'], PROFILES[PARENT][contract.TRAFFIC_WAIVER]['id'])
        self.assertTrue(waiver['decision'].startswith(('PENDING', 'APPROVED')), waiver['decision'])
        self.assertEqual(contract.traffic_waiver_pending(dict(PROFILES[SHIP], name=SHIP)), waiver['decision'].startswith('PENDING'))
        self.assertFalse(contract.traffic_waiver_pending(dict(PROFILES[PARENT], name=PARENT)), 'the production profile stays approved')
        self.assertTrue(PROFILES[PARENT][contract.TRAFFIC_WAIVER]['decision'].startswith('APPROVED'))
        self.assertEqual(sorted(name for name, body in PROFILES.items() if contract.TRAFFIC_WAIVER in body), sorted([fusion.NAMESPACE + 'traffic', PARENT]))
        self.assertGreaterEqual(len(waiver['evidence']), 5)
        self.assertEqual(waiver['decision'], SPEC['waiver']['decision'])

    def test_the_text_of_the_profile_is_public_safe(self):
        text = json.dumps([PROFILES[SHIP]['description'], PROFILES[SHIP][contract.TRAFFIC_WAIVER]])
        for word in fusion.SHIP_FORBIDDEN:
            self.assertNotIn(word, text)


class ContractTests(unittest.TestCase):
    def problems(self, profile):
        environ = {'QWEN_C2_SERVING': '1'}
        contract.apply_environment(profile, environ)
        found = []
        for check in (lambda: contract.gate_problems(profile, environ) + contract.waiver_problems(profile, environ) + contract.traffic_waiver_problems(profile),
                      lambda: contract.prefix_reuse_problems(profile), lambda: contract.levern_problems(profile), lambda: contract.multi_problems(profile),
                      lambda: contract.w2_problems(profile, environ), lambda: contract.upload_problems(profile), lambda: contract.drafter_problems(profile),
                      lambda: contract.parked_problems(profile, environ), lambda: contract.mesh_problems(profile)):
            found += check()
        return found

    def test_a_pending_profile_boots_in_a_gate_or_a_smoke_with_every_boot_check_clean(self):
        self.assertEqual(self.problems(dict(PROFILES[SHIP], name=SHIP)), [])

    def test_the_boot_checks_refuse_what_the_waiver_does_not_name(self):
        profile = copy.deepcopy(PROFILES[SHIP])
        profile['name'] = SHIP
        del profile[contract.TRAFFIC_WAIVER]['levers']['QWEN_FAST_TP4_SDPA']
        self.assertTrue(self.problems(profile), 'a lever the profile sets and the waiver does not name stays refused')
        profile = copy.deepcopy(PROFILES[SHIP])
        profile['name'] = SHIP
        profile['env']['QWEN_FAST_DRAFT_REDUCE_AUDIT'] = '1'
        profile['env']['QWEN_FAST_DEVICE_ZEROS_AUDIT'] = '1'
        self.assertTrue(self.problems(profile), 'an audit on a traffic profile is refused')

    def with_profiles(self, decision):
        root = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, str(root), True)
        (root / 'scripts' / 'ci').mkdir(parents=True)
        data = json.loads(PROFILES_PATH.read_text(encoding='utf-8'))
        data['profiles'][SHIP][contract.TRAFFIC_WAIVER]['decision'] = decision
        (root / 'scripts' / 'ci' / 'qwen_c2_profiles.json').write_text(json.dumps(data, indent=2) + '\n', encoding='utf-8')
        return root

    def test_the_bake_refuses_a_pending_decision_and_accepts_an_approved_one(self):
        with self.assertRaises(job.JobError) as caught:
            job.read_bake({'C2_BAKE_DEFAULT_PROFILE': SHIP}, ['build'], 'quad', root=self.with_profiles('PENDING: requested on 2026-10-11 for the owner\'s decision'))
        self.assertIn('not APPROVED', str(caught.exception))
        self.assertEqual(job.read_bake({'C2_BAKE_DEFAULT_PROFILE': SHIP}, ['build'], 'quad', root=self.with_profiles('APPROVED by the owner 2026-10-11: ship it')), SHIP)

    def test_the_production_profile_still_bakes(self):
        self.assertEqual(job.read_bake({'C2_BAKE_DEFAULT_PROFILE': PARENT}, ['build'], 'quad'), PARENT)


class ApprovalTests(unittest.TestCase):
    def copy_manifests(self):
        directory = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, str(directory), True)
        for path in fusion.MANIFESTS.glob('*.json'):
            shutil.copy(str(path), str(directory / path.name))
        return directory

    def test_approve_ship_changes_the_decision_sentence_and_nothing_else(self):
        directory = self.copy_manifests()
        before = {path.name: path.read_text(encoding='utf-8') for path in directory.glob('*.json')}
        was = fusion.approve_ship('APPROVED by the owner 2026-10-11 (remote control): ship the op-fusion levers', directory)
        self.assertEqual(was, SPEC['waiver']['decision'])
        after = {path.name: path.read_text(encoding='utf-8') for path in directory.glob('*.json')}
        self.assertEqual([name for name in after if after[name] != before[name]], ['WP0.json'])
        old, new = json.loads(before['WP0.json']), json.loads(after['WP0.json'])
        self.assertTrue(new['ship'][0]['waiver']['decision'].startswith('APPROVED'))
        new['ship'][0]['waiver']['decision'] = old['ship'][0]['waiver']['decision']
        self.assertEqual(new, old)
        plan = fusion.normalise(fusion.read_manifests(directory))
        profile = fusion.generate_profiles(json.loads(PROFILES_PATH.read_text(encoding='utf-8')), plan)['profiles'][SHIP]
        self.assertTrue(profile[contract.TRAFFIC_WAIVER]['decision'].startswith('APPROVED'))
        self.assertEqual({key: value for key, value in profile.items() if key != contract.TRAFFIC_WAIVER}, {key: value for key, value in PROFILES[SHIP].items() if key != contract.TRAFFIC_WAIVER})
        self.assertEqual({key: value for key, value in profile[contract.TRAFFIC_WAIVER].items() if key != 'decision'},
                         {key: value for key, value in PROFILES[SHIP][contract.TRAFFIC_WAIVER].items() if key != 'decision'})
        self.assertFalse(contract.traffic_waiver_pending(dict(profile, name=SHIP)))

    def test_it_refuses_a_sentence_that_is_not_a_decision_or_names_a_host_a_digest_or_a_path(self):
        directory = self.copy_manifests()
        for sentence in ('yes', 'The owner approved', '', 'APPROVED see ' + 'sha256' + ':abcd', 'APPROVED by ' + 'me' + '@' + 'there', 'APPROVED in /' + 'home' + '/x', 'APPROVED at 192' + '.168.1.1'):
            with self.subTest(sentence=sentence), self.assertRaises(fusion.ManifestError):
                fusion.approve_ship(sentence, directory)
        self.assertEqual({path.name: path.read_text(encoding='utf-8') for path in directory.glob('*.json')},
                         {path.name: path.read_text(encoding='utf-8') for path in fusion.MANIFESTS.glob('*.json')})

    def test_it_reverts_with_pending_and_refuses_a_manifest_it_cannot_rewrite_exactly(self):
        directory = self.copy_manifests()
        fusion.approve_ship('APPROVED by the owner 2026-10-11: ok', directory)
        fusion.approve_ship(SPEC['waiver']['decision'], directory)
        self.assertEqual((directory / 'WP0.json').read_text(encoding='utf-8'), (fusion.MANIFESTS / 'WP0.json').read_text(encoding='utf-8'))
        text = (directory / 'WP0.json').read_text(encoding='utf-8')
        (directory / 'WP0.json').write_text(text.replace('\n ', '\n   ', 1), encoding='utf-8')
        with self.assertRaises(fusion.ManifestError):
            fusion.approve_ship('APPROVED by the owner: ok', directory)

    def test_the_command_line_flips_the_decision_in_a_scratch_copy_and_regenerates_the_profile(self):
        directory = self.copy_manifests()
        checked_in = PROFILES_PATH.read_text(encoding='utf-8')
        profiles = Path(tempfile.mkdtemp()) / 'profiles.json'
        self.addCleanup(shutil.rmtree, str(profiles.parent), True)
        shutil.copy(str(PROFILES_PATH), str(profiles))
        with mock.patch.object(fusion, 'PROFILES', profiles), mock.patch.dict(fusion.PATHS, {'profiles': profiles}), \
                mock.patch.object(fusion, 'read_texts', lambda: dict(profiles=profiles.read_text(encoding='utf-8'), smoke=fusion.SMOKE.read_text(encoding='utf-8'),
                                                                     overlay=fusion.OVERLAY.read_text(encoding='utf-8'), cpu=fusion.CPU_WORKFLOW.read_text(encoding='utf-8'),
                                                                     addresses=fusion.ADDRESSES.read_text(encoding='utf-8'))), \
                mock.patch.dict(fusion.PATHS, {'smoke': Path(tempfile.mkdtemp()) / 'smoke.py', 'overlay': Path(tempfile.mkdtemp()) / 'overlay.txt',
                                               'cpu': Path(tempfile.mkdtemp()) / 'cpu.yml', 'addresses': Path(tempfile.mkdtemp()) / 'addresses.py'}):
            for name in ('smoke', 'overlay', 'cpu', 'addresses'):
                self.addCleanup(shutil.rmtree, str(fusion.PATHS[name].parent), True)
            self.assertEqual(fusion.main(['--approve-ship', 'APPROVED by the owner 2026-10-11: ok', '--manifests', str(directory)]), 0)
        data = json.loads(profiles.read_text(encoding='utf-8'))
        self.assertTrue(data['profiles'][SHIP][contract.TRAFFIC_WAIVER]['decision'].startswith('APPROVED'))
        self.assertEqual(PROFILES_PATH.read_text(encoding='utf-8'), checked_in, 'the checked-in profile was not touched')


class PackTests(unittest.TestCase):
    NAMES = ('B0-ship-build', 'G1-ship-gate', 'GX0-exact-control', 'GX1-exact-ship', 'G2-prefix-hit', 'SR-platform-replay')

    def test_the_pack_is_what_the_generator_makes(self):
        wanted = jobs_gen.generate(PLAN)
        found = {path.name: path.read_text(encoding='utf-8') for path in PACK.glob('*') if path.is_file()}
        self.assertEqual(found, {name.split('/', 1)[1]: text for name, text in wanted.items() if name.startswith('ship/')})
        self.assertEqual(sorted(name[:-4] for name in found if name.endswith('.env')), sorted(self.NAMES))

    def run_job(self, name):
        return subprocess.run([sys.executable, '-s', str(HERE / 'c2_serving_job.py'), str(PACK / (name + '.env')), str(PROFILES_PATH)], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                              cwd=str(REPO))

    def test_every_gate_parses_and_the_build_is_refused_exactly_while_the_decision_is_pending(self):
        pending = PROFILES[SHIP][contract.TRAFFIC_WAIVER]['decision'].startswith('PENDING')
        for name in self.NAMES:
            with self.subTest(name=name):
                result = self.run_job(name)
                if name == 'B0-ship-build' and pending:
                    self.assertEqual(result.returncode, 1)
                    self.assertIn('not APPROVED', (result.stdout + result.stderr).decode('utf-8', 'replace'))
                else:
                    self.assertEqual(result.returncode, 0, result.stderr.decode('utf-8', 'replace'))

    def test_the_build_bakes_the_ship_profile_and_the_gates_name_it(self):
        self.assertEqual(pack_values('B0-ship-build')['C2_BAKE_DEFAULT_PROFILE'], SHIP)
        self.assertEqual(pack_values('B0-ship-build')['C2_ACTIONS'], 'build')
        self.assertEqual(pack_values('G1-ship-gate')['C2_PROFILE'], SHIP)
        self.assertEqual(pack_values('GX1-exact-ship')['C2_PROFILE'], SHIP)
        self.assertEqual(pack_values('G2-prefix-hit')['C2_PREFIX_PROFILE'], SHIP)
        self.assertEqual(pack_values('SR-platform-replay')['C2_REPLAY_PROFILE'], SHIP)
        self.assertEqual(pack_values('SR-platform-replay')['C2_PLATFORM_IMAGE'], 'local:thin-layer')
        for name in self.NAMES:
            self.assertEqual(pack_values(name)['C2_IMAGE_TAG'], jobs_gen.SHIP_IMAGE, name)

    def test_the_telemetry_sidecar_runs_where_the_action_supports_it_and_only_there(self):
        for name in ('G1-ship-gate', 'GX0-exact-control', 'GX1-exact-ship'):
            self.assertEqual(pack_values(name).get('C2_TELEMETRY'), '1', name)
        for name in ('B0-ship-build', 'G2-prefix-hit', 'SR-platform-replay'):
            self.assertNotIn('C2_TELEMETRY', pack_values(name), name)
        # the job reader is why: it refuses the sidecar on an action with no smoke or gate step
        values = dict(pack_values('G2-prefix-hit'), C2_TELEMETRY='1')
        with self.assertRaises(job.JobError):
            job.read_job(values, job.profile_names(str(PROFILES_PATH)), envs=job.profile_envs(str(PROFILES_PATH)))

    def test_the_exactness_pair_is_the_same_prompts_on_one_image_for_the_production_profile_and_the_ship_profile(self):
        control, ship = pack_values('GX0-exact-control'), pack_values('GX1-exact-ship')
        self.assertEqual(control['C2_PROFILE'], PARENT)
        self.assertEqual(ship['C2_PROFILE'], SHIP)
        for key in ('C2_SMOKE_TESTS', 'C2_IMAGE_TAG', 'C2_ACTIONS', 'C2_CARDS', 'C2_BOX_MINUTES'):
            self.assertEqual(control[key], ship[key], key)
        self.assertEqual(ship['C2_SMOKE_TESTS'], jobs_gen.GX_TESTS)
        self.assertTrue(set(('parked_equal', 'levern_equal', 'levern_equal_busy')) <= set(ship['C2_SMOKE_TESTS'].split(',')))
        self.assertIn('C2_SMOKE_PARTIAL', ship)

    def test_the_gate_is_the_eight_seat_262k_gate_with_the_stall_shape_and_the_engine_reuse_tests(self):
        tests = pack_values('G1-ship-gate')['C2_SMOKE_TESTS'].split(',')
        for name in ('concurrent8_steady', 'concurrent8_skew', 'stall8_cold262k', 'parked_abort_reuse', 'parked_turns', 'levern_seed_stops', 'concurrent8_drain'):
            self.assertIn(name, tests)
        reference = {}
        for line in (REPO / 'scripts' / 'ci' / 'references' / 'tp4-ship-ln-w2-er-jobs' / 'G1-ship-gate.env').read_text(encoding='utf-8').splitlines():
            if line.startswith('C2_SMOKE_TESTS='):
                reference = set(line.split('=', 1)[1].split(','))
        self.assertTrue(reference <= set(tests), 'every test the w2-er G1 ran')

    def test_the_order_lists_the_six_jobs_in_the_stop_order_the_readme_says(self):
        rows = [line.split() for line in (PACK / 'ORDER.txt').read_text(encoding='utf-8').splitlines() if line and not line.startswith('#')]
        self.assertEqual([row[0] for row in rows], list(self.NAMES))
        self.assertEqual({row[0]: row[1] for row in rows}, {'B0-ship-build': 'stop', 'G1-ship-gate': 'stop', 'GX0-exact-control': 'stop', 'GX1-exact-ship': 'stop',
                                                              'G2-prefix-hit': 'soft', 'SR-platform-replay': 'stop'})
        text = (PACK / 'ORDER.txt').read_text(encoding='utf-8')
        for word in ('--approve-ship', 'owner_traffic_waiver.decision', 'PENDING', 'BOOTS', '# NEEDS SR <- G1 GX1'):
            self.assertIn(word, text)

    def test_the_replay_header_says_what_it_needs_without_naming_an_image(self):
        text = (PACK / 'SR-platform-replay.env').read_text(encoding='utf-8')
        for word in ('THIN LAYER', 'source image', 'tracked record', 'IMAGE DEFAULT'):
            self.assertIn(word, text)
        for template in self.NAMES:
            body = (PACK / (template + '.env')).read_text(encoding='utf-8')
            for word in fusion.SHIP_FORBIDDEN:
                self.assertNotIn(word, body, (template, word))

    def test_the_main_order_points_at_the_ship_pack_and_the_pack_is_no_package_folder(self):
        text = (HERE / 'references' / 'fusion-jobs' / 'ORDER.txt').read_text(encoding='utf-8')
        self.assertIn('THE SHIP PACK (references/fusion-jobs/ship', text)
        self.assertNotIn('fusion-jobs/ship,', text.split('THE PACKAGES')[1].split('\n')[0])


class ReadingTests(unittest.TestCase):
    def test_parked_compare_judges_the_ship_arm_with_the_lever_band_and_the_control_with_the_production_one(self):
        calls = []

        def judge(env, text, smoke=None, levers=(), lever_audits=()):
            calls.append((sorted(levers), sorted(lever_audits)))
            return [], {}

        with mock.patch.object(parked_judge, 'judge', judge):
            parked_compare.compare('', '', 'control container', 'parked container', 'exact', env_of(SHIP))
            parked_compare.compare('', '', 'control container', 'parked container', 'exact', env_of(PARENT))
            parked_compare.compare('', '', 'control container', 'parked container', 'exact', None)
        self.assertEqual(calls[0], (c2_smoke_check.fusion_arm_flags(env_of(SHIP))[0], []))
        self.assertTrue(calls[0][0])
        self.assertEqual(calls[1], ([], []))
        self.assertEqual(calls[2], ([], []))

    def test_the_measured_engine_of_the_shipped_set_is_inside_the_lever_band_and_outside_the_production_one(self):
        low, high = parked_judge.engine_band(*c2_smoke_check.fusion_arm_flags(env_of(SHIP)))
        self.assertTrue(low <= 0.418 <= high)
        self.assertEqual(parked_judge.engine_band(), parked_judge.ENGINE_GB)
        self.assertFalse(parked_judge.ENGINE_GB[0] <= 0.418)
        self.assertEqual(parked_judge.ENGINE_GB_LEVERS, (0.41, 0.51))


if __name__ == '__main__':
    unittest.main()
