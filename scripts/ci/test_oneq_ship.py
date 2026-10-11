"""The oneq ship candidate (tp4/oneq-ship): the traffic profile twin, its gate pack, the RQ1/RQ2 re-record of the packed-any evidence on the K64j-OQ binary, and how the
admission takes the records once they land.

  - ...-fx-traffic-oneq is GENERATED (make_fusion_profiles.py, the ship entry of WPP.json) = the fusion ship profile plus QWEN_FAST_SDPA_PF_ONEQ=1 and nothing else, with its own
    owner waiver, PENDING; approve_ship decides one ship entry by name.
  - references/fusion-jobs/ship-traffic-oneq is its gate pack (B0 bakes the K64j-OQ graft; GX0's control is the fusion ship profile).
  - references/fusion-jobs/WPP/RQ0-RQ2 are the card-M jobs that make the K64j-OQ records; record_packed_any_evidence_tp4.py --binary k64j-oq writes them as FILES OF THEIR OWN
    (K64j's stay and keep qualifying K64j); packed_any_admission.own_record makes a K64j-OQ process read its own record, strictly, once its pin is set.
  - What lands is exactly: the two record files and their two pins (None until then). Every test here holds both states: the committed one (no K64j-OQ record yet, or the
    recorded one) and a simulated landing in a scratch directory.

Run from scripts/ci:  python -B -m unittest test_oneq_ship
"""

import copy
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import c2_serving_job as job  # noqa: E402
import make_fusion_jobs as jobs_gen  # noqa: E402
import make_fusion_profiles as fusion  # noqa: E402
import packed_any_admission as admission  # noqa: E402
import record_packed_any_evidence_tp4 as recorder  # noqa: E402
import serving_c2_contract as contract  # noqa: E402
from test_packed_any_admission import Lines, M3  # noqa: E402
import test_record_packed_any_evidence_tp4 as sample  # noqa: E402

PROFILES_PATH = HERE / 'qwen_c2_profiles.json'
PROFILES = json.loads(PROFILES_PATH.read_text(encoding='utf-8'))['profiles']
PLAN = fusion.normalise(fusion.read_manifests())
SHIP = fusion.NAMESPACE + 'traffic'
ONEQ_SHIP = fusion.NAMESPACE + 'traffic-oneq'
SPEC = next(item for item in PLAN['ship'] if item['name'] == 'traffic-oneq')
PACK = HERE / 'references' / 'fusion-jobs' / 'ship-traffic-oneq'
WPP = HERE / 'references' / 'fusion-jobs' / 'WPP'
K64J = admission.K64J_TTNNCPP_SHA256
OQ = admission.K64J_OQ_TTNNCPP_SHA256
ENV = admission.RUNTIME_BINARY_ENV
GATE = {admission.GATE_ENV: '1'}
FLAG = 'QWEN_FAST_SDPA_PF_ONEQ'
LANDED = admission.EVIDENCE_TP4_K64J_OQ_SHA256 is not None      # read at import, before any test patches the pins: have the K64j-OQ records landed in this checkout


def env_values(path):
    values = {}
    for line in Path(path).read_text(encoding='utf-8').splitlines():
        if line and not line.startswith('#'):
            key, _sep, value = line.partition('=')
            values[key] = value
    return values


def run_job(path, profiles=PROFILES_PATH):
    return subprocess.run([sys.executable, '-s', str(HERE / 'c2_serving_job.py'), str(path), str(profiles)], stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=str(REPO))


class ProfileTests(unittest.TestCase):
    def test_it_is_the_fusion_ship_profile_plus_the_oneq_flag_and_nothing_else(self):
        ship, oneq = PROFILES[SHIP], PROFILES[ONEQ_SHIP]
        self.assertEqual({key: value for key, value in oneq['env'].items() if ship['env'].get(key) != value}, {FLAG: '1'})
        self.assertEqual(sorted(set(ship['env']) - set(oneq['env'])), [])
        self.assertNotIn(FLAG, ship['env'])
        for key in ship:
            if key not in ('env', 'description', contract.TRAFFIC_WAIVER):
                self.assertEqual(oneq[key], ship[key], key)
        self.assertEqual(sorted(set(oneq) - set(ship)), [])

    def test_it_is_a_traffic_profile_without_audit_or_gate_marker(self):
        oneq = PROFILES[ONEQ_SHIP]
        self.assertNotIn('gate_only', oneq)
        self.assertTrue(oneq['description'].startswith('The SHIP CANDIDATE'))
        self.assertIn('K64j-OQ', oneq['description'])
        self.assertIn('QWEN_FAST_SDPA_PF_ONEQ=1', oneq['description'])
        self.assertNotIn('QWEN_FAST_SDPA_PF_ONEQ_AUDIT', oneq['env'])

    def test_it_is_generated_not_hand_edited(self):
        self.assertIn(ONEQ_SHIP, fusion.ship_names())
        self.assertNotIn(ONEQ_SHIP, fusion.twin_names())
        self.assertEqual(fusion.generate_profiles(json.loads(PROFILES_PATH.read_text(encoding='utf-8')), PLAN)['profiles'][ONEQ_SHIP], PROFILES[ONEQ_SHIP])

    def test_it_has_its_own_waiver_and_the_decision_is_pending_until_the_owner_decides(self):
        waiver = PROFILES[ONEQ_SHIP][contract.TRAFFIC_WAIVER]
        self.assertEqual(contract.traffic_waiver_problems(dict(PROFILES[ONEQ_SHIP], name=ONEQ_SHIP)), [])
        self.assertNotEqual(waiver['id'], PROFILES[SHIP][contract.TRAFFIC_WAIVER]['id'])
        self.assertEqual(waiver['levers'], PROFILES[SHIP][contract.TRAFFIC_WAIVER]['levers'])
        self.assertTrue(waiver['decision'].startswith('PENDING'), waiver['decision'])
        self.assertTrue(contract.traffic_waiver_pending(dict(PROFILES[ONEQ_SHIP], name=ONEQ_SHIP)))
        self.assertEqual(waiver['decision'], SPEC['waiver']['decision'])
        self.assertGreaterEqual(len(waiver['evidence']), 3)
        self.assertIn('inherits in full', waiver['waives'])
        self.assertEqual(sorted(name for name, body in PROFILES.items() if contract.TRAFFIC_WAIVER in body), sorted([SHIP, ONEQ_SHIP, fusion.PARENT]))
        text = json.dumps([PROFILES[ONEQ_SHIP]['description'], waiver])
        for word in fusion.SHIP_FORBIDDEN:
            self.assertNotIn(word, text)

    def test_the_bake_refuses_it_while_pending_and_takes_it_once_approved(self):
        with self.assertRaises(job.JobError) as caught:
            job.read_bake({'C2_BAKE_DEFAULT_PROFILE': ONEQ_SHIP}, ['build'], 'quad')
        self.assertIn('not APPROVED', str(caught.exception))
        root = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, str(root), True)
        (root / 'scripts' / 'ci').mkdir(parents=True)
        data = json.loads(PROFILES_PATH.read_text(encoding='utf-8'))
        data['profiles'][ONEQ_SHIP][contract.TRAFFIC_WAIVER]['decision'] = 'APPROVED by the owner 2026-10-11: ship it'
        (root / 'scripts' / 'ci' / 'qwen_c2_profiles.json').write_text(json.dumps(data, indent=2) + '\n', encoding='utf-8')
        self.assertEqual(job.read_bake({'C2_BAKE_DEFAULT_PROFILE': ONEQ_SHIP}, ['build'], 'quad', root=str(root)), ONEQ_SHIP)

    def test_the_manifest_entry_asks_for_what_the_profile_and_the_pack_carry(self):
        self.assertEqual((SPEC['source'], SPEC['env'], SPEC['bake_graft'], SPEC['control']), ('all', {FLAG: '1'}, 'K64j-OQ', 'traffic'))
        self.assertEqual(SPEC['file'], 'WPP.json')


class ApprovalTests(unittest.TestCase):
    def copy_manifests(self):
        directory = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, str(directory), True)
        for path in fusion.MANIFESTS.glob('*.json'):
            shutil.copy(str(path), str(directory / path.name))
        return directory

    def test_with_two_ship_entries_the_command_must_name_one(self):
        directory = self.copy_manifests()
        with self.assertRaises(fusion.ManifestError) as caught:
            fusion.approve_ship('APPROVED by the owner: ok', directory)
        self.assertIn('name the one to decide with --ship', str(caught.exception))
        self.assertIn('traffic', str(caught.exception))
        self.assertIn('traffic-oneq', str(caught.exception))
        with self.assertRaises(fusion.ManifestError) as caught:
            fusion.approve_ship('APPROVED by the owner: ok', directory, 'nope')
        self.assertIn("no ship entry named 'nope'", str(caught.exception))

    def test_it_decides_the_named_entry_and_changes_no_other_byte(self):
        directory = self.copy_manifests()
        before = {path.name: path.read_text(encoding='utf-8') for path in directory.glob('*.json')}
        was = fusion.approve_ship('APPROVED by the owner 2026-10-12 (remote control): ship the oneq candidate', directory, 'traffic-oneq')
        self.assertEqual(was, SPEC['waiver']['decision'])
        after = {path.name: path.read_text(encoding='utf-8') for path in directory.glob('*.json')}
        self.assertEqual([name for name in after if after[name] != before[name]], ['WPP.json'])
        old, new = json.loads(before['WPP.json']), json.loads(after['WPP.json'])
        self.assertTrue(new['ship'][0]['waiver']['decision'].startswith('APPROVED'))
        new['ship'][0]['waiver']['decision'] = old['ship'][0]['waiver']['decision']
        self.assertEqual(new, old)
        plan = fusion.normalise(fusion.read_manifests(directory))
        profiles = fusion.generate_profiles(json.loads(PROFILES_PATH.read_text(encoding='utf-8')), plan)['profiles']
        self.assertFalse(contract.traffic_waiver_pending(dict(profiles[ONEQ_SHIP], name=ONEQ_SHIP)))
        self.assertEqual(profiles[SHIP], PROFILES[SHIP], 'the fusion ship profile is untouched')
        self.assertEqual({key: value for key, value in profiles[ONEQ_SHIP].items() if key != contract.TRAFFIC_WAIVER},
                         {key: value for key, value in PROFILES[ONEQ_SHIP].items() if key != contract.TRAFFIC_WAIVER})
        fusion.approve_ship(SPEC['waiver']['decision'], directory, 'traffic-oneq')
        self.assertEqual((directory / 'WPP.json').read_text(encoding='utf-8'), before['WPP.json'])

    def test_the_other_entry_is_decided_in_its_own_manifest(self):
        directory = self.copy_manifests()
        fusion.approve_ship('APPROVED by the owner 2026-10-12: ok', directory, 'traffic')
        changed = [path.name for path in directory.glob('*.json') if path.read_text(encoding='utf-8') != (fusion.MANIFESTS / path.name).read_text(encoding='utf-8')]
        self.assertEqual(changed, ['WP0.json'])

    def test_a_ship_entry_naming_a_control_that_is_not_another_ship_is_refused(self):
        directory = self.copy_manifests()
        data = json.loads((directory / 'WPP.json').read_text(encoding='utf-8'))
        data['ship'][0]['control'] = 'traffic-oneq'
        (directory / 'WPP.json').write_text(json.dumps(data, indent=1, ensure_ascii=False) + '\n', encoding='utf-8')
        with self.assertRaises(fusion.ManifestError) as caught:
            fusion.normalise(fusion.read_manifests(directory))
        self.assertIn('not another ship entry', str(caught.exception))
        data['ship'][0]['control'] = None
        data['ship'][0]['bake_graft'] = 'K64k'
        (directory / 'WPP.json').write_text(json.dumps(data, indent=1, ensure_ascii=False) + '\n', encoding='utf-8')
        with self.assertRaises(fusion.ManifestError) as caught:
            fusion.normalise(fusion.read_manifests(directory))
        self.assertIn('bake_graft', str(caught.exception))


class ShipPackTests(unittest.TestCase):
    NAMES = ('B0-ship-build', 'G1-ship-gate', 'GX0-exact-control', 'GX1-exact-ship', 'G2-prefix-hit', 'SR-platform-replay')

    def test_the_pack_is_what_the_generator_makes_and_the_first_pack_is_unchanged_but_for_its_approve_command(self):
        wanted = jobs_gen.generate(PLAN)
        found = {path.name: path.read_text(encoding='utf-8') for path in PACK.glob('*') if path.is_file()}
        self.assertEqual(found, {name.split('/', 1)[1]: text for name, text in wanted.items() if name.startswith('ship-traffic-oneq/')})
        self.assertEqual(sorted(name[:-4] for name in found if name.endswith('.env')), sorted(self.NAMES))
        self.assertEqual(jobs_gen.stale(jobs_gen.FOLDER, wanted), [])
        first = (HERE / 'references' / 'fusion-jobs' / 'ship' / 'ORDER.txt').read_text(encoding='utf-8')
        self.assertIn('--approve-ship "APPROVED ..." --ship traffic: it edits the ship entry named traffic of scripts/ci/fusion-wp/WP0.json', first)
        self.assertIn('--approve-ship "APPROVED ..." --ship traffic-oneq: it edits the ship entry named traffic-oneq of scripts/ci/fusion-wp/WPP.json',
                      (PACK / 'ORDER.txt').read_text(encoding='utf-8'))

    def test_the_build_bakes_the_oneq_graft_and_the_oneq_ship_profile(self):
        values = env_values(PACK / 'B0-ship-build.env')
        self.assertEqual((values['C2_ACTIONS'], values['C2_BAKE_DEFAULT_PROFILE'], values['C2_BAKE_GRAFT']), ('build', ONEQ_SHIP, 'K64j-OQ'))
        self.assertNotIn('C2_BAKE_GRAFT', env_values(HERE / 'references' / 'fusion-jobs' / 'ship' / 'B0-ship-build.env'), 'the first pack still bakes K64j')
        self.assertIn('RQ1 and RQ2', (PACK / 'B0-ship-build.env').read_text(encoding='utf-8'))

    def test_the_exactness_pair_is_the_fusion_ship_against_the_oneq_ship_on_one_image(self):
        self.assertEqual(env_values(PACK / 'GX0-exact-control.env')['C2_PROFILE'], SHIP)
        self.assertEqual(env_values(PACK / 'GX1-exact-ship.env')['C2_PROFILE'], ONEQ_SHIP)
        self.assertEqual(env_values(HERE / 'references' / 'fusion-jobs' / 'ship' / 'GX0-exact-control.env')['C2_PROFILE'], fusion.PARENT)
        images = {env_values(path)['C2_IMAGE_TAG'] for path in PACK.glob('*.env')}
        self.assertEqual(images, {jobs_gen.ship_image(SPEC)})
        self.assertNotEqual(images, {jobs_gen.SHIP_IMAGE})
        self.assertEqual(env_values(PACK / 'G1-ship-gate.env')['C2_PROFILE'], ONEQ_SHIP)
        self.assertEqual(env_values(PACK / 'G2-prefix-hit.env')['C2_PREFIX_PROFILE'], ONEQ_SHIP)
        self.assertEqual(env_values(PACK / 'SR-platform-replay.env')['C2_REPLAY_PROFILE'], ONEQ_SHIP)

    def test_every_template_parses_and_the_build_is_refused_exactly_while_the_decision_is_pending(self):
        for name in self.NAMES:
            code = run_job(PACK / (name + '.env')).returncode
            with self.subTest(name=name):
                self.assertEqual(code, 1 if name == 'B0-ship-build' else 0)
        refused = run_job(PACK / 'B0-ship-build.env')
        self.assertIn(b'not APPROVED', refused.stderr)
        # (the approved side: ProfileTests.test_the_bake_refuses_it_while_pending_and_takes_it_once_approved)
        self.assertIn('C2_BAKE_GRAFT=K64j-OQ', (PACK / 'B0-ship-build.env').read_text(encoding='utf-8'))
        values = job.parse_env((PACK / 'B0-ship-build.env').read_text(encoding='utf-8'))
        self.assertEqual(job.read_bake_graft(values, ['build']), 'K64j-OQ')


class PublicTests(unittest.TestCase):
    PATTERN = re.compile(r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}|/home/|/Users/|[A-Za-z]:[\\/]|\.local\b|\.lan\b|\bssh\b|ghcr\.io|docker\.io|sha256:[0-9a-f]{12}|spark-|thatch@')

    def test_the_second_pack_the_rq_jobs_the_records_text_and_the_readme_name_no_host_address_registry_or_home_path(self):
        paths = [path for path in sorted(PACK.glob('*'))] + sorted(WPP.glob('RQ*.env')) + [WPP / 'ORDER.txt', HERE / 'fusion-wp' / 'WPP.json',
                                                                                           REPO / 'optimisation' / 'ttnn-op' / 'sdpa_prefill_oneq' / 'README.md']
        for path in paths:
            text = path.read_text(encoding='utf-8')
            with self.subTest(path.name):
                self.assertIsNone(self.PATTERN.search(text), self.PATTERN.search(text) and self.PATTERN.search(text).group(0))
                self.assertNotIn('\r', text)


class RequalificationTemplatesTests(unittest.TestCase):
    RQ1 = ('RQ1-W1-watcher-cb1-cb2a', 'RQ1-F1-cb1', 'RQ1-F2-cb2a', 'RQ1-W2-reader-watcher', 'RQ1-F3-cb2b')
    RQ2 = ('RQ2-W1-watcher-cb1', 'RQ2-F1-cb1', 'RQ2-F2-cb2a', 'RQ2-W2-reader-watcher', 'RQ2-F3-cb2b')
    OLD_131K = ('EV-W1-cb1-cb2a-watcher', 'EV-F1-cb1', 'EV-F2-cb2a', 'EV-W2-cb2b-watcher', 'EV-F3-cb2b')
    OLD_262K = ('E2w-card-watcher', 'E2a-cb1', 'E2b-cb2a', 'E2xw-reader-watcher', 'E2c-cb2b')

    def old(self, folder, name):
        return env_values(HERE / 'references' / folder / (name + '.env'))

    def test_every_rq_template_parses(self):
        for path in sorted(WPP.glob('RQ*.env')):
            with self.subTest(path=path.name):
                self.assertEqual(run_job(path).returncode, 0)
        self.assertEqual(sorted(path.stem for path in WPP.glob('RQ*.env')), sorted(('RQ0-quad-reset',) + self.RQ1 + self.RQ2))

    def test_the_arguments_are_the_k64j_templates_arguments_on_the_k64j_oq_graft_by_name(self):
        by_name = ('KOPGRAFT64=', 'EXPECT_TTNNCPP_SHA256=', 'IMAGE=', 'IMAGE_C2_TAG=', 'KOPGRAFT64_NAME=')
        for new_names, old_names, folder in ((self.RQ1, self.OLD_131K, 'tp4-s2-serve-jobs'), (self.RQ2, self.OLD_262K, 'tp4-262k8-jobs')):
            for new, old in zip(new_names, old_names):
                with self.subTest(new=new):
                    mine, theirs = env_values(WPP / (new + '.env')), self.old(folder, old)
                    for key in ('C2_CARDS', 'C2_ACTIONS', 'C2_CARDM_HARNESS', 'C2_CARDM_ARGS'):
                        self.assertEqual(mine[key], theirs[key], key)
                    words = lambda text: sorted(word for word in text.split() if not word.startswith(by_name))
                    self.assertEqual(words(mine['C2_CARDM_ENV']), words(theirs['C2_CARDM_ENV']))

    def test_the_graft_the_digest_and_the_image_are_plain_literal_words_for_a_public_tag_commit(self):
        """No placeholder, no path, no registry: the graft is a directory NAME under the runner's home (KOPGRAFT64_NAME), the digest the literal K64j-OQ one already in tracked
        code, the image its C2 tag (IMAGE_C2_TAG = C2_IMAGE_TAG = tp4-oneq-1, the image OQ-B0 builds)."""
        for path in sorted(WPP.glob('RQ*.env')):
            text = path.read_text(encoding='utf-8')
            values = env_values(path)
            with self.subTest(path=path.name):
                self.assertNotIn('@', text)
                self.assertEqual(values['C2_IMAGE_TAG'], 'tp4-oneq-1')
                if values['C2_ACTIONS'] != 'cardm':
                    continue
                pairs = dict(word.partition('=')[::2] for word in values['C2_CARDM_ENV'].split())
                self.assertEqual(pairs['KOPGRAFT64_NAME'], 'opgraft-K64j-OQ')
                self.assertNotIn('KOPGRAFT64', pairs)
                self.assertEqual(pairs['EXPECT_TTNNCPP_SHA256'], OQ)
                self.assertEqual(pairs['IMAGE_C2_TAG'], 'tp4-oneq-1')
                self.assertNotIn('IMAGE', pairs)
                for word in values['C2_CARDM_ENV'].split():
                    self.assertTrue(job.PLAIN_WORD.fullmatch(word), word)

    def test_the_262k_jobs_name_the_window_and_the_131k_ones_do_not(self):
        for name in self.RQ2:
            self.assertIn('--capacity 262144', env_values(WPP / (name + '.env'))['C2_CARDM_ARGS'], name)
        for name in self.RQ1:
            self.assertNotIn('--capacity', env_values(WPP / (name + '.env'))['C2_CARDM_ARGS'], name)

    def test_the_watchers_run_before_their_full_jobs_and_the_reset_is_quad(self):
        reset = env_values(WPP / 'RQ0-quad-reset.env')
        self.assertEqual((reset['C2_CARDS'], reset['C2_ACTIONS']), ('quad', 'reset'))
        for name in self.RQ1 + self.RQ2:
            env = env_values(WPP / (name + '.env'))
            self.assertEqual(env['C2_CARDS'], 'pair', name)
            self.assertEqual('WATCHER=1' in env['C2_CARDM_ENV'], '-W' in name, name)

    def test_the_order_has_the_chain_the_minutes_and_the_totals_the_readme_states(self):
        text = (WPP / 'ORDER.txt').read_text(encoding='utf-8')
        rows = {line.split()[0]: line.split() for line in text.splitlines() if line.startswith('RQ')}
        self.assertEqual(sorted(rows), sorted(('RQ0-quad-reset',) + self.RQ1 + self.RQ2))
        minutes = {name: int(row[3]) for name, row in rows.items()}
        self.assertEqual(minutes['RQ0-quad-reset'] + sum(minutes[name] for name in self.RQ1), 48)
        self.assertEqual(sum(minutes[name] for name in self.RQ2), 88)
        self.assertEqual(sum(minutes.values()), 136)
        self.assertEqual({row[2] for row in rows.values()}, {'tp4-oneq-1'})
        for name in self.RQ1 + self.RQ2:
            self.assertTrue(any(line.startswith('# NEEDS %s <- ' % name) for line in text.splitlines()), name)
        # the minutes are the K64j templates' own estimates
        for new_names, old_names, folder in ((self.RQ1, self.OLD_131K, 'tp4-s2-serve-jobs'), (self.RQ2, self.OLD_262K, 'tp4-262k8-jobs')):
            order = {line.split()[0]: int(line.split()[3]) for line in (HERE / 'references' / folder / 'ORDER.txt').read_text(encoding='utf-8').splitlines()
                     if line and not line.startswith('#')}
            for new, old in zip(new_names, old_names):
                self.assertEqual(minutes[new], order[old], new)


def admit_boot(environ, max_position=None):
    """admit() as a four-card boot with `environ` over the extent environment, the runtime check answering for the binary named in it: -> (record, log lines). A traffic
    boot has no QWEN_C2_GATE; the E1 record and the 262k window are whatever the checkout holds unless the caller patches page_width_tp4."""
    env = {admission.FLAG: '1', 'QWEN_FAST_ANY_REQUEST': '1', 'QWEN_FAST_REPLAY_GROUP_ROWS': '8', 'QWEN_SDPA_TREE_SCRATCH_ROUNDS': '1', 'QWEN_FAST_SDPA_MODES': 'tail,share',
           'QWEN_FAST_TP': '4'}
    env.update(environ)
    if max_position:
        env['QWEN_FAST_MAX_POSITION'] = str(max_position)
    lines = Lines()
    with mock.patch.dict(admission._STATE, clear=True), mock.patch.object(
            admission, 'check_runtime', lambda root, binaries=None, served=None: dict(binaries={'a': served or K64J, 'b': served or K64J}, kernels={}, literals=[])):
        record = admission.admit('/opt/tt-metal', m3=M3, environ=env, log=lines)
    return record, lines


def landed_files(directory):
    """The state after the records land: the committed K64j records copied with their binary reset to K64j-OQ, in `directory`, and the constants that name them."""
    paths = {}
    for capacity, source, name in ((admission.CAPACITY_131K, admission.EVIDENCE_TP4, 'packed_any_evidence_tp4_k64j_oq.json'),
                                   (admission.CAPACITY_262K, admission.EVIDENCE_TP4_262K, 'packed_any_evidence_tp4_262144_k64j_oq.json')):
        text = source.read_text(encoding='utf-8')
        assert text.count(K64J) == 1
        path = Path(directory) / name
        path.write_text(text.replace(K64J, OQ), encoding='utf-8', newline=chr(10))
        paths[capacity] = (path, hashlib.sha256(path.read_bytes()).hexdigest())
    return paths


class OwnRecordTests(unittest.TestCase):
    """packed_any_admission.own_record: a K64j-OQ process reads its own record per capacity, strictly; K64j never does."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix='oneq-ship-'))
        self.addCleanup(shutil.rmtree, str(self.tmp), True)
        self.files = landed_files(self.tmp)
        self.none = [mock.patch.object(admission, 'EVIDENCE_TP4_K64J_OQ_SHA256', None), mock.patch.object(admission, 'EVIDENCE_TP4_262K_K64J_OQ_SHA256', None)]
        for patch in self.none:
            patch.start()
            self.addCleanup(patch.stop)

    def land(self, *capacities):
        for capacity in capacities:
            path, digest = self.files[capacity]
            if capacity == admission.CAPACITY_131K:
                patches = [mock.patch.object(admission, 'EVIDENCE_TP4_K64J_OQ', path), mock.patch.object(admission, 'EVIDENCE_TP4_K64J_OQ_SHA256', digest)]
            else:
                patches = [mock.patch.object(admission, 'EVIDENCE_TP4_262K_K64J_OQ', path), mock.patch.object(admission, 'EVIDENCE_TP4_262K_K64J_OQ_SHA256', digest)]
            for patch in patches:
                patch.start()
                self.addCleanup(patch.stop)

    def test_nothing_is_recorded_yet_the_k64j_records_are_the_only_ones_and_the_pins_are_none(self):
        # (holds in the committed state before the records land; after they land test_the_committed_records_are_consistent takes over)
        if LANDED:
            self.skipTest('the K64j-OQ records have landed')
        self.assertIsNone(admission.own_record(4, admission.CAPACITY_131K, OQ))
        self.assertFalse(admission.EVIDENCE_TP4_K64J_OQ.exists())
        self.assertFalse(admission.EVIDENCE_TP4_262K_K64J_OQ.exists())

    def test_without_a_pin_every_binary_reads_the_k64j_records(self):
        for capacity in admission.CAPACITIES:
            for binary in (None, K64J, OQ):
                self.assertEqual(admission.evidence_path(4, capacity, binary), admission.evidence_path(4, capacity))
                self.assertEqual(admission.evidence_pin(4, capacity, binary), admission.evidence_pin(4, capacity))

    def test_with_a_pin_only_k64j_oq_at_four_cards_reads_its_own_file_per_capacity(self):
        self.land(admission.CAPACITY_262K)
        path, digest = self.files[admission.CAPACITY_262K]
        self.assertEqual(admission.evidence_path(4, admission.CAPACITY_262K, OQ), path)
        self.assertEqual(admission.evidence_pin(4, admission.CAPACITY_262K, OQ), digest)
        self.assertEqual(admission.evidence_path(4, admission.CAPACITY_262K, K64J), admission.EVIDENCE_TP4_262K)
        self.assertEqual(admission.evidence_path(4, admission.CAPACITY_262K), admission.EVIDENCE_TP4_262K)
        self.assertEqual(admission.evidence_path(4, admission.CAPACITY_131K, OQ), admission.EVIDENCE_TP4, 'the 131,328 record has no K64j-OQ file yet')
        self.assertEqual(admission.evidence_path(2, admission.CAPACITY_262K, OQ), admission.EVIDENCE_TP4_262K, 'the pair has no K64j-OQ record')

    def check(self, capacity, environ):
        return admission.check_evidence(tp=4, capacity=capacity, sources_root=str(HERE), environ=environ)

    def test_a_traffic_boot_on_k64j_oq_passes_on_its_own_record_and_k64j_still_passes_on_k64js(self):
        self.land(*admission.CAPACITIES)
        for capacity in admission.CAPACITIES:
            with self.subTest(capacity=capacity):
                record = self.check(capacity, {ENV: OQ})
                self.assertEqual(record['binary']['ttnncpp_sha256'], OQ)
                self.assertEqual(self.check(capacity, {ENV: K64J})['binary']['ttnncpp_sha256'], K64J)
                self.assertEqual(self.check(capacity, {})['binary']['ttnncpp_sha256'], K64J)
        window = admission.check_window_262k(sources_root=str(HERE), writer_state=(True, []), environ={ENV: OQ})
        self.assertEqual(window['binary']['ttnncpp_sha256'], OQ)

    def test_the_gate_flag_does_not_change_which_record_a_k64j_oq_process_reads(self):
        self.land(admission.CAPACITY_131K)
        self.assertEqual(self.check(admission.CAPACITY_131K, dict(GATE, **{ENV: OQ}))['binary']['ttnncpp_sha256'], OQ)

    def test_a_capacity_without_its_own_record_still_borrows_in_a_gate_boot_only(self):
        self.land(admission.CAPACITY_131K)
        with self.assertRaises(admission.AdmissionRefused) as caught:
            self.check(admission.CAPACITY_262K, {ENV: OQ})
        self.assertIn('binary: the evidence qualified %s, not K64j-OQ %s' % (K64J[:16], OQ[:16]), str(caught.exception))
        self.assertEqual(self.check(admission.CAPACITY_262K, dict(GATE, **{ENV: OQ}))['binary']['ttnncpp_sha256'], K64J)

    def test_an_own_record_that_names_k64j_is_not_a_record_of_k64j_oq(self):
        path, _ = self.files[admission.CAPACITY_131K]
        wrong = Path(self.tmp) / 'wrong.json'
        wrong.write_text(admission.EVIDENCE_TP4.read_text(encoding='utf-8'), encoding='utf-8', newline=chr(10))
        with mock.patch.object(admission, 'EVIDENCE_TP4_K64J_OQ', wrong), \
                mock.patch.object(admission, 'EVIDENCE_TP4_K64J_OQ_SHA256', hashlib.sha256(wrong.read_bytes()).hexdigest()):
            with self.assertRaises(admission.AdmissionRefused) as caught:
                self.check(admission.CAPACITY_131K, {ENV: OQ})
            self.assertIn('binary: the evidence qualified %s, not K64j-OQ %s' % (K64J[:16], OQ[:16]), str(caught.exception))
            # in a gate boot a K64j record in the K64j-OQ slot is still K64j's record, and K64j-OQ may borrow it by the reviewed equivalence (never in traffic, above)
            self.assertEqual(self.check(admission.CAPACITY_131K, dict(GATE, **{ENV: OQ}))['binary']['ttnncpp_sha256'], K64J)

    def test_a_tampered_own_record_fails_its_pin(self):
        self.land(admission.CAPACITY_131K)
        path, _ = self.files[admission.CAPACITY_131K]
        path.write_text(path.read_text(encoding='utf-8').replace('"PASS"', '"PASS "', 1), encoding='utf-8', newline=chr(10))
        with self.assertRaises(admission.AdmissionRefused) as caught:
            self.check(admission.CAPACITY_131K, {ENV: OQ})
        self.assertIn('not the reviewed', str(caught.exception))

    def admit(self, environ, max_position=None):
        return admit_boot(environ, max_position)

    def test_the_admit_of_a_traffic_boot_passes_on_the_own_record_logs_its_pin_and_needs_no_equivalence(self):
        self.land(admission.CAPACITY_131K)
        record, lines = self.admit({ENV: OQ})
        self.assertNotIn('binary_equivalence', record)
        self.assertEqual(len(lines), 1)
        self.assertIn('packed-any admission passed: K64j-OQ %s x2' % OQ[:16], lines[0])
        self.assertIn('evidence %s' % self.files[admission.CAPACITY_131K][1][:16], lines[0])
        for line in lines:
            self.assertLess(len(line), 250)

    def test_the_admit_at_262k_needs_the_own_record_of_the_window(self):
        with mock.patch('page_width_tp4.evidence_state', return_value=(True, [])):
            self.land(admission.CAPACITY_131K)
            with self.assertRaises(admission.AdmissionRefused):
                self.admit({ENV: OQ}, max_position=262144)
            self.land(admission.CAPACITY_262K)
            record, lines = self.admit({ENV: OQ}, max_position=262144)
        self.assertEqual(record['capacity'], 262144)
        self.assertTrue(lines[-1].endswith(' capacity=262144'), lines[-1])


class RecorderTests(unittest.TestCase):
    """record_packed_any_evidence_tp4.py --binary k64j-oq: the K64j-OQ file and its pin, in one go, and nothing of K64j's."""

    def setUp(self):
        self.work = sample.Workspace(self)
        # the scratch copy of the admission starts with the K64j-OQ pins unset whether or not the records have landed in this checkout
        text = self.work.admission.read_text(encoding='utf-8')
        for name in ('EVIDENCE_TP4_K64J_OQ_SHA256', 'EVIDENCE_TP4_262K_K64J_OQ_SHA256'):
            text, count = re.subn(r"^(%s = )(?:None|'[0-9a-f]{64}')" % name, r"\1None", text, flags=re.M)
            assert count == 1
        self.work.admission.write_text(text, encoding='utf-8', newline=chr(10))
        self.target = self.work.dir / 'packed_any_evidence_tp4_k64j_oq.json'
        patch = mock.patch.object(admission, 'EVIDENCE_TP4_K64J_OQ', self.target)
        patch.start()
        self.addCleanup(patch.stop)

    def oq(self, report):
        report = copy.deepcopy(report)
        report['binary']['sha256'] = OQ
        return report

    def argv(self, reports=('cb1', 'cb2a', 'cb2b'), extra=()):
        args = ['--admission', str(self.work.admission), '--sources-root', str(HERE), '--binary', 'k64j-oq']
        builders = dict(cb1=('card-ev-f1.json', sample.cb1_report), cb2a=('card-ev-f2.json', sample.cb2a_report), cb2b=('reader-ev-f3.json', sample.cb2b_report))
        for name in reports:
            file_name, make = builders[name]
            args += ['--%s' % name, self.work.report(file_name, self.oq(make())), '--%s-run' % name, '2001', '--%s-tag' % name, 'experiment/c2-serving-v951']
        if 'cb2b' in reports:
            args += ['--cb2b-commit', sample.COMMIT, '--cb2b-image', 'tp4-oneq-1']
        return args + list(extra)

    def pin(self, name):
        return re.search(r"^%s = (None|'[0-9a-f]{64}')" % name, self.work.admission.read_text(encoding='utf-8'), re.M).group(1)

    def test_three_reports_on_k64j_oq_make_its_own_record_file_and_set_its_own_pin(self):
        k64j_pin_before = self.work.pin()
        lines = []
        self.assertEqual(recorder.main(self.argv(), out=lines.append), 0, lines)
        payload = self.target.read_bytes()
        self.assertNotIn(b'\r', payload)
        record = json.loads(payload.decode('utf-8'))
        self.assertEqual(record['binary']['ttnncpp_sha256'], OQ)
        self.assertIn('K64j-OQ', record['binary']['note'])
        self.assertEqual([record['sections'][name]['status'] for name in admission.SECTIONS], ['PASS'] * 3)
        self.assertIn('EVIDENCE_TP4_K64J_OQ_SHA256', record['what'])
        self.assertNotIn("(EVIDENCE_TP4_SHA256)", record['what'])
        self.assertEqual(self.pin('EVIDENCE_TP4_K64J_OQ_SHA256'), "'%s'" % hashlib.sha256(payload).hexdigest())
        self.assertEqual(self.work.pin(), k64j_pin_before, "K64j's pin is not touched")
        self.assertEqual(self.pin('EVIDENCE_TP4_262K_K64J_OQ_SHA256'), 'None', "the other capacity's K64j-OQ pin is not touched")
        self.assertEqual(self.work.evidence.read_bytes(), sample.SKELETON_TP4_TEXT.encode('utf-8') if hasattr(sample, 'SKELETON_TP4_TEXT') else self.work.evidence.read_bytes())
        # the record qualifies a TRAFFIC boot on K64j-OQ, through the admission's own gate
        admission.check_evidence(str(self.target), expected_sha256=hashlib.sha256(payload).hexdigest(), sources_root=str(HERE), tp=4, environ={ENV: OQ})
        self.assertEqual(admission.evidence_problems(record, str(HERE), tp=4, environ={ENV: OQ}), [])
        self.assertTrue(admission.evidence_problems(record, str(HERE), tp=4, environ={ENV: K64J}), 'and not a boot on K64j')

    def test_the_first_k64j_oq_record_needs_all_three_sections(self):
        for reports in (('cb1',), ('cb1', 'cb2a'), ('cb2b',)):
            lines = []
            with self.subTest(reports=reports):
                self.assertEqual(recorder.main(self.argv(reports), out=lines.append), 1)
                self.assertIn('needs all three of --cb1, --cb2a and --cb2b', lines[0])
                self.assertFalse(self.target.exists())
        self.assertEqual(self.pin('EVIDENCE_TP4_K64J_OQ_SHA256'), 'None')

    def test_reports_made_on_k64j_are_not_a_k64j_oq_record(self):
        argv = [word for word in self.argv()]
        k64j = copy.deepcopy(sample.cb1_report())
        path = self.work.report('card-ev-f1.json', k64j)
        argv[argv.index('--cb1') + 1] = path
        lines = []
        self.assertEqual(recorder.main(argv, out=lines.append), 1)
        self.assertTrue(any('the loaded binary is %s, not K64j-OQ %s' % (K64J[:16], OQ[:16]) in line for line in lines), lines)
        self.assertFalse(self.target.exists())

    def test_a_later_run_updates_the_existing_k64j_oq_record_in_place(self):
        self.assertEqual(recorder.main(self.argv(), out=lambda line: None), 0)
        first = self.target.read_bytes()
        self.assertEqual(recorder.main(self.argv(('cb2a',)), out=lambda line: None), 0)
        record = json.loads(self.target.read_bytes().decode('utf-8'))
        self.assertEqual(record['binary']['ttnncpp_sha256'], OQ)
        self.assertEqual([record['sections'][name]['status'] for name in admission.SECTIONS], ['PASS'] * 3)
        self.assertEqual(self.pin('EVIDENCE_TP4_K64J_OQ_SHA256'), "'%s'" % hashlib.sha256(self.target.read_bytes()).hexdigest())
        self.assertTrue(first)

    def test_the_k64j_default_still_writes_the_k64j_file_and_pin(self):
        pins_before = {name: self.pin(name) for name in ('EVIDENCE_TP4_K64J_OQ_SHA256', 'EVIDENCE_TP4_262K_K64J_OQ_SHA256')}
        argv = ['--evidence', str(self.work.evidence), '--admission', str(self.work.admission), '--sources-root', str(HERE),
                '--cb1', self.work.report('card-ev-f1.json', sample.cb1_report()), '--cb1-run', '1', '--cb1-tag', 'experiment/c2-serving-v901']
        self.assertEqual(recorder.main(argv, out=lambda line: None), 0)
        self.assertEqual({name: self.pin(name) for name in pins_before}, pins_before)
        self.assertFalse(self.target.exists())

    def test_pin_name_and_repin_follow_the_binary_and_the_capacity(self):
        self.assertEqual(recorder.pin_name(admission.CAPACITY_131K), 'EVIDENCE_TP4_SHA256')
        self.assertEqual(recorder.pin_name(admission.CAPACITY_262K), 'EVIDENCE_TP4_262K_SHA256')
        self.assertEqual(recorder.pin_name(admission.CAPACITY_131K, OQ), 'EVIDENCE_TP4_K64J_OQ_SHA256')
        self.assertEqual(recorder.pin_name(admission.CAPACITY_262K, OQ), 'EVIDENCE_TP4_262K_K64J_OQ_SHA256')
        original = self.work.admission.read_text(encoding='utf-8')
        for capacity, name in ((admission.CAPACITY_131K, 'EVIDENCE_TP4_K64J_OQ_SHA256'), (admission.CAPACITY_262K, 'EVIDENCE_TP4_262K_K64J_OQ_SHA256')):
            changed = recorder.repin(str(self.work.admission), 'ab' * 32, capacity, OQ).decode('utf-8')
            old_lines, new_lines = original.splitlines(), changed.splitlines()
            differing = [(a, b) for a, b in zip(old_lines, new_lines) if a != b]
            self.assertEqual(len(old_lines), len(new_lines))
            self.assertEqual(differing, [('%s = None' % name, "%s = '%s'" % (name, 'ab' * 32))])


@unittest.skipUnless(LANDED, 'the K64j-OQ records have not landed in this checkout')
class TrafficBootOnTheCommittedRecordsTests(unittest.TestCase):
    """With the RQ records landed, a TRAFFIC boot on K64j-OQ passes the admission on the committed files at their committed pins: no patch of a constant, no gate switch, no
    equivalence, the real E1 record. The same boots on K64j still pass on K64j's records."""

    def test_a_traffic_boot_on_k64j_oq_is_admitted_at_131k_on_its_own_record(self):
        record, lines = admit_boot({ENV: OQ})
        self.assertNotIn('binary_equivalence', record, 'its own record, not K64j\'s by the gate-boot equivalence')
        self.assertEqual(record['evidence']['binary']['ttnncpp_sha256'], OQ)
        self.assertEqual(len(lines), 1)
        self.assertIn('packed-any admission passed: K64j-OQ %s x2' % OQ[:16], lines[0])
        self.assertIn('evidence %s' % admission.EVIDENCE_TP4_K64J_OQ_SHA256[:16], lines[0])
        self.assertEqual(hashlib.sha256(admission.EVIDENCE_TP4_K64J_OQ.read_bytes()).hexdigest(), admission.EVIDENCE_TP4_K64J_OQ_SHA256)

    def test_a_traffic_boot_on_k64j_oq_is_admitted_at_262k_on_its_own_record_and_the_real_e1(self):
        record, lines = admit_boot({ENV: OQ}, max_position=262144)
        self.assertEqual(record['capacity'], 262144)
        self.assertNotIn('binary_equivalence', record)
        self.assertEqual(record['evidence']['binary']['ttnncpp_sha256'], OQ)
        self.assertEqual(record['evidence']['capacity'], 262144)
        self.assertEqual(len(lines), 1)
        self.assertIn('packed-any admission passed: K64j-OQ %s x2' % OQ[:16], lines[0])
        self.assertIn('evidence %s' % admission.EVIDENCE_TP4_262K_K64J_OQ_SHA256[:16], lines[0])
        self.assertTrue(lines[0].endswith(' capacity=262144'), lines[0])
        self.assertEqual(hashlib.sha256(admission.EVIDENCE_TP4_262K_K64J_OQ.read_bytes()).hexdigest(), admission.EVIDENCE_TP4_262K_K64J_OQ_SHA256)

    def test_the_guard_of_every_other_four_card_process_admits_it_too(self):
        for max_position in (None, 262144):
            env = {'QWEN_FAST_TP': '4', ENV: OQ}
            if max_position:
                env['QWEN_FAST_MAX_POSITION'] = str(max_position)
            with self.subTest(max_position=max_position):
                self.assertEqual(admission.tp_guard(env, log=Lines(), sources_root=str(HERE)), [])

    def test_a_gate_boot_on_k64j_oq_reads_the_same_own_record(self):
        record, lines = admit_boot(dict(GATE, **{ENV: OQ}), max_position=262144)
        self.assertNotIn('binary_equivalence', record)
        self.assertEqual(record['evidence']['binary']['ttnncpp_sha256'], OQ)

    def test_k64j_still_serves_traffic_on_its_own_records_at_both_capacities(self):
        for max_position, pin in ((None, admission.EVIDENCE_TP4_SHA256), (262144, admission.EVIDENCE_TP4_262K_SHA256)):
            with self.subTest(max_position=max_position):
                record, lines = admit_boot({ENV: K64J}, max_position=max_position)
                self.assertEqual(record['evidence']['binary']['ttnncpp_sha256'], K64J)
                self.assertIn('packed-any admission passed: K64j %s x2' % K64J[:16], lines[0])
                self.assertIn('evidence %s' % pin[:16], lines[0])

    def test_a_binary_that_is_neither_is_still_refused_by_name(self):
        with self.assertRaises(admission.AdmissionRefused) as caught:
            admit_boot({ENV: 'a' * 64})
        self.assertIn("not K64j's %s or K64j-OQ's %s" % (K64J[:16], OQ[:16]), str(caught.exception))


class LandingTests(unittest.TestCase):
    """The committed state, before and after the records land: whatever is pinned is there, at its pin, naming the binary it is for."""

    def test_the_committed_records_are_consistent(self):
        for capacity, k64j_path, k64j_pin, own_path, own_pin in (
                (admission.CAPACITY_131K, admission.EVIDENCE_TP4, admission.EVIDENCE_TP4_SHA256, admission.EVIDENCE_TP4_K64J_OQ, admission.EVIDENCE_TP4_K64J_OQ_SHA256),
                (admission.CAPACITY_262K, admission.EVIDENCE_TP4_262K, admission.EVIDENCE_TP4_262K_SHA256, admission.EVIDENCE_TP4_262K_K64J_OQ,
                 admission.EVIDENCE_TP4_262K_K64J_OQ_SHA256)):
            with self.subTest(capacity=capacity):
                self.assertEqual(hashlib.sha256(k64j_path.read_bytes()).hexdigest(), k64j_pin)
                self.assertEqual(json.loads(k64j_path.read_text(encoding='utf-8'))['binary']['ttnncpp_sha256'], K64J, "K64j's record names K64j and stays")
                if own_pin is None:
                    self.assertFalse(own_path.exists(), 'a K64j-OQ record without its pin is a record nothing reads')
                    continue
                self.assertEqual(hashlib.sha256(own_path.read_bytes()).hexdigest(), own_pin)
                record = json.loads(own_path.read_text(encoding='utf-8'))
                self.assertEqual(record['binary']['ttnncpp_sha256'], OQ)
                self.assertEqual(admission.evidence_problems(record, str(HERE), tp=4, capacity=capacity, environ={ENV: OQ}), [])
                self.assertEqual([record['sections'][name]['status'] for name in admission.SECTIONS], ['PASS'] * 3)
                text = own_path.read_text(encoding='utf-8')
                self.assertNotIn('/' + 'home' + '/', text)

    def test_the_two_new_files_and_pins_are_all_a_landing_changes(self):
        """The landing is the recorder's: the two K64j-OQ record files and the two None pins. Nothing else names them."""
        source = (HERE / 'packed_any_admission.py').read_text(encoding='utf-8')
        for name in ('EVIDENCE_TP4_K64J_OQ_SHA256', 'EVIDENCE_TP4_262K_K64J_OQ_SHA256'):
            self.assertEqual(len(re.findall(r"^%s = " % name, source, re.M)), 1)
        self.assertEqual(sorted(path.name for path in HERE.glob('packed_any_evidence*.json')),
                         sorted(['packed_any_evidence.json', 'packed_any_evidence_tp4.json', 'packed_any_evidence_tp4_262144.json']
                                + [path.name for path in (admission.EVIDENCE_TP4_K64J_OQ, admission.EVIDENCE_TP4_262K_K64J_OQ) if path.exists()]))

    def test_a_pinned_k64j_oq_record_is_carried_by_the_image_overlay(self):
        """The overlay manifest is the one image list: a pin without the file's overlay line gives a K64j-OQ image a pin and no record (every attach then refuses, a gate boot too)."""
        lines = {line.strip() for line in (REPO / 'docker' / 'qwen-c2-overlay.txt').read_text(encoding='utf-8').splitlines()}
        for path, pin in ((admission.EVIDENCE_TP4_K64J_OQ, admission.EVIDENCE_TP4_K64J_OQ_SHA256),
                          (admission.EVIDENCE_TP4_262K_K64J_OQ, admission.EVIDENCE_TP4_262K_K64J_OQ_SHA256)):
            with self.subTest(path=path.name):
                if pin is not None:
                    self.assertIn('scripts/ci/' + path.name, lines)
                else:
                    self.assertNotIn('scripts/ci/' + path.name, lines, 'an overlay line for a file that does not exist yet')
        for path in (admission.EVIDENCE_TP4, admission.EVIDENCE_TP4_262K):
            self.assertIn('scripts/ci/' + path.name, lines)

    def test_the_k64j_only_tests_hold_in_both_states(self):
        """A K64j process reads K64j's records whatever else is committed (the whole of the existing admission tests rests on it)."""
        for capacity in admission.CAPACITIES:
            self.assertEqual(admission.evidence_path(4, capacity, K64J), admission.evidence_path(4, capacity))
            self.assertEqual(admission.evidence_pin(4, capacity, K64J), admission.evidence_pin(4, capacity))


if __name__ == '__main__':
    unittest.main()
