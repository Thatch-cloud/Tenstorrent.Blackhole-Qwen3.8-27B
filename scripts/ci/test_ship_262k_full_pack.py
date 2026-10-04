"""The ship/262k-full pack (scripts/ci/references/tp4-ship-262k-full-jobs): every template parses with c2_serving_job.py, the order file is consistent, and
the public templates name no rig, card, host, registry or digest.

The three profiles (c2-packed-tp4-8x262k-ship-full, -ship-full-audit, -ship-full-final-hold-gate) land with the integration merge. While one is absent the
test parses against a temporary profiles file in which the missing profile is cloned from a stand-in that exists (the shipping profile for ship-full, the
best-time gate profile for the two gate twins), so the template's own keys, actions and plans are still checked."""

import copy
import json
import os
import re
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import c2_serving_job as job  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
FOLDER = os.path.join(HERE, 'references', 'tp4-ship-262k-full-jobs')
PROFILES_PATH = os.path.join(HERE, 'qwen_c2_profiles.json')
IMAGE = 'tp4-serve-10'
FULL = 'c2-packed-tp4-8x262k-ship-full'
AUDIT = FULL + '-audit'
FAULT = FULL + '-final-hold-gate'
STAND_IN = {FULL: 'c2-packed-tp4-8x262k-ship', AUDIT: 'c2-packed-tp4-8x262k-best-time-gate', FAULT: 'c2-packed-tp4-8x262k-best-time-gate'}
PLACEHOLDERS = {'@THIN_LAYER_IMAGE@': 'thin-layer-image-ref'}
BANNED = re.compile(r'blackhole-[A-Za-z0-9]{8,}|thatch\.local|\d{1,3}(\.\d{1,3}){3}|sha256:[0-9a-f]{16}|[0-9a-f]{40,}|/dev/tenstorrent|home/|zot\.')
ORDERED = ('X0B-status-rescan-reset-build', 'A1-audited-attach-smoke', 'S0-baked-default-smoke', 'H1-hang-shape-ship-full', 'H2-hang-shape-ship-full',
           'P1-prefix-exactness-lifecycle', 'F1-levern-final-hold-fault', 'L8-ladder8-past-131k', 'C16-churn16', 'M1-cold262k-stall-soft',
           'SR10-platform-replay', 'Z-reset')
SOFT = ('M1-cold262k-stall-soft', 'Z-reset')
ACTIONS = {'X0B-status-rescan-reset-build': 'status rescan reset build', 'P1-prefix-exactness-lifecycle': 'reset prefix', 'L8-ladder8-past-131k': 'reset gate',
           'C16-churn16': 'reset gate', 'SR10-platform-replay': 'reset replay', 'Z-reset': 'status reset'}
SMOKE_TESTS = 'warmup,coding,concurrent8_steady,concurrent8_code_32k,concurrent8_code_equal'


def text_of(name):
    with open(os.path.join(FOLDER, name + '.env'), encoding='utf-8') as handle:
        return handle.read()


def read_order():
    with open(os.path.join(FOLDER, 'ORDER.txt'), encoding='utf-8') as handle:
        return [line.split() for line in handle.read().splitlines() if line.strip() and not line.startswith('#')]


class Pack(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with open(PROFILES_PATH, encoding='utf-8') as handle:
            data = json.load(handle)
        cls.missing = [name for name in STAND_IN if name not in data['profiles']]
        for name in cls.missing:
            data['profiles'][name] = copy.deepcopy(data['profiles'][STAND_IN[name]])
        cls.tmp = tempfile.TemporaryDirectory()
        os.makedirs(os.path.join(cls.tmp.name, 'scripts', 'ci'))
        cls.path = os.path.join(cls.tmp.name, 'scripts', 'ci', 'qwen_c2_profiles.json')
        with open(cls.path, 'w', encoding='utf-8') as handle:
            json.dump(data, handle)
        cls.profiles = sorted(data['profiles'])
        cls.meshes = job.profile_meshes(cls.path)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def parsed(self, name):
        text = text_of(name)
        for placeholder, value in PLACEHOLDERS.items():
            text = text.replace(placeholder, value)
        return job.read_job(job.parse_env(text), self.profiles, root=self.tmp.name, meshes=self.meshes)

    def test_every_template_is_in_the_order_once_with_four_columns_and_the_one_image(self):
        rows = read_order()
        self.assertTrue(all(len(row) == 4 for row in rows))
        self.assertEqual([row[0] for row in rows], list(ORDERED))
        self.assertEqual(sorted(row[0] for row in rows), sorted(n[:-4] for n in os.listdir(FOLDER) if n.endswith('.env')))
        for name, mode, image, minutes in rows:
            self.assertEqual(image, IMAGE)
            self.assertTrue(minutes.isdigit() and 4 <= int(minutes) <= 180, minutes)
            want = 'pre' if name == ORDERED[0] else 'soft' if name in SOFT else 'stop'
            self.assertEqual(mode, want, name)

    def test_every_env_parses_with_the_job_reader(self):
        for name in ORDERED:
            with self.subTest(job=name):
                out = self.parsed(name)
                self.assertEqual(out['tag'], IMAGE)
                self.assertEqual(out['cards'], 'quad')
                if name in ACTIONS:
                    self.assertEqual(out['actions'], ACTIONS[name])

    def test_no_agent_actions_and_only_the_first_job_builds(self):
        for name in ORDERED:
            actions = self.parsed(name)['actions'].split()
            self.assertFalse({'agentstop', 'platform', 'unserve'} & set(actions), name)
            self.assertEqual('build' in actions, name == ORDERED[0], name)

    def test_the_build_bakes_ship_full_and_s0_serves_it(self):
        self.assertEqual(self.parsed(ORDERED[0])['bake_default_profile'], FULL)
        self.assertEqual(self.parsed('S0-baked-default-smoke')['profile'], FULL)

    def test_a1_and_s0_run_the_same_five_tests_and_h_runs_use_ship_full(self):
        for name in ('A1-audited-attach-smoke', 'S0-baked-default-smoke'):
            self.assertEqual(self.parsed(name)['tests'], SMOKE_TESTS)
        self.assertEqual(self.parsed('A1-audited-attach-smoke')['profile'], AUDIT)
        for name in ('H1-hang-shape-ship-full', 'H2-hang-shape-ship-full'):
            self.assertEqual(self.parsed(name)['profile'], FULL)
            self.assertIn('concurrent8_drain', self.parsed(name)['tests'])

    def test_the_gate_jobs(self):
        out = self.parsed('L8-ladder8-past-131k')
        self.assertEqual((out['profile'], out['gate_plan']), (AUDIT, 'matrix'))
        lengths = [int(n) for n in out['gate_lengths'].split(',')]
        self.assertEqual(len(lengths), 8)
        self.assertEqual(sum(1 for n in lengths if n > 131072), 4)
        out = self.parsed('C16-churn16')
        self.assertEqual((out['profile'], out['gate_plan']), (AUDIT, 'churn'))
        self.assertEqual(len(out['gate_lengths'].split(',')), 16)
        self.assertEqual(self.parsed('F1-levern-final-hold-fault')['profile'], FAULT)
        self.assertEqual(self.parsed('M1-cold262k-stall-soft')['profile'], FULL)

    def test_the_prefix_job(self):
        out = self.parsed('P1-prefix-exactness-lifecycle')
        self.assertEqual(out['prefix_plan'], 'exactness-shared,lifecycle-evict')
        self.assertEqual(out['prefix_profile'], AUDIT)

    def test_the_replay_names_the_thin_layer_placeholder_and_the_budget_smoke(self):
        self.assertIn('@THIN_LAYER_IMAGE@', text_of('SR10-platform-replay'))
        out = self.parsed('SR10-platform-replay')
        self.assertEqual((out['replay_profile'], out['replay_budget_smoke']), (FULL, '1'))

    def test_templates_are_lf_and_name_nothing_private(self):
        for name in os.listdir(FOLDER):
            with open(os.path.join(FOLDER, name), 'rb') as handle:
                raw = handle.read()
            self.assertNotIn(b'\r', raw, name)
            self.assertIsNone(BANNED.search(raw.decode('utf-8')), name)

    def test_profile_existence_is_checked_once_the_integration_lands(self):
        if self.missing:
            self.skipTest('profiles not merged yet: %s (parsed against stand-ins)' % ', '.join(self.missing))
        with open(PROFILES_PATH, encoding='utf-8') as handle:
            profiles = json.load(handle)['profiles']
        self.assertTrue(profiles[FAULT].get('gate_only') and profiles[AUDIT].get('gate_only'))
        self.assertFalse(profiles[FULL].get('gate_only'))


if __name__ == '__main__':
    unittest.main()
