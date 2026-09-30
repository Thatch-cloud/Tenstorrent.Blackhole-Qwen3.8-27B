"""The four-card tp4/next window: its job templates (scripts/ci/references/tp4-next-jobs), their order, and the profiles they use.

Every template parses with c2_serving_job, names no rig, card, address, registry or digest, uses ONE image tag, resets before it opens
the cards (once), and states its duration and its stop rule. The best-config jobs serve the best profiles (audited for the smoke and
the exactness jobs, timed for N4); the smokes name only tests c2_serving_smoke knows; the lanes jobs name the lanes plans. No template
pushes a tag: the window's driver does, from a throwaway commit.
"""

import json
import os
import re
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import c2_serving_job as job  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
FOLDER = os.path.join(HERE, 'references', 'tp4-next-jobs')
with open(os.path.join(HERE, 'qwen_c2_profiles.json'), encoding='utf-8') as _handle:
    PROFILES = json.load(_handle)
NAMES = sorted(PROFILES['profiles'])
DEVICE_STEPS = ('cardm', 'smoke', 'gate', 'prefix', 'fabric', 'replay')
BANNED = re.compile(r'blackhole-[A-Za-z0-9]{8,}|thatch\.local|\d{1,3}(\.\d{1,3}){3}|sha256:[0-9a-f]{16}|[0-9a-f]{40,}|'
                    r'/dev/tenstorrent|home/|zot\.')
TAG = 'tp4-next-1'
EXPECTED = {
    'N0-build': ('c2-packed-tp4-best', 'stop'),
    'N1-best-gate-smoke-audited': ('c2-packed-tp4-best-gate', 'stop'),
    'N2-s3a-matrix-best-gate': ('c2-packed-tp4-best-gate', 'stop'),
    'N3-staggered-trigger-best-gate': ('c2-packed-tp4-best-gate', 'soft'),
    'N4-best-smoke-timed': ('c2-packed-tp4-best', 'soft'),
    'N5-lanes-round-timing': ('c2-packed-tp4-gate', 'soft'),
    'N6-lanes-timing-l3': ('c2-packed-tp4-gate', 'soft'),
    'N7-lanes-exact-l1': ('c2-packed-tp4-gate', 'soft'),
}
PLANS = {'N2-s3a-matrix-best-gate': 'matrix', 'N3-staggered-trigger-best-gate': 'staggered',
         'N5-lanes-round-timing': 'round-timing', 'N6-lanes-timing-l3': 'lanes-timing', 'N7-lanes-exact-l1': 'lanes-exact'}
SMOKES = {'N1-best-gate-smoke-audited': 'warmup,coding,concurrent4,concurrent4_steady,steady_resend',
          'N4-best-smoke-timed': 'warmup,coding,concurrent4_steady'}


def read_order():
    with open(os.path.join(FOLDER, 'ORDER.txt'), encoding='utf-8') as handle:
        return [tuple(line.split()) for line in handle.read().splitlines() if line.strip() and not line.startswith('#')]


def text_of(name):
    with open(os.path.join(FOLDER, name + '.env'), encoding='utf-8') as handle:
        return handle.read()


def parsed(name):
    values = job.parse_env(text_of(name))
    return values, job.read_job(values, NAMES)


class OrderTests(unittest.TestCase):
    def test_every_template_is_in_the_order_once_and_the_order_names_no_other(self):
        on_disk = sorted(name[:-4] for name in os.listdir(FOLDER) if name.endswith('.env'))
        ordered = [name for name, unused in read_order()]
        self.assertEqual(sorted(ordered), on_disk)
        self.assertEqual(len(set(ordered)), len(ordered))
        self.assertEqual(sorted(ordered), sorted(EXPECTED))

    def test_the_order_and_the_stop_rules(self):
        order = read_order()
        self.assertEqual([name for name, unused in order], sorted(EXPECTED))
        for name, mode in order:
            self.assertEqual(mode, EXPECTED[name][1], name)
        self.assertEqual([name for name, mode in order if mode == 'stop'],
                         ['N0-build', 'N1-best-gate-smoke-audited', 'N2-s3a-matrix-best-gate'])

    def test_the_order_states_the_durations_and_the_tag(self):
        with open(os.path.join(FOLDER, 'ORDER.txt'), encoding='utf-8') as handle:
            text = handle.read()
        self.assertIsNone(BANNED.search(text))
        self.assertIn(TAG, text)
        for name in EXPECTED:
            self.assertRegex(text, r'#\s+%s\s+.*\d+' % name.split('-')[0])


class TemplateTests(unittest.TestCase):
    def test_every_template_parses_names_no_host_and_uses_the_one_image_tag(self):
        for name, (profile, unused) in EXPECTED.items():
            with self.subTest(template=name):
                values, outputs = parsed(name)
                self.assertIsNone(BANNED.search(text_of(name)), name)
                self.assertEqual(outputs['tag'], TAG)
                self.assertEqual(outputs['cards'], 'quad')
                self.assertEqual(outputs['profile'], profile)
                self.assertEqual(set(re.findall(r'@[A-Z0-9_]+@', text_of(name))), set())
                self.assertIn('PLACEHOLDER', text_of(name))

    def test_the_build_opens_no_card_and_a_run_resets_first_and_opens_the_cards_once(self):
        for name in EXPECTED:
            actions = parsed(name)[1]['actions'].split()
            with self.subTest(template=name):
                if name == 'N0-build':
                    self.assertEqual(actions, ['status', 'reset', 'build'])
                    continue
                self.assertEqual(len(set(actions) & set(DEVICE_STEPS)), 1)
                self.assertEqual(actions[0], 'reset', 'links train only at board init: a four-card run follows a reset')

    def test_every_template_states_its_duration_and_its_stop_rule(self):
        for name, (unused, mode) in EXPECTED.items():
            text = text_of(name)
            with self.subTest(template=name):
                self.assertRegex(text, r'# Duration \(estimate')
                self.assertRegex(text, r'\d+ / \d+ / \d+ min|most 90|\d+-\d+ min')
                self.assertIn('# STOP' if mode == 'stop' else '# SOFT', text)
                if name != 'N0-build':
                    self.assertIn('# Read', text)

    def test_no_template_pushes_a_tag_or_names_a_remote(self):
        for name in EXPECTED:
            text = text_of(name)
            self.assertNotRegex(text, r'git push|git tag |gh workflow run|plink|ssh ')
            for line in text.splitlines():
                if 'experiment/' in line:
                    self.assertIn('push a tag experiment/c2-serving-vN', line, 'only the generic instruction, never a named tag')

    def test_the_smokes_name_tests_the_smoke_knows(self):
        with open(os.path.join(HERE, 'c2_serving_smoke.py'), encoding='utf-8') as handle:
            smoke = handle.read()
        for name, tests in SMOKES.items():
            outputs = parsed(name)[1]
            with self.subTest(template=name):
                self.assertEqual(outputs['actions'].split(), ['reset', 'smoke'])
                self.assertEqual(outputs['tests'], tests)
                for test in tests.split(','):
                    self.assertTrue("record('%s'" % test in smoke, test)

    def test_the_gate_jobs_name_their_plans(self):
        for name, plan in PLANS.items():
            values, outputs = parsed(name)
            with self.subTest(template=name):
                self.assertEqual(outputs['actions'].split(), ['reset', 'gate'])
                self.assertEqual(values['C2_GATE_PLAN'], plan)
        values = parsed('N2-s3a-matrix-best-gate')[0]
        self.assertEqual(values['C2_GATE_LENGTHS'], '4096,16384,32768,60000')
        self.assertEqual(values['C2_GATE_MAX_TOKENS'], '512')
        self.assertNotIn('C2_GATE_POLICY', values, 'strict is the default: no decision record')


class ProfileTests(unittest.TestCase):
    def env(self, name):
        return PROFILES['profiles'][parsed(name)[1]['profile']]['env']

    def test_the_audited_jobs_serve_the_audited_best_and_the_timed_jobs_the_timed_best(self):
        flags = ('QWEN_FAST_TP4_COMMIT_LANES', 'QWEN_FAST_TP4_SHARD_VALUES', 'QWEN_FAST_TP4_GDN_GLUE',
                 'QWEN_FAST_TP4_GDN_BLOCK_CONV', 'QWEN_FAST_TP4_ATTN_FOLD', 'QWEN_FAST_FUSED_COMMIT',
                 'QWEN_FAST_FUSED_COMMIT_LIVE_BANKS', 'QWEN_FAST_QUAD_DRAFT')
        for name in ('N0-build', 'N1-best-gate-smoke-audited', 'N2-s3a-matrix-best-gate', 'N3-staggered-trigger-best-gate',
                     'N4-best-smoke-timed'):
            env = self.env(name)
            audited = name not in ('N0-build', 'N4-best-smoke-timed')
            with self.subTest(template=name):
                for flag in flags:
                    self.assertEqual(env.get(flag), '1', flag)
                for key in ('QWEN_FAST_VERIFY_T1_AUDIT', 'QWEN_FAST_VERIFY_T2_AUDIT'):
                    self.assertEqual(env[key], '1' if audited else '0')
                self.assertEqual(env.get('QWEN_FAST_TP4_VGLUE_AUDIT'), '1' if audited else None)
                self.assertEqual(env.get('QWEN_FAST_FUSED_COMMIT_AUDIT'), '1' if audited else None)

    def test_every_best_arm_is_a_gate_only_four_card_profile(self):
        for name, (profile, unused) in EXPECTED.items():
            entry = PROFILES['profiles'][profile]
            self.assertIs(entry['gate_only'], True, name)
            self.assertEqual(entry['mesh_device'], 'P150x4', name)

    def test_the_lanes_plans_are_the_lanes_gate_plans(self):
        import c2_serving_gate  # noqa: F401
        for name in ('N5-lanes-round-timing', 'N6-lanes-timing-l3', 'N7-lanes-exact-l1'):
            self.assertIn(parsed(name)[0]['C2_GATE_PLAN'], job.ALL_GATE_PLANS)
            self.assertIn('PROFILE NOTE', text_of(name))

    def test_the_cpu_suite_runs_this_module(self):
        with open(os.path.join(HERE, '..', '..', '.github', 'workflows', 'qwen-integration-cpu.yml'), encoding='utf-8') as handle:
            self.assertIn('test_tp4_next_window', handle.read())


if __name__ == '__main__':
    unittest.main()
