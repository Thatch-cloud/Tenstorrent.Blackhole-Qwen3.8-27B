"""The four-card verify-glue window: its job templates (scripts/ci/references/tp4-vglue-jobs), their order, and the profiles they use
(tp4/vglue).

The templates are public, so they name no rig, card, address, registry or digest. Every template parses with c2_serving_job, opens the
cards in ONE step and resets first, uses ONE image tag, the smokes name only tests c2_serving_smoke knows, the audited gate arm serves
the audited vglue profile and the timed arms the timed ones, and each per-lever arm serves the profile whose env carries exactly that
lever. No template pushes a tag: the window's driver does, from a throwaway commit.
"""

import json
import os
import re
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import c2_serving_job as job  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
FOLDER = os.path.join(HERE, 'references', 'tp4-vglue-jobs')
with open(os.path.join(HERE, 'qwen_c2_profiles.json'), encoding='utf-8') as _handle:
    PROFILES = json.load(_handle)
NAMES = sorted(PROFILES['profiles'])
DEVICE_STEPS = ('cardm', 'smoke', 'gate', 'prefix', 'fabric', 'replay')
BANNED = re.compile(r'blackhole-[A-Za-z0-9]{8,}|thatch\.local|\d{1,3}(\.\d{1,3}){3}|sha256:[0-9a-f]{16}|[0-9a-f]{40,}|'
                    r'/dev/tenstorrent|home/|zot\.')
EXPECTED = {
    'G0-build': 'c2-packed-tp4-speed-vglue',
    'G1-quad-smoke-timed': 'c2-packed-tp4-speed-vglue',
    'G2-s3a-matrix-gate': 'c2-packed-tp4-gate-vglue',
    'G3z-baseline-smoke-timed': 'c2-packed-tp4-speed',
    'G3a-c1a-smoke-timed': 'c2-packed-tp4-speed-vglue-c1a',
    'G3b-v4a-smoke-timed': 'c2-packed-tp4-speed-vglue-v4a',
    'G3c-v2-smoke-timed': 'c2-packed-tp4-speed-vglue-v2',
    'G3d-v1-smoke-timed': 'c2-packed-tp4-speed-vglue-v1',
    'G3e-v3a-smoke-timed': 'c2-packed-tp4-speed-vglue-v3a',
}
LEVER_OF = {'G3a-c1a-smoke-timed': {'QWEN_FAST_TP4_COMMIT_LANES'}, 'G3b-v4a-smoke-timed': {'QWEN_FAST_TP4_SHARD_VALUES'},
            'G3c-v2-smoke-timed': {'QWEN_FAST_TP4_GDN_GLUE'},
            'G3d-v1-smoke-timed': {'QWEN_FAST_TP4_GDN_GLUE', 'QWEN_FAST_TP4_GDN_BLOCK_CONV'},
            'G3e-v3a-smoke-timed': {'QWEN_FAST_TP4_ATTN_FOLD'}}
SMOKES = tuple(name for name in EXPECTED if 'smoke' in name)


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

    def test_build_then_the_timed_smoke_then_exactness_then_the_ab_arms(self):
        order = read_order()
        self.assertEqual([name for name, unused in order][:3], ['G0-build', 'G1-quad-smoke-timed', 'G2-s3a-matrix-gate'])
        self.assertEqual([name for name, mode in order if mode == 'stop'],
                         ['G0-build', 'G1-quad-smoke-timed', 'G2-s3a-matrix-gate'], 'a lever that is not exact is not a saving')
        self.assertEqual([name for name, mode in order if mode == 'soft'],
                         ['G3z-baseline-smoke-timed', 'G3a-c1a-smoke-timed', 'G3b-v4a-smoke-timed', 'G3c-v2-smoke-timed',
                          'G3d-v1-smoke-timed', 'G3e-v3a-smoke-timed'])
        self.assertEqual({mode for unused, mode in order}, {'stop', 'soft'})

    def test_every_template_parses_names_no_host_and_uses_the_one_image_tag(self):
        for name, unused in read_order():
            with self.subTest(template=name):
                values, outputs = parsed(name)
                self.assertIsNone(BANNED.search(text_of(name)), name)
                self.assertEqual(outputs['tag'], 'tp4-vglue-1')
                self.assertEqual(outputs['cards'], 'quad')
                self.assertEqual(outputs['profile'], EXPECTED[name])
                self.assertEqual(set(re.findall(r'@[A-Z0-9_]+@', text_of(name))), set())
        self.assertIn('PLACEHOLDER', text_of('G0-build'), 'the image tag is a placeholder, and G0 says so')
        with open(os.path.join(FOLDER, 'ORDER.txt'), encoding='utf-8') as handle:
            self.assertIsNone(BANNED.search(handle.read()))

    def test_the_build_opens_no_card_and_a_run_resets_first_and_opens_the_cards_once(self):
        for name, unused in read_order():
            actions = parsed(name)[1]['actions'].split()
            with self.subTest(template=name):
                if name.startswith('G0'):
                    self.assertEqual(actions, ['status', 'reset', 'build'])
                    continue
                self.assertEqual(len(set(actions) & set(DEVICE_STEPS)), 1)
                self.assertEqual(actions[0], 'reset', 'links train only at board init: a four-card run follows a reset')

    def test_every_template_states_its_duration_and_its_stop_rule(self):
        for name, mode in read_order():
            text = text_of(name)
            with self.subTest(template=name):
                self.assertRegex(text, r'# Duration \(estimate')
                self.assertRegex(text, r'\d+ / \d+ / \d+ min|most 90|\d+-\d+ min')
                self.assertIn('# STOP' if mode == 'stop' or name.startswith('G0') else '# SOFT', text)
                if name != 'G0-build':
                    self.assertIn('# Read:', text)

    def test_no_template_pushes_a_tag_or_names_a_remote(self):
        for name, unused in read_order():
            text = text_of(name)
            self.assertNotRegex(text, r'git push|git tag |gh workflow run|plink|ssh ')
            for line in text.splitlines():
                if 'experiment/' in line:
                    self.assertIn('push a tag experiment/c2-serving-vN', line, 'only the generic instruction, never a named tag')


class SmokeTests(unittest.TestCase):
    def test_the_smokes_name_tests_the_smoke_knows_and_run_the_same_three(self):
        with open(os.path.join(HERE, 'c2_serving_smoke.py'), encoding='utf-8') as handle:
            smoke = handle.read()
        tests = set()
        for name in SMOKES:
            outputs = parsed(name)[1]
            self.assertEqual(outputs['actions'].split(), ['reset', 'smoke'], name)
            listed = outputs['tests'].split(',')
            for test in listed:
                self.assertTrue("record('%s'" % test in smoke, (name, test))
            tests.add(tuple(listed))
        self.assertEqual(tests, {('warmup', 'coding', 'concurrent4')}, 'the A/B smokes run the same tests as G1')

    def test_the_matrix_is_the_s3a_matrix_with_the_strict_policy(self):
        values, outputs = parsed('G2-s3a-matrix-gate')
        self.assertEqual(outputs['actions'].split(), ['reset', 'gate'])
        self.assertEqual(values['C2_GATE_PLAN'], 'matrix')
        self.assertEqual(values['C2_GATE_LENGTHS'], '4096,16384,32768,60000')
        self.assertEqual(values['C2_GATE_MAX_TOKENS'], '512')
        self.assertNotIn('C2_GATE_POLICY', values, 'strict is the default: no decision record')


class ProfileTests(unittest.TestCase):
    def env(self, name):
        return PROFILES['profiles'][parsed(name)[1]['profile']]['env']

    def test_the_timed_arms_have_the_audits_off_and_the_gate_arm_on(self):
        for name in EXPECTED:
            env = self.env(name)
            audited = name == 'G2-s3a-matrix-gate'
            with self.subTest(template=name):
                for key in ('QWEN_FAST_VERIFY_T1_AUDIT', 'QWEN_FAST_VERIFY_T2_AUDIT'):
                    self.assertEqual(env[key], '1' if audited else '0')
                self.assertEqual(env.get('QWEN_FAST_TP4_VGLUE_AUDIT'), '1' if audited else None)

    def test_each_per_lever_arm_turns_on_exactly_its_lever(self):
        flags = ('QWEN_FAST_TP4_COMMIT_LANES', 'QWEN_FAST_TP4_SHARD_VALUES', 'QWEN_FAST_TP4_GDN_GLUE',
                 'QWEN_FAST_TP4_GDN_BLOCK_CONV', 'QWEN_FAST_TP4_ATTN_FOLD')
        for name, levers in LEVER_OF.items():
            env = self.env(name)
            with self.subTest(template=name):
                self.assertEqual({flag for flag in flags if env.get(flag) == '1'}, levers)
        self.assertEqual({flag for flag in flags if self.env('G3z-baseline-smoke-timed').get(flag)}, set(),
                         'the control carries no lever')
        for name in ('G0-build', 'G1-quad-smoke-timed', 'G2-s3a-matrix-gate'):
            self.assertEqual({flag for flag in flags if self.env(name).get(flag) == '1'}, set(flags), name)

    def test_every_arm_is_a_gate_only_four_card_profile(self):
        for name in EXPECTED:
            profile = PROFILES['profiles'][EXPECTED[name]]
            self.assertIs(profile['gate_only'], True, name)
            self.assertEqual(profile['mesh_device'], 'P150x4', name)


class SmokeRuleTests(unittest.TestCase):
    def test_a_fallen_back_lever_fails_the_smoke(self):
        import c2_smoke_check

        line = '[PINDIAG] tp4 vglue fell back site=sampler reason=RuntimeError: gather'
        problems, facts = c2_smoke_check.check('', 'ok'+'\n' + line + '\n', False)
        self.assertTrue(any('a vglue lever fell back' in text and 'site=sampler' in text for text in problems), problems)
        problems, facts = c2_smoke_check.check('', 'ok\n', False)
        self.assertFalse(any('vglue lever' in text for text in problems))

    def test_the_window_builds_from_the_stack_fix_merge(self):
        for name in ('G0-build.env', 'ORDER.txt'):
            with open(os.path.join(FOLDER, name), encoding='utf-8') as handle:
                self.assertIn('tp4/stack-fix', handle.read(), name)


if __name__ == '__main__':
    unittest.main()
