"""The tp4-serve-7 window: its job templates (scripts/ci/references/tp4-serve7-jobs), their order and what each asks of the tree.

tp4-serve-7 puts the verified winning recipe into production: the profile c2-packed-tp4 with the verify audits OFF and the pinned
sampler recorded in the verify trace (verified on tp4-serve-6 as the gate-only c2-packed-tp4-speed-strace). B7 builds the image and smokes
the production profile on every hang shape, M7 is the S3a matrix gate on the production profile, SR7 is the platform replay of the thin
layer image (the one @...@ placeholder, filled by whoever drives the window), H7 hands the cards back. The templates are public, so they
name no rig, card, address, registry or digest."""

import json
import os
import re
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import c2_serving_job as job  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
FOLDER = os.path.join(HERE, 'references', 'tp4-serve7-jobs')
FOLDER5 = os.path.join(HERE, 'references', 'tp4-serve5-jobs')
PROFILES_PATH = os.path.join(HERE, 'qwen_c2_profiles.json')
BUILD_SCRIPT = os.path.join(HERE, 'build-c2-serving-image.sh')
BANNED = re.compile(r'blackhole-[A-Za-z0-9]{8,}|thatch\.local|\d{1,3}(\.\d{1,3}){3}|sha256:[0-9a-f]{16}|[0-9a-f]{40,}|'
                    r'/dev/tenstorrent|home/|zot\.')
PLACEHOLDERS = {'@THIN_LAYER_IMAGE@': 'thin-layer-image-ref'}
IMAGE = 'tp4-serve-7'
ORDERED = ('B7-build-smoke', 'M7-matrix', 'SR7-platform-replay', 'H7-handback-reset')
PRODUCTION = 'c2-packed-tp4'
# B5's list, plus the hang shapes it did not run, each inserted where the smoke executes it.
B7_TESTS = ['warmup', 'coding', 'long_real_text', 'concurrent4', 'concurrent4_v164order', 'concurrent4_steady', 'steady_resend',
            'tool_call', 'stream_tool_call', 'stream_reasoning', 'refused_n2', 'alive_after_refusal', 'stream_dropped',
            'alive_after_drop', 'concurrent4_solo', 'replay_concurrent4', 'concurrent4_code', 'concurrent4_code_equal',
            'concurrent8_code']


def profiles():
    with open(PROFILES_PATH, encoding='utf-8') as handle:
        return json.load(handle)['profiles']


def read_order():
    with open(os.path.join(FOLDER, 'ORDER.txt'), encoding='utf-8') as handle:
        return [line.split() for line in handle.read().splitlines() if line.strip() and not line.startswith('#')]


def text_of(name, folder=FOLDER):
    with open(os.path.join(folder, name + '.env'), encoding='utf-8') as handle:
        return handle.read()


def parsed(name, folder=FOLDER):
    text = text_of(name, folder)
    for placeholder, value in PLACEHOLDERS.items():
        text = text.replace(placeholder, value)
    return job.read_job(job.parse_env(text), sorted(profiles()))


class OrderTests(unittest.TestCase):
    def test_every_template_is_in_the_order_once_with_four_columns_and_the_order_names_no_other(self):
        rows = read_order()
        self.assertTrue(all(len(row) == 4 for row in rows))
        on_disk = sorted(name[:-4] for name in os.listdir(FOLDER) if name.endswith('.env'))
        ordered = [row[0] for row in rows]
        self.assertEqual(sorted(ordered), on_disk)
        self.assertEqual(ordered, list(ORDERED))

    def test_the_modes_images_and_minutes(self):
        for name, mode, image, minutes in read_order():
            with self.subTest(job=name):
                self.assertEqual(mode, 'stop')
                self.assertEqual(image, IMAGE)
                self.assertEqual(parsed(name)['tag'], IMAGE)
                self.assertTrue(minutes.isdigit() and 10 <= int(minutes) <= 80, minutes)

    def test_the_build_is_the_first_job_and_the_only_one(self):
        self.assertIn('build', parsed('B7-build-smoke')['actions'].split())
        for name in ORDERED[1:]:
            self.assertNotIn('build', parsed(name)['actions'].split(), name)

    def test_the_hand_back_is_last_and_places_the_new_image_only_after_all_three_passed(self):
        self.assertEqual(read_order()[-1][0], 'H7-handback-reset')
        with open(os.path.join(FOLDER, 'ORDER.txt'), encoding='utf-8') as handle:
            order = handle.read()
        self.assertIn('runs even when a stop job', order)
        text = text_of('H7-handback-reset')
        self.assertEqual(parsed('H7-handback-reset')['actions'], 'status reset')
        for word in ('B7, M7 and SR7 all passed', 'NEVER place a gate arm', 'fabric', ':latest retag', 'admin API'):
            self.assertIn(word, text)


class TemplateTests(unittest.TestCase):
    def test_every_template_parses_and_is_lf_and_names_no_card_host_address_registry_or_digest(self):
        for name in ORDERED:
            with self.subTest(template=name):
                self.assertEqual(parsed(name)['cards'], 'quad')
                with open(os.path.join(FOLDER, name + '.env'), 'rb') as handle:
                    self.assertNotIn(b'\r', handle.read(), name)
                self.assertIsNone(BANNED.search(text_of(name)), name)

    def test_only_the_replay_names_a_placeholder_and_it_is_the_thin_layer_image(self):
        for name in ORDERED:
            found = set(re.findall(r'@[A-Z0-9_]+@', text_of(name)))
            self.assertEqual(found, {'@THIN_LAYER_IMAGE@'} if name == 'SR7-platform-replay' else set(), name)
        self.assertTrue(parsed('SR7-platform-replay')['platform_image'])

    def test_every_device_job_resets_the_cards_first_and_opens_them_in_one_step(self):
        for name in ORDERED:
            actions = parsed(name)['actions'].split()
            with self.subTest(template=name):
                self.assertLessEqual(len(set(actions) & set(('cardm', 'smoke', 'gate', 'prefix', 'fabric', 'replay'))), 1)
                self.assertIn('reset', actions, 'links train only at board init: a four-card job follows a reset')

    def test_the_profiles_and_actions_are_the_plans_and_every_job_runs_the_production_profile(self):
        known = profiles()
        want = {'B7-build-smoke': 'status reset build smoke', 'M7-matrix': 'reset gate', 'SR7-platform-replay': 'reset replay',
                'H7-handback-reset': 'status reset'}
        for name, actions in want.items():
            with self.subTest(template=name):
                self.assertEqual(parsed(name)['actions'], actions)
        for name in ('B7-build-smoke', 'M7-matrix'):
            self.assertEqual(parsed(name)['profile'], PRODUCTION, name)
        self.assertEqual(parsed('SR7-platform-replay')['replay_profile'], PRODUCTION)
        self.assertNotIn('gate_only', known[PRODUCTION])

    def test_the_production_profile_is_the_recipe_the_window_exercises(self):
        env = profiles()[PRODUCTION]['env']
        self.assertEqual((env['QWEN_FAST_VERIFY_T1_AUDIT'], env['QWEN_FAST_VERIFY_T2_AUDIT']), ('0', '0'))
        self.assertEqual(env['QWEN_FAST_PACKED_SAMPLER_IN_TRACE'], '1')
        self.assertEqual(env['QWEN_FAST_BUDGET_CAP'], '1')
        self.assertNotIn('QWEN_C2_GATE_PROFILE', env)
        self.assertEqual((profiles()[PRODUCTION]['max_prompt_tokens'], profiles()[PRODUCTION]['min_answer_tokens']), (123136, 8192))


class SmokeTests(unittest.TestCase):
    def test_b7_is_b5s_list_plus_the_four_hang_shapes_in_the_smokes_order(self):
        listed = parsed('B7-build-smoke')['tests'].split(',')
        self.assertEqual(listed, B7_TESTS)
        b5 = parsed('B5-build-smoke', FOLDER5)['tests'].split(',')
        self.assertEqual([test for test in listed if test in b5], b5, 'B5\'s list, in B5\'s order')
        self.assertEqual([test for test in listed if test not in b5], ['concurrent4_v164order', 'replay_concurrent4', 'concurrent8_code'])

    def test_every_named_test_is_one_the_smoke_knows_and_runs_in_the_order_the_template_lists_them(self):
        with open(os.path.join(HERE, 'c2_serving_smoke.py'), encoding='utf-8') as handle:
            smoke = handle.read()
        executed = re.findall(r"^\s*record\('([a-z0-9_]+)'", smoke, re.M)
        listed = parsed('B7-build-smoke')['tests'].split(',')
        for test in listed:
            self.assertIn("'%s'" % test, smoke)
        self.assertEqual([test for test in executed if test in listed], listed)

    def test_the_matrix_is_the_s3a_gate(self):
        values = parsed('M7-matrix')
        self.assertEqual((values['gate_plan'], values['gate_lengths'], values['gate_max_tokens']), ('matrix', '4096,16384,32768,60000', '512'))


class BuildScriptTests(unittest.TestCase):
    def test_the_build_is_prune_safe_the_image_is_tagged_before_provenance_and_the_provisional_tag_is_always_removed(self):
        with open(BUILD_SCRIPT, encoding='utf-8') as handle:
            text = handle.read()
        build = text.index('docker build')
        provisional_tag = text.index('docker tag "$built" "$provisional"')
        provenance = text.index('c2_image_provenance.py" "${provenance[@]}"')
        final = text.index('docker tag "$built" "$image"')
        self.assertIn('--tag "$provisional"', text[build:provisional_tag])
        self.assertLess(build, provisional_tag)
        self.assertLess(provisional_tag, provenance)
        self.assertLess(provenance, final)
        self.assertEqual(text.count('provisional="$image-unverified"'), 1)
        self.assertIn('docker rmi "$provisional"', text[text.index('cleanup() {'):text.index('trap cleanup EXIT')])
        self.assertIn('docker rmi "$provisional"', text[final:])
        self.assertIn('set -euo pipefail', text)


if __name__ == '__main__':
    unittest.main()
