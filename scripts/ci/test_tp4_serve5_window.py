"""The tp4-serve-6 window: its job templates (scripts/ci/references/tp4-serve5-jobs), their order and what each asks of the tree.

B5 builds the image tp4-serve-6 (the capture plug seal fix and ttexalens in the image) and smokes the production profile; FX5a..FX5e are
the fix arm (audits off, the capture plug, the stall watch) run five times IDENTICALLY, the acceptance bar being five consecutive passes;
D5 repeats the diagnosis with working triage and runs only if an FX5 run fails. Every job resets the cards first. The templates are
public, so they name no rig, card, address, registry or digest."""

import json
import os
import re
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import c2_serving_job as job  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
FOLDER = os.path.join(HERE, 'references', 'tp4-serve5-jobs')
FOLDER4 = os.path.join(HERE, 'references', 'tp4-serve4-jobs')
PROFILES_PATH = os.path.join(HERE, 'qwen_c2_profiles.json')
BANNED = re.compile(r'blackhole-[A-Za-z0-9]{8,}|thatch\.local|\d{1,3}(\.\d{1,3}){3}|sha256:[0-9a-f]{16}|[0-9a-f]{40,}|'
                    r'/dev/tenstorrent|home/|zot\.|@[A-Z0-9_]+@')
IMAGE = 'tp4-serve-6'
FX5 = ('FX5a-fix-arm', 'FX5b-fix-arm', 'FX5c-fix-arm', 'FX5d-fix-arm', 'FX5e-fix-arm')
ORDERED = ('B5-build-smoke',) + FX5 + ('D5-diag-repro', 'H5-handback-reset')
WORK = ORDERED[:-1]
HANG_SHAPES = ['warmup', 'coding', 'long_real_text', 'concurrent4', 'concurrent4_v164order', 'concurrent4_steady',
               'replay_concurrent4', 'concurrent4_code']
DEVICE_STEPS = ('cardm', 'smoke', 'gate', 'prefix', 'fabric', 'replay')


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
    return job.read_job(job.parse_env(text_of(name, folder)), sorted(profiles()))


def tests_of(name):
    return parsed(name)['tests'].split(',')


class OrderTests(unittest.TestCase):
    def test_every_template_is_in_the_order_once_with_four_columns_and_the_order_names_no_other(self):
        rows = read_order()
        self.assertTrue(all(len(row) == 4 for row in rows))
        on_disk = sorted(name[:-4] for name in os.listdir(FOLDER) if name.endswith('.env'))
        ordered = [row[0] for row in rows]
        self.assertEqual(sorted(ordered), on_disk)
        self.assertEqual(ordered, list(ORDERED))

    def test_the_modes_images_and_minutes(self):
        modes = {}
        for name, mode, image, minutes in read_order():
            modes[name] = mode
            with self.subTest(job=name):
                self.assertIn(mode, ('stop', 'optional'))
                self.assertEqual(image, IMAGE)
                self.assertEqual(parsed(name)['tag'], IMAGE)
                self.assertTrue(minutes.isdigit() and 10 <= int(minutes) <= 80, minutes)
        self.assertEqual(modes['B5-build-smoke'], 'stop')
        self.assertEqual(modes['H5-handback-reset'], 'stop')
        for name in FX5 + ('D5-diag-repro',):
            self.assertEqual(modes[name], 'optional', name)

    def test_the_build_is_the_first_job_and_the_only_one(self):
        self.assertEqual(ORDERED[0], 'B5-build-smoke')
        self.assertIn('build', parsed('B5-build-smoke')['actions'].split())
        for name in ORDERED[1:]:
            self.assertNotIn('build', parsed(name)['actions'].split(), name)

    def test_the_hand_back_is_last_and_the_order_says_it_runs_even_when_a_stop_job_halts_the_window(self):
        self.assertEqual(read_order()[-1][0], 'H5-handback-reset')
        with open(os.path.join(FOLDER, 'ORDER.txt'), encoding='utf-8') as handle:
            order = handle.read()
        self.assertIn('runs even when a stop job', order)
        text = text_of('H5-handback-reset')
        self.assertEqual(parsed('H5-handback-reset')['actions'], 'status reset')
        for word in ('AUDITED production image', 'NEVER place a gate arm', 'fabric', ':latest retag', 'admin API'):
            self.assertIn(word, text)

    def test_the_order_says_five_consecutive_passes_on_the_same_image_and_d5_runs_only_after_a_failure(self):
        with open(os.path.join(FOLDER, 'ORDER.txt'), encoding='utf-8') as handle:
            order = handle.read()
        self.assertIn('FIVE consecutive', order)
        self.assertIn('same image', order)
        self.assertIn('BOTH hang shapes', order)
        self.assertIn('ONLY if an FX5 run fails or stalls', order)


class TemplateTests(unittest.TestCase):
    def test_every_template_parses_and_is_lf_and_names_no_card_host_address_registry_or_digest(self):
        for name in ORDERED:
            with self.subTest(template=name):
                self.assertEqual(parsed(name)['cards'], 'quad')
                with open(os.path.join(FOLDER, name + '.env'), 'rb') as handle:
                    self.assertNotIn(b'\r', handle.read(), name)
                self.assertIsNone(BANNED.search(text_of(name)), name)

    def test_every_job_resets_the_cards_first_and_opens_them_in_one_step(self):
        for name in WORK:
            actions = parsed(name)['actions'].split()
            with self.subTest(template=name):
                self.assertLessEqual(len(set(actions) & set(DEVICE_STEPS)), 1)
                self.assertIn('reset', actions, 'links train only at board init: a four-card job follows a reset')
                self.assertLess(actions.index('reset'), actions.index('smoke'))

    def test_the_profiles_and_actions_are_the_plans(self):
        known = profiles()
        want = {'B5-build-smoke': ('status reset build smoke', 'c2-packed-tp4'), 'D5-diag-repro': ('reset smoke', 'c2-packed-tp4-diag')}
        want.update({name: ('reset smoke', 'c2-packed-tp4-speed-fix') for name in FX5})
        for name, (actions, profile) in want.items():
            with self.subTest(template=name):
                self.assertEqual((parsed(name)['actions'], parsed(name)['profile']), (actions, profile))
                self.assertIn(profile, known)
        self.assertNotIn('gate_only', known['c2-packed-tp4'])
        for name in FX5 + ('D5-diag-repro',):
            self.assertIs(known[parsed(name)['profile']].get('gate_only'), True, name)

    def test_the_fix_arm_turns_the_plug_and_the_stall_watch_on(self):
        env = profiles()['c2-packed-tp4-speed-fix']['env']
        self.assertEqual(env.get('QWEN_FAST_CAPTURE_PLUG'), '1')
        self.assertTrue(env.get('QWEN_FAST_STALL_DEADLINE_S'))


class FixArmTests(unittest.TestCase):
    def test_the_five_fix_jobs_are_identical_but_for_their_letter(self):
        texts = [text_of(name).replace('# FX5%s:' % name[3], '# FX5x:') for name in FX5]
        self.assertEqual(len(set(texts)), 1)
        for name in FX5:
            self.assertIn('FX5%s:' % name[3], text_of(name))

    def test_the_fix_arm_runs_exactly_the_hang_shapes_in_order(self):
        for name in FX5:
            self.assertEqual(tests_of(name), HANG_SHAPES, name)
        tests = tests_of(FX5[0])
        self.assertLess(tests.index('concurrent4'), tests.index('concurrent4_steady'), 'the v164 sequence')
        self.assertIn('replay_concurrent4', tests)

    def test_b5_keeps_the_b4_test_list(self):
        self.assertEqual(tests_of('B5-build-smoke'), parsed('B4-build-smoke', FOLDER4)['tests'].split(','))

    def test_d5_is_d0s_reproduction(self):
        self.assertEqual(tests_of('D5-diag-repro'), parsed('D0-diag-repro', FOLDER4)['tests'].split(','))


class SmokeTests(unittest.TestCase):
    def test_every_named_test_is_one_the_smoke_knows_and_runs_in_the_order_the_template_lists_them(self):
        with open(os.path.join(HERE, 'c2_serving_smoke.py'), encoding='utf-8') as handle:
            smoke = handle.read()
        executed = re.findall(r"^\s*record\('([a-z0-9_]+)'", smoke, re.M)
        for name in WORK:
            listed = tests_of(name)
            with self.subTest(template=name):
                for test in listed:
                    self.assertIn("'%s'" % test, smoke)
                self.assertEqual([test for test in executed if test in listed], listed)


if __name__ == '__main__':
    unittest.main()
