"""The tp4-serve-4 window: its job templates (scripts/ci/references/tp4-serve4-jobs), their order and what each asks of the tree.

B4 builds the image tp4-serve-4 and smokes the production profile; FX is the fix arm (audits off, the tail caps, the capture plug, the
stall watch and the handle guard), run five times in a row to accept a fix; F1 and F2 are the audit factorial's arms with the instruments
on. Every job resets the cards first. The templates are public, so they name no rig, card, address, registry or digest."""

import json
import os
import re
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import c2_serving_job as job  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
FOLDER = os.path.join(HERE, 'references', 'tp4-serve4-jobs')
with open(os.path.join(HERE, 'qwen_c2_profiles.json'), encoding='utf-8') as _handle:
    PROFILES = json.load(_handle)
NAMES = sorted(PROFILES['profiles'])
DEVICE_STEPS = ('cardm', 'smoke', 'gate', 'prefix', 'fabric', 'replay')
BANNED = re.compile(r'blackhole-[A-Za-z0-9]{8,}|thatch\.local|\d{1,3}(\.\d{1,3}){3}|sha256:[0-9a-f]{16}|[0-9a-f]{40,}|'
                    r'/dev/tenstorrent|home/|zot\.|@[A-Z0-9_]+@')
IMAGE = 'tp4-serve-4'
ORDERED = ('B4-build-smoke', 'FX-fix-arm', 'F1-diag-t1', 'F2-diag-t2')
SS = ['warmup', 'coding', 'long_real_text', 'concurrent4', 'concurrent4_steady', 'steady_resend', 'tool_call', 'stream_tool_call',
      'stream_reasoning', 'refused_n2', 'alive_after_refusal', 'stream_dropped', 'alive_after_drop']
CODE = ['concurrent4_code', 'concurrent4_code_equal']
DIAG_TESTS = ['warmup', 'coding', 'long_real_text', 'concurrent4_v164order', 'concurrent4_steady', 'replay_concurrent4']


def read_order():
    with open(os.path.join(FOLDER, 'ORDER.txt'), encoding='utf-8') as handle:
        return [line.split() for line in handle.read().splitlines() if line.strip() and not line.startswith('#')]


def text_of(name):
    with open(os.path.join(FOLDER, name + '.env'), encoding='utf-8') as handle:
        return handle.read()


def parsed(name):
    return job.read_job(job.parse_env(text_of(name)), NAMES)


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
        for name, mode, image, minutes in read_order():
            with self.subTest(job=name):
                self.assertIn(mode, ('stop', 'optional'))
                self.assertEqual(image, IMAGE)
                self.assertEqual(parsed(name)['tag'], IMAGE)
                self.assertTrue(minutes.isdigit() and 10 <= int(minutes) <= 80, minutes)
        modes = {row[0]: row[1] for row in read_order()}
        self.assertEqual(modes['B4-build-smoke'], 'stop')
        self.assertEqual(modes['FX-fix-arm'], 'optional')

    def test_the_order_says_five_consecutive_fix_runs_are_the_acceptance(self):
        with open(os.path.join(FOLDER, 'ORDER.txt'), encoding='utf-8') as handle:
            order = handle.read()
        self.assertIn('FIVE consecutive', order)
        self.assertIn('both hang shapes'.replace('both', 'BOTH'), order)
        self.assertIn('same image', order)


class TemplateTests(unittest.TestCase):
    def test_every_template_parses_and_is_lf_and_names_no_card_host_address_registry_digest_or_placeholder(self):
        for name in ORDERED:
            with self.subTest(template=name):
                self.assertEqual(parsed(name)['cards'], 'quad')
                with open(os.path.join(FOLDER, name + '.env'), 'rb') as handle:
                    self.assertNotIn(b'\r', handle.read(), name)
                self.assertIsNone(BANNED.search(text_of(name)), name)

    def test_every_job_resets_the_cards_first_and_opens_them_in_one_step(self):
        for name in ORDERED:
            actions = parsed(name)['actions'].split()
            with self.subTest(template=name):
                self.assertLessEqual(len(set(actions) & set(DEVICE_STEPS)), 1)
                self.assertIn('reset', actions, 'links train only at board init: a four-card job follows a reset')
                self.assertLess(actions.index('reset'), actions.index('smoke'))

    def test_the_profiles_and_actions_are_the_plans(self):
        want = {'B4-build-smoke': ('status reset build smoke', 'c2-packed-tp4'),
                'FX-fix-arm': ('reset smoke', 'c2-packed-tp4-speed-fix'),
                'F1-diag-t1': ('reset smoke', 'c2-packed-tp4-diag-t1'),
                'F2-diag-t2': ('reset smoke', 'c2-packed-tp4-diag-t2')}
        for name, (actions, profile) in want.items():
            outputs = parsed(name)
            with self.subTest(template=name):
                self.assertEqual((outputs['actions'], outputs['profile']), (actions, profile))
                self.assertIn(profile, PROFILES['profiles'])

    def test_only_the_production_profile_is_a_traffic_profile(self):
        self.assertNotIn('gate_only', PROFILES['profiles'][parsed('B4-build-smoke')['profile']])
        for name in ('FX-fix-arm', 'F1-diag-t1', 'F2-diag-t2'):
            self.assertIs(PROFILES['profiles'][parsed(name)['profile']].get('gate_only'), True, name)

    def test_the_image_is_built_by_the_first_job_only(self):
        for name in ORDERED[1:]:
            self.assertNotIn('build', parsed(name)['actions'].split(), name)
        self.assertIn('build', parsed('B4-build-smoke')['actions'].split())


class SmokeTests(unittest.TestCase):
    def test_every_named_test_is_one_the_smoke_knows_and_runs_in_the_order_the_template_lists_them(self):
        with open(os.path.join(HERE, 'c2_serving_smoke.py'), encoding='utf-8') as handle:
            smoke = handle.read()
        executed = re.findall(r"^\s*record\('([a-z0-9_]+)'", smoke, re.M)
        for name in ORDERED:
            listed = tests_of(name)
            with self.subTest(template=name):
                for test in listed:
                    self.assertIn("'%s'" % test, smoke)
                self.assertEqual([test for test in executed if test in listed], listed)

    def test_b4_keeps_the_ss_list_then_the_solo_references_then_the_coding_tests(self):
        self.assertEqual(tests_of('B4-build-smoke'), SS + ['concurrent4_solo'] + CODE)

    def test_fx_keeps_the_t2_list_v164s_sequence_then_solo_then_the_replay_shape_then_the_coding_tests(self):
        self.assertEqual(tests_of('FX-fix-arm'), SS + ['concurrent4_solo', 'replay_concurrent4'] + CODE)
        self.assertLess(tests_of('FX-fix-arm').index('concurrent4'), tests_of('FX-fix-arm').index('concurrent4_steady'))

    def test_the_factorial_arms_run_v164s_sequence(self):
        for name in ('F1-diag-t1', 'F2-diag-t2'):
            self.assertEqual(tests_of(name), DIAG_TESTS, name)


if __name__ == '__main__':
    unittest.main()
