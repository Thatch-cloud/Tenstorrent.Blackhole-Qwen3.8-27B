"""The tp4/sampler window: its job templates (scripts/ci/references/tp4-sampler-jobs), their order and what each asks of the tree.

SB builds the image tp4-sampler-1 and smokes the production profile; SPa..SPe and STa..STe are the sampler isolation arms (the pinned
sampler once in the packed warm forward; the pinned sampler's ops also inside the packed verify capture) on the diag twins, five
repeats each; SPt and STt time each arm on the speed twin; SH hands the cards back. Every job resets the cards first. The templates are
public, so they name no rig, card, address, registry or digest."""

import json
import os
import re
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import c2_serving_job as job  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
FOLDER = os.path.join(HERE, 'references', 'tp4-sampler-jobs')
with open(os.path.join(HERE, 'qwen_c2_profiles.json'), encoding='utf-8') as _handle:
    PROFILES = json.load(_handle)
NAMES = sorted(PROFILES['profiles'])
DEVICE_STEPS = ('cardm', 'smoke', 'gate', 'prefix', 'fabric', 'replay')
BANNED = re.compile(r'blackhole-[A-Za-z0-9]{8,}|thatch\.local|\d{1,3}(\.\d{1,3}){3}|sha256:[0-9a-f]{16}|[0-9a-f]{40,}|'
                    r'/dev/tenstorrent|home/|zot\.|@[A-Z0-9_]+@')
IMAGE = 'tp4-sampler-1'
PREWARM_REPEATS = ('SPa-diag-sprewarm', 'SPb-diag-sprewarm', 'SPc-diag-sprewarm', 'SPd-diag-sprewarm', 'SPe-diag-sprewarm')
TRACE_REPEATS = ('STa-diag-strace', 'STb-diag-strace', 'STc-diag-strace', 'STd-diag-strace', 'STe-diag-strace')
TIMING = ('SPt-speed-sprewarm', 'STt-speed-strace')
ORDERED = ('SB-build-smoke',) + PREWARM_REPEATS + TRACE_REPEATS + TIMING + ('SH-handback-reset',)
WORK = ORDERED[:-1]
SS = ['warmup', 'coding', 'long_real_text', 'concurrent4', 'concurrent4_steady', 'steady_resend', 'tool_call', 'stream_tool_call',
      'stream_reasoning', 'refused_n2', 'alive_after_refusal', 'stream_dropped', 'alive_after_drop']
CODE = ['concurrent4_code', 'concurrent4_code_equal']
# The hang shapes in the order the smoke runs them (it runs concurrent4 before concurrent4_v164order, both before concurrent4_steady).
SHAPES = ['warmup', 'coding', 'long_real_text', 'concurrent4', 'concurrent4_v164order', 'concurrent4_steady', 'replay_concurrent4']
TIMING_TESTS = ['warmup', 'coding', 'concurrent4_code', 'concurrent4_code_equal', 'concurrent8_code']
PROFILE_OF = {'SB-build-smoke': 'c2-packed-tp4',
              **{name: 'c2-packed-tp4-diag-sprewarm' for name in PREWARM_REPEATS},
              **{name: 'c2-packed-tp4-diag-strace' for name in TRACE_REPEATS},
              'SPt-speed-sprewarm': 'c2-packed-tp4-speed-sprewarm', 'STt-speed-strace': 'c2-packed-tp4-speed-strace'}


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
        self.assertEqual(modes['SB-build-smoke'], 'stop')
        self.assertEqual(modes['SH-handback-reset'], 'stop')
        for name in ORDERED[1:-1]:
            self.assertEqual(modes[name], 'optional', name)

    def test_the_hand_back_is_last_and_the_order_says_it_runs_even_when_a_stop_job_halts_the_window(self):
        self.assertEqual(read_order()[-1][0], 'SH-handback-reset')
        with open(os.path.join(FOLDER, 'ORDER.txt'), encoding='utf-8') as handle:
            order = handle.read()
        self.assertIn('runs even when a stop job', order)
        text = text_of('SH-handback-reset')
        self.assertEqual(parsed('SH-handback-reset')['actions'], 'status reset')
        self.assertEqual(parsed('SH-handback-reset')['cards'], 'quad')
        for word in ('AUDITED production image', 'NEVER place a gate arm', 'fabric', ':latest retag', 'admin API'):
            self.assertIn(word, text)

    def test_the_order_says_five_consecutive_runs_per_arm_and_names_both_hang_shapes(self):
        with open(os.path.join(FOLDER, 'ORDER.txt'), encoding='utf-8') as handle:
            order = handle.read()
        for word in ('FIVE consecutive', 'same image', 'concurrent4_steady', 'replay_concurrent4', 'concurrent4_v164order'):
            self.assertIn(word, order)
        self.assertEqual(len(PREWARM_REPEATS), 5)
        self.assertEqual(len(TRACE_REPEATS), 5)


class TemplateTests(unittest.TestCase):
    def test_every_template_parses_and_is_lf_and_names_no_card_host_address_registry_digest_or_placeholder(self):
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
        for name, profile in PROFILE_OF.items():
            outputs = parsed(name)
            with self.subTest(template=name):
                self.assertEqual(outputs['actions'], 'status reset build smoke' if name == 'SB-build-smoke' else 'reset smoke')
                self.assertEqual(outputs['profile'], profile)
                self.assertIn(profile, PROFILES['profiles'])

    def test_only_the_production_profile_is_a_traffic_profile(self):
        self.assertNotIn('gate_only', PROFILES['profiles'][parsed('SB-build-smoke')['profile']])
        for name in ORDERED[1:-1]:
            self.assertIs(PROFILES['profiles'][parsed(name)['profile']].get('gate_only'), True, name)

    def test_the_diag_repeats_run_the_diag_twins_and_the_timing_jobs_the_speed_twins(self):
        for name in PREWARM_REPEATS + TRACE_REPEATS:
            self.assertTrue(parsed(name)['profile'].startswith('c2-packed-tp4-diag-s'), name)
        for name in TIMING:
            self.assertTrue(parsed(name)['profile'].startswith('c2-packed-tp4-speed-s'), name)
        self.assertEqual({parsed(name)['profile'] for name in PREWARM_REPEATS}, {'c2-packed-tp4-diag-sprewarm'})
        self.assertEqual({parsed(name)['profile'] for name in TRACE_REPEATS}, {'c2-packed-tp4-diag-strace'})

    def test_each_arms_template_names_its_flag(self):
        for name in PREWARM_REPEATS + ('SPt-speed-sprewarm',):
            self.assertIn('QWEN_FAST_PACKED_SAMPLER_PREWARM=1', text_of(name), name)
            self.assertNotIn('QWEN_FAST_PACKED_SAMPLER_IN_TRACE=1', text_of(name), name)
        for name in TRACE_REPEATS + ('STt-speed-strace',):
            self.assertIn('QWEN_FAST_PACKED_SAMPLER_IN_TRACE=1', text_of(name), name)
            self.assertNotIn('QWEN_FAST_PACKED_SAMPLER_PREWARM=1', text_of(name), name)

    def test_the_image_is_built_by_the_first_job_only(self):
        for name in ORDERED[1:]:
            self.assertNotIn('build', parsed(name)['actions'].split(), name)
        self.assertIn('build', parsed('SB-build-smoke')['actions'].split())


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

    def test_sb_keeps_b4s_list(self):
        self.assertEqual(tests_of('SB-build-smoke'), SS + ['concurrent4_solo'] + CODE)

    def test_the_diag_repeats_run_the_hang_shapes_and_the_timing_jobs_the_coding_tests(self):
        for name in PREWARM_REPEATS + TRACE_REPEATS:
            self.assertEqual(tests_of(name), SHAPES, name)
        for name in TIMING:
            self.assertEqual(tests_of(name), TIMING_TESTS, name)
            for test in ('concurrent4_code', 'concurrent4_code_equal', 'concurrent8_code'):
                self.assertIn(test, tests_of(name))


if __name__ == '__main__':
    unittest.main()
