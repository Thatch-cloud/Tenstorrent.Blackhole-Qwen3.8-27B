"""Tonight's four-card window (tp4/serve-3): its job templates (scripts/ci/references/tp4-tonight-jobs), their order and what each asks
of the tree.

T1 builds the image tp4-serve-3 and smokes the production configuration, T2 and T3 are the like-for-like speed arms (audits off with the
tail caps; G1), T4 the formal exactness of the caps, T5 / T5b / T5c the hang diagnosis and T6 the hand-back. The templates are public,
so they name no rig, card, address, registry or digest."""

import json
import os
import re
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import c2_serving_gate as gate  # noqa: E402
import c2_serving_job as job  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
FOLDER = os.path.join(HERE, 'references', 'tp4-tonight-jobs')
with open(os.path.join(HERE, 'qwen_c2_profiles.json'), encoding='utf-8') as _handle:
    PROFILES = json.load(_handle)
NAMES = sorted(PROFILES['profiles'])
DEVICE_STEPS = ('cardm', 'smoke', 'gate', 'prefix', 'fabric', 'replay')
BANNED = re.compile(r'blackhole-[A-Za-z0-9]{8,}|thatch\.local|\d{1,3}(\.\d{1,3}){3}|sha256:[0-9a-f]{16}|[0-9a-f]{40,}|'
                    r'/dev/tenstorrent|home/|zot\.|@[A-Z0-9_]+@')
IMAGE = 'tp4-serve-3'
ORDERED = ('T1-build-smoke', 'T2-smoke-audits-off', 'T3-g1-smoke', 'T4-gate-matrix', 'T5-diag-repro', 'T5b-diag-t1', 'T5c-diag-t2',
           'T6-handback-reset')
# v166's SS list, in its order: the like-for-like part of T1 and T2.
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
        self.assertEqual(modes['T1-build-smoke'], 'stop')
        self.assertEqual(modes['T6-handback-reset'], 'stop')
        self.assertEqual(modes['T5-diag-repro'], 'optional', 'T5 is a localisation job: its failure is a finding, not a stop')

    def test_the_diagnosis_is_the_last_device_work_before_the_hand_back(self):
        names = [row[0] for row in read_order()]
        self.assertEqual(names[-1], 'T6-handback-reset')
        self.assertLess(names.index('T5-diag-repro'), names.index('T5b-diag-t1'))
        self.assertLess(names.index('T5-diag-repro'), names.index('T5c-diag-t2'))
        self.assertLess(max(names.index(name) for name in names[:3]), names.index('T5-diag-repro'),
                        'T1-T3 are the service and speed jobs: the job that can wedge a board comes after them')

    def test_the_window_fits_the_plans_hours_of_runner_time(self):
        total = sum(int(row[3]) for row in read_order())
        self.assertTrue(180 <= total <= 300, total)


class TemplateTests(unittest.TestCase):
    def test_every_template_parses(self):
        for name in ORDERED:
            with self.subTest(template=name):
                self.assertEqual(parsed(name)['cards'], 'quad')

    def test_the_templates_are_lf_and_name_no_card_host_address_registry_digest_or_placeholder(self):
        for name in ORDERED:
            with open(os.path.join(FOLDER, name + '.env'), 'rb') as handle:
                self.assertNotIn(b'\r', handle.read(), name)
            self.assertIsNone(BANNED.search(text_of(name)), name)

    def test_a_quad_job_resets_first_and_opens_the_cards_in_one_step(self):
        for name in ORDERED:
            actions = parsed(name)['actions'].split()
            with self.subTest(template=name):
                self.assertLessEqual(len(set(actions) & set(DEVICE_STEPS)), 1)
                self.assertIn('reset', actions, 'links train only at board init: a four-card job follows a reset')

    def test_the_profiles_and_actions_are_the_plans(self):
        want = {'T1-build-smoke': ('status reset build smoke', 'c2-packed-tp4'),
                'T2-smoke-audits-off': ('reset smoke', 'c2-packed-tp4-speed'),
                'T3-g1-smoke': ('reset smoke', 'general-tp4'),
                'T4-gate-matrix': ('reset gate', 'c2-packed-tp4'),
                'T5-diag-repro': ('reset smoke', 'c2-packed-tp4-diag'),
                'T5b-diag-t1': ('reset smoke', 'c2-packed-tp4-diag-t1'),
                'T5c-diag-t2': ('reset smoke', 'c2-packed-tp4-diag-t2'),
                'T6-handback-reset': ('status reset', 'general')}
        for name, (actions, profile) in want.items():
            outputs = parsed(name)
            with self.subTest(template=name):
                self.assertEqual((outputs['actions'], outputs['profile']), (actions, profile))
                if profile != 'general':
                    self.assertIn(profile, PROFILES['profiles'])

    def test_the_gate_only_profiles_are_the_speed_and_diag_arms_and_the_traffic_profile_is_not(self):
        for name in ('T2-smoke-audits-off', 'T5-diag-repro', 'T5b-diag-t1', 'T5c-diag-t2'):
            self.assertIs(PROFILES['profiles'][parsed(name)['profile']].get('gate_only'), True, name)
        for name in ('T1-build-smoke', 'T4-gate-matrix'):
            self.assertNotIn('gate_only', PROFILES['profiles'][parsed(name)['profile']], name)
        self.assertNotIn('gate_only', PROFILES['profiles']['general-tp4'])


class SmokeTests(unittest.TestCase):
    def test_every_named_test_is_one_the_smoke_knows(self):
        with open(os.path.join(HERE, 'c2_serving_smoke.py'), encoding='utf-8') as handle:
            smoke = handle.read()
        for name in ('T1-build-smoke', 'T2-smoke-audits-off', 'T3-g1-smoke', 'T5-diag-repro', 'T5b-diag-t1', 'T5c-diag-t2'):
            for test in tests_of(name):
                with self.subTest(template=name, test=test):
                    self.assertIn("'%s'" % test, smoke)

    def test_t1_and_t2_keep_v166s_list_and_order_first_then_the_solo_references_then_the_coding_text_tests(self):
        t1, t2 = tests_of('T1-build-smoke'), tests_of('T2-smoke-audits-off')
        self.assertEqual(t1, SS + ['concurrent4_solo'] + CODE)
        self.assertEqual(t2, SS + ['concurrent4_solo', 'replay_concurrent4'] + CODE)
        self.assertEqual(t1[:len(SS)], t2[:len(SS)], 'T2 is T1 on the audits-off profile: the same like-for-like part in the same order')

    def test_the_smoke_runs_each_templates_tests_in_the_order_the_template_lists_them(self):
        # The smoke ignores the order of the list it is given: it runs record() calls in source order, filtered by the list.
        with open(os.path.join(HERE, 'c2_serving_smoke.py'), encoding='utf-8') as handle:
            executed = re.findall(r"^\s*record\('([a-z0-9_]+)'", handle.read(), re.M)
        for name in ('T1-build-smoke', 'T2-smoke-audits-off', 'T3-g1-smoke', 'T5-diag-repro', 'T5b-diag-t1', 'T5c-diag-t2'):
            with self.subTest(template=name):
                listed = tests_of(name)
                self.assertEqual([test for test in executed if test in listed], listed)
        for name in ('T1-build-smoke', 'T2-smoke-audits-off'):
            with self.subTest(template=name):
                ran = [test for test in executed if test in tests_of(name)]
                self.assertEqual(ran[:len(SS)], SS, "v166's like-for-like list, unchanged and first")

    def test_t3_runs_the_coding_text_tests_so_s2_and_g1_compare_on_coding_text(self):
        t3 = tests_of('T3-g1-smoke')
        self.assertEqual(t3, ['warmup', 'coding', 'long_real_text', 'concurrent4', 'concurrent4_steady'] + CODE)

    def test_the_diagnosis_runs_v164s_sequence_in_v164s_order(self):
        for name in ('T5-diag-repro', 'T5b-diag-t1', 'T5c-diag-t2'):
            self.assertEqual(tests_of(name), DIAG_TESTS, name)

    def test_the_smoke_check_judges_the_opt_in_tests_the_templates_name(self):
        import c2_smoke_check as check
        for name in ('concurrent4_solo', 'replay_concurrent4', 'concurrent4_v164order') + tuple(CODE):
            self.assertIn(name, check.CORE)


class GateTests(unittest.TestCase):
    def test_t4_is_the_matrix_on_the_traffic_profile_at_the_plans_lengths(self):
        outputs = parsed('T4-gate-matrix')
        self.assertEqual((outputs['gate_plan'], outputs['gate_lengths'], outputs['gate_max_tokens'], outputs['gate_jit']),
                         ('matrix', '4096,16384,32768,60000', '512', 'record'))
        lines = []
        code = gate.main(['--image', 'img', '--profile', outputs['profile'], '--plan', outputs['gate_plan'], '--cards', 'quad',
                          '--dry-run', '--results', os.path.join(HERE, 'no-results'), '--profiles',
                          os.path.join(HERE, 'qwen_c2_profiles.json'), '--max-tokens', outputs['gate_max_tokens'],
                          '--jit', outputs['gate_jit'], '--lengths', outputs['gate_lengths']], log=lines.append)
        self.assertEqual(code, 0, lines)
        arms = [json.loads(line)['docker'] for line in lines[1:]]
        self.assertTrue(arms)
        for argv in arms:
            # the profile's env (the caps among it) is applied inside the container by the profile name
            self.assertIn('QWEN_C2_PROFILE=c2-packed-tp4', argv)


if __name__ == '__main__':
    unittest.main()
