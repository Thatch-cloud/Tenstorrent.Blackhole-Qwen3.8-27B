"""The tp4-seats8 window: its job templates (scripts/ci/references/tp4-seats8-jobs), their order and what each asks of the tree.

tp4/seats8 serves eight seats on two 64-row M3 blocks (QWEN_FAST_M3_BLOCKS=2). The window has two stages: a PROBE first (the production profile
with the flag off on the new image, the first eight-seat attach on the audited gate profile, one paired timing, memory at eight live: the cheapest
kill signals), then the QUALIFICATION (the S3a matrix at eight users on the gate and the traffic profile, five audits-off hang runs on the diag
twin, staggered, churn of 16 sessions over 8 seats, the second timing pair). The handback is last and runs even when a stop job halts the window.
The templates are public, so they name no rig, card, address, registry or digest."""

import json
import os
import re
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import c2_serving_gate as driver  # noqa: E402
import c2_serving_job as job  # noqa: E402
import test_c2_serving_gate as base  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
FOLDER = os.path.join(HERE, 'references', 'tp4-seats8-jobs')
with open(os.path.join(HERE, 'qwen_c2_profiles.json'), encoding='utf-8') as _handle:
    PROFILES = json.load(_handle)
NAMES = sorted(PROFILES['profiles'])
DEVICE_STEPS = ('cardm', 'smoke', 'gate', 'prefix', 'fabric', 'replay')
BANNED = re.compile(r'blackhole-[A-Za-z0-9]{8,}|thatch\.local|\d{1,3}(\.\d{1,3}){3}|sha256:[0-9a-f]{16}|[0-9a-f]{40,}|'
                    r'/dev/tenstorrent|home/|zot\.|@[A-Z0-9_]+@')
IMAGE = 'tp4-seats8'
PRODUCTION = 'c2-packed-tp4'
EIGHT, EIGHT_GATE, EIGHT_TIME, EIGHT_DIAG = ('c2-packed-tp4-8', 'c2-packed-tp4-8-gate', 'c2-packed-tp4-8-time-gate',
                                             'c2-packed-tp4-8-diag-strace')
FOUR_TIME = 'c2-packed-tp4-speed-strace'
PROBE = ('S8-0-flag-off-smoke', 'S8-1-seats8-attach-audited', 'S8-6-memory8', 'S8-7a-seats4-timed', 'S8-7b-seats8-timed')
HANG = tuple('S8-3%s-hang8-strace' % letter for letter in 'abcde')
QUALIFICATION = ('S8-2a-matrix8-gate', 'S8-2b-matrix8-traffic') + HANG + ('S8-4-staggered8', 'S8-5-churn16', 'S8-7c-seats4-timed',
                                                                         'S8-7d-seats8-timed')
HANDBACK = 'H8-handback-reset'
ORDERED = PROBE + QUALIFICATION + (HANDBACK,)
WORK = ORDERED[:-1]
TIMED = ('S8-7a-seats4-timed', 'S8-7b-seats8-timed', 'S8-7c-seats4-timed', 'S8-7d-seats8-timed')
PROFILE_OF = {'S8-0-flag-off-smoke': PRODUCTION, 'S8-1-seats8-attach-audited': EIGHT_GATE, 'S8-7a-seats4-timed': FOUR_TIME,
              'S8-7b-seats8-timed': EIGHT_TIME, 'S8-6-memory8': EIGHT, 'S8-2a-matrix8-gate': EIGHT_GATE,
              'S8-2b-matrix8-traffic': EIGHT, 'S8-4-staggered8': EIGHT_GATE, 'S8-5-churn16': EIGHT_GATE,
              'S8-7c-seats4-timed': FOUR_TIME, 'S8-7d-seats8-timed': EIGHT_TIME, **{name: EIGHT_DIAG for name in HANG}}
S80_TESTS = ['warmup', 'coding', 'concurrent4_steady', 'steady_resend', 'replay_concurrent4', 'concurrent4_code_equal']
S81_TESTS = ['warmup', 'coding', 'concurrent4_solo', 'concurrent4_code_equal', 'concurrent8_code', 'concurrent8_code_equal',
             'concurrent5_split', 'concurrent8_drain']
HANG_TESTS = ['warmup', 'concurrent4_steady', 'concurrent8_steady', 'steady_resend', 'replay_concurrent4', 'replay_concurrent8',
              'concurrent5_split', 'concurrent8_drain']
TIMED_TESTS = ['warmup', 'coding', 'concurrent4_code_equal', 'concurrent4_code_32k', 'concurrent8_code', 'concurrent8_code_32k']
MATRIX_LENGTHS = '4096,8192,16384,24576,32768,49152,60000,120000'


def read_order():
    with open(os.path.join(FOLDER, 'ORDER.txt'), encoding='utf-8') as handle:
        return [line.split() for line in handle.read().splitlines() if line.strip() and not line.startswith('#')]


def order_text():
    with open(os.path.join(FOLDER, 'ORDER.txt'), encoding='utf-8') as handle:
        return handle.read()


def text_of(name):
    with open(os.path.join(FOLDER, name + '.env'), encoding='utf-8') as handle:
        return handle.read()


def parsed(name):
    return job.read_job(job.parse_env(text_of(name)), NAMES)


class OrderTests(unittest.TestCase):
    def test_every_template_is_in_the_order_once_with_four_columns_and_the_order_names_no_other(self):
        rows = read_order()
        self.assertTrue(all(len(row) == 4 for row in rows))
        on_disk = sorted(name[:-4] for name in os.listdir(FOLDER) if name.endswith('.env'))
        ordered = [row[0] for row in rows]
        self.assertEqual(sorted(ordered), on_disk)
        self.assertEqual(ordered, list(ORDERED))

    def test_the_modes_images_and_minutes(self):
        step_minutes = base.WorkflowTests.budget_literals()['step_minutes']
        for name, mode, image, minutes in read_order():
            with self.subTest(job=name):
                self.assertEqual(mode, 'optional' if name in TIMED else 'stop')
                self.assertEqual(image, IMAGE)
                self.assertEqual(parsed(name)['tag'], IMAGE)
                self.assertTrue(minutes.isdigit() and 10 <= int(minutes) <= step_minutes, minutes)

    def test_two_stages_the_probe_first_with_the_one_paired_timing_right_after_the_first_attach(self):
        ordered = [row[0] for row in read_order()]
        self.assertEqual(ordered[:5], list(PROBE))
        self.assertEqual(ordered[5:-1], list(QUALIFICATION))
        self.assertEqual(ordered.index('S8-6-memory8'), ordered.index('S8-1-seats8-attach-audited') + 1)
        self.assertEqual(ordered.index('S8-7a-seats4-timed'), ordered.index('S8-6-memory8') + 1)
        self.assertEqual(ordered.index('S8-7b-seats8-timed'), ordered.index('S8-7a-seats4-timed') + 1)
        self.assertLess(ordered.index('S8-6-memory8'), ordered.index('S8-2a-matrix8-gate'))
        text = order_text()
        for word in ('PROBE', 'QUALIFICATION', 'about 4 hours', 'kill signal'):
            self.assertIn(word, text)

    def test_the_hand_back_is_last_runs_even_after_a_stop_and_re_places_todays_production_recipe_never_a_gate_arm(self):
        self.assertEqual(read_order()[-1][0], HANDBACK)
        self.assertIn('runs even when a stop job halts the window', order_text())
        self.assertEqual(parsed(HANDBACK)['actions'], 'status reset')
        text = text_of(HANDBACK)
        for word in ("TODAY'S PRODUCTION RECIPE", 'NEVER a gate arm', ':latest', 'admin API', 'fabric', 'runs even when a stop job'):
            self.assertIn(word, text)
        self.assertNotIn('C2_PROFILE=', text, 'the handback serves nothing')


class TemplateTests(unittest.TestCase):
    def test_every_template_parses_and_is_lf_and_names_no_card_host_address_registry_digest_or_placeholder(self):
        for name in ORDERED:
            with self.subTest(template=name):
                self.assertEqual(parsed(name)['cards'], 'quad')
                with open(os.path.join(FOLDER, name + '.env'), 'rb') as handle:
                    self.assertNotIn(b'\r', handle.read(), name)
                self.assertIsNone(BANNED.search(text_of(name)), name)
        with open(os.path.join(FOLDER, 'ORDER.txt'), 'rb') as handle:
            data = handle.read()
        self.assertNotIn(b'\r', data)
        self.assertIsNone(BANNED.search(data.decode('utf-8')))

    def test_every_job_resets_the_cards_first_and_opens_them_in_one_step(self):
        for name in ORDERED:
            actions = parsed(name)['actions'].split()
            with self.subTest(template=name):
                self.assertLessEqual(len(set(actions) & set(DEVICE_STEPS)), 1)
                self.assertIn('reset', actions, 'links train only at board init: a four-card job follows a reset')

    def test_no_job_builds_and_only_the_first_takes_production_down(self):
        for name in ORDERED:
            self.assertNotIn('build', parsed(name)['actions'].split(), name)
        for name in ORDERED[1:]:
            self.assertNotIn('unserve', parsed(name)['actions'].split(), name)
        actions = parsed(ORDERED[0])['actions'].split()
        self.assertLess(actions.index('unserve'), actions.index('reset'))
        self.assertIn('PRODUCTION IS LIVE ON THE CARDS', order_text())
        self.assertIn('admin API', text_of(ORDERED[0]))
        self.assertIn('ONLY job that takes production down', text_of(ORDERED[0]))

    def test_the_actions_and_profiles_are_the_plans(self):
        for name in WORK:
            with self.subTest(template=name):
                self.assertEqual(parsed(name)['profile'], PROFILE_OF[name])
                self.assertIn(PROFILE_OF[name], NAMES)
        self.assertEqual(parsed('S8-0-flag-off-smoke')['actions'], 'status unserve reset smoke')
        for name in ORDERED[1:]:
            want = 'status reset' if name == HANDBACK else ('reset gate' if parsed(name)['gate_plan'] != 'bringup' else 'reset smoke')
            self.assertEqual(parsed(name)['actions'], want, name)

    def test_only_the_traffic_profiles_are_not_gate_only_and_production_is_the_production_profile_unchanged(self):
        self.assertNotIn('gate_only', PROFILES['profiles'][PRODUCTION])
        self.assertNotIn('gate_only', PROFILES['profiles'][EIGHT])
        traffic = [name for name in WORK if 'gate_only' not in PROFILES['profiles'][parsed(name)['profile']]]
        self.assertEqual(traffic, [name for name in WORK if PROFILE_OF[name] in (PRODUCTION, EIGHT)])
        for name in WORK:
            if PROFILE_OF[name] not in (PRODUCTION, EIGHT):
                self.assertIs(PROFILES['profiles'][PROFILE_OF[name]].get('gate_only'), True, name)

    def test_the_flag_is_on_exactly_in_the_eight_seat_profiles_the_window_serves_and_off_in_s8_0(self):
        flag = 'QWEN_FAST_M3_BLOCKS'
        self.assertNotIn(flag, PROFILES['profiles'][parsed('S8-0-flag-off-smoke')['profile']]['env'])
        for name in WORK[1:]:
            env = PROFILES['profiles'][parsed(name)['profile']]['env']
            with self.subTest(template=name):
                if PROFILE_OF[name] in (EIGHT, EIGHT_GATE, EIGHT_TIME, EIGHT_DIAG):
                    self.assertEqual(env[flag], '2')
                else:
                    self.assertNotIn(flag, env)


class SmokeTests(unittest.TestCase):
    @staticmethod
    def executed():
        with open(os.path.join(HERE, 'c2_serving_smoke.py'), encoding='utf-8') as handle:
            return re.findall(r"^\s*record\('([a-z0-9_]+)'", handle.read(), re.M)

    def test_the_lists_are_the_plans(self):
        self.assertEqual(parsed('S8-0-flag-off-smoke')['tests'].split(','), S80_TESTS)
        self.assertEqual(parsed('S8-1-seats8-attach-audited')['tests'].split(','), S81_TESTS)
        for name in HANG:
            self.assertEqual(parsed(name)['tests'].split(','), HANG_TESTS, name)
        for name in TIMED:
            self.assertEqual(parsed(name)['tests'].split(','), TIMED_TESTS, name)

    def test_every_named_test_is_one_the_smoke_knows_and_runs_in_the_order_the_template_lists_them(self):
        executed = self.executed()
        for name in WORK:
            listed = parsed(name)['tests'].split(',') if parsed(name)['tests'] else []
            for test in listed:
                self.assertIn(test, executed, '%s: %s' % (name, test))
            self.assertEqual([test for test in executed if test in listed], listed, name)

    def test_the_hang_runs_are_five_with_the_four_user_factorial_shapes_and_their_eight_user_versions(self):
        self.assertEqual(len(HANG), 5)
        for shape in ('concurrent4_steady', 'replay_concurrent4', 'steady_resend', 'concurrent8_steady', 'replay_concurrent8',
                      'concurrent5_split', 'concurrent8_drain'):
            self.assertIn(shape, HANG_TESTS)
        for name in HANG:
            self.assertEqual(parsed(name)['profile'], EIGHT_DIAG)
            self.assertEqual(parsed(name)['actions'], 'reset smoke', 'each run follows its own reset')
            text = text_of(name)
            for word in ('NO QUALIFIED HANG FALLBACK AT 8 SEATS', 'FIVE CONSECUTIVE', 'QWEN_FAST_STALL_DEADLINE_S=120',
                         'QWEN_FAST_CCL_HANDLE_LOG=1', 'QWEN_FAST_CCL_HANDLE_GUARD=log', 'QWEN_FAST_SEQ_STAGE_LOG=1', 'QWEN_FAST_TRACE_CENSUS=1'):
                self.assertIn(word, text, name)

    def test_s8_0_is_the_production_profile_with_the_flag_off_and_the_four_user_shapes(self):
        values = parsed('S8-0-flag-off-smoke')
        self.assertEqual(values['profile'], PRODUCTION)
        for shape in ('concurrent4_code_equal', 'concurrent4_steady', 'replay_concurrent4', 'steady_resend'):
            self.assertIn(shape, values['tests'].split(','))
        text = text_of('S8-0-flag-off-smoke')
        for word in ('flag OFF', 'blocks=2', 'STOPS the window'):
            self.assertIn(word, text)

    def test_the_two_timed_arms_differ_only_in_the_seat_flag_and_the_seats(self):
        a, b = PROFILES['profiles'][FOUR_TIME], PROFILES['profiles'][EIGHT_TIME]
        self.assertEqual({k: v for k, v in b['env'].items() if a['env'].get(k) != v}, {'QWEN_FAST_M3_BLOCKS': '2'})
        self.assertEqual(set(a['env']) ^ set(b['env']), {'QWEN_FAST_M3_BLOCKS'})
        for name in TIMED:
            self.assertNotIn('c2-packed-tp4-time-gate', text_of(name), name)

    def test_the_timing_is_an_abab_of_four_seats_against_eight_on_the_same_tests(self):
        self.assertEqual([parsed(name)['profile'] for name in TIMED], [FOUR_TIME, EIGHT_TIME, FOUR_TIME, EIGHT_TIME])
        self.assertEqual([row[0] for row in read_order() if row[0] in TIMED], ['S8-7a-seats4-timed', 'S8-7b-seats8-timed',
                                                                                'S8-7c-seats4-timed', 'S8-7d-seats8-timed'])
        for name in TIMED:
            self.assertIs(PROFILES['profiles'][parsed(name)['profile']]['env']['QWEN_FAST_VERIFY_T1_AUDIT'], '0')
            text = text_of(name)
            for word in ('2 x P(4) + 10 ms', 'PAIRED', 'block_trace_split'):
                self.assertIn(word, text, name)


class KillSignalTests(unittest.TestCase):
    def test_the_earliest_kill_signals_of_the_review_are_in_the_read_outs(self):
        s81 = text_of('S8-1-seats8-attach-audited')
        for word in ('extent block engaged: blocks=2 segments=4,4', 'ZERO programs after block A', '22.7 to 23.0 GB per chip',
                     'ZERO audit mismatches', 'block-B users', 'EQUAL to S8-0', 'QWEN_FAST_TRACE_CENSUS=1', 'NOT CARRIED'):
            self.assertIn(word, s81)
        s86 = text_of('S8-6-memory8')
        for word in ('below 3 GB per chip', 'EIGHT LIVE', 'dram-after-engine'):
            self.assertIn(word, s86)
        self.assertEqual(parsed('S8-6-memory8')['gate_plan'], 'memory')

    def test_the_census_flag_is_not_in_the_eight_seat_gate_profile_the_note_says_so(self):
        # a job carries a profile and no environment: the note in S8-1 is true only while the profile lacks the flag
        env = PROFILES['profiles'][EIGHT_GATE]['env']
        self.assertNotIn('QWEN_FAST_TRACE_CENSUS', env)


class GatePlanTests(unittest.TestCase):
    def lengths(self, name):
        return [int(part) for part in parsed(name)['gate_lengths'].split(',')]

    def test_the_matrix_and_staggered_plans_are_eight_users_at_the_eight_lengths(self):
        for name, plan, tokens in (('S8-2a-matrix8-gate', 'matrix', '512'), ('S8-2b-matrix8-traffic', 'matrix', '512'),
                                   ('S8-4-staggered8', 'staggered', '4096')):
            outputs = parsed(name)
            with self.subTest(template=name):
                self.assertEqual((outputs['gate_plan'], outputs['gate_lengths'], outputs['gate_max_tokens']), (plan, MATRIX_LENGTHS, tokens))
                self.assertEqual(len(self.lengths(name)), 8)

    def test_the_churn_is_sixteen_sessions_over_eight_seats_with_at_least_eight_replacements(self):
        lengths = self.lengths('S8-5-churn16')
        self.assertEqual(len(lengths), 16)
        self.assertGreaterEqual(len(lengths) - driver.profile_seats(PROFILES, EIGHT_GATE), 8)
        self.assertLess(lengths[-1], 2048, 'the last user is short: the one-bucket ladder under churn')
        notes = []
        arms = driver.plan_arms('churn', EIGHT_GATE, PROFILES, lengths=lengths, max_tokens=1024, notes=notes)
        self.assertEqual(notes, [])
        self.assertEqual(base.parse_harness(arms[0][1]).users, 16)

    def test_every_gate_job_fits_the_gate_step_and_its_job_minutes(self):
        step = base.WorkflowTests.budget_literals()['step']
        for name in WORK:
            outputs = parsed(name)
            if outputs['gate_plan'] == 'bringup':
                continue
            plan = outputs['gate_plan']
            lengths = [int(part) for part in outputs['gate_lengths'].split(',')] if outputs['gate_lengths'] else None
            kwargs = dict(lengths=lengths, max_tokens=int(outputs['gate_max_tokens'])) if lengths else {}
            arms = driver.plan_arms(plan, outputs['profile'], PROFILES, **kwargs)
            with self.subTest(template=name):
                self.assertLessEqual(driver.worst_case_seconds([plan], {plan: arms}), step)


class EightSeatProfilesAreTheSeatsProfilesTests(unittest.TestCase):
    def test_every_eight_seat_profile_the_window_uses_reads_eight_seats_and_the_four_seat_ones_four(self):
        for name in WORK:
            profile = parsed(name)['profile']
            self.assertEqual(driver.profile_seats(PROFILES, profile), 8 if profile.startswith('c2-packed-tp4-8') else 4, name)


if __name__ == '__main__':
    unittest.main()
