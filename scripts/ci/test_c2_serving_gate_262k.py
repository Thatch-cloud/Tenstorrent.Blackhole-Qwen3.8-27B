"""The gate sets of the 262,144-token window (tp4/seats8-262k, B6): the ladders, the churn, the memory plan's users, the KV reservation's
judgement and the job key that carries C2_GATE_MEMORY_USERS.

L9a and L9b run the strict matrix on LADDER4_262K and LADDER8_262K; C9 churns sixteen users over eight seats and must SEE the reservation hold
and its release (else NOT_EXERCISED); M9a runs ONE cold 253,920-token prompt (C2_GATE_MEMORY_USERS=1) and M9b eight prompts of 157,000 that
reserve 21,688 of the pool's 21,759 blocks, so the pool is full and nothing holds. Every number below is arithmetic on serving_kv_reservation, not a
measurement; what the hardware says is the windows'."""

import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import c2_serving_gate as driver  # noqa: E402
import c2_serving_job as job  # noqa: E402
import lever_n_m3native_gate as harness  # noqa: E402
import serving_kv_reservation as kv  # noqa: E402
import test_c2_serving_gate as base  # noqa: E402

ROOT = HERE.parent.parent
PROFILES = json.loads((HERE / 'qwen_c2_profiles.json').read_text(encoding='utf-8'))
EIGHT_GATE, EIGHT, EIGHT_TIME, FOUR_GATE = ('c2-packed-tp4-8x262k-gate', 'c2-packed-tp4-8x262k', 'c2-packed-tp4-8x262k-time-gate',
                                            'c2-packed-tp4-262k-gate')
POOL = 21760 - 1


def parse(arm):
    return base.parse_harness(arm[1])


class LadderTests(unittest.TestCase):
    def test_the_ladders_are_the_262k_gate_profiles_rungs(self):
        self.assertEqual(driver.LADDER4_262K, (4096, 32768, 131072, 261856))
        self.assertEqual(driver.LADDER8_262K, (4096, 32768, 131072, 261856, 16384, 65536, 200000, 253920))
        self.assertEqual(driver.LADDER8_262K[:4], driver.LADDER4_262K)
        for name in (EIGHT_GATE, FOUR_GATE):
            context, ceiling, room = driver.profile_limits(PROFILES, name)
            self.assertEqual((context, ceiling, room), (262144, 16384, 261856))
            self.assertEqual(max(driver.LADDER8_262K), 261856)
            self.assertEqual(room, max(driver.LADDER4_262K), 'the top rung is the contract\'s largest prompt less 256 of answer')
        self.assertEqual(driver.profile_headroom(PROFILES, EIGHT_GATE), 32)
        self.assertEqual(driver.profile_headroom(PROFILES, 'c2-packed-tp4-8-gate'), 0)

    def test_the_four_seat_ladder_is_a_strict_matrix_of_four_users_with_256_tokens(self):
        concurrent, solo = driver.plan_arms('matrix', FOUR_GATE, PROFILES, lengths=list(driver.LADDER4_262K), max_tokens=256)
        self.assertEqual((parse(concurrent).users, parse(concurrent).max_tokens), (4, 256))
        self.assertEqual(parse(concurrent).prompt_lengths, list(driver.LADDER4_262K))
        self.assertEqual(parse(solo).sequential_users, 4)

    def test_the_eight_seat_ladder_is_eight_users_and_one_more_token_is_refused(self):
        concurrent, solo = driver.plan_arms('matrix', EIGHT_GATE, PROFILES, lengths=list(driver.LADDER8_262K), max_tokens=256)
        self.assertEqual(parse(concurrent).users, 8)
        self.assertEqual(parse(solo).sequential_users, 8)
        with self.assertRaisesRegex(driver.PlanError, 'exceed 261856'):
            driver.plan_arms('matrix', EIGHT_GATE, PROFILES, lengths=[261857] + [4096] * 7, max_tokens=256)
        # the traffic profile's edge is 253,920: the top gate rung is past it
        with self.assertRaisesRegex(driver.PlanError, 'exceed 253920'):
            driver.plan_arms('matrix', EIGHT, PROFILES, lengths=[261856] + [4096] * 7, max_tokens=256)

    def test_the_answer_clamp_is_the_window_less_the_headroom(self):
        # 261,856 + 256 = 262,112: the last token the clamp hands out; one more answer token would be refused by the plan
        driver.plan_arms('matrix', EIGHT_GATE, PROFILES, lengths=[261856] + [4096] * 7, max_tokens=256)
        with self.assertRaisesRegex(driver.PlanError, 'would cut that answer'):
            driver.plan_arms('matrix', EIGHT_GATE, PROFILES, lengths=[261856] + [4096] * 7, max_tokens=257)

    def test_the_four_seat_131k_defaults_are_untouched(self):
        for name in ('c2-packed-tp4-gate', 'c2-packed-tp4-8-gate'):
            context, ceiling, room = driver.profile_limits(PROFILES, name)
            self.assertEqual((context, ceiling, room), (131328, 16384, 131072))
            self.assertEqual(driver.profile_headroom(PROFILES, name), 0)

    def test_the_ladder_reservations_fit_their_pools_without_a_hold(self):
        self.assertEqual(sum(kv.request_blocks(length, 256) for length in driver.LADDER8_262K), 15135)
        self.assertLessEqual(15135, POOL)
        # the four-seat ladder is fully reserved (4 x 4,096 blocks, no reservation admission): the four rungs are one request each
        self.assertEqual(sum(kv.request_blocks(length, 256) for length in driver.LADDER4_262K), 6739)
        self.assertLessEqual(max(kv.request_blocks(length, 256) for length in driver.LADDER4_262K), 4097)


class ChurnTests(unittest.TestCase):
    def test_the_262k_set_is_sixteen_users_eight_replacements_every_budget_1024(self):
        lengths, budgets = driver.CHURN_262K
        self.assertEqual((len(lengths), len(budgets)), (16, 16))
        self.assertEqual(set(budgets), {1024})
        self.assertEqual(len(lengths) - 8, driver.CHURN_MIN_REPLACEMENTS)
        self.assertEqual(lengths[:3], (253920,) * 3)
        self.assertEqual(lengths[-1], 1536)
        self.assertLess(lengths[-1], 2048, 'a short user last: the one-bucket ladder under churn')
        self.assertTrue(all(length <= 253920 for length in lengths))

    def test_it_is_the_default_at_a_262k_window_on_eight_seats_only(self):
        self.assertEqual(driver.churn_defaults(8, 262144), driver.CHURN_262K)
        self.assertEqual(driver.churn_defaults(8), (driver.CHURN_LENGTHS_8, driver.CHURN_MAX_TOKENS_8))
        self.assertEqual(driver.churn_defaults(8, 131328), (driver.CHURN_LENGTHS_8, driver.CHURN_MAX_TOKENS_8))
        self.assertEqual(driver.churn_defaults(4, 262144), (driver.CHURN_LENGTHS, driver.CHURN_MAX_TOKENS))
        self.assertEqual(driver.churn_defaults(4), (driver.CHURN_LENGTHS, driver.CHURN_MAX_TOKENS))

    def test_the_plan_on_a_262k_profile_is_the_262k_set_and_on_the_131k_profile_what_it_was(self):
        churn, = driver.plan_arms('churn', EIGHT_GATE, PROFILES)
        options = parse(churn)
        self.assertEqual((options.users, options.alive_check), (16, 8))
        self.assertEqual(options.prompt_lengths, list(driver.CHURN_LENGTHS_262K))
        self.assertEqual(options.events['max_tokens'], {index: 1024 for index in range(16)})
        old, = driver.plan_arms('churn', 'c2-packed-tp4-8-gate', PROFILES)
        self.assertEqual(parse(old).prompt_lengths, list(driver.CHURN_LENGTHS_8))

    def test_the_first_six_long_users_fit_the_pool_and_the_seventh_is_held(self):
        lengths, budgets = driver.CHURN_262K
        reserved = [kv.request_blocks(length, budget) for length, budget in zip(lengths, budgets)]
        self.assertEqual(sum(reserved[:6]), 21384)
        self.assertLessEqual(sum(reserved[:6]), POOL)
        self.assertEqual(sum(reserved[:7]), 23450)
        self.assertGreater(sum(reserved[:7]), POOL, 'the seventh waits for a departure: the hold the churn must see')

    def test_every_churn_length_is_one_the_contract_admits_on_the_gate_and_the_traffic_profile(self):
        for name in (EIGHT_GATE, EIGHT):
            room = driver.profile_limits(PROFILES, name)[2]
            for length in driver.CHURN_LENGTHS_262K:
                self.assertLessEqual(length, room, name)


def report_with(record=None, **extra):
    s2 = {}
    if record is not None:
        s2['kv_reservation'] = record
    return dict(s2=s2, **extra)


def record(**changes):
    value = dict(installed=1, holds=2, released=2, fits=14, too_large=0, unavailable=0, carried=1, pools=[POOL],
                 over_pool=[], over_pool_count=0, hold_that_fits=[], hold_that_fits_count=0, max_reserved=3985,
                 max_running=19000, lines=[])
    value.update(changes)
    return value


class KvJudgementTests(unittest.TestCase):
    def test_a_clean_record_has_no_problem(self):
        self.assertEqual(driver.kv_reservation_problems(report_with(record())), [])
        self.assertEqual(driver.kv_reservation_checks(report_with(record()), True, require_hold=True), ([], []))

    def test_a_report_without_the_reservation_judges_nothing_unless_the_profile_names_it(self):
        self.assertEqual(driver.kv_reservation_problems(report_with()), [])
        self.assertEqual(driver.kv_reservation_problems({}), [])
        self.assertEqual(driver.kv_reservation_checks(report_with(), False, require_hold=True), ([], []))
        problems, shortfalls = driver.kv_reservation_checks(report_with(), True)
        self.assertEqual(len(problems), 1)
        self.assertIn('never logged the reservation\'s install line', problems[0])

    def test_each_wrong_line_is_a_problem(self):
        cases = {
            'admitted past the pool': dict(over_pool_count=1, over_pool=[dict(request='a', reserved=10, running=POOL, pool=POOL)]),
            'a hold that fits': dict(hold_that_fits_count=1, hold_that_fits=[dict(request='a')]),
            'too large': dict(too_large=1),
            'unreadable pool': dict(unavailable=2),
        }
        for label, changes in cases.items():
            with self.subTest(label):
                problems = driver.kv_reservation_problems(report_with(record(**changes)))
                self.assertEqual(len(problems), 1, problems)
                self.assertEqual(driver.arm_problems('arm', report_with(record(**changes))), ['arm: %s' % problems[0]])

    def test_the_churn_must_see_a_hold_and_its_release_or_it_is_not_exercised(self):
        for changes, word in ((dict(holds=0, released=0), '0 holds'), (dict(holds=3, released=0), 'none released')):
            problems, shortfalls = driver.kv_reservation_checks(report_with(record(**changes)), True, require_hold=True)
            self.assertEqual(problems, [])
            self.assertEqual(len(shortfalls), 1, shortfalls)
            self.assertIn(word, shortfalls[0])
            self.assertIn('NOT_EXERCISED', shortfalls[0])
        # the memory arms do not require one: M9b's pool is full and nothing holds
        self.assertEqual(driver.kv_reservation_checks(report_with(record(holds=0, released=0)), True, require_hold=False), ([], []))

    def test_the_expected_path_follows_the_profiles_flag(self):
        for name in (EIGHT, EIGHT_GATE, EIGHT_TIME, 'c2-packed-tp4-8x262k-diag-strace'):
            self.assertTrue(driver.kv_reservation_expected(PROFILES, name), name)
        for name in (FOUR_GATE, 'c2-packed-tp4', 'c2-packed-tp4-8', 'c2-packed-tp4-8-gate', 'no-such-profile'):
            self.assertFalse(driver.kv_reservation_expected(PROFILES, name), name)
        self.assertFalse(driver.kv_reservation_expected(None, EIGHT))

    def test_run_arm_adds_the_executed_path_problem_to_an_arm_that_never_installed_the_reservation(self):
        def runner(report, profile):
            return SimpleNamespace(profiles=PROFILES, profile=profile, judges=lambda *a, **k: False,
                                   run=lambda *a, **k: report)

        spec = ('arm', ['--users', '1'], 100)
        report = driver.run_arm(runner(report_with(), EIGHT_GATE), 'matrix', spec)
        self.assertEqual(len(report['c2_gate_problems']), 1)
        self.assertIn('never logged the reservation\'s install line', report['c2_gate_problems'][0])
        self.assertEqual(len(driver.arm_problems('arm', report)), 1)
        clean = driver.run_arm(runner(report_with(record()), EIGHT_GATE), 'matrix', spec)
        self.assertNotIn('c2_gate_problems', clean)
        other = driver.run_arm(runner(report_with(), 'c2-packed-tp4-8-gate'), 'matrix', spec)
        self.assertNotIn('c2_gate_problems', other, 'a profile without the flag is judged as it always was')
        self.assertIsNone(driver.run_arm(runner(None, EIGHT_GATE), 'matrix', spec))


LOG = '''[PINDIAG] kv reservation installed on vllm_tt_plugin.scheduler.TTScheduler: pool from block_pool.num_gpu_blocks less the null block, block=64 lookahead=32 spare=1
[PINDIAG] kv reservation fit request=r0 reserved=3985 running=0 pool=21759 decodes=0
[PINDIAG] kv reservation fit request=r1 reserved=3985 running=3985 pool=21759 decodes=1
[PINDIAG] kv reservation hold request=r7 reserved=2066 running=21384 pool=21759 decodes=6
[PINDIAG] kv reservation carried finished=['r2'] past the discarded prefill pass into the decode-only step
[PINDIAG] kv reservation released request=r7 reserved=2066 running=17399 pool=21759
'''


class HarnessParseTests(unittest.TestCase):
    def test_the_server_logs_lines_become_the_record_the_gate_judges(self):
        parsed = harness.kv_reservations(LOG)
        self.assertEqual((parsed['installed'], parsed['holds'], parsed['released'], parsed['fits'], parsed['carried']),
                         (1, 1, 1, 2, 1))
        self.assertEqual((parsed['pools'], parsed['over_pool_count'], parsed['hold_that_fits_count']), ([21759], 0, 0))
        self.assertEqual((parsed['max_reserved'], parsed['max_running']), (3985, 21384))
        self.assertEqual(driver.kv_reservation_checks(report_with(parsed), True, require_hold=True), ([], []))

    def test_an_admission_past_the_pool_and_a_hold_that_fits_are_read_as_such(self):
        bad = LOG + ('[PINDIAG] kv reservation fit request=r9 reserved=3985 running=20000 pool=21759 decodes=5\n'
                     '[PINDIAG] kv reservation hold request=r8 reserved=100 running=100 pool=21759 decodes=2\n'
                     '[PINDIAG] kv reservation too large request=r10 reserved=30000 pool=21759: held\n')
        parsed = harness.kv_reservations(bad)
        self.assertEqual((parsed['over_pool_count'], parsed['hold_that_fits_count'], parsed['too_large']), (1, 1, 1))
        self.assertEqual(len(driver.kv_reservation_problems(report_with(parsed))), 3)

    def test_no_kv_lines_is_an_empty_record_and_dram_lines_are_not_kv_lines(self):
        empty = harness.kv_reservations('[PINDIAG] dram hold prompt=123136 largest_free=900.0MB need=1300.0MB request=a decodes=1 '
                                        'free=2.0GB trace_largest_free=49.0MB short=free\n')
        self.assertEqual((empty['installed'], empty['holds'], empty['released'], empty['fits']), (0, 0, 0, 0))
        self.assertEqual(driver.kv_reservation_problems(report_with(empty)), [])
        self.assertEqual(harness.KV_HOLD_MARKER, kv.HOLD_PREFIX)
        self.assertTrue(kv.INSTALLED_LINE.startswith(harness.KV_INSTALLED_MARKER))
        self.assertTrue(kv.RELEASED_LINE.startswith(harness.KV_RELEASED_MARKER))
        self.assertTrue(kv.FIT_LINE.startswith(harness.KV_FIT_MARKER))
        self.assertTrue(kv.TOO_LARGE_LINE.startswith(harness.KV_TOO_LARGE_MARKER))
        self.assertTrue(kv.CARRIED_LINE.startswith(harness.KV_CARRIED_MARKER))
        self.assertTrue(kv.UNAVAILABLE_LINE.startswith(harness.KV_UNAVAILABLE_MARKER))

    def test_the_formatted_lines_parse(self):
        lines = '\n'.join([kv.HOLD_LINE.format('rq', 61, 61, 100, 1), kv.RELEASED_LINE.format('rq', 61, 0, 100),
                           kv.FIT_LINE.format('rq2', 14, 61, 100, 1)])
        parsed = harness.kv_reservations(lines)
        self.assertEqual((parsed['holds'], parsed['released'], parsed['fits'], parsed['pools']), (1, 1, 1, [100]))

    def test_the_dram_rule_still_reads_dram_lines_only(self):
        report = {'s2': dict(dram_hold=harness.dram_holds(LOG), kv_reservation=harness.kv_reservations(LOG),
                             before=None)}
        self.assertEqual(report['s2']['dram_hold']['holds'], 0, 'a KV hold with a seat free is no DRAM hold')
        problems, _shortfalls = driver.memory_s2_checks('churn', report, 8)
        self.assertFalse([problem for problem in problems if 'DRAM held a prompt with a seat free' in problem])


class MemoryUsersTests(unittest.TestCase):
    def test_m9a_is_one_cold_prompt_and_m9b_is_eight_that_fill_the_pool(self):
        concurrent, short = driver.plan_arms('memory', EIGHT, PROFILES, memory_prompt=253920, memory_users=1)
        options = parse(concurrent)
        self.assertEqual((options.users, options.prompt_lengths, options.max_tokens), (1, [253920], 8192))
        self.assertEqual(parse(short).users, 1)
        full, short8 = driver.plan_arms('memory', EIGHT, PROFILES, memory_prompt=157000)
        options = parse(full)
        self.assertEqual((options.users, options.prompt_lengths, options.max_tokens), (8, [157000] * 8, 16384))
        self.assertEqual(8 * kv.request_blocks(157000, 16384), 21688)
        self.assertLessEqual(21688, POOL)
        self.assertGreater(21688 + kv.request_blocks(157000, 16384), POOL, 'a ninth would not fit: the pool is full')

    def test_the_default_is_still_the_profiles_seats(self):
        for name, seats in (('c2-packed-tp4-8-gate', 8), ('c2-packed-tp4-gate', 4), (EIGHT_GATE, 8), (FOUR_GATE, 4)):
            for arm in driver.plan_arms('memory', name, PROFILES):
                self.assertEqual(parse(arm).users, seats, name)

    def test_more_users_than_seats_or_none_are_refused(self):
        for bad in (9, 0, -1, 1.5):
            with self.subTest(users=bad), self.assertRaisesRegex(driver.PlanError, 'memory --memory-users'):
                driver.plan_arms('memory', EIGHT, PROFILES, memory_users=bad)

    def test_the_verdict_judges_engines_against_the_arms_own_users(self):
        arm = dict(dram=dict(engines=1, min_free_gb=3.1, min_largest_free_mb=900), streams=[], alive=True)
        self.assertEqual(driver.memory_verdict(dict(arm), users=1)['verdict'] in ('PASS', 'FAIL'), True)
        verdict = driver.memory_verdict(dict(arm), users=8)
        self.assertIn('1 dram-after-engine lines for 8 users', ' '.join(verdict['problems']))
        self.assertEqual([problem for problem in driver.memory_verdict(dict(arm), users=1)['problems']
                          if 'dram-after-engine' in problem], [])

    def test_the_cli_carries_the_flag(self):
        parser = driver.build_parser()
        options = parser.parse_args(['--image', 'x', '--profile', EIGHT, '--plan', 'memory', '--memory-users', '1',
                                     '--memory-prompt', '253920', '--results', 'r'])
        self.assertEqual((options.memory_users, options.memory_prompt), (1, 253920))
        self.assertIsNone(parser.parse_args(['--image', 'x', '--profile', EIGHT, '--plan', 'memory', '--results', 'r']).memory_users)


class JobKeyTests(unittest.TestCase):
    PLAIN = 'C2_IMAGE_TAG=tp4-262k8-1\nC2_CARDS=quad\nC2_ACTIONS=reset gate\nC2_PROFILE=%s\nC2_GATE_PLAN=memory\n' % EIGHT

    def parsed(self, extra=''):
        return job.read_job(job.parse_env(self.PLAIN + extra), sorted(PROFILES['profiles']), root=str(ROOT))

    def test_the_key_is_an_output_and_empty_by_default(self):
        self.assertEqual(self.parsed()['gate_memory_users'], '')
        outputs = self.parsed('C2_GATE_MEMORY_USERS=1\nC2_GATE_MEMORY_PROMPT=253920\n')
        self.assertEqual((outputs['gate_memory_users'], outputs['gate_memory_prompt']), ('1', '253920'))

    def test_a_bad_value_is_refused(self):
        for bad in ('0', '-1', 'one', '1.5'):
            with self.subTest(value=bad), self.assertRaises(job.JobError):
                self.parsed('C2_GATE_MEMORY_USERS=%s\n' % bad)

    def test_the_workflow_hands_it_to_the_gate(self):
        text = (ROOT / '.github' / 'workflows' / 'qwen-c2-serving.yml').read_text(encoding='utf-8')
        self.assertIn('GATE_MEMORY_USERS: ${{ steps.job.outputs.gate_memory_users }}', text)
        self.assertIn('${GATE_MEMORY_USERS:+--memory-users "$GATE_MEMORY_USERS"}', text)
        self.assertIn('${GATE_MEMORY_PROMPT:+--memory-prompt "$GATE_MEMORY_PROMPT"}', text)

    def test_the_job_docs_name_the_key(self):
        self.assertIn('C2_GATE_MEMORY_USERS', job.__doc__)


if __name__ == '__main__':
    unittest.main()
