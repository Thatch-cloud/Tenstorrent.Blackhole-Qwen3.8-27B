"""The tp4/next-2 window: its job templates (scripts/ci/references/tp4-next2-jobs), their order, and the profiles they use.

tp4/next-2 is tp4/serve-4 (the hang-fix arms) merged with tp4/next (the round-time levers). N0 builds the image tp4-next-2 and smokes
the PRODUCTION profile (the merge must not have changed it); N1 is the audited smoke of the best gate profile; N2 and N3 are the strict
exactness plans (S3a matrix, staggered trigger) on it; N4a..N4d are the paired timing ABAB of the timed best against its control, the
speed twin, each carrying hang fix A (QWEN_FAST_PACKED_SAMPLER_IN_TRACE), on coding prompts (4 x 4k, 4 x 32k); N5a..N5d are the same ABAB
carrying hang fix B (QWEN_FAST_REQUEST_SHARD_ARGMAX), run only if R1..R1e all completed (every timed audits-off pair carries a hang fix); R1 (five runs), R2, R2b, R0 and R3 (an ABAB; optional, after N0) test the request engine's shard argmax
(QWEN_FAST_REQUEST_SHARD_ARGMAX) on the hang shapes, audited, and timed against the speed twin; N6 hands the cards back. Every job resets the cards first. The templates are public,
so they name no rig, card, address, registry or digest."""

import json
import os
import re
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import c2_serving_job as job  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
FOLDER = os.path.join(HERE, 'references', 'tp4-next2-jobs')
with open(os.path.join(HERE, 'qwen_c2_profiles.json'), encoding='utf-8') as _handle:
    PROFILES = json.load(_handle)
NAMES = sorted(PROFILES['profiles'])
DEVICE_STEPS = ('cardm', 'smoke', 'gate', 'prefix', 'fabric', 'replay')
BANNED = re.compile(r'blackhole-[A-Za-z0-9]{8,}|thatch\.local|\d{1,3}(\.\d{1,3}){3}|sha256:[0-9a-f]{16}|[0-9a-f]{40,}|'
                    r'/dev/tenstorrent|home/|zot\.|@[A-Z0-9_]+@')
IMAGE = 'tp4-next-2'
TIMED_A = ('N4a-speed-timed', 'N4b-best-timed', 'N4c-speed-timed', 'N4d-best-timed')
TIMED_B = ('N5a-speed-rshard-timed', 'N5b-best-rshard-timed', 'N5c-speed-rshard-timed', 'N5d-best-rshard-timed')
TIMED = TIMED_A + TIMED_B
HANDBACK = 'N6-handback-reset'
CONTROL, BEST, BEST_GATE, PRODUCTION = 'c2-packed-tp4-speed', 'c2-packed-tp4-best', 'c2-packed-tp4-best-gate', 'c2-packed-tp4'
R1_RUNS = ('R1-diag-rshard', 'R1b-diag-rshard', 'R1c-diag-rshard', 'R1d-diag-rshard', 'R1e-diag-rshard')
R0_R3 = ('R0-speed-base', 'R3-speed-rshard', 'R0b-speed-base', 'R3b-speed-rshard')
RSHARD_JOBS = ('R2b-rshard-audit-hang-shapes', 'R2-rshard-audit') + R1_RUNS + R0_R3
STOP_JOBS = ('N1-best-gate-smoke-audited', 'N2-s3a-matrix-best-gate', 'N3-staggered-trigger-best-gate')
ORDERED = ('N0-build-smoke',) + RSHARD_JOBS + STOP_JOBS + TIMED + (HANDBACK,)
WORK = ORDERED[:-1]
RSHARD_ARM = 'QWEN_FAST_REQUEST_SHARD_ARGMAX'
HANG_SHAPES = ['warmup', 'coding', 'long_real_text', 'concurrent4', 'concurrent4_v164order', 'concurrent4_steady', 'replay_concurrent4']
RSHARD_TIMING_TESTS = ['warmup', 'coding', 'concurrent4_code', 'concurrent4_code_equal', 'concurrent8_code']
PROFILE_OF = {'N0-build-smoke': PRODUCTION, 'R2-rshard-audit': 'c2-packed-tp4-gate-rshard-audit',
              'R2b-rshard-audit-hang-shapes': 'c2-packed-tp4-diag-t1-rshard-audit',
              **{name: 'c2-packed-tp4-diag-rshard' for name in R1_RUNS},
              **{name: ('c2-packed-tp4-speed-strace' if name.startswith('R0') else 'c2-packed-tp4-speed-rshard') for name in R0_R3}, 'N1-best-gate-smoke-audited': BEST_GATE, 'N2-s3a-matrix-best-gate': BEST_GATE,
              'N3-staggered-trigger-best-gate': BEST_GATE,
              'N4a-speed-timed': 'c2-packed-tp4-speed-strace', 'N4b-best-timed': 'c2-packed-tp4-best-strace',
              'N4c-speed-timed': 'c2-packed-tp4-speed-strace', 'N4d-best-timed': 'c2-packed-tp4-best-strace',
              'N5a-speed-rshard-timed': 'c2-packed-tp4-speed-rshard', 'N5b-best-rshard-timed': 'c2-packed-tp4-best-rshard',
              'N5c-speed-rshard-timed': 'c2-packed-tp4-speed-rshard', 'N5d-best-rshard-timed': 'c2-packed-tp4-best-rshard'}
SS = ['warmup', 'coding', 'long_real_text', 'concurrent4', 'concurrent4_steady', 'steady_resend', 'tool_call', 'stream_tool_call',
      'stream_reasoning', 'refused_n2', 'alive_after_refusal', 'stream_dropped', 'alive_after_drop']
CODE = ['concurrent4_code', 'concurrent4_code_equal']
TIMING_TESTS = ['warmup', 'coding', 'concurrent4_code_equal', 'concurrent4_code_32k']
LEVERS = ('QWEN_FAST_TP4_COMMIT_LANES', 'QWEN_FAST_TP4_SHARD_VALUES', 'QWEN_FAST_TP4_GDN_GLUE', 'QWEN_FAST_TP4_GDN_BLOCK_CONV',
          'QWEN_FAST_TP4_ATTN_FOLD')
FUSED = ('QWEN_FAST_FUSED_COMMIT', 'QWEN_FAST_FUSED_COMMIT_INPLACE', 'QWEN_FAST_FUSED_COMMIT_LIVE_BANKS')
QUAD = 'QWEN_FAST_QUAD_DRAFT'
TAIL_CAPS = ('QWEN_FAST_BUDGET_CAP', 'QWEN_FAST_SEQ_DEADLINE_S')


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


def env_of(name):
    return PROFILES['profiles'][name]['env']


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
                self.assertTrue(minutes.isdigit() and 15 <= int(minutes) <= 180, minutes)
        modes = {row[0]: row[1] for row in read_order()}
        for name in ('N0-build-smoke',) + STOP_JOBS + (HANDBACK,):
            self.assertEqual(modes[name], 'stop', name)
        for name in TIMED + RSHARD_JOBS:
            self.assertEqual(modes[name], 'optional', name)

    def test_the_hand_back_is_last_and_the_order_says_it_runs_even_when_a_stop_job_halts_the_window(self):
        self.assertEqual(read_order()[-1][0], HANDBACK)
        with open(os.path.join(FOLDER, 'ORDER.txt'), encoding='utf-8') as handle:
            order = handle.read()
        self.assertIn('runs even when a stop job', order)
        text = text_of(HANDBACK)
        self.assertEqual(parsed(HANDBACK)['actions'], 'status reset')
        self.assertEqual(parsed(HANDBACK)['cards'], 'quad')
        for word in ('AUDITED production image', 'NEVER place a gate arm', 'fabric', ':latest retag', 'admin API'):
            self.assertIn(word, text)

    def test_the_order_names_the_control_the_hang_fix_precondition_and_the_exactness_before_timing_rule(self):
        with open(os.path.join(FOLDER, 'ORDER.txt'), encoding='utf-8') as handle:
            order = handle.read()
        for word in ('not c2-packed-tp4-time-gate', 'EVERY TIMED AUDITS-OFF PAIR CARRIES A HANG FIX', 'S0t', 'tp4-serve-6', 'hang-fix arm',
                     'AND c2-packed-tp4-best', 'N4 runs after the stop jobs', 'ONLY IF R1, R1b, R1c, R1d', 'ABAB',
                     'NOTHING COMBINED HAS RUN ON A CARD'):
            self.assertIn(word, order)
        names = [row[0] for row in read_order()]
        for exact in ('N2-s3a-matrix-best-gate', 'N3-staggered-trigger-best-gate'):
            for timed in TIMED:
                self.assertLess(names.index(exact), names.index(timed))


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
                self.assertEqual(len(set(actions) & set(DEVICE_STEPS)), 1)
                self.assertIn('reset', actions, 'links train only at board init: a four-card job follows a reset')
                device = [step for step in actions if step in DEVICE_STEPS][0]
                self.assertLess(actions.index('reset'), actions.index(device))

    def test_the_actions_and_profiles_are_the_plans(self):
        for name, profile in PROFILE_OF.items():
            outputs = parsed(name)
            with self.subTest(template=name):
                self.assertEqual(outputs['profile'], profile)
                self.assertIn(profile, PROFILES['profiles'])
                self.assertEqual(outputs['actions'], {'N0-build-smoke': 'status reset build smoke', 'N2-s3a-matrix-best-gate': 'reset gate',
                                                      'N3-staggered-trigger-best-gate': 'reset gate'}.get(name, 'reset smoke'))

    def test_the_image_is_built_by_the_first_job_only(self):
        for name in ORDERED[1:]:
            self.assertNotIn('build', parsed(name)['actions'].split(), name)
        self.assertIn('build', parsed('N0-build-smoke')['actions'].split())

    def test_only_the_production_profile_is_a_traffic_profile(self):
        self.assertNotIn('gate_only', PROFILES['profiles'][parsed('N0-build-smoke')['profile']])
        for name in WORK[1:]:
            self.assertIs(PROFILES['profiles'][parsed(name)['profile']].get('gate_only'), True, name)

    def test_the_exactness_plans_are_s3a_s_matrix_and_the_staggered_trigger_at_s3a_s_lengths(self):
        for name, plan, tokens in (('N2-s3a-matrix-best-gate', 'matrix', '512'), ('N3-staggered-trigger-best-gate', 'staggered', '4096')):
            outputs = parsed(name)
            with self.subTest(template=name):
                self.assertEqual(outputs['gate_plan'], plan)
                self.assertEqual(outputs['gate_lengths'], '4096,16384,32768,60000')
                self.assertEqual(outputs['gate_max_tokens'], tokens)
        for name in ('N0-build-smoke', 'N1-best-gate-smoke-audited') + TIMED:
            self.assertEqual(parsed(name)['gate_plan'], 'bringup', name)


class ProfileTests(unittest.TestCase):
    def test_the_production_profile_carries_no_lever(self):
        env = env_of(PRODUCTION)
        for flag in LEVERS + FUSED + (QUAD, 'QWEN_FAST_TP4_VGLUE_AUDIT', 'QWEN_FAST_SOLO_LANE', 'QWEN_FAST_LANE'):
            self.assertIn(env.get(flag, '0'), ('0', ''), flag)
        # tp4-serve-7: production serves with the verify audits OFF and the pinned sampler recorded in the verify trace
        for flag in ('QWEN_FAST_VERIFY_T1_AUDIT', 'QWEN_FAST_VERIFY_T2_AUDIT'):
            self.assertEqual(env[flag], '0', 'the production profile serves with the audits off (serve-7)')
        self.assertEqual(env['QWEN_FAST_PACKED_SAMPLER_IN_TRACE'], '1')

    def test_the_timed_best_is_its_control_plus_the_levers_and_nothing_else(self):
        control, best = env_of(CONTROL), env_of(BEST)
        differing = {key for key in set(control) | set(best) if control.get(key) != best.get(key)}
        self.assertEqual(differing, set(LEVERS) | set(FUSED) | {QUAD})
        for flag in LEVERS + FUSED + (QUAD,):
            self.assertEqual(best[flag], '1', flag)
        for flag in TAIL_CAPS:
            self.assertEqual((control[flag], best[flag]), (PROFILES['profiles'][PRODUCTION]['env'][flag],) * 2, 'both arms carry the caps')
        for flag in ('QWEN_FAST_VERIFY_T1_AUDIT', 'QWEN_FAST_VERIFY_T2_AUDIT'):
            self.assertEqual((control[flag], best[flag]), ('0', '0'), 'audits off in both timed arms')
        for key in set(PROFILES['profiles'][CONTROL]) | set(PROFILES['profiles'][BEST]):
            if key not in ('env', 'description'):
                self.assertEqual(PROFILES['profiles'][CONTROL].get(key), PROFILES['profiles'][BEST].get(key), key)

    def test_the_audited_best_matches_the_timed_best_in_arithmetic_and_scheduling(self):
        timed, audited = env_of(BEST), env_of(BEST_GATE)
        differing = {key for key in set(timed) | set(audited) if timed.get(key) != audited.get(key)}
        self.assertEqual(differing, {'QWEN_FAST_VERIFY_T1_AUDIT', 'QWEN_FAST_VERIFY_T2_AUDIT', 'QWEN_FAST_FUSED_COMMIT_AUDIT',
                                     'QWEN_FAST_DRAFT_SINGLES_AUDIT', 'QWEN_FAST_TP4_VGLUE_AUDIT'})

    def test_the_control_is_the_speed_twin_the_old_control_is_not_used_here(self):
        for name in TIMED:
            self.assertNotEqual(parsed(name)['profile'], 'c2-packed-tp4-time-gate', name)
        for jobs, fix in ((TIMED_A, 'strace'), (TIMED_B, 'rshard')):
            self.assertEqual([parsed(name)['profile'] for name in jobs],
                             ['%s-%s' % (CONTROL, fix), '%s-%s' % (BEST, fix)] * 2, 'A B A B')

    def test_no_timed_audits_off_arm_is_without_a_hang_fix_and_each_pair_differs_in_the_levers_alone(self):
        flags = {'strace': 'QWEN_FAST_PACKED_SAMPLER_IN_TRACE', 'rshard': RSHARD_ARM}
        levers = {key for key in set(env_of(CONTROL)) | set(env_of(BEST)) if env_of(CONTROL).get(key) != env_of(BEST).get(key)}
        for jobs, fix in ((TIMED_A, 'strace'), (TIMED_B, 'rshard')):
            control, best = env_of('%s-%s' % (CONTROL, fix)), env_of('%s-%s' % (BEST, fix))
            self.assertEqual({key for key in set(control) | set(best) if control.get(key) != best.get(key)}, levers, fix)
            for name in jobs:
                env = env_of(parsed(name)['profile'])
                self.assertEqual(env[flags[fix]], '1', name)
                self.assertEqual((env['QWEN_FAST_VERIFY_T1_AUDIT'], env['QWEN_FAST_VERIFY_T2_AUDIT']), ('0', '0'), name)
                self.assertEqual([other for other in flags.values() if other != flags[fix] and env.get(other) == '1'], [], name)
            self.assertEqual(env_of('c2-packed-tp4-best-' + fix), dict(env_of(BEST), **{flags[fix]: '1'}))
            self.assertEqual(env_of('c2-packed-tp4-speed-' + fix), dict(env_of(CONTROL), **{flags[fix]: '1'}))
        for name in (CONTROL, BEST):
            for flag in flags.values():
                self.assertNotIn(flag, env_of(name), 'the un-fixed pair is never timed: it hangs audits-off')

    def test_the_rshard_pair_runs_after_the_strace_pair_before_the_hand_back_and_says_it_needs_r1_to_r1e(self):
        names = [row[0] for row in read_order()]
        self.assertEqual(names[names.index(TIMED_A[0]):], list(TIMED_A + TIMED_B + (HANDBACK,)))
        for name in TIMED_B:
            self.assertIn('R1..R1e all completed', text_of(name))
            self.assertIn('S0t', text_of(name))
        for name in TIMED_A:
            self.assertIn('S0t', text_of(name))


class RequestShardJobTests(unittest.TestCase):
    """R1 (x5), R2, R2b, R0, R3 (an ABAB): the request engine's shard argmax (QWEN_FAST_REQUEST_SHARD_ARGMAX) on the hang shapes, audited, and timed paired."""

    def test_the_jobs_follow_n0_in_the_order_and_are_optional(self):
        names = [row[0] for row in read_order()]
        self.assertEqual(names[:1 + len(RSHARD_JOBS)], ['N0-build-smoke'] + list(RSHARD_JOBS))
        self.assertEqual(names[1:3], ['R2b-rshard-audit-hang-shapes', 'R2-rshard-audit'], 'the audited arms run before the unaudited repeats')
        modes = {row[0]: row[1] for row in read_order()}
        for name in RSHARD_JOBS:
            self.assertEqual(modes[name], 'optional', name)

    def test_r1_is_five_runs_each_a_template_of_its_own_on_the_hang_shapes_and_the_timing_runs_only_after_all_five(self):
        self.assertEqual(len(R1_RUNS), 5)
        names = [row[0] for row in read_order()]
        for name in R1_RUNS:
            self.assertEqual(tests_of(name), HANG_SHAPES, name)
            self.assertEqual(parsed(name)['profile'], 'c2-packed-tp4-diag-rshard', name)
            self.assertEqual(parsed(name)['actions'], 'reset smoke', 'each run follows its own card reset')
            for timing in R0_R3:
                self.assertLess(names.index(name), names.index(timing), 'R0 and R3 run after the R1 repeats')
        with open(os.path.join(FOLDER, 'ORDER.txt'), encoding='utf-8') as handle:
            order = handle.read()
        self.assertIn('ONLY AFTER ALL FIVE R1 RUNS COMPLETED', order)
        self.assertIn('all five R1 runs', text_of('R3-speed-rshard'))
        self.assertIn('five consecutive runs', text_of('R1-diag-rshard').lower())

    def test_r2b_audits_rows_1_2_and_4_on_the_hang_shapes_with_the_caps_off_and_the_read_out_requires_each_width(self):
        self.assertEqual(tests_of('R2b-rshard-audit-hang-shapes'), HANG_SHAPES)
        env = env_of('c2-packed-tp4-diag-t1-rshard-audit')
        self.assertEqual(env['QWEN_FAST_BUDGET_CAP'], '0', 'the caps are off, or no rows=1 or rows=2 capture is ever replayed')
        self.assertEqual((env[RSHARD_ARM], env['QWEN_FAST_REQUEST_SHARD_AUDIT']), ('1', '1'))
        text = text_of('R2b-rshard-audit-hang-shapes')
        self.assertIn('audit exact=True rows=1', text)
        for rows in (2, 4):
            self.assertIn("'rows=%d'" % rows, text)
        self.assertIn('REQUIRED', text)
        self.assertIn('R2b', text_of('R2-rshard-audit'), 'R2 says it does not cover rows 1 and 2')
        self.assertIn('width-4', text_of('R2-rshard-audit'))

    def test_r2_runs_the_production_smoke_on_the_audited_arm(self):
        self.assertEqual(tests_of('R2-rshard-audit'), SS + ['concurrent4_solo'] + CODE)
        self.assertEqual(tests_of('R2-rshard-audit'), tests_of('N0-build-smoke'), 'the production smoke of the serve window (B5)')

    def test_the_timing_is_an_abab_of_the_speed_twin_without_and_with_the_flag_on_the_same_tests(self):
        names = [row[0] for row in read_order()]
        # A is fix A (the in-trace sampler), not the bare speed twin: that deadlocked on these tests (S0t, tp4-serve-6).
        self.assertEqual([parsed(name)['profile'] for name in R0_R3], ['c2-packed-tp4-speed-strace', 'c2-packed-tp4-speed-rshard'] * 2)
        for name in R0_R3:
            self.assertEqual(tests_of(name), RSHARD_TIMING_TESTS, name)
        control, armed = env_of('c2-packed-tp4-speed-strace'), env_of('c2-packed-tp4-speed-rshard')
        self.assertEqual({key for key in set(control) | set(armed) if control.get(key) != armed.get(key)},
                         {RSHARD_ARM, 'QWEN_FAST_PACKED_SAMPLER_IN_TRACE'})
        self.assertEqual(names[names.index('R0-speed-base'):][:4], list(R0_R3), 'A B A B, back to back')

    def test_only_the_arm_profiles_carry_the_flag_and_the_audit_is_the_audited_arms(self):
        for name in RSHARD_JOBS:
            profile = parsed(name)['profile']
            carries = env_of(profile).get(RSHARD_ARM) == '1'
            self.assertEqual(carries, not name.startswith('R0'), name)
        self.assertEqual(env_of('c2-packed-tp4-gate-rshard-audit')['QWEN_FAST_REQUEST_SHARD_AUDIT'], '1')
        for name in ('c2-packed-tp4-diag-rshard', 'c2-packed-tp4-speed-rshard'):
            self.assertNotIn('QWEN_FAST_REQUEST_SHARD_AUDIT', env_of(name))
        for flag in (RSHARD_ARM, 'QWEN_FAST_REQUEST_SHARD_AUDIT'):
            for profile in (PRODUCTION, BEST, BEST_GATE):
                self.assertNotIn(flag, env_of(profile), profile)

    def test_the_audited_hang_shape_arm_is_the_t1_diag_arm_plus_the_two_flags_and_nothing_else(self):
        base, arm = env_of('c2-packed-tp4-diag-t1'), env_of('c2-packed-tp4-diag-t1-rshard-audit')
        self.assertEqual({key for key in set(base) | set(arm) if base.get(key) != arm.get(key)}, {RSHARD_ARM, 'QWEN_FAST_REQUEST_SHARD_AUDIT'})


class SmokeTests(unittest.TestCase):
    def test_every_named_test_is_one_the_smoke_knows_and_runs_in_the_order_the_template_lists_them(self):
        with open(os.path.join(HERE, 'c2_serving_smoke.py'), encoding='utf-8') as handle:
            smoke = handle.read()
        executed = re.findall(r"^\s*record\('([a-z0-9_]+)'", smoke, re.M)
        for name in ('N0-build-smoke', 'N1-best-gate-smoke-audited') + RSHARD_JOBS + TIMED:
            listed = tests_of(name)
            with self.subTest(template=name):
                for test in listed:
                    self.assertIn("'%s'" % test, smoke)
                self.assertEqual([test for test in executed if test in listed], listed)

    def test_n0_keeps_the_production_smoke_of_the_serve_window(self):
        self.assertEqual(tests_of('N0-build-smoke'), SS + ['concurrent4_solo'] + CODE)

    def test_the_audited_smoke_includes_the_4k_coding_reference_the_timed_best_is_held_against(self):
        self.assertEqual(tests_of('N1-best-gate-smoke-audited'),
                         ['warmup', 'coding', 'concurrent4', 'concurrent4_steady', 'steady_resend', 'concurrent4_code_equal',
                          'concurrent4_code_32k'])

    def test_the_four_timing_jobs_run_the_same_coding_tests_4x4k_and_4x32k(self):
        for name in TIMED:
            self.assertEqual(tests_of(name), TIMING_TESTS, name)

    def test_the_32k_test_is_four_users_of_32k_judged_as_a_code_answer(self):
        import c2_smoke_check

        with open(os.path.join(HERE, 'c2_serving_smoke.py'), encoding='utf-8') as handle:
            smoke = handle.read()
        self.assertIn('code_prompts((32768,) * 4)', smoke)
        for names in (c2_smoke_check.CONCURRENT_TESTS, c2_smoke_check.CORE, c2_smoke_check.TEXT_TESTS):
            self.assertIn('concurrent4_code_32k', names)


if __name__ == '__main__':
    unittest.main()
