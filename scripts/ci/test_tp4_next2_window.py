"""The tp4/next-2 window: its job templates (scripts/ci/references/tp4-next2-jobs), their order, and the profiles they use.

tp4/next-2 is tp4/serve-4 (the hang-fix arms) merged with tp4/next (the round-time levers). N0 builds the image tp4-next-2 and smokes
the PRODUCTION profile (the merge must not have changed it); N1 is the audited smoke of the best gate profile; N2 and N3 are the strict
exactness plans (S3a matrix, staggered trigger) on it; N4a..N4d are the paired timing ABAB of the timed best against its control, the
speed twin, on coding prompts (4 x 4k, 4 x 32k); R1, R2, R0 and R3 (optional, after N0) test the request engine's shard argmax
(QWEN_FAST_REQUEST_SHARD_ARGMAX) on the hang shapes, audited, and timed against the speed twin; N5 hands the cards back. Every job resets the cards first. The templates are public,
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
TIMED = ('N4a-speed-timed', 'N4b-best-timed', 'N4c-speed-timed', 'N4d-best-timed')
CONTROL, BEST, BEST_GATE, PRODUCTION = 'c2-packed-tp4-speed', 'c2-packed-tp4-best', 'c2-packed-tp4-best-gate', 'c2-packed-tp4'
RSHARD_JOBS = ('R1-diag-rshard', 'R2-rshard-audit', 'R0-speed-base', 'R3-speed-rshard')
ORDERED = ('N0-build-smoke',) + RSHARD_JOBS + ('N1-best-gate-smoke-audited', 'N2-s3a-matrix-best-gate', 'N3-staggered-trigger-best-gate') + TIMED + (
    'N5-handback-reset',)
WORK = ORDERED[:-1]
RSHARD_ARM = 'QWEN_FAST_REQUEST_SHARD_ARGMAX'
HANG_SHAPES = ['warmup', 'coding', 'long_real_text', 'concurrent4', 'concurrent4_v164order', 'concurrent4_steady', 'replay_concurrent4']
RSHARD_TIMING_TESTS = ['warmup', 'coding', 'concurrent4_code', 'concurrent4_code_equal', 'concurrent8_code']
PROFILE_OF = {'N0-build-smoke': PRODUCTION, 'R1-diag-rshard': 'c2-packed-tp4-diag-rshard', 'R2-rshard-audit': 'c2-packed-tp4-gate-rshard-audit',
              'R0-speed-base': CONTROL, 'R3-speed-rshard': 'c2-packed-tp4-speed-rshard', 'N1-best-gate-smoke-audited': BEST_GATE, 'N2-s3a-matrix-best-gate': BEST_GATE,
              'N3-staggered-trigger-best-gate': BEST_GATE, 'N4a-speed-timed': CONTROL, 'N4b-best-timed': BEST,
              'N4c-speed-timed': CONTROL, 'N4d-best-timed': BEST}
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
        for name in ('N0-build-smoke',) + ORDERED[5:8] + ('N5-handback-reset',):
            self.assertEqual(modes[name], 'stop', name)
        for name in TIMED + RSHARD_JOBS:
            self.assertEqual(modes[name], 'optional', name)

    def test_the_hand_back_is_last_and_the_order_says_it_runs_even_when_a_stop_job_halts_the_window(self):
        self.assertEqual(read_order()[-1][0], 'N5-handback-reset')
        with open(os.path.join(FOLDER, 'ORDER.txt'), encoding='utf-8') as handle:
            order = handle.read()
        self.assertIn('runs even when a stop job', order)
        text = text_of('N5-handback-reset')
        self.assertEqual(parsed('N5-handback-reset')['actions'], 'status reset')
        self.assertEqual(parsed('N5-handback-reset')['cards'], 'quad')
        for word in ('AUDITED production image', 'NEVER place a gate arm', 'fabric', ':latest retag', 'admin API'):
            self.assertIn(word, text)

    def test_the_order_names_the_control_the_hang_fix_precondition_and_the_exactness_before_timing_rule(self):
        with open(os.path.join(FOLDER, 'ORDER.txt'), encoding='utf-8') as handle:
            order = handle.read()
        for word in ('c2-packed-tp4-speed, not c2-packed-tp4-time-gate', 'hang-fix arm', 'AND c2-packed-tp4-best',
                     'ABAB', 'NOTHING COMBINED HAS RUN ON A CARD'):
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
        for flag in ('QWEN_FAST_VERIFY_T1_AUDIT', 'QWEN_FAST_VERIFY_T2_AUDIT'):
            self.assertEqual(env[flag], '1', 'the production profile serves with the audits on')

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
        self.assertEqual([parsed(name)['profile'] for name in TIMED], [CONTROL, BEST, CONTROL, BEST], 'A B A B')


class RequestShardJobTests(unittest.TestCase):
    """R1, R2, R0, R3: the request engine's shard argmax (QWEN_FAST_REQUEST_SHARD_ARGMAX) on the hang shapes, audited, and timed paired."""

    def test_the_four_jobs_follow_n0_in_the_order_and_are_optional(self):
        names = [row[0] for row in read_order()]
        self.assertEqual(names[:5], ['N0-build-smoke', 'R1-diag-rshard', 'R2-rshard-audit', 'R0-speed-base', 'R3-speed-rshard'])
        modes = {row[0]: row[1] for row in read_order()}
        for name in RSHARD_JOBS:
            self.assertEqual(modes[name], 'optional', name)

    def test_r1_runs_the_hang_shapes_on_the_diag_arm_and_r2_the_production_smoke_on_the_audited_arm(self):
        self.assertEqual(tests_of('R1-diag-rshard'), HANG_SHAPES)
        self.assertEqual(tests_of('R2-rshard-audit'), SS + ['concurrent4_solo'] + CODE)
        self.assertEqual(tests_of('R2-rshard-audit'), tests_of('N0-build-smoke'), 'the production smoke of the serve window (B5)')

    def test_r0_is_the_speed_twin_and_r3_the_speed_twin_plus_the_flag_on_the_same_tests(self):
        self.assertEqual(parsed('R0-speed-base')['profile'], CONTROL)
        self.assertEqual(tests_of('R0-speed-base'), RSHARD_TIMING_TESTS)
        self.assertEqual(tests_of('R3-speed-rshard'), RSHARD_TIMING_TESTS)
        control, armed = env_of(CONTROL), env_of('c2-packed-tp4-speed-rshard')
        self.assertEqual({key for key in set(control) | set(armed) if control.get(key) != armed.get(key)}, {RSHARD_ARM})
        names = [row[0] for row in read_order()]
        self.assertEqual(names.index('R0-speed-base') + 1, names.index('R3-speed-rshard'), 'the baseline runs just before R3')

    def test_only_the_arm_profiles_carry_the_flag_and_the_audit_is_the_audited_arms(self):
        for name in RSHARD_JOBS:
            profile = parsed(name)['profile']
            carries = env_of(profile).get(RSHARD_ARM) == '1'
            self.assertEqual(carries, name != 'R0-speed-base', name)
        self.assertEqual(env_of('c2-packed-tp4-gate-rshard-audit')['QWEN_FAST_REQUEST_SHARD_AUDIT'], '1')
        for name in ('c2-packed-tp4-diag-rshard', 'c2-packed-tp4-speed-rshard'):
            self.assertNotIn('QWEN_FAST_REQUEST_SHARD_AUDIT', env_of(name))
        for flag in (RSHARD_ARM, 'QWEN_FAST_REQUEST_SHARD_AUDIT'):
            for profile in (PRODUCTION, BEST, BEST_GATE):
                self.assertNotIn(flag, env_of(profile), profile)


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
