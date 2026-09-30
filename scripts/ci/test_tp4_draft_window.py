"""The four-card batched-draft window: its job templates (scripts/ci/references/tp4-draft-jobs), their order, and the profiles and smoke
test they use (tp4/draft).

The templates are public, so they name no rig, card, address, registry or digest. Every template parses with c2_serving_job, opens
the cards in ONE step and resets first, uses ONE image tag, the smokes name only tests c2_serving_smoke knows (the steady four-user
test is opt-in, named), the audited arms serve the audited draft profiles and the timed arms the timed ones (each the speed or gate
profile plus the quad flag), the exactness matrix serves four steady users - the only mix the quad forms on - two of them the speed
window's E1 prompts, and the offline compares the templates name exist in speed_window_compare."""

import json
import os
import re
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import c2_serving_gate as gate  # noqa: E402
import c2_serving_job as job  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
FOLDER = os.path.join(HERE, 'references', 'tp4-draft-jobs')
SPEED_FOLDER = os.path.join(HERE, 'references', 'tp4-speed-jobs')
with open(os.path.join(HERE, 'qwen_c2_profiles.json'), encoding='utf-8') as _handle:
    PROFILES = json.load(_handle)
NAMES = sorted(PROFILES['profiles'])
DEVICE_STEPS = ('cardm', 'smoke', 'gate', 'prefix', 'fabric', 'replay')
BANNED = re.compile(r'blackhole-[A-Za-z0-9]{8,}|thatch\.local|\d{1,3}(\.\d{1,3}){3}|sha256:[0-9a-f]{16}|[0-9a-f]{40,}|'
                    r'/dev/tenstorrent|home/|zot\.')
SMOKES = ('D1-quad-smoke', 'D2-pairs-smoke', 'D4-quad-smoke-timed', 'D5-pairs-smoke-timed')
EXPECTED_PROFILES = {'D0-build': 'c2-packed-tp4-speed-quad', 'D1-quad-smoke': 'c2-packed-tp4-gate-quad',
                     'D2-pairs-smoke': 'c2-packed-tp4-gate-pairs', 'D3-quad-matrix': 'c2-packed-tp4-gate-quad',
                     'D4-quad-smoke-timed': 'c2-packed-tp4-speed-quad', 'D5-pairs-smoke-timed': 'c2-packed-tp4-speed-pairs'}
QUAD, SINGLES = 'QWEN_FAST_QUAD_DRAFT', 'QWEN_FAST_DRAFT_SINGLES_AUDIT'


def read_order(folder=FOLDER):
    with open(os.path.join(folder, 'ORDER.txt'), encoding='utf-8') as handle:
        return [tuple(line.split()) for line in handle.read().splitlines() if line.strip() and not line.startswith('#')]


def text_of(name, folder=FOLDER):
    with open(os.path.join(folder, name + '.env'), encoding='utf-8') as handle:
        return handle.read()


def parsed(name, folder=FOLDER):
    values = job.parse_env(text_of(name, folder))
    return values, job.read_job(values, NAMES)


def order_text():
    with open(os.path.join(FOLDER, 'ORDER.txt'), encoding='utf-8') as handle:
        return handle.read()


def env_of(name):
    return PROFILES['profiles'][parsed(name)[1]['profile']]['env']


class OrderTests(unittest.TestCase):
    def test_every_template_is_in_the_order_once_and_the_order_names_no_other(self):
        on_disk = sorted(name[:-4] for name in os.listdir(FOLDER) if name.endswith('.env'))
        ordered = [name for name, _ in read_order()]
        self.assertEqual(sorted(ordered), on_disk)
        self.assertEqual(len(set(ordered)), len(ordered))

    def test_the_stages_build_then_the_audited_smokes_then_exactness_then_the_timed_arms(self):
        order = read_order()
        self.assertEqual([name.split('-')[0] for name, _ in order], ['D0', 'D1', 'D2', 'D4', 'D5', 'D3'])
        self.assertEqual(order[0], ('D0-build', 'stop'))
        self.assertEqual([name for name, mode in order if mode == 'soft'],
                         ['D2-pairs-smoke', 'D4-quad-smoke-timed', 'D5-pairs-smoke-timed', 'D3-quad-matrix'])
        self.assertEqual({mode for _, mode in order}, {'stop', 'soft'})
        self.assertEqual([name for name, mode in order if mode == 'stop'],
                         ['D0-build', 'D1-quad-smoke'], 'D3 is soft and last: its text policy can fail on the verify side')

    def test_every_template_parses_names_no_host_and_uses_the_one_image_tag(self):
        for name, _ in read_order():
            with self.subTest(template=name):
                values, outputs = parsed(name)
                self.assertIsNone(BANNED.search(text_of(name)), name)
                self.assertEqual(outputs['tag'], 'tp4-draft-1')
                self.assertEqual(outputs['cards'], 'quad')
                self.assertEqual(outputs['profile'], EXPECTED_PROFILES[name])
                self.assertEqual(set(re.findall(r'@[A-Z0-9_]+@', text_of(name))), set(), 'no placeholder: no one-card job here')
        self.assertIn('PLACEHOLDER', text_of('D0-build'), 'the image tag is a placeholder, and D0 says so')
        self.assertIsNone(BANNED.search(order_text()))

    def test_the_build_opens_no_card_and_a_run_resets_first_and_opens_the_cards_once(self):
        for name, _ in read_order():
            _, outputs = parsed(name)
            actions = outputs['actions'].split()
            with self.subTest(template=name):
                if name.startswith('D0'):
                    self.assertEqual(actions, ['status', 'reset', 'build'])
                    continue
                self.assertEqual(len(set(actions) & set(DEVICE_STEPS)), 1)
                self.assertEqual(actions[0], 'reset', 'links train only at board init: a four-card run follows a reset')

    def test_every_template_states_its_duration_and_its_stop_rule(self):
        for name, mode in read_order():
            text = text_of(name)
            with self.subTest(template=name):
                self.assertRegex(text, r'# Duration \(estimate')
                self.assertRegex(text, r'\d+ / \d+ / \d+ min|most 90')
                self.assertIn('# STOP' if mode == 'stop' or name.startswith('D0') else '# SOFT', text)
                if name != 'D0-build':
                    self.assertIn('# Read:', text)


class SmokeTests(unittest.TestCase):
    def source(self):
        with open(os.path.join(HERE, 'c2_serving_smoke.py'), encoding='utf-8') as handle:
            return handle.read()

    def test_the_smokes_name_tests_the_smoke_knows_and_run_the_same_four(self):
        smoke = self.source()
        tests = set()
        for name in SMOKES:
            _, outputs = parsed(name)
            self.assertEqual(outputs['actions'].split(), ['reset', 'smoke'], name)
            listed = outputs['tests'].split(',')
            for test in listed:
                self.assertTrue("record('%s'" % test in smoke, (name, test))
            tests.add(tuple(listed))
        self.assertEqual(tests, {('warmup', 'coding', 'concurrent4', 'concurrent4_steady')}, 'the A/B smokes run the same tests')

    def test_the_steady_test_is_opt_in_and_the_only_one_that_can_form_a_batch(self):
        smoke = self.source()
        self.assertIn("if ONLY and 'concurrent4_steady' in ONLY:", smoke)
        # four prompts of 14,000 characters (about 3,500 tokens): every user past the 2,048-row window from its first round
        self.assertIn('span = 14000', smoke)
        self.assertIn('for index in range(4)', smoke)
        compile(smoke, 'c2_serving_smoke.py', 'exec')   # the rig host's python 3.7 takes the same syntax: no walrus, no match


class ProfileTests(unittest.TestCase):
    def test_the_window_serves_the_four_draft_profiles_and_no_other(self):
        served = {parsed(name)[1]['profile'] for name, _ in read_order()}
        self.assertEqual(served, {'c2-packed-tp4-speed-quad', 'c2-packed-tp4-speed-pairs', 'c2-packed-tp4-gate-quad',
                                  'c2-packed-tp4-gate-pairs'})
        for name in served:
            self.assertEqual(PROFILES['profiles'][name]['mesh_device'], 'P150x4', name)
            self.assertIs(PROFILES['profiles'][name]['gate_only'], True, name)

    def test_the_flag_is_on_in_the_quad_jobs_and_off_in_the_pairs_jobs(self):
        self.assertEqual([(name, env_of(name)[QUAD]) for name in SMOKES + ('D3-quad-matrix', 'D0-build')],
                         [('D1-quad-smoke', '1'), ('D2-pairs-smoke', '0'), ('D4-quad-smoke-timed', '1'),
                          ('D5-pairs-smoke-timed', '0'), ('D3-quad-matrix', '1'), ('D0-build', '1')])
        for on, off in (('D1-quad-smoke', 'D2-pairs-smoke'), ('D4-quad-smoke-timed', 'D5-pairs-smoke-timed')):
            self.assertEqual(parsed(on)[1]['tests'], parsed(off)[1]['tests'])
            left, right = env_of(on), env_of(off)
            self.assertEqual({key for key in set(left) | set(right) if left.get(key) != right.get(key)}, {QUAD}, (on, off))

    def test_the_audited_jobs_carry_the_singles_audit_and_the_timed_ones_no_audit_at_all(self):
        for name in ('D1-quad-smoke', 'D2-pairs-smoke', 'D3-quad-matrix'):
            env = env_of(name)
            self.assertEqual((env[SINGLES], env['QWEN_FAST_VERIFY_T1_AUDIT'], env['QWEN_FAST_VERIFY_T2_AUDIT']), ('all', '1', '1'), name)
        for name in ('D4-quad-smoke-timed', 'D5-pairs-smoke-timed', 'D0-build'):
            env = env_of(name)
            self.assertNotIn(SINGLES, env, name)
            self.assertEqual((env['QWEN_FAST_VERIFY_T1_AUDIT'], env['QWEN_FAST_VERIFY_T2_AUDIT']), ('0', '0'), name)

    def test_the_four_card_quad_is_served_without_the_live_banks_the_pair_needed(self):
        for name in EXPECTED_PROFILES:
            self.assertEqual(env_of(name)['QWEN_FAST_FUSED_COMMIT_LIVE_BANKS'], '0', name)
            self.assertEqual(env_of(name)['QWEN_FAST_TP'], '4', name)


class MatrixTests(unittest.TestCase):
    def test_the_exactness_matrix_serves_four_steady_users_two_of_them_the_speed_windows_e1_prompts(self):
        outputs = parsed('D3-quad-matrix')[1]
        lengths = [int(part) for part in outputs['gate_lengths'].split(',')]
        self.assertEqual(len(lengths), 4, 'four users, four seats: the concurrent arm is a packed-4 round')
        self.assertTrue(all(length > 2048 for length in lengths), 'past the 2,048-row draft window: the quad forms')
        speed = parsed('E1-matrix-slide-on', SPEED_FOLDER)[1]
        e1 = [int(part) for part in speed['gate_lengths'].split(',')]
        self.assertEqual(lengths[2:], e1[2:], 'users 2 and 3 are E1\'s: the corpus fixes a prompt by user index and length')
        self.assertEqual(outputs['gate_max_tokens'], speed['gate_max_tokens'], 'the same answer budget: the texts compare')
        self.assertEqual((outputs['gate_plan'], outputs['gate_jit']), ('matrix', 'record'))
        self.assertEqual(outputs['gate_audits'], '', 'the prestage and fused-commit audits are not ported to four cards')
        self.assertIn(job.parse_env(text_of('D3-quad-matrix'))['C2_GATE_MAX_TOKENS'], ('256',))

    def test_the_gate_plan_is_accepted_on_the_profile_and_fits_the_step(self):
        outputs = parsed('D3-quad-matrix')[1]
        lengths = [int(part) for part in outputs['gate_lengths'].split(',')]
        arms = gate.plan_arms('matrix', outputs['profile'], PROFILES, lengths, int(outputs['gate_max_tokens']), None, [], {})
        self.assertEqual([arm[0] for arm in arms], ['matrix-concurrent', 'matrix-solo'])
        worst = 2 * sum(arm[2] + gate.ARM_OVERHEAD_SECONDS for arm in arms)
        self.assertLessEqual(worst, 380 * 60 - 600)

    def test_a_gate_only_profile_is_booted_with_the_gate_switch_in_the_dry_run_argv(self):
        outputs = parsed('D3-quad-matrix')[1]
        lines = []
        code = gate.main(['--image', 'img', '--profile', outputs['profile'], '--plan', 'matrix', '--cards', 'quad', '--dry-run',
                          '--results', os.path.join(HERE, 'no-results'), '--profiles', os.path.join(HERE, 'qwen_c2_profiles.json'),
                          '--lengths', outputs['gate_lengths'], '--max-tokens', outputs['gate_max_tokens'],
                          '--jit', outputs['gate_jit'] or 'auto'], log=lines.append)
        self.assertEqual(code, 0)
        arms = [json.loads(line)['docker'] for line in lines[1:]]
        self.assertTrue(arms)
        for argv in arms:
            self.assertIn('QWEN_C2_GATE=1', argv)
            self.assertIn('QWEN_C2_PROFILE=%s' % outputs['profile'], argv)


class CompareTests(unittest.TestCase):
    def test_the_offline_commands_the_templates_name_are_the_compares_own(self):
        order = order_text()
        matrix = text_of('D3-quad-matrix')
        with open(os.path.join(HERE, 'speed_window_compare.py'), encoding='utf-8') as handle:
            compare = handle.read()
        for text in (order, matrix):
            for flag in re.findall(r'speed_window_compare\.py [^\n]*?(--[a-z-]+)', text):
                self.assertIn("'%s'" % flag, compare, flag)
        for flag in ('--position-keyed', '--strict-concurrent-prefixes', '--users'):
            self.assertIn(flag, order)
            self.assertIn(flag, matrix)
            self.assertIn("'%s'" % flag, compare)
        for text in (order, matrix):
            self.assertNotIn('speed_window_compare.py <D3 results> --batched-vs-singles', text)
        self.assertIn('<E1 results> <D3 results> --users 2,3 --position-keyed --strict-concurrent-prefixes', matrix)

    def test_the_stop_rules_the_smokes_name_are_enforced_by_the_smoke_check(self):
        import c2_smoke_check as check

        for name in ('D1-quad-smoke', 'D2-pairs-smoke', 'D4-quad-smoke-timed', 'D5-pairs-smoke-timed'):
            _, outputs = parsed(name)
            self.assertIn('concurrent4_steady', outputs['tests'].split(','), name)
            self.assertIn(check.STEADY_TEST, check.CORE)
        for name, wants_quad in (('D1-quad-smoke', True), ('D2-pairs-smoke', False), ('D4-quad-smoke-timed', True),
                                 ('D5-pairs-smoke-timed', False)):
            self.assertEqual(env_of(name)[check.QUAD_FLAG] == '1', wants_quad, name)
            self.assertIn('ENFORCES' if name.startswith(('D2', 'D4', 'D5')) else 'ENFORCE', text_of(name))


if __name__ == '__main__':
    unittest.main()
