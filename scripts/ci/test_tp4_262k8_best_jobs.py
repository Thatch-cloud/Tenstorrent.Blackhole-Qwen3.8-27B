"""tp4/262k8: the eight-seat 262k development window's job pack (references/tp4-262k8-best-jobs).

Every template parses with the job parser, names the one image, never touches the node agent (no agentstop, agentstart, unserve, platform), names no
production tag, bakes nothing, and the ORDER.txt lines match the files. The first quad job carries a rescan before its reset. The eight-user ops profile
(ops_profile_plan.plan_users) runs eight users on an eight-seat profile and the four it always ran on every other. The templates are public, so they name
no rig, card, address, registry or digest."""

import json
import os
import re
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import c2_serving_job as job  # noqa: E402
import ops_profile_plan  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
FOLDER = os.path.join(HERE, 'references', 'tp4-262k8-best-jobs')
PROFILES_PATH = os.path.join(HERE, 'qwen_c2_profiles.json')
IMAGE = 'tp4-262k8-best-1'
BANNED = re.compile(r'blackhole-[A-Za-z0-9]{8,}|thatch\.local|\d{1,3}(\.\d{1,3}){3}|sha256:[0-9a-f]{16}|[0-9a-f]{40,}|/dev/tenstorrent|home/|zot\.')
CONTROL_GATE = 'c2-packed-tp4-8x262k-gate'
CONTROL, BEST, AUDIT, TIMED = ('c2-packed-tp4-8x262k-time-gate', 'c2-packed-tp4-8x262k-best', 'c2-packed-tp4-8x262k-best-audit',
                               'c2-packed-tp4-8x262k-best-time-gate')
EXPECTED = {
    'X0-status-rescan-reset': ('status rescan reset', None, 'stop'), 'B0-build': ('build', 'c2-packed-tp4', 'stop'),
    'S0c-control-attach-smoke': ('reset smoke', CONTROL_GATE, 'stop'), 'S0-audited-attach-smoke': ('reset smoke', AUDIT, 'stop'),
    'H1-hang-shapes-best': ('reset smoke', TIMED, 'stop'), 'H2-hang-shapes-best': ('reset smoke', TIMED, 'stop'),
    'H3-hang-shapes-best': ('reset smoke', TIMED, 'stop'),
    'H4-stall8-cold262k-best': ('reset smoke', TIMED, 'stop'),
    'T1-timed-A-control-8x262k-time-gate': ('reset smoke', CONTROL, 'soft'), 'T2-timed-B-best-8x262k-time-gate': ('reset smoke', TIMED, 'soft'),
    'T3-timed-A-control-8x262k-time-gate': ('reset smoke', CONTROL, 'soft'), 'T4-timed-B-best-8x262k-time-gate': ('reset smoke', TIMED, 'soft'),
    'P1-best-8-user-profile': ('status reset gate', TIMED, 'soft'),
    'L1-ladder8-best': ('reset gate', AUDIT, 'soft'), 'C1-churn16-best': ('reset gate', BEST, 'soft'), 'Z-reset': ('status reset', None, 'soft'),
}
AGENT_ACTIONS = {'agentstop', 'agentstart', 'unserve', 'platform', 'replay', 'priority', 'cardm'}
LADDER = '4096,32768,131072,261856,16384,65536,200000,253920'


def profiles():
    with open(PROFILES_PATH, encoding='utf-8') as handle:
        return json.load(handle)['profiles']


def text_of(name):
    with open(os.path.join(FOLDER, name + '.env'), encoding='utf-8') as handle:
        return handle.read()


def parsed(name):
    return job.read_job(job.parse_env(text_of(name)), sorted(profiles()), root=ROOT)


def order_lines():
    with open(os.path.join(FOLDER, 'ORDER.txt'), encoding='utf-8') as handle:
        return [line.split() for line in handle.read().splitlines() if line.strip() and not line.startswith('#')]


def needs_lines():
    found = {}
    with open(os.path.join(FOLDER, 'ORDER.txt'), encoding='utf-8') as handle:
        for line in handle.read().splitlines():
            match = re.match(r'# NEEDS (.+) <- (.+)$', line)
            if match:
                for name in match.group(1).split():
                    found.setdefault(name, set()).update(match.group(2).split())
    return found


class JobPackTests(unittest.TestCase):
    def test_order_lists_exactly_the_templates_in_the_asked_order(self):
        lines = order_lines()
        names = [line[0] for line in lines]
        self.assertEqual(names, list(EXPECTED))
        files = sorted(name[:-4] for name in os.listdir(FOLDER) if name.endswith('.env'))
        self.assertEqual(files, sorted(names))
        for name, mode, image, minutes in lines:
            self.assertEqual((mode, image), (EXPECTED[name][2], IMAGE), name)
            self.assertTrue(minutes.isdigit() and int(minutes) > 0, name)

    def test_every_template_parses_with_the_expected_actions_and_profile(self):
        for name, (actions, profile, _mode) in EXPECTED.items():
            with self.subTest(name):
                result = parsed(name)
                self.assertEqual(result['actions'], actions)
                self.assertEqual(result['cards'], 'quad')
                self.assertEqual(result['tag'], IMAGE)
                if profile:
                    self.assertEqual(result['profile'], profile)

    def test_the_window_never_touches_the_node_agent_or_production(self):
        for name in EXPECTED:
            with self.subTest(name):
                result = parsed(name)
                self.assertFalse(AGENT_ACTIONS & set(result['actions'].split()))
                self.assertEqual(result['bake_default_profile'] or '', '')
        self.assertNotIn(IMAGE, job.PROTECTED)
        self.assertFalse(IMAGE.startswith(job.PROTECTED_PREFIXES))

    def test_the_first_quad_job_rescans_before_it_resets_and_the_window_ends_with_a_reset(self):
        actions = parsed(order_lines()[0][0])['actions'].split()
        self.assertLess(actions.index('rescan'), actions.index('reset'))
        self.assertEqual(order_lines()[-1][0], 'Z-reset')
        with open(PROFILES_PATH, encoding='utf-8') as handle:
            self.assertEqual(json.load(handle)['default'], 'c2-packed-tp4')

    def test_the_hang_shape_arms_carry_the_eight_seat_shapes_and_the_audits_off_profile(self):
        for name in ('H1-hang-shapes-best', 'H2-hang-shapes-best', 'H3-hang-shapes-best'):
            tests = parsed(name)['tests'].replace(' ', ',').split(',')
            for shape in ('concurrent8_steady', 'steady_resend', 'replay_concurrent8', 'concurrent8_code_equal'):
                self.assertIn(shape, tests, name)
            env = profiles()[parsed(name)['profile']]['env']
            self.assertEqual((env['QWEN_FAST_VERIFY_T1_AUDIT'], env['QWEN_FAST_VERIFY_T2_AUDIT']), ('0', '0'))

    def test_the_audited_smoke_runs_the_quad_steady_mix(self):
        tests = parsed('S0-audited-attach-smoke')['tests'].replace(' ', ',').split(',')
        self.assertIn('concurrent8_steady', tests)
        self.assertIn('concurrent8_code_equal', tests)

    def test_the_timed_pairs_alternate_abab_on_the_same_eight_user_coding_tests_and_differ_in_the_profile_alone(self):
        names = [name for name in EXPECTED if name.startswith('T')]
        self.assertEqual([parsed(name)['profile'] for name in names], [CONTROL, TIMED, CONTROL, TIMED])
        self.assertEqual(len({parsed(name)['tests'] for name in names}), 1)
        tests = parsed(names[0])['tests'].replace(' ', ',').split(',')
        self.assertTrue({'concurrent8_code_32k', 'concurrent8_code_128k'} <= set(tests))
        found = profiles()
        for key in ('QWEN_FAST_VERIFY_T1_AUDIT', 'QWEN_FAST_VERIFY_T2_AUDIT'):
            self.assertEqual(found[CONTROL]['env'][key], found[TIMED]['env'][key])

    def test_the_ladder_is_the_262k_ladder_on_the_audited_best_arm_with_the_pool_it_fits(self):
        result = parsed('L1-ladder8-best')
        self.assertEqual(result['profile'], AUDIT)
        text = text_of('L1-ladder8-best')
        self.assertIn('C2_GATE_PLAN=matrix', text)
        self.assertIn('C2_GATE_LENGTHS=' + LADDER, text)
        self.assertIn('C2_GATE_MAX_TOKENS=256', text)
        self.assertIn('C2_GATE_AUDITS=extent', text)

    def test_s0c_runs_the_same_smoke_as_s0_on_the_control_and_s0_needs_it(self):
        self.assertEqual(parsed('S0c-control-attach-smoke')['tests'], parsed('S0-audited-attach-smoke')['tests'])
        needs = needs_lines()
        self.assertEqual(needs['S0'], {'S0c'})
        self.assertIn('S0c', text_of('S0-audited-attach-smoke'))

    def test_the_soft_long_jobs_run_after_the_timing_and_the_profile_and_depend_on_the_gates(self):
        names = [line[0] for line in order_lines()]
        for late in ('L1-ladder8-best', 'C1-churn16-best'):
            self.assertGreater(names.index(late), names.index('P1-best-8-user-profile'))
        needs = needs_lines()
        for job_name in ('T1', 'T2', 'T3', 'T4', 'P1'):
            self.assertEqual(needs[job_name], {'S0', 'H1', 'H2', 'H3', 'H4'})
        for job_name in ('L1', 'C1'):
            self.assertEqual(needs[job_name], {'S0', 'H1', 'H2', 'H3', 'H4'})
        self.assertNotIn('L1', set().union(*needs.values()), 'nothing waits on the five-hour ladder')

    def test_the_cold_arrival_and_the_churn_are_the_262k_shapes_of_the_smaller_pool(self):
        self.assertEqual(parsed('H4-stall8-cold262k-best')['tests'], 'warmup,stall8_cold262k')
        text = text_of('C1-churn16-best')
        self.assertIn('C2_GATE_PLAN=churn', text)
        self.assertIn('C2_GATE_LENGTHS=253920,253920,253920,200000,200000,200000,131072,65536,32768,16384,8192,4096,253920,200000,131072,1536', text)

    def test_the_read_rule_converts_the_ledger_to_f8_and_the_p1_minutes_cover_two_arms(self):
        with open(os.path.join(FOLDER, 'ORDER.txt'), encoding='utf-8') as handle:
            order = handle.read()
        self.assertIn('1.979 GB', order)
        self.assertIn('1.979 GB', text_of('S0-audited-attach-smoke'))
        self.assertEqual((19968 - 16416) * 557056, 1978662912)
        minutes = {line[0]: int(line[3]) for line in order_lines()}
        self.assertGreaterEqual(minutes['P1-best-8-user-profile'], 2 * 105 + 20)

    def test_the_profile_job_names_the_ops_plans_on_the_timed_arm(self):
        text = text_of('P1-best-8-user-profile')
        self.assertIn('C2_GATE_PLAN=ops-twin,ops-trace', text)
        self.assertEqual(parsed('P1-best-8-user-profile')['profile'], TIMED)

    def test_no_hostname_address_registry_or_digest(self):
        for name in os.listdir(FOLDER):
            with open(os.path.join(FOLDER, name), encoding='utf-8') as handle:
                text = handle.read()
            self.assertIsNone(BANNED.search(text), name)
            self.assertNotIn('\r', text, name)


class EightUserOpsPlanTests(unittest.TestCase):
    class Gate:
        STAGGER = 2

        class PlanError(Exception):
            pass

        class Arm:
            def __init__(self, name, args, timeout, rerun, judged, role, ops):
                self.name, self.args, self.ops = name, args, ops

        @staticmethod
        def profile_limits(found, name):
            return 262144, 16384, 262144

        @staticmethod
        def check_lengths(*args):
            return None

        @staticmethod
        def check_budget(*args):
            return None

        @staticmethod
        def common_args(profile, context, stream, readiness=0):
            return ['--profile', profile]

    def arm(self, name):
        document = {'profiles': profiles()}
        arms = ops_profile_plan.plan_arms('ops-twin', name, document, self.Gate)
        return arms[0]

    def test_an_eight_seat_profile_gets_eight_users_and_the_ops_record_says_so(self):
        arm = self.arm(TIMED)
        args = arm.args
        self.assertEqual(args[args.index('--users') + 1], '8')
        self.assertEqual(args[args.index('--prompt-lengths') + 1], ','.join(['4096'] * 8))
        self.assertEqual(args[args.index('--user-ignore-eos') + 1], '0,1,2,3,4,5,6,7')
        self.assertEqual(arm.ops['users'], 8)

    def test_a_four_seat_profile_is_exactly_what_it_was(self):
        arm = self.arm('c2-packed-tp4-speed-strace')
        args = arm.args
        self.assertEqual(args[args.index('--users') + 1], '4')
        self.assertEqual(args[args.index('--prompt-lengths') + 1], '4096,4096,4096,4096')
        self.assertEqual(args[args.index('--user-ignore-eos') + 1], '0,1,2,3')
        self.assertNotIn('users', arm.ops)


if __name__ == '__main__':
    unittest.main()
