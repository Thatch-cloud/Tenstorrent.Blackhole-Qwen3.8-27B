"""tp4/lever-n: the Lever N development window's job pack (references/tp4-lever-n-jobs).

Every template parses with the job parser, names the one image, never touches the node agent (no agentstop, agentstart, unserve, platform), names no production tag,
bakes nothing, and the ORDER.txt lines match the files. The first quad job carries a rescan before its reset. The exactness pair (A0c control, A1 interleaved) runs the
same tests, the stall ABAB alternates control and interleaved on the same tests, the hang shapes run three times on the audits-off interleaved arm, and every test a template
names is one the smoke defines and the check judges. The templates are public, so they name no rig, card, address, registry or digest."""

import ast
import json
import os
import re
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import c2_serving_job as job  # noqa: E402
import c2_smoke_check as check  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
FOLDER = os.path.join(HERE, 'references', 'tp4-lever-n-jobs')
PROFILES_PATH = os.path.join(HERE, 'qwen_c2_profiles.json')
SMOKE = os.path.join(HERE, 'c2_serving_smoke.py')
IMAGE = 'tp4-lever-n-1'
BANNED = re.compile(r'blackhole-[A-Za-z0-9]{8,}|thatch\.local|\d{1,3}(\.\d{1,3}){3}|sha256:[0-9a-f]{16}|[0-9a-f]{40,}|/dev/tenstorrent|home/|zot\.')
P = 'c2-packed-tp4-8x262k-best'
CONTROL, CONTROL_AUDIT, TIMED, R1, AUDIT, HOLD, FOREIGN, HANG = (
    P + '-time-gate', P + '-levern-control-audit', P + '-levern-time-gate', P + '-levern-r1-time-gate', P + '-levern-audit',
    P + '-levern-final-hold-time-gate', P + '-levern-foreign-time-gate', P + '-levern-hang-gate')
EXPECTED = {
    'X0-status-rescan-reset': ('status rescan reset', None, 'stop'), 'B0-build': ('build', 'c2-packed-tp4', 'stop'),
    'A0c-control-audit-attach': ('reset smoke', CONTROL_AUDIT, 'stop'), 'A1-levern-audited-attach': ('reset smoke', AUDIT, 'stop'),
    'H1-hang-shapes-levern': ('reset smoke', HANG, 'stop'), 'H2-hang-shapes-levern': ('reset smoke', HANG, 'stop'),
    'H3-hang-shapes-levern': ('reset smoke', HANG, 'stop'),
    'F1-final-hold-fault': ('reset smoke', HOLD, 'soft'),
    'S1-stall-A-control': ('reset smoke', CONTROL, 'soft'), 'S2-stall-B-levern': ('reset smoke', TIMED, 'soft'),
    'S3-stall-A-control': ('reset smoke', CONTROL, 'soft'), 'S4-stall-B-levern': ('reset smoke', TIMED, 'soft'),
    'S5-stall-C-levern-r1': ('reset smoke', R1, 'soft'),
    'F2-foreign-fault': ('reset smoke', FOREIGN, 'soft'), 'Z-reset': ('status reset', None, 'soft'),
}
AGENT_ACTIONS = {'agentstop', 'agentstart', 'unserve', 'platform', 'replay', 'priority', 'cardm'}


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


def tests_of(name):
    return [test for test in parsed(name)['tests'].replace(' ', ',').split(',') if test]


def smoke_functions():
    with open(SMOKE, encoding='utf-8') as handle:
        tree = ast.parse(handle.read())
    return {node.name for node in tree.body if isinstance(node, ast.FunctionDef)}


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

    def test_every_profile_it_names_is_a_gate_only_profile_and_none_is_production(self):
        found = profiles()
        for name in EXPECTED:
            profile = parsed(name)['profile']
            if profile and name != 'B0-build' and 'C2_PROFILE=' in text_of(name):
                with self.subTest(name):
                    self.assertTrue(found[profile].get('gate_only'), profile)
                    self.assertNotIn(profile, ('c2-packed-tp4', 'c2-packed-tp4-8x262k-ship'))

    def test_every_test_a_template_names_is_defined_by_the_smoke(self):
        defined = smoke_functions()
        always_defined = {'warmup', 'coding', 'concurrent8_steady', 'concurrent8_drain', 'stall8_cold128k', 'stall8_cold262k'}
        for name in EXPECTED:
            for test in tests_of(name):
                with self.subTest(name=name, test=test):
                    if test in always_defined:
                        continue
                    self.assertIn(test, defined)
        # the stall tests and the shapes the pack runs are judged by the check
        for test in ('levern_equal', 'levern_equal_busy', 'levern_equal_long', 'levern_decoder_finishes', 'levern_all_decoders_finish',
                     'levern_cancel_mid_prefill', 'levern_arrival_during_prefill', 'levern_seed_stops'):
            self.assertIn(test, check.LEVERN_ROW_TESTS + check.LEVERN_USER_TESTS)
        self.assertEqual(set(check.STALL_TESTS), {'stall8_cold128k', 'stall8_cold262k'})

    def test_the_exactness_pair_runs_the_same_tests_and_the_interleaved_arm_needs_the_control(self):
        self.assertEqual(parsed('A0c-control-audit-attach')['tests'], parsed('A1-levern-audited-attach')['tests'])
        for needed in ('levern_equal', 'levern_equal_busy', 'levern_equal_long'):
            self.assertIn(needed, tests_of('A1-levern-audited-attach'))
        self.assertEqual(needs_lines()['A1'], {'A0c'})
        self.assertIn('levern_compare.py', text_of('A1-levern-audited-attach'))
        found = profiles()
        self.assertEqual(found[CONTROL_AUDIT]['env']['QWEN_FAST_VERIFY_T1_AUDIT'], found[AUDIT]['env']['QWEN_FAST_VERIFY_T1_AUDIT'])

    def test_the_hang_shapes_run_three_times_on_the_audits_off_interleaved_arm(self):
        for name in ('H1-hang-shapes-levern', 'H2-hang-shapes-levern', 'H3-hang-shapes-levern'):
            tests = tests_of(name)
            for shape in ('levern_decoder_finishes', 'levern_all_decoders_finish', 'levern_cancel_mid_prefill', 'levern_arrival_during_prefill',
                          'levern_seed_stops', 'concurrent8_steady'):
                self.assertIn(shape, tests, name)
            env = profiles()[parsed(name)['profile']]['env']
            self.assertEqual((env['QWEN_FAST_VERIFY_T1_AUDIT'], env['QWEN_FAST_VERIFY_T2_AUDIT']), ('0', '0'))
        self.assertEqual(len({parsed(name)['tests'] for name in ('H1-hang-shapes-levern', 'H2-hang-shapes-levern', 'H3-hang-shapes-levern')}), 1)

    def test_the_stall_pairs_alternate_abab_on_the_same_tests_and_both_cold_lengths(self):
        names = ['S1-stall-A-control', 'S2-stall-B-levern', 'S3-stall-A-control', 'S4-stall-B-levern']
        self.assertEqual([parsed(name)['profile'] for name in names], [CONTROL, TIMED, CONTROL, TIMED])
        self.assertEqual(len({parsed(name)['tests'] for name in names + ['S5-stall-C-levern-r1']}), 1)
        self.assertEqual(tests_of('S1-stall-A-control'), ['warmup', 'stall8_cold128k', 'stall8_cold262k'])
        found = profiles()
        for key in ('QWEN_FAST_VERIFY_T1_AUDIT', 'QWEN_FAST_VERIFY_T2_AUDIT', 'QWEN_FAST_KV_RESERVATION'):
            self.assertEqual(found[CONTROL]['env'][key], found[TIMED]['env'][key])
            self.assertEqual(found[CONTROL]['env'][key], found[R1]['env'][key])
        self.assertEqual(found[CONTROL]['engine']['num-gpu-blocks-override'], found[TIMED]['engine']['num-gpu-blocks-override'])

    def test_the_negative_controls_run_on_their_own_fault_profiles(self):
        self.assertEqual(profiles()[HOLD]['env']['QWEN_FAST_LEVERN_FAULT'], 'final-hold')
        self.assertEqual(profiles()[FOREIGN]['env']['QWEN_FAST_LEVERN_FAULT'], 'foreign')
        self.assertIn('levern_decoder_finishes', tests_of('F1-final-hold-fault'))
        self.assertIn('levern_equal', tests_of('F2-foreign-fault'))
        self.assertIn('fails by design', text_of('F2-foreign-fault').lower())
        self.assertEqual(EXPECTED['F2-foreign-fault'][2], 'soft')
        names = [line[0] for line in order_lines()]
        self.assertEqual(names[-2], 'F2-foreign-fault', 'the job that kills its engine runs last of the engine jobs')

    def test_the_dependencies(self):
        needs = needs_lines()
        self.assertEqual(needs['H1'], {'A1'})
        self.assertEqual(needs['F1'], {'A1'})
        self.assertEqual(needs['F2'], {'A1'})
        for stall in ('S1', 'S2', 'S3', 'S4', 'S5'):
            self.assertEqual(needs[stall], {'A1', 'H1', 'H2', 'H3'})
        names = [line[0] for line in order_lines()]
        for late in ('S1-stall-A-control', 'F2-foreign-fault'):
            self.assertGreater(names.index(late), names.index('H3-hang-shapes-levern'))
        self.assertNotIn('F2', set().union(*needs.values()), 'nothing waits on the job that kills its engine')

    def test_the_compare_commands_the_pack_names_exist(self):
        order = open(os.path.join(FOLDER, 'ORDER.txt'), encoding='utf-8').read()
        for command in ('levern_compare.py',):
            self.assertIn(command, order + text_of('A1-levern-audited-attach') + text_of('S1-stall-A-control'))
        self.assertTrue(os.path.exists(os.path.join(HERE, 'levern_compare.py')))

    def test_the_rescan_and_the_standing_rules_are_in_the_order_header(self):
        order = open(os.path.join(FOLDER, 'ORDER.txt'), encoding='utf-8').read()
        for needle in ('NO agentstart, NO agentstop', 'UNQUALIFIED', 'PAIRED', 'qwen-cpu-suite.yml'):
            self.assertIn(needle, order)

    def test_no_hostname_address_registry_or_digest(self):
        for name in os.listdir(FOLDER):
            with open(os.path.join(FOLDER, name), encoding='utf-8') as handle:
                text = handle.read()
            self.assertIsNone(BANNED.search(text), name)
            self.assertNotIn('\r', text, name)

    def test_the_minutes_cover_what_each_job_runs(self):
        minutes = {line[0]: int(line[3]) for line in order_lines()}
        self.assertGreaterEqual(minutes['A0c-control-audit-attach'], 90)
        self.assertGreaterEqual(minutes['A1-levern-audited-attach'], 90)
        for stall in ('S1-stall-A-control', 'S2-stall-B-levern', 'S3-stall-A-control', 'S4-stall-B-levern', 'S5-stall-C-levern-r1'):
            self.assertGreaterEqual(minutes[stall], 60)


if __name__ == '__main__':
    unittest.main()
