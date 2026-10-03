"""tp4/packed-prefix: the sticky-session development window's job pack (references/tp4-packed-prefix-jobs).

Every template parses with the job parser, names the one image, never touches the node agent, names no production tag, bakes
nothing, and the ORDER.txt lines match the files; the first quad job carries a rescan before its reset; the prefix jobs name
sticky-session profiles with their no-reuse controls; the agent-turn replay alternates the control and the prefix arm ABAB on
the timed pair with eight agents. The templates are public, so they name no rig, card, address, registry or digest."""

import json
import os
import re
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import c2_prefix_gate as prefix_gate  # noqa: E402
import c2_serving_job as job  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
FOLDER = os.path.join(HERE, 'references', 'tp4-packed-prefix-jobs')
PROFILES_PATH = os.path.join(HERE, 'qwen_c2_profiles.json')
IMAGE = 'tp4-packed-prefix-1'
BANNED = re.compile(r'blackhole-[A-Za-z0-9]{8,}|thatch\.local|\d{1,3}(\.\d{1,3}){3}|sha256:[0-9a-f]{16}|[0-9a-f]{40,}|/dev/tenstorrent|home/|zot\.')
CONTROL, TWIN = 'c2-packed-tp4-8x262k-best', 'c2-packed-tp4-8x262k-prefix-gate'
CONTROL_T, TWIN_T = 'c2-packed-tp4-8x262k-best-time-gate', 'c2-packed-tp4-8x262k-prefix-time-gate'
EXPECTED = {
    'X0-status-rescan-reset': ('status rescan reset', 'stop'), 'B0-build': ('build', 'stop'),
    'A0-control-attach-smoke': ('reset smoke', 'stop'), 'A1-audited-attach-smoke': ('reset smoke', 'stop'),
    'A2-bringup': ('reset prefix', 'stop'), 'E1-exactness-eager': ('reset prefix', 'stop'),
    'E2-exactness-shared8': ('reset prefix', 'stop'), 'R0-agent-turns-both': ('reset prefix', 'soft'),
    'R1-turns-A-control': ('reset prefix', 'soft'), 'R2-turns-B-prefix': ('reset prefix', 'soft'),
    'R3-turns-A-control': ('reset prefix', 'soft'), 'R4-turns-B-prefix': ('reset prefix', 'soft'),
    'Z-reset': ('status reset', 'soft'),
}
AGENT_ACTIONS = {'agentstop', 'agentstart', 'unserve', 'platform', 'replay', 'priority', 'cardm'}


def profiles():
    with open(PROFILES_PATH, encoding='utf-8') as handle:
        return json.load(handle)


def text_of(name):
    with open(os.path.join(FOLDER, name + '.env'), encoding='utf-8') as handle:
        return handle.read()


def parsed(name):
    return job.read_job(job.parse_env(text_of(name)), sorted(profiles()['profiles']), root=ROOT)


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
            self.assertEqual((mode, image), (EXPECTED[name][1], IMAGE), name)
            self.assertTrue(minutes.isdigit() and int(minutes) > 0, name)

    def test_every_template_parses_on_the_quad_with_the_expected_actions(self):
        for name, (actions, _mode) in EXPECTED.items():
            with self.subTest(name):
                result = parsed(name)
                self.assertEqual(result['actions'], actions)
                self.assertEqual(result['cards'], 'quad')
                self.assertEqual(result['tag'], IMAGE)

    def test_the_window_never_touches_the_node_agent_or_production(self):
        for name in EXPECTED:
            with self.subTest(name):
                result = parsed(name)
                self.assertFalse(AGENT_ACTIONS & set(result['actions'].split()))
                self.assertEqual(result['bake_default_profile'] or '', '')
        self.assertNotIn(IMAGE, job.PROTECTED)
        self.assertFalse(IMAGE.startswith(job.PROTECTED_PREFIXES))
        self.assertEqual(profiles()['default'], 'c2-packed-tp4')

    def test_the_first_quad_job_rescans_before_it_resets_and_the_window_ends_with_a_reset(self):
        actions = parsed(order_lines()[0][0])['actions'].split()
        self.assertLess(actions.index('rescan'), actions.index('reset'))
        self.assertEqual(order_lines()[-1][0], 'Z-reset')

    def test_the_attach_pair_is_the_control_then_the_twin_on_the_same_smoke(self):
        control, twin = parsed('A0-control-attach-smoke'), parsed('A1-audited-attach-smoke')
        self.assertEqual((control['profile'], twin['profile']), (CONTROL, TWIN))
        self.assertEqual(control['tests'], twin['tests'])
        self.assertIn('concurrent8_code_equal', control['tests'])
        self.assertEqual(needs_lines()['A1'], {'A0'})
        self.assertIn('ENGAGED', text_of('A1-audited-attach-smoke'))

    def test_the_prefix_jobs_name_a_sticky_profile_and_its_no_reuse_control(self):
        found = profiles()
        for name in ('A2-bringup', 'E1-exactness-eager', 'E2-exactness-shared8', 'R0-agent-turns-both', 'R1-turns-A-control',
                     'R2-turns-B-prefix', 'R3-turns-A-control', 'R4-turns-B-prefix'):
            with self.subTest(name):
                result = parsed(name)
                mine = found['profiles'][result['prefix_profile']]
                control = found['profiles'][result['prefix_baseline']]
                self.assertTrue(prefix_gate.is_sticky_profile(mine))
                self.assertFalse(prefix_gate.is_prefix_profile(control))
                self.assertIs(mine.get('gate_only'), True)
                self.assertEqual(mine['mesh_device'], 'P150x4')
                # every plan the job names builds its arms on those two profiles
                for plan in result['prefix_plan'].split(','):
                    arms = prefix_gate.plan_arms(plan, result['prefix_profile'], result['prefix_baseline'], found)
                    self.assertTrue(arms, plan)

    def test_the_audited_jobs_use_the_audited_pair_and_the_timed_jobs_the_timed_pair(self):
        for name in ('A2-bringup', 'E1-exactness-eager', 'E2-exactness-shared8'):
            result = parsed(name)
            self.assertEqual((result['prefix_profile'], result['prefix_baseline']), (TWIN, CONTROL), name)
        for name in ('R0-agent-turns-both', 'R1-turns-A-control', 'R2-turns-B-prefix', 'R3-turns-A-control', 'R4-turns-B-prefix'):
            result = parsed(name)
            self.assertEqual((result['prefix_profile'], result['prefix_baseline']), (TWIN_T, CONTROL_T), name)
            self.assertEqual(result['prefix_agents'], '8', name)
        found = profiles()['profiles']
        for key in ('QWEN_FAST_VERIFY_T1_AUDIT', 'QWEN_FAST_VERIFY_T2_AUDIT'):
            self.assertEqual(found[TWIN_T]['env'][key], '0')
            self.assertEqual(found[CONTROL_T]['env'][key], '0')

    def test_the_bringup_the_exactness_arms_and_the_replay_run_the_plans_they_are_named_for(self):
        self.assertEqual(parsed('A2-bringup')['prefix_plan'], 'bringup')
        self.assertEqual(parsed('E1-exactness-eager')['prefix_plan'], 'exactness-eager')
        self.assertEqual(parsed('E2-exactness-shared8')['prefix_plan'], 'exactness-shared')
        self.assertEqual(parsed('R0-agent-turns-both')['prefix_plan'], 'agent-turns')

    def test_the_replay_pairs_alternate_abab_with_the_control_first(self):
        plans = [parsed(name)['prefix_plan'] for name in ('R1-turns-A-control', 'R2-turns-B-prefix', 'R3-turns-A-control',
                                                           'R4-turns-B-prefix')]
        self.assertEqual(plans, ['agent-turns-baseline', 'agent-turns-prefix'] * 2)
        needs = needs_lines()
        for name in ('R0', 'R1', 'R2', 'R3', 'R4'):
            self.assertEqual(needs[name], {'A1', 'A2'})
        for name in ('A2', 'E1', 'E2'):
            self.assertEqual(needs[name], {'A1'})
        self.assertIn('prefix_agent_turns.py', text_of('R2-turns-B-prefix'))

    def test_the_pack_runs_its_plans_inside_the_job_budget(self):
        minutes = {line[0]: int(line[3]) for line in order_lines()}
        found = profiles()
        for name in ('A2-bringup', 'E1-exactness-eager', 'E2-exactness-shared8', 'R0-agent-turns-both', 'R1-turns-A-control'):
            result = parsed(name)
            arms = []
            for plan in result['prefix_plan'].split(','):
                arms += prefix_gate.plan_arms(plan, result['prefix_profile'], result['prefix_baseline'], found)
            worst = prefix_gate.worst_case_seconds({'plan': arms})
            self.assertLessEqual(worst, 380 * 60 - 600, name)
            self.assertGreater(minutes[name], 0)

    def test_no_hostname_address_registry_or_digest(self):
        for name in os.listdir(FOLDER):
            with open(os.path.join(FOLDER, name), encoding='utf-8') as handle:
                text = handle.read()
            self.assertIsNone(BANNED.search(text), name)
            self.assertNotIn('\r', text, name)


if __name__ == '__main__':
    unittest.main()
