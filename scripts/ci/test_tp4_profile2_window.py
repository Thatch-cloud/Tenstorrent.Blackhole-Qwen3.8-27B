"""The tp4/next-3-prof window: its job templates (scripts/ci/references/tp4-profile2-jobs), their order, and what they run.

The window takes production down (PA0: agentstop, unserve), builds the tp4-prof-2 image and profiles the production-shaped four-card
verify (P2: c2-packed-tp4-speed-strace, the gate plans ops-twin and ops-trace), and resets the cards (PH). The fabric re-measure, the
agentstart and the owner's /deploy that bring production back are outside the templates. The templates are public, so they name no
rig, card, address, registry or digest."""

import json
import os
import re
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import c2_serving_gate as driver  # noqa: E402
import c2_serving_job as job  # noqa: E402
import ops_profile_plan as ops  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
FOLDER = os.path.join(HERE, 'references', 'tp4-profile2-jobs')
WORKFLOW = os.path.join(ROOT, '.github', 'workflows', 'qwen-c2-serving.yml')
CPU_WORKFLOW = os.path.join(ROOT, '.github', 'workflows', 'qwen-integration-cpu.yml')
with open(os.path.join(HERE, 'qwen_c2_profiles.json'), encoding='utf-8') as _handle:
    PROFILES = json.load(_handle)
NAMES = sorted(PROFILES['profiles'])
BANNED = re.compile(r'blackhole-[A-Za-z0-9]{8,}|thatch\.local|\d{1,3}(\.\d{1,3}){3}|sha256:[0-9a-f]{16}|[0-9a-f]{40,}|'
                    r'/dev/tenstorrent|home/|zot\.|@[A-Z0-9_]+@|[A-Za-z]:[\\/]')
IMAGE = 'tp4-prof-2'
BUILD, STOP, PROFILE, HANDBACK = 'PB0-build', 'PA0-agent-stop', 'P2-trace-profile', 'PH-handback-reset'
ORDERED = (BUILD, STOP, PROFILE, HANDBACK)
SERVED = 'c2-packed-tp4'
TIMED = 'c2-packed-tp4-speed-strace'
DEVICE_STEPS = ('cardm', 'smoke', 'gate', 'prefix', 'fabric', 'replay', 'probe', 'drift', 'platform', 'priority')


def read(path):
    with open(path, encoding='utf-8') as handle:
        return handle.read()


def read_order():
    return [line.split() for line in read(os.path.join(FOLDER, 'ORDER.txt')).splitlines()
            if line.strip() and not line.startswith('#')]


def text_of(name):
    return read(os.path.join(FOLDER, name + '.env'))


def parsed(name):
    return job.read_job(job.parse_env(text_of(name)), NAMES)


def env_of(name):
    return PROFILES['profiles'][name]['env']


class OrderTests(unittest.TestCase):
    def test_every_template_is_in_the_order_once_with_four_columns_and_the_order_names_no_other(self):
        rows = read_order()
        self.assertTrue(all(len(row) == 4 for row in rows))
        on_disk = sorted(name[:-4] for name in os.listdir(FOLDER) if name.endswith('.env'))
        self.assertEqual([row[0] for row in rows], list(ORDERED))
        self.assertEqual(sorted(row[0] for row in rows), on_disk)

    def test_the_modes_images_and_minutes(self):
        for name, mode, image, minutes in read_order():
            with self.subTest(job=name):
                self.assertEqual(mode, 'stop')
                self.assertEqual(image, IMAGE)
                self.assertEqual(parsed(name)['tag'], IMAGE)
                self.assertTrue(minutes.isdigit() and 5 <= int(minutes) <= 210, minutes)

    def test_the_order_carries_the_booking_the_hand_back_and_the_traps(self):
        order = read(os.path.join(FOLDER, 'ORDER.txt'))
        for word in ('PRODUCTION IS LIVE ON THE CARDS', "'agentstop unserve'", 'PB0 runs FIRST', 'FRESH tag', 'book 2 hours', 'HAND-BACK', 'fabric', 'agentstart',
                     'asks the owner for /deploy', 'No template', 'deploys anything', 'runs even when a stop job', 'NEVER CANCEL P2',
                     'root-owned', 'EACCES', 'DISK', '85%', 'NEVER delete the P8 base', 'qwen-fast-serving:ci-be9e184e',
                     'HOST MEMORY', 'LAUNCHED argv', 'THE READ-OUT', 'VALIDITY', 'At least 20', 'ops-trace texts == ops-twin',
                     '3%', 'trace pick', 'cpp_device_perf_report.csv.gz', 'v170', 'weight matmuls 17.0', 'gdn.recurrence 9.29',
                     'attn.sdpa 7.12', 'collectives 4.63', 'sampler 1.73', 'publication', 'drafters 24.75', 'commits 2.6',
                     'period 124.5', 'op-support'):
            self.assertIn(word, order)

    def test_the_stop_job_is_first_and_the_hand_back_is_last(self):
        rows = read_order()
        self.assertEqual((rows[0][0], rows[1][0], rows[-1][0]), (BUILD, STOP, HANDBACK), 'the build runs first, production still serving')


class TemplateTests(unittest.TestCase):
    def test_every_template_parses_is_lf_and_names_no_card_host_address_registry_digest_or_local_path(self):
        for name in ORDERED:
            with self.subTest(template=name):
                self.assertEqual(parsed(name)['cards'], 'quad')
                with open(os.path.join(FOLDER, name + '.env'), 'rb') as handle:
                    self.assertNotIn(b'\r', handle.read(), name)
                self.assertIsNone(BANNED.search(text_of(name)), name)
        with open(os.path.join(FOLDER, 'ORDER.txt'), 'rb') as handle:
            self.assertNotIn(b'\r', handle.read())
        self.assertIsNone(BANNED.search(read(os.path.join(FOLDER, 'ORDER.txt'))))

    def test_the_build_job_is_build_only_and_the_profile_job_no_longer_builds(self):
        outputs = parsed(BUILD)
        self.assertEqual((outputs['actions'], outputs['cards'], outputs['tag']), ('build', 'quad', IMAGE))
        for word in ('BEFORE PA0', 'FRESH', 'already exists', 'PLACEHOLDER', 'production still serves'):
            self.assertIn(word, text_of(BUILD))
        for name in ORDERED:
            self.assertEqual('build' in parsed(name)['actions'].split(), name == BUILD, name)

    def test_the_first_production_job_stops_the_agent_and_opens_no_board(self):
        outputs = parsed(STOP)
        self.assertEqual(outputs['actions'], 'status agentstop unserve')
        self.assertFalse(set(outputs['actions'].split()) & set(DEVICE_STEPS + ('reset', 'build', 'agentstart', 'push')))
        text = text_of(STOP)
        for word in ('PRODUCTION IS LIVE ON THE CARDS', 'agentstop'):
            self.assertIn(word, text)

    def test_only_the_first_job_stops_the_agent_and_none_starts_it_or_deploys(self):
        for name in ORDERED:
            actions = parsed(name)['actions'].split()
            with self.subTest(template=name):
                self.assertEqual('agentstop' in actions, name == STOP)
                self.assertEqual('agentstart' in actions, name == HANDBACK)
                self.assertNotIn('replay', actions)
                self.assertNotIn('push', actions)

    def test_the_profile_job_is_status_reset_gate_on_the_timed_production_profile(self):
        outputs = parsed(PROFILE)
        self.assertEqual(outputs['actions'], 'status reset gate')
        self.assertEqual((outputs['profile'], outputs['tag']), (TIMED, IMAGE))
        self.assertEqual(outputs['gate_plan'].replace(' ', ','), 'ops-twin,ops-trace')
        self.assertEqual(outputs['gate_jit'], 'record')
        text = text_of(PROFILE)
        for word in ('NEVER CANCEL', 'root-owned', 'qwen-fast-serving:ci-be9e184e', '85% disk', 'op-support 20000'):
            self.assertIn(word, text)
        actions = outputs['actions'].split()
        self.assertLess(actions.index('reset'), actions.index('gate'))

    def test_the_hand_back_is_status_reset_fabric_agentstart_and_the_owner_deploys(self):
        self.assertEqual(parsed(HANDBACK)['actions'], 'status reset fabric agentstart')
        text = text_of(HANDBACK)
        for word in ('fabric re-measure', 'agentstart', "OWNER's /deploy", 'never deploys anything', 'NEVER place a gate arm',
                     'runs even when a stop job', 'audits off'):
            self.assertIn(word, text)

    def test_the_templates_carry_no_job_key_the_plan_fixes_in_code(self):
        for name in ORDERED:
            for key in ('C2_GATE_LENGTHS', 'C2_GATE_MAX_TOKENS', 'C2_GATE_PAIRS', 'C2_GATE_MEMORY_PROMPT'):
                self.assertNotIn(key + '=', text_of(name))


class ProfileTests(unittest.TestCase):
    def test_the_profile_is_production_plus_the_gate_marker_and_nothing_else_in_the_flags_that_matter(self):
        served, timed = env_of(SERVED), env_of(TIMED)
        for flag in ('QWEN_FAST_TP', 'QWEN_FAST_VERIFY_T1_AUDIT', 'QWEN_FAST_VERIFY_T2_AUDIT',
                     'QWEN_FAST_PACKED_SAMPLER_IN_TRACE', 'QWEN_FAST_BUDGET_CAP', 'QWEN_FAST_QUAD_DRAFT',
                     'QWEN_FAST_TP_KV_SLIDE'):
            self.assertEqual(timed.get(flag), served.get(flag), flag)
        self.assertEqual((timed['QWEN_FAST_TP'], timed['QWEN_FAST_VERIFY_T1_AUDIT'], timed['QWEN_FAST_VERIFY_T2_AUDIT'],
                          timed['QWEN_FAST_PACKED_SAMPLER_IN_TRACE']), ('4', '0', '0', '1'))
        self.assertIs(PROFILES['profiles'][TIMED].get('gate_only'), True)
        self.assertNotIn('gate_only', PROFILES['profiles'][SERVED])
        self.assertEqual(PROFILES['profiles'][TIMED]['mesh_device'], 'P150x4')

    def test_no_checked_in_profile_carries_a_profiler_variable_and_the_plan_accepts_the_profile(self):
        ops.check_profiles(PROFILES)
        ops.check_timed_profile(PROFILES, TIMED)
        for plan in ops.PLANS:
            with self.subTest(plan=plan):
                arm, = ops.plan_arms(plan, TIMED, PROFILES, driver)
                args = list(arm[1])
                if plan in ops.PREFILL_PLANS:          # the prefill pair: one user, one 131,072-token prompt (test_ops_profile_plan holds it)
                    self.assertEqual(args[args.index('--prompt-lengths') + 1], '131072')
                    self.assertEqual(args[args.index('--users') + 1], '1')
                else:
                    self.assertEqual(args[args.index('--prompt-lengths') + 1], '4096,4096,4096,4096')
                    self.assertEqual(args[args.index('--users') + 1], '4')
                self.assertFalse(arm.judged)

    def test_the_gate_orders_the_ops_plans_last_and_the_twin_first(self):
        job.check_ops_order(['ops-twin', 'ops-trace'])
        with self.assertRaises(job.JobError):
            job.check_ops_order(['ops-trace', 'ops-twin'])


class WorkflowTests(unittest.TestCase):
    """The workflow's always() hand-back step beside the agentstop and agentstart steps: one order, no overlap."""

    @classmethod
    def setUpClass(cls):
        cls.text = read(WORKFLOW)

    def index(self, name):
        found = self.text.index('- name: %s' % name)
        return found

    def test_the_handback_step_runs_after_the_gate_and_before_the_upload_and_the_agent_restart(self):
        stop = self.index('Stop the node agent')
        hand_back = self.index('Hand back the TP4 op-profile output')
        start = self.index('Start the node agent')
        upload = self.text.index('actions/upload-artifact')
        self.assertLess(stop, hand_back)
        self.assertLess(hand_back, start)
        self.assertLess(hand_back, upload)
        self.assertLess(self.text.index('python3 scripts/ci/c2_serving_gate.py'), hand_back)

    def test_the_handback_step_is_always_and_only_for_ops_plans_and_the_agent_steps_keep_their_conditions(self):
        step = self.text[self.index('Hand back the TP4 op-profile output'):self.index('Prefix-reuse gates in the agent')]
        self.assertIn("if: always() && contains(steps.job.outputs.actions, 'gate') && contains(steps.job.outputs.gate_plan, 'ops-')",
                      step)
        self.assertIn('ops_profile_plan.py handback', step)
        self.assertIn('qwen-c2-gate-ops-', step)
        self.assertIn("if: contains(steps.job.outputs.actions, 'agentstop')", self.text)
        self.assertIn("if: ${{ !cancelled() && contains(steps.job.outputs.actions, 'agentstart') }}", self.text)

    def test_the_artifact_upload_takes_the_whole_results_tree_so_the_compressed_report_travels(self):
        upload = self.text[self.text.index('actions/upload-artifact'):]
        self.assertIn('path: ${{ runner.temp }}/c2-results', upload)
        self.assertEqual(ops.OPS_SUBDIR, 'ops')
        self.assertEqual(ops.CSV_GZ, 'cpp_device_perf_report.csv.gz')

    def test_the_new_test_modules_are_on_the_cpu_allowlist(self):
        cpu = read(CPU_WORKFLOW)
        for module in ('test_ops_profile_plan', 'test_tp4_profile_report', 'test_tp4_profile2_window'):
            self.assertRegex(cpu, r'\b%s\b' % module)


if __name__ == '__main__':
    unittest.main()
