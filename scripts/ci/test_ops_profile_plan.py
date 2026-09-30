"""ops_profile_plan (the TP4 op-profile gate plans) and its wiring into c2_serving_job, c2_serving_gate and the workflow."""
import copy
import gzip
import json
import os
import re
import sys
import tempfile
import unittest

import c2_serving_gate as driver
import c2_serving_job as job
import ops_profile_plan as ops
import test_tp4_attach_profile as attach
import test_tp4_profile_report as synthetic
from test_c2_serving_gate import CHECKOUT_PROFILES, FakeDocker, V235

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
WORKFLOW = os.path.join(ROOT, '.github', 'workflows', 'qwen-c2-serving.yml')
TEMPLATE_DIR = os.path.join(HERE, 'references', 'tp4-profile-jobs')
PROFILE = 'c2-packed-tp4-speed'
CARDS = ['/dev/tenstorrent/%d' % n for n in range(4)]
CONFIGURATION = {'QWEN_FAST_TP': '4', 'QWEN_FAST_VERIFY_T1_AUDIT': '0', 'QWEN_FAST_VERIFY_T2_AUDIT': '0',
                 'QWEN_FAST_PROFILE_DUMP_EVERY': '2', 'TT_METAL_PROFILER_DIR': ops.PROFILE_DIR,
                 'TT_METAL_DEVICE_PROFILER': '1', 'TT_METAL_PROFILER_TRACE_TRACKING': '1',
                 'TT_METAL_PROFILER_MID_RUN_DUMP': '1', 'TT_METAL_PROFILER_CPP_POST_PROCESS': '1',
                 'TT_METAL_CACHE': ops.SCRATCH_CACHE, 'QWEN_FAST_PROFILED_BLOCK_STREAM': '1'}
TEXTS = ['alpha', 'beta', 'gamma', 'delta']


def report(texts=TEXTS, configuration=None, **extra):
    data = dict(gate_passed=True, users=4, qwen_configuration=configuration or dict(CONFIGURATION),
                streams=[dict(text=text, completion_tokens=64, finish_reason='length') for text in texts],
                platform=dict(served_profile=PROFILE, served_argv=[], problems=[]), real_text_stream_problems=[],
                flag_markers=dict(missing=[]), dram=dict(events=[], engines=0))
    data.update(extra)
    return data


class Arms(object):
    """driver.PlanError and the driver's own helpers, as ops_profile_plan asks for them."""


class PlanTests(unittest.TestCase):
    def arms(self, plan, profile=PROFILE, profiles=None):
        return ops.plan_arms(plan, profile, profiles or CHECKOUT_PROFILES, driver)

    def test_the_plans_are_the_jobs(self):
        self.assertEqual(ops.PLANS, job.OPS_GATE_PLANS)
        self.assertEqual(ops.PLANS, ('ops-twin', 'ops-trace'))
        self.assertIn('ops-twin', job.ALL_GATE_PLANS)

    def test_the_shape_is_fixed_in_code(self):
        for plan in ops.PLANS:
            arm, = self.arms(plan)
            args = list(arm[1])
            self.assertEqual(args[args.index('--prompt-lengths') + 1], '4096,8192,16384,24576')
            self.assertEqual(args[args.index('--users') + 1], '4')
            self.assertEqual(args[args.index('--max-tokens') + 1], '64')
            self.assertEqual(args[args.index('--user-max-tokens') + 1], '0:192')
            self.assertEqual(args[args.index('--user-ignore-eos') + 1], '0,1,2,3')
            self.assertIn('--prompt-source', args)
            self.assertEqual(args[args.index('--prompt-source') + 1], 'real-text')
            self.assertFalse(arm.judged)
            self.assertFalse(driver.arm_runs(plan, arm) - 1)          # never re-run

    def test_the_twin_adds_nothing_and_the_trace_arm_adds_the_profiler(self):
        twin, = self.arms('ops-twin')
        trace, = self.arms('ops-trace')
        self.assertEqual(twin.extra['ops']['env'], ())
        self.assertIsNone(twin.extra['ops']['tracy'])
        env = dict(trace.extra['ops']['env'])
        self.assertEqual(env['QWEN_FAST_PROFILE_DUMP_EVERY'], '2')
        for name in ('TT_METAL_DEVICE_PROFILER', 'TT_METAL_PROFILER_TRACE_TRACKING', 'TT_METAL_PROFILER_MID_RUN_DUMP',
                     'TT_METAL_PROFILER_CPP_POST_PROCESS', 'TTNN_OP_PROFILER'):
            self.assertEqual(env[name], '1')
        self.assertEqual(env['TT_METAL_PROFILER_DIR'], ops.PROFILE_DIR)
        self.assertNotIn('QWEN_PREFILL_PROFILE_FLUSH', env)          # never run at op-support 20000

    def test_the_attach_test_carries_the_same_profiler_environment(self):
        self.assertEqual(attach.OPS_PROFILER_ENV, ops.PROFILER_ENV)

    def test_the_tracy_recipe_is_v138s(self):
        self.assertEqual(ops.tracy_args(), ['-p', '--check-exit-code', '--disable-device-data-dump-to-files',
                                            '--disable-device-data-push-to-tracy', '--dump-device-data-mid-run',
                                            '--op-support-count', '20000', '-o', ops.PROFILE_DIR])

    def test_op_support_above_20000_is_refused(self):
        for count in ('200000', '20001', 'many', '0'):
            args = ops.tracy_args()
            args[args.index('--op-support-count') + 1] = count
            with self.assertRaises(ops.OpsPlanError, msg=count):
                ops.check_tracy(args)
        ops.check_tracy(['--disable-device-data-dump-to-files', '--op-support-count', '10000', '-o', ops.PROFILE_DIR])

    def test_tracy_without_the_dump_suppression_or_another_output_is_refused(self):
        with self.assertRaises(ops.OpsPlanError):
            ops.check_tracy(['--op-support-count', '20000', '-o', ops.PROFILE_DIR])
        with self.assertRaises(ops.OpsPlanError):
            ops.check_tracy(['--disable-device-data-dump-to-files', '--op-support-count', '20000', '-o', '/tmp/x'])
        with self.assertRaises(ops.OpsPlanError):
            ops.check_tracy(ops.tracy_args() + ['--enable-sum-profiling'])

    def test_an_environment_name_outside_the_list_is_refused(self):
        with self.assertRaises(ops.OpsPlanError):
            ops.check_env((('QWEN_FAST_EXTENT_AUDIT', '1'),))
        with self.assertRaises(ops.OpsPlanError):
            ops.check_env((('QWEN_FAST_PROFILE_DUMP_EVERY', 'two'),))
        with self.assertRaises(ops.OpsPlanError):
            ops.check_env((('QWEN_FAST_PROFILE_DUMP_EVERY', '0'),))
        self.assertEqual(ops.check_env(ops.PROFILER_ENV), ops.PROFILER_ENV)

    def test_only_the_timed_four_card_profile_is_served(self):
        with self.assertRaisesRegex(driver.PlanError, 'QWEN_FAST_VERIFY_T1_AUDIT'):
            self.arms('ops-trace', profile='c2-packed-tp4-gate')          # audits on: not the timed trace
        with self.assertRaisesRegex(driver.PlanError, 'QWEN_FAST_TP'):
            self.arms('ops-trace', profile='c2-packed')
        with self.assertRaisesRegex(driver.PlanError, 'not in the image'):
            self.arms('ops-trace', profile='nope')

    def test_a_served_profile_that_carries_a_profiler_variable_is_refused(self):
        profiles = copy.deepcopy(CHECKOUT_PROFILES)
        profiles['profiles']['general']['env']['TT_METAL_DEVICE_PROFILER'] = '1'
        with self.assertRaisesRegex(driver.PlanError, 'never runs the profiler'):
            self.arms('ops-trace', profiles=profiles)

    def test_no_checked_in_profile_carries_a_profiler_variable(self):
        ops.check_profiles(CHECKOUT_PROFILES)

    def test_the_lengths_must_fit_the_profile(self):
        profiles = copy.deepcopy(CHECKOUT_PROFILES)
        profiles['profiles'][PROFILE]['max_prompt_tokens'] = 20000
        with self.assertRaisesRegex(driver.PlanError, 'exceed'):
            self.arms('ops-twin', profiles=profiles)

    def test_the_arm_limits_cover_readiness_stream_and_close(self):
        self.assertEqual(ops.ARM_SECONDS['ops-twin'], 1800 + 1800 + 300)
        self.assertEqual(ops.ARM_SECONDS['ops-trace'], 1800 + 3600 + 900)


class DockerShapeTests(unittest.TestCase):
    def run_argv(self, plan, arm_dir='/r/ops'):
        arm, = ops.plan_arms(plan, PROFILE, CHECKOUT_PROFILES, driver)
        prepared = ops.planned(arm_dir, arm.extra['ops'])
        return driver.gate_run('img', 'qwen-c2-gate-%s' % plan, PROFILE, CARDS, '/checkout', arm_dir, list(arm[1]),
                               ops=prepared, gate_only=True), arm

    def test_the_twin_argv_is_the_unprofiled_one(self):
        argv, arm = self.run_argv('ops-twin')
        plain = driver.gate_run('img', 'qwen-c2-gate-ops-twin', PROFILE, CARDS, '/checkout', '/r/ops', list(arm[1]),
                                gate_only=True)
        self.assertEqual(argv, plain)
        self.assertNotIn('tracy', argv)
        self.assertIn('-B', argv[argv.index('python3'):])
        self.assertFalse([a for a in argv if a.startswith('TT_METAL_DEVICE_PROFILER')])

    def test_the_trace_argv_wraps_the_harness_in_tracy_and_mounts_the_profile_dir(self):
        argv, arm = self.run_argv('ops-trace', arm_dir='/r/ops-trace')
        entry = argv.index('python3')
        self.assertEqual(argv[entry + 1:entry + 5], ['img', '-B', '-m', 'tracy'])
        wrapped = argv[entry + 2:]
        self.assertEqual(wrapped[wrapped.index('--op-support-count') + 1], '20000')
        self.assertLess(wrapped.index('-o'), wrapped.index('/bench/lever_n_m3native_gate.py'))
        self.assertEqual(argv[argv.index('-o', entry) + 1], ops.PROFILE_DIR)
        self.assertIn('type=bind,src=%s,dst=%s' % (os.path.join('/r/ops-trace', ops.PROFILE_SUBDIR), ops.PROFILE_DIR), argv)
        for name, value in ops.PROFILER_ENV:
            self.assertIn('%s=%s' % (name, value), argv)
        # the results mount and the profile mount are two mounts
        self.assertEqual(len([a for a in argv if a.startswith('type=bind') and 'dst=%s' % driver.RESULTS_IN_CONTAINER in a]), 1)

    def test_agent_shape_refuses_an_unlisted_ops_variable(self):
        with self.assertRaises(driver.PlanError):
            driver.agent_shape('img', 'n', PROFILE, CARDS, ops_env=(('QWEN_FAST_EXTENT_AUDIT', '1'),))


class JobTests(unittest.TestCase):
    PROFILES = CHECKOUT_PROFILES['profiles']

    def read(self, **values):
        base = {'C2_ACTIONS': 'status reset build gate', 'C2_IMAGE_TAG': 'tp4-prof-1', 'C2_CARDS': 'quad',
                'C2_PROFILE': PROFILE, 'C2_GATE_PLAN': 'ops-twin,ops-trace', 'C2_GATE_JIT': 'record'}
        base.update(values)
        return job.read_job(base, self.PROFILES)

    def test_the_two_plans_in_order_are_accepted(self):
        self.assertEqual(self.read()['gate_plan'], 'ops-twin,ops-trace')
        self.assertEqual(self.read(C2_GATE_PLAN='ops-twin')['gate_plan'], 'ops-twin')

    def test_the_trace_needs_the_twin_first(self):
        for plan in ('ops-trace', 'ops-trace,ops-twin'):
            with self.assertRaisesRegex(job.JobError, 'ops-twin before it'):
                self.read(C2_GATE_PLAN=plan)

    def test_an_ops_plan_goes_after_every_judged_plan_and_only_once(self):
        with self.assertRaisesRegex(job.JobError, 'after every judged plan'):
            self.read(C2_GATE_PLAN='ops-twin,matrix')
        with self.assertRaisesRegex(job.JobError, 'twice'):
            self.read(C2_GATE_PLAN='ops-twin,ops-twin')
        self.assertEqual(self.read(C2_GATE_PLAN='matrix,ops-twin,ops-trace')['gate_plan'], 'matrix,ops-twin,ops-trace')

    def test_a_pair_profile_cannot_ride_a_quad_job(self):
        with self.assertRaises(job.JobError):
            self.read(C2_PROFILE='c2-packed')

    def test_the_committed_template_is_a_valid_job(self):
        values = {}
        path = os.path.join(TEMPLATE_DIR, 'P1-trace-profile.env')
        with open(path, encoding='utf-8') as handle:
            for line in handle:
                line = line.strip()
                if line and not line.startswith('#'):
                    key, _, value = line.partition('=')
                    values[key] = value
        outputs = job.read_job(values, self.PROFILES)
        self.assertEqual((outputs['cards'], outputs['profile'], outputs['gate_plan'], outputs['actions']),
                         ('quad', PROFILE, 'ops-twin,ops-trace', 'status reset build gate'))
        self.assertEqual(outputs['gate_jit'], 'record')
        self.assertEqual(values['C2_IMAGE_TAG'], 'tp4-prof-1')


class TemplateFolderTests(unittest.TestCase):
    BANNED = re.compile(r'blackhole-[A-Za-z0-9]{8,}|thatch\.local|\d{1,3}(\.\d{1,3}){3}|sha256:[0-9a-f]{16}|[0-9a-f]{40,}|'
                        r'/dev/tenstorrent|home/|zot\.')

    def order(self):
        with open(os.path.join(TEMPLATE_DIR, 'ORDER.txt'), encoding='utf-8') as handle:
            return [tuple(line.split()) for line in handle.read().splitlines() if line.strip() and not line.startswith('#')]

    def test_every_template_is_in_the_order_once(self):
        on_disk = sorted(name[:-4] for name in os.listdir(TEMPLATE_DIR) if name.endswith('.env'))
        self.assertEqual(sorted(name for name, _ in self.order()), on_disk)
        self.assertEqual(self.order(), [('P1-trace-profile', 'stop')])

    def test_the_templates_are_public_safe(self):
        for name in os.listdir(TEMPLATE_DIR):
            with open(os.path.join(TEMPLATE_DIR, name), encoding='utf-8') as handle:
                self.assertIsNone(self.BANNED.search(handle.read()), name)

    def test_the_template_builds_resets_first_and_never_names_a_tag_to_push(self):
        with open(os.path.join(TEMPLATE_DIR, 'P1-trace-profile.env'), encoding='utf-8') as handle:
            text = handle.read()
        self.assertNotIn('experiment/c2-serving-v', text.replace('a tag experiment/c2-serving-vN', ''))
        actions = job.parse_env(text)['C2_ACTIONS'].split()
        self.assertEqual(actions[:2], ['status', 'reset'])
        self.assertLess(actions.index('build'), actions.index('gate'))


class WorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with open(WORKFLOW, encoding='utf-8') as handle:
            cls.text = handle.read()

    def step(self, name):
        start = self.text.index('      - name: %s' % name)
        end = self.text.find('\n      - name: ', start + 10)
        return self.text[start:end]

    def test_the_handback_step_runs_however_the_gate_ended_and_before_the_upload(self):
        step = self.step('Hand back the TP4 op-profile output')
        self.assertIn("if: always() && contains(steps.job.outputs.actions, 'gate') && contains(steps.job.outputs.gate_plan, 'ops-')",
                      step)
        self.assertIn("grep '^qwen-c2-gate-ops-'", step)
        self.assertIn('scripts/ci/ops_profile_plan.py handback --results "$RUNNER_TEMP/c2-results/gate"', step)
        self.assertLess(self.text.index('Hand back the TP4 op-profile output'), self.text.index('name: Upload results'))
        self.assertGreater(self.text.index('Hand back the TP4 op-profile output'),
                           self.text.index('Run the gate in the agent\'s container shape'))

    def test_the_gate_step_removes_every_gate_container_including_the_ops_ones(self):
        self.assertIn("grep '^qwen-c2-gate-' | xargs -r docker rm -f", self.text)
        self.assertTrue((driver.CONTAINER_PREFIX + 'ops-trace').startswith('qwen-c2-gate-ops-'))


class VerdictTests(unittest.TestCase):
    def arm(self, **ops_fields):
        return dict(exit=0, ops=dict(kind='ops', **ops_fields), ops_env=[list(pair) for pair in ops.PROFILER_ENV])

    def log(self, readbacks=25, audit=False):
        text = ops.READBACK_MARKER + ' 2 (QWEN_FAST_PROFILE_DUMP_EVERY=2)\n'
        return text * readbacks + ('[PINDIAG] verify t2 audit ok\n' if audit else '')

    def test_a_twin_that_served_four_texts_passes(self):
        result = ops.verdict('ops-twin', report(), dict(exit=0, ops=dict(kind='twin')), None, log_text=self.log(0))
        self.assertEqual(result['verdict'], 'PASS')
        self.assertEqual(len(result['texts']), 4)

    def test_the_trace_arm_passes_with_the_twin_texts(self):
        result = ops.verdict('ops-trace', report(), self.arm(), report(), log_text=self.log())
        self.assertEqual((result['verdict'], result['reason']), ('PASS', None))

    def test_texts_that_differ_from_the_twin_fail_it(self):
        result = ops.verdict('ops-trace', report(), self.arm(), report(texts=['alpha', 'beta', 'gamma', 'other']),
                             log_text=self.log())
        self.assertEqual(result['verdict'], 'FAIL')
        self.assertIn('profiling changed the arithmetic', result['reason'])

    def test_no_twin_is_not_exercised(self):
        result = ops.verdict('ops-trace', report(), self.arm(), None, log_text=self.log())
        self.assertEqual(result['verdict'], 'NOT_EXERCISED')

    def test_an_audit_line_or_an_audited_configuration_fails_it(self):
        result = ops.verdict('ops-trace', report(), self.arm(), report(), log_text=self.log(audit=True))
        self.assertEqual(result['verdict'], 'FAIL')
        config = dict(CONFIGURATION, QWEN_FAST_VERIFY_T1_AUDIT='1')
        result = ops.verdict('ops-twin', report(configuration=config), dict(exit=0, ops=dict(kind='twin')), None,
                             log_text=self.log(0))
        self.assertEqual(result['verdict'], 'FAIL')
        self.assertIn('not the timed four-card profile', result['reason'])

    def test_a_profiler_variable_that_never_arrived_fails_it(self):
        config = dict(CONFIGURATION)
        del config['QWEN_FAST_PROFILE_DUMP_EVERY']
        result = ops.verdict('ops-trace', report(configuration=config), self.arm(), report(), log_text=self.log())
        self.assertEqual(result['verdict'], 'FAIL')
        self.assertIn('QWEN_FAST_PROFILE_DUMP_EVERY=2 never reached the container', result['reason'])

    def test_too_few_readbacks_is_a_note_and_the_plan_still_passes(self):
        result = ops.verdict('ops-trace', report(), self.arm(), report(), log_text=self.log(3))
        self.assertEqual(result['verdict'], 'PASS')
        self.assertTrue(any('read-back lines' in line for line in result['lines']))

    def test_a_short_report_is_soft(self):
        result = ops.verdict('ops-trace', report(), self.arm(validity=dict(problems=['x'], ok=False)), report(),
                             log_text=self.log())
        self.assertEqual(result['verdict'], 'PASS')
        self.assertIn('note: validity: x', result['lines'])

    def test_the_disk_guard_and_a_missing_report(self):
        self.assertEqual(ops.verdict('ops-trace', report(), self.arm(disk_guard='full'), report())['verdict'],
                         'NOT_EXERCISED')
        self.assertEqual(ops.verdict('ops-trace', None, self.arm(), report())['verdict'], 'FAIL')

    def test_a_stream_without_text_fails(self):
        broken = report()
        broken['streams'][2] = dict(finish_reason='length')
        self.assertEqual(ops.verdict('ops-twin', broken, dict(exit=0, ops=dict(kind='twin')), None,
                                     log_text=self.log(0))['verdict'], 'FAIL')


class AroundTheArmTests(unittest.TestCase):
    def test_the_disk_guard_stops_the_container_past_its_cap_or_the_disk_limit(self):
        stopped = []
        big = ops.DiskGuard('/p', 100, lambda: stopped.append(1), usage=lambda path: (100, 10, 90), size=lambda path: 101)
        self.assertIn('reached', big.check())
        full = ops.DiskGuard('/p', 100, lambda: None, usage=lambda path: (100, 90, 10), size=lambda path: 1)
        self.assertIn('the disk reached 90.0% used', full.check())
        fine = ops.DiskGuard('/p', 100, lambda: None, usage=lambda path: (100, 10, 90), size=lambda path: 1)
        self.assertIsNone(fine.check())
        running = ops.DiskGuard('/p', 100, lambda: stopped.append(1), usage=lambda path: (100, 10, 90),
                                size=lambda path: 101, interval=0.01, log=lambda text: None).start()
        deadline = 0
        while not running.tripped and deadline < 200:
            import time
            time.sleep(0.01)
            deadline += 1
        self.assertIn('reached', running.finish())
        self.assertEqual(stopped, [1])

    def test_the_disk_refusal(self):
        self.assertIsNone(ops.disk_problem('/r', 4 * ops.GB, usage=lambda path: (1000 * ops.GB, 100 * ops.GB, 0)))
        self.assertIn('past 80%', ops.disk_problem('/r', 4 * ops.GB, usage=lambda path: (100 * ops.GB, 78 * ops.GB, 0)))

    def test_the_handback_keeps_the_report_compressed_and_prunes_the_rest(self):
        with tempfile.TemporaryDirectory() as results:
            logs = os.path.join(results, 'ops-trace', ops.PROFILE_SUBDIR, '.logs')
            os.makedirs(logs)
            with open(os.path.join(logs, ops.CSV_NAME), 'w') as handle:
                handle.write('a,b\n' + '1,2\n' * (2 * 1024 * 1024))            # above KEEP_BYTES
            with open(os.path.join(logs, 'small.log'), 'w') as handle:
                handle.write('x')
            handed, lines = [], []
            done = ops.handback_results(results, 'img', handback=lambda image, path: handed.append(path), log=lines.append)
            self.assertEqual(handed, [os.path.join(results, 'ops-trace', ops.PROFILE_SUBDIR)])
            self.assertEqual([entry['path'] for entry in done['ops-trace']], [os.path.join('.logs', ops.CSV_NAME)])
            self.assertFalse(os.path.exists(os.path.join(logs, ops.CSV_NAME)))
            self.assertTrue(os.path.exists(os.path.join(logs, 'small.log')))
            with gzip.open(os.path.join(results, ops.OPS_SUBDIR, ops.CSV_GZ), 'rt') as handle:
                self.assertTrue(handle.read(8).startswith('a,b'))

    def test_a_handback_that_raises_is_reported_and_the_prune_still_runs(self):
        def broken(image, path):
            raise RuntimeError('docker down')
        with tempfile.TemporaryDirectory() as results:
            os.makedirs(os.path.join(results, 'ops-trace', ops.PROFILE_SUBDIR))
            lines = []
            done = ops.handback_results(results, 'img', handback=broken, log=lines.append)
            self.assertIn('ops-trace', done)
            self.assertTrue(any('hand-back failed: docker down' in line for line in lines))

    def test_handback_arm_never_raises_and_skips_the_twin(self):
        with tempfile.TemporaryDirectory() as arm_dir:
            os.makedirs(os.path.join(arm_dir, ops.PROFILE_SUBDIR))
            self.assertIsNone(ops.handback_arm('img', arm_dir, dict(kind='twin'), lambda i, p: 1 / 0))
            self.assertTrue(ops.handback_arm('img', arm_dir, dict(kind='ops'), lambda i, p: 1 / 0).startswith('failed'))

    def test_prepare_arm_makes_the_profile_dir_writable_for_the_containers_root(self):
        with tempfile.TemporaryDirectory() as directory:
            arm = os.path.join(directory, 'ops-trace')
            os.makedirs(arm)
            prepared = ops.prepare_arm(arm, ops.plan_arms('ops-trace', PROFILE, CHECKOUT_PROFILES, driver)[0].extra['ops'])
            self.assertTrue(os.path.isdir(os.path.join(arm, ops.PROFILE_SUBDIR)))
            self.assertEqual(len(prepared['mounts']), 2)
            twin = os.path.join(directory, 'ops-twin')
            os.makedirs(twin)
            ops.prepare_arm(twin, ops.plan_arms('ops-twin', PROFILE, CHECKOUT_PROFILES, driver)[0].extra['ops'])
            self.assertFalse(os.path.exists(os.path.join(twin, ops.PROFILE_SUBDIR)))

    def test_finish_arm_compresses_analyses_and_prunes(self):
        class Analyse(object):
            @staticmethod
            def analyse_files(csv_path, **kwargs):
                Analyse.seen = (csv_path, kwargs)
                return dict(validity=dict(ok=True, problems=[]))

            @staticmethod
            def render_markdown(data):
                return '# report\nline\n'

        with tempfile.TemporaryDirectory() as results:
            arm = os.path.join(results, 'ops-trace')
            logs = os.path.join(arm, ops.PROFILE_SUBDIR, '.logs')
            os.makedirs(logs)
            with open(os.path.join(logs, ops.CSV_NAME), 'w') as handle:
                handle.write('a,b\n1,2\n')
            summary = ops.finish_arm(arm, results, twin_arm_dir=os.path.join(results, 'ops-twin'), log=lambda t: None,
                                     analyse=Analyse)
            self.assertEqual(summary['report'], 'ops/tp4-profile-report.json')
            self.assertEqual(summary['validity']['ok'], True)
            out = os.path.join(results, ops.OPS_SUBDIR)
            self.assertEqual(sorted(os.listdir(out)), sorted([ops.CSV_GZ, ops.REPORT_JSON, ops.REPORT_MD, ops.PRUNED]))
            csv_path, kwargs = Analyse.seen
            self.assertEqual(csv_path, os.path.join(out, ops.CSV_GZ))
            self.assertEqual(kwargs['server_log'], os.path.join(arm, 'server.log'))
            self.assertEqual(kwargs['twin_json'], os.path.join(results, 'ops-twin', 'm3native-gate.json'))

    def test_finish_arm_says_so_when_tracy_wrote_no_report(self):
        with tempfile.TemporaryDirectory() as results:
            arm = os.path.join(results, 'ops-trace')
            os.makedirs(os.path.join(arm, ops.PROFILE_SUBDIR))
            summary = ops.finish_arm(arm, results, log=lambda t: None)
            self.assertIn('no cpp_device_perf_report.csv', summary['problem'])

    def test_an_analysis_that_raises_is_a_problem_not_a_crash(self):
        class Broken(object):
            @staticmethod
            def analyse_files(csv_path, **kwargs):
                raise ValueError('bad csv')

        with tempfile.TemporaryDirectory() as results:
            arm = os.path.join(results, 'ops-trace')
            logs = os.path.join(arm, ops.PROFILE_SUBDIR, '.logs')
            os.makedirs(logs)
            with open(os.path.join(logs, ops.CSV_NAME), 'w') as handle:
                handle.write('a,b\n')
            summary = ops.finish_arm(arm, results, log=lambda t: None, analyse=Broken)
            self.assertIn('analysis failed', summary['problem'])


class EndToEndTests(unittest.TestCase):
    """The gate driver over both plans with a fake docker, as the job runs them."""

    def drive(self, twin_texts=TEXTS, trace_texts=TEXTS, readbacks=25, write_csv=True, extra_args=()):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, 'profiles.json')
            with open(path, 'w', encoding='utf-8') as handle:
                json.dump(CHECKOUT_PROFILES, handle)
            results = os.path.join(directory, 'results')
            reports = {'ops-twin': lambda n: report(texts=twin_texts), 'ops-trace': lambda n: report(texts=trace_texts)}
            log_text = 'round\n' + (ops.READBACK_MARKER + ' 2 (QWEN_FAST_PROFILE_DUMP_EVERY=2)\n') * readbacks
            fake = FakeDocker(reports, {'ops-twin': 'x\n', 'ops-trace': log_text})

            def execute(arguments, stdout_path, timeout, name):
                code = fake(arguments, stdout_path, timeout, name)
                arm_dir = os.path.dirname(stdout_path)
                if name.endswith('ops-trace') and write_csv:
                    logs = os.path.join(arm_dir, ops.PROFILE_SUBDIR, '.logs')
                    os.makedirs(logs, exist_ok=True)
                    synthetic.build(sessions=('1', '2', '3')).write(logs, ops.CSV_NAME)
                return code

            lines = []
            code = driver.main(['--image', 'zot/img:c2', '--results', results, '--profiles', path, '--profile', PROFILE,
                                '--plan', 'ops-twin,ops-trace', '--cards', 'quad', '--jit', 'record'] + list(extra_args),
                               execute=execute, devices=CARDS, log=lines.append, containers=lambda: [],
                               corpus=lambda: dict(V235['real_text']['corpus']))
            with open(os.path.join(results, 'c2-gate-summary.json'), encoding='utf-8') as handle:
                summary = json.load(handle)
            listing = sorted(os.listdir(results))
            ops_dir = sorted(os.listdir(os.path.join(results, 'ops'))) if os.path.isdir(os.path.join(results, 'ops')) else []
            trace_dir = sorted(os.listdir(os.path.join(results, 'ops-trace')))
            with open(os.path.join(results, 'ops-trace', 'docker-run.json'), encoding='utf-8') as handle:
                argv = json.load(handle)
            with open(os.path.join(results, 'ops-twin', 'docker-run.json'), encoding='utf-8') as handle:
                twin_argv = json.load(handle)
        return code, summary, lines, listing, ops_dir, trace_dir, fake.calls, argv, twin_argv

    def test_both_plans_pass_and_the_artifact_holds_the_report(self):
        code, summary, lines, listing, ops_dir, trace_dir, calls, argv, twin_argv = self.drive()
        self.assertEqual(code, 0, lines)
        self.assertEqual([c['arm'] for c in calls], ['ops-twin', 'ops-trace'])
        self.assertEqual(summary['results']['ops-twin']['verdict'], 'PASS')
        self.assertEqual(summary['results']['ops-trace']['verdict'], 'PASS')
        self.assertEqual(sorted(ops_dir), sorted([ops.CSV_GZ, ops.REPORT_JSON, ops.REPORT_MD, ops.PRUNED]))
        self.assertIn('tracy', argv)
        self.assertNotIn('tracy', twin_argv)
        self.assertEqual(summary['arms']['ops-trace']['ops']['kind'], 'ops')
        self.assertEqual(summary['arms']['ops-twin']['ops'], dict(kind='twin', disk_guard=None, handback=None))
        self.assertTrue(summary['arms']['ops-trace']['ops']['validity']['ok'])
        self.assertLess(summary['worst_case_seconds'], 3 * 3600)

    def test_texts_that_differ_fail_the_profiled_plan_and_the_gate_exit(self):
        code, summary, lines, *_ = self.drive(trace_texts=['alpha', 'beta', 'gamma', 'x'])
        self.assertEqual(code, 1)
        self.assertEqual(summary['results']['ops-trace']['verdict'], 'FAIL')
        self.assertEqual(summary['results']['ops-twin']['verdict'], 'PASS')

    def test_a_run_with_no_report_from_tracy_still_passes_and_says_so(self):
        code, summary, lines, listing, ops_dir, *_ = self.drive(write_csv=False)
        self.assertEqual(code, 0, lines)
        self.assertIn('no cpp_device_perf_report.csv', summary['arms']['ops-trace']['ops']['problem'])

    def test_a_pair_profile_is_refused_before_any_container_starts(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, 'profiles.json')
            with open(path, 'w', encoding='utf-8') as handle:
                json.dump(CHECKOUT_PROFILES, handle)
            lines = []
            code = driver.main(['--image', 'zot/img:c2', '--results', os.path.join(directory, 'r'), '--profiles', path,
                                '--profile', 'c2-packed', '--plan', 'ops-twin'], log=lines.append)
            self.assertEqual(code, 2)
            self.assertIn('QWEN_FAST_TP', lines[0])

    def test_dry_run_prints_the_tracy_argv_without_touching_the_disk(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, 'profiles.json')
            with open(path, 'w', encoding='utf-8') as handle:
                json.dump(CHECKOUT_PROFILES, handle)
            lines = []
            results = os.path.join(directory, 'r')
            code = driver.main(['--image', 'zot/img:c2', '--results', results, '--profiles', path, '--profile', PROFILE,
                                '--plan', 'ops-twin,ops-trace', '--cards', 'quad', '--dry-run'], log=lines.append)
            self.assertEqual(code, 0, lines)
            arms = [json.loads(line) for line in lines[1:]]
            self.assertEqual([a['arm'] for a in arms], ['ops-twin', 'ops-trace'])
            self.assertIn('tracy', arms[1]['docker'])
            self.assertNotIn('tracy', arms[0]['docker'])
            self.assertFalse(os.path.exists(os.path.join(results, 'ops-trace', ops.PROFILE_SUBDIR)))


if __name__ == '__main__':
    unittest.main()
