"""The lanes window's gate (c2_lanes_plans and the harness's --user-lane), held on CPU: the arms each plan builds, that the harness
parses every one of them, the lane mark on the wire, the report the harness adds, and each plan's verdicts through the whole driver with
injected docker and server logs.

The lane lines in a fixture's server log are the ones the whole-stack fakes log (lanes_fakes.World runs the real hook, gate, controller
and packed step at the design's round times), so what the gate reads here is what a served arm's log carries. Nothing here opens a
device or runs docker."""

import io
import json
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from types import SimpleNamespace
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import c2_lanes_plans as plans  # noqa: E402
import c2_serving_gate as driver  # noqa: E402
import c2_serving_job as job  # noqa: E402
import lanes_report  # noqa: E402
import lever_n_m3native_gate as harness  # noqa: E402
import longctx_cycle_bench as bench  # noqa: E402
import test_c2_serving_gate as base  # noqa: E402
import test_c2_serving_gate_s2 as s2t  # noqa: E402
from lanes_fakes import World  # noqa: E402
import lanes_sim  # noqa: E402
from serving_fast_lane import LaneConfig  # noqa: E402
from test_fast_lane_hook import make, run  # noqa: E402

PROFILES = s2t.PROFILES
V235 = base.V235
EXACT_LENGTHS = list(plans.EXACT_LENGTHS)


def parse_harness(args):
    with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
        return harness.parse_options(list(args))


def lane_world(schedule, standard=3, seconds=40.0, lever='phase1', context='32k', tau=12.1, **options):
    """The lane lines of a run: the whole stack on a virtual clock at the design's round times, the ratio on `schedule`."""
    config = LaneConfig.from_environment({'QWEN_FAST_LANE_SCHEDULE': schedule}, seats=4)
    world = make(lever, context, standard, config=config, max_tokens=12000, tau_fast=tau, tau_standard=tau, **options)
    run(world, seconds=seconds)
    return world


WARM_LINE = 'INFO [PINDIAG] four-card eager prefill warmed before the packed traces: page_table_blocks=2052 slots=4 programs=117->503 ms=9000'
PREFILL_LINE = 'INFO [PINDIAG] four-card prefill programs=503->503 window=0 prompt=4096'


class Fixtures(object):
    """Server logs and arm reports the plans' arms would leave."""

    _worlds = {}

    @classmethod
    def lines(cls, schedule=plans.EXACT_SCHEDULE, standard=3, tau=12.1, context='32k', seconds=40.0):
        key = (schedule, standard, tau, context, seconds)
        if key not in cls._worlds:
            cls._worlds[key] = list(lane_world(schedule, standard, seconds, tau=tau, context=context).lines)
        return cls._worlds[key]

    @staticmethod
    def solo_route_lines():
        return ['INFO [SOLO-LANE] round=1 slot=0 request=cmpl-x position=4100 rows=16 block=solo']

    @classmethod
    def log(cls, users=4, lanes=None, solo=False, positions=None, rounds=12, **lane_options):
        """An arm's server log: the S2 arm's lines (s2_log) plus the lane lines and the solo-lane route line when asked."""
        text = s2t.s2_log(users=users, rounds=rounds, positions=positions or EXACT_LENGTHS[:users])
        # the four-card prefill tripwire (c2_serving_gate.prefill_tripwire_check, on every four-card S2 arm): the warm line before
        # the packed traces and a prefill that compiled nothing beyond its window snapshot
        text = '\n'.join((WARM_LINE, PREFILL_LINE, text)) if WARM_LINE not in text else text
        extra = []
        if solo:
            extra += cls.solo_route_lines()
        if lanes is not None:
            extra += ['INFO ' + line for line in (lanes if not isinstance(lanes, str) else cls.lines(lanes, **lane_options))]
        return text + '\n'.join(extra) + '\n'

    @staticmethod
    def streams(report, timelines=None):
        for index, stream in enumerate(report['streams']):
            if timelines and index < len(timelines):
                stream['chunk_s'], stream['chunk_tokens'] = timelines[index]
        return report

    @classmethod
    def report(cls, log_text, profile, env=(), lengths=None, users=4, texts=None, sequential=0, max_tokens=1024, fast=(),
               timelines=None, finish='stop', completion=300, user_max_tokens=None):
        report = s2t.s2_arm_report(log_text, profile, env=env, texts=texts, users=users, lengths=lengths,
                                   finish=finish, completion=completion, max_tokens=max_tokens, sequential=sequential,
                                   chunk=s2t.chunk_every(10))
        if user_max_tokens:
            for user, budget in user_max_tokens.items():
                report['streams'][user]['completion_tokens'] = budget
        cls.streams(report, timelines)
        options = SimpleNamespace(lanes={user: 'fast' for user in fast})
        harness.add_lanes_report(report, options, log_text, report['streams'])
        report['gate_passed'] = not report['flag_markers']['missing']
        return report


def texts_of(users=4):
    return [V235['streams'][index % len(V235['streams'])]['text'] for index in range(users)]


class ArmTests(unittest.TestCase):
    def arms(self, plan, lengths=None):
        return driver.plan_arms(plan, 'c2-packed-tp4-gate', PROFILES, lengths, 4096, None, [])

    def test_every_arm_of_every_plan_is_parsed_by_the_harness_it_runs_on(self):
        for plan in plans.PLANS:
            for spec in self.arms(plan):
                with self.subTest(plan=plan, arm=spec[0]):
                    options = parse_harness(spec[1])
                    self.assertEqual(options.prompt_source, 'real-text')
                    self.assertEqual(options.expect_profile, spec.profile)
                    self.assertGreater(spec[2], 0)

    def test_the_plans_are_in_the_jobs_vocabulary_and_are_not_the_s2_plans(self):
        self.assertEqual(job.LANES_GATE_PLANS, plans.PLANS)
        self.assertFalse(set(plans.PLANS) & set(job.S2_GATE_PLANS + job.GATE_PLANS))
        self.assertEqual(set(job.ALL_GATE_PLANS), set(job.GATE_PLANS + job.S2_GATE_PLANS + plans.PLANS))
        self.assertIn('lanes_report.py', driver.BENCH_SCRIPTS)

    def test_lanes_exact_is_the_reference_d0_and_the_lanes_on_the_same_prompts(self):
        ref, d0, lanes = self.arms('lanes-exact')
        self.assertEqual([spec[0] for spec in (ref, d0, lanes)], ['ref-solo', 'd0-solo', 'lanes-mixed'])
        self.assertEqual([spec.profile for spec in (ref, d0, lanes)],
                         ['c2-packed-tp4-gate', 'c2-packed-tp4-solo-gate', 'c2-packed-tp4-lanes-gate'])
        prompts = [spec[1][spec[1].index('--prompt-lengths') + 1] for spec in (ref, d0, lanes)]
        self.assertEqual(prompts, ['4096,16384,32768,60000'] * 3)
        for spec in (ref, d0):
            self.assertEqual(spec[1][spec[1].index('--users'):][:4], ['--users', '1', '--sequential-users', '4'])
            self.assertNotIn('--user-lane', spec[1])
        self.assertEqual(lanes[1][lanes[1].index('--user-lane') + 1], '0:fast')
        self.assertIn(('QWEN_FAST_LANE_SCHEDULE', plans.EXACT_SCHEDULE), lanes.env)
        for spec in (ref, d0, lanes):
            self.assertIn((harness.EXTENT_AUDIT_FLAG, '1'), spec.env, 'every arm of the exactness plan is audited')
        self.assertEqual([spec.role for spec in (ref, d0, lanes)], ['solo', 'd0', 'lanes'])

    def test_the_exactness_schedule_reaches_every_ratio_the_window_needs(self):
        from serving_fast_lane import parse_schedule
        entries = parse_schedule(plans.EXACT_SCHEDULE, 3)
        self.assertEqual([k for k, _ in entries], [2.0, 1.0, 0.5, 0.0, None])
        self.assertEqual([frames for _, frames in entries][:4], [25] * 4)
        self.assertGreaterEqual(25, lanes_report.MIN_STRETCH_ROUNDS, 'a stretch is read from 24 packed rounds')
        self.assertEqual(len(parse_schedule(plans.SWEEP_SCHEDULE, 3)), 6)

    def test_round_timing_is_the_ladders_and_the_lone_arms(self):
        specs = {spec[0]: spec for spec in self.arms('round-timing')}
        self.assertEqual(list(specs), ['padded-4k', 'padded-32k', 'base-lone-4k', 'd0-lone-4k', 'd0-lone-32k'])
        ladder = specs['padded-4k']
        args = ladder[1]
        self.assertEqual(args[args.index('--user-max-tokens') + 1], plans.LADDER_BUDGETS)
        self.assertEqual(args[args.index('--user-ignore-eos') + 1], '0,1,2,3')
        self.assertEqual(int(args[args.index('--max-tokens') + 1]), plans.LADDER_MAX_TOKENS)
        budgets = dict(part.split(':') for part in plans.LADDER_BUDGETS.split(','))
        self.assertLess(int(budgets['3']), int(budgets['2']))
        self.assertLess(int(budgets['2']), int(budgets['1']))
        self.assertLess(int(budgets['1']), plans.LADDER_MAX_TOKENS, 'each user ends in turn: live 4, 3, 2, 1')
        self.assertEqual(specs['d0-lone-4k'].profile, 'c2-packed-tp4-solo-time-gate')
        self.assertEqual(specs['base-lone-4k'].profile, 'c2-packed-tp4-time-gate')
        for name in ('d0-lone-4k', 'base-lone-4k', 'd0-lone-32k'):
            self.assertEqual(specs[name][1][specs[name][1].index('--users'):][:4], ['--users', '1', '--sequential-users', '4'])
        for spec in specs.values():
            self.assertEqual(spec.env, (), 'a timing arm adds no audit')

    def test_lanes_timing_is_one_fast_user_beside_three_two_and_one_and_the_more_plan_the_rest(self):
        names = lambda plan: [spec[0] for spec in self.arms(plan)]
        self.assertEqual(names('lanes-timing'), ['lanes-n3-4k', 'lanes-n2-4k', 'lanes-n1-4k', 'lanes-n3-32k'])
        self.assertEqual(names('lanes-timing-more'), ['lanes-n2-32k', 'lanes-n1-32k', 'lanes-n3-120k'])
        for plan in ('lanes-timing', 'lanes-timing-more'):
            for spec in self.arms(plan):
                args = spec[1]
                users = int(args[args.index('--users') + 1])
                self.assertEqual(int(spec[0].split('-n')[1][0]), users - 1, spec[0])
                self.assertEqual(args[args.index('--user-lane') + 1], '0:fast')
                self.assertEqual(spec.profile, 'c2-packed-tp4-lanes-time-gate')
                self.assertEqual(spec.env, ((plans.SCHEDULE_ENV, plans.SWEEP_SCHEDULE),))
                self.assertNotIn('--user-ignore-eos', args, 'real text to its natural end: tau is what the bars hang on')

    def test_every_arm_env_is_one_an_arm_may_add(self):
        for plan in plans.PLANS:
            for spec in self.arms(plan):
                for name, _ in spec.env:
                    self.assertIn(name, driver.ARM_ENV_NAMES)

    def test_a_profile_edited_away_from_its_plan_refuses_it_before_any_container(self):
        edited = json.loads(json.dumps(PROFILES))
        del edited['profiles']['c2-packed-tp4-lanes-gate']['env']['QWEN_FAST_LANE']
        with self.assertRaisesRegex(driver.PlanError, 'QWEN_FAST_LANE is off'):
            driver.plan_arms('lanes-exact', 'c2-packed-tp4-gate', edited, None, 4096, None, [])
        edited = json.loads(json.dumps(PROFILES))
        edited['profiles']['c2-packed-tp4-time-gate']['env']['QWEN_FAST_VERIFY_T1_AUDIT'] = '1'
        edited['profiles']['c2-packed-tp4-time-gate']['env']['QWEN_FAST_VERIFY_T2_AUDIT'] = '1'
        with self.assertRaisesRegex(driver.PlanError, 'verify audits are on'):
            driver.plan_arms('round-timing', 'c2-packed-tp4-gate', edited, None, 4096, None, [])
        edited = json.loads(json.dumps(PROFILES))
        del edited['profiles']['c2-packed-tp4-solo-gate']
        with self.assertRaisesRegex(driver.PlanError, 'do not carry'):
            driver.plan_arms('lanes-exact', 'c2-packed-tp4-gate', edited, None, 4096, None, [])
        edited = json.loads(json.dumps(PROFILES))
        edited['profiles']['c2-packed-tp4-gate']['env']['QWEN_FAST_SOLO_LANE'] = '1'
        with self.assertRaisesRegex(driver.PlanError, 'QWEN_FAST_SOLO_LANE is on'):
            driver.plan_arms('lanes-exact', 'c2-packed-tp4-gate', edited, None, 4096, None, [])

    def test_lengths_replace_the_exactness_prompts_within_two_to_four_users(self):
        notes = []
        arms = driver.plan_arms('lanes-exact', 'c2-packed-tp4-gate', PROFILES, [4096, 8192], 4096, None, notes)
        self.assertTrue(notes)
        self.assertEqual(arms[0][1][arms[0][1].index('--sequential-users') + 1], '2')
        with self.assertRaisesRegex(driver.PlanError, 'two to four users'):
            driver.plan_arms('lanes-exact', 'c2-packed-tp4-gate', PROFILES, [4096], 4096, None, [])

    def test_the_worst_cases_fit_a_hosted_step_that_the_s3a_matrix_fits(self):
        # the S2 window's matrix arms are 2 x 2 x (5400 + 120) s = 22,080 s = 368 min; a step allows 380
        for plan in plans.PLANS:
            arms = {plan: self.arms(plan)}
            self.assertLessEqual(driver.worst_case_seconds([plan], arms), 380 * 60, plan)

    def test_dry_run_prints_the_docker_argv_of_each_arm_with_the_lane_knob_and_the_gate_switch(self):
        lines = []
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, 'profiles.json')
            with open(path, 'w', encoding='utf-8') as handle:
                json.dump(PROFILES, handle)
            code = driver.main(['--profile', 'c2-packed-tp4-gate', '--plan', 'lanes-exact,lanes-timing', '--cards', 'quad',
                                '--image', 'img:x', '--results', os.path.join(directory, 'r'), '--profiles', path, '--dry-run',
                                '--jit', 'record'], log=lines.append)
        self.assertEqual(code, 0)
        arms = [json.loads(line) for line in lines if line.startswith('{') and '"arm"' in line]
        by_name = {entry['arm']: entry['docker'] for entry in arms}
        self.assertEqual(sorted(by_name), sorted(['ref-solo', 'd0-solo', 'lanes-mixed', 'lanes-n3-4k', 'lanes-n2-4k', 'lanes-n1-4k',
                                                  'lanes-n3-32k']))
        mixed = by_name['lanes-mixed']
        self.assertIn('QWEN_C2_PROFILE=c2-packed-tp4-lanes-gate', mixed)
        self.assertIn('QWEN_FAST_LANE_SCHEDULE=%s' % plans.EXACT_SCHEDULE, mixed)
        self.assertIn('QWEN_C2_GATE=1', mixed)
        self.assertIn('lanes_report.py', ' '.join(mixed))
        self.assertIn('--user-lane', mixed)
        self.assertNotIn('--user-lane', by_name['ref-solo'])
        self.assertIn('QWEN_C2_PROFILE=c2-packed-tp4-solo-gate', by_name['d0-solo'])
        self.assertIn('QWEN_C2_PROFILE=c2-packed-tp4-lanes-time-gate', by_name['lanes-n3-4k'])


class ClaimTests(unittest.TestCase):
    def test_a_lanes_divergence_fails_it_is_never_not_comparable(self):
        import real_text_compare
        from serving_fast_lane import FLAG_NAMES
        for name in FLAG_NAMES + ('QWEN_FAST_SOLO_LANE',):
            self.assertIn(name, real_text_compare.S2_EXACT_CLAIMS)
        self.assertEqual(sorted(real_text_compare.LANES_EXACT_CLAIMS), sorted(FLAG_NAMES + ('QWEN_FAST_SOLO_LANE',)))
        first = dict(qwen_configuration={'QWEN_FAST_LANE': '1', 'QWEN_FAST_SOLO_LANE': '1', 'QWEN_FAST_LANE_SCHEDULE': 'auto'})
        second = dict(qwen_configuration={})
        self.assertEqual(real_text_compare.arithmetic_diff(first, second), {})


class HarnessTests(unittest.TestCase):
    def test_the_lane_option_parses_and_names_one_fast_stream_at_most(self):
        self.assertEqual(harness.user_lanes('0:fast,2:standard', 4), {0: 'fast', 2: 'standard'})
        self.assertEqual(harness.user_lanes(None, 4), {})
        for bad, message in (('4:fast', 'names no stream'), ('x:fast', 'not USER:LANE'), ('0:quick', 'not one of'),
                             ('0:fast,0:standard', 'twice'), ('0:fast,1:fast', 'at most one stream may be fast')):
            with self.subTest(text=bad):
                with self.assertRaisesRegex(ValueError, message):
                    harness.user_lanes(bad, 4)

    def test_a_bad_lane_option_is_a_parser_error_and_a_good_one_reaches_the_streams_keywords(self):
        args = ['--users', '2', '--prompt-source', 'real-text', '--allow-missing-references', '--eos', 'stop', '--user-lane', '0:fast']
        options = parse_harness(args)
        self.assertEqual(options.lanes, {0: 'fast'})
        self.assertEqual(harness.user_stream(options, 0, {})[1], {'lane': 'fast'})
        self.assertEqual(harness.user_stream(options, 1, {})[1], {})
        with self.assertRaises(SystemExit):
            parse_harness(args[:-1] + ['0:fast,1:fast'])

    def test_without_the_option_the_streams_keywords_are_what_they_were(self):
        options = parse_harness(['--users', '2'])
        self.assertEqual(options.lanes, {})
        self.assertEqual(harness.user_stream(options, 0, {'a': 1}), (options.max_tokens, {'a': 1}))

    def test_the_lane_mark_rides_vllm_xargs_and_no_other_payload_changes(self):
        bodies = []

        class Response(object):
            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            def __iter__(self):
                return iter(())

        def fake(request, timeout=None):
            bodies.append(json.loads(request.data.decode()))
            return Response()

        with mock.patch.object(bench, 'urlopen', fake):
            plain, marked = [None], [None]
            bench.stream_once(1, [1, 2], 4, plain, 0)
            bench.stream_once(1, [1, 2], 4, marked, 0, lane='fast')
        self.assertNotIn('vllm_xargs', bodies[0])
        self.assertEqual(bodies[1]['vllm_xargs'], {'qwen_lane': 'fast'})
        self.assertEqual({key: value for key, value in bodies[1].items() if key != 'vllm_xargs'}, bodies[0])

    def test_an_arm_that_marked_a_stream_fast_gets_a_lanes_record_and_its_problems_reach_gate_passed(self):
        report = dict(streams=[{}, {}], flag_markers=dict(found={}, missing=[]))
        options = SimpleNamespace(lanes={0: 'fast'})
        harness.add_lanes_report(report, options, 'nothing of the lanes here', report['streams'])
        self.assertIn('lanes', report)
        self.assertTrue(report['flag_markers']['missing'])
        self.assertTrue(any('never came up' in problem for problem in report['flag_markers']['missing']))

    def test_an_arm_with_no_lane_mark_and_no_lane_lines_keeps_exactly_its_keys(self):
        report = dict(streams=[{}], flag_markers=dict(found={}, missing=[]))
        harness.add_lanes_report(report, SimpleNamespace(lanes={}), 'INFO [PACKED] request=x', report['streams'])
        self.assertEqual(sorted(report), ['flag_markers', 'streams'])

    def test_an_arm_that_ran_the_runtime_with_no_mark_is_still_read(self):
        report = dict(streams=[{}], flag_markers=dict(found={}, missing=[]))
        harness.add_lanes_report(report, SimpleNamespace(lanes={}), '[LANE] engaged seats=4 ratio=auto', report['streams'])
        self.assertIn('lanes', report)


class Runs(object):
    """Drive the whole gate on one lanes plan with injected docker, per-arm reports and server logs."""

    @staticmethod
    def go(test, plan, reports, logs, profile='c2-packed-tp4-gate', argv=()):
        every = {}
        every_logs = {}
        for name, factory in reports.items():
            every[name] = factory
            every[name + '-rerun'] = factory
        for name, text in logs.items():
            every_logs[name] = text
            every_logs[name + '-rerun'] = text
        return s2t.Driver(test).run(['--profile', profile, '--plan', plan, '--cards', 'quad', '--jit', 'record'] + list(argv),
                                    every, every_logs, profiles=PROFILES)


class ExactTests(unittest.TestCase):
    ENV = ((harness.EXTENT_AUDIT_FLAG, '1'),)

    def arms(self, texts=None, lanes_log=None, d0_log=None, ref_log=None, lanes_texts=None, fast=(0,), d0_texts=None):
        texts = texts or texts_of()
        lanes_env = self.ENV + (('QWEN_FAST_LANE_SCHEDULE', plans.EXACT_SCHEDULE),)
        ref_log = ref_log or Fixtures.log()
        d0_log = d0_log or Fixtures.log(solo=True)
        lanes_log = lanes_log or Fixtures.log(lanes=plans.EXACT_SCHEDULE)
        reports = {
            'ref-solo': lambda n: Fixtures.report(ref_log, 'c2-packed-tp4-gate', self.ENV, EXACT_LENGTHS, texts=texts, sequential=4),
            'd0-solo': lambda n: Fixtures.report(d0_log, 'c2-packed-tp4-solo-gate', self.ENV, EXACT_LENGTHS,
                                                 texts=d0_texts or texts, sequential=4),
            'lanes-mixed': lambda n: Fixtures.report(lanes_log, 'c2-packed-tp4-lanes-gate', lanes_env, EXACT_LENGTHS,
                                                     texts=lanes_texts or texts, fast=fast)}
        logs = {'ref-solo': ref_log, 'd0-solo': d0_log, 'lanes-mixed': lanes_log}
        return reports, logs

    def run_exact(self, **kwargs):
        reports, logs = self.arms(**kwargs)
        return Runs.go(self, 'lanes-exact', reports, logs)

    def test_every_lane_and_d0_equal_to_the_per_request_engines_passes_and_prints_the_sweep(self):
        code, summary, calls, lines, _ = self.run_exact()
        result = summary['results']['lanes-exact']
        self.assertEqual((code, result['verdict']), (0, 'PASS'), result.get('lines'))
        self.assertEqual([call['arm'] for call in calls], ['ref-solo', 'd0-solo', 'lanes-mixed'])
        self.assertEqual(result['facts']['ratios_read'], ['0', '0.5', '1', '2', 'auto'])
        self.assertGreater(result['facts']['lanes_rounds']['solo'], 100)
        self.assertTrue(any(line.startswith('lanes BEST') for line in result['lines']))

    def test_a_divergent_lane_is_rerun_with_its_reference_once_then_fails(self):
        texts = texts_of()
        bad = list(texts)
        bad[1] = bad[1][:120] + '#' + bad[1][121:]
        code, summary, calls, lines, records = self.run_exact(lanes_texts=bad)
        result = summary['results']['lanes-exact']
        self.assertEqual((code, result['verdict']), (1, 'FAIL'), result.get('lines'))
        self.assertEqual([call['arm'] for call in calls],
                         ['ref-solo', 'd0-solo', 'lanes-mixed', 'ref-solo-rerun', 'lanes-mixed-rerun'])
        self.assertEqual(result['arms']['d0']['verdict'], 'PASS')

    def test_a_divergent_d0_and_a_divergent_lane_rerun_the_reference_only_once(self):
        bad = texts_of()
        bad[2] = bad[2][:90] + '#' + bad[2][91:]
        code, summary, calls, _, _ = self.run_exact(lanes_texts=bad, d0_texts=bad)
        self.assertEqual([call['arm'] for call in calls].count('ref-solo-rerun'), 1)
        self.assertEqual(summary['results']['lanes-exact']['verdict'], 'FAIL')

    def test_lane_lines_in_the_reference_arms_log_mean_it_did_not_run_its_own_path(self):
        leaked = Fixtures.log(lanes=Fixtures.lines()[:6])
        code, summary, _, _, _ = self.run_exact(ref_log=leaked)
        result = summary['results']['lanes-exact']
        self.assertEqual(result['verdict'], 'FAIL')
        self.assertTrue(any('ref-solo' in problem and 'leaves QWEN_FAST_LANE off' in problem for problem in result['s2_problems']))

    def test_solo_lane_lines_in_the_reference_arms_log_fail_it(self):
        code, summary, _, _, _ = self.run_exact(ref_log=Fixtures.log(solo=True))
        self.assertTrue(any('QWEN_FAST_SOLO_LANE off' in problem for problem in summary['results']['lanes-exact']['s2_problems']))

    def test_d0_that_never_served_a_lone_round_proves_nothing_and_is_not_exercised(self):
        code, summary, _, _, _ = self.run_exact(d0_log=Fixtures.log(solo=False))
        result = summary['results']['lanes-exact']
        self.assertEqual(result['verdict'], 'NOT_EXERCISED')
        self.assertTrue(any('one-user block never served a lone round' in text for text in result['shortfalls']))

    def test_lanes_that_barely_alternated_are_not_exercised(self):
        quiet = Fixtures.log(lanes='0@200,auto', seconds=6.0)
        code, summary, _, _, _ = self.run_exact(lanes_log=quiet)
        result = summary['results']['lanes-exact']
        self.assertEqual(result['verdict'], 'NOT_EXERCISED', result['lines'])
        self.assertTrue(any('barely alternated' in text or 'never held' in text for text in result['shortfalls']))

    def test_a_fast_mark_that_never_reached_the_worker_fails_by_name(self):
        lines = [line.replace('asked=fast', 'asked=standard').replace('granted=fast', 'granted=standard')
                 for line in Fixtures.lines()]
        code, summary, _, _, _ = self.run_exact(lanes_log=Fixtures.log(lanes=lines))
        result = summary['results']['lanes-exact']
        self.assertEqual(result['verdict'], 'FAIL')
        self.assertTrue(any('never reached the worker' in problem for problem in result['s2_problems']), result['s2_problems'])

    def test_a_desync_line_fails_the_plan(self):
        lines = list(Fixtures.lines()) + ['[LANE-DESYNC] the lane gate planned [u1] and the step scheduled [u1, u2]: the scheduler '
                                          'did not hide [u2]']
        code, summary, _, _, _ = self.run_exact(lanes_log=Fixtures.log(lanes=lines))
        self.assertEqual(summary['results']['lanes-exact']['verdict'], 'FAIL')

    def test_a_schedule_the_worker_never_read_fails_by_name(self):
        lines = [line.replace('ratio=schedule:', 'ratio=auto ignored:') for line in Fixtures.lines()]
        code, summary, _, _, _ = self.run_exact(lanes_log=Fixtures.log(lanes=lines))
        self.assertTrue(any('never reached the worker' in problem and 'SCHEDULE' in problem
                            for problem in summary['results']['lanes-exact']['s2_problems']),
                        summary['results']['lanes-exact']['s2_problems'])


class TimingTests(unittest.TestCase):
    def lanes_arm(self, standard, tau=12.1, context='32k', label=None, schedule=plans.SWEEP_SCHEDULE, seconds=60.0, log=None,
                  lengths=None):
        name = 'lanes-n%d-%s' % (standard, label or ('4k' if context == '4k' else '32k'))
        lengths = lengths or [4096 if context == '4k' else 32768] * (1 + standard)
        text = log or Fixtures.log(users=1 + standard, lanes=Fixtures.lines(schedule, standard, tau, context, seconds),
                                   positions=lengths)
        env = ((plans.SCHEDULE_ENV, schedule),)
        return name, text, (lambda n: Fixtures.report(text, 'c2-packed-tp4-lanes-time-gate', env, lengths, users=1 + standard,
                                                      fast=(0,), max_tokens=plans.TIMING_MAX_TOKENS))

    def run_lanes(self, standards=(3, 2, 1, 3), **kwargs):
        reports, logs = {}, {}
        contexts = ['4k', '4k', '4k', '32k']
        for standard, context in zip(standards, contexts):
            name, text, factory = self.lanes_arm(standard, context=context, **kwargs)
            reports[name], logs[name] = factory, text
        return Runs.go(self, 'lanes-timing', reports, logs)

    def test_every_arm_read_passes_and_prints_each_stretch_and_the_best_cell_per_users(self):
        code, summary, calls, lines, _ = self.run_lanes()
        result = summary['results']['lanes-timing']
        self.assertEqual((code, result['verdict']), (0, 'PASS'), result.get('lines'))
        self.assertEqual([call['arm'] for call in calls], ['lanes-n3-4k', 'lanes-n2-4k', 'lanes-n1-4k', 'lanes-n3-32k'])
        target = [line for line in result['lines'] if ' TARGET n=' in line]
        self.assertEqual(len(target), 4)
        self.assertTrue(all(line.endswith(': MET') for line in target), target)
        self.assertEqual(sorted(result['facts']), ['lanes-n1-4k', 'lanes-n2-4k', 'lanes-n3-32k', 'lanes-n3-4k'])
        for facts in result['facts'].values():
            self.assertEqual([entry['label'] for entry in facts['best']], ['both'])

    def test_a_missed_bar_is_a_result_not_a_failure(self):
        code, summary, _, _, _ = self.run_lanes(tau=4.9)
        result = summary['results']['lanes-timing']
        self.assertEqual((code, result['verdict']), (0, 'PASS'), result.get('lines'))
        self.assertTrue(any(line.endswith(': MISSED') or line.endswith(': MISSED (fast)') for line in result['lines']
                            if ' TARGET n=' in line), [line for line in result['lines'] if 'TARGET' in line])

    def test_a_stretch_too_short_to_read_is_not_exercised_and_named(self):
        code, summary, _, _, _ = self.run_lanes(seconds=4.0)
        result = summary['results']['lanes-timing']
        self.assertEqual(result['verdict'], 'NOT_EXERCISED', result['lines'])
        self.assertTrue(any('unread' in text or 'no stretch was read' in text for text in result['shortfalls']))


class RoundTests(unittest.TestCase):
    @staticmethod
    def ladder_report(text, medians=(('4p', 40, 100.0, 8.0), ('3p', 40, 95.0, 8.0), ('2p', 40, 80.0, 8.0), ('1s', 60, 250.0, 3.0)),
                      length=4096):
        report = Fixtures.report(text, 'c2-packed-tp4-time-gate', (), [length] * 4, max_tokens=plans.LADDER_MAX_TOKENS,
                                 finish='length', completion=plans.LADDER_MAX_TOKENS)
        budgets = dict(part.split(':') for part in plans.LADDER_BUDGETS.split(','))
        for user, budget in budgets.items():
            report['streams'][int(user)]['completion_tokens'] = int(budget)
        groups = [dict(live=int(key[0]), packed=key[1] == 'p', rounds=rounds, median_round_ms=median,
                       tokens_per_user_per_round=tau) for key, rounds, median, tau in medians]
        report['acceptance'] = dict(rounds_by_live=dict(groups=groups))
        return report

    @staticmethod
    def lone_report(text, profile, rate, tau, lengths, users=4, completion=700):
        report = Fixtures.report(text, profile, (), lengths, users=users, max_tokens=plans.LONE_MAX_TOKENS, sequential=users,
                                 finish='stop', completion=completion)
        for stream in report['streams']:
            step = tau / rate
            seconds = [0.0, 2.0] + [2.0 + step * (index + 1) for index in range(60)]
            counts = [1, 1 + int(tau)] + [1 + int(tau) * (index + 2) for index in range(60)]
            stream['chunk_s'], stream['chunk_tokens'] = seconds, counts
        return report

    def go(self, d0_rate=120.0, base_rate=30.0, ladder=None):
        lengths4, lengths32 = [4096] * 4, [32768] * 4
        plain = Fixtures.log(users=4, positions=lengths4)
        solo = Fixtures.log(users=4, positions=lengths4, solo=True)
        solo32 = Fixtures.log(users=4, positions=lengths32, solo=True)
        plain32 = Fixtures.log(users=4, positions=lengths32)
        reports = {
            'padded-4k': lambda n: self.ladder_report(plain, **({} if ladder is None else dict(medians=ladder))),
            'padded-32k': lambda n: self.ladder_report(plain32, length=32768, **({} if ladder is None else dict(medians=ladder))),
            'base-lone-4k': lambda n: self.lone_report(plain, 'c2-packed-tp4-time-gate', base_rate, 3.0, lengths4),
            'd0-lone-4k': lambda n: self.lone_report(solo, 'c2-packed-tp4-solo-time-gate', d0_rate, 8.0, lengths4),
            'd0-lone-32k': lambda n: self.lone_report(solo32, 'c2-packed-tp4-solo-time-gate', d0_rate * 0.9, 8.0, lengths32)}
        logs = {'padded-4k': plain, 'padded-32k': plain32, 'base-lone-4k': plain, 'd0-lone-4k': solo, 'd0-lone-32k': solo32}
        return Runs.go(self, 'round-timing', reports, logs)

    def test_the_ladders_medians_and_the_lone_rates_are_printed_and_d0_is_set_against_the_engines(self):
        code, summary, calls, _, _ = self.go()
        result = summary['results']['round-timing']
        self.assertEqual((code, result['verdict']), (0, 'PASS'), result.get('lines'))
        text = '\n'.join(result['lines'])
        self.assertIn('padded-4k: 4p rounds 40 median 100.0 ms', text)
        self.assertIn('padded-32k: 1s rounds 60 median 250.0 ms', text)
        self.assertRegex(text, r'D0 vs per-request engines at 4096: 120\.\d+ vs 30\.\d+ tok/s \(x4\.0\d\)')
        self.assertEqual(result['facts']['d0-lone-4k']['mean_tau'], 8.0)
        self.assertEqual(result['facts']['padded-4k']['ladder']['4p']['median_round_ms'], 100.0)

    def test_a_ladder_phase_with_too_few_rounds_is_not_exercised(self):
        code, summary, _, _, _ = self.go(ladder=(('4p', 40, 100.0, 8.0), ('3p', 10, 95.0, 8.0), ('2p', 40, 80.0, 8.0)))
        result = summary['results']['round-timing']
        self.assertEqual(result['verdict'], 'NOT_EXERCISED')
        self.assertTrue(any('3p' in text and 'P(n) unread' in text for text in result['shortfalls']))

    def test_a_d0_arm_with_no_solo_route_line_is_not_exercised(self):
        lengths4 = [4096] * 4
        plain = Fixtures.log(users=4, positions=lengths4)
        reports = {name: (lambda n, name=name: self.ladder_report(plain)) for name in ('padded-4k', 'padded-32k')}
        reports['base-lone-4k'] = lambda n: self.lone_report(plain, 'c2-packed-tp4-time-gate', 30.0, 3.0, lengths4)
        reports['d0-lone-4k'] = lambda n: self.lone_report(plain, 'c2-packed-tp4-solo-time-gate', 100.0, 8.0, lengths4)
        reports['d0-lone-32k'] = lambda n: self.lone_report(plain, 'c2-packed-tp4-solo-time-gate', 100.0, 8.0, [4096] * 4)
        logs = {name: plain for name in reports}
        code, summary, _, _, _ = Runs.go(self, 'round-timing', reports, logs)
        result = summary['results']['round-timing']
        self.assertEqual(result['verdict'], 'FAIL' if result.get('s2_problems') else 'NOT_EXERCISED', result['lines'])
        self.assertTrue(any('D0 never served a lone round' in text for text in result['shortfalls']))


if __name__ == '__main__':
    unittest.main()
