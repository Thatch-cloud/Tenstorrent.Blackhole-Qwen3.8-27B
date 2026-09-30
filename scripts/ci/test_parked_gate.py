"""Stage E, E8: the parked plans of c2_serving_gate, the job keys, the harness's sequential --start-order, and the
prefix gate's parked reading - plans built against the checkout's real profiles, judged on synthetic arm reports."""
import copy
import io
import json
import os
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import c2_serving_gate as gate  # noqa: E402
import c2_serving_job as job  # noqa: E402
import parked_judge  # noqa: E402
import parked_judge as judge  # noqa: E402
import parked_markers as markers  # noqa: E402
import serving_parked_engines as parked  # noqa: E402
from test_parked_judge import encoded  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
PROFILES = os.path.join(HERE, 'qwen_c2_profiles.json')
with open(PROFILES, encoding='utf-8') as handle:
    PROFILE_SET = json.load(handle)
PARKED, TWIN = 'c2-packed-prefix-parked-gate', 'c2-packed-prefix-gate'
PLANS = job.PARKED_GATE_PLANS
S2 = dict(ballast_mb=1500)


def arms_of(plan, profile=PARKED, **s2):
    return gate.plan_arms(plan, profile, PROFILE_SET, s2=dict(S2, **s2))


def by_name(arms):
    return dict((arm[0], arm) for arm in arms)


def attach_log(k=4, warm=2048, prewarm=7, rebinds=(), unparked=(), extra=()):
    lines = ['[PINDIAG] parked engines built k=%d of 4 attach_ms=1.0 trace_used=1 free=2 largest_free=3' % k,
             '[PINDIAG] parked drafter warm P=%d ms=1.0 window=256 prewarm_pairs=%d' % (warm, prewarm),
             '[PINDIAG] publish prewarm pairs=16:1-3 count=%d ms=1.0 program_cache=1->2' % prewarm]
    for index, request in enumerate(rebinds):
        lines.append('[PINDIAG] parked rebind req=%s slot=0 gen=%d P=300 budget=16 ms=1.0 single_rebuilt=0 window=256'
                     % (request, index + 2))
    lines += ['[PINDIAG] parked slot %d unparked: %s' % item for item in unparked]
    return '\n'.join(lines + list(extra))


def synthetic(spec, text_of=None, paths=None, mean=3.5, parked_record=None, extra=None):
    """A report an arm of `spec` could have written: asked()'s streams, budgets and lengths, one text per user."""
    want = gate.asked(spec[1])
    count = want['streams']
    texts = [(text_of or (lambda user: 'text %d' % user))(user) for user in range(count)]
    report = dict(
        streams=[dict(text=text, finish_reason='length', completion_tokens=want['budgets'][user])
                 for user, text in enumerate(texts)],
        real_text=dict(users=[dict(user=i, prompt_sha256='%064x' % (i + 1), prompt_tokens=want['lengths'][i])
                              for i in range(count)], prompt_lengths=want['lengths']),
        comparisons=[dict(user=i, prompt_sha256='%064x' % (i + 1), max_tokens=want['budgets'][i]) for i in range(count)],
        max_tokens=want['max_tokens'], users=1, qwen_configuration=dict(QWEN_FAST_SDPA_PF='1'), alive=True,
        s2=dict(paths=dict(users=dict((str(i), dict(encoded=encoded(*(paths or {}).get(i, [('S', 10, 3, 3), ('P', 13, 4, 4)]))))
                                      for i in range(count)))),
        acceptance=dict(users=[dict(user=i, full_draft=dict(rounds=40, mean_emitted=mean)) for i in range(count)]))
    if parked_record is not None:
        report['c2_gate_parked'] = parked_record
    report.update(extra or {})
    return report


class Feed:
    """run_arm stand-in: each arm's report from a table, in the order the plan asks."""

    def __init__(self, table):
        self.table, self.asked = table, []

    def __call__(self, runner, plan, spec, suffix=''):
        self.asked.append(spec[0] + suffix)
        value = self.table[spec[0]]
        return value(spec) if callable(value) else value


class PlanTests(unittest.TestCase):
    def test_every_parked_plan_builds_on_the_parked_profile_and_fits_a_step(self):
        for plan in PLANS:
            with self.subTest(plan=plan):
                arms = arms_of(plan)
                self.assertTrue(arms)
                self.assertLess(gate.worst_case_seconds([plan], {plan: arms}), 380 * 60 - 600)
                for arm in arms:
                    self.assertIn(arm.profile, (PARKED, TWIN))
                    self.assertEqual(arm.extra['parked']['on'], arm.profile == PARKED, arm[0])
                    for name, _ in arm.env:
                        self.assertIn(name, gate.ARM_ENV_NAMES)

    def test_the_parked_plans_need_a_parked_profile_with_a_flag_off_twin(self):
        for profile in ('c2-packed-prefix', 'c2-packed', 'c2'):
            with self.subTest(profile=profile), self.assertRaisesRegex(gate.PlanError, 'parked-engines plan'):
                arms_of('parked-exact', profile)
        broken = copy.deepcopy(PROFILE_SET)
        del broken['profiles'][TWIN]
        with self.assertRaisesRegex(gate.PlanError, 'no flag-off twin'):
            gate.plan_arms('parked-exact', PARKED, broken)

    def test_the_ballast_plan_needs_its_size(self):
        with self.assertRaisesRegex(gate.PlanError, 'needs --ballast-mb'):
            gate.plan_arms('parked-ballast', PARKED, PROFILE_SET, s2={})
        arms = by_name(arms_of('parked-ballast', ballast_mb=1500))
        self.assertEqual(dict(arms['ballast-arrival'].env)['QWEN_FAST_GATE_DRAM_BALLAST'], str(1500 * 2 ** 20))

    def test_exact_chain_arms_send_the_same_prompts_in_three_orders(self):
        arms = by_name(arms_of('parked-exact'))
        desc, asc, shuf = (arms[name][1] for name in ('exact-p-desc', 'exact-p-asc', 'exact-p-shuf'))
        for flag in ('--prompt-lengths', '--user-max-tokens', '--user-ignore-eos', '--sequential-users'):
            self.assertEqual(desc[desc.index(flag) + 1], asc[asc.index(flag) + 1], flag)
            self.assertEqual(desc[desc.index(flag) + 1], shuf[shuf.index(flag) + 1], flag)
        self.assertNotIn('--start-order', desc)
        self.assertEqual(asc[asc.index('--start-order') + 1], ','.join(map(str, reversed(range(13)))))
        self.assertEqual(arms['exact-f-desc'].profile, TWIN)
        self.assertEqual(arms['exact-f-desc'][1][arms['exact-f-desc'][1].index('--expect-profile') + 1], TWIN)
        self.assertEqual(arms['exact-first-a'][1][arms['exact-first-a'][1].index('--expect-profile') + 1], PARKED)
        self.assertEqual(sorted(name for name in arms if name.startswith('exact-first')),
                         ['exact-first-a', 'exact-first-b', 'exact-first-f'])
        self.assertEqual(gate.asked(desc)['lengths'][0], 123136)

    def test_the_drafter_plan_runs_both_controls_on_the_same_script(self):
        arms = by_name(arms_of('parked-drafter'))
        base = arms['drafter-f-solo'][1]
        for name in ('drafter-p-solo', 'drafter-neg-carry', 'drafter-neg-drafter'):
            args = arms[name][1]
            self.assertEqual(args[args.index('--prompt-lengths') + 1], base[base.index('--prompt-lengths') + 1])
            self.assertIn('--sequential-users', args)
        self.assertEqual(dict(arms['drafter-neg-carry'].env)['QWEN_FAST_PARKED_NEGATIVE'], 'carry')
        self.assertEqual(dict(arms['drafter-neg-drafter'].env)['QWEN_FAST_PARKED_NEGATIVE'], 'drafter')
        self.assertEqual(dict(arms['drafter-p-live'].env)['QWEN_FAST_SHARD_CHECK'], '1')
        self.assertEqual(dict(arms['drafter-f-live'].env)['QWEN_FAST_SHARD_CHECK'], '1')
        self.assertNotIn('QWEN_FAST_PARKED_NEGATIVE', dict(arms['drafter-p-solo'].env))

    def test_churn_is_two_hundred_admissions_over_four_arms(self):
        arms = arms_of('parked-churn')
        self.assertEqual(sum(gate.asked(arm[1])['streams'] for arm in arms), 200)
        scripts = set(arm[1][arm[1].index('--prompt-lengths') + 1] for arm in arms)
        self.assertEqual(len(scripts), 4)
        for arm in arms:
            self.assertEqual(arm[1][arm[1].index('--alive-check') + 1], '4')

    def test_the_corner_puts_the_arrival_behind_four_decoders_and_a_drop(self):
        arms = by_name(arms_of('parked-corner'))
        args = arms['corner-p'][1]
        self.assertEqual(gate.asked(args)['lengths'], list(gate.CORNER_LENGTHS))
        self.assertEqual(args[args.index('--drops') + 1], gate.CORNER_DROP)
        self.assertEqual(arms['corner-f'].profile, TWIN)

    def test_every_planned_arm_passes_the_contract_the_server_boots_under(self):
        import serving_c2_contract as contract
        for plan in list(PLANS) + ['warm']:
            for arm in arms_of(plan):
                served = contract.load_profile(PROFILES, arm.profile)
                self.assertEqual(contract.parked_problems(served, dict(arm.env)), [], (plan, arm[0], arm.profile))

    def test_a_parked_plan_refuses_a_profile_that_is_not_a_gate_profile(self):
        with self.assertRaisesRegex(gate.PlanError, 'gate-only knobs'):
            arms_of('parked-exact', PARKED[:-len('-gate')])

    def test_the_warm_plan_on_the_parked_profile_is_g_e0(self):
        arms = by_name(arms_of('warm'))
        self.assertEqual(arms['warm-solo'].profile, PARKED[:-len('-gate')], 'the traffic profile: no gate knob reaches it')
        self.assertEqual(arms['warm-4x131072'].profile, PARKED)
        self.assertTrue(all(arm.extra['parked']['on'] for arm in arms.values()))
        lengths = gate.asked(arms['warm-solo'][1])['lengths']
        self.assertTrue(set(parked_judge.RUNGS) <= set(lengths), 'every G-E1 rung is warmed')
        self.assertNotIn(('QWEN_FAST_PARKED_AUDIT', '1'), arms['warm-solo'].env)
        self.assertIn(('QWEN_FAST_PARKED_AUDIT', '1'), arms['warm-4x131072'].env)
        # on the sticky profile it is what it was
        for arm in gate.plan_arms('warm', TWIN, PROFILE_SET):
            self.assertFalse(arm.extra.get('parked'))
            self.assertNotIn(('QWEN_FAST_PARKED_AUDIT', '1'), arm.env)


class ParkedArmCheckTests(unittest.TestCase):
    def test_a_clean_parked_arm_and_a_clean_flag_off_arm(self):
        problems, missing, record = gate.parked_arm_check('arm', attach_log(rebinds=['a', 'b']), dict(on=True))
        self.assertEqual((problems, missing), ([], []))
        self.assertEqual(record['facts']['rebinds'], 2)
        self.assertEqual(gate.parked_arm_check('arm', 'nothing parked here', dict(on=False))[:2], ([], []))

    def test_a_flag_off_arm_that_shows_parked_lines_is_refused(self):
        problems, _, _ = gate.parked_arm_check('arm', attach_log(), dict(on=False))
        self.assertEqual(len(problems), 1)
        self.assertIn('leaves QWEN_FAST_PARKED_ENGINES off', problems[0])

    def test_the_gate_only_controls_must_print_exactly_what_the_arm_set(self):
        log = attach_log(extra=['[PINDIAG] parked negative control mode=carry (gate only: x)'])
        self.assertEqual(gate.parked_arm_check('arm', log, dict(on=True, negative='carry'))[0], [])
        self.assertEqual(len(gate.parked_arm_check('arm', log, dict(on=True))[0]), 1)
        self.assertEqual(len(gate.parked_arm_check('arm', log, dict(on=True, negative='drafter'))[0]), 1)
        self.assertEqual(len(gate.parked_arm_check('arm', attach_log(), dict(on=True, negative='carry'))[0]), 1)
        fault = attach_log(extra=['[PINDIAG] parked negative control fault=park (gate only: y)'])
        self.assertEqual(gate.parked_arm_check('arm', fault, dict(on=True, fault=True))[0], [])

    def test_the_ballast_line_is_required_when_asked_and_refused_when_not(self):
        log = attach_log(extra=['[PINDIAG] gate dram ballast bytes=1 buffers=1 (gate only; unread) x'])
        self.assertEqual(gate.parked_arm_check('arm', log, dict(on=True, ballast=True))[0], [])
        self.assertEqual(len(gate.parked_arm_check('arm', attach_log(), dict(on=True, ballast=True))[0]), 1)
        self.assertEqual(len(gate.parked_arm_check('arm', log, dict(on=True))[0]), 1)

    def test_g_e1_zero_counts_and_a_stopped_or_unparked_set(self):
        log = attach_log(extra=['[PACKED-PROPOSE] pair=[0, 1] fallback=dram_reserve'])
        self.assertEqual(len(gate.parked_arm_check('arm', log, dict(on=True, zero_counts=True))[0]), 1)
        self.assertEqual(gate.parked_arm_check('arm', log, dict(on=True))[0], [])
        self.assertEqual(len(gate.parked_arm_check('arm', attach_log(k=3), dict(on=True))[0]), 1)
        unparked = attach_log(unparked=[(1, 'x')])
        self.assertEqual(len(gate.parked_arm_check('arm', unparked, dict(on=True))[0]), 1)
        self.assertEqual(gate.parked_arm_check('arm', unparked, dict(on=True, allow_unparks=True))[0], [])
        self.assertIn('no server.log', gate.parked_arm_check('arm', None, dict(on=True))[0][0])


class RunnerTests(unittest.TestCase):
    """Runner.run with the docker step replaced: the parked check reaches the report and the arm's problems."""

    def test_the_arms_log_is_read_with_the_parked_markers(self):
        with tempfile.TemporaryDirectory() as results:
            runner = gate.Runner('img', PARKED, results, HERE, ['/a', '/b'], '/hub',
                                 execute=lambda arguments, stdout_path, timeout, name: self.write(results, name, stdout_path),
                                 log=lambda text: None, containers=lambda: [], profiles=PROFILE_SET)
            report = runner.run('arm', ['--users', '1'], 10, profile=PARKED, parked=dict(on=True))
            self.assertIsNotNone(report)
            self.assertEqual(report['c2_gate_parked']['facts']['built'][0]['k'], 3)
            self.assertTrue(any('built 3 of 4' in text for text in report['c2_gate_problems']), report['c2_gate_problems'])

    @staticmethod
    def write(results, name, stdout_path):
        arm = name.replace(gate.CONTAINER_PREFIX, '')
        with open(os.path.join(results, arm, 'server.log'), 'w') as handle:
            handle.write(attach_log(k=3))
        with open(stdout_path, 'w') as handle:
            handle.write('%s\n%s\n%s\n' % (gate.harness.BEGIN, json.dumps(dict(streams=[], gate_passed=True)),
                                          gate.harness.END))
        return 0


class JudgeRunTests(unittest.TestCase):
    """The plans' runners on synthetic reports (run_arm patched)."""

    def run_plan(self, plan, table, **s2):
        arms = arms_of(plan, **s2)
        feed = Feed(table)
        with patch.object(gate, 'run_arm', feed):
            result = gate.run_parked_plan(plan, None, PROFILE_SET, arms)
        return result, feed, by_name(arms)

    def exact_table(self, arms, change=None, drafting=None):
        table = {}
        for arm in arms:
            table[arm[0]] = (lambda spec, arm=arm: synthetic(spec, text_of=(change or {}).get(arm[0])
                                                              or (lambda user: 'ref %d' % user),
                                                              paths=(drafting or {}).get(arm[0])))
        return table

    def test_parked_exact_passes_when_every_order_matches_the_twin(self):
        arms = arms_of('parked-exact')
        result, feed, _ = self.run_plan('parked-exact', self.exact_table(arms))
        self.assertEqual(result['verdict'], 'PASS', result['lines'])
        self.assertEqual(feed.asked, [arm[0] for arm in arms])
        self.assertEqual(result['facts']['exact-p-shuf']['drafting_identical'], True)

    def test_a_request_that_ends_at_its_first_token_owes_no_rebind_and_no_path_records(self):
        arms = arms_of('parked-exact')
        ended = [i for i, b in enumerate(gate.asked(arms[0][1])['budgets']) if b == 1]
        self.assertTrue(ended, 'the 2048 rung has a budget of 1')

        def report(spec, rebinds):
            made = synthetic(spec, parked_record=dict(on=True, facts=dict(rebinds=rebinds)) if spec.role == 'p' else None)
            for user in ended:
                made['s2']['paths']['users'].pop(str(user), None)
            return made
        streams = gate.asked(arms[0][1])['streams']
        table = dict((arm[0], (lambda spec, arm=arm: report(spec, streams - len(ended)))) for arm in arms)
        result, _, _ = self.run_plan('parked-exact', table)
        self.assertEqual(result['verdict'], 'PASS', result['lines'])
        table = dict((arm[0], (lambda spec, arm=arm: report(spec, streams))) for arm in arms)
        result, _, _ = self.run_plan('parked-exact', table)
        self.assertNotEqual(result['verdict'], 'PASS', 'a rebind for a request that never reached the engine is a miscount')

    def test_parked_exact_fails_on_one_changed_token_and_on_one_changed_round(self):
        arms = arms_of('parked-exact')
        result, _, _ = self.run_plan('parked-exact', self.exact_table(
            arms, change={'exact-p-asc': lambda user: 'ref %d' % user if user != 4 else 'other'}))
        self.assertEqual(result['verdict'], 'FAIL')
        self.assertTrue(any('exact-p-asc against exact-f-desc: user 4 is DIVERGED' in text
                            for text in result['s2_problems']), result['s2_problems'])
        result, _, _ = self.run_plan('parked-exact', self.exact_table(
            arms, drafting={'exact-p-desc': {3: [('S', 10, 1, 1), ('P', 11, 1, 1)]}}))
        self.assertEqual(result['verdict'], 'FAIL')
        self.assertTrue(any('user 3 drafts differently' in text for text in result['s2_problems']))

    def test_parked_exact_fails_on_a_first_request_that_differs(self):
        arms = arms_of('parked-exact')
        result, _, _ = self.run_plan('parked-exact', self.exact_table(
            arms, change={'exact-first-b': lambda user: 'wrong'}))
        self.assertEqual(result['verdict'], 'FAIL')
        self.assertTrue(any('exact-first-b against exact-first-f' in text for text in result['s2_problems']))

    def test_the_ballast_arms_must_exercise_the_rebind(self):
        arms = arms_of('parked-ballast')

        def table(facts):
            return dict((arm[0], (lambda spec: synthetic(spec, parked_record=dict(on=True, facts=facts)))) for arm in arms)
        result, _, _ = self.run_plan('parked-ballast', table(dict(rebinds=1, peak_held_max=90_000_000)))
        self.assertEqual(result['verdict'], 'PASS', result)
        result, _, _ = self.run_plan('parked-ballast', table(dict(rebinds=0)))
        self.assertNotEqual(result['verdict'], 'PASS')
        self.assertTrue(any('did not exercise R' in text for text in result.get('s2_shortfalls', []) or [str(result)]),
                        result)

    def test_a_missing_arm_report_fails_the_plan(self):
        arms = arms_of('parked-exact')
        table = self.exact_table(arms)
        table['exact-p-desc'] = None
        result, _, _ = self.run_plan('parked-exact', table)
        self.assertEqual(result['verdict'], 'FAIL')

    def drafter_table(self, arms, carry_text=None, drafter_paths=None, live_mean=3.5, carry_paths=None):
        stale = {0: [('S', 10, 1, 1), ('P', 11, 1, 1)]}
        table = {}
        for arm in arms:
            name = arm[0]
            if name == 'drafter-neg-carry':
                table[name] = (lambda spec: synthetic(spec, text_of=carry_text or (lambda user: 'ref %d' % user
                                                                                 if user else 'broken')))
            elif name == 'drafter-neg-drafter':
                table[name] = (lambda spec: synthetic(spec, text_of=lambda user: 'ref %d' % user,
                                                      paths=stale if drafter_paths is None else drafter_paths))
            elif name == 'drafter-p-live':
                table[name] = (lambda spec: synthetic(spec, text_of=lambda user: 'ref %d' % user, mean=live_mean))
            else:
                table[name] = (lambda spec: synthetic(spec, text_of=lambda user: 'ref %d' % user))
        return table

    def test_parked_drafter_passes_with_both_controls_doing_their_jobs(self):
        arms = arms_of('parked-drafter')
        result, _, _ = self.run_plan('parked-drafter', self.drafter_table(arms))
        self.assertEqual(result['verdict'], 'PASS', result['lines'] + result.get('s2_problems', []))
        self.assertTrue(any('drafter-neg-carry' in line and 'did its job' in line for line in result['lines']))
        self.assertFalse(result['facts']['drafter-neg-drafter']['equivalence']['solo']['identical'])

    def test_parked_drafter_fails_a_control_that_does_not_do_its_job(self):
        arms = arms_of('parked-drafter')
        result, _, _ = self.run_plan('parked-drafter', self.drafter_table(
            arms, carry_text=lambda user: 'ref %d' % user))
        self.assertEqual(result['verdict'], 'FAIL')
        self.assertTrue(any('kept every token' in text for text in result['s2_problems']))
        result, _, _ = self.run_plan('parked-drafter', self.drafter_table(arms, drafter_paths={}))
        self.assertEqual(result['verdict'], 'FAIL')
        self.assertTrue(any('could not be judged' in text or 'invisible' in text for text in result['s2_problems']))
        same = {user: [('S', 10, 3, 3), ('P', 13, 4, 4)] for user in range(8)}
        result, _, _ = self.run_plan('parked-drafter', self.drafter_table(arms, drafter_paths=same))
        self.assertEqual(result['verdict'], 'FAIL')
        self.assertTrue(any('invisible to the equivalence judge' in text for text in result['s2_problems']))

    def test_parked_drafter_fails_a_concurrent_acceptance_off_by_more_than_two_percent(self):
        arms = arms_of('parked-drafter')
        result, _, _ = self.run_plan('parked-drafter', self.drafter_table(arms, live_mean=5.0))
        self.assertEqual(result['verdict'], 'FAIL')
        self.assertTrue(any('concurrent acceptance' in text for text in result['s2_problems']))

    def test_parked_lifecycle_judges_the_fault_arm(self):
        arms = arms_of('parked-lifecycle')
        good = dict(built=[dict(k=4)], unparked=[dict(slot=0, reason='injected park fault (gate only)')], reparked=1,
                    rebinds=5)
        solo = synthetic(by_name(arms)['parked-lifecycle-solo'], text_of=lambda user: 'ref %d' % user)
        for facts, verdict in ((good, 'PASS'), (dict(good, reparked=0), 'NOT_EXERCISED'),
                               (dict(good, unparked=[]), 'FAIL')):
            with self.subTest(verdict=verdict), patch.object(gate, 'run_lifecycle', lambda plan, runner, specs: dict(
                    verdict='PASS', lines=[])), patch.object(gate, 'arm_report', lambda runner, name: solo), \
                    patch.object(gate, 'run_arm', lambda runner, plan, spec, suffix='': synthetic(
                        spec, text_of=lambda user: 'ref %d' % user, parked_record=dict(facts=facts, on=True))):
                result = gate.run_parked_plan('parked-lifecycle', None, PROFILE_SET, arms)
            self.assertEqual(result['verdict'], verdict, result['lines'] + result.get('s2_problems', []))

    def test_parked_churn_judges_the_idle_return_and_the_rebinds(self):
        arms = arms_of('parked-churn')
        record = lambda spec: dict(facts=dict(rebinds=gate.asked(spec[1])['streams'], unparked=[]), on=True)

        def table(drift, odd=None):
            return dict((arm[0], (lambda spec, arm=arm: synthetic(spec, parked_record=record(spec), extra=dict(
                ledger=dict(idle_drift_gb=odd if odd is not None and arm[0] == arms[0][0] else drift))))) for arm in arms)
        # (the floors themselves are memory_s2_checks', held by the S2 gate's tests)
        with patch.object(gate, 'memory_s2_checks', lambda *args, **kwargs: ([], [])):
            result, _, _ = self.run_plan('parked-churn', table({'0': 0.004, '1': 0.0}))
            self.assertNotEqual(result['verdict'], 'FAIL', result.get('s2_problems'))
            # the first prefill leaves a step behind the first reading: every arm at -0.12 GB is no leak
            result, _, _ = self.run_plan('parked-churn', table({'0': -0.12, '1': -0.12}))
            self.assertNotEqual(result['verdict'], 'FAIL', result.get('s2_problems'))
            # one arm off the others by more than 16 MB is one
            result, _, _ = self.run_plan('parked-churn', table({'0': -0.12, '1': -0.12}, odd={'0': -0.05, '1': -0.12}))
            self.assertEqual(result['verdict'], 'FAIL')
            self.assertTrue(any('from the plan' in text for text in result['s2_problems']), result['s2_problems'])
            # and a drift no step explains fails whichever arms share it
            result, _, _ = self.run_plan('parked-churn', table({'0': 0.6, '1': 0.6}))
            self.assertEqual(result['verdict'], 'FAIL')

    def test_idle_rebuilds_widen_an_arms_band_by_the_single_they_made(self):
        drifts = {'a': {'0': -0.12}, 'b': {'0': -0.12}, 'c': {'0': -0.12 + 0.227}}
        self.assertEqual(len(judge.idle_return_across(drifts)[0]), 1)
        self.assertEqual(judge.idle_return_across(drifts, {'c': 1}), ([], []))
        self.assertEqual(len(judge.idle_return_across({'a': None})[1]), 1)

    def test_parked_corner_and_ballast_fail_only_on_a_death_or_a_refusal(self):
        arms = arms_of('parked-corner')
        table = dict((arm[0], (lambda spec: synthetic(spec, text_of=lambda user: 'ref %d' % user))) for arm in arms)
        result, _, _ = self.run_plan('parked-corner', table)
        self.assertEqual(result['verdict'], 'NOT_EXERCISED', 'no quad departure and no drop fired in the synthetic')
        table['corner-p'] = lambda spec: synthetic(spec, text_of=lambda user: 'ref %d' % user, extra=dict(fatal='oom'))
        result, _, _ = self.run_plan('parked-corner', table)
        self.assertEqual(result['verdict'], 'FAIL')
        arms = arms_of('parked-ballast')
        refused = dict((arm[0], (lambda spec: synthetic(spec, extra=dict(s2=dict(quarantined=1))))) for arm in arms)
        result, _, _ = self.run_plan('parked-ballast', refused)
        self.assertEqual(result['verdict'], 'FAIL')
        fine = dict((arm[0], (lambda spec: synthetic(spec, parked_record=dict(on=True, facts=dict(
            rebinds=1, peak_held_max=90_000_000))))) for arm in arms)
        result, _, _ = self.run_plan('parked-ballast', fine)
        self.assertEqual(result['verdict'], 'PASS', result['lines'])


class GeZeroTests(unittest.TestCase):
    def test_the_warm_plan_records_the_attach_and_advises_the_ballast(self):
        built = dict(k=4, attach_ms=12000.0, free=1_400_000_000, largest_free=1_300_000_000, trace_used=206_000_000)
        record = dict(on=True, facts=dict(built=[built], p7p=[dict(chip=0, free_gb=1.40, largest_free_mb=1300.0),
                                                              dict(chip=1, free_gb=1.42, largest_free_mb=1310.0)],
                                          peak_held_max=90_000_000, peaks=5, rebinds=4,
                                          single_rebuilt=[dict(trace_delta=4_000_000, dram_delta=227_000_000)],
                                          single_rebuilt_count=1))
        reports = {'warm-4x%d' % gate.BRINGUP_PROMPT: dict(c2_gate_parked=record), 'warm-solo': dict(c2_gate_parked=record)}
        problems, shortfalls, facts, lines = gate.parked_g_e0(reports)
        self.assertEqual((problems, shortfalls), ([], []))
        self.assertIn('ballast_advice_bytes', facts)
        self.assertTrue(any('C2_GATE_BALLAST_MB=' in line for line in lines))
        self.assertTrue(any('R measured' in line for line in lines))
        bare = {'warm-4x%d' % gate.BRINGUP_PROMPT: dict(c2_gate_parked=dict(on=True, facts=dict(built=[built])))}
        _, shortfalls, _, _ = gate.parked_g_e0(bare)
        self.assertEqual(len(shortfalls), 2, 'no P7p and no peak reading are unrecorded, never a pass')
        solo = {'warm-solo': dict(c2_gate_parked=dict(on=True, facts=dict(built=[built])))}
        self.assertEqual(len(gate.parked_g_e0(solo)[1]), 1, 'the solo warm runs no audit: only its P7p is owed')

    def test_a_measurement_above_the_admission_constants_fails(self):
        import serving_prefill_admission as admission
        built = dict(k=4, attach_ms=1.0, free=1, largest_free=1, trace_used=1)
        p7p = [dict(chip=0, free_gb=1.4, largest_free_mb=1300.0)]

        def run(peak, dram, trace):
            record = dict(on=True, facts=dict(built=[built], p7p=p7p, peak_held_max=peak, peaks=1, rebinds=1,
                                              single_rebuilt=[dict(trace_delta=trace, dram_delta=dram)],
                                              single_rebuilt_count=1))
            return gate.parked_g_e0({'warm-4x%d' % gate.BRINGUP_PROMPT: dict(c2_gate_parked=record)})[0]

        self.assertEqual(run(admission.PARKED_REBIND_BYTES, admission.MEASURED_SINGLE_CAPTURE_BYTES,
                             admission.SINGLE_TRACE_BYTES), [])
        problems = run(admission.PARKED_REBIND_BYTES + 1, admission.MEASURED_SINGLE_CAPTURE_BYTES + 1,
                       admission.SINGLE_TRACE_BYTES + 1)
        self.assertEqual(len(problems), 3, problems)


class JobTests(unittest.TestCase):
    def values(self, **changes):
        with open(os.path.join(HERE, '..', '..', '.github', 'c2-serving-job.env'), encoding='utf-8') as handle:
            text = handle.read()
        values = job.parse_env(text)
        values.update(changes)
        return values

    def outputs(self, **changes):
        return job.read_job(self.values(**changes), job.profile_names(PROFILES))

    def test_the_template_carries_the_key_empty(self):
        self.assertEqual(self.values()['C2_GATE_BALLAST_MB'], '')
        self.assertEqual(self.outputs()['gate_ballast_mb'], '')

    def test_the_parked_plans_are_known_and_the_ballast_key_goes_with_its_plan(self):
        for plan in PLANS[:-1]:
            self.assertEqual(self.outputs(C2_ACTIONS='status reset gate', C2_GATE_PLAN=plan,
                                          C2_PROFILE=PARKED)['gate_plan'], plan)
        with self.assertRaisesRegex(job.JobError, 'needs C2_GATE_BALLAST_MB'):
            self.outputs(C2_ACTIONS='status reset gate', C2_GATE_PLAN='parked-ballast', C2_PROFILE=PARKED)
        self.assertEqual(self.outputs(C2_ACTIONS='status reset gate', C2_GATE_PLAN='parked-ballast',
                                      C2_PROFILE=PARKED, C2_GATE_BALLAST_MB='1400')['gate_ballast_mb'], '1400')
        with self.assertRaisesRegex(job.JobError, 'belongs to parked-ballast'):
            self.outputs(C2_ACTIONS='status reset gate', C2_GATE_PLAN='parked-exact', C2_GATE_BALLAST_MB='1400')
        with self.assertRaisesRegex(job.JobError, 'at most'):
            self.outputs(C2_ACTIONS='status reset gate', C2_GATE_PLAN='parked-ballast', C2_GATE_BALLAST_MB='9999')

    def test_the_gate_takes_the_ballast_and_refuses_a_bad_one(self):
        logs = []
        with tempfile.TemporaryDirectory() as results:
            code = gate.main(['--image', 'img', '--profile', PARKED, '--plan', 'parked-ballast', '--results', results,
                              '--profiles', PROFILES, '--ballast-mb', '1500', '--dry-run'], log=logs.append,
                             devices=['/a', '/b'])
            self.assertEqual(code, 0, logs)
            refused = gate.main(['--image', 'img', '--profile', PARKED, '--plan', 'parked-ballast', '--results', results,
                                 '--profiles', PROFILES, '--ballast-mb', '999999', '--dry-run'], log=logs.append)
            self.assertEqual(refused, 2)
            self.assertEqual(gate.main(['--image', 'img', '--profile', PARKED, '--plan', 'parked-ballast',
                                        '--results', results, '--profiles', PROFILES, '--dry-run'], log=logs.append), 2)

    def test_the_workflow_passes_the_key_to_the_gate(self):
        with open(os.path.join(HERE, '..', '..', '.github', 'workflows', 'qwen-c2-serving.yml'),
                  encoding='utf-8') as handle:
            text = handle.read()
        self.assertIn('GATE_BALLAST_MB: ${{ steps.job.outputs.gate_ballast_mb }}', text)
        self.assertIn('${GATE_BALLAST_MB:+--ballast-mb "$GATE_BALLAST_MB"}', text)


class HarnessTests(unittest.TestCase):
    def test_a_sequential_run_takes_a_start_order_and_sends_in_it(self):
        import lever_n_m3native_gate as harness

        options = harness.parse_options(['--users', '1', '--sequential-users', '4', '--start-order', '3,1,0,2'])
        self.assertEqual(harness.sequential_order(options), [3, 1, 0, 2])
        plain = harness.parse_options(['--users', '1', '--sequential-users', '4'])
        self.assertEqual(harness.sequential_order(plain), [0, 1, 2, 3])
        for argv in (['--users', '1', '--sequential-users', '4', '--start-order', '0,1,2'],
                     ['--users', '1', '--sequential-users', '4', '--start-order', '0,1,1,2'],
                     ['--users', '4', '--start-order', '3,2,1,0']):
            with self.subTest(argv=argv), patch('sys.stderr', new_callable=io.StringIO), self.assertRaises(SystemExit):
                harness.parse_options(argv)

    def test_the_trace_region_reports_its_smallest_use_too(self):
        import lever_n_m3native_gate as harness

        region = harness.trace_region('')
        self.assertIsNone(region['min_used_gb'])
        self.assertIn('min_used_gb', region)


class PrefixGateTests(unittest.TestCase):
    def test_a_parked_arm_is_read_with_the_parked_markers(self):
        import c2_prefix_gate as prefix

        arm = dict(served_env=dict(QWEN_FAST_PARKED_ENGINES='1', QWEN_FAST_ANY_REQUEST='1'), env=(), sticky=False)
        problems, missing, lines, _ = prefix.s2_findings(arm, attach_log(k=3), {}, [])
        self.assertTrue(any(text.startswith('parked:') and 'built 3 of 4' in text for text in problems), problems)
        self.assertTrue(any(line.startswith('parked:') for line in lines))
        arm = dict(served_env=dict(QWEN_FAST_ANY_REQUEST='1'), env=(), sticky=False)
        problems, _, _, _ = prefix.s2_findings(arm, attach_log(), {}, [])
        self.assertTrue(any('leaves QWEN_FAST_PARKED_ENGINES off' in text for text in problems), problems)


if __name__ == '__main__':
    unittest.main()
