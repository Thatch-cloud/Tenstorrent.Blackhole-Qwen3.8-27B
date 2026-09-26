"""S2 gate tooling (s2-design.md W11), held on CPU: the S2 plans' arms, the harness's S2 record, the path
records and four-live rate, the strict policy with its divergence records and the D-c(i) switch, the S2
verdicts, and the whole driver with injected docker, server logs and kernel-cache counter.

Nothing here opens a device or runs docker. The S2 profiles are W8's; until they land in the checkout this
module builds them as the design defines them (s2_profiles), and uses the checkout's once they exist."""

import datetime
import io
import json
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import acceptance_report  # noqa: E402
import c2_serving_gate as driver  # noqa: E402
import c2_serving_job as job  # noqa: E402
import lever_n_m3native_gate as gate  # noqa: E402
import real_text_compare as compare  # noqa: E402
import test_c2_serving_gate as base  # noqa: E402

V235 = base.V235
AUDIT = gate.EXTENT_AUDIT_FLAG
CAPTURE = gate.CAPTURE_POSITION_FLAG
FORCE = gate.FORCE_CAP_FLAG
EXTENT = gate.EXTENT_REPLAY_FLAG
BLOCK_KEYS = ('QWEN_FAST_PACKED_STEP', 'QWEN_FAST_PADDED_BLOCK')


def s2_profiles():
    """The checkout's profiles with c2-packed and c2-packed-gate as s2-design 1.2 defines them (W8's own when
    the checkout carries them): c2-packed is c2 without its two block overrides plus the flag, c2-packed-gate is
    c2-gate plus the flag, and both keep their parent's limits and exact's engine."""
    profiles = base.profiles_with_c2()
    if 'c2-packed' not in profiles['profiles']:
        packed = json.loads(json.dumps(profiles['profiles']['c2']))
        for key in BLOCK_KEYS:
            packed['env'].pop(key, None)
        packed['env'][EXTENT] = '1'
        profiles['profiles']['c2-packed'] = packed
    if 'c2-packed-gate' not in profiles['profiles']:
        packed_gate = json.loads(json.dumps(profiles['profiles']['c2-gate']))
        packed_gate['env'][EXTENT] = '1'
        profiles['profiles']['c2-packed-gate'] = packed_gate
    return profiles


PROFILES = s2_profiles()
BASE_TIME = datetime.datetime(2026, 9, 27, 1, 0, 0)


def stamp(seconds):
    return (BASE_TIME + datetime.timedelta(seconds=seconds)).strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]


def execute_line(seconds, live):
    return ('%s | INFO     | serving_worker_hook:execute_model:272 - [PHASE] execute total=%d new=0 cached=%d spec=%d '
            'finished=0 preempted=0' % (stamp(seconds), 16 * live, live, 15 * live))


def request_id(user):
    return 'cmpl-%032x' % (user + 1)


def engine_id(user):
    return request_id(user) + '-0-abcd1234'


def s2_log(on=True, users=4, rounds=12, positions=None, emitted=5, round_ms=170.0, audit_ms=2.0, audit=True,
           mismatch=False, drop_audit=0, words_ok=None, cap=None, admission=True, packed=True, extra=(), families=None,
           ladder='[2048]', sequential_rows=4):
    """A server log of an S2 arm: the C2-any lines, the S2 attach lines when `on`, and `rounds` decode steps -
    packed (four [PACKED] lines, a packed extent round line and an audit line each) or, `packed` False, one
    [SEQ-PUBLISH] step line per user - on one timestamped clock, `round_ms` apart."""
    lines = base.any_request_log(requests=users, ladder=ladder).splitlines()
    if on:
        if admission:
            lines.append('INFO [PINDIAG] packed-any admission admitted runtime=152951c1 evidence=ok')
        lines += ['INFO [PINDIAG] sdpa qwen-modes binary /opt/tt-metal/ttnn/_ttnncpp.so carries the [QWEN-SDPA] branch',
                  'INFO [PINDIAG] sdpa qwen-modes modes=extent,share,slice,tail rows=16 capacity=131328 bundles=[2] '
                  'flags=[\'0x27\'] mask=narrow',
                  'INFO [QWEN-SDPA] flags=0x27 kv_share=1 q_slice=1',
                  'INFO [QWEN-SDPA] q-slice rows_per_kv=48',
                  'INFO [QWEN-SDPA] runtime-extent entries=2',
                  'INFO [PINDIAG] extent replay engaged segments=4 flags=[0x27] mask=narrow capacity=131328']
    lines += list(extra)
    at = 0.0
    position = list(positions or [131072] * users)
    for index in range(rounds):
        lines.append(execute_line(at, users))
        if packed:
            fams = families[index % len(families)] if families else [gate.extent_of(p) for p in position]
            for user in range(users):
                suffix = '' if cap is None else ' cap=%d' % (cap if not callable(cap) else cap(position[user]))
                lines.append('INFO [PACKED] request=%s segment=%d position=%d prefix=%d emitted=%d predictions=[1, 2, 3]%s'
                             % (engine_id(user), user, position[user], emitted, emitted, suffix))
            if on:
                lines.append('INFO [PINDIAG] packed extent round round=%d live=%d families=[%s] idle=[] capped=[]'
                             % (index + 1, users, ','.join(str(f) for f in fams)))
                if audit and index >= drop_audit:
                    ok = users if words_ok is None else words_ok
                    lines.append('INFO [EXTENT-AUDIT] round=%d segments=%d words_ok=%d cur_pos_ok=%d mask_ok=1 '
                                 'tables_ok=1 ms=%.2f' % (index + 1, users, ok, users, audit_ms))
                if mismatch and index == 3:
                    lines.append('WARNING [EXTENT-AUDIT] MISMATCH round=4 segment=2 word=131072 expected=0')
            position = [p + emitted for p in position]
        else:
            for user in range(users):
                lines.append('INFO [SEQ-PUBLISH] request=%s rows=%d prefix=3 step_ms=40.00 cache=5->5'
                             % (engine_id(user), sequential_rows))
                lines.append('INFO [SEQ-PUBLISH] request=%s stages features=0.10' % engine_id(user))
            position = [p + 3 for p in position]
        at += (round_ms + (audit_ms if on and audit and packed else 0.0)) / 1000.0
    lines.append(execute_line(at, users))
    lines.append('INFO trigger received signal=SIGTERM')
    return '\n'.join(lines) + '\n'


def configuration(profile, env=()):
    return dict(base.served_configuration(profile, PROFILES), **dict(env))


def s2_arm_report(log_text, profile, env=(), texts=None, users=4, lengths=None, finish='length', completion=256,
                  max_tokens=256, sequential=0, chunk=None):
    """A gate report as the harness would print it for an S2 arm: the served report fixture, the streams' ids,
    the configuration the knobs reach, and report['s2'] computed from `log_text` by the harness itself."""
    if texts is None and users > len(V235['streams']):
        texts = [V235['streams'][index % len(V235['streams'])]['text'] for index in range(users)]
    report = base.served_report(texts=texts, profile=profile, users=users, lengths=lengths, finish=finish,
                                completion=completion, max_tokens=max_tokens,
                                configuration=configuration(profile, env),
                                argv=base.served_argv(profile, PROFILES))
    if lengths is not None:
        for index, entry in enumerate(report['real_text']['users']):
            entry['prompt_tokens'] = lengths[index]
            entry['prompt_sha256'] = 'sha-%d-%d' % (index, lengths[index])
        for stream, length in zip(report['streams'], lengths):
            stream['prompt_tokens'] = length
    for index, stream in enumerate(report['streams']):
        stream['request_id'] = request_id(index)
        if chunk is not None:
            stream['chunk_chars'], stream['chunk_tokens'] = chunk(stream.get('text') or '')
    report['sequential_users'] = sequential
    environ = dict(configuration(profile, env))
    gate.add_s2_report(report, environ, log_text, report['streams'],
                       lengths or [131072] * users)
    report['gate_passed'] = not report['flag_markers']['missing']
    return report


class ArmTests(unittest.TestCase):
    def test_an_arm_is_a_three_tuple_to_everything_that_unpacks_it(self):
        arm = driver.Arm('x', ['--users', '4'], 60, profile='c2-packed', env=((AUDIT, '1'),), rerun=True, role='on',
                         pair=2)
        name, args, timeout = arm
        self.assertEqual((name, args, timeout), ('x', ['--users', '4'], 60))
        self.assertEqual(arm, ('x', ['--users', '4'], 60))
        self.assertEqual((arm.profile, arm.env, arm.rerun, arm.role, arm.extra), ('c2-packed', ((AUDIT, '1'),), True,
                                                                                  'on', dict(pair=2)))
        self.assertEqual(driver.arm_runs('warm', arm), 2)
        self.assertEqual(driver.arm_runs('matrix', ('a', [], 1)), 2)
        self.assertEqual(driver.arm_runs('memory', ('a', [], 1)), 1)

    def test_the_s1_plans_off_the_flag_are_plain_tuples_and_the_agent_argv_unchanged(self):
        for plan in job.GATE_PLANS:
            with self.subTest(plan=plan):
                arms = driver.plan_arms(plan, 'c2', PROFILES)
                self.assertTrue(all(type(arm) is tuple for arm in arms))
        argv = driver.gate_run('img', 'n', 'c2', ['/dev/a'], '/c', '/r', ['--x'])
        self.assertEqual(argv, driver.gate_run('img', 'n', 'c2', ['/dev/a'], '/c', '/r', ['--x'], env=()))
        self.assertFalse(any(part.startswith('QWEN_FAST_EXTENT') for part in argv))

    def test_an_arm_may_add_only_gate_knobs(self):
        argv = driver.agent_shape('img', 'n', 'c2-packed', ['/dev/a'], env=((AUDIT, '1'), (CAPTURE, '16384')))
        self.assertEqual(argv[-4:], ['-e', '%s=1' % AUDIT, '-e', '%s=16384' % CAPTURE])
        self.assertLess(argv.index('QWEN_C2_PROFILE=c2-packed'), argv.index('%s=1' % AUDIT))
        for name in ('QWEN_FAST_PACKED_STEP', EXTENT, 'QWEN_FAST_OUTPUT_BUDGET'):
            with self.subTest(name=name), self.assertRaises(driver.PlanError):
                driver.agent_shape('img', 'n', 'c2', [], env=((name, '1'),))

    def test_the_s1_plans_on_an_s2_profile_audit_every_arm(self):
        for plan in job.GATE_PLANS:
            with self.subTest(plan=plan):
                arms = driver.plan_arms(plan, 'c2-packed-gate' if plan == 'bringup' else 'c2-packed', PROFILES)
                self.assertTrue(arms and all(arm.env[:1] == ((AUDIT, '1'),) for arm in arms))
        matrix = driver.plan_arms('matrix', 'c2-packed', PROFILES, s2=dict(audits='all'))
        self.assertEqual(matrix[0].env, driver.AUDIT_ENV + driver.G4_ALL_AUDITS)
        memory = driver.plan_arms('memory', 'c2-packed', PROFILES, s2=dict(audits='all'))
        self.assertEqual(memory[0].env, driver.AUDIT_ENV, 'the other audits only on the G4 serving arms')


class PlanTests(unittest.TestCase):
    def parse(self, arm):
        return base.parse_harness(arm[1])

    def test_control_is_abab_on_its_two_fixed_profiles_at_v235s_shape(self):
        arms = driver.plan_arms('control', 'c2-packed', PROFILES)
        self.assertEqual([arm[0] for arm in arms], ['control-off-1', 'control-on-1', 'control-off-2', 'control-on-2'])
        self.assertEqual([arm.profile for arm in arms], ['c2-gate', 'c2-packed-gate'] * 2)
        self.assertEqual([arm.env for arm in arms], [(), driver.AUDIT_ENV] * 2)
        for arm in arms:
            options = self.parse(arm)
            self.assertEqual((options.users, options.prompt_tokens, options.max_tokens, options.stagger),
                             (4, 131072, 256, 0.25))
            self.assertEqual(options.expect_profile, arm.profile)
        self.assertEqual(len(driver.plan_arms('control', 'c2', PROFILES, s2=dict(pairs=1))), 2)
        self.assertIn('c2-packed-gate', driver.REFERENCE_PROFILES)

    def test_forced_cap_caps_one_arm_only(self):
        off, on = driver.plan_arms('forced-cap', 'c2-packed', PROFILES)
        self.assertEqual((off[0], on[0], off.profile, on.profile), ('forced-cap-off', 'forced-cap-on', 'c2-packed-gate',
                                                                    'c2-packed-gate'))
        self.assertEqual(off.env, driver.AUDIT_ENV)
        self.assertEqual(on.env, driver.AUDIT_ENV + ((FORCE, '8'),))
        self.assertEqual(off[1], on[1])

    def test_control_below_keeps_every_ticket_in_its_family(self):
        arms = driver.plan_arms('control-below', 'c2-packed', PROFILES)
        self.assertEqual(len(arms), 8)
        for arm in arms:
            family = arm.extra['family']
            options = self.parse(arm)
            self.assertEqual(options.prompt_tokens, family - 256)
            self.assertEqual(options.max_tokens, driver.BELOW_MAX_TOKENS)
            self.assertEqual(arm.env[0], (CAPTURE, str(family - 256)))
            self.assertEqual(arm.env[1:], () if arm.role == 'off' else driver.AUDIT_ENV)
            last_start = options.prompt_tokens + options.max_tokens - 1
            self.assertLessEqual(last_start + 16, family, 'the last ticket stays in family F')
            self.assertEqual(gate.extent_of(options.prompt_tokens), family)
        self.assertEqual(sorted(set(arm.extra['family'] for arm in arms)), [4352, 16640])
        for families in ([4000], [20224], [3840]):
            with self.subTest(families=families), self.assertRaises(driver.PlanError):
                driver.plan_arms('control-below', 'c2', PROFILES, s2=dict(families=families))

    def test_the_g4_plans_serve_the_s2_profile_as_asked(self):
        mixed = driver.plan_arms('mixed', 'c2-packed', PROFILES)
        self.assertEqual([arm[0] for arm in mixed], ['mixed-concurrent', 'mixed-solo'])
        self.assertEqual([arm.rerun for arm in mixed], [True, True])
        self.assertEqual(self.parse(mixed[0]).prompt_lengths, list(driver.MIXED_LENGTHS))
        self.assertEqual(self.parse(mixed[1]).sequential_users, 4)
        second = driver.plan_arms('mixed', 'c2-packed', PROFILES, lengths=[5000, 40000, 90000, 123136])
        self.assertEqual(self.parse(second[0]).prompt_lengths, [5000, 40000, 90000, 123136])
        notes = []
        boundaries = driver.plan_arms('boundaries', 'c2-packed', PROFILES, notes=notes)
        options = self.parse(boundaries[0])
        self.assertEqual((options.max_tokens, options.events['ignore_eos']), (8192, [0, 1, 2, 3]))
        self.assertEqual(self.parse(boundaries[1]).events['ignore_eos'], [0, 1, 2, 3], 'the solo arm ignores EOS too')
        self.assertTrue(notes)
        staggered = driver.plan_arms('staggered', 'c2-packed', PROFILES)
        self.assertEqual([arm.role for arm in staggered], ['concurrent', 'solo', 'probe'])
        self.assertEqual(self.parse(staggered[0]).stagger, driver.STAGGER_SECONDS)
        self.assertEqual(staggered[2].env, driver.AUDIT_ENV + driver.PADDED_PROBE_ENV)
        self.assertFalse(staggered[2].rerun)
        short = driver.plan_arms('short', 'c2-packed', PROFILES)
        self.assertEqual(self.parse(short[0]).prompt_lengths, [60, 255, 2047, 120000])

    def test_the_128000_prompt_is_refused_on_c2_packed_before_any_container(self):
        with self.assertRaises(driver.PlanError) as caught:
            driver.plan_arms('mixed', 'c2-packed', PROFILES, lengths=[1536, 20000, 60000, 128000])
        self.assertIn('123136', str(caught.exception))
        # c2-packed-gate has no prompt cap: 128000 fits with 256 out, never with 4096.
        driver.plan_arms('mixed', 'c2-packed-gate', PROFILES, lengths=[1536, 128000], max_tokens=256)
        with self.assertRaises(driver.PlanError):
            driver.plan_arms('mixed', 'c2-packed-gate', PROFILES, lengths=[1536, 128000])

    def test_an_s2_plan_needs_the_s2_profiles(self):
        with self.assertRaises(driver.PlanError) as caught:
            driver.plan_arms('mixed', 'c2', PROFILES)
        self.assertIn('S2 plan', str(caught.exception))
        bare = base.profiles_with_c2()
        bare['profiles'].pop('c2-packed-gate', None)
        for plan in ('control', 'forced-cap', 'control-below', 'warm'):
            with self.subTest(plan=plan), self.assertRaises(driver.PlanError) as caught:
                driver.plan_arms(plan, 'c2', bare)
            self.assertIn('W8', str(caught.exception))
        swapped = s2_profiles()
        swapped['profiles']['c2-gate']['env'][EXTENT] = '1'
        with self.assertRaises(driver.PlanError):
            driver.plan_arms('control', 'c2', swapped)

    def test_warm_covers_every_length_a_judged_arm_serves_and_is_never_judged(self):
        warm = driver.plan_arms('warm', 'c2', PROFILES)
        warm_off = driver.plan_arms('warm-off', 'c2', PROFILES)
        self.assertEqual([arm[0] for arm in warm], ['warm-solo', 'warm-4x131072', 'warm-4x16384', 'warm-4x4096'])
        self.assertEqual([arm.profile for arm in warm_off], ['c2-gate'] * 3)
        self.assertTrue(all(arm.judged is False for arm in warm + warm_off))
        solo = self.parse(warm[0])
        served = set(solo.prompt_lengths)
        for plan in ('mixed', 'short', 'boundaries', 'staggered', 'churn', 'permuted', 'lifecycle-arrival'):
            for arm in driver.plan_arms(plan, 'c2-packed', PROFILES):
                self.assertLessEqual(set(self.parse(arm).prompt_lengths or []), served, (plan, arm[0]))
        self.assertLessEqual(set(driver.LIFECYCLE_LENGTHS), served)
        self.assertLessEqual({family - 256 for family in driver.BELOW_FAMILIES}, set(
            self.parse(arm).prompt_tokens for arm in warm[2:]))
        for index, length in enumerate(solo.prompt_lengths):
            if length < 2048:
                self.assertEqual(solo.events['max_tokens'][index], driver.WARM_SHORT_TOKENS)
                self.assertIn(index, solo.events['ignore_eos'])
                self.assertGreater(length + driver.WARM_SHORT_TOKENS, 2048, 'decodes past 2048 of history')
        self.assertEqual([arm.env[0][0] for arm in warm[2:]], [CAPTURE, CAPTURE])

    def test_churn_lifecycle_arrival_and_permuted(self):
        churn, = driver.plan_arms('churn', 'c2-packed', PROFILES)
        options = self.parse(churn)
        self.assertGreaterEqual(options.users, 8)
        self.assertEqual(options.alive_check, 4)
        self.assertTrue(any(length < 2048 for length in options.prompt_lengths))
        self.assertTrue(all(100000 <= length <= 123136 for length in options.prompt_lengths if length >= 2048))
        event, solo = driver.plan_arms('lifecycle-arrival', 'c2-packed', PROFILES)
        self.assertEqual(self.parse(event).events['drops'], {1: ('build', 0)})
        self.assertEqual(self.parse(solo).sequential_users, 6)
        forward, reverse = driver.plan_arms('permuted', 'c2-packed', PROFILES)
        self.assertEqual(self.parse(reverse).start_order, [3, 2, 1, 0])

    def test_every_s2_plan_fits_the_workflows_gate_step_alone(self):
        step = base.WorkflowTests.budget_literals()['step']
        arms = dict((plan, driver.plan_arms(plan, 'c2-packed', PROFILES)) for plan in job.S2_GATE_PLANS)
        for plan in job.S2_GATE_PLANS:
            with self.subTest(plan=plan):
                self.assertLessEqual(driver.worst_case_seconds([plan], arms), step)
        self.assertEqual(driver.worst_case_seconds(['staggered'], arms), 5 * (4200 + driver.ARM_OVERHEAD_SECONDS),
                         'the probe arm is never re-run')


class HarnessTests(unittest.TestCase):
    def test_the_extent_flag_promises_modes_0x27_and_f22(self):
        environ = dict(QWEN_FAST_SDPA_MODES='tail,share,slice')
        self.assertEqual(gate.sdpa_mode_markers(gate.sdpa_mode_names(environ))[2], '[QWEN-SDPA] flags=0x7 ')
        environ[EXTENT] = '1'
        markers = gate.sdpa_mode_markers(gate.sdpa_mode_names(environ))
        self.assertIn('[PINDIAG] sdpa qwen-modes modes=extent,share,slice,tail ', markers)
        self.assertIn('[QWEN-SDPA] flags=0x27 ', markers)
        self.assertIn(gate.SDPA_EXTENT_MARKER, markers)
        found = gate.flag_marker_report(environ, 4, s2_log())
        self.assertEqual(found['found']['QWEN_FAST_SDPA_MODES'], dict((marker, True) for marker in markers))

    def s2(self, log_text, env=(), on=True, streams=None, lengths=None):
        environ = dict(QWEN_FAST_SDPA_MODES='tail,share,slice')
        if on:
            environ[EXTENT] = '1'
        environ.update(env)
        streams = streams or [dict(request_id=request_id(u)) for u in range(4)]
        return gate.s2_report(environ, log_text, streams, lengths or [131072] * 4)

    def test_a_clean_flag_on_log_has_no_problem(self):
        report = self.s2(s2_log(), env=((AUDIT, '1'),))
        self.assertEqual(report['problems'], [])
        self.assertEqual(report['rounds']['count'], 12)
        self.assertEqual((report['extent_audit']['lines'], report['extent_audit']['mismatches']), (12, 0))
        self.assertEqual(report['extent_audit']['median_ms'], 2.0)
        self.assertEqual(report['live4']['timed_rounds'], 12)
        self.assertEqual(report['live4']['median_round_ms'], 172.0)
        self.assertEqual(report['live4']['net_median_round_ms'], 170.0)
        self.assertEqual(report['paths']['users']['0']['packed'], 12)
        self.assertEqual(report['paths']['position_mismatch_count'], 0)
        self.assertTrue(all(report['markers'].values()))

    def test_audit_mismatch_missing_and_incomplete_lines_fail(self):
        on = ((AUDIT, '1'),)
        self.assertIn('MISMATCH', ' '.join(self.s2(s2_log(mismatch=True), env=on)['problems']))
        self.assertIn('went unaudited', ' '.join(self.s2(s2_log(drop_audit=1), env=on)['problems']))
        self.assertIn('did not read back', ' '.join(self.s2(s2_log(words_ok=3), env=on)['problems']))
        self.assertEqual(self.s2(s2_log(audit=False))['problems'], [], 'no audit asked, none needed')

    def test_the_attach_lines_are_required_under_the_flag(self):
        missing = self.s2(s2_log(admission=False))['problems']
        self.assertTrue(any('packed-any admission' in problem for problem in missing))
        text = s2_log().replace('[PINDIAG] packed extent round', '[PINDIAG] packed other round')
        self.assertTrue(any('did not run the extent verify' in problem for problem in self.s2(text)['problems']))

    def test_s2_lines_off_the_flag_are_a_leak(self):
        report = self.s2(s2_log(on=True), on=False)
        self.assertTrue(any('this is not the profile' in problem for problem in report['problems']))
        self.assertFalse(gate.s2_relevant(dict(QWEN_FAST_SDPA_MODES='tail'), s2_log(on=False)))
        self.assertTrue(gate.s2_relevant({CAPTURE: '16384'}, ''))

    def test_packed_below_128_cap_refusals_deadlines_and_an_unlogged_knob_fail(self):
        low = self.s2(s2_log(positions=[100, 131072, 131072, 131072], rounds=2), lengths=[100, 131072, 131072, 131072])
        self.assertTrue(any('below position 128' in problem for problem in low['problems']))
        text = s2_log(extra=['WARNING [PINDIAG] packed extent cap refused segment=1 prefix=7 limit=6',
                             'ERROR [PINDIAG] replay deadline exceeded round=9 segments=4 families=[256]'])
        problems = ' '.join(self.s2(text)['problems'])
        self.assertIn('block backstop', problems)
        self.assertIn('deadline', problems)
        knob = self.s2(s2_log(on=False), env=((CAPTURE, '16384'),), on=False)
        self.assertTrue(any('never reached the block' in problem for problem in knob['problems']))
        taken = self.s2(s2_log(on=False, extra=['INFO [PINDIAG] packed capture position override=16384 (gate only)']),
                        env=((CAPTURE, '16384'),), on=False)
        self.assertEqual(taken['problems'], [])

    def test_add_s2_report_puts_its_problems_where_gate_passed_reads_them(self):
        report = dict(flag_markers=dict(found={}, missing=['x']))
        gate.add_s2_report(report, {EXTENT: '1'}, s2_log(admission=False), [], None)
        self.assertEqual(report['flag_markers']['missing'][0], 'x')
        self.assertTrue(any('admission' in line for line in report['flag_markers']['missing'][1:]))
        untouched = dict(flag_markers=dict(found={}, missing=[]))
        gate.add_s2_report(untouched, {}, s2_log(on=False), [], None)
        self.assertEqual(untouched, dict(flag_markers=dict(found={}, missing=[])), 'a flag-off arm keeps its keys')

    def test_extent_rounds_refused_rounds_and_before_points(self):
        text = '\n'.join([
            'INFO [PINDIAG] packed extent round round=1 live=4 families=[256,2304,16640,131328] idle=[] capped=[2:6]',
            'INFO [PINDIAG] packed extent round round=2 live=2 families=[512,512] idle=[0,32] capped=[]',
            'WARNING [PACKED] A round the block cannot serve (x) holds tickets no request engine captured (y)',
            'WARNING [PINDIAG] packed refused round aborted 1/2 FINISHED_ABORTED via the request quarantine, engine kept: '
            'request=a',
            'WARNING [PINDIAG] packed refused round aborted 2/2 FINISHED_ABORTED via the request quarantine, engine kept: '
            'request=b',
            '[MEMLEDGER] phase=prefill point=before prompt=120000 chip0 allocated=1.000GB free=2.000GB '
            'largest_free=1500.0MB total=34.000GB known=1.000GB residual=0.000GB',
            '[MEMLEDGER] phase=prefill point=before prompt=60 chip1 allocated=1.000GB free=2.000GB '
            'largest_free=400.0MB total=34.000GB known=1.000GB residual=0.000GB',
            '[MEMLEDGER] phase=quad point=before round=4 estimate=0.700GB chip0 allocated=1.000GB free=2.000GB '
            'largest_free=900.0MB total=34.000GB known=1.000GB residual=0.000GB',
            '[MEMLEDGER] phase=engine point=before req=abc chip0 allocated=1.000GB free=2.000GB largest_free=1100.0MB '
            'total=34.000GB known=1.000GB residual=0.000GB'])
        rounds = gate.extent_rounds(text)
        self.assertEqual((rounds['count'], rounds['max_families'], rounds['multi_family_rounds']), (2, 4, 1))
        self.assertEqual((rounds['capped_segments'], rounds['idle_rounds'], rounds['by_live']), (1, 1, {'4': 1, '2': 1}))
        refused = gate.refused_rounds_report(text)
        self.assertEqual((refused['refused'], refused['complete_groups'], refused['accounted']), (1, 1, True))
        self.assertFalse(gate.refused_rounds_report(text.replace('2/2 FINISHED', '9/9 FINISHED'))['accounted'])
        before = gate.before_points(text)
        self.assertEqual((before['points'], before['ops']), (4, ['engine', 'prefill', 'quad']))
        self.assertEqual(before['floor_gb'], 0.1, 'the engine build: 1.1 GB largest free less its 1.0 GB')
        single = [gate.before_points(line)['floor_point'] for line in text.split('\n') if 'MEMLEDGER' in line]
        self.assertEqual([(point['op'], point['estimate_gb'], point['margin_gb']) for point in single],
                         [('prefill', 0.3, 1.2), ('prefill', 0.0, 0.4), ('quad', 0.7, 0.2), ('engine', 1.0, 0.1)])


class PathTests(unittest.TestCase):
    def test_packed_and_sequential_rounds_chain_positions_from_the_prompt(self):
        streams = [dict(request_id=request_id(0)), dict(request_id=request_id(1))]
        text = '\n'.join([
            '[SEQ-PUBLISH] request=%s rows=4 prefix=3 step_ms=1.00 cache=1->1' % engine_id(0),
            '[SEQ-PUBLISH] request=%s stages features=0.10' % engine_id(0),
            '[PACKED] request=%s segment=0 position=1003 prefix=16 emitted=16 predictions=[1] cap=16' % engine_id(0),
            '[PACKED] request=%s segment=1 position=2000 prefix=2 emitted=2 predictions=[4, 5]' % engine_id(1),
            '[SEQUENTIAL] request=%s position=1019 prefix=4' % engine_id(0),
            '[PACKED] request=%s segment=0 position=9999 prefix=1 emitted=1 predictions=[1]' % 'cmpl-nobody-0-x',
            '[SEQ-PUBLISH] request=%s rows=n/a prefix=n/a step_ms=1.00 cache=n/a->n/a' % engine_id(1),
            '[SEQ-PUBLISH] request=%s rows=4 prefix=4 step_ms=1.00 cache=1->1' % engine_id(1)])
        found = acceptance_report.path_records(text, streams, [1000, 2000])
        user0, user1 = found['users'][0], found['users'][1]
        self.assertEqual([(r['path'], r['position'], r['prefix']) for r in user0],
                         [('S', 1000, 3), ('P', 1003, 16), ('S', 1019, 4)])
        self.assertEqual(user0[1]['cap'], 16)
        self.assertEqual([(r['path'], r['position']) for r in user1], [('P', 2000), ('S', 2002), ('S', None)])
        self.assertEqual((found['position_checks'], found['position_mismatch_count']), (3, 0))
        self.assertEqual(found['unattributed'], ['cmpl-nobody-0-x'])
        encoded = acceptance_report.encode_paths(user0)
        self.assertEqual(encoded, 'S1000:3:-:4:- P1003:16:16:-:16 S1019:4:-:-:-')
        decoded = compare.decode_paths(encoded)
        self.assertEqual([(r['path'], r['position'], r['prefix'], r['emitted'], r['rows'], r['cap']) for r in decoded],
                         [('S', 1000, 3, None, 4, None), ('P', 1003, 16, 16, None, 16), ('S', 1019, 4, None, None, None)])
        wrong = acceptance_report.path_records(text, streams, [999, 2000])
        self.assertEqual(wrong['position_mismatches'][0], dict(user=0, round=1, chained=1002, logged=1003))

    def test_decode_steps_and_the_four_live_rate(self):
        rate = acceptance_report.live_rate(s2_log(rounds=10, emitted=6, round_ms=200.0, audit_ms=4.0), 4)
        self.assertEqual((rate['rounds'], rate['timed_rounds'], rate['tokens_per_user_per_round']), (10, 10, 6.0))
        self.assertEqual((rate['median_round_ms'], rate['median_audit_ms'], rate['net_median_round_ms']),
                         (204.0, 4.0, 200.0))
        self.assertEqual((rate['per_user_tok_s'], rate['net_per_user_tok_s']), (29.41, 30.0))
        self.assertIsNone(acceptance_report.live_rate(s2_log(packed=False), 4), 'sequential rounds are not packed')
        self.assertIsNone(acceptance_report.live_rate('no steps', 4))


def chunk_every(size):
    """Detail-stream chunks of `size` characters, one token each after a one-token seed chunk."""
    def split(text):
        chars = [1] + [size] * ((len(text) - 1) // size) + ([(len(text) - 1) % size] if (len(text) - 1) % size else [])
        return chars, [1] * len(chars)
    return split


def paths_report(texts, encoded, profile='c2-packed', env=((AUDIT, '1'),), chunk=None):
    report = base.served_report(texts=texts, profile=profile, configuration=configuration(profile, env),
                                users=len(texts), finish='stop', completion=300, max_tokens=4096)
    for index, stream in enumerate(report['streams']):
        stream['request_id'] = request_id(index)
        stream['chunk_chars'], stream['chunk_tokens'] = (chunk or chunk_every(10))(stream['text'])
    report['s2'] = dict(problems=[], paths=dict(users=dict((str(user), dict(encoded=text))
                                                           for user, text in enumerate(encoded))))
    return report


# One user's records: a packed arm's 16-row rounds, a solo arm's 4-row ones (enough of both to commit token 10).
PACKED_PATHS = 'P131072:16:16:-:16 P131088:16:16:-:16'
SOLO_PATHS = 'S131072:4:-:4:- S131076:4:-:4:- S131080:4:-:4:- S131084:4:-:4:-'


class PolicyTests(unittest.TestCase):
    def texts(self, diverge=None, at=100):
        texts = [s['text'][:400] for s in V235['streams']]
        if diverge is not None:
            texts[diverge] = texts[diverge][:at] + '#' + texts[diverge][at + 1:]
        return texts

    def test_an_s2_claim_never_makes_a_divergence_not_comparable(self):
        on = base.served_report(texts=self.texts(1), configuration=dict(configuration('c2-gate'), **{EXTENT: '1'}))
        off = base.served_report(texts=self.texts(), configuration=configuration('c2-gate'))
        self.assertEqual(compare.arithmetic_diff(on, off), {})
        self.assertEqual(compare.policy_pass(on, off)['users'][1]['verdict'], 'DIVERGED')
        for knob in (CAPTURE, FORCE):
            self.assertEqual(compare.arithmetic_diff(dict(on, qwen_configuration=dict(on['qwen_configuration'],
                                                                                       **{knob: '8'})), off), {})
        other = dict(on, qwen_configuration=dict(on['qwen_configuration'], QWEN_FAST_QUAD_DRAFT='0'))
        self.assertEqual(compare.policy_pass(other, off)['users'][1]['verdict'], 'NOT_COMPARABLE',
                         'any other arithmetic flag still is one')

    def test_a_divergence_record_names_t_star_its_rounds_positions_families_and_paths(self):
        # Seed chunk 1 char, then 10 chars per one-token chunk: character 100 is token 10.
        concurrent = paths_report(self.texts(2), ['P131072:5:5:-:16'] * 4)
        concurrent['s2']['paths']['users']['2'] = dict(encoded='P131072:4:4:-:16 P131076:4:4:-:16 P131080:4:4:-:16')
        solo = paths_report(self.texts(), ['S131072:4:-:4:-'] * 4)
        solo['s2']['paths']['users']['2'] = dict(encoded='S131072:3:-:4:- S131075:3:-:4:- S131078:3:-:4:- '
                                                       'S131081:3:-:4:-')
        record = compare.divergence_record(concurrent, solo, 2)
        self.assertEqual((record['character'], record['t_star'], record['t_star_chunk']), (100, [10, 11], 10))
        self.assertEqual(record['concurrent'], dict(round=2, path='P', position=131080, extent=131328, rows=None, cap=16,
                                                    prefix=4))
        self.assertEqual((record['solo']['round'], record['solo']['path'], record['solo']['position']), (3, 'S', 131081))
        self.assertTrue(record['cross_path'])
        self.assertIsNone(record['margin'])
        prefill = compare.divergence_record(paths_report(self.texts(2, at=0), ['P1:1:1:-:-'] * 4), solo, 2)
        self.assertEqual(prefill['concurrent'], dict(round='prefill', path='prefill'))

    def test_the_strict_policy_fails_a_reproduced_cross_path_divergence_and_writes_its_record(self):
        concurrent = paths_report(self.texts(2), [PACKED_PATHS] * 4)
        solo = paths_report(self.texts(), [SOLO_PATHS] * 4)
        result = compare.exactness_policy(concurrent, solo, (concurrent, solo))
        self.assertEqual(result['verdict'], 'FAIL')
        self.assertEqual(result['relaxation'], 'strict')
        self.assertEqual([(r['user'], r['verdict'], r['cross_path']) for r in result['records']], [(2, 'DIVERGED', True)])
        self.assertIn('record user 2 (DIVERGED): t*=[10, 11]', compare.render_policy(result))
        plain = compare.exactness_policy(base.served_report(texts=self.texts(2)), base.served_report(texts=self.texts()))
        self.assertNotIn('records', plain, 'a report without path records keeps today\'s result exactly')

    def test_dc_i_only_relaxes_a_cross_path_divergence(self):
        concurrent = paths_report(self.texts(2), [PACKED_PATHS] * 4)
        cross = paths_report(self.texts(), [SOLO_PATHS] * 4)
        relaxed = compare.exactness_policy(concurrent, cross, (concurrent, cross), relaxation='dc-i')
        self.assertEqual((relaxed['verdict'], relaxed['users'][2]['verdict']), ('NOT_COMPARABLE', 'NOT_COMPARABLE'))
        self.assertIn('D-c(i)', relaxed['users'][2]['reason'])
        self.assertEqual(relaxed['relaxation'], 'dc-i')
        same = paths_report(self.texts(), [PACKED_PATHS] * 4)
        self.assertEqual(compare.exactness_policy(concurrent, same, (concurrent, same), relaxation='dc-i')['verdict'],
                         'FAIL', 'a same-path divergence FAILS under any policy')
        unknown = dict(cross)
        unknown['s2'] = dict(problems=[])
        self.assertEqual(compare.exactness_policy(concurrent, unknown, (concurrent, unknown), relaxation='dc-i')['verdict'],
                         'FAIL', 'unknown paths fail closed')
        with self.assertRaises(ValueError):
            compare.exactness_policy(concurrent, cross, relaxation='relaxed')

    def test_same_path_users(self):
        first = paths_report(self.texts(), ['P1:1:1:-:-', 'S1:1:-:-:-', 'P5:2:2:-:-', 'P9:1:1:-:-'])
        second = paths_report(self.texts(), ['P1:1:1:-:-', 'P1:1:1:-:-', 'P5:2:2:-:-', 'S9:1:-:-:-'])
        self.assertEqual(compare.same_path_users(first, second), [0, 2])
        self.assertIsNone(compare.same_path_users(first, base.served_report()))

    def test_the_cli_relaxation_needs_the_decision(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, 'a.json')
            with open(path, 'w', encoding='utf-8') as handle:
                json.dump(base.served_report(), handle)
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                compare.main([path, path, '--policy', '--relaxation', 'dc-i'])
            with redirect_stdout(io.StringIO()) as out:
                self.assertEqual(compare.main([path, path, '--policy', '--relaxation', 'dc-i', '--decision', 'doc#1']), 0)
            self.assertIn('POLICY PASS', out.getvalue())


class JobTests(unittest.TestCase):
    def read(self, **values):
        return job.read_job(dict(C2_IMAGE_TAG='s2-k64j', **values), base.CHECKOUT_PROFILES['profiles'].keys())

    def test_the_s2_plans_and_keys_parse(self):
        outputs = self.read(C2_GATE_PLAN='warm control forced-cap', C2_GATE_PAIRS='1', C2_GATE_FAMILIES='16640, 4352',
                            C2_GATE_JIT='judge', C2_GATE_AUDITS='all')
        self.assertEqual(outputs['gate_plan'], 'warm,control,forced-cap')
        self.assertEqual((outputs['gate_pairs'], outputs['gate_families'], outputs['gate_jit'], outputs['gate_audits']),
                         ('1', '16640,4352', 'judge', 'all'))
        self.assertEqual((outputs['gate_policy'], outputs['gate_policy_decision']), ('', ''))
        relaxed = self.read(C2_GATE_POLICY='dc-i', C2_GATE_POLICY_DECISION='docs/decision-dc.md#i')
        self.assertEqual(relaxed['gate_policy'], 'dc-i')
        for plan in job.S2_GATE_PLANS:
            self.assertEqual(self.read(C2_GATE_PLAN=plan)['gate_plan'], plan)
        defaults = self.read()
        self.assertEqual([defaults[key] for key in ('gate_pairs', 'gate_families', 'gate_jit', 'gate_policy',
                                                    'gate_policy_decision', 'gate_audits')], [''] * 6)

    def test_what_the_s2_keys_refuse(self):
        for values in (dict(C2_GATE_PAIRS='0'), dict(C2_GATE_PAIRS='5'), dict(C2_GATE_FAMILIES='4000'),
                       dict(C2_GATE_FAMILIES='16896'), dict(C2_GATE_FAMILIES='4352,4352'), dict(C2_GATE_JIT='maybe'),
                       dict(C2_GATE_POLICY='relaxed'), dict(C2_GATE_POLICY='dc-i'),
                       dict(C2_GATE_POLICY='dc-i', C2_GATE_POLICY_DECISION='a b'), dict(C2_GATE_AUDITS='none'),
                       dict(C2_GATE_PLAN='control soak')):
            with self.subTest(values=values), self.assertRaises(job.JobError):
                self.read(**values)

    def test_the_gate_step_passes_every_s2_key(self):
        text = base.WorkflowTests.text()
        step = text[text.index('- name: Run the gate'):text.index('- name: Replay')]
        for option, output in (('--pairs', 'gate_pairs'), ('--families', 'gate_families'), ('--jit', 'gate_jit'),
                               ('--policy', 'gate_policy'), ('--policy-decision', 'gate_policy_decision'),
                               ('--audits', 'gate_audits')):
            name = output.upper()
            self.assertIn('%s: ${{ steps.job.outputs.%s }}' % (name, output), step)
            self.assertIn('${%s:+%s "$%s"}' % (name, option, name), step)


class Driver(object):
    """The driver with FakeDocker, S2 profiles and an optional kernel-cache counter."""

    def __init__(self, test):
        self.test = test

    def run(self, argv, reports, server_logs, cache=None, profiles=None):
        profiles = profiles or PROFILES
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, 'profiles.json')
            with open(path, 'w', encoding='utf-8') as handle:
                json.dump(profiles, handle)
            results = os.path.join(directory, 'results')
            docker, lines = base.FakeDocker(reports, server_logs), []
            code = driver.main(argv + ['--image', 'zot/img:s2', '--results', results, '--profiles', path],
                               execute=docker, devices=['/dev/tenstorrent/3', '/dev/tenstorrent/1'], log=lines.append,
                               containers=lambda: [], corpus=lambda: dict(V235['real_text']['corpus']),
                               cache_entries=cache)
            summary = None
            summary_path = os.path.join(results, 'c2-gate-summary.json')
            if os.path.isfile(summary_path):
                with open(summary_path, encoding='utf-8') as handle:
                    summary = json.load(handle)
            records = {}
            for name in os.listdir(results) if os.path.isdir(results) else []:
                if name.endswith('-divergence-records.json'):
                    with open(os.path.join(results, name), encoding='utf-8') as handle:
                        records[name] = json.load(handle)
        return code, summary, docker.calls, lines, records


class ControlDriverTests(unittest.TestCase):
    def control(self, on_log=None, off_log=None, on_env=((AUDIT, '1'),), off_texts=None, pairs='2', cache=None,
                on_logs=None):
        """The control plan with every flag-off arm on `off_log` and every flag-on arm on `on_log`, or on its own
        log in `on_logs` (arm name -> server log; its report is built from the same log)."""
        on_log = on_log or s2_log()
        off_log = off_log or s2_log(on=False, round_ms=172.0)
        logs, factories = {}, {}
        for index in range(1, int(pairs) + 1):
            off_name, on_name = 'control-off-%d' % index, 'control-on-%d' % index
            mine = (on_logs or {}).get(on_name, on_log)
            logs[off_name], logs[on_name] = off_log, mine
            factories[off_name] = lambda n, t=off_texts: s2_arm_report(off_log, 'c2-gate', texts=t)
            factories[on_name] = lambda n, text=mine: s2_arm_report(text, 'c2-packed-gate', env=on_env)
        return Driver(self).run(['--profile', 'c2-packed', '--plan', 'control', '--pairs', pairs], factories, logs,
                                cache=cache)

    def test_an_abab_that_reproduces_v235_within_the_tolerance_passes(self):
        code, summary, calls, lines, _ = self.control()
        result = summary['results']['control']
        self.assertEqual((code, result['verdict']), (0, 'PASS'), result.get('lines'))
        self.assertEqual([c['arm'] for c in calls], ['control-off-1', 'control-on-1', 'control-off-2', 'control-on-2'])
        on_argv = calls[1]['arguments']
        self.assertIn('QWEN_C2_PROFILE=c2-packed-gate', on_argv)
        self.assertIn('%s=1' % AUDIT, on_argv)
        self.assertIn('QWEN_C2_PROFILE=c2-gate', calls[0]['arguments'])
        self.assertNotIn('%s=1' % AUDIT, calls[0]['arguments'])
        self.assertEqual([(p['pair'], p['ratio']) for p in result['pairs']], [(1, 0.9884), (2, 0.9884)])
        self.assertEqual(summary['policy'], 'strict')
        self.assertEqual(summary['s2_exit_blockers'], [])

    def test_a_slower_flag_on_round_fails(self):
        code, summary, _, _, _ = self.control(on_log=s2_log(round_ms=180.0))
        result = summary['results']['control']
        self.assertEqual((code, result['verdict']), (1, 'FAIL'))
        self.assertIn('past 1.02', ' '.join(result['s2_problems']))

    def test_the_tolerance_must_hold_in_every_pair(self):
        code, summary, _, _, _ = self.control(on_logs={'control-on-2': s2_log(round_ms=180.0)})
        result = summary['results']['control']
        self.assertEqual(result['verdict'], 'FAIL')
        self.assertEqual([p['pair'] for p in result['pairs']], [1, 2])
        self.assertTrue(all('pair 2' in problem for problem in result['s2_problems']), result['s2_problems'])

    def test_a_pair_without_a_timed_four_live_round_is_not_exercised(self):
        code, summary, _, _, _ = self.control(off_log=s2_log(on=False, packed=False), pairs='1')
        result = summary['results']['control']
        self.assertEqual(result['verdict'], 'NOT_EXERCISED')
        self.assertIn('flag-off', ' '.join(result['shortfalls']))

    def test_an_audit_mismatch_or_a_missing_audit_line_fails(self):
        for log_text in (s2_log(mismatch=True), s2_log(drop_audit=2)):
            code, summary, _, _, _ = self.control(on_log=log_text, pairs='1')
            self.assertEqual(summary['results']['control']['verdict'], 'FAIL')
            on = summary['results']['control']['arms']['control-on-1']
            self.assertTrue(any('EXTENT_AUDIT' in p for p in on['platform_problems']), on['platform_problems'])

    def test_a_marker_leak_into_the_flag_off_arm_fails(self):
        code, summary, _, _, _ = self.control(off_log=s2_log(on=True), pairs='1')
        off = summary['results']['control']['arms']['control-off-1']
        self.assertEqual(off['verdict'], 'FAIL')
        self.assertTrue(any('not the profile' in p for p in off['platform_problems']))

    def test_a_knob_that_never_reached_the_container_fails(self):
        code, summary, _, _, _ = self.control(on_env=(), pairs='1')
        on = summary['results']['control']['arms']['control-on-1']
        self.assertTrue(any('never reached the container' in p for p in on['platform_problems']))

    def test_a_judged_arm_that_compiled_fails_and_the_growth_is_recorded(self):
        counts = iter(range(100, 200, 3))
        code, summary, _, lines, _ = self.control(pairs='1', cache=lambda: next(counts))
        self.assertEqual(summary['results']['control']['verdict'], 'FAIL')
        self.assertEqual(summary['arms']['control-off-1']['kernel_cache'], dict(before=100, after=103, added=3,
                                                                                 judged=True))
        self.assertEqual(summary['arms']['control-on-1']['kernel_cache'], dict(before=106, after=109, added=3,
                                                                                judged=True))
        self.assertTrue(any('kernel cache 106 -> 109 entries (+3, judged)' in line for line in lines))
        on = summary['results']['control']['arms']['control-on-1']
        self.assertTrue(any('the kernel cache grew by 3 entries' in p for p in on['platform_problems']))
        flat = lambda: 100
        self.assertEqual(self.control(pairs='1', cache=flat)[1]['results']['control']['verdict'], 'PASS')

    def test_forced_cap(self):
        capped = s2_log(cap=8)
        uncapped = s2_log(cap=16)
        def run(on_log):
            return Driver(self).run(['--profile', 'c2-packed', '--plan', 'forced-cap'], {
                'forced-cap-off': lambda n: s2_arm_report(uncapped, 'c2-packed-gate', env=((AUDIT, '1'),)),
                'forced-cap-on': lambda n: s2_arm_report(on_log, 'c2-packed-gate', env=((AUDIT, '1'), (FORCE, '8')))},
                {'forced-cap-off': uncapped, 'forced-cap-on': on_log})[1]['results']['forced-cap']
        self.assertEqual(run(capped)['verdict'], 'PASS')
        self.assertEqual(run(uncapped)['verdict'], 'FAIL', 'a cap of 16 under the forced cap of 8')
        self.assertEqual(run(s2_log())['verdict'], 'NOT_EXERCISED', 'no cap= in the [PACKED] lines')


class BelowDriverTests(unittest.TestCase):
    def below(self, off_log=None, on_log=None, on_texts=None):
        prompt = 16384
        knob = ((CAPTURE, str(prompt)),)
        override = ['INFO [PINDIAG] packed capture position override=%d (gate only)' % prompt]
        off_log = off_log or s2_log(on=False, positions=[prompt] * 4, extra=override)
        on_log = on_log or s2_log(positions=[prompt] * 4, extra=override)
        lengths = [prompt] * 4
        return Driver(self).run(['--profile', 'c2-packed', '--plan', 'control-below', '--families', '16640',
                                 '--pairs', '1'], {
            'below-16640-off-1': lambda n: s2_arm_report(off_log, 'c2-gate', env=knob, lengths=lengths, max_tokens=224,
                                                         completion=224),
            'below-16640-on-1': lambda n: s2_arm_report(on_log, 'c2-packed-gate', env=knob + ((AUDIT, '1'),),
                                                        lengths=lengths, max_tokens=224, completion=224, texts=on_texts)},
            {'below-16640-off-1': off_log, 'below-16640-on-1': on_log})[1]['results']['control-below']

    def test_identical_arms_with_the_flag_off_block_packed_in_the_family_pass(self):
        result = self.below()
        self.assertEqual(result['verdict'], 'PASS', result['lines'])
        self.assertEqual(result['pairs']['16640-1']['flag_off_packed_rounds'], 48)

    def test_a_flag_off_block_that_never_packed_decides_nothing(self):
        prompt = 16384
        override = ['INFO [PINDIAG] packed capture position override=%d (gate only)' % prompt]
        result = self.below(off_log=s2_log(on=False, packed=False, positions=[prompt] * 4, extra=override))
        self.assertEqual(result['verdict'], 'NO_DECISION')

    def test_a_divergence_across_the_arms_fails(self):
        texts = [s['text'] for s in V235['streams']]
        texts[3] = texts[3][:50] + '#' + texts[3][51:]
        self.assertEqual(self.below(on_texts=texts)['verdict'], 'FAIL')


class ServingDriverTests(unittest.TestCase):
    LENGTHS = [1536, 20000, 60000, 120000]

    def g4(self, plan, concurrent_log, solo_log, concurrent_texts=None, extra_reports=None, extra_logs=None,
           max_tokens=4096, argv=(), cache=None, lengths=None):
        lengths = lengths or self.LENGTHS
        env = ((AUDIT, '1'),)
        reports = {
            '%s-concurrent' % plan: lambda n: s2_arm_report(concurrent_log, 'c2-packed', env=env, lengths=lengths,
                                                           max_tokens=max_tokens, finish='stop', completion=300,
                                                           texts=concurrent_texts, chunk=chunk_every(10)),
            '%s-solo' % plan: lambda n: s2_arm_report(solo_log, 'c2-packed', env=env, lengths=lengths,
                                                     max_tokens=max_tokens, finish='stop', completion=300,
                                                     sequential=4, chunk=chunk_every(10))}
        reports.update(extra_reports or {})
        logs = {'%s-concurrent' % plan: concurrent_log, '%s-solo' % plan: solo_log}
        logs.update(extra_logs or {})
        for name in list(reports):
            reports[name + '-rerun'] = reports[name]
            logs[name + '-rerun'] = logs.get(name)
        return Driver(self).run(['--profile', 'c2-packed', '--plan', plan, '--max-tokens', str(max_tokens)] + list(argv),
                                reports, logs, cache=cache)

    def mixed_log(self, **kwargs):
        kwargs.setdefault('positions', self.LENGTHS)
        kwargs.setdefault('families', [[1792, 20224, 60160, 120064]])
        return s2_log(**kwargs)

    def solo_log(self):
        return s2_log(packed=False, positions=self.LENGTHS)

    def test_mixed_passes_with_two_families_a_clean_audit_and_the_rate(self):
        code, summary, calls, lines, _ = self.g4('mixed', self.mixed_log(), self.solo_log())
        result = summary['results']['mixed']
        self.assertEqual((code, result['verdict']), (0, 'PASS'), result['lines'])
        self.assertEqual(result['facts']['live4']['net_per_user_tok_s'], 29.41)
        self.assertEqual([c['arm'] for c in calls], ['mixed-concurrent', 'mixed-solo'])

    def test_mixed_without_a_two_family_round_is_not_exercised(self):
        same = s2_log(positions=[1536] * 4)
        code, summary, _, _, _ = self.g4('mixed', same, self.solo_log())
        self.assertEqual(summary['results']['mixed']['verdict'], 'NOT_EXERCISED')

    def test_a_slow_four_live_round_fails_the_rate(self):
        code, summary, _, _, _ = self.g4('mixed', self.mixed_log(round_ms=500.0), self.solo_log())
        result = summary['results']['mixed']
        self.assertEqual(result['verdict'], 'FAIL')
        self.assertIn('not above general', ' '.join(result['s2_problems']))

    def test_a_reproduced_divergence_fails_with_its_record_written(self):
        texts = [s['text'] for s in V235['streams']]
        texts[1] = texts[1][:120] + '#' + texts[1][121:]
        code, summary, calls, lines, records = self.g4('mixed', self.mixed_log(), self.solo_log(),
                                                      concurrent_texts=texts)
        result = summary['results']['mixed']
        self.assertEqual(result['verdict'], 'FAIL')
        self.assertEqual([c['arm'] for c in calls], ['mixed-concurrent', 'mixed-solo', 'mixed-concurrent-rerun',
                                                     'mixed-solo-rerun'])
        record, = records['mixed-divergence-records.json']
        self.assertEqual((record['user'], record['verdict'], record['t_star']), (1, 'DIVERGED', [12, 13]))
        self.assertEqual((record['concurrent']['path'], record['solo']['path'], record['cross_path']), ('P', 'S', True))
        self.assertEqual(summary['s2_exit_blockers'], [], 'a FAIL is no blocker: it fails outright')

    def test_under_dc_i_a_cross_path_divergence_blocks_exit_with_its_record(self):
        texts = [s['text'] for s in V235['streams']]
        texts[1] = texts[1][:120] + '#' + texts[1][121:]
        code, summary, _, lines, records = self.g4('mixed', self.mixed_log(), self.solo_log(), concurrent_texts=texts,
                                                  argv=['--policy', 'dc-i', '--policy-decision', 'user-dc-2026-09-27'])
        self.assertEqual(summary['results']['mixed']['verdict'], 'NOT_COMPARABLE')
        blocker, = summary['s2_exit_blockers']
        self.assertEqual((blocker['plan'], blocker['user'], blocker['verdict']), ('mixed', 1, 'NOT_COMPARABLE'))
        self.assertTrue(any('S2 EXIT BLOCKED' in line for line in lines))
        refused = Driver(self).run(['--profile', 'c2-packed', '--plan', 'mixed', '--policy', 'dc-i'], {}, {})
        self.assertEqual(refused[0], 2, 'dc-i without the decision is refused')

    def test_short_needs_the_one_bucket_ladder_no_hold_and_the_floor(self):
        lengths = [60, 255, 2047, 120000]
        floor = ('[MEMLEDGER] phase=prefill point=before prompt=120000 chip0 allocated=1.000GB free=3.000GB '
                 'largest_free=%s total=34.000GB known=1.000GB residual=0.000GB')
        good = s2_log(positions=[128, 255, 2047, 120000], extra=[floor % '1500.0MB'])
        solo = s2_log(packed=False, positions=lengths, extra=[floor % '1500.0MB'])
        self.assertEqual(self.g4('short', good, solo, lengths=lengths)[1]['results']['short']['verdict'], 'PASS')
        held = s2_log(positions=[128, 255, 2047, 120000],
                      extra=[floor % '1500.0MB', 'INFO [PINDIAG] dram hold prompt=120000 largest_free=900MB need=1300MB'])
        result = self.g4('short', held, solo, lengths=lengths)[1]['results']['short']
        self.assertEqual(result['verdict'], 'FAIL')
        self.assertIn('dram hold', ' '.join(result['s2_problems']))
        ladder = s2_log(positions=[128, 255, 2047, 120000], extra=[floor % '1500.0MB'], ladder='[256, 512, 1024, 2048]')
        self.assertEqual(self.g4('short', ladder, solo, lengths=lengths)[1]['results']['short']['verdict'], 'FAIL')
        low = s2_log(positions=[128, 255, 2047, 120000], extra=[floor % '400.0MB'])
        self.assertEqual(self.g4('short', low, solo, lengths=lengths)[1]['results']['short']['verdict'], 'FAIL')

    def test_boundaries_need_cap_events_and_crossings(self):
        lengths = list(driver.BOUNDARY_LENGTHS)
        capped = lambda position: min(16, gate.extent_of(position) - position)
        crossing = s2_log(positions=lengths, rounds=70, emitted=67, cap=capped)
        solo = s2_log(packed=False, positions=lengths)
        result = self.g4('boundaries', crossing, solo, max_tokens=8192, lengths=lengths)[1]['results']['boundaries']
        self.assertEqual(result['verdict'], 'PASS', result['lines'])
        few = s2_log(positions=lengths, rounds=10, emitted=5, cap=capped)
        result = self.g4('boundaries', few, solo, max_tokens=8192, lengths=lengths)[1]['results']['boundaries']
        self.assertEqual(result['verdict'], 'NOT_EXERCISED')

    def test_staggered_needs_padded_rounds_and_a_clean_probe(self):
        padded = s2_log(positions=self.LENGTHS, extra=[
            'INFO [PINDIAG] packed extent round round=99 live=2 families=[1792,20224] idle=[0,32] capped=[]',
            'INFO [EXTENT-AUDIT] round=99 segments=4 words_ok=4 cur_pos_ok=4 mask_ok=1 tables_ok=1 ms=2.00'])
        probe_log = padded + 'INFO [PINDIAG] padded probe round=3 live=0,1,2 exact=1 trace_ms=5.0 idle_carry_intact=1\n'
        probe = lambda n: dict(s2_arm_report(probe_log, 'c2-packed', env=((AUDIT, '1'), ('QWEN_FAST_PADDED_PROBE', '1')),
                                             lengths=self.LENGTHS, max_tokens=4096, finish='stop', completion=300),
                               flag_markers=dict(missing=[], padded_probe=[dict(round=3, exact='1')]))
        code, summary, calls, _, _ = self.g4('staggered', padded, self.solo_log(),
                                            extra_reports={'staggered-probe': probe},
                                            extra_logs={'staggered-probe': probe_log})
        self.assertEqual([c['arm'] for c in calls], ['staggered-concurrent', 'staggered-solo', 'staggered-probe'])
        self.assertEqual(summary['results']['staggered']['verdict'], 'PASS', summary['results']['staggered']['lines'])
        self.assertIn('QWEN_FAST_PADDED_PROBE=1', calls[2]['arguments'])
        code, summary, _, _, _ = self.g4('staggered', s2_log(positions=self.LENGTHS), self.solo_log(),
                                        extra_reports={'staggered-probe': probe}, extra_logs={'staggered-probe': probe_log})
        self.assertEqual(summary['results']['staggered']['verdict'], 'NOT_EXERCISED', 'no round at two or three live')


class LifecycleMemoryChurnTests(unittest.TestCase):
    def run_plan(self, plan, factories, logs):
        return Driver(self).run(['--profile', 'c2-packed', '--plan', plan], factories, logs)[1]['results'][plan]

    def lifecycle_reports(self, log_text, narrowed=True):
        text = log_text + ('WARNING [PINDIAG] packed survivor narrowed request=%s rows=16->4 reason=x\n' % engine_id(1)
                           if narrowed else '')
        events = {'lifecycle-drops': {0: 'drop', 1: 'drop', 2: 'drop', 3: 'drop', 4: 'cut256', 5: 'ignore'},
                  'lifecycle-edges': {0: 'cancel', 1: 'drop', 2: 'drop', 3: 'drop', 4: 'one'}, 'lifecycle-solo': {}}

        def arm(name):
            def build(n):
                report = base.lifecycle_report(events[name], profile='c2-packed')
                report['qwen_configuration'] = configuration('c2-packed', ((AUDIT, '1'),))
                gate.add_s2_report(report, report['qwen_configuration'], text, report['streams'], None)
                return report
            return build
        return dict((name, arm(name)) for name in events), dict((name, text) for name in events)

    def test_lifecycle_on_c2_packed_needs_a_narrowed_survivor(self):
        factories, logs = self.lifecycle_reports(s2_log(rounds=2))
        passed = self.run_plan('lifecycle', factories, logs)
        self.assertEqual(passed['verdict'], 'PASS', passed['lines'])
        self.assertEqual(passed['facts'], dict(narrowed=2))
        factories, logs = self.lifecycle_reports(s2_log(rounds=2), narrowed=False)
        result = self.run_plan('lifecycle', factories, logs)
        self.assertIn('D1 was not exercised', ' '.join(result['shortfalls']))
        self.assertEqual(result['verdict'], 'NOT_EXERCISED')

    def test_an_unaccounted_refused_round_fails_the_lifecycle(self):
        text = s2_log(rounds=2) + 'WARNING [PACKED] A round the block cannot serve (r) holds tickets no request engine\n'
        factories, logs = self.lifecycle_reports(text)
        result = self.run_plan('lifecycle', factories, logs)
        self.assertEqual(result['verdict'], 'FAIL')
        self.assertIn('W5b', ' '.join(result['s2_problems']))

    def test_memory_on_c2_packed_fails_on_a_hold(self):
        floor = ('[MEMLEDGER] phase=prefill point=before prompt=123136 chip0 allocated=1.000GB free=3.000GB '
                 'largest_free=1500.0MB total=34.000GB known=1.000GB residual=0.000GB')
        def arm(log_text, users=4):
            return lambda n: s2_arm_report(log_text, 'c2-packed', env=((AUDIT, '1'),), users=users,
                                           lengths=[123136] * users, max_tokens=8192, completion=8192)
        good = s2_log(positions=[123136] * 4, extra=[floor])
        short_log = s2_log(positions=[128] * 4, extra=[floor])   # packed only from 128 on (the admission floor)
        clean = self.run_plan('memory', {'memory-concurrent': arm(good), 'memory-short': lambda n: s2_arm_report(
            short_log, 'c2-packed', env=((AUDIT, '1'),), lengths=[60] * 4, max_tokens=16384, completion=16384)},
            {'memory-concurrent': good, 'memory-short': short_log})
        self.assertEqual(clean['verdict'], 'PASS', clean['lines'])
        held = good + 'INFO [PINDIAG] dram hold prompt=123136 largest_free=700MB need=1300MB\n'
        result = self.run_plan('memory', {'memory-concurrent': arm(held), 'memory-short': lambda n: s2_arm_report(
            short_log, 'c2-packed', env=((AUDIT, '1'),), lengths=[60] * 4, max_tokens=16384, completion=16384)},
            {'memory-concurrent': held, 'memory-short': short_log})
        self.assertEqual(result['verdict'], 'FAIL')

    def churn(self, log_text):
        lengths = list(driver.CHURN_LENGTHS)
        def report(n):
            one = s2_arm_report(log_text, 'c2-packed', env=((AUDIT, '1'),), users=len(lengths), lengths=lengths,
                                max_tokens=2304, finish='length')
            for stream, budget in zip(one['streams'], driver.CHURN_MAX_TOKENS):
                stream['completion_tokens'] = budget
            one['alive'] = True
            return one
        return self.run_plan('churn', {'churn': report}, {'churn': log_text})

    def test_churn_needs_dead_traces_released(self):
        floor = ('[MEMLEDGER] phase=prefill point=before prompt=110000 chip0 allocated=1.000GB free=3.000GB '
                 'largest_free=1500.0MB total=34.000GB known=1.000GB residual=0.000GB')
        quads = ['INFO [QUAD-DRAFT] round=4 built=1 ms=60.0']
        released = ['INFO [PACKED-PROPOSE] released quad=1 pairs=2'] * 5
        passed = self.churn(s2_log(extra=[floor] + quads + released))
        self.assertEqual(passed['verdict'], 'PASS', passed['lines'])
        self.assertEqual(passed['facts']['replacements'], 5)
        self.assertEqual(self.churn(s2_log(extra=[floor] + quads))['verdict'], 'FAIL')
        self.assertEqual(self.churn(s2_log(extra=[floor]))['verdict'], 'NOT_EXERCISED')

    def test_warm_records_the_cache_and_never_judges_it(self):
        counts = iter(range(0, 1000, 50))
        logs, factories = {}, {}
        for arm in driver.plan_arms('warm', 'c2', PROFILES):
            options = base.parse_harness(arm[1])
            lengths = options.prompt_lengths or [options.prompt_tokens] * options.users
            streams = options.sequential_users or options.users
            log_text = s2_log(users=min(streams, 4), extra=['INFO [PINDIAG] packed capture position override=%s'
                                                            % dict(arm.env).get(CAPTURE, '0')])
            logs[arm[0]] = log_text

            def build(n, log_text=log_text, arm=arm, lengths=lengths, streams=streams, options=options):
                want = driver.asked(arm[1])
                report = s2_arm_report(log_text, arm.profile, env=arm.env, users=streams, lengths=lengths,
                                       max_tokens=options.max_tokens, completion=options.max_tokens)
                for stream, budget in zip(report['streams'], want['budgets']):
                    stream['completion_tokens'] = budget
                return report
            factories[arm[0]] = build
        code, summary, _, lines, _ = Driver(self).run(['--profile', 'c2', '--plan', 'warm'], factories, logs,
                                                      cache=lambda: next(counts))
        result = summary['results']['warm']
        self.assertEqual(result['verdict'], 'PASS', result['lines'])
        self.assertEqual(result['arms']['warm-solo']['kernel_cache'], dict(before=0, after=50, added=50, judged=False))

    def test_permuted_counts_same_path_users(self):
        lengths = ServingDriverTests.LENGTHS
        log_text = s2_log(positions=lengths)
        report = lambda n: s2_arm_report(log_text, 'c2-packed', env=((AUDIT, '1'),), lengths=lengths, max_tokens=4096,
                                         finish='stop', completion=300)
        result = self.run_plan('permuted', {'permuted-forward': report, 'permuted-reverse': report},
                               {'permuted-forward': log_text, 'permuted-reverse': log_text})
        self.assertEqual((result['verdict'], result['same_path_users']), ('PASS', [0, 1, 2, 3]))
        solo_log = s2_log(packed=False, positions=lengths)
        other = lambda n: s2_arm_report(solo_log, 'c2-packed', env=((AUDIT, '1'),), lengths=lengths, max_tokens=4096,
                                        finish='stop', completion=300)
        result = self.run_plan('permuted', {'permuted-forward': report, 'permuted-reverse': other},
                               {'permuted-forward': log_text, 'permuted-reverse': solo_log})
        self.assertEqual(result['verdict'], 'NOT_EXERCISED')


if __name__ == '__main__':
    unittest.main()
