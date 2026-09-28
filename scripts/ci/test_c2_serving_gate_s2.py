"""S2 gate tooling (s2-design.md W11), held on CPU: the S2 plans' arms, the harness's S2 record, the path
records and four-live rate, the strict policy with its divergence records and the D-c(i) switch, the S2
verdicts, and the whole driver with injected docker, server logs and kernel-cache counter.

Nothing here opens a device or runs docker. The S2 profiles are W8's; until they land in the checkout this
module builds them as the design defines them (s2_profiles), and uses the checkout's once they exist.

Every server-log line a fixture writes is rendered from its producer's own format (the W* constants below,
copied verbatim with their sources), never invented: the W11 review found four formats the gate read one way
and W3/W6 write another. ProducerContractTests renders each producer's real lines and parses them with the
harness whenever the checkout carries that producer (W3 on s2/w3-block-step, W6 on s2/w6-memory), and holds
the copies here equal to the producers' constants."""

import datetime
import inspect
import io
import json
import math
import os
import random
import re
import statistics
import sys
import tempfile
import types
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

# The producers' line formats, verbatim (ProducerContractTests holds each against its producer).
W3_ROUND_FORMAT = '%s round=%d live=%d families=[%s] idle=[%s] capped=[%s]'   # packed_verifier.note_extent_round
W3_AUDIT_FORMAT = ('%s round=%d segments=%d words_ok=%d cur_pos_ok=%d mask_ok=%d tables_ok=%d rotated=%d '
                   'ms=%.2f')                                                  # packed_verifier.audit_extent
W3_MISMATCH_FORMAT = '%s round=%d at=%s'                                       # packed_verifier.audit_extent
W3_AUDIT_MARKER, W3_MISMATCH_MARKER = '[EXTENT-AUDIT]', '[EXTENT-AUDIT] MISMATCH'
# serving_prefill_admission (W6b): the hold's lines and the wrapper's decision line (once per distinct state).
W6_HOLD_LINE = '[PINDIAG] dram hold prompt={} largest_free={} need={} request={} decodes={}'
W6_RELEASED_HOLD_LINE = '[PINDIAG] dram hold released prompt={} largest_free={} need={} request={}'
W6_LIFTED_LINE = ('[PINDIAG] dram hold lifted prompt={} largest_free={} need={} request={}: no decode is left to '
                  'free DRAM, so the prompt is admitted and the bridge backstop decides')
W6_UNAVAILABLE_LINE = '[PINDIAG] dram hold unavailable request={}: {} (not held)'
W6_DEFERRED_LINE = ('[PINDIAG] dram admission deferred one step prompt={} largest_free={} need={} request={} '
                    'finished={}: the reading still counts their engines, which this step detaches first')
W6_CARRIED_LINE = ('[PINDIAG] dram admission carried finished={} past the discarded prefill pass into the '
                   'decode-only step')
W6_BUCKETS_BUILT = '[PINDIAG] proposal buckets built request='    # serving_request_factory.PROPOSAL_BUCKETS_BUILT
DECISION_LINE = '[PINDIAG] one fresh prefill per step: partials={} decodes={} gate_held={} allowed={} hidden={}'
# memory_ledger (W6d): MemoryLedger._before's line, cut at LINE_BUDGET onto '[MEMLEDGER] ...' lines by .log.
W6_BEFORE_FORMAT = '[MEMLEDGER] before op=%s chip%d largest_free=%s free=%s estimate=%s margin=%s floor=%s %s'
W6_BEFORE_UNREAD_FORMAT = '[MEMLEDGER] before op=%s dram unavailable (%s)'
LEDGER_LINE_BUDGET, LEDGER_CONTINUATION = 180, '[MEMLEDGER] ...'
W6_RELEASED_LINE = '[PACKED-PROPOSE] released quad={quad} pairs={pairs}'     # dflash_packed_proposal_coordinator
QUAD_ROUND_FORMAT = '[QUAD-DRAFT] round={round} built={built} ms={ms}'        # quad_draft.ROUND_LINE
PHASE_EXECUTE_FORMAT = '[PHASE] execute total={} new={} cached={} spec={} finished={} preempted={}'   # worker hook
# publication_warm (B6): the extent block's attach-time eager publication warm, as warmed_line renders the M3 plan.
WARM_LINE = ('[PINDIAG] eager publication warmed: 71 shapes in 1830.4 ms packed=0,16,32,48:1-16 '
             'sequential=1:1,2:1-2,4:1-4 merge_release=1 fused_steady_state=1 program_cache=412->655')
WARM_SUMMARY = dict(shapes=71, ms=1830.4, packed='0,16,32,48:1-16', sequential='1:1,2:1-2,4:1-4', merge_release=True,
                    fused_steady_state=True, program_cache=(412, 655))
WARM_SKIPPED_LINE = '[PINDIAG] eager publication warm skipped reason=no-collectives'
# serving_request_factory's any-request engine line: the ladder is a tuple through loguru's {} (W6a).
SINGLE_LADDER, SHORT_LADDER = '{}'.format((2048,)), '{}'.format((256, 512, 1024, 2048))
ENGINE_PEAK = 1000 * 10 ** 6       # serving_prefill_admission.engine_build_peak(): 800 MB resident + 200 MB transient
QUAD_ESTIMATE = 450 * 2 ** 20      # quad_draft.QUAD_CAPTURE_BYTES_EST (0.450 GiB), W6d's quad estimate
# G3's flag phases (c2_serving_gate.FLAG_PHASES): each round's lines as their producers write them.
PHASE_END_FORMAT = '[PHASE] {} {} end {:.1f} ms'                                  # serving_worker_hook.phase
VERIFY_SPLIT_FORMAT = ('[PACKED-PHASE] round=%d users=%d bind_ms=%.2f input_ms=%.2f trace_ms=%.2f '
                       'sync_ms=%.2f readback_ms=%.2f')                             # packed_verifier.verify
FENCES_FORMAT = ('[PACKED-FENCES] round={round} fence={fence} validated={validated} replay_ms={replay_ms} '
                 'commit_sync_ms={commit_sync_ms} path={path} prestage_ms={prestage_ms} diff_ms={diff_ms} '
                 'write_ms={write_ms}')                                             # serving_packed_step.FENCES_LINE
WINDOW_MARKER = '[PACKED-PRESTAGE-WINDOW]'                                         # verify_prestage.WINDOW_MARKER
WINDOW_FORMAT = '%s round=%d buffers=%d ms=%.2f live=%s'                           # verify_prestage.BlockPrestage.prestage
GDN_MARKER = '[PACKED-GDN-AFTER-PAIRS]'                                            # early_draft.GDN_MARKER
FLUSH_FORMAT = '%s round=%d commits=%d site=%s enqueue_ms=%.2f segments=%s'        # packed_verifier...flush_commits
# One four-live round as v61's flag-off arms logged it (medians, ms): F = 124.2 verify + 4 x 0.3 commits + 0.03 flush +
# 7.0 window.
BASE_PHASES = dict(trace=119.6, sync=0.6, readback=0.8, diff=1.6, write=1.0, bookkeeping=0.6, commit=0.3, flush=0.03,
                   window=7.0)
BASE_F = 132.43
ROUND_MS = 165
# The flag-on timing arm of a control run that PASSes: two pairs bound c within 2% only when it reads below about
# -1.4% of a 172 ms round (the arms' own offsets, CONTROL_ARM_SIGMA_MS, add ~3.35% to the bound), so this fixture's
# flag makes F 6.5 ms quicker (its window pre-stage 0.5 ms, not 7.0).
QUICKER_ON = dict(window=0.5)


def mb(value):
    """memory_ledger._mb and serving_prefill_admission._megabytes."""
    return '%.1fMB' % (value / 1e6)


def gb(value):
    return '%.3fGB' % (value / 1e9)


def round_line(number, families, idle=(), capped=()):
    """W3's line for a packed round: live segment i at family families[i] (segment:E), idle segments, and
    (segment, limit) caps."""
    return 'INFO ' + W3_ROUND_FORMAT % (gate.S2_ROUND_MARKER, number, len(families),
                                        ','.join('%d:%d' % pair for pair in enumerate(families)),
                                        ','.join(str(segment) for segment in idle),
                                        ','.join('%d:%d' % pair for pair in capped))


def audit_line(number, segments=4, words_ok=None, cur_pos_ok=None, rotated=0, ms=2.0):
    return 'INFO ' + W3_AUDIT_FORMAT % (W3_AUDIT_MARKER, number, segments, segments if words_ok is None else words_ok,
                                        segments if cur_pos_ok is None else cur_pos_ok, 1, 1, rotated, ms)


def mismatch_line(number, at='word:2'):
    return 'WARNING ' + W3_MISMATCH_FORMAT % (W3_MISMATCH_MARKER, number, at)


def hold_line(prompt, largest, need, request, decodes):
    return 'INFO ' + W6_HOLD_LINE.format(prompt, mb(largest), mb(need), request, decodes)


def decision_line(decodes, allowed=0, hidden=True, partials=0, held=False):
    return 'INFO ' + DECISION_LINE.format(partials, decodes, held, allowed, hidden)


def deferred_line(prompt, largest, need, request, finished):
    return 'INFO ' + W6_DEFERRED_LINE.format(prompt, mb(largest), mb(need), request, list(finished))


def buckets_line(user, contexts=(2048,)):
    """W6a's executed-path line: the buckets read from the built capture (a tuple through loguru's {})."""
    return 'INFO ' + W6_BUCKETS_BUILT + '{} contexts={}'.format(request_id(user), tuple(contexts))


def ledger_lines(message):
    """memory_ledger.MemoryLedger.log: the message cut at LINE_BUDGET, the rest on continuation lines."""
    head, rest = message[:LEDGER_LINE_BUDGET], message[LEDGER_LINE_BUDGET:]
    lines = ['INFO ' + head]
    width = LEDGER_LINE_BUDGET - len(LEDGER_CONTINUATION)
    while rest:
        lines.append('INFO ' + LEDGER_CONTINUATION + rest[:width])
        rest = rest[width:]
    return lines


def before_lines(op, largest, estimate, point=None, chips=(0, 1), free=3e9, trace=None):
    """W6d's before point, one line (or more) per chip: `largest` free bytes (one value, or one per chip), the
    operation's estimate, the running floor as the margin itself; `trace` (used, largest free) bytes, or None for
    'trace=unavailable'."""
    label = op if point is None else '%s point=%s' % (op, point)
    lines = []
    for index, chip in enumerate(chips):
        free_block = largest[index] if isinstance(largest, (list, tuple)) else largest
        margin = free_block - estimate
        text = 'trace=unavailable' if trace is None else 'trace_used=%s trace_largest_free=%s' % (mb(trace[0]),
                                                                                                 mb(trace[1]))
        lines += ledger_lines(W6_BEFORE_FORMAT % (label, chip, mb(free_block), gb(free), mb(estimate), mb(margin),
                                                  mb(margin), text))
    return lines


def prefill_point(prompt, largest_mb, chip=0):
    """The ledger's phase point ahead of a prefill (serving_runtime: record('prefill', point='before prompt=N'))."""
    return ('INFO [MEMLEDGER] phase=prefill point=before prompt=%d chip%d allocated=1.000GB free=3.000GB '
            'largest_free=%.1fMB total=34.000GB known=1.000GB residual=0.000GB' % (prompt, chip, largest_mb))


def engine_points(users=4, largest=1500e6):
    """W6d's before point ahead of each user's engine build, healthy unless `largest` says otherwise."""
    return [line for user in range(users) for line in before_lines('engine', largest, ENGINE_PEAK,
                                                                   point='req=%s' % engine_id(user)[-12:])]


def released_line(quad, pairs=()):
    return 'INFO ' + W6_RELEASED_LINE.format(quad=quad, pairs=[list(pair) for pair in pairs])


def quad_line(number, built=0, ms=1.5):
    return 'INFO ' + QUAD_ROUND_FORMAT.format(round=number, built=built, ms=ms)


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
    # To the millisecond loguru prints, rounded: a clock summed from float round times never loses one to truncation.
    return (BASE_TIME + datetime.timedelta(seconds=round(seconds, 3))).strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]


def execute_line(seconds, live, finished=()):
    """serving_worker_hook's step line: finished= is the step's sorted finished request ids."""
    return '%s | INFO     | serving_worker_hook:execute_model:272 - ' % stamp(seconds) + PHASE_EXECUTE_FORMAT.format(
        16 * live, 0, live, 15 * live, sorted(finished), [])


def request_id(user):
    return 'cmpl-%032x' % (user + 1)


def engine_id(user):
    return request_id(user) + '-0-abcd1234'


def flag_phase_lines(number, users, values, window=True, drop=()):
    """The producers' flag-phase lines of one packed round (block round `number`): `values` is BASE_PHASES with the
    round's changes (bookkeeping is the verify's rest, commit each user's, flush the round's deferred-commit flush),
    `window` False for a round whose next step is not packed (nothing is pre-staged), `drop` the lines left out
    ('verify', 'split', 'fences', 'commit', 'window', 'flush')."""
    ids = ','.join(engine_id(user) for user in range(users))
    verify = sum(values[name] for name in ('trace', 'sync', 'readback', 'diff', 'write', 'bookkeeping'))
    lines = []
    if 'split' not in drop:
        lines.append('INFO ' + VERIFY_SPLIT_FORMAT % (number, users, 0.0, values['diff'] + values['write'] + 0.1,
                                                      values['trace'], values['sync'], values['readback']))
    if 'verify' not in drop:
        lines.append('INFO ' + PHASE_END_FORMAT.format('packed_verify', ids, verify))
    if 'commit' not in drop:
        lines += ['INFO ' + PHASE_END_FORMAT.format('packed_commit', engine_id(user), values['commit'])
                  for user in range(users)]
    if 'fences' not in drop:
        lines.append('INFO ' + FENCES_FORMAT.format(round=number, fence='replay', validated=1, replay_ms='0.06',
                                                    commit_sync_ms='0.01', path='diff', prestage_ms='6.00',
                                                    diff_ms='%.2f' % values['diff'], write_ms='%.2f' % values['write']))
    if window and 'window' not in drop:
        lines.append('INFO ' + WINDOW_FORMAT % (WINDOW_MARKER, number + 1, 148, values['window'], users))
    if 'flush' not in drop:
        lines.append('INFO ' + FLUSH_FORMAT % (GDN_MARKER, number, users, 'window', values['flush'],
                                               ','.join('%d:5' % user for user in range(users))))
    return lines


def quicker_on_log(**changes):
    """An unaudited flag-on timing arm whose flag phases read QUICKER_ON (a control run that PASSes)."""
    return s2_log(audit=False, phases=lambda k: (QUICKER_ON, ()), **changes)


def s2_log(on=True, users=4, rounds=12, positions=None, emitted=5, round_ms=170.0, audit_ms=2.0, audit=True,
           mismatch=False, drop_audit=0, words_ok=None, cap=None, admission=True, packed=True, extra=(), families=None,
           ladder=SINGLE_LADDER, sequential_rows=4, buckets=(2048,), warm=True, phases=None):
    """A server log of an S2 arm: the C2-any lines, the S2 attach lines when `on`, and `rounds` decode steps -
    packed (four [PACKED] lines, a packed extent round line and an audit line each) or, `packed` False, one
    [SEQ-PUBLISH] step line per user - on one timestamped clock, `round_ms` apart (a number, or per round a function
    of its index). `phases`: None (no flag-phase lines), True (BASE_PHASES every round) or a function of the round's
    index returning (changes to BASE_PHASES, lines to drop): the flag-phase lines of every packed round
    (flag_phase_lines; the last round's next step is not packed, so it pre-stages nothing)."""
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
        if warm:
            lines.append('INFO ' + WARM_LINE)
        if buckets is not None:
            lines += [buckets_line(user, buckets) for user in range(users)]
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
                lines.append(round_line(index + 1, fams))
                if audit and index >= drop_audit:
                    lines.append(audit_line(index + 1, users, words_ok=words_ok, rotated=index % users, ms=audit_ms))
                if mismatch and index == 3:
                    lines.append(mismatch_line(4, 'word:2,cur_pos:2'))
            if phases is not None:
                changes, drop = ({}, ()) if phases is True else phases(index)
                lines += flag_phase_lines(index + 1, users, dict(BASE_PHASES, **changes), window=index < rounds - 1,
                                          drop=drop)
            position = [p + emitted for p in position]
        else:
            for user in range(users):
                lines.append('INFO [SEQ-PUBLISH] request=%s rows=%d prefix=3 step_ms=40.00 cache=5->5'
                             % (engine_id(user), sequential_rows))
                lines.append('INFO [SEQ-PUBLISH] request=%s stages features=0.10' % engine_id(user))
            position = [p + 3 for p in position]
        whole = round_ms(index) if callable(round_ms) else round_ms
        at += (whole + (audit_ms if on and audit and packed else 0.0)) / 1000.0
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

    def test_the_kernel_cache_is_found_on_the_hub_and_counted(self):
        self.assertEqual(driver.host_cache_dir('/experiment-cache/kernels-qwen-abc-pf-def', hub='/h'),
                         os.path.join('/h', '.qwen-c2', 'kernels-qwen-abc-pf-def'))
        self.assertEqual(driver.host_cache_dir('/models/cache', hub='/h'), os.path.join('/h', 'cache'))
        self.assertIsNone(driver.host_cache_dir('/root/.cache/tt-metal-cache', hub='/h'), 'a tmpfs dies with the arm')
        self.assertIsNone(driver.host_cache_dir(None))
        with tempfile.TemporaryDirectory() as directory:
            os.makedirs(os.path.join(directory, 'kernels', 'sdpa', '1234'))
            for name in ('a.o', 'b.elf'):
                with open(os.path.join(directory, 'kernels', 'sdpa', '1234', name), 'w') as handle:
                    handle.write('x')
            self.assertEqual(driver.count_entries(directory), 2)
        self.assertIsNone(driver.count_entries(None))

    def test_whose_kernel_cache_growth_is_judged(self):
        runner = driver.Runner('img', 'c2', '/r', '/c', [], profiles=PROFILES)
        self.assertFalse(runner.judges('matrix', 'c2'), 'S1 on an S1 profile: recorded only (auto)')
        self.assertTrue(runner.judges('matrix', 'c2-packed'))
        self.assertTrue(runner.judges('control', 'c2-gate'), 'every arm of an S2 plan, flag-off ones too')
        self.assertFalse(runner.judges('warm', 'c2-packed-gate'))
        self.assertFalse(runner.judges('control', 'c2-gate', judged=False))
        runner.jit = 'judge'
        self.assertTrue(runner.judges('bringup', 'exact'), 'M2: the exact bring-up adds nothing to the cache')
        self.assertFalse(runner.judges('warm-off', 'c2-gate'))
        runner.jit = 'record'
        self.assertFalse(runner.judges('control', 'c2-packed-gate'))
        self.assertIsNone(runner.relaxation('c2-packed'), 'strict unless asked')
        runner.policy, runner.decision = 'dc-i', 'user-decision'
        self.assertEqual((runner.relaxation('c2-packed'), runner.relaxation('c2')), ('dc-i', None))

    def test_mixed_always_runs_the_three_other_audits(self):
        # M7: zero PRESTAGE/PAIR_MASK/FUSED_COMMIT audit mismatches - vacuous unless the audits run.
        for audits in (None, 'extent', 'all'):
            with self.subTest(audits=audits):
                mixed = driver.plan_arms('mixed', 'c2-packed', PROFILES, s2=dict(audits=audits))
                self.assertEqual([arm.env for arm in mixed], [driver.AUDIT_ENV + driver.G4_ALL_AUDITS] * 2)
        short = driver.plan_arms('short', 'c2-packed', PROFILES)
        self.assertEqual([arm.env for arm in short], [driver.AUDIT_ENV] * 2, 'the others only with --audits all')
        self.assertEqual(driver.plan_arms('short', 'c2-packed', PROFILES, s2=dict(audits='all'))[0].env,
                         driver.AUDIT_ENV + driver.G4_ALL_AUDITS)

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
        self.assertEqual([arm[0] for arm in arms], ['control-off-1', 'control-on-1', 'control-off-2', 'control-on-2',
                                                    'control-audit'])
        self.assertEqual([arm.profile for arm in arms], ['c2-gate', 'c2-packed-gate'] * 2 + ['c2-packed-gate'])
        # The pairs time the flag without its audit (runs 36270917139, 36272682217: the audit's cost reaches past
        # the ms its line reports); the audited arm, after them, judges exactness and the audit only.
        self.assertEqual([arm.env for arm in arms], [(), ()] * 2 + [driver.AUDIT_ENV])
        self.assertEqual([arm.role for arm in arms], ['off', 'on'] * 2 + ['audit'])
        self.assertEqual([arm.extra.get('pair') for arm in arms], [1, 1, 2, 2, None])
        self.assertEqual(arms[1][1], arms[4][1], 'the audited arm serves the timing arms\' requests')
        for arm in arms:
            options = self.parse(arm)
            self.assertEqual((options.users, options.prompt_tokens, options.max_tokens, options.stagger),
                             (4, 131072, 256, 0.25))
            self.assertEqual(options.expect_profile, arm.profile)
        self.assertEqual([arm[0] for arm in driver.plan_arms('control', 'c2', PROFILES, s2=dict(pairs=1))],
                         ['control-off-1', 'control-on-1', 'control-audit'])
        self.assertEqual(driver.worst_case_seconds(['control'], dict(control=arms)),
                         5 * (driver.ARM_SECONDS['control'] + driver.ARM_OVERHEAD_SECONDS))
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
        self.assertEqual([arm.profile for arm in warm_off], ['c2-gate'] * 3 + ['exact'])
        # M2 judges exact's bring-up for zero new entries: warm-off compiles exact at exactly M2's shape.
        exact = warm_off[-1]
        self.assertEqual((exact[0], exact.env, exact.judged), ('warm-off-exact-4x131072', (), False))
        self.assertEqual(exact[1], driver.v235_args(PROFILES, 'exact', 'warm-off'))
        m2, = driver.plan_arms('bringup', 'exact', PROFILES)
        self.assertEqual(self.parse(exact).prompt_tokens, self.parse(m2).prompt_tokens)
        self.assertEqual(self.parse(exact).max_tokens, self.parse(m2).max_tokens)
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
        notes = []
        churn, = driver.plan_arms('churn', 'c2-packed', PROFILES, notes=notes)
        options = self.parse(churn)
        self.assertGreaterEqual(options.users - options.alive_check, 8, 'M11: at least eight replacements')
        self.assertEqual((options.users, options.alive_check, notes), (12, 4, []))
        self.assertTrue(any(length < 2048 for length in options.prompt_lengths))
        self.assertTrue(all(100000 <= length <= 123136 for length in options.prompt_lengths if length >= 2048))
        self.assertEqual(options.events['max_tokens'], dict(enumerate(driver.CHURN_MAX_TOKENS)))
        driver.plan_arms('churn', 'c2-packed', PROFILES, lengths=[110000] * 9, notes=notes)
        self.assertIn('fewer than the 8 M11 asks for', ' '.join(notes))
        event, solo = driver.plan_arms('lifecycle-arrival', 'c2-packed', PROFILES)
        self.assertEqual(self.parse(event).events['drops'], {1: ('build', 0)})
        self.assertEqual(event.extra, dict(expect_live={1: 1}), 'user 0 decodes when user 1 is dropped')
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

    def test_the_eager_publication_warm_is_required_with_the_flag_and_leaks_without_it(self):
        report = self.s2(s2_log())
        self.assertTrue(report['markers']['warm'])
        self.assertEqual(report['publication_warm'], dict(
            warmed=[dict(shapes=71, ms=1830.4, packed='0,16,32,48:1-16', sequential='1:1,2:1-2,4:1-4',
                         program_cache='412->655')], skipped=[]))
        missing = self.s2(s2_log(warm=False))
        self.assertFalse(missing['markers']['warm'])
        self.assertEqual([problem for problem in missing['problems'] if gate.S2_WARM_MARKER in problem],
                         ['%s=1: no "%s" line: the eager publication was not warmed at attach (B6), so a refused '
                          'fused commit or a sequential step can compile mid-request' % (EXTENT, gate.S2_WARM_MARKER)])
        skipped = self.s2(s2_log(warm=False, extra=['INFO ' + WARM_SKIPPED_LINE]))
        self.assertIn('(INFO %s)' % WARM_SKIPPED_LINE, ' '.join(skipped['problems']))
        self.assertEqual(skipped['publication_warm']['skipped'], ['INFO ' + WARM_SKIPPED_LINE])
        for line in (WARM_LINE, WARM_SKIPPED_LINE):
            with self.subTest(line=line):
                leaked = self.s2(s2_log(on=False, extra=['INFO ' + line]), on=False)
                self.assertIn('carries S2 markers', ' '.join(leaked['problems']))
                self.assertTrue(gate.s2_relevant({}, 'INFO ' + line))

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
        lines = [
            round_line(1, [256, 2304, 16640, 131328], capped=[(2, 6)]),
            round_line(2, [512, 512], idle=[2, 3]),
            'WARNING [PACKED] A round the block cannot serve (x) holds tickets no request engine captured (y)',
            'WARNING [PINDIAG] packed refused round aborted 1/2 FINISHED_ABORTED via the request quarantine, engine kept: '
            'request=a',
            'WARNING [PINDIAG] packed refused round aborted 2/2 FINISHED_ABORTED via the request quarantine, engine kept: '
            'request=b',
            prefill_point(120000, 1500.0, chip=0), prefill_point(60, 400.0, chip=1)]
        lines += before_lines('quad', 900e6, QUAD_ESTIMATE, point='slots=0,1,2,3', chips=(0,))
        lines += before_lines('engine', 1100e6, ENGINE_PEAK, point='req=abc', chips=(0,))
        text = '\n'.join(lines)
        rounds = gate.extent_rounds(text)
        self.assertEqual((rounds['count'], rounds['max_families'], rounds['multi_family_rounds']), (2, 4, 1))
        self.assertEqual((rounds['capped_segments'], rounds['idle_rounds'], rounds['by_live']), (1, 1, {'4': 1, '2': 1}))
        self.assertEqual(rounds['families_seen'], [256, 512, 2304, 16640, 131328], 'the E after each segment:')
        refused = gate.refused_rounds_report(text)
        self.assertEqual((refused['refused'], refused['complete_groups'], refused['accounted']), (1, 1, True))
        self.assertFalse(gate.refused_rounds_report(text.replace('2/2 FINISHED', '9/9 FINISHED'))['accounted'])
        before = gate.before_points(text)
        self.assertEqual((before['points'], before['ops']), (4, ['engine', 'prefill', 'quad']))
        self.assertEqual(before['floor_gb'], 0.1, 'the engine build: 1.1 GB largest free less its 1.0 GB')
        single = [gate.before_points(line)['floor_point'] for line in text.split('\n') if 'MEMLEDGER' in line]
        self.assertEqual([(point['op'], point['estimate_gb'], point['margin_gb']) for point in single],
                         [('prefill', 0.3, 1.2), ('prefill', 0.0, 0.4), ('quad', 0.4719, 0.4281), ('engine', 1.0, 0.1)])


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
                                                    'gate_policy_decision', 'gate_audits', 'gate_control_pool')],
                         [''] * 7)
        pooled = self.read(C2_GATE_PLAN='control', C2_GATE_CONTROL_POOL='scripts/ci/references/c2-serving/g3-first.json')
        self.assertEqual(pooled['gate_control_pool'], 'scripts/ci/references/c2-serving/g3-first.json')

    def test_what_the_s2_keys_refuse(self):
        for values in (dict(C2_GATE_PAIRS='0'), dict(C2_GATE_PAIRS='5'), dict(C2_GATE_FAMILIES='4000'),
                       dict(C2_GATE_FAMILIES='16896'), dict(C2_GATE_FAMILIES='4352,4352'), dict(C2_GATE_JIT='maybe'),
                       dict(C2_GATE_POLICY='relaxed'), dict(C2_GATE_POLICY='dc-i'),
                       dict(C2_GATE_POLICY='dc-i', C2_GATE_POLICY_DECISION='a b'), dict(C2_GATE_AUDITS='none'),
                       dict(C2_GATE_PLAN='control soak'), dict(C2_GATE_CONTROL_POOL='g3-first.json'),
                       dict(C2_GATE_CONTROL_POOL='scripts/ci/references/c2-serving/../../x.json'),
                       dict(C2_GATE_CONTROL_POOL='/tmp/control-phase-rounds.json'),
                       dict(C2_GATE_CONTROL_POOL='scripts/ci/references/c2-serving/g3 first.json')):
            with self.subTest(values=values), self.assertRaises(job.JobError):
                self.read(**values)

    def test_the_gate_step_passes_every_s2_key(self):
        text = base.WorkflowTests.text()
        step = text[text.index('- name: Run the gate'):text.index('- name: Replay')]
        for option, output in (('--pairs', 'gate_pairs'), ('--families', 'gate_families'), ('--jit', 'gate_jit'),
                               ('--policy', 'gate_policy'), ('--policy-decision', 'gate_policy_decision'),
                               ('--audits', 'gate_audits'), ('--control-pool', 'gate_control_pool')):
            name = output.upper()
            self.assertIn('%s: ${{ steps.job.outputs.%s }}' % (name, output), step)
            self.assertIn('${%s:+%s "$%s"}' % (name, option, name), step)


def flat_cache():
    """A kernel cache no arm grows: every judged arm's growth is counted, and is zero."""
    return 100


class Driver(object):
    """The driver with FakeDocker, S2 profiles and a kernel-cache counter (flat_cache unless given; None: none)."""

    def __init__(self, test):
        self.test = test

    def run(self, argv, reports, server_logs, cache=flat_cache, profiles=None):
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
                if name.endswith('-divergence-records.json') or name == driver.CONTROL_PHASE_RECORD:
                    with open(os.path.join(results, name), encoding='utf-8') as handle:
                        records[name] = json.load(handle)
        return code, summary, docker.calls, lines, records


class ControlDriverTests(unittest.TestCase):
    def control(self, on_log=None, off_log=None, on_env=(), off_texts=None, pairs='2', cache=flat_cache,
                on_logs=None, audit_log=None, audit_env=((AUDIT, '1'),), off_logs=None, extra=()):
        """The control plan with every flag-off arm on `off_log`, every flag-on timing arm on `on_log` (unaudited),
        either on its own log in `off_logs` / `on_logs` (arm name -> server log; its report is built from the same
        log), and the audited arm on `audit_log`; `on_env` and `audit_env` are what reaches each report's harness,
        `extra` more gate arguments. The timing arms' logs carry the flag phases (BASE_PHASES; the flag-on arm's
        QUICKER_ON, a run that passes) unless given."""
        on_log = on_log or quicker_on_log()
        off_log = off_log or s2_log(on=False, round_ms=172.0, phases=True)
        audit_log = audit_log or s2_log()
        logs, factories = {}, {}
        for index in range(1, int(pairs) + 1):
            off_name, on_name = 'control-off-%d' % index, 'control-on-%d' % index
            mine = (on_logs or {}).get(on_name, on_log)
            theirs = (off_logs or {}).get(off_name, off_log)
            logs[off_name], logs[on_name] = theirs, mine
            factories[off_name] = lambda n, text=theirs, t=off_texts: s2_arm_report(text, 'c2-gate', texts=t)
            factories[on_name] = lambda n, text=mine: s2_arm_report(text, 'c2-packed-gate', env=on_env)
        logs[driver.CONTROL_AUDIT_ARM] = audit_log
        factories[driver.CONTROL_AUDIT_ARM] = lambda n: s2_arm_report(audit_log, 'c2-packed-gate', env=audit_env)
        return Driver(self).run(['--profile', 'c2-packed', '--plan', 'control', '--pairs', pairs] + list(extra),
                                factories, logs, cache=cache)

    def test_an_abab_that_reproduces_v235_within_the_tolerance_passes(self):
        code, summary, calls, lines, _ = self.control()
        result = summary['results']['control']
        self.assertEqual((code, result['verdict']), (0, 'PASS'), result.get('lines'))
        self.assertEqual([c['arm'] for c in calls], ['control-off-1', 'control-on-1', 'control-off-2', 'control-on-2',
                                                     'control-audit'])
        on_argv, audit_argv = calls[1]['arguments'], calls[4]['arguments']
        self.assertIn('QWEN_C2_PROFILE=c2-packed-gate', on_argv)
        self.assertNotIn('%s=1' % AUDIT, on_argv, 'the timing arm runs without the audit')
        self.assertIn('QWEN_C2_PROFILE=c2-packed-gate', audit_argv)
        self.assertIn('%s=1' % AUDIT, audit_argv)
        self.assertIn('QWEN_C2_PROFILE=c2-gate', calls[0]['arguments'])
        self.assertNotIn('%s=1' % AUDIT, calls[0]['arguments'])
        self.assertEqual([(p['pair'], p['on_ms'], p['off_ms'], p['ratio']) for p in result['pairs']],
                         [(1, 170.0, 172.0, 0.9884), (2, 170.0, 172.0, 0.9884)])
        self.assertEqual(result['audited'], dict(arm='control-audit', median_round_ms=172.0, median_audit_ms=2.0))
        self.assertIn('pair 1: flag-off 172.0 ms, flag-on 170.0 ms (unaudited): 0.9884 x (net, at most 1.05)',
                      result['lines'])
        self.assertEqual(result['arms']['control-audit']['verdict'], 'PASS')
        self.assertEqual(summary['policy'], 'strict')
        self.assertEqual(summary['s2_exit_blockers'], [])
        # The primary: every round of both pairs paired, F 6.5 ms quicker with the flag (QUICKER_ON) but in the last
        # round of each pair (nothing pre-staged on either side); the bound is c plus the arms' offsets at the floor.
        statistic = result['flag_phase']['statistic']
        self.assertEqual((statistic['verdict'], statistic['rounds'], statistic['pairs'], statistic['cost']),
                         ('PASS', 24, 2, round(-6.5 / 172.0, 6)))
        arms = driver.CONTROL_BOUND_Z * driver.CONTROL_ARM_SIGMA_MS / 172.0
        self.assertEqual((statistic['arm_sigma_ms'], statistic['arm_term']), (3.5, round(arms, 6)))
        self.assertTrue(statistic['cost'] + arms - 1e-6 <= statistic['upper'] <= 0.02, statistic)
        self.assertEqual((statistic['trimmed_low'], statistic['trimmed_high']), (2, 2))
        self.assertEqual(result['flag_phase']['arm_sigma'], dict(floor_ms=3.5, seen_ms=0.0, used_ms=3.5,
                                                                 reads_ms=[0.0, 0.0]))
        self.assertEqual(os.path.basename(result['phase_record']), driver.CONTROL_PHASE_RECORD)

    def test_a_slower_flag_on_round_fails_the_net(self):
        code, summary, _, _, _ = self.control(on_log=quicker_on_log(round_ms=185.0))
        result = summary['results']['control']
        self.assertEqual((code, result['verdict']), (1, 'FAIL'))
        self.assertIn('the net', ' '.join(result['s2_problems']))
        self.assertIn('past 1.05', ' '.join(result['s2_problems']))
        self.assertEqual(result['flag_phase']['statistic']['verdict'], 'PASS', 'the flag phases are the same')

    def test_the_net_is_1_05_of_the_unaudited_round(self):
        for round_ms, verdict in ((180.0, 'PASS'), (181.0, 'FAIL')):
            with self.subTest(round_ms=round_ms):
                result = self.control(on_log=quicker_on_log(round_ms=round_ms))[1]['results']['control']
                self.assertEqual(result['verdict'], verdict, result['lines'])
                self.assertEqual([p['ratio'] for p in result['pairs']], [round(round_ms / 172.0, 4)] * 2)

    def test_one_pair_or_three_is_never_a_pass(self):
        # The rule times ABAB, two pairs (CONTROL_PAIRS). Under the first coding --pairs 1 (C2_GATE_PAIRS=1, which the
        # job accepts for any plan) PASSed on one pair's rounds, half the paired set.
        for pairs in ('1', '3'):
            with self.subTest(pairs=pairs):
                code, summary, _, _, records = self.control(pairs=pairs)
                result = summary['results']['control']
                self.assertEqual((code, result['verdict']), (1, 'NOT_EXERCISED'), result['lines'])
                self.assertIn("the rule times 2 pairs (ABAB) and this run has %s: its paired set is not the rule's, "
                              "never a pass" % pairs, result['shortfalls'])
                self.assertFalse(result['flag_phase']['decided'])
                self.assertEqual(len(records[driver.CONTROL_PHASE_RECORD]['pairs']), int(pairs))
        # A cost past 2% still FAILs on whatever pairs were measured.
        slow = s2_log(audit=False, phases=lambda k: (dict(bookkeeping=6.0), ()))
        result = self.control(on_log=slow, pairs='1')[1]['results']['control']
        self.assertEqual(result['verdict'], 'FAIL')

    def test_the_audited_arm_is_never_timed(self):
        # The audit lengthens its rounds past the ms its line reports; control-audit judges exactness only - its flag
        # phases, however slow, are never read.
        slow = s2_log(round_ms=200.0, audit_ms=9.0, phases=lambda k: (dict(bookkeeping=40.0), ()))
        code, summary, _, _, records = self.control(audit_log=slow)
        result = summary['results']['control']
        self.assertEqual((code, result['verdict']), (0, 'PASS'), result['lines'])
        self.assertEqual([p['ratio'] for p in result['pairs']], [0.9884, 0.9884])
        self.assertEqual(result['audited'], dict(arm='control-audit', median_round_ms=209.0, median_audit_ms=9.0))
        statistic = result['flag_phase']['statistic']
        self.assertEqual((statistic['cost'], statistic['rounds']), (round(-6.5 / 172.0, 6), 24))
        self.assertEqual([pair['pair'] for pair in records[driver.CONTROL_PHASE_RECORD]['pairs']], [1, 2])

    def test_a_timing_arm_that_ran_the_audit_fails(self):
        # Its round is not the flag's: the pair is refused, whichever way the audit got there.
        for on_env in ((), ((AUDIT, '1'),)):
            with self.subTest(on_env=on_env):
                code, summary, _, _, _ = self.control(on_log=s2_log(), on_env=on_env, pairs='1')
                result = summary['results']['control']
                self.assertEqual((code, result['verdict']), (1, 'FAIL'))
                self.assertEqual(result['pairs'], [])
                self.assertIn('pair 1: the flag-on timing arm ran the extent audit (12 "[EXTENT-AUDIT]" lines)',
                              ' '.join(result['s2_problems']))

    def test_the_tolerance_must_hold_in_every_pair(self):
        code, summary, _, _, _ = self.control(on_logs={'control-on-2': quicker_on_log(round_ms=185.0)})
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
            code, summary, _, _, _ = self.control(audit_log=log_text, pairs='1')
            self.assertEqual(summary['results']['control']['verdict'], 'FAIL')
            audited = summary['results']['control']['arms']['control-audit']
            self.assertTrue(any('EXTENT_AUDIT' in p for p in audited['platform_problems']), audited['platform_problems'])
            self.assertEqual(summary['results']['control']['arms']['control-on-1']['verdict'], 'PASS')

    def test_a_marker_leak_into_the_flag_off_arm_fails(self):
        code, summary, _, _, _ = self.control(off_log=s2_log(on=True), pairs='1')
        off = summary['results']['control']['arms']['control-off-1']
        self.assertEqual(off['verdict'], 'FAIL')
        self.assertTrue(any('not the profile' in p for p in off['platform_problems']))

    def test_a_knob_that_never_reached_the_container_fails(self):
        code, summary, _, _, _ = self.control(audit_env=(), pairs='1')
        result = summary['results']['control']
        self.assertEqual(result['verdict'], 'FAIL')
        audited = result['arms']['control-audit']
        self.assertTrue(any('never reached the container' in p for p in audited['platform_problems']))
        self.assertIn('control-audit: the extent audit did not run', ' '.join(result['s2_problems']))

    def test_a_judged_arm_whose_cache_could_not_be_counted_is_not_exercised(self):
        # Kernel-cache growth unknown (no counter: the cache is not on the hub, or sudo find failed): never PASS.
        code, summary, _, lines, _ = self.control(pairs='1', cache=None)
        result = summary['results']['control']
        self.assertEqual((code, result['verdict']), (1, 'NOT_EXERCISED'), result['lines'])
        self.assertTrue(any('could not be counted (no counter' in s for s in result['shortfalls']), result['shortfalls'])
        failing = iter([100, None, 100, 100, 100, 100])
        result = self.control(pairs='1', cache=lambda: next(failing))[1]['results']['control']
        self.assertEqual(result['verdict'], 'NOT_EXERCISED')
        self.assertTrue(any('control-off-1' in s and 'a count failed' in s for s in result['shortfalls']))

    def test_a_flag_on_arm_without_a_packed_extent_round_fails(self):
        # G3 must not pass with every flag-on round sequential.
        code, summary, _, _, _ = self.control(on_log=s2_log(packed=False, audit=False), pairs='1')
        result = summary['results']['control']
        self.assertEqual(result['verdict'], 'FAIL')
        on = result['arms']['control-on-1']
        self.assertEqual(on['verdict'], 'FAIL')
        self.assertIn('problem: the flag-on arm served no packed extent round', on['lines'])
        self.assertEqual(result['arms']['control-off-1']['verdict'], 'PASS')
        result = self.control(audit_log=s2_log(packed=False), pairs='1')[1]['results']['control']
        self.assertEqual(result['verdict'], 'FAIL')
        self.assertIn('problem: the flag-on arm served no packed extent round', result['arms']['control-audit']['lines'])

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
        self.assertEqual(self.control(cache=flat)[1]['results']['control']['verdict'], 'PASS')

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

    def test_forced_cap_compares_the_capped_text_with_the_uncapped(self):
        texts = [s['text'] for s in V235['streams']]
        texts[2] = texts[2][:90] + '#' + texts[2][91:]
        capped = s2_log(cap=8)
        result = Driver(self).run(['--profile', 'c2-packed', '--plan', 'forced-cap'], {
            'forced-cap-off': lambda n: s2_arm_report(s2_log(cap=16), 'c2-packed-gate', env=((AUDIT, '1'),)),
            'forced-cap-on': lambda n: s2_arm_report(capped, 'c2-packed-gate', env=((AUDIT, '1'), (FORCE, '8')),
                                                     texts=texts)},
            {'forced-cap-off': s2_log(cap=16), 'forced-cap-on': capped})[1]['results']['forced-cap']
        self.assertEqual(result['verdict'], 'FAIL')
        self.assertTrue(any('user 2: the capped arm is DIVERGED against the uncapped' in p and 'Q5' in p
                            for p in result['s2_problems']), result['s2_problems'])


# G3's rule scenarios: a four-live round of ROUND_MS, BASE_F of it in the flag phases, of which the replay is device
# time and the rest (HOST_IN_PHASES) host time, like the round outside the phases (HOST_OUTSIDE).
HOST_IN_PHASES = BASE_F - BASE_PHASES['trace'] - BASE_PHASES['sync']
HOST_OUTSIDE = ROUND_MS - BASE_F
FEW_ROUNDS = (4, 11, 18, 25, 32)                  # 5 of 36


def rule_log(on, rounds=36, offset=0.0, cost=None, outside=None, spikes=None, seed=0, drop=None, emitted=5,
             f_offset=0.0, window_ms=None, flush=None):
    """A timing arm: `rounds` four-live rounds with `offset` (a fraction: the arm's host slowdown - its round that
    much longer, the host part of the flag phases taking its share), `f_offset` ms more of the arm's own host offset
    inside the flag phases every round (runs 36270917139-36276533585: up to 3.7 ms of an arm's 5), `cost(k)` ms inside
    the flag phases (the verify's rest), `window_ms` the window pre-stage's ms when not BASE_PHASES' (a flag that does
    less there), `flush(k)` the deferred-commit flush's ms when not BASE_PHASES', `outside(k)` ms in the round outside
    the phases, `spikes` {k: ms} of host stalls inside the verify, a small fixed jitter per round (`seed`), and
    `drop(k)` the round's flag-phase lines left out."""
    share = offset * ROUND_MS * HOST_IN_PHASES / (HOST_IN_PHASES + HOST_OUTSIDE)

    def inside(k):
        return (cost(k) if cost else 0.0) + (spikes or {}).get(k, 0.0) + f_offset

    def phases(k):
        jitter = (((k * 7 + seed) % 5) - 2) * 0.1
        changes = dict(bookkeeping=round(BASE_PHASES['bookkeeping'] + share + inside(k) + jitter, 1))
        if window_ms is not None:
            changes['window'] = window_ms
        if flush is not None:
            changes['flush'] = flush(k)
        return changes, drop(k) if drop else ()

    def whole(k):
        quicker = BASE_PHASES['window'] - window_ms if window_ms is not None and k < rounds - 1 else 0.0
        slower = flush(k) - BASE_PHASES['flush'] if flush is not None else 0.0
        return (ROUND_MS * (1 + offset) + inside(k) + (outside(k) if outside else 0.0) - quicker + slower +
                (((k * 3 + seed) % 5) - 2))
    return s2_log(on=on, audit=False, rounds=rounds, round_ms=whole, phases=phases, emitted=emitted)


def rule_arms(off=None, on=None, rerun=False, **arms):
    """The four timing arms of two pairs, each rule_log(**off) / rule_log(**on) with its own jitter (another for a
    `rerun`, so a re-run never serves its first run's rounds to the millisecond), or as given."""
    built, shift = {}, 2 if rerun else 0
    for index in (1, 2):
        built['control-off-%d' % index] = rule_log(False, seed=2 * index - 1 + shift, **(off or {}))
        built['control-on-%d' % index] = rule_log(True, seed=2 * index + shift, **(on or {}))
    built.update(('control-%s' % name.replace('_', '-'), text) for name, text in arms.items())
    return built


def wide_cost(k):
    """~1.8% of the round on average, spread +-8.75 ms (NOT_RESOLVED: c within 2%, its upper bound not)."""
    return 3.0 + 0.5 * (((k * 13) % 36) - 17.5)


QUICKER = dict(window_ms=0.5)       # a flag that does 6.5 ms less in the window pre-stage: c about -3.9%, a PASS


class FlagPhaseRuleTests(unittest.TestCase):
    """G3's pre-registered timing rule (c2_serving_gate's control): the flag phases paired round by round, the
    trimmed-mean primary and the untrimmed mean with their bounds (the rounds' bootstrap and the arms' offsets), the
    untrimmed guard, the 1.05 net, two pairs, the one pooled re-run bound to its first run. The tests named for a
    defect reproduce what a skeptic's review of the first coding (rule g3-flag-phase-2026-09-28) found."""

    def rule(self, arms, extra=()):
        """The control plan over `arms` (rule_arms); (code, result, records, lines)."""
        off = dict((name, text) for name, text in arms.items() if '-off-' in name)
        on = dict((name, text) for name, text in arms.items() if '-on-' in name)
        code, summary, _, lines, records = ControlDriverTests.control(self, off_logs=off, on_logs=on, extra=extra)
        return code, (summary or {}).get('results', {}).get('control'), records, lines

    def statistic(self, result):
        return result['flag_phase']['statistic']

    # -- the phases, round by round --------------------------------------------------------------------------------
    def test_each_phase_is_read_from_its_own_line(self):
        values = dict(trace=119.5, sync=0.7, readback=0.9, diff=1.7, write=1.1, bookkeeping=2.0, commit=0.4, flush=0.2,
                      window=6.5)
        rounds = driver.flag_phase_rounds(s2_log(on=False, rounds=3, phases=lambda k: (values, ())))
        self.assertEqual(len(rounds), 3)
        first, last = rounds[0], rounds[-1]
        self.assertEqual(first['phases'], dict(replay=120.2, staging=2.8, readback=0.9, bookkeeping=2.0, commit=1.8,
                                               window=6.5))
        self.assertEqual((first['flag_ms'], first['round_ms'], first['missing']), (134.2, 170.0, []))
        self.assertEqual(first['fingerprint'], [[0, 131072, 5], [1, 131072, 5], [2, 131072, 5], [3, 131072, 5]])
        self.assertEqual(rounds[1]['fingerprint'][0], [0, 131077, 5])
        # The last round's next step is not packed: nothing was pre-staged, and nothing is missing.
        self.assertEqual((last['phases']['window'], last['flag_ms'], last['missing']), (0.0, 127.7, []))

    def test_a_line_absent_or_repeated_leaves_its_round_unmeasured(self):
        for drop, where in ((('fences',), 1), (('verify',), 1), (('split',), 0), (('commit',), 2), (('window',), 0),
                            (('flush',), 1)):
            with self.subTest(drop=drop):
                rounds = driver.flag_phase_rounds(s2_log(on=False, rounds=3, phases=lambda k: (
                    {}, drop if k == where else ())))
                self.assertIsNone(rounds[where]['flag_ms'])
                self.assertTrue(rounds[where]['missing'])
                self.assertEqual([r['missing'] for i, r in enumerate(rounds) if i != where], [[], []])
        text = s2_log(on=False, rounds=3, phases=True)
        window = [line for line in text.split('\n') if WINDOW_MARKER in line][0]
        dropped = 'INFO %s round=2 dropped=epoch detail=-' % WINDOW_MARKER
        for changed in (text.replace(window, dropped), text.replace(window, window + '\n' + window)):
            self.assertTrue(driver.flag_phase_rounds(changed)[0]['missing'])
        cut = text.split('\n')
        last_step = max(i for i, line in enumerate(cut) if '[PHASE] execute total=' in line)
        rounds = driver.flag_phase_rounds('\n'.join(cut[:last_step] + cut[last_step + 1:]))
        self.assertEqual(rounds[-1]['missing'], ['its whole round (no timed decode step after it)'])

    def test_defect_the_deferred_commit_flush_is_part_of_the_commit_phase(self):
        # packed_verifier.flush_commits enqueues the round's commit traces inside replay_deadline('flush'), which only
        # the flag arms; the first coding's F left its enqueue_ms out (+0.05 ms a round in runs 36270917139-
        # 36276533585, every flag-on arm above every flag-off one).
        text = s2_log(on=False, rounds=3, phases=lambda k: (dict(flush=0.25), ()))
        rounds = driver.flag_phase_rounds(text)
        self.assertEqual([r['phases']['commit'] for r in rounds], [1.45, 1.45, 1.45])
        flush = [line for line in text.split('\n') if GDN_MARKER in line][1]
        self.assertIn('round=2 commits=4 site=window enqueue_ms=0.25 segments=0:5,1:5,2:5,3:5', flush)
        # Absent, repeated, dropped (a block that failed with commits held) or at an R1 backstop site: unmeasured.
        dropped = 'INFO %s round=2 commits=0 site=window enqueue_ms=0.00 segments=0:5 dropped=4 reason=block-failed' % (
            GDN_MARKER)
        for changed in (text.replace(flush, ''), text.replace(flush, flush + '\n' + flush), text.replace(flush, dropped),
                        text.replace(flush, flush.replace('site=window', 'site=verify'))):
            found = driver.flag_phase_rounds(changed)
            self.assertIsNone(found[1]['flag_ms'])
            self.assertTrue(any('PACKED-GDN-AFTER-PAIRS' in gap for gap in found[1]['missing']), found[1]['missing'])
        # site=end (the end of the early draft) is in the step that decided the commits: measured.
        self.assertEqual(driver.flag_phase_rounds(text.replace(flush, flush.replace('site=window', 'site=end')))[1][
            'phases']['commit'], 1.45)
        # A flag whose whole cost is in the flush (3% of the round) FAILs; the first coding read c 0 for it.
        code, result, _, _ = self.rule(rule_arms(on=dict(flush=lambda k: 5.0)))
        self.assertEqual((code, result['verdict']), (1, 'FAIL'), result['lines'])
        self.assertIn('the flag phases: c 3.0', ' '.join(result['s2_problems']))
        self.assertTrue(result['flag_phase']['phases']['commit']['trimmed_mean_ms'] > 4.9)

    def test_the_pre_registered_constants(self):
        self.assertEqual((driver.CONTROL_RULE, driver.CONTROL_PAIRS, driver.CONTROL_COST_LIMIT,
                          driver.CONTROL_TRIM_PERCENT, driver.CONTROL_BOUND_PERCENT, driver.CONTROL_BOUND_Z,
                          driver.CONTROL_BOOTSTRAP, driver.CONTROL_SEED, driver.CONTROL_ARM_SIGMA_MS,
                          driver.CONTROL_NET_TOLERANCE),
                         ('g3-flag-phase-2026-09-28-r2', 2, 0.02, 10, 95, 1.6449, 10000, 20260928, 3.5, 1.05))
        # CONTROL_BOUND_Z is the standard normal's CONTROL_BOUND_PERCENT percentile (statistics.NormalDist: 3.8+).
        self.assertEqual(round(statistics.NormalDist().inv_cdf(driver.CONTROL_BOUND_PERCENT / 100.0), 4),
                         driver.CONTROL_BOUND_Z)
        self.assertEqual(driver.FLAG_PHASES, ('replay', 'staging', 'readback', 'bookkeeping', 'commit', 'window'))

    def test_the_trim_the_bootstrap_and_its_bounds(self):
        self.assertEqual([driver.trimmed_mean(list(range(n)))[1] for n in (9, 10, 36, 72, 144)], [0, 1, 3, 7, 14])
        self.assertEqual(driver.trimmed_mean([100.0] + [1.0] * 8 + [-50.0]), (1.0, 1))
        rounds = [(wide_cost(k), 165.0 + k % 3) for k in range(36)]
        once, again = driver.flag_phase_cost(rounds, resamples=200), driver.flag_phase_cost(rounds, resamples=200)
        self.assertEqual(once, again, 'a fixed seed: the same bound every time')
        rng, draws, untrimmed = random.Random(driver.CONTROL_SEED), [], []
        for _ in range(200):
            sample = [rounds[int(rng.random() * 36)] for _ in range(36)]
            median = statistics.median([w for _, w in sample])
            draws.append(driver.trimmed_mean([d for d, _ in sample])[0] / median)
            untrimmed.append(sum(d for d, _ in sample) / 36 / median)
        self.assertEqual(once['round_upper'], round(sorted(draws)[189], 6), 'the 95th percentile: draw 190 of 200')
        self.assertEqual(once['untrimmed_round_upper'], round(sorted(untrimmed)[189], 6))
        self.assertEqual((once['trimmed_low'], once['trimmed_high'], len(once['trimmed_high_ms'])), (3, 3, 3))
        # Each bound adds the arms' offsets to the rounds': Z sds of sigma * sqrt(2 / pairs) over the median round.
        median = statistics.median([w for _, w in rounds])
        for pairs, sigma in ((2, 3.5), (4, 3.5), (2, 6.0)):
            with self.subTest(pairs=pairs, sigma=sigma):
                cost = driver.flag_phase_cost(rounds, pairs=pairs, arm_sigma_ms=sigma, resamples=200)
                arms = driver.CONTROL_BOUND_Z * sigma * math.sqrt(2.0 / pairs) / median
                self.assertEqual(cost['arm_term'], round(arms, 6))
                for estimate, alone, bound in (('cost', 'round_upper', 'upper'),
                                               ('untrimmed', 'untrimmed_round_upper', 'untrimmed_upper')):
                    self.assertAlmostEqual(cost[bound], cost[estimate] + math.sqrt(
                        (cost[alone] - cost[estimate]) ** 2 + arms ** 2), places=5)
                    self.assertTrue(cost[bound] >= cost[alone], 'never below the bootstrap\'s own percentile')
        # The arms' sd: the floor, raised by the A/A reads (each carries two arms' offsets), never lowered.
        self.assertEqual(driver.flag_phase_sigma([]), (3.5, None))
        self.assertEqual(driver.flag_phase_sigma([1.0, -1.0]), (3.5, math.sqrt(0.5)))
        self.assertEqual(driver.flag_phase_sigma([8.0, -8.0]), (math.sqrt(32.0), math.sqrt(32.0)))

    # -- the scenarios the rule was approved against ---------------------------------------------------------------
    def test_a_0_4_percent_flag_with_3_percent_arm_offsets_and_host_spikes_is_not_resolved(self):
        # The flag as measured (~0.4%) is never FAILed, and two pairs cannot bound it within 2% once the arms' own
        # offsets are in the bound; the first coding PASSed it on the rounds' bootstrap alone. Pooled with a like
        # re-run it is still NOT_RESOLVED: the user decides.
        flag = lambda k: 0.7                                                    # 0.42% of the round
        early, late = {3: 40.0, 20: 90.0}, {9: 90.0, 27: 40.0}                   # host stalls in every arm
        def arms(shift):
            return rule_arms(off_1=rule_log(False, spikes=early, seed=1 + shift),
                             on_1=rule_log(True, offset=0.03, cost=flag, spikes=late, seed=2 + shift),
                             off_2=rule_log(False, offset=0.03, spikes=late, seed=3 + shift),
                             on_2=rule_log(True, cost=flag, spikes=early, seed=4 + shift))
        code, result, records, lines = self.rule(arms(0))
        self.assertEqual((code, result['verdict']), (1, 'NOT_RESOLVED'), result['lines'])
        statistic = self.statistic(result)
        self.assertEqual((statistic['rounds'], statistic['pairs'], statistic['trimmed_low']), (72, 2, 7))
        self.assertTrue(0.003 < statistic['cost'] < 0.006, statistic)
        self.assertTrue(statistic['round_upper'] <= 0.02 < statistic['upper'], 'the first coding\'s PASS: %s' % statistic)
        self.assertEqual([p['ratio'] for p in result['pairs']], [1.0333, 0.9735])
        self.assertTrue(any('7 rounds trimmed from each tail' in line for line in result['lines']), result['lines'])
        record = records[driver.CONTROL_PHASE_RECORD]
        self.assertEqual((record['plan_verdict'], record['timing_verdict'], record['pooled']),
                         ('NOT_RESOLVED', 'NOT_RESOLVED', False))
        self.assertEqual([len(pair['rounds']) for pair in record['pairs']], [36, 36])
        code, result, _, _ = self.pooled(record, arms(2))
        self.assertEqual((code, result['verdict']), (1, 'NOT_RESOLVED'), result['lines'])
        self.assertEqual((self.statistic(result)['rounds'], self.statistic(result)['pairs']), (144, 4))
        self.assertTrue(result['flag_phase']['escalate'])

    def test_a_flag_that_saves_time_passes_despite_arm_offsets_and_host_spikes(self):
        early, late = {3: 40.0, 20: 90.0}, {9: 90.0, 27: 40.0}
        arms = rule_arms(off_1=rule_log(False, spikes=early, seed=1),
                         on_1=rule_log(True, offset=0.03, spikes=late, seed=2, **QUICKER),
                         off_2=rule_log(False, offset=0.03, spikes=late, seed=3),
                         on_2=rule_log(True, spikes=early, seed=4, **QUICKER))
        code, result, records, _ = self.rule(arms)
        self.assertEqual((code, result['verdict']), (0, 'PASS'), result['lines'])
        statistic = self.statistic(result)
        self.assertTrue(statistic['cost'] < -0.03 and statistic['upper'] <= 0.02, statistic)
        self.assertEqual(result['flag_phase']['arm_sigma']['used_ms'], 3.5, 'the A/A reads stay under the floor')
        self.assertEqual(records[driver.CONTROL_PHASE_RECORD]['plan_verdict'], 'PASS')

    def test_defect_a_cost_the_arms_offsets_hide_is_never_passed(self):
        # A 3% flag cost with both flag-off arms 3.5 ms slower in F (an arm's own host offset, which no resampling of
        # rounds sees): c reads ~0.9% with a tight round bound - the first coding's PASS. The arms' sd (the floor: the
        # A/A reads are 0 here, both flag-off arms alike) holds it at NOT_RESOLVED.
        code, result, _, _ = self.rule(rule_arms(off=dict(f_offset=3.5), on=dict(cost=lambda k: 5.0)))
        statistic = self.statistic(result)
        self.assertEqual((code, result['verdict']), (1, 'NOT_RESOLVED'), result['lines'])
        self.assertTrue(0.005 < statistic['cost'] < 0.012, statistic)
        self.assertTrue(statistic['round_upper'] <= 0.02 < statistic['upper'], statistic)
        self.assertEqual(result['flag_phase']['arm_sigma']['used_ms'], 3.5)

    def test_defect_the_runs_own_a_a_reads_raise_the_arms_offset(self):
        # Pair 2's arms both 12 ms slower in F: its D is pair 1's, but the A/A reads (+12 ms flag-off and flag-on) say
        # an arm's offset here is ~8.5 ms, not the floor's 3.5 - the flag that saves 6.5 ms is no longer bounded.
        arms = rule_arms(on=QUICKER, off_2=rule_log(False, seed=3, f_offset=12.0),
                         on_2=rule_log(True, seed=4, f_offset=12.0, **QUICKER))
        code, result, _, _ = self.rule(arms)
        self.assertEqual((code, result['verdict']), (1, 'NOT_RESOLVED'), result['lines'])
        sigma = result['flag_phase']['arm_sigma']
        self.assertTrue(8.0 < sigma['seen_ms'] == sigma['used_ms'] < 9.0, sigma)
        self.assertEqual(len(sigma['reads_ms']), 2)
        self.assertTrue(self.statistic(result)['cost'] < -0.03)

    def test_a_3_percent_uniform_cost_inside_the_phases_fails(self):
        code, result, _, _ = self.rule(rule_arms(on=dict(cost=lambda k: 5.0)))
        self.assertEqual((code, result['verdict']), (1, 'FAIL'))
        self.assertTrue(self.statistic(result)['cost'] > 0.02)
        self.assertIn('the flag phases: c 3.0', ' '.join(result['s2_problems']))

    def test_a_3_percent_cost_on_5_of_36_rounds_fails_on_the_untrimmed_mean(self):
        # The trim's known hole: a cost in few rounds is trimmed away. On 5 rounds of every pair the trimmed c is
        # 1.1% (its round bound past 2%); on 5 rounds of one pair it is ~0 (its round bound within 2%).
        both = lambda k: 35.6 if k in FEW_ROUNDS else 0.0                       # 3% of every pair's rounds
        one = lambda k: 71.3 if k in FEW_ROUNDS else 0.0                        # 3% of both pairs' rounds
        for arms, round_bound in ((rule_arms(on=dict(cost=both)), False), (rule_arms(on_1=rule_log(True, cost=one,
                                                                                                   seed=2)), True)):
            with self.subTest(round_bound=round_bound):
                code, result, _, _ = self.rule(arms)
                self.assertEqual((code, result['verdict']), (1, 'FAIL'))
                statistic = self.statistic(result)
                self.assertTrue(statistic['untrimmed'] > 0.02 and statistic['cost'] <= 0.02, statistic)
                self.assertEqual(statistic['round_upper'] <= 0.02, round_bound)
                self.assertIn('the untrimmed cost', ' '.join(result['s2_problems']))

    def test_defect_a_cost_on_the_flag_on_stall_rounds_is_never_passed(self):
        # The skeptic's attack on run 36276533585: the flag-off arms' stalls are larger, so the untrimmed mean reads
        # far below 0 and the guard is blind, while the flag's cost sits on the flag-on arms' own stall rounds, which
        # the trim drops. Here a flag that saves 6.5 ms a round but costs 1 s on each flag-on arm's one stall round (a
        # 0.5 s host stall) - 12.9% of the round in all - reads c -3.9%: the trimmed bound (with the arms' offsets)
        # PASSes it. The untrimmed mean's bound does not.
        stalls_on = {20: 500.0 + 1000.0}
        stalls_off = {10: 2000.0}
        arms = rule_arms(off=dict(spikes=stalls_off), on=dict(spikes=stalls_on, **QUICKER))
        code, result, _, _ = self.rule(arms)
        statistic = self.statistic(result)
        self.assertEqual((code, result['verdict']), (1, 'NOT_RESOLVED'), result['lines'])
        self.assertTrue(statistic['cost'] < -0.03 and statistic['upper'] <= 0.02, 'c alone: a PASS %s' % statistic)
        self.assertTrue(statistic['untrimmed'] < 0 and statistic['untrimmed_upper'] > 0.02, statistic)
        self.assertIn('the untrimmed cost\'s', statistic['why'])

    def test_a_6_percent_cost_outside_the_phases_fails_by_the_net(self):
        code, result, _, _ = self.rule(rule_arms(on=dict(outside=lambda k: 0.06 * ROUND_MS)))
        self.assertEqual((code, result['verdict']), (1, 'FAIL'))
        statistic = self.statistic(result)
        self.assertTrue(abs(statistic['cost']) < 0.002 and statistic['verdict'] != 'FAIL', 'the flag phases never see '
                                                                                           'it: %s' % statistic)
        self.assertEqual([p['ratio'] for p in result['pairs']], [1.0606, 1.0606])
        self.assertEqual(len([p for p in result['s2_problems'] if 'the net' in p and 'past 1.05' in p]), 2)

    # -- what the approved FAIL side still does to a flag that costs nothing (the user's decision) -----------------
    def test_pinned_host_stalls_on_the_flag_on_arms_alone_fail_the_untrimmed_guard(self):
        # What calibration found (runs 36270917139, 36272682217, 36276533585): log-write stalls of 0.1-1.1 s inside
        # ~10% of the verifies, in either arm. The trim drops them; the untrimmed mean does not - so stalls that happen
        # to fall on the flag-on arms FAIL a flag that costs nothing. Pinned here as the rule's behaviour: dropping the
        # guard is not stricter, so it waits for the user.
        stalls = {2: 135.0, 8: 190.0, 14: 1100.0, 24: 90.0}
        code, result, _, _ = self.rule(rule_arms(on=dict(spikes=stalls)))
        statistic = self.statistic(result)
        self.assertEqual((code, result['verdict']), (1, 'FAIL'))
        self.assertTrue(statistic['cost'] < 0.02 < statistic['untrimmed'], statistic)

    def test_pinned_the_arms_offsets_can_fail_a_flag_that_costs_nothing(self):
        # FAIL iff c > 2% (approved) reads c alone, and c carries the arms' offsets: both flag-on arms 4 ms slower in F
        # (~1.1 sds of a pair's offset at the floor) FAIL a flag that costs nothing. Judging the FAIL side on a bound
        # is not stricter, so it waits for the user.
        code, result, _, _ = self.rule(rule_arms(on=dict(f_offset=4.0)))
        self.assertEqual((code, result['verdict']), (1, 'FAIL'), result['lines'])
        self.assertIn('the flag phases: c 2.4', ' '.join(result['s2_problems']))

    # -- what is not judged -------------------------------------------------------------------------------------
    def test_mismatched_rounds_are_not_comparable(self):
        code, result, records, _ = self.rule(rule_arms(on_2=rule_log(True, emitted=4, seed=4)))
        self.assertEqual((code, result['verdict']), (1, 'NOT_COMPARABLE'), result['lines'])
        self.assertTrue(any(line.startswith('pair 2: NOT_COMPARABLE - four-live packed round 0 is') for line in
                            result['lines']), result['lines'])
        self.assertEqual(records[driver.CONTROL_PHASE_RECORD]['plan_verdict'], 'NOT_COMPARABLE')
        self.assertFalse(result['flag_phase']['decided'], 'pair 1 alone is not the paired set')
        self.assertTrue(any('not decided (the paired set is not complete; it would read NOT_RESOLVED)' in line
                            for line in result['lines']), result['lines'])

    def test_every_pair_serves_the_same_rounds(self):
        # Both arms of pair 2 alike, but not pair 1's rounds: NOT_COMPARABLE; the same rounds stopping early: short.
        code, result, _, _ = self.rule(rule_arms(off_2=rule_log(False, emitted=4, seed=3),
                                                 on_2=rule_log(True, emitted=4, seed=4)))
        self.assertEqual((code, result['verdict']), (1, 'NOT_COMPARABLE'), result['lines'])
        self.assertTrue(any(line.startswith('pair 2: NOT_COMPARABLE - not pair 1\'s rounds: four-live packed round 0')
                            for line in result['lines']), result['lines'])
        code, result, _, _ = self.rule(rule_arms(off_2=rule_log(False, rounds=30, seed=3),
                                                 on_2=rule_log(True, rounds=30, seed=4, **QUICKER), on_1=rule_log(
                                                     True, seed=2, **QUICKER)))
        self.assertEqual((code, result['verdict']), (1, 'NOT_EXERCISED'), result['lines'])
        self.assertIn('pair 2: not pair 1\'s rounds: control-off-1 has 36 four-live packed rounds and control-off-2 30',
                      result['shortfalls'])

    def test_short_or_untimed_rounds_are_a_shortfall_never_a_pass(self):
        cut = rule_log(True, seed=2).split('\n')
        last_step = max(i for i, line in enumerate(cut) if '[PHASE] execute total=' in line)
        for label, on_1 in (('short', rule_log(True, rounds=30, seed=2)),
                            ('untimed', '\n'.join(cut[:last_step] + cut[last_step + 1:])),
                            ('no rounds', s2_log(audit=False, packed=False))):
            with self.subTest(label=label):
                code, result, _, _ = self.rule(rule_arms(on_1=on_1))
                self.assertIn(result['verdict'], ('NOT_EXERCISED', 'FAIL'), result['lines'])
                self.assertNotEqual(code, 0)
                self.assertTrue(any(s.startswith('pair 1: ') for s in result['shortfalls']), result['shortfalls'])
        code, result, _, _ = self.rule(rule_arms(on_1=rule_log(True, rounds=30, seed=2)))
        self.assertEqual(result['verdict'], 'NOT_EXERCISED')
        self.assertIn('pair 1: the flag-off arm has 36 four-live packed rounds and the flag-on arm 30: only 30 pair',
                      result['shortfalls'])

    def test_a_phase_absent_from_a_log_is_a_shortfall_never_a_pass(self):
        for label, drop in (('every [PACKED-FENCES] line', lambda k: ('fences',)),
                            ('one due window pre-stage', lambda k: ('window',) if k == 7 else ()),
                            ('one round\'s flush', lambda k: ('flush',) if k == 7 else ())):
            with self.subTest(label=label):
                code, result, _, _ = self.rule(rule_arms(on=QUICKER, on_2=rule_log(True, seed=4, drop=drop,
                                                                                    **QUICKER)))
                self.assertEqual((code, result['verdict']), (1, 'NOT_EXERCISED'), result['lines'])
                self.assertTrue(any('not measured whole' in s for s in result['shortfalls']), result['shortfalls'])
                self.assertEqual(result['flag_phase']['verdict'], 'NOT_EXERCISED')

    def test_the_breakdowns_and_the_a_a_contrasts_are_recorded_and_judge_only_through_the_arms_offset(self):
        # Pair 2's arms both 3% slower on the host: the A/A reads (off-2 - off-1, on-2 - on-1) show it, the pairs do
        # not; the reads (~1.35 ms) stay under the floor, so the bound is the floor's.
        code, result, records, _ = self.rule(rule_arms(on=QUICKER, off_2=rule_log(False, offset=0.03, seed=3),
                                                       on_2=rule_log(True, offset=0.03, seed=4, **QUICKER)))
        self.assertEqual((code, result['verdict']), (0, 'PASS'), result['lines'])
        flag_phase = result['flag_phase']
        off, on = flag_phase['aa']
        self.assertEqual([(entry['arms'], entry['state'], entry['rounds']) for entry in (off, on)],
                         [('control-off-2 - control-off-1', 'off', 36), ('control-on-2 - control-on-1', 'on', 36)])
        self.assertTrue(off['phases']['bookkeeping']['trimmed_mean_ms'] > 1.0, off)
        self.assertTrue(1.0 < off['trimmed_ms'] < 2.0 and 1.0 < on['trimmed_ms'] < 2.0, (off, on))
        self.assertEqual(flag_phase['arm_sigma']['used_ms'], 3.5)
        self.assertEqual(sorted(flag_phase['phases']), sorted(driver.FLAG_PHASES + ('flag_ms',)))
        self.assertEqual([entry['pair'] for entry in flag_phase['pairs']], [1, 2])
        self.assertTrue(any(line.startswith('A/A control-off-2 - control-off-1: 36 rounds') for line in result['lines']))
        self.assertEqual([entry['arms'] for entry in records[driver.CONTROL_PHASE_RECORD]['aa']],
                         ['control-off-2 - control-off-1', 'control-on-2 - control-on-1'])

    # -- NOT_RESOLVED and the one re-run ---------------------------------------------------------------------------
    def pooled(self, first, second_arms, extra=()):
        """The re-run of `first` (a phase record) on `second_arms`, pooled: (code, result, records, lines)."""
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, driver.CONTROL_PHASE_RECORD)
            with open(path, 'w', encoding='utf-8') as handle:
                json.dump(first, handle)
            return self.rule(second_arms, extra=['--control-pool', path] + list(extra))

    def test_not_resolved_asks_for_exactly_one_pooled_re_run(self):
        code, result, records, _ = self.rule(rule_arms(on=dict(cost=wide_cost)))
        statistic = self.statistic(result)
        self.assertEqual((code, result['verdict']), (1, 'NOT_RESOLVED'), result['lines'])
        self.assertTrue(statistic['cost'] <= 0.02 < statistic['upper'] and statistic['untrimmed'] <= 0.02, statistic)
        self.assertTrue(any('exactly one re-run of this plan, decided on both runs pooled' in line
                            for line in result['lines']), result['lines'])
        self.assertFalse(result['flag_phase']['escalate'])
        record = records[driver.CONTROL_PHASE_RECORD]
        self.assertEqual((record['rule'], record['plan_verdict'], record['pooled']),
                         (driver.CONTROL_RULE, 'NOT_RESOLVED', False))
        # What binds a re-run to it: this run's identity, the image, profile and arms, and the rounds it served.
        self.assertEqual(len(record['run']['nonce']), 16)
        self.assertEqual((record['binding']['image'], record['binding']['profile']), ('zot/img:s2', 'c2-packed'))
        self.assertEqual([arm[0] for arm in record['binding']['arms']], ['control-off-1', 'control-on-1',
                                                                         'control-off-2', 'control-on-2',
                                                                         'control-audit'])
        self.assertEqual(len(record['sequence']), 36)

    def test_the_one_re_run_is_decided_on_both_runs_pooled(self):
        first = self.rule(rule_arms(on=dict(cost=wide_cost)))[2][driver.CONTROL_PHASE_RECORD]
        code, result, records, lines = self.pooled(first, rule_arms(on=QUICKER, rerun=True))
        self.assertEqual((code, result['verdict']), (0, 'PASS'), result['lines'])
        self.assertEqual((self.statistic(result)['rounds'], self.statistic(result)['pairs']), (144, 4))
        pooled = records[driver.CONTROL_PHASE_RECORD]
        self.assertTrue(pooled['pooled'])
        self.assertEqual(pooled['pooled_with']['run'], first['run'])
        self.assertEqual(len(pooled['pooled_with']['sha256']), 64)
        self.assertNotEqual(pooled['run']['nonce'], first['run']['nonce'])
        code, result, records, _ = self.pooled(first, rule_arms(on=dict(cost=wide_cost), rerun=True))
        self.assertEqual((code, result['verdict']), (1, 'NOT_RESOLVED'), result['lines'])
        self.assertTrue(result['flag_phase']['escalate'])
        self.assertTrue(any('escalate to the user - no further re-run' in line for line in result['lines']))
        # The pooled record can never be pooled again.
        code, result, _, lines = self.pooled(records[driver.CONTROL_PHASE_RECORD], rule_arms())
        self.assertEqual((code, result), (2, None))
        self.assertTrue(any('the one re-run is spent' in line for line in lines), lines)

    def test_defect_a_second_re_run_of_one_first_run_shows_as_one(self):
        # The gate keeps no state between runs, so it cannot refuse a second re-run of a first run it never saw spent;
        # the first coding also left nothing to find one by. Now every pooled record (and the summary, and its line)
        # names the first run it spent and that record's sha256: two re-runs of one first run are two records naming
        # the same first run.
        first = self.rule(rule_arms(on=dict(cost=wide_cost)))[2][driver.CONTROL_PHASE_RECORD]
        spent = []
        for arms in (rule_arms(on=dict(cost=wide_cost), rerun=True), rule_arms(on=QUICKER, rerun=True)):
            code, result, records, _ = self.pooled(first, arms)
            spent.append(records[driver.CONTROL_PHASE_RECORD]['pooled_with'])
            self.assertTrue(any('the one re-run of the first run %s' % first['run']['nonce'] in line and
                                'any other pooled record naming this first run is a second re-run, void' in line
                                for line in result['lines']), result['lines'])
        self.assertEqual([(entry['run'], entry['sha256']) for entry in spent],
                         [(first['run'], spent[0]['sha256'])] * 2)

    def test_defect_a_run_is_never_pooled_with_itself(self):
        # The first run's own logs judged again as its "re-run": the same rounds to the millisecond.
        arms = rule_arms(on=dict(cost=wide_cost))
        first = self.rule(arms)[2][driver.CONTROL_PHASE_RECORD]
        code, result, _, _ = self.pooled(first, arms)
        self.assertEqual((code, result['verdict']), (1, 'NOT_EXERCISED'), result['lines'])
        self.assertIn('pooled: this run\'s paired rounds are the first run\'s own - a run is never pooled with itself',
                      result['shortfalls'])

    def test_what_the_re_run_refuses(self):
        passed = self.rule(rule_arms(on=QUICKER))[2][driver.CONTROL_PHASE_RECORD]
        unresolved = self.rule(rule_arms(on=dict(cost=wide_cost)))[2][driver.CONTROL_PHASE_RECORD]
        other_image = dict(unresolved, binding=dict(unresolved['binding'], image='zot/img:other'))
        one_pair = dict(unresolved, pairs=unresolved['pairs'][:1])
        no_run = dict(unresolved, run=None)
        for label, record, extra, why in (
                ('a run that was not NOT_RESOLVED', passed, (), 'only an unresolved run is re-run'),
                ('another rule', dict(unresolved, rule='g3-whole-round'), (), 'was judged under rule'),
                ('the first coding\'s rule', dict(unresolved, rule='g3-flag-phase-2026-09-28'), (),
                 'was judged under rule'),
                ('another image', other_image, (), 'it differs in image (first run: image zot/img:other'),
                ('another plan shape', unresolved, ('--pairs', '1'), 'it differs in arms'),
                ('another number of pairs', one_pair, (), 'carries 1 timed pairs, not the rule\'s 2'),
                ('no run identity', no_run, (), 'names no run identity')):
            with self.subTest(label=label):
                code, result, _, lines = self.pooled(record, rule_arms(), extra=extra)
                self.assertEqual((code, result), (2, None))
                self.assertTrue(any(why in line for line in lines), lines)
        # The same image and arms, but other rounds (another shape of the prompts): NOT_COMPARABLE once run.
        shifted = dict(unresolved, sequence=[[[s, p + 1, e] for s, p, e in rounds] for rounds in unresolved['sequence']])
        code, result, _, _ = self.pooled(shifted, rule_arms(on=QUICKER, rerun=True))
        self.assertEqual((code, result['verdict']), (1, 'NOT_COMPARABLE'), result['lines'])
        self.assertTrue(any(line.startswith('pooled: NOT_COMPARABLE - four-live packed round 0') for line in
                            result['lines']), result['lines'])
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, 'first.json')
            with open(path, 'w', encoding='utf-8') as handle:
                json.dump(unresolved, handle)
            code, summary, calls, lines, _ = Driver(self).run(['--profile', 'c2-packed', '--plan', 'forced-cap',
                                                               '--control-pool', path], {}, {})
        self.assertEqual((code, calls), (2, []))
        self.assertTrue(any('run no control' in line for line in lines), lines)


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

    def test_a_flag_off_packed_round_outside_family_f_fails(self):
        prompt = 16384
        override = ['INFO [PINDIAG] packed capture position override=%d (gate only)' % prompt]
        outside = s2_log(on=False, positions=[prompt, prompt, prompt, 16630], rounds=12, extra=override)
        result = self.below(off_log=outside)
        self.assertEqual(result['verdict'], 'FAIL')
        self.assertIn('flag-off: packed rounds outside family 16640 (user 3 round', ' '.join(
            result['pairs']['16640-1']['problems']))


class ServingDriverTests(unittest.TestCase):
    LENGTHS = [1536, 20000, 60000, 120000]

    def g4(self, plan, concurrent_log, solo_log, concurrent_texts=None, extra_reports=None, extra_logs=None,
           max_tokens=4096, argv=(), cache=flat_cache, lengths=None):
        lengths = lengths or self.LENGTHS
        # What the plan's arms add, as the harness's configuration records it (mixed: all four audits, M7).
        env = driver.AUDIT_ENV + (driver.G4_ALL_AUDITS if plan in driver.ALL_AUDIT_PLANS else ())
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

    SHORT = [60, 255, 2047, 120000]

    def short(self, extra=(), ladder=SINGLE_LADDER, points=None, buckets=(2048,)):
        """The short plan with both arms' logs carrying `extra`, W6d's engine points (healthy unless `points`), the
        prefill point and W6a's built-bucket lines (`buckets`, None for none); the verdict."""
        ledger = [prefill_point(120000, 1500.0)] + (engine_points() if points is None else list(points))
        concurrent = s2_log(positions=[128, 255, 2047, 120000], extra=ledger + list(extra), ladder=ladder,
                            buckets=buckets)
        solo = s2_log(packed=False, positions=self.SHORT, extra=ledger + list(extra), ladder=ladder, buckets=buckets)
        return self.g4('short', concurrent, solo, lengths=self.SHORT)[1]['results']['short']

    def test_short_needs_the_one_bucket_ladder_no_hold_and_the_floor(self):
        self.assertEqual(self.short()['verdict'], 'PASS', self.short()['lines'])
        held = self.short(extra=[hold_line(120000, 900e6, 1568.4e6, engine_id(3), 3), decision_line(3)])
        deferred = self.short(extra=[deferred_line(120000, 900e6, 1568.4e6, engine_id(3), [engine_id(0)]),
                                     decision_line(3)])
        self.assertEqual(deferred['verdict'], 'PASS', 'a deferred step (a stale reading) is no hold')
        self.assertEqual(held['verdict'], 'FAIL')
        self.assertIn('seat free (decodes [3] of 4 seats)', ' '.join(held['s2_problems']))
        self.assertEqual(self.short(ladder=SHORT_LADDER)['verdict'], 'FAIL')
        self.assertEqual(self.short(extra=[prefill_point(120000, 400.0)])['verdict'], 'FAIL', 'prefill floor 0.1 GB')

    def test_short_reads_the_ladder_as_w6_logs_it(self):
        # W6a logs a tuple through loguru ('(2048,)'); a wrong ladder is named, an unlogged one fails.
        wrong = self.short(ladder=SHORT_LADDER)
        self.assertIn('[256, 512, 1024, 2048]', ' '.join(wrong['s2_problems']))
        self.assertEqual(self.short(ladder='unavailable (ValueError)')['verdict'], 'FAIL')
        report = gate.s2_report({EXTENT: '1'}, s2_log(ladder=SINGLE_LADDER), [], None)
        self.assertEqual((report['ladders'], report['buckets_built']), ([[2048]] * 4, [[2048]] * 4))

    def test_short_judges_the_buckets_the_engines_built(self):
        # The executed path (W6a's 'proposal buckets built' line, read from the capture): the ladder line is only
        # computed from the environment.
        built = self.short(buckets=(256, 512, 1024, 2048))
        self.assertEqual(built['verdict'], 'FAIL')
        self.assertIn('built proposal buckets [\'[256, 512, 1024, 2048]\']', ' '.join(built['s2_problems']))
        unseen = self.short(buckets=None)
        self.assertEqual(unseen['verdict'], 'NOT_EXERCISED')
        self.assertIn('0 "[PINDIAG] proposal buckets built" lines for 4 engines', ' '.join(unseen['shortfalls']))

    def test_short_judges_w6s_engine_margin_not_only_the_prefill_points(self):
        # W6 logs margin=-700.0MB ahead of an engine build; the prefill points alone are healthy.
        starved = self.short(points=engine_points(largest=300e6))
        self.assertEqual(starved['verdict'], 'FAIL')
        self.assertIn('below 0.25 (engine req=', ' '.join(starved['s2_problems']))
        self.assertIn('-0.700 GB', ' '.join(starved['s2_problems']))
        unread = self.short(extra=['INFO ' + W6_BEFORE_UNREAD_FORMAT % ('quad point=slots=0,1,2,3', 'no statistics')])
        self.assertEqual(unread['verdict'], 'NOT_EXERCISED', 'a before-point that read no DRAM is never a pass')
        self.assertIn('read no DRAM', ' '.join(unread['shortfalls']))
        blind = self.short(points=())
        self.assertEqual(blind['verdict'], 'NOT_EXERCISED', 'no engine point: the floor saw no engine build')

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
        padded = s2_log(positions=self.LENGTHS, extra=[round_line(99, [1792, 20224], idle=[2, 3]), audit_line(99)])
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

    def arrival(self, live):
        """lifecycle-arrival with user 1's build drop fired at `live` live streams."""
        text = s2_log(rounds=2)

        def arm(name):
            def build(n):
                report = base.lifecycle_report({1: 'drop'} if name == 'lifecycle-arrival' else {}, profile='c2-packed')
                if name == 'lifecycle-arrival':
                    event = report['lifecycle']['events']['1']
                    event.update(kind='build', spec='build', phase='build', live=live)
                    report['user_events']['drops'] = {'1': 'build'}
                report['qwen_configuration'] = configuration('c2-packed', ((AUDIT, '1'),))
                gate.add_s2_report(report, report['qwen_configuration'], text, report['streams'], None)
                return report
            return build
        names = ('lifecycle-arrival', 'lifecycle-arrival-solo')
        return self.run_plan('lifecycle-arrival', dict((name, arm(name)) for name in names),
                             dict((name, text) for name in names))

    def test_lifecycle_arrival_needs_user_0_decoding_when_user_1_is_dropped(self):
        self.assertEqual(self.arrival(1)['verdict'], 'PASS', self.arrival(1)['lines'])
        early = self.arrival(0)
        self.assertEqual(early['verdict'], 'NOT_EXERCISED', 'user 1 dropped before user 0 decoded: not VR4:150')
        self.assertIn('fired at 0 live streams, not 1', ' '.join(early['shortfalls']))

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

    def memory(self, extra=()):
        floor = [prefill_point(123136, 1500.0)] + engine_points()

        def arm(log_text, users=4):
            return lambda n: s2_arm_report(log_text, 'c2-packed', env=((AUDIT, '1'),), users=users,
                                           lengths=[123136] * users, max_tokens=8192, completion=8192)
        good = s2_log(positions=[123136] * 4, extra=floor + list(extra))
        short_log = s2_log(positions=[128] * 4, extra=floor)   # packed only from 128 on (the admission floor)
        return self.run_plan('memory', {'memory-concurrent': arm(good), 'memory-short': lambda n: s2_arm_report(
            short_log, 'c2-packed', env=((AUDIT, '1'),), lengths=[60] * 4, max_tokens=16384, completion=16384)},
            {'memory-concurrent': good, 'memory-short': short_log})

    def test_memory_on_c2_packed_fails_on_a_hold(self):
        clean = self.memory()
        self.assertEqual(clean['verdict'], 'PASS', clean['lines'])
        held = self.memory([hold_line(123136, 700e6, 1568.4e6, engine_id(3), 3), decision_line(3)])
        self.assertEqual(held['verdict'], 'FAIL')
        # A deferred step logs the wrapper's held decision state too (allowed=0, hidden): it is no hold.
        deferred = self.memory([deferred_line(123136, 700e6, 1568.4e6, engine_id(3), [engine_id(0)]),
                                decision_line(3), 'INFO ' + W6_CARRIED_LINE.format([engine_id(0)])])
        self.assertEqual(deferred['verdict'], 'PASS', deferred['lines'])
        self.assertEqual(self.memory([W6_LIFTED_LINE.format(123136, mb(700e6), mb(1568.4e6), engine_id(3))])['verdict'],
                         'FAIL', 'a lifted hold: admitted though it did not fit')
        unavailable = self.memory(['INFO ' + W6_UNAVAILABLE_LINE.format(engine_id(3), 'no reading')])
        self.assertEqual(unavailable['verdict'], 'NOT_EXERCISED', 'a hold that read no DRAM is unjudged, never a pass')
        released = self.memory([hold_line(123136, 700e6, 1568.4e6, engine_id(3), 4), decision_line(4),
                                'INFO ' + W6_RELEASED_HOLD_LINE.format(123136, mb(1700e6), mb(1568.4e6), engine_id(3))])
        self.assertEqual(released['verdict'], 'PASS', 'every seat decoding: the prompt waited for a seat, not DRAM')

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

    @staticmethod
    def churn_log(quad_departures=8, release=1, others=3, holds=True, extra=()):
        """A churn arm's log: the ledger's points, `quad_departures` departures each while a quad was formed (a quad
        round, then the detach's release line - quad=`release` (one value, or one per departure), or none for None
        - just ahead of the step's [PHASE] line), then `others` departures with no quad (the last wave), and -
        `holds` - the fifth user's W6b holds while every seat decodes (decodes=4, the seats: no failed fit)."""
        lines = [prefill_point(110000, 1500.0)] + engine_points()
        for index in range(quad_departures):
            lines.append(quad_line(10 * index + 1, built=1))
            lines.append(quad_line(10 * index + 2, built=0))
            if holds:
                lines += [hold_line(120000, 700e6, 1568.4e6, engine_id(index + 4), 4), decision_line(4)]
            quad = release[index] if isinstance(release, list) else release
            if quad is not None:
                lines.append(released_line(quad, [(0, 1), (2, 3)]))
            lines.append(execute_line(100.0 + index, 4, finished=[engine_id(index)]))
            if holds:
                lines.append('INFO ' + W6_RELEASED_HOLD_LINE.format(120000, mb(1900e6), mb(1568.4e6), engine_id(index + 4)))
        for index in range(quad_departures, quad_departures + others):
            lines.append(released_line(0))
            lines.append(execute_line(200.0 + index, 3, finished=[engine_id(index)]))
        return s2_log(extra=lines + list(extra))

    def test_churn_needs_dead_traces_released(self):
        passed = self.churn(self.churn_log())
        self.assertEqual(passed['verdict'], 'PASS', passed['lines'])
        self.assertEqual(passed['facts']['replacements'], 8)
        self.assertEqual((passed['facts']['releases']['quad_departures'], passed['facts']['releases']['departures']),
                         (8, 11))
        self.assertEqual(self.churn(self.churn_log(release=None))['verdict'], 'FAIL', 'no release line at all')
        stale = self.churn(self.churn_log(release=0))
        self.assertEqual(stale['verdict'], 'FAIL', 'released quad=0 while the quad was formed')
        self.assertIn('released quad=0', ' '.join(stale['s2_problems']))
        # One departure of eight left its quad behind: the others' releases do not cover it.
        one = self.churn(self.churn_log(release=[1, 1, 1, 0, 1, 1, 1, 1]))
        self.assertEqual(one['verdict'], 'FAIL')
        self.assertIn('1 of 8 departures while a quad was formed', ' '.join(one['s2_problems']))
        self.assertEqual(self.churn(self.churn_log(release=[1, 1, 1, None, 1, 1, 1, 1]))['verdict'], 'FAIL')
        self.assertEqual(self.churn(self.churn_log(quad_departures=0, others=11))['verdict'], 'NOT_EXERCISED')
        self.assertEqual(self.churn(self.churn_log(quad_departures=4, others=1))['verdict'], 'NOT_EXERCISED',
                         'five departures for eight replacements: the seats did not churn')

    def test_churn_holds_while_every_seat_decodes_are_no_failure(self):
        # The review's M11 false FAIL: nine-plus users on four seats, the fifth prompt held at decodes=4.
        self.assertEqual(self.churn(self.churn_log(holds=True))['verdict'], 'PASS')
        freed = self.churn(self.churn_log(extra=[hold_line(120000, 700e6, 1568.4e6, engine_id(9), 3)]))
        self.assertEqual(freed['verdict'], 'FAIL', 'held with a seat free (decodes=3)')
        self.assertIn('decodes [3] of 4 seats', ' '.join(freed['s2_problems']))
        deferred = self.churn(self.churn_log(extra=[deferred_line(120000, 700e6, 1568.4e6, engine_id(9),
                                                                  [engine_id(1)]), decision_line(3)]))
        self.assertEqual(deferred['verdict'], 'PASS', 'a deferred step behind a departure is no hold')
        region = self.churn(self.churn_log(extra=before_lines('quad', 900e6, QUAD_ESTIMATE, point='slots=0,1,2,3',
                                                              trace=(260e6, 8e6))))
        self.assertEqual(region['facts']['trace_region']['readings'], 2)
        self.assertIn('2 readings, at most 0.26 GB used', ' '.join(region['lines']))
        self.assertIn('unavailable at', ' '.join(self.churn(self.churn_log())['lines']))

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

    def test_a_same_path_divergence_fails_the_permuted_arm(self):
        lengths = ServingDriverTests.LENGTHS
        log_text = s2_log(positions=lengths)
        texts = [s['text'] for s in V235['streams']]
        texts[1] = texts[1][:70] + '#' + texts[1][71:]
        forward = lambda n: s2_arm_report(log_text, 'c2-packed', env=((AUDIT, '1'),), lengths=lengths, max_tokens=4096,
                                          finish='stop', completion=300)
        reverse = lambda n: s2_arm_report(log_text, 'c2-packed', env=((AUDIT, '1'),), lengths=lengths, max_tokens=4096,
                                          finish='stop', completion=300, texts=texts)
        result = self.run_plan('permuted', {'permuted-forward': forward, 'permuted-reverse': reverse},
                               {'permuted-forward': log_text, 'permuted-reverse': log_text})
        self.assertEqual(result['verdict'], 'FAIL')
        self.assertTrue(any(p.startswith('user 1 took the same path in both arms') and 'FAILS under any policy' in p
                            for p in result['s2_problems']), result['s2_problems'])


class FormatTests(unittest.TestCase):
    """The harness reads each producer's line as the producer writes it (the W11 review's defects 1-5, 8, 10)."""

    def s2(self, text, env=((AUDIT, '1'),)):
        environ = dict(QWEN_FAST_SDPA_MODES='tail,share,slice')
        environ[EXTENT] = '1'
        environ.update(env)
        return gate.s2_report(environ, text, [dict(request_id=request_id(u)) for u in range(4)], [131072] * 4)

    def test_w3s_audit_line_with_rotated_is_read_by_field_name(self):
        text = '\n'.join([round_line(5, [131328] * 4), audit_line(5, rotated=1, ms=2.13)])
        audit = gate.extent_audit(text)
        self.assertEqual((audit['lines'], audit['malformed'], audit['incomplete'], audit['median_ms']), (1, 0, 0, 2.13))
        self.assertEqual([p for p in self.s2(text)['problems'] if AUDIT in p], [])
        # A field W3 adds later never hides the line; one it drops is named, never read as a clean round.
        self.assertEqual(gate.extent_audit(audit_line(5).replace(' ms=', ' reads=12 ms='))['lines'], 1)
        cut = re.sub(r' cur_pos_ok=[0-9]+', '', audit_line(5))
        problems = ' '.join(self.s2(round_line(5, [131328] * 4) + '\n' + cut)['problems'])
        self.assertIn('1 audit lines without round/segments/words_ok/cur_pos_ok', problems)
        incomplete = round_line(5, [131328] * 4) + '\n' + audit_line(5, words_ok=3)
        self.assertIn('did not read back', ' '.join(self.s2(incomplete)['problems']))

    def test_an_unaudited_round_is_named_by_its_number(self):
        text = '\n'.join([round_line(5, [131328] * 4), audit_line(5), round_line(6, [131328] * 4), audit_line(5)])
        self.assertIn('unaudited rounds [6]', ' '.join(self.s2(text)['problems']))

    def test_w3s_families_are_the_extents_after_each_segment(self):
        # The review's probe: a same-family four-live round and a same-family padded round are one family each.
        same = '\n'.join([round_line(5, [131328] * 4), round_line(6, [20224, 20224], idle=[2, 3], capped=[(1, 6)])])
        rounds = gate.extent_rounds(same)
        self.assertEqual((rounds['multi_family_rounds'], rounds['max_families'], rounds['families_seen']),
                         (0, 1, [20224, 131328]))
        self.assertEqual((rounds['capped_segments'], rounds['idle_rounds']), (1, 1))
        mixed = gate.extent_rounds(round_line(7, [1792, 20224, 60160, 120064]))
        self.assertEqual((mixed['multi_family_rounds'], mixed['max_families']), (1, 4))
        self.assertEqual(gate.round_families('131328,4352'), [131328, 4352], 'a bare E is read too')

    def test_w6s_proposal_ladder_is_read_as_its_tuple(self):
        self.assertEqual([gate.ladder_of(text) for text in (SINGLE_LADDER, SHORT_LADDER, '[2048]', 'unavailable (X)')],
                         [[2048], [256, 512, 1024, 2048], [2048], 'unavailable (X)'])
        log_text = base.any_request_log(ladder=SINGLE_LADDER) + base.any_request_log(ladder=SHORT_LADDER)
        self.assertEqual(self.s2(log_text)['ladders'], [[2048]] * 4 + [[256, 512, 1024, 2048]] * 4)

    def test_w6s_before_points_negative_margins_unread_points_and_continuations(self):
        # The review's probe: W6's own line, margin=-700.0MB ahead of an engine build, here long enough to continue.
        lines = before_lines('engine', 300e6, ENGINE_PEAK, point='req=%s' % engine_id(0)[-12:], trace=(268.4e6, 12.5e6))
        self.assertTrue(any(LEDGER_CONTINUATION in line for line in lines), 'past the ledger\'s line budget')
        lines += ['INFO ' + W6_BEFORE_UNREAD_FORMAT % ('quad point=slots=0,1,2,3', 'no statistics'),
                  'INFO [MEMLEDGER] phase=prefill point=before prompt=60 dram unavailable (no statistics)']
        before = gate.before_points('\n'.join(lines))
        self.assertEqual((before['points'], before['judged'], before['unread'], before['by_op']), (2, 2, 2, {'engine': 2}))
        self.assertEqual((before['floor_gb'], before['logged_floor_gb']), (-0.7, -0.7))
        self.assertEqual((before['floor_point']['op'], before['floor_point']['detail']), ('engine', 'req=1-0-abcd1234'))
        region = gate.trace_region('\n'.join(lines))
        self.assertEqual((region['readings'], region['max_used_gb'], region['min_largest_free_gb']), (2, 0.2684, 0.0125))
        blind = gate.trace_region('\n'.join(before_lines('quad', 900e6, QUAD_ESTIMATE)))
        self.assertEqual((blind['readings'], blind['unavailable']), (0, 2))

    def test_w6s_dram_hold_lines(self):
        text = '\n'.join([deferred_line(120000, 700e6, 1568.4e6, 'r5', ['r1']), decision_line(3),
                          'INFO ' + W6_CARRIED_LINE.format(['r1']), hold_line(120000, 700e6, 1568.4e6, 'r5', 3),
                          decision_line(2, held=True), decision_line(3, allowed=1, hidden=False),
                          'INFO ' + W6_RELEASED_HOLD_LINE.format(120000, mb(1700e6), mb(1568.4e6), 'r5'),
                          'INFO ' + W6_LIFTED_LINE.format(60, mb(700e6), mb(1256e6), 'r6'),
                          'INFO ' + W6_UNAVAILABLE_LINE.format('r7', 'no reading')])
        hold = gate.dram_holds(text)
        self.assertEqual((hold['holds'], hold['hold_decodes']), (1, [3]),
                         'only DRAM_HOLD lines: a deferred step and the wrapper\'s decision lines are none')
        self.assertEqual((hold['released'], hold['lifted'], hold['unavailable'], hold['deferred'], hold['carried']),
                         (1, 1, 1, 1, 1))
        self.assertEqual([line.split('] ', 1)[1][:25] for line in hold['lines']], ['dram hold prompt=120000 l'],
                         'released, lifted and unavailable lines are not hold lines')
        healthy = dict(floor_gb=1.0, by_op=dict(engine=8))
        problems, shortfalls = driver.memory_s2_checks('arm', dict(s2=dict(dram_hold=hold, before=healthy,
                                                                           extent_replay=True)), 4)
        self.assertEqual(len(problems), 2, problems)
        self.assertIn('seat free (decodes [3] of 4 seats)', problems[0])
        self.assertIn('dram hold lifted', problems[1])
        self.assertEqual(len(shortfalls), 1)
        # W6's first cut asked with every seat decoding too: such a hold is the fifth prompt waiting for a seat.
        waiting = gate.dram_holds('\n'.join([hold_line(120000, 700e6, 1568.4e6, 'r5', 4), decision_line(4)]))
        self.assertEqual(driver.memory_s2_checks('arm', dict(s2=dict(dram_hold=waiting, before=healthy,
                                                                     extent_replay=True)), 4), ([], []),
                         'every seat decoding: the fifth prompt waits for a seat, and no fit failed')

    def test_w6c_releases_are_matched_to_departures_while_a_quad_was_formed(self):
        text = '\n'.join([
            quad_line(1, built=1), released_line(1, [(0, 1), (2, 3)]), execute_line(1.0, 4, finished=[engine_id(0)]),
            released_line(0), execute_line(2.0, 3, finished=[engine_id(1)]),              # no quad formed: none owed
            quad_line(2), released_line(0, [(0, 1)]), execute_line(3.0, 4, finished=[engine_id(2)]),   # owed: quad=0
            quad_line(3), execute_line(4.0, 4, finished=[engine_id(3), engine_id(4)]),       # owed: no release line
            quad_line(4), released_line(1), released_line(0), execute_line(5.0, 4, finished=[engine_id(5), engine_id(6)]),
            execute_line(6.0, 4)])
        releases = gate.proposal_releases(text)
        self.assertEqual((releases['lines'], releases['quad'], releases['pairs'], releases['quad_rounds']), (5, 2, 3, 4))
        self.assertEqual((releases['departures'], releases['departure_steps'], releases['quad_departures'],
                          releases['unreleased']), (7, 5, 4, 2))
        self.assertIn('released quad=0', releases['unreleased_steps'][0])
        self.assertIn('no release line', releases['unreleased_steps'][1])
        self.assertEqual([gate.pair_count(text) for text in ('[[0, 1], [2, 3]]', '[]', '[[1, 3]]', '2')], [2, 0, 1, 2])

    def test_the_trace_region_is_recorded_from_w6s_fields(self):
        text = '\n'.join(before_lines('pair', 700e6, 44e6, point='slots=0,1', trace=(100e6, 150e6)))
        report = self.s2(text)
        self.assertEqual((report['trace_region']['readings'], report['trace_region']['max_used_gb']), (2, 0.1))
        self.assertEqual(len(report['trace_region_lines']), 2)
        self.assertIn('2 readings, at most 0.1 GB used', driver.trace_region_text(report['trace_region']))
        self.assertEqual(driver.trace_region_text(None), 'not logged (Q18)')


class HostCheckTests(unittest.TestCase):
    def test_the_host_reads_s2_markers_itself_without_the_harness_record(self):
        leaked = driver.s2_log_check(s2_log(on=True), False, (), dict(qwen_configuration={}))
        self.assertTrue(any('this is not the profile' in p for p in leaked), leaked)
        self.assertEqual(driver.s2_log_check(s2_log(on=False), False, (), dict(qwen_configuration={})), [])
        missing = driver.s2_log_check(s2_log(on=True), True, (), dict(qwen_configuration={}))
        self.assertTrue(any('carries no S2 record' in p for p in missing), missing)
        self.assertEqual(driver.s2_log_check(s2_log(on=True), True, (), dict(qwen_configuration={},
                                                                            s2=dict(problems=['x']))), ['S2: x'])

    def test_only_an_s2_run_or_an_explicit_jit_counts_the_kernel_cache(self):
        self.assertFalse(driver.counts_cache(list(job.GATE_PLANS), PROFILES, 'c2', 'auto'))
        self.assertFalse(driver.counts_cache(['bringup'], PROFILES, 'exact', 'auto'))
        self.assertTrue(driver.counts_cache(['matrix'], PROFILES, 'c2-packed', 'auto'))
        self.assertTrue(driver.counts_cache(['bringup', 'control'], PROFILES, 'c2', 'auto'))
        self.assertTrue(driver.counts_cache(['bringup'], PROFILES, 'exact', 'judge'))
        self.assertTrue(driver.counts_cache(['bringup'], PROFILES, 'exact', 'record'))

    def main_run(self, argv, reports, server_logs):
        """The driver's main with its real kernel-cache path (no cache_entries, no execute): docker is FakeDocker
        through Runner._execute, the image's cache lookup and count recorded."""
        from unittest import mock
        looked = []

        def image_cache(image, hub=driver.HUB):
            looked.append(image)
            return '/hub/.qwen-c2/kernels-x'
        docker = base.FakeDocker(reports, server_logs)
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, 'profiles.json')
            with open(path, 'w', encoding='utf-8') as handle:
                json.dump(PROFILES, handle)
            results = os.path.join(directory, 'results')
            with mock.patch.object(driver, 'image_kernel_cache', image_cache), \
                    mock.patch.object(driver, 'count_entries', lambda path: 100), \
                    mock.patch.object(driver.Runner, '_execute', staticmethod(docker)):
                driver.main(argv + ['--image', 'zot/img:s2', '--results', results, '--profiles', path],
                            devices=['/dev/tenstorrent/3', '/dev/tenstorrent/1'], log=lambda line: None,
                            containers=lambda: [], corpus=lambda: dict(V235['real_text']['corpus']))
            with open(os.path.join(results, 'c2-gate-summary.json'), encoding='utf-8') as handle:
                summary = json.load(handle)
        return looked, summary

    def test_an_s1_run_reads_no_kernel_cache_and_gains_no_cache_keys(self):
        report = lambda n: base.served_report(profile='c2-gate', configuration=configuration('c2-gate'),
                                              argv=base.served_argv('c2-gate', PROFILES))
        looked, summary = self.main_run(['--profile', 'c2-gate', '--plan', 'bringup'], {'bringup-concurrent': report},
                                        {'bringup-concurrent': base.any_request_log()})
        self.assertEqual(looked, [], 'no docker image inspect, no find: S1 as before W11')
        self.assertNotIn('kernel_cache', summary['arms']['bringup-concurrent'])
        self.assertNotIn('kernel_cache', summary)
        off, on = s2_log(on=False, round_ms=172.0, phases=True), quicker_on_log()
        reports, logs = {'control-audit': lambda n: s2_arm_report(s2_log(), 'c2-packed-gate', env=((AUDIT, '1'),))}, \
            {'control-audit': s2_log()}
        for index in (1, 2):
            reports['control-off-%d' % index] = lambda n: s2_arm_report(off, 'c2-gate')
            reports['control-on-%d' % index] = lambda n: s2_arm_report(on, 'c2-packed-gate')
            logs.update({'control-off-%d' % index: off, 'control-on-%d' % index: on})
        looked, summary = self.main_run(['--profile', 'c2-packed', '--plan', 'control'], reports, logs)
        self.assertEqual(looked, ['zot/img:s2'])
        self.assertEqual(summary['arms']['control-on-1']['kernel_cache'], dict(before=100, after=100, added=0,
                                                                                judged=True))
        self.assertEqual(summary['results']['control']['verdict'], 'PASS', summary['results']['control']['lines'])


def producer(name, attribute):
    """The checkout's `name` module when it carries `attribute` (its S2 work item merged), else None."""
    try:
        module = __import__(name)
    except Exception:
        return None
    return module if hasattr(module, attribute) else None


def source_literal(function, prefix):
    """The one string literal in `function`'s source that starts with `prefix` (quotes stripped)."""
    found = re.findall(r"'(%s[^']*)'" % re.escape(prefix), inspect.getsource(function))
    assert len(found) == 1, (prefix, found)
    return found[0]


def rendered(lines):
    return '\n'.join('INFO ' + line for line in lines)


class ProducerContractTests(unittest.TestCase):
    """Each producer's own lines through the harness: rendered by the producer's code (or its format literal, where
    the line needs a device), parsed here, and the fixtures' copies held equal. A producer whose work item is not
    merged into this checkout skips; merged, it is live (W3: packed_verifier, W6: serving_prefill_admission,
    memory_ledger.MemoryLedger.before, the coordinator's RELEASED_LINE, serving_request_factory's ladder)."""

    def test_the_eager_publication_warm_lines(self):
        import publication_warm
        self.assertEqual((publication_warm.MARKER, publication_warm.SKIPPED_MARKER),
                         (gate.S2_WARM_MARKER, gate.S2_WARM_SKIPPED_MARKER))
        self.assertEqual(publication_warm.warmed_line(WARM_SUMMARY), WARM_LINE)
        self.assertEqual(gate.publication_warm_report(rendered([publication_warm.warmed_line(WARM_SUMMARY)]))['warmed'],
                         [dict(shapes=71, ms=1830.4, packed='0,16,32,48:1-16', sequential='1:1,2:1-2,4:1-4',
                               program_cache='412->655')])
        lines = []
        block = types.SimpleNamespace(extent=True)
        self.assertIsNone(publication_warm.warm(block, operations=None, mesh=None, pool=None, shared_weights=None,
                                                collectives=None, log=lines.append))
        self.assertEqual(lines, [WARM_SKIPPED_LINE])

    def test_quad_rounds_and_step_lines(self):
        import quad_draft
        import serving_worker_hook
        self.assertEqual(quad_draft.ROUND_LINE, QUAD_ROUND_FORMAT)
        self.assertTrue(gate.QUAD_ROUND_LINE.search(quad_draft.ROUND_LINE.format(round=7, built=1, ms=61.25)))
        self.assertEqual(quad_draft.QUAD_CAPTURE_BYTES_EST, QUAD_ESTIMATE)
        self.assertEqual(source_literal(serving_worker_hook.FastWorkerHook._execute, '[PHASE] execute '),
                         PHASE_EXECUTE_FORMAT)
        line = PHASE_EXECUTE_FORMAT.format(64, 0, 4, 60, sorted(['cmpl-b-0-x', 'cmpl-a-0-y']), [])
        self.assertEqual(gate.finished_ids(gate.PHASE_FINISHED.search(line).group(1)), ['cmpl-a-0-y', 'cmpl-b-0-x'])

    def test_the_flag_phase_lines(self):
        # G3's flag phases (c2_serving_gate.FLAG_PHASES): each producer's own format, and the gate reads what it renders.
        import early_draft
        import packed_verifier
        import serving_packed_step
        import serving_worker_hook
        import verify_prestage
        self.assertEqual(source_literal(serving_worker_hook.phase, '[PHASE] {} {} end'), PHASE_END_FORMAT)
        verify = re.sub(r"'\s*\n\s*'", '', inspect.getsource(packed_verifier.PackedVerifierEngine.verify))
        self.assertIn("'%s'" % VERIFY_SPLIT_FORMAT, verify)
        self.assertEqual(serving_packed_step.FENCES_LINE, FENCES_FORMAT)
        self.assertEqual((verify_prestage.WINDOW_MARKER, source_literal(verify_prestage.BlockPrestage.prestage,
                                                                        '%s round=%d buffers=')),
                         (WINDOW_MARKER, WINDOW_FORMAT))
        self.assertEqual((early_draft.GDN_MARKER, source_literal(packed_verifier.PackedVerifierEngine.flush_commits,
                                                                 '%s round=%d commits=%d site=')),
                         (GDN_MARKER, FLUSH_FORMAT))
        self.assertEqual(early_draft.IN_STEP_SITES, driver.FLUSH_SITES)
        fields = serving_packed_step.fences_fields(
            dict(replay_fence='replay', replay_fence_ms=0.06, validated_this_round=True,
                 prestage=dict(path='diff', prestage_ms=5.9, diff_ms=1.62, write_ms=0.48)),
            types.SimpleNamespace(rounds=36, commit_block_ms=[0.0, 0.0, 0.0, 0.01]), [3, 2, 1, 0])
        ids = ','.join(engine_id(user) for user in range(4))
        stretch = [execute_line(0.0, 4)] + ['INFO [PACKED] request=%s segment=%d position=131072 prefix=1 emitted=1'
                                            % (engine_id(user), user) for user in range(4)] + [
            'INFO ' + VERIFY_SPLIT_FORMAT % (36, 4, 0.0, 2.23, 119.53, 0.62, 3.23),
            'INFO ' + PHASE_END_FORMAT.format('packed_verify', ids, 126.1)] + [
            'INFO ' + PHASE_END_FORMAT.format('packed_commit', engine_id(user), 0.3) for user in range(4)] + [
            'INFO ' + serving_packed_step.FENCES_LINE.format(**fields),
            'INFO ' + WINDOW_FORMAT % (verify_prestage.WINDOW_MARKER, 37, 144, 5.8, 4),
            'INFO ' + FLUSH_FORMAT % (early_draft.GDN_MARKER, 36, 4, 'window', 0.07, '3:4,2:4,1:1,0:5'),
            execute_line(0.165, 4)]
        found, = driver.flag_phase_rounds('\n'.join(stretch))
        self.assertEqual(found['missing'], [])
        self.assertEqual(found['phases'], dict(replay=120.15, staging=2.1, readback=3.23, bookkeeping=0.62, commit=1.27,
                                               window=5.8))
        self.assertEqual((found['flag_ms'], found['round_ms']), (133.17, 165.0))

    def test_the_prefill_admission_decision_line(self):
        import serving_prefill_admission
        self.assertEqual(source_literal(serving_prefill_admission, '[PINDIAG] one fresh prefill per step: '),
                         DECISION_LINE)

    def test_the_ledgers_prefill_phase_point(self):
        import memory_ledger
        lines = []
        ledger = memory_ledger.MemoryLedger(None, None, log=lines.append, emit=lambda text: None)
        ledger.reading = lambda: [dict(chip=chip, largest_free=largest, free=3 * 10 ** 9, allocated=30 * 10 ** 9,
                                       total=34 * 10 ** 9, banks=8) for chip, largest in ((0, 1500 * 10 ** 6),
                                                                                         (1, 400 * 10 ** 6))]
        ledger.phase('prefill', point='before prompt=120000')
        before = gate.before_points(rendered(lines))
        self.assertEqual(before['points'], 2)
        point = before['floor_point']
        self.assertEqual((point['op'], point['chip'], point['estimate_gb'], point['margin_gb']), ('prefill', 1, 0.3, 0.1))

    def test_w6d_before_points(self):
        memory_ledger = producer('memory_ledger', 'BEFORE_MARKER')
        if memory_ledger is None:
            self.skipTest('W6 (s2/w6-memory) is not merged into this checkout')
        self.assertEqual((memory_ledger.LINE_BUDGET, memory_ledger.BEFORE_MARKER), (LEDGER_LINE_BUDGET,
                                                                                   '[MEMLEDGER] before op='))
        lines = []
        ledger = memory_ledger.MemoryLedger(None, None, log=lines.append, emit=lambda text: None)
        ledger.reading = lambda: [dict(chip=0, largest_free=300 * 10 ** 6, free=1200 * 10 ** 6, allocated=30 * 10 ** 9,
                                       total=32 * 10 ** 9, banks=8),
                                  dict(chip=1, largest_free=900 * 10 ** 6, free=1300 * 10 ** 6, allocated=30 * 10 ** 9,
                                       total=32 * 10 ** 9, banks=8)]
        ledger.trace_reading = lambda: [dict(chip=chip, allocated=268400000, largest_free=12500000, free=0, total=0,
                                             banks=1) for chip in (0, 1)]
        ledger.before('engine', estimate=ENGINE_PEAK, point='req=%s' % engine_id(0)[-12:])
        ledger.trace_reading = lambda: {'unavailable': 'no TRACE view'}
        ledger.before('quad', estimate=QUAD_ESTIMATE, point='slots=0,1,2,3')
        ledger.reading = lambda: {'unavailable': 'no statistics'}
        ledger.before('pair', estimate=44 * 10 ** 6, point='slots=0,1')
        text = rendered(lines)
        before = gate.before_points(text)
        self.assertEqual((before['points'], before['unread'], before['by_op']), (4, 1, dict(engine=2, quad=2)))
        self.assertEqual(before['floor_gb'], -0.7, 'W6\'s own margin below zero, never passed over')
        self.assertEqual(before['logged_floor_gb'], round(min(ledger.floor.values()) / 1e9, 4))
        region = gate.trace_region(text)
        self.assertEqual((region['readings'], region['unavailable'], region['min_largest_free_gb']), (2, 2, 0.0125))

    def test_w6b_dram_hold_through_the_wrapper(self):
        admission = producer('serving_prefill_admission', 'DRAM_HOLD_LINE')
        if admission is None:
            self.skipTest('W6 (s2/w6-memory) is not merged into this checkout')
        self.assertEqual((admission.DRAM_HOLD_LINE, admission.DRAM_RELEASED_LINE, admission.DRAM_LIFTED_LINE,
                          admission.DRAM_UNAVAILABLE_LINE, admission.DRAM_HOLD),
                         (W6_HOLD_LINE, W6_RELEASED_HOLD_LINE, W6_LIFTED_LINE, W6_UNAVAILABLE_LINE,
                          gate.DRAM_HOLD_MARKER))
        if hasattr(admission, 'DRAM_DEFERRED_LINE'):
            self.assertEqual((admission.DRAM_DEFERRED_LINE, admission.DRAM_CARRIED_LINE),
                             (W6_DEFERRED_LINE, W6_CARRIED_LINE))
        self.assertEqual(admission.engine_build_peak(), ENGINE_PEAK)

        class Queue(list):
            def prepend_requests(self, other):
                self[:0] = list(other)

            def peek_request(self):
                return self[0]
        lines, fits = [], [False]
        wrapped = admission.wrap(lambda scheduler: None, queue_factory=lambda scheduler: Queue(),
                                 log=lambda message, *values: lines.append(message.format(*values)))
        request = types.SimpleNamespace(request_id='r5', num_prompt_tokens=120000)
        scheduler = types.SimpleNamespace(waiting=Queue([request]), max_num_running_reqs=4, finished_req_ids=set(),
                                          running=[types.SimpleNamespace(is_prefill_chunk=False)] * 4)
        holder = types.SimpleNamespace(admits=lambda prompt: (fits[0], dict(largest_free=700 * 10 ** 6 if not fits[0]
                                                                             else 1700 * 10 ** 6, need=1568400000)))
        healthy = dict(floor_gb=1.0, by_op=dict(engine=8))

        def judged():
            hold = gate.dram_holds(rendered(lines))
            return hold, driver.memory_s2_checks('arm', dict(s2=dict(dram_hold=hold, before=healthy,
                                                                     extent_replay=True)), 4)[0]
        from unittest import mock
        with mock.patch.dict(sys.modules, {admission.DRAM_KEY: holder}):
            wrapped(scheduler)                       # every seat decoding: the fifth prompt waits for a seat
            every_seat = judged()
            scheduler.running = scheduler.running[:3]
            scheduler.finished_req_ids = {'cmpl-gone'}
            wrapped(scheduler)                       # a seat freed on a stale reading: deferred (W6 71d9aded) or held
            stale = judged()
            scheduler.finished_req_ids = set()
            wrapped(scheduler)                       # a fresh reading, still no fit: a hold with a seat free
            fresh = judged()
            fits[0] = True
            wrapped(scheduler)                       # fits: released
        self.assertEqual(every_seat[1], [], 'no fit failed while every seat decodes')
        if hasattr(admission, 'DRAM_DEFERRED_LINE'):
            self.assertEqual((stale[0]['holds'], stale[0]['deferred'], stale[1]), (0, 1, []), 'deferred: no hold')
        self.assertEqual(fresh[0]['hold_decodes'][-1], 3)
        self.assertTrue(fresh[1] and 'decodes [3] of 4 seats' in fresh[1][0], fresh[1])
        self.assertEqual(judged()[0]['released'], 1)

    def test_w6c_release_line(self):
        coordinator = producer('dflash_packed_proposal_coordinator', 'RELEASED_LINE')
        if coordinator is None:
            self.skipTest('W6 (s2/w6-memory) is not merged into this checkout')
        self.assertEqual(coordinator.RELEASED_LINE, W6_RELEASED_LINE)
        line = coordinator.RELEASED_LINE.format(quad=1, pairs=[list(group) for group in ((0, 1), (2, 3))])
        match = gate.RELEASED_LINE.search(line)
        self.assertEqual((match.group(1), gate.pair_count(match.group(2))), ('1', 2))

    def test_w6a_proposal_ladder(self):
        factory = producer('serving_request_factory', 'single_bucket_contexts')
        if factory is None:
            self.skipTest('W6 (s2/w6-memory) is not merged into this checkout')
        from unittest import mock
        found = []
        for flag in ('1', '0'):
            with mock.patch.dict(os.environ, {EXTENT: flag}):
                text = 'proposal ladder {}'.format(factory._proposal_ladder(60, 4096))
            found.append(gate.ladder_of(gate.PROPOSAL_LADDER.search(text).group(1)))
        if hasattr(factory, 'built_proposal_buckets'):
            self.assertEqual(factory.PROPOSAL_BUCKETS_BUILT, W6_BUCKETS_BUILT)
            device = types.SimpleNamespace(proposal_capture=types.SimpleNamespace(buckets={2048: object()}))
            line = (factory.PROPOSAL_BUCKETS_BUILT + '{} contexts={}').format('cmpl-x', factory.built_proposal_buckets(
                device))
            match = gate.PROPOSAL_BUCKETS_BUILT_LINE.search(line)
            self.assertEqual((match.group(1), gate.ladder_of(match.group(2))), ('cmpl-x', [2048]))
        self.assertEqual(found, [[2048], [256, 512, 1024, 2048]])

    def test_w3_extent_round_line(self):
        verifier = producer('packed_verifier', 'EXTENT_ROUND_MARKER')
        if verifier is None or not hasattr(verifier.PackedVerifierEngine, 'note_extent_round'):
            self.skipTest('W3 (s2/w3-block-step) is not merged into this checkout')
        import extent_attention_replay
        self.assertEqual((verifier.EXTENT_ROUND_MARKER, verifier.EXTENT_CAP_REFUSED_MARKER,
                          verifier.REPLAY_DEADLINE_MARKER),
                         (gate.S2_ROUND_MARKER, gate.CAP_REFUSED_MARKER, gate.DEADLINE_MARKER))
        self.assertEqual(source_literal(verifier.PackedVerifierEngine.note_extent_round, '%s round=%d live='),
                         W3_ROUND_FORMAT)
        lines = []
        engine = types.SimpleNamespace(rounds=4, rows_per_user=16, extent_counts=dict(rounds=0, cap_events=0,
                                                                                         mixed_rounds=0),
                                       accept_limit=lambda start: min(16, extent_attention_replay.extent(start) - start))
        from unittest import mock
        with mock.patch.object(verifier, 'diagnostic', lines.append):
            verifier.PackedVerifierEngine.note_extent_round(
                engine, {0: (None, 131072), 1: (None, 131080), 2: (None, 131090), 3: (None, 131320)}, [0, 1, 2, 3], [])
            verifier.PackedVerifierEngine.note_extent_round(engine, {0: (None, 20000), 1: (None, 20010)}, [0, 1], [2, 3])
            verifier.PackedVerifierEngine.note_extent_round(engine, {0: (None, 1500), 1: (None, 60000)}, [0, 1], [2, 3])
        rounds = gate.extent_rounds(rendered(lines))
        self.assertEqual((rounds['count'], rounds['multi_family_rounds']), (3, engine.extent_counts['mixed_rounds']))
        self.assertEqual((rounds['multi_family_rounds'], rounds['families_seen']), (1, [1536, 20224, 60160, 131328]))
        self.assertEqual((rounds['capped_segments'], engine.extent_counts['cap_events']), (1, 1))

    def test_w3_extent_audit_lines(self):
        verifier = producer('packed_verifier', 'EXTENT_AUDIT_MISMATCH_MARKER')
        if verifier is None or not hasattr(verifier.PackedVerifierEngine, 'audit_extent'):
            self.skipTest('W3 (s2/w3-block-step) is not merged into this checkout')
        self.assertEqual((verifier.EXTENT_AUDIT_MARKER, verifier.EXTENT_AUDIT_MISMATCH_MARKER),
                         (W3_AUDIT_MARKER, W3_MISMATCH_MARKER))
        audit = verifier.PackedVerifierEngine.audit_extent
        self.assertEqual(source_literal(audit, '%s round=%d segments='), W3_AUDIT_FORMAT)
        self.assertEqual(source_literal(audit, '%s round=%d at='), W3_MISMATCH_FORMAT)
        text = '\n'.join([round_line(9, [131328] * 4), 'INFO ' + W3_AUDIT_FORMAT % (
            verifier.EXTENT_AUDIT_MARKER, 9, 4, 4, 4, 1, 1, 2, 1.25)])
        found = gate.extent_audit(text)
        self.assertEqual((found['lines'], found['malformed'], found['incomplete'], found['median_ms']), (1, 0, 0, 1.25))
        mismatch = 'WARNING ' + W3_MISMATCH_FORMAT % (verifier.EXTENT_AUDIT_MISMATCH_MARKER, 9, 'mask:2')
        self.assertEqual(gate.extent_audit(text + '\n' + mismatch)['mismatches'], 1)


if __name__ == '__main__':
    unittest.main()
