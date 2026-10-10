"""Octo-T8, the eight-seat adaptive round shape (QWEN_FAST_OCTO), and the lone-user packed policy (QWEN_FAST_SOLO_PACKED). BOTH GATE ONLY, both default off.

THE SHAPE. At eight seats the round is two 64-row M3 blocks (4 users x 16 rows each) back to back, and the measured 8-live round is two ~50 ms
traces, ~7 ms of host verify work, two quad drafts (~43 ms) and ~6 ms of residual: 157-161 ms at 4k, 177 ms at 33k. A seat commits 3.3-4.8 tokens
a round at 8 live and only 5% of seat-rounds emit more than 8 (capping at 8 rows keeps 0.85-0.99 of the tokens). The single-block trace costs
49.1 ms at 2 seats and 49.7 at 4: the rows are nearly free, the PASS is what costs. So at 5+ seats live the round can be ONE 64-row pass of
8 seats x 8 rows (7 proposals + the seed) instead of two passes of 4 x 16, and the second trace (~50 ms) and half the verify host work go away
for ~8 fewer committed rows per seat-round. At <= 4 live nothing changes: M3 at 16 rows per seat.

  - The rows per pass stay 64, so every matmul, norm, collective and sampler program is M3's (packed_shapes.octo_shape: 8 users x 8 rows).
  - The drafts do not change: the pair/quad draft passes run per M3 block as they do (QWEN_FAST_QUAD_DRAFT_BLOCKS=2); a seat's ticket is its
    draft cut to 7 proposals on the host (GreedySession.propose(max_rows=8), the same cut a narrowed ticket takes).
  - The block is a THIRD packed block over pool slots 0..7, SHARING their carries with blocks A and B by identity (PackedVerifierEngine.segment_of),
    exactly as the one-user D0 block shares slot 0's with M3's segment 0. Which block a round runs on is the PackedStep's policy
    (serving_packed_step.PackedStep.proposal_groups / route_octo); each switch of shape bumps the fixture write epoch (note_switch).

EXACTNESS (owner rule: STRICT). Only the target's own argmax is ever committed; speculation changes only HOW MANY tokens a round commits. The octo
round is the M3 round at another row geometry: GDN state commits at the accepted prefix (the block's per-user commit traces: 8 users x prefixes 1..8),
K/V rows past the frontier are overwritten later, the drafter's history takes verified features only (the block's taps at the segment's 8 rows), every
shape switch bumps the fixture epoch, and native GDN slot 0 stays untrusted after a packed step (verifier_engine.note_packed_step).

FLAGS (default off; strict values):
    QWEN_FAST_OCTO            off | live | alternate. `live`: every round with enough live seats runs octo. `alternate`: every ELIGIBLE round
                              switches shape (octo, m3, octo, m3, ...), so both arms are timed in pairs on one boot.
    QWEN_FAST_OCTO_MIN_LIVE   the fewest live seats a round needs for octo (default 6 = the eight segments less the two idle ones page 0 can hold).
                              5 is the design's number and is refused until a third idle target exists (IDLE below).
    QWEN_FAST_SOLO_PACKED     0 | 1. A lone live user (and two or three in one block) takes the padded 64-row T16 block instead of the 1/2/4-row
                              per-request engines.
Lines (server log): octo_markers.py.

WHAT IS BUILT AND WHAT IS NOT QUALIFIED. Every DEVICE_PIECES entry is built on the host and tested on CPU (the flag no longer refuses by name): the attach builds the
third block (serving_runtime), the batched GDN launch of the TP4 sibling takes eight users, the octo extent readers and the pool's (8, 8) extent storage serve
one eight-row group per bundle (flags 0x21), the octo block keeps its own publication warm and the fused commit counts it. NOTHING of it has run on a card. The K64j kernels and
factory are UNCHANGED (0x21 is an existing flag set; what no card has run is a bundle of one eight-row group), so there is no new graft binary. What each card job must establish is
UNQUALIFIED_ITEMS, logged one per line at every admission; the order of the jobs is references/tp4-octo-jobs/ORDER.txt (Q1 first). The admission still REFUSES by name when any
piece is unbuilt (device_gaps), a minimum live count below IDLE_CAPACITY_NOTE's limit, and any run that is not a gate run of a gate-only profile.
"""

import os
import time

OCTO_FLAG = 'QWEN_FAST_OCTO'
MIN_LIVE_FLAG = 'QWEN_FAST_OCTO_MIN_LIVE'
SOLO_PACKED_FLAG = 'QWEN_FAST_SOLO_PACKED'
MODES = ('off', 'live', 'alternate')
USERS = 8
ROWS = 8
# The design asks for 5 live seats; the block can serve a round with at most `idle_capacity()` idle segments (PackedVerifierEngine.MAX_IDLE_SEGMENTS: page
# 0 has two 32-row tile rows, and the T2 chained K/V write allows one writer per (page, tile row)), which with eight segments is six live seats.
MIN_LIVE_TARGET = 5
MIN_LIVE_DEFAULT = 6
# The pool slot count a lone user pads from: the 4-user M3 block (its lone user leaves three idle segments).
M3_USERS = 4

GATE_ENV = 'QWEN_C2_GATE'
GATE_PROFILE_ENV = 'QWEN_C2_GATE_PROFILE'
M3_BLOCKS_FLAG = 'QWEN_FAST_M3_BLOCKS'

MARKER = '[OCTO]'
ADMITTED_MARKER = MARKER + ' admitted'
UNQUALIFIED_MARKER = MARKER + ' UNQUALIFIED (gate only)'
REFUSED_MARKER = MARKER + ' refused'
ROUND_LINE = ('[OCTO] round={round} shape={shape} planned={planned} live={live} rows={rows} eligible={eligible} counted={counted} '
              'octo_rounds={octo_rounds} m3_rounds={m3_rounds} committed={committed} step_ms={step_ms:.1f} gap_ms={gap_ms:.1f}')
PROGRAMS_LINE = '[OCTO] programs shape={shape} round={round} switch={switch} before={before} after={after}'
SOLO_PACKED_MARKER = '[SOLO-PACKED]'

# What the octo block needs of the environment: name, wanted value, why.
REQUIRED_ENV = (
    ('QWEN_FAST_PACKED_STEP', '1', 'the packed step routes the shapes'),
    ('QWEN_FAST_TP', '4', 'a four-card shape; the pair serves its seats as it always did'),
    (M3_BLOCKS_FLAG, '2', 'octo is a THIRD block beside the two M3 blocks (4+ live seats and every fallback keep them)'),
    ('QWEN_FAST_FOUR_AS_TWO', '0', 'M3 blocks of 64 rows, not 32-row pairs'),
    ('QWEN_FAST_EXTENT_REPLAY', '1', 'the extent readers (K64j) serve any position at 8-row segments'),
    ('QWEN_FAST_SINGLE_GATEUP', '1', 'one MLP arithmetic at M = 64 in every shape: native w1/w3 (exactness E1)'),
    ('QWEN_FAST_QUAD_DRAFT', '1', 'the drafts stay per quad, cut on the host to 8 rows'),
    ('QWEN_FAST_QUAD_DRAFT_BLOCKS', '2', 'one quad draft per M3 block, unchanged'),
    ('QWEN_FAST_KV_RESERVATION', '1', 'the KV pool is lowered to fund the third block: the reservation admission reads that pool'),
)
# Flags the octo block cannot sit beside.
EXCLUDED_FLAGS = (
    ('QWEN_FAST_SOLO_LANE', 'the D0 one-user block is built beside ONE M3 block over slot 0 (serving_solo_lane refuses two)'),
    ('QWEN_FAST_LANE', 'the fast lane rides on the D0 block'),
)
# Flags whose OFF values are more than '0' (sdpa_long_tp.OFF_VALUES): refused when set to anything else. The long-context SDPA configurations are qualified at flags 0x23 (G8B2)
# only, and sdpa_multi_tp tiles 16-row users: neither takes an octo (0x21, eight-row) segment.
EXCLUDED_OFF_VALUES = (
    ('QWEN_FAST_TP4_SDPA', ('', '0', 'off'), 'the long-context SDPA configurations (sdpa_long_tp, sdpa_multi_tp) are qualified at 0x23 and 16-row users, not the octo 0x21 eight-row segment'),
)

# The device pieces the octo block needs, by key -> what must exist (a requirement, so a refusal reads right when a piece is missing). `device_gaps` lists every key not built;
# the admission refuses while any is. BUILT (below) declares them; the tests (test_octo, test_octo_block, test_extent_attention_octo_tp, test_octo_attach) are what make each claim true.
DEVICE_PIECES = (
    ('attach-build', 'serving_runtime must build the third packed block: PackedVerifierEngine(shape=octo_shape, pool_slots=0..7, defer_capture) built after blocks A and B and '
                     'captured with them in complete_blocks_two_phase (its fixture, taps, checkpoints and extent words allocated and warmed BEFORE any capture), over extent '
                     'storage the pool lends for a (8, 8) shape (ServingBufferPool packed_shapes / packed_replicas), carries_in_place proven, bound into the PackedStep with its OctoState'),
    ('gdn-batch', 'the batched GDN launch must carry eight users: gdn_user_batch_tp.MAX_USERS (the unpinned TP4 sibling; the pair\'s gdn_user_batch is hash-pinned at 4) is 8, so '
                  'one launch is 8 users x 12 heads = 96 of the 110 cores, one wave'),
    ('attention-8row', 'the extent readers and the pool must take an eight-row segment as ONE group per bundle (K64j G8B1, flags 0x21: tail | extent, no KV share): '
                       'extent_attention_octo_tp.OctoSegmentReader / build_packed behind the fold twin, ServingBufferPool extent storage for the (8, 8) shape (extent_bundle_batches(8, 8) is (1,)), '
                       'the block fold taking bundles of one; the K64j binary and kernels are unchanged and the pinned four-card reader untouched'),
    ('publication-8row', 'the octo block must keep its own publication warm: complete_blocks_two_phase skips it for every block after the first except the octo block, whose plan '
                         '(publication_warm.plan) is 64 packed shapes of which 32 no M3 warm ran (segment offsets 8/24/40/56 x prefixes 1..8), so the first octo round compiles nothing'),
    ('fused-third-block', 'the fused commit must be built over a third block: 8 T_proj traces and 64 slide traces (fused_commit_tp) and the pre-stage state of a third fixture '
                          '(PackedStep includes the block in verify_prestage.engage_two_block)'),
)

# Built pieces, by key: the declaration the admission reads (and 'gdn-batch' is also read from the module's own limit, so it cannot be claimed while the code says otherwise).
BUILT = {'attach-build', 'gdn-batch', 'attention-8row', 'publication-8row', 'fused-third-block'}

# What no card has qualified, one line each at every admission: (what must hold, the card job that establishes it). The first four are the strict-exactness ones.
UNQUALIFIED_ITEMS = (
    ("octo rounds == M3 rounds == the per-request text (strict greedy exactness)", 'A1 E1 E2'),
    ("K64j one eight-row group per bundle (0x21, G8B1) == the native one-row decode", 'Q1w Q1a Q1b'),
    ("the 64-row block at 8 users x 8 rows: matmul rows == the 4 x 16 block's", 'E1'),
    ("a shape switch compiles nothing (publication warm at 8-row offsets)", 'A1 H1'),
    ("the batched GDN launch at 8 users (96 cores, one wave) == four users", 'Q2w Q2'),
    ("the fused commit of a third block, and its pre-stage", 'A1 E1'),
)

# An idle segment writes its K/V to page 0 and page 0 has two 32-row tile rows (PackedVerifierEngine.MAX_IDLE_SEGMENTS = 2; the T2 chained K/V write allows one
# writer per (page, tile row)): octo serves 6 or 7 live seats, not the design's 5, and a lone user cannot pad the 4-user block. A third idle target (a reserved
# scratch page) is a card question; both rules below read the block's own constant, so they lift the day it changes.
IDLE_CAPACITY_NOTE = 'page 0 has two tile rows (PackedVerifierEngine.MAX_IDLE_SEGMENTS)'


def gate_run(environ=None):
    """Whether this process is a gate run of a gate-only profile: the workflow's switch (QWEN_C2_GATE=1) AND the profile's own marker (QWEN_C2_GATE_PROFILE=1).
    serving_solo_lane.gate_run's rule, without its import (test_octo pins the two equal)."""
    environ = os.environ if environ is None else environ
    return environ.get(GATE_ENV) == '1' and environ.get(GATE_PROFILE_ENV) == '1'


def octo_mode(environ=None):
    """QWEN_FAST_OCTO, strictly: unset is off; off, live and alternate are the values; anything else - an empty value included - is a configuration error
    naming the flag (never a silent off)."""
    value = (os.environ if environ is None else environ).get(OCTO_FLAG, 'off')
    if value not in MODES:
        raise ValueError('%s must be one of %s, got %r' % (OCTO_FLAG, '|'.join(MODES), value))
    return value


def octo_min_live(environ=None):
    """QWEN_FAST_OCTO_MIN_LIVE: the fewest live seats a round needs for the octo shape, a decimal integer from 1 to 7 (default MIN_LIVE_DEFAULT)."""
    text = (os.environ if environ is None else environ).get(MIN_LIVE_FLAG, str(MIN_LIVE_DEFAULT))
    if type(text) is not str or not text.isdigit() or text != str(int(text)) or not 1 <= int(text) < USERS:
        raise ValueError('%s must be a decimal integer from 1 to %d, got %r' % (MIN_LIVE_FLAG, USERS - 1, text))
    return int(text)


def solo_packed_requested(environ=None):
    """QWEN_FAST_SOLO_PACKED=1, strictly: unset or '0' is off, '1' is on, anything else is a configuration error."""
    value = (os.environ if environ is None else environ).get(SOLO_PACKED_FLAG, '0')
    if value not in ('0', '1'):
        raise ValueError('%s must be 0 or 1, got %r' % (SOLO_PACKED_FLAG, value))
    return value == '1'


def idle_capacity():
    """The most idle segments one packed block can hold: PackedVerifierEngine.MAX_IDLE_SEGMENTS (2: page 0's two tile rows)."""
    try:
        from packed_verifier import PackedVerifierEngine

        value = getattr(PackedVerifierEngine, 'MAX_IDLE_SEGMENTS', None)
    except ImportError:
        value = None
    # (a class a test replaced reports no integer: the documented two)
    return value if type(value) is int else 2


def gdn_batch_users():
    """The users one batched GDN launch carries at four cards (gdn_user_batch_tp.MAX_USERS, the unpinned TP4 sibling: 8 users x 12 heads = 96 of the 110 cores; the pair's
    hash-pinned gdn_user_batch keeps 4), or 0 when the module cannot be read."""
    try:
        import gdn_user_batch_tp

        return int(gdn_user_batch_tp.MAX_USERS)
    except ImportError:
        return 0


def device_gaps():
    """[(key, what remains)] of DEVICE_PIECES not built. 'gdn-batch' reads the pinned module's own limit, so it cannot be claimed built while the code still
    says otherwise; the rest are built when BUILT names them."""
    gaps = []
    for key, text in DEVICE_PIECES:
        if key == 'gdn-batch':
            built = key in BUILT and gdn_batch_users() >= USERS
        else:
            built = key in BUILT
        if not built:
            gaps.append((key, text))
    return gaps


def _log(template, *values):
    text = template.format(*values)
    try:
        from loguru import logger
    except ImportError:
        print(text, flush=True)
        return
    logger.info(template, *values)


def emit(text):
    """One line into the server log: loguru where it exists, stdout otherwise. Never raises."""
    try:
        try:
            from loguru import logger
        except ImportError:
            print(text, flush=True)
        else:
            logger.info('{}', text)
    except BaseException:
        pass


def octo_admission(m3, environ=None, *, log=None):
    """None while the flag is off (nothing else is read). Else the record of what is admitted, or ValueError naming EVERY reason the attach is refused. `m3` is
    serving_runtime.m3_shape's (met, description).

    Refused unless: the shape is the two-M3-block eight-seat shape; every REQUIRED_ENV holds; no EXCLUDED_FLAGS flag is on; the minimum live count leaves no more idle
    segments than a block holds; the process is a gate run of a gate-only profile; and no DEVICE_PIECES piece is missing (`device_gaps`)."""
    environ = os.environ if environ is None else environ
    mode = octo_mode(environ)
    if mode == 'off':
        return None
    log = _log if log is None else log
    minimum = octo_min_live(environ)
    met, shape = m3
    problems = []
    if not met:
        problems.append('octo-T8 is built beside the two 64-row M3 blocks of eight seats (users=8 FOUR_AS_TWO=0 PACKED_STEP=1 M3_BLOCKS=2), not %s' % shape)
    for name, wanted, why in REQUIRED_ENV:
        if environ.get(name) != wanted:
            problems.append('%s=%s, not %s: %s' % (name, environ.get(name, '(unset)'), wanted, why))
    for name, why in EXCLUDED_FLAGS:
        if environ.get(name, '0') not in ('0', ''):
            problems.append('%s=%s: octo cannot sit beside it (%s)' % (name, environ.get(name), why))
    for name, off_values, why in EXCLUDED_OFF_VALUES:
        if environ.get(name, '').strip().lower() not in off_values:
            problems.append('%s=%s: octo cannot sit beside it (%s)' % (name, environ.get(name), why))
    idle = USERS - minimum
    capacity = idle_capacity()
    if idle > capacity:
        problems.append('%s=%d leaves %d idle segments and a block holds %d (%s): the design\'s %d live seats need a third idle target'
                        % (MIN_LIVE_FLAG, minimum, idle, capacity, IDLE_CAPACITY_NOTE, MIN_LIVE_TARGET))
    if not gate_run(environ):
        problems.append('%s is admitted only in a gate run of a gate-only four-card profile (QWEN_C2_GATE=1 and QWEN_C2_GATE_PROFILE=1): no card run has qualified it'
                        % OCTO_FLAG)
    for key, text in device_gaps():
        problems.append('device piece %s is not built: %s' % (key, text))
    if problems:
        for problem in problems:
            log('{}: {}', REFUSED_MARKER, problem)
        raise ValueError('%s=%s is refused: %s' % (OCTO_FLAG, mode, '; '.join(problems)))
    unqualified = ['%s [job %s]' % item for item in UNQUALIFIED_ITEMS]
    for index, piece in enumerate(unqualified, 1):
        log('{} ({}/{}): {}', UNQUALIFIED_MARKER, index, len(unqualified), piece)
    log('{} mode={} rows={} users={} min_live={} (gate only)', ADMITTED_MARKER, mode, ROWS, USERS, minimum)
    return dict(mode=mode, rows=ROWS, users=USERS, min_live=minimum, unqualified=unqualified)


def solo_packed_admission(m3, environ=None, *, log=None):
    """None while QWEN_FAST_SOLO_PACKED is off. Else the record (min_users 1: the padded block is built to serve a lone live user), or ValueError naming every reason.

    A lone user in the 4-user M3 block leaves THREE idle segments. Page 0 holds two (PackedVerifierEngine.MAX_IDLE_SEGMENTS), and the block refuses a
    padded_min_users below users - 2 at construction. So the lone user is refused until a third idle target exists (the same limit octo's 5-live
    round meets, IDLE_CAPACITY_NOTE); two or three live users in a block are the padded rounds QWEN_FAST_PADDED_BLOCK=1 already serves, which this flag does not need."""
    environ = os.environ if environ is None else environ
    if not solo_packed_requested(environ):
        return None
    log = _log if log is None else log
    met, shape = m3
    problems = []
    if not met:
        problems.append('the lone-user padded round is the 64-row M3 block\'s (users=4 or 8 with M3_BLOCKS=2, FOUR_AS_TWO=0, PACKED_STEP=1), not %s' % shape)
    for name, wanted, why in (('QWEN_FAST_PACKED_STEP', '1', 'the packed step routes the rounds'),
                              ('QWEN_FAST_TP', '4', 'a four-card shape'),
                              ('QWEN_FAST_EXTENT_REPLAY', '1', 'the extent block\'s idle segments are ordinary users at start 0 or 32'),
                              ('QWEN_FAST_PADDED_BLOCK', '1', 'the block must be built to serve padded rounds'),
                              ('QWEN_FAST_SINGLE_GATEUP', '1', 'one MLP arithmetic at M = 64')):
        if environ.get(name) != wanted:
            problems.append('%s=%s, not %s: %s' % (name, environ.get(name, '(unset)'), wanted, why))
    if environ.get('QWEN_FAST_SOLO_LANE', '0') not in ('0', ''):
        problems.append('QWEN_FAST_SOLO_LANE=%s: the D0 one-user block and the padded lone round are two answers to one question' % environ.get('QWEN_FAST_SOLO_LANE'))
    idle = M3_USERS - 1
    capacity = idle_capacity()
    if idle > capacity:
        problems.append('a lone user leaves %d idle segments of the 4-user block and a block holds %d (%s): the block refuses padded_min_users '
                        'below %d at construction' % (idle, capacity, IDLE_CAPACITY_NOTE, M3_USERS - capacity))
    if not gate_run(environ):
        problems.append('%s is admitted only in a gate run of a gate-only four-card profile (QWEN_C2_GATE=1 and QWEN_C2_GATE_PROFILE=1)' % SOLO_PACKED_FLAG)
    if problems:
        for problem in problems:
            log('{}: {}', SOLO_PACKED_MARKER + ' refused', problem)
        raise ValueError('%s=1 is refused: %s' % (SOLO_PACKED_FLAG, '; '.join(problems)))
    log('{} admitted min_users=1 (gate only)', SOLO_PACKED_MARKER)
    return dict(min_users=1)


def program_count(block):
    """The mesh's program-cache entry count as the block's model reports it, or None where nothing reports it (a test double). Never raises."""
    try:
        count = getattr(getattr(getattr(block, 'model', None), 'mesh_device', None), 'num_program_cache_entries', None)
        return int(count()) if callable(count) else None
    except Exception:
        return None


class OctoState:
    """The octo policy's process state, host only: the mode, the alternation, the per-shape round counters and the lines.

    plan() is pure with respect to the alternation (an early draft may be discarded and drafted again): the turn only advances in finish(), when a round that
    was ELIGIBLE and ran as PLANNED completes. A round that was not eligible, or that fell back to the other shape, consumes no turn, so the counted rounds of
    an alternate boot strictly alternate octo, m3, octo, m3 (octo_judge checks it from the log)."""

    def __init__(self, mode, min_live, *, clock=None, programs=None, log=None):
        if mode not in ('live', 'alternate'):
            raise ValueError('An octo state needs mode live or alternate, got %r' % (mode,))
        if type(min_live) is not int or not 1 <= min_live < USERS:
            raise ValueError('min_live must be an integer from 1 to %d, got %r' % (USERS - 1, min_live))
        self.mode, self.min_live = mode, min_live
        self.clock = time.perf_counter if clock is None else clock
        self.programs = programs
        self.log = emit if log is None else log
        self.next_shape = 'octo'          # alternate: the shape of the next eligible round
        self.planned, self.plan_eligible = 'm3', False
        self.last_ran = None              # the last packed shape that ran ('octo' or 'm3'), for the programs line
        self.announced = None             # the last packed shape announced or run, for the epoch bump
        self.rounds = dict(octo=0, m3=0, seq=0)
        self.counted = dict(octo=0, m3=0)
        self.round = 0
        self.last_end = None
        self.started = self.before = None

    def plan(self, eligible):
        """The shape the coming round's tickets are drafted for: 'octo' or 'm3'. Not eligible is always 'm3'."""
        eligible = bool(eligible)
        shape = 'm3'
        if eligible:
            shape = 'octo' if self.mode == 'live' else self.next_shape
        self.planned, self.plan_eligible = shape, eligible
        return shape

    def begin(self):
        """Called as a round starts: its start time and the program-cache count before it."""
        self.started = self.clock()
        self.before = self.programs() if callable(self.programs) else None

    def finish(self, ran, *, live, rows, committed):
        """Called when a round ended, `ran` its shape ('octo', 'm3' or 'seq'). Counts it, advances an alternation, writes its lines. Returns the line fields."""
        if ran not in ('octo', 'm3', 'seq'):
            raise ValueError('A round ran as octo, m3 or seq, got %r' % (ran,))
        ended = self.clock()
        started = ended if self.started is None else self.started
        gap_ms = 0.0 if self.last_end is None else max(0.0, (started - self.last_end) * 1000)
        step_ms = max(0.0, (ended - started) * 1000)
        self.last_end = ended
        after = self.programs() if callable(self.programs) else None
        eligible = self.plan_eligible
        counted = eligible and ran == self.planned
        self.round += 1
        self.rounds[ran] += 1
        if counted:
            self.counted[ran] += 1
            if self.mode == 'alternate':
                self.next_shape = 'm3' if ran == 'octo' else 'octo'
        fields = dict(round=self.round, shape=ran, planned=self.planned, live=live, rows=rows, eligible=int(eligible), counted=int(counted),
                      octo_rounds=self.rounds['octo'], m3_rounds=self.rounds['m3'], committed=committed, step_ms=step_ms, gap_ms=gap_ms)
        self.log(ROUND_LINE.format(**fields))
        if ran != 'seq' and ran != self.last_ran:
            self.log(PROGRAMS_LINE.format(shape=ran, round=self.round, switch=int(self.last_ran is not None),
                                          before='None' if self.before is None else self.before, after='None' if after is None else after))
            self.last_ran = ran
        self.started = self.before = None
        return fields

    def switched(self, shape):
        """Whether `shape` ('octo' or 'm3') differs from the last packed shape announced or run; records it. The epoch bump of a switch (PackedStep.note_switch)."""
        previous, self.announced = self.announced, shape
        return previous is not None and previous != shape
