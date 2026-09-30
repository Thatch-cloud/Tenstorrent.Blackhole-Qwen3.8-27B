"""D0, the one-user lane: a 16-row packed block a lone user's rounds run on (QWEN_FAST_SOLO_LANE, GATE ONLY).

Today a lone user has no packed block. The M3 block pads to two or three live users but not one
(packed_verifier.MAX_IDLE_SEGMENTS: page 0 has two tile rows), so a lone user decodes on the per-request 1/2/4-row
engines - at most four tokens per step, each step a full weight pass (packed_shapes.sequential_capture_rows). At TP4 that
is a third of what one T16 round commits (the 200 tok/s plan's D0). This adds ONE more packed block beside M3, built at
attach over pool slot 0, whose shape is one T16 user in 16 rows (packed_shapes.solo_shape). It is the same class as
M3 - PackedVerifierEngine, the K64j extent reader, the per-segment carry and commit traces - so nothing here is a new
kernel and no pinned source changes (test_tp2_pins).

What routes to it (serving_packed_step.PackedStep, given `solo=`): a round whose ONLY entry is the request that borrowed
pool slot 0's carry, drafted at 16 rows, at a position the extent block admits. Every other round is what it was: two
or three live users are M3's padded rounds, four its full round, and a lone user in any other slot (a survivor of a
departed user 0) still takes the per-request engines - the block binds its segment to slot 0's carry by identity
(PackedVerifierEngine.segment_of), and a second solo block per slot would cost DRAM the fast lane does not need
(serving_fast_lane reserves slot 0 for the fast request, so on the lanes profile the lone user IS in slot 0).

Why the arithmetic is the same as M3's (design LN-0, exactness E1/E2): the image serves the M3 block on native w1/w3
(QWEN_FAST_SINGLE_GATEUP=1: no w_gate_up, no FusedT16Arm), and a 16-row verify then runs the same native MLP path at
M = 16 as M3 does at M = 64. Whether every matmul row is byte-identical at M = 16 and M = 64 is a card question
(in0_block_w and the K split pinned equal): the gate's lanes-exact plan runs every user alone on this block (arm d0-solo) and
alone on the per-request engines (arm ref-solo), every audit on, and until the texts are equal the lane is unqualified.

GATE ONLY, the way the four-card extent path is (packed_any_admission.unqualified_allowed): the flag is refused
outside a gate run of a gate-only four-card profile, and an admitted attach logs every missing piece as UNQUALIFIED.

Flags (default off; the attach refuses anything but 0 or 1):
    QWEN_FAST_SOLO_LANE  1 builds the block. Admitted only at the 64-row M3 block with the padded block, the extent
                         path, single gate/up (E1), four cards, and no fused commit.
Lines (server log):
    [SOLO-LANE] admitted ...            once, at attach
    [SOLO-LANE] UNQUALIFIED ...         one per missing piece (gate only)
    [SOLO-LANE] round=<n> slot=<s> ...  the routing decisions a gate reads (serving_packed_step, once per state)
"""

import os

SOLO_LANE_FLAG = 'QWEN_FAST_SOLO_LANE'
GATE_ENV = 'QWEN_C2_GATE'
GATE_PROFILE_ENV = 'QWEN_C2_GATE_PROFILE'
# The pool slot the block is bound to: the fast request's slot on the lanes profile, and the first arrival's otherwise
# (ServingBufferPool.acquire lends the lowest free slot).
SOLO_SLOT = 0
MARKER = '[SOLO-LANE]'
ADMITTED_MARKER = MARKER + ' admitted'
UNQUALIFIED_MARKER = MARKER + ' UNQUALIFIED (gate only)'
ROUTE_MARKER = MARKER + ' round'
SKIP_MARKER = MARKER + ' skipped'
# Flags the solo block cannot take. The fused commit (QWEN_FAST_FUSED_COMMIT) builds a per-block fused-commit object over the
# attach's collectives, which the attach hands the M3 block only, and its TP4 port has not landed (the four-card profiles
# switch it off): refused, so the attach says so instead of the solo block building without what the flag promises.
# The round-fence flags the image sets (QWEN_FAST_PRESTAGE, QWEN_FAST_ROUND_FENCES, QWEN_FAST_EARLY_DRAFT,
# QWEN_FAST_GDN_AFTER_PAIRS) are NOT refused: each block reads them at construction, so the solo block is built with the
# same pre-stage, round fences and deferred commits, and PackedStep picks the block of the COMING round for the drafts' fence
# window (while_waiting) and arms and flushes every block (all_blocks); a round on the other block than the last one's bumps
# the fixture write epoch (note_switch).
UNSUPPORTED_FLAGS = ('QWEN_FAST_FUSED_COMMIT',)
# What the solo block needs of the environment, beyond what packed_any_admission already requires of the extent path:
# name, wanted value, why.
REQUIRED_ENV = (
    ('QWEN_FAST_EXTENT_REPLAY', '1', 'the solo block is an extent block: any-position attention through the K64j readers'),
    ('QWEN_FAST_PADDED_BLOCK', '1', 'the M3 block serves two and three live users beside it (padded rounds)'),
    ('QWEN_FAST_SINGLE_GATEUP', '1', 'the solo block and M3 must run one MLP arithmetic: native w1/w3 (exactness E1)'),
    ('QWEN_FAST_FOUR_AS_TWO', '0', 'one 64-row M3 block, not two 32-row blocks'),
)


def gate_run(environ=None):
    """Whether this process is a gate run of a gate-only profile: the workflow's switch (QWEN_C2_GATE=1, which the contract
    demands to boot a gate-only profile) AND the profile's own marker (QWEN_C2_GATE_PROFILE=1, set by no traffic profile).
    packed_any_admission.unqualified_allowed's rule, without its import: that module ships in the C2 overlay only, and this
    one is copied into the P8 tree too (test_solo_lane pins the two equal)."""
    environ = os.environ if environ is None else environ
    return environ.get(GATE_ENV) == '1' and environ.get(GATE_PROFILE_ENV) == '1'


def solo_lane_requested(environ=None):
    """Whether QWEN_FAST_SOLO_LANE=1, strictly: unset or '0' is off, '1' is on, anything else is a configuration error
    (never a silent off)."""
    value = (os.environ if environ is None else environ).get(SOLO_LANE_FLAG, '0')
    if value not in ('0', '1'):
        raise ValueError('%s must be 0 or 1, got %r' % (SOLO_LANE_FLAG, value))
    return value == '1'


def _log(template, *values):
    text = template.format(*values)
    try:
        from loguru import logger
    except ImportError:
        print(text, flush=True)
        return
    logger.info(template, *values)


def solo_lane_admission(m3, environ=None, *, log=None):
    """None while the flag is off (nothing else is read). Else the record of what is admitted, or ValueError naming every
    reason the attach is refused. `m3` is serving_runtime.m3_shape's (met, description).

    Refused unless: the shape is the 64-row M3 block; every REQUIRED_ENV holds; no UNSUPPORTED_FLAGS flag is on; the width
    is four cards (a two-card image has no four-card qualification to wait for, and the pair's pinned sources are not
    this lane's to widen); and the process is a gate run of a gate-only profile (packed_any_admission.
    unqualified_allowed). An admitted attach logs one UNQUALIFIED line for each piece nothing has qualified yet."""
    environ = os.environ if environ is None else environ
    if not solo_lane_requested(environ):
        return None
    log = _log if log is None else log
    import tp_shapes

    met, shape = m3
    problems = []
    if not met:
        problems.append('the solo lane is built beside the 64-row M3 block (users=4 FOUR_AS_TWO=0 PACKED_STEP=1), not %s'
                        % shape)
    for name, wanted, why in REQUIRED_ENV:
        if environ.get(name) != wanted:
            problems.append('%s=%s, not %s: %s' % (name, environ.get(name, '(unset)'), wanted, why))
    for name in UNSUPPORTED_FLAGS:
        if environ.get(name, '0') not in ('0', ''):
            problems.append('%s=%s: the solo block cannot take it (the fused commit is built for the M3 block alone)'
                            % (name, environ.get(name)))
    tp = tp_shapes.requested_tp(environ)
    if tp == tp_shapes.PAIR:
        problems.append('%s is a four-card lane (QWEN_FAST_TP=4); the pair serves its lone user on the engines it always did'
                        % SOLO_LANE_FLAG)
    if not gate_run(environ):
        problems.append('%s is admitted only in a gate run of a gate-only four-card profile (QWEN_C2_GATE=1 and '
                        'QWEN_C2_GATE_PROFILE=1): no card run has qualified it' % SOLO_LANE_FLAG)
    if problems:
        for problem in problems:
            log('{} refused: {}', MARKER, problem)
        raise ValueError('%s=1 is refused: %s' % (SOLO_LANE_FLAG, '; '.join(problems)))
    unqualified = ['M16 == M64 byte gate (every matmul row identical at 16 and 64 rows)',
                   'the solo extent reader is the M3 segment reader (one program, flags 0x23)',
                   'lone slot-0 round == the packed-2 and the per-request-engine text (audited)']
    for index, piece in enumerate(unqualified, 1):
        log('{} ({}/{}): {}', UNQUALIFIED_MARKER, index, len(unqualified), piece)
    log('{} slot={} rows=16 users=1 (gate only)', ADMITTED_MARKER, SOLO_SLOT)
    return dict(slot=SOLO_SLOT, rows=16, users=1, unqualified=unqualified)


def solo_admits(solo, request):
    """Whether `solo` can serve this request's next 16-row round: its engine borrowed the slot the block is bound to
    and its frontier is inside what the extent block admits. Never raises."""
    try:
        solo.segment_of(request.engine)
    except ValueError:
        return False
    from serving_packed_step import admitted

    return admitted(solo, request.session.position)


def solo_proposal_rows(solo, requests):
    """The ticket width of a round the solo block would serve, or None. Exactly ONE live request (finished ones aside),
    bound to the block's slot, with at least a block's worth of tokens left and a frontier the block admits - proposal_rows'
    own per-request conditions, for one user."""
    live = [request for request in requests if not request.session.finished]
    if len(live) != 1:
        return None
    request = live[0]
    rows = solo.shape.rows_per_user
    session = request.session
    if session.max_new_tokens - len(session.emitted) < rows or not solo_admits(solo, request):
        return None
    return rows


def solo_serves(solo, entries):
    """Whether this round's entries are one 16-row ticket the solo block serves: the routing decision at the step."""
    entries = list(entries)
    if len(entries) != 1:
        return False
    entry = entries[0]
    if len(entry['ticket'].tokens) != solo.shape.rows_per_user:
        return False
    try:
        solo.segment_of(entry['request'].engine)
    except ValueError:
        return False
    return True
