"""One device step for every packed user: one verify trace, then each user's own commit.

`serving_packed_bridge.execute_packed_decode` takes the device step as a parameter, and
`serving_sequential_step.sequential_packed_step` is the correct-but-slow one: one pass
over the 19.92 GB of dense weights per user per round. This step serves every admitted
user from ONE pass, through `packed_verifier.PackedVerifierEngine`: the block stages each
user's tokens, positions and pages into that user's segment, runs its one trace and hands
back each user's predictions; each user then decides and publishes exactly as
`serving_fast_request.FastRequest.step` does, through `VerifierEngine.adopt_packed`,
whose publication is the block's per-user commit into that user's own carry.

Same interface and contract as the sequential step: entries in the SCHEDULER's order,
one CommittedOutput per entry in that order, each carrying its entry's request id.
Predictions, taps and commits are indexed by the entry's SEGMENT - the pool slot whose
carry its engine borrowed (`PackedVerifierEngine.segment_of`, returned per entry as
`metrics['segments'][i]`) - never by its index in the entries: probe 35436807668 saw the
pair presented as ['B', 'A'].

Nothing here is written for two users: the block's shape (packed_shapes.PackedShape) says
how many entries a round takes - two T16 users in the 32-row M1 block, four in the 64-row
M3 block - and each entry's draft proposal stays its own pass in the hook (one per bridge,
serving_worker_hook._drafts), before this one verify.

A round the block cannot serve goes to the sequential step WHOLE: a ticket narrower than
the block's rows per user (greedy_session narrows the last block of a budget; under
QWEN_FAST_BUDGET_CAP it does not: a user with any budget left keeps the block's width and its
commit alone is cut at the budget, commit_limit), more or
fewer entries than the block's users (the survivors after a partner finished - one of two,
or one to three of four), an engine the block was not captured against, or a cancellation
already raised before any device work. The two are never mixed within a round. Either way
native GDN slot 0 is trusted by nobody afterwards (`verifier_engine.note_packed_step`):
the sequential step restores each user's carry first, and the packed trace restores every
segment's carry inside itself.

S2 D1 (survivor narrowing): a round drafted at the block's width that the block then does not
serve - a partner aborted after the drafts (vLLM tells the worker one step late), or a padded
check failed at the step - holds tickets its engines never captured (beside the 64-row block
they capture 1, 2 and 4 rows). Each such ticket is cut to the widest width its own engine
serves (`narrow_round`, GreedySession.narrow; one NARROWED_MARKER line per request) and the
round goes to the sequential step like any other. Only a ticket nothing narrower serves still
refuses the round (`refuse_round`), and under the D2 request quarantine its requests then end
as FINISHED_ABORTED in the same engine step instead of failing the engine one step later.

Every user's commit runs in entries order; the last one is the round's fence. The block
knows it is last - `PackedVerifierEngine.commit_user` fences exactly the commit that
empties its pending segments - so this step adds nothing for that beyond committing every
segment, even on cancellation (prefix 0) or after a failure (prefix 0 for what was left).

S2 W4 (the extent block, packed_verifier's S2 paragraph): which positions a block serves is the
block's own `admits` (any start with 128 <= start, start + rows <= C on the extent block; its one
family otherwise), asked before drafting (proposal_rows), for the padded skip line and - extent only
- at the step (ineligible, so a ticket outside it is narrowed rather than refused by the verify).
Each user commits at most the block's `accept_limit` rows (commit_limit, GreedySession.commit
max_rows): the extent path's rows at or past their family end are never committed, so the block's
own backstop in commit_user is unreachable. QWEN_FAST_GATE_FORCE_CAP (gate only) caps every packed
commit lower still, for the ticket-width arm (design Q5). A block without the methods (every block
before S2) is asked exactly what it was before.

Octo-T8 (QWEN_FAST_OCTO=live|alternate, GATE ONLY, default off; serving_octo): beside the two M3 blocks a THIRD packed block of eight users x eight
rows (packed_shapes.octo_shape) over pool slots 0..7, sharing their carries by identity. When enough seats are live (serving_octo.octo_min_live) and the block
admits every member, the coming round's tickets are drafted at EIGHT rows for it (`proposal_groups`: one group, the octo block, every live request) and the
round runs as ONE 64-row pass (`route_octo`) in place of the two M3 passes; otherwise the groups, the tickets and the step are exactly the two-block ones.
Under `alternate` the shape switches on every eligible round. A round whose tickets are not all octo-width, or that the octo block does not serve at the
step, goes to `packed_device_step`'s narrowing and the sequential step like any other. Each switch of shape bumps the fixture write epoch (`announce_shape`,
before the drafts' fence window, exactly as D0's `announce_round`). With no octo block (the default) none of this exists: `octo` is None and every path below
is what it was.

Variable-user rounds (M2, QWEN_FAST_PADDED_BLOCK=1 at the 64-row block, default off): a block
built with `padded_min_users` also serves padded_min_users <= n < users live requests as one
pass, the other segments idle (packed_verifier.py, VARIABLE-USER ROUNDS). proposal_rows drafts
such a round at the block's width once every live request passes the usual per-request checks
AND the padded ones (`padded_refusal`: the idle slot rule and no page 0 in a live table's used
range - a page-0 hit drafts the round narrow, fail closed); kv_guard fills the idle segments
from the block's idle_inputs before the T2 tile-row check, so it never skips a padded round;
ineligible repeats all of it at the step. Every round of that many live users that goes
sequential instead logs why, and whether it was eligible (`note_padded_skip`). Nothing else
here changes: the verify stages the idle segments and commits them at prefix 0 itself, and
every entry of the round is a live one, committed as always. Without the flag no path below
differs from before it existed.
"""

import os
import time
from types import SimpleNamespace

from attention_mask_replay import validate_ticket
from serving_fast_request import CommittedOutput, budget_cap_enabled
from serving_sequential_step import describe as describe_sequential, sequential_packed_step
from serving_worker_hook import note_fixture_writer, phase
import memory_ledger
import round_host
import stall_watch
import trace_census
import verifier_engine
import w2_switch
from verify_prestage import entry_line, hostgap_log_enabled

# Under QWEN_FAST_PACKED_AUDIT=1, one line per user per round for the token-exact gate
# (docs/packed-device-step-plan-2026-09-20.md section 6): which segment served which
# request at which position, what it accepted and emitted, and the first predictions.
AUDIT_LINE = ('[PACKED] request={request} segment={segment} position={position} '
              'prefix={prefix} emitted={emitted} predictions={predictions}')
# S2 W4: appended to AUDIT_LINE whenever the round's commit had a cap (commit_limit: an extent block,
# or the gate-only forced cap), so a family block's line is byte for byte what it was.
AUDIT_CAP = ' cap={cap}'
# QWEN_FAST_BUDGET_CAP: one line per commit the remaining budget cut (commit_limit's `budget`).
BUDGET_CAP_MARKER = '[PINDIAG] packed budget cap'
BUDGET_CAP_LINE = BUDGET_CAP_MARKER + ' request={request} segment={segment} position={position} remaining={remaining} limit={limit}'
# QWEN_FAST_GATE_FORCE_CAP (S2 design Q5, M4's forced-cap arm; GATE ONLY, unset by default): every
# packed commit capped at this many rows, to show that commit granularity does not change the text.
FORCE_CAP_FLAG = 'QWEN_FAST_GATE_FORCE_CAP'
FORCE_CAP_MARKER = '[PINDIAG] packed forced cap='

# Under QWEN_FAST_PACKED_AUDIT=1, one line per ROUND (not per user, unlike AUDIT_LINE
# above): the packed_commit phase's host wall time split into commit_entry's own
# sub-phases, one array entry per entry in the round's scheduler order - adopt_ms
# (engine.adopt_packed), session_ms (session.commit or session.abort, which is where
# DFlashRequestRuntime.publish, VerifierEngine.publish and
# packed_verifier.PackedVerifierEngine.commit_user all run), block_ms (the portion of
# session_ms that was packed_verifier.PackedVerifierEngine.commit_block_ms for that
# entry's segment - RetainedGDNBlock.commit_user's own host cost, gdn_records.py,
# dominated by validate_bindings' native-buffer address re-checks) and other_ms
# (whatever of commit_entry's own wall time neither adopt_ms nor session_ms accounts
# for: CommittedOutput construction, the [PACKED] audit line above, record_timing).
COMMIT_HOST_LINE = ('[PACKED-COMMIT-HOST] round={round} adopt_ms=[{adopt}] block_ms=[{block}] '
                     'session_ms=[{session}] other_ms=[{other}]')

# Under QWEN_FAST_PACKED_AUDIT=1, one line per ROUND breaking session_ms's own host cost
# further, into DFlashRequestRuntime.publish's own named stages (dflash_request_runtime.py:
# publish, its publication_stage(name, prefix) seam - a nullcontext there by default, timed
# here instead): 'features' (VerifierEngine.verified_features_for_publication), 'prepare_
# history' (DFlashDevice.prepare_publication - project_features, the history concat/pad/
# copy, DraftKVHistory.prepare), 'publish_target' (VerifierEngine.publish, which is where
# block_ms's own packed_verifier.PackedVerifierEngine.commit_user runs), and 'commit_
# history' (DFlashDevice.commit_publication, a pointer swap). One array entry per entry in
# the round's scheduler order, 0.0 for an entry whose accepted prefix was 0 (publish()
# skips 'features'/'prepare_history' entirely then, per dflash_request_runtime.py:78).
PUBLISH_STAGES = ('features', 'prepare_history', 'publish_target', 'commit_history')
PUBLISH_LINE = '[PACKED-PUBLISH] round={round} stages={stages}'


def audit_enabled():
    return os.environ.get('QWEN_FAST_PACKED_AUDIT') == '1'


def audit_log(message, **values):
    """Short lines: the log capture truncates around 250 characters."""
    try:
        from loguru import logger
    except ImportError:
        print(message.format(**values), flush=True)
        return
    logger.info(message, **values)


class _StageTimer:
    """One with-block's wall time, written into `sink[name]` on exit - the
    publication_stage(name, prefix) seam's real implementation, installed only for the
    span of one commit_entry() call (install_stage_timer below)."""
    __slots__ = ('name', 'sink', 'started', 'context')

    def __init__(self, name, sink, context=None):
        self.name, self.sink, self.context = name, sink, context

    def __enter__(self):
        # QWEN_FAST_SEQ_STAGE_LOG: the stage's 'begin' line, flushed before it runs (trace_census.stage), so a hang inside a
        # publication stage names it; context is the runtime's (request id, rows), None when the flag is off.
        if self.context is not None:
            trace_census.stage(*self.context, self.name)
        self.started = time.perf_counter()
        return self

    def __exit__(self, *exc_info):
        self.sink[self.name] = (time.perf_counter() - self.started) * 1000
        return False


def install_stage_timer(runtime, sink):
    """Rebind runtime.publication_stage (an instance attribute, shadowing
    DFlashRequestRuntime's own nullcontext-returning method) so every `with self.
    publication_stage(name, prefix):` inside publish() times itself into `sink[name]`.
    Returns a restore callable. Safe on any object - including a fake runtime whose
    class defines no publication_stage at all - since this only ever sets and later
    deletes an INSTANCE attribute."""
    if 'publication_stage' in runtime.__dict__:
        raise ValueError('runtime.publication_stage is already overridden')

    context = trace_census.runtime_context(runtime) if trace_census.stage_log_enabled() else None

    def stage(name, prefix):
        return _StageTimer(name, sink, context)

    runtime.publication_stage = stage

    def restore():
        del runtime.publication_stage

    return restore


def format_publish_stages(publish_stage_timings):
    parts = []
    for name in PUBLISH_STAGES:
        values = publish_stage_timings.get(name)
        if values is None:
            continue
        parts.append('%s: [%s]' % (name, ','.join('%.2f' % value for value in values)))
    return '{%s}' % ', '.join(parts)


def forced_cap(environ=None):
    """QWEN_FAST_GATE_FORCE_CAP (gate only): None unset, else the cap every packed commit takes at most,
    a decimal integer from 1 to 32; anything else is a configuration error (PackedStep refuses it at
    attach, before any request)."""
    import re

    text = (os.environ if environ is None else environ).get(FORCE_CAP_FLAG)
    if text is None:
        return None
    if type(text) is not str or re.fullmatch('[1-9][0-9]?', text) is None or int(text) > 32:
        raise ValueError('%s must be a row count from 1 to 32, got %r' % (FORCE_CAP_FLAG, text))
    return int(text)


def admitted(block, position):
    """Whether `block` serves a live ticket at `position`: the block's own admits when it has one
    (packed_verifier.PackedVerifierEngine: the extent path's range on an S2 block, its one family
    otherwise), else - a block without it - the family check on its replay_capacity, as before S2,
    and no check at all without that."""
    admits = getattr(block, 'admits', None)
    if callable(admits):
        return bool(admits(position))
    capacity = getattr(block, 'replay_capacity', None)
    if capacity is None:
        return True
    try:
        validate_ticket(position, block.shape.rows_per_user, capacity, short_context=False)
    except ValueError:
        return False
    return True


def commit_limit(block, ticket, budget=None):
    """How many of this ticket's rows its user may commit, or None for all of them: the block's
    accept_limit at the ticket's position (S2: min(rows, E - start) on the extent block; None on a
    family block, or a block without it), then at most QWEN_FAST_GATE_FORCE_CAP when that is set,
    then - `budget`, the tokens the request may still emit (QWEN_FAST_BUDGET_CAP; None: no budget
    cap) - at most that, by min, so a user at its last tokens commits only them while the others
    commit whole. A budget under one is a request the scheduler no longer owes a token: it raises
    (the round fails closed) rather than commit one past vLLM's budget."""
    accept = getattr(block, 'accept_limit', None)
    limit = accept(ticket.position) if callable(accept) else None
    forced = forced_cap()
    if forced is not None:
        limit = min(len(ticket.tokens) if limit is None else limit, forced)
    if budget is not None:
        if budget < 1:
            raise ValueError('A packed commit needs a budget of at least one token, got %r' % (budget,))
        if budget < (len(ticket.tokens) if limit is None else limit):
            limit = budget
    return limit


def proposal_rows(block, requests):
    """The ticket width every live request drafts for the coming round, asked by the worker
    hook before any proposal: the block's rows per user when the block will serve the round
    as one pass - every live request present (exactly the block's users, finished ones
    aside), each with at least that many tokens left, each engine bound to a segment, each
    frontier inside the block's native chunk family - else None, and each engine proposes
    at its own captured width.

    `block` is either one packed verify block or a sequence of them
    (QWEN_FAST_FOUR_AS_TWO's pair of 32-row blocks for four users, instead of the single
    64-row block). Eligibility is checked PER BLOCK: every live request must be bound to
    SOME configured block, and THAT block's own full user set must be live, each with a
    block's worth of tokens left and inside its family - so a round where one block's pair
    is intact and another's partner has already finished still answers the shared width
    here (the finished partner is not live, so its block is never even asked to be full),
    but a request bound to no configured block, or whose own block is missing one of its
    users, makes the WHOLE round None, exactly as a lone block always required all its own
    users present. Every configured block shares one rows-per-user in practice; if they
    ever did not, the mismatch answers None rather than picking one width over another.

    QWEN_FAST_BUDGET_CAP: "a block's worth of tokens left" becomes "any token left". A user near
    the end of its budget stays in the block's round at the block's width and its commit alone is
    cut at the budget (commit_limit), where it used to send the whole round, every other user
    included, down the one-pass-per-user sequential step for the tail. vLLM's sync scheduler offers
    the rows regardless of max_tokens, and the commit at a prefix within the budget is one of the
    block's existing traces, so nothing is captured or allocated for it.

    Beside the 64-row block (or the two 32-row blocks together, the same total rows) the
    per-request engines may capture only the sequential widths
    (packed_shapes.sequential_capture_rows), so a block-served round MUST be drafted at the
    block's width and a sequential round (survivors, a last narrow block) at the engines';
    a ticket drafted for a block that then does not serve it has no capture anywhere
    (`unservable`). The draft returns exactly the rows asked of it
    (dflash_request_runtime.propose), so the decision here is the ticket width.

    A block whose K/V write runs one chain per user (QWEN_FAST_VERIFY_T2 #2, `kv_chains`)
    serves the round only while no two users write one (physical page, tile row): checked
    HERE, before drafting, from each request's frontier and its engine's page table
    (`kv_shared_at_proposal`), so a conflicting round is drafted at the engines' widths and
    the exact sequential step serves it. The step's own check (`ineligible`) is the backstop.
    """
    blocks = tuple(block) if isinstance(block, (list, tuple)) else (block,)
    live = [request for request in requests if not request.session.finished]
    if not live:
        return None
    members = {}
    order = []
    for request in live:
        matched = None
        for candidate in blocks:
            try:
                candidate.segment_of(request.engine)
            except ValueError:
                continue
            matched = candidate
            break
        if matched is None:
            return None
        key = id(matched)
        if key not in members:
            members[key] = (matched, [])
            order.append(key)
        members[key][1].append(request)
    width = None
    for key in order:
        matched, group = members[key]
        shape = matched.shape
        if w2_switch.routes_sequential(matched):
            # The W2 kill switch (w2.off) latched: the round is drafted at the engines' own widths for the exact sequential step.
            return None
        padded = len(group) != shape.users and pads(matched, len(group))
        if len(group) != shape.users and not padded:
            return None
        if width is None:
            width = shape.rows_per_user
        elif width != shape.rows_per_user:
            return None
        for request in group:
            session = request.session
            if session.max_new_tokens - len(session.emitted) < (1 if budget_cap_enabled() else shape.rows_per_user):
                return None
            # The block's own admits (S2 W4): the extent path's range, or its one family.
            if not admitted(matched, session.position):
                return None
        # QWEN_FAST_PADDED_BLOCK: the idle slot rule and page 0, before the T2 tile rows (a
        # page-0 hit would otherwise surface there as a shared tile row).
        if padded and padded_refused_at_proposal(group, matched) is not None:
            return None
        if getattr(matched, 'kv_chains', False) and kv_shared_at_proposal(group, matched) is not None:
            return None
    return width


def proposal_groups(blocks, requests):
    """QWEN_FAST_M3_BLOCKS=2: the coming round's ticket width PER BLOCK, as [(block, rows, requests)] - `rows` the
    block's rows per user for every member when THAT block serves its own members as one pass, else None: each
    member then drafts at its engine's own width and the exact sequential step serves it - and one last
    (None, rows None, rest) entry for every request no block serves (finished, or bound to none). Every request of
    `requests` is in exactly one entry; the blocks come in their configured order, a block with no member is absent.

    This is `proposal_rows` asked of each block's group alone: the same eligibility, one block at a time. Where
    `proposal_rows` answers one width for the whole round, so one lone member (a 4+1 split) sends every user to the
    sequential step, here block A's four users stay one 16-row pass and only the lone user of block B narrows - the
    step already partitions a round by block (`packed_device_rounds`), so a lone member's engine-width ticket and the
    other block's 16-row tickets run in one round, each exactly once."""
    blocks = tuple(blocks)
    members = {id(block): [] for block in blocks}
    rest = []
    for request in requests:
        if request.session.finished:
            rest.append(request)
            continue
        for block in blocks:
            try:
                block.segment_of(request.engine)
            except ValueError:
                continue
            members[id(block)].append(request)
            break
        else:
            rest.append(request)
    groups = []
    for block in blocks:
        group = members[id(block)]
        if not group:
            continue
        groups.append((block, block_rows_for(block, group), group))
    if rest:
        groups.append((None, None, rest))
    return groups


def block_rows_for(block, group):
    """`proposal_rows`' decision for ONE block over its own live members: the block's rows per user, or None. See
    proposal_rows for each condition; this is the body of its per-block loop."""
    shape = block.shape
    if w2_switch.routes_sequential(block):
        return None
    padded = len(group) != shape.users and pads(block, len(group))
    if len(group) != shape.users and not padded:
        return None
    for request in group:
        session = request.session
        if session.max_new_tokens - len(session.emitted) < (1 if budget_cap_enabled() else shape.rows_per_user):
            return None
        if not admitted(block, session.position):
            return None
    if padded and padded_refused_at_proposal(group, block) is not None:
        return None
    if getattr(block, 'kv_chains', False) and kv_shared_at_proposal(group, block) is not None:
        return None
    return shape.rows_per_user


def pads(block, count):
    """Whether `block` serves a round of `count` live requests padded (QWEN_FAST_PADDED_BLOCK:
    packed_verifier.PackedVerifierEngine.pads, padded_min_users <= count < users). False for a
    block without the method or built without the flag, so every caller is today's."""
    padded = getattr(block, 'pads', None)
    return callable(padded) and bool(padded(count))


def padded_log(text, once=False):
    """One padded-path line into the server log (loguru, else stdout); `once`: once per process
    per text, for the lines the hook's every-tick policy question would otherwise repeat."""
    import verify_trace_t2

    if once:
        verify_trace_t2.log_once(text, key=('padded', text))
    else:
        verify_trace_t2.log_line(text)


def padded_refusal(owners, block):
    """Why the block cannot serve these owners - [(engine, frontier position)], a padded round's
    live requests - as one padded round, or None: the block's own padded_refusal over what it
    would stage (the count, the idle slot rule, page 0 in a live table's used range), each
    owner at the segment its engine is bound to, None for every idle segment. Fails closed:
    two owners through one segment are a reason."""
    users = [None] * block.shape.users
    for engine, position in owners:
        segment = block.segment_of(engine)
        if users[segment] is not None:
            return 'padded round with two live requests through segment %d' % segment
        users[segment] = (None, position, getattr(engine, 'pages', None))
    return block.padded_refusal(users)


def padded_marker(reason):
    from packed_verifier import PADDED_PAGE0_MARKER, PADDED_REFUSED_MARKER

    return PADDED_PAGE0_MARKER if reason.startswith('page0') else PADDED_REFUSED_MARKER


def padded_refused_at_proposal(requests, block):
    """`padded_refusal` over the live requests' frontiers, before the round is drafted: a reason
    drafts it at the engines' own widths, for the exact sequential step. Logged once per reason
    (the hook asks every tick)."""
    reason = padded_refusal([(request.engine, request.session.position) for request in requests], block)
    if reason is not None:
        padded_log('%s site=proposal_rows %s' % (padded_marker(reason), reason), once=True)
    return reason


def unservable(entries):
    """Entries whose ticket no capture of their own engine holds: a round drafted for the
    block that the block will not serve cannot go to the sequential step either."""
    refused = []
    for entry in entries:
        serves = getattr(entry['request'].engine, 'serves', None)
        if callable(serves) and not serves(entry['ticket']):
            refused.append('request=%s rows=%d' % (str(entry['request_id'])[:48], len(entry['ticket'].tokens)))
    return refused


_SOLO_NOTED = set()


def note_solo_route(solo, entries):
    """D0: one line per (request, position family) the solo block takes a round for - the routing a gate reads
    (serving_solo_lane.ROUTE_MARKER). Bounded: the first round of each request, then one per 4096 positions."""
    try:
        from serving_solo_lane import ROUTE_MARKER, SOLO_SLOT

        entry = entries[0]
        key = (str(entry['request_id'])[:48], entry['ticket'].position // 4096)
        if key in _SOLO_NOTED:
            return
        _SOLO_NOTED.add(key)
        padded_log('%s=%d slot=%d request=%s position=%d rows=%d block=solo' % (
            ROUTE_MARKER, getattr(solo, 'rounds', 0) + 1, SOLO_SLOT, key[0], entry['ticket'].position,
            len(entry['ticket'].tokens)))
    except Exception:
        pass


def note_solo_skipped(solo, requests, rows):
    """D0: why a lone live request is NOT on the solo block (logged once per reason): its engine borrowed another slot
    than the block's, or its frontier is outside what the extent block admits. Nothing for two or more live requests
    (M3's rounds) or none. Never raises."""
    try:
        from serving_solo_lane import SKIP_MARKER, SOLO_SLOT

        live = [request for request in requests if not request.session.finished]
        if rows is not None or len(live) != 1:
            return
        request = live[0]
        try:
            solo.segment_of(request.engine)
            reason = 'position %d outside the extent block or under 16 tokens of budget' % request.session.position
        except ValueError:
            reason = 'the request borrowed a pool slot other than %d' % SOLO_SLOT
        padded_log('%s request=%s: %s' % (SKIP_MARKER, str(request.session.request_id)[:48], reason), once=True)
    except Exception:
        pass


class PackedStep:
    """The packed device step bound to its block (or blocks - QWEN_FAST_FOUR_AS_TWO's pair
    of 32-row blocks for four users): the step itself, and the per-round ticket-width policy
    the worker hook asks before drafting (`proposal_rows`).

    With exactly one block this runs the SAME `packed_device_step` a lone block always
    did - `self.block` still names it, as callers that built a `PackedStep` over one block
    have always relied on. With several, each round is partitioned by block and run through
    `packed_device_rounds` instead; `self.block` is then None, since no one block owns the
    round."""

    def __init__(self, blocks, *, solo=None, per_block_widths=False, octo=None, octo_state=None):
        if blocks is None:
            raise ValueError('A packed verify block is required')
        self.blocks = tuple(blocks) if isinstance(blocks, (list, tuple)) else (blocks,)
        if not self.blocks:
            raise ValueError('At least one packed verify block is required')
        # QWEN_FAST_M3_BLOCKS=2 (serving_runtime): the coming round's ticket width is decided PER BLOCK
        # (`proposal_groups`), so a block that cannot serve its own members as one pass narrows only them. False, the
        # default, is the one width for the whole round every caller below has always read.
        if type(per_block_widths) is not bool or (per_block_widths and (len(self.blocks) < 2 or solo is not None)):
            raise ValueError('per_block_widths is an explicit bool, for several blocks and no solo block')
        self.per_block_widths = per_block_widths
        self.block = self.blocks[0] if len(self.blocks) == 1 else None
        # D0 (QWEN_FAST_SOLO_LANE, serving_solo_lane; default off): the one-user 16-row block a lone slot-0 user's
        # rounds run on. It is NOT one of `blocks`: proposal_rows and group_by_block match a request to the first
        # block whose carry it borrowed, and the solo block shares slot 0's carry with M3's segment 0, so it is
        # consulted by name - after the blocks for the coming round's width, before them for the step. None, every
        # path below is what it was.
        if solo is not None and (solo in self.blocks or solo.shape.users != 1 or solo.shape.rows_per_user != 16):
            raise ValueError('The solo block is a separate one-user 16-row packed block')
        self.solo = solo
        self.last_solo = None
        self.route = None
        # Octo-T8 (QWEN_FAST_OCTO, serving_octo; default off): the third block of eight users x eight rows beside the TWO M3 blocks, and the policy state that
        # decides, round by round, which shape the tickets are drafted for. Both None, every path below is what it was.
        if (octo is None) != (octo_state is None):
            raise ValueError('The octo block and its state (serving_octo.OctoState) go together')
        if octo is not None:
            shape = octo.shape
            if (not per_block_widths or len(self.blocks) != 2 or solo is not None or octo in self.blocks
                    or (shape.users, shape.rows_per_user) != (8, 8)):
                raise ValueError('The octo block is a separate eight-user eight-row block beside two M3 blocks that decide their widths per block, and no solo block')
        self.octo, self.octo_state = octo, octo_state
        if len(self.blocks) > 1:
            # tp4/hostgap: the two-block pre-stage and the per-block epochs, engaged (or refused, with the reason) once, here, by
            # the flags; host only. Neither flag set, nothing changes. Stage 0 also labels the blocks for its lines.
            # The octo block (when there is one) is a third block of the same pre-stage: its own fixture, its own snapshot.
            import verify_prestage

            staged = self.blocks if octo is None else self.blocks + (octo,)
            verify_prestage.engage_two_block(staged)
            if verify_prestage.hostgap_log_enabled():
                for index, block in enumerate(staged):
                    verify_prestage.block_label(block, index)
        # tp4/round-host: the six flags read strictly at the attach (a malformed one, or an audit without a lever, fails here) and the
        # engaged line written once. Nothing set, nothing logged.
        round_host.engage()
        # QWEN_FAST_GATE_FORCE_CAP (gate only): refused here, at attach, if malformed; logged once when set.
        cap = forced_cap()
        if cap is not None:
            audit_log(FORCE_CAP_MARKER + '{cap} (gate only)', cap=cap)

    def __call__(self, entries, *, cancelled):
        # QWEN_FAST_STALL_DEADLINE_S: one packed round (or its sequential fallback) is one watched scope; without the flag
        # scope() is a nullcontext and this is the routing below.
        with stall_watch.scope('step', 'packed round users=%d' % len(entries)):
            if self.octo is not None:
                return self.route_octo(entries, cancelled)
            if self.solo is not None:
                return self.route_round(entries, cancelled)
            return self.run_blocks(entries, cancelled)

    def route_round(self, entries, cancelled):
        """A step with the solo block: the round runs on it when its one entry is a 16-row slot-0 ticket, else on M3 (or the
        sequential step, as ever). `route` records where it went - 'solo', 'packed' or 'sequential', by which block's round
        counter moved - for the lane telemetry (serving_worker_hook reads it after the step)."""
        from serving_solo_lane import solo_serves

        counts = [getattr(block, 'rounds', 0) for block in self.blocks], getattr(self.solo, 'rounds', 0)
        solo_round = solo_serves(self.solo, entries)
        self.note_switch(solo_round)
        self.route = None
        try:
            if solo_round:
                note_solo_route(self.solo, entries)
                return packed_device_step(entries, cancelled=cancelled, block=self.solo)
            return self.run_blocks(entries, cancelled)
        finally:
            after = [getattr(block, 'rounds', 0) for block in self.blocks], getattr(self.solo, 'rounds', 0)
            self.route = ('solo' if after[1] != counts[1] else 'packed' if after[0] != counts[0] else 'sequential')

    def route_octo(self, entries, cancelled):
        """A step with the octo block: the round runs on it when every entry is an octo-width ticket (the drafts were made for it, `proposal_groups`), else on the
        two M3 blocks (or the sequential step) as ever. `route` records where it went - 'octo', 'packed' or 'sequential', by which block's round counter moved -
        and the octo state counts the round and writes its line AFTER it ran: a line that says the octo block executed, not that it was mounted."""
        entries = list(entries)
        state = self.octo_state
        octo_round = octo_serves(self.octo, entries)
        rows = len(entries[0]['ticket'].tokens) if entries else 0
        counts = [getattr(block, 'rounds', 0) for block in self.blocks], getattr(self.octo, 'rounds', 0)
        self.announce_shape('octo' if octo_round else 'm3')
        self.route = None
        state.begin()
        outputs = None
        try:
            # ORDERING INVARIANT (packed_device_rounds): no block's verify may replay between another block's verify and that block's commit flush, so every other
            # block still holding deferred commits is flushed BEFORE this round's verify.
            running = (self.octo,) if octo_round else self.blocks
            held = [block for block in self.all_blocks() if block not in running and getattr(block, 'deferred_commits', None)]
            if held:
                flush_blocks_deferred(held, 'verify')
            if octo_round:
                outputs = packed_device_step(entries, cancelled=cancelled, block=self.octo)
            else:
                outputs = self.run_blocks(entries, cancelled)
            return outputs
        finally:
            after = [getattr(block, 'rounds', 0) for block in self.blocks], getattr(self.octo, 'rounds', 0)
            ran = 'octo' if after[1] != counts[1] else 'm3' if after[0] != counts[0] else 'seq'
            self.route = {'octo': 'octo', 'm3': 'packed', 'seq': 'sequential'}[ran]
            if outputs is not None:
                state.finish(ran, live=len(entries), rows=rows, committed=sum(len(output.token_ids) for output in outputs))

    def announce_shape(self, shape):
        """The coming round's shape ('octo' or 'm3'), known when the drafts plan it (`proposal_groups`, before their fence window) and again when the step routes it:
        a switch from the last announced shape bumps the fixture write epoch (note_fixture_writer), exactly as D0's note_switch does. The epoch bump of a switch belongs
        BEFORE the window that pre-stages the coming verify, not at the step after it; the step's own call then finds the shape already noted and does not bump again.
        Over-bumping is the safe direction: a plan that does not hold bumps again when the next plan differs. Host only."""
        if self.octo_state.switched(shape):
            note_fixture_writer('octo-switch')

    def octo_rows(self, live, blocked=None):
        """The ticket width of a round the octo block would serve over these live requests (its rows per user), or None: too few live (serving_octo.octo_min_live), a
        member the hook's budget narrowing would cut at this width (`blocked`, a set of request ids), a member not bound to the block, or the block's own per-member rules
        (block_rows_for: the padded count and idle-segment rules, tokens left, the extent block's frontier range, page 0, the K/V tile rows)."""
        if len(live) < self.octo_state.min_live:
            return None
        if blocked and any(id(request) in blocked for request in live):
            return None
        for request in live:
            try:
                self.octo.segment_of(request.engine)
            except ValueError:
                return None
        return block_rows_for(self.octo, live)

    def octo_groups(self, groups, requests, blocked=None):
        """`proposal_groups` with the octo block: the coming round's shape is planned (OctoState.plan), announced, and when it is 'octo' the groups are ONE entry for
        the octo block (every live request, at its 8 rows) and one for the requests no block serves (finished ones); otherwise `groups`, the two-block answer, as it was."""
        live = [request for request in requests if not request.session.finished]
        rows = self.octo_rows(live, blocked) if live else None
        shape = self.octo_state.plan(rows is not None)
        self.announce_shape(shape)
        if shape != 'octo':
            return groups
        rest = [request for request in requests if request.session.finished]
        return [(self.octo, rows, live)] + ([(None, None, rest)] if rest else [])

    def solo_rows(self, request):
        """The ticket width the solo block would serve this ONE request at (16), or None: not its slot, no token budget left,
        or a frontier the extent block does not admit. None without a solo block."""
        if self.solo is None:
            return None
        from serving_solo_lane import solo_proposal_rows

        return solo_proposal_rows(self.solo, [request])

    def run_blocks(self, entries, cancelled):
        if len(self.blocks) == 1:
            return packed_device_step(entries, cancelled=cancelled, block=self.blocks[0])
        return packed_device_rounds(entries, cancelled=cancelled, blocks=self.blocks)

    def note_switch(self, solo_round):
        """D0: a round on the other packed block than the last one's. Both blocks read and write slot 0's carry and
        the model's native GDN slot, and each pre-stages its own next verify (verify_prestage): the switch bumps the
        fixture write epoch, exactly as any other writer to what a pre-staged verify relies on does
        (serving_worker_hook.note_fixture_writer), so the next verify restages in full. Host only; a step with no solo
        block never gets here."""
        previous, self.last_solo = self.last_solo, bool(solo_round)
        if previous is not None and previous != self.last_solo:
            note_fixture_writer('lane-switch')

    def announce_round(self, solo_round):
        """The coming round's block, known when the lanes plan it (the drafts, before their fence window): the epoch bump of a
        switch belongs HERE, before that window pre-stages the coming verify, not at the step after it - a bump at the step
        throws away the pre-stage the window just made, and every round after a switch restages in full. The step's own
        note_switch then finds the block already noted and does not bump again. Over-bumping is the safe direction: a plan
        that does not hold bumps again when the next plan differs, and the step's check still catches the rest."""
        if self.solo is not None:
            self.note_switch(solo_round)

    def proposal_rows(self, requests):
        rows = proposal_rows(self.blocks, requests)
        if rows is None and self.solo is not None:
            from serving_solo_lane import solo_proposal_rows

            rows = solo_proposal_rows(self.solo, requests)
            note_solo_skipped(self.solo, requests, rows)
        return rows

    def proposal_groups(self, requests, *, blocked=None):
        """[(block, rows, requests)] for the coming round (module `proposal_groups`), when this step decides its widths
        per block (QWEN_FAST_M3_BLOCKS=2); None otherwise, and the worker hook then asks `proposal_rows` as it always
        did. With the octo block (QWEN_FAST_OCTO) the round may instead be ONE group for it (octo_groups); `blocked` is
        then the set of request ids the hook's budget narrowing would cut at the octo width."""
        if not self.per_block_widths:
            return None
        groups = proposal_groups(self.blocks, requests)
        if self.octo is None:
            return groups
        return self.octo_groups(groups, requests, blocked)

    def while_waiting_groups(self, groups):
        """The drafts' fence window for a per-block round: one window per block whose members are drafted at its width
        (verify_prestage.WhileWaiting), composed (verify_prestage.CompositeWindow) when more than one block runs
        packed - so every packed block's round fence is armed and every block's fused tables are staged. None when no
        block runs packed or none holds a flag the window serves, and the hook passes nothing.

        The verify pre-stage is written only when ONE block runs packed in the round: the fixture write epoch is one
        counter for the process (verify_prestage.bump), a pre-stage and every verify bump it, so a second block's
        pre-stage or the first block's verify would invalidate the first's snapshot before it is read - the verify then
        takes the full stage, today's, with the pre-stage's host work spent for nothing.

        tp4/hostgap (verify_prestage.engage_two_block, attach): QWEN_FAST_TP4_TWO_BLOCK_PRESTAGE=1 pre-stages the ONE block
        that verifies first (the first, in this step's block order, whose members run packed - the block packed_device_rounds
        verifies first), under the very same epoch: nothing bumps it between that pre-stage and that verify (the other block's
        pre-stage is left out, and the first block's own diff comes after it consumed its snapshot). With
        QWEN_FAST_TP4_PRESTAGE_BLOCK_EPOCHS=1 as well, every packed block is pre-staged, each against its own epoch. Neither
        set, today's rule."""
        packed = [(block, members) for block, rows, members in groups if block is not None and rows is not None]
        from verify_prestage import two_block_mode

        mode = two_block_mode()
        first_block = None
        if mode == 'first' and len(packed) > 1:
            order = {id(block): index for index, block in enumerate(self.blocks)}
            first_block = min((block for block, members in packed), key=lambda block: order.get(id(block), len(order)))
        windows = []
        for block, members in packed:
            if (getattr(block, 'prestaged', None) is None and not getattr(block, 'round_fences', False)
                    and getattr(block, 'fused', None) is None):
                continue
            from verify_prestage import WhileWaiting

            windows.append(WhileWaiting(block, members, prestage=len(packed) == 1 or mode == 'blocks'
                                        or (mode == 'first' and block is first_block)))
        if not windows:
            return None
        if len(windows) == 1:
            return windows[0]
        from verify_prestage import CompositeWindow

        return CompositeWindow(windows)

    def while_waiting(self, requests):
        """Round-fence plan H1a: the drafts' fence-window callable for this step's one block
        (verify_prestage.WhileWaiting: the next verify's pre-stage under QWEN_FAST_PRESTAGE, the
        arming of its replay under QWEN_FAST_ROUND_FENCES, and - H1b, QWEN_FAST_FUSED_COMMIT - the
        next round's T_proj RoPE tables), for the live `requests` the hook's policy just drafted a
        block round for. None with several blocks (QWEN_FAST_FOUR_AS_TWO) or a block built with
        none of the flags - the hook then passes nothing."""
        block = self.block
        if self.solo is not None:
            from serving_solo_lane import solo_proposal_rows

            if solo_proposal_rows(self.solo, requests) is not None:
                block = self.solo    # a solo round's window is the solo block's own: its pre-stage, its round fence
        if block is None or (getattr(block, 'prestaged', None) is None and not getattr(block, 'round_fences', False)
                             and getattr(block, 'fused', None) is None):
            return None
        from verify_prestage import WhileWaiting

        return WhileWaiting(block, requests)

    def all_blocks(self):
        """Every block a round may run on: the packed blocks, then the solo block (D0) or the octo block when there is one. A block armed
        for deferred commits that does not run the round is disarmed by the flush (PackedVerifierEngine.flush_commits)."""
        extra = tuple(block for block in (self.solo, self.octo) if block is not None)
        return self.blocks + extra if extra else self.blocks

    def arm_deferred_commits(self):
        """Round-fence plan H2 (early_draft.py, QWEN_FAST_GDN_AFTER_PAIRS): ask every block to defer its
        coming round's GDN commit traces (packed_verifier.PackedVerifierEngine.arm_deferred_commits) -
        asked only by early_draft.EarlyDraft.execute, which flushes them inside the same execute_model.
        True when a block will defer; a block built without the flag never does."""
        armed = False
        for block in self.all_blocks():
            arm = getattr(block, 'arm_deferred_commits', None)
            if callable(arm) and arm():
                armed = True
        return armed

    def flush_deferred_commits(self, site):
        """Round-fence plan H2: enqueue every block's deferred GDN commit traces (flush_commits; a no-op
        for a block holding none). Returns how many were enqueued."""
        return flush_blocks_deferred(self.all_blocks(), site)


def flush_blocks_deferred(blocks, site):
    """Flush every block's deferred GDN commits, ATTEMPTING EVERY BLOCK even when one raises (the first error is
    raised after the last attempt): a block skipped because an earlier one failed would keep retained states that a
    later block's verify replay overwrites (the retained buffers of one block sit in the holes of another's verify
    trace). Returns how many traces were enqueued."""
    count, first_error = 0, None
    for block in blocks:
        flush = getattr(block, 'flush_commits', None)
        if callable(flush):
            try:
                count += flush(site) or 0
            except BaseException as error:
                if first_error is None:
                    first_error = error
    if first_error is not None:
        raise first_error
    return count


def octo_serves(octo, entries):
    """Whether this round is one the octo block serves: every entry an octo-width ticket (8 rows) of a request bound to the block. The routing decision at the
    step: the tickets were drafted for the shape (`PackedStep.proposal_groups`), so the width is what says which shape the round was planned as. An entry the block
    does not hold, or a ticket of any other width, is not an octo round (the block's own `ineligible` then narrows or refuses what it must)."""
    entries = list(entries)
    if not entries:
        return False
    rows = octo.shape.rows_per_user
    for entry in entries:
        if len(entry['ticket'].tokens) != rows:
            return False
        try:
            octo.segment_of(entry['request'].engine)
        except ValueError:
            return False
    return True


def ineligible(entries, block):
    """Why the block cannot serve this round as one pass, or None when it can."""
    if w2_switch.routes_sequential(block):
        # The W2 kill switch (w2.off): the packed block's traces carry W2, so the round goes to the sequential step (the served SDPA and conv gates).
        return w2_switch.REASON
    shape = block.shape
    padded = len(entries) != shape.users and pads(block, len(entries))
    if len(entries) != shape.users and not padded:
        return 'entries=%d block_users=%d' % (len(entries), shape.users)
    for entry in entries:
        rows = len(entry['ticket'].tokens)
        if rows != shape.rows_per_user:
            return 'request=%s rows=%d rows_per_user=%d' % (str(entry['request_id'])[:48], rows, shape.rows_per_user)
        try:
            block.segment_of(entry['request'].engine)
        except ValueError:
            return 'request=%s engine not bound to the block' % str(entry['request_id'])[:48]
        if getattr(block, 'extent', False) and not admitted(block, entry['ticket'].position):
            # S2 W4: a ticket the extent path does not serve (below 128, or past C) - D1 narrows it for
            # the sequential step rather than the verify refusing the whole round.
            return 'request=%s position=%d outside the extent path' % (str(entry['request_id'])[:48],
                                                                        entry['ticket'].position)
    if padded:
        # QWEN_FAST_PADDED_BLOCK: the backstop behind proposal_rows, over the round's tickets.
        reason = padded_refusal([(entry['request'].engine, entry['ticket'].position) for entry in entries], block)
        if reason is not None:
            padded_log('%s site=ineligible %s' % (padded_marker(reason), reason))
            return reason
    if getattr(block, 'kv_chains', False):
        return kv_shared(entries, block)
    return None


def pad_sentinels_enabled():
    """Sticky sessions (serving_fast_policy.STICKY_SESSIONS_FLAG): the guard maps a row past an engine's
    bound blocks to a per-request sentinel page (pad_sentinel_table). Off, the guard reads the
    engines' tables exactly as it always did."""
    from serving_fast_policy import sticky_sessions_enabled

    return sticky_sessions_enabled()


def pad_sentinel_table(table, segment):
    """An engine's page table as the K/V guard should read it under prefix caching: the entries past
    its bound blocks - the page binding pads them with the request's FIRST block
    (serving_page_binding.VerifierPageBinding.refresh, serving_runtime's bridge factory) - mapped to
    a sentinel page of this segment's own, -(segment + 1), which no real page and no other segment
    names.

    Why: at proposal time (kv_shared_at_proposal) a round's 16 rows are mapped through the table of
    the LAST refresh, so a row past it maps to the pad. With prefix caching off that pad is the
    request's own first block and can never alias another user's; with it on, same-tenant requests
    share their first block (a cached prefix), and two of them crossing a 64-token boundary in one
    round would hit (blocks[0], tile row 0) together - a KV_SHARED conflict that is not real (the
    rows are written only after vLLM appends their own block and the binding refreshes; the refresh
    refuses a table that does not cover every row, and the step-time guard reads that refreshed
    table, where no row is past the bound). The pad is recognised without the binding: the bound
    blocks are unique (validate_blocks), so the first entry after index 0 equal to entry 0 starts
    the pad. Returns a one-row list of ints; anything that is not a table is returned unchanged,
    so the guard still fails closed on it."""
    try:
        row = table[0]
        values = [int(value) for value in (row.tolist() if hasattr(row, 'tolist') else row)]
    except (IndexError, TypeError, ValueError):
        return table
    if not values:
        return table
    bound = next((index for index in range(1, len(values)) if values[index] == values[0]), len(values))
    return [values[:bound] + [-(segment + 1)] * (len(values) - bound)]


def kv_guard(owners, block):
    """QWEN_FAST_VERIFY_T2 (#2, verify_trace_t2): the reason the block's per-user K/V chains
    cannot serve these owners - [(engine, frontier position)] - or None. The same host values
    packed_host_inputs stages: each user's rows_per_user positions from its frontier, through
    its engine's one page table, in SEGMENT order. Fails closed: a position its table cannot
    map (or an engine with no table) is a reason, never an exception past the step.

    A padded round (QWEN_FAST_PADDED_BLOCK: fewer owners than segments, a count the block
    pads) is checked with its idle segments filled from the block's own idle_inputs - their
    page-0 tile rows are written in the same chain as the live ones - rather than skipped as a
    round with an empty segment would be."""
    import verify_trace_t2

    users = [None] * block.shape.users
    sentinels = pad_sentinels_enabled()
    for engine, position in owners:
        segment = block.segment_of(engine)
        table = getattr(engine, 'pages', None)
        if sentinels:
            table = pad_sentinel_table(table, segment)
        users[segment] = (range(position, position + block.shape.rows_per_user), table)
    if any(user is None for user in users) and pads(block, len(owners)):
        live = tuple(segment for segment, user in enumerate(users) if user is not None)
        if len(live) != len(owners):
            return 'verify t2 kv padded round with two owners through one segment'
        try:
            fills = block.idle_inputs(live)
        except ValueError as refusal:
            return 'verify t2 kv padded idle segments refused: %s' % (refusal,)
        for segment, (tokens, start, table) in fills.items():
            users[segment] = (range(start, start + block.shape.rows_per_user), table)
    if any(user is None for user in users):
        return None  # two owners through one segment: the block's own checks refuse the round
    try:
        conflict = verify_trace_t2.kv_conflict([(positions, table[0]) for positions, table in users])
    except (IndexError, TypeError, ValueError) as error:
        return 'verify t2 kv tile rows unmapped: %s' % (error,)
    return None if conflict is None else verify_trace_t2.kv_conflict_reason(conflict)


def kv_shared_at_proposal(requests, block):
    """`kv_guard` over the live requests' frontiers, before the round is drafted
    (proposal_rows): a reason there drafts the round at the engines' own widths, so the exact
    sequential step serves it. The engines' page tables are the last refresh's; a block vLLM
    appends for this round's rows is that request's own (allocations past a request's cached
    prefix are disjoint), and an unrefreshed entry names the request's own first block
    (serving_page_binding), so neither can hide a conflict between two users. Under sticky
    sessions two same-tenant requests share their cached first block, so kv_guard reads an
    unrefreshed entry as a per-request sentinel page instead (pad_sentinel_table) - it would
    otherwise report a conflict on the shared pad that no write makes. Logged once per reason:
    the hook asks every tick."""
    import verify_trace_t2

    reason = kv_guard([(request.engine, request.session.position) for request in requests], block)
    if reason is not None:
        verify_trace_t2.log_once('%s site=proposal_rows %s' % (verify_trace_t2.KV_SHARED, reason))
    return reason


def kv_shared(entries, block):
    """`kv_guard` over the round's tickets, at the step: the backstop behind proposal_rows.
    Beside the 64-row block (the only one that chains) the per-request engines capture only
    the sequential widths (1, 2, 4), so the round's 16-row tickets have no capture of their
    own: packed_device_step then cuts each to the widest width its engine serves and hands
    the round to the sequential step, the block never touched (`unservable`, `narrow_round`,
    S2 D1). Only a ticket nothing narrower serves still refuses the round (`refuse_round` -
    every request fails, nothing is written; under the D2 request quarantine each then ends
    FINISHED_ABORTED, S2 W5b). A conflict here means the tables changed between the drafting
    and the step in a way vLLM's disjoint, append-only allocation does not produce; KV_SHARED
    fails the gated arm either way."""
    import verify_trace_t2

    reason = kv_guard([(entry['request'].engine, entry['ticket'].position) for entry in entries], block)
    if reason is not None:
        verify_trace_t2.log_line('%s site=ineligible %s' % (verify_trace_t2.KV_SHARED, reason))
    return reason


def validate_packed_entries(entries):
    """FastRequest.step's own refusal, before any device work: one live owner with its own
    prepared ticket, for every entry of the round - regardless of which block, if any, will
    serve it, so this runs once over the whole round rather than once per block group."""
    for entry in entries:
        request, ticket = entry['request'], entry['ticket']
        if ticket.request_id != entry['request_id']:
            raise ValueError('Each packed entry must carry its own prepared ticket')
        if request.closed or request.busy or request.cancelled or request.session.pending is not ticket:
            raise ValueError('One live owner with its prepared ticket required for every packed entry')


def run_verified_block(entries, *, cancelled, block):
    """The block's own round once `packed_device_step` (or `packed_device_rounds`, per block
    group) has decided it will serve these entries as one pass: the verify, then each
    entry's own commit, in entries order."""
    requests = [entry['request'] for entry in entries]
    for request in requests:
        request.busy = True
    try:
        # Under QWEN_FAST_PHASE_LOG the same begin/end lines the hook writes around each
        # proposal, so a hang says which phase stalled and whose.
        ids = ','.join(str(entry['request_id'])[:48] for entry in entries)
        verify_started = time.perf_counter()
        # The block stages every entry's inputs itself (packed_verifier.py, verify).
        predictions, metrics = phase('packed_verify', ids, lambda: block.verify(entries))
        verified = time.perf_counter()
        round_host.ledger.note_verify(metrics, (verified - verify_started) * 1000)
        segments = metrics['segments']
        if len(predictions) != len(entries) or len(segments) != len(entries):
            raise ValueError('The packed block must return one prediction list and one segment per entry')
        outputs = []
        # Only collected under the audit gate: appending a dict per entry is cheap, but
        # there is no reason to pay even that when nobody will read it.
        # tp4/round-host LEAN: the per-user [PACKED-COMMIT-HOST] and the splits behind it are not written (the ledger line carries the time).
        audit = audit_enabled()
        commit_host_timings = [] if audit and not round_host.lean_enabled() else None
        # {stage_name: [ms per entry, in entries order]} - built up one entry at a time
        # (commit_entry appends its own dict per call) so the round's line stays
        # positional with commit_host_timings and predictions/segments above; an entry
        # whose publish() skipped a stage (prefix 0) contributes 0.0 for it, not a gap.
        publish_stage_timings = {} if audit_enabled() else None
        for entry, segment, rows in zip(entries, segments, predictions):
            # entries[i] is served by metrics['segments'][i] and predictions[i]: the
            # segment its engine's carry is bound to, never its index in the entries
            output = phase('packed_commit', entry['request_id'],
                           lambda entry=entry, segment=segment, rows=rows: commit_entry(
                               entry, block, segment, rows, cancelled=cancelled, metrics=metrics,
                               verify_started=verify_started, verified=verified,
                               commit_host_timings=commit_host_timings, publish_stage_timings=publish_stage_timings))
            outputs.append(output)
        if commit_host_timings:
            audit_log(COMMIT_HOST_LINE, round=getattr(block, 'rounds', 0),
                      adopt=','.join('%.2f' % item['adopt_ms'] for item in commit_host_timings),
                      block=','.join('%.2f' % item['block_ms'] for item in commit_host_timings),
                      session=','.join('%.2f' % item['session_ms'] for item in commit_host_timings),
                      other=','.join('%.2f' % item['other_ms'] for item in commit_host_timings))
        if publish_stage_timings:
            audit_log(PUBLISH_LINE, round=getattr(block, 'rounds', 0),
                      stages=format_publish_stages(publish_stage_timings))
        if publish_stage_timings and PUBLISH_SPLIT_KEY in publish_stage_timings:
            for index, splits in enumerate(publish_stage_timings[PUBLISH_SPLIT_KEY]):
                audit_log(PUBLISH_SPLIT_LINE, round=getattr(block, 'rounds', 0), entry=index,
                          splits=format_publish_splits(splits))
        if audit and ('replay_fence' in metrics or 'prestage' in metrics):
            # Round-fence plan H1a only (a block built with QWEN_FAST_PRESTAGE or
            # QWEN_FAST_ROUND_FENCES puts these in its metrics): FENCES_LINE, once per round.
            audit_log(FENCES_LINE, **fences_fields(metrics, block, segments))
        # QWEN_FAST_MEMORY_LEDGER=1 only, and once per process: P12, after the first packed
        # round's verify and every commit have returned - outside any capture - to catch the
        # buffers the first round allocates lazily (packed proposals, publication).
        memory_ledger.first_packed_round(packed_block=block, round_requests=[entry['request'] for entry in entries])
        round_host.ledger.note_commit((time.perf_counter() - verified) * 1000)
        w2_switch.note_round(block)
        return outputs
    except BaseException:
        fail_round(entries, block)
        raise
    finally:
        # Slot 0 holds the last committed segment's user or nobody's committed state.
        verifier_engine.note_packed_step()
        for request in requests:
            request.busy = False


def packed_device_step(entries, *, cancelled, block):
    """Step every packed request through one verify pass, committing in the scheduler's order."""
    entries = list(entries)
    if not entries:
        raise ValueError('A packed step needs at least one admitted request')
    if not callable(cancelled) or block is None:
        raise ValueError('A cancellation callback and the packed verify block are required')
    validate_packed_entries(entries)
    started = time.perf_counter() if hostgap_log_enabled() else None
    reason = ineligible(entries, block)
    if started is not None and reason is None:
        entry_line((time.perf_counter() - started) * 1000)
    if reason is not None:
        # A round drafted for the block (proposal_rows) whose entries changed before the
        # step: a partner aborted after the drafts (vLLM names it one step late: the 2->1,
        # 3->1 and 4->1 aborts), a padded check failed here, or a scheduler race (run
        # 35535533720). Its block-width tickets have no capture of their own engines, and a
        # session cannot re-propose while one is pending, so S2 D1 narrows each to the widest
        # width its engine serves (narrow_round) and the round goes to the sequential step
        # below. Only a ticket nothing narrower serves still refuses the round (refuse_round),
        # before any device work and without raising past the step: a race must never crash
        # the engine for every OTHER live user.
        refused = unservable(entries)
        if refused:
            entries, failure = narrow_round(entries, reason)
            if failure is not None:
                return refuse_round(entries, block, reason, refused)
    elif cancelled():
        # Each request's own step answers a cancellation without touching the device.
        reason = 'cancelled before the verify'
    if reason is not None:
        if pads(block, len(entries)):
            note_padded_skip(entries, block, reason)
        return sequential_packed_step(entries, cancelled=cancelled)
    return run_verified_block(entries, cancelled=cancelled, block=block)


def remaining_budget(entry):
    """The tokens this entry's request may still emit: its session's own budget (what
    proposal_rows reads), and no more than vLLM's (serving_worker_hook.real_remaining_budget,
    what the hook narrows a round on) when the entry carries its bridge."""
    from serving_worker_hook import real_remaining_budget

    session = entry['request'].session
    remaining = session.max_new_tokens - len(session.emitted)
    bridge = entry.get('bridge')
    real = real_remaining_budget(bridge) if bridge is not None else None
    return remaining if real is None else min(remaining, real)


def note_padded_skip(entries, block, reason):
    """QWEN_FAST_PADDED_BLOCK: one line for a round of padded_min_users..users-1 live requests
    that the block did not serve, with whether it could have. eligible=1 when every entry had a
    block round of budget left and a frontier inside the block's family - what proposal_rows
    asks before anything padded-specific - so the gate can hold the padded rounds against every
    eligible round (a narrow tail round is eligible=0). Never raises."""
    try:
        from packed_verifier import PADDED_SKIPPED_MARKER

        rows = block.shape.rows_per_user
        eligible = True
        for entry in entries:
            if remaining_budget(entry) < (1 if budget_cap_enabled() else rows):
                eligible = False
                break
            if not admitted(block, entry['request'].session.position):
                eligible = False
                break
        padded_log('%s live=%d eligible=%d reason=%s' % (PADDED_SKIPPED_MARKER, len(entries), int(eligible),
                                                         str(reason).replace(' ', '_')[:120]))
    except Exception:
        pass


def group_by_block(entries, blocks):
    """Partition entries by the block their engine is bound to (segment_of), preserving each
    group's relative entries-order and the order blocks are first seen in; entries bound to
    none of `blocks` come back separately, in their own relative order."""
    groups, order, unbound = {}, [], []
    for entry in entries:
        matched = None
        for block in blocks:
            try:
                block.segment_of(entry['request'].engine)
            except ValueError:
                continue
            matched = block
            break
        if matched is None:
            unbound.append(entry)
            continue
        key = id(matched)
        if key not in groups:
            groups[key] = (matched, [])
            order.append(key)
        groups[key][1].append(entry)
    return [groups[key] for key in order], unbound


def packed_device_rounds(entries, *, cancelled, blocks):
    """One round over SEVERAL packed blocks (QWEN_FAST_FOUR_AS_TWO's pair of 32-row blocks
    for four users, instead of one 64-row block): partition the entries by the block their
    engine is bound to, then run each block's own round - exactly as `packed_device_step`
    runs a lone block's round, `ineligible`/`unservable`/`refuse_round`/cancellation and all
    - in the blocks' configured order (block A's verify and commits, then block B's).

    Eligibility is PER BLOCK, not whole-round: a block whose group is not its own full user
    set (one partner already finished, or missing entirely) falls to the sequential step for
    just its own live member(s), while another block whose group IS complete still runs
    packed in the same round. Entries bound to no configured block go to that same sequential
    batch. Every request is served exactly once; the returned outputs always follow
    `entries`' own order, whichever block (or the sequential step) actually produced them.

    ORDERING INVARIANT (the retained GDN states, conv states and histories of a block, and its trace outputs, are
    allocated inside its own capture and almost certainly sit in the holes of the EARLIER block's verify trace): no
    block's verify may replay between another block's verify and that block's commit flush. So every block still
    holding deferred commits is flushed BEFORE any verify of this round, and the in-step, `finally` and reconcile
    flushes attempt every block."""
    # tp4/hostgap stage 0 (QWEN_FAST_TP4_HOSTGAP_LOG): this step's own checks before its first verify, for the [PACKED-ENTRY] line.
    log = hostgap_log_enabled()
    checks_started = time.perf_counter() if log else 0.0
    checks_ms = 0.0
    entries = list(entries)
    if not entries:
        raise ValueError('A packed step needs at least one admitted request')
    blocks = tuple(blocks)
    if not callable(cancelled) or not blocks:
        raise ValueError('A cancellation callback and at least one packed verify block are required')
    validate_packed_entries(entries)
    order = {entry['request_id']: index for index, entry in enumerate(entries)}
    groups, sequential_entries = group_by_block(entries, blocks)
    by_block = {id(matched): (matched, group_entries) for matched, group_entries in groups}
    outputs_by_id = {}
    if log:
        checks_ms = (time.perf_counter() - checks_started) * 1000
    if len(blocks) > 1:
        flush_blocks_deferred([block for block in blocks if getattr(block, 'deferred_commits', None)], 'verify')
    if log:
        checks_started = time.perf_counter()
    for block in blocks:
        group = by_block.get(id(block))
        if group is None:
            continue
        matched, group_entries = group
        reason = ineligible(group_entries, block)
        if reason is not None:
            # Same rule as the single-block step, applied to this block's own group: its
            # tickets are narrowed to widths their engines serve (S2 D1) and join the
            # sequential batch; a group with a ticket nothing narrower serves is refused
            # (failed) here rather than raised past the step for every OTHER live user or block.
            refused = unservable(group_entries)
            if refused:
                group_entries, failure = narrow_round(group_entries, reason)
                if failure is not None:
                    for output in refuse_round(group_entries, block, reason, refused):
                        outputs_by_id[output.request_id] = output
                    continue
        elif cancelled():
            reason = 'cancelled before the verify'
        if reason is not None:
            sequential_entries.extend(group_entries)
            continue
        if log:
            # The first verify of the step: everything before it was entry (the line is written once, with that block's checks).
            checks_ms += (time.perf_counter() - checks_started) * 1000
            entry_line(checks_ms)
            log = False
        for output in run_verified_block(group_entries, cancelled=cancelled, block=block):
            outputs_by_id[output.request_id] = output
    if sequential_entries:
        # Back into the round's own entries order: a block's fallback group is appended
        # after entries no block claimed at all, which need not be their relative order.
        sequential_entries.sort(key=lambda entry: order[entry['request_id']])
        for output in sequential_packed_step(sequential_entries, cancelled=cancelled):
            outputs_by_id[output.request_id] = output
    return [outputs_by_id[entry['request_id']] for entry in entries]


def commit_entry(entry, block, segment, rows, *, cancelled, metrics, verify_started, verified,
                 commit_host_timings=None, publish_stage_timings=None):
    """FastRequest.step from its verify readback on, for one user of the block."""
    request, ticket, request_id = entry['request'], entry['ticket'], entry['request_id']
    session, engine, runtime = request.session, request.engine, request.runtime
    started = time.perf_counter()
    # Verified on the block, in this segment: the publication is the block's per-user commit.
    adopt_started = time.perf_counter()
    engine.adopt_packed(ticket, block, segment)
    adopt_ms = (time.perf_counter() - adopt_started) * 1000
    # QWEN_FAST_PACKED_AUDIT: times DFlashRequestRuntime.publish's own named stages
    # (dflash_request_runtime.publication_stage's seam) for this one user's commit only -
    # installed and restored around session.commit/session.abort below, never left on
    # runtime past this call. stage_sink stays {} (every PUBLISH_STAGES entry reported as
    # 0.0) for a runtime whose publish() never calls publication_stage at all (a fake in a
    # test that does not model it) or an entry whose accepted prefix is 0.
    stage_sink = {} if publish_stage_timings is not None else None
    restore_stage_timer = install_stage_timer(runtime, stage_sink) if stage_sink is not None else None
    # QWEN_FAST_PIPELINED_PUBLISH merges prepare_publication's own three device fences
    # into one, and QWEN_FAST_TRACED_PUBLISH cuts its op count in the steady state
    # (dflash_traced_publish.install_publish_options composes both into ONE installer,
    # so neither flag's effect is silently dropped by the other overwriting drafter.
    # prepare_publication after it - see that module for why) - installed on the
    # drafter only if this runtime actually exposes one (a dspark request's runtime,
    # or a test fake that does not model prepare_publication, has none), and only if
    # at least one of the two flags is actually on.
    drafter = getattr(runtime, 'drafter', None)
    restore_merge_release = None
    if drafter is not None:
        from dflash_pipelined_publish import pipelined_publish_enabled
        from dflash_traced_publish import traced_publish_enabled, install_publish_options

        merge_release, fused_steady_state = pipelined_publish_enabled(), traced_publish_enabled()
        # Round-fence plan H1b (fused_commit.py, QWEN_FAST_FUSED_COMMIT): a block that built its
        # fused commit wraps the same installers - today's publication for any round its guards
        # refuse. Without it (every block built without the flag) this is today's call.
        fused = getattr(block, 'fused', None)
        if fused is not None:
            from fused_commit import install_fused_commit

            restore_merge_release = install_fused_commit(drafter, fused, segment, merge_release=merge_release,
                                                         fused_steady_state=fused_steady_state)
        elif merge_release or fused_steady_state:
            restore_merge_release = install_publish_options(drafter,
                merge_release=merge_release, fused_steady_state=fused_steady_state)
    # QWEN_FAST_ROUND_B1 (M0a): this user's publication splits, only when the stage timer
    # above is on too (QWEN_FAST_PACKED_AUDIT) - PUBLISH_SPLIT_LINE.
    split_sink = split_token = None
    if stage_sink is not None and os.environ.get('QWEN_FAST_ROUND_B1') == '1' and not round_host.lean_enabled():
        from dflash_traced_publish import PUBLICATION_SPLITS

        split_sink = {}
        split_token = PUBLICATION_SPLITS.set(split_sink)
    try:
        # session.commit and session.abort both end in runtime.publish - DFlashRequestRuntime.
        # publish (dflash_request_runtime.py) - which runs VerifierEngine.publish and, through
        # it, packed_verifier.PackedVerifierEngine.commit_user: this one interval covers all
        # of that, for either outcome.
        session_started = time.perf_counter()
        # S2 W4: the rows this user may commit (the extent block's boundary cap, the gate's forced
        # cap), or None - every block before S2 - for the call exactly as it was.
        budget = remaining_budget(entry) if budget_cap_enabled() else None
        limit = commit_limit(block, ticket, budget)
        if cancelled():
            session.abort(request_id, ticket, runtime.publish)
            request.cancelled = True
            decision = None
            output = CommittedOutput(request_id, (), session.position, True, True)
        elif limit is None:
            decision = session.commit(request_id, ticket, rows, runtime.publish)
            output = CommittedOutput(request_id, tuple(decision.emitted), session.position, session.finished)
        else:
            decision = session.commit(request_id, ticket, rows, runtime.publish, max_rows=limit)
            output = CommittedOutput(request_id, tuple(decision.emitted), session.position, session.finished)
        session_ms = (time.perf_counter() - session_started) * 1000
    finally:
        if restore_merge_release is not None:
            restore_merge_release()
        if restore_stage_timer is not None:
            restore_stage_timer()
        if split_token is not None:
            from dflash_traced_publish import PUBLICATION_SPLITS

            PUBLICATION_SPLITS.reset(split_token)
    finished = time.perf_counter()
    if audit_enabled():
        if budget is not None and limit == budget and budget < len(ticket.tokens):
            audit_log(BUDGET_CAP_LINE, request=str(request_id)[:48], segment=segment, position=ticket.position,
                      remaining=budget, limit=limit)
        audit_log(AUDIT_LINE if limit is None else AUDIT_LINE + AUDIT_CAP, request=str(request_id)[:48],
                  segment=segment, position=ticket.position, prefix=0 if decision is None else decision.state_rows,
                  emitted=0 if decision is None else len(decision.emitted), predictions=list(rows[:8]),
                  **({} if limit is None else dict(cap=limit)))
    if decision is not None and getattr(request, 'collect_timings', False):
        record_timing(request, ticket, decision, metrics, segment,
                      verify_started=verify_started, verified=verified, started=started, finished=finished)
    if commit_host_timings is not None:
        # block_ms is the portion of session_ms already attributed to
        # RetainedGDNBlock.commit_user's own host cost (packed_verifier.py's
        # commit_block_ms, indexed the same way predictions/segments are - by THIS
        # entry's segment, never its index in the round); other_ms is whatever of this
        # call's own wall time neither adopt_ms nor session_ms accounts for.
        block_ms = getattr(block, 'commit_block_ms', None)
        block_ms = block_ms[segment] if block_ms is not None else 0.0
        other_ms = max((finished - started) * 1000 - adopt_ms - session_ms, 0.0)
        commit_host_timings.append(dict(adopt_ms=adopt_ms, block_ms=block_ms, session_ms=session_ms, other_ms=other_ms))
    if publish_stage_timings is not None:
        for name in PUBLISH_STAGES:
            publish_stage_timings.setdefault(name, []).append(stage_sink.get(name, 0.0))
        if split_sink is not None:
            publish_stage_timings.setdefault(PUBLISH_SPLIT_KEY, []).append(split_sink)
    return output


def record_timing(request, ticket, decision, metrics, segment, *, verify_started, verified, started, finished):
    """The block FastRequest.step records under QWEN_FAST_PHASE_TIMING, for the packed round:
    the verify is the shared pass, the commit is this user's own, and the cycle runs from
    this user's proposal (or its last commit) to its commit."""
    prepared, drafted = request.prepared_timing
    cycle_start = prepared if request.last_commit_time is None else request.last_commit_time
    request.timings.append(dict(position=ticket.position, rows=len(ticket.tokens),
        committed=len(decision.emitted), draft_ms=(drafted - prepared) * 1000,
        verify_host_ms=(verified - verify_started) * 1000,
        commit_host_ms=(finished - started) * 1000,
        cycle_ms=(finished - cycle_start) * 1000,
        outside_phases_ms=((prepared - cycle_start) + (verify_started - drafted)) * 1000,
        verifier=dict(metrics, packed=True, segment=segment)))
    request.last_commit_time = finished
    request.prepared_timing = None


def fail_round(entries, block):
    """FastRequest.step's failure path, for every request of the round: an adopted ticket is
    aborted through the block (prefix 0 runs no trace) and its session marked failed; one
    not yet adopted fails its verification. Then every segment still undecided is released
    at prefix 0, so the block ends the round idle or failed but never verified - a verified
    block refuses the next round and refuses to close. Secondary failures are dropped in
    favour of the one propagating."""
    for entry in entries:
        request, ticket, request_id = entry['request'], entry['ticket'], entry['request_id']
        session, engine = request.session, request.engine
        if session.phase != 'pending' or session.pending is not ticket:
            continue
        if engine.phase == 'verified':
            try:
                session.abort(request_id, ticket, request.runtime.publish)
            except BaseException:
                pass
            finally:
                session.phase = 'failed'
        else:
            session.fail_verification(request_id, ticket)
    for segment in sorted(getattr(block, 'pending_segments', ())):
        try:
            block.commit_user(segment, 0)
        except BaseException:
            pass


def refuse_round(entries, block, reason, refused):
    """A round drafted for the block that the block cannot serve, whose tickets no
    request engine captures either, even narrowed (`narrow_round`) - so the sequential
    step cannot take it either.

    `serving_worker_hook.discard_stale_ticket` keeps every live request's pending
    ticket at one width per step, and `narrow_round` serves the survivors of an abort
    the drafts could not see, so reaching this means a ticket no narrower width of its
    own engine serves, not that the hardware or a request failed. Fails every request of
    the round exactly as `fail_round` always has, but returns their outputs instead of
    raising past the step, so this costs the round's requests rather than the engine
    for every other live user (run 35535533720). A failed session cannot draft again,
    so without more the engine still dies one step later, at the next draft (VR4
    finding 2); `abort_refused` (S2 W5b) ends these requests as FINISHED_ABORTED
    through the D2 request quarantine in this same engine step when its scheduler-side
    consumer is installed (QWEN_FAST_ANY_REQUEST), and without it changes nothing."""
    message = ('A round the block cannot serve (%s) holds tickets no request engine captured (%s): '
              'it was drafted for the block but its entries changed before the step'
              % (reason, '; '.join(refused)))
    try:
        from loguru import logger
        logger.warning('[PACKED] {}', message)
    except ImportError:
        print('[PACKED] %s' % message, flush=True)
    fail_round(entries, block)
    abort_refused(entries, message)
    return [CommittedOutput(entry['request_id'], (), entry['request'].session.position, True, True)
            for entry in entries]


def describe():
    """What this costs, beside what it falls back to."""
    return dict(name='packed', weight_passes_per_round='one for all users',
                per_user_rate='the block rate, shared by every user', batched=True,
                fallback=describe_sequential())


# QWEN_FAST_ROUND_B1 (M0a), under QWEN_FAST_PACKED_AUDIT=1: one line per entry of the round,
# in the same scheduler order as PUBLISH_LINE's arrays, splitting that entry's
# 'prepare_history' into its phases (dflash_traced_publish.PUBLICATION_SPLIT_NAMES: 'proj'
# project_features, 'hist' the feature-history write, 'kv' kv_history.prepare, 'sync' and
# 'rel' prepare_publication's own fence and release; then inside kv_history.prepare on the
# slide path 'kv_in' project_inputs, 'kv_proj' the K/V projections, 'kv_build'/'kv_exec'
# the slide transports built and run, 'kv_sync' and 'kv_rel'). Per entry rather than per
# stage so the line stays under the log capture's ~250 characters. Defined here, at the end,
# so every line above keeps its number - loguru prints it in each [PACKED*] line's prefix.
PUBLISH_SPLIT_KEY = 'round_b1_splits'
PUBLISH_SPLIT_LINE = '[PACKED-PUBLISH-SPLIT] round={round} entry={entry} {splits}'


def format_publish_splits(splits):
    from dflash_traced_publish import PUBLICATION_SPLIT_NAMES

    return ' '.join('%s=%.2f' % (name, splits.get(name, 0.0)) for name in PUBLICATION_SPLIT_NAMES)


# Round-fence plan H1a (verify_prestage.py), under QWEN_FAST_PACKED_AUDIT=1 and only for a block
# built with QWEN_FAST_PRESTAGE or QWEN_FAST_ROUND_FENCES: one line per ROUND, after its commits,
# in the key=value form round_timeline.py parses. fence: how this round's replay was armed - 'f9'
# (the drafts' fence, verify_prestage.WhileWaiting.fenced), 'replay' (one fence at replay, its
# cost in replay_ms), 'first' (the block's first round runs its trace plainly) or '-' without
# QWEN_FAST_ROUND_FENCES; validated: whether the round's commits skipped their first validate
# (validated_this_round, plumbed from the verify's replay into packed_verifier.commit_user);
# commit_sync_ms: the round's last commit beyond its own trace (its fence, ~0 once F8 moves);
# path/prestage_ms/diff_ms/write_ms: the verify's input path under QWEN_FAST_PRESTAGE ('off'
# without) - the window's pre-stage cost that this round used, the verify-time recompute and
# diff, and the verify-time write. Defined here, at the end, like PUBLISH_SPLIT_LINE.
FENCES_LINE = ('[PACKED-FENCES] round={round} fence={fence} validated={validated} replay_ms={replay_ms} '
               'commit_sync_ms={commit_sync_ms} path={path} prestage_ms={prestage_ms} diff_ms={diff_ms} '
               'write_ms={write_ms}')


def fences_fields(metrics, block, segments):
    """FENCES_LINE's fields for this round, from the verify's metrics and the block."""
    prestage = metrics.get('prestage') or {}
    commit_block_ms = getattr(block, 'commit_block_ms', None)
    last = segments[-1] if segments else None
    commit_sync = commit_block_ms[last] if commit_block_ms is not None and last is not None else 0.0
    return dict(round=getattr(block, 'rounds', 0), fence=metrics.get('replay_fence') or '-',
                validated=int(bool(metrics.get('validated_this_round', False))),
                replay_ms='%.2f' % float(metrics.get('replay_fence_ms') or 0.0),
                commit_sync_ms='%.2f' % float(commit_sync), path=prestage.get('path', 'off'),
                prestage_ms='%.2f' % float(prestage.get('prestage_ms', 0.0)),
                diff_ms='%.2f' % float(prestage.get('diff_ms', 0.0)),
                write_ms='%.2f' % float(prestage.get('write_ms', 0.0)))


# S2 D1 (design section 4, W5a/W5b): survivor narrowing and the abort of a refused round, defined
# here at the end beside PUBLISH_SPLIT_LINE. Unlike that one, S2 did NOT keep the lines above at
# their numbers: the module docstring's S2 paragraph (from :33), the SimpleNamespace import and the
# narrowing branches in packed_device_step, packed_device_rounds and refuse_round shift every line
# after them, so match a [PACKED*] line's loguru prefix, or a traceback, against the source the
# image actually carries, never a pre-S2 copy.
#
# One NARROWED_MARKER line per request whose ticket was cut, written only once every cut the round
# needed has succeeded and the round goes to the sequential step - so a count of them is the count
# of narrowed survivors the engine actually served, never one that refuse_round then failed (M6's
# "survivor narrowed" check); NARROWING_REFUSED_MARKER when a ticket could not be cut and the round
# went to refuse_round; one ABORTED_LINE per request refuse_round ended through the D2 request
# quarantine. The log capture truncates around 250 characters and loguru's prefix takes some:
# ABORTED_LINE stays within 180 for any request id (cut to 48 as everywhere here), its id last;
# the other two carry their free text (the reason, the failure) last, so a cut loses only that.
NARROWED_MARKER = '[PINDIAG] packed survivor narrowed'
NARROWING_REFUSED_MARKER = '[PINDIAG] packed survivor narrowing refused'
ABORTED_MARKER = '[PINDIAG] packed refused round aborted'
ABORTED_LINE = (ABORTED_MARKER + ' {index}/{count} FINISHED_ABORTED via the request quarantine, engine kept: '
                'request={request}')
NARROW_WIDTHS = (32, 16, 8, 4, 2, 1)


def step_log(text):
    """One narrowing or abort line into the server log (loguru, else stdout)."""
    try:
        from loguru import logger
    except ImportError:
        print(text, flush=True)
        return
    logger.warning('{}', text)


def servable_width(engine, ticket):
    """The widest verifier bucket no wider than `ticket` whose cut of it `engine` serves (its own
    captures hold it: VerifierEngine.serves, which reads only the rows and the position), or None -
    also for an engine without the serves contract."""
    serves = getattr(engine, 'serves', None)
    if not callable(serves):
        return None
    for rows in NARROW_WIDTHS:
        if rows > len(ticket.tokens):
            continue
        cut = SimpleNamespace(request_id=ticket.request_id, epoch=getattr(ticket, 'epoch', None),
                              position=ticket.position, tokens=tuple(ticket.tokens[:rows]),
                              source=getattr(ticket, 'source', None), match_length=getattr(ticket, 'match_length', 0))
        if serves(cut):
            return rows
    return None


def narrow_round(entries, reason):
    """S2 D1 (W5a): every entry whose ticket its own engine does not serve, cut to the widest width
    that engine does (GreedySession.narrow: same position and seed, its leading proposals, a new
    epoch), so the round can go to the sequential step instead of refuse_round. Returns
    (entries, None) - new entry dicts carrying the narrowed tickets, the others as they were - or
    (entries, why) when one cannot be cut: every width is planned before any session changes, so
    then nothing was narrowed, unless a session refused midway, whose entries narrowed before it are
    returned with their live tickets for refuse_round to fail. An entry already servable (or whose
    engine has no serves contract) keeps its ticket. The NARROWED_MARKER lines are written only on
    (entries, None): an entry cut before a later one refused is failed with the round, not served."""
    widths = []
    for entry in entries:
        engine, ticket = entry['request'].engine, entry['ticket']
        serves = getattr(engine, 'serves', None)
        if not callable(serves) or serves(ticket):
            widths.append(None)
            continue
        width = servable_width(engine, ticket)
        if width is None:
            why = 'request=%s rows=%d no narrower width its engine captures' % (
                str(entry['request_id'])[:48], len(ticket.tokens))
            step_log('%s %s' % (NARROWING_REFUSED_MARKER, why))
            return entries, why
        widths.append(width)
    narrowed, lines = list(entries), []
    for index, (entry, width) in enumerate(zip(entries, widths)):
        if width is None:
            continue
        rows = len(entry['ticket'].tokens)
        try:
            ticket = entry['request'].session.narrow(entry['request_id'], entry['ticket'], width)
        except Exception as failure:
            why = 'request=%s rows=%d->%d %s: %s' % (str(entry['request_id'])[:48], rows, width,
                                                     type(failure).__name__, str(failure)[:80])
            step_log('%s %s' % (NARROWING_REFUSED_MARKER, why))
            # the entries already cut go to refuse_round with the rest: none is served, so the
            # NARROWED_MARKER lines buffered for them are dropped
            return narrowed, why
        narrowed[index] = dict(entry, ticket=ticket)
        lines.append('%s request=%s rows=%d->%d reason=%s' % (NARROWED_MARKER, str(entry['request_id'])[:48], rows,
                                                              width, str(reason).replace(' ', '_')[:120]))
    for line in lines:
        step_log(line)
    return narrowed, None


def abort_refused(entries, reason):
    """S2 W5b: end a refused round's requests as FINISHED_ABORTED in this engine step, keeping the
    engine. Each is registered with the D2 request quarantine (serving_request_quarantine.register),
    whose scheduler-side wrapper finishes it right after this step's output, zero new tokens and
    all (it adds the finishing EngineCoreOutput itself), and frees its KV blocks; the next step
    names it in finished_req_ids and the lifecycle detaches its bridge like any finished request.
    Its session is marked finished, so the drafts between the two (inside this step under
    QWEN_FAST_EARLY_DRAFT, else post_step's take_draft_token_ids) skip it - FastRunnerBridge.drafts
    returns None for a finished session, proposal_rows counts only unfinished ones - instead of
    raising on its failed session (VR4 finding 2).

    Only when the quarantine's consumer is installed in this process (QWEN_FAST_ANY_REQUEST's
    serving_lifecycle installs it on the scheduler class the engine runs). Without it - the exact
    profile, any image without the module, a non-string request id - nothing is registered or
    marked and refuse_round is exactly what it was. Otherwise EVERY request of the round is
    registered and marked - the servable, the narrowed and the unservable alike, since fail_round
    failed each one's session - with one ABORTED_LINE each. Returns the request ids registered."""
    try:
        import serving_request_quarantine as quarantine
    except ImportError:
        return ()
    installed = getattr(quarantine, 'consumer_installed', None)
    if not callable(installed) or not installed():
        return ()
    ids = [entry['request_id'] for entry in entries]
    if not ids or any(not isinstance(request_id, str) or not request_id for request_id in ids):
        return ()
    for index, entry in enumerate(entries, 1):
        quarantine.register(entry['request_id'], 'packed round refused: %s' % str(reason)[:200])
        entry['request'].session.finished = True
        step_log(ABORTED_LINE.format(index=index, count=len(entries), request=entry['request_id'][:48]))
    return tuple(ids)
