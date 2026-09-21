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
the block's rows per user (greedy_session narrows the last block of a budget), more or
fewer entries than the block's users (the survivors after a partner finished - one of two,
or one to three of four), an engine the block was not captured against, or a cancellation
already raised before any device work. The two are never mixed within a round. Either way
native GDN slot 0 is trusted by nobody afterwards (`verifier_engine.note_packed_step`):
the sequential step restores each user's carry first, and the packed trace restores every
segment's carry inside itself.

Every user's commit runs in entries order; the last one is the round's fence. The block
knows it is last - `PackedVerifierEngine.commit_user` fences exactly the commit that
empties its pending segments - so this step adds nothing for that beyond committing every
segment, even on cancellation (prefix 0) or after a failure (prefix 0 for what was left).
"""

import os
import time

from attention_mask_replay import validate_ticket
from serving_fast_request import CommittedOutput
from serving_sequential_step import describe as describe_sequential, sequential_packed_step
from serving_worker_hook import phase
import verifier_engine

# Under QWEN_FAST_PACKED_AUDIT=1, one line per user per round for the token-exact gate
# (docs/packed-device-step-plan-2026-09-20.md section 6): which segment served which
# request at which position, what it accepted and emitted, and the first predictions.
AUDIT_LINE = ('[PACKED] request={request} segment={segment} position={position} '
              'prefix={prefix} emitted={emitted} predictions={predictions}')

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
    __slots__ = ('name', 'sink', 'started')

    def __init__(self, name, sink):
        self.name, self.sink = name, sink

    def __enter__(self):
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

    def stage(name, prefix):
        return _StageTimer(name, sink)

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

    Beside the 64-row block (or the two 32-row blocks together, the same total rows) the
    per-request engines may capture only the sequential widths
    (packed_shapes.sequential_capture_rows), so a block-served round MUST be drafted at the
    block's width and a sequential round (survivors, a last narrow block) at the engines';
    a ticket drafted for a block that then does not serve it has no capture anywhere
    (`unservable`). The draft returns exactly the rows asked of it
    (dflash_request_runtime.propose), so the decision here is the ticket width.
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
        if len(group) != shape.users:
            return None
        if width is None:
            width = shape.rows_per_user
        elif width != shape.rows_per_user:
            return None
        capacity = getattr(matched, 'replay_capacity', None)
        for request in group:
            session = request.session
            if session.max_new_tokens - len(session.emitted) < shape.rows_per_user:
                return None
            if capacity is not None:
                try:
                    validate_ticket(session.position, shape.rows_per_user, capacity, short_context=False)
                except ValueError:
                    return None
    return width


def unservable(entries):
    """Entries whose ticket no capture of their own engine holds: a round drafted for the
    block that the block will not serve cannot go to the sequential step either."""
    refused = []
    for entry in entries:
        serves = getattr(entry['request'].engine, 'serves', None)
        if callable(serves) and not serves(entry['ticket']):
            refused.append('request=%s rows=%d' % (str(entry['request_id'])[:48], len(entry['ticket'].tokens)))
    return refused


class PackedStep:
    """The packed device step bound to its block (or blocks - QWEN_FAST_FOUR_AS_TWO's pair
    of 32-row blocks for four users): the step itself, and the per-round ticket-width policy
    the worker hook asks before drafting (`proposal_rows`).

    With exactly one block this runs the SAME `packed_device_step` a lone block always
    did - `self.block` still names it, as callers that built a `PackedStep` over one block
    have always relied on. With several, each round is partitioned by block and run through
    `packed_device_rounds` instead; `self.block` is then None, since no one block owns the
    round."""

    def __init__(self, blocks):
        if blocks is None:
            raise ValueError('A packed verify block is required')
        self.blocks = tuple(blocks) if isinstance(blocks, (list, tuple)) else (blocks,)
        if not self.blocks:
            raise ValueError('At least one packed verify block is required')
        self.block = self.blocks[0] if len(self.blocks) == 1 else None

    def __call__(self, entries, *, cancelled):
        if len(self.blocks) == 1:
            return packed_device_step(entries, cancelled=cancelled, block=self.blocks[0])
        return packed_device_rounds(entries, cancelled=cancelled, blocks=self.blocks)

    def proposal_rows(self, requests):
        return proposal_rows(self.blocks, requests)


def ineligible(entries, block):
    """Why the block cannot serve this round as one pass, or None when it can."""
    shape = block.shape
    if len(entries) != shape.users:
        return 'entries=%d block_users=%d' % (len(entries), shape.users)
    for entry in entries:
        rows = len(entry['ticket'].tokens)
        if rows != shape.rows_per_user:
            return 'request=%s rows=%d rows_per_user=%d' % (str(entry['request_id'])[:48], rows, shape.rows_per_user)
        try:
            block.segment_of(entry['request'].engine)
        except ValueError:
            return 'request=%s engine not bound to the block' % str(entry['request_id'])[:48]
    return None


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
        segments = metrics['segments']
        if len(predictions) != len(entries) or len(segments) != len(entries):
            raise ValueError('The packed block must return one prediction list and one segment per entry')
        outputs = []
        # Only collected under the audit gate: appending a dict per entry is cheap, but
        # there is no reason to pay even that when nobody will read it.
        commit_host_timings = [] if audit_enabled() else None
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
    reason = ineligible(entries, block)
    if reason is not None:
        # A round drafted for the block (proposal_rows) whose entries changed before the
        # step: its tickets have no capture anywhere, and the session cannot re-propose
        # (fail_verification is final), so the round is failed here, before any device
        # work, with the reason - rather than by the engine's own refusal one step later.
        # serving_worker_hook.discard_stale_ticket keeps every live request's pending
        # ticket at one width per step, so this is unreachable in normal steady-state
        # and transition operation; kept only as a last-resort guard, it degrades this
        # round (refuse_round) instead of raising past the step - a scheduler race must
        # never crash the engine for every OTHER live user (run 35535533720).
        refused = unservable(entries)
        if refused:
            return refuse_round(entries, block, reason, refused)
    elif cancelled():
        # Each request's own step answers a cancellation without touching the device.
        reason = 'cancelled before the verify'
    if reason is not None:
        return sequential_packed_step(entries, cancelled=cancelled)
    return run_verified_block(entries, cancelled=cancelled, block=block)


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
    `entries`' own order, whichever block (or the sequential step) actually produced them."""
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
    for block in blocks:
        group = by_block.get(id(block))
        if group is None:
            continue
        matched, group_entries = group
        reason = ineligible(group_entries, block)
        if reason is not None:
            # Same degrade-not-crash rule as the single-block step, applied to this block's
            # own group: a mixed group with no fallback capture anywhere is refused (failed)
            # here rather than raised past the step for every OTHER live user or block.
            refused = unservable(group_entries)
            if refused:
                for output in refuse_round(group_entries, block, reason, refused):
                    outputs_by_id[output.request_id] = output
                continue
        elif cancelled():
            reason = 'cancelled before the verify'
        if reason is not None:
            sequential_entries.extend(group_entries)
            continue
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
        if merge_release or fused_steady_state:
            restore_merge_release = install_publish_options(drafter,
                merge_release=merge_release, fused_steady_state=fused_steady_state)
    try:
        # session.commit and session.abort both end in runtime.publish - DFlashRequestRuntime.
        # publish (dflash_request_runtime.py) - which runs VerifierEngine.publish and, through
        # it, packed_verifier.PackedVerifierEngine.commit_user: this one interval covers all
        # of that, for either outcome.
        session_started = time.perf_counter()
        if cancelled():
            session.abort(request_id, ticket, runtime.publish)
            request.cancelled = True
            decision = None
            output = CommittedOutput(request_id, (), session.position, True, True)
        else:
            decision = session.commit(request_id, ticket, rows, runtime.publish)
            output = CommittedOutput(request_id, tuple(decision.emitted), session.position, session.finished)
        session_ms = (time.perf_counter() - session_started) * 1000
    finally:
        if restore_merge_release is not None:
            restore_merge_release()
        if restore_stage_timer is not None:
            restore_stage_timer()
    finished = time.perf_counter()
    if audit_enabled():
        audit_log(AUDIT_LINE, request=str(request_id)[:48], segment=segment, position=ticket.position,
                  prefix=0 if decision is None else decision.state_rows,
                  emitted=0 if decision is None else len(decision.emitted), predictions=list(rows[:8]))
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
    request engine captures either - so the sequential step cannot take it either.

    `serving_worker_hook.discard_stale_ticket` keeps every live request's pending
    ticket at one width per step, so this is unreachable in normal steady-state and
    transition operation: reaching it means a scheduler race put a mixed round here
    anyway, not that the hardware or a request failed. Fails every request of the
    round exactly as `fail_round` always has, but returns their outputs instead of
    raising past the step, so the race costs this round for these requests rather
    than the engine for every other live user (run 35535533720)."""
    message = ('A round the block cannot serve (%s) holds tickets no request engine captured (%s): '
              'it was drafted for the block but its entries changed before the step'
              % (reason, '; '.join(refused)))
    try:
        from loguru import logger
        logger.warning('[PACKED] {}', message)
    except ImportError:
        print('[PACKED] %s' % message, flush=True)
    fail_round(entries, block)
    return [CommittedOutput(entry['request_id'], (), entry['request'].session.position, True, True)
            for entry in entries]


def describe():
    """What this costs, beside what it falls back to."""
    return dict(name='packed', weight_passes_per_round='one for all users',
                per_user_rate='the block rate, shared by every user', batched=True,
                fallback=describe_sequential())
