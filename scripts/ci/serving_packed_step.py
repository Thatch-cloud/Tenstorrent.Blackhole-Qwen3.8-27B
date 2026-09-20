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


def proposal_rows(block, requests):
    """The ticket width every live request drafts for the coming round, asked by the worker
    hook before any proposal: the block's rows per user when the block will serve the round
    as one pass - every live request present (exactly the block's users, finished ones
    aside), each with at least that many tokens left, each engine bound to a segment, each
    frontier inside the block's native chunk family - else None, and each engine proposes
    at its own captured width.

    Beside the 64-row block the per-request engines capture only the sequential widths
    (packed_shapes.sequential_capture_rows), so a block-served round MUST be drafted at the
    block's width and a sequential round (survivors, a last narrow block) at the engines';
    a ticket drafted for the block that the block then does not serve has no capture
    anywhere (`unservable`). The draft returns exactly the rows asked of it
    (dflash_request_runtime.propose), so the decision here is the ticket width.
    """
    shape = block.shape
    live = [request for request in requests if not request.session.finished]
    if len(live) != shape.users:
        return None
    capacity = getattr(block, 'replay_capacity', None)
    for request in live:
        session = request.session
        if session.max_new_tokens - len(session.emitted) < shape.rows_per_user:
            return None
        try:
            block.segment_of(request.engine)
            if capacity is not None:
                validate_ticket(session.position, shape.rows_per_user, capacity, short_context=False)
        except ValueError:
            return None
    return shape.rows_per_user


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
    """The packed device step bound to its block: the step itself, and the per-round
    ticket-width policy the worker hook asks before drafting (`proposal_rows`)."""

    def __init__(self, block):
        if block is None:
            raise ValueError('A packed verify block is required')
        self.block = block

    def __call__(self, entries, *, cancelled):
        return packed_device_step(entries, cancelled=cancelled, block=self.block)

    def proposal_rows(self, requests):
        return proposal_rows(self.block, requests)


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


def packed_device_step(entries, *, cancelled, block):
    """Step every packed request through one verify pass, committing in the scheduler's order."""
    entries = list(entries)
    if not entries:
        raise ValueError('A packed step needs at least one admitted request')
    if not callable(cancelled) or block is None:
        raise ValueError('A cancellation callback and the packed verify block are required')
    for entry in entries:
        request, ticket = entry['request'], entry['ticket']
        if ticket.request_id != entry['request_id']:
            raise ValueError('Each packed entry must carry its own prepared ticket')
        # FastRequest.step's own refusal, before any device work
        if request.closed or request.busy or request.cancelled or request.session.pending is not ticket:
            raise ValueError('One live owner with its prepared ticket required for every packed entry')
    reason = ineligible(entries, block)
    if reason is not None:
        # A round drafted for the block (proposal_rows) whose entries changed before the
        # step: its tickets have no capture anywhere, and the session cannot re-propose
        # (fail_verification is final), so the round is failed here, before any device
        # work, with the reason - rather than by the engine's own refusal one step later.
        refused = unservable(entries)
        if refused:
            fail_round(entries, block)
            raise ValueError('A round the block cannot serve (%s) holds tickets no request engine captured (%s): '
                             'it was drafted for the block but its entries changed before the step'
                             % (reason, '; '.join(refused)))
    elif cancelled():
        # Each request's own step answers a cancellation without touching the device.
        reason = 'cancelled before the verify'
    if reason is not None:
        return sequential_packed_step(entries, cancelled=cancelled)
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
        for entry, segment, rows in zip(entries, segments, predictions):
            # entries[i] is served by metrics['segments'][i] and predictions[i]: the
            # segment its engine's carry is bound to, never its index in the entries
            output = phase('packed_commit', entry['request_id'],
                           lambda entry=entry, segment=segment, rows=rows: commit_entry(
                               entry, block, segment, rows, cancelled=cancelled, metrics=metrics,
                               verify_started=verify_started, verified=verified))
            outputs.append(output)
        return outputs
    except BaseException:
        fail_round(entries, block)
        raise
    finally:
        # Slot 0 holds the last committed segment's user or nobody's committed state.
        verifier_engine.note_packed_step()
        for request in requests:
            request.busy = False


def commit_entry(entry, block, segment, rows, *, cancelled, metrics, verify_started, verified):
    """FastRequest.step from its verify readback on, for one user of the block."""
    request, ticket, request_id = entry['request'], entry['ticket'], entry['request_id']
    session, engine, runtime = request.session, request.engine, request.runtime
    started = time.perf_counter()
    # Verified on the block, in this segment: the publication is the block's per-user commit.
    engine.adopt_packed(ticket, block, segment)
    if cancelled():
        session.abort(request_id, ticket, runtime.publish)
        request.cancelled = True
        decision = None
        output = CommittedOutput(request_id, (), session.position, True, True)
    else:
        decision = session.commit(request_id, ticket, rows, runtime.publish)
        output = CommittedOutput(request_id, tuple(decision.emitted), session.position, session.finished)
    finished = time.perf_counter()
    if audit_enabled():
        audit_log(AUDIT_LINE, request=str(request_id)[:48], segment=segment, position=ticket.position,
                  prefix=0 if decision is None else decision.state_rows,
                  emitted=0 if decision is None else len(decision.emitted), predictions=list(rows[:8]))
    if decision is not None and getattr(request, 'collect_timings', False):
        record_timing(request, ticket, decision, metrics, segment,
                      verify_started=verify_started, verified=verified, started=started, finished=finished)
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


def describe():
    """What this costs, beside what it falls back to."""
    return dict(name='packed', weight_passes_per_round='one for all users',
                per_user_rate='the block rate, shared by every user', batched=True,
                fallback=describe_sequential())
