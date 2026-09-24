"""Round-fence plan H1a: pre-stage the next packed verify inside the drafts' fence window, guarded
by a VALUE DIFF, and the round's fence diet. Three flags, every one default off:

  QWEN_FAST_PRESTAGE        the pre-stage (this module, packed_verifier.verify)
  QWEN_FAST_PRESTAGE_AUDIT  reads back a rotating AUDIT_BUFFERS of the staged buffers after each
                            verify-time write and compares them with the verify-time values
  QWEN_FAST_ROUND_FENCES    the fence diet (gdn_records.RetainedGDNBlock, packed_verifier)

WHAT THE PRE-STAGE BUYS. packed_verifier.stage_packed restages ~140 captured buffers before every
verify (tokens, positions, rotary tables, singleton positions, per-row and per-tile page tables,
each reader's positions word and bundle tables) and fences (F1); verify() first re-checks every
binding (bind_ms). All of it but the tokens depends only on each user's next frontier and page
table, both known once the round's commits are done. So the drafts' fence window - the host
waiting for the two pair proposals (dflash_packed_proposal_coordinator.prepare) - computes and
writes every verify input except the tokens (`BlockPrestage.prestage`, with no fence of its own:
the window's fence F9 follows it) and validates the block's bindings.

WHY A VALUE DIFF. The pages are refreshed at execute time, after the window (serving_packed_bridge,
page_binding.refresh), and an unrefreshed page entry names the user's own first block, so a missed
page change would write K/V into the wrong place. The pre-stage therefore keeps no key of what it
thinks the values depend on: it keeps the raw host VALUES it wrote, per destination (`Snapshot`).
At verify the host values are recomputed from the verify-time tickets and tables - the ground
truth, through the same packed_values stage_packed uses, T2 K/V guard included - and only the
buffers whose value differs are copied: the tokens always, a page change a few more. Nothing is
fenced (F1 goes: the trace follows on the same in-order CQ0), and verify() skips validate_bindings
(it ran in the window).

THE ONE INVARIANT LEFT is that nothing else wrote the fixture's buffers between the pre-stage and
the verify. The WRITE EPOCH enforces it: every other writer of fixture inputs bumps it (`bump`) -
the full stage_packed (so the padded probe's restage), a prefill or a Lever N prefill chunk
(serving_lifecycle, the worker hook's pass-through), an admission or a detach (the hook), and the
verify itself - and a snapshot whose epoch moved takes today's full stage_packed. A snapshot serves
one verify at most.

FAILURE. A pre-stage that raises drops its snapshot (the epoch was bumped before its first copy)
and never fails the round: the next verify takes the full path, which restages every buffer. The
pre-stage never poisons a reader and never moves a reader's `start` (both only at verify, as
today). The diff write poisons on failure exactly as stage_packed does.

THE AUDIT (QWEN_FAST_PRESTAGE_AUDIT, with QWEN_FAST_PRESTAGE). After every verify-time write, diff
or full, AUDIT_BUFFERS destinations in rotation are read back from both chips and compared with
the verify-time values. A mismatch is logged (AUDIT_MARKER ... mismatches=N) and the round is
restaged in full before its trace, so the round stays exact; the gate fails the arm on it.

ROUND FENCES (QWEN_FAST_ROUND_FENCES; gdn_records.RetainedGDNBlock.use_round_fences). F3: the
retained block's replay drops the fence after the (blocking) trace and the second validate. F8: the
round's last commit no longer fences; the replay is armed by the drafts' fence F9
(`WhileWaiting.fenced` -> note_round_fence) or, when no draft fenced, by one fence at replay. The
first commit skips its validate when the round's replay validated (validated_this_round). A replay
still needs every segment decided, and a poisoned block still refuses.
"""

import os
import sys
import time

PRESTAGE_FLAG = 'QWEN_FAST_PRESTAGE'
PRESTAGE_AUDIT_FLAG = 'QWEN_FAST_PRESTAGE_AUDIT'
ROUND_FENCES_FLAG = 'QWEN_FAST_ROUND_FENCES'

# Once at attach, from the block that engaged the flag (packed_verifier).
ENGAGED_MARKER = '[PINDIAG] verify prestage engaged'
FENCES_ENGAGED_MARKER = '[PINDIAG] round fences engaged'
# Per round: the window's pre-stage (or why it dropped), the verify's path, the audit, and the
# step's fence line (serving_packed_step, under QWEN_FAST_PACKED_AUDIT).
WINDOW_MARKER = '[PACKED-PRESTAGE-WINDOW]'
MARKER = '[PACKED-PRESTAGE]'
AUDIT_MARKER = '[PACKED-PRESTAGE-AUDIT]'
FENCES_MARKER = '[PACKED-FENCES]'
AUDIT_BUFFERS = 8


def _flag(name, environ):
    value = (os.environ if environ is None else environ).get(name, '0')
    if value not in ('0', '1'):
        raise ValueError('%s must be 0 or 1' % name)
    return value == '1'


def enabled(environ=None):
    """QWEN_FAST_PRESTAGE=1. Any value but 0 or 1 is a configuration error."""
    return _flag(PRESTAGE_FLAG, environ)


def audit_enabled(environ=None):
    """QWEN_FAST_PRESTAGE_AUDIT=1 under QWEN_FAST_PRESTAGE=1 (alone it audits nothing: the gate
    and the arm refuse it; here it is inert)."""
    return _flag(PRESTAGE_AUDIT_FLAG, environ) and enabled(environ)


def round_fences_enabled(environ=None):
    """QWEN_FAST_ROUND_FENCES=1."""
    return _flag(ROUND_FENCES_FLAG, environ)


def any_enabled(environ=None):
    """Whether the worker hook has a window callable to build at all (either flag)."""
    return enabled(environ) or round_fences_enabled(environ)


# The fixture write epoch: one counter per process (every packed block's fixture lives in this
# process), bumped by every writer of fixture inputs other than the pre-stage it invalidates.
_EPOCH = [0, 'start']


def epoch():
    return _EPOCH[0]


def last_bump():
    return _EPOCH[1]


def bump(reason):
    """Another writer touched (or may have touched) a packed fixture's inputs: every snapshot
    taken before now is stale. Host only, never raises."""
    _EPOCH[0] += 1
    _EPOCH[1] = str(reason)


def log_line(text):
    """One line into the server log: loguru where it exists, stderr otherwise. Never raises."""
    try:
        try:
            from loguru import logger
        except ImportError:
            print(text, file=sys.stderr, flush=True)
        else:
            logger.info('{}', text)
    except BaseException:
        pass


def same_value(expected, actual):
    """Whether two host tensors hold the same values in the same shape and dtype."""
    import torch

    return (actual.dtype == expected.dtype and tuple(actual.shape) == tuple(expected.shape)
            and bool(torch.equal(actual, expected)))


def same_readback(expected, actual):
    """Whether a chip's readback holds the staged values (element for element; float64 holds
    every int32, uint32 and bfloat16 value exactly)."""
    import torch

    expected, actual = expected.reshape(-1), actual.reshape(-1)
    if actual.numel() != expected.numel():
        return False
    try:
        return bool(torch.equal(actual.to(torch.float64), expected.to(torch.float64)))
    except (RuntimeError, TypeError):
        return actual.tolist() == expected.tolist()


class Snapshot:
    """What one pre-stage wrote: every destination of the fixture's staging list, in order, and
    the raw host value written into each - None for the tokens, never pre-staged."""

    __slots__ = ('epoch', 'destinations', 'values', 'buffers', 'ms', 'round')

    def __init__(self, epoch_value, destinations, values, buffers, ms, round_number):
        self.epoch, self.destinations, self.values = epoch_value, tuple(destinations), list(values)
        self.buffers, self.ms, self.round = buffers, ms, round_number


class BlockPrestage:
    """The pre-stage state of one PackedVerifierEngine (built by it under QWEN_FAST_PRESTAGE)."""

    def __init__(self, block, *, audit):
        self.block, self.audit = block, bool(audit)
        self.snapshot = None
        self.dropped = 'no-snapshot'
        # The host tensors of the last unfenced write, kept alive until the next write replaces
        # them - after at least one fence (F9, or the verify's blocking trace).
        self.inflight = ()
        self.cursor = 0
        self.counts = dict(prestaged=0, dropped=0, diff=0, full=0, audited=0, mismatches=0)
        # The last verify's split, for serving_packed_step's fence line.
        self.last = dict(path='off', buffers=0, reason='-', prestage_ms=0.0, diff_ms=0.0, write_ms=0.0)

    # -- the window ---------------------------------------------------------------------------
    def drop(self, reason):
        self.snapshot = None
        self.dropped = str(reason).replace(' ', '_')[:120]
        self.counts['dropped'] += 1

    def users_for(self, requests):
        """The next round's users in segment order, from the live requests' frontiers and page
        tables - (placeholder tokens, start, table) per live segment, idle segments filled as
        the verify fills them - and the live count. Raises for a round this block will not
        serve as one pass."""
        block = self.block
        live = [request for request in requests if not getattr(request.session, 'finished', False)]
        users = [None] * block.users
        for request in live:
            segment = block.segment_of(request.engine)
            if users[segment] is not None:
                raise ValueError('two live requests through segment %d' % segment)
            users[segment] = ((0,) * block.rows_per_user, request.session.position, request.engine.pages)
        segments = tuple(segment for segment, user in enumerate(users) if user is not None)
        if len(segments) != block.users:
            if not block.pads(len(segments)):
                raise ValueError('%d live requests is not a round of this block' % len(segments))
            users = block.padded_users(users, segments)
        return users, len(segments)

    def prestage_requests(self, requests):
        """The window's pre-stage for these requests (serving_worker_hook, through WhileWaiting)."""
        users, live = self.users_for(requests)
        self.prestage(users, live=live)

    def prestage(self, users, live=None):
        """Write every verify input but the tokens for `users` (segment order), validate the
        bindings, and keep the snapshot. No fence: the window's F9 follows."""
        from packed_verifier import packed_values, write_packed

        block = self.block
        self.snapshot = None
        self.dropped = 'prestage-incomplete'
        round_number = block.rounds + 1
        if block.phase != 'idle':
            raise ValueError('the block is %s, not idle' % block.phase)
        # Invalidates any older snapshot before the first copy: a pre-stage that fails part way
        # leaves nothing a verify could diff against.
        bump('prestage')
        started = time.perf_counter()
        block.validate_bindings()
        values, readers = packed_values(block.operations, block.model, block.fixture, block.shape, users, guard=False)
        tokens = block.fixture.tokens
        indices = [index for index, value in enumerate(values) if value[0] is not tokens]
        self.inflight = write_packed(block.operations, block.model, values, readers, indices=indices, fence=False,
                                     poison=False)
        ms = (time.perf_counter() - started) * 1000
        self.snapshot = Snapshot(epoch(), [value[0] for value in values],
                                 [None if value[0] is tokens else value[1:] for value in values],
                                 len(indices), ms, round_number)
        self.dropped = None
        self.counts['prestaged'] += 1
        log_line('%s round=%d buffers=%d ms=%.2f live=%s' % (WINDOW_MARKER, round_number, len(indices), ms,
                                                            '-' if live is None else live))

    # -- the verify ---------------------------------------------------------------------------
    def usable(self):
        """(snapshot, None) when the snapshot can serve this verify, else (None, reason)."""
        snapshot = self.snapshot
        if snapshot is None:
            return None, self.dropped or 'no-snapshot'
        if snapshot.epoch != epoch():
            return None, 'epoch:%s' % last_bump()
        return snapshot, None

    def stage(self, entries, segments, snapshot, reason):
        """The verify-time write: the diff against `snapshot` when there is one, else today's
        full stage_packed. Returns the buffers written; logs MARKER; audits."""
        from packed_verifier import packed_values, write_packed

        block = self.block
        round_number = block.rounds + 1
        users = block.segment_users(entries, segments)
        if len(segments) < block.users:
            users = block.padded_users(users, segments)
        self.snapshot = None
        self.dropped = 'no-snapshot'
        prestage_ms = 0.0 if snapshot is None else snapshot.ms
        started = time.perf_counter()
        values = None
        if snapshot is not None:
            values, readers = packed_values(block.operations, block.model, block.fixture, block.shape, users)
            if len(values) != len(snapshot.destinations) or any(
                    value[0] is not destination for value, destination in zip(values, snapshot.destinations)):
                snapshot, reason = None, 'destinations'
        if snapshot is None:
            diffed = started
            written = block.stage_packed_inputs(entries)
            path = 'full'
            self.counts['full'] += 1
        else:
            changed = [index for index, (value, kept) in enumerate(zip(values, snapshot.values))
                       if kept is None or kept[1:] != value[2:] or not same_value(kept[0], value[1])]
            diffed = time.perf_counter()
            try:
                self.inflight = write_packed(block.operations, block.model, values, readers, indices=changed,
                                             fence=False)
            except BaseException:
                bump('verify-failed')
                raise
            for own, user in zip(readers, users, strict=True):
                own.start = user[1]
            bump('verify')
            written, path, reason = len(changed), 'diff', '-'
            self.counts['diff'] += 1
        finished = time.perf_counter()
        self.last = dict(path=path, buffers=written, reason=reason, prestage_ms=prestage_ms,
                         diff_ms=(diffed - started) * 1000, write_ms=(finished - diffed) * 1000)
        log_line('%s round=%d path=%s buffers=%d reason=%s live=%d' % (
            MARKER, round_number, path, written, str(reason).replace(' ', '_')[:120], len(segments)))
        if self.audit:
            self.audit_round(users, round_number, path)
        return written

    def audit_round(self, users, round_number, path):
        """QWEN_FAST_PRESTAGE_AUDIT: AUDIT_BUFFERS destinations in rotation, read back from both
        chips, against the verify-time values; a mismatch restages the round in full."""
        from packed_verifier import packed_values, stage_packed

        block = self.block
        operations = block.operations
        values, unused = packed_values(operations, block.model, block.fixture, block.shape, users, guard=False)
        count = len(values)
        # `first` is where this round's rotation starts (the marker's first=), not the smallest
        # index checked: past the end of the list the eight wrap round to index 0.
        first = self.cursor % count if count else -1
        indices = sorted({(self.cursor + offset) % count for offset in range(min(AUDIT_BUFFERS, count))})
        self.cursor = (self.cursor + AUDIT_BUFFERS) % count if count else 0
        mismatched = []
        for index in indices:
            destination, value = values[index][0], values[index][1]
            for chip, shard in enumerate(operations.get_device_tensors(destination)):
                if not same_readback(value, operations.to_torch(shard)):
                    mismatched.append('%d.%d' % (index, chip))
        self.counts['audited'] += len(indices)
        self.counts['mismatches'] += len(mismatched)
        log_line('%s round=%d path=%s checked=%d first=%d mismatches=%d%s' % (
            AUDIT_MARKER, round_number, path, len(indices), first, len(mismatched),
            (' at=%s' % ','.join(mismatched[:8])) if mismatched else ''))
        if mismatched:
            # The round must stay exact whatever the audit found: every buffer again, fenced
            # (stage_packed bumps the epoch and sets every reader's start).
            stage_packed(operations, block.model, block.fixture, block.shape, users)


class WhileWaiting:
    """What the drafts' fence window runs for one packed block: `coordinator.prepare(...,
    while_waiting=this)` calls it just before its fence F9, after both pairs are enqueued, and
    `fenced()` right after that fence. Built by serving_packed_step.PackedStep.while_waiting only
    when the coming round is this block's packed round and a flag is on."""

    def __init__(self, block, requests):
        self.block, self.requests = block, list(requests)
        self.token = None

    def __call__(self):
        block = self.block
        if getattr(block, 'round_fences', False):
            # Taken before the fence: only commits enqueued before it may be armed by it.
            self.token = block.fence_token()
        prestaged = getattr(block, 'prestaged', None)
        if prestaged is not None:
            prestaged.prestage_requests(self.requests)

    def drop(self, failure):
        """The pre-stage raised (the coordinator catches it): its snapshot is gone and the round
        goes on - the verify takes the full path."""
        prestaged = getattr(self.block, 'prestaged', None)
        if prestaged is None:
            return
        reason = 'prestage-failed:%s' % type(failure).__name__
        prestaged.drop(reason)
        log_line('%s round=%d dropped=%s detail=%s' % (WINDOW_MARKER, self.block.rounds + 1, reason,
                                                       str(failure).replace(' ', '_')[:120]))

    def fenced(self):
        """F9 has just drained CQ0: arm the retained block's replay (round fences)."""
        if self.token is not None:
            self.block.note_round_fence(self.token)
            self.token = None
