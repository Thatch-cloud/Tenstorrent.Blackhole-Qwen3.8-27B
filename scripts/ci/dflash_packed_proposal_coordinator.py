"""QWEN_FAST_PACKED_PROPOSAL round-level wiring: pair eligible bridges by their fixed
pool slot (dflash_packed_proposal.FOUR_AS_TWO_PAIRS), run one traced packed pass per
full pair (dflash_proposal_trace.PreparedPackedDFlashProposal) instead of two separate
single-user passes, and hand each pair's per-user result back through the SAME
has_pending(seed)/finish(count) surface DFlashDevice.propose() already calls on
device.proposal_capture - so neither DFlashDevice nor a bridge's own drafts() needs to
change at all.

Entirely inert with the flag off: PackedProposalCoordinator is only ever constructed
from inside serving_worker_hook.FastWorkerHook._drafts's QWEN_FAST_PACKED_PROPOSAL
branch, so a hook that never takes that branch never creates one, never stores one on
itself, and never touches a device's proposal_capture - the flag-off path is exactly
prepare_pipelined_drafts(), unchanged.

Packing is gated to history_rows == 2048 on BOTH paired devices (steady-state decode,
past the prefill ramp) - see PreparedPackedDFlashProposal's own docstring for why a
narrower, still-growing history is deliberately left unpacked rather than recaptured
every round.
"""

import os
import time


AUDIT_FLAG = 'QWEN_FAST_PACKED_AUDIT'
AUDIT_LINE = '[PACKED-PROPOSE] round={round} pairs={pairs} propose_ms={propose_ms}'
PACKED_CONTEXT = 2048


def audit_enabled(environ=None):
    return (os.environ if environ is None else environ).get(AUDIT_FLAG) == '1'


def audit_log(message, **values):
    """One loguru INFO line, brace-formatted; plain print where loguru is absent
    (host tests) - dflash_device.pindiag's own pattern."""
    try:
        from loguru import logger
    except ImportError:
        print(message.format(**values), flush=True)
        return
    logger.info(message, **values)


def packable(device_a, device_b):
    """Both paired devices at the permanent steady-state context, both running the
    qualified native-proposal path, and both carrying a real single-user
    proposal_capture (QWEN_FAST_EAGER_PROPOSAL off) for _PackedCaptureView to fall
    through to - a device with none (proposal_capture is None: DFlashDevice.
    prepare_device already declines it unconditionally) must never be wrapped, or a
    later round that falls back to its single-user path (its partner finished,
    breaking the pair) would delegate has_pending()/finish()/propose() onto None
    and crash. draft_kv_history.DraftKVHistory requires history_rows ==
    min(position, 2048), so history_rows == 2048 is monotonic and permanent once
    reached - this never flips back to False for a pair that has started packing."""
    return (getattr(device_a, 'history_rows', None) == PACKED_CONTEXT
            and getattr(device_b, 'history_rows', None) == PACKED_CONTEXT
            and getattr(device_a, 'native_proposal_attention', False)
            and getattr(device_b, 'native_proposal_attention', False)
            and getattr(device_a, 'proposal_capture', None) is not None
            and getattr(device_b, 'proposal_capture', None) is not None)


class _PackedCaptureView:
    """Transparent proxy for device.proposal_capture, installed on each half of a
    packed pair for the rounds it packs. Presents has_pending(seed)/finish(count)
    over the shared PreparedPackedDFlashProposal when THIS round's packed
    prepare_device() is still pending, and falls through to the device's own
    original single-user capture for anything else - a seed this round's pack did
    not prepare, or any other proposal_capture method DFlashDevice.propose()/close()
    reaches. has_pending(seed) and the finish(count) that follows it are always
    called back to back with nothing in between (DFlashDevice.propose()), so caching
    which side matched between the two calls is safe."""

    def __init__(self, trace, which, original):
        self._trace, self._which, self._original = trace, which, original
        self._matched = False

    def has_pending(self, seed):
        self._matched = self._trace.has_pending(self._which, seed)
        if self._matched:
            return True
        original_has_pending = getattr(self._original, 'has_pending', None)
        return bool(callable(original_has_pending) and original_has_pending(seed))

    def finish(self, count):
        if self._matched:
            self._matched = False
            return self._trace.finish(self._which, count)
        return self._original.finish(count)

    def propose(self, seed, count):
        return self._original.propose(seed, count)

    def prepare_device(self, seed):
        return self._original.prepare_device(seed)

    def discard_pending(self):
        # The shared trace's own pending is this pair's, not just this side's -
        # PackedProposalCoordinator.prepare() discards it directly (both sides) on a
        # phase-A failure, bypassing this proxy entirely, so this only ever needs to
        # forward to the per-device single-user capture underneath.
        original_discard = getattr(self._original, 'discard_pending', None)
        if callable(original_discard):
            original_discard()

    def close(self):
        # The shared trace outlives any one device - PackedProposalCoordinator owns
        # closing it (at hook teardown, or when a pair is retired) - so a per-device
        # close() must only close this device's OWN single-user capture underneath.
        return self._original.close()


def _install(device, trace, which):
    """Wrap device.proposal_capture in a _PackedCaptureView for `which` side of
    `trace`, unwrapping any previous view first so repeated rounds never nest -
    always exactly one proxy layer over the true single-user PreparedDFlashProposal."""
    current = device.proposal_capture
    original = current._original if isinstance(current, _PackedCaptureView) else current
    device.proposal_capture = _PackedCaptureView(trace, which, original)


class PackedProposalCoordinator:
    """Owns one PreparedPackedDFlashProposal per FOUR_AS_TWO_PAIRS fixed slot pair for
    the life of this object (one per FastWorkerHook, created lazily on first use),
    rebuilding a pair's trace only when the devices occupying that pair's slots
    change identity (a finished request releases its slot, a new one acquires it) -
    never once per round. A slot pair with only one member active this round, or not
    yet at the packable steady-state context, runs each member's OWN
    device.prepare_device() unchanged, exactly as prepare_pipelined_drafts always
    has; this coordinator never touches a bridge outside a full, packable pair."""

    def __init__(self):
        self.pairs = {}
        self.rounds = 0

    def close(self):
        for _, _, trace in self.pairs.values():
            trace.close()
        self.pairs.clear()

    def _trace_for(self, pair, device_a, device_b):
        cached = self.pairs.get(pair)
        if cached is not None:
            old_a, old_b, trace = cached
            if old_a is device_a and old_b is device_b and not device_a.closed and not device_b.closed:
                return trace
            trace.close()
            del self.pairs[pair]
        from dflash_proposal_trace import PreparedPackedDFlashProposal

        trace = PreparedPackedDFlashProposal(device_a, device_b)
        self.pairs[pair] = (device_a, device_b, trace)
        return trace

    def prepare(self, bridges):
        """Phase A, packed variant: pair eligible bridges by fixed pool slot, run one
        traced pass per full packable pair, prepare_device() unchanged for anything
        left over (an unpaired bridge, a lone survivor of a broken pair, a pair not
        yet past the ramp), then one shared fence covering every device this round
        actually touched - prepare_pipelined_drafts' own contract, so phase B's
        per-bridge drafts() loop needs no changes at all to read any of it back.

        A device that raises while being prepared fails the round exactly as its own
        drafts() call would have, but only after every other already-prepared
        device's enqueued work is fenced and its pending discarded - mirroring
        prepare_pipelined_drafts' own exception path."""
        from dflash_packed_proposal import pair_slots
        from serving_worker_hook import phase, pipelined_device

        self.rounds += 1
        round_number = self.rounds
        entries = []
        for bridge in bridges:
            device = pipelined_device(bridge)
            if device is None:
                continue
            slot = getattr(getattr(device, 'pool_slot', None), 'index', None)
            entries.append(dict(bridge=bridge, device=device, slot=slot, seed=bridge.request.session.seed))
        by_slot = {entry['slot']: entry for entry in entries if entry['slot'] is not None}
        unpaired = [entry for entry in entries if entry['slot'] is None]
        groups = pair_slots(by_slot) if by_slot else []

        prepared, fence = [], None
        pair_labels, pair_ms = [], []

        def prepare_single(entry):
            if entry['device'].prepare_device(entry['seed']):
                prepared.append(entry['device'])
                return True
            return False

        try:
            for group in groups:
                if len(group) == 2:
                    slot_a, slot_b = group
                    entry_a, entry_b = by_slot[slot_a], by_slot[slot_b]
                    device_a, device_b = entry_a['device'], entry_b['device']
                    if packable(device_a, device_b):
                        started = time.perf_counter() if audit_enabled() else None
                        trace = self._trace_for(group, device_a, device_b)
                        ids = '%s,%s' % (entry_a['bridge'].request.session.request_id,
                                         entry_b['bridge'].request.session.request_id)
                        ready = phase('propose_pair', ids,
                                      lambda: trace.prepare_device(entry_a['seed'], entry_b['seed']))
                        if ready:
                            _install(device_a, trace, 'a')
                            _install(device_b, trace, 'b')
                            prepared.extend((device_a, device_b))
                            if fence is None:
                                fence = (device_a.operations, device_a.mesh)
                            if started is not None:
                                pair_labels.append([slot_a, slot_b])
                                pair_ms.append((time.perf_counter() - started) * 1000)
                            continue
                    for entry in (entry_a, entry_b):
                        if prepare_single(entry) and fence is None:
                            fence = (entry['device'].operations, entry['device'].mesh)
                else:
                    entry = by_slot[group[0]]
                    if prepare_single(entry) and fence is None:
                        fence = (entry['device'].operations, entry['device'].mesh)
            for entry in unpaired:
                if prepare_single(entry) and fence is None:
                    fence = (entry['device'].operations, entry['device'].mesh)
        except BaseException:
            if fence is not None:
                fence[0].synchronize_device(fence[1])
            # A packed pair's two devices share ONE trace - discard it once, not once
            # per device, or the second discard_pending() call raises into an
            # exception handler that is already unwinding one.
            discarded_traces = set()
            for device in prepared:
                capture = device.proposal_capture
                if isinstance(capture, _PackedCaptureView):
                    if id(capture._trace) not in discarded_traces:
                        discarded_traces.add(id(capture._trace))
                        capture._trace.discard_pending()
                else:
                    discard = getattr(capture, 'discard_pending', None)
                    if callable(discard):
                        discard()
            raise
        if fence is not None:
            fence[0].synchronize_device(fence[1])
        if audit_enabled() and pair_labels:
            audit_log(AUDIT_LINE, round=round_number, pairs=pair_labels,
                      propose_ms=['%.1f' % value for value in pair_ms])
        return prepared
