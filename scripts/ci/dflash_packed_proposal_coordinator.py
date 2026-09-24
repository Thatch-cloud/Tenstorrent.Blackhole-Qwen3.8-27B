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
# One line per pair whose first (or any) traced execution raised - run 35581352016
# (image v73): a pair formed the moment packable() first saw both devices at
# history_rows == 2048, and dflash_device.DFlashDevice.execute_proposal refused the
# packed cached_history it was handed ('Every prepared learned layer requires a
# committed K/V cache', dflash_device.py:661-662). See packable()'s own docstring for
# the precondition this line's fallback pair failed anyway, on hardware, despite
# passing every host-side check this module can perform.
PAIR_FALLBACK_LINE = '[PACKED-PROPOSE] pair={pair} fallback={fallback}'
# One line whenever _ensure_single_user rebuilds a device's own single-user
# PreparedDFlashProposal after _release_single_user freed it - a fallback after
# release is exactly the case run 35585107688 needs visible.
RECAPTURE_LINE = '[PACKED-PROPOSE] recapture slot={slot}'
PACKED_CONTEXT = 2048

DRAM_RESERVE_FLAG = 'QWEN_FAST_PACKED_PROPOSAL_DRAM_RESERVE_MB'
DRAM_RESERVE_DEFAULT_MB = 256

# Variable-user packed rounds, M0's fallback (QWEN_FAST_PAIRS_PACKED_ONLY=1, default off): in a
# round the packed step's policy will not serve as one pass (serving_worker_hook._drafts'
# packed_rows is None: a user finished, or one is within a block round of its budget), every
# member of a slot pair proposes on its own single-user capture, as an unpaired bridge does -
# the pair trace runs only in packed rounds. For the case M0's mask audit finds the mask intact
# while the pair users' tail acceptance still collapses: single-proposal users hold 2.76-4.0
# tokens per step in those same steps. The hook passes packed_round only while the flag is on;
# PAIRS_PACKED_ONLY_MARKER is logged once per process, the first round it unpairs a pair.
PAIRS_PACKED_ONLY_FLAG = 'QWEN_FAST_PAIRS_PACKED_ONLY'
PAIRS_PACKED_ONLY_MARKER = '[PINDIAG] pairs packed only'
_PAIRS_PACKED_ONLY_NOTED = []


def pairs_packed_only_enabled(environ=None):
    """QWEN_FAST_PAIRS_PACKED_ONLY=1."""
    return (os.environ if environ is None else environ).get(PAIRS_PACKED_ONLY_FLAG) == '1'


def unpair_groups(groups, round_number):
    """QWEN_FAST_PAIRS_PACKED_ONLY: every member of every group on its own, in group order.
    Logs PAIRS_PACKED_ONLY_MARKER once per process, the first time a full pair is split."""
    pairs = [list(group) for group in groups if len(group) == 2]
    if pairs and not _PAIRS_PACKED_ONLY_NOTED:
        _PAIRS_PACKED_ONLY_NOTED.append(round_number)
        message = '%s: round=%s unpaired=%s (the policy serves this round sequentially)' % (
            PAIRS_PACKED_ONLY_MARKER, round_number, pairs)
        try:
            from loguru import logger
        except ImportError:
            print(message, flush=True)
        else:
            logger.info('{}', message)
    return [(slot,) for group in groups for slot in group]

# One packed pair's own placeholder buffers at the only packable geometry (steady
# state, 2048-row context, T16 block_rows=16): identifiers (1,32) uint32 (~128 B,
# negligible), mask (1,1,32,4160) bf16 (~266 KB), rope.q (2x(1,1,32,128) bf16,
# ~16 KB), rope.k (2x(1,1,4160,128) bf16, ~2.03 MB), rope.live_k (2x(1,1,32,128)
# bf16, ~16 KB), cached_history (2 users x 5 layers x 2 k/v x (1,4,2048,128) bf16,
# ~40 MB - the dominant term). Precisely computed from dflash_proposal_trace.
# PreparedPackedDFlashProposal._bucket's own upload calls and dflash_batched_mask.
# segments' span formula (see the report for the full derivation) - a LOWER BOUND:
# it does not include the trace's own internal compute/intermediate buffers
# (embeddings, 5 layers of attention and MLP intermediates) that capture_operation
# retains for replay, which are not sized from source here.
PAIR_CAPTURE_PLACEHOLDER_BYTES = 128 + 266_240 + 16_384 + 2_129_920 + 16_384 + 41_943_040

# The (1,1,2048,5120) bf16 transient every prepare_publication call allocates and
# releases (the 'padded'/'combined' tensor, general or fused_steady_state path
# alike, dflash_device.py) - exactly the shape and size of the allocation that
# killed run 35585107688's engine (20,971,520 B). A fallback's own single-user
# commit needs this same allocation regardless of packing, so the reserve below
# accounts for BOTH paired users needing it in the worst case (the pair capture
# fails and both fall back to their own commit in the same round).
PREPARE_PUBLICATION_TRANSIENT_BYTES = 2048 * 5120 * 2


def estimated_pair_capture_bytes():
    return PAIR_CAPTURE_PLACEHOLDER_BYTES + 2 * PREPARE_PUBLICATION_TRANSIENT_BYTES


def dram_reserve_bytes(environ=None):
    """QWEN_FAST_PACKED_PROPOSAL_DRAM_RESERVE_MB: an integer number of megabytes,
    default 256. Read at each round rather than at import, so the value a test
    sets is the one the round sees."""
    value = (os.environ if environ is None else environ).get(DRAM_RESERVE_FLAG, str(DRAM_RESERVE_DEFAULT_MB))
    try:
        megabytes = int(value)
    except (TypeError, ValueError):
        raise ValueError('%s must be an integer number of megabytes' % DRAM_RESERVE_FLAG)
    if megabytes < 0:
        raise ValueError('%s must not be negative' % DRAM_RESERVE_FLAG)
    return megabytes * 1024 * 1024


def dram_headroom(device):
    """The smallest largest-contiguous-free-DRAM-block across this device's chips,
    or None if the allocator statistics are unavailable. serving_buffer_pool.
    dram_statistics's own contract is 'a diagnostic, never a gate': unavailable
    statistics here fall through to packing exactly as the reserve check being off
    would, rather than refusing to ever pack in an environment (or a host test)
    that cannot read the allocator at all."""
    from serving_buffer_pool import dram_statistics

    tensor = getattr(device, 'history', None)
    if tensor is None:
        return None
    statistics = dram_statistics(device.operations, tensor)
    if isinstance(statistics, dict):
        return None
    return min(chip['largest_free'] for chip in statistics)


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


def _committed_kv_history(device):
    """The exact precondition dflash_device.DFlashDevice.execute_proposal itself
    checks before accepting ANY packed cached_history for this device
    (dflash_device.py:661-662): `self.kv_history is None or len(cached_history) !=
    len(self.layers)` raises 'Every prepared learned layer requires a committed K/V
    cache'. There is no separate kv_history.committed flag anywhere in draft_kv_history.
    py or dflash_device.py - DraftKVHistory.active is populated once, at construction,
    from the prefill features, not by any later commit, so this is a structural check
    (kv_history exists and holds one active bank per prepared draft layer), not a
    timing one. Checked here so a device this admits is a device execute_proposal's
    OWN guard would structurally accept too - though see PAIR_FALLBACK_LINE and
    PackedProposalCoordinator.prepare(): run 35581352016 hit that same guard anyway,
    on hardware, for a pair both members of which pass every check in this function."""
    kv_history = getattr(device, 'kv_history', None)
    if kv_history is None:
        return False
    active = getattr(kv_history, 'active', None)
    layers = getattr(device, 'layers', None)
    return active is not None and layers is not None and len(active) == len(layers)


def packable(device_a, device_b):
    """Both paired devices at the permanent steady-state context, both running the
    qualified native-proposal path, both carrying a committed K/V history
    (_committed_kv_history, the same precondition execute_proposal itself checks),
    and both carrying a real single-user proposal_capture (QWEN_FAST_EAGER_PROPOSAL
    off) for _PackedCaptureView to fall through to - a device with none
    (proposal_capture is None: DFlashDevice.prepare_device already declines it
    unconditionally) must never be wrapped, or a later round that falls back to its
    single-user path (its partner finished, breaking the pair) would delegate
    has_pending()/finish()/propose() onto None and crash. draft_kv_history.
    DraftKVHistory requires history_rows == min(position, 2048), so history_rows ==
    2048 is monotonic and permanent once reached - this never flips back to False for
    a pair that has started packing.

    Passing every check here is NOT a guarantee execute_proposal will accept the
    pack - PackedProposalCoordinator.prepare() treats a pair's first traced execution
    as fallible regardless, and falls the pair back to single-user for that one round
    on any exception (PAIR_FALLBACK_LINE)."""
    return (getattr(device_a, 'history_rows', None) == PACKED_CONTEXT
            and getattr(device_b, 'history_rows', None) == PACKED_CONTEXT
            and getattr(device_a, 'native_proposal_attention', False)
            and getattr(device_b, 'native_proposal_attention', False)
            and _committed_kv_history(device_a)
            and _committed_kv_history(device_b)
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
        # _original is None whenever PackedProposalCoordinator._release_single_user
        # has freed it and no fallback has needed it back since (this pair packed
        # every round from then until the request itself ended) - nothing to close.
        if self._original is not None:
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
    has; this coordinator never touches a bridge outside a full, packable pair.

    self.pairs[pair] holds (device_a, device_b, trace, released): `released` is True
    once the pair's first successful capture has freed both devices' own single-user
    PreparedDFlashProposal (_release_single_user) - run 35585107688: a packed pair's
    placeholder set duplicates the two single-user traced-proposal placeholder sets,
    so keeping all three allocated at once is pure DRAM duplication once the pair is
    doing the same job. _ensure_single_user rebuilds a released capture, once, the
    first time ANY later round needs the single-user fallback again (the pair's own
    capture failing, or the pair breaking up when a partner finishes)."""

    def __init__(self):
        self.pairs = {}
        self.rounds = 0

    def close(self):
        for _, _, trace, _ in self.pairs.values():
            trace.close()
        self.pairs.clear()

    def _cached_trace(self, pair, device_a, device_b):
        """The pair's already-captured trace if the SAME two devices still occupy
        it and neither has closed, else None - a fresh capture (and the DRAM
        reserve check that must gate one) is needed either way."""
        cached = self.pairs.get(pair)
        if cached is None:
            return None
        old_a, old_b, trace, released = cached
        if old_a is device_a and old_b is device_b and not device_a.closed and not device_b.closed:
            return trace
        return None

    def _trace_for(self, pair, device_a, device_b):
        trace = self._cached_trace(pair, device_a, device_b)
        if trace is not None:
            return trace
        if pair in self.pairs:
            self.pairs[pair][2].close()
            del self.pairs[pair]
        from dflash_proposal_trace import PreparedPackedDFlashProposal

        trace = PreparedPackedDFlashProposal(device_a, device_b)
        self.pairs[pair] = (device_a, device_b, trace, False)
        return trace

    def _release_single_user(self, device):
        """Free device's own single-user PreparedDFlashProposal (reached either
        directly, if no pair has ever formed for it, or through the _PackedCaptureView
        _install put in its place) once its pair has captured successfully - see the
        class docstring for why keeping both is pure duplication. Marks the device so
        _ensure_single_user knows this - and ONLY this - is why its capture (or its
        view's _original) reads None afterwards; a device whose capture is None for
        any OTHER reason (QWEN_FAST_EAGER_PROPOSAL, or one this coordinator never
        touched at all) must never be treated as needing a rebuild. Idempotent: a
        device whose capture is already released is left alone."""
        capture = device.proposal_capture
        if isinstance(capture, _PackedCaptureView):
            if capture._original is not None:
                capture._original.close()
                capture._original = None
                device._packed_capture_released = True
        elif capture is not None:
            capture.close()
            device.proposal_capture = None
            device._packed_capture_released = True

    def _ensure_single_user(self, device):
        """Rebuild device's own single-user PreparedDFlashProposal if
        _release_single_user freed it (device._packed_capture_released, set ONLY by
        that method - never inferred from proposal_capture being None alone, which
        is also the ordinary QWEN_FAST_EAGER_PROPOSAL state this coordinator must
        leave untouched) - reached either as device.proposal_capture directly (no
        pair ever formed for this device) or as the _original a _PackedCaptureView
        falls through to (a pair released it, then broke up or its capture failed
        this round) - device.prepare_device() delegates to whichever of the two it
        finds, and DFlashDevice.propose()'s own fallback (self.proposal_capture.
        propose(...), reached through the SAME view when has_pending() answers
        False) would otherwise run onto None. max_new_tokens=1 reproduces the SAME
        single 2048-row bucket the original ladder would have narrowed to by now
        regardless of its real value - dflash_proposal_inputs.proposal_contexts(
        position, max_new_tokens) returns exactly (2048,) whenever position >= 2048
        (steady state, the only state packable() ever admits a device in) for ANY
        max_new_tokens satisfying its own bound (1 <= max_new_tokens <= 262144 -
        position - 32) - the real original value is not stored anywhere on
        DFlashDevice and is not needed to reproduce its ladder."""
        if not getattr(device, '_packed_capture_released', False):
            return False
        capture = device.proposal_capture
        view = capture if isinstance(capture, _PackedCaptureView) else None
        from dflash_proposal_trace import PreparedDFlashProposal

        rebuilt = PreparedDFlashProposal(device, max_new_tokens=1)
        if view is not None:
            view._original = rebuilt
        else:
            device.proposal_capture = rebuilt
        device._packed_capture_released = False
        if audit_enabled():
            audit_log(RECAPTURE_LINE, slot=getattr(getattr(device, 'pool_slot', None), 'index', None))
        return True

    def prepare(self, bridges, packed_round=None, while_waiting=None):
        """Phase A, packed variant: pair eligible bridges by fixed pool slot, run one
        traced pass per full packable pair, prepare_device() unchanged for anything
        left over (an unpaired bridge, a lone survivor of a broken pair, a pair not
        yet past the ramp), then one shared fence covering every device this round
        actually touched - prepare_pipelined_drafts' own contract, so phase B's
        per-bridge drafts() loop needs no changes at all to read any of it back.

        A device that raises while being prepared fails the round exactly as its own
        drafts() call would have, but only after every other already-prepared
        device's enqueued work is fenced and its pending discarded - mirroring
        prepare_pipelined_drafts' own exception path.

        `packed_round` (QWEN_FAST_PAIRS_PACKED_ONLY only; the hook passes nothing
        otherwise): False when the packed step's policy serves this round sequentially,
        and then, with the flag on, no pair forms - every member prepares on its own
        (unpair_groups).

        `while_waiting` (round-fence plan H1a, verify_prestage.WhileWaiting; the hook passes
        it only when a flag is on and the coming round is the block's packed round): called
        just before this round's one fence, after every pair and single is enqueued, so its
        host work hides under their device time - and only when there is a fence to follow
        it. An Exception it raises is handed to its `drop` and never fails the round. Right
        after the fence its `fenced` is called (the fence has drained everything enqueued
        before it)."""
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
        if packed_round is False and pairs_packed_only_enabled():
            groups = unpair_groups(groups, round_number)

        prepared, fence = [], None
        pair_labels, pair_ms = [], []
        # QWEN_FAST_ROUND_B1 (C1): the traces this round prepared, selected together after
        # the fence below. None with the flag off, and nothing then reads it.
        batched = [] if os.environ.get('QWEN_FAST_ROUND_B1') == '1' else None

        def prepare_single(entry):
            device = entry['device']
            # A no-op unless a pair had already released this device's own single-
            # user capture (PackedProposalCoordinator._release_single_user) and this
            # is the first round since that needs it back - see _ensure_single_user's
            # own docstring for why this must run before prepare_device() below.
            self._ensure_single_user(device)
            if device.prepare_device(entry['seed']):
                prepared.append(device)
                return True
            return False

        try:
            for group in groups:
                if len(group) == 2:
                    slot_a, slot_b = group
                    entry_a, entry_b = by_slot[slot_a], by_slot[slot_b]
                    device_a, device_b = entry_a['device'], entry_b['device']
                    if packable(device_a, device_b):
                        fresh_build = self._cached_trace(group, device_a, device_b) is None
                        headroom_ok = True
                        if fresh_build:
                            # Only a FRESH capture allocates new placeholder buffers -
                            # replaying an already-built trace does not - so the
                            # reserve check only ever gates the first round a pair
                            # forms, never every round after (run 35585107688).
                            headroom = dram_headroom(device_a)
                            if headroom is not None and headroom < estimated_pair_capture_bytes() + dram_reserve_bytes():
                                headroom_ok = False
                                if audit_enabled():
                                    audit_log(PAIR_FALLBACK_LINE, pair=[slot_a, slot_b], fallback='dram_reserve')
                        if headroom_ok:
                            started = time.perf_counter() if audit_enabled() else None
                            trace = self._trace_for(group, device_a, device_b)
                            from dflash_proposal_trace import pair_mask_audit_enabled

                            if pair_mask_audit_enabled():
                                # QWEN_FAST_PAIR_MASK_AUDIT (M0): this round's number for the
                                # trace's '[PACKED-PROPOSE] mask round=' lines.
                                trace.round_number = round_number
                            ids = '%s,%s' % (entry_a['bridge'].request.session.request_id,
                                             entry_b['bridge'].request.session.request_id)
                            try:
                                ready = phase('propose_pair', ids,
                                              lambda: trace.prepare_device(entry_a['seed'], entry_b['seed']))
                            except Exception as failure:
                                # The four-user gate must never lose the engine to a
                                # proposal-path refusal (run 35581352016: packable() passed
                                # every host-side check yet execute_proposal still raised on
                                # the pair's first traced execution) - this pair falls back
                                # to single-user for THIS round only. _bucket()'s own
                                # exception handling already removed any partially-built
                                # bucket (and every placeholder it uploaded, run
                                # 35585107688) before re-raising, so the trace object itself
                                # stays valid; discard_pending() is a safe no-op here
                                # (prepare_device() never reached setting self._pending
                                # before its own _bucket() call raised) but is called
                                # anyway, matching every other failure path in this module.
                                # packable() is re-evaluated fresh next round - a permanent-
                                # shaped failure (see packable()'s own docstring) then just
                                # falls back every round, harmlessly, rather than being
                                # retried once and blacklisted.
                                trace.discard_pending()
                                if audit_enabled():
                                    audit_log(PAIR_FALLBACK_LINE, pair=[slot_a, slot_b],
                                              fallback='%s: %s' % (type(failure).__name__, str(failure)[:160]))
                                ready = False
                        else:
                            ready = False
                        if ready:
                            _install(device_a, trace, 'a')
                            _install(device_b, trace, 'b')
                            prepared.extend((device_a, device_b))
                            if batched is not None:
                                batched.append(([slot_a, slot_b], trace))
                            if fence is None:
                                fence = (device_a.operations, device_a.mesh)
                            if started is not None:
                                pair_labels.append([slot_a, slot_b])
                                pair_ms.append((time.perf_counter() - started) * 1000)
                            cached = self.pairs.get(group)
                            if cached is not None and not cached[3]:
                                self._release_single_user(device_a)
                                self._release_single_user(device_b)
                                self.pairs[group] = (cached[0], cached[1], cached[2], True)
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
        if fence is not None and while_waiting is not None:
            run_while_waiting(while_waiting)
        if fence is not None:
            fence[0].synchronize_device(fence[1])
            if while_waiting is not None:
                note_fenced(while_waiting)
        if batched:
            select_round(batched, prepared, round_number)
        if audit_enabled() and pair_labels:
            audit_log(AUDIT_LINE, round=round_number, pairs=pair_labels,
                      propose_ms=['%.1f' % value for value in pair_ms])
        return prepared


# QWEN_FAST_ROUND_B1 (C1), under QWEN_FAST_PACKED_AUDIT=1: one line per round, the host
# time of the pairs' readback (collect) and of the batched selection, and how many selector
# calls that took (one unless two pairs lend different codebooks).
SELECT_LINE = '[PACKED-SELECT] round={round} pairs={pairs} users={users} calls={calls} collect_ms={collect_ms} select_ms={select_ms}'


def _discard_round(prepared):
    """The exception path of PackedProposalCoordinator.prepare, for a failure after its
    fence: every prepared device's pending work is dropped, a shared pair trace once.

    Unlike that path, a device wearing a _PackedCaptureView also has the view's own
    single-user capture discarded (the view forwards discard_pending to it): a device whose
    pair did not pack this round is prepared through prepare_single, and its pending then
    lives in that capture, not in the pair trace. A no-op for a device the pair prepared
    (its single-user capture holds nothing pending, or was released)."""
    discarded_traces = set()
    for device in prepared:
        capture = device.proposal_capture
        if isinstance(capture, _PackedCaptureView):
            if id(capture._trace) not in discarded_traces:
                discarded_traces.add(id(capture._trace))
                capture._trace.discard_pending()
            capture.discard_pending()
        else:
            discard = getattr(capture, 'discard_pending', None)
            if callable(discard):
                discard()


def select_round(batched, prepared, round_number):
    """QWEN_FAST_ROUND_B1 (C1): after the round's one fence, read every prepared pair back
    (PreparedPackedDFlashProposal.collect) and select all of their users in ONE FP64
    selector call (dflash_packed_proposal.select_packed_batched), where each pair's first
    finish() in phase B used to select its own two users - and adopt the tokens into each
    trace, so phase B's finish() returns them unchanged. Pairs are batched only when they
    select against the very same codebook objects (the lent draft weights), checked by
    identity; a pair that does not share them gets a call of its own.

    A failure here fails the round as it would have failed in phase B, after dropping
    every prepared device's pending work - the device queue is already fenced.

    Under QWEN_FAST_ROUND_B1_AUDIT each pair is then read back and selected again through
    PreparedPackedDFlashProposal.audit_selection (select_device_outputs, the flag-off path)
    and must give the tokens it adopted (dflash_packed_proposal.ROUND_B1_AUDIT_FLAG)."""
    from dflash_packed_proposal import note_round_b1, round_b1_audit_enabled, select_packed_batched

    audit = round_b1_audit_enabled()
    started = time.perf_counter()
    try:
        collected = [(labels, trace, trace.collect()) for labels, trace in batched]
        collected_at = time.perf_counter()
        groups = []
        for labels, trace, result in collected:
            device = trace.device_a
            for group in groups:
                if group['predecessors'] is device.predecessors and group['successors'] is device.successors:
                    group['members'].append((trace, result))
                    break
            else:
                groups.append(dict(predecessors=device.predecessors, successors=device.successors,
                                   members=[(trace, result)]))
        for group in groups:
            parts = [part for _, result in group['members'] for part in result['parts']]
            seeds = [seed for _, result in group['members'] for seed in result['seeds']]
            counts = [count for _, result in group['members'] for count in result['counts']]
            tokens = select_packed_batched(parts, seeds, counts, group['predecessors'], group['successors'])
            offset = 0
            for trace, result in group['members']:
                users = len(result['parts'])
                trace.adopt(tokens[offset:offset + users])
                if audit:
                    _audit_selection(trace, tokens[offset:offset + users])
                offset += users
    except BaseException:
        _discard_round(prepared)
        raise
    note_round_b1('batched-select')
    if audit:
        from dflash_packed_proposal import note_round_b1_audit

        note_round_b1_audit()
    if audit_enabled():
        finished = time.perf_counter()
        audit_log(SELECT_LINE, round=round_number, pairs=[labels for labels, _, _ in collected],
                  users=sum(len(result['parts']) for _, _, result in collected), calls=len(groups),
                  collect_ms='%.2f' % ((collected_at - started) * 1000),
                  select_ms='%.2f' % ((finished - collected_at) * 1000))


def run_while_waiting(while_waiting):
    """Round-fence plan H1a: the window's callable, just before the round's fence. An Exception
    goes to its `drop` (the pre-stage's snapshot is dropped, the next verify takes the full
    path) and never fails the round; a failing `drop` is swallowed too."""
    try:
        while_waiting()
    except Exception as failure:
        drop = getattr(while_waiting, 'drop', None)
        if callable(drop):
            try:
                drop(failure)
            except Exception:
                pass


def note_fenced(while_waiting):
    """Round-fence plan H1a: right after the round's fence, the window callable's `fenced` (it
    arms the packed block's next replay under QWEN_FAST_ROUND_FENCES). A failure only leaves
    that replay to pay its own fence, so it never fails the round."""
    fenced = getattr(while_waiting, 'fenced', None)
    if callable(fenced):
        try:
            fenced()
        except Exception as failure:
            try:
                audit_log('[PACKED-PRESTAGE-WINDOW] fenced failed: {failure}', failure=repr(failure)[:160])
            except Exception:
                pass


def _audit_selection(trace, adopted):
    """QWEN_FAST_ROUND_B1_AUDIT (C1): the pair's own flag-off selection against the batched
    tokens it adopted."""
    from dflash_packed_proposal import round_b1_audit_count, round_b1_audit_mismatch

    reference = tuple(tuple(tokens) for tokens in trace.audit_selection())
    adopted = tuple(tuple(tokens) for tokens in adopted)
    if reference != adopted:
        round_b1_audit_mismatch('C1', 'batched=%s per-user=%s' % (adopted, reference))
    round_b1_audit_count('select', len(adopted))
