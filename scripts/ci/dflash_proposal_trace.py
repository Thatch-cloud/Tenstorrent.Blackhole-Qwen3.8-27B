"""Pre-verifier capture of fixed-context five-layer proposal computation."""

import os
from types import SimpleNamespace

from attention_batch import capture_operation
from dflash_proposal_inputs import proposal_contexts, proposal_inputs
from gdn_multitoken_conv import addresses, release_owned


# Variable-user packed rounds, M0 (the pair-mask audit and refresh). A packed pair's bucket
# uploads its mask once, at the lazy bucket build, and every replay reads it from then on;
# every other input the pair trace reads is re-uploaded (identifiers, rope.q, rope.live_k) or
# re-copied (cached_history) each round. v185 builds the pair buckets after every engine's
# 4-row verify traces were captured, and serving_buffer_pool documents what that order does
# to a buffer allocated into an earlier trace's freed holes: that trace's replay overwrites
# it. The 4-row traces first replay at the first sequential round, where the pair users'
# acceptance collapses (1.0-1.5 tokens per step against 2.76-3.0 for single-proposal users).
# Both flags are read at each use, never at import; with both unset nothing below runs.
#   QWEN_FAST_PAIR_MASK_REFRESH=1  copy the kept host mask into bucket.mask on every
#       _update, beside the other per-round copies (266 KB per pair), deferred path included.
#       PAIR_MASK_REFRESH_MARKER is logged once per process at the first refresh.
#   QWEN_FAST_PAIR_MASK_AUDIT=1    a diagnostic arm's read-back: before any copy of the
#       round, bucket.mask is read from both chips and compared with the host mask, bit for
#       bit; one PAIR_MASK_AUDIT_LINE per chip. The round is the coordinator's
#       ([PACKED-PROPOSE] round=, set on the trace as round_number), the pair its pool slots.
PAIR_MASK_REFRESH_FLAG = 'QWEN_FAST_PAIR_MASK_REFRESH'
PAIR_MASK_AUDIT_FLAG = 'QWEN_FAST_PAIR_MASK_AUDIT'
PAIR_MASK_REFRESH_MARKER = '[PINDIAG] pair mask refresh'
PAIR_MASK_AUDIT_LINE = '[PACKED-PROPOSE] mask round=%s pair=[%s,%s] intact=%d mismatched=%d chip=%d'
_PAIR_MASK_REFRESH_NOTED = []


def pair_mask_refresh_enabled():
    """QWEN_FAST_PAIR_MASK_REFRESH=1."""
    return os.environ.get(PAIR_MASK_REFRESH_FLAG) == '1'


def pair_mask_audit_enabled():
    """QWEN_FAST_PAIR_MASK_AUDIT=1."""
    return os.environ.get(PAIR_MASK_AUDIT_FLAG) == '1'


def _log_line(message):
    """One INFO line into the server log: loguru where it exists, stdout otherwise. Never raises."""
    try:
        try:
            from loguru import logger
        except ImportError:
            print(message, flush=True)
        else:
            logger.info('{}', message)
    except BaseException:
        pass


def _slot_of(device):
    return getattr(getattr(device, 'pool_slot', None), 'index', None)


def _live_bank_history(device_a, device_b):
    """Round-fence plan H1b, F4: the pair's cached_history as the pool's active banks under
    QWEN_FAST_FUSED_COMMIT_LIVE_BANKS (fused_commit.live_bank_history), else None. The flag is read
    first and fused_commit imported only when it is set."""
    if os.environ.get('QWEN_FAST_FUSED_COMMIT_LIVE_BANKS') != '1':
        return None
    from fused_commit import live_bank_history

    return live_bank_history(device_a, device_b)


class PreparedDFlashProposal:
    def __init__(self, device, *, max_new_tokens):
        import torch

        self.device, self.operations, self.mesh = device, device.operations, device.mesh
        self.buckets, self.owned, self.checks = {}, [], []
        self.cache_checks = []
        self.kv_history = getattr(device, 'kv_history', None)
        self.closed = False
        # QWEN_FAST_PIPELINED_PROPOSALS (serving_worker_hook.prepare_pipelined_drafts):
        # (seed, bucket, owned) once prepare_device() has enqueued this proposal's
        # device work and is waiting on finish() to read it back - None the rest of
        # the time, including on the unmodified single-phase propose() path below.
        self._pending = None
        operations = self.operations
        def upload(value, *, identifiers=False):
            tensor = operations.from_torch(value, device=self.mesh,
                dtype=operations.uint32 if identifiers else operations.bfloat16,
                layout=operations.ROW_MAJOR_LAYOUT if identifiers else operations.TILE_LAYOUT,
                memory_config=operations.DRAM_MEMORY_CONFIG, mesh_mapper=operations.ReplicateTensorToMesh(self.mesh))
            self.owned.append(tensor)
            return tensor
        try:
            for context in proposal_contexts(device.position, max_new_tokens):
                host = proposal_inputs(0, device.position, device.history_rows, device.block_rows, context)
                bucket = SimpleNamespace(context=context, identifiers=upload(host['identifiers'], identifiers=True),
                    history=upload(torch.zeros((1, 1, context + 32, 5120), dtype=torch.bfloat16)),
                    mask=upload(host['mask']), rope={name: tuple(upload(value) for value in host['rope'][name]) for name in ('q', 'k')},
                    trace=None, outputs=None, owned=[])
                bucket.cached_history = None if self.kv_history is None else [{name: upload(
                    torch.zeros((1, 4, context, 128), dtype=torch.bfloat16)) for name in ('k', 'v')}
                    for layer in self.kv_history.active]
                bucket.inputs = [bucket.identifiers, bucket.history, bucket.mask, *bucket.rope['q'], *bucket.rope['k']]
                if bucket.cached_history is not None:
                    bucket.inputs.extend(value for layer in bucket.cached_history for value in layer.values())
                bucket.addresses = [addresses(operations, value) for value in bucket.inputs]
                self.buckets[context] = bucket
            for bucket in self.buckets.values():
                self.update(bucket, 0)
                transient, retain = device.temporaries([*device.owned, *self.owned])
                try:
                    self.execute(bucket, transient, retain)
                    operations.synchronize_device(self.mesh)
                finally:
                    release_owned(operations, transient)
                bucket.owned, retain = device.temporaries([*device.owned, *self.owned])
                bucket.trace, bucket.outputs = capture_operation(operations, self.mesh,
                    lambda: self.execute(bucket, bucket.owned, retain))
                operations.execute_trace(self.mesh, bucket.trace, cq_id=0, blocking=True)
        except BaseException:
            self.close()
            raise

    def execute(self, bucket, owned, retain, *, audit_convolution=False, use_cache=True):
        return self.device.execute_proposal(bucket.identifiers, bucket.history, bucket.mask, bucket.rope,
            context=bucket.context, owned=owned, retain=retain, stage=lambda name, **values: None, audit=False,
            **(dict(audit_convolution=True) if audit_convolution else {}),
            **(dict(cached_history=bucket.cached_history) if use_cache and bucket.cached_history is not None else {}))

    def update(self, bucket, seed, *, defer_finish=False):
        """Copy this proposal's inputs onto `bucket` for `seed`. By default (`defer_finish`
        False, every existing caller) this synchronizes the device, checks the prepared
        addresses did not move and releases its own transients before returning - exactly
        as before. `defer_finish=True` (QWEN_FAST_PIPELINED_PROPOSALS, prepare_device()
        below) enqueues the same copies without waiting on any of that, and hands the
        transients back instead of releasing them, so several requests' updates can share
        one synchronize_device (serving_worker_hook.prepare_pipelined_drafts) instead of
        paying one each; the caller must finish() (or discard_pending()) what this
        returns before the device can safely reuse the storage it protects."""
        operations, device = self.operations, self.device
        if getattr(device, 'live_query_qk', False):
            device.validated_live_masks.discard(addresses(operations, bucket.mask))
        if getattr(device, 'native_proposal_attention', False):
            device.validated_native_proposal_masks.discard(addresses(operations, bucket.mask))
        if self.kv_history is not None and (self.kv_history.pending is not None
                or self.kv_history.position != device.position or self.kv_history.history_rows != device.history_rows):
            raise ValueError('Proposal replay requires a fully committed matching K/V frontier')
        host = proposal_inputs(seed, device.position, device.history_rows, device.block_rows, bucket.context)
        if getattr(device, 'live_query_qk', False):
            from draft_live_qk import validate_live_qk_mask

            validate_live_qk_mask(host['mask'])
        if getattr(device, 'native_proposal_attention', False):
            if device.block_rows == 16:
                from dflash_t16_native_scope import require_active
                from dflash_t16_native_attention import validate_mask

                require_active()
            else:
                from proposal_native_attention import validate_mask

            validate_mask(host['mask'])
        sources = [host['identifiers'], host['mask'], *host['rope']['q'], *host['rope']['k']]
        destinations = [bucket.identifiers, bucket.mask, *bucket.rope['q'], *bucket.rope['k']]
        for value, destination in zip(sources, destinations, strict=True):
            payload = operations.from_torch(value, dtype=destination.dtype, layout=destination.layout,
                mesh_mapper=operations.ReplicateTensorToMesh(self.mesh))
            operations.copy_host_to_device_tensor(payload, destination)
        # Pooled, the K/V banks read below are lent (kv_history.borrowed), not owned
        # (kv_history.owned is empty under the pool): run 35561480877 died on the
        # FIRST traced proposal with 'input_tensor.is_allocated()' reading active[name]
        # below, because a context=2048 bucket slices a bank's full row extent - the
        # same identity the bank already has - and unprotected here, retain() queued
        # that reslice as a temporary for this call's own release_owned() to free the
        # live pool bank out from under every later proposal. kv_history.temporaries()
        # already protects kv_history.borrowed the same way; this call must too.
        owned, retain = device.temporaries([device.history, device.spare_history, *self.owned,
            *(self.kv_history.owned if self.kv_history is not None else []),
            *(self.kv_history.borrowed if self.kv_history is not None else [])])
        def copy_history_and_cache():
            if self.kv_history is None or device.progress is not None:
                if getattr(device, 'history_stale', False):
                    # Set only under QWEN_FAST_ROUND_B1 (C7), which stopped writing the
                    # feature history this branch would copy (dflash_device.DFlashDevice.
                    # _prepare_publication_round_b1).
                    raise ValueError('The feature history is stale (QWEN_FAST_ROUND_B1 C7) and cannot be read')
                history = retain(operations.slice(device.history, (0, 0, 0, 0), (1, 1, bucket.context, 5120)))
                padded = retain(operations.pad(history, [(0, 0), (0, 0), (0, 32), (0, 0)], 0.0))
                operations.copy(padded, bucket.history)
            if self.kv_history is not None:
                for active, destination in zip(self.kv_history.active, bucket.cached_history, strict=True):
                    for name in ('k', 'v'):
                        value = retain(operations.slice(active[name], (0, 0, 0, 0), (1, 4, bucket.context, 128)))
                        operations.copy(value, destination[name])
        if defer_finish:
            try:
                copy_history_and_cache()
            except BaseException:
                # Nothing will call finish() for this attempt, so there is no later
                # point that releases these - unlike the non-deferred path's finally
                # below, this must release right here or leak them.
                release_owned(operations, owned)
                raise
            return owned
        try:
            copy_history_and_cache()
            operations.synchronize_device(self.mesh)
            if [addresses(operations, value) for value in bucket.inputs] != bucket.addresses:
                raise AssertionError('Prepared proposal input addresses moved')
            if getattr(device, 'live_query_qk', False):
                device.validated_live_masks.add(addresses(operations, bucket.mask))
            if getattr(device, 'native_proposal_attention', False):
                device.validated_native_proposal_masks.add(addresses(operations, bucket.mask))
        finally:
            release_owned(operations, owned)

    def prepare_device(self, seed):
        """Phase A of QWEN_FAST_PIPELINED_PROPOSALS (serving_worker_hook.
        prepare_pipelined_drafts): enqueue this proposal's copies and its trace
        replay without waiting on the device, deferring update()'s
        synchronize_device, its address-identity assertion and its transient
        release to finish(). Returns False - the caller falls back to this
        bridge's normal blocking drafts() - when there is nothing to prewarm:
        this capture is closed, or its device audits its proposal (progress is
        not None; the audit path reads back inline and cannot defer). A prepare
        left over from a round the harness chose not to consume (GreedySession.
        propose's `limit > 1` gate, a request within one token of the session's
        256-slot ceiling - real_remaining_budget's docstring) is discarded here
        rather than left to raise, so one skipped round does not refuse every
        later prewarm for that request."""
        if self.closed or self.device.progress is not None:
            return False
        if self._pending is not None:
            self.discard_pending()
        device, operations = self.device, self.operations
        bucket = next((value for context, value in self.buckets.items() if context >= device.history_rows), None)
        if bucket is None:
            raise ValueError('Committed history exceeds prepared request contexts')
        owned = self.update(bucket, seed, defer_finish=True)
        operations.execute_trace(self.mesh, bucket.trace, cq_id=0, blocking=False)
        self._pending = (seed, bucket, owned)
        return True

    def has_pending(self, seed):
        return self._pending is not None and self._pending[0] == seed

    def finish(self, count):
        """Phase B counterpart to prepare_device(): update()'s deferred tail (the
        address-identity assertion, validated-mask bookkeeping and transient
        release) followed by propose()'s own tail (the trace's host readback and
        selection) - run once the caller's single shared synchronize_device has
        already fenced every prepared request's device work, not just this one's,
        so this issues no synchronize_device of its own."""
        if self._pending is None:
            raise ValueError('No prepared proposal is pending')
        seed, bucket, owned = self._pending
        self._pending = None
        operations, device = self.operations, self.device
        try:
            if [addresses(operations, value) for value in bucket.inputs] != bucket.addresses:
                raise AssertionError('Prepared proposal input addresses moved')
            if getattr(device, 'live_query_qk', False):
                device.validated_live_masks.add(addresses(operations, bucket.mask))
            if getattr(device, 'native_proposal_attention', False):
                device.validated_native_proposal_masks.add(addresses(operations, bucket.mask))
        finally:
            release_owned(operations, owned)
        return device.select_proposal(bucket.outputs, seed, count)

    def discard_pending(self):
        """Release a prepare_device() that will never be finish()ed: a phase-A
        failure fencing every OTHER bridge's already-enqueued work before it
        re-raises (serving_worker_hook.prepare_pipelined_drafts), a stale prewarm
        prepare_device() itself finds still pending, or close(). Callers other
        than prepare_device() must synchronize_device first, exactly as finish()
        requires - this only releases host-side bookkeeping and transients, it
        does not fence the device."""
        if self._pending is None:
            return
        _, _, owned = self._pending
        self._pending = None
        release_owned(self.operations, owned)

    def propose(self, seed, count):
        import torch

        if self.closed:
            raise ValueError('Closed proposal capture cannot replay')
        device, operations = self.device, self.operations
        bucket = next((value for context, value in self.buckets.items() if context >= device.history_rows), None)
        if bucket is None:
            raise ValueError('Committed history exceeds prepared request contexts')
        self.update(bucket, seed)
        expected = None
        if device.progress is not None:
            owned, retain = device.temporaries([*device.owned, *self.owned])
            try:
                outputs = self.execute(bucket, owned, retain, audit_convolution=getattr(device, 'fused_convolution', False))
                operations.synchronize_device(self.mesh)
                expected = device.proposal_snapshot(outputs)
                if self.kv_history is not None:
                    control = self.execute(bucket, owned, retain, use_cache=False)
                    operations.synchronize_device(self.mesh)
                    original = device.proposal_snapshot(control)
                    if len(original) != len(expected) or any(not torch.equal(left, right)
                            for left, right in zip(original, expected, strict=True)):
                        raise AssertionError('Cached learned proposal differs from full historical projection')
                    self.cache_checks.append(dict(position=device.position, context=bucket.context,
                        tensors=len(expected), exact=True))
            finally:
                release_owned(operations, owned)
        operations.execute_trace(self.mesh, bucket.trace, cq_id=0, blocking=True)
        if expected is not None:
            actual = device.proposal_snapshot(bucket.outputs)
            if len(actual) != len(expected) or any(not torch.equal(left, right) for left, right in zip(actual, expected, strict=True)):
                raise AssertionError('Captured learned proposal differs from the same fixed-context eager execution')
            self.checks.append(dict(position=device.position, context=bucket.context, tensors=len(actual), exact=True))
        return device.select_proposal(bucket.outputs, seed, count)

    def close(self):
        if self.closed:
            return
        self.operations.synchronize_device(self.mesh)
        # After the sync above, exactly as finish()/discard_pending() require.
        self.discard_pending()
        for bucket in self.buckets.values():
            if bucket.trace is not None:
                self.operations.release_trace(self.mesh, bucket.trace)
                bucket.trace = None
        for bucket in self.buckets.values():
            if getattr(self.device, 'live_query_qk', False):
                self.device.validated_live_masks.discard(addresses(self.operations, bucket.mask))
            if getattr(self.device, 'native_proposal_attention', False):
                self.device.validated_native_proposal_masks.discard(addresses(self.operations, bucket.mask))
            release_owned(self.operations, bucket.owned)
            bucket.owned.clear()
        release_owned(self.operations, self.owned)
        self.owned.clear()
        self.closed = True


class PreparedPackedDFlashProposal:
    """QWEN_FAST_PACKED_PROPOSAL: one traced two-user packed proposal for a single
    FOUR_AS_TWO_PAIRS slot pair (dflash_packed_proposal.FOUR_AS_TWO_PAIRS), mirroring
    PreparedDFlashProposal's single-user capture above but built over
    dflash_packed_proposal.propose_packed's own cached-path geometry
    (packed_identifiers, dflash_batched_mask.batched_attention_mask/
    packed_rope_tables/live_key_rope) instead of duplicating it - the CACHED path
    only, so no history tensor is ever built here (propose_packed's own module
    docstring: 'that would be 42 MB of DRAM nobody touches').

    Keyed at capture by the pair's OWN geometry: block_rows (both devices' qualified
    T16 width) and each user's history_rows - dflash_batched_mask packs by KEY
    SEGMENT, sized from the exact context, so a (1024, 2048) pair needs a different
    capture from a (2048, 2048) one. Built lazily, one bucket per distinct
    (history_rows_a, history_rows_b) this SPECIFIC slot pair actually reaches, kept
    for the life of this object.

    dflash_packed_proposal_coordinator.PackedProposalCoordinator is the only caller,
    and it gates packing to history_rows == 2048 on BOTH devices before ever calling
    prepare_device() here - draft_kv_history.DraftKVHistory requires history_rows ==
    min(position, 2048), so that is a PERMANENT, monotonic state every request
    eventually reaches and never leaves. Below 2048, history_rows changes almost every
    round (by however many tokens that round accepted), so an exact-context bucket key
    would recapture on nearly every round of the ramp - capture is a blocking,
    expensive device operation, so this class does not by itself avoid that cost; the
    coordinator's gate is what keeps it out of the ramp entirely. This class stays
    correct for any (history_rows_a, history_rows_b) pair regardless - the gate is a
    policy choice above it, not a correctness requirement of the geometry here.

    device_a and device_b must share weights (dflash_device.SharedDraftWeights.lend,
    the only way four concurrent requests share the five DFlash2 layers today):
    device_a is used as execute_proposal's weight/layer owner (propose_packed's own
    `device` argument), device_b contributes only its own K/V cache and position.

    Uncertified: host construction and the mocked device-call sequence only (matching
    dflash_packed_proposal.ProposePackedTests' own mocked-device pattern). Nothing
    here has run on a device or against a real ttnn runtime."""

    def __init__(self, device_a, device_b):
        if device_a.operations is not device_b.operations or device_a.mesh is not device_b.mesh:
            raise ValueError('Both paired devices must share one mesh and runtime')
        if (device_a.block_rows != device_b.block_rows
                or not getattr(device_a, 'native_proposal_attention', False)
                or not getattr(device_b, 'native_proposal_attention', False)):
            raise ValueError('Both paired devices must share the qualified native-proposal block width')
        if device_a.kv_history is None or device_b.kv_history is None:
            raise ValueError('Both paired devices require a committed K/V cache')
        self.device_a, self.device_b = device_a, device_b
        self.operations, self.mesh = device_a.operations, device_a.mesh
        self.block_rows = device_a.block_rows
        self.buckets, self.owned = {}, []
        self.closed = False
        # (seed_a, seed_b, bucket, owned) once prepare_device() has enqueued this
        # pair's device work and is waiting on finish() (from EACH side) to read it
        # back - None the rest of the time. Mirrors PreparedDFlashProposal._pending,
        # but one shared pending covers both users: the trace replay is one device
        # call, not two.
        self._pending = None
        # QWEN_FAST_PAIR_MASK_AUDIT: the coordinator's round, which it sets here before
        # prepare_device() only while the audit is on, for PAIR_MASK_AUDIT_LINE. Read by
        # nothing else.
        self.round_number = None

    def _upload(self, value, *, identifiers=False):
        operations = self.operations
        tensor = operations.from_torch(value, device=self.mesh,
            dtype=operations.uint32 if identifiers else operations.bfloat16,
            layout=operations.ROW_MAJOR_LAYOUT if identifiers else operations.TILE_LAYOUT,
            memory_config=operations.DRAM_MEMORY_CONFIG, mesh_mapper=operations.ReplicateTensorToMesh(self.mesh))
        self.owned.append(tensor)
        return tensor

    def _bucket(self, context_a, context_b):
        key = (context_a, context_b)
        bucket = self.buckets.get(key)
        if bucket is not None:
            return bucket
        import torch
        from dflash_batched_mask import batched_attention_mask, packed_rope_tables, live_key_rope
        from dflash_packed_proposal import packed_identifiers
        from dflash_t16_native_attention import validate_mask

        operations, device = self.operations, self.device_a
        # Every self._upload() below appends into self.owned (the trace's own
        # persistent placeholder list, not the transient/capture-scoped owned lists
        # further down) - snapshotted here so a failure anywhere in this build can
        # release exactly what THIS attempt added and nothing from any other bucket
        # already built on this trace. Run 35585107688: without this, a failed
        # build left its identifiers/mask/rope/cached_history placeholders (the
        # ~42 MB per attempt the report accounts for) allocated forever - three
        # consecutive failed attempts for one pair leaked three full sets on top of
        # the DRAM an already-96%-full device had none of left, which is what
        # starved the 20 MB prepare_publication transient that actually killed it.
        placeholder_mark = len(self.owned)
        try:
            # Placeholder users: seed 0, and a position equal to the bucket's own
            # context - only the SHAPE of the geometry matters at capture time
            # (proposal_inputs' own convention, PreparedDFlashProposal above);
            # update() re-uploads every round's real seeds and real positions onto
            # these same fixed tensors before every replay.
            placeholder_users = [dict(position=context_a, history_rows=context_a),
                                 dict(position=context_b, history_rows=context_b)]
            host_mask = batched_attention_mask([context_a, context_b], self.block_rows)
            validate_mask(host_mask, contexts=[context_a, context_b])
            tables = packed_rope_tables(placeholder_users, self.block_rows)
            live = live_key_rope(placeholder_users, self.block_rows)
            # host_mask is kept (a host tensor, 266 KB at the packable geometry): the
            # mask depends on the bucket's contexts alone, so it is every round's mask -
            # what QWEN_FAST_PAIR_MASK_REFRESH copies back and QWEN_FAST_PAIR_MASK_AUDIT
            # compares with. Nothing reads it with both flags off.
            # Round-fence plan H1b, F4 (QWEN_FAST_FUSED_COMMIT_LIVE_BANKS, default off): the pair
            # reads the two pool slots' ACTIVE banks - which the in-place fused commit slides and
            # never swaps - instead of per-round copies into ~40 MB of its own placeholders. None
            # without the flag: the placeholders below, exactly as before.
            live_banks = _live_bank_history(self.device_a, self.device_b) if context_a == context_b == 2048 else None
            bucket = SimpleNamespace(context=(context_a, context_b), host_mask=host_mask,
                identifiers=self._upload(packed_identifiers([0, 0], self.block_rows), identifiers=True),
                mask=self._upload(host_mask),
                rope=dict(q=tuple(self._upload(value) for value in tables['q']),
                          k=tuple(self._upload(value) for value in tables['k']),
                          live_k=tuple(self._upload(value) for value in live)),
                cached_history=live_banks if live_banks is not None else [
                    [{name: self._upload(torch.zeros((1, 4, context, 128), dtype=torch.bfloat16))
                      for name in ('k', 'v')} for _ in self.device_a.kv_history.active]
                    for context in (context_a, context_b)],
                trace=None, outputs=None, owned=[], tokens=None, consumed=set())
            if live_banks is not None:
                bucket.live_banks = True
            bucket.inputs = [bucket.identifiers, bucket.mask, *bucket.rope['q'], *bucket.rope['k'], *bucket.rope['live_k'],
                *(value for cache in bucket.cached_history for layer in cache for value in layer.values())]
            bucket.addresses = [addresses(operations, value) for value in bucket.inputs]
            device.validated_native_proposal_masks.add(addresses(operations, bucket.mask))
            self.buckets[key] = bucket
            if live_banks is not None:
                from fused_commit import note_live_banks

                note_live_banks(self.pair_label(), bucket.context)
            self._update(bucket, 0, 0)
            # F4: the lent banks are protected like the placeholders they replace, so no temporary
            # of the pass can ever queue one for release. Without the flag this list is today's.
            lent = [] if live_banks is None else [value for cache in live_banks for layer in cache
                                                  for value in layer.values()]
            transient, retain = device.temporaries([device.history, device.spare_history,
                self.device_b.history, self.device_b.spare_history, *self.owned, *lent])
            try:
                self._execute(bucket, transient, retain)
                operations.synchronize_device(self.mesh)
            finally:
                release_owned(operations, transient)
            bucket.owned, retain = device.temporaries([device.history, device.spare_history,
                self.device_b.history, self.device_b.spare_history, *self.owned, *lent])
            bucket.trace, bucket.outputs = capture_operation(operations, self.mesh,
                lambda: self._execute(bucket, bucket.owned, retain))
            operations.execute_trace(self.mesh, bucket.trace, cq_id=0, blocking=True)
        except BaseException:
            self.buckets.pop(key, None)
            built = locals().get('bucket')
            if built is not None:
                # Reads built.mask's address (validated_native_proposal_masks.discard)
                # and releases built.owned (the capture-scoped intermediates, disjoint
                # from the placeholders below) BEFORE any placeholder is deallocated -
                # built.mask is itself one of the placeholders released next.
                device.validated_native_proposal_masks.discard(addresses(operations, built.mask))
                release_owned(operations, built.owned)
            leaked = self.owned[placeholder_mark:]
            del self.owned[placeholder_mark:]
            release_owned(operations, leaked)
            raise
        return bucket

    def _execute(self, bucket, owned, retain):
        rope = dict(q=bucket.rope['q'], k=bucket.rope['k'], live_k=bucket.rope['live_k'])
        users = [dict(position=self.device_a.position, history_rows=bucket.context[0]),
                 dict(position=self.device_b.position, history_rows=bucket.context[1])]
        return self.device_a.execute_proposal(bucket.identifiers, None, bucket.mask, rope, context=None,
            pack=users, cached_history=bucket.cached_history, owned=owned, retain=retain,
            stage=lambda name, **values: None, audit=False)

    def _update(self, bucket, seed_a, seed_b, *, defer_finish=False):
        from dflash_batched_mask import packed_rope_tables, live_key_rope
        from dflash_packed_proposal import packed_identifiers

        device_a, device_b, operations = self.device_a, self.device_b, self.operations
        context_a, context_b = bucket.context
        if device_a.history_rows != context_a or device_b.history_rows != context_b:
            raise ValueError("Packed proposal replay requires both users at this bucket's committed context")
        if (device_a.kv_history.pending is not None or device_b.kv_history.pending is not None
                or device_a.position - device_a.history_rows < 0 or device_b.position - device_b.history_rows < 0):
            raise ValueError('Packed proposal replay requires a fully committed matching K/V frontier')
        users = [dict(position=device_a.position, history_rows=context_a),
                 dict(position=device_b.position, history_rows=context_b)]
        if pair_mask_audit_enabled():
            # Before any copy of this round: what every replay since the last refresh (or
            # since the build) left in the mask.
            self.audit_mask(bucket)
        identifiers_host = packed_identifiers([seed_a, seed_b], self.block_rows)
        if os.environ.get('QWEN_FAST_ROUND_B1') == '1':
            # QWEN_FAST_ROUND_B1 (C8). The packed cached trace never reads rope['k']:
            # draft_attention_branch takes rope['live_k'] whenever the block is packed
            # and only shape-checks rope['k'], so the two (1, 1, 4160, 128) uploads into
            # it were dead. The key tables are still built in full, once, and the live
            # rows sliced out of them (live_key_rope_from) - rope_tables at the live
            # start alone can round differently in bf16.
            from dflash_batched_mask import live_key_rope_from
            from dflash_packed_proposal import note_round_b1, round_b1_audit_enabled

            note_round_b1('pair-update')
            tables = packed_rope_tables(users, self.block_rows)
            live = live_key_rope_from(tables['k'], users, self.block_rows)
            if round_b1_audit_enabled():
                _audit_live_key_rope(live, users, self.block_rows)
            sources = [identifiers_host, *tables['q'], *live]
            destinations = [bucket.identifiers, *bucket.rope['q'], *bucket.rope['live_k']]
        else:
            tables = packed_rope_tables(users, self.block_rows)
            live = live_key_rope(users, self.block_rows)
            sources = [identifiers_host, *tables['q'], *tables['k'], *live]
            destinations = [bucket.identifiers, *bucket.rope['q'], *bucket.rope['k'], *bucket.rope['live_k']]
        if pair_mask_refresh_enabled():
            # QWEN_FAST_PAIR_MASK_REFRESH (M0): the build's own mask again, every round,
            # through the same copy as every other per-round input - on the deferred path
            # too - so whatever overwrote it since is healed before this round's replay.
            # The address-identity check below covers it unchanged (bucket.mask is one of
            # bucket.inputs).
            sources.append(bucket.host_mask)
            destinations.append(bucket.mask)
            self._note_refresh(bucket)
        for value, destination in zip(sources, destinations, strict=True):
            payload = operations.from_torch(value, dtype=destination.dtype, layout=destination.layout,
                mesh_mapper=operations.ReplicateTensorToMesh(self.mesh))
            operations.copy_host_to_device_tensor(payload, destination)
        owned, retain = device_a.temporaries([device_a.history, device_a.spare_history,
            device_b.history, device_b.spare_history, *self.owned,
            *device_a.kv_history.owned, *device_a.kv_history.borrowed,
            *device_b.kv_history.owned, *device_b.kv_history.borrowed])

        live_banks = getattr(bucket, 'live_banks', False)

        def copy_cache():
            normalised = 0
            for index, device in enumerate((device_a, device_b)):
                context = bucket.context[index]
                for active, destination in zip(device.kv_history.active, bucket.cached_history[index], strict=True):
                    for name in ('k', 'v'):
                        if live_banks and active[name] is destination[name]:
                            # F4: the pair reads this live bank itself - nothing to copy.
                            continue
                        # F4 with the device's live bank on the pool's spare side (the ramp left an
                        # odd swap count, before its first in-place commit): the live rows go into
                        # the pool's active bank - the device's free spare - which the pair reads.
                        value = retain(operations.slice(active[name], (0, 0, 0, 0), (1, 4, context, 128)))
                        operations.copy(value, destination[name])
                        normalised += 1
            if live_banks and normalised:
                from fused_commit import LIVE_BANKS_NORMALISED, log_line

                log_line('%s pair=%s normalised=%d' % (LIVE_BANKS_NORMALISED, self.pair_label(), normalised))
        if defer_finish:
            try:
                copy_cache()
            except BaseException:
                # Nothing will call finish() for this attempt, so this must release
                # right here or leak them - PreparedDFlashProposal.update()'s own
                # defer_finish path, above.
                release_owned(operations, owned)
                raise
            return owned
        try:
            copy_cache()
            operations.synchronize_device(self.mesh)
            if [addresses(operations, value) for value in bucket.inputs] != bucket.addresses:
                raise AssertionError('Prepared packed proposal input addresses moved')
        finally:
            release_owned(operations, owned)

    def pair_label(self):
        """This pair's pool slots, as the coordinator labels it ([slot_a, slot_b])."""
        return [_slot_of(self.device_a), _slot_of(self.device_b)]

    def _note_refresh(self, bucket):
        """PAIR_MASK_REFRESH_MARKER, once per process, at the first refresh."""
        if _PAIR_MASK_REFRESH_NOTED:
            return False
        _PAIR_MASK_REFRESH_NOTED.append(True)
        slot_a, slot_b = self.pair_label()
        _log_line('%s engaged pair=[%s,%s] context=%s,%s bytes=%d' % (
            PAIR_MASK_REFRESH_MARKER, slot_a, slot_b, bucket.context[0], bucket.context[1],
            bucket.host_mask.numel() * bucket.host_mask.element_size()))
        return True

    def audit_mask(self, bucket):
        """QWEN_FAST_PAIR_MASK_AUDIT: bucket.mask read back from each chip and compared with
        the kept host mask, bit for bit (int16 views of both). Logs one PAIR_MASK_AUDIT_LINE
        per chip and returns [(chip, intact, mismatched)]. A chip whose read-back has another
        shape counts every element as mismatched. Host reads only: it changes nothing."""
        import torch

        operations = self.operations
        expected = bucket.host_mask.contiguous().view(torch.int16)
        slot_a, slot_b = self.pair_label()
        results = []
        for chip, part in enumerate(operations.get_device_tensors(bucket.mask)):
            actual = operations.to_torch(part).contiguous().view(torch.int16)
            if tuple(actual.shape) != tuple(expected.shape):
                mismatched = max(int(actual.numel()), int(expected.numel()))
            else:
                mismatched = int((actual != expected).sum())
            intact = int(mismatched == 0)
            results.append((chip, intact, mismatched))
            _log_line(PAIR_MASK_AUDIT_LINE % (self.round_number, slot_a, slot_b, intact, mismatched, chip))
        return results

    def prepare_device(self, seed_a, seed_b):
        """Enqueue this pair's copies and its trace replay without waiting on the
        device, deferring update()'s synchronize_device and address-identity check to
        finish() - PreparedDFlashProposal.prepare_device()'s own contract, for both
        users at once. Returns False when there is nothing to prewarm: closed, either
        device mid-publication or already closed, or either device under audit
        (progress is not None, which reads intermediates back inline and cannot
        defer) - the caller then falls back to each device's own prepare_device()."""
        if self.closed:
            return False
        if self._pending is not None:
            self.discard_pending()
        device_a, device_b = self.device_a, self.device_b
        if (device_a.closed or device_b.closed or device_a.pending is not None or device_b.pending is not None
                or device_a.progress is not None or device_b.progress is not None):
            return False
        bucket = self._bucket(device_a.history_rows, device_b.history_rows)
        owned = self._update(bucket, seed_a, seed_b, defer_finish=True)
        self.operations.execute_trace(self.mesh, bucket.trace, cq_id=0, blocking=False)
        self._pending = (seed_a, seed_b, bucket, owned)
        return True

    def has_pending(self, which, seed):
        if self._pending is None:
            return False
        seed_a, seed_b, bucket, owned = self._pending
        if which in bucket.consumed:
            # This side's own half already came back through finish() this round;
            # the OTHER side's may still be outstanding, but this one is not pending
            # again until the coordinator's next prepare_device().
            return False
        return seed == (seed_a if which == 'a' else seed_b)

    def finish(self, which, count):
        """Counterpart to prepare_device(): the deferred address-identity check
        followed by the trace's host readback and BOTH users' selection, run once
        (on whichever side's finish() is called first) - the caller's own shared
        synchronize_device (PackedProposalCoordinator.prepare) has already fenced
        this pair's device work, so this issues no synchronize_device of its own.
        The second side's finish() just reads its own half back out."""
        if self._pending is None:
            raise ValueError('No prepared packed proposal is pending')
        seed_a, seed_b, bucket, owned = self._pending
        if bucket.tokens is None:
            operations = self.operations
            if [addresses(operations, value) for value in bucket.inputs] != bucket.addresses:
                release_owned(operations, owned)
                raise AssertionError('Prepared packed proposal input addresses moved')
            release_owned(operations, owned)
            from dflash_packed_proposal import select_device_outputs

            bucket.tokens = select_device_outputs(self.device_a, bucket.outputs, (seed_a, seed_b),
                (self.block_rows - 1, self.block_rows - 1), 2, self.block_rows)
        tokens = bucket.tokens[0 if which == 'a' else 1]
        bucket.consumed.add(which)
        if len(bucket.consumed) == 2:
            self._pending = None
            bucket.tokens, bucket.consumed = None, set()
        return tokens[:count]

    def collect(self):
        """QWEN_FAST_ROUND_B1 (C1): the first finish()'s own tail up to, not including,
        the selection - the deferred address-identity check, the transient release and
        the head readback - so the coordinator can select every pair of the round in one
        call and hand the tokens back through adopt(). Run after the caller's shared
        fence, exactly as finish() is. Returns this pair's per-user selector parts with
        the seeds and counts finish() would have selected them with.

        The pending transients are released here, as finish() releases them, and the
        pending entry keeps an empty list in their place, so a later discard_pending()
        cannot release them twice."""
        if self._pending is None:
            raise ValueError('No prepared packed proposal is pending')
        seed_a, seed_b, bucket, owned = self._pending
        if bucket.tokens is not None or bucket.consumed:
            raise ValueError('This packed proposal was already selected')
        operations = self.operations
        self._pending = (seed_a, seed_b, bucket, [])
        moved = [addresses(operations, value) for value in bucket.inputs] != bucket.addresses
        release_owned(operations, owned)
        if moved:
            raise AssertionError('Prepared packed proposal input addresses moved')
        from dflash_packed_proposal import read_device_outputs

        parts = read_device_outputs(self.device_a, bucket.outputs, 2, self.block_rows)
        return dict(parts=parts, seeds=(seed_a, seed_b), counts=(self.block_rows - 1, self.block_rows - 1))

    def audit_selection(self):
        """QWEN_FAST_ROUND_B1_AUDIT (C1): the tokens finish() would have selected itself
        for this pending pair - its outputs read back again and selected through
        select_device_outputs, the flag-off path. Host work only; run after collect(),
        before either side's finish(), while the outputs still hold this round's replay."""
        if self._pending is None:
            raise ValueError('No prepared packed proposal is pending')
        seed_a, seed_b, bucket, _ = self._pending
        from dflash_packed_proposal import select_device_outputs

        return select_device_outputs(self.device_a, bucket.outputs, (seed_a, seed_b),
            (self.block_rows - 1, self.block_rows - 1), 2, self.block_rows)

    def adopt(self, tokens):
        """QWEN_FAST_ROUND_B1 (C1): take both users' tokens from the round's batched
        selection. finish() then returns them exactly as if it had selected them itself -
        it already skips the selection whenever bucket.tokens is set."""
        if self._pending is None:
            raise ValueError('No prepared packed proposal is pending')
        _, _, bucket, _ = self._pending
        tokens = tuple(tokens)
        if bucket.tokens is not None or bucket.consumed or len(tokens) != 2:
            raise ValueError('Both users of one pending packed proposal must adopt one selection')
        bucket.tokens = tokens

    def discard_pending(self):
        """Release a prepare_device() that will never be finish()ed by either side:
        a phase-A failure (PackedProposalCoordinator.prepare), a stale prewarm this
        object itself finds still pending, or close(). Callers other than
        prepare_device() must synchronize_device first, exactly as finish() requires
        - this only releases host-side bookkeeping and transients, it does not fence
        the device."""
        if self._pending is None:
            return
        _, _, bucket, owned = self._pending
        self._pending = None
        bucket.tokens, bucket.consumed = None, set()
        release_owned(self.operations, owned)

    def close(self):
        if self.closed:
            return
        self.operations.synchronize_device(self.mesh)
        self.discard_pending()
        for bucket in self.buckets.values():
            if bucket.trace is not None:
                self.operations.release_trace(self.mesh, bucket.trace)
                bucket.trace = None
            self.device_a.validated_native_proposal_masks.discard(addresses(self.operations, bucket.mask))
            release_owned(self.operations, bucket.owned)
            bucket.owned.clear()
        release_owned(self.operations, self.owned)
        self.owned.clear()
        self.buckets.clear()
        self.closed = True


def _audit_live_key_rope(live, users, block_rows):
    """QWEN_FAST_ROUND_B1_AUDIT (C8): the live key rows sliced from the pair's one table
    build against live_key_rope's own build, bit for bit."""
    from dflash_batched_mask import live_key_rope
    from dflash_packed_proposal import round_b1_audit_count, round_b1_audit_mismatch, same_bits

    reference = live_key_rope(users, block_rows)
    if len(live) != len(reference) or not all(same_bits(mine, theirs) for mine, theirs in zip(live, reference)):
        round_b1_audit_mismatch('C8', 'live key RoPE differs for users %s' % (users,))
    round_b1_audit_count('rope')
