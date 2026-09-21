"""Pre-verifier capture of fixed-context five-layer proposal computation."""

from types import SimpleNamespace

from attention_batch import capture_operation
from dflash_proposal_inputs import proposal_contexts, proposal_inputs
from gdn_multitoken_conv import addresses, release_owned


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
