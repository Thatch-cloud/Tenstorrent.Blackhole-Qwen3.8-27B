"""Single-request captured verifier buckets; prepare only after that request's prefill."""

import os
import time
from pathlib import Path
from contextlib import ExitStack

from attention_batch import capture_operation
from force_argmax import sample_rows
from gdn_commit_dma import prepare
from gdn_multitoken_conv import addresses, release_owned
from model_batch import ModelBatch
from verifier_inputs import stage_inputs


# Which engine's state native GDN slot 0 holds right now. Runs 35477522469 and
# 35479238722 alternated two engines through that one slot with nothing restoring
# it between them, so each user's second block ran on the other user's recurrence.
_resident = None


def note_prefill():
    """A prefill overwrote slot 0: no engine is resident until one restores or publishes."""
    global _resident
    _resident = None


def note_packed_step():
    """A packed block rewrote slot 0 - its segment restores inside the verify trace and
    its per-user commit DMAs - so no engine is resident: a later sequential step of any
    user restores its carry first (packed_verifier.py; the packed step calls this too)."""
    global _resident
    _resident = None


class PackedFeatureView:
    """The synthetic bucket's feature capture for an adopted packed segment: the block's
    block-row taps at this user's row offset (packed_verifier.PackedVerifierEngine.features)."""

    def __init__(self, block, segment):
        self.block, self.segment = block, segment

    def outputs(self):
        return self.block.features(self.segment)

    def close(self):
        pass


def carry_log(message, **values):
    """One line per carry copy, so a stalled run shows whether the copies finished."""
    try:
        from loguru import logger
    except ImportError:
        print(message.format(**values), flush=True)
        return
    logger.info(message, **values)


# Every width a verify fixture can be captured at. 64 is the M3 block - four T16 users
# sharing one weight pass (docs/packed-device-step-plan-2026-09-20.md section 5) - and is
# reachable only through an explicit cap: the default cap stays 32, so every per-request
# engine captures exactly the widths it did before.
VERIFY_WIDTHS = (1, 2, 4, 8, 16, 32, 64)
BLOCK_WIDTHS = (16, 32, 64)


def capture_widths(position, capacity, verifier_rows, remaining, max_verify_rows=32):
    if type(max_verify_rows) is not int or max_verify_rows not in VERIFY_WIDTHS:
        raise ValueError('Explicit supported verification width cap required')
    if any(type(value) is not int for value in (position, capacity, verifier_rows, remaining)):
        raise ValueError('Integer request geometry required')
    if position < 0 or remaining < 1 or position + remaining > capacity or verifier_rows not in BLOCK_WIDTHS:
        raise ValueError('The complete decode budget must fit the request page capacity')
    return tuple(rows for rows in VERIFY_WIDTHS if rows <= min(verifier_rows, remaining, max_verify_rows))


def capture_bucket_rows(verifier_rows, remaining, max_verify_rows=32):
    """Every width a request's captures can ask for, with multiplicity, over every position.

    The serving pool allocates one BucketSlot per entry before any request exists, so it
    cannot know the position. Without replay there is one capture per width. With replay
    (attention_request_plan.capture_plan) widths below eight are captured once and each
    wider one once per 256-position mask family whose share of the budget holds a whole
    block: a budget of `remaining` tokens starting anywhere touches at most
    ceil((remaining - 1) / 256) + 1 families, and its disjoint shares of them hold at
    most remaining // rows blocks. test_verifier_carry checks the bound against the
    plan itself, at every offset within a family.
    """
    if type(remaining) is not int or remaining < 1:
        raise ValueError('Positive decode budget required')
    widths = capture_widths(0, remaining, verifier_rows, remaining, max_verify_rows)
    families = (remaining + 254) // 256 + 1
    return (tuple(rows for rows in widths if rows < 8)
            + tuple(rows for rows in widths if rows >= 8 for family in range(min(families, remaining // rows))))


def validate_snapshots(snapshots, helpers, name):
    """A pooled GDN snapshot set: one list per helper, shaped like the helper's live state."""
    snapshots = [list(snapshot) for snapshot in snapshots]
    if len(snapshots) != len(helpers) or any(len(snapshot) != len(helper.live)
                                             for snapshot, helper in zip(snapshots, helpers, strict=True)):
        raise ValueError('Pooled %s snapshots must hold one slot-zero state per GDN helper' % name)
    return snapshots


def validate_replay_options(attention_replay, attention_mask_once, replay_group_rows):
    if type(attention_mask_once) is not bool or (attention_mask_once and not attention_replay):
        raise ValueError('Shared attention masks require explicit replay attention')
    if type(replay_group_rows) is not int or replay_group_rows not in (4, 8):
        raise ValueError('Replay group width must be integer four or eight')
    if replay_group_rows == 8 and (not attention_replay or os.environ.get('QWEN_SDPA_TREE_SCRATCH_ROUNDS') != '1'):
        raise ValueError('Eight-row replay requires explicit replay attention and native compact scratch')


class VerifierEngine:
    def __init__(self, model, session, pages, helpers, *, sampler=None, norm_batch=False, attention_replay=False,
                 attention_mask_once=False, replay_group_rows=4, max_verify_rows=32, retain_mtp_hidden=False,
                 native_sampling_rows=False, short_context=False, attention_audit=False, retain_feature_taps=(),
                 commit_only_gdn=False, before_capture=None, target_attention_t16=False, storage=None):
        import ttnn

        if before_capture is not None and not callable(before_capture):
            raise ValueError('Callable pre-capture preparation required')
        # Pooled verifier storage (serving_buffer_pool.VerifierSlot): allocated at attach,
        # before any request trace, and lent for this request's life. Everything this
        # engine keeps across steps - initial and carried GDN state, per-width checkpoints,
        # feature taps, MTP hidden and the fixtures' inputs - comes from it, and close()
        # frees none of it. Allocated here instead, they would land in the holes an
        # earlier request's verify trace baked and its replays would overwrite them.
        if storage is not None and not callable(getattr(storage, 'take', None)):
            raise ValueError('Pooled verifier storage must lend width buckets')
        self.storage, self.borrowed = storage, []

        if type(commit_only_gdn) is not bool:
            raise ValueError('Explicit commit-only GDN policy required')
        self.commit_only_gdn = commit_only_gdn

        if type(native_sampling_rows) is not bool or (native_sampling_rows and sampler is None):
            raise ValueError('Native-row sampling requires an explicit device sampler')
        self.native_sampling_rows = native_sampling_rows
        if type(short_context) is not bool or (short_context and
                (max_verify_rows != 8 or not norm_batch or not native_sampling_rows or replay_group_rows != 4)):
            raise ValueError('Short-context verifier requires native-row sampling, batched norm and a T8 cap')
        self.short_context = short_context
        if type(attention_audit) is not bool or (attention_audit and not (short_context and attention_replay)):
            raise ValueError('Attention diagnostics require explicit short-context replay')
        self.attention_audit = attention_audit
        if type(retain_mtp_hidden) is not bool:
            raise ValueError('Explicit boolean MTP hidden retention required')
        self.retain_mtp_hidden = retain_mtp_hidden
        self.retain_feature_taps = tuple(retain_feature_taps)
        if (len(set(self.retain_feature_taps)) != len(self.retain_feature_taps)
                or any(type(index) is not int or not 0 <= index < len(model.layers) for index in self.retain_feature_taps)):
            raise ValueError('Unique native target-layer feature taps required')
        if type(norm_batch) is not bool:
            raise ValueError('Explicit boolean norm-batch selection required')
        if type(attention_replay) is not bool or (attention_replay and not norm_batch):
            raise ValueError('Explicit replay attention requires norm batching')
        validate_replay_options(attention_replay, attention_mask_once, replay_group_rows)
        from target_t16_attention_gate import validate_request_option
        validate_request_option(target_attention_t16, rows=max_verify_rows, position=session.position,
            remaining=session.max_new_tokens - len(session.emitted), replay=attention_replay,
            norm_batch=norm_batch, native_sampling=native_sampling_rows,
            group_rows=replay_group_rows, short_context=short_context)
        if target_attention_t16:
            from target_t16_attention_gate import qualify
            qualify(Path(__file__).parent)
        self.target_attention_t16 = target_attention_t16
        if attention_replay and max_verify_rows != 32 and not short_context and not target_attention_t16:
            raise ValueError('Width-cap experiment currently requires native attention')
        self.norm_batch = norm_batch
        self.attention_replay = attention_replay
        self.attention_mask_once = attention_mask_once
        self.replay_group_rows = replay_group_rows
        if session.phase != 'idle' or session.pending is not None or session.finished or len(helpers) != 48:
            raise ValueError('An unfinished prefilled request and all native GDN helpers are required')
        if len(pages.shape) != 2 or pages.shape[0] != 1:
            raise ValueError('One request page table required')
        widths = capture_widths(session.position, pages.shape[1] * 64, session.verifier_rows,
                                session.max_new_tokens - len(session.emitted), max_verify_rows)
        self.model, self.session, self.pages, self.helpers = model, session, pages, helpers
        self.operations, self.mesh, self.sampler = ttnn, model.mesh_device, sampler
        self.position = session.position
        self.phase, self.pending = 'preparing', None
        self.initial, self.buckets = [], {}
        self.carry, self.carry_addresses = [], []
        self.mtp_row_reader = None
        self.pending_key = None
        self.replay_plan = None
        if attention_replay:
            from attention_request_plan import capture_plan
            self.replay_plan = capture_plan(session.position, pages.shape[1] * 64, session.verifier_rows,
                session.max_new_tokens - len(session.emitted), max_verify_rows=max_verify_rows, short_context=short_context)
        self.native_addresses = [[addresses(ttnn, value) for value in helper.live] for helper in helpers]
        self.widths = widths
        started = time.perf_counter()
        session.begin_preparation(session.request_id)
        # The prefill before this engine and the capture below both rewrite slot 0:
        # whoever was resident no longer is, even if construction fails part way.
        note_prefill()
        try:
            if storage is None:
                for helper in helpers:
                    self.initial.append(helper.allocate())
            else:
                self.initial = self.borrow(validate_snapshots(storage.initial, helpers, 'initial'))
            # Before any capture, like every buffer a trace may see. Allocated after
            # capture it would sit where a trace's temporaries were and be overwritten
            # by the next replay; it is seeded after the last restore_initial below.
            self.allocate_carry()
            for helper, snapshot in zip(helpers, self.initial, strict=True):
                helper.save(snapshot)
            captures = [(rows, rows, self.position) for rows in self.widths] if self.replay_plan is None else [
                (capture.key, capture.rows, capture.position) for capture in self.replay_plan.captures]
            # Every bucket claims its pooled width first, so a request the pool cannot
            # hold is refused before anything is allocated or captured.
            slots = {key: None if storage is None else storage.take(rows) for key, rows, position in captures}
            for key, rows, position in captures:
                bucket = dict(rows=rows, capture_position=position, checkpoints=[], fixture=None, trace=None, output=None, commits={}, first=True)
                self.buckets[key] = bucket
                if self.retain_feature_taps:
                    from prepared_target_features import PreparedTargetFeatures

                    bucket['target_features'] = []
                    if slots[key] is None:
                        import torch

                        for index in self.retain_feature_taps:
                            feature = ttnn.from_torch(torch.zeros((1, 1, rows, 5120), dtype=torch.bfloat16),
                                device=self.mesh, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                                memory_config=ttnn.DRAM_MEMORY_CONFIG, mesh_mapper=ttnn.ShardTensorToMesh(self.mesh, dim=3))
                            bucket['target_features'].append(feature)
                    else:
                        if len(slots[key].target_features) < len(self.retain_feature_taps):
                            raise ValueError('Pooled %d-row bucket holds %d feature taps; this request retains %d'
                                             % (rows, len(slots[key].target_features), len(self.retain_feature_taps)))
                        bucket['target_features'] = self.borrow(list(slots[key].target_features[:len(self.retain_feature_taps)]))
                    bucket['feature_capture'] = PreparedTargetFeatures(model, self.retain_feature_taps,
                        bucket['target_features'], copy=ttnn.copy,
                        storage_ids=lambda value: tuple(enumerate(addresses(ttnn, value))))
            for key, rows, position in captures:
                bucket = self.buckets[key]
                if retain_mtp_hidden:
                    from mtp_hidden_capture import MTPHiddenCapture

                    if slots[key] is None:
                        import torch

                        hidden = ttnn.from_torch(torch.zeros((1, 1, rows, 5120), dtype=torch.bfloat16),
                            device=self.mesh, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                            memory_config=ttnn.DRAM_MEMORY_CONFIG, mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh))
                    else:
                        if slots[key].mtp_hidden is None:
                            raise ValueError('Pooled %d-row bucket holds no MTP hidden; this request retains it' % rows)
                        (hidden,) = self.borrow([slots[key].mtp_hidden])
                    bucket['mtp_hidden'] = hidden
                    bucket['mtp_capture'] = MTPHiddenCapture(model, hidden, copy=ttnn.copy,
                        storage_ids=lambda value: addresses(ttnn, value))
                if slots[key] is None:
                    for helper in helpers:
                        bucket['checkpoints'].append(helper.allocate())
                else:
                    bucket['checkpoints'] = self.borrow(validate_snapshots(slots[key].checkpoints, helpers, 'checkpoint'))
                bucket['fixture'] = self.fixture(rows, bucket['checkpoints'], retain=rows > 1, position=position,
                                                 storage=None if slots[key] is None else slots[key].batch)
            if retain_mtp_hidden:
                from mtp_hidden_rows import MTPHiddenRows
                self.mtp_row_reader = MTPHiddenRows(ttnn, self.mesh, [bucket['mtp_hidden'] for bucket in self.buckets.values()])
            if before_capture is not None:
                before_capture(self)
            for bucket in self.buckets.values():
                rows = bucket['rows']
                self.restore_initial()
                warm = self.fixture(rows, bucket['checkpoints'], retain=self.commit_only_gdn and rows > 1,
                    position=bucket['capture_position'])
                result = None
                try:
                    result = self.operation(warm, hidden_capture=bucket.get('mtp_capture'),
                        feature_capture=bucket.get('feature_capture'))
                    ttnn.synchronize_device(self.mesh)
                finally:
                    if result is not None:
                        release_owned(ttnn, [value for value in result if value is not None])
                    warm.close()
            if self.mtp_row_reader is not None:
                self.mtp_row_reader.prepare()
            for bucket in self.buckets.values():
                rows = bucket['rows']
                self.restore_initial()
                bucket['trace'], bucket['output'] = capture_operation(ttnn, self.mesh,
                    lambda bucket=bucket: self.operation(bucket['fixture'], hidden_capture=bucket.get('mtp_capture'),
                        feature_capture=bucket.get('feature_capture')))
                if rows > 1:
                    layers = [[*state.entry, result['states'], *result['packed_conv_states'],
                               state.gdn.rec_state, *state.gdn.conv_states, *checkpoint]
                              for state, result, checkpoint in bucket['fixture'].retained.records]
                    publications = {prefix: prepare(self.mesh, layers, prefix) for prefix in range(rows + 1)}
                    for publication in publications.values():
                        publication()
                    ttnn.synchronize_device(self.mesh)
                    for prefix, publication in publications.items():
                        bucket['commits'][prefix], unused = capture_operation(ttnn, self.mesh, publication)
            self.restore_initial()
            ttnn.synchronize_device(self.mesh)
            self.validate_bindings()
            # slot 0 holds this request's own prefill state again: seed the carry from it
            self.save_carry()
            self.setup_ms = (time.perf_counter() - started) * 1000
            session.finish_preparation(session.request_id)
            self.phase = 'idle'
        except BaseException:
            self.phase = 'failed'
            session.fail_preparation(session.request_id)
            self.close()
            raise

    def borrow(self, value):
        """Record lent storage so close() frees none of it; a snapshot set is a list of lists."""
        for item in value:
            if isinstance(item, list):
                self.borrowed.extend(item)
            else:
                self.borrowed.append(item)
        return value

    def fixture(self, rows, checkpoints, *, retain, position=None, pack=None, storage=None):
        # Packed, the rows belong to several requests: ModelBatch takes their
        # frontiers, page tables and per-layer GDN state from the pack, and the
        # position and pages below apply only to the unpacked case.
        # Only when packed: the image may run this module over a ModelBatch that
        # predates the pack= parameter (run 35481140763 died on it at admission).
        # Likewise storage=, the pooled fixture inputs, only when the engine is pooled.
        return ModelBatch(self.model, [1] * rows, self.position if position is None else position, self.pages, self.helpers, checkpoints,
            0 if rows == 1 else rows, **({'pack': pack} if pack is not None else {}),
            **({'storage': storage} if storage is not None else {}),
            serial_sdpa=True, compact_gdn=True, reuse_gdn_input=True,
            skip_row_clones=True, hoist_row_layout=True, device_loop_gdn=True, compact_prologue=True,
            batch_conv=True, packed_checkpoints=True, retain_records=retain, ordered_cache=True,
            norm_batch=self.norm_batch, attention_replay=getattr(self, 'attention_replay', False),
            attention_mask_once=getattr(self, 'attention_mask_once', False),
            replay_group_rows=getattr(self, 'replay_group_rows', 4),
            short_context=getattr(self, 'short_context', False) and getattr(self, 'attention_replay', False),
            attention_audit=getattr(self, 'attention_audit', False),
            **(dict(commit_only_gdn=True) if getattr(self, 'commit_only_gdn', False) and rows > 1 else {}))

    def proposal_rows(self):
        remaining = self.session.max_new_tokens - len(self.session.emitted)
        if getattr(self, 'replay_plan', None) is not None:
            return self.replay_plan.max_rows(self.position, remaining)
        return max(rows for rows in self.widths if rows <= remaining)

    def bucket_key(self, ticket):
        if getattr(self, 'replay_plan', None) is None:
            return len(ticket.tokens)
        remaining = self.session.max_new_tokens - len(self.session.emitted)
        return self.replay_plan.select(ticket.position, len(ticket.tokens), remaining).key

    def operation(self, fixture, *, hidden_capture=None, feature_capture=None):
        logits = None
        try:
            with ExitStack() as captures:
                for capture in (hidden_capture, feature_capture):
                    if capture is not None:
                        captures.enter_context(capture.capture())
                logits = fixture.run(sharded_logits=self.sampler is not None)
            ids = sample_rows(self.sampler, logits, fixture.rows, self.operations,
                native_rows=self.native_sampling_rows) if self.sampler is not None else None
            return logits, ids
        except BaseException:
            if logits is not None:
                self.operations.deallocate(logits)
            raise

    def restore_initial(self):
        for helper, snapshot in zip(self.helpers, self.initial, strict=True):
            helper.restore(snapshot)

    def allocate_carry(self):
        # One slot-zero snapshot per layer, seeded by save_carry once the layer holds
        # this request's state. Where each sits is recorded: a restore or save that
        # would read or write anywhere else is refused on the host before any copy.
        # Pooled, the carry is the slot's: allocated at attach, before ANY request's
        # trace, not merely before this one's - an earlier request's replay could
        # otherwise overwrite it between this engine's steps.
        storage = getattr(self, 'storage', None)
        if storage is None:
            self.carry = [helper.allocate() for helper in self.helpers]
        else:
            self.carry = self.borrow(validate_snapshots(storage.carry, self.helpers, 'carry'))
        self.carry_addresses = self.slot_addresses()

    def slot_addresses(self):
        return [[addresses(self.operations, value) for value in slot] for slot in self.carry]

    def copy_carry(self, operation, source):
        if self.slot_addresses() != self.carry_addresses:
            raise ValueError('Carried GDN state moved under the engine')
        # Eager: 48 launches per call. Capturing them as one trace is the follow-up.
        logging = os.environ.get('QWEN_FAST_CARRY_LOG') == '1'
        request, layers = str(self.session.request_id)[:48], len(self.carry)
        if logging:
            carry_log('[CARRY] op={op} request={request}{origin} layers={layers} begin', op=operation,
                request=request, origin='' if source is None else ' from=' + source, layers=layers)
        started = time.perf_counter()
        for helper, slot in zip(self.helpers, self.carry, strict=True):
            getattr(helper, operation)(slot)
        enqueued = time.perf_counter()
        if logging:
            # Fenced only when asked, so a normal run keeps its fencing as it is.
            self.operations.synchronize_device(self.mesh)
            carry_log('[CARRY] op={op} request={request} layers={layers} enqueue_ms={enqueue:.1f} fence_ms={fence:.1f}',
                op=operation, request=request, layers=layers, enqueue=(enqueued - started) * 1000,
                fence=(time.perf_counter() - enqueued) * 1000)

    def restore_carry(self):
        # Nothing to do while this engine is resident, so a lone request never pays.
        global _resident
        if not getattr(self, 'carry', None) or _resident is self:
            return False
        previous = 'none' if _resident is None else str(_resident.session.request_id)[:48]
        # Claimed before the copies, so a failure part way still sends the other
        # engine through a restore next time rather than trusting old residency.
        _resident = self
        self.copy_carry('restore', previous)
        return True

    def save_carry(self):
        global _resident
        _resident = self
        if getattr(self, 'carry', None):
            self.copy_carry('save', None)

    def validate_bindings(self):
        if any(helper.gdn.B != 8 or not helper.gdn._stable_state for helper in self.helpers):
            raise ValueError('Native stable B8 state contract changed')
        if [[addresses(self.operations, value) for value in helper.live] for helper in self.helpers] != self.native_addresses:
            raise ValueError('Native GDN buffers changed under captured verifier')

    def verify(self, ticket):
        self.session.check_ticket(self.session.request_id, ticket)
        key = self.bucket_key(ticket)
        if self.phase != 'idle' or self.pending is not None or ticket.position != self.position or key not in self.buckets:
            raise ValueError('An idle engine and its next supported request ticket are required')
        self.phase, self.pending = 'verifying', ticket
        self.pending_key = key
        bucket = self.buckets[key]
        try:
            binding_started = time.perf_counter()
            self.validate_bindings()
            carry_started = time.perf_counter()
            restored = self.restore_carry()
            started = time.perf_counter()
            stage_inputs(bucket['fixture'], ticket.tokens, ticket.position)
            staged = time.perf_counter()
            trace_ms = 0.0
            def operation():
                nonlocal trace_ms
                trace_started = time.perf_counter()
                result = self.operations.execute_trace(self.mesh, bucket['trace'], cq_id=0, blocking=True)
                trace_ms += (time.perf_counter() - trace_started) * 1000
                return result
            if bucket['first'] or bucket['fixture'].retained is None:
                operation()
                self.operations.synchronize_device(self.mesh)
            else:
                bucket['fixture'].retained.replay(operation)
            if getattr(self, 'attention_audit', False) and bucket['fixture'].replay_reader is not None:
                bucket['fixture'].replay_reader.audit.check(ticket.position, ticket.tokens)
            replay_finished = time.perf_counter()
            logits, ids = bucket['output']
            tensor = logits if ids is None else ids
            parts = self.operations.get_device_tensors(tensor)
            if len(parts) != 2:
                raise AssertionError('Two chip-local outputs required')
            host = self.operations.to_torch(parts[0])
            predictions = (host.reshape(len(ticket.tokens), self.model.args.vocab_size).float().argmax(dim=-1)
                           if ids is None else host.reshape(-1)[:len(ticket.tokens)]).tolist()
            finished = time.perf_counter()
            if len(predictions) != len(ticket.tokens):
                raise AssertionError('Missing target prediction rows')
            bucket['first'] = False
            self.phase = 'verified'
            return predictions, dict(input_ms=(staged - started) * 1000,
                verify_readback_ms=(finished - staged) * 1000,
                binding_validation_ms=(carry_started - binding_started) * 1000,
                carry_restore_ms=(started - carry_started) * 1000, carry_restored=restored,
                singleton_position_uploads=getattr(bucket['fixture'], 'last_singleton_uploads', None),
                blocking_trace_host_ms=trace_ms,
                replay_checks_sync_ms=(replay_finished - staged) * 1000 - trace_ms,
                output_readback_host_ms=(finished - replay_finished) * 1000)
        except BaseException:
            self.phase = 'failed'
            self.session.fail_verification(self.session.request_id, ticket)
            raise

    def verified_mtp_hidden(self, ticket):
        self.session.check_ticket(self.session.request_id, ticket)
        if (self.phase != 'verified' or self.pending is not ticket
                or not getattr(self, 'retain_mtp_hidden', False)):
            raise ValueError('MTP hidden requires its live verified ticket and opt-in retention')
        return self.buckets[self.pending_key]['mtp_capture'].output()

    def verified_mtp_hidden_for_publication(self, ticket):
        if (self.phase != 'verified' or self.pending is not ticket or self.session.pending is not ticket
                or self.session.phase != 'committing' or not self.retain_mtp_hidden):
            raise ValueError('MTP publication hidden requires the current committing request')
        return self.buckets[self.pending_key]['mtp_capture'].output()

    def mtp_row(self, source, row):
        if (self.phase != 'verified' or self.session.phase != 'committing'
                or self.mtp_row_reader is None or source is not self.buckets[self.pending_key]['mtp_hidden']):
            raise ValueError('Hidden row extraction is restricted to current target publication')
        return self.mtp_row_reader(source, row)

    def verified_features_for_publication(self, ticket):
        if (self.phase != 'verified' or self.pending is not ticket or self.session.pending is not ticket
                or self.session.phase != 'committing' or not self.retain_feature_taps):
            raise ValueError('Feature publication requires the current committing verifier ticket')
        return self.buckets[self.pending_key]['feature_capture'].outputs()

    def adopt_packed(self, ticket, block, segment):
        """This ticket was verified by a packed block (packed_verifier.PackedVerifierEngine),
        in that block's `segment`: become 'verified' on it, so DFlashRequestRuntime.publish
        runs unchanged. The synthetic bucket's features are the block's taps at this
        segment and its publication is the block's per-user commit; the carry is not
        saved afterwards, because that commit wrote the accepted state into it."""
        self.session.check_ticket(self.session.request_id, ticket)
        if self.phase != 'idle' or self.pending is not None or ticket.position != self.position:
            raise ValueError('Only an idle engine at the ticket frontier can adopt a packed verification')
        if (type(segment) is not int or segment < 0 or not callable(getattr(block, 'features', None))
                or not callable(getattr(block, 'commit_user', None))
                or getattr(block, 'rows_per_user', None) != len(ticket.tokens)):
            raise ValueError('A packed block segment holding exactly this ticket\'s rows is required')
        key = ('packed', segment)
        self.buckets[key] = dict(rows=len(ticket.tokens), capture_position=ticket.position, checkpoints=[],
            fixture=None, trace=None, output=None, commits={}, first=False, target_features=[],
            feature_capture=PackedFeatureView(block, segment), packed=(block, segment))
        self.pending_key = key
        self.phase, self.pending = 'verified', ticket

    def publish(self, prefix):
        global _resident
        ticket = self.pending
        if self.phase != 'verified' or ticket is None or self.session.pending is not ticket or self.session.phase != 'committing':
            raise ValueError('Publication requires the live verified ticket during its owner decision')
        if type(prefix) is not int or not 0 <= prefix <= len(ticket.tokens):
            raise ValueError('Selected prefix outside verified block')
        self.phase = 'committing'
        bucket = self.buckets[self.pending_key]
        packed = bucket.get('packed')
        try:
            if packed is not None:
                block, segment = packed
                block.commit_user(segment, prefix)
                # The block's commit wrote this user's accepted state into its carry, so
                # there is nothing to save; and slot 0 holds the last committed segment (or,
                # at prefix 0, whatever the block left), which no engine may trust.
                _resident = None
                del self.buckets[self.pending_key]
            else:
                if bucket['fixture'].retained is not None:
                    bucket['fixture'].retained.commit(prefix, dma=True, synchronize=True,
                        publication=lambda selected: self.operations.execute_trace(self.mesh, bucket['commits'][selected], cq_id=0, blocking=True))
                else:
                    self.validate_bindings()
                    if prefix == 0:
                        for helper, snapshot in zip(self.helpers, bucket['checkpoints'], strict=True):
                            helper.restore(snapshot)
                    self.operations.synchronize_device(self.mesh)
                self.save_carry()
            self.position += prefix
            self.phase, self.pending = 'idle', None
            self.pending_key = None
        except BaseException:
            self.phase = 'failed'
            raise

    def close(self):
        global _resident
        if self.phase == 'closed':
            return
        if self.phase not in ('idle', 'preparing', 'failed'):
            raise ValueError('Finish or abort the pending verifier block before closing')
        self.operations.synchronize_device(self.mesh)
        if getattr(self, 'mtp_row_reader', None) is not None:
            self.mtp_row_reader.close()
        for bucket in self.buckets.values():
            for trace in bucket['commits'].values():
                self.operations.release_trace(self.mesh, trace)
            if bucket['trace'] is not None:
                self.operations.release_trace(self.mesh, bucket['trace'])
        # Lent storage is the pool's: it goes back with the slot, never through deallocate.
        borrowed = {id(value) for value in getattr(self, 'borrowed', ())}

        def owned(values):
            return [value for value in values if id(value) not in borrowed]

        for bucket in self.buckets.values():
            if bucket.get('feature_capture') is not None:
                bucket['feature_capture'].close()
            release_owned(self.operations, owned(bucket.get('target_features', [])))
            if bucket.get('mtp_hidden') is not None:
                release_owned(self.operations, owned([bucket['mtp_hidden']]))
            if bucket['output'] is not None:
                release_owned(self.operations, [value for value in bucket['output'] if value is not None])
            if bucket['fixture'] is not None:
                bucket['fixture'].close()
            release_owned(self.operations, owned(value for snapshot in bucket['checkpoints'] for value in snapshot))
        release_owned(self.operations, owned(value for snapshot in self.initial for value in snapshot))
        release_owned(self.operations, owned(value for slot in getattr(self, 'carry', ()) for value in slot))
        self.carry, self.carry_addresses = [], []
        if getattr(self, 'borrowed', None):
            self.borrowed.clear()
        if _resident is self:
            _resident = None
        self.buckets.clear()
        self.initial.clear()
        self.phase, self.pending = 'closed', None
