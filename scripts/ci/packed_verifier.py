"""One packed verify block for every admitted user: its fixture, its trace, its per-user
commit traces, and the staging and readback that serve them.

WHAT IT BUYS. `serving_sequential_step` spends one pass over the 19.92 GB of dense
weights per user per round. This block runs ONE 32-row verify serving every user:
the weights are read once, the 48 GDN layers run one input and one output projection
over the whole block, and only the elementwise recurrence runs once per segment
(gdn_device_loop_state.DeviceLoopState.decode, segments=). The per-user work that
remains - the draft proposal and the feature publication - stays per user and is
untouched here.

CONSTRUCTION ORDER (docs/packed-device-step-plan-2026-09-20.md section 4;
serving_buffer_pool.py). Everything this engine keeps across rounds - the fixture's
32-row inputs including its 32 per-row page tables, the five (1, 1, 32, 5120) feature
taps, users x 48 GDN checkpoint sets, the retained block's per-layer entry and
histories, the verify trace and every commit trace - must exist BEFORE any trace that
will replay does. A request's verify trace bakes the addresses of the intermediates
its capture frees; a buffer allocated afterwards lands in those holes and every
replay of that trace overwrites it, per chip. On the serving path that means the
block is built at attach, AFTER ServingBufferPool (serving_runtime.py) and AFTER
PreparedDraftWeights, and BEFORE the lifecycle admits a request. What can be seen of
that order from here is enforced: construction is refused while any pool slot is
lent (a request device exists, so its traces may too), when the pool or the shared
draft weights are closed or the weights are not yet uploaded, and while a verifier
engine is resident in native GDN slot 0 (a request engine exists).

WHICH ROWS ARE WHOSE. Segment u of the block is rows [rows_per_user * u,
rows_per_user * (u + 1)). The verify trace restores segment u's GDN state from the
pool's slot-u carry (serving_buffer_pool.VerifierSlot.carry) inside the trace, before
that segment's recurrence, and segment u's commit traces DMA the accepted prefix
state straight back into the same carry. Both addresses are baked at attach, so a
request is served by the segment whose carry its engine borrowed - `segment_of` -
and NOT by its index in the scheduler's entries (probe 35436807668 saw the pair
presented as ['B', 'A']). What follows the ENTRIES order is everything the caller
sees: `verify(entries)` returns one prediction list per entry in entries order,
`stage_packed_inputs(entries)` writes each entry's tokens, positions and pages into
that entry's segment rows, and `features` and `commit_user` take the segment that
`segment_of` named for that entry's engine. Neither registry nor pool order is ever
used to index a result.

THE CARRY. Sequentially, each engine restores its carry into slot 0 before its trace
and saves slot 0 back after its commit, on the host, fenced (verifier_engine.py). Here
the restore is captured in the trace and the save IS the commit DMA, so no host copy
remains. Native slot 0 is left holding the last committed segment's user (or, after
a prefix-0 commit, whatever the block's recurrence left): `verifier_engine`'s
residency is cleared before the trace runs and after every packed commit, so a later
sequential step of any user restores first (`verifier_engine.note_packed_step`).
"""

from contextlib import ExitStack
import time
from types import SimpleNamespace
from typing import NamedTuple

from attention_batch import capture_operation
from force_argmax import sample_rows
from gdn_commit_dma import prepare
from gdn_multitoken_conv import addresses, release_owned
from model_batch import ModelBatch
from prepared_target_features import PreparedTargetFeatures
from serving_fast_policy import OUTPUT_BUDGET
from verifier_engine import note_packed_step, note_prefill
import verifier_engine
from verifier_inputs import host_inputs, validate_tokens
from verifier_pack import GDN_LAYERS, build_pack, participant

FEATURE_WIDTH = 5120
LEGAL_WIDTHS = (1, 2, 4, 8, 16, 32)


class PackedShape(NamedTuple):
    """The block geometry every shape-dependent choice is keyed on: M1 is
    (2, 16, 32, page_width, page_width * 64); M2 adds (4, 8, 32, ...) and M3 needs 64 rows."""
    users: int
    rows_per_user: int
    block_rows: int
    page_width: int
    capacity: int


def validate_shape(shape):
    if (not isinstance(shape, PackedShape) or any(type(value) is not int for value in shape)
            or shape.users < 1 or shape.rows_per_user not in LEGAL_WIDTHS
            or shape.block_rows not in (2, 4, 8, 16, 32) or shape.users * shape.rows_per_user != shape.block_rows
            or shape.page_width < 68 or shape.capacity != shape.page_width * 64):
        raise ValueError('A packed shape needs users x rows_per_user == block_rows within the 32-row block, '
                         'and a page-table width of whole 64-token pages')
    return shape


def m1_shape(page_width):
    """Two T16 users in one 32-row block over the serving page-table width."""
    return validate_shape(PackedShape(2, 16, 32, page_width, page_width * 64))


def segment_rows(shape, segment):
    if type(segment) is not int or not 0 <= segment < shape.users:
        raise ValueError('Segment outside the packed block')
    return segment * shape.rows_per_user, (segment + 1) * shape.rows_per_user


class PackedFeatureTaps(tuple):
    """The block's five 32-row taps with one user's row offset attached, so that
    DFlashDevice.project_features (dflash_device.py, row_offset=) slices that user's rows
    while DFlashRequestRuntime.publish passes the taps through unchanged."""

    def __new__(cls, taps, *, row_offset, rows):
        self = super().__new__(cls, taps)
        self.row_offset, self.rows = row_offset, rows
        return self


def validate_commit_layers(layers, carry):
    """One user's commit layers as the retained block builds them (gdn_records.RetainedGDNBlock
    .segment_layers): 48 records of the twenty tensors gdn_commit_dma.prepare takes, each
    ending in that user's CARRY - the checkpoint destination, so the accepted prefix state
    lands where the next round's segment restore reads it."""
    layers = [list(layer) for layer in layers]
    if (len(layers) != GDN_LAYERS or any(len(layer) != 20 for layer in layers)
            or any(len(layer[15:]) != len(slot) or any(a is not b for a, b in zip(layer[15:], slot))
                   for layer, slot in zip(layers, carry, strict=True))):
        raise ValueError("A packed user's commit must write into that user's own carry in every GDN layer")
    return layers


def packed_host_inputs(users, shape, rope_dim, theta, vocab_size):
    """Host tokens (block_rows, 1), positions (block_rows,), cos/sin (1, block_rows, 1, rope)
    and per-row page tables (block_rows, page_width), segment by segment."""
    import torch

    if len(users) != shape.users or any(user is None for user in users):
        raise ValueError('One (tokens, start, pages) per segment required')
    tokens, positions, cos, sin, pages = [], [], [], [], []
    for user_tokens, start, table in users:
        user_tokens = list(user_tokens)
        validate_tokens(user_tokens, shape.rows_per_user, start, vocab_size, shape.capacity)
        if (getattr(table, 'ndim', None) != 2 or tuple(table.shape) != (1, shape.page_width)
                or table.dtype != torch.int32):
            raise ValueError('Each packed user stages its own (1, page_width) int32 page table')
        token_values, user_positions, user_cos, user_sin = host_inputs(user_tokens, start, rope_dim, theta)
        tokens.append(token_values)
        positions.append(user_positions)
        cos.append(user_cos)
        sin.append(user_sin)
        pages.append(table.repeat(shape.rows_per_user, 1))
    return (torch.cat(tokens), torch.cat(positions), torch.cat(cos, dim=1), torch.cat(sin, dim=1),
            torch.cat(pages))


def stage_packed(operations, model, fixture, shape, users):
    """Write every user's inputs into the captured fixture's buffers, segment by segment.

    verifier_inputs.stage_inputs cannot serve a packed block: it builds one arange from
    one start and never restages pages. Here the tokens, the per-user positions, the rotary
    tables, every singleton position, the (block_rows, page_width) table AND every per-row
    table (attention_batch.SerialAttentionReader reads row_pages[i]) are restaged each
    round, from each user's own host page table. Returns the number of buffers written.
    """
    if fixture.rows != shape.block_rows or len(fixture.singleton_positions) != shape.block_rows \
            or len(fixture.row_pages) != shape.block_rows:
        raise ValueError('Every per-row attention position and page table of the block must be retained')
    tokens, positions, cos, sin, pages = packed_host_inputs(users, shape, model.args.rope_head_dim,
                                                            model.args.rope_theta, model.args.vocab_size)
    values = [(fixture.tokens, tokens, operations.uint32, operations.ROW_MAJOR_LAYOUT),
              (fixture.positions, positions, operations.int32, operations.ROW_MAJOR_LAYOUT),
              (fixture.cos, cos, operations.bfloat16, operations.TILE_LAYOUT),
              (fixture.sin, sin, operations.bfloat16, operations.TILE_LAYOUT),
              (fixture.pages, pages, operations.int32, operations.ROW_MAJOR_LAYOUT)]
    values.extend((destination, positions[index:index + 1], operations.int32, operations.ROW_MAJOR_LAYOUT)
                  for index, destination in enumerate(fixture.singleton_positions))
    values.extend((destination, pages[index:index + 1], operations.int32, operations.ROW_MAJOR_LAYOUT)
                  for index, destination in enumerate(fixture.row_pages))
    if any(tuple(destination.shape) != tuple(value.shape) or destination.dtype != dtype or destination.layout != layout
           for destination, value, dtype, layout in values):
        raise ValueError('Staged packed inputs must preserve every captured tensor signature')
    destinations = [destination for destination, value, dtype, layout in values]
    before = [addresses(operations, destination) for destination in destinations]
    if any(len({pair[chip] for pair in before}) != len(before) for chip in range(2)):
        raise ValueError('Two independent chip-local buffers per packed input required')
    staged = [operations.from_torch(value, device=None, dtype=dtype, layout=layout,
                                    mesh_mapper=operations.ReplicateTensorToMesh(model.mesh_device))
              for destination, value, dtype, layout in values]
    try:
        for source, destination in zip(staged, destinations, strict=True):
            operations.copy_host_to_device_tensor(source, destination)
    finally:
        operations.synchronize_device(model.mesh_device)
    if [addresses(operations, destination) for destination in destinations] != before:
        raise AssertionError('Packed input staging replaced a captured buffer')
    return len(destinations)


class PackedVerifierEngine:
    """Owner of one packed verify block: build at attach, then per round
    `verify(entries)` -> per-entry predictions, `features(segment)` for each user's
    publication, `commit_user(segment, prefix)` for each user's decision."""

    def __init__(self, operations, model, helpers, sampler, *, pool, shared_weights, shape, feature_taps,
                 capture_position=None):
        import torch

        self.shape = validate_shape(shape)
        self.rows_per_user, self.block_rows, self.users = shape.rows_per_user, shape.block_rows, shape.users
        helpers = tuple(helpers)
        if len(helpers) != GDN_LAYERS or sampler is None:
            raise ValueError('All native GDN helpers and the pinned force-argmax sampler are required')
        self.feature_taps = tuple(feature_taps)
        if (not self.feature_taps or len(set(self.feature_taps)) != len(self.feature_taps)
                or any(type(index) is not int or not 0 <= index < len(model.layers) for index in self.feature_taps)):
            raise ValueError('Unique native target-layer feature taps required')
        if capture_position is None:
            capture_position = shape.capacity - OUTPUT_BUDGET
        if type(capture_position) is not int or capture_position < 0 or capture_position + shape.rows_per_user > shape.capacity:
            raise ValueError('Capture position must leave one segment within the page capacity')
        self.capture_position = capture_position
        # The construction-order rule, as far as the pool and the weights show it.
        if (getattr(pool, 'closed', True) or getattr(pool, 'helpers', None) is None
                or getattr(pool, 'page_width', None) != shape.page_width
                or len(getattr(pool, 'slots', ())) < shape.users):
            raise ValueError('An open serving buffer pool with verifier storage for every packed user, '
                             'at this page-table width, is required')
        if any(slot.lent for slot in pool.slots):
            raise ValueError('The packed block must be built before any request exists: a pool slot is lent')
        if (getattr(shared_weights, 'closed', True) or not callable(getattr(shared_weights, 'lend', None))
                or not getattr(shared_weights, 'tensors', None)):
            raise ValueError('The packed block must be built after the shared draft weights are uploaded')
        if verifier_engine._resident is not None:
            raise ValueError('The packed block must be built before any request engine exists: one is resident')
        self.operations, self.model, self.mesh, self.sampler = operations, model, model.mesh_device, sampler
        self.helpers = helpers
        self.name = 'PackedVerifierEngine@%x users=%d rows=%d' % (id(self), shape.users, shape.rows_per_user)
        self.carries = []
        for index in range(shape.users):
            carry = pool.slots[index].verifier.carry
            if len(carry) != GDN_LAYERS or any(len(snapshot) != len(helper.live)
                                                for snapshot, helper in zip(carry, helpers, strict=True)):
                raise ValueError('Pooled carry %d must hold one slot-zero snapshot per GDN helper' % index)
            self.carries.append([list(snapshot) for snapshot in carry])
        self.native_addresses = [[addresses(operations, value) for value in helper.live] for helper in helpers]
        self.carry_addresses = self.slot_addresses()
        self.checkpoints, self.taps, self.owned = [], [], []
        self.feature_capture = self.fixture = None
        self.trace = self.output = None
        self.commits = [{} for user in range(shape.users)]
        self.phase, self.first, self.rounds = 'preparing', True, 0
        self.pending_segments = set()
        started = time.perf_counter()
        try:
            # Allocated before ANY capture, like everything a trace may see.
            for user in range(shape.users):
                self.checkpoints.append([helper.allocate() for helper in helpers])
            for index in self.feature_taps:
                tap = operations.from_torch(torch.zeros((1, 1, shape.block_rows, FEATURE_WIDTH), dtype=torch.bfloat16),
                    device=self.mesh, dtype=operations.bfloat16, layout=operations.TILE_LAYOUT,
                    memory_config=operations.DRAM_MEMORY_CONFIG, mesh_mapper=operations.ShardTensorToMesh(self.mesh, dim=3))
                self.taps.append(tap)
            self.feature_capture = PreparedTargetFeatures(model, self.feature_taps, self.taps, copy=operations.copy,
                storage_ids=lambda value: tuple(enumerate(addresses(operations, value))))
            # The capture's placeholder users: every row reads physical page 0 at the
            # capture position. Positions, rotary tables and every page table are
            # restaged before each verify; only the addresses and shapes are baked.
            placeholders = [SimpleNamespace(position=capture_position,
                                            pages=torch.zeros((1, shape.page_width), dtype=torch.int32))
                            for user in range(shape.users)]
            warm = self.build_fixture(placeholders)
            result = None
            try:
                result = self.operation(warm)
                operations.synchronize_device(self.mesh)
            finally:
                if result is not None:
                    release_owned(operations, [value for value in result if value is not None])
                warm.close()
            self.fixture = self.build_fixture(placeholders)
            self.trace, self.output = capture_operation(operations, self.mesh, lambda: self.operation(self.fixture))
            retained = self.fixture.retained
            if len(retained.records) != GDN_LAYERS:
                raise ValueError('The captured packed block must retain every GDN layer')
            for user in range(shape.users):
                layers = validate_commit_layers(retained.segment_layers(user), self.carries[user])
                # Prefix 0 is a no-op on the carry (the deferred decode never advanced it, and
                # the segment's entry IS the carry), so no trace is captured for it: the
                # retained block's commit_user publishes nothing at prefix 0.
                publications = {prefix: prepare(self.mesh, layers, prefix) for prefix in range(1, shape.rows_per_user + 1)}
                for publication in publications.values():
                    publication()
                operations.synchronize_device(self.mesh)
                for prefix, publication in publications.items():
                    self.commits[user][prefix], unused = capture_operation(operations, self.mesh, publication)
            operations.synchronize_device(self.mesh)
            self.validate_bindings()
            # The captures rewrote slot 0: nobody is resident.
            note_prefill()
            self.setup_ms = (time.perf_counter() - started) * 1000
            self.phase = 'idle'
        except BaseException:
            self.phase = 'failed'
            self.close()
            raise

    def build_fixture(self, placeholders):
        """The packed ModelBatch: retained records for the per-user commits; commit-only GDN,
        which packed is the DEFERRED decode (gdn_device_loop_state: one block-start entry per
        user, nothing restored or advanced at decode, every decision made by commit_user
        after the readback); no replay attention (the per-row SerialAttentionReader takes
        singleton_positions and row_pages, which are restaged per user); and no T16 gate
        (it pins rows == 16)."""
        pack = build_pack([participant(placeholder, self.rows_per_user, 0, self.checkpoints[user], self.carries[user])
                           for user, placeholder in enumerate(placeholders)], block_rows=self.block_rows)
        return ModelBatch(self.model, [1] * self.block_rows, self.capture_position, placeholders[0].pages, self.helpers,
            self.checkpoints[0], self.block_rows, pack=pack,
            serial_sdpa=True, compact_gdn=True, reuse_gdn_input=True,
            skip_row_clones=True, hoist_row_layout=True, device_loop_gdn=True, compact_prologue=True,
            batch_conv=True, packed_checkpoints=True, retain_records=True, ordered_cache=True,
            norm_batch=True, attention_replay=False, attention_mask_once=False, replay_group_rows=4,
            short_context=False, attention_audit=False, commit_only_gdn=True)

    def operation(self, fixture):
        logits = None
        try:
            with ExitStack() as captures:
                captures.enter_context(self.feature_capture.capture())
                logits = fixture.run(sharded_logits=True)
            ids = sample_rows(self.sampler, logits, self.block_rows, self.operations, native_rows=False)
            return logits, ids
        except BaseException:
            if logits is not None:
                self.operations.deallocate(logits)
            raise

    def slot_addresses(self):
        return [[[addresses(self.operations, value) for value in snapshot] for snapshot in carry] for carry in self.carries]

    def validate_bindings(self):
        if any(helper.gdn.B != 8 or not helper.gdn._stable_state for helper in self.helpers):
            raise ValueError('Native stable B8 state contract changed')
        if [[addresses(self.operations, value) for value in helper.live] for helper in self.helpers] != self.native_addresses:
            raise ValueError('Native GDN buffers changed under the packed block')
        if self.slot_addresses() != self.carry_addresses:
            raise ValueError('A carried GDN state moved under the packed block')

    def segment_of(self, engine):
        """The segment whose carry this request's engine borrowed."""
        carry = getattr(engine, 'carry', None) or ()
        for segment, own in enumerate(self.carries):
            if len(carry) == len(own) and all(len(mine) == len(theirs) and all(a is b for a, b in zip(mine, theirs))
                                              for mine, theirs in zip(own, carry)):
                return segment
        raise ValueError('The request engine borrows no carry this block restores: it was not admitted '
                         'through a pool slot the block was captured against')

    def segments(self, entries):
        """One segment per entry, in entries order; every segment exactly once."""
        entries = list(entries)
        if len(entries) != self.users:
            raise ValueError('The packed block serves exactly %d users; %d entries given' % (self.users, len(entries)))
        segments = tuple(self.segment_of(entry['request'].engine) for entry in entries)
        if sorted(segments) != list(range(self.users)):
            raise ValueError('Two packed entries were admitted through the same pool slot')
        return segments

    def stage_packed_inputs(self, entries):
        entries = list(entries)
        segments = self.segments(entries)
        users = [None] * self.users
        for entry, segment in zip(entries, segments):
            ticket, engine = entry['ticket'], entry['request'].engine
            users[segment] = (tuple(ticket.tokens), ticket.position, engine.pages)
        return stage_packed(self.operations, self.model, self.fixture, self.shape, users)

    def verify(self, entries):
        """One trace for every entry. Returns (predictions, metrics): predictions[i] is
        entries[i]'s rows_per_user target ids; metrics['segments'][i] is entries[i]'s segment."""
        entries = list(entries)
        if self.phase != 'idle':
            raise ValueError('An idle packed block is required: every segment of the last round must be committed')
        segments = self.segments(entries)
        for entry, segment in zip(entries, segments):
            request, ticket = entry['request'], entry['ticket']
            engine = request.engine
            engine.session.check_ticket(engine.session.request_id, ticket)
            if (ticket.request_id != entry['request_id'] or engine.phase != 'idle' or engine.pending is not None
                    or ticket.position != engine.position or len(ticket.tokens) != self.rows_per_user):
                raise ValueError('Every packed entry needs an idle engine and a full %d-row ticket at its frontier'
                                 % self.rows_per_user)
        self.phase = 'verifying'
        try:
            binding_started = time.perf_counter()
            self.validate_bindings()
            started = time.perf_counter()
            staged = self.stage_packed_inputs(entries)
            staged_at = time.perf_counter()
            # Claimed before the trace: its segment restores rewrite slot 0, so no engine
            # may trust its residency from here on, even if the trace fails part way.
            note_packed_step()
            trace_ms = 0.0

            def operation():
                nonlocal trace_ms
                trace_started = time.perf_counter()
                result = self.operations.execute_trace(self.mesh, self.trace, cq_id=0, blocking=True)
                trace_ms += (time.perf_counter() - trace_started) * 1000
                return result

            if self.first:
                operation()
                self.operations.synchronize_device(self.mesh)
            else:
                self.fixture.retained.replay(operation)
            replayed = time.perf_counter()
            logits, ids = self.output
            parts = self.operations.get_device_tensors(ids)
            if len(parts) != 2:
                raise AssertionError('Two chip-local outputs required')
            host = self.operations.to_torch(parts[0]).reshape(-1)[:self.block_rows].tolist()
            if len(host) != self.block_rows:
                raise AssertionError('Missing packed prediction rows')
            predictions = [host[slice(*segment_rows(self.shape, segment))] for segment in segments]
            finished = time.perf_counter()
            self.first = False
            self.pending_segments = set(segments)
            self.rounds += 1
            self.phase = 'verified'
            return predictions, dict(segments=segments, staged_buffers=staged,
                binding_validation_ms=(started - binding_started) * 1000,
                input_ms=(staged_at - started) * 1000, verify_readback_ms=(finished - staged_at) * 1000,
                blocking_trace_host_ms=trace_ms, replay_checks_sync_ms=(replayed - staged_at) * 1000 - trace_ms,
                output_readback_host_ms=(finished - replayed) * 1000, users=self.users)
        except BaseException:
            self.phase = 'failed'
            for entry in entries:
                session, ticket = entry['request'].session, entry['ticket']
                if session.phase == 'pending' and session.pending is ticket:
                    session.fail_verification(session.request_id, ticket)
            raise

    def check_segment(self, segment):
        segment_rows(self.shape, segment)
        if self.phase != 'verified' or segment not in self.pending_segments:
            raise ValueError('Segment %d has no verified, uncommitted block' % segment)

    def features(self, segment):
        """The block's taps at this user's row offset, for the user's own publication."""
        self.check_segment(segment)
        start, unused = segment_rows(self.shape, segment)
        return PackedFeatureTaps(self.feature_capture.outputs(), row_offset=start, rows=self.rows_per_user)

    def execute_commit(self, segment, prefix):
        if prefix:
            self.operations.execute_trace(self.mesh, self.commits[segment][prefix], cq_id=0, blocking=True)

    def commit_user(self, segment, prefix):
        """Commit one user's decision: its own prefix trace writes the accepted state into
        native slot 0 and into that user's carry; prefix 0 runs nothing."""
        self.check_segment(segment)
        if type(prefix) is not int or not 0 <= prefix <= self.rows_per_user:
            raise ValueError('Selected prefix outside the user segment')
        try:
            self.fixture.retained.commit_user(segment, prefix, dma=True, synchronize=True,
                publication=lambda selected: self.execute_commit(segment, selected))
            self.pending_segments.discard(segment)
            if not self.pending_segments:
                self.phase = 'idle'
        except BaseException:
            self.phase = 'failed'
            raise
        finally:
            # Slot 0 holds this segment's user, or after prefix 0 nobody's committed state.
            note_packed_step()

    def describe(self):
        operations = self.operations
        return dict(name='packed', shape=self.shape._asdict(), capture_position=self.capture_position,
            weight_passes_per_round='one for every user', batched=True,
            verify_traces=1 if self.trace is not None else 0,
            commit_traces=sum(len(commits) for commits in self.commits),
            taps=[list(addresses(operations, tap)) for tap in self.taps],
            checkpoints=[list(addresses(operations, checkpoints[0][0])) for checkpoints in self.checkpoints],
            carries=[list(carry[0][0]) for carry in self.carry_addresses],
            setup_ms=getattr(self, 'setup_ms', None), rounds=self.rounds)

    def close(self):
        if self.phase == 'closed':
            return
        if self.phase not in ('idle', 'preparing', 'failed'):
            raise ValueError('Commit every segment of the pending packed block before closing')
        operations = self.operations
        operations.synchronize_device(self.mesh)
        for commits in self.commits:
            for trace in commits.values():
                operations.release_trace(self.mesh, trace)
            commits.clear()
        if self.trace is not None:
            operations.release_trace(self.mesh, self.trace)
            self.trace = None
        if self.feature_capture is not None:
            self.feature_capture.close()
        if self.output is not None:
            release_owned(operations, [value for value in self.output if value is not None])
            self.output = None
        if self.fixture is not None:
            self.fixture.close()
            self.fixture = None
        release_owned(operations, self.taps)
        release_owned(operations, [value for checkpoints in self.checkpoints for snapshot in checkpoints for value in snapshot])
        self.taps, self.checkpoints = [], []
        # The carries are the pool's: lent to the requests, never freed here.
        self.carries, self.carry_addresses = [], []
        self.pending_segments.clear()
        self.phase = 'closed'
