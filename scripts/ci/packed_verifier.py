"""One packed verify block for every admitted user: its fixture, its trace, its per-user
commit traces, and the staging and readback that serve them.

WHAT IT BUYS. `serving_sequential_step` spends one pass over the 19.92 GB of dense
weights per user per round. This block runs ONE block_rows-row verify serving every
user - 32 rows for two T16 users (M1), 64 for four (M3, packed_shapes.m3_shape): the
weights are read once, the 48 GDN layers run one input and one output projection over
the whole block, and only the elementwise recurrence runs once per segment
(gdn_device_loop_state.DeviceLoopState.decode, segments=). The per-user work that
remains - the draft proposal and the feature publication - stays per user and is
untouched here.

THE SHAPE. Every per-segment structure scales from the `PackedShape` (packed_shapes.py):
segments bound to pool slots by carry identity, users x 48 checkpoint sets, five
(1, 1, block_rows, 5120) taps, one replay reader per user, one positions word and one
bundle-table set per user, a block_rows-row staging batch, users x rows_per_user commit
traces, block_rows sampled ids. Nothing here is written for two users. At 64 rows the
K/V write of every full-attention layer runs as two 32-row tiles of the audited ordered
kernel (packed_cache_writer.py), because the per-row serial adapters of the pinned
attention_batch.py stop at 32 rows; the fixture keeps each tile's own positions word and
page-table rows, restaged here with everything else.

CONSTRUCTION ORDER (docs/packed-device-step-plan-2026-09-20.md section 4;
serving_buffer_pool.py). Everything this engine keeps across rounds - the fixture's
block_rows-row inputs including its per-row page tables and cache tiles, the five
(1, 1, block_rows, 5120) feature taps, users x 48 GDN checkpoint sets, the retained
block's per-layer entries and histories, the verify trace and every commit trace - must
exist BEFORE any trace that
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

THE ATTENTION (M1b). Run 35497378631 measured the packed verify at 150.6 ms against
67 ms for a 16-row sequential verify, and the cost model puts ~56 ms of the difference
in the attention: 16 full-attention layers each running 32 per-row serial SDPA launches
at 32K context, where the sequential verify's bundled replay reader (four-row groups)
takes 9.9 ms for all 16 layers. The pinned reader (attention_replay.py, frozen-recipe
evidence) serves ONE user: one start word, one page table, bundles over its rows. So the
block's fixture builds one reader PER USER, each over that user's own start word and its
own page tables, captured in the block's native chunk family (the capture position's,
which the serving pin - position 32768, budget 256 - makes the whole decode's), and
dispatches each layer's 32-row query a segment at a time
(pooled_attention_replay.PackedReplayAttentionReader): user u's rows through user u's
bundles, results concatenated in row order. The masks are recomputed inside the trace
from each reader's positions word. The readers' per-bundle page tables are the pool's
(serving_buffer_pool.PackedReplayTables, one set per user for this shape, lent once to
the block), and `stage_packed` restages every user's positions word and bundle tables
before each verify along with its rows. Expected: attention back to ~2 x 9.9 ms, about
46 ms saved per two-user round.

THE CARRY. Sequentially, each engine restores its carry into slot 0 before its trace
and saves slot 0 back after its commit, on the host, fenced (verifier_engine.py). Here
the restore is captured in the trace and the save IS the commit DMA, so no host copy
remains. Native slot 0 is left holding the last committed segment's user (or, after
a prefix-0 commit, whatever the block's recurrence left): `verifier_engine`'s
residency is cleared before the trace runs and after every packed commit, so a later
sequential step of any user restores first (`verifier_engine.note_packed_step`).
"""

from contextlib import ExitStack
import os
import sys
import time
import traceback
from types import SimpleNamespace

from attention_batch import capture_operation
from attention_mask_replay import validate_ticket
from force_argmax import sample_rows
from gdn_commit_dma import prepare
from gdn_multitoken_conv import addresses, release_owned
from model_batch import ModelBatch
from packed_cache_writer import tile_rows
# The shape and its validation live in packed_shapes.py (one definition for M1 and M3);
# the names stay importable from here.
from packed_shapes import PackedShape, ROWS_PER_USER as LEGAL_WIDTHS, m1_shape, m3_shape, segment_rows, validate_shape
from prepared_target_features import PreparedTargetFeatures
from serving_fast_policy import OUTPUT_BUDGET
from verifier_engine import note_packed_step, note_prefill
import verifier_engine
from verifier_inputs import host_inputs, validate_tokens
from verifier_pack import GDN_LAYERS, build_pack, participant

FEATURE_WIDTH = 5120


def diagnostic(text):
    """One [PINDIAG] line into the server log: loguru where it exists, stderr otherwise
    (the lane attaches both). Never raises: a failed report must not mask the failure
    it reports."""
    try:
        try:
            from loguru import logger
        except ImportError:
            print(text, file=sys.stderr, flush=True)
        else:
            logger.info('{}', text)
    except BaseException:
        pass


class PackedFeatureTaps(tuple):
    """The block's five block_rows-row taps with one user's row offset attached, so that
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
    round, from each user's own host page table; when the fixture's attention is the
    per-user replay reader (M1b), each user's positions word and its reader's per-bundle
    page tables, the way verifier_inputs.stage_inputs stages one request's word and
    VerifierPageBinding.refresh rewrites its tables; and, beyond one 32-row tile (M3),
    each cache tile's own positions word and page-table rows for the tile-by-tile K/V
    write (packed_cache_writer.py). One fence for all of it. Returns the number of
    buffers written.
    """
    import torch

    if fixture.rows != shape.block_rows or len(fixture.singleton_positions) != shape.block_rows \
            or len(fixture.row_pages) != shape.block_rows:
        raise ValueError('Every per-row attention position and page table of the block must be retained')
    reader = getattr(fixture, 'replay_reader', None)
    readers = () if reader is None else tuple(reader.readers)
    if reader is not None and len(readers) != shape.users:
        raise ValueError('One replay reader per packed user required')
    tiles = tuple(getattr(fixture, 'cache_tiles', None) or ())
    if tiles and tuple(tile.rows for tile in tiles) != tile_rows(shape.block_rows):
        raise ValueError('The cache tiles must cover the block in 32-row tiles')
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
    for tile in tiles:
        first, last = tile.rows
        values.append((tile.positions, positions[first:last].contiguous(), operations.int32, operations.ROW_MAJOR_LAYOUT))
        values.append((tile.pages, pages[first:last].contiguous(), operations.int32, operations.ROW_MAJOR_LAYOUT))
    for own, (user_tokens, start, table) in zip(readers, users, strict=True):
        # Host only, before any copy: this user's ticket inside its reader's family.
        own.validate(start)
        words = torch.zeros(8, dtype=torch.int32)
        words[0] = start
        values.append((own.positions, words, operations.int32, operations.ROW_MAJOR_LAYOUT))
        values.extend((entry[1], table[:, :own.capacity // 64].repeat(len(entry[0]), 1).contiguous(),
                       operations.int32, operations.ROW_MAJOR_LAYOUT) for entry in own.metadata)
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
        try:
            for source, destination in zip(staged, destinations, strict=True):
                operations.copy_host_to_device_tensor(source, destination)
        finally:
            operations.synchronize_device(model.mesh_device)
        if [addresses(operations, destination) for destination in destinations] != before:
            raise AssertionError('Packed input staging replaced a captured buffer')
    except BaseException:
        # A half-staged word or table poisons the readers, as stage_inputs poisons one.
        for own in readers:
            own.failed = True
        raise
    for own, (user_tokens, start, table) in zip(readers, users, strict=True):
        own.start = start
    return len(destinations)


PROFILE_DUMP_ROUND = 'QWEN_FAST_PROFILE_DUMP_ROUND'


def dump_device_profiler_after_round(operations, mesh, rounds, environ=None):
    """Under the m3native profile arm only (QWEN_FAST_PROFILE_DUMP_ROUND=N): read the device
    profiler buffers back once, right after packed round N, so cpp_device_perf_report.csv
    exists before anything later in the run can end the process uncleanly. The pinned
    profiler writes that CSV only on a clean device close or on ttnn.ReadDeviceProfiler,
    and run 35564623068 profiled seven packed rounds then died (a user finishing first)
    with nothing written. Returns True when the dump was issued; inert when unset."""
    import os

    value = (os.environ if environ is None else environ).get(PROFILE_DUMP_ROUND, '')
    if not value:
        return False
    try:
        target = int(value)
    except ValueError:
        raise ValueError('%s must be a round number; got %r' % (PROFILE_DUMP_ROUND, value))
    if rounds != target:
        return False
    reader = getattr(operations, 'ReadDeviceProfiler', None)
    if reader is None:
        diagnostic('[PINDIAG] %s=%d but the runtime has no ReadDeviceProfiler; no device dump' % (PROFILE_DUMP_ROUND, target))
        return False
    reader(mesh)
    diagnostic('[PINDIAG] device profiler read back after packed round %d (%s)' % (rounds, PROFILE_DUMP_ROUND))
    return True


class PackedVerifierEngine:
    """Owner of one packed verify block: build at attach, then per round
    `verify(entries)` -> per-entry predictions, `features(segment)` for each user's
    publication, `commit_user(segment, prefix)` for each user's decision."""

    def __init__(self, operations, model, helpers, sampler, *, pool, shared_weights, shape, feature_taps,
                 capture_position=None, pool_slots=None):
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
        # Which pool slots this block's segments carry: slots 0..users-1 in segment order by
        # default - the only binding a single serving block ever needed. Given explicitly
        # when several blocks share one pool (serving_runtime's QWEN_FAST_FOUR_AS_TWO, two
        # 32-row M1 blocks over a four-slot pool): each block claims its own disjoint slots
        # - (0, 1) and (2, 3) - so its carries restore from and commit to the users admitted
        # through THOSE slots, and `segment_of` never matches a request bound to the other
        # block's slots.
        if pool_slots is None:
            pool_slots = tuple(range(shape.users))
        else:
            pool_slots = tuple(pool_slots)
        if (len(pool_slots) != shape.users or len(set(pool_slots)) != shape.users
                or any(type(index) is not int or not 0 <= index < len(pool.slots) for index in pool_slots)):
            raise ValueError('One distinct pool slot per packed user, within the pool, is required')
        self.pool_slots = pool_slots
        if (getattr(shared_weights, 'closed', True) or not callable(getattr(shared_weights, 'lend', None))
                or not getattr(shared_weights, 'tensors', None)):
            raise ValueError('The packed block must be built after the shared draft weights are uploaded')
        if verifier_engine._resident is not None:
            raise ValueError('The packed block must be built before any request engine exists: one is resident')
        # The attention (M1b): one bundled replay reader per user, every one captured in the
        # block's native chunk family - the capture position's, as for one request - over
        # the pool's table sets for this shape (serving_buffer_pool.PackedReplayTables), lent
        # once to this block. Refused, like everything above, before anything is allocated.
        self.replay = None
        self.replay_capacity = (capture_position // 256 + 1) * 256
        packed = pool.packed_replay(shape.users, shape.rows_per_user)
        if self.replay_capacity not in packed.replay_pages:
            raise ValueError('The pool holds packed replay page tables for families %r; the block captures in family %d'
                             % (sorted(packed.replay_pages), self.replay_capacity))
        validate_ticket(capture_position, shape.rows_per_user, self.replay_capacity, short_context=False)
        self.replay = packed.take()
        self.replay_tables = [list(tables) for tables in packed.replay_pages[self.replay_capacity]]
        self.replay_addresses = [[addresses(operations, table) for table in tables] for tables in self.replay_tables]
        self.operations, self.model, self.mesh, self.sampler = operations, model, model.mesh_device, sampler
        self.helpers = helpers
        self.name = 'PackedVerifierEngine@%x users=%d rows=%d' % (id(self), shape.users, shape.rows_per_user)
        self.carries = []
        for segment, index in enumerate(pool_slots):
            carry = pool.slots[index].verifier.carry
            if len(carry) != GDN_LAYERS or any(len(snapshot) != len(helper.live)
                                                for snapshot, helper in zip(carry, helpers, strict=True)):
                raise ValueError('Pooled carry %d must hold one slot-zero snapshot per GDN helper' % segment)
            self.carries.append([list(snapshot) for snapshot in carry])
        self.native_addresses = [[addresses(operations, value) for value in helper.live] for helper in helpers]
        self.carry_addresses = self.slot_addresses()
        self.initial, self.checkpoints, self.taps, self.owned = [], [], [], []
        self.feature_capture = self.fixture = None
        self.trace = self.output = None
        self.commits = [{} for user in range(shape.users)]
        self.phase, self.first, self.rounds = 'preparing', True, 0
        self.pending_segments = set()
        # QWEN_FAST_PIPELINED_COMMITS=1: each user's commit trace is enqueued
        # (blocking=False) instead of replayed one at a time; read once here rather than
        # per call, since a round's four commits must agree on one mode (see execute_commit,
        # commit_user). commit_timings is this round's per-segment device/enqueue time,
        # indexed by segment, printed as one line under QWEN_FAST_PACKED_AUDIT=1.
        self.pipelined_commits = os.environ.get('QWEN_FAST_PIPELINED_COMMITS') == '1'
        self.commit_timings = [0.0] * shape.users
        # This round's per-segment HOST cost of the RetainedGDNBlock.commit_user call
        # itself (gdn_records.py), beyond its device commit trace: call_ms - commit_ms,
        # dominated by validate_bindings' native-buffer address re-checks. Always
        # collected (a handful of perf_counter calls and float subtractions - the same
        # cost class as commit_timings above, already collected unconditionally),
        # printed only under QWEN_FAST_PACKED_AUDIT=1 by serving_packed_step.py's
        # run_verified_block, alongside commit_entry's own adopt/session/other split, as
        # '[PACKED-COMMIT-HOST]'.
        self.commit_block_ms = [0.0] * shape.users
        self.stage = 'allocating'
        started = time.perf_counter()
        try:
            # Allocated before ANY capture, like everything a trace may see. The initial
            # snapshot is slot 0 as attach found it, put back once the captures are done.
            self.initial = [helper.allocate() for helper in helpers]
            for helper, snapshot in zip(helpers, self.initial, strict=True):
                helper.save(snapshot)
            # The pack's per-user checkpoints: demanded by verifier_pack and validate_pack,
            # never written by the deferred decode (its decisions go to the carries).
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
            self.stage = 'warm forward'
            warm = self.build_fixture(placeholders)
            result = None
            try:
                result = self.operation(warm)
                self.stage = 'warm forward fence'
                operations.synchronize_device(self.mesh)
            finally:
                if result is not None:
                    release_owned(operations, [value for value in result if value is not None])
                warm.close()
            self.stage = 'verify trace capture'
            self.fixture = self.build_fixture(placeholders)
            self.trace, self.output = capture_operation(operations, self.mesh, lambda: self.operation(self.fixture))
            retained = self.fixture.retained
            if len(retained.records) != GDN_LAYERS:
                raise ValueError('The captured packed block must retain every GDN layer')
            self.stage = 'commit trace capture'
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
            # Warming the commit traces wrote every carry and slot 0 (as verifier_engine's
            # own warming does): slot 0 goes back to what attach found, and the carries go
            # back to the zeros the pool lends - a request's engine seeds its own on
            # admission (VerifierEngine.save_carry), after the pool zeroes the slot again.
            self.stage = 'reseeding'
            self.reseed()
            operations.synchronize_device(self.mesh)
            self.stage = 'validating bindings'
            self.validate_bindings()
            # The captures rewrote slot 0: nobody is resident.
            note_prefill()
            self.setup_ms = (time.perf_counter() - started) * 1000
            self.phase = 'idle'
        except BaseException as failure:
            self.phase = 'failed'
            # Logged BEFORE close: run 35505708710 (image v50) raised on the host inside the
            # 64-row warm forward after the device had hung on an enqueued op; close then
            # blocked in the device fence for the rest of the timeout and the cause was
            # never logged (serving_runtime's own [PINDIAG] line comes after this close).
            self.report_failure(failure)
            self.close(wait=False)
            raise

    def report_failure(self, failure):
        """The construction failure and its traceback into the log, so the next run's log
        shows the cause even when the device never completes the work already enqueued."""
        try:
            summary = '%s: %s' % (type(failure).__name__, str(failure)[:300])
            diagnostic('[PINDIAG] packed block warm failed with %s; stage %s; closing without the device fence'
                       % (summary, self.stage))
            diagnostic('[PINDIAG] packed block failure traceback\n'
                       + ''.join(traceback.format_exception(type(failure), failure, failure.__traceback__)))
        except BaseException:
            pass

    def build_fixture(self, placeholders):
        """The packed ModelBatch: retained records for the per-user commits; commit-only GDN,
        which packed is the DEFERRED decode (gdn_device_loop_state: one block-start entry per
        user, nothing restored or advanced at decode, every decision made by commit_user
        after the readback); replay attention as one bundled reader PER USER over the
        pool's lent table sets (M1b; four-row groups, masks refreshed in-trace per layer as
        the sequential verify does); and no T16 gate (it pins rows == 16)."""
        pack = build_pack([participant(placeholder, self.rows_per_user, 0, self.checkpoints[user], self.carries[user])
                           for user, placeholder in enumerate(placeholders)], block_rows=self.block_rows)
        return ModelBatch(self.model, [1] * self.block_rows, self.capture_position, placeholders[0].pages, self.helpers,
            self.checkpoints[0], self.block_rows, pack=pack, packed_replay_pages=self.replay_tables,
            serial_sdpa=True, compact_gdn=True, reuse_gdn_input=True,
            skip_row_clones=True, hoist_row_layout=True, device_loop_gdn=True, compact_prologue=True,
            batch_conv=True, packed_checkpoints=True, retain_records=True, ordered_cache=True,
            norm_batch=True, attention_replay=True, attention_mask_once=False, replay_group_rows=4,
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

    def reseed(self):
        """Undo what warming the commit traces wrote: slot 0 from the initial snapshot,
        every carry to zero."""
        if self.slot_addresses() != self.carry_addresses:
            raise ValueError('A carried GDN state moved under the packed block')
        for helper, snapshot in zip(self.helpers, self.initial, strict=True):
            helper.restore(snapshot)
        for carry in self.carries:
            for snapshot in carry:
                for value in snapshot:
                    self.operations.full_like(value, 0.0, optional_tensor=value)

    def slot_addresses(self):
        return [[[addresses(self.operations, value) for value in snapshot] for snapshot in carry] for carry in self.carries]

    def validate_bindings(self):
        if any(helper.gdn.B != 8 or not helper.gdn._stable_state for helper in self.helpers):
            raise ValueError('Native stable B8 state contract changed')
        if [[addresses(self.operations, value) for value in helper.live] for helper in self.helpers] != self.native_addresses:
            raise ValueError('Native GDN buffers changed under the packed block')
        if self.slot_addresses() != self.carry_addresses:
            raise ValueError('A carried GDN state moved under the packed block')
        if [[addresses(self.operations, table) for table in tables] for tables in self.replay_tables] != self.replay_addresses:
            raise ValueError('A pooled replay page table moved under the packed block')

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
            # Each user's reader is captured in the block's one family; a ticket outside it
            # (never under the serving pin: position 32768, budget 256) has no trace here.
            try:
                validate_ticket(ticket.position, self.rows_per_user, self.replay_capacity, short_context=False)
            except ValueError:
                raise ValueError("Packed ticket of request %s at %d leaves the block's native chunk family [%d, %d)"
                                 % (str(entry['request_id'])[:48], ticket.position, self.replay_capacity - 256,
                                    self.replay_capacity)) from None
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
            self.commit_timings = [0.0] * self.users
            self.commit_block_ms = [0.0] * self.users
            self.rounds += 1
            self.phase = 'verified'
            dump_device_profiler_after_round(self.operations, self.mesh, self.rounds)
            metrics = dict(segments=segments, staged_buffers=staged,
                binding_validation_ms=(started - binding_started) * 1000,
                input_ms=(staged_at - started) * 1000, verify_readback_ms=(finished - staged_at) * 1000,
                blocking_trace_host_ms=trace_ms, replay_checks_sync_ms=(replayed - staged_at) * 1000 - trace_ms,
                output_readback_host_ms=(finished - replayed) * 1000, users=self.users)
            # Under QWEN_FAST_PACKED_AUDIT=1: the round's host-visible phase split, so a slow
            # round's log says whether the cost sits in staging, the blocking trace replay
            # (attention, GDN, MLP, norm and the LM head are one opaque number here - a
            # captured trace's Python call graph runs once at attach, not per round; this
            # cannot attribute time inside it) or readback. Pure logging: the returned dict
            # is unchanged in keys or values.
            if os.environ.get('QWEN_FAST_PACKED_AUDIT') == '1':
                diagnostic('[PACKED-PHASE] round=%d users=%d bind_ms=%.2f input_ms=%.2f trace_ms=%.2f '
                           'sync_ms=%.2f readback_ms=%.2f'
                           % (self.rounds, self.users, metrics['binding_validation_ms'], metrics['input_ms'],
                              metrics['blocking_trace_host_ms'], metrics['replay_checks_sync_ms'],
                              metrics['output_readback_host_ms']))
            return predictions, metrics
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
        """This user's own prefix trace. Blocking (the default), this call IS the device
        replay and the elapsed time is the trace time (commit_trace_ms). Under
        QWEN_FAST_PIPELINED_COMMITS=1 (self.pipelined_commits) the trace is enqueued
        instead (blocking=False) and the elapsed time is only the host-side submission
        (commit_enqueue_ms) - the real device time for all four users' traces is folded
        into the one trailing fence commit_user measures as sync_ms. Returns the elapsed
        ms, or None when prefix 0 ran no trace (the caller's `mode=` audit field, not this
        return value, says which of the two the number means)."""
        if not prefix:
            return None
        blocking = not self.pipelined_commits
        started = time.perf_counter()
        self.operations.execute_trace(self.mesh, self.commits[segment][prefix], cq_id=0, blocking=blocking)
        return (time.perf_counter() - started) * 1000

    def commit_user(self, segment, prefix):
        """Commit one user's decision: its own prefix trace writes the accepted state into
        native slot 0 and into that user's carry; prefix 0 runs nothing.

        Blocking (the default), each commit trace already fences the host, so the extra
        `synchronize=last` fence below costs nothing extra. Under
        QWEN_FAST_PIPELINED_COMMITS=1, execute_commit enqueues this segment's trace without
        blocking; RetainedGDNBlock.commit_user (gdn_records.py) calls its own
        `self.operations.synchronize_device(mesh)` right after the publication callback
        returns, WHEN `synchronize=True` - i.e. on the round's LAST segment here - which
        fences the whole command queue (cq_id=0) that all four of this round's enqueued
        commit traces were submitted to, in this same per-segment call order. So by the
        time `self.fixture.retained.commit_user(...)` returns below, every enqueued commit
        trace of the round has completed - no separate trailing sync is added here; the
        existing `synchronize=last` plumbing already provides it."""
        self.check_segment(segment)
        if type(prefix) is not int or not 0 <= prefix <= self.rows_per_user:
            raise ValueError('Selected prefix outside the user segment')
        # Each commit trace already blocks the host; the one fence that matters is the
        # last user's, which arms the retained block's replay for the next round.
        last = self.pending_segments == {segment}
        commit_ms = 0.0

        def publish(selected):
            nonlocal commit_ms
            commit_ms = self.execute_commit(segment, selected) or 0.0

        try:
            call_started = time.perf_counter()
            self.fixture.retained.commit_user(segment, prefix, dma=True, synchronize=last, publication=publish)
            call_ms = (time.perf_counter() - call_started) * 1000
            self.commit_timings[segment] = commit_ms
            # This segment's own call cost beyond its device commit trace: bookkeeping
            # plus RetainedGDNBlock.commit_user's validate_bindings (gdn_records.py) and,
            # on the round's last segment, the trailing fence (see sync_ms below, the
            # same quantity for that one segment). Collected for every segment, not just
            # the last, so serving_packed_step.py's per-user [PACKED-COMMIT-HOST] line
            # can attribute this round's host time to validate_bindings specifically.
            self.commit_block_ms[segment] = max(call_ms - commit_ms, 0.0)
            self.pending_segments.discard(segment)
            if not self.pending_segments:
                self.phase = 'idle'
                if os.environ.get('QWEN_FAST_PACKED_AUDIT') == '1':
                    # sync_ms is what the LAST segment's own call spent beyond its own
                    # commit_ms: bookkeeping plus (when pipelined) the trailing fence that
                    # drains the whole round's four enqueued traces; near zero when blocking,
                    # since the trace replay above already drained the queue.
                    sync_ms = self.commit_block_ms[segment]
                    diagnostic('[PACKED-COMMIT] round=%d mode=%s commit_ms=[%s] sync_ms=%.2f'
                               % (self.rounds, 'pipelined' if self.pipelined_commits else 'blocking',
                                  ','.join('%.2f' % value for value in self.commit_timings), sync_ms))
        except BaseException:
            self.phase = 'failed'
            raise
        finally:
            # Slot 0 holds this segment's user, or after prefix 0 nobody's committed state.
            note_packed_step()

    def describe(self):
        operations = self.operations
        return dict(name='packed', shape=self.shape._asdict(), capture_position=self.capture_position,
            pool_slots=list(self.pool_slots),
            weight_passes_per_round='one for every user', batched=True,
            verify_traces=1 if self.trace is not None else 0,
            commit_traces=sum(len(commits) for commits in self.commits),
            taps=[list(addresses(operations, tap)) for tap in self.taps],
            checkpoints=[list(addresses(operations, checkpoints[0][0])) for checkpoints in self.checkpoints],
            carries=[list(carry[0][0]) for carry in self.carry_addresses],
            attention=dict(reader='per-user bundled replay', family=self.replay_capacity, replay_group_rows=4,
                           bundles_per_user=[len(tables) for tables in self.replay_tables],
                           tables=[[list(address) for address in tables] for tables in self.replay_addresses]),
            setup_ms=getattr(self, 'setup_ms', None), rounds=self.rounds)

    def close(self, *, wait=True):
        """Release the block. `wait=False` is the failed construction: the device may still be
        running, or hung on, the work the failed forward enqueued, so nothing that can block
        is done - no device fence, and the traces (if any were captured) are abandoned rather
        than released, since a trace release may fence. Everything else is released as
        usual: the frees are host-side allocator bookkeeping (run 35505708710 freed the warm
        fixture's buffers with the device hung and returned), the pool's tables are handed
        back, and a process whose attach failed does not allocate again, so a freed address
        cannot be re-issued under a kernel that is still running. Every other close keeps
        the fence: an idle block has nothing in flight and returns at once; a block failed
        in verify or commit was fenced by its blocking trace or its staging before it
        failed."""
        if self.phase == 'closed':
            return
        if self.phase not in ('idle', 'preparing', 'failed'):
            raise ValueError('Commit every segment of the pending packed block before closing')
        operations = self.operations
        if wait:
            operations.synchronize_device(self.mesh)
        abandoned = sum(len(commits) for commits in self.commits) + (self.trace is not None)
        for commits in self.commits:
            if wait:
                for trace in commits.values():
                    operations.release_trace(self.mesh, trace)
            commits.clear()
        if self.trace is not None:
            if wait:
                operations.release_trace(self.mesh, self.trace)
            self.trace = None
        if not wait:
            diagnostic('[PINDIAG] packed block closed without the device fence at stage %s; %d captured trace(s) abandoned'
                       % (self.stage, abandoned))
        if self.feature_capture is not None:
            self.feature_capture.close()
        if self.output is not None:
            release_owned(operations, [value for value in self.output if value is not None])
            self.output = None
        if self.fixture is not None:
            self.fixture.close()
            self.fixture = None
        # The readers' page tables are the pool's: handed back, never freed here.
        if getattr(self, 'replay', None) is not None:
            self.replay.release()
            self.replay = None
        self.replay_tables, self.replay_addresses = [], []
        release_owned(operations, self.taps)
        release_owned(operations, [value for checkpoints in self.checkpoints for snapshot in checkpoints for value in snapshot])
        release_owned(operations, [value for snapshot in self.initial for value in snapshot])
        self.taps, self.checkpoints, self.initial = [], [], []
        # The carries are the pool's: lent to the requests, never freed here.
        self.carries, self.carry_addresses = [], []
        self.pending_segments.clear()
        self.phase = 'closed'
