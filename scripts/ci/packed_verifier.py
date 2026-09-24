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
Under QWEN_FAST_VERIFY_T1=1 the batched recurrence reads every carry in place and the
trace writes neither the entries nor slot 0 (verify_trace_t1, cut #3).

VARIABLE-USER ROUNDS (M2, QWEN_FAST_PADDED_BLOCK=1, default off; admitted by serving_runtime at
the 64-row M3 block only, which passes `padded_min_users`). A round with padded_min_users <= n
< users live entries is served by the SAME captured trace: each live entry in its own segment
as always, every other segment staged idle (`idle_inputs`: tokens 1 at the family start, an
all-zero page table, so its K/V lands in physical page 0 - vLLM's null block - on a tile row of
its own). Run v188 (R1, image P1) replayed live patterns {0,1}, {1,3} and {0,1,2} this way and
found the live rows bit-identical to the all-live replay on both chips, every idle carry intact
and 140.2 ms per replay. Page 0 holds two 32-row tile rows, so at most two segments are idle
(MAX_IDLE_SEGMENTS) and padded_min_users is at least users - 2. What an idle segment must never
do is commit: after the readback it decides at prefix 0 (`commit_user(idle, 0)`), which writes
no state, no carry and no K/V - it only gives the retained block the decision every segment
owes before the next replay - and it runs before any live commit, so the round's last LIVE
commit keeps the fence. Its pool slot must be unlent, or the trace must read every carry in
place (QWEN_FAST_VERIFY_T1 #3, both halves: `carries_in_place`), in which case the trace writes
neither that carry nor native slot 0 and a request admitted into the slot meanwhile is safe.
No live table may hold page 0 inside the range it reads and writes (`padded_refusal`, fail
closed). serving_packed_step decides the round (proposal_rows, ineligible, kv_guard); `verify`
repeats the checks before anything is staged, as the backstop. With the flag off every path
here is the one that ran before it existed.
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
import verify_trace_t1
import verify_trace_t2

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
    if getattr(fixture, 'kv_chains', False):
        # QWEN_FAST_VERIFY_T2 (#2): the chained K/V write is exact only while no two users write
        # one (page, tile row). serving_packed_step.proposal_rows drafts such a round for the
        # sequential step and ineligible refuses one first seen at the step, so neither reaches
        # here; this is the fail-closed backstop, before any copy.
        guard = verify_trace_t2.block_users(positions, pages, shape.rows_per_user, shape.users)
        conflict = verify_trace_t2.kv_conflict(guard)
        if conflict is not None:
            verify_trace_t2.log_line('%s site=stage_packed %s' % (verify_trace_t2.KV_SHARED,
                                                                  verify_trace_t2.kv_conflict_reason(conflict)))
            raise ValueError('The chained K/V write needs disjoint cache tile rows: %s'
                             % verify_trace_t2.kv_conflict_reason(conflict))
        if verify_trace_t2.audit_enabled():
            verify_trace_t2.log_line('%s kv_rows_per_user=%s' % (verify_trace_t2.AUDIT_MARKER, ','.join(
                str(len(verify_trace_t2.kv_tile_rows(user_positions, table))) for user_positions, table in guard)))
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


REPLAY_GROUP_ROWS_FLAG = 'QWEN_FAST_REPLAY_GROUP_ROWS'


def replay_group_rows(environ=None):
    """The opt-in flag for the packed block's per-user replay reader group-row width
    (model_batch.ModelBatch's replay_group_rows, which accepts only 4 or 8; 8 requires
    attention_replay, already True for this fixture). Default '4' is today's grouping,
    byte-identical; anything but '4' or '8' is a configuration error rather than a silent
    fallback, the same pattern as gdn_user_batch.enabled."""
    value = (os.environ if environ is None else environ).get(REPLAY_GROUP_ROWS_FLAG, '4')
    if value not in ('4', '8'):
        raise ValueError('%s must be 4 or 8' % REPLAY_GROUP_ROWS_FLAG)
    return int(value)


PADDED_BLOCK_FLAG = 'QWEN_FAST_PADDED_BLOCK'
PADDED_MIN_USERS_FLAG = 'QWEN_FAST_PADDED_BLOCK_MIN_USERS'
PADDED_MIN_USERS_DEFAULT = 2
# The lines the variable-user M2 path logs (lever_n_m3native_gate reads every one): the block's
# admission once at attach; one line per padded round, and one per round of padded_min_users..
# users-1 live users the block did NOT serve padded (serving_packed_step.note_padded_skip, with
# whether it was eligible); a refusal by the idle slot rule or the count, a page-0 refusal, and
# an idle segment asked to commit a prefix (never: the block refuses it).
PADDED_ADMITTED_MARKER = '[PINDIAG] packed padded block admitted'
PADDED_ROUND_MARKER = '[PINDIAG] packed padded round'
PADDED_SKIPPED_MARKER = '[PINDIAG] packed padded skipped'
PADDED_REFUSED_MARKER = '[PINDIAG] packed padded refused'
PADDED_PAGE0_MARKER = '[PINDIAG] packed padded page0'
PADDED_IDLE_COMMIT_MARKER = '[PINDIAG] packed padded idle commit'


def padded_block_min_users(environ=None):
    """QWEN_FAST_PADDED_BLOCK (variable-user packed rounds M2): None unless the flag is '1', else
    QWEN_FAST_PADDED_BLOCK_MIN_USERS as an int (default 2), which the block then checks against
    its own users. A flag value other than '0' or '1', or a minimum that is not a decimal integer,
    is a configuration error rather than a silent fallback (the gdn_user_batch.enabled pattern).
    The minimum is not read at all while the flag is off."""
    environ = os.environ if environ is None else environ
    value = environ.get(PADDED_BLOCK_FLAG, '0')
    if value not in ('0', '1'):
        raise ValueError('%s must be 0 or 1' % PADDED_BLOCK_FLAG)
    if value != '1':
        return None
    text = environ.get(PADDED_MIN_USERS_FLAG, str(PADDED_MIN_USERS_DEFAULT))
    if type(text) is not str or not text.isdigit() or text != str(int(text)):
        raise ValueError('%s must be a decimal integer' % PADDED_MIN_USERS_FLAG)
    return int(text)


def page_zero_index(table, start, rows):
    """The first index inside [0, (start + rows + 63) // 64) at which this (1, page_width) page
    table holds physical page 0, or None: every page a user at `start` reads or writes in a
    `rows`-row round (padded_probe.used_pages). Raises ValueError when the table cannot map that
    range, and whatever indexing raises for a table that is not one."""
    used = (int(start) + int(rows) + 63) // 64
    values = table[0][:used]
    values = values.tolist() if hasattr(values, 'tolist') else list(values)
    if len(values) < used:
        raise ValueError('a %d-entry page table cannot map positions [0, %d)' % (len(values), int(start) + int(rows)))
    for index, value in enumerate(values):
        if int(value) == 0:
            return index
    return None


class PackedVerifierEngine:
    """Owner of one packed verify block: build at attach, then per round
    `verify(entries)` -> per-entry predictions, `features(segment)` for each user's
    publication, `commit_user(segment, prefix)` for each user's decision."""

    def __init__(self, operations, model, helpers, sampler, *, pool, shared_weights, shape, feature_taps,
                 capture_position=None, pool_slots=None, padded_min_users=None):
        import torch

        self.shape = validate_shape(shape)
        self.rows_per_user, self.block_rows, self.users = shape.rows_per_user, shape.block_rows, shape.users
        # QWEN_FAST_PADDED_BLOCK (variable-user rounds M2): the fewest live users a round of this
        # block may serve, the rest of its segments idle; None, the default, serves exactly
        # `users`, as always. Passed by serving_runtime only where it admits the flag. At most
        # MAX_IDLE_SEGMENTS idle segments fit page 0, and at least one user must be live.
        if padded_min_users is not None and (
                type(padded_min_users) is not int
                or not max(1, shape.users - self.MAX_IDLE_SEGMENTS) <= padded_min_users < shape.users):
            raise ValueError('padded_min_users must be an integer in [%d, %d) for a %d-user block; got %r'
                             % (max(1, shape.users - self.MAX_IDLE_SEGMENTS), shape.users, shape.users,
                                padded_min_users))
        self.padded_min_users = padded_min_users
        self.padded_rounds = 0
        self.idle_segments = frozenset()
        # Whether the captured trace reads every carry in place (QWEN_FAST_VERIFY_T1 #3, both
        # halves, in every GDN layer): set from the capture's own counts (note_verify_t1).
        self.carries_in_place = False
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
        # Each segment's pool slot itself, for the padded idle slot rule (padded_refusal):
        # whether a request holds it now.
        self.segment_slots = tuple(pool.slots[index] for index in pool_slots)
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
        # QWEN_FAST_REPLAY_GROUP_ROWS: read once here, like pipelined_commits above, and
        # stored for the block's whole life so build_fixture (the trace capture) and
        # describe() (the diagnostic dict) always agree on the value actually in use.
        self.replay_group_rows = replay_group_rows()
        # QWEN_FAST_VERIFY_T1 (verify_trace_t1): read once here, like the flags above, so the
        # capture and every round's readback agree. #10 shares each reader's mask across the
        # forward (build_fixture); #8a samples each chip's vocab shard and combines on the host
        # (operation, shard_predictions) whenever the pinned sampler is plain greedy argmax.
        # QWEN_FAST_VERIFY_T1_SKIP leaves either out (verify_trace_t1.cut).
        self.verify_t1 = verify_trace_t1.enabled()
        self.mask_once = verify_trace_t1.cut('mask_once')
        shard_cut = verify_trace_t1.cut('shard_argmax')
        self.shard_problem = verify_trace_t1.shard_sampling_problem(sampler) if shard_cut else None
        self.shard_argmax = shard_cut and self.shard_problem is None
        self.shard_audit = self.shard_argmax and verify_trace_t1.audit_enabled()
        # QWEN_FAST_VERIFY_T2 (verify_trace_t2): read once here too. #2 (kv_chains) makes the
        # warm fixture's K/V write one chain (build_fixture); the fixture decides the writer
        # (model_batch) and this block reads what it became (kv_chains, warm_kv_chains). The
        # windows audit runs only if the capture engaged #1.
        self.verify_t2 = verify_trace_t2.enabled()
        self.kv_chains_cut = verify_trace_t2.cut('kv_chains')
        self.warm_kv_chains = False
        self.windows_audit = False
        # QWEN_FAST_PADDED_PROBE (variable-user packed rounds M1, padded_probe.py): read once
        # here, like the flags above. Off, padded_probe is never imported and verify() is
        # today's.
        self.padded_probe = os.environ.get('QWEN_FAST_PADDED_PROBE') == '1'
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
            # QWEN_FAST_VERIFY_T2 (#2): the placeholders put every user on page 0's first tile
            # row, which per-user chains would write concurrently; the warm forward - the one
            # eager forward with placeholders - writes K/V as ONE chain over all rows instead:
            # the same kernels and compile args (so the capture's per-user chains are the same
            # programs), the served row order and the served page-0 bytes. The capture is
            # recorded, never run, with placeholders.
            warm = self.build_fixture(placeholders, warm=True)
            self.warm_kv_chains = self.kv_chains_cut and bool(getattr(warm, 'kv_chains', False))
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
            if self.verify_t1:
                verify_trace_t1.take()  # count only what the captured forward engages
            if self.verify_t2:
                verify_trace_t2.take()
            self.trace, self.output = capture_operation(operations, self.mesh, lambda: self.operation(self.fixture))
            if self.verify_t1:
                self.note_verify_t1(verify_trace_t1.take())
            if self.verify_t2:
                self.note_verify_t2(verify_trace_t2.take())
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
            if self.padded_min_users is not None:
                diagnostic('%s min_users=%d users=%d max_idle=%d carries_in_place=%d'
                           % (PADDED_ADMITTED_MARKER, self.padded_min_users, self.users, self.MAX_IDLE_SEGMENTS,
                              int(self.carries_in_place)))
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

    def build_fixture(self, placeholders, warm=False):
        """The packed ModelBatch: retained records for the per-user commits; commit-only GDN,
        which packed is the DEFERRED decode (gdn_device_loop_state: one block-start entry per
        user, nothing restored or advanced at decode, every decision made by commit_user
        after the readback); replay attention as one bundled reader PER USER over the
        pool's lent table sets (M1b; four-row groups, masks refreshed in-trace per layer as
        the sequential verify does - or, under QWEN_FAST_VERIFY_T1 (#10), once per forward:
        each mask is a function of its reader's positions word alone, staged before the
        replay and written by nothing inside it); and no T16 gate (it pins rows == 16)."""
        pack = build_pack([participant(placeholder, self.rows_per_user, 0, self.checkpoints[user], self.carries[user])
                           for user, placeholder in enumerate(placeholders)], block_rows=self.block_rows)
        return ModelBatch(self.model, [1] * self.block_rows, self.capture_position, placeholders[0].pages, self.helpers,
            self.checkpoints[0], self.block_rows, pack=pack, packed_replay_pages=self.replay_tables,
            serial_sdpa=True, compact_gdn=True, reuse_gdn_input=True,
            skip_row_clones=True, hoist_row_layout=True, device_loop_gdn=True, compact_prologue=True,
            batch_conv=True, packed_checkpoints=True, retain_records=True, ordered_cache=True,
            norm_batch=True, attention_replay=True, attention_mask_once=self.mask_once,
            replay_group_rows=self.replay_group_rows,
            short_context=False, attention_audit=False, commit_only_gdn=True,
            **({'kv_single_chain': True} if warm and self.kv_chains_cut else {}))

    def operation(self, fixture):
        logits = None
        try:
            with ExitStack() as captures:
                captures.enter_context(self.feature_capture.capture())
                logits = fixture.run(sharded_logits=True)
            if self.shard_argmax:
                return (logits, *self.sample_shards(logits))
            ids = sample_rows(self.sampler, logits, self.block_rows, self.operations, native_rows=False)
            return logits, ids
        except BaseException:
            if logits is not None:
                self.operations.deallocate(logits)
            raise

    def sample_shards(self, logits):
        """QWEN_FAST_VERIFY_T1 (#8a): each chip's argmax and max over its own vocab shard, no
        gather; under QWEN_FAST_VERIFY_T1_AUDIT also the pinned sampler's ids, for the audit."""
        ids, values = verify_trace_t1.sample_shards(self.operations, logits, self.block_rows)
        if not self.shard_audit:
            return ids, values
        try:
            reference = sample_rows(self.sampler, logits, self.block_rows, self.operations, native_rows=False)
        except BaseException:
            release_owned(self.operations, [ids, values])
            raise
        return ids, values, reference

    def shard_predictions(self):
        """The block's ids from the per-chip (id, max) pairs: shard 1 only where its max is
        strictly greater (verify_trace_t1.combine_shards). Audited, every row is compared with
        the pinned sampler's id from the same replay."""
        ids, values = self.output[1], self.output[2]
        id_parts = self.operations.get_device_tensors(ids)
        value_parts = self.operations.get_device_tensors(values)
        if len(id_parts) != 2 or len(value_parts) != 2:
            raise AssertionError('Two chip-local outputs required')
        chip_ids = [self.operations.to_torch(part).reshape(-1)[:self.block_rows] for part in id_parts]
        chip_values = [self.operations.to_torch(part).reshape(-1)[:self.block_rows] for part in value_parts]
        if any(len(value) != self.block_rows for value in (*chip_ids, *chip_values)):
            raise AssertionError('Missing packed prediction rows')
        host = verify_trace_t1.combine_shards(chip_ids, chip_values).tolist()
        if self.shard_audit:
            reference = self.operations.to_torch(self.operations.get_device_tensors(self.output[3])[0])
            verify_trace_t1.audit_round(host, reference.reshape(-1)[:self.block_rows].tolist())
        return host

    def note_verify_t1(self, counts):
        """VERIFY_T1_MARKER once per captured verify trace: which T1 cuts the capture engaged."""
        # #3 in every GDN layer, both halves: the trace writes no carry and not native slot 0
        # (the padded idle slot rule, padded_refusal).
        self.carries_in_place = (counts.get('direct_carry') == GDN_LAYERS and counts.get('last_carry') == GDN_LAYERS)
        fields = dict(mask_once=int(bool(getattr(self.fixture, 'attention_mask_once', False))),
                      shard_argmax=int(self.shard_argmax), audit=int(self.shard_audit),
                      direct_carry=counts.get('direct_carry', 0), last_carry=counts.get('last_carry', 0),
                      coalesced=counts.get('coalesced', 0),
                      coalesce_fallback=counts.get('coalesce_fallback', 0))
        diagnostic(verify_trace_t1.engaged_line('packed_verify', **fields))
        if self.shard_problem is not None:
            diagnostic('%s: %s' % (verify_trace_t1.KEPT_SAMPLER, self.shard_problem))

    def note_verify_t2(self, counts):
        """verify_trace_t2.MARKER once per captured verify trace: what the captured forward
        engaged (#1 per GDN layer, #2 per K/V write), what the capture fixture was built with, and
        whether the warm forward wrote K/V as one chain (warm_chain=single) or served (none)."""
        fixture = self.fixture
        self.windows_audit = verify_trace_t2.audit_enabled() and counts.get('windows', 0) > 0
        diagnostic(verify_trace_t2.engaged_line(
            'packed_verify', windows=counts.get('windows', 0), windows_fallback=counts.get('windows_fallback', 0),
            kv_chains=counts.get('kv_chains', 0), kv_fallback=getattr(fixture, 'kv_fallback', 0),
            kv_rows=getattr(fixture, 'kv_rows', 0),
            warm_chain='single' if self.warm_kv_chains else 'none', audit=int(self.windows_audit)))

    @property
    def kv_chains(self):
        """Whether the captured fixture writes K/V through per-user chains (verify_trace_t2 #2):
        serving_packed_step.proposal_rows (before drafting) and ineligible (at the step) then
        check every round's tile rows before the verify."""
        return bool(getattr(self.fixture, 'kv_chains', False))

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

    def pads(self, count):
        """Whether a round of `count` live users is one this block serves padded
        (QWEN_FAST_PADDED_BLOCK): padded_min_users <= count < users. Always False without it."""
        return self.padded_min_users is not None and self.padded_min_users <= count < self.users

    def segments(self, entries):
        """One segment per entry, in entries order; every segment exactly once - or, for a
        padded round (`pads`), at most once, the segments no entry holds idle."""
        entries = list(entries)
        if len(entries) != self.users and not self.pads(len(entries)):
            if self.padded_min_users is not None:
                raise ValueError('The padded packed block serves %d to %d users; %d entries given'
                                 % (self.padded_min_users, self.users, len(entries)))
            raise ValueError('The packed block serves exactly %d users; %d entries given' % (self.users, len(entries)))
        segments = tuple(self.segment_of(entry['request'].engine) for entry in entries)
        if len(entries) == self.users:
            if sorted(segments) != list(range(self.users)):
                raise ValueError('Two packed entries were admitted through the same pool slot')
        elif len(set(segments)) != len(segments):
            raise ValueError('Two packed entries were admitted through the same pool slot')
        return segments

    def segment_users(self, entries, segments):
        """Each entry's (tokens, start, pages) at its segment: what stage_packed stages. A
        segment no entry holds (a padded round's idle one) is None."""
        users = [None] * self.users
        for entry, segment in zip(entries, segments):
            ticket, engine = entry['ticket'], entry['request'].engine
            users[segment] = (tuple(ticket.tokens), ticket.position, engine.pages)
        return users

    def padded_users(self, users, live_segments):
        """`users` (segment_users) with every idle segment filled from idle_inputs."""
        fills = self.idle_inputs(live_segments)
        return [fills[segment] if user is None else user for segment, user in enumerate(users)]

    def stage_packed_inputs(self, entries):
        entries = list(entries)
        segments = self.segments(entries)
        users = self.segment_users(entries, segments)
        if len(segments) < self.users:
            # A padded round (QWEN_FAST_PADDED_BLOCK): every segment no entry holds is staged idle.
            users = self.padded_users(users, segments)
        return stage_packed(self.operations, self.model, self.fixture, self.shape, users)

    def padded_refusal(self, users):
        """Why these users - in segment order, (tokens, start, table) for each live segment and
        None for each idle one - cannot be served as one padded round, or None. Host only, and
        nothing is staged. The rules (module docstring, VARIABLE-USER ROUNDS):
          - the count: `pads(live)`;
          - the idle slot rule: each idle segment's pool slot is unlent, or the captured trace
            reads every carry in place (`carries_in_place`, VERIFY_T1 #3 in every layer);
          - page 0: no live table holds physical page 0 inside its used range [0, (start + rows
            + 63) // 64), which the idle segments write. Fails closed: a table that cannot map
            that range is a page-0 reason too.
        Every page-0 reason starts 'page0', the marker its callers log it under."""
        users = list(users)
        live = [segment for segment, user in enumerate(users) if user is not None]
        if len(users) != self.users or not self.pads(len(live)):
            return 'padded round of %d live users outside [%s, %d)' % (len(live), self.padded_min_users, self.users)
        if not self.carries_in_place:
            for segment, user in enumerate(users):
                if user is None and getattr(self.segment_slots[segment], 'lent', True):
                    return ('idle segment %d: pool slot %d is lent and the verify trace moves carries '
                            '(QWEN_FAST_VERIFY_T1 #3 not in every layer)' % (segment, self.pool_slots[segment]))
        for segment in live:
            tokens, start, table = users[segment]
            try:
                index = page_zero_index(table, start, self.rows_per_user)
            except (IndexError, TypeError, ValueError, AttributeError) as error:
                return 'page0 unmapped: segment %d position %s: %s' % (segment, start, error)
            if index is not None:
                return 'page0 in a live table: segment %d position %d page_index %d' % (segment, int(start), index)
        return None

    # Variable-user packed rounds (M1): the most idle segments one block can hold. An idle
    # segment writes its rows' K/V through an all-zero page table - physical page 0, vLLM's
    # null block, which the capture's placeholders already write - and page 0 has two 32-row
    # tile rows. Two idle segments sit on disjoint ones, as the T2 chained K/V write needs
    # (verify_trace_t2.kv_conflict: one writer per (page, tile row)); a third would share one.
    MAX_IDLE_SEGMENTS = 2

    def idle_inputs(self, live_segments):
        """{segment: (tokens, start, pages)} for every segment of the block NOT in
        `live_segments`, the j-th of them (in segment order) as tokens (1,) * rows_per_user
        from start F + 32 * (j % 2), F = replay_capacity - 256 (the start of the block's
        native chunk family, so every reader accepts it), through an all-zero (1, page_width)
        int32 table: page 0, tile row j % 2 - two idle segments never share a tile row. Host
        only; what stage_packed takes in an idle segment's place. Refuses live segments that
        are not distinct segments of this block, and a third idle segment."""
        import torch

        live = tuple(live_segments)
        if len(set(live)) != len(live) or any(type(segment) is not int or not 0 <= segment < self.users
                                              for segment in live):
            raise ValueError('Live segments must be distinct segments of this %d-user block: %r' % (self.users, live))
        idle = [segment for segment in range(self.users) if segment not in live]
        if len(idle) > self.MAX_IDLE_SEGMENTS:
            raise ValueError('At most %d idle segments: page 0 holds two 32-row tile rows, so idle segments %s '
                             'would share one' % (self.MAX_IDLE_SEGMENTS, idle))
        first = self.replay_capacity - 256
        return {segment: ((1,) * self.rows_per_user, first + 32 * (index % 2),
                          torch.zeros((1, self.shape.page_width), dtype=torch.int32))
                for index, segment in enumerate(idle)}

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
        idle = tuple(segment for segment in range(self.users) if segment not in segments)
        if idle:
            # A padded round (QWEN_FAST_PADDED_BLOCK): the fail-closed backstop behind
            # serving_packed_step's proposal_rows and ineligible - host only, before anything is
            # staged or claimed, so a refusal leaves the block idle.
            reason = self.padded_refusal(self.segment_users(entries, segments))
            if reason is not None:
                diagnostic('%s site=verify %s' % (PADDED_PAGE0_MARKER if reason.startswith('page0')
                                                  else PADDED_REFUSED_MARKER, reason))
                raise ValueError('The padded packed block cannot serve this round: %s' % reason)
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
            if self.shard_argmax:
                host = self.shard_predictions()
            else:
                logits, ids = self.output
                parts = self.operations.get_device_tensors(ids)
                if len(parts) != 2:
                    raise AssertionError('Two chip-local outputs required')
                host = self.operations.to_torch(parts[0]).reshape(-1)[:self.block_rows].tolist()
            if len(host) != self.block_rows:
                raise AssertionError('Missing packed prediction rows')
            if self.windows_audit:
                # QWEN_FAST_VERIFY_T2_AUDIT (G1 for #1): this replay's packed windows against the
                # served ones built beside them in the same trace - every GDN layer on round 1,
                # then two per round in rotation (verify_trace_t2.audit_layers).
                verify_trace_t2.audit_round(self.operations, self.fixture.retained.records, self.rounds + 1)
            predictions = [host[slice(*segment_rows(self.shape, segment))] for segment in segments]
            finished = time.perf_counter()
            if self.padded_probe:
                # QWEN_FAST_PADDED_PROBE (M1): after this round's readback, before its commit.
                # The predictions above are already on the host; the probe ends with this
                # round's own inputs restaged and replayed, and proves the replay bit-identical.
                import padded_probe

                padded_probe.after_readback(self, self.segment_users(entries, segments), self.rounds + 1)
            self.first = False
            self.pending_segments = set(segments) | set(idle)
            self.idle_segments = frozenset(idle)
            self.commit_timings = [0.0] * self.users
            self.commit_block_ms = [0.0] * self.users
            self.rounds += 1
            self.phase = 'verified'
            for segment in idle:
                # A padded round's idle segment decides at prefix 0: no commit trace exists for
                # 0 and the retained block publishes nothing at it - no state, no carry, no K/V -
                # so this only gives the retained block the decision each segment owes before
                # the next replay. Before any live commit, so the round's last LIVE commit keeps
                # the synchronize=last fence (commit_user: the one that empties the pending set).
                self.commit_user(segment, 0)
            dump_device_profiler_after_round(self.operations, self.mesh, self.rounds)
            metrics = dict(segments=segments, staged_buffers=staged,
                binding_validation_ms=(started - binding_started) * 1000,
                input_ms=(staged_at - started) * 1000, verify_readback_ms=(finished - staged_at) * 1000,
                blocking_trace_host_ms=trace_ms, replay_checks_sync_ms=(replayed - staged_at) * 1000 - trace_ms,
                output_readback_host_ms=(finished - replayed) * 1000, users=self.users)
            if self.padded_min_users is not None:
                # QWEN_FAST_PADDED_BLOCK only: the round's live and idle segments (every round, the
                # all-live ones included), into each user's phase-timing record.
                metrics.update(live=len(segments), idle=list(idle))
            if idle:
                self.padded_rounds += 1
                diagnostic('%s live=%d round=%d segments=%s idle=%s padded=%d'
                           % (PADDED_ROUND_MARKER, len(segments), self.rounds,
                              ','.join(str(segment) for segment in sorted(segments)),
                              ','.join(str(segment) for segment in idle), self.padded_rounds))
            # Under QWEN_FAST_PACKED_AUDIT=1: the round's host-visible phase split, so a slow
            # round's log says whether the cost sits in staging, the blocking trace replay
            # (attention, GDN, MLP, norm and the LM head are one opaque number here - a
            # captured trace's Python call graph runs once at attach, not per round; this
            # cannot attribute time inside it) or readback. Pure logging: the returned dict
            # is unchanged in keys or values.
            if os.environ.get('QWEN_FAST_PACKED_AUDIT') == '1':
                # Under QWEN_FAST_PADDED_BLOCK the live and idle segments close the line, after
                # every field the existing parsers read (lever_n_m3native_profile_report).
                diagnostic('[PACKED-PHASE] round=%d users=%d bind_ms=%.2f input_ms=%.2f trace_ms=%.2f '
                           'sync_ms=%.2f readback_ms=%.2f'
                           % (self.rounds, self.users, metrics['binding_validation_ms'], metrics['input_ms'],
                              metrics['blocking_trace_host_ms'], metrics['replay_checks_sync_ms'],
                              metrics['output_readback_host_ms'])
                           + ('' if self.padded_min_users is None else ' live=%d idle=%s' % (
                               len(segments), ','.join(str(segment) for segment in idle) or '-')))
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
        if self.idle_segments and segment in self.idle_segments and prefix != 0:
            # A padded round's idle segment commits nothing, ever (verify decides it at 0), and
            # an attempt is logged for the gate before anything else is checked.
            diagnostic('%s refused segment=%s prefix=%s round=%d' % (PADDED_IDLE_COMMIT_MARKER, segment, prefix,
                                                                      self.rounds))
            raise ValueError('Idle segment %s of a padded round commits nothing; prefix %r refused' % (segment, prefix))
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
            attention=dict(reader='per-user bundled replay', family=self.replay_capacity, replay_group_rows=self.replay_group_rows,
                           bundles_per_user=[len(tables) for tables in self.replay_tables],
                           tables=[[list(address) for address in tables] for tables in self.replay_addresses]),
            setup_ms=getattr(self, 'setup_ms', None), rounds=self.rounds,
            **({} if self.padded_min_users is None else dict(padded=dict(
                min_users=self.padded_min_users, max_idle=self.MAX_IDLE_SEGMENTS,
                carries_in_place=self.carries_in_place, rounds=self.padded_rounds))))

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
