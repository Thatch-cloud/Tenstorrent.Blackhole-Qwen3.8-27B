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

SEVERAL BLOCKS (QWEN_FAST_M3_BLOCKS=2, serving_runtime.complete_blocks_two_phase): the rule holds across blocks, not
only within one. Block B built after block A's capture would allocate its fixture inputs, taps, checkpoints and
extent words in the holes A's capture freed, and every replay of A would overwrite them. So a block built with
defer_capture=True only allocates (initial snapshot, checkpoints, taps) in its constructor; the caller then runs
`warm_and_fixture` of EVERY block (the warm forward, the captured fixture, the extent readers' words and masks),
then `capture_traces` of every block, then `finish_construction` of every block (publication warm, reseed, binding
check). Built whole (the default), the construction is those phases in that order inside one call, as it always was.

ONE KNOWN EXCEPTION (QWEN_FAST_TP4_RS_UNIT_MAJOR_AUDIT, a gate-only audit): its clones of the unit-major and split results are
persistent buffers allocated inside each block's capture, so the second block's can sit in holes the first block's capture
freed, and the first block's replay overwrites them. Served data is untouched. The audit is therefore read only right after
its own block's replay (tile_collective_tp.audit_replayed / audit_round, which refuses any other order): never overlap two
blocks' replays while it is on.

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

ROUND-FENCE PLAN H1a (verify_prestage.py; QWEN_FAST_PRESTAGE, _PRESTAGE_AUDIT and
QWEN_FAST_ROUND_FENCES, every one default off). stage_packed is split into packed_values (host)
and write_packed (copies); with the flags off it is the same calls in the same order. Under
QWEN_FAST_PRESTAGE the drafts' fence window pre-stages every input of the next verify but its
tokens (verify_prestage.BlockPrestage.prestage), and verify() - while the fixture write epoch
has not moved - skips validate_bindings and writes only the buffers whose recomputed value
differs, with no fence; anything else takes today's full path. Under QWEN_FAST_ROUND_FENCES the
retained block drops the fence and the second validate after the blocking trace (F3), the last
commit leaves its fence to the drafts' F9 or the next replay (F8), and the first commit skips
its validate when the round's replay ran one (validated_this_round).

ROUND-FENCE PLAN H1b (fused_commit.py; QWEN_FAST_FUSED_COMMIT and its _INPLACE, _LIVE_BANKS and
_AUDIT sub-flags, every one default off). The block owns the fused commit: each segment's RoPE
tables and K/V deltas are allocated right after the taps - before the warm forward and every
capture (R2) - and each segment's projection trace (T_proj) and, in place, its sixteen slide
traces are captured after the GDN commit traces, separate from them. serving_packed_step's
commit_entry installs the publication overrides (fused_commit.install_fused_commit) around each
user's commit; the verify, the readback and the GDN commits here are unchanged. Without the flag
`fused` is None, nothing is allocated or captured, and every path is today's.

ROUND-FENCE PLAN H2 (early_draft.py; QWEN_FAST_GDN_AFTER_PAIRS under QWEN_FAST_EARLY_DRAFT, default
off; refused without QWEN_FAST_ROUND_FENCES). A round the early draft armed (arm_deferred_commits)
decides every commit as today but execute_commit only records its (segment, prefix): flush_commits
enqueues the GDN commit traces, in decision order and today's blocking mode, once the next round's
pairs have been read back (or at the end of the early draft), inside the same execute_model, and
re-owes the retained block's fence (gdn_records.RetainedGDNBlock.note_deferred_publications) so the
next replay pays it. verify() flushes anything still held first (site=verify, R1 broken, logged).
Without the flags nothing is armed and every path is today's.

S2 C2-PACKED-ANY (design s2-design.md W3; QWEN_FAST_EXTENT_REPLAY=1 builds the pool with
extent_replay, serving_buffer_pool.PackedExtentStorage, and the block keys on that storage alone).
The block serves every live user at its OWN 256-key family E = (start // 256 + 1) * 256 through one
captured program (K64j flag 0x20): model_batch builds extent_attention_replay.PackedExtentReplayReader
over the pool's full-width tables and cur_pos words instead of the per-family readers, the capacity
is the whole table, C = page_width * 64, and:
  - `admits(position)` (128 <= start, start + rows <= C) replaces validate_ticket - in
    serving_packed_step before drafting and at the step, and in verify() as the raising backstop;
  - packed_values stages each extent reader's own values: its word (start & 255), each bundle's
    cur_pos (E - 1) and full-width table, from the one helper that computes word and cur_pos;
  - an idle segment is an ordinary user at start 0 or 32 on the zero table (E = 256), on page 0
    tile row 0 or 1 exactly as before; the page-0 rule also checks every page the extent reads,
    [0, E // 64) (design 2.6);
  - `accept_limit(position)` = min(rows, E - start): rows at or past E see all of [0, E) and never
    their own key, so the session commits at most that many (serving_packed_step.commit_entry,
    GreedySession.commit max_rows) and commit_user refuses more - before its try, so the refusal
    leaves the block verified - as the backstop (design 2.4);
  - one EXTENT_ROUND_MARKER line per packed round, from verify: the executed path's own proof;
  - QWEN_FAST_EXTENT_AUDIT=1 (gate profiles only): after each replay, every segment's word and
    cur_pos, and in rotation one segment's narrow masks and tables, read back from both chips
    against the host values; a mismatch is logged and the round restaged in full (audit_extent);
  - a replay deadline (QWEN_FAST_REPLAY_DEADLINE_S, default 30 s) around every verify and commit
    trace: a replay that overruns it logs the families it held and ends the process (exit 70)
    rather than wedging it (ReplayDeadline);
  - validate_bindings also checks the cur_pos words, and each reader's word and masks, unmoved;
  - QWEN_FAST_PACKED_CAPTURE_POSITION (serving_runtime, gate only) moves the capture position;
  - after the last capture, publication_warm.warm (B6): the drafter's eager publication - a refused
    fused commit's and every sequential step's - published and discarded on scratch at each segment's
    row offset x every prefix and each pooled width x its prefixes, so none compiles after attach.
Without the pool's extent storage none of this is built or read, and every path is today's.
"""

from contextlib import ExitStack, contextmanager, nullcontext
import os
import sys
import threading
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
from serving_fast_policy import PACKED_FAMILY_TOKENS
from verifier_engine import note_packed_step, note_prefill
import verifier_engine
from verifier_inputs import host_inputs, validate_tokens
from verifier_pack import GDN_LAYERS, build_pack, participant
import gdn_seq_block
import round_host
import verify_prestage
import capture_plug
import tp_shapes
import tile_collective_tp
import tp4_vglue
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


def audit_shard_values(operations, output, block_rows, chip_values):
    """QWEN_FAST_TP4_SHARD_VALUES under QWEN_FAST_TP4_VGLUE_AUDIT: each chip's gathered maxima against the ttnn.max taken beside
    them in the same trace (verify_trace_t1.shard_values), every row. Nothing to compare when the lever is off; when it is on under
    the audit and its gather fell back (no reference was recorded) that is a failure, V4a was not audited."""
    import tp4_sampdraft

    if tp4_sampdraft.enabled(tp4_sampdraft.SHARD_ARGMAX):
        import tp4_shard_argmax

        if tp4_shard_argmax.produced(output[2]):
            # QWEN_FAST_TP4_SHARD_ARGMAX made these maxima (its kernel supersedes the gather); its own audit
            # (QWEN_FAST_TP4_SHARD_ARGMAX_AUDIT) compares them with today's path, row by row
            return
    reference = verify_trace_t1.VALUE_REFERENCES.get(id(output[2]))
    if reference is None:
        if tp4_vglue.audit_enabled() and tp4_vglue.enabled(tp4_vglue.SHARD_VALUES):
            # the lever is on and audited but the gather fell back: V4a was not exercised, so the gate must not pass it
            message = '%s site=sampler no reference: the gather fell back, V4a was not audited' % tp4_vglue.AUDIT_MISMATCH
            diagnostic(message)
            raise AssertionError(message)
        return
    for chip, (part, gathered) in enumerate(zip(operations.get_device_tensors(reference), chip_values)):
        rows = verify_trace_t1.compare_values(gathered, operations.to_torch(part).reshape(-1)[:block_rows])
        if rows:
            message = '%s site=sampler chip=%d rows=%s' % (tp4_vglue.AUDIT_MISMATCH, chip, rows[:8])
            diagnostic(message)
            raise AssertionError(message)
    diagnostic('%s site=sampler shard_values exact=True rows=%d' % (tp4_vglue.AUDIT_MARKER, block_rows))


def audit_sampdraft(operations, output, block_rows, chip_ids, chip_values):
    """The sampler lever's audit (tp4_sampdraft; a no-op unless QWEN_FAST_TP4_SHARD_ARGMAX_AUDIT is on, which the pair cannot have):
    the kernel's per-chip ids and values against today's from the same capture, every row. The drafter audits (conv, heads) are not
    here: a drafter bucket compares its own held pairs right after its own replay (dflash_proposal_trace.compare_draft_audit), because
    a pair allocated after the verify trace was captured can be overwritten by a verify replay."""
    import tp4_sampdraft

    if tp4_sampdraft.audit_enabled(tp4_sampdraft.SHARD_ARGMAX_AUDIT):
        import tp4_shard_argmax

        tp4_shard_argmax.audit_round(operations, output[2], block_rows, chip_ids, chip_values)


def mlp_audit():
    """tp4_mlp_gateup (op-fusion WP4) when one of its flags is in the environment (the rule two_tile_decode.bind_two_tile_mlp uses), else None: the module is imported only
    then, and its audit_claim / audit_replayed / audit_round / audit_release hold nothing (and return 0) with the audit flags off."""
    if not any(name.startswith('QWEN_FAST_MLP_') for name in os.environ):       # (the five names two_tile_decode.MLP_LEVER_FLAGS lists; no other QWEN_FAST_MLP_ name exists)
        return None
    import tp4_mlp_gateup

    return tp4_mlp_gateup


def release_sampdraft_audit(operations, values=None):
    """Free what the sampler audit holds for the trace output `values`: today's sampler outputs (and the V4a reference registered
    for them). The drafter audits' pairs belong to their buckets and are freed with the bucket's trace."""
    import tp4_sampdraft

    if tp4_sampdraft.enabled(tp4_sampdraft.SHARD_ARGMAX):
        import tp4_shard_argmax

        tp4_shard_argmax.release_audit(operations, values)


def reserve_sampdraft(operations, mesh):
    """At a block's warm, before any trace of any block is captured: validate the samp-draft flags (an audit without its lever fails
    here, not at the first round's readback) and reserve the shard-argmax partials buffer (a buffer allocated inside the verify capture
    would change that trace's hole layout). Returns True when a buffer holder was taken (release with release_sampdraft_reserve)."""
    import tp4_sampdraft

    tp4_sampdraft.validate()
    if not tp4_sampdraft.enabled(tp4_sampdraft.SHARD_ARGMAX):
        return False
    import tp4_shard_argmax

    tp4_shard_argmax.reserve(operations, mesh)
    return True


def release_sampdraft_reserve(operations, mesh):
    """At a block's close, after its traces are released: give back the partials buffer holder reserve_sampdraft took."""
    import tp4_shard_argmax

    tp4_shard_argmax.release_reserved(operations, mesh)


def release_vglue_audit(operations, fixture, values=None):
    """QWEN_FAST_TP4_VGLUE_AUDIT: free what the audit holds outside `owned` - each retained record's held tensors (tp4_vglue) and
    the sampler's ttnn.max reference for the trace output `values` - before the fixture that owns the records closes."""
    if not tp4_vglue.audit_enabled():
        return
    held = []
    retained = getattr(fixture, 'retained', None)
    if retained is not None:
        held += [value for state, result, checkpoint in retained.records for value in tp4_vglue.audit_held_of(result)]
    reference = verify_trace_t1.VALUE_REFERENCES.pop(id(values), None) if values is not None else None
    if reference is not None:
        held.append(reference)
    if held:
        release_owned(operations, held)


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


def packed_host_tokens(users, shape, vocab_size):
    """packed_host_inputs' tokens alone (block_rows, 1), int32: the same refusals in the same order (the segment count, then each user's
    validate_tokens), the same per-user tensors concatenated. tp4/round-host KEYED stages nothing else when the key stands."""
    import torch

    if len(users) != shape.users or any(user is None for user in users):
        raise ValueError('One (tokens, start, pages) per segment required')
    tokens = []
    for user_tokens, start, table in users:
        user_tokens = list(user_tokens)
        validate_tokens(user_tokens, shape.rows_per_user, start, vocab_size, shape.capacity)
        tokens.append(torch.tensor(user_tokens, dtype=torch.int32).reshape(len(user_tokens), 1))
    return torch.cat(tokens)


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

    Round-fence plan H1a (verify_prestage.py): split into `packed_values` (every host check
    and value, the T2 K/V guard included) and `write_packed` (the copies), which the
    pre-stage and the verify-time diff also use; this whole-block write is the same calls in
    the same order as before the split. Every call bumps the fixture write epoch
    (verify_prestage.bump): a snapshot taken before it no longer describes the buffers.
    """
    # tp4/hostgap: with the per-block epochs engaged only THIS fixture's snapshot goes stale; otherwise the global bump, as ever.
    verify_prestage.bump_fixture(fixture, 'stage_packed')
    values, readers = packed_values(operations, model, fixture, shape, users)
    write_packed(operations, model, values, readers)
    for own, (user_tokens, start, table) in zip(readers, users, strict=True):
        own.start = start
    return len(values)


def packed_values(operations, model, fixture, shape, users, *, guard=True):
    """stage_packed's host half: every check it makes before its first copy, and the
    (destination, value, dtype, layout) list it writes, in its order, with the readers whose
    words and tables are in it. `guard=False` (the pre-stage only: its tokens are placeholders
    and its tables may predate the execute-time refresh) leaves out the T2 K/V guard, which
    every verify-time call keeps."""
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
    if guard and getattr(fixture, 'kv_chains', False):
        # QWEN_FAST_VERIFY_T2 (#2): the chained K/V write is exact only while no two users write
        # one (page, tile row). serving_packed_step.proposal_rows drafts such a round for the
        # sequential step and ineligible refuses one first seen at the step, so neither reaches
        # here; this is the fail-closed backstop, before any copy.
        guarded = verify_trace_t2.block_users(positions, pages, shape.rows_per_user, shape.users)
        conflict = verify_trace_t2.kv_conflict(guarded)
        if conflict is not None:
            verify_trace_t2.log_line('%s site=stage_packed %s' % (verify_trace_t2.KV_SHARED,
                                                                  verify_trace_t2.kv_conflict_reason(conflict)))
            raise ValueError('The chained K/V write needs disjoint cache tile rows: %s'
                             % verify_trace_t2.kv_conflict_reason(conflict))
        if verify_trace_t2.audit_enabled():
            verify_trace_t2.log_line('%s kv_rows_per_user=%s' % (verify_trace_t2.AUDIT_MARKER, ','.join(
                str(len(verify_trace_t2.kv_tile_rows(user_positions, table))) for user_positions, table in guarded)))
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
        if getattr(own, 'runtime_extent', False):
            # S2: the extent reader's own values for this start - its word (start & 255), then per
            # bundle its cur_pos (E - 1) and its full-width table - word and cur_pos from one helper.
            values.extend(own.stage_values(start, table))
            continue
        words = torch.zeros(8, dtype=torch.int32)
        words[0] = start
        values.append((own.positions, words, operations.int32, operations.ROW_MAJOR_LAYOUT))
        values.extend((entry[1], table[:, :own.capacity // 64].repeat(len(entry[0]), 1).contiguous(),
                       operations.int32, operations.ROW_MAJOR_LAYOUT) for entry in own.metadata)
    if any(tuple(destination.shape) != tuple(value.shape) or destination.dtype != dtype or destination.layout != layout
           for destination, value, dtype, layout in values):
        raise ValueError('Staged packed inputs must preserve every captured tensor signature')
    return values, readers


def write_packed(operations, model, values, readers, *, indices=None, fence=True, poison=True):
    """stage_packed's device half: copy `values` (every one, or the `indices` of them) into
    their captured buffers, the addresses checked unmoved, one fence unless `fence=False` (the
    pre-stage, whose window fence F9 follows; the verify-time diff, whose trace follows on the
    same in-order CQ0). A failed copy poisons the readers unless `poison=False` (the pre-stage:
    its failure only drops its snapshot, and the verify then restages everything). Returns the
    staged host tensors, which the caller keeps alive while a copy may be unfenced."""
    chosen = values if indices is None else [values[index] for index in indices]
    destinations = [destination for destination, value, dtype, layout in chosen]
    before = [addresses(operations, destination) for destination in destinations]
    chips = tp_shapes.chip_count()
    if any(len(pair) != chips for pair in before):
        raise ValueError('%s chip-local addresses per packed input required' % tp_shapes.count_word())
    if any(len({pair[chip] for pair in before}) != len(before) for chip in range(chips)):
        raise ValueError('%s independent chip-local buffers per packed input required' % tp_shapes.count_word())
    staged = [operations.from_torch(value, device=None, dtype=dtype, layout=layout,
                                    mesh_mapper=operations.ReplicateTensorToMesh(model.mesh_device))
              for destination, value, dtype, layout in chosen]
    try:
        try:
            for source, destination in zip(staged, destinations, strict=True):
                operations.copy_host_to_device_tensor(source, destination)
        finally:
            if fence:
                operations.synchronize_device(model.mesh_device)
        if [addresses(operations, destination) for destination in destinations] != before:
            raise AssertionError('Packed input staging replaced a captured buffer')
    except BaseException:
        # A half-staged word or table poisons the readers, as stage_inputs poisons one.
        if poison:
            for own in readers:
                own.failed = True
        raise
    return staged


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


PROFILE_DUMP_EVERY = 'QWEN_FAST_PROFILE_DUMP_EVERY'
# One process-wide count of verify replays of ANY kind (a packed round, a sequential step): the TP4 op profile's
# read-back cadence (docs/tp4-profile.md). The device profiler's per-core buffer holds op-support (20000) programs and
# nothing is read back until a drain, so a read-back every Nth replay keeps every window inside it.
_replays_seen = 0


def dump_device_profiler_every(operations, mesh, environ=None):
    """Under the TP4 op-profile arm only (QWEN_FAST_PROFILE_DUMP_EVERY=N): count this verify replay and, on every Nth,
    read the device profiler buffers back (ttnn.ReadDeviceProfiler). Called once per verify replay of any kind, right
    after it (packed_verifier's round, verifier_engine_tp's sequential step), so the count is one process-wide
    sequence. Returns True when a read-back was issued; inert (and counting nothing) when unset."""
    import os

    global _replays_seen
    value = (os.environ if environ is None else environ).get(PROFILE_DUMP_EVERY, '')
    if not value:
        return False
    try:
        every = int(value)
    except ValueError:
        raise ValueError('%s must be a positive replay count; got %r' % (PROFILE_DUMP_EVERY, value))
    if every < 1:
        raise ValueError('%s must be a positive replay count; got %r' % (PROFILE_DUMP_EVERY, value))
    _replays_seen += 1
    if _replays_seen % every:
        return False
    reader = getattr(operations, 'ReadDeviceProfiler', None)
    if reader is None:
        diagnostic('[PINDIAG] %s=%d but the runtime has no ReadDeviceProfiler; no device dump' % (PROFILE_DUMP_EVERY, every))
        return False
    reader(mesh)
    diagnostic('[PINDIAG] device profiler read back after replay %d (%s=%d)' % (_replays_seen, PROFILE_DUMP_EVERY, every))
    return True


REPLAY_GROUP_ROWS_FLAG = 'QWEN_FAST_REPLAY_GROUP_ROWS'


SAMPLER_PREWARM_FLAG = 'QWEN_FAST_PACKED_SAMPLER_PREWARM'
SAMPLER_IN_TRACE_FLAG = 'QWEN_FAST_PACKED_SAMPLER_IN_TRACE'
SAMPLER_ARM_MARKER = '[PINDIAG] packed sampler arm'


def sampler_arm_enabled(flag, environ=None):
    """QWEN_FAST_PACKED_SAMPLER_PREWARM=1 / QWEN_FAST_PACKED_SAMPLER_IN_TRACE=1: exactly '1' engages the arm (default off)."""
    return (os.environ if environ is None else environ).get(flag) == '1'


def sampler_arm_requested(environ=None):
    return sampler_arm_enabled(SAMPLER_PREWARM_FLAG, environ) or sampler_arm_enabled(SAMPLER_IN_TRACE_FLAG, environ)


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


def fused_commit_requested(environ=None):
    """Round-fence plan H1b: QWEN_FAST_FUSED_COMMIT is '1' (fused_commit validates the value when it
    builds; this only decides whether to import it)."""
    return (os.environ if environ is None else environ).get('QWEN_FAST_FUSED_COMMIT') == '1'


def gdn_after_pairs_requested(environ=None):
    """Round-fence plan H2: QWEN_FAST_GDN_AFTER_PAIRS is set to anything but 0 (early_draft validates
    the value and its parent flag; this only decides whether to import it)."""
    return (os.environ if environ is None else environ).get('QWEN_FAST_GDN_AFTER_PAIRS', '0') != '0'


# S2 (design W3): the extent block's lines, its gate-only audit and its replay deadline. The gate (W11)
# reads the markers; QWEN_FAST_EXTENT_AUDIT is never in the image ENV or a traffic profile.
EXTENT_ROUND_MARKER = '[PINDIAG] packed extent round'
EXTENT_CAP_REFUSED_MARKER = '[PINDIAG] packed extent cap refused'
EXTENT_AUDIT_FLAG = 'QWEN_FAST_EXTENT_AUDIT'
EXTENT_AUDIT_MARKER = '[EXTENT-AUDIT]'
EXTENT_AUDIT_MISMATCH_MARKER = '[EXTENT-AUDIT] MISMATCH'
REPLAY_DEADLINE_FLAG = 'QWEN_FAST_REPLAY_DEADLINE_S'
REPLAY_DEADLINE_DEFAULT_S = 30.0
REPLAY_DEADLINE_MARKER = '[PINDIAG] replay deadline exceeded'
REPLAY_DEADLINE_EXIT_CODE = 70


def extent_audit_enabled(environ=None):
    """QWEN_FAST_EXTENT_AUDIT (S2, gate only): '1' audits every packed round of an extent block
    (PackedVerifierEngine.audit_extent), '0' or unset does not; any other value is a configuration
    error. Read by an extent block only."""
    value = (os.environ if environ is None else environ).get(EXTENT_AUDIT_FLAG, '0')
    if value not in ('0', '1'):
        raise ValueError('%s must be 0 or 1' % EXTENT_AUDIT_FLAG)
    return value == '1'


def replay_deadline_seconds(environ=None):
    """QWEN_FAST_REPLAY_DEADLINE_S (S2): how long one verify or commit replay of an extent block may
    run before ReplayDeadline ends the process; default 30 (a round is ~0.17 s). A positive decimal
    number of seconds; anything else is a configuration error. Read by an extent block only."""
    import re

    text = (os.environ if environ is None else environ).get(REPLAY_DEADLINE_FLAG)
    if text is None:
        return REPLAY_DEADLINE_DEFAULT_S
    if type(text) is not str or re.fullmatch('[0-9]+([.][0-9]+)?', text) is None or float(text) <= 0:
        raise ValueError('%s must be a positive number of seconds, got %r' % (REPLAY_DEADLINE_FLAG, text))
    return float(text)


class ReplayDeadline:
    """S2's replay deadline (design W3, B5): one daemon watchdog per extent block, armed around each
    verify and commit replay (`armed`). A replay still running `seconds` after it was armed logs
    REPLAY_DEADLINE_MARKER with what it was replaying (round, trace, segments, families) and the
    stack of the thread that armed it, then ends the process with exit code 70: a hung card then
    ends the engine with the families it held in the log, instead of wedging it for the rest of the
    run (the E < 4096 stale-writer zone of K64j's probe README is what this covers; card-M Z passed
    all fifteen families). Whether the platform's restart recovers a hung card without an M+A reset
    is UNVERIFIED (design Q21).

    Host only. The thread starts at the first arming and sleeps while nothing is armed; a scope
    armed inside another is a no-op (the outer one covers it).

    The watchdog is a Python thread, so it runs only while the replaying thread does not hold the GIL.
    Every armed scope holds ttnn.execute_trace calls and nothing else (test_packed_extent_block pins the
    three scope bodies), and ttnn.execute_trace releases the GIL for the whole call, its blocking wait on
    the card included: its nanobind binding carries nb::call_guard<nb::gil_scoped_release>
    (ttnn/cpp/ttnn-nanobind/operations/trace.cpp:53-63 at the image's tt-metal 9f9cd4fd, v0.77.0-rc1,
    re-exported unwrapped by ttnn/ttnn/__init__.py:149-155). A replay hung on the card therefore leaves
    this thread free to fire; test_packed_extent_block shows it end a real process with exit 70 while
    the main thread is blocked in C with the GIL released, and, as the control, a call that KEPT the GIL
    starving it until the call returned. So anything armed here must be a GIL-releasing call; a signal
    timeout (docs/gotchas.md, 'Hangs need an external watchdog') is a different mechanism - its handler
    runs only on the main thread between bytecodes - and does not apply to a thread."""

    def __init__(self, seconds, *, exit=None, log=None, clock=None):
        if type(seconds) not in (int, float) or not 0 < seconds < float('inf'):
            raise ValueError('A positive, finite replay deadline is required, got %r' % (seconds,))
        self.seconds = float(seconds)
        self.exit = os._exit if exit is None else exit
        self.log = log
        self.clock = time.monotonic if clock is None else clock
        self.condition = threading.Condition()
        self.current = None
        self.thread = None
        self.closed = False
        self.fired = None

    @contextmanager
    def armed(self, what):
        with self.condition:
            if self.closed:
                raise RuntimeError('The replay deadline of a closed packed block cannot be armed')
            nested = self.current is not None
            if not nested:
                self.current = (self.clock() + self.seconds, str(what), threading.get_ident())
                if self.thread is None:
                    self.thread = threading.Thread(target=self.watch, name='qwen-replay-deadline', daemon=True)
                    self.thread.start()
                self.condition.notify_all()
        try:
            yield
        finally:
            if not nested:
                with self.condition:
                    self.current = None
                    self.condition.notify_all()

    def watch(self):
        while True:
            with self.condition:
                while self.current is None and not self.closed:
                    self.condition.wait()
                if self.closed:
                    return
                deadline, what, ident = self.current
                remaining = deadline - self.clock()
                if remaining > 0:
                    self.condition.wait(remaining)
                    continue
                self.fired = what
            self.fire(what, ident)
            return

    def fire(self, what, ident):
        """The deadline passed with the replay still armed: the line, the replaying thread's stack,
        then the exit. The exit is unconditional: a failed report must not keep a wedged engine."""
        try:
            log = self.log or diagnostic
            frame = sys._current_frames().get(ident)
            stack = ''.join(traceback.format_stack(frame)) if frame is not None else '(the replaying thread is gone)\n'
            log('%s %s seconds=%g' % (REPLAY_DEADLINE_MARKER, what, self.seconds))
            log('[PINDIAG] replay deadline traceback of the replaying thread\n' + stack)
        except BaseException:
            pass
        finally:
            self.exit(REPLAY_DEADLINE_EXIT_CODE)

    def close(self):
        with self.condition:
            self.closed = True
            self.current = None
            self.condition.notify_all()


def same_bits(expected, actual):
    """Whether a readback holds exactly these BF16 bit patterns (the narrow mask: 0x0000 and 0xff80)."""
    import torch

    try:
        expected = expected.to(torch.bfloat16).contiguous().reshape(-1)
        actual = actual.to(torch.bfloat16).contiguous().reshape(-1)
        return actual.numel() == expected.numel() and bool(torch.equal(actual.view(torch.int16),
                                                                          expected.view(torch.int16)))
    except (RuntimeError, TypeError, AttributeError):
        return False


def page_zero_index(table, start, rows, *, extent=False):
    """The first index inside [0, (start + rows + 63) // 64) at which this (1, page_width) page
    table holds physical page 0, or None: every page a user at `start` reads or writes in a
    `rows`-row round (padded_probe.used_pages). Raises ValueError when the table cannot map that
    range, and whatever indexing raises for a table that is not one.

    S2 (`extent=True`, design 2.6): also every page the extent SDPA reads, [0, E // 64) with
    E = (start // 256 + 1) * 256 - a page-0 key there sits at a masked position, and a NaN one would
    survive the -inf add. Rows of a ticket crossing E still write past it, so the range is the
    larger of the two."""
    limit = int(start) + int(rows)
    if extent:
        limit = max(limit, (int(start) // 256 + 1) * 256)
    used = (limit + 63) // 64
    values = table[0][:used]
    values = values.tolist() if hasattr(values, 'tolist') else list(values)
    if len(values) < used:
        raise ValueError('a %d-entry page table cannot map positions [0, %d)' % (len(values), limit))
    for index, value in enumerate(values):
        if int(value) == 0:
            return index
    return None


class PackedVerifierEngine:
    """Owner of one packed verify block: build at attach, then per round
    `verify(entries)` -> per-entry predictions, `features(segment)` for each user's
    publication, `commit_user(segment, prefix)` for each user's decision."""

    def __init__(self, operations, model, helpers, sampler, *, pool, shared_weights, shape, feature_taps,
                 capture_position=None, pool_slots=None, padded_min_users=None, collectives=None,
                 defer_capture=False):
        import torch

        # QWEN_FAST_M3_BLOCKS=2 (serving_runtime.complete_blocks_two_phase): True builds only the first part of the
        # construction here - the persistent allocations (the initial snapshot, the checkpoints, the taps) - and leaves
        # `warm_and_fixture`, `capture_traces` and `finish_construction` to the caller, so several blocks can allocate
        # and warm first and capture afterwards (CONSTRUCTION ORDER, module docstring). False, the default, is the whole
        # construction in this call, in the order it always had.
        if type(defer_capture) is not bool:
            raise ValueError('defer_capture must be an explicit bool')
        self.defer_capture = defer_capture
        self.build = None
        self.shape = validate_shape(shape)
        # QWEN_FAST_CAPTURE_PLUG (TP4 only, off by default): the capture zone, opened before the warm forward and sealed after
        # the last capture (capture_plug); None otherwise.
        self.plug = None
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
            # The packed block's one native chunk family is the last 256 positions
            # (attention_mask_replay.validate_ticket), whatever the per-request budget.
            capture_position = shape.capacity - PACKED_FAMILY_TOKENS
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
        # S2 (QWEN_FAST_EXTENT_REPLAY=1, design W3): a pool built with extent_replay lends full-width
        # tables and cur_pos words instead of per-family tables, and the block is then the extent one:
        # one capture serving every user at its own family (module docstring). Keyed on the pool's
        # storage alone, and only on an explicit True. Everything below reads `extent`; False, every
        # path is today's.
        self.extent = getattr(pool, 'extent_replay', False) is True
        self.extent_storage = self.extent_cur_pos = self.cur_pos_addresses = self.reader_addresses = None
        self.extent_audit, self.deadline = False, None
        self.round_starts = {}
        self.extent_audit_cursor = 0
        self.extent_counts = dict(rounds=0, cap_events=0, mixed_rounds=0, cap_refused=0, audit_mismatches=0)
        if self.extent:
            self.take_extent_storage(operations, pool, shape, capture_position)
        else:
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
        self.captured_reader = None       # (the fixture's replay reader, its multi launch) the verify trace was captured on
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
        # QWEN_FAST_PACKED_SAMPLER_PREWARM / QWEN_FAST_PACKED_SAMPLER_IN_TRACE (the audits-off hang's
        # isolation arms, default OFF): what the T1 audit's pinned sampler protects, without its per-round
        # readback and compare. Both reuse the audit's sampler call (sample_shards below) and engage only
        # beside the shard argmax with the audit off; they never change a token (shard_predictions reads
        # output[1] and output[2] only, and the audit's compare stays gated on shard_audit).
        self.sampler_prewarm = self.shard_argmax and not self.shard_audit and sampler_arm_enabled(SAMPLER_PREWARM_FLAG)
        self.sampler_in_trace = self.shard_argmax and not self.shard_audit and sampler_arm_enabled(SAMPLER_IN_TRACE_FLAG)
        # QWEN_FAST_VERIFY_T2 (verify_trace_t2): read once here too. #2 (kv_chains) makes the
        # warm fixture's K/V write one chain (build_fixture); the fixture decides the writer
        # (model_batch) and this block reads what it became (kv_chains, warm_kv_chains). The
        # windows audit runs only if the capture engaged #1.
        self.verify_t2 = verify_trace_t2.enabled()
        self.kv_chains_cut = verify_trace_t2.cut('kv_chains')
        self.warm_kv_chains = False
        self.windows_audit = False
        # QWEN_FAST_GDN_SEQ_BLOCK_AUDIT (K5-A, gdn_seq_block): read once here too. The captured
        # forward's audited layers hold the audit's tensors (gdn_user_batch_conv), and every replay
        # compares them (gdn_seq_block.audit_round). () without QWEN_FAST_GDN_SEQ_BLOCK=1.
        self.seq_block_audit = gdn_seq_block.audit_active()
        # QWEN_FAST_PADDED_PROBE (variable-user packed rounds M1, padded_probe.py): read once
        # here, like the flags above. Off, padded_probe is never imported and verify() is
        # today's.
        self.padded_probe = os.environ.get('QWEN_FAST_PADDED_PROBE') == '1'
        # Round-fence plan H1a (verify_prestage.py; every flag default off): read once here, like
        # the flags above. QWEN_FAST_PRESTAGE gives the block its pre-stage state (the drafts'
        # window writes every input of the next verify but its tokens; verify() then writes only
        # what differs), QWEN_FAST_PRESTAGE_AUDIT its read-back audit; QWEN_FAST_ROUND_FENCES puts
        # the captured retained block on the fence diet (F3, F8; set after the capture, below).
        # Off, `prestaged` is None, `round_fences` False, and every path below is today's.
        self.round_fences = verify_prestage.round_fences_enabled()
        self.prestaged = (verify_prestage.BlockPrestage(self, audit=verify_prestage.audit_enabled())
                          if verify_prestage.enabled() else None)
        self.validated_this_round = False
        # Round-fence plan H1b (fused_commit.py; QWEN_FAST_FUSED_COMMIT and its sub-flags, every one
        # default off): the fused commit, built right after the taps - its RoPE tables and deltas
        # predate the verify capture (R2) - and captured after the GDN commit traces. None without
        # the flag or when fused_commit refused it; every path below is then today's.
        self.fused = None
        # S2 B6 (publication_warm.py): the summary of the eager publication warm the extent block runs at
        # attach, after its last capture; None without the extent block, or when the warm was skipped.
        self.publication_warm = None
        # complete_blocks_two_phase sets this False on every block after the first: the plan is the same 71 shapes and the
        # program cache is shared, so a second warm compiles nothing and only cycles the shared drafter collectives again.
        self.warm_publication = True
        # Round-fence plan H2 (early_draft.py; QWEN_FAST_GDN_AFTER_PAIRS under QWEN_FAST_EARLY_DRAFT, default
        # off): read once here, like the flags above. The block defers a round's GDN commit traces only
        # when the early draft arms it (arm_deferred_commits) and only on the fence diet, whose owed fence
        # the next replay pays; without QWEN_FAST_ROUND_FENCES it is refused (logged at the end of attach).
        self.gdn_after_pairs, self.gdn_after_pairs_refusal = False, None
        self.defer_armed = self.deferring = False
        self.deferred_commits = []
        if gdn_after_pairs_requested():
            import early_draft

            if early_draft.gdn_after_pairs_enabled():
                if self.round_fences:
                    self.gdn_after_pairs = True
                else:
                    self.gdn_after_pairs_refusal = 'round-fences-off'
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
            if fused_commit_requested():
                # H1b, R2: allocated here, before the warm forward and every capture.
                import fused_commit

                self.stage = 'fused commit storage'
                self.fused = fused_commit.build(self, operations=operations, mesh=self.mesh, pool=pool,
                                                shared_weights=shared_weights, collectives=collectives,
                                                diagnostic=diagnostic)
            # The capture's placeholder users: every row reads physical page 0 at the
            # capture position. Positions, rotary tables and every page table are
            # restaged before each verify; only the addresses and shapes are baked.
            placeholders = [SimpleNamespace(position=capture_position,
                                            pages=torch.zeros((1, shape.page_width), dtype=torch.int32))
                            for user in range(shape.users)]
            # What the later phases of the construction need of this call's arguments.
            self.build = SimpleNamespace(operations=operations, pool=pool, shared_weights=shared_weights,
                                         collectives=collectives, placeholders=placeholders, started=started,
                                         stage='allocated')
            if defer_capture:
                self.stage = 'allocated'
                return
            self._warm_and_fixture()
            self._capture_traces()
            self._finish_construction()
        except BaseException as failure:
            self.phase = 'failed'
            # Logged BEFORE close: run 35505708710 (image v50) raised on the host inside the
            # 64-row warm forward after the device had hung on an enqueued op; close then
            # blocked in the device fence for the rest of the timeout and the cause was
            # never logged (serving_runtime's own [PINDIAG] line comes after this close).
            self.report_failure(failure)
            self.close(wait=False)
            raise

    # The construction's phases after the allocations, each a method so that a caller building several blocks can run
    # every block's first phase before any block's second (serving_runtime.complete_blocks_two_phase). __init__ runs
    # them in order when the block is not deferred - the order this construction always had - and the caller runs them
    # one public wrapper at a time when it is. A phase that fails closes the block without the device fence, as
    # __init__ does for its own.
    def phase_guard(self, name, expected, following, step):
        if self.phase in ('failed', 'closed'):
            raise ValueError('The packed block is %s: %s cannot run' % (self.phase, name))
        if self.build is None or self.build.stage != expected:
            raise ValueError('%s needs a block that finished %r, not %r'
                             % (name, expected, None if self.build is None else self.build.stage))
        try:
            step()
            if self.build is not None:
                self.build.stage = following
        except BaseException as failure:
            self.phase = 'failed'
            self.report_failure(failure)
            self.close(wait=False)
            raise

    def warm_and_fixture(self):
        """Second phase of a deferred block: the warm forward and the captured fixture (the extent readers' words and
        masks included), so every persistent allocation the block's traces will bake exists. No capture yet."""
        self.phase_guard('warm_and_fixture', 'allocated', 'fixture', self._warm_and_fixture)

    def capture_traces(self):
        """Third phase of a deferred block: the verify trace and every commit trace. No persistent allocation of any
        block may follow, so the caller runs this for every block only after every block's warm_and_fixture."""
        self.phase_guard('capture_traces', 'fixture', 'captured', self._capture_traces)

    def finish_construction(self):
        """Last phase of a deferred block: the publication warm, the reseed, the binding check; the block is idle."""
        self.phase_guard('finish_construction', 'captured', 'done', self._finish_construction)

    def _warm_and_fixture(self):
        operations, shape, placeholders = self.build.operations, self.shape, self.build.placeholders
        if self.defer_capture and os.environ.get('QWEN_FAST_TP', '2') == '4' and capture_plug.config() is not None:
            raise ValueError('QWEN_FAST_CAPTURE_PLUG is not supported for blocks built in two phases '
                             '(QWEN_FAST_M3_BLOCKS=2): one zone cannot span the captures of several blocks')
        self.sampdraft_reserved = reserve_sampdraft(operations, self.mesh)
        self.open_capture_plug(operations)
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
            result = self.operation(warm, warm=True)
            self.stage = 'warm forward fence'
            operations.synchronize_device(self.mesh)
            # QWEN_FAST_TP4_RS_UNIT_MAJOR_AUDIT: this eager forward's unit-major reductions against the split's, now (nothing
            # held when the audit is off).
            tile_collective_tp.audit_claim(warm, 'warm')
            tile_collective_tp.audit_round(operations, warm, 0)
            mlp = mlp_audit()
            if mlp is not None:
                # QWEN_FAST_MLP_CFG_AUDIT / QWEN_FAST_MLP_GATEUP_AUDIT (tp4_mlp_gateup, op-fusion WP4): the lever's MLP outputs against the served forward's, held beside them
                mlp.audit_claim(warm, 'warm')
                mlp.audit_round(operations, warm, 0)
        finally:
            tile_collective_tp.audit_claim(warm, 'warm')
            tile_collective_tp.audit_release(operations, warm)
            mlp = mlp_audit()
            if mlp is not None:
                mlp.audit_claim(warm, 'warm')
                mlp.audit_release(operations, warm)
            if result is not None:
                release_sampdraft_audit(operations, result[2] if len(result) > 2 else None)
                release_vglue_audit(operations, warm, result[2] if len(result) > 2 else None)
                release_owned(operations, [value for value in result if value is not None])
            else:
                release_vglue_audit(operations, warm)
            warm.close()
        self.stage = 'verify trace capture'
        self.fixture = self.build_fixture(placeholders)
        if self.extent:
            # S2: each extent reader's own word and narrow masks, allocated by this fixture before
            # any trace: validate_bindings checks them unmoved from here on.
            self.reader_addresses = self.extent_reader_addresses()

    def _capture_traces(self):
        operations, shape = self.build.operations, self.shape
        if self.verify_t1:
            verify_trace_t1.take()  # count only what the captured forward engages
        if self.verify_t2:
            verify_trace_t2.take()
        tp4_vglue.take()
        try:
            self.trace, self.output = capture_operation(operations, self.mesh, lambda: self.operation(self.fixture))
            captured = getattr(self.fixture, 'replay_reader', None)
            self.captured_reader = None if captured is None else (captured, getattr(captured, 'multi', None))
        finally:
            # also after a failed capture: its audited clones belong to the fixture, released when it closes
            tile_collective_tp.audit_claim(self.fixture, 'capture')
            mlp = mlp_audit()
            if mlp is not None:
                mlp.audit_claim(self.fixture, 'capture')
        if self.verify_t1:
            self.note_verify_t1(verify_trace_t1.take())
        if self.verify_t2:
            self.note_verify_t2(verify_trace_t2.take())
        self.note_vglue(tp4_vglue.take())
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
        if self.fused is not None:
            # H1b: every segment's T_proj, then (in place) every (segment, prefix) slide trace,
            # next to the GDN commit traces and after the verify trace.
            self.stage = 'fused commit capture'
            self.fused.capture(capture_operation)
        self.seal_capture_plug()

    def _finish_construction(self):
        build = self.build
        operations, pool, shared_weights, collectives, started = (build.operations, build.pool, build.shared_weights,
                                                                  build.collectives, build.started)
        shape = self.shape
        if self.extent and getattr(self, 'warm_publication', True):
            # S2 B6 (publication_warm.py): today's eager publication - what a packed round the fused
            # commit refuses and every sequential step run - published and discarded once at every shape
            # serving can ask of it (each segment's row offset x prefix, each pooled width x prefix), on
            # scratch, fenced and released here, so no serving path compiles a publication program after
            # attach. After every capture; a raise fails the attach like every stage here.
            import publication_warm

            self.stage = 'eager publication warm'
            self.publication_warm = publication_warm.warm(self, operations=operations, mesh=self.mesh, pool=pool,
                                                          shared_weights=shared_weights, collectives=collectives,
                                                          log=diagnostic)
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
        if self.round_fences:
            # Only this block's retained records: a sequential engine's never take the diet.
            self.fixture.retained.use_round_fences()
            diagnostic('%s users=%d' % (verify_prestage.FENCES_ENGAGED_MARKER, self.users))
        if self.prestaged is not None:
            diagnostic('%s users=%d audit=%d' % (verify_prestage.ENGAGED_MARKER, self.users,
                                                 int(self.prestaged.audit)))
        if verify_prestage.hostgap_log_enabled():
            # tp4/hostgap stage 0: the gen-2 collection log, once per process (idempotent).
            verify_prestage.install_gc_log()
            # tp4/fx-wph: the host-gap instrument's wrappers on the host calls and its collection counter (hostgap_instr), once per process (idempotent); it reads the
            # probe flags strictly, here.
            import hostgap_instr

            hostgap_instr.install(self.operations)
            hostgap_instr.probe_enabled()
            hostgap_instr.probe_every()
        if self.fused is not None:
            diagnostic(self.fused.engaged_line())
        if self.gdn_after_pairs or self.gdn_after_pairs_refusal is not None:
            import early_draft

            if self.gdn_after_pairs:
                diagnostic('%s users=%d pipelined=%d' % (early_draft.GDN_ENGAGED_MARKER, self.users,
                                                         int(self.pipelined_commits)))
            else:
                diagnostic('%s users=%d reason=%s' % (early_draft.GDN_REFUSED_MARKER, self.users,
                                                      self.gdn_after_pairs_refusal))
        # Nothing of the construction's arguments is kept once the block is idle.
        self.build = None

    def open_capture_plug(self, operations):
        """QWEN_FAST_CAPTURE_PLUG=1 at QWEN_FAST_TP=4: the capture zone (capture_plug.Zone), opened before the block's first
        eager forward so the warm forward, the fixture and every capture allocate in it. A failure raises: the attach fails
        closed. Without the flag, nothing."""
        if os.environ.get('QWEN_FAST_TP', '2') != '4':
            return
        settings = capture_plug.config()
        if settings is None:
            return
        import trace_census

        self.plug_sequence = trace_census.SEQUENCE
        self.plug = capture_plug.Plug.packed(settings, operations, self.mesh, diagnostic)
        self.plug.open()

    def seal_capture_plug(self):
        """After the block's last capture: plug every hole the captures freed, then let the later allocations in below the zone.
        With the graph census on, every recorded freed extent of these captures is held against the zone's top (fail closed)."""
        if self.plug is None:
            return
        import trace_census

        self.stage = 'capture plug seal'
        self.plug.seal()
        self.plug.verify_extents([extent for trace in trace_census.TRACES if trace['seq'] > self.plug_sequence
                                  for extent in trace['ranges']])
        capture_plug.PACKED.append(self.plug)

    def take_extent_storage(self, operations, pool, shape, capture_position):
        """S2 (design W3): take the pool's extent storage for this shape (serving_buffer_pool.
        PackedExtentStorage: per user, per bundle of the extent layout, a full-width table and a
        cur_pos word), set the capacity every extent reader of the block takes - the whole table,
        C = page_width * 64 - and read the extent block's own flags. Refused before anything is taken:
        a capture start the extent path does not admit, a replay group width other than K64j's
        qualified eight, a bad QWEN_FAST_EXTENT_AUDIT or QWEN_FAST_REPLAY_DEADLINE_S."""
        import extent_attention_replay

        self.replay_capacity = shape.page_width * 64
        if not extent_attention_replay.admits(capture_position, shape.rows_per_user, self.replay_capacity):
            raise ValueError('The extent block captures at a start the extent path admits (%d <= start, start + %d <= %d); '
                             'got %r' % (extent_attention_replay.MIN_LIVE_START, shape.rows_per_user,
                                         self.replay_capacity, capture_position))
        if replay_group_rows() != extent_attention_replay.EXTENT_GROUP_ROWS:
            raise ValueError('The extent block is qualified at eight-row replay groups only (K64j CB1, G8B2): %s=8 required'
                             % REPLAY_GROUP_ROWS_FLAG)
        self.extent_audit = extent_audit_enabled()
        self.deadline = ReplayDeadline(replay_deadline_seconds())
        storage = pool.packed_extent(shape.users, shape.rows_per_user)
        if getattr(storage, 'users', None) != shape.users or getattr(storage, 'rows', None) != shape.rows_per_user:
            raise ValueError('The pool lent extent storage for %r x T%r users; the block is %d x T%d'
                             % (getattr(storage, 'users', None), getattr(storage, 'rows', None), shape.users,
                                shape.rows_per_user))
        self.replay = storage.take()
        self.extent_storage = [list(pairs) for pairs in storage.segment_storage()]
        self.replay_tables = [list(tables) for tables in storage.tables]
        self.extent_cur_pos = [list(values) for values in storage.cur_pos]
        self.replay_addresses = [[addresses(operations, table) for table in tables] for tables in self.replay_tables]
        self.cur_pos_addresses = [[addresses(operations, value) for value in values] for values in self.extent_cur_pos]

    def extent_reader_addresses(self):
        """Per segment, the chip addresses of its extent reader's own positions word and narrow masks."""
        return [[addresses(self.operations, own.positions)]
                + [addresses(self.operations, entry[2]) for entry in own.metadata]
                for own in self.fixture.replay_reader.readers]

    def admits(self, position):
        """Whether this block serves a live ticket at `position`. S2's extent path: any integer start
        with 128 <= start and start + rows_per_user <= C (extent_attention_replay.admits), each at its
        own family. Without it, the block's one captured family (validate_ticket), as always. Host
        only: serving_packed_step asks it before drafting (proposal_rows) and at the step (ineligible,
        extent only), and verify() repeats it as the raising backstop."""
        if getattr(self, 'extent', False):
            import extent_attention_replay

            return extent_attention_replay.admits(position, self.rows_per_user, self.replay_capacity)
        try:
            validate_ticket(position, self.rows_per_user, self.replay_capacity, short_context=False)
        except ValueError:
            return False
        return True

    def accept_limit(self, position):
        """How many rows of a ticket at `position` may commit, or None for all of them. S2's extent
        path: min(rows_per_user, E - position) - a row at or past its family end E sees all of [0, E)
        and never its own key, so it is capped like a rejected draft (design 2.4). The family block's
        tickets never cross its family (validate_ticket): None."""
        if not getattr(self, 'extent', False):
            return None
        import extent_attention_replay

        return extent_attention_replay.accept_limit(position, self.rows_per_user)

    def replay_deadline(self, trace, segments, prefix=None):
        """The replay deadline's scope for one verify, commit or flush replay (S2), or a no-op for a
        block without one."""
        deadline = getattr(self, 'deadline', None)
        if deadline is None:
            return nullcontext()
        starts = getattr(self, 'round_starts', None) or {}
        segments = tuple(segments)
        what = 'round=%d trace=%s segments=%s families=%s' % (
            self.rounds + (1 if trace == 'verify' else 0), trace, ','.join(str(segment) for segment in segments),
            ','.join(str((starts[segment] // 256 + 1) * 256) if segment in starts else '-' for segment in segments))
        if prefix is not None:
            what += ' prefix=%d' % prefix
        return deadline.armed(what)

    def note_extent_round(self, users, segments, idle):
        """S2: one EXTENT_ROUND_MARKER line per packed round, from verify after its replay - the executed
        path, not a flag (memory graft-mounted-is-not-graft-executed): each live segment's family, the
        idle segments, and each live segment whose commit the family end caps (segment:limit)."""
        import extent_attention_replay

        live = sorted(segments)
        families = [extent_attention_replay.extent(users[segment][1]) for segment in live]
        capped = [(segment, self.accept_limit(users[segment][1])) for segment in live]
        capped = [(segment, limit) for segment, limit in capped if limit < self.rows_per_user]
        self.extent_counts['rounds'] += 1
        self.extent_counts['cap_events'] += len(capped)
        if len(set(families)) > 1:
            self.extent_counts['mixed_rounds'] += 1
        diagnostic('%s round=%d live=%d families=[%s] idle=[%s] capped=[%s]' % (
            EXTENT_ROUND_MARKER, self.rounds + 1, len(live),
            ','.join('%d:%d' % pair for pair in zip(live, families)), ','.join(str(segment) for segment in idle),
            ','.join('%d:%d' % pair for pair in capped)))

    def audit_extent(self, users, round_number):
        """QWEN_FAST_EXTENT_AUDIT (gate profiles only; design 2.2 step 7, A1). After the replay, read
        back from both chips every segment's positions word and cur_pos words, and - in rotation, one
        segment a round - that segment's narrow masks and full-width tables, and compare them with
        the host's own values: extent_values(start), the pinned mask kernel's host transliteration at
        capacity 256 (narrow_mask_host) and the staged table. One EXTENT_AUDIT_MARKER line per packed
        round; any mismatch also logs EXTENT_AUDIT_MISMATCH_MARKER and restages the round in full, so
        the next verify reads what was meant (the gate fails the arm on the line). The reads are after
        the replay: they perturb timing, never arithmetic, and the line carries their ms. Returns the
        mismatches, as 'what:segment' strings."""
        import torch

        import extent_attention_replay
        from verify_prestage import same_readback

        started = time.perf_counter()
        operations = self.operations
        readers = list(self.fixture.replay_reader.readers)
        rotated = self.extent_audit_cursor % len(readers)
        self.extent_audit_cursor += 1
        counts = dict(words=0, cur_pos=0, mask=0, tables=0)
        mismatched = []

        def chips(tensor):
            return [operations.to_torch(shard) for shard in operations.get_device_tensors(tensor)]

        for segment, (own, (tokens, start, table)) in enumerate(zip(readers, users, strict=True)):
            word, position = extent_attention_replay.extent_values(start)
            words = torch.zeros(8, dtype=torch.int32)
            words[0] = word
            if all(same_readback(words, value) for value in chips(own.positions)):
                counts['words'] += 1
            else:
                mismatched.append('word:%d' % segment)
            if all(same_readback(torch.full((len(entry[0]),), position, dtype=torch.int32), value)
                   for entry, cur_pos in zip(own.metadata, own.cur_pos, strict=True) for value in chips(cur_pos)):
                counts['cur_pos'] += 1
            else:
                mismatched.append('cur_pos:%d' % segment)
            if segment != rotated:
                continue
            masks = tables = True
            for bundle, pages, mask, config in own.metadata:
                expected = extent_attention_replay.narrow_mask_host(word, bundle[0]['rows'], len(bundle),
                                                                    bundle[0]['offset'])
                masks = masks and all(same_bits(expected, value) for value in chips(mask))
                host = table[:, :own.page_width].repeat(len(bundle), 1)
                tables = tables and all(same_readback(host, value) for value in chips(pages))
            counts['mask'] += int(masks)
            counts['tables'] += int(tables)
            if not masks:
                mismatched.append('mask:%d' % segment)
            if not tables:
                mismatched.append('table:%d' % segment)
        if mismatched:
            self.extent_counts['audit_mismatches'] += len(mismatched)
            diagnostic('%s round=%d at=%s' % (EXTENT_AUDIT_MISMATCH_MARKER, round_number, ','.join(mismatched[:8])))
            # Every buffer again, fenced: the next verify reads what was meant (stage_packed bumps the
            # pre-stage epoch and sets every reader's start).
            stage_packed(operations, self.model, self.fixture, self.shape, users)
        diagnostic('%s round=%d segments=%d words_ok=%d cur_pos_ok=%d mask_ok=%d tables_ok=%d rotated=%d ms=%.2f' % (
            EXTENT_AUDIT_MARKER, round_number, len(readers), counts['words'], counts['cur_pos'], counts['mask'],
            counts['tables'], rotated, (time.perf_counter() - started) * 1000))
        return mismatched

    def check_extent_cap(self, segment, prefix):
        """S2's block backstop (design 2.4): a live segment commits at most accept_limit(its round start)
        rows. The session's own cap (GreedySession.commit max_rows, serving_packed_step.commit_entry) is
        the control and makes this unreachable (test_packed_extent_step's property test); a firing still
        fails every request of the round (fail_round), so it is logged first. Host only, and called
        before commit_user's try, so a refusal leaves the block verified - fail_round then releases every
        segment at prefix 0 - rather than failed."""
        if prefix == 0 or segment in self.idle_segments:
            return
        start = self.round_starts.get(segment)
        limit = None if start is None else self.accept_limit(start)
        if limit is None or prefix > limit:
            self.extent_counts['cap_refused'] += 1
            diagnostic('%s round=%d segment=%d start=%s prefix=%d limit=%s'
                       % (EXTENT_CAP_REFUSED_MARKER, self.rounds, segment, start, prefix, limit))
            raise ValueError('Packed extent segment %d at %s commits at most %s rows; prefix %d refused'
                             % (segment, start, limit, prefix))

    def extent_attention(self):
        """describe()'s attention entry for the extent block."""
        operations = self.operations
        reader = getattr(self.fixture, 'replay_reader', None) if self.fixture is not None else None
        flags = [value for own in getattr(reader, 'readers', ()) for value in getattr(own, 'sdpa_modes_applied', ())]
        return dict(reader='per-user extent replay', capacity=self.replay_capacity, mask='narrow',
                    replay_group_rows=self.replay_group_rows, flags=['0x%x' % value for value in flags],
                    bundles_per_user=[len(tables) for tables in self.replay_tables],
                    tables=[[list(address) for address in tables] for tables in self.replay_addresses],
                    cur_pos=[[list(addresses(operations, value)) for value in values] for values in self.extent_cur_pos],
                    audit=self.extent_audit, deadline_s=None if self.deadline is None else self.deadline.seconds,
                    **self.extent_counts)

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
            self.checkpoints[0], self.block_rows, pack=pack,
            # S2: the extent readers over the pool's full-width tables and cur_pos words, else today's.
            **({'packed_extent': self.extent_storage} if getattr(self, 'extent', False)
               else {'packed_replay_pages': self.replay_tables}),
            serial_sdpa=True, compact_gdn=True, reuse_gdn_input=True,
            skip_row_clones=True, hoist_row_layout=True, device_loop_gdn=True, compact_prologue=True,
            batch_conv=True, packed_checkpoints=True, retain_records=True, ordered_cache=True,
            norm_batch=True, attention_replay=True, attention_mask_once=self.mask_once,
            replay_group_rows=self.replay_group_rows,
            short_context=False, attention_audit=False, commit_only_gdn=True,
            **({'kv_single_chain': True} if warm and self.kv_chains_cut else {}))

    def operation(self, fixture, warm=False):
        logits = None
        try:
            with ExitStack() as captures:
                captures.enter_context(self.feature_capture.capture())
                logits = fixture.run(sharded_logits=True)
            if self.shard_argmax:
                return (logits, *self.sample_shards(logits, warm=warm))
            ids = sample_rows(self.sampler, logits, self.block_rows, self.operations, native_rows=False)
            return logits, ids
        except BaseException:
            if logits is not None:
                self.operations.deallocate(logits)
            raise

    def sample_shards(self, logits, warm=False):
        """QWEN_FAST_VERIFY_T1 (#8a): each chip's argmax and max over its own vocab shard, no
        gather; under QWEN_FAST_VERIFY_T1_AUDIT also the pinned sampler's ids, for the audit. The
        sampler arms run the same pinned sampler call without the audit: PREWARM in the eager warm
        forward only (the result is released with the warm forward's outputs), IN_TRACE in the warm
        forward and the capture (output[3] is held like the audit's, never read back here)."""
        ids, values = verify_trace_t1.sample_shards(self.operations, logits, self.block_rows)
        if not (self.shard_audit or self.sampler_in_trace or (warm and self.sampler_prewarm)):
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
        if len(id_parts) != tp_shapes.chip_count() or len(value_parts) != tp_shapes.chip_count():
            raise AssertionError('%s chip-local outputs required' % tp_shapes.count_word())
        reads_started = time.perf_counter()
        if os.environ.get('QWEN_FAST_BATCHED_READS', '0') != '0':
            # tp4/fx-wph QWEN_FAST_BATCHED_READS (batched_reads_tp, imported only with the flag): the same eight shards as two mesh reads (or non-blocking copies and one
            # fence); the per-chip tensors it hands back are the served reads' bytes, cut the same way.
            import batched_reads_tp

            chip_ids, chip_values = batched_reads_tp.verify_reads(self.operations, self.mesh, ids, values, self.block_rows)
        else:
            chip_ids = [self.operations.to_torch(part).reshape(-1)[:self.block_rows] for part in id_parts]
            chip_values = [self.operations.to_torch(part).reshape(-1)[:self.block_rows] for part in value_parts]
        reads_ms = (time.perf_counter() - reads_started) * 1000
        if any(len(value) != self.block_rows for value in (*chip_ids, *chip_values)):
            raise AssertionError('Missing packed prediction rows')
        host = verify_trace_t1.combine_shards(chip_ids, chip_values).tolist()
        audit_sampdraft(self.operations, self.output, self.block_rows, chip_ids, chip_values)
        audit_shard_values(self.operations, self.output, self.block_rows, chip_values)
        if self.shard_audit:
            reference = self.operations.to_torch(self.operations.get_device_tensors(self.output[3])[0])
            verify_trace_t1.audit_round(host, reference.reshape(-1)[:self.block_rows].tolist())
        # tp4/hostgap stage 0: the eight reads against the combine and the audits (read by note_hostgap_verify).
        self.readback_split = (reads_ms, (time.perf_counter() - reads_started) * 1000 - reads_ms)
        return host

    def note_hostgap_verify(self, segments, snapshot, input_ms, bind_ms, stage_cpu_ms, rest_cpu_ms, readback_ms,
                            readback_cpu_ms=0.0):
        """Stage 0 (QWEN_FAST_TP4_HOSTGAP_LOG): this verify's staging path and its host split, on a line of its own - the block, the
        path the verify-time stage took, bind and input wall time beside the staging's thread CPU time (GC and host compute count
        there, a descheduled thread does not), the prediction readback split into its reads and its combine and audits
        (shard_predictions; both 0 on the unsharded readback), and the CPU time of everything after the staging up to the replay's end (after_stage_cpu_ms: the fence and the trace's
        blocking wait, which can spin) and, apart, of the readback (readback_cpu_ms). Never raises."""
        try:
            prestaged = self.prestaged
            path = 'off' if prestaged is None else prestaged.last['path']
            reason = '-' if prestaged is None else str(prestaged.last['reason']).replace(' ', '_')[:120]
            reads_ms, checks_ms = getattr(self, 'readback_split', (0.0, 0.0))
            diagnostic('%s block=%s round=%d live=%d path=%s reason=%s bind_ms=%.2f input_ms=%.2f stage_cpu_ms=%.2f reads_ms=%.2f '
                       'checks_ms=%.2f readback_ms=%.2f after_stage_cpu_ms=%.2f readback_cpu_ms=%.2f'
                       % (verify_prestage.HOSTGAP_VERIFY_MARKER, verify_prestage.block_label(self), self.rounds + 1,
                          len(segments), path, reason, bind_ms, input_ms, stage_cpu_ms, reads_ms, checks_ms, readback_ms,
                          rest_cpu_ms, readback_cpu_ms))
        except Exception:
            pass

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
        if self.sampler_prewarm or self.sampler_in_trace or sampler_arm_requested():
            diagnostic('%s prewarm=%d in_trace=%d requested=%d shard_argmax=%d audit=%d'
                       % (SAMPLER_ARM_MARKER, int(self.sampler_prewarm), int(self.sampler_in_trace),
                          int(sampler_arm_requested()), int(self.shard_argmax), int(self.shard_audit)))

    def note_vglue(self, counts):
        """tp4_vglue.ENGAGED once per captured verify trace, when any lever is on: what the captured forward engaged
        (attention layers folded, sampler value gathers) and what fell back."""
        levers = tp4_vglue.engaged_levers()
        if not levers:
            return
        diagnostic(tp4_vglue.marker('packed_verify', levers=','.join(name.replace('QWEN_FAST_TP4_', '').lower() for name in levers),
                                    **{name: counts[name] for name in sorted(counts)}))

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
        if getattr(self, 'extent', False):
            # S2: the pool's cur_pos words, and each extent reader's own word and narrow masks.
            if [[addresses(self.operations, value) for value in values] for values in self.extent_cur_pos] \
                    != self.cur_pos_addresses:
                raise ValueError('A pooled extent cur_pos word moved under the packed block')
            if (self.reader_addresses is not None and self.fixture is not None
                    and self.extent_reader_addresses() != self.reader_addresses):
                raise ValueError("An extent reader's positions word or narrow mask moved under the packed block")

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
                index = page_zero_index(table, start, self.rows_per_user,
                                        **({'extent': True} if getattr(self, 'extent', False) else {}))
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
        are not distinct segments of this block, and a third idle segment.

        S2's extent block: F = 0, so an idle segment is an ordinary user at start 0 or 32 in family
        E = 256 on the zero table (design 1.4 #1) - the same page-0 tile row as F = C - 256, since C
        is a multiple of 64 - and its SDPA reads one 256-key chunk instead of the whole table."""
        import torch

        live = tuple(live_segments)
        if len(set(live)) != len(live) or any(type(segment) is not int or not 0 <= segment < self.users
                                              for segment in live):
            raise ValueError('Live segments must be distinct segments of this %d-user block: %r' % (self.users, live))
        idle = [segment for segment in range(self.users) if segment not in live]
        if len(idle) > self.MAX_IDLE_SEGMENTS:
            raise ValueError('At most %d idle segments: page 0 holds two 32-row tile rows, so idle segments %s '
                             'would share one' % (self.MAX_IDLE_SEGMENTS, idle))
        first = 0 if getattr(self, 'extent', False) else self.replay_capacity - 256
        return {segment: ((1,) * self.rows_per_user, first + 32 * (index % 2),
                          torch.zeros((1, self.shape.page_width), dtype=torch.int32))
                for index, segment in enumerate(idle)}

    def verify(self, entries):
        """One trace for every entry. Returns (predictions, metrics): predictions[i] is
        entries[i]'s rows_per_user target ids; metrics['segments'][i] is entries[i]'s segment."""
        entries = list(entries)
        if getattr(self, 'deferred_commits', None):
            # Round-fence plan H2's backstop: GDN commits deferred by the last round and never flushed
            # inside its execute_model (R1 broken) go ahead of this round's trace - logged, so the gate
            # fails the arm - rather than being lost.
            self.flush_commits('verify')
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
            if getattr(self, 'extent', False):
                # S2: any start the extent path admits, each at its own family; proposal_rows and
                # ineligible (serving_packed_step) asked the same admits first, so this is the backstop.
                if not self.admits(ticket.position):
                    raise ValueError('Packed ticket of request %s at %r is outside the extent path: 128 <= start and '
                                     'start + %d <= %d required' % (str(entry['request_id'])[:48], ticket.position,
                                                                  self.rows_per_user, self.replay_capacity))
                continue
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
        extent_users = None
        if getattr(self, 'extent', False):
            # S2: every segment's start this round, live and idle, before anything is staged - the
            # commit cap (check_extent_cap), the deadline's line and the extent audit read them.
            extent_users = self.segment_users(entries, segments)
            if idle:
                extent_users = self.padded_users(extent_users, segments)
            self.round_starts = {segment: user[1] for segment, user in enumerate(extent_users)}
        self.phase = 'verifying'
        self.validated_this_round = False
        try:
            binding_started = time.perf_counter()
            # QWEN_FAST_PRESTAGE (verify_prestage): a snapshot the fixture write epoch still vouches
            # for skips the binding check (the drafts' window ran it) and writes only what differs;
            # anything else is today's check and full stage_packed.
            snapshot = reason = None
            if self.prestaged is not None:
                snapshot, reason = self.prestaged.usable()
            if snapshot is None:
                self.validate_bindings()
            # QWEN_FAST_TP4_SDPA=multi bakes the readers' positions, table and cur_pos buffers into its programs at the attach; a replay never comes
            # back to Python, so a rebinding after the attach is only visible here, on the host, before the replay (a few identity comparisons).
            # A reader (or multi launch) REPLACED after the capture would answer None to rebound_reason (its own multi is None, or its own buffers are
            # the ones it was built on) while the trace replays programs built on the old one's buffers: the objects themselves are compared first.
            reader = getattr(self.fixture, 'replay_reader', None)
            captured = self.captured_reader
            if captured is not None and (reader is not captured[0] or getattr(reader, 'multi', None) is not captured[1]):
                raise RuntimeError('The fixture replay reader (or its multi launch) was replaced after the verify trace was captured: the trace replays programs '
                                   'built on the buffers of the old reader')
            rebound_reason = getattr(reader, 'rebound_reason', None)
            if rebound_reason is not None:
                rebound = rebound_reason()
                if rebound:
                    raise RuntimeError(rebound)
            started = time.perf_counter()
            hostgap = verify_prestage.hostgap_log_enabled()
            cpu_started = verify_prestage.thread_ms() if hostgap else 0.0
            with verify_prestage.hostgap_span('verify_stage', block=verify_prestage.block_label(self)):
                if self.prestaged is None:
                    staged = self.stage_packed_inputs(entries)
                else:
                    staged = self.prestaged.stage(entries, segments, snapshot, reason)
            staged_at = time.perf_counter()
            cpu_staged = verify_prestage.thread_ms() if hostgap else 0.0
            # Claimed before the trace: its segment restores rewrite slot 0, so no engine
            # may trust its residency from here on, even if the trace fails part way.
            note_packed_step()
            trace_ms = 0.0

            def operation():
                nonlocal trace_ms
                trace_started = time.perf_counter()
                with self.replay_deadline('verify', segments):
                    result = self.operations.execute_trace(self.mesh, self.trace, cq_id=0, blocking=True)
                trace_ms += (time.perf_counter() - trace_started) * 1000
                return result

            first = self.first
            if first:
                operation()
                self.operations.synchronize_device(self.mesh)
            else:
                if (snapshot is not None and verify_prestage.window_validate_enabled() and self.prestaged is not None
                        and self.prestaged.last['path'] == 'diff'):
                    # tp4/hostgap 1c (QWEN_FAST_TP4_WINDOW_VALIDATE): the window validated this block's bindings and the epoch
                    # vouches that nothing that can move a native buffer happened since, so the replay skips its own second check.
                    # Audited, the skipped check still runs, here, as a shadow (a failure raises exactly as the check would).
                    if self.prestaged.audit or self.prestaged.full_audit:
                        self.fixture.retained.validate_bindings()
                        diagnostic('[PACKED-PRESTAGE-SHADOW] round=%d retained_bindings=ok' % (self.rounds + 1))
                    with verify_prestage.skip_next_binding_check(self.fixture.retained):
                        self.fixture.retained.replay(operation)
                else:
                    self.fixture.retained.replay(operation)
                # QWEN_FAST_ROUND_FENCES: the replay validated every native binding right before
                # this trace, and nothing moves one between here and the round's commits.
                self.validated_this_round = self.round_fences
            tile_collective_tp.audit_replayed(self.fixture)
            mlp = mlp_audit()
            if mlp is not None:
                mlp.audit_replayed(self.fixture)
            replayed = time.perf_counter()
            cpu_replayed = verify_prestage.thread_ms() if hostgap else 0.0
            if self.shard_argmax:
                with verify_prestage.hostgap_span('readback', block=verify_prestage.block_label(self)):
                    host = self.shard_predictions()
            else:
                logits, ids = self.output
                parts = self.operations.get_device_tensors(ids)
                if len(parts) != tp_shapes.chip_count():
                    raise AssertionError('%s chip-local outputs required' % tp_shapes.count_word())
                host = self.operations.to_torch(parts[0]).reshape(-1)[:self.block_rows].tolist()
            if len(host) != self.block_rows:
                raise AssertionError('Missing packed prediction rows')
            if self.windows_audit:
                # QWEN_FAST_VERIFY_T2_AUDIT (G1 for #1): this replay's packed windows against the
                # served ones built beside them in the same trace - every GDN layer on round 1,
                # then two per round in rotation (verify_trace_t2.audit_layers).
                verify_trace_t2.audit_round(self.operations, self.fixture.retained.records, self.rounds + 1)
            if getattr(self, 'seq_block_audit', ()):
                # QWEN_FAST_GDN_SEQ_BLOCK_AUDIT: this replay's K5-A prefix states and gated output
                # against the served launch's on the same inputs, every audited layer.
                gdn_seq_block.audit_round(self.operations, self.fixture.retained.records, self.seq_block_audit,
                                          self.rounds + 1)
            if tp4_vglue.audit_enabled():
                # QWEN_FAST_TP4_VGLUE_AUDIT: each engaged GDN lever's output against the served path's, held beside it.
                # (the octo block's 8 x 8 users run the served path at the glue sites, tp4_vglue.EIGHT_ROW_SERVED: nothing of the levers to compare - unless
                # QWEN_FAST_OCTO_GLUE8 makes them native (octo_glue8), when the audited entries exist like an M3 block's and a declined lever is a failure)
                tp4_vglue.audit_round(self.operations, self.fixture.retained.records, self.rounds + 1,
                                      served_only=(self.users, self.rows_per_user) == (8, 8) and os.environ.get('QWEN_FAST_OCTO_GLUE8', '0') == '0')
                if os.environ.get('QWEN_FAST_TP4_CONV_GATES_SPREAD_AUDIT', '0') != '0':
                    # QWEN_FAST_TP4_CONV_GATES_SPREAD_AUDIT (gdn_conv_gates_spread): the F1 launch's three outputs (conv, beta, g) per audited layer, every chip.
                    import gdn_conv_gates_spread

                    gdn_conv_gates_spread.audit_round(self.operations, self.fixture.retained.records, self.rounds + 1)
            # QWEN_FAST_TP4_RS_UNIT_MAJOR_AUDIT: this replay's unit-major reductions against the split's held beside them.
            tile_collective_tp.audit_round(self.operations, self.fixture, self.rounds + 1)
            mlp = mlp_audit()
            if mlp is not None:
                # QWEN_FAST_MLP_*_AUDIT (tp4_mlp_gateup): this replay's MLP outputs against the served forward's, read right after the block's own replay
                mlp.audit_round(self.operations, self.fixture, self.rounds + 1)
            sdpa_audit = getattr(getattr(self.fixture, 'replay_reader', None), 'sdpa_audit_round', None)
            if sdpa_audit is not None:
                # QWEN_FAST_TP4_SDPA_AUDIT (sdpa_multi_tp, gate profiles only): the multi launch's counters against the per-user
                # launches run beside it in this replay. The reader twin's method returns 0 at once unless the audit is on.
                sdpa_audit(self.rounds + 1)
            if os.environ.get('QWEN_FAST_KV_PAGE_WRITER_AUDIT', '0') not in ('', '0'):
                # QWEN_FAST_KV_PAGE_WRITER_AUDIT (kv_page_writer_tp4, op-fusion WP2, gate profiles only): the page writers' counters after this replay, once per
                # replay - zero mismatched words and at least one valid unit per chip, or the mismatch line and a raise. Imported only with the flag.
                import kv_page_writer_tp4

                kv_page_writer_tp4.audit_round(self.operations, self.rounds + 1)
            predictions = [host[slice(*segment_rows(self.shape, segment))] for segment in segments]
            finished = time.perf_counter()
            if hostgap:
                self.note_hostgap_verify(segments, snapshot, (staged_at - started) * 1000,
                                         (started - binding_started) * 1000, cpu_staged - cpu_started,
                                         cpu_replayed - cpu_staged, (finished - replayed) * 1000,
                                         verify_prestage.thread_ms() - cpu_replayed)
            if extent_users is not None:
                # S2, after the replay and outside the round's phase timings: the executed path's line,
                # then (QWEN_FAST_EXTENT_AUDIT, gate profiles only) the read-back audit.
                self.note_extent_round(extent_users, segments, idle)
                if self.extent_audit:
                    self.audit_extent(extent_users, self.rounds + 1)
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
            if getattr(self, 'gdn_after_pairs', False):
                # Round-fence plan H2: this round's GDN commit traces are deferred when the early draft
                # armed the block for it (it flushes them inside this execute_model); else today's.
                self.deferring, self.defer_armed, self.deferred_commits = self.defer_armed, False, []
            for segment in idle:
                # A padded round's idle segment decides at prefix 0: no commit trace exists for
                # 0 and the retained block publishes nothing at it - no state, no carry, no K/V -
                # so this only gives the retained block the decision each segment owes before
                # the next replay. Before any live commit, so the round's last LIVE commit keeps
                # the synchronize=last fence (commit_user: the one that empties the pending set).
                self.commit_user(segment, 0)
            dump_device_profiler_after_round(self.operations, self.mesh, self.rounds)
            dump_device_profiler_every(self.operations, self.mesh)
            metrics = dict(segments=segments, staged_buffers=staged,
                binding_validation_ms=(started - binding_started) * 1000,
                input_ms=(staged_at - started) * 1000, verify_readback_ms=(finished - staged_at) * 1000,
                blocking_trace_host_ms=trace_ms, replay_checks_sync_ms=(replayed - staged_at) * 1000 - trace_ms,
                output_readback_host_ms=(finished - replayed) * 1000, users=self.users)
            if self.padded_min_users is not None:
                # QWEN_FAST_PADDED_BLOCK only: the round's live and idle segments (every round, the
                # all-live ones included), into each user's phase-timing record.
                metrics.update(live=len(segments), idle=list(idle))
            if self.round_fences:
                # QWEN_FAST_ROUND_FENCES only: how this round's replay was armed - by the drafts'
                # fence F9 ('f9'), by one fence at replay ('replay', its cost in replay_fence_ms),
                # or the first round's plain trace ('first') - and whether its commits may skip
                # the first validate (serving_packed_step's FENCES_MARKER line).
                retained = self.fixture.retained
                metrics.update(validated_this_round=self.validated_this_round,
                               replay_fence='first' if first else getattr(retained, 'replay_fence', None),
                               replay_fence_ms=0.0 if first else getattr(retained, 'replay_fence_ms', 0.0))
            if self.prestaged is not None:
                # QWEN_FAST_PRESTAGE only: this verify's path and split (verify_prestage.BlockPrestage.last).
                metrics.update(prestage=dict(self.prestaged.last))
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
        if getattr(self, 'deferring', False):
            # Round-fence plan H2 (QWEN_FAST_GDN_AFTER_PAIRS): decided now, enqueued by flush_commits
            # after the next round's pairs have been read back.
            self.deferred_commits.append((segment, prefix))
            return 0.0
        blocking = not self.pipelined_commits
        started = time.perf_counter()
        with self.replay_deadline('commit', (segment,), prefix):
            self.operations.execute_trace(self.mesh, self.commits[segment][prefix], cq_id=0, blocking=blocking)
        return (time.perf_counter() - started) * 1000

    def arm_deferred_commits(self):
        """Round-fence plan H2 (early_draft.EarlyDraft.execute, through serving_packed_step.PackedStep):
        defer the coming round's GDN commit traces - its caller flushes them inside the same
        execute_model. True when armed; False, and nothing armed, without QWEN_FAST_GDN_AFTER_PAIRS."""
        if not getattr(self, 'gdn_after_pairs', False):
            return False
        self.defer_armed = True
        return True

    def flush_commits(self, site='end'):
        """Round-fence plan H2: enqueue the round's deferred GDN commit traces, in decision order and
        today's blocking mode (QWEN_FAST_PIPELINED_COMMITS: enqueued), then re-owe the retained block's
        fence (gdn_records.RetainedGDNBlock.note_deferred_publications) so the next replay pays it. Also
        disarms a round that never verified. `site` says where: 'window' (after the pairs' readback),
        'end' (the end of the early draft), 'reconcile' or 'verify' (R1's backstops). A block that
        failed or closed drops what it held (logged): its requests fail with it. Returns how many
        traces were enqueued."""
        self.defer_armed = False
        pending = list(getattr(self, 'deferred_commits', None) or ())
        self.deferring, self.deferred_commits = False, []
        if not pending:
            return 0
        import early_draft

        fixture = self.fixture
        retained = getattr(fixture, 'retained', None) if fixture is not None else None
        segments = ','.join('%d:%d' % pair for pair in pending)
        if (self.phase in ('failed', 'closed') or retained is None or getattr(retained, 'poisoned', False)
                or getattr(retained, 'closed', False)):
            diagnostic('%s round=%d commits=0 site=%s enqueue_ms=0.00 segments=%s dropped=%d reason=block-%s'
                       % (early_draft.GDN_MARKER, self.rounds, site, segments, len(pending), self.phase))
            return 0
        blocking = not self.pipelined_commits
        started = time.perf_counter()
        try:
            with self.replay_deadline('flush', [segment for segment, prefix in pending]):
                for segment, prefix in pending:
                    self.operations.execute_trace(self.mesh, self.commits[segment][prefix], cq_id=0, blocking=blocking)
        except BaseException:
            # A half-enqueued round of carries cannot be replayed over (gdn_records' own rule).
            retained.poisoned = True
            self.phase = 'failed'
            raise
        finally:
            note_packed_step()
        retained.note_deferred_publications()
        diagnostic('%s round=%d commits=%d site=%s enqueue_ms=%.2f segments=%s' % (
            early_draft.GDN_MARKER, self.rounds, len(pending), site, (time.perf_counter() - started) * 1000,
            segments))
        return len(pending)

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
        if getattr(self, 'extent', False):
            # S2: no live segment commits a row at or past its family end (design 2.4). Before the try:
            # a refusal must not also mark the block failed.
            self.check_extent_cap(segment, prefix)
        # Each commit trace already blocks the host; the one fence that matters is the
        # last user's, which arms the retained block's replay for the next round.
        last = self.pending_segments == {segment}
        commit_ms = 0.0

        def publish(selected):
            nonlocal commit_ms
            commit_ms = self.execute_commit(segment, selected) or 0.0

        try:
            call_started = time.perf_counter()
            if self.round_fences:
                # QWEN_FAST_ROUND_FENCES: the last commit leaves its fence to the drafts' F9 (or the
                # next replay), and the first skips its validate when this round's replay ran one.
                self.fixture.retained.commit_user(segment, prefix, dma=True, synchronize=last, publication=publish,
                                                  validated_this_round=self.validated_this_round)
            else:
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
                if os.environ.get('QWEN_FAST_PACKED_AUDIT') == '1' and not round_host.lean_enabled():
                    # sync_ms is what the LAST segment's own call spent beyond its own
                    # commit_ms: bookkeeping plus (when pipelined) the trailing fence that
                    # drains the whole round's four enqueued traces; near zero when blocking,
                    # since the trace replay above already drained the queue.
                    sync_ms = self.commit_block_ms[segment]
                    # mode=deferred: round-fence plan H2 held this round's traces for flush_commits.
                    diagnostic('[PACKED-COMMIT] round=%d mode=%s commit_ms=[%s] sync_ms=%.2f'
                               % (self.rounds, 'deferred' if getattr(self, 'deferring', False)
                                  else 'pipelined' if self.pipelined_commits else 'blocking',
                                  ','.join('%.2f' % value for value in self.commit_timings), sync_ms))
        except BaseException:
            self.phase = 'failed'
            raise
        finally:
            # Slot 0 holds this segment's user, or after prefix 0 nobody's committed state.
            note_packed_step()

    def fence_token(self):
        """QWEN_FAST_ROUND_FENCES: the retained block's commit count, taken by the drafts' window
        (verify_prestage.WhileWaiting) BEFORE its fence F9, so that fence arms only the commits
        enqueued ahead of it."""
        return self.fixture.retained.commit_serial

    def note_round_fence(self, token):
        """QWEN_FAST_ROUND_FENCES: F9 has just drained CQ0 behind every commit `token` counted -
        arm the next replay (gdn_records.RetainedGDNBlock.note_round_fence). False, and nothing
        armed, without the flag or when anything moved since the token."""
        if not self.round_fences or self.fixture is None:
            return False
        return self.fixture.retained.note_round_fence(token)

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
            attention=self.extent_attention() if getattr(self, 'extent', False) else
            dict(reader='per-user bundled replay', family=self.replay_capacity, replay_group_rows=self.replay_group_rows,
                           bundles_per_user=[len(tables) for tables in self.replay_tables],
                           tables=[[list(address) for address in tables] for tables in self.replay_addresses]),
            setup_ms=getattr(self, 'setup_ms', None), rounds=self.rounds,
            **({} if self.padded_min_users is None else dict(padded=dict(
                min_users=self.padded_min_users, max_idle=self.MAX_IDLE_SEGMENTS,
                carries_in_place=self.carries_in_place, rounds=self.padded_rounds))),
            **({} if self.prestaged is None else dict(prestage=dict(self.prestaged.counts, audit=self.prestaged.audit))),
            **({} if not self.round_fences else dict(round_fences=True)),
            **({} if self.fused is None else dict(fused_commit=self.fused.describe())),
            **({} if getattr(self, 'publication_warm', None) is None else dict(publication_warm=dict(
                self.publication_warm, program_cache=list(self.publication_warm['program_cache'])))),
            **({} if not getattr(self, 'gdn_after_pairs', False) else dict(gdn_after_pairs=True)))

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
        if getattr(self, 'fused', None) is not None:
            # H1b: its traces (abandoned without the fence, like the block's own) and its buffers.
            abandoned += self.fused.trace_count()
            self.fused.close(wait=wait)
            self.fused = None
        if not wait:
            diagnostic('[PINDIAG] packed block closed without the device fence at stage %s; %d captured trace(s) abandoned'
                       % (self.stage, abandoned))
        if self.feature_capture is not None:
            self.feature_capture.close()
        shard_values = self.output[2] if self.output is not None and len(self.output) > 2 else None
        if self.output is not None:
            release_owned(operations, [value for value in self.output if value is not None])
            self.output = None
        release_sampdraft_audit(operations, shard_values)
        if getattr(self, 'sampdraft_reserved', False):
            # after the traces that use it (released above, or abandoned by an unfenced close, whose buffers are freed like this one)
            release_sampdraft_reserve(operations, self.mesh)
            self.sampdraft_reserved = False
        if self.fixture is not None:
            release_vglue_audit(operations, self.fixture, shard_values)
            tile_collective_tp.audit_release(operations, self.fixture)
            mlp = mlp_audit()
            if mlp is not None:
                mlp.audit_release(operations, self.fixture)
            self.fixture.close()
            self.fixture = None
        if self.plug is not None:
            # After the traces and the captured buffers: the plugs sat under them. A block closed without the device fence
            # (a failed attach: the device may still run work above these blocks) frees nothing, ever.
            if wait:
                self.plug.close()
                if self.plug in capture_plug.PACKED:
                    capture_plug.PACKED.remove(self.plug)
            else:
                self.plug.abandon()
            self.plug = None
        # The readers' page tables are the pool's: handed back, never freed here.
        if getattr(self, 'replay', None) is not None:
            self.replay.release()
            self.replay = None
        self.replay_tables, self.replay_addresses = [], []
        if getattr(self, 'deadline', None) is not None:
            # S2: the extent block's watchdog thread (the pool's cur_pos words went back above).
            self.deadline.close()
        if getattr(self, 'extent', False):
            self.extent_storage, self.extent_cur_pos, self.cur_pos_addresses = None, [], []
        release_owned(operations, self.taps)
        release_owned(operations, [value for checkpoints in self.checkpoints for snapshot in checkpoints for value in snapshot])
        release_owned(operations, [value for snapshot in self.initial for value in snapshot])
        self.taps, self.checkpoints, self.initial = [], [], []
        # The carries are the pool's: lent to the requests, never freed here.
        self.carries, self.carry_addresses = [], []
        self.pending_segments.clear()
        self.phase = 'closed'
