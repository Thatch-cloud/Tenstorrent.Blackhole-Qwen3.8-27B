"""Verify-trace tuning T1 (QWEN_FAST_VERIFY_T1=1): the exact cuts of waves 1 and 2 of the
verify-trace tuning spec, for the packed 64-row verify (four T16 users, K64f + C1c).

Off by default and read at each use, never at import. With the flag unset every path below
is the one that ran before it existed. The cuts, where they live, and why each is exact:

  #11 (wave 1, graft)  lever_n_m3native_patch section A2: the attn_qkv and MLP gate M = 64
      matmul configs on a wider N partition. Same builder, in0_block_w, fused activation and
      call-site compute config, so every output tile is reduced over K in the same blocks on
      one core (class B*, byte compare in verify_t1_device_compare.py).
  #10 packed_verifier.build_fixture: the replay readers' attention masks refreshed once per
      forward (model_batch's shared_masks) instead of before every attention layer. The mask
      is a pure function of each reader's positions word, which is staged before the replay
      and nothing inside it writes, so every SDPA call reads the same mask bytes (class B).
  #3  gdn_device_loop_state._recurrence_user_batched: every packed user's recurrence and
      windows read that user's CARRY directly instead of a block-start entry copied from it.
      The entry was a byte copy of the carry (copy_compact, or restore + save through native
      slot 0 for the last user) and only the launch reads it, so the recurrence reads the
      same bytes from another address (class B). Nothing inside the verify trace writes a
      carry (the commit traces do, after the readback). The entries stay allocated: the
      commit DMA's 20-tensor list names them, but gdn_commit_dma.cpp reads an entry only at
      prefix 0, and prefix 0 never publishes (gdn_records.commit_user returns first;
      packed_verifier captures prefixes 1..rows only). Dropping the last user's restore/save
      as well is safe because native slot 0 has no reader that trusts it after a packed
      step: verifier_engine's residency is cleared before the verify trace and after every
      packed commit (note_packed_step), so any later sequential step restores its own carry
      first; the commit DMA only writes slot 0; and every commit at prefix > 0 rewrites slot
      0 in full afterwards anyway (see test_verify_trace_t1.DirectCarryTests).
  #12 gdn_user_batch.build_program: the batched recurrence program built from rectangle core
      ranges, with one kernel descriptor per role over all four users' cores whenever their
      compile-time arguments are identical. Every core keeps its kernel, its compile-time
      arguments and its runtime arguments byte for byte (class B).
  #8a packed_verifier.operation: the sampling tail as a per-chip argmax over the local vocab
      shard plus the max value, combined on the host (chip 1 only when its max is strictly
      greater, NaN handled as torch.argmax does), instead of the pinned sampler's AllGather,
      untilize and ArgMax over the gathered 248320-wide row, twice. The pinned sampler equals
      torch.argmax's first occurrence over the whole row (sampling-links.py, including cross-
      shard ties), and first occurrence over [shard 0 | shard 1] is exactly this combine.
      Class B*: the device ArgMax's first-occurrence rule within a shard and ttnn.max's value
      must be confirmed on hardware - QWEN_FAST_VERIFY_T1_AUDIT=1 runs the pinned sampler
      beside it in the same trace and compares every row, every round.

Markers: packed_verifier logs VERIFY_T1_MARKER once per captured verify trace with what
engaged; the model_config graft logs it (site=matmul_configs) where the configs are built.
The m3native gate requires it whenever the arm passes the flag.

QWEN_FAST_VERIFY_T1_SKIP (read only while the flag is on, so flag-off stays byte-identical):
a comma list of CUTS to leave out, so one image can bisect the cuts and a wave-1 arm can
declare that it expects no wave-2 cut. 'last_carry' leaves out only #3's second half (the
last user's restore/save through native slot 0); 'direct_carry' leaves out all of #3. An
unknown name raises: a typo must never silently run every cut.
"""

import os

FLAG = 'QWEN_FAST_VERIFY_T1'
AUDIT_FLAG = 'QWEN_FAST_VERIFY_T1_AUDIT'
SKIP_FLAG = 'QWEN_FAST_VERIFY_T1_SKIP'
MARKER = '[PINDIAG] verify t1 engaged'
AUDIT_MARKER = '[PINDIAG] verify t1 audit'
AUDIT_MISMATCH = '[PINDIAG] verify t1 audit mismatch'
KEPT_SAMPLER = '[PINDIAG] verify t1 kept the pinned sampler'
# #11 (the graft), #10, #3, #3's second half, #12, #8a.
CUTS = ('matmul_configs', 'mask_once', 'direct_carry', 'last_carry', 'coalesce', 'shard_argmax')
# What the packed verify engages (all but the graft's #11): skipping all of these is a wave-1 arm.
WAVE2_CUTS = ('mask_once', 'direct_carry', 'coalesce', 'shard_argmax')

# The TP2 Qwen vocabulary: one 124160-wide shard per chip, 248320 in all, no padding.
SHARD_WIDTH = 124160
VOCABULARY = 2 * SHARD_WIDTH

_COUNTS = {}
_AUDIT = dict(rounds=0, rows=0)


def enabled():
    """QWEN_FAST_VERIFY_T1=1."""
    return os.environ.get('QWEN_FAST_VERIFY_T1') == '1'


def audit_enabled():
    """QWEN_FAST_VERIFY_T1_AUDIT=1 beside QWEN_FAST_VERIFY_T1=1: a correctness arm, never a timed one."""
    return enabled() and os.environ.get('QWEN_FAST_VERIFY_T1_AUDIT') == '1'


def skipped():
    """The cuts QWEN_FAST_VERIFY_T1_SKIP names (empty when the flag is off). Raises on a name
    that is not a cut."""
    if not enabled():
        return frozenset()
    names = frozenset(name.strip() for name in os.environ.get('QWEN_FAST_VERIFY_T1_SKIP', '').split(',') if name.strip())
    unknown = sorted(names.difference(CUTS))
    if unknown:
        raise ValueError('%s names no cut: %s (cuts: %s)' % (SKIP_FLAG, ','.join(unknown), ','.join(CUTS)))
    return names


def cut(name):
    """QWEN_FAST_VERIFY_T1=1 and `name` not in QWEN_FAST_VERIFY_T1_SKIP."""
    if name not in CUTS:
        raise ValueError('Unknown verify t1 cut %r' % (name,))
    return enabled() and name not in skipped()


def note(cut, count=1):
    """Count one engagement of `cut` (reported, then reset, by the next take())."""
    _COUNTS[cut] = _COUNTS.get(cut, 0) + count


def take():
    """The engagements counted since the last take(), and a fresh count."""
    counts = dict(_COUNTS)
    _COUNTS.clear()
    return counts


def log_line(message):
    """One line into the server log: loguru where it exists, stdout otherwise. Never raises."""
    try:
        try:
            from loguru import logger
        except ImportError:
            print(message, flush=True)
        else:
            logger.info('{}', message)
    except BaseException:
        pass


def engaged_line(site, **fields):
    return '%s site=%s %s' % (MARKER, site, ' '.join('%s=%s' % (name, fields[name]) for name in sorted(fields)))


# --------------------------------------------------------------------------------------
# #12: rectangle core ranges.
# --------------------------------------------------------------------------------------

def rectangles(points):
    """Disjoint inclusive (x0, y0, x1, y1) rectangles covering exactly `points`.

    Column runs of consecutive y first, then runs of equal y-span in adjacent columns merged.
    gdn_user_batch.core_shares puts worker w at (w // rows, w % rows), so one user's 24
    workers are at most three rectangles and all four users' 96 are two."""
    points = [tuple(point) for point in points]
    if not points or len(set(points)) != len(points):
        raise ValueError('Distinct worker cores required')
    columns = {}
    for horizontal, vertical in points:
        columns.setdefault(horizontal, []).append(vertical)
    runs = []
    for horizontal in sorted(columns):
        verticals = sorted(columns[horizontal])
        first = previous = verticals[0]
        for vertical in verticals[1:]:
            if vertical != previous + 1:
                runs.append((horizontal, first, previous))
                first = vertical
            previous = vertical
        runs.append((horizontal, first, previous))
    merged = []
    for horizontal, first, last in runs:
        for index, (x0, y0, x1, y1) in enumerate(merged):
            if x1 == horizontal - 1 and (y0, y1) == (first, last):
                merged[index] = (x0, y0, horizontal, y1)
                break
        else:
            merged.append((horizontal, first, horizontal, last))
    covered = sorted((x, y) for x0, y0, x1, y1 in merged for x in range(x0, x1 + 1) for y in range(y0, y1 + 1))
    if covered != sorted(points):
        raise AssertionError('Rectangles must cover exactly the worker cores')
    return merged


def rectangle_set(operations, points):
    return operations.CoreRangeSet([operations.CoreRange(operations.CoreCoord(x0, y0), operations.CoreCoord(x1, y1))
                                    for x0, y0, x1, y1 in rectangles(points)])


# --------------------------------------------------------------------------------------
# #8a: per-shard argmax with a host combine.
# --------------------------------------------------------------------------------------

def shard_sampling_problem(sampler):
    """Why the per-shard argmax cannot stand in for the pinned sampler, or None.

    The same guard as force_argmax.sample_rows' native-row experiment: the pinned greedy
    force-argmax sampler, the unpadded TP2 Qwen vocabulary, and nothing that the sampler would
    apply beyond an argmax (penalties, log-probabilities, a request seed)."""
    tt_sampling = getattr(sampler, 'tt_sampling', None)
    if tt_sampling is None or not getattr(tt_sampling, 'force_argmax_sampling', False):
        return 'the pinned sampler is not the greedy force-argmax sampler'
    if getattr(tt_sampling, 'vocab_size', None) != VOCABULARY or getattr(tt_sampling, 'padded_vocab_size', None) != VOCABULARY:
        return 'vocabulary %r padded %r is not the unpadded TP2 %d' % (
            getattr(tt_sampling, 'vocab_size', None), getattr(tt_sampling, 'padded_vocab_size', None), VOCABULARY)
    if getattr(sampler, '_penalties_active', True) or getattr(sampler, '_log_probs_active', False):
        return 'penalties or log-probabilities are active'
    seeds = getattr(sampler, 'seed_manager', None)
    if seeds is None or seeds.has_active_request_seed():
        return 'a request seed is active'
    return None


def sample_shards(operations, logits, rows):
    """Per chip, on its own vocab shard: the first-occurrence argmax index (uint32) and the max
    value (bf16), each (1, 1, rows, 1). Nothing crosses chips; the host combines.

    Only bf16 TILE logits: a block-float shard would come back from ttnn.max packed with a
    shared exponent, which can round the max and flip a near tie across shards."""
    shape = tuple(logits.shape)
    if len(shape) != 4 or shape[:3] != (1, 1, rows) or shape[3] != SHARD_WIDTH:
        raise ValueError('Per-shard argmax needs the pre-gather (1, 1, %d, %d) vocab shard; got %r'
                         % (rows, SHARD_WIDTH, shape))
    dtype, layout = getattr(logits, 'dtype', None), getattr(logits, 'layout', None)
    if dtype != operations.bfloat16 or layout != operations.TILE_LAYOUT:
        raise ValueError('Per-shard argmax needs bf16 TILE logits; got %r %r' % (dtype, layout))
    dram = operations.DRAM_MEMORY_CONFIG
    row_major = operations.to_layout(logits, operations.ROW_MAJOR_LAYOUT, memory_config=dram)
    ids = values = None
    try:
        ids = operations.argmax(row_major, dim=3, keepdim=True, use_multicore=True, memory_config=dram)
        values = operations.max(logits, dim=3, keepdim=True, memory_config=dram)
        return ids, values
    except BaseException:
        if ids is not None:
            operations.deallocate(ids)
        raise
    finally:
        if row_major is not logits:
            operations.deallocate(row_major)


def combine_shards(chip_ids, chip_values, shard_width=SHARD_WIDTH):
    """The first-occurrence argmax over [shard 0 | shard 1] from each shard's own argmax and max.

    Shard 1 wins a row only when its max is strictly greater than shard 0's, so a tie keeps the
    earlier index, as torch.argmax (and the pinned sampler, sampling-links.py) do. -0 and +0
    compare equal. NaN is greater than everything, as torch.argmax treats it: shard 1 wins only
    when its max is NaN and shard 0's is not. Returns an int64 tensor of global vocabulary ids."""
    import torch

    if len(chip_ids) != 2 or len(chip_values) != 2:
        raise ValueError('Two chip-local vocab shards required')
    ids = [torch.as_tensor(value).reshape(-1).to(torch.int64) for value in chip_ids]
    values = [torch.as_tensor(value).reshape(-1).to(torch.float32) for value in chip_values]
    rows = ids[0].numel()
    if rows == 0 or any(value.numel() != rows for value in (*ids, *values)):
        raise ValueError('Every shard must report one id and one value per row')
    if any(bool(((value < 0) | (value >= shard_width)).any()) for value in ids):
        raise ValueError('A shard-local argmax lies outside its %d-wide shard' % shard_width)
    first, second = values
    later = (second > first) | (torch.isnan(second) & ~torch.isnan(first))
    return torch.where(later, ids[1] + shard_width, ids[0])


def audit_round(combined, reference):
    """QWEN_FAST_VERIFY_T1_AUDIT: the combined ids against the pinned sampler's, every row.
    Logs AUDIT_MARKER '<n> exact=True rows=<total>' or AUDIT_MISMATCH and raises."""
    combined, reference = [int(value) for value in combined], [int(value) for value in reference]
    if len(combined) != len(reference):
        raise AssertionError('The audit compares %d combined rows with %d sampled' % (len(combined), len(reference)))
    rows = [row for row, (mine, pinned) in enumerate(zip(combined, reference)) if mine != pinned]
    if rows:
        message = '%s round=%d rows=%s shard=%s sampler=%s' % (
            AUDIT_MISMATCH, _AUDIT['rounds'] + 1, rows[:8], [combined[row] for row in rows[:8]],
            [reference[row] for row in rows[:8]])
        log_line(message)
        raise AssertionError(message)
    _AUDIT['rounds'] += 1
    _AUDIT['rows'] += len(combined)
    log_line('%s %d exact=True rows=%d' % (AUDIT_MARKER, _AUDIT['rounds'], _AUDIT['rows']))
    return True
