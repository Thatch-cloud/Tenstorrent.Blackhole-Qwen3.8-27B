"""QWEN_FAST_PAIR_ROW_EXACT (default off): the packed pair's draft SDPA folded, so that a user in pair row 1
attends exactly as it would drafting alone.

THE EFFECT (h1a-draft-race.md sections 2-3). The pair drafter drafts a user in pair ROW 1 (block rows 16-31)
differently from the same user in row 0 or drafting alone. The difference is deterministic and does not depend
on the partner. Row 1 is never better and sometimes clearly worse. Mean emitted per packed round, row 0 or alone
against row 1: u3 at 131k 6.47 vs 5.40, u0 at 32k 5.43 vs 5.27, u3 at 32k 7.93 vs 7.57. Drafting alone equals
row 0 exactly. Verification is exact, so the loss is acceptance, never text.

THE MECHANISM (M1, by elimination). The pair's draft SDPA (dflash_t16_native_attention.attention, which calls
draft_attention.draft_sdpa) is ONE call over K = 4160 keys: two 2080-key segments [a | b], k_chunk 32,
q_chunk 32, so all 32 rows share one work unit and one key order. Row 0's own keys come first. Row 1's first 65
chunks are fully masked, so its first live chunk runs the online-softmax merge against a max, sum and output
carried from 65 masked chunks, where the single-user trace (K = 2080) starts fresh. Every other op on rows
16-31 is row-local or an exact bf16 copy, and row 1's host inputs (mask block, query and live-key RoPE) are
bit-identical to the single-user trace's. The card-B probe (optimisation/ttnn-op/pair_row_probe) settles the
residue itself.

THE FIX (R1g, the GQA fold). Today's K/V assembly is kept unchanged and reinterpreted with one KV head per user
segment. Per chip (TP2: 16 query heads, 4 KV heads, group 4):
  K, V  (1, 4, 4160, 128) -> (1, 8, 2080, 128)    a view: KV head 2h+u is user u's segment of head h
  Q     (1, 16, 32, 128)  -> (1, 32, 32, 128)     Q head 8h+4u+j is head 4h+j with user u's rows at rows 0-15
                                                  (u=1: rows 16-31 moved up to 0-15, rows 0-15 down to 16-31)
  mask  the served single-user mask (1, 1, 32, 2080) = batched_attention_mask([2048], 16), over every head
  SDPA  NQH 32, NKH 8: group 4 again, so Q head 8h+4u+j reads KV head 2h+u; draft_sdpa's configuration verbatim
  unfold  O (1, 32, 32, 128) -> (1, 16, 32, 128): rows 0-15 of the u=0 heads, then rows 0-15 of the u=1 heads
Every (Q head, KV head) work unit is then the single-user trace's unit for that user: the same 16 query rows at
the same tile rows, the same K/V at every visible key, the same mask tiles, chunks 0..64 in order, the same
kernel, program configuration and buffer formats. Only two things differ, and neither is read: the values of
masked keys (row 0 == alone already proves they are irrelevant; pair row 0's segment pad carries its own live
rows 0-15 and the 65 trailing chunks the partner's keys) and the pad query rows 16-31, which are row-local and
discarded.

COST. About seven small ops per layer (the reshapes are views): two slices and a concat to shift Q, a concat to
fold it, two slices and a concat to unfold the output, all on tensors of 128 KB or less. The SDPA goes from 16
work units x 130 chunks to 32 units x 65 chunks: its per-core loop halves (global Q scheduling spreads the 32
units over 32 of the 64 cores). The per-round mask refresh uploads 133 KB instead of 266 KB. The probe times the
pair pass both ways. DRAM: the trace retains ten 128-256 KiB tensors per layer where today's native call retains
one 128 KiB output, so +1.125 MiB per layer per chip, about 5.6 MiB per pair trace and 11 MiB per chip with two
pairs. The new SDPA shape (32/8 heads) and the new slice/concat shapes JIT-compile at the first folded bucket's
eager pass, inside that round's propose_pair.

SCOPE. Two T16 users, both at the full 2048-row history: the only packable geometry
(dflash_packed_proposal_coordinator.PACKED_CONTEXT). dflash_proposal_trace decides it once per pair bucket; the
capture and every replay of that bucket agree. draft_attention.py, dflash_t16_native_attention.py and
dflash_attention_mask.py are hashed sources of the T16 admission, so this module reads nothing from them and
changes none of them; test_pair_row_exact pins folded_sdpa's arguments to draft_sdpa's.

Off (unset or '0'): nothing imports this module on the serving path and every call is today's. '1' folds.
Anything else raises ValueError at the pair bucket build. The packed proposal coordinator turns ANY bucket-build
failure into a per-round fallback (that pair drafts on its single-user captures; PAIR_FALLBACK_LINE under
QWEN_FAST_PACKED_AUDIT), so a bad value does not stop the engine: it quietly stops packing. The arm passes only
'1'. MARKER is logged once per process, after the first folded bucket has captured and replayed; the m3native
gate requires it when the arm passes the flag, so a fold that never builds, for any reason, fails the gate.
"""

import os

FLAG = 'QWEN_FAST_PAIR_ROW_EXACT'
MARKER = '[PINDIAG] pair row exact engaged'
# The one packable geometry: two T16 users, each a 2048-row history plus its own 16-row block, aligned to 32.
CONTEXT = 2048
BLOCK_ROWS = 16
SPAN = 2080
HEAD_DIM = 128
# Per chip. The fold doubles both head counts, so the GQA group - and with it every head's KV head - is kept.
QUERY_HEADS, KEY_HEADS = 16, 4
GROUP = QUERY_HEADS // KEY_HEADS
FOLDED_QUERY_HEADS, FOLDED_KEY_HEADS = 2 * QUERY_HEADS, 2 * KEY_HEADS
_NOTED = []


def enabled(environ=None):
    """QWEN_FAST_PAIR_ROW_EXACT: '1' folds, unset or '0' does not, anything else raises."""
    value = (os.environ if environ is None else environ).get(FLAG, '0')
    if value not in ('0', '1'):
        raise ValueError('%s must be 0 or 1, got %r' % (FLAG, value))
    return value == '1'


def folds(contexts, block_rows):
    """Whether a pack can fold: exactly two T16 users, both at the full 2048-row history."""
    return (type(block_rows) is int and block_rows == BLOCK_ROWS
            and tuple(contexts) == (CONTEXT, CONTEXT))


def engages(contexts, block_rows, environ=None):
    """The flag set and a foldable pair: what dflash_proposal_trace decides once per pair bucket."""
    return enabled(environ) and folds(contexts, block_rows)


def require_fold(contexts, block_rows):
    if not folds(contexts, block_rows):
        raise ValueError('%s folds a pair of two %d-row T16 users only, not contexts %r at %r rows'
                         % (FLAG, CONTEXT, tuple(contexts), block_rows))


def fold_mask(contexts, block_rows=BLOCK_ROWS):
    """The folded SDPA's host mask: the served single-user mask, (1, 1, 32, 2080). It is also each user's own
    block of today's packed mask - batched_attention_mask's rows u*16..u*16+15 over segment u - so every folded
    head sees exactly the keys its user sees in the pair. Validate it with the single-user rule
    (dflash_t16_native_attention.validate_mask without contexts)."""
    require_fold(contexts, block_rows)
    from dflash_batched_mask import batched_attention_mask

    return batched_attention_mask([contexts[0]], block_rows)


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


def note(pair, contexts, *, log=None):
    """MARKER once per process, at the first bucket that folds. Returns whether it logged."""
    if _NOTED:
        return False
    _NOTED.append(tuple(pair))
    (log or _log_line)('%s pair=[%s,%s] context=%s,%s heads=%d/%d keys=%d' % (
        MARKER, pair[0], pair[1], contexts[0], contexts[1], FOLDED_QUERY_HEADS, FOLDED_KEY_HEADS, SPAN))
    return True


def validate_fold(operations, query, key, value, mask):
    """The packed pair's operands as execute_attention_branch assembles them, and the single-user mask."""
    if (tuple(query.shape) != (1, QUERY_HEADS, 32, HEAD_DIM)
            or tuple(key.shape) != (1, KEY_HEADS, 2 * SPAN, HEAD_DIM) or tuple(value.shape) != tuple(key.shape)
            or tuple(mask.shape) != (1, 1, 32, SPAN)):
        raise ValueError('The folded pair SDPA takes the packed (1, 16, 32, 128) query, the two-segment '
                         '(1, 4, 4160, 128) keys and values and the single-user (1, 1, 32, 2080) mask')
    if any(tensor.dtype != operations.bfloat16 for tensor in (query, key, value, mask)):
        raise ValueError('BF16 folded draft attention operands required')
    if any(tensor.layout != operations.TILE_LAYOUT or tensor.memory_config() != operations.DRAM_MEMORY_CONFIG
           for tensor in (query, key, value, mask)):
        raise ValueError('Interleaved tiled DRAM operands required')


def fold_query(operations, query, retain):
    """(1, 16, 32, 128) -> (1, 32, 32, 128): Q head 8h+4u+j is head 4h+j with user u's 16 rows at rows 0-15.

    The u=0 half is the query as it is; the u=1 half is the query with its row halves swapped, so user b's rows
    sit at rows 0-15 exactly as they do in b's single-user trace. The row moves are bf16 copies."""
    memory = operations.DRAM_MEMORY_CONFIG
    upper = retain(operations.slice(query, (0, 0, 0, 0), (1, QUERY_HEADS, BLOCK_ROWS, HEAD_DIM)))
    lower = retain(operations.slice(query, (0, 0, BLOCK_ROWS, 0), (1, QUERY_HEADS, 2 * BLOCK_ROWS, HEAD_DIM)))
    shifted = retain(operations.concat([lower, upper], dim=2, memory_config=memory))
    groups = [retain(operations.reshape(value, (KEY_HEADS, GROUP, 32, HEAD_DIM))) for value in (query, shifted)]
    folded = retain(operations.concat(groups, dim=1, memory_config=memory))
    return retain(operations.reshape(folded, (1, FOLDED_QUERY_HEADS, 32, HEAD_DIM)))


def fold_keys(operations, tensor, retain):
    """(1, 4, 4160, 128) -> (1, 8, 2080, 128), a view: each head's 130 tiles are [segment a | segment b], so KV
    head 2h+u is user u's segment of head h."""
    return retain(operations.reshape(tensor, (1, FOLDED_KEY_HEADS, SPAN, HEAD_DIM)))


def unfold_output(operations, output, retain):
    """(1, 32, 32, 128) -> (1, 16, 32, 128): head 4h+j's rows 0-15 from folded head 8h+j (user a) and its rows
    16-31 from folded head 8h+4+j's rows 0-15 (user b) - the packed layout the rest of the branch reads."""
    grouped = retain(operations.reshape(output, (KEY_HEADS, 2 * GROUP, 32, HEAD_DIM)))
    halves = []
    for user in range(2):
        rows = retain(operations.slice(grouped, (0, user * GROUP, 0, 0),
                                       (KEY_HEADS, (user + 1) * GROUP, BLOCK_ROWS, HEAD_DIM)))
        halves.append(retain(operations.reshape(rows, (1, QUERY_HEADS, BLOCK_ROWS, HEAD_DIM))))
    return retain(operations.concat(halves, dim=2, memory_config=operations.DRAM_MEMORY_CONFIG))


def folded_sdpa(operations, query, key, value, mask):
    """draft_attention.draft_sdpa's call, verbatim (streaming=False, key_chunk_size=32), on the folded heads:
    HiFi4, no math approximation, fp32 destination accumulation, no packer L1 accumulation, an 8x8 grid, q and
    k chunks of 32, exact exp, scale 128^-0.5, DRAM output. draft_sdpa itself refuses 32 query heads
    (validate_attention), and it is a hashed source of the T16 admission, so it is not widened; the test pins
    the two argument lists equal."""
    kernel = operations.WormholeComputeKernelConfig(math_fidelity=operations.MathFidelity.HiFi4,
        math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=False)
    program = operations.SDPAProgramConfig(compute_with_storage_grid_size=(8, 8), q_chunk_size=32,
        k_chunk_size=32, exp_approx_mode=False)
    return operations.transformer.scaled_dot_product_attention(query, key, value,
        attn_mask=mask, is_causal=False, scale=128 ** -0.5, program_config=program,
        compute_kernel_config=kernel, memory_config=operations.DRAM_MEMORY_CONFIG)


def fold_attention(operations, query, key, value, mask, retain, *, mask_validated=False):
    """The packed pair's draft attention, folded: the (1, 16, 32, 128) output today's native call returns,
    with each user's rows computed exactly as its single-user trace computes them."""
    if mask_validated is not True:
        raise ValueError('Validate the single-user host mask before upload and replay')
    validate_fold(operations, query, key, value, mask)
    folded = fold_query(operations, query, retain)
    keys = fold_keys(operations, key, retain)
    values = fold_keys(operations, value, retain)
    attention = retain(folded_sdpa(operations, folded, keys, values, mask))
    return unfold_output(operations, attention, retain)
