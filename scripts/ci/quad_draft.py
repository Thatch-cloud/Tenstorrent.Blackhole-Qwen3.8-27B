"""QWEN_FAST_QUAD_DRAFT (default off): one four-user, 64-row draft pass in place of the two packed pair passes,
in a round where all four users are live and packable (quad-draft-plan.md, Q4-core).

WHAT RUNS. The pair pass (dflash_device.execute_proposal over PreparedPackedDFlashProposal, the pair's draft SDPA
folded by pair_row_exact) is run once over a 64-row block holding the four users at rows 0/16/32/48:
  - every row-local op (embedding, norms, RoPE, typecast/add/silu/multiply, create/concat heads) at 64 rows;
  - the nine explicit matmul programs (conv-kernel x2, q, k, v, o, gate, up, down, selector) at per_core_M=2,
    everything else of each program as today;
  - the K/V: one 12-piece plan, every user's pad rows taken from rows [32p, 32p + 16) of its pair p, so each
    user's key segment is byte for byte the one its pair assembly builds (key_value_plan, R2);
  - the draft SDPA: QWEN_FAST_QUAD_SDPA=fold (the default) folds each 32-row half with pair_row_exact.fold_query,
    unchanged, concatenates the halves on the head axis and runs pair_row_exact.folded_sdpa, verbatim, at 64
    query / 16 KV heads (fold_attention); =pairs runs pair_row_exact.fold_attention on each half (pairs_attention);
  - the fused conv: QWEN_FAST_QUAD_CONV=110 (the default, E1b) runs quad_conv_io.cpp - the card-B probe's 64-row
    copy of the served I/O kernel, promoted byte for byte and pinned by CONV_KERNEL_SHA256 - over 320 pages on
    110 workers (the 11x10 grid; 0-99 take three pages, 100-109 two) with the served compute kernel; =80 (E1) the
    same kernel on the served 80 workers; =halves (C0) the served 32-row call on each half;
  - the LM head and top-k: today's 32-row call on each half (H0), each chunk's values and uint16 indices then
    concatenated on dim 2 (head_candidates), so the readback stays 18 reads.
The readback merges each half with merge_chunk_candidates(block_rows=32), stitches [half 0 | a dummy row | half 1]
(the dummy is merged index 31, user 2's anchor row, which no user slice reads) and splits with
split_selection(block_width=64) (read_quad_outputs): every user's selector features, candidates and scores are
the rows its pair's read_device_outputs returns.

WHY IT IS EXACT (E2, per user against the pair-fold control). The LM head, top-k, merge and host inputs are
today's calls on tile-aligned halves; the K/V bytes are the two pair assemblies' (C5); the folds are the pair's
own functions plus a head-axis concat; the conv I/O kernel moves the same integers per page. The widened
matmuls, norms, rotary and head ops were proven byte for byte on card B (Q0a 22/22, the K-split control firing
12/12 per matmul program), the quad fold 48/48 against the pair fold and single-user (Q0b; the restated control:
every user followed by masked keys, 30/30), E1 and E1b 12/12 (Q0d), the direct uint16 concat 12/12 (Q0c-lite),
all on commit ede1fd1a's probe (optimisation/ttnn-op/quad_draft_probe). In model the shadow audit
(QWEN_FAST_QUAD_DRAFT_AUDIT) replays the pair traces in the same round and compares every user's result.

WHEN IT RUNS. dflash_packed_proposal_coordinator engages it before the pair loop only when the flag is 1, the
round's groups are exactly [(0, 1), (2, 3)], both pairs are packable, every requirement below holds and it has
not given up. 3-live and 2-live rounds keep today's pairs. A failed build or replay falls that round back to the
pairs (FALLBACK_LINE); GIVE_UP_FAILURES consecutive failures disable it for the process (DISABLED_MARKER), which
the gate fails on. A fresh build needs QUAD_CAPTURE_BYTES_EST plus the packed reserve of free DRAM (est. until
the quad_built ledger point measures it).

REQUIRES QWEN_FAST_PACKED_PROPOSAL, QWEN_FAST_PAIR_ROW_EXACT, QWEN_FAST_ROUND_B1 (the batched selection) and
QWEN_FAST_FUSED_COMMIT_LIVE_BANKS (the quad reads the pool's live banks) at 1, the fused conv, four devices on one
mesh sharing one weight set, and a compute grid that holds the conv's cores; otherwise the quad is disabled with
the reason and the pairs run.

THE FLAGS, read at each round, never at import. QWEN_FAST_QUAD_DRAFT: '0' or unset and nothing imports this
module; '1' on; anything else raises ValueError (a configuration error, as QWEN_FAST_PACKED_PROPOSAL's).
QWEN_FAST_QUAD_SDPA fold|pairs, QWEN_FAST_QUAD_CONV 110|80|halves (defaults: what Q0 proved),
QWEN_FAST_QUAD_DRAFT_AUDIT all|N (every quad round, or the first N): the pair traces are replayed after the quad
in the same round, before the fence, read back beside the quad's own readback (before the round's deferred GDN
commits are enqueued), and every user's draft tokens, candidates, scores and selector features, and the raw
per-chip values, indices and features, are compared bit for bit - one AUDIT_LINE per audited round.
It never compares the quad with itself, and it spends back what the quad saves: a correctness arm only.

T16 PINS. The quad calls neither dflash_t16_native_attention.attention nor draft_attention.draft_sdpa; it reads
validate_mask and pair_row_exact (not hashed). The SDPA program is the pinned one, called at NQH 64 / NKH 16 (the
owner accepted it on the card-B byte proof, as the pair fold's 32/8).
"""

import hashlib
import os
from pathlib import Path
import time
from types import SimpleNamespace

FLAG = 'QWEN_FAST_QUAD_DRAFT'
SDPA_FLAG = 'QWEN_FAST_QUAD_SDPA'
CONV_FLAG = 'QWEN_FAST_QUAD_CONV'
AUDIT_FLAG = 'QWEN_FAST_QUAD_DRAFT_AUDIT'
REQUIRED_FLAGS = ('QWEN_FAST_PACKED_PROPOSAL', 'QWEN_FAST_PAIR_ROW_EXACT', 'QWEN_FAST_ROUND_B1',
                  'QWEN_FAST_FUSED_COMMIT_LIVE_BANKS')
SDPA_MODES = ('fold', 'pairs')
CONV_MODES = ('110', '80', 'halves')

# Logged once per process, after the first quad bucket has captured and replayed; the m3native gate requires it
# exactly once when the arm passes the flag.
MARKER = '[PINDIAG] quad draft engaged'
DISABLED_MARKER = '[PINDIAG] quad draft disabled'
# dflash_packed_proposal_coordinator.audit_log (brace) templates: per quad round under QWEN_FAST_PACKED_AUDIT, and
# per round the quad fell back to the pairs (always).
ROUND_LINE = '[QUAD-DRAFT] round={round} built={built} ms={ms}'
FALLBACK_LINE = '[QUAD-DRAFT] fallback round={round} reason={reason}'
RELEASE_LINE = '[QUAD-DRAFT] released pairs={pairs} headroom={headroom}'
# %-templates: QWEN_FAST_PAIR_MASK_AUDIT's read-back of the quad's mask, and the shadow audit.
MASK_AUDIT_LINE = '[QUAD-DRAFT] mask round=%s intact=%d mismatched=%d chip=%d'
AUDIT_LINE = '[QUAD-AUDIT] round=%s equal=%d stage=%s users=%d checks=%d'

# The one geometry: four T16 users at the full 2048-row history, rows 0/16/32/48 of a 64-row block.
CONTEXT, BLOCK, SPAN = 2048, 16, 2080
USERS, ROWS = 4, 64
SLOTS = (0, 1, 2, 3)
PAIRS = ((0, 1), (2, 3))
HEADS, KV_HEADS, HEAD_DIM = 16, 4, 128
GROUP = HEADS // KV_HEADS
QUAD_KEYS = USERS * SPAN                      # 8320
QUAD_QUERY_HEADS, QUAD_KEY_HEADS = 4 * HEADS, 4 * KV_HEADS
HIDDEN = 5120
TILE_PAGES = HIDDEN // 32                     # 160 pages per tile row of the hidden block
# The promoted conv I/O kernel: optimisation/ttnn-op/quad_draft_probe/quad_conv_io.cpp at ede1fd1a, the bytes Q0d
# ran on card B (JOB/quad/ship.sha256). Its header still reads 'NOT SHIPPED': the bytes are kept as probed.
CONV_KERNEL = 'quad_conv_io.cpp'
CONV_KERNEL_SHA256 = '8c4e8f93f8ced3a3c8ef8f551b8f711e5fdc03e7bcd0aeeade3e15ee94e0492b'
CONV_COMPUTE = 'draft_convolution_fused_compute.cpp'
# workers, the grid they tile, and worker w's core: E1b column-major on 11x10, E1 the served 8x10 order.
CONV_VARIANTS = {
    '110': dict(workers=110, grid=(11, 10), core=lambda worker: (worker // 10, worker % 10)),
    '80': dict(workers=80, grid=(8, 10), core=lambda worker: (worker % 8, worker // 8)),
}
# A fresh quad capture's DRAM per chip: placeholders ~0.2 MB plus the trace-retained intermediates, 0.3-0.45 GB
# (est., plan section 3.9). The coordinator refuses a fresh build below this plus the packed reserve; the
# quad_built ledger point measures the real figure.
QUAD_CAPTURE_BYTES_EST = 450 * 2 ** 20
# Plan section 3.9: below this free after the quad captures, the pair traces are released (rebuilt at the next
# 3-live round, 81-143 ms measured) - never while the audit needs them.
PAIR_RELEASE_BELOW_BYTES = 600 * 10 ** 6
GIVE_UP_FAILURES = 2
_NOTED = []
_KERNEL_CHECKED = []


def _flag(name, environ):
    return (os.environ if environ is None else environ).get(name)


def enabled(environ=None):
    """QWEN_FAST_QUAD_DRAFT: '1' on, unset or '0' off, anything else raises."""
    value = _flag(FLAG, environ)
    value = '0' if value is None else value
    if value not in ('0', '1'):
        raise ValueError('%s must be 0 or 1, got %r' % (FLAG, value))
    return value == '1'


def sdpa_mode(environ=None):
    """QWEN_FAST_QUAD_SDPA: 'fold' (unset, the default) or 'pairs'."""
    value = _flag(SDPA_FLAG, environ) or 'fold'
    if value not in SDPA_MODES:
        raise ValueError('%s must be one of %s, got %r' % (SDPA_FLAG, '|'.join(SDPA_MODES), value))
    return value


def conv_mode(environ=None):
    """QWEN_FAST_QUAD_CONV: '110' (unset, the default: E1b), '80' (E1) or 'halves' (C0)."""
    value = _flag(CONV_FLAG, environ) or '110'
    if value not in CONV_MODES:
        raise ValueError('%s must be one of %s, got %r' % (CONV_FLAG, '|'.join(CONV_MODES), value))
    return value


def audit_rounds(environ=None):
    """QWEN_FAST_QUAD_DRAFT_AUDIT: None (unset or empty: no audit), 'all', or a positive count N (the first N quad
    rounds)."""
    value = _flag(AUDIT_FLAG, environ)
    if value is None or value == '':
        return None
    if value == 'all':
        return value
    if not value.isdigit() or value != str(int(value)) or int(value) < 1:
        raise ValueError("%s must be 'all' or a positive count, got %r" % (AUDIT_FLAG, value))
    return int(value)


def audit_selected(quad_round, environ=None):
    """Whether the quad round numbered `quad_round` (1-based) is audited."""
    rounds = audit_rounds(environ)
    return rounds == 'all' or (rounds is not None and quad_round <= rounds)


def missing_requirements(environ=None):
    """The required flags not set to 1, in REQUIRED_FLAGS order."""
    environ = os.environ if environ is None else environ
    return [name for name in REQUIRED_FLAGS if environ.get(name) != '1']


def log_line(message):
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


def note(slots, sdpa, conv, *, log=None):
    """MARKER once per process, at the first quad bucket that captured and replayed. Returns whether it logged."""
    if _NOTED:
        return False
    _NOTED.append(tuple(slots))
    heads = '%d/%d' % (QUAD_QUERY_HEADS, QUAD_KEY_HEADS) if sdpa == 'fold' else '%d/%dx2' % (2 * HEADS, 2 * KV_HEADS)
    (log or log_line)('%s slots=[%s] heads=%s rows=%d sdpa=%s conv=%s' % (
        MARKER, ','.join(str(slot) for slot in slots), heads, ROWS, sdpa, conv))
    return True


def require_quad(contexts, block_rows):
    if type(block_rows) is not int or block_rows != BLOCK or tuple(contexts) != (CONTEXT,) * USERS:
        raise ValueError('%s packs four %d-row T16 users at the %d-row history only, not contexts %r at %r rows'
                         % (FLAG, BLOCK, CONTEXT, tuple(contexts), block_rows))


# ---------------------------------------------------------------------------------------------
# The K/V plan (plan section 3.4).
# ---------------------------------------------------------------------------------------------

def quad_key_value_plan():
    """The 12 pieces of the quad key axis, in order: for each user u (pair p = u // 2) its cached bank (2048 rows),
    its live rows [16u, 16u + 16) of the 64-row block, then 16 pad rows taken from rows [32p, 32p + 16) - exactly
    what the pair assembly puts in the pad (its `start = 0` is row 0 of the PAIR's own 32-row block). Every segment
    is then byte-identical to its pair assembly's segment (R2). The card-B probe's plan, verbatim."""
    plan = []
    for user in range(USERS):
        pair = user // 2
        plan.append(dict(kind='cached', user=user, rows=CONTEXT))
        plan.append(dict(kind='live', user=user, rows=BLOCK, source=slice(BLOCK * user, BLOCK * (user + 1))))
        plan.append(dict(kind='pad', user=user, rows=SPAN - CONTEXT - BLOCK,
                         source=slice(32 * pair, 32 * pair + BLOCK)))
    if sum(piece['rows'] for piece in plan) != QUAD_KEYS:
        raise AssertionError('The quad plan must cover 8320 keys')
    return plan


def key_value_plan(contexts, block_rows):
    """draft_attention_branch's (plan, spans, key_rows) for the quad: quad_key_value_plan and one 2080-key span
    per user, its rows the user's 16 rows of the 64-row block."""
    require_quad(contexts, block_rows)
    spans = [dict(user=user, context=CONTEXT, offset=SPAN * user, span=SPAN, rows=slice(BLOCK * user, BLOCK * (user + 1)),
                  keys=slice(SPAN * user, SPAN * (user + 1))) for user in range(USERS)]
    return quad_key_value_plan(), spans, QUAD_KEYS


# ---------------------------------------------------------------------------------------------
# The draft SDPA (plan section 3.3): the four-way fold, or the pair fold per half.
# ---------------------------------------------------------------------------------------------

def validate_quad(operations, query, key, value, mask):
    if (tuple(query.shape) != (1, HEADS, ROWS, HEAD_DIM) or tuple(key.shape) != (1, KV_HEADS, QUAD_KEYS, HEAD_DIM)
            or tuple(value.shape) != tuple(key.shape) or tuple(mask.shape) != (1, 1, 32, SPAN)):
        raise ValueError('The quad fold takes the (1, 16, 64, 128) query, the four-segment (1, 4, 8320, 128) keys '
                         'and values and the single-user (1, 1, 32, 2080) mask')
    if any(tensor.dtype != operations.bfloat16 for tensor in (query, key, value, mask)):
        raise ValueError('BF16 quad draft attention operands required')
    if any(tensor.layout != operations.TILE_LAYOUT or tensor.memory_config() != operations.DRAM_MEMORY_CONFIG
           for tensor in (query, key, value, mask)):
        raise ValueError('Interleaved tiled DRAM operands required')


def quad_fold_query(operations, query, retain):
    """(1, 16, 64, 128) -> (1, 64, 32, 128). Each tile-aligned half is folded by pair_row_exact.fold_query,
    unchanged, to (1, 32, 32, 128) (head 8h + 4u' + j); each is read as (4, 8, 32, 128) and the halves are
    concatenated on dim 1, so folded head 16h + 8p + 4u' + j = 16h + 4u + j with u = 2p + u'. GQA group 4 then
    maps it to KV head 4h + u: user u's segment of head h."""
    from pair_row_exact import fold_query

    memory = operations.DRAM_MEMORY_CONFIG
    grouped = []
    for pair in range(2):
        half = retain(operations.slice(query, (0, 0, 32 * pair, 0), (1, HEADS, 32 * (pair + 1), HEAD_DIM)))
        folded = fold_query(operations, half, retain)
        grouped.append(retain(operations.reshape(folded, (KV_HEADS, 2 * GROUP, 32, HEAD_DIM))))
    combined = retain(operations.concat(grouped, dim=1, memory_config=memory))
    return retain(operations.reshape(combined, (1, QUAD_QUERY_HEADS, 32, HEAD_DIM)))


def quad_fold_keys(operations, tensor, retain):
    """(1, 4, 8320, 128) -> (1, 16, 2080, 128), a view: KV head 4h + u is user u's segment of head h."""
    return retain(operations.reshape(tensor, (1, QUAD_KEY_HEADS, SPAN, HEAD_DIM)))


def quad_unfold_output(operations, output, retain):
    """(1, 64, 32, 128) -> (1, 16, 64, 128): per half p, folded heads [16h + 8p, 16h + 8p + 8) are pair p's
    (1, 32, 32, 128) fold output, which pair_row_exact.unfold_output turns into the pair's packed rows."""
    from pair_row_exact import unfold_output

    grouped = retain(operations.reshape(output, (KV_HEADS, QUAD_QUERY_HEADS // KV_HEADS, 32, HEAD_DIM)))
    halves = []
    for pair in range(2):
        part = retain(operations.slice(grouped, (0, 8 * pair, 0, 0), (KV_HEADS, 8 * (pair + 1), 32, HEAD_DIM)))
        folded = retain(operations.reshape(part, (1, 2 * HEADS, 32, HEAD_DIM)))
        halves.append(unfold_output(operations, folded, retain))
    return retain(operations.concat(halves, dim=2, memory_config=operations.DRAM_MEMORY_CONFIG))


def fold_attention(operations, query, key, value, mask, retain, *, mask_validated=False):
    """QWEN_FAST_QUAD_SDPA=fold: every (Q head, KV head) unit is the pair fold's unit for that user, byte for byte
    in Q, K, V and mask (given the plan's pads); only the compile-time NQH/NKH differ (64 / 16)."""
    from pair_row_exact import folded_sdpa

    if mask_validated is not True:
        raise ValueError('Validate the single-user host mask before upload and replay')
    validate_quad(operations, query, key, value, mask)
    folded = quad_fold_query(operations, query, retain)
    keys = quad_fold_keys(operations, key, retain)
    values = quad_fold_keys(operations, value, retain)
    attention = retain(folded_sdpa(operations, folded, keys, values, mask))
    return quad_unfold_output(operations, attention, retain)


def pairs_attention(operations, query, key, value, mask, retain, *, mask_validated=False):
    """QWEN_FAST_QUAD_SDPA=pairs: pair_row_exact.fold_attention on each half - pair p's 32 query rows and its two
    key segments, keys [4160p, 4160p + 4160), which are byte for byte its own pair assembly."""
    from pair_row_exact import fold_attention as pair_fold

    if mask_validated is not True:
        raise ValueError('Validate the single-user host mask before upload and replay')
    validate_quad(operations, query, key, value, mask)
    halves = []
    for pair in range(2):
        rows = retain(operations.slice(query, (0, 0, 32 * pair, 0), (1, HEADS, 32 * (pair + 1), HEAD_DIM)))
        keys, values = (retain(operations.slice(tensor, (0, 0, 2 * SPAN * pair, 0),
                                                (1, KV_HEADS, 2 * SPAN * (pair + 1), HEAD_DIM)))
                        for tensor in (key, value))
        halves.append(pair_fold(operations, rows, keys, values, mask, retain, mask_validated=True))
    return retain(operations.concat(halves, dim=2, memory_config=operations.DRAM_MEMORY_CONFIG))


def quad_head_map():
    """{folded query head: (h, u, j, kv head)}: the fold's head arithmetic."""
    out = {}
    for h in range(KV_HEADS):
        for u in range(USERS):
            for j in range(GROUP):
                out[16 * h + 4 * u + j] = (h, u, j, 4 * h + u)
    return out


# ---------------------------------------------------------------------------------------------
# The 64-row variants of the bundle-only helpers (draft_kv_projection, draft_head_layout, feature_collective): the
# same calls, at 64 rows. Those files are in neither image copy list and stay untouched.
# ---------------------------------------------------------------------------------------------

def _tiled_dram(operations, tensors):
    return all(tensor.dtype == operations.bfloat16 and tensor.layout == operations.TILE_LAYOUT
               and tensor.memory_config() == operations.DRAM_MEMORY_CONFIG for tensor in tensors)


def split_projected_heads(operations, query, key, value, retain):
    """draft_head_layout.split_projected_heads at 64 rows: query, keys and values share the 64 rows, so there is
    no query pad and no slice back - its 32-row packed call's exact sequence."""
    if (tuple(query.shape) != (1, 1, ROWS, 2048) or tuple(key.shape) != (1, 1, ROWS, 512)
            or tuple(value.shape) != tuple(key.shape) or not _tiled_dram(operations, (query, key, value))):
        raise ValueError('64-row BF16 DRAM projections for 16 query and four KV heads required')
    combined_kv = retain(operations.concat([key, value], dim=3, memory_config=operations.DRAM_MEMORY_CONFIG))
    heads = operations.experimental.nlp_create_qkv_heads(query, combined_kv,
        num_heads=16, num_kv_heads=4, transpose_k_heads=False, memory_config=operations.DRAM_MEMORY_CONFIG)
    query_heads, key_heads, value_heads = (retain(tensor) for tensor in heads)
    return dict(q=query_heads, k=key_heads, v=value_heads)


def project_key_value(operations, inputs, query, cosine_sine, retain, *, parameters):
    """draft_kv_projection.project_key_value at 64 rows: the same program (per_core_M = rows // 32 = 2), the
    64-row head split, then today's k norm, rotary and typecast."""
    if (parameters.get('operations') is not operations or parameters.get('native_head_layout') is not True
            or tuple(inputs.shape) != (1, 1, ROWS, HIDDEN) or tuple(query.shape) != (1, 1, ROWS, 2048)
            or len(cosine_sine) != 2 or any(tuple(table.shape) != (1, 1, ROWS, HEAD_DIM) for table in cosine_sine)
            or not _tiled_dram(operations, (inputs, query, *cosine_sine))):
        raise ValueError('Owned 64-row BF16 tiled K/V rows, the live-key rotary tables and native head parameters '
                         'required')
    kernel = parameters['kernel']
    program = operations.MatmulMultiCoreReuseMultiCast1DProgramConfig(compute_with_storage_grid_size=(8, 8),
        in0_block_w=4, out_subblock_h=1, out_subblock_w=1, per_core_M=ROWS // 32,
        per_core_N=1, fuse_batch=True, fused_activation=None, mcast_in0=True)
    flat = {}
    for name in ('k', 'v'):
        projected = retain(operations.matmul(inputs, parameters['projections'][name], dtype=operations.float32,
            compute_kernel_config=kernel, program_config=program, memory_config=operations.DRAM_MEMORY_CONFIG))
        flat[name] = retain(operations.typecast(projected, operations.bfloat16))
    heads = split_projected_heads(operations, query, flat['k'], flat['v'], retain)
    normalized = retain(operations.rms_norm(heads['k'], epsilon=1e-6, weight=parameters['head_norms']['k'],
        compute_kernel_config=kernel, memory_config=operations.DRAM_MEMORY_CONFIG))
    wide = [retain(operations.typecast(value, operations.float32)) for value in (normalized, *cosine_sine)]
    rotated = retain(operations.experimental.rotary_embedding_hf(*wide, is_decode_mode=False,
        compute_kernel_config=kernel, memory_config=operations.DRAM_MEMORY_CONFIG))
    return dict(q=heads['q'], k=retain(operations.typecast(rotated, operations.bfloat16)), v=heads['v'])


def concatenate_query_heads(operations, value, retain):
    """draft_head_layout.concatenate_query_heads at 64 rows."""
    if tuple(value.shape) != (1, HEADS, ROWS, HEAD_DIM) or not _tiled_dram(operations, (value,)):
        raise ValueError('64-row BF16 DRAM query heads required')
    return retain(operations.experimental.nlp_concat_heads(value, memory_config=operations.DRAM_MEMORY_CONFIG))


def gather_add_projection(operations, mesh, collectives, value, *, retain_temporaries=None, observe=None):
    """feature_collective.gather_add_projection at 64 rows, trace-owned only (the served branches always pass
    retain_temporaries): the dim-0 all-gather, one slice per chip, the fp32 add."""
    from projection_link_policy import projection_links

    links = projection_links()
    if (list(mesh.shape) != [1, 2] or tuple(value.shape) != (1, 1, ROWS, HIDDEN) or value.dtype != operations.float32
            or not callable(retain_temporaries)):
        raise ValueError('64 rows of full-width FP32 TP2 projection and a trace lifetime owner required')
    temporaries = []

    def retain(tensor):
        temporaries.append(tensor)
        retain_temporaries(tensor)
        return tensor

    gathered = operations.experimental.all_gather_async(value,
        persistent_output_buffer=None, dim=0,
        multi_device_global_semaphore=collectives.get_and_cycle_ag_semaphore_handles(),
        barrier_semaphore=collectives.get_and_cycle_barrier_semaphore_handle(), num_links=links,
        memory_config=operations.DRAM_MEMORY_CONFIG, topology=operations.Topology.Linear,
        chunks_per_sync=10, num_workers_per_link=2, num_buffers_per_channel=2)
    retain(gathered)
    if observe is not None:
        observe('gathered', gathered)
    for chip in range(2):
        retain(operations.slice(gathered, (chip, 0, 0, 0), (chip + 1, 1, ROWS, HIDDEN)))
    return operations.add(temporaries[1], temporaries[2], dtype=operations.float32,
                          memory_config=operations.DRAM_MEMORY_CONFIG)


# ---------------------------------------------------------------------------------------------
# The fused conv (plan section 3.5).
# ---------------------------------------------------------------------------------------------

def seam_word(boundaries, rows):
    """draft_convolution_fused.seam_mask for rows <= 32: bit r set where a segment begins (bit 0 always)."""
    spans = tuple(tuple(span) for span in (boundaries or ((0, rows),)))
    if not spans or spans[0][0] != 0 or spans[-1][1] != rows or any(
            spans[index][1] != spans[index + 1][0] or spans[index][0] >= spans[index][1]
            for index in range(len(spans) - 1)) or spans[-1][0] >= spans[-1][1]:
        raise ValueError('Ordered contiguous packed segment spans covering the block required')
    word = 0
    for start, _ in spans:
        word |= 1 << start
    return word


def seam_words(boundaries, rows):
    """The conv I/O kernel's per-tile-row seam words (low, high). A 64-row block must begin a segment at row 32:
    no causal carry may cross a tile row (the kernel shifts within a tile), which the four users at rows
    0/16/32/48 satisfy."""
    spans = tuple(tuple(span) for span in (boundaries or ((0, rows),)))
    if rows <= 32:
        return seam_word(spans, rows), 0
    if rows != ROWS or not any(start == 32 for start, _ in spans):
        raise ValueError('A 64-row conv block must begin a segment at row 32')
    low = tuple((start, stop) for start, stop in spans if stop <= 32)
    high = tuple((start - 32, stop - 32) for start, stop in spans if start >= 32)
    if sum(stop - start for start, stop in low + high) != rows:
        raise ValueError('A segment crosses the tile-row boundary')
    return seam_word(low, 32), seam_word(high, 32)


def conv_kernel_path():
    """The promoted I/O kernel beside this module, refused unless its bytes are the probed ones."""
    path = Path(__file__).with_name(CONV_KERNEL)
    if not _KERNEL_CHECKED:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != CONV_KERNEL_SHA256:
            raise ValueError('%s is not the card-B qualified kernel (sha256 %s, pinned %s)'
                             % (CONV_KERNEL, digest, CONV_KERNEL_SHA256))
        _KERNEL_CHECKED.append(digest)
    return path


def conv_pages(workers, rows=ROWS):
    """{worker: [pages]}: page = worker, worker + workers, ... below 160 per tile row."""
    total = TILE_PAGES * ((rows + 31) // 32)
    return {worker: list(range(worker, total, workers)) for worker in range(workers)}


def column_ranges(coordinates):
    """Merge core coordinates into per-column (x, y0, y1) runs."""
    runs = []
    for x, y in sorted(coordinates):
        if runs and runs[-1][0] == x and runs[-1][2] == y - 1:
            runs[-1] = (x, runs[-1][1], y)
        else:
            runs.append((x, y, y))
    return runs


def core_ranges(operations, coordinates):
    return operations.CoreRangeSet([operations.CoreRange(operations.CoreCoord(x, y0), operations.CoreCoord(x, y1))
                                    for x, y0, y1 in column_ranges(coordinates)])


def grid_fits(mesh, conv):
    """Whether the mesh's compute grid holds the conv variant's cores: True for 'halves' (the served call),
    None when the grid cannot be read."""
    if conv == 'halves':
        return True
    try:
        size = mesh.compute_with_storage_grid_size()
        grid = int(size.x), int(size.y)
    except Exception:  # noqa: BLE001 - a diagnostic read, never a refusal by itself
        return None
    need = CONV_VARIANTS[conv]['grid']
    return grid[0] >= need[0] and grid[1] >= need[1]


def validate_conv_shapes(hidden, dynamic, base):
    if (tuple(hidden.shape) != (1, 1, ROWS, HIDDEN) or len(dynamic) != 2 or len(base) != 2
            or any(tuple(value.shape) != (1, 1, ROWS, 320) for value in dynamic)
            or any(tuple(value.shape) != (1, 1, 1, HIDDEN) for value in base)):
        raise ValueError('A 64-row hidden block, two 64-row group-16 dynamic kernels and two full-width base '
                         'kernels required')


def quad_fused_convolution(operations, mesh, hidden, dynamic, base, *, boundaries, conv='110'):
    """draft_convolution_fused.fused_convolution at 64 rows (E1b on 110 workers, E1 on 80): the promoted I/O
    kernel reads each page's tiles and seam word, the served compute kernel does the per-page arithmetic with one
    compile arg per group of workers taking the same number of pages. Per chip, runtime args are the six buffer
    addresses + [rows, worker, workers, low seams, high seams]."""
    validate_conv_shapes(hidden, dynamic, base)
    if conv not in CONV_VARIANTS:
        raise ValueError('The quad conv program runs on 110 or 80 workers, not %r' % (conv,))
    low, high = seam_words(boundaries, ROWS)
    tensors = [hidden, *dynamic, *base]
    if list(mesh.shape) != [1, 2] or not _tiled_dram(operations, tensors):
        raise ValueError('Two-chip interleaved DRAM BF16 convolution operands required')
    parts = [operations.get_device_tensors(value) for value in tensors]
    if any(len(shards) != 2 for shards in parts):
        raise ValueError('Both operand shards required')
    kernel = conv_kernel_path()
    spec = CONV_VARIANTS[conv]
    workers = spec['workers']
    coordinates = [spec['core'](worker) for worker in range(workers)]
    pages = conv_pages(workers)
    output = operations.empty(tuple(hidden.shape), dtype=operations.bfloat16, layout=operations.TILE_LAYOUT,
        device=mesh, memory_config=operations.DRAM_MEMORY_CONFIG)
    tensors.append(output)
    parts.append(operations.get_device_tensors(output))
    cores = core_ranges(operations, coordinates)
    buffers = [operations.CBDescriptor(total_size=2048 * count, core_ranges=cores,
        format_descriptors=[operations.CBFormatDescriptor(buffer_index=index, data_format=operations.bfloat16,
            page_size=2048, tile=operations.TileDescriptor(operations.Tile([32, 32])))])
        for index, count in ((0, 7), (1, 2), (16, 1))]
    groups = {}
    for worker, owned in pages.items():
        groups.setdefault(len(owned), []).append(coordinates[worker])
    computes = [operations.KernelDescriptor(kernel_source=str(Path(__file__).with_name(CONV_COMPUTE)),
        core_ranges=core_ranges(operations, group), compile_time_args=[count],
        config=operations.ComputeConfigDescriptor(math_fidelity=operations.MathFidelity.HiFi4, fp32_dest_acc_en=True,
                                                  math_approx_mode=False))
        for count, group in sorted(groups.items())]
    program = operations.MeshProgramDescriptor()
    try:
        for chip in range(2):
            local = [shards[chip] for shards in parts]
            if local[-1].buffer_address() in {value.buffer_address() for value in local[:-1]}:
                raise ValueError('Convolution output must not alias borrowed inputs')
            runtime = operations.RuntimeArgs()
            for worker, (x, y) in enumerate(coordinates):
                runtime[x][y] = [value.buffer_address() for value in local] + [ROWS, worker, workers, low, high]
            reader = operations.KernelDescriptor(kernel_source=str(kernel), core_ranges=cores,
                compile_time_args=[argument for value in local
                                   for argument in operations.TensorAccessorArgs(value).get_compile_time_args()],
                runtime_args=runtime, config=operations.DataMovementConfigDescriptor(
                    processor=operations.DataMovementProcessor.RISCV_0, noc=operations.NOC.RISCV_0_default))
            coordinate = operations.MeshCoordinate(0, chip)
            program[operations.MeshCoordinateRange(coordinate, coordinate)] = operations.ProgramDescriptor(
                kernels=[reader, *computes], cbs=buffers)
        operations.generic_op(tensors, program)
    except BaseException:
        operations.deallocate(output)
        raise
    return output


def halves_convolution(operations, mesh, hidden, dynamic, base, *, boundaries, retain):
    """QWEN_FAST_QUAD_CONV=halves (C0): the served 32-row fused call on each tile-aligned half, then a concat."""
    from draft_convolution_fused import fused_convolution

    validate_conv_shapes(hidden, dynamic, base)
    seam_words(boundaries, ROWS)
    spans = tuple(tuple(span) for span in boundaries)
    parts = []
    for half in range(2):
        start, stop = 32 * half, 32 * (half + 1)
        rows = retain(operations.slice(hidden, (0, 0, start, 0), (1, 1, stop, HIDDEN)))
        kernels = [retain(operations.slice(value, (0, 0, start, 0), (1, 1, stop, 320))) for value in dynamic]
        local = tuple((low - start, high - start) for low, high in spans if start <= low and high <= stop)
        parts.append(retain(fused_convolution(operations, mesh, rows, kernels, base, boundaries=local)))
    return operations.concat(parts, dim=2, memory_config=operations.DRAM_MEMORY_CONFIG)


# ---------------------------------------------------------------------------------------------
# The head (plan section 3.6).
# ---------------------------------------------------------------------------------------------

def concat_candidates(operations, first, second, retain):
    """Each chunk's values and uint16 indices of two 32-row halves concatenated on dim 2 (Q0c-lite: bitwise)."""
    memory = operations.DRAM_MEMORY_CONFIG
    out = []
    for top, bottom in zip(first, second, strict=True):
        if (top['start'], top['stop']) != (bottom['start'], bottom['stop']):
            raise ValueError('The halves must carry the same candidate chunks')
        values = retain(operations.concat([top['values'], bottom['values']], dim=2, memory_config=memory))
        indices = retain(operations.concat([top['indices'], bottom['indices']], dim=2, memory_config=memory))
        out.append(dict(start=top['start'], stop=top['stop'], values=values, indices=indices))
    return out


def head_candidates(operations, model, normalized, owned, retain):
    """H0: draft_shared_head.shared_head_candidates, unchanged, on each tile-aligned 32-row half, then the
    candidate concat."""
    from draft_shared_head import shared_head_candidates

    if tuple(normalized.shape) != (1, 1, ROWS, HIDDEN):
        raise ValueError('The 64-row learned-normalized block required')
    halves = []
    for half in range(2):
        block = retain(operations.slice(normalized, (0, 0, 32 * half, 0), (1, 1, 32 * (half + 1), HIDDEN)))
        halves.append(shared_head_candidates(operations, model, block, owned))
    return concat_candidates(operations, halves[0], halves[1], retain)


class QuadPass:
    """What execute_proposal and the two branches take from the quad (their `quad` keyword; None, the default,
    changes nothing): the 64 rows, the K/V plan, the SDPA, the 64-row helpers, the conv and the head."""

    rows = ROWS
    mask_rows = SPAN

    def __init__(self, sdpa='fold', conv='110'):
        if sdpa not in SDPA_MODES or conv not in CONV_MODES:
            raise ValueError('Quad SDPA %s and conv %s required, got %r / %r'
                             % ('|'.join(SDPA_MODES), '|'.join(CONV_MODES), sdpa, conv))
        self.sdpa, self.conv = sdpa, conv

    key_value_plan = staticmethod(key_value_plan)
    project_key_value = staticmethod(project_key_value)
    concatenate_query_heads = staticmethod(concatenate_query_heads)
    gather_add_projection = staticmethod(gather_add_projection)
    head_candidates = staticmethod(head_candidates)

    def attention(self, operations, query, key, value, mask, retain, *, mask_validated=False):
        attend = fold_attention if self.sdpa == 'fold' else pairs_attention
        return attend(operations, query, key, value, mask, retain, mask_validated=mask_validated)

    def convolution(self, operations, mesh, hidden, dynamic, base, *, fp32_intermediates=False,
                    retain_temporaries=None, boundaries=None):
        """The branches' convolution_operation: checked_convolution's contract (exact FP32 arithmetic, a lifetime
        owner, the output retained), at 64 rows."""
        if fp32_intermediates is not True or not callable(retain_temporaries) or boundaries is None:
            raise ValueError('The quad conv requires exact FP32 arithmetic, a lifetime owner and the user seams')
        if self.conv == 'halves':
            output = halves_convolution(operations, mesh, hidden, dynamic, base, boundaries=boundaries,
                                        retain=retain_temporaries)
        else:
            output = quad_fused_convolution(operations, mesh, hidden, dynamic, base, boundaries=boundaries,
                                            conv=self.conv)
        retain_temporaries(output)
        return output


# ---------------------------------------------------------------------------------------------
# Host inputs and the readback.
# ---------------------------------------------------------------------------------------------

def quad_users(devices, context=CONTEXT):
    return [dict(position=device.position, history_rows=context) for device in devices]


def quad_rope(users):
    """The quad's rope.q and live_k: each pair's QWEN_FAST_ROUND_B1 (C8) build - packed_rope_tables over the pair,
    live_key_rope_from its key tables - concatenated on the row axis, so rows [32p, 32p + 32) are byte for byte
    what pair p uploads. Returns (query tables, live key tables), each (cos, sin) of (1, 1, 64, 128)."""
    import torch

    from dflash_batched_mask import live_key_rope_from, packed_rope_tables

    users = list(users)
    if len(users) != USERS:
        raise ValueError('Four users required')
    query, live = [], []
    for first, second in PAIRS:
        members = [users[first], users[second]]
        tables = packed_rope_tables(members, BLOCK)
        query.append(tables['q'])
        live.append(live_key_rope_from(tables['k'], members, BLOCK))
    joined = lambda parts: tuple(torch.cat([part[index] for part in parts], dim=2) for index in (0, 1))
    return joined(query), joined(live)


def read_quad_outputs(device, outputs):
    """read_device_outputs for the quad: the 18 reads (4 chunks x 2 chips x values and indices, the features on 2
    chips), each half merged by merge_chunk_candidates(block_rows=32) exactly as its pair merges, stitched
    [half 0 | a dummy row | half 1] (the dummy is merged index 31 - block row 32, user 2's anchor - which no user
    slice reads), and split with split_selection(block_width=64): four users' (features, candidates, scores)."""
    import torch

    from dflash_packed_proposal import split_selection
    from draft_shared_head import merge_chunk_candidates

    operations = device.operations
    halves = ([], [])
    for chunk in outputs.chunks:
        values = operations.get_device_tensors(chunk['values'])
        indices = operations.get_device_tensors(chunk['indices'])
        if len(values) != 2 or len(indices) != 2:
            raise AssertionError('Both learned head shards required')
        for chip in range(2):
            host_values = operations.to_torch(values[chip]).float().reshape(ROWS, 16)
            host_indices = operations.to_torch(indices[chip]).long().reshape(ROWS, 16)
            for half in range(2):
                rows = slice(32 * half, 32 * (half + 1))
                halves[half].append(dict(chip=chip, start=chunk['start'], stop=chunk['stop'],
                                         values=host_values[rows], indices=host_indices[rows]))
    merged = [merge_chunk_candidates(list(part), block_rows=32) for part in halves]
    candidates = torch.cat([merged[0][0], torch.zeros_like(merged[0][0][:, :1]), merged[1][0]], dim=1)
    unary = torch.cat([merged[0][1], torch.zeros_like(merged[0][1][:, :1]), merged[1][1]], dim=1)
    parts = [operations.to_torch(value) for value in operations.get_device_tensors(outputs.projected)]
    if len(parts) != 2 or not torch.equal(*parts):
        raise AssertionError('Replicated learned selector features differ')
    hidden = parts[0].reshape(1, ROWS, 256)
    return split_selection(hidden, candidates, unary, USERS, BLOCK, block_width=ROWS)


def select_quad_outputs(device, outputs, seeds, counts):
    """read_quad_outputs, then today's per-user selector (select_packed): the flag-off selection."""
    from dflash_packed_proposal import select_packed

    return select_packed(read_quad_outputs(device, outputs), seeds, counts, device.predecessors, device.successors)


# ---------------------------------------------------------------------------------------------
# The trace (plan section 3.7).
# ---------------------------------------------------------------------------------------------

class PreparedQuadDFlashProposal:
    """One traced four-user 64-row proposal over pool slots 0-3: dflash_proposal_trace.PreparedPackedDFlashProposal
    for four devices. One bucket, (2048,) * 4, built lazily at the first prepare_device: placeholders (ids (1, 64),
    the folded single-user mask (1, 1, 32, 2080) registered on devices[0], rope.q and live_k 2 x (1, 1, 64, 128);
    no rope.k - nothing reads it), cached_history the pool's live banks (fused_commit.live_bank_history per pair),
    then an eager warm-up, the capture, one blocking replay, the marker and the quad_built ledger point.
    devices[0] owns the weights (all four share them). collect(), adopt() and finish() take `which` in 0..3.

    Under QWEN_FAST_QUAD_DRAFT_AUDIT the coordinator attaches the round's two pair traces, already prepared with
    the same seeds (attach_audit); collect() takes every device read the audit compares (read_audit) beside the
    quad's own readback - before select_round's after_collect enqueues the round's deferred GDN commits - and
    run_audit compares those host copies after the selection; discard_pending discards the pairs with the quad's
    own pending, so every failure path releases them."""

    def __init__(self, devices, *, sdpa=None, conv=None):
        devices = tuple(devices)
        if len(devices) != USERS:
            raise ValueError('Four devices required')
        first = devices[0]
        if any(device.operations is not first.operations or device.mesh is not first.mesh for device in devices):
            raise ValueError('All four devices must share one mesh and runtime')
        if any(device.block_rows != BLOCK or not getattr(device, 'native_proposal_attention', False)
               for device in devices):
            raise ValueError('All four devices must run the qualified native-proposal T16 block')
        if any(device.kv_history is None for device in devices):
            raise ValueError('All four devices require a committed K/V cache')
        self.devices = devices
        self.operations, self.mesh = first.operations, first.mesh
        self.block_rows = BLOCK
        self.quad = QuadPass(sdpa_mode() if sdpa is None else sdpa, conv_mode() if conv is None else conv)
        self.buckets, self.owned = {}, []
        self.closed = False
        self._pending = None
        self._audit = None
        # snapshot_audit's host copy of the quad's outputs, before the audit's pair replays; read_audit's host
        # copies (or 'read-error:<type>'), taken by collect() while an audit is attached.
        self._audit_snapshot = None
        self._audit_reads = None
        self.last_built = False
        # QWEN_FAST_PAIR_MASK_AUDIT: the coordinator's round, set before prepare_device only while it is on.
        self.round_number = None

    @property
    def device_a(self):
        """The weight and codebook owner (select_round groups by its predecessors)."""
        return self.devices[0]

    def pair_label(self):
        from dflash_proposal_trace import _slot_of

        return [_slot_of(device) for device in self.devices]

    def _upload(self, value, *, identifiers=False):
        operations = self.operations
        tensor = operations.from_torch(value, device=self.mesh,
            dtype=operations.uint32 if identifiers else operations.bfloat16,
            layout=operations.ROW_MAJOR_LAYOUT if identifiers else operations.TILE_LAYOUT,
            memory_config=operations.DRAM_MEMORY_CONFIG, mesh_mapper=operations.ReplicateTensorToMesh(self.mesh))
        self.owned.append(tensor)
        return tensor

    def _protected(self, bucket=None):
        """What no temporary of the pass or of an update may ever queue for release: every device's feature
        history pair, the placeholders and the lent live banks."""
        protected = []
        for device in self.devices:
            protected.extend((device.history, device.spare_history))
        protected.extend(self.owned)
        if bucket is not None:
            protected.extend(value for cache in bucket.cached_history for layer in cache for value in layer.values())
        return protected

    def _live_banks(self):
        from fused_commit import live_bank_history

        banks = []
        for first, second in PAIRS:
            pair = live_bank_history(self.devices[first], self.devices[second])
            if pair is None:
                raise ValueError('The quad draft reads the pool live banks: QWEN_FAST_FUSED_COMMIT_LIVE_BANKS and '
                                 'pooled five-layer caches required')
            banks.extend(pair)
        return banks

    def _bucket(self):
        key = (CONTEXT,) * USERS
        bucket = self.buckets.get(key)
        if bucket is not None:
            self.last_built = False
            return bucket
        from dflash_packed_proposal import packed_identifiers
        from dflash_t16_native_attention import validate_mask
        from gdn_multitoken_conv import addresses, release_owned
        from pair_row_exact import fold_mask

        operations, device = self.operations, self.devices[0]
        placeholder_mark = len(self.owned)
        try:
            host_mask = fold_mask((CONTEXT, CONTEXT), BLOCK)
            validate_mask(host_mask)
            query, live = quad_rope([dict(position=CONTEXT, history_rows=CONTEXT)] * USERS)
            bucket = SimpleNamespace(context=key, host_mask=host_mask,
                identifiers=self._upload(packed_identifiers([0] * USERS, BLOCK, block_width=ROWS), identifiers=True),
                mask=self._upload(host_mask),
                rope=dict(q=tuple(self._upload(value) for value in query),
                          live_k=tuple(self._upload(value) for value in live)),
                cached_history=self._live_banks(), trace=None, outputs=None, owned=[], tokens=None, consumed=set(),
                parts=None)
            bucket.inputs = [bucket.identifiers, bucket.mask, *bucket.rope['q'], *bucket.rope['live_k'],
                *(value for cache in bucket.cached_history for layer in cache for value in layer.values())]
            bucket.addresses = [addresses(operations, value) for value in bucket.inputs]
            device.validated_native_proposal_masks.add(addresses(operations, bucket.mask))
            self.buckets[key] = bucket
            from fused_commit import note_live_banks

            note_live_banks(self.pair_label(), bucket.context)
            self._update(bucket, (0,) * USERS)
            transient, retain = device.temporaries(self._protected(bucket))
            try:
                self._execute(bucket, transient, retain)
                operations.synchronize_device(self.mesh)
            finally:
                release_owned(operations, transient)
            bucket.owned, retain = device.temporaries(self._protected(bucket))
            from attention_batch import capture_operation

            bucket.trace, bucket.outputs = capture_operation(operations, self.mesh,
                lambda: self._execute(bucket, bucket.owned, retain))
            operations.execute_trace(self.mesh, bucket.trace, cq_id=0, blocking=True)
            note(self.pair_label(), self.quad.sdpa, self.quad.conv)
            import memory_ledger

            # The quad's own buffers only (its placeholders and the capture's retained intermediates): the delta
            # against the previous point is what a fresh quad costs (QUAD_CAPTURE_BYTES_EST until measured).
            memory_ledger.record('quad_built', point='slots=%s' % ','.join(str(slot) for slot in self.pair_label()),
                                 quad_placeholders=list(self.owned), quad_intermediates=list(bucket.owned))
        except BaseException:
            self.buckets.pop(key, None)
            built = locals().get('bucket')
            if built is not None:
                device.validated_native_proposal_masks.discard(addresses(operations, built.mask))
                if built.trace is not None:
                    operations.release_trace(self.mesh, built.trace)
                    built.trace = None
                release_owned(operations, built.owned)
            leaked = self.owned[placeholder_mark:]
            del self.owned[placeholder_mark:]
            release_owned(operations, leaked)
            raise
        self.last_built = True
        return bucket

    def _execute(self, bucket, owned, retain):
        rope = dict(q=bucket.rope['q'], live_k=bucket.rope['live_k'])
        return self.devices[0].execute_proposal(bucket.identifiers, None, bucket.mask, rope, context=None,
            pack=quad_users(self.devices), cached_history=bucket.cached_history, owned=owned, retain=retain,
            stage=lambda name, **values: None, audit=False, quad=self.quad)

    def _update(self, bucket, seeds, *, defer_finish=False):
        from dflash_packed_proposal import note_round_b1, packed_identifiers, round_b1_audit_enabled
        from dflash_proposal_trace import _audit_live_key_rope, pair_mask_audit_enabled, pair_mask_refresh_enabled
        from gdn_multitoken_conv import addresses, release_owned

        devices, operations = self.devices, self.operations
        if any(device.history_rows != CONTEXT for device in devices):
            raise ValueError("Quad proposal replay requires every user at the bucket's committed context")
        if any(device.kv_history.pending is not None or device.position - device.history_rows < 0
               for device in devices):
            raise ValueError('Quad proposal replay requires a fully committed matching K/V frontier')
        users = quad_users(devices)
        if pair_mask_audit_enabled():
            self.audit_mask(bucket)
        note_round_b1('quad-update')
        query, live = quad_rope(users)
        if round_b1_audit_enabled():
            # C8, per pair: the uploaded live-key rows [32p, 32p + 32) against live_key_rope's own build.
            for pair, (first, second) in enumerate(PAIRS):
                rows = tuple(table[:, :, 32 * pair:32 * (pair + 1)] for table in live)
                _audit_live_key_rope(rows, [users[first], users[second]], BLOCK)
        sources = [packed_identifiers(list(seeds), BLOCK, block_width=ROWS), *query, *live]
        destinations = [bucket.identifiers, *bucket.rope['q'], *bucket.rope['live_k']]
        if pair_mask_refresh_enabled():
            sources.append(bucket.host_mask)
            destinations.append(bucket.mask)
            self._note_refresh(bucket)
        for value, destination in zip(sources, destinations, strict=True):
            payload = operations.from_torch(value, dtype=destination.dtype, layout=destination.layout,
                mesh_mapper=operations.ReplicateTensorToMesh(self.mesh))
            operations.copy_host_to_device_tensor(payload, destination)
        protected = self._protected()
        for device in devices:
            protected.extend((*device.kv_history.owned, *device.kv_history.borrowed))
        owned, retain = devices[0].temporaries(protected)

        def copy_cache():
            normalised = 0
            for device, cache in zip(devices, bucket.cached_history, strict=True):
                for active, destination in zip(device.kv_history.active, cache, strict=True):
                    for name in ('k', 'v'):
                        if active[name] is destination[name]:
                            # F4: the quad reads this live bank itself - nothing to copy.
                            continue
                        # The device's live bank on the pool's spare side (an odd swap count before its first
                        # in-place commit): its rows go into the pool's active bank, which the quad reads - the
                        # pair's own normalisation (dflash_proposal_trace._update).
                        value = retain(operations.slice(active[name], (0, 0, 0, 0), (1, 4, CONTEXT, 128)))
                        operations.copy(value, destination[name])
                        normalised += 1
            if normalised:
                from fused_commit import LIVE_BANKS_NORMALISED, log_line as fused_log

                fused_log('%s pair=%s normalised=%d' % (LIVE_BANKS_NORMALISED, self.pair_label(), normalised))
        if defer_finish:
            try:
                copy_cache()
            except BaseException:
                release_owned(operations, owned)
                raise
            return owned
        try:
            copy_cache()
            operations.synchronize_device(self.mesh)
            if [addresses(operations, value) for value in bucket.inputs] != bucket.addresses:
                raise AssertionError('Prepared quad proposal input addresses moved')
        finally:
            release_owned(operations, owned)

    def _note_refresh(self, bucket):
        """dflash_proposal_trace's PAIR_MASK_REFRESH_MARKER, once per process across pairs and the quad: the gate
        requires it whenever the refresh flag is set, and the quad may be the first to refresh."""
        from dflash_proposal_trace import _PAIR_MASK_REFRESH_NOTED, PAIR_MASK_REFRESH_MARKER, _log_line

        if _PAIR_MASK_REFRESH_NOTED:
            return False
        _PAIR_MASK_REFRESH_NOTED.append(True)
        _log_line('%s engaged pair=%s context=%s bytes=%d' % (
            PAIR_MASK_REFRESH_MARKER, self.pair_label(), ','.join(str(value) for value in bucket.context),
            bucket.host_mask.numel() * bucket.host_mask.element_size()))
        return True

    def audit_mask(self, bucket):
        """QWEN_FAST_PAIR_MASK_AUDIT for the quad's mask: read back from each chip, compared bit for bit with the
        kept host mask; one MASK_AUDIT_LINE per chip. Host reads only."""
        import torch

        operations = self.operations
        expected = bucket.host_mask.contiguous().view(torch.int16)
        results = []
        for chip, part in enumerate(operations.get_device_tensors(bucket.mask)):
            actual = operations.to_torch(part).contiguous().view(torch.int16)
            if tuple(actual.shape) != tuple(expected.shape):
                mismatched = max(int(actual.numel()), int(expected.numel()))
            else:
                mismatched = int((actual != expected).sum())
            results.append((chip, int(mismatched == 0), mismatched))
            log_line(MASK_AUDIT_LINE % (self.round_number, int(mismatched == 0), mismatched, chip))
        return results

    def prepare_device(self, seeds):
        """Enqueue the quad's copies and its replay without waiting on the device (the pair's prepare_device for
        four users). False when there is nothing to prewarm: closed, or any device closed, mid-publication or
        under audit."""
        if self.closed:
            return False
        if self._pending is not None:
            self.discard_pending()
        seeds = tuple(seeds)
        if len(seeds) != USERS:
            raise ValueError('One anchor per quad user required')
        if any(device.closed or device.pending is not None or device.progress is not None for device in self.devices):
            return False
        bucket = self._bucket()
        owned = self._update(bucket, seeds, defer_finish=True)
        self.operations.execute_trace(self.mesh, bucket.trace, cq_id=0, blocking=False)
        self._pending = (seeds, bucket, owned)
        return True

    def has_pending(self, which, seed):
        if self._pending is None:
            return False
        seeds, bucket, _ = self._pending
        if which in bucket.consumed:
            return False
        return seed == seeds[which]

    def finish(self, which, count):
        """The pair's finish() for four users: the first call checks the addresses, releases the transients and
        selects (unless the round's batched selection was adopted); each user reads its own tokens."""
        if self._pending is None:
            raise ValueError('No prepared quad proposal is pending')
        seeds, bucket, owned = self._pending
        if bucket.tokens is None:
            from gdn_multitoken_conv import addresses, release_owned

            operations = self.operations
            moved = [addresses(operations, value) for value in bucket.inputs] != bucket.addresses
            release_owned(operations, owned)
            if moved:
                raise AssertionError('Prepared quad proposal input addresses moved')
            bucket.tokens = select_quad_outputs(self.devices[0], bucket.outputs, seeds, (BLOCK - 1,) * USERS)
        tokens = bucket.tokens[which]
        bucket.consumed.add(which)
        if len(bucket.consumed) == USERS:
            self._pending = None
            bucket.tokens, bucket.consumed, bucket.parts = None, set(), None
        return tokens[:count]

    def collect(self):
        """QWEN_FAST_ROUND_B1 (C1), as the pair's: the address check, the transient release and the readback, with
        the seeds and counts finish() would select with. The parts are kept for the shadow audit."""
        if self._pending is None:
            raise ValueError('No prepared quad proposal is pending')
        seeds, bucket, owned = self._pending
        if bucket.tokens is not None or bucket.consumed:
            raise ValueError('This quad proposal was already selected')
        from gdn_multitoken_conv import addresses, release_owned

        operations = self.operations
        self._pending = (seeds, bucket, [])
        moved = [addresses(operations, value) for value in bucket.inputs] != bucket.addresses
        release_owned(operations, owned)
        if moved:
            raise AssertionError('Prepared quad proposal input addresses moved')
        bucket.parts = read_quad_outputs(self.devices[0], bucket.outputs)
        if isinstance(self._audit, list):
            # QWEN_FAST_QUAD_DRAFT_AUDIT: the pairs' readback and every raw read the audit compares, here and not
            # after the selection - select_round's after_collect (early_draft, QWEN_FAST_GDN_AFTER_PAIRS) enqueues
            # the round's deferred GDN commit traces right after this collect, and a trace replay can write into
            # a buffer allocated in its freed holes (dflash_proposal_trace, M0). A failure is the audit's
            # verdict, never the round's.
            try:
                self._audit_reads = read_audit(self, self._audit)
            except Exception as failure:  # noqa: BLE001 - run_audit logs it as equal=0
                self._audit_reads = 'read-error:%s' % type(failure).__name__
        return dict(parts=bucket.parts, seeds=seeds, counts=(BLOCK - 1,) * USERS)

    def audit_selection(self):
        """QWEN_FAST_ROUND_B1_AUDIT (C1): the tokens finish() would have selected itself."""
        if self._pending is None:
            raise ValueError('No prepared quad proposal is pending')
        seeds, bucket, _ = self._pending
        return select_quad_outputs(self.devices[0], bucket.outputs, seeds, (BLOCK - 1,) * USERS)

    def adopt(self, tokens):
        if self._pending is None:
            raise ValueError('No prepared quad proposal is pending')
        _, bucket, _ = self._pending
        tokens = tuple(tokens)
        if bucket.tokens is not None or bucket.consumed or len(tokens) != USERS:
            raise ValueError('All four users of one pending quad proposal must adopt one selection')
        bucket.tokens = tokens

    def snapshot_audit(self):
        """QWEN_FAST_QUAD_DRAFT_AUDIT, before the audit's pair traces are prepared: fence the quad's replay and keep
        its raw outputs (every chunk's values and indices and the features, per chip) as that replay left them.
        The pair replays that follow run only in an audited round, and a quad output the capture was given out of
        a pair capture's freed holes would take their writes; compare_with_pairs then names it (quad-overwritten)
        instead of comparing bytes the pairs wrote. An audit arm's fence, never a timed one's."""
        if self._pending is None:
            raise ValueError('Snapshot a pending quad proposal')
        _, bucket, _ = self._pending
        self._audit_snapshot = None
        self.operations.synchronize_device(self.mesh)
        self._audit_snapshot = raw_outputs(self.operations, bucket.outputs)

    def attach_audit(self, pairs):
        """The round's pair traces, prepared with the quad's seeds before the fence ([(labels, trace)]), or a reason
        string when they could not be."""
        if self._pending is None:
            raise ValueError('Attach the audit to a pending quad proposal')
        self._audit = pairs

    def run_audit(self, round_number):
        """Compare the attached pair traces with this round's quad (compare_with_pairs, on the reads collect()
        took), log AUDIT_LINE, discard the pairs' pending. Never raises: a failure to read or compare is equal=0
        with its stage."""
        audit, self._audit = self._audit, None
        reads, self._audit_reads = self._audit_reads, None
        if audit is None:
            self._audit_snapshot = None
            return None
        if isinstance(audit, str):
            result = (False, audit, 0)
        else:
            try:
                result = (False, reads, 0) if isinstance(reads, str) else compare_with_pairs(self, audit, reads=reads)
            except Exception as failure:  # noqa: BLE001 - the audit line is the verdict
                result = (False, 'error:%s' % type(failure).__name__, 0)
            finally:
                self._audit_snapshot = None
                for _, pair in audit:
                    try:
                        pair.discard_pending()
                    except Exception:  # noqa: BLE001
                        pass
        equal, stage, checks = result
        log_line(AUDIT_LINE % (round_number, int(bool(equal)), stage, USERS, checks))
        return result

    def discard_pending(self):
        audit, self._audit = self._audit, None
        self._audit_snapshot = self._audit_reads = None
        if isinstance(audit, list):
            for _, pair in audit:
                pair.discard_pending()
        if self._pending is None:
            return
        from gdn_multitoken_conv import release_owned

        _, bucket, owned = self._pending
        self._pending = None
        bucket.tokens, bucket.consumed, bucket.parts = None, set(), None
        release_owned(self.operations, owned)

    def close(self):
        if self.closed:
            return
        from gdn_multitoken_conv import addresses, release_owned

        self.operations.synchronize_device(self.mesh)
        self.discard_pending()
        for bucket in self.buckets.values():
            if bucket.trace is not None:
                self.operations.release_trace(self.mesh, bucket.trace)
                bucket.trace = None
            self.devices[0].validated_native_proposal_masks.discard(addresses(self.operations, bucket.mask))
            release_owned(self.operations, bucket.owned)
            bucket.owned.clear()
        release_owned(self.operations, self.owned)
        self.owned.clear()
        self.buckets.clear()
        self.closed = True


# ---------------------------------------------------------------------------------------------
# The shadow audit (QWEN_FAST_QUAD_DRAFT_AUDIT).
# ---------------------------------------------------------------------------------------------

def _same(left, right):
    import torch

    if left.dtype != right.dtype or tuple(left.shape) != tuple(right.shape):
        return False
    image = {torch.bfloat16: torch.int16, torch.float16: torch.int16, torch.float32: torch.int32,
             torch.float64: torch.int64}.get(left.dtype)
    if image is None:
        return torch.equal(left, right)
    return torch.equal(left.contiguous().view(image), right.contiguous().view(image))


def _audited_pairs(quad, pairs):
    """The audit's pairs, refused unless they are the two pair traces (0, 1) and (2, 3) - never the quad."""
    if [list(labels) for labels, _ in pairs] != [list(pair) for pair in PAIRS]:
        raise ValueError('The audit needs the two pair traces (0, 1) and (2, 3)')
    if any(pair is quad or isinstance(pair, PreparedQuadDFlashProposal) for _, pair in pairs):
        raise ValueError('The audit compares the quad with the pair traces, never with itself')
    return pairs


def raw_outputs(operations, outputs):
    """A proposal's raw outputs as host copies: every chunk's values and indices and the features, per chip."""
    def host(tensor):
        return [operations.to_torch(shard) for shard in operations.get_device_tensors(tensor)]

    return dict(chunks=[{key: host(chunk[key]) for key in ('values', 'indices')} for chunk in outputs.chunks],
                projected=host(outputs.projected))


def read_audit(quad, pairs):
    """Every device read the audit compares, taken at once after the round's fence: each pair's collect() (its
    address check, transient release and readback, as select_round collects a pair) and the per-chip values,
    indices and replicated features of the quad and of both pairs, as host copies, beside the quad's snapshot
    from before the pairs replayed (snapshot_audit; required). PreparedQuadDFlashProposal.collect takes it beside
    the quad's own readback, before anything else is enqueued, so the comparison after the selection reads
    nothing a later replay could have written."""
    if quad._pending is None:
        raise ValueError('The audited quad round is not pending')
    _audited_pairs(quad, pairs)
    snapshot = getattr(quad, '_audit_snapshot', None)
    if snapshot is None:
        raise ValueError("The audit needs the quad's outputs as its replay left them (snapshot_audit)")
    _, bucket, _ = quad._pending
    operations = quad.operations
    reads = dict(quad=raw_outputs(operations, bucket.outputs), snapshot=snapshot, pairs=[])
    for labels, pair in pairs:
        result = pair.collect()
        # The pair's pending after its collect(): (seed_a, seed_b, bucket, []).
        reads['pairs'].append(dict(labels=list(labels), result=result, device=pair.device_a,
                                   raw=raw_outputs(operations, pair._pending[2].outputs)))
    return reads


def compare_with_pairs(quad, pairs, reads=None):
    """The quad's round against the two pair traces replayed in the same round: (equal, first differing stage,
    comparisons made). First the quad's raw outputs as it was read back against its snapshot from before the
    pairs replayed (a pair replay that wrote into them is quad-overwritten, never compared); then per user u of
    pair p (row r = u % 2): its selector features, candidates, scores and draft tokens against the pair's; per
    chip, every chunk's values and indices and the replicated features, rows [32p, 32p + 32) of the quad's
    against the pair's. `reads` is read_audit's (collect() takes it); None reads now. The pairs must be pair
    traces, never the quad itself."""
    from dflash_packed_proposal import select_packed_batched

    if quad._pending is None:
        raise ValueError('The audited quad round is not pending')
    seeds, bucket, _ = quad._pending
    if bucket.parts is None or bucket.tokens is None:
        raise ValueError('The audited quad round was not collected and selected')
    _audited_pairs(quad, pairs)
    reads = read_audit(quad, pairs) if reads is None else reads
    if [read['labels'] for read in reads['pairs']] != [list(labels) for labels, _ in pairs]:
        raise ValueError('The audit reads are not these pairs')
    checks = 0

    def verdict(stage):
        return False, stage, checks

    before, after = reads['snapshot'], reads['quad']
    for number, (then, now) in enumerate(zip(before['chunks'], after['chunks'], strict=True)):
        for key in ('values', 'indices'):
            for chip, (left, right) in enumerate(zip(then[key], now[key], strict=True)):
                checks += 1
                if not _same(left.contiguous(), right.contiguous()):
                    return verdict('quad-overwritten:%s:chunk%d:chip%d' % (key, number, chip))
    for chip, (left, right) in enumerate(zip(before['projected'], after['projected'], strict=True)):
        checks += 1
        if not _same(left.contiguous(), right.contiguous()):
            return verdict('quad-overwritten:projected:chip%d' % chip)
    for index, read in enumerate(reads['pairs']):
        labels, result, device = read['labels'], read['result'], read['device']
        pair_seeds = tuple(result['seeds'])
        if pair_seeds != tuple(seeds[user] for user in labels):
            return verdict('seeds:pair%d' % index)
        tokens = select_packed_batched(result['parts'], result['seeds'], result['counts'], device.predecessors,
                                       device.successors)
        for row, user in enumerate(labels):
            mine, theirs = bucket.parts[user], result['parts'][row]
            for stage, key in (('features', 'hidden'), ('candidates', 'candidates'), ('scores', 'unary')):
                checks += 1
                if not _same(mine[key], theirs[key]):
                    return verdict('%s:u%d' % (stage, user))
            checks += 1
            if tuple(bucket.tokens[user]) != tuple(tokens[row]):
                return verdict('tokens:u%d' % user)
        rows = slice(32 * index, 32 * (index + 1))
        for number, (chunk, theirs) in enumerate(zip(reads['quad']['chunks'], read['raw']['chunks'], strict=True)):
            for key in ('values', 'indices'):
                for chip, (left, right) in enumerate(zip(chunk[key], theirs[key], strict=True)):
                    checks += 1
                    mine = left.reshape(ROWS, 16)[rows]
                    if not _same(mine.contiguous(), right.reshape(32, 16).contiguous()):
                        return verdict('%s:chunk%d:chip%d:pair%d' % (key, number, chip, index))
        for chip, (left, right) in enumerate(zip(reads['quad']['projected'], read['raw']['projected'], strict=True)):
            checks += 1
            mine = left.reshape(ROWS, 256)[rows]
            if not _same(mine.contiguous(), right.reshape(32, 256).contiguous()):
                return verdict('projected:chip%d:pair%d' % (chip, index))
    return True, 'all', checks


# ---------------------------------------------------------------------------------------------
# The coordinator's engage-time checks.
# ---------------------------------------------------------------------------------------------

def refusal(devices, batched, environ=None):
    """Why the quad cannot serve these four packable devices (a permanent configuration reason: the coordinator
    disables it with the reason), or None."""
    missing = missing_requirements(environ)
    if missing:
        return 'requires ' + ','.join('%s=1' % name for name in missing)
    if batched is None:
        return 'requires the batched selection (QWEN_FAST_ROUND_B1=1)'
    first = devices[0]
    if any(device.operations is not first.operations or device.mesh is not first.mesh for device in devices):
        return 'the four devices do not share one mesh'
    if any(getattr(device, 'block_rows', None) != BLOCK for device in devices):
        return 'the four devices are not T16'
    if any(not getattr(device, 'fused_convolution', False) for device in devices):
        return 'requires the fused learned convolution'
    if any(device.layers is not first.layers or device.predecessors is not first.predecessors
           or device.successors is not first.successors for device in devices):
        return 'the four devices do not share one draft weight set'
    conv = conv_mode(environ)
    if grid_fits(first.mesh, conv) is False:
        return 'the compute grid cannot hold QWEN_FAST_QUAD_CONV=%s' % conv
    sdpa_mode(environ)
    return None


def elapsed_ms(started):
    return '%.1f' % ((time.perf_counter() - started) * 1000)
