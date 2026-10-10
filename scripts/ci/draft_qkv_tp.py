"""The drafter's q, k and v projections as ONE matmul launch, and the head split of the result as one launch (QWEN_FAST_DRAFT_QKV1, default off; F-F3c of the op-fusion programme).

WHY. The device profile of the shipped stack (run 38030902670, trace of the quad) shows each drafter layer projecting the same 64-row block three times: q (52.0 us on 32 cores), k (46.5 us on 8)
and v (46.6 us on 8), each followed by a typecast to bfloat16, then the head split (D2c's tile copy, 2.7 us). The three matmuls read the same `prepared` input, and a matmul at per_core_N = 1 gives
every output tile column its own core, K loop and accumulator: its bits do not depend on how many other columns the launch holds. One launch over a load-time concatenation of the three weights is
48 columns (32 + 8 + 8) on 48 cores, an estimated 55-60 us, one typecast, and one tile-copy launch that places the q, k and v heads. About -85 us and four launches a layer, -0.55 ms a quad.

WHAT. `fused_host_weight` concatenates, per chip, the transposed q, k and v weights column-wise ((5120, 1024 + 256 + 256) a chip) in the host's upload order, so the sharded tensor holds
[q | k | v] on every chip; draft_attention_branch.prepare_attention_branch uploads it beside the separate weights when the flag is on (the separate q, k and v stay: the pair path, the single-user
path and the singles audit run on them, and the lever costs 42 MB (bf8) or 79 MB (bf16) of DRAM a chip over five layers - that is the price of keeping the served paths). `project` runs the fused
matmul with the SAME program config, kernel config and fp32 output as the separate ones, the typecast, and the head split through draft_permute_tp's kernel (a tile permutation: head tile
(h, row tile, dt) <- column tile 4 h + dt of the q columns, 32 + 4 h + dt of the k columns, 40 + 4 h + dt of the v columns). The drafter's norm, rotary and typecasts after the split are the served ops.

EXACT BY CONSTRUCTION. Output columns of a 1D mcast matmul at per_core_N = 1 are independent (same in0_block_w, out_subblock, fp32 dest, packer L1 accumulation off, same activation mcast
order); the typecast is elementwise; the split moves whole tiles. The audit (QWEN_FAST_DRAFT_QKV1_AUDIT, in the bucket's eager warm pass) compares the three split heads with the served
projections' split heads byte for byte on every chip; the draft-singles audit and the accepted-prefix compare are the end-to-end gate.

FLAGS (strict 0 or 1, QWEN_FAST_TP=4 only): QWEN_FAST_DRAFT_QKV1, QWEN_FAST_DRAFT_QKV1_AUDIT (needs the lever). Markers: '[PINDIAG] tp4 draft qkv1 engaged ...', '... fell back ...',
'... audit exact=True ...', '... audit mismatch ...'. A call this module cannot take (no fused weight prepared, another row count, a layout) returns None and the caller runs the served ops.

Stdlib only at import, py 3.7 (torch inside the host-weight builder's caller).
"""

import draft_permute_tp as permute
import tp_shapes

FLAG = 'QWEN_FAST_DRAFT_QKV1'
AUDIT_FLAG = 'QWEN_FAST_DRAFT_QKV1_AUDIT'
ENGAGED = '[PINDIAG] tp4 draft qkv1 engaged'
FALLBACK = '[PINDIAG] tp4 draft qkv1 fell back'
AUDIT = '[PINDIAG] tp4 draft qkv1 audit'
MISMATCH = '[PINDIAG] tp4 draft qkv1 audit mismatch'
RUNTIME_FILES = ('draft_qkv_tp.py',)
TILE = 32
HEAD_DIM = 128
COLUMN_TILES = HEAD_DIM // TILE
GRID = (8, 8)                   # the served projections' program grid


def enabled(environ=None):
    return permute.lever_enabled(FLAG, environ)


def audit_enabled(environ=None):
    return permute.lever_audit_enabled(AUDIT_FLAG, FLAG, environ)


def hook_enabled(environ=None):
    """What the attention branch asks: the lever is on and this call is not a served reference."""
    return permute._SERVED['depth'] == 0 and enabled(environ)


def fused_host_weight(torch, weights, chips):
    """(chips * hidden, q + k + v columns) host weight: chip c's rows are [q_c^T | k_c^T | v_c^T] where x_c is the c-th chunk of the layer's x_proj.weight over its output rows. Sharded on dim 0
    over the mesh it leaves each chip the (hidden, q + k + v) weight whose first columns are the chip's query heads, then its key heads, then its value heads - the separate weights' columns."""
    parts = []
    for name in ('q', 'k', 'v'):
        parts.append([chunk.T.contiguous() for chunk in weights['layers.0.self_attn.%s_proj.weight' % name].chunk(chips, dim=0)])
    return torch.cat([torch.cat([parts[0][chip], parts[1][chip], parts[2][chip]], dim=1) for chip in range(chips)], dim=0)


def split_records(query_heads, kv_heads, rows):
    """The head split as tile moves from one (1, 1, rows, (query_heads + 2 kv_heads) * 128) projection: source 0; destinations 0 / 1 / 2 are the q heads (1, query_heads, rows, 128), the k heads and the v
    heads (1, kv_heads, rows, 128). Head tile (h, row tile, dt) <- column tile (4 h + dt) + the tensor's column offset; consecutive dt are consecutive pages on both sides."""
    if type(rows) is not int or rows < TILE or rows % TILE:
        raise permute.Unsupported('rows must be a whole number of %d-row tiles, got %r' % (TILE, rows))
    row_tiles = rows // TILE
    width = (query_heads + 2 * kv_heads) * COLUMN_TILES
    emit = permute.Emitter()
    for destination, count, offset in ((0, query_heads, 0), (1, kv_heads, query_heads * COLUMN_TILES), (2, kv_heads, (query_heads + kv_heads) * COLUMN_TILES)):
        for head in range(count):
            for row_tile in range(row_tiles):
                for part in range(COLUMN_TILES):
                    emit.tile(destination, (head * row_tiles + row_tile) * COLUMN_TILES + part, 0, row_tile * width + offset + COLUMN_TILES * head + part, False)
    return emit.done()


def _placement_problem(operations, tensor):
    if tensor.dtype != operations.bfloat16 or tensor.layout != operations.TILE_LAYOUT:
        return 'the projection is not bfloat16 TILE'
    if tensor.memory_config() != operations.DRAM_MEMORY_CONFIG:
        return 'the projection is not interleaved DRAM'
    return None


def project(operations, prepared, retain, *, parameters, project, rows, served, site, processors=permute.PROCESSORS, chips=None):
    """The q, k and v head tensors dict(q=, k=, v=) of the 64-row `prepared` block: one fused matmul, one typecast, one split launch; or None when this call cannot take it (the caller
    then runs the served projections). `project(value, weight, grid, rows, columns)` is the branch's own matmul closure (the served program config and kernel config); `served()` the served
    projections' split heads (the audit's reference, run only in the eager warm pass)."""
    found = tp_shapes.active()
    heads, kv_heads = found.draft_heads, found.draft_kv_heads
    weight = (parameters.get('projections') or {}).get('qkv')
    reason = None
    if weight is None:
        reason = 'no fused q|k|v weight was prepared (the flag was off at the attach)'
    elif tuple(prepared.shape) != (1, 1, rows, tp_shapes.HIDDEN):
        reason = 'the block %r is not (1, 1, %d, 5120)' % (tuple(prepared.shape), rows)
    elif rows % TILE:
        reason = 'rows %r are not whole tiles' % (rows,)
    else:
        reason = _placement_problem(operations, prepared)
    per_lane = size = lane_rows = records = mesh = None
    if reason is None:
        try:
            records = permute._cached(('qkv', heads, kv_heads, rows), lambda: split_records(heads, kv_heads, rows))
            mesh = prepared.device()
            per_lane, size, lane_rows = permute._plan_for(mesh, records, 1, 3, processors)
        except permute.Unsupported as failure:
            reason = str(failure)
    if reason is not None:
        permute.note(FALLBACK, 'site=%s reason=%s' % (site, reason))
        return None
    projected = project(prepared, weight, GRID, rows, 1)
    flat = retain(operations.typecast(projected, operations.bfloat16))
    outputs = [operations.empty((1, count, rows, HEAD_DIM), dtype=operations.bfloat16, layout=operations.TILE_LAYOUT, device=mesh,
                                memory_config=operations.DRAM_MEMORY_CONFIG) for count in (heads, kv_heads, kv_heads)]
    try:
        lanes, cores = permute.launch(operations, mesh, [flat], outputs, per_lane, size, lane_rows, processors=processors, chips=chips)
    except permute.Unsupported as failure:
        for tensor in outputs:
            operations.deallocate(tensor)
        permute.note(FALLBACK, 'site=%s reason=%s' % (site, failure))
        return None
    except BaseException:
        for tensor in outputs:
            operations.deallocate(tensor)
        raise
    made = dict((name, retain(tensor)) for name, tensor in zip('qkv', outputs))
    if permute.auditing(AUDIT_FLAG, FLAG):
        with permute.served_only():
            reference = served()
        permute.compare(operations, [(name, made[name], reference[name]) for name in 'qkv'], 'heads', site, audit=AUDIT, mismatch=MISMATCH)
    permute.note(ENGAGED, 'site=%s rows=%d columns=%d heads=%d/%d records=%d lanes=%d cores=%d' % (
        site, rows, (heads + 2 * kv_heads) * COLUMN_TILES, heads, kv_heads, len(records), lanes, cores))
    return made
