"""Q4 probe-local candidates: the pieces of one four-user, 64-row draft pass that Q0 qualifies on card B.

NOT SHIPPED. These are the probe's copies of what quad-draft-plan.md section 3 would add to scripts/ci/quad_draft.py:
  - the K/V plan with pair-identical pads (section 3.4, R2);
  - the four-way GQA fold built from pair_row_exact's own, unchanged functions (section 3.3, R9);
  - the dense, unfolded 4-segment mask (Q0b's negative control);
  - the fused conv's program for ONE chip: the served 32-row call (draft_convolution_fused.fused_convolution's
    program, field for field, at mesh coordinate (0, 0)) and E1 / E1b, which run quad_conv_io.cpp - the probe-local
    64-row copy of the served I/O kernel - with the served compute kernel untouched (section 3.5);
  - the H0 candidate concat (section 3.6, Q0c-lite).
Every function takes the ttnn module as `operations` and a `retain` callable, as the served code does, so the torch
stand-in in test_quad_draft_probe.py drives them unchanged. Nothing here imports ttnn.
"""

from pathlib import Path

CONTEXT, BLOCK, SPAN = 2048, 16, 2080
USERS, ROWS = 4, 64
HEADS, KV_HEADS, HEAD_DIM = 16, 4, 128
GROUP = HEADS // KV_HEADS
QUAD_KEYS = USERS * SPAN                      # 8320
QUAD_QUERY_HEADS, QUAD_KEY_HEADS = 4 * HEADS, 4 * KV_HEADS
HIDDEN = 5120
TILE_PAGES = HIDDEN // 32                     # 160 pages per tile row of the hidden block
DYNAMIC_TILES = 320 // 32                     # 10 dynamic-kernel tiles per tile row
QUAD_CONV_KERNEL = 'quad_conv_io.cpp'
SERVED_CONV_IO = 'draft_convolution_fused_io.cpp'
SERVED_CONV_COMPUTE = 'draft_convolution_fused_compute.cpp'
# The conv variants: workers, the grid they tile, and how worker w maps to its core.
#   served32  the served call: 80 workers on the 8x10 grid, core (w % 8, w // 8), 160 pages (2 each)
#   E1        80 workers on the same cores, 320 pages (4 each)
#   E1b       110 workers on the 11x10 grid, column-major core (w // 10, w % 10), 320 pages: workers 0-99 take
#             three, workers 100-109 (column 10) take two
CONV_VARIANTS = {
    'served32': dict(workers=80, grid=(8, 10), core=lambda w: (w % 8, w // 8), kernel=SERVED_CONV_IO),
    'E1': dict(workers=80, grid=(8, 10), core=lambda w: (w % 8, w // 8), kernel=QUAD_CONV_KERNEL),
    'E1b': dict(workers=110, grid=(11, 10), core=lambda w: (w // 10, w % 10), kernel=QUAD_CONV_KERNEL),
}


# ---------------------------------------------------------------------------------------------
# K/V assembly (section 3.4).
# ---------------------------------------------------------------------------------------------

def quad_key_value_plan():
    """The 12 pieces of the quad key axis, in order: for each user u (pair p = u // 2) its cached bank (2048 rows),
    its live rows [16u, 16u + 16) of the 64-row block, then 16 pad rows taken from rows [32p, 32p + 16) - exactly
    what the pair assembly puts in the pad (its `start = 0` is row 0 of the PAIR's own 32-row block). Every segment
    is then byte-identical to its pair assembly's segment (R2)."""
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


def host_assembly(torch, caches, block, plan=None):
    """The quad K/V assembled on the host from the same pieces: {'k': (1, 4, 8320, 128), 'v': ...}."""
    plan = quad_key_value_plan() if plan is None else plan
    out = {}
    for name in 'kv':
        pieces = []
        for part in plan:
            if part['kind'] == 'cached':
                pieces.append(caches[part['user']][name])
            else:
                pieces.append(block[name][:, :, part['source'].start:part['source'].stop])
        out[name] = torch.cat(pieces, dim=2).contiguous()
    return out


def assemble_quad(operations, caches, block, retain, *, plan=None):
    """The device assembly: 12 pieces per K and V, one concat each (today's untilize -> RM concat -> tilize path)."""
    plan = quad_key_value_plan() if plan is None else plan
    heads = {}
    for name in 'kv':
        pieces = []
        for part in plan:
            if part['kind'] == 'cached':
                pieces.append(caches[part['user']][name])
                continue
            start, stop = part['source'].start, part['source'].stop
            pieces.append(retain(operations.slice(block[name], (0, 0, start, 0), (1, KV_HEADS, stop, HEAD_DIM))))
        heads[name] = retain(operations.concat(pieces, dim=2, memory_config=operations.DRAM_MEMORY_CONFIG))
    return heads['k'], heads['v']


# ---------------------------------------------------------------------------------------------
# The four-way fold (section 3.3).
# ---------------------------------------------------------------------------------------------

def validate_quad(operations, query, key, value, mask):
    if (tuple(query.shape) != (1, HEADS, ROWS, HEAD_DIM) or tuple(key.shape) != (1, KV_HEADS, QUAD_KEYS, HEAD_DIM)
            or tuple(value.shape) != tuple(key.shape) or tuple(mask.shape) != (1, 1, 32, SPAN)):
        raise ValueError('The quad fold takes the (1, 16, 64, 128) query, the four-segment (1, 4, 8320, 128) keys '
                         'and values and the single-user (1, 1, 32, 2080) mask')
    if any(tensor.dtype != operations.bfloat16 for tensor in (query, key, value, mask)):
        raise ValueError('BF16 quad draft attention operands required')


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


def quad_fold_attention(operations, query, key, value, mask, retain, *, mask_validated=False):
    """The quad's draft attention: every (Q head, KV head) unit is the pair fold's unit for that user, byte for
    byte in Q, K, V and mask (given the plan's pads); only the compile-time NQH/NKH differ (64 / 16)."""
    from pair_row_exact import folded_sdpa

    if mask_validated is not True:
        raise ValueError('Validate the single-user host mask before upload and replay')
    validate_quad(operations, query, key, value, mask)
    folded = quad_fold_query(operations, query, retain)
    keys = quad_fold_keys(operations, key, retain)
    values = quad_fold_keys(operations, value, retain)
    attention = retain(folded_sdpa(operations, folded, keys, values, mask))
    return quad_unfold_output(operations, attention, retain)


def dense_quad_mask(torch):
    """Q0b's negative control: the dense, unfolded 4-segment mask (1, 1, 64, 8320). Rows [16u, 16u + 16) see
    segment u exactly as the single-user mask's live rows see its 2080 keys; nothing else is visible."""
    from dflash_batched_mask import batched_attention_mask

    single = batched_attention_mask([CONTEXT], BLOCK)
    mask = torch.full((1, 1, ROWS, QUAD_KEYS), float('-inf'), dtype=torch.bfloat16)
    for user in range(USERS):
        mask[:, :, BLOCK * user:BLOCK * (user + 1), SPAN * user:SPAN * (user + 1)] = single[:, :, :BLOCK]
    return mask


def quad_head_map():
    """{folded query head: (h, u, j, kv head)}: the fold's head arithmetic, for the CPU test."""
    out = {}
    for h in range(KV_HEADS):
        for u in range(USERS):
            for j in range(GROUP):
                out[16 * h + 4 * u + j] = (h, u, j, 4 * h + u)
    return out


# ---------------------------------------------------------------------------------------------
# The fused conv on one chip (section 3.5; Q0d).
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
    """E1/E1b's per-tile-row seam words (low, high) for a 1-64 row block. A 64-row block must begin a segment at
    row 32: no causal carry may cross a tile row (the kernel shifts within a tile), which the four users at rows
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


def conv_pages(variant, rows):
    """{worker: [pages]} for a variant: page = worker, worker + workers, ... below 160 per tile row."""
    spec = CONV_VARIANTS[variant]
    total = TILE_PAGES * ((rows + 31) // 32)
    if variant == 'served32' and rows > 32:
        raise ValueError('The served conv takes at most 32 rows')
    return {worker: list(range(worker, total, spec['workers'])) for worker in range(spec['workers'])}


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


def conv_program(operations, shards, variant, *, kernel_dir, served_dir, seams, compute_groups=None):
    """One chip's fused-conv program at mesh coordinate (0, 0). `shards` are the six single-device tensors
    [hidden, dynamic0, dynamic1, base0, base1, output]. `seams`: the served call's one word, or E1/E1b's
    (low, high) words. The served call is draft_convolution_fused.fused_convolution's program for one chip, field
    for field (CBs 0/1/16 of 7/2/1 pages, compute compile args [2], runtime args addresses + [rows, worker,
    seams]). E1/E1b run quad_conv_io.cpp (runtime args addresses + [rows, worker, workers, low, high]) and the
    served compute kernel with one compile arg per group of workers that take the same number of pages.
    `compute_groups` overrides that grouping (the CPU test's fault)."""
    spec = CONV_VARIANTS[variant]
    rows = shards[0].shape[2]
    pages = conv_pages(variant, rows)
    coordinates = [spec['core'](worker) for worker in range(spec['workers'])]
    cores = core_ranges(operations, coordinates)
    buffers = [operations.CBDescriptor(total_size=2048 * count, core_ranges=cores,
        format_descriptors=[operations.CBFormatDescriptor(buffer_index=index, data_format=operations.bfloat16,
            page_size=2048, tile=operations.TileDescriptor(operations.Tile([32, 32])))])
        for index, count in ((0, 7), (1, 2), (16, 1))]
    served = Path(served_dir)
    if compute_groups is None:
        compute_groups = {}
        for worker, owned in pages.items():
            compute_groups.setdefault(len(owned), []).append(coordinates[worker])
    computes = [operations.KernelDescriptor(kernel_source=str(served / SERVED_CONV_COMPUTE),
        core_ranges=core_ranges(operations, group), compile_time_args=[count], config=operations.ComputeConfigDescriptor(
            math_fidelity=operations.MathFidelity.HiFi4, fp32_dest_acc_en=True, math_approx_mode=False))
        for count, group in sorted(compute_groups.items())]
    addresses = [value.buffer_address() for value in shards]
    if addresses[-1] in addresses[:-1]:
        raise ValueError('Convolution output must not alias borrowed inputs')
    runtime = operations.RuntimeArgs()
    for worker, (x, y) in enumerate(coordinates):
        if variant == 'served32':
            runtime[x][y] = addresses + [rows, worker, int(seams)]
        else:
            low, high = seams
            runtime[x][y] = addresses + [rows, worker, spec['workers'], int(low), int(high)]
    kernel = Path(kernel_dir) / spec['kernel'] if variant != 'served32' else served / SERVED_CONV_IO
    reader = operations.KernelDescriptor(kernel_source=str(kernel), core_ranges=cores,
        compile_time_args=[argument for value in shards
                           for argument in operations.TensorAccessorArgs(value).get_compile_time_args()],
        runtime_args=runtime, config=operations.DataMovementConfigDescriptor(
            processor=operations.DataMovementProcessor.RISCV_0, noc=operations.NOC.RISCV_0_default))
    program = operations.MeshProgramDescriptor()
    coordinate = operations.MeshCoordinate(0, 0)
    program[operations.MeshCoordinateRange(coordinate, coordinate)] = operations.ProgramDescriptor(
        kernels=[reader, *computes], cbs=buffers)
    return program


def fits_grid(variant, grid):
    """Whether a device's compute grid (x, y) holds the variant's cores."""
    need = CONV_VARIANTS[variant]['grid']
    return grid[0] >= need[0] and grid[1] >= need[1]


# ---------------------------------------------------------------------------------------------
# H0 candidate concat (section 3.6; Q0c-lite).
# ---------------------------------------------------------------------------------------------

def concat_candidates(operations, first, second, retain, *, indices_u32=False):
    """Each chunk's values and indices of two 32-row halves concatenated on dim 2 (the device side of the H0
    readback). `indices_u32` casts the uint16 indices to uint32 first (the plan's fallback)."""
    memory = operations.DRAM_MEMORY_CONFIG
    out = []
    for top, bottom in zip(first, second):
        if (top['start'], top['stop']) != (bottom['start'], bottom['stop']):
            raise ValueError('The halves must carry the same candidate chunks')
        values = retain(operations.concat([top['values'], bottom['values']], dim=2, memory_config=memory))
        parts = [top['indices'], bottom['indices']]
        if indices_u32:
            parts = [retain(operations.typecast(part, operations.uint32)) for part in parts]
        indices = retain(operations.concat(parts, dim=2, memory_config=memory))
        out.append(dict(start=top['start'], stop=top['stop'], values=values, indices=indices))
    return out
