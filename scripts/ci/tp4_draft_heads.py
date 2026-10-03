"""The drafter's head split and head merge as one tile-copy launch each at four cards (QWEN_FAST_TP4_DRAFT_HEADS, default off; D2c).

WHY. The drafter's attention branch runs, per layer, a K|V concat (Concat, 1.7 us) and nlp_create_qkv_heads (one core, 19.8 us) to
split the projections into heads, and nlp_concat_heads (one core, 13.3 us) to merge the attention result: about 35 us a layer, five
layers a pass. The pair path also pads the 32-row query to the key rows and slices the heads back.

WHAT. With head_dim 128 (four whole tiles) and a row count that is a whole number of tiles, both ops move no element inside a tile.
For query / key / value (1, 1, rows, heads * 128) the head split is the tile permutation

    heads tile (h, rt, dt)  <-  projection tile (rt, 4 h + dt)         (the K and V heads from the key and the value tensor)

and the merge is its inverse. One generic_op (draft_heads_copy.cpp) copies each tile from the source page to the destination page
over up to 24 cores, reading key and value directly (no concat) and the query's own 32 rows (no pad, no slice). The bytes are the
ops' bytes: the drafter's proposals are identical. test_tp4_draft_heads holds the permutation against a torch reshape / permute
reference of both ops at the drafter's per-chip head counts.

FLAG OFF. The twins (draft_head_layout_tp, quad_draft_tp) call their served functions exactly as before and this module is not
imported. A call this module cannot take (another dtype, layout or placement, head_dim, a row count that is not a whole number of
tiles, more tasks than a core's argument budget) is handed to the served function with a 'fell back' line.

Stdlib only at import, py 3.7.
"""

from pathlib import Path

import tp4_sampdraft
import tp_shapes

KERNEL = 'draft_heads_copy.cpp'
TILE = 32
HEAD_DIM = 128
HEAD_TILES = HEAD_DIM // TILE            # 4 tiles per head
TILE_BYTES = 2048
TASK_WORDS = 4
LANES = 8
MAX_CORES = 24
GRID_WIDTH = 8
MAX_ARGUMENT_WORDS = 256


class Unsupported(ValueError):
    """The launch this planner would build does not fit the kernel: the caller takes the served path and says so."""


def split_tasks(heads, kv_heads, query_rows, kv_rows):
    """The head split as tile moves (source slot, source page, destination slot, destination page): slots 0 / 1 / 2 are the query,
    the key and the value on the source side and the query heads, key heads and value heads on the destination side."""
    for rows in (query_rows, kv_rows):
        if type(rows) is not int or rows < TILE or rows % TILE:
            raise Unsupported('rows must be a whole number of %d-row tiles, got %r' % (TILE, rows))
    tasks = []
    for slot, count, rows in ((0, heads, query_rows), (1, kv_heads, kv_rows), (2, kv_heads, kv_rows)):
        row_tiles = rows // TILE
        width_tiles = count * HEAD_TILES
        for head in range(count):
            for row_tile in range(row_tiles):
                for part in range(HEAD_TILES):
                    tasks.append((slot, row_tile * width_tiles + HEAD_TILES * head + part, slot,
                                  (head * row_tiles + row_tile) * HEAD_TILES + part))
    return tasks


def merge_tasks(heads, rows):
    """The head merge (nlp_concat_heads) as tile moves: (heads (1, heads, rows, 128)) page -> (1, 1, rows, heads * 128) page."""
    if type(rows) is not int or rows < TILE or rows % TILE:
        raise Unsupported('rows must be a whole number of %d-row tiles, got %r' % (TILE, rows))
    row_tiles = rows // TILE
    width_tiles = heads * HEAD_TILES
    return [(0, (head * row_tiles + row_tile) * HEAD_TILES + part, 0, row_tile * width_tiles + HEAD_TILES * head + part)
            for head in range(heads) for row_tile in range(row_tiles) for part in range(HEAD_TILES)]


def distribute(tasks, cores):
    """Deal the tasks over `cores` cores in contiguous runs (each core's tasks are one run of one destination)."""
    if not tasks:
        raise Unsupported('At least one move required')
    count = max(1, min(cores, len(tasks)))
    size = -(-len(tasks) // count)
    runs = [tasks[index:index + size] for index in range(0, len(tasks), size)]
    if 1 + TASK_WORDS * max(len(run) for run in runs) > MAX_ARGUMENT_WORDS:
        raise Unsupported('%d tasks do not fit %d cores at %d words each' % (len(tasks), cores, TASK_WORDS))
    return runs


def capacity(runs):
    """The most tasks any core carries: the last compile-time arg of the launch, and every core's runtime-arg list is padded to
    1 + TASK_WORDS * capacity (generic_op's program cache does not hash runtime-arg lengths)."""
    return max(len(run) for run in runs)


def runtime_arguments(runs, sources, destinations):
    """Each core's [tasks, then four words per task, zero padded to 1 + TASK_WORDS * capacity] from per-chip address lists."""
    size = 1 + TASK_WORDS * capacity(runs)
    lists = []
    for run in runs:
        words = [len(run)]
        for source, source_page, destination, destination_page in run:
            words += [sources[source], source_page, destinations[destination], destination_page]
        lists.append(words + [0] * (size - len(words)))
    return lists


def _tiled_dram(operations, tensors):
    return all(tensor.dtype == operations.bfloat16 and tensor.layout == operations.TILE_LAYOUT
               and tensor.memory_config() == operations.DRAM_MEMORY_CONFIG for tensor in tensors)


def launch(operations, mesh, sources, destinations, tasks):
    """One generic_op copying `tasks` from `sources` to `destinations` (tensors, indexed by slot), per chip."""
    chips = tp_shapes.chip_count()
    source_shards = [operations.get_device_tensors(tensor) for tensor in sources]
    destination_shards = [operations.get_device_tensors(tensor) for tensor in destinations]
    if any(len(shards) != chips for shards in source_shards + destination_shards):
        raise Unsupported('%s chips required' % tp_shapes.all_chips())
    runs = distribute(tasks, MAX_CORES)
    points = [(index % GRID_WIDTH, index // GRID_WIDTH) for index in range(len(runs))]
    import verify_trace_t1

    cores = verify_trace_t1.rectangle_set(operations, points)
    buffer = operations.CBDescriptor(total_size=LANES * TILE_BYTES, core_ranges=cores,
        format_descriptors=[operations.CBFormatDescriptor(buffer_index=0, data_format=operations.bfloat16,
            page_size=TILE_BYTES, tile=operations.TileDescriptor(operations.Tile([32, 32])))])
    program = operations.MeshProgramDescriptor()
    for chip in range(chips):
        local_sources = [shards[chip] for shards in source_shards]
        local_destinations = [shards[chip] for shards in destination_shards]
        source_layouts = [list(operations.TensorAccessorArgs(tensor).get_compile_time_args()) for tensor in local_sources]
        destination_layouts = [list(operations.TensorAccessorArgs(tensor).get_compile_time_args())
                               for tensor in local_destinations]
        if any(layout != source_layouts[0] for layout in source_layouts) or any(
                layout != destination_layouts[0] for layout in destination_layouts):
            raise Unsupported('Every source (and every destination) of one launch must share an accessor layout')
        runtime = operations.RuntimeArgs()
        arguments = runtime_arguments(runs, [tensor.buffer_address() for tensor in local_sources],
                                      [tensor.buffer_address() for tensor in local_destinations])
        for (x, y), words in zip(points, arguments):
            runtime[x][y] = words
        descriptor = operations.KernelDescriptor(kernel_source=str(Path(__file__).with_name(KERNEL)), core_ranges=cores,
            compile_time_args=[*source_layouts[0], *destination_layouts[0], capacity(runs)], runtime_args=runtime,
            config=operations.DataMovementConfigDescriptor(processor=operations.DataMovementProcessor.RISCV_0,
                                                           noc=operations.NOC.RISCV_0_default))
        coordinate = operations.MeshCoordinate(0, chip)
        program[operations.MeshCoordinateRange(coordinate, coordinate)] = operations.ProgramDescriptor(
            kernels=[descriptor], cbs=[buffer])
    seen, io = set(), []
    for tensor in [*sources, *destinations]:
        if id(tensor) not in seen:
            seen.add(id(tensor))
            io.append(tensor)
    operations.generic_op(io, program)
    return len(runs)


def split_heads(operations, query, key, value, retain, *, served, site):
    """(query heads, key heads, value heads) as dict(q=, k=, v=) for (1, 1, rows, heads * 128) projections: the tile-copy launch,
    or `served()` (the twin's served split) when this call cannot take it."""
    found = tp_shapes.active()
    heads, kv_heads = found.draft_heads, found.draft_kv_heads
    query_rows = query.shape[2] if len(query.shape) == 4 else 0
    kv_rows = key.shape[2] if len(key.shape) == 4 else 0
    reason = None
    if (tuple(query.shape) != (1, 1, query_rows, heads * HEAD_DIM) or tuple(key.shape) != (1, 1, kv_rows, kv_heads * HEAD_DIM)
            or tuple(value.shape) != tuple(key.shape)):
        reason = 'shapes %r %r %r are not (1, 1, rows, heads * %d)' % (tuple(query.shape), tuple(key.shape),
                                                                       tuple(value.shape), HEAD_DIM)
    elif not _tiled_dram(operations, (query, key, value)):
        reason = 'an operand is not interleaved DRAM bfloat16 tiled'
    tasks = None
    if reason is None:
        try:
            tasks = split_tasks(heads, kv_heads, query_rows, kv_rows)
            distribute(tasks, MAX_CORES)
        except Unsupported as failure:
            reason = str(failure)
    if reason is not None:
        tp4_sampdraft.note(tp4_sampdraft.HEADS_FALLBACK, 'site=%s reason=%s' % (site, reason))
        return served()
    mesh = query.device()
    outputs = [operations.empty((1, count, rows, HEAD_DIM), dtype=operations.bfloat16, layout=operations.TILE_LAYOUT,
                                device=mesh, memory_config=operations.DRAM_MEMORY_CONFIG)
               for count, rows in ((heads, query_rows), (kv_heads, kv_rows), (kv_heads, kv_rows))]
    try:
        cores = launch(operations, mesh, [query, key, value], outputs, tasks)
    except BaseException:
        for tensor in outputs:
            operations.deallocate(tensor)
        raise
    query_heads, key_heads, value_heads = (retain(tensor) for tensor in outputs)
    tp4_sampdraft.note(tp4_sampdraft.HEADS_ENGAGED, 'site=%s op=split rows=%dx%d heads=%d/%d tiles=%d cores=%d' % (
        site, query_rows, kv_rows, heads, kv_heads, len(tasks), cores))
    return dict(q=query_heads, k=key_heads, v=value_heads)


def merge_heads(operations, value, retain, *, served, site):
    """nlp_concat_heads for (1, heads, rows, 128): the tile-copy launch, or `served()` when this call cannot take it."""
    found = tp_shapes.active()
    heads = found.draft_heads
    rows = value.shape[2] if len(value.shape) == 4 else 0
    reason = None
    if tuple(value.shape) != (1, heads, rows, HEAD_DIM):
        reason = 'shape %r is not (1, %d, rows, %d)' % (tuple(value.shape), heads, HEAD_DIM)
    elif not _tiled_dram(operations, (value,)):
        reason = 'the operand is not interleaved DRAM bfloat16 tiled'
    tasks = None
    if reason is None:
        try:
            tasks = merge_tasks(heads, rows)
            distribute(tasks, MAX_CORES)
        except Unsupported as failure:
            reason = str(failure)
    if reason is not None:
        tp4_sampdraft.note(tp4_sampdraft.HEADS_FALLBACK, 'site=%s reason=%s' % (site, reason))
        return served()
    mesh = value.device()
    output = operations.empty((1, 1, rows, heads * HEAD_DIM), dtype=operations.bfloat16, layout=operations.TILE_LAYOUT,
                              device=mesh, memory_config=operations.DRAM_MEMORY_CONFIG)
    try:
        cores = launch(operations, mesh, [value], [output], tasks)
    except BaseException:
        operations.deallocate(output)
        raise
    result = retain(output)
    tp4_sampdraft.note(tp4_sampdraft.HEADS_ENGAGED, 'site=%s op=merge rows=%d heads=%d tiles=%d cores=%d' % (
        site, rows, heads, len(tasks), cores))
    return result
