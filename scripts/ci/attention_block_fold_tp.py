"""QWEN_FAST_TP4_ATTN_FOLD (tp4/vglue V3a): the packed block's attention fold-in and fold-out as one launch each.

Served (extent_attention_replay_tp.PackedExtentReplayReader), per attention layer and per packed user segment: a query
Slice, one fold DMA per eight-row group and a Concat into the bundle's stacked (1, 2, 48, 256) query; after the SDPA, a
Slice per group, one inverse fold DMA per group and a Concat back to the segment's rows; then one Concat over the users:
37 glue ops per layer. Every one of them is data movement (slices and concats along the untiled dim 1, and the fold's
row copies), so replacing them with a launch that performs the same row copies is byte-identical by construction; the
SDPA launches, their configs, masks, page tables and cur_pos words are untouched.

  fold-in   ONE launch reads the block query (1, R, 6, 256) at each group's token offset and writes every bundle's
            stacked query (each an output of its own).
  fold-out  ONE launch reads each bundle's SDPA result at the group's batch index and writes the block output
            (1, R, 6, 256) in the memory config the served final Concat wrote.

The kernel is attention_block_fold_tp.cpp: attention_fold_dma_tp.cpp's tile body, looped over a per-core task list whose
entries carry the source and destination buffer, source page base and destination page base. The planners here are pure
(no ttnn) so the CPU tests can run the kernel's index map against the served composition.

Stdlib only at import, py 3.7; ttnn is imported inside the launch builders.
"""

from pathlib import Path

import tp_kernels
import tp_shapes

KERNEL = 'attention_block_fold_tp.cpp'
TILE_COLUMNS = 8                # 256 / 32 tile columns per token
TASK_WORDS = 6                  # source address, destination address, rows, source base, destination base, task
GROUP_ROWS = 8                  # the extent replay's qualified group width


def head_rows():
    return tp_shapes.active().attn_fold_rows


def group_tiles(rows):
    """Tile rows of one folded group (rows * 6 head rows padded to 32): 2 for the qualified eight-row group."""
    return (rows * head_rows() + 31) // 32


class Chunk(object):
    """One SDPA bundle of the block: the segment's first block token, each group's (token offset, rows) inside the
    segment. The served reader dispatches bundle after bundle, group after group, in ascending offsets."""

    def __init__(self, first, groups):
        self.first, self.groups = first, tuple(groups)
        self.rows = groups[0][1]
        if not self.groups or any(rows != self.rows for offset, rows in self.groups) or not 1 <= self.rows <= 8:
            raise ValueError('One bundle of equal groups of one to eight rows required')

    @property
    def batches(self):
        return len(self.groups)

    def tokens(self):
        """The block tokens this bundle covers, in group then token order."""
        return [self.first + offset + token for offset, rows in self.groups for token in range(rows)]


def chunks_of(segments, bundles):
    """Chunks from the block's segments [(first, last)] and each segment's bundles (a list of groups, each a dict with
    'offset' and 'rows', as extent_attention_replay.LAYOUT returns them). Raises when the bundles do not tile the block
    in row order, because the served final Concat is what put the rows in order and this launch writes them by token."""
    if len(segments) != len(bundles):
        raise ValueError('One bundle list per segment required')
    chunks = []
    for (first, last), segment_bundles in zip(segments, bundles):
        for bundle in segment_bundles:
            chunks.append(Chunk(first, [(group['offset'], group['rows']) for group in bundle]))
    covered = [token for chunk in chunks for token in chunk.tokens()]
    total = segments[-1][1] if segments else 0
    if covered != list(range(total)):
        raise ValueError('Bundles must tile the block rows in ascending order')
    return chunks


def forward_tasks(chunks):
    """Per chunk, its fold-in tasks (source base, destination base, task, rows): the served launch's `offset` is the
    group's block token, its output page is the group's batch slot in the stacked tensor plus the tile row and column."""
    plan = []
    for chunk in chunks:
        tiles = group_tiles(chunk.rows)
        plan.append([(chunk.first + offset, batch * tiles * TILE_COLUMNS, task, chunk.rows)
                     for batch, (offset, rows) in enumerate(chunk.groups) for task in range(tiles * TILE_COLUMNS)])
    return plan


def inverse_tasks(chunks):
    """Per chunk, its fold-out tasks (source base, destination base, task, rows): the source is the group's batch of the
    bundle result (its tile rows start batch * tiles), the destination the block output at the group's first token."""
    plan = []
    for chunk in chunks:
        tiles = group_tiles(chunk.rows)
        plan.append([(batch * tiles, (chunk.first + offset) * TILE_COLUMNS, task, chunk.rows)
                     for batch, (offset, rows) in enumerate(chunk.groups) for task in range(chunk.rows * TILE_COLUMNS)])
    return plan


def distribute(tasks, cores):
    """Round-robin the flat task list over `cores` cores: a list of per-core task lists, no core empty when
    len(tasks) >= cores."""
    count = min(len(tasks), cores)
    if count == 0:
        raise ValueError('At least one fold task required')
    return [tasks[index::count] for index in range(count)]


def runtime_arguments(inverse, per_core):
    """The runtime args of each core: [inverse, tasks, then the six words of each task]."""
    return [[int(inverse), len(core)] + [word for task in core for word in task] for core in per_core]


def flat_tasks(plan, sources, destinations):
    """[(source address, destination address, rows, source base, destination base, task)] over every chunk's plan."""
    flat = []
    for chunk_plan, source, destination in zip(plan, sources, destinations):
        for source_base, destination_base, task, rows in chunk_plan:
            flat.append((source, destination, rows, source_base, destination_base, task))
    return flat


def total_tokens(chunks):
    """The block rows the chunks cover (they tile 0..R-1, chunks_of checks)."""
    if not chunks:
        return 0
    last = chunks[-1]
    return last.first + last.groups[-1][0] + last.groups[-1][1]


def problem(query, chunks, memory_config, ttnn):
    """Why the block fold cannot serve this call (None when it can): only the geometry and placement the launch is
    written for. A problem means the served per-segment path runs instead, logged by the caller."""
    total = total_tokens(chunks)
    if tuple(query.shape) != (1, total, head_rows(), 256):
        return 'query %r is not (1, %d, %d, 256)' % (tuple(query.shape), total, head_rows())
    if query.dtype != ttnn.bfloat16 or query.layout != ttnn.TILE_LAYOUT:
        return 'query is not bf16 TILE'
    interleaved = (ttnn.DRAM_MEMORY_CONFIG, getattr(ttnn, 'L1_MEMORY_CONFIG', ttnn.DRAM_MEMORY_CONFIG))
    if query.memory_config() not in interleaved:
        return 'query is not interleaved DRAM or L1'
    if memory_config not in interleaved:
        return 'output memory config is not interleaved DRAM or L1'
    if any(chunk.rows != GROUP_ROWS or chunk.batches != 2 for chunk in chunks):
        return 'bundles are not two eight-row groups'
    return None


def _cores(mesh, tasks):
    grid = mesh.compute_with_storage_grid_size()
    return [(index % grid.x, index // grid.x) for index in range(min(tasks, grid.x * grid.y))]


def _launch(mesh, sources, destinations, plan, *, inverse, tag):
    """One generic_op: the per-chip programs of `sources` (one tensor per chunk, or the one shared tensor repeated) and
    `destinations` (likewise). Every chunk's source tensors share one accessor layout and so do the destinations."""
    import ttnn

    import verify_trace_t1

    chips = tp_shapes.chip_count()
    source_shards = [ttnn.get_device_tensors(tensor) for tensor in sources]
    destination_shards = [ttnn.get_device_tensors(tensor) for tensor in destinations]
    if any(len(shards) != chips for shards in source_shards + destination_shards):
        raise ValueError('%s chips required' % tp_shapes.all_chips())
    rows = max(chunk_rows for chunk_plan in plan for *unused, chunk_rows in chunk_plan)
    points = _cores(mesh, sum(len(chunk_plan) for chunk_plan in plan))
    cores = verify_trace_t1.rectangle_set(ttnn, points)
    buffer = ttnn.CBDescriptor(total_size=(max(4, rows) + 1) * 2048, core_ranges=cores,
        format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=0, data_format=ttnn.bfloat16,
            page_size=2048, tile=ttnn.TileDescriptor(ttnn.Tile([32, 32])))])
    program = ttnn.MeshProgramDescriptor()
    for chip in range(chips):
        local_sources = [shards[chip] for shards in source_shards]
        local_destinations = [shards[chip] for shards in destination_shards]
        source_layouts = [ttnn.TensorAccessorArgs(tensor).get_compile_time_args() for tensor in local_sources]
        destination_layouts = [ttnn.TensorAccessorArgs(tensor).get_compile_time_args() for tensor in local_destinations]
        if any(layout != source_layouts[0] for layout in source_layouts) or any(
                layout != destination_layouts[0] for layout in destination_layouts):
            raise ValueError('Every %s buffer must share one accessor layout' % tag)
        flat = flat_tasks(plan, [tensor.buffer_address() for tensor in local_sources],
                          [tensor.buffer_address() for tensor in local_destinations])
        per_core = distribute(flat, len(points))
        descriptor = ttnn.KernelDescriptor(kernel_source=str(Path(__file__).with_name(KERNEL)), core_ranges=cores,
            defines=tp_kernels.fold_defines(),
            compile_time_args=[*source_layouts[0], *destination_layouts[0]],
            config=ttnn.DataMovementConfigDescriptor(processor=ttnn.DataMovementProcessor.RISCV_0,
                                                     noc=ttnn.NOC.RISCV_0_default))
        runtime = ttnn.RuntimeArgs()
        for (x, y), arguments in zip(points, runtime_arguments(inverse, per_core)):
            runtime[x][y] = arguments
        descriptor.runtime_args = runtime
        coordinate = ttnn.MeshCoordinate(0, chip)
        program[ttnn.MeshCoordinateRange(coordinate, coordinate)] = ttnn.ProgramDescriptor(kernels=[descriptor], cbs=[buffer])
    seen, io = set(), []
    for tensor in [*sources, *destinations]:
        if id(tensor) not in seen:
            seen.add(id(tensor))
            io.append(tensor)
    ttnn.generic_op(io, program)


def fold_in(mesh, query, chunks, owned):
    """The stacked (1, batches, rows * 6, 256) bf16 TILE DRAM query of every chunk, from one launch."""
    import ttnn

    found = tp_shapes.active()
    stacked = []
    for chunk in chunks:
        tensor = ttnn.empty((1, chunk.batches, chunk.rows * found.attn_fold_rows, 256), dtype=ttnn.bfloat16,
                            layout=ttnn.TILE_LAYOUT, device=mesh, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        owned.append(tensor)
        stacked.append(tensor)
    plan = forward_tasks(chunks)
    _launch(mesh, [query] * len(chunks), stacked, plan, inverse=False, tag='fold-in')
    return stacked


def fold_out(mesh, results, chunks, memory_config, owned):
    """The block output (1, R, 6, 256) bf16 TILE in `memory_config`, from one launch over every chunk's result."""
    import ttnn

    total = total_tokens(chunks)
    output = ttnn.empty((1, total, head_rows(), 256), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh,
                        memory_config=memory_config)
    owned.append(output)
    plan = inverse_tasks(chunks)
    _launch(mesh, list(results), [output] * len(chunks), plan, inverse=True, tag='fold-out')
    return output
