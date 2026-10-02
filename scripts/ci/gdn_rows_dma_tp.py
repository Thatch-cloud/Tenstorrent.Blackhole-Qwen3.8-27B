"""The half-tile row mover behind the packed GDN block's split, merge and window stacking (tp4/vglue V2, V1).

The packed verify projects all four users' 16-row segments as ONE (1, 64, W) block, and the recurrence wants one (1, 16, .)
tensor per user. The served path cuts the block per user (a tile Slice for users 0 and 2, whose rows start on a tile
boundary, and an untilize / slice / tilize round trip for users 1 and 3) and joins the four gated outputs with an untilize,
a concat and a tilize: 17 ops per GDN layer. In a TILE tensor a 32-row tile holds two users, faces 0-1 (bytes 0-1023) the
even one and faces 2-3 the odd one, so cutting and joining is moving half tiles. gdn_rows_dma_tp.cpp composes one
destination tile from two half-tile sources (each raw, canonicalised or zero); this module plans those moves and builds the
launches.

Canonicalisation. The served round trip through untilize and tilize maps a bf16 with a zero exponent (-0 and every
denormal) to +0 (gdn_prefill_conv_exact.py: "canonical on both state paths", measured on card M). Users 1 and 3 of the
split therefore come out canonical and users 0 and 2 raw, and every merged output is canonical. The kernel applies that
rule to exactly those halves; the rule is a compile-time define (CANON_DENORM) so an edge probe on a card can flip it.

Pure planners first (no ttnn), the launch builder last. A task is
    (destination index, destination page, (source index, source page, mode), (source index, source page, mode))
with the two sources composing rows 0-15 and 16-31, and mode 0 = zeros, 1 / 2 = raw half 0 / 1, 3 / 4 = canonical half 0 / 1.

Stdlib only at import, py 3.7.
"""

from pathlib import Path

import tp_shapes

KERNEL = 'gdn_rows_dma_tp.cpp'
USER_ROWS = 16                  # a packed user's segment
TILE_ROWS = 32
TASK_WORDS = 8
LANES = 8
MAX_ARGUMENT_WORDS = 256        # a core's runtime args
SCRATCH_BYTES = LANES * 2048

ZERO, RAW, CANON = 0, 1, 3      # mode = ZERO, or RAW / CANON + the source half


class Unsupported(ValueError):
    """The launch this planner would build does not fit the kernel (task list too long, layouts differ): the caller takes
    the served path and says so."""


def mode(half, canonical):
    if half not in (0, 1):
        raise ValueError('A tile has two halves')
    return (CANON if canonical else RAW) + half


def served_canon(user):
    """Whether the served split canonicalises this user's piece: a slice that starts inside a tile (users 1 and 3) goes
    through untilize / slice / tilize, one that starts on a tile boundary (users 0 and 2) is a raw tile slice."""
    return (user * USER_ROWS) % TILE_ROWS != 0


def position(user):
    """(tile row, half) of a user's 16 rows in the 64-row block."""
    start = user * USER_ROWS
    return start // TILE_ROWS, (start % TILE_ROWS) // USER_ROWS


def tile_columns(width):
    return (width + 31) // 32


def split_pieces(users, width):
    """Block -> one (1, 16, width) piece per user. Source 0 is the block, destination u is user u's piece. The piece's
    rows 0-15 are the user's rows (canonical for users 1 and 3), rows 16-31 zero."""
    columns = tile_columns(width)
    tasks = []
    for user in range(users):
        row, half = position(user)
        for column in range(columns):
            tasks.append((user, column, (0, row * columns + column, mode(half, served_canon(user))), (0, 0, ZERO)))
    return tasks


def canon_block(users, width):
    """Block -> the same block with users 1 and 3 canonical. Source 0 is the block, destination 0 the copy. Each tile
    takes both halves from the same source tile."""
    columns = tile_columns(width)
    rows = (users * USER_ROWS + TILE_ROWS - 1) // TILE_ROWS
    tasks = []
    for row in range(rows):
        for column in range(columns):
            page = row * columns + column
            first, second = 2 * row, 2 * row + 1
            second_half = (0, page, mode(1, served_canon(second))) if second < users else (0, 0, ZERO)
            tasks.append((0, page, (0, page, mode(0, served_canon(first))), second_half))
    return tasks


def merge_outputs(users, width):
    """The users' (1, 16, width) outputs -> the (1, 16 * users, width) block, every user canonical (the served concat
    untilizes, joins and tilizes them all). Source u is user u's output, destination 0 the block."""
    columns = tile_columns(width)
    rows = (users * USER_ROWS + TILE_ROWS - 1) // TILE_ROWS
    tasks = []
    for row in range(rows):
        for column in range(columns):
            first, second = 2 * row, 2 * row + 1
            tasks.append((0, row * columns + column, (first, column, mode(0, True)),
                          (second, column, mode(0, True)) if second < users else (0, 0, ZERO)))
    return tasks


def stack_users(users, width, canonical=False):
    """The users' (1, 16, width) tensors -> one (1, 16 * users, width) block, raw (windows) or canonical. Source u is
    user u's tensor, destination 0 the block."""
    columns = tile_columns(width)
    rows = (users * USER_ROWS + TILE_ROWS - 1) // TILE_ROWS
    tasks = []
    for row in range(rows):
        for column in range(columns):
            first, second = 2 * row, 2 * row + 1
            tasks.append((0, row * columns + column, (first, column, mode(0, canonical)),
                          (second, column, mode(0, canonical)) if second < users else (0, 0, ZERO)))
    return tasks


def unstack_users(users, block_columns, first_column, columns):
    """`columns` tile columns of a (1, 16 * users, .) block starting at tile column `first_column` -> one (1, 16, .) tensor
    per user (rows 0-15 the user's rows, rows 16-31 zero). Source 0 is the block, destination u the user's tensor."""
    tasks = []
    for user in range(users):
        row, half = position(user)
        for column in range(columns):
            tasks.append((user, column, (0, row * block_columns + first_column + column, mode(half, False)), (0, 0, ZERO)))
    return tasks


def distribute(tasks, cores):
    """Round-robin the flat task list over at most `cores` cores."""
    count = min(len(tasks), cores)
    if count == 0:
        raise Unsupported('At least one move required')
    per_core = [tasks[index::count] for index in range(count)]
    if 1 + TASK_WORDS * max(len(core) for core in per_core) > MAX_ARGUMENT_WORDS:
        raise Unsupported('%d tasks do not fit %d cores at %d words each' % (len(tasks), cores, TASK_WORDS))
    return per_core


def capacity(per_core):
    """The most tasks any core carries. It is the LAST compile-time arg of the launch (so it is part of the program cache
    key) and every core's runtime-arg list is padded to 1 + TASK_WORDS * capacity words: generic_op caches on the kernel,
    defines, compile-time args and cores but NOT on runtime-arg lengths, and a cache hit writes this launch's lists into
    the cached slots (hardware window N1: a 1800-task unstack hit the 640-task window stack's 49-word slots)."""
    return max(len(core) for core in per_core)


def runtime_arguments(per_core, sources, destinations):
    """Each core's [tasks, then eight words per task, zero padded to 1 + TASK_WORDS * capacity] from per-launch address
    lists (`sources`, `destinations`: one buffer address per tensor index, for one chip)."""
    size = 1 + TASK_WORDS * capacity(per_core)
    lists = []
    for core in per_core:
        words = [len(core)]
        for destination, page, first, second in core:
            words += [destinations[destination], page]
            for source, source_page, source_mode in (first, second):
                words += [sources[source], source_page, source_mode]
        lists.append(words + [0] * (size - len(words)))
    return lists


def problem(tensors, ttnn):
    """Why these tensors cannot take part in one launch (None when they can): same dtype and layout, interleaved."""
    for tensor in tensors:
        if tensor.dtype != ttnn.bfloat16 or tensor.layout != ttnn.TILE_LAYOUT:
            return 'a tensor is not bf16 TILE'
        if tensor.memory_config() not in (ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG):
            return 'a tensor is not interleaved DRAM or L1'
    return None


def launch(mesh, sources, destinations, tasks, *, canon_denorm=True):
    """One generic_op that runs `tasks` over per-chip programs. Every source shares one accessor layout and every
    destination another; anything else is Unsupported."""
    import ttnn

    import verify_trace_t1

    chips = tp_shapes.chip_count()
    reason = problem([*sources, *destinations], ttnn)
    if reason is not None:
        raise Unsupported(reason)
    source_shards = [ttnn.get_device_tensors(tensor) for tensor in sources]
    destination_shards = [ttnn.get_device_tensors(tensor) for tensor in destinations]
    if any(len(shards) != chips for shards in source_shards + destination_shards):
        raise Unsupported('%s chips required' % tp_shapes.all_chips())
    grid = mesh.compute_with_storage_grid_size()
    count = min(len(tasks), grid.x * grid.y)
    points = [(index % grid.x, index // grid.x) for index in range(count)]
    per_core = distribute(tasks, count)
    cores = verify_trace_t1.rectangle_set(ttnn, points)
    buffer = ttnn.CBDescriptor(total_size=SCRATCH_BYTES, core_ranges=cores,
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
            raise Unsupported('Every source (and every destination) of one launch must share an accessor layout')
        descriptor = ttnn.KernelDescriptor(kernel_source=str(Path(__file__).with_name(KERNEL)), core_ranges=cores,
            defines=[('CANON_DENORM', '1' if canon_denorm else '0')],
            compile_time_args=[*source_layouts[0], *destination_layouts[0], capacity(per_core)],
            config=ttnn.DataMovementConfigDescriptor(processor=ttnn.DataMovementProcessor.RISCV_0,
                                                     noc=ttnn.NOC.RISCV_0_default))
        runtime = ttnn.RuntimeArgs()
        arguments = runtime_arguments(per_core, [tensor.buffer_address() for tensor in local_sources],
                                      [tensor.buffer_address() for tensor in local_destinations])
        for (x, y), words in zip(points, arguments):
            runtime[x][y] = words
        descriptor.runtime_args = runtime
        coordinate = ttnn.MeshCoordinate(0, chip)
        program[ttnn.MeshCoordinateRange(coordinate, coordinate)] = ttnn.ProgramDescriptor(kernels=[descriptor], cbs=[buffer])
    seen, io = set(), []
    for tensor in [*sources, *destinations]:
        if id(tensor) not in seen:
            seen.add(id(tensor))
            io.append(tensor)
    ttnn.generic_op(io, program)
