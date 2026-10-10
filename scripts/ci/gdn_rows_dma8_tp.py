"""The quarter-tile row mover behind the octo block's GDN split, merge and window stacking (QWEN_FAST_OCTO_GLUE8, tp4/octo-2 lever 2).

gdn_rows_dma_tp.py moves the half tiles of the M3 block's sixteen-row users (V2, V1). The octo block (docs/tp4-octo.md) packs EIGHT users of EIGHT rows into the same
64-row block: user u sits in tile row u // 4, quarter u % 4 (rows 8q .. 8q + 7). An eight-row user is a quarter tile, which that kernel does not move, so the octo block ran the
SERVED ops at those sites (a tile Slice or an untilize / slice / tilize round trip per user, an untilize / concat / tilize for the join). This module is the same lever for
quarter tiles: gdn_rows_dma8_tp.cpp composes one destination tile from four quarter-tile sources (each raw, canonicalised or zero); this module plans those moves and builds
the launches. gdn_rows_dma_tp is untouched and the M3 path never imports this one.

Canonicalisation, per user position. The served round trip through untilize and tilize maps a bf16 with a zero exponent (-0 and every denormal) to +0 (gdn_prefill_conv_exact.py:
"canonical on both state paths", measured on card M). It happens to a slice that starts INSIDE a tile and not to one that starts on a tile boundary (the V2 audit's rule for users 1
and 3 of the M3 block). At eight rows a user starts at row 8u, so users 0 and 4 start on a tile boundary and come out RAW, and users 1, 2, 3, 5, 6 and 7 start inside a tile and come
out CANONICAL; every merged output is canonical. This is the sixteen-row rule extended by analogy (the slice path does not look at the slice length); no card has measured it
at eight rows, which is exactly what QWEN_FAST_TP4_VGLUE_AUDIT compares in-trace (piece user k, merged output) and the edge probe would settle.

Pure planners first (no ttnn), the launch builder last. A task is
    (destination index, destination page, (quarter 0 source, quarter 1 source, quarter 2 source, quarter 3 source))
with each source (source index, source page, mode): destination quarter j (rows 8j .. 8j + 7) takes it. mode 0 = zeros, 1..4 = raw quarter mode - 1 of the source page, 5..8 =
canonical quarter mode - 5. The kernel's runtime args carry the source and destination ADDRESS tables once and five words per task (index, mode and page packed), so the longest list a
core can carry is (256 - 1 - NSRC - NDST) // 5 tasks (36 for the 8-source, 64-destination unstack: 3,960 tasks on 110 cores against its 3,600).

Stdlib only at import, py 3.7.
"""

from pathlib import Path

from gdn_rows_dma_tp import Unsupported, problem, tile_columns        # one exception type and one placement rule for both movers
import tp_shapes

KERNEL = 'gdn_rows_dma8_tp.cpp'
USER_ROWS = 8                   # an octo user's segment
TILE_ROWS = 32
QUARTERS = TILE_ROWS // USER_ROWS
USERS = 8                       # the octo block
TASK_WORDS = 5
LANES = 8
MAX_ARGUMENT_WORDS = 256        # a core's runtime args
SCRATCH_BYTES = LANES * 2048
QUARTER_STRIDE = 256            # bytes between the quarters of one face
CHUNK_BYTES = 256

Z = (0, 0, 0)                   # a zero source
RAW, CANON = 1, 5               # mode = RAW / CANON + the source quarter
BLOCK_CANON_USERS = (1, 2, 3, 5, 6, 7)


def mode(quarter, canonical):
    if quarter not in range(QUARTERS):
        raise ValueError('A tile has four quarters')
    return (CANON if canonical else RAW) + quarter


def served_canon(user):
    """Whether the served split canonicalises this user's piece: a slice that starts inside a tile (users 1, 2, 3, 5, 6, 7) goes through untilize / slice / tilize, one that
    starts on a tile boundary (users 0 and 4) is a raw tile slice."""
    return (user * USER_ROWS) % TILE_ROWS != 0


def position(user):
    """(tile row, quarter) of a user's 8 rows in the 64-row block."""
    start = user * USER_ROWS
    return start // TILE_ROWS, (start % TILE_ROWS) // USER_ROWS


def quarter_offsets(quarter):
    """The two byte offsets (left face, right face) of quarter `quarter` inside a 2048-byte tile: the kernel's quarter_offset."""
    left = (quarter >> 1) * 1024 + (quarter & 1) * QUARTER_STRIDE
    return left, left + 512


def tile_rows_of(users):
    return (users * USER_ROWS + TILE_ROWS - 1) // TILE_ROWS


def split_pieces(users, width):
    """Block -> one (1, 8, width) piece per user. Source 0 is the block, destination u is user u's piece. The piece's rows 0-7 are the user's rows (canonical for the users that
    start inside a tile), rows 8-31 zero."""
    columns = tile_columns(width)
    tasks = []
    for user in range(users):
        row, quarter = position(user)
        for column in range(columns):
            tasks.append((user, column, ((0, row * columns + column, mode(quarter, served_canon(user))), Z, Z, Z)))
    return tasks


def canon_block(users, width):
    """Block -> the same block with the users that start inside a tile canonical. Source 0 is the block, destination 0 the copy. Each tile takes its four quarters from the same
    source tile."""
    columns = tile_columns(width)
    tasks = []
    for row in range(tile_rows_of(users)):
        for column in range(columns):
            page = row * columns + column
            quarters = []
            for j in range(QUARTERS):
                user = row * QUARTERS + j
                quarters.append((0, page, mode(j, served_canon(user))) if user < users else Z)
            tasks.append((0, page, tuple(quarters)))
    return tasks


def merge_outputs(users, width):
    """The users' (1, 8, width) outputs -> the (1, 8 * users, width) block, every user canonical (the served concat untilizes, joins and tilizes them all). Source u is user u's
    output, destination 0 the block."""
    return stack_users(users, width, canonical=True)


def stack_users(users, width, canonical=False):
    """The users' (1, 8, width) tensors -> one (1, 8 * users, width) block, raw (windows) or canonical. Source u is user u's tensor, destination 0 the block."""
    columns = tile_columns(width)
    tasks = []
    for row in range(tile_rows_of(users)):
        for column in range(columns):
            quarters = []
            for j in range(QUARTERS):
                user = row * QUARTERS + j
                quarters.append((user, column, mode(0, canonical)) if user < users else Z)
            tasks.append((0, row * columns + column, tuple(quarters)))
    return tasks


def unstack_users(users, block_columns, first_column, columns):
    """`columns` tile columns of a (1, 8 * users, .) block starting at tile column `first_column` -> one (1, 8, .) tensor per user (rows 0-7 the user's rows, rows 8-31 zero).
    Source 0 is the block, destination u the user's tensor."""
    tasks = []
    for user in range(users):
        row, quarter = position(user)
        for column in range(columns):
            tasks.append((user, column, ((0, row * block_columns + first_column + column, mode(quarter, False)), Z, Z, Z)))
    return tasks


def capacity_for(sources, destinations):
    """The most tasks one core's runtime args can carry beside the two address tables."""
    return (MAX_ARGUMENT_WORDS - 1 - sources - destinations) // TASK_WORDS


def distribute(tasks, cores, sources=1, destinations=1):
    """Round-robin the flat task list over at most `cores` cores; Unsupported when a core's list does not fit its argument words."""
    count = min(len(tasks), cores)
    if count == 0:
        raise Unsupported('At least one move required')
    per_core = [tasks[index::count] for index in range(count)]
    if max(len(core) for core in per_core) > capacity_for(sources, destinations):
        raise Unsupported('%d tasks do not fit %d cores at %d words each beside %d + %d addresses' % (len(tasks), cores, TASK_WORDS, sources, destinations))
    return per_core


def capacity(per_core):
    """The most tasks any core carries. It is a compile-time arg of the launch (so it is part of the program cache key) and every core's runtime-arg list is padded to
    1 + NSRC + NDST + TASK_WORDS * capacity words: generic_op caches on the kernel, defines, compile-time args and cores but NOT on runtime-arg lengths, and a cache hit writes
    this launch's lists into the cached slots (hardware window N1 of the half-tile mover: a 1800-task unstack hit a 640-task window stack's 49-word slots)."""
    return max(len(core) for core in per_core)


def pack_head(destination, page):
    if not (0 <= destination < 256 and 0 <= page < 65536):
        raise ValueError('destination index and page out of range')
    return (destination << 16) | page


def pack_source(source):
    index, page, source_mode = source
    if source_mode == 0:
        return 0
    if not (0 <= index < 256 and 0 <= page < 65536 and 1 <= source_mode <= 8):
        raise ValueError('source index, page or mode out of range')
    return (index << 24) | (source_mode << 16) | page


def runtime_arguments(per_core, sources, destinations):
    """Each core's [tasks, source addresses, destination addresses, five words per task], zero padded to 1 + NSRC + NDST + TASK_WORDS * capacity, from per-launch address lists
    (`sources`, `destinations`: one buffer address per tensor index, for one chip)."""
    size = 1 + len(sources) + len(destinations) + TASK_WORDS * capacity(per_core)
    lists = []
    for core in per_core:
        words = [len(core)] + [int(address) for address in sources] + [int(address) for address in destinations]
        for destination, page, quarters in core:
            words.append(pack_head(destination, page))
            words.extend(pack_source(source) for source in quarters)
        lists.append(words + [0] * (size - len(words)))
    return lists


def launch(mesh, sources, destinations, tasks, *, canon_denorm=True):
    """One generic_op that runs `tasks` over per-chip programs. Every source shares one accessor layout and every destination another; anything else is Unsupported."""
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
    per_core = distribute(tasks, count, len(sources), len(destinations))
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
            compile_time_args=[*source_layouts[0], *destination_layouts[0], len(sources), len(destinations), capacity(per_core)],
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
