"""The packed block's ordered K/V write with one semaphore chain PER USER (verify-trace T2 cut #2).

WHAT IT REPLACES. packed_cache_writer.SegmentedOrderedCacheWriter writes the 64-row block's
K/V as two launches of the audited ordered kernel per cache, one per 32-row tile, each after a
whole-tile slice of the prepared K/V, each ONE semaphore chain over its 32 rows - two users -
at about 3.2 us a step (101.99 us a launch, 16 layers x 2 caches x 2 tiles per replay). Two
users never share a cache tile row, so the cross-user half of each chain buys nothing.

WHAT RUNS INSTEAD. One launch per cache over all 64 rows, 64 cores (an 8x8 grid, core i at
(i % 8, i // 8), one row each), one chain per packed user's segment: the chain waits on no
predecessor at a segment's first row and signals no successor at its last. The kernels are
ordered_cache.load_kernels' hash-pinned sources, imported and never copied (ordered_cache.py is
frozen writer evidence); the compile arguments are the served lists (ordered_cache.update:128,
131, 139) with rows = 64, so only reader arg 7 (index_stick_size_B, rows * 4 = 256 bytes of
positions) differs. The block-level positions word (64,) and page-table rows (64, W) are the
fixture's own, staged every round with the tile metadata from the same host tensors behind one
fence (packed_verifier.stage_packed); input page (first + i) * 8 + t is exactly what the served
slice would have put at page i * 8 + t, so no slice is needed.

CB16 stays at the served 256 pages (`min(rows, 32) * 8`, not `rows * 8`): the compute kernel
pushes 8 pages per head and the writer pops them before the next, so no more than 16 are ever
outstanding and the capacity only changes back-pressure, never a value. The per-core CB
footprint is therefore the audited launch's own (390,160 bytes at page width 2052).

WHY THE BYTES ARE THE SERVED ONES. Each 32-row cache tile row X = (physical page, head, tile
row) ends as RMW_{r_k}(...RMW_{r_1}(X0)) over the rows r_1 < ... < r_k that target it, in row
order: served, because one chain runs every row in index order; chained, because all rows
targeting X are in ONE user's chain, in the same relative order - PROVIDED no two users target
the same (page, tile row). That is verify_trace_t2.kv_conflict, checked on the host before the
round is drafted (serving_packed_step.proposal_rows: a conflict drafts it at the engines' own
widths and the exact sequential step serves it), again at the step (ineligible: a conflict
first seen there refuses the round - nothing is written) and, fail-closed, before any staging
copy (packed_verifier.stage_packed). The untilize / tilize
round trips inside each RMW are the same kernel code on the same bytes in the same order in
both paths. QWEN_FAST_VERIFY_T2_KV_ROWS=32 keeps the served tiles and slices and chains each
tile's rows per user: two 32-row launches per cache, no image rebuild.

The descriptors are rebuilt on every call, exactly as ordered_cache.update does, so the
program-cache behaviour is the served one.
"""

KV_HEADS, KV_WIDTH = 32, 256
TILE_ROWS = 32
LAUNCH_ROWS = (64, 32)
GRID_WIDTH = 8
# ordered_cache.update's CB16 at its 32-row audited launch (rows * 8 pages of 1088 bytes).
CB16_SERVED = 256
CACHE_PAGE, INPUT_PAGE = 1088, 2048
NEGATIVE_CONTROLS = ('nochain', 'index', 'conflict')
COMPUTE_ARGS = (0, 1, 24, 25, 26, 16, 8, 2)


class Unsupported(ValueError):
    """The chained writer cannot serve this block (grid, width, spans): the served segmented
    writer is built instead, before any warm forward or capture."""


# ---------------------------------------------------------------------------------------------
# Pure planners (CPU-tested).
# ---------------------------------------------------------------------------------------------

def validate_spans(rows, spans):
    """Contiguous, non-empty (first, last) spans covering [0, rows) in order."""
    spans = tuple((int(first), int(last)) for first, last in spans)
    cursor = 0
    for first, last in spans:
        if first != cursor or last <= first:
            raise Unsupported('Chains need contiguous non-empty segments covering the block from row 0')
        cursor = last
    if cursor != rows or not spans:
        raise Unsupported('Chains must cover exactly the %d launch rows' % rows)
    return spans


def chain_args(rows, spans):
    """Per row i: (wait, signal, next). wait(i) = i is not a span start; signal(i) = i + 1 is not
    its span's end; next = i + 1, or i itself for a span's tail (whose signal is 0)."""
    spans = validate_spans(rows, spans)
    starts = {first for first, last in spans}
    tails = {last - 1 for first, last in spans}
    return [(int(row not in starts), int(row not in tails), row if row in tails else row + 1) for row in range(rows)]


def tile_spans(spans, first, last):
    """The spans inside rows [first, last), shifted to the tile's own row numbers. A span that
    crosses the tile boundary is split: the two launches run in command-queue order, so its
    rows still land in order."""
    out = []
    for start, stop in spans:
        start, stop = max(start, first), min(stop, last)
        if start < stop:
            out.append((start - first, stop - first))
    return tuple(out)


def apply_negative(args, spans, negative):
    """Harness-only runtime-arg perturbations (no kernel defines). 'nochain': every wait and
    signal 0, so the rows of one tile race; 'index': the WRITER's my_batch_idx of rows i and i+1
    of the first span swapped, so row i's input lands at row i+1's position (deterministic);
    'conflict': no change here - the harness gives two users one tile row."""
    rows = len(args)
    index = list(range(rows))
    args = list(args)
    if negative is None or negative == 'conflict':
        return args, index
    if negative == 'nochain':
        return [(0, 0, row) for row in range(rows)], index
    if negative == 'index':
        first, last = spans[0]
        if last - first < 2:
            raise ValueError('the index control needs a chain of two rows')
        index[first], index[first + 1] = index[first + 1], index[first]
        return args, index
    raise ValueError('unknown negative control %r' % (negative,))


def reader_compile_args(rows, width):
    """ordered_cache.update's reader list (line 128) at `rows`: only arg 7 (rows * 4) moves."""
    return [0, 1, 1, 2, 0, 8, 0, rows * 4, 1, 2, 64, 2, width, 0, width * 4, 3, 2, 0, 0]


def writer_compile_args(width):
    """ordered_cache.update's writer list (line 131), unchanged."""
    return [16, 24, 25, 26, 1, 2, 0, 8, 512, 1, 2, 64, 2, width, 3, 2, 0, 0]


def cb_table(width, cb16_pages=CB16_SERVED):
    """(indices, pages, dtype name, page bytes, tiled) as ordered_cache.update builds them, CB16
    at `cb16_pages`."""
    return [((0,), 16, 'bfloat8_b', CACHE_PAGE, True), ((1,), 8, 'bfloat16', INPUT_PAGE, True),
            ((24, 25), 16, 'bfloat16', INPUT_PAGE, True), ((26,), 16, 'bfloat16', INPUT_PAGE, True),
            ((16,), cb16_pages, 'bfloat8_b', CACHE_PAGE, True), ((2,), 1, 'int32', 4096, False),
            ((3,), 1, 'int32', width * 4, False)]


def cb_bytes(width, cb16_pages=CB16_SERVED):
    return sum(pages * size for indices, pages, dtype, size, tiled in cb_table(width, cb16_pages))


def core_points(rows):
    return [(row % GRID_WIDTH, row // GRID_WIDTH) for row in range(rows)]


def grid_problem(grid_x, grid_y, rows):
    if grid_x < GRID_WIDTH or grid_y < rows // GRID_WIDTH:
        return 'a %d-row chained launch needs an %dx%d worker grid; this one is %dx%d' % (
            rows, GRID_WIDTH, rows // GRID_WIDTH, grid_x, grid_y)
    return None


def validate_chained(cache, packed, positions, pages, rows):
    """ordered_cache.validate_shapes, widened from 32 to the chained launch rows (64 or 32)."""
    from ordered_cache import page_width_admitted

    if rows not in LAUNCH_ROWS or tuple(packed) != (1, rows, KV_HEADS, KV_WIDTH):
        raise ValueError('Chained K/V writes take (1, 64 or 32, 32, 256) prepared tiles; got %r' % (tuple(packed),))
    if len(cache) != 4 or cache[0] < 1 or tuple(cache[1:]) != (2, 64, 256):
        raise ValueError('Expected two-head 64-row BF8 paged cache')
    if (tuple(positions) != (rows,) or len(pages) != 2 or pages[0] != rows
            or not page_width_admitted(pages[1]) or pages[1] > cache[0]):
        raise ValueError('Paired position vector and page-table rows required')
    return rows


def mesh_coordinates(shape):
    rows, cols = shape
    return [(index // cols, index % cols) for index in range(rows * cols)]


# ---------------------------------------------------------------------------------------------
# The launch.
# ---------------------------------------------------------------------------------------------

def update_chained(mesh, cache, packed, positions, pages, kernels, spans, *, cb16_pages=CB16_SERVED, negative=None,
                   operations=None):
    """One chained ordered write of `packed`'s rows into `cache`, row i on core (i % 8, i // 8),
    chained within each span. `kernels` is ordered_cache.load_kernels' output."""
    if operations is None:
        import ttnn as operations
    from verify_trace_t1 import rectangle_set

    rows = packed.shape[1] if len(packed.shape) == 4 else 0
    validate_chained(tuple(cache.shape), tuple(packed.shape), tuple(positions.shape), tuple(pages.shape), rows)
    tensors = [cache, packed, positions, pages]
    if any(value.memory_config() != operations.DRAM_MEMORY_CONFIG for value in tensors):
        raise ValueError('Interleaved DRAM buffers required')
    if cache.dtype != operations.bfloat8_b or packed.dtype != operations.bfloat16 or any(
            value.layout != operations.TILE_LAYOUT for value in tensors[:2]):
        raise ValueError('Native BF8 cache and BF16 input tiles required')
    if any(value.dtype != operations.int32 or value.layout != operations.ROW_MAJOR_LAYOUT for value in tensors[2:]):
        raise ValueError('Int32 row-major metadata required')
    if type(cb16_pages) is not int or cb16_pages < 16:
        raise ValueError('CB16 needs at least the 16 pages the kernels can hold')
    spans = validate_spans(rows, spans)
    try:
        shape = tuple(mesh.shape)
    except (AttributeError, TypeError):
        raise Unsupported('no mesh shape') from None
    if shape not in ((1, 1), (1, 2)):
        raise Unsupported('mesh %s is not [1, 1] or [1, 2]' % (shape,))
    coordinates = mesh_coordinates(shape)
    shards = [operations.get_device_tensors(value) for value in tensors]
    if any(len(parts) != len(coordinates) for parts in shards):
        raise ValueError('Every tensor must have one shard per mesh device')
    grid = mesh.compute_with_storage_grid_size()
    problem = grid_problem(grid.x, grid.y, rows)
    if problem is not None:
        raise Unsupported(problem)
    points = core_points(rows)
    cores = rectangle_set(operations, points)
    coordinates_of = [operations.CoreCoord(x, y) for x, y in points]
    width = pages.padded_shape[-1]
    buffers = []
    dtypes = dict(bfloat8_b=operations.bfloat8_b, bfloat16=operations.bfloat16, int32=operations.int32)
    for indices, count, dtype, size, tiled in cb_table(width, cb16_pages):
        formats = [operations.CBFormatDescriptor(buffer_index=index, data_format=dtypes[dtype], page_size=size,
                   **(dict(tile=operations.TileDescriptor(operations.Tile([32, 32]))) if tiled else {}))
                   for index in indices]
        buffers.append(operations.CBDescriptor(total_size=count * size, core_ranges=cores, format_descriptors=formats))
    semaphore = operations.SemaphoreDescriptor(id=0, core_ranges=cores, initial_value=0)
    chains, writer_index = apply_negative(chain_args(rows, spans), spans, negative)
    program = operations.MeshProgramDescriptor()
    for chip, (row_index, col_index) in enumerate(coordinates):
        local = [parts[chip] for parts in shards]
        addresses = [value.buffer_address() for value in local]
        if len(set(addresses)) != 4:
            raise ValueError('Cache, packed input and metadata must not alias')
        reader_args = reader_compile_args(rows, width)
        for index in (0, 2, 3, 1):
            reader_args.extend(operations.TensorAccessorArgs(local[index]).get_compile_time_args())
        writer_args = writer_compile_args(width)
        writer_args.extend(operations.TensorAccessorArgs(local[0]).get_compile_time_args())
        device = local[0].device()
        descriptors = []
        for role, args, config in (
            ('reader', reader_args, operations.DataMovementConfigDescriptor(
                processor=operations.DataMovementProcessor.RISCV_1, noc=operations.NOC.RISCV_1_default)),
            ('writer', writer_args, operations.DataMovementConfigDescriptor(
                processor=operations.DataMovementProcessor.RISCV_0, noc=operations.NOC.RISCV_0_default)),
            ('compute', list(COMPUTE_ARGS), operations.ComputeConfigDescriptor(fp32_dest_acc_en=False)),
        ):
            runtime = operations.RuntimeArgs()
            for row, core in enumerate(coordinates_of):
                wait, signal, successor = chains[row]
                if role == 'reader':
                    runtime[core.x][core.y] = [addresses[0], 0, addresses[2], row, addresses[3], wait, addresses[1]]
                elif role == 'writer':
                    target = device.worker_core_from_logical_core(coordinates_of[successor])
                    runtime[core.x][core.y] = [addresses[0], 0, 0, writer_index[row], signal, target.x, target.y]
                else:
                    runtime[core.x][core.y] = []
            descriptor = operations.KernelDescriptor(kernel_source=kernels[role],
                source_type=operations.KernelDescriptor.SourceType.SOURCE_CODE, core_ranges=cores,
                compile_time_args=args, config=config)
            descriptor.runtime_args = runtime
            descriptors.append(descriptor)
        coordinate = operations.MeshCoordinate(row_index, col_index)
        program[operations.MeshCoordinateRange(coordinate, coordinate)] = operations.ProgramDescriptor(
            kernels=descriptors, cbs=buffers, semaphores=[semaphore])
    operations.generic_op(tensors, program)


# ---------------------------------------------------------------------------------------------
# The block's writer.
# ---------------------------------------------------------------------------------------------

class ChainedOrderedCacheWriter:
    """attention_batch.OrderedCacheWriter's call contract over the packed block: one chained
    launch per cache (launch_rows 64), or per 32-row tile over the served tile metadata and
    slices (launch_rows 32). Counts `calls` as the served writers do (model_batch checks two
    per layer per forward) and notes 'kv_chains' once per call (32 per captured forward)."""

    def __init__(self, mesh, operations, kernels, *, positions, pages, spans, tiles, launch_rows=64):
        from ordered_cache import page_width_admitted
        from packed_cache_writer import validate_tiles

        if launch_rows not in LAUNCH_ROWS:
            raise Unsupported('launch rows %r is not 64 or 32' % (launch_rows,))
        rows = positions.shape[0] if len(positions.shape) == 1 else 0
        if (rows != 64 or positions.dtype != operations.int32 or positions.layout != operations.ROW_MAJOR_LAYOUT
                or len(pages.shape) != 2 or pages.shape[0] != rows or pages.dtype != operations.int32
                or pages.layout != operations.ROW_MAJOR_LAYOUT):
            raise Unsupported('the chained writer serves the 64-row block staged as row-major int32 metadata')
        if not page_width_admitted(pages.shape[1]):
            raise Unsupported('page width %r is not admitted' % (pages.shape[1],))
        self.spans = validate_spans(rows, spans)
        self.tiles, block_rows, width = validate_tiles(operations, tiles)
        if block_rows != rows or width != pages.shape[1]:
            raise Unsupported('the tile metadata must cover the same block at the same page width')
        grid = mesh.compute_with_storage_grid_size()
        problem = grid_problem(grid.x, grid.y, launch_rows)
        if problem is not None:
            raise Unsupported(problem)
        self.mesh, self.operations, self.kernels = mesh, operations, kernels
        self.positions, self.pages, self.rows, self.page_width = positions, pages, rows, pages.shape[1]
        self.launch_rows = launch_rows
        self.calls = 0

    def __call__(self, cache, packed, *, update_idxs_tensor, page_table):
        from verify_trace_t2 import note

        operations = self.operations
        shape = tuple(packed.shape)
        if shape != (1, self.rows, KV_HEADS, KV_WIDTH):
            raise ValueError('Unexpected native prepared KV shape for the %d-row block: %r' % (self.rows, shape))
        # The model passes the block's whole positions word and page table - the tensors this
        # writer was built over - so only the geometry is checked (packed_cache_writer:84-89).
        if tuple(update_idxs_tensor.shape) != (self.rows,) or tuple(page_table.shape) != (self.rows, self.page_width):
            raise ValueError('The block positions and page table must cover every row')
        converted = packed.memory_config() != operations.DRAM_MEMORY_CONFIG
        interleaved = operations.to_memory_config(packed, operations.DRAM_MEMORY_CONFIG) if converted else packed
        try:
            if self.launch_rows == self.rows:
                update_chained(self.mesh, cache, interleaved, self.positions, self.pages, self.kernels, self.spans,
                               operations=operations)
            else:
                for entry in self.tiles:
                    first, last = entry.rows
                    piece = operations.slice(interleaved, (0, first, 0, 0), (1, last, KV_HEADS, KV_WIDTH),
                                             memory_config=operations.DRAM_MEMORY_CONFIG)
                    try:
                        update_chained(self.mesh, cache, piece, entry.positions, entry.pages, self.kernels,
                                       tile_spans(self.spans, first, last), operations=operations)
                    finally:
                        operations.deallocate(piece)
        finally:
            if converted:
                operations.deallocate(interleaved)
        self.calls += 1
        note('kv_chains')

