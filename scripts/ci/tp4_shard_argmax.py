"""The per-shard argmax of the TP4 verify trace as small launches (QWEN_FAST_TP4_SHARD_ARGMAX, default off; S1).

WHY. verify_trace_t1.sample_shards takes each chip's (rows, 62,080) bf16 TILE logits shard through an untilize (46 us), ttnn.argmax
(940 us: element-serial on the data-movement cores, about 26 ns an element) and, for the shard maxima, either a 2-core ttnn.max
(787 us) or the V4a gather (QWEN_FAST_TP4_SHARD_VALUES): about 1.0 ms of the 56-59 ms verify trace plus the gather, 1.8 ms in the
production recipe's profile.

WHAT. tp4_shard_argmax_scan.cpp scans the TILE logits in place on every worker of the device's compute grid (two data-movement RISC-Vs
each: 220 tasks on the 11 x 10 grid, 260 on 13 x 10), each task a run of tile columns of one tile row (rows 33..64: one tile row per
RISC; up to 32 rows: the two RISCs split the column run in halves), and writes per row one word (column << 16) | bits to a partials
page. A fold then takes, per row, the partials of the row in ascending column order with a strict greater-than: either
tp4_shard_argmax_fold.cpp (one core, the baseline) or, behind QWEN_FAST_TP4_SHARD_ARGMAX_FOLD2=1, tp4_shard_argmax_fold2.cpp (a
two-level tree: FOLD2_GROUPS cores each fold a contiguous block of the tasks, then one core folds their winners; the same answer,
because a strict-greater scan over a sequence equals the scan over its contiguous blocks' winners in block order). The answer is the
first index holding the maximum, the rule of torch.argmax, of the pinned sampler and of combine_shards: the combined global ids are
identical to today's for every row without a NaN, and a NaN row follows torch (the first NaN wins). The values are the maximum
element's own bits; the only possible bit difference from ttnn.max is -0 against +0 when a row's maximum is zero, which
combine_shards compares equal.

WHY THE TREE (TS2). The only timed ABAB so far (TS1-TS4, four 8-user runs) shows S1 saving 0.70-0.78 ms of trace_ms at live=4 in every
pair, against the 1.03 ms of Untilize + ArgMax + Gather it replaces (measured): scan + fold cost 0.29 ms, not the 80-150 us designed. The split
of those 0.29 ms between scan and fold is NOT measured (card-M harness: optimisation/ttnn-op/shard_argmax, arms scan / scan + fold / scan + tree);
a cycle count puts the one-core fold (7,040 record steps at 64 rows x 110 tasks, an estimated 25 cycles each) at about 0.13 ms, the serial
half, and the scan (about 9,000 words a RISC-V at an estimated 12 cycles) at about 0.1 ms. TS2's lower aggregate tok/s was the host (input_ms
+2.3 ms, readback +0.9 ms in its first two tests, while its trace_ms was lower than the control's), not S1: TS4 repeats it at parity.

CONTRACT AND READBACK LAYOUT (the one layout S1 and the W3 mesh-read work (tp4_mesh_read) share). Per chip, three ROW_MAJOR DRAM tensors,
each ONE page of 64 words, shaped (1, 1, 1, 64), with the same shape on every chip so that a mesh-level read with a dim-0 concat composer
returns chip c's part as rows [c, c + 1):
    ids     uint32, 256 B: row r = the shard-local column of row r's maximum (< 62,080);
    values  bf16,   128 B: row r = the maximum element's own bits;
    words   uint32, 256 B: row r = (column << 16) | bits, ids and values joined, so ONE mesh read can carry both
            (ids = word >> 16, bits = word & 0xffff; `decode_words` is the host side; read a uint32 tensor that torch returns as int32
            through `& 0xffffffff`: columns above 32,767 set the sign bit).
Rows at and past `rows` are zero in all three. sample() returns (ids, values) like sample_shards, and `words_for(values)` returns the third
tensor (the packed block frees it with the other two through release_audit). Every consumer reads to_torch(part).reshape(-1)[:rows]
(PackedVerifierEngine.shard_predictions, verifier_engine_tp.shard_predictions), so the shape is invisible to them; the MR reader may read
`words` once instead of `ids` and `values` twice when its probe (MR4) says the joined word wins.

GRID. The worker rectangle is read from the device (`mesh.compute_with_storage_grid_size()`, 11 x 10 before the 2026-10-10 firmware unlock,
13 x 10 after), at reserve(): 110 or 130 workers, 220 or 260 tasks, partials buffer (1, 1, 2 x workers, 32). A mesh that cannot say (a fake)
gets 11 x 10. Everything else in this module derives from that grid, nothing is a literal 110.

PERSISTENT SCRATCH. The partials buffer (and, with FOLD2, the group buffer) is allocated once per mesh by reserve(), which the packed block calls at its
warm, before any trace is captured, and freed by release_reserved() when the last block closes: a buffer allocated inside the verify capture would
change that trace's hole layout (the traced-publish hang mechanism). One buffer serves every block: replays are serial on one queue and the
fold consumes the partials inside the same trace that wrote them. A call on a mesh with no reservation (the request engine, a
caller that never reserved) is a logged fall-back, never an allocation inside a capture.

FALLBACK. Any call it cannot take (another shard shape, dtype, layout, placement, more than 64 rows, a tile-column plan whose runs
exceed the scan kernel's scratch) is logged once and answered by today's untilize / argmax / max / gather path.

AUDIT (QWEN_FAST_TP4_SHARD_ARGMAX_AUDIT=1, needs the lever). Today's path runs beside the kernel in the same capture on the same
logits, its (ids, values) are held, and every round PackedVerifierEngine.shard_predictions calls audit_round(), which compares every
row of every chip: ids exactly, values as numbers (-0 == +0, NaN with NaN), and the words tensor against the kernel's own ids and value bits
(exactly). A difference logs SARG_MISMATCH and raises.

Stdlib only at import, py 3.7.
"""

import os
import sys
from pathlib import Path

import tp4_sampdraft
import tp_shapes

SCAN_KERNEL = 'tp4_shard_argmax_scan.cpp'
FOLD_KERNEL = 'tp4_shard_argmax_fold.cpp'
FOLD2_KERNEL = 'tp4_shard_argmax_fold2.cpp'
FOLD2_FLAG = 'QWEN_FAST_TP4_SHARD_ARGMAX_FOLD2'
WORKERS = 110                      # the 11 x 10 grid's workers: the default when the device cannot say (a fake mesh)
GRID = (11, 10)                    # worker w sits at (w // gy, w % gy), the quad conv's E1b order
TILE = 32
MAX_ROWS = 64
MAX_TILES = 18                     # the scan kernel's scratch: tiles of one task
PAGE_WORDS = 32                    # a partials page: one word per row of a tile row
PAGE_BYTES = PAGE_WORDS * 4
TASKS = 2 * WORKERS                # two RISC-Vs per worker, whatever the row count (on the default grid)
OUTPUT_WORDS = 64                  # ids / values / words page: 64 words / 64 bf16 / 64 words
SCAN_CB_PAGES = MAX_TILES + 1      # the tiles plus the result page
TILE_BYTES = 2048
IDS_BYTES, VALUES_BYTES, WORDS_BYTES = 256, 128, 256
FOLD_CB_PAGES = 16                 # the one-core fold's scratch on the default grid: 220 partials pages (28,160 B) plus the three outputs
FOLD2_GROUPS = 8                   # level-1 cores of the tree fold
GROUP_PAGE_WORDS = 64              # a group page: one record per row of the block
GROUP_PAGE_BYTES = GROUP_PAGE_WORDS * 4

# Held (today's ids, today's values) of the audit, keyed by id(values), and the audit's round counter. PRODUCED holds id(values) of every
# output a kernel launch made (not today's path), so the V4a value audit (packed_verifier.audit_shard_values), which compares gathered
# maxima with a ttnn.max taken beside them in served_shards, knows these are not its outputs and leaves them to this module's audit.
REFERENCES = {}
PRODUCED = {}                      # id(values) -> the values tensor itself, so a freed tensor's id cannot be reused while it is listed
WORDS = {}                         # id(values) -> the joined (column << 16 | bits) tensor of the same launch (freed by release_audit)
_STATE = {'rounds': 0}
_RESERVED = {}                     # 'mesh' -> [partials tensor, holders, mesh, group buffer or None, (grid x, y), fold2]: one mesh per process
                                   # (logits.device() need not be the same Python object as the block's mesh, so identity is not the key)


def fold2_enabled(environ=None):
    """QWEN_FAST_TP4_SHARD_ARGMAX_FOLD2: the tree fold instead of the one-core fold. Strict (unset or 0 off, 1 on, anything else raises);
    raises at the pair and without S1 itself (the flag would select nothing)."""
    source = os.environ if environ is None else environ
    value = source.get(FOLD2_FLAG)
    if value is None or value == '0':
        return False
    if value != '1':
        raise ValueError('%s must be 0 or 1, got %r' % (FOLD2_FLAG, value))
    if tp_shapes.chip_count(source) == tp_shapes.PAIR:
        raise ValueError('%s is a TP4 lever: it needs QWEN_FAST_TP=4, this process serves the pair' % FOLD2_FLAG)
    if source.get(tp4_sampdraft.SHARD_ARGMAX) != '1':
        raise ValueError('%s=1 needs %s=1: it only chooses how S1 folds' % (FOLD2_FLAG, tp4_sampdraft.SHARD_ARGMAX))
    return True


def grid_of(mesh):
    """The (x, y) compute grid of the device behind `mesh`: 11 x 10 before the 2026-10-10 firmware unlock, 13 x 10 after. A mesh that
    cannot say (a fake) gets the default 11 x 10, which every grid has."""
    try:
        grid = mesh.compute_with_storage_grid_size()
        x, y = int(grid.x), int(grid.y)
    except Exception:  # noqa: BLE001 - the default is a rectangle every grid contains
        return GRID
    if x < 1 or y < 1:
        return GRID
    return (x, y)


def group_pages_bound(task_pages, groups):
    """The most partials pages one level-1 core reads, over both row regimes (one tile row of `task_pages` tasks, or two of half that):
    the kernel's scratch and its FOLD2_GROUP_PAGES define."""
    half = task_pages // 2
    return max(-(-task_pages // groups), 2 * -(-half // groups))


def group_plan(per_tile_row, groups=FOLD2_GROUPS):
    """[(group, first task, one past the last task)] of one tile row's tasks: contiguous, ascending, every group non-empty. Raises ValueError
    when there are fewer tasks than groups."""
    if type(groups) is not int or groups < 1 or per_tile_row < groups:
        raise ValueError('%r tasks per tile row cannot feed %r fold groups' % (per_tile_row, groups))
    return [(group, group * per_tile_row // groups, (group + 1) * per_tile_row // groups) for group in range(groups)]


def reserve(operations, mesh):
    """Allocate the process's scratch on `mesh` (or add a holder to the one that exists): the partials buffer (1, 1, 2 x workers, 32)
    and, under QWEN_FAST_TP4_SHARD_ARGMAX_FOLD2, the group buffer (1, 1, FOLD2_GROUPS, 64). Call before any trace is captured; pair it
    with release_reserved. Returns the partials tensor."""
    entry = _RESERVED.get('mesh')
    if entry is None:
        grid = grid_of(mesh)
        fold2 = fold2_enabled()
        tasks = 2 * grid[0] * grid[1]
        tensor = operations.empty((1, 1, tasks, PAGE_WORDS), dtype=operations.uint32, layout=operations.ROW_MAJOR_LAYOUT,
                                  device=mesh, memory_config=operations.DRAM_MEMORY_CONFIG)
        groups = None
        if fold2:
            try:
                groups = operations.empty((1, 1, FOLD2_GROUPS, GROUP_PAGE_WORDS), dtype=operations.uint32,
                                          layout=operations.ROW_MAJOR_LAYOUT, device=mesh, memory_config=operations.DRAM_MEMORY_CONFIG)
            except BaseException:
                operations.deallocate(tensor)
                raise
        entry = _RESERVED['mesh'] = [tensor, 0, mesh, groups, grid, fold2]
    entry[1] += 1
    return entry[0]


def release_reserved(operations, mesh):
    """Drop one holder of this mesh's scratch; the last holder frees it. A mesh with no reservation is a no-op. Call only after
    every trace that reads or writes the buffers is released."""
    entry = _RESERVED.get('mesh')
    if entry is None:
        return
    entry[1] -= 1
    if entry[1] <= 0:
        del _RESERVED['mesh']
        for tensor in (entry[0], entry[3]):
            if tensor is None:
                continue
            try:
                operations.deallocate(tensor)
            except BaseException:
                pass


def column_runs(tile_columns, workers=WORKERS):
    """[(first, last)] per worker: contiguous runs of tile columns in ascending order, the first `extra` workers one column longer
    (1,940 columns over 110 workers: 70 runs of 18 and 40 of 17; over 130: 120 of 15 and 10 of 14)."""
    if tile_columns < workers:
        raise ValueError('%d tile columns for %d workers' % (tile_columns, workers))
    base, extra = divmod(tile_columns, workers)
    runs, start = [], 0
    for worker in range(workers):
        stop = start + base + (1 if worker < extra else 0)
        runs.append((start, stop))
        start = stop
    return runs


def plan(rows, tile_columns, workers=WORKERS):
    """The task list for a (rows, 32 * tile_columns) shard on `workers` workers: (task page, worker, role, tile row, first column, one past
    the last column, live rows), in the order the fold reads them, plus (tasks per tile row, tile rows). Raises ValueError when the
    shard does not fit the kernels."""
    if type(rows) is not int or not 1 <= rows <= MAX_ROWS:
        raise ValueError('rows must be 1 to %d, got %r' % (MAX_ROWS, rows))
    runs = column_runs(tile_columns, workers)
    if max(last - first for first, last in runs) > MAX_TILES:
        raise ValueError('a run of %d tile columns does not fit the scan scratch of %d tiles' % (
            max(last - first for first, last in runs), MAX_TILES))
    tile_rows = (rows + TILE - 1) // TILE
    tasks = []
    if tile_rows == 2:
        for tile_row in range(2):
            live = min(TILE, rows - TILE * tile_row)
            for worker, (first, last) in enumerate(runs):
                tasks.append((tile_row * workers + worker, worker, tile_row, tile_row, first, last, live))
        return tasks, workers, 2
    if min(last - first for first, last in runs) < 2:
        # at up to 32 rows each RISC takes half of a worker's run: an empty half would write a candidate that never was one
        raise ValueError('%d tile columns leave a worker run too short to split between its two RISC-Vs' % tile_columns)
    for worker, (first, last) in enumerate(runs):
        middle = first + (last - first + 1) // 2
        tasks.append((2 * worker, worker, 0, 0, first, middle, rows))
        tasks.append((2 * worker + 1, worker, 1, 0, middle, last, rows))
    return tasks, 2 * workers, 1


def problem(operations, logits, rows, grid=GRID):
    """Why these logits cannot take the kernels (a short reason), or None."""
    shape = tuple(logits.shape)
    width = tp_shapes.vocab_shard()
    if tp_shapes.chip_count() != 4:
        return 'not four cards'
    if len(shape) != 4 or shape[:3] != (1, 1, rows) or shape[3] != width:
        return 'shape %r is not (1, 1, %d, %d)' % (shape, rows, width)
    if type(rows) is not int or not 1 <= rows <= MAX_ROWS:
        return 'rows %r is not 1 to %d' % (rows, MAX_ROWS)
    if width % TILE:
        return 'shard width %d is not a whole number of tiles' % width
    if getattr(logits, 'dtype', None) != operations.bfloat16 or getattr(logits, 'layout', None) != operations.TILE_LAYOUT:
        return 'logits are not bfloat16 TILE'
    try:
        if logits.memory_config() != operations.DRAM_MEMORY_CONFIG:
            return 'logits are not interleaved DRAM'
    except Exception:  # noqa: BLE001 - a diagnostic read
        return 'logits placement unreadable'
    try:
        plan(rows, width // TILE, grid[0] * grid[1])
    except ValueError as failure:
        return str(failure)
    return None


def _cores(operations, count_x, count_y):
    return operations.CoreRangeSet([operations.CoreRange(operations.CoreCoord(0, 0),
                                                         operations.CoreCoord(count_x - 1, count_y - 1))])


def _cb(operations, cores, index, pages, data_format):
    return operations.CBDescriptor(total_size=TILE_BYTES * pages, core_ranges=cores,
        format_descriptors=[operations.CBFormatDescriptor(buffer_index=index, data_format=data_format,
            page_size=TILE_BYTES, tile=operations.TileDescriptor(operations.Tile([32, 32])))])


def _pages(nbytes):
    """CB pages (2 KB) that hold `nbytes`."""
    return -(-nbytes // TILE_BYTES)


def fold_scratch_pages(task_pages):
    """The one-core fold's CB pages: every partials page plus the three output pages, and never fewer than the 16 the 11 x 10 grid always had."""
    return max(FOLD_CB_PAGES, _pages(task_pages * PAGE_BYTES + IDS_BYTES + VALUES_BYTES + WORDS_BYTES))


def launch(operations, logits, rows, partials, groups_buffer, ids, values, words, grid, fold2, scan_only=False):
    """The scan launch and the fold launch(es) over every chip of `logits` (four in a verify, one in the card-M harness): the scan
    reads `logits` and writes `partials`; the one-core fold (or, with `fold2`, the two levels through `groups_buffer`) writes `ids`,
    `values` and `words`. All are whole mesh tensors. Returns the launch count (2, or 3 for the tree; 1 with `scan_only`, which the card-M
    harness uses to time the scan apart from the fold). Raises ValueError for a shard the kernels cannot hold (plan)."""
    logit_shards = operations.get_device_tensors(logits)
    partial_shards = operations.get_device_tensors(partials)
    group_shards = operations.get_device_tensors(groups_buffer) if groups_buffer is not None else None
    id_shards = operations.get_device_tensors(ids)
    value_shard_list = operations.get_device_tensors(values)
    word_shards = operations.get_device_tensors(words)
    chips = len(logit_shards)
    workers_x, workers_y = grid
    workers = workers_x * workers_y
    tile_columns = tp_shapes.vocab_shard() // TILE
    tasks, per_tile_row, tile_rows = plan(rows, tile_columns, workers)
    task_pages = 2 * workers
    if fold2 and group_shards is None:
        raise ValueError('the tree fold needs the group buffer reserve() allocates under %s' % FOLD2_FLAG)

    cores = _cores(operations, workers_x, workers_y)
    scan = operations.MeshProgramDescriptor()
    for chip in range(chips):
        logit, partial = logit_shards[chip], partial_shards[chip]
        arguments = [list(operations.TensorAccessorArgs(logit).get_compile_time_args()),
                     list(operations.TensorAccessorArgs(partial).get_compile_time_args())]
        kernels = []
        for role in (0, 1):
            runtime = operations.RuntimeArgs()
            for task, worker, task_role, tile_row, first, last, live in tasks:
                if task_role == role:
                    runtime[worker // workers_y][worker % workers_y] = [
                        logit.buffer_address(), partial.buffer_address(), tile_row, first, last, live, task, role,
                        tile_columns]
            kernels.append(operations.KernelDescriptor(
                kernel_source=str(Path(__file__).with_name(SCAN_KERNEL)), core_ranges=cores,
                compile_time_args=[*arguments[0], *arguments[1]], runtime_args=runtime,
                config=operations.DataMovementConfigDescriptor(
                    processor=(operations.DataMovementProcessor.RISCV_0, operations.DataMovementProcessor.RISCV_1)[role],
                    noc=(operations.NOC.RISCV_0_default, operations.NOC.RISCV_1_default)[role])))
        coordinate = operations.MeshCoordinate(0, chip)
        scan[operations.MeshCoordinateRange(coordinate, coordinate)] = operations.ProgramDescriptor(
            kernels=kernels, cbs=[_cb(operations, cores, index, SCAN_CB_PAGES, operations.bfloat16) for index in (0, 1)])
    operations.generic_op([logits, partials], scan)
    if scan_only:
        return 1

    one = _cores(operations, 1, 1)
    if not fold2:
        fold = operations.MeshProgramDescriptor()
        for chip in range(chips):
            partial, ident, value, word = partial_shards[chip], id_shards[chip], value_shard_list[chip], word_shards[chip]
            runtime = operations.RuntimeArgs()
            runtime[0][0] = [partial.buffer_address(), ident.buffer_address(), value.buffer_address(), word.buffer_address(), rows,
                             per_tile_row, tile_rows]
            kernel = operations.KernelDescriptor(kernel_source=str(Path(__file__).with_name(FOLD_KERNEL)), core_ranges=one,
                compile_time_args=[argument for tensor in (partial, ident, value, word)
                                   for argument in operations.TensorAccessorArgs(tensor).get_compile_time_args()],
                defines=[('FOLD_MAX_PAGES', str(task_pages))],
                runtime_args=runtime, config=operations.DataMovementConfigDescriptor(
                    processor=operations.DataMovementProcessor.RISCV_0, noc=operations.NOC.RISCV_0_default))
            coordinate = operations.MeshCoordinate(0, chip)
            fold[operations.MeshCoordinateRange(coordinate, coordinate)] = operations.ProgramDescriptor(
                kernels=[kernel], cbs=[_cb(operations, one, 0, fold_scratch_pages(task_pages), operations.bfloat16)])
        operations.generic_op([partials, ids, values, words], fold)
        return 2

    groups = FOLD2_GROUPS
    blocks = group_plan(per_tile_row, groups)
    group_pages = group_pages_bound(task_pages, groups)
    level1 = operations.MeshProgramDescriptor()
    group_cores = _cores(operations, groups, 1)
    for chip in range(chips):
        partial, group_buffer = partial_shards[chip], group_shards[chip]
        runtime = operations.RuntimeArgs()
        for group, _first, _last in blocks:
            runtime[group][0] = [partial.buffer_address(), group_buffer.buffer_address(), rows, per_tile_row, tile_rows, groups, group]
        kernel = operations.KernelDescriptor(kernel_source=str(Path(__file__).with_name(FOLD2_KERNEL)), core_ranges=group_cores,
            compile_time_args=[argument for tensor in (partial, group_buffer)
                               for argument in operations.TensorAccessorArgs(tensor).get_compile_time_args()],
            defines=[('FOLD2_LEVEL', '1'), ('FOLD2_GROUP_PAGES', str(group_pages))],
            runtime_args=runtime, config=operations.DataMovementConfigDescriptor(
                processor=operations.DataMovementProcessor.RISCV_0, noc=operations.NOC.RISCV_0_default))
        coordinate = operations.MeshCoordinate(0, chip)
        level1[operations.MeshCoordinateRange(coordinate, coordinate)] = operations.ProgramDescriptor(
            kernels=[kernel], cbs=[_cb(operations, group_cores, 0, _pages(group_pages * PAGE_BYTES + GROUP_PAGE_BYTES),
                                       operations.bfloat16)])
    operations.generic_op([partials, groups_buffer], level1)

    level2 = operations.MeshProgramDescriptor()
    for chip in range(chips):
        group_buffer, ident, value, word = group_shards[chip], id_shards[chip], value_shard_list[chip], word_shards[chip]
        runtime = operations.RuntimeArgs()
        runtime[0][0] = [group_buffer.buffer_address(), ident.buffer_address(), value.buffer_address(), word.buffer_address(), rows, groups]
        kernel = operations.KernelDescriptor(kernel_source=str(Path(__file__).with_name(FOLD2_KERNEL)), core_ranges=one,
            compile_time_args=[argument for tensor in (group_buffer, ident, value, word)
                               for argument in operations.TensorAccessorArgs(tensor).get_compile_time_args()],
            defines=[('FOLD2_LEVEL', '2'), ('FOLD2_GROUP_PAGES', str(group_pages))],
            runtime_args=runtime, config=operations.DataMovementConfigDescriptor(
                processor=operations.DataMovementProcessor.RISCV_0, noc=operations.NOC.RISCV_0_default))
        coordinate = operations.MeshCoordinate(0, chip)
        level2[operations.MeshCoordinateRange(coordinate, coordinate)] = operations.ProgramDescriptor(
            kernels=[kernel], cbs=[_cb(operations, one, 0, _pages(groups * GROUP_PAGE_BYTES + IDS_BYTES + VALUES_BYTES + WORDS_BYTES),
                                       operations.bfloat16)])
    operations.generic_op([groups_buffer, ids, values, words], level2)
    return 3


def sample(operations, logits, rows, served=None):
    """(ids, values) per chip by the scan and fold launches, or None (after one logged line) when this call cannot take them.
    `served()` is today's (ids, values) call, run beside under the audit. The joined words tensor of the same launch is
    `words_for(values)`."""
    reserved = _RESERVED.get('mesh')
    reason = problem(operations, logits, rows, reserved[4] if reserved is not None else GRID)
    if reason is not None:
        tp4_sampdraft.note(tp4_sampdraft.SARG_FALLBACK, 'rows=%d reason=%s' % (rows if type(rows) is int else -1, reason))
        return None
    audit = tp4_sampdraft.audit_enabled(tp4_sampdraft.SHARD_ARGMAX_AUDIT)
    chips = tp_shapes.chip_count()
    mesh = logits.device()
    logit_shards = operations.get_device_tensors(logits)
    if len(logit_shards) != chips:
        tp4_sampdraft.note(tp4_sampdraft.SARG_FALLBACK, 'rows=%d reason=logits are not one shard per chip' % rows)
        return None
    if reserved is None:
        tp4_sampdraft.note(tp4_sampdraft.SARG_FALLBACK, 'rows=%d reason=no partials buffer reserved before capture (only the packed '
                           'block reserves one)' % rows)
        return None
    partials, groups_buffer, grid, fold2 = reserved[0], reserved[3], reserved[4], reserved[5]
    dram = operations.DRAM_MEMORY_CONFIG
    ids = values = words = None
    try:
        ids = operations.empty((1, 1, 1, OUTPUT_WORDS), dtype=operations.uint32, layout=operations.ROW_MAJOR_LAYOUT,
                               device=mesh, memory_config=dram)
        values = operations.empty((1, 1, 1, OUTPUT_WORDS), dtype=operations.bfloat16, layout=operations.ROW_MAJOR_LAYOUT,
                                  device=mesh, memory_config=dram)
        words = operations.empty((1, 1, 1, OUTPUT_WORDS), dtype=operations.uint32, layout=operations.ROW_MAJOR_LAYOUT,
                                 device=mesh, memory_config=dram)
        launch(operations, logits, rows, partials, groups_buffer, ids, values, words, grid, fold2)
    except BaseException:
        for tensor in (ids, values, words):
            if tensor is not None:
                operations.deallocate(tensor)
        raise
    if audit and served is not None:
        try:
            REFERENCES[id(values)] = served()
        except BaseException:
            for tensor in (ids, values, words):
                operations.deallocate(tensor)
            raise
    PRODUCED[id(values)] = values
    WORDS[id(values)] = words
    workers = grid[0] * grid[1]
    tasks, _, _ = plan(rows, tp_shapes.vocab_shard() // TILE, workers)
    tp4_sampdraft.note(tp4_sampdraft.SARG_ENGAGED, 'rows=%d workers=%d tasks=%d fold=%d audit=%d words=1' % (
        rows, workers, len(tasks), 2 if fold2 else 1, int(audit)))
    return ids, values


def words_for(values):
    """The joined (column << 16) | bits tensor of the launch that made `values` (the second output of sample), or None."""
    return WORDS.get(id(values)) if values is not None else None


def pack_words(ids, values):
    """The host word the kernels write for each row: (column << 16) | the value's bf16 bits. `ids` and `values` are the host tensors a
    reader gets for one chip, already cut to the live rows. Returns an int64 tensor."""
    import torch

    ids = torch.as_tensor(ids).reshape(-1).to(torch.int64) & 0xFFFFFFFF
    values = torch.as_tensor(values).reshape(-1)
    if values.dtype != torch.bfloat16:
        values = values.to(torch.bfloat16)
    bits = values.contiguous().view(torch.int16).to(torch.int64) & 0xFFFF
    return (ids << 16) | bits


def decode_words(words, rows=None):
    """(ids, values) from a chip's host words tensor, the reader of the joined layout: ids int64 (the shard-local columns) and values
    bf16 (the maxima's own bits). `words` is what to_torch returns for the uint32 tensor (int32 or wider: a column above 32,767 sets
    the sign bit), cut to `rows` when given."""
    import torch

    flat = torch.as_tensor(words).reshape(-1)
    if rows is not None:
        flat = flat[:rows]
    wide = flat.to(torch.int64) & 0xFFFFFFFF
    bits = wide & 0xFFFF
    signed = torch.where(bits >= 0x8000, bits - 0x10000, bits).to(torch.int16)
    return wide >> 16, signed.contiguous().view(torch.bfloat16)


def compare(left, right):
    """The audit's rule for one chip's (ids, values) against today's, per row: ids exactly; values as numbers (-0 == +0), both NaN
    equal. `left` and `right` are (ids, values) torch tensors already cut to the live rows. Returns the differing rows."""
    import torch

    left_ids = torch.as_tensor(left[0]).reshape(-1).to(torch.int64)
    right_ids = torch.as_tensor(right[0]).reshape(-1).to(torch.int64)
    left_values = torch.as_tensor(left[1]).reshape(-1).to(torch.float32)
    right_values = torch.as_tensor(right[1]).reshape(-1).to(torch.float32)
    if left_ids.numel() != right_ids.numel() or left_values.numel() != right_values.numel() \
            or left_ids.numel() != left_values.numel():
        raise AssertionError('The shard argmax audit compares %d/%d ids and %d/%d values' % (
            left_ids.numel(), right_ids.numel(), left_values.numel(), right_values.numel()))
    same_values = (left_values == right_values) | (torch.isnan(left_values) & torch.isnan(right_values))
    same = (left_ids == right_ids) & same_values
    return [row for row, ok in enumerate(same.tolist()) if not ok]


def audit_round(operations, values, rows, chip_ids, chip_values):
    """One audited round: every chip's kernel (ids, values), already read to the host and cut to `rows`, against today's from the
    same capture, every row; and the kernel's joined words tensor against its own ids and value bits, exactly. A no-op when `values` is
    not an audited output (the audit is off). Logs the audit marker; a difference logs the mismatch marker and raises AssertionError."""
    reference = REFERENCES.get(id(values))
    if reference is None:
        if tp4_sampdraft.audit_enabled(tp4_sampdraft.SHARD_ARGMAX_AUDIT):
            message = '%s no reference: the audited kernel call recorded none' % tp4_sampdraft.SARG_MISMATCH
            tp4_sampdraft.log_line(message)
            raise AssertionError(message)
        return
    ids, held = reference
    _STATE['rounds'] += 1
    id_parts = operations.get_device_tensors(ids)
    value_parts = operations.get_device_tensors(held)
    if len(id_parts) != len(chip_ids) or len(value_parts) != len(chip_values):
        raise AssertionError('The shard argmax audit needs the served outputs of every chip')
    for chip in range(len(chip_ids)):
        served = (operations.to_torch(id_parts[chip]).reshape(-1)[:rows], operations.to_torch(value_parts[chip]).reshape(-1)[:rows])
        differing = compare((chip_ids[chip], chip_values[chip]), served)
        if differing:
            message = '%s round=%d chip=%d rows=%s kernel=%s served=%s' % (
                tp4_sampdraft.SARG_MISMATCH, _STATE['rounds'], chip, differing[:8],
                [int(chip_ids[chip][row]) for row in differing[:8]], [int(served[0][row]) for row in differing[:8]])
            tp4_sampdraft.log_line(message)
            raise AssertionError(message)
    joined = WORDS.get(id(values))
    if joined is not None:
        word_parts = operations.get_device_tensors(joined)
        if len(word_parts) != len(chip_ids):
            raise AssertionError('The shard argmax audit needs the words of every chip')
        for chip in range(len(chip_ids)):
            held_words = operations.to_torch(word_parts[chip]).reshape(-1)[:rows]
            kernel_words = pack_words(chip_ids[chip], chip_values[chip])
            got = held_words.to(kernel_words.dtype) & 0xFFFFFFFF
            wrong = [row for row, ok in enumerate((got == kernel_words).tolist()) if not ok]
            if wrong:
                message = '%s round=%d chip=%d words rows=%s' % (tp4_sampdraft.SARG_MISMATCH, _STATE['rounds'], chip, wrong[:8])
                tp4_sampdraft.log_line(message)
                raise AssertionError(message)
    tp4_sampdraft.log_line('%s %d exact=True rows=%d chips=%d' % (tp4_sampdraft.SARG_AUDIT, _STATE['rounds'], rows,
                                                                 len(chip_ids)))


def produced(values):
    """Whether `values` is the maxima output of a kernel launch (not of today's path)."""
    return values is not None and id(values) in PRODUCED


def release_audit(operations, values):
    """Forget the trace output `values`, free the words tensor of its launch and today's (ids, values) the audit holds for it (nothing is
    held with the audit off)."""
    joined = None
    if values is not None:
        PRODUCED.pop(id(values), None)
        joined = WORDS.pop(id(values), None)
    if joined is not None:
        try:
            operations.deallocate(joined)
        except BaseException:
            pass
    reference = REFERENCES.pop(id(values), None) if values is not None else None
    if reference is not None:
        # today's path's maxima may also be the V4a audit's own reference entry (verify_trace_t1.shard_values registers
        # VALUE_REFERENCES[id(served values)] under QWEN_FAST_TP4_VGLUE_AUDIT, and release_vglue_audit pops only the kernel's id): pop it
        # here so a stale id cannot match a later tensor, and free its ttnn.max reference with the pair.
        module = sys.modules.get('verify_trace_t1')
        stale = module.VALUE_REFERENCES.pop(id(reference[1]), None) if module is not None else None
        for tensor in (*reference, stale):
            if tensor is not None:
                try:
                    operations.deallocate(tensor)
                except BaseException:
                    pass
