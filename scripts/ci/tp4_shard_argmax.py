"""The per-shard argmax of the TP4 verify trace as two small launches (QWEN_FAST_TP4_SHARD_ARGMAX, default off; S1).

WHY. verify_trace_t1.sample_shards takes each chip's (rows, 62,080) bf16 TILE logits shard through an untilize (46 us), ttnn.argmax
(940 us: element-serial on the data-movement cores, about 26 ns an element) and, for the shard maxima, either a 2-core ttnn.max
(787 us) or the V4a gather (QWEN_FAST_TP4_SHARD_VALUES): about 1.0 ms of the 56-59 ms verify trace plus the gather, 1.8 ms in the
production recipe's profile.

WHAT. tp4_shard_argmax_scan.cpp scans the TILE logits in place on 110 workers (two data-movement RISC-Vs each, 220 tasks), each task a
run of tile columns of one tile row (rows 33..64: one tile row per RISC; up to 32 rows: the two RISCs split the column run in
halves), and writes per row one word (column << 16) | bits to a partials page. tp4_shard_argmax_fold.cpp folds the partials of each
row in ascending column order with a strict greater-than, on one core, and writes the ids and the values. The answer is the first
index holding the maximum, the rule of torch.argmax, of the pinned sampler and of combine_shards: the combined global ids are
identical to today's for every row without a NaN, and a NaN row follows torch (the first NaN wins). The values are the maximum
element's own bits; the only possible bit difference from ttnn.max is -0 against +0 when a row's maximum is zero, which
combine_shards compares equal.

CONTRACT. It returns (ids, values) like sample_shards: ids uint32 and values bf16, ROW_MAJOR, one page each per chip, shaped
(1, 1, 1, 64) (page = 64 words, 256 B and 128 B: whole DRAM alignment units, which a 4-byte row page of the old (1, 1, rows, 1)
shape is not). Rows at and past `rows` are zero. Every consumer reads them with to_torch(part).reshape(-1)[:rows]
(PackedVerifierEngine.shard_predictions, verifier_engine_tp.shard_predictions), so the shape change is invisible to them.

PERSISTENT SCRATCH. The partials buffer is allocated once per mesh by reserve(), which the packed block calls at its warm, before any trace
is captured, and freed by release_reserved() when the last block closes: a buffer allocated inside the verify capture would change
that trace's hole layout (the traced-publish hang mechanism). One buffer serves every block: replays are serial on one queue and
the fold consumes the partials inside the same trace that wrote them. A call on a mesh with no reservation (the request engine, a
caller that never reserved) is a logged fall-back, never an allocation inside a capture.

FALLBACK. Any call it cannot take (another shard shape, dtype, layout, placement, more than 64 rows, a tile-column plan whose runs
exceed the scan kernel's scratch) is logged once and answered by today's untilize / argmax / max / gather path.

AUDIT (QWEN_FAST_TP4_SHARD_ARGMAX_AUDIT=1, needs the lever). Today's path runs beside the kernel in the same capture on the same
logits, its (ids, values) are held, and every round PackedVerifierEngine.shard_predictions calls audit_round(), which compares every
row of every chip: ids exactly, values as numbers (-0 == +0, NaN with NaN). A difference logs SARG_MISMATCH and raises.

Stdlib only at import, py 3.7.
"""

import sys
from pathlib import Path

import tp4_sampdraft
import tp_shapes

SCAN_KERNEL = 'tp4_shard_argmax_scan.cpp'
FOLD_KERNEL = 'tp4_shard_argmax_fold.cpp'
WORKERS = 110
GRID = (11, 10)                    # worker w sits at (w // 10, w % 10), the quad conv's E1b order
TILE = 32
MAX_ROWS = 64
MAX_TILES = 18                     # the scan kernel's scratch: tiles of one task
PAGE_WORDS = 32                    # a partials page: one word per row of a tile row
PAGE_BYTES = PAGE_WORDS * 4
TASKS = 2 * WORKERS                # two RISC-Vs per worker, whatever the row count
OUTPUT_WORDS = 64                  # ids / values page: 64 words / 64 bf16
SCAN_CB_PAGES = MAX_TILES + 1      # the tiles plus the result page
FOLD_CB_PAGES = 16                 # 32 KB: 220 partials pages (28,160 B) plus the 256 B and 128 B outputs
TILE_BYTES = 2048

# Held (today's ids, today's values) of the audit, keyed by id(values), and the audit's round counter. PRODUCED holds id(values) of every
# output a kernel launch made (not today's path), so the V4a value audit (packed_verifier.audit_shard_values), which compares gathered
# maxima with a ttnn.max taken beside them in served_shards, knows these are not its outputs and leaves them to this module's audit.
REFERENCES = {}
PRODUCED = {}                      # id(values) -> the values tensor itself, so a freed tensor's id cannot be reused while it is listed
_STATE = {'rounds': 0}
_RESERVED = {}                     # 'mesh' -> [partials tensor, holders, mesh]: one mesh per process (logits.device() need not be the
                                   # same Python object as the block's mesh, so identity is not the key)


def reserve(operations, mesh):
    """Allocate the process's partials buffer on `mesh` (or add a holder to the one that exists). Call before any trace is captured; pair it with
    release_reserved. Returns the tensor."""
    entry = _RESERVED.get('mesh')
    if entry is None:
        tensor = operations.empty((1, 1, TASKS, PAGE_WORDS), dtype=operations.uint32, layout=operations.ROW_MAJOR_LAYOUT,
                                  device=mesh, memory_config=operations.DRAM_MEMORY_CONFIG)
        entry = _RESERVED['mesh'] = [tensor, 0, mesh]
    entry[1] += 1
    return entry[0]


def release_reserved(operations, mesh):
    """Drop one holder of this mesh's partials buffer; the last holder frees it. A mesh with no reservation is a no-op. Call only after
    every trace that reads or writes the buffer is released."""
    entry = _RESERVED.get('mesh')
    if entry is None:
        return
    entry[1] -= 1
    if entry[1] <= 0:
        del _RESERVED['mesh']
        try:
            operations.deallocate(entry[0])
        except BaseException:
            pass


def column_runs(tile_columns, workers=WORKERS):
    """[(first, last)] per worker: contiguous runs of tile columns in ascending order, the first `extra` workers one column longer
    (1,940 columns over 110 workers: 70 runs of 18 and 40 of 17)."""
    if tile_columns < workers:
        raise ValueError('%d tile columns for %d workers' % (tile_columns, workers))
    base, extra = divmod(tile_columns, workers)
    runs, start = [], 0
    for worker in range(workers):
        stop = start + base + (1 if worker < extra else 0)
        runs.append((start, stop))
        start = stop
    return runs


def plan(rows, tile_columns):
    """The task list for a (rows, 32 * tile_columns) shard: (task page, worker, role, tile row, first column, one past the last
    column, live rows), in the order the fold reads them, plus (tasks per tile row, tile rows). Raises ValueError when the shard does
    not fit the kernels."""
    if type(rows) is not int or not 1 <= rows <= MAX_ROWS:
        raise ValueError('rows must be 1 to %d, got %r' % (MAX_ROWS, rows))
    runs = column_runs(tile_columns)
    if max(last - first for first, last in runs) > MAX_TILES:
        raise ValueError('a run of %d tile columns does not fit the scan scratch of %d tiles' % (
            max(last - first for first, last in runs), MAX_TILES))
    tile_rows = (rows + TILE - 1) // TILE
    tasks = []
    if tile_rows == 2:
        for tile_row in range(2):
            live = min(TILE, rows - TILE * tile_row)
            for worker, (first, last) in enumerate(runs):
                tasks.append((tile_row * WORKERS + worker, worker, tile_row, tile_row, first, last, live))
        return tasks, WORKERS, 2
    if min(last - first for first, last in runs) < 2:
        # at up to 32 rows each RISC takes half of a worker's run: an empty half would write a candidate that never was one
        raise ValueError('%d tile columns leave a worker run too short to split between its two RISC-Vs' % tile_columns)
    for worker, (first, last) in enumerate(runs):
        middle = first + (last - first + 1) // 2
        tasks.append((2 * worker, worker, 0, 0, first, middle, rows))
        tasks.append((2 * worker + 1, worker, 1, 0, middle, last, rows))
    return tasks, 2 * WORKERS, 1


def problem(operations, logits, rows):
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
        plan(rows, width // TILE)
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


def sample(operations, logits, rows, served=None):
    """(ids, values) per chip by the scan and fold launches, or None (after one logged line) when this call cannot take them.
    `served()` is today's (ids, values) call, run beside under the audit."""
    reason = problem(operations, logits, rows)
    if reason is not None:
        tp4_sampdraft.note(tp4_sampdraft.SARG_FALLBACK, 'rows=%d reason=%s' % (rows if type(rows) is int else -1, reason))
        return None
    audit = tp4_sampdraft.audit_enabled(tp4_sampdraft.SHARD_ARGMAX_AUDIT)
    chips = tp_shapes.chip_count()
    mesh = logits.device()
    tile_columns = tp_shapes.vocab_shard() // TILE
    tasks, per_tile_row, tile_rows = plan(rows, tile_columns)
    logit_shards = operations.get_device_tensors(logits)
    if len(logit_shards) != chips:
        tp4_sampdraft.note(tp4_sampdraft.SARG_FALLBACK, 'rows=%d reason=logits are not one shard per chip' % rows)
        return None
    reserved = _RESERVED.get('mesh')
    if reserved is None:
        tp4_sampdraft.note(tp4_sampdraft.SARG_FALLBACK, 'rows=%d reason=no partials buffer reserved before capture (only the packed '
                           'block reserves one)' % rows)
        return None
    partials = reserved[0]
    dram = operations.DRAM_MEMORY_CONFIG
    ids = values = None
    try:
        ids = operations.empty((1, 1, 1, OUTPUT_WORDS), dtype=operations.uint32, layout=operations.ROW_MAJOR_LAYOUT,
                               device=mesh, memory_config=dram)
        values = operations.empty((1, 1, 1, OUTPUT_WORDS), dtype=operations.bfloat16, layout=operations.ROW_MAJOR_LAYOUT,
                                  device=mesh, memory_config=dram)
        partial_shards = operations.get_device_tensors(partials)
        id_shards = operations.get_device_tensors(ids)
        value_shards = operations.get_device_tensors(values)

        workers = _cores(operations, GRID[0], GRID[1])
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
                        runtime[worker // GRID[1]][worker % GRID[1]] = [
                            logit.buffer_address(), partial.buffer_address(), tile_row, first, last, live, task, role,
                            tile_columns]
                kernels.append(operations.KernelDescriptor(
                    kernel_source=str(Path(__file__).with_name(SCAN_KERNEL)), core_ranges=workers,
                    compile_time_args=[*arguments[0], *arguments[1]], runtime_args=runtime,
                    config=operations.DataMovementConfigDescriptor(
                        processor=(operations.DataMovementProcessor.RISCV_0, operations.DataMovementProcessor.RISCV_1)[role],
                        noc=(operations.NOC.RISCV_0_default, operations.NOC.RISCV_1_default)[role])))
            coordinate = operations.MeshCoordinate(0, chip)
            scan[operations.MeshCoordinateRange(coordinate, coordinate)] = operations.ProgramDescriptor(
                kernels=kernels, cbs=[_cb(operations, workers, index, SCAN_CB_PAGES, operations.bfloat16) for index in (0, 1)])
        operations.generic_op([logits, partials], scan)

        one = _cores(operations, 1, 1)
        fold = operations.MeshProgramDescriptor()
        for chip in range(chips):
            partial, ident, value = partial_shards[chip], id_shards[chip], value_shards[chip]
            runtime = operations.RuntimeArgs()
            runtime[0][0] = [partial.buffer_address(), ident.buffer_address(), value.buffer_address(), rows,
                             per_tile_row, tile_rows]
            kernel = operations.KernelDescriptor(kernel_source=str(Path(__file__).with_name(FOLD_KERNEL)), core_ranges=one,
                compile_time_args=[argument for tensor in (partial, ident, value)
                                   for argument in operations.TensorAccessorArgs(tensor).get_compile_time_args()],
                runtime_args=runtime, config=operations.DataMovementConfigDescriptor(
                    processor=operations.DataMovementProcessor.RISCV_0, noc=operations.NOC.RISCV_0_default))
            coordinate = operations.MeshCoordinate(0, chip)
            fold[operations.MeshCoordinateRange(coordinate, coordinate)] = operations.ProgramDescriptor(
                kernels=[kernel], cbs=[_cb(operations, one, 0, FOLD_CB_PAGES, operations.bfloat16)])
        operations.generic_op([partials, ids, values], fold)
    except BaseException:
        for tensor in (ids, values):
            if tensor is not None:
                operations.deallocate(tensor)
        raise
    if audit and served is not None:
        try:
            REFERENCES[id(values)] = served()
        except BaseException:
            operations.deallocate(ids)
            operations.deallocate(values)
            raise
    PRODUCED[id(values)] = values
    tp4_sampdraft.note(tp4_sampdraft.SARG_ENGAGED, 'rows=%d workers=%d tasks=%d fold=1 audit=%d' % (
        rows, WORKERS, len(tasks), int(audit)))
    return ids, values


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
    same capture, every row. A no-op when `values` is not an audited output (the audit is off). Logs the audit marker; a
    difference logs the mismatch marker and raises AssertionError."""
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
    tp4_sampdraft.log_line('%s %d exact=True rows=%d chips=%d' % (tp4_sampdraft.SARG_AUDIT, _STATE['rounds'], rows,
                                                                 len(chip_ids)))


def produced(values):
    """Whether `values` is the maxima output of a kernel launch (not of today's path)."""
    return values is not None and id(values) in PRODUCED


def release_audit(operations, values):
    """Forget the trace output `values` and free today's (ids, values) the audit holds for it (nothing is held with the audit off)."""
    if values is not None:
        PRODUCED.pop(id(values), None)
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
