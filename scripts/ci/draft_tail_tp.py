"""The drafter MLP branch's SwiGLU and residual tail as one launch each at four cards (QWEN_FAST_DRAFT_TAIL, default off; tp4/fx-wp6, F-F3a).

WHY. Per layer the drafter MLP branch runs draft_mlp.swiglu_device - seven typecasts, a silu and a multiply (nine launches, 54 us on the v676
quad: 0.270 ms a quad over five layers) - and, after the learned convolution, the residual add as two typecasts, an fp32 add and a typecast
(four launches, 0.137 ms a quad per branch). Every one of them reads a DRAM tensor and writes one, for a value that is a single tile of
arithmetic.

WHAT. draft_tail_tp_swiglu_compute.cpp does the whole SwiGLU on one pair of tiles in the destination registers: bf16-round the gate, silu it,
bf16-round the activation, bf16-round the up, multiply, bf16-round the product, pack bf16. draft_tail_tp_residual_compute.cpp does the
residual: widen both bf16 tiles, fp32-add, bf16-round, pack. Both are read by draft_tail_tp_io.cpp (two operand tiles per task, any page
stride, so a fused gate|up matmul output is read in place) and written by draft_fuse_out.cpp, over min(tiles, grid, WORKER_CAP) cores of the grid
the device reports.

EXACTNESS. Bit-identical, by construction. The served compositions pass every intermediate through a DRAM tensor of the stated dtype: the
steps that change a value are the fp32 -> bf16 roundings (ttnn.typecast = typecast_tile<Float32, Float16_b>), the silu (ttnn.silu on an fp32
tensor = silu_tile, no approximation mode), the multiply and the add (binary_ng's SFPU multiply and add of two fp32 tensors); the bf16 -> fp32
widenings and the DRAM round trips are lossless, so a register that holds the rounded value as an fp32 lane is the widened tensor. The new
kernels execute those same primitives in the same order. CPU: the dataflow is transliterated and held against the served composition for every
rounding point, including ties, overflow to infinity, -0 and a reversed order (test_draft_tail_tp); the silu is the one primitive a CPU cannot
reproduce bit for bit, so the CPU test shares it between both sides and the card audit is what holds it: QWEN_FAST_DRAFT_TAIL_AUDIT=1 runs the
served composition on the very same operands beside each launch and compares every chip's bytes (draft_fusion_tp's eager audit).

CALLERS. draft_mlp_branch.execute_mlp_branch calls swiglu() / swiglu_fused() and residual() when the flag is on (the MLP branch's own file).
The attention branch ends in the same residual chain; its file is WP7's, and it gets the same one-line residual() hook there. A call this module cannot take
(another shape, dtype, layout or placement, an unreadable grid, a launch that raises) is a logged fall-back to the served ops.

Stdlib only at import, py 3.7. The flag is strict 0 or 1 (draft_fusion_tp) and refused at the pair.
"""

from pathlib import Path

import draft_fusion_tp as fusion

IO_KERNEL = 'draft_tail_tp_io.cpp'
SWIGLU_KERNEL = 'draft_tail_tp_swiglu_compute.cpp'
RESIDUAL_KERNEL = 'draft_tail_tp_residual_compute.cpp'
OUT_KERNEL = 'draft_fuse_out.cpp'
HIDDEN = 5120
FP32_TILE_BYTES = 4096
BF16_TILE_BYTES = 2048
INPUT_PAGES = 2                              # per operand: the next tile is read while this one is computed
OUTPUT_PAGES = 2
MAX_ROWS = 64

RUNTIME_FILES = ('draft_tail_tp.py', IO_KERNEL, SWIGLU_KERNEL, RESIDUAL_KERNEL, OUT_KERNEL, 'draft_fusion_tp.py')


def enabled(environ=None):
    return fusion.enabled(fusion.TAIL, environ)


def plan(rows, columns, grid):
    """The launch geometry for a (rows, 32 * columns) result: tile rows, the tasks (one per output tile, row-major), the workers and the
    runs [(first task, count)]. ValueError for a row count or width the kernels do not take."""
    if type(rows) is not int or not 1 <= rows <= MAX_ROWS:
        raise ValueError('rows must be 1 to %d, got %r' % (MAX_ROWS, rows))
    if type(columns) is not int or columns < 1:
        raise ValueError('columns (tiles across) must be a positive integer, got %r' % (columns,))
    tile_rows = (rows + 31) // 32
    tasks = tile_rows * columns
    workers = fusion.worker_count(tasks, grid)
    return dict(rows=rows, columns=columns, tile_rows=tile_rows, tasks=tasks, workers=workers, runs=fusion.plan_runs(tasks, workers))


def _tiled_dram(operations, tensor, dtype):
    if getattr(tensor, 'dtype', None) != dtype or getattr(tensor, 'layout', None) != operations.TILE_LAYOUT:
        return False
    try:
        return tensor.memory_config() == operations.DRAM_MEMORY_CONFIG
    except Exception:  # noqa: BLE001 - a diagnostic read
        return False


def swiglu_problem(operations, mesh, gate, up, *, fused=False):
    """Why these projections cannot take the launch (a short reason), or None. `fused`: `gate` is the (1, 1, rows, 2 * width) gate|up output
    and `up` is None."""
    shape = tuple(gate.shape)
    if len(shape) != 4 or shape[:2] != (1, 1) or not 1 <= shape[2] <= MAX_ROWS:
        return 'shape %r is not (1, 1, rows <= %d, width)' % (shape, MAX_ROWS)
    if fused:
        if shape[3] % 64:
            return 'the fused width %d is not two whole tile columns of halves' % shape[3]
    else:
        if tuple(up.shape) != shape:
            return 'gate %r and up %r differ' % (shape, tuple(up.shape))
        if shape[3] % 32:
            return 'the width %d is not whole tiles' % shape[3]
    for tensor in ((gate,) if fused else (gate, up)):
        if not _tiled_dram(operations, tensor, operations.float32):
            return 'a projection is not float32 TILE interleaved DRAM'
    if fusion.grid_of(mesh) is None:
        return 'the compute grid cannot be read'
    return None


def residual_problem(operations, mesh, finished, hidden):
    shape = tuple(finished.shape)
    if len(shape) != 4 or shape[:2] != (1, 1) or shape[3] != HIDDEN or not 1 <= shape[2] <= MAX_ROWS:
        return 'shape %r is not (1, 1, rows <= %d, %d)' % (shape, MAX_ROWS, HIDDEN)
    if tuple(hidden.shape) != shape:
        return 'finished %r and hidden %r differ' % (shape, tuple(hidden.shape))
    for tensor in (finished, hidden):
        if not _tiled_dram(operations, tensor, operations.bfloat16):
            return 'a block is not bfloat16 TILE interleaved DRAM'
    if fusion.grid_of(mesh) is None:
        return 'the compute grid cannot be read'
    return None


def _launch(operations, mesh, compute_source, left, right, shape, found, *, left_bytes, right_bytes, out_dtype, out_bytes, left_stride,
            right_stride, right_column, fp32_inputs):
    """One tail launch: a new `shape` tensor of `out_dtype`, tile (row, column) computed from the left tensor's page row * left_stride + column
    and the right tensor's page row * right_stride + right_column + column. Returns the output (the caller owns it)."""
    grid = fusion.grid_of(mesh)
    chips = len(operations.get_device_tensors(left))
    output = operations.empty(tuple(shape), dtype=out_dtype, layout=operations.TILE_LAYOUT, device=mesh,
                              memory_config=operations.DRAM_MEMORY_CONFIG)
    try:
        cores = fusion.core_ranges(operations, grid, found['workers'])
        coordinates = fusion.coordinates(grid, found['workers'])
        data_in = operations.float32 if fp32_inputs else operations.bfloat16
        buffers = [fusion.circular_buffer(operations, cores, 0, INPUT_PAGES, data_in, left_bytes),
                   fusion.circular_buffer(operations, cores, 1, INPUT_PAGES, data_in, right_bytes),
                   fusion.circular_buffer(operations, cores, 16, OUTPUT_PAGES, out_dtype, out_bytes)]
        config = fusion.fp32_compute_config(operations, (0, 1) if fp32_inputs else ())
        lefts, rights = operations.get_device_tensors(left), operations.get_device_tensors(right)
        outs = operations.get_device_tensors(output)
        if len(lefts) != chips or len(rights) != chips or len(outs) != chips:
            raise ValueError('one shard per chip required')
        program = operations.MeshProgramDescriptor()
        for chip in range(chips):
            local_left, local_right, local_out = lefts[chip], rights[chip], outs[chip]
            if local_out.buffer_address() in (local_left.buffer_address(), local_right.buffer_address()):
                raise ValueError('The tail output must not alias its operands')
            reading, writing, computing = operations.RuntimeArgs(), operations.RuntimeArgs(), operations.RuntimeArgs()
            for (x, y), (first, count) in zip(coordinates, found['runs']):
                reading[x][y] = [local_left.buffer_address(), local_right.buffer_address(), first, count, left_stride, right_stride,
                                 right_column, found['columns']]
                writing[x][y] = [local_out.buffer_address(), first, count]
                computing[x][y] = [count]
            reader = operations.KernelDescriptor(kernel_source=str(Path(__file__).with_name(IO_KERNEL)), core_ranges=cores,
                compile_time_args=list(operations.TensorAccessorArgs(local_left).get_compile_time_args())
                + list(operations.TensorAccessorArgs(local_right).get_compile_time_args()) + [left_bytes, right_bytes],
                runtime_args=reading, config=operations.DataMovementConfigDescriptor(
                    processor=operations.DataMovementProcessor.RISCV_0, noc=operations.NOC.RISCV_0_default))
            writer = operations.KernelDescriptor(kernel_source=str(Path(__file__).with_name(OUT_KERNEL)), core_ranges=cores,
                compile_time_args=list(operations.TensorAccessorArgs(local_out).get_compile_time_args()) + [out_bytes],
                runtime_args=writing, config=operations.DataMovementConfigDescriptor(
                    processor=operations.DataMovementProcessor.RISCV_1, noc=operations.NOC.RISCV_1_default))
            compute = operations.KernelDescriptor(kernel_source=str(Path(__file__).with_name(compute_source)), core_ranges=cores,
                compile_time_args=[], runtime_args=computing, config=config)
            coordinate = operations.MeshCoordinate(0, chip)
            program[operations.MeshCoordinateRange(coordinate, coordinate)] = operations.ProgramDescriptor(
                kernels=[reader, writer, compute], cbs=buffers)
        operations.generic_op([left] + ([] if right is left else [right]) + [output], program)      # a fused gate|up is one tensor read twice
    except BaseException:
        operations.deallocate(output)
        raise
    return output


def swiglu_launch(operations, mesh, gate, up, rows, width):
    """The SwiGLU launch on separate gate and up tensors: a new (1, 1, rows, width) bf16 tensor."""
    columns = width // 32
    found = plan(rows, columns, fusion.grid_of(mesh))
    return _launch(operations, mesh, SWIGLU_KERNEL, gate, up, (1, 1, rows, width), found, left_bytes=FP32_TILE_BYTES,
                   right_bytes=FP32_TILE_BYTES, out_dtype=operations.bfloat16, out_bytes=BF16_TILE_BYTES, left_stride=columns,
                   right_stride=columns, right_column=0, fp32_inputs=True)


def swiglu_fused_launch(operations, mesh, gate_up, rows, width):
    """The SwiGLU launch reading the gate half and the up half of one fused (1, 1, rows, 2 * width) projection in place."""
    columns = width // 32
    found = plan(rows, columns, fusion.grid_of(mesh))
    return _launch(operations, mesh, SWIGLU_KERNEL, gate_up, gate_up, (1, 1, rows, width), found, left_bytes=FP32_TILE_BYTES,
                   right_bytes=FP32_TILE_BYTES, out_dtype=operations.bfloat16, out_bytes=BF16_TILE_BYTES, left_stride=2 * columns,
                   right_stride=2 * columns, right_column=columns, fp32_inputs=True)


def residual_launch(operations, mesh, finished, hidden, rows):
    columns = HIDDEN // 32
    found = plan(rows, columns, fusion.grid_of(mesh))
    return _launch(operations, mesh, RESIDUAL_KERNEL, finished, hidden, (1, 1, rows, HIDDEN), found, left_bytes=BF16_TILE_BYTES,
                   right_bytes=BF16_TILE_BYTES, out_dtype=operations.bfloat16, out_bytes=BF16_TILE_BYTES, left_stride=columns,
                   right_stride=columns, right_column=0, fp32_inputs=False)


def served_residual(operations, finished, hidden, retain):
    """The served residual tail (draft_mlp_branch / draft_attention_branch), op for op."""
    wide = [retain(operations.typecast(value, operations.float32)) for value in (finished, hidden)]
    summed = retain(operations.add(*wide, dtype=operations.float32))
    return retain(operations.typecast(summed, operations.bfloat16))


def served_halves(operations, gate_up, rows, retain):
    """A fused gate|up projection as the two tensors the served SwiGLU takes: two tile-aligned column slices (exact copies)."""
    width = tuple(gate_up.shape)[3] // 2
    return [retain(operations.slice(gate_up, (0, 0, 0, offset), (1, 1, rows, offset + width))) for offset in (0, width)]


def _engaged(site, rows, kind, tasks, workers):
    fusion.STATS['tail'] += 1
    fusion.note(fusion.TAIL_ENGAGED, 'kind=%s site=%s rows=%d tasks=%d workers=%d' % (kind, site, rows, tasks, workers))


def _audited(operations, mesh, kind, site, rows, mine, make_reference, **fields):
    """The eager audit: `make_reference(keep)` is the served composition on the same operands; compared on every chip, then freed."""
    operations.synchronize_device(mesh)
    held = []

    def keep(tensor):
        held.append(tensor)
        return tensor

    try:
        reference = make_reference(keep)
        operations.synchronize_device(mesh)
        bad = fusion.differing_chips(operations, mine, reference)
    finally:
        for tensor in reversed(held):
            operations.deallocate(tensor)
    fusion.audit_result(fusion.TAIL, fusion.TAIL_AUDIT_LINE, fusion.TAIL_MISMATCH, site, rows, bad, fusion.audited_total(fusion.TAIL),
                        kind=kind, **fields)


def swiglu(operations, mesh, gate, up, retain, *, served, site='mlp'):
    """draft_mlp.swiglu_device(operations, gate, up, retain) as one launch (or `served(operations, gate, up, retain)` when it cannot be)."""
    reason = swiglu_problem(operations, mesh, gate, up)
    if reason is not None:
        fusion.fell_back(fusion.TAIL_FALLBACK, site + '.swiglu', reason)
        return served(operations, gate, up, retain)
    rows, width = tuple(gate.shape)[2], tuple(gate.shape)[3]
    try:
        output = swiglu_launch(operations, mesh, gate, up, rows, width)
    except Exception as failure:  # noqa: BLE001 - the served ops can still run on the same operands
        fusion.fell_back(fusion.TAIL_FALLBACK, site + '.swiglu', 'the launch failed: %s: %s' % (type(failure).__name__, str(failure)[:120]))
        return served(operations, gate, up, retain)
    if fusion.audit_enabled(fusion.TAIL_AUDIT) and fusion.audit_due(fusion.TAIL, site + '.swiglu', rows):
        _audited(operations, mesh, 'swiglu', site, rows, output, lambda keep: served(operations, gate, up, keep))
    found = plan(rows, width // 32, fusion.grid_of(mesh))
    _engaged(site, rows, 'swiglu', found['tasks'], found['workers'])
    return retain(output)


def swiglu_fused(operations, mesh, gate_up, retain, *, served, site='mlp'):
    """The SwiGLU of a fused gate|up projection: one launch reading both halves in place, or (when it cannot be) the two column slices and
    `served(operations, gate, up, retain)`."""
    reason = swiglu_problem(operations, mesh, gate_up, None, fused=True)
    rows = tuple(gate_up.shape)[2]
    if reason is not None:
        fusion.fell_back(fusion.TAIL_FALLBACK, site + '.swiglu', reason)
        gate, up = served_halves(operations, gate_up, rows, retain)
        return served(operations, gate, up, retain)
    width = tuple(gate_up.shape)[3] // 2
    try:
        output = swiglu_fused_launch(operations, mesh, gate_up, rows, width)
    except Exception as failure:  # noqa: BLE001
        fusion.fell_back(fusion.TAIL_FALLBACK, site + '.swiglu', 'the launch failed: %s: %s' % (type(failure).__name__, str(failure)[:120]))
        gate, up = served_halves(operations, gate_up, rows, retain)
        return served(operations, gate, up, retain)
    if fusion.audit_enabled(fusion.TAIL_AUDIT) and fusion.audit_due(fusion.TAIL, site + '.swiglu', rows):
        def reference(keep):
            gate, up = served_halves(operations, gate_up, rows, keep)
            return served(operations, gate, up, keep)
        _audited(operations, mesh, 'swiglu', site, rows, output, reference, fused=1)
    found = plan(rows, width // 32, fusion.grid_of(mesh))
    _engaged(site, rows, 'swiglu', found['tasks'], found['workers'])
    return retain(output)


def residual(operations, mesh, finished, hidden, retain, *, site='mlp'):
    """The residual tail `typecast(add(typecast(finished), typecast(hidden)))` as one launch (or the served four ops when it cannot be)."""
    reason = residual_problem(operations, mesh, finished, hidden)
    if reason is not None:
        fusion.fell_back(fusion.TAIL_FALLBACK, site + '.residual', reason)
        return served_residual(operations, finished, hidden, retain)
    rows = tuple(finished.shape)[2]
    try:
        output = residual_launch(operations, mesh, finished, hidden, rows)
    except Exception as failure:  # noqa: BLE001
        fusion.fell_back(fusion.TAIL_FALLBACK, site + '.residual', 'the launch failed: %s: %s' % (type(failure).__name__, str(failure)[:120]))
        return served_residual(operations, finished, hidden, retain)
    if fusion.audit_enabled(fusion.TAIL_AUDIT) and fusion.audit_due(fusion.TAIL, site + '.residual', rows):
        _audited(operations, mesh, 'residual', site, rows, output, lambda keep: served_residual(operations, finished, hidden, keep))
    found = plan(rows, HIDDEN // 32, fusion.grid_of(mesh))
    _engaged(site, rows, 'residual', found['tasks'], found['workers'])
    return retain(output)
