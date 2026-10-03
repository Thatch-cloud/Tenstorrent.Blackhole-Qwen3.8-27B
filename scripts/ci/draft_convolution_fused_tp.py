"""draft_convolution_fused at any served width.

The pair's module is pinned (recorded evidence hashes its bytes; test_tp2_pins) and stays as it was;
tp_addresses rebinds the names below to these at four cards only. Each is the pair's function with its literal chip and
head counts read from tp_shapes; at two chips it would be call for call the pinned one."""

from pathlib import Path
from draft_convolution import validate_boundaries, validate_shapes
from draft_convolution_fused import seam_mask
import tp_shapes


def fused_convolution(operations, mesh, hidden, dynamic, base, *, boundaries=None):
    """The fused convolution: the served call, or with QWEN_FAST_TP4_DRAFT_CONV=1 (D2a, tp4_draft_conv) the same convolution on a
    rewritten I/O stage that hands the compute kernel byte-identical tiles. Flag off, this is served_fused_convolution."""
    import tp4_sampdraft

    if not tp4_sampdraft.enabled(tp4_sampdraft.DRAFT_CONV):
        return served_fused_convolution(operations, mesh, hidden, dynamic, base, boundaries=boundaries)
    import tp4_draft_conv

    rows = validate_shapes(hidden, dynamic, base)
    coordinates = [(worker % 8, worker // 8) for worker in range(80)]
    return tp4_draft_conv.convolution(
        operations, mesh, hidden, dynamic, base, rows=rows, seams_low=seam_mask(boundaries, rows), seams_high=0, workers=80,
        coordinates=coordinates, label='pair',
        core_set=lambda group: _pair_cores(operations, group),
        served=lambda: served_fused_convolution(operations, mesh, hidden, dynamic, base, boundaries=boundaries))


def _pair_cores(operations, group):
    """The pair's one 8 x 10 core range, whatever group is asked for: the pair's 160 pages over 80 workers are one page-count group (two
    pages each), so a group that is not all 80 workers would be a plan this range cannot express."""
    if len(group) != 80:
        raise ValueError('The pair conv runs one group of 80 workers on its 8 x 10 range, not %d' % len(group))
    return operations.CoreRangeSet([operations.CoreRange(operations.CoreCoord(0, 0), operations.CoreCoord(7, 9))])


def served_fused_convolution(operations, mesh, hidden, dynamic, base, *, boundaries=None):
    rows = validate_shapes(hidden, dynamic, base)
    seams = seam_mask(boundaries, rows)
    tensors = [hidden, *dynamic, *base]
    chips = tp_shapes.chip_count()
    if list(mesh.shape) != [1, chips] or any(value.dtype != operations.bfloat16
            or value.layout != operations.TILE_LAYOUT or value.memory_config() != operations.DRAM_MEMORY_CONFIG
            for value in tensors):
        raise ValueError('Two-chip interleaved DRAM BF16 convolution operands required')
    parts = [operations.get_device_tensors(value) for value in tensors]
    if any(len(shards) != chips for shards in parts):
        raise ValueError('%s operand shards required' % tp_shapes.all_chips())
    output = operations.empty(tuple(hidden.shape), dtype=operations.bfloat16, layout=operations.TILE_LAYOUT,
        device=mesh, memory_config=operations.DRAM_MEMORY_CONFIG)
    tensors.append(output)
    parts.append(operations.get_device_tensors(output))
    cores = operations.CoreRangeSet([operations.CoreRange(operations.CoreCoord(0, 0), operations.CoreCoord(7, 9))])
    buffers = [operations.CBDescriptor(total_size=2048 * count, core_ranges=cores,
        format_descriptors=[operations.CBFormatDescriptor(buffer_index=index, data_format=operations.bfloat16,
            page_size=2048, tile=operations.TileDescriptor(operations.Tile([32, 32])))])
        for index, count in ((0, 7), (1, 2), (16, 1))]
    compute = operations.KernelDescriptor(kernel_source=str(Path(__file__).with_name('draft_convolution_fused_compute.cpp')),
        core_ranges=cores, compile_time_args=[2], config=operations.ComputeConfigDescriptor(
            math_fidelity=operations.MathFidelity.HiFi4, fp32_dest_acc_en=True, math_approx_mode=False))
    program = operations.MeshProgramDescriptor()
    try:
        for chip in range(chips):
            local = [shards[chip] for shards in parts]
            if local[-1].buffer_address() in {value.buffer_address() for value in local[:-1]}:
                raise ValueError('Convolution output must not alias borrowed inputs')
            runtime = operations.RuntimeArgs()
            for worker in range(80):
                runtime[worker % 8][worker // 8] = [value.buffer_address() for value in local] + [rows, worker, seams]
            reader = operations.KernelDescriptor(kernel_source=str(Path(__file__).with_name('draft_convolution_fused_io.cpp')),
                core_ranges=cores, compile_time_args=[argument for value in local
                    for argument in operations.TensorAccessorArgs(value).get_compile_time_args()], runtime_args=runtime,
                config=operations.DataMovementConfigDescriptor(processor=operations.DataMovementProcessor.RISCV_0,
                    noc=operations.NOC.RISCV_0_default))
            coordinate = operations.MeshCoordinate(0, chip)
            program[operations.MeshCoordinateRange(coordinate, coordinate)] = operations.ProgramDescriptor(
                kernels=[reader, compute], cbs=buffers)
        operations.generic_op(tensors, program)
    except BaseException:
        operations.deallocate(output)
        raise
    return output
