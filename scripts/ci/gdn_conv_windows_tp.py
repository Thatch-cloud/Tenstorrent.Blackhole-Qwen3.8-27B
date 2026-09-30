"""gdn_conv_windows.build_windows at any served width: one arithmetic-free DMA launch builds all four causal windows.

The pair's module is sha256-pinned (gdn_direct_window_report, the frozen combined evidence) and stays as it was;
tp_addresses rebinds its build_windows to this one at four cards. The launch is call for call the pair's, with the
qkv width, the chip count and the kernel source (the `_tp` sibling with page counts from defines) from tp_shapes.
"""

from pathlib import Path

import tp_kernels
import tp_shapes


def build_windows(mesh, projected, history):
    import ttnn
    from gdn_multitoken_conv import validate_projected

    rows = validate_projected(tuple(projected.shape), history)
    inputs = [projected, *history]
    if any(value.dtype != ttnn.bfloat16 or value.layout != ttnn.TILE_LAYOUT or
           value.memory_config() not in (ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG) for value in inputs):
        raise ValueError('Interleaved BF16 tiled inputs required')
    grid = mesh.compute_with_storage_grid_size()
    if grid.x < 8 or grid.y < 6:
        raise ValueError('Window DMA requires the audited 8x6 worker grid')
    outputs = []
    try:
        for slot in range(4):
            outputs.append(ttnn.empty((1, rows, tp_shapes.active().gdn_qkv), device=mesh, dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG))
        tensors = inputs + outputs
        shards = [ttnn.get_device_tensors(value) for value in tensors]
        chips = tp_shapes.chip_count()
        if any(len(parts) != chips for parts in shards):
            raise ValueError('%s chips required' % tp_shapes.all_chips())
        cores = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 5))])
        buffer = ttnn.CBDescriptor(total_size=6 * 2048, core_ranges=cores,
            format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=0, data_format=ttnn.bfloat16,
                page_size=2048, tile=ttnn.TileDescriptor(ttnn.Tile([32, 32])))])
        program = ttnn.MeshProgramDescriptor()
        for chip in range(chips):
            local = [parts[chip] for parts in shards]
            addresses = [value.buffer_address() for value in local]
            if len(set(addresses)) != len(addresses):
                raise ValueError('Immutable input and mutable windows must not alias')
            descriptor = ttnn.KernelDescriptor(kernel_source=tp_kernels.source(Path(__file__).with_name('gdn_conv_windows.cpp')),
                core_ranges=cores,
                defines=tp_kernels.defines(),
                compile_time_args=[argument for value in local for argument in ttnn.TensorAccessorArgs(value).get_compile_time_args()],
                config=ttnn.DataMovementConfigDescriptor(processor=ttnn.DataMovementProcessor.RISCV_0,
                                                         noc=ttnn.NOC.RISCV_0_default))
            runtime = ttnn.RuntimeArgs()
            for worker in range(48):
                runtime[worker % 8][worker // 8] = addresses + [rows, worker]
            descriptor.runtime_args = runtime
            coordinate = ttnn.MeshCoordinate(0, chip)
            program[ttnn.MeshCoordinateRange(coordinate, coordinate)] = ttnn.ProgramDescriptor(kernels=[descriptor], cbs=[buffer])
        ttnn.generic_op(tensors, program)
        return outputs
    except BaseException:
        for value in outputs:
            ttnn.deallocate(value)
        raise
