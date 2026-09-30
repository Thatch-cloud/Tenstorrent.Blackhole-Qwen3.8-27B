"""Restore any post-verification convolution prefix from four packed window tensors."""

from pathlib import Path

import tp_kernels
import tp_shapes


def validate_prefix(source_shapes, destination_shapes, prefix):
    rows = source_shapes[0][1] if len(source_shapes) == 4 and len(source_shapes[0]) == 3 else 0
    if type(rows) is not int or rows not in (1, 2, 4, 8, 16, 32):
        raise ValueError('Supported packed convolution width required')
    if any(tuple(shape) != (1, rows, tp_shapes.active().gdn_qkv) for shape in source_shapes):
        raise ValueError('Four identically shaped packed convolution states required')
    if len(destination_shapes) != 4 or any(tuple(shape) != (1, 1, tp_shapes.active().gdn_qkv) for shape in destination_shapes):
        raise ValueError('Four compact convolution destinations required')
    if type(prefix) is not int or not 1 <= prefix <= rows:
        raise ValueError('Nonzero prefix within packed windows required')
    return rows


def copy_prefix(mesh, source, destination, prefix, *, reuse_zero_tile=False):
    import ttnn

    if type(reuse_zero_tile) is not bool:
        raise ValueError('Explicit zero-tile reuse policy required')
    validate_prefix([tuple(value.shape) for value in source], [tuple(value.shape) for value in destination], prefix)
    tensors = [*source, *destination]
    if any(value.dtype != ttnn.bfloat16 or value.layout != ttnn.TILE_LAYOUT or
           value.memory_config() != ttnn.DRAM_MEMORY_CONFIG for value in tensors):
        raise ValueError('Interleaved DRAM BF16 tiles required')
    shards = [ttnn.get_device_tensors(value) for value in tensors]
    chips = tp_shapes.chip_count()
    if any(len(parts) != chips for parts in shards):
        raise ValueError('%s chips required' % tp_shapes.all_chips())
    grid = mesh.compute_with_storage_grid_size()
    if grid.x < 8 or grid.y < 6:
        raise ValueError('Prefix DMA requires the audited 8x6 worker grid')
    cores = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 5))])
    buffer = ttnn.CBDescriptor(total_size=4096, core_ranges=cores,
        format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=0, data_format=ttnn.bfloat16,
            page_size=2048, tile=ttnn.TileDescriptor(ttnn.Tile([32, 32])))])
    program = ttnn.MeshProgramDescriptor()
    for chip in range(chips):
        local = [parts[chip] for parts in shards]
        addresses = [value.buffer_address() for value in local]
        if len(set(addresses)) != len(addresses):
            raise ValueError('Packed snapshots and restore destinations must not alias')
        descriptor = ttnn.KernelDescriptor(kernel_source=tp_kernels.source(Path(__file__).with_suffix('.cpp')),
            core_ranges=cores,
            defines=([('QWEN_PREFIX_REUSE_ZERO_TILE', '1')] if reuse_zero_tile else []) + tp_kernels.defines(),
            compile_time_args=[argument for index in (0, 4) for argument in ttnn.TensorAccessorArgs(local[index]).get_compile_time_args()],
            config=ttnn.DataMovementConfigDescriptor(processor=ttnn.DataMovementProcessor.RISCV_0,
                                                     noc=ttnn.NOC.RISCV_0_default))
        runtime = ttnn.RuntimeArgs()
        for worker in range(48):
            runtime[worker % 8][worker // 8] = addresses + [prefix, worker]
        descriptor.runtime_args = runtime
        coordinate = ttnn.MeshCoordinate(0, chip)
        program[ttnn.MeshCoordinateRange(coordinate, coordinate)] = ttnn.ProgramDescriptor(kernels=[descriptor], cbs=[buffer])
    ttnn.generic_op(tensors, program)
