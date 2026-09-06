"""Restore any post-verification convolution prefix from four packed window tensors."""

from pathlib import Path


def validate_prefix(source_shapes, destination_shapes, prefix):
    rows = source_shapes[0][1] if len(source_shapes) == 4 and len(source_shapes[0]) == 3 else 0
    if type(rows) is not int or rows not in (1, 2, 4, 8, 16, 32):
        raise ValueError('Supported packed convolution width required')
    if any(tuple(shape) != (1, rows, 5120) for shape in source_shapes):
        raise ValueError('Four identically shaped packed convolution states required')
    if len(destination_shapes) != 4 or any(tuple(shape) != (1, 1, 5120) for shape in destination_shapes):
        raise ValueError('Four compact convolution destinations required')
    if type(prefix) is not int or not 1 <= prefix <= rows:
        raise ValueError('Nonzero prefix within packed windows required')
    return rows


def copy_prefix(mesh, source, destination, prefix):
    import ttnn

    validate_prefix([tuple(value.shape) for value in source], [tuple(value.shape) for value in destination], prefix)
    tensors = [*source, *destination]
    if any(value.dtype != ttnn.bfloat16 or value.layout != ttnn.TILE_LAYOUT or
           value.memory_config() != ttnn.DRAM_MEMORY_CONFIG for value in tensors):
        raise ValueError('Interleaved DRAM BF16 tiles required')
    shards = [ttnn.get_device_tensors(value) for value in tensors]
    if any(len(parts) != 2 for parts in shards):
        raise ValueError('Both chips required')
    grid = mesh.compute_with_storage_grid_size()
    if grid.x < 8 or grid.y < 6:
        raise ValueError('Prefix DMA requires the audited 8x6 worker grid')
    cores = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 5))])
    buffer = ttnn.CBDescriptor(total_size=4096, core_ranges=cores,
        format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=0, data_format=ttnn.bfloat16,
            page_size=2048, tile=ttnn.TileDescriptor(ttnn.Tile([32, 32])))])
    program = ttnn.MeshProgramDescriptor()
    for chip in range(2):
        local = [parts[chip] for parts in shards]
        addresses = [value.buffer_address() for value in local]
        if len(set(addresses)) != len(addresses):
            raise ValueError('Packed snapshots and restore destinations must not alias')
        descriptor = ttnn.KernelDescriptor(kernel_source=str(Path(__file__).with_suffix('.cpp')),
            core_ranges=cores,
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
