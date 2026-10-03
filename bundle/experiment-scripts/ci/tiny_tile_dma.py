"""Exact T8 BF16 face copies between preallocated 16-row and 32-row tiles."""

from pathlib import Path


def validate_layout(source_shape, destination_shape, source_tile, destination_tile):
    if (tuple(source_shape) != tuple(destination_shape) or len(source_shape) != 4
            or tuple(source_shape)[:3] != (1, 1, 8) or source_shape[-1] not in (5120, 8704)
            or (tuple(source_tile), tuple(destination_tile)) not in (
                ((16, 32), (32, 32)), ((32, 32), (16, 32)))):
        raise ValueError('Matching T8 MLP shapes and opposite 16/32-row native tiles required')
    return source_shape[-1] // 32


def copy_live_rows(mesh, source, destination):
    import ttnn

    pages = validate_layout(tuple(source.shape), tuple(destination.shape), source.tile.tile_shape, destination.tile.tile_shape)
    tensors = [source, destination]
    if any(value.dtype != ttnn.bfloat16 or value.layout != ttnn.TILE_LAYOUT
            or value.memory_config() != ttnn.L1_MEMORY_CONFIG
            or value.tile.transpose_of_faces or value.tile.transpose_within_face for value in tensors):
        raise ValueError('Non-transposed interleaved L1 BF16 tiles required')
    shards = [ttnn.get_device_tensors(value) for value in tensors]
    if any(len(parts) != 2 for parts in shards):
        raise ValueError('Both chips required')
    grid = mesh.compute_with_storage_grid_size()
    if (grid.x, grid.y) != (11, 10):
        raise ValueError('Qualified worker-dispatch grid required')
    cores = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(10, 3))])
    buffer = ttnn.CBDescriptor(total_size=2048, core_ranges=cores,
        format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=0, data_format=ttnn.bfloat16,
            page_size=2048, tile=ttnn.TileDescriptor(ttnn.Tile((32, 32))))])
    program = ttnn.MeshProgramDescriptor()
    for chip in range(2):
        local_source, local_destination = (parts[chip] for parts in shards)
        source_address, destination_address = local_source.buffer_address(), local_destination.buffer_address()
        if source_address == destination_address:
            raise ValueError('DMA source and destination must not alias')
        descriptor = ttnn.KernelDescriptor(kernel_source=str(Path(__file__).with_suffix('.cpp')), core_ranges=cores,
            compile_time_args=[argument for value in (local_source, local_destination)
                for argument in ttnn.TensorAccessorArgs(value).get_compile_time_args()],
            config=ttnn.DataMovementConfigDescriptor(processor=ttnn.DataMovementProcessor.RISCV_0,
                noc=ttnn.NOC.RISCV_0_default))
        runtime = ttnn.RuntimeArgs()
        for worker in range(44):
            runtime[worker % 11][worker // 11] = [source_address, destination_address,
                source.tile.tile_shape[0] * 64, destination.tile.tile_shape[0] * 64, pages, worker]
        descriptor.runtime_args = runtime
        coordinate = ttnn.MeshCoordinate(0, chip)
        program[ttnn.MeshCoordinateRange(coordinate, coordinate)] = ttnn.ProgramDescriptor(kernels=[descriptor], cbs=[buffer])
    ttnn.generic_op(tensors, program)
    return destination
