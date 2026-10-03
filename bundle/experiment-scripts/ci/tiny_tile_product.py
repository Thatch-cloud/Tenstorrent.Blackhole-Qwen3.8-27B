"""Native BF16 SFPU multiplication with explicit 16-row tile descriptors."""

from pathlib import Path


def validate_layout(shapes, tiles):
    if (len(shapes) != 3 or len(tiles) != 3
            or any(tuple(shape) != (1, 1, 8, 8704) for shape in shapes)
            or any(tuple(tile) != (16, 32) for tile in tiles)):
        raise ValueError('Three matching T8 MLP hidden tensors with 16-row tiles required')
    return 272


def multiply(mesh, gate, up, destination):
    import ttnn

    tensors = [gate, up, destination]
    pages = validate_layout([tuple(value.shape) for value in tensors], [value.tile.tile_shape for value in tensors])
    if any(value.dtype != ttnn.bfloat16 or value.layout != ttnn.TILE_LAYOUT
            or value.memory_config() != ttnn.L1_MEMORY_CONFIG
            or value.tile.transpose_of_faces or value.tile.transpose_within_face for value in tensors):
        raise ValueError('Non-transposed interleaved L1 BF16 tiles required')
    shards = [ttnn.get_device_tensors(value) for value in tensors]
    grid = mesh.compute_with_storage_grid_size()
    if any(len(parts) != 2 for parts in shards) or (grid.x, grid.y) != (11, 10):
        raise ValueError('Two chips on the qualified worker-dispatch grid required')
    cores = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(10, 3))])
    buffers = [ttnn.CBDescriptor(total_size=2048, core_ranges=cores,
        format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=index, data_format=ttnn.bfloat16,
            page_size=1024, tile=ttnn.TileDescriptor(ttnn.Tile((16, 32))))]) for index in (0, 1, 16)]
    compute_runtime = ttnn.RuntimeArgs()
    for worker in range(44):
        compute_runtime[worker % 11][worker // 11] = [(pages + 43 - worker) // 44]
    config = ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi4,
        fp32_dest_acc_en=False, math_approx_mode=False)
    modes = [ttnn.UnpackToDestMode.Default] * 64
    modes[0] = modes[1] = ttnn.UnpackToDestMode.UnpackToDestFp32
    config.unpack_to_dest_mode.extend(modes)
    compute = ttnn.KernelDescriptor(kernel_source=str(Path(__file__).with_name('tiny_tile_product_compute.cpp')),
        core_ranges=cores, runtime_args=compute_runtime, config=config)
    program = ttnn.MeshProgramDescriptor()
    for chip in range(2):
        local = [parts[chip] for parts in shards]
        addresses = [value.buffer_address() for value in local]
        if len(set(addresses)) != 3:
            raise ValueError('Product inputs and destination must not alias')
        runtime = ttnn.RuntimeArgs()
        for worker in range(44):
            runtime[worker % 11][worker // 11] = addresses + [pages, worker]
        reader = ttnn.KernelDescriptor(kernel_source=str(Path(__file__).with_name('tiny_tile_product_io.cpp')),
            core_ranges=cores, runtime_args=runtime,
            compile_time_args=[argument for value in local for argument in ttnn.TensorAccessorArgs(value).get_compile_time_args()],
            config=ttnn.DataMovementConfigDescriptor(processor=ttnn.DataMovementProcessor.RISCV_0,
                noc=ttnn.NOC.RISCV_0_default))
        coordinate = ttnn.MeshCoordinate(0, chip)
        program[ttnn.MeshCoordinateRange(coordinate, coordinate)] = ttnn.ProgramDescriptor(kernels=[reader, compute], cbs=buffers)
    ttnn.generic_op(tensors, program)
    return destination
