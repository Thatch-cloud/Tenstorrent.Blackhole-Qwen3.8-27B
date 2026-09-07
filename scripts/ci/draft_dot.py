"""Bounded-memory SFPU batched dot-product experiment; no production speed claim."""

from pathlib import Path


def dot_geometry(left_shape, right_shape, worker_limit=64):
    if type(worker_limit) is not int or worker_limit not in (64, 80, 110):
        raise ValueError('Worker limit must be64,80 or110')
    if (len(left_shape) != 4 or len(right_shape) != 4 or tuple(left_shape[:3]) != (1, 16, 32)
            or tuple(right_shape[:2]) != (1, 16) or left_shape[3] != right_shape[3]
            or any(type(value) is not int or value % 32 or not 32 <= value <= 2080
                for value in (left_shape[3], right_shape[2]))):
        raise ValueError('TP2 head-local FP32 dot geometry with tile-aligned width/keys32..2080 required')
    tasks = 16 * (right_shape[2] // 32)
    return min(worker_limit, tasks), right_shape[2] // 32, left_shape[3] // 32


def dot_core_layout(workers):
    if type(workers) is not int or not 16 <= workers <= 110:
        raise ValueError('Bounded active worker count required')
    columns = 11 if workers > 80 else 8
    full_rows, remainder = divmod(workers, columns)
    return columns, full_rows, remainder


def dot_buffer_tiles(width_tiles, cache_tiles):
    if type(cache_tiles) is not bool or type(width_tiles) is not int or not 1 <= width_tiles <= 65:
        raise ValueError('Explicit cache policy and bounded reduction width required')
    return ((0, width_tiles if cache_tiles else 2), (1, 2), (2, width_tiles + 1 if cache_tiles else 2), (16, 1))


def fused_dot(mesh, left, right, owned, *, cache_tiles=False, worker_limit=64):
    import ttnn

    workers, key_tiles, width_tiles = dot_geometry(tuple(left.shape), tuple(right.shape), worker_limit)
    columns, full_rows, remainder = dot_core_layout(workers)
    grid = mesh.compute_with_storage_grid_size()
    if grid.x < columns or grid.y < full_rows + bool(remainder):
        raise ValueError('Requested dot workers exceed the available compute grid')
    buffer_tiles = dot_buffer_tiles(width_tiles, cache_tiles)
    for tensor in (left, right):
        if tensor.dtype != ttnn.float32 or tensor.layout != ttnn.TILE_LAYOUT or tensor.memory_config() != ttnn.DRAM_MEMORY_CONFIG:
            raise ValueError('Interleaved FP32 operands required')
    output = ttnn.empty((1, 16, 32, right.shape[2]), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT,
        device=mesh, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    owned.append(output)
    ranges = [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(columns - 1, full_rows - 1))]
    if remainder:
        ranges.append(ttnn.CoreRange(ttnn.CoreCoord(0, full_rows), ttnn.CoreCoord(remainder - 1, full_rows)))
    cores = ttnn.CoreRangeSet(ranges)
    buffers = [ttnn.CBDescriptor(total_size=4096 * count, core_ranges=cores,
        format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=index, data_format=ttnn.float32,
            page_size=4096, tile=ttnn.TileDescriptor(ttnn.Tile([32, 32])))])
        for index, count in buffer_tiles]
    config = ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi4,
        fp32_dest_acc_en=True, math_approx_mode=False)
    modes = [ttnn.UnpackToDestMode.Default] * 64
    modes[0] = modes[1] = ttnn.UnpackToDestMode.UnpackToDestFp32
    config.unpack_to_dest_mode.extend(modes)
    program = ttnn.MeshProgramDescriptor()
    for chip, shards in enumerate(zip(*(ttnn.get_device_tensors(tensor) for tensor in (left, right, output)), strict=True)):
        runtime = ttnn.RuntimeArgs()
        for worker in range(workers):
            runtime[worker % columns][worker // columns] = [tensor.buffer_address() for tensor in shards] + [worker, workers, key_tiles, width_tiles, int(cache_tiles)]
        reader = ttnn.KernelDescriptor(kernel_source=str(Path(__file__).with_name('draft_dot_io.cpp')), core_ranges=cores,
            compile_time_args=[argument for tensor in shards for argument in ttnn.TensorAccessorArgs(tensor).get_compile_time_args()],
            runtime_args=runtime, config=ttnn.DataMovementConfigDescriptor(processor=ttnn.DataMovementProcessor.RISCV_0,
                noc=ttnn.NOC.RISCV_0_default))
        compute = ttnn.KernelDescriptor(kernel_source=str(Path(__file__).with_name('draft_dot_compute.cpp')),
            core_ranges=cores, runtime_args=runtime, config=config)
        coordinate = ttnn.MeshCoordinate(0, chip)
        program[ttnn.MeshCoordinateRange(coordinate, coordinate)] = ttnn.ProgramDescriptor(kernels=[reader, compute], cbs=buffers)
    ttnn.generic_op([left, right, output], program)
    return output
