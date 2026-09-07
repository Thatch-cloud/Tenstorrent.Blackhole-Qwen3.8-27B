"""Bounded-memory SFPU batched dot-product experiment; no production speed claim."""

from pathlib import Path


def dot_geometry(left_shape, right_shape):
    if (len(left_shape) != 4 or len(right_shape) != 4 or tuple(left_shape[:3]) != (1, 16, 32)
            or tuple(right_shape[:2]) != (1, 16) or left_shape[3] != right_shape[3]
            or any(type(value) is not int or value % 32 or not 32 <= value <= 2080
                for value in (left_shape[3], right_shape[2]))):
        raise ValueError('TP2 head-local FP32 dot geometry with tile-aligned width/keys32..2080 required')
    tasks = 16 * (right_shape[2] // 32)
    return min(64, tasks), right_shape[2] // 32, left_shape[3] // 32


def fused_dot(mesh, left, right, owned):
    import ttnn

    workers, key_tiles, width_tiles = dot_geometry(tuple(left.shape), tuple(right.shape))
    for tensor in (left, right):
        if tensor.dtype != ttnn.float32 or tensor.layout != ttnn.TILE_LAYOUT or tensor.memory_config() != ttnn.DRAM_MEMORY_CONFIG:
            raise ValueError('Interleaved FP32 operands required')
    output = ttnn.empty((1, 16, 32, right.shape[2]), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT,
        device=mesh, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    owned.append(output)
    cores = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, workers // 8 - 1))])
    buffers = [ttnn.CBDescriptor(total_size=4096 * count, core_ranges=cores,
        format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=index, data_format=ttnn.float32,
            page_size=4096, tile=ttnn.TileDescriptor(ttnn.Tile([32, 32])))])
        for index, count in ((0, 2), (1, 2), (2, 2), (16, 1))]
    config = ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi4,
        fp32_dest_acc_en=True, math_approx_mode=False)
    modes = [ttnn.UnpackToDestMode.Default] * 64
    modes[0] = modes[1] = ttnn.UnpackToDestMode.UnpackToDestFp32
    config.unpack_to_dest_mode.extend(modes)
    program = ttnn.MeshProgramDescriptor()
    for chip, shards in enumerate(zip(*(ttnn.get_device_tensors(tensor) for tensor in (left, right, output)), strict=True)):
        runtime = ttnn.RuntimeArgs()
        for worker in range(workers):
            runtime[worker % 8][worker // 8] = [tensor.buffer_address() for tensor in shards] + [worker, workers, key_tiles, width_tiles]
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
