"""Experimental single-dispatch FP32 row sum; simulator gate required before hardware."""

from pathlib import Path


def row_sum(mesh, value, owned):
    import ttnn

    shape = tuple(value.shape)
    if (len(shape) != 4 or shape[:3] != (1, 16, 32) or shape[3] % 32
            or not 32 <= shape[3] <= 2080 or value.dtype != ttnn.float32
            or value.layout != ttnn.TILE_LAYOUT or value.memory_config() != ttnn.DRAM_MEMORY_CONFIG):
        raise ValueError('Interleaved FP32 draft head rows with padded width32..2080 required')
    output = ttnn.empty((1, 16, 32, 1), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT,
        device=mesh, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    owned.append(output)
    cores = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 1))])
    buffers = [ttnn.CBDescriptor(total_size=4096 * count, core_ranges=cores,
        format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=index, data_format=ttnn.float32,
            page_size=4096, tile=ttnn.TileDescriptor(ttnn.Tile([32, 32])))]) for index, count in ((0, 2), (16, 1))]
    config = ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi4,
        fp32_dest_acc_en=True, math_approx_mode=False)
    modes = [ttnn.UnpackToDestMode.Default] * 64
    modes[0] = ttnn.UnpackToDestMode.UnpackToDestFp32
    config.unpack_to_dest_mode.extend(modes)
    compute = ttnn.KernelDescriptor(kernel_source=str(Path(__file__).with_name('draft_row_sum_compute.cpp')),
        core_ranges=cores, compile_time_args=[shape[3] // 32], config=config)
    program = ttnn.MeshProgramDescriptor()
    for chip, (local_input, local_output) in enumerate(zip(ttnn.get_device_tensors(value), ttnn.get_device_tensors(output), strict=True)):
        runtime = ttnn.RuntimeArgs()
        for task in range(16):
            runtime[task % 8][task // 8] = [local_input.buffer_address(), local_output.buffer_address(), task, shape[3] // 32]
        reader = ttnn.KernelDescriptor(kernel_source=str(Path(__file__).with_name('draft_row_sum_io.cpp')),
            core_ranges=cores, compile_time_args=ttnn.TensorAccessorArgs(local_input).get_compile_time_args()
                + ttnn.TensorAccessorArgs(local_output).get_compile_time_args(), runtime_args=runtime,
            config=ttnn.DataMovementConfigDescriptor(processor=ttnn.DataMovementProcessor.RISCV_0,
                noc=ttnn.NOC.RISCV_0_default))
        coordinate = ttnn.MeshCoordinate(0, chip)
        program[ttnn.MeshCoordinateRange(coordinate, coordinate)] = ttnn.ProgramDescriptor(kernels=[reader, compute], cbs=buffers)
    ttnn.generic_op([value, output], program)
    return output
