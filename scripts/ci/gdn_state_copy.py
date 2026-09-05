"""Copy slot zero between interleaved BF16 state and compact snapshots, without arithmetic."""

from pathlib import Path


def page_counts(shapes):
    if len(shapes) != 5 or tuple(shapes[0]) != (1, 24, 128, 128):
        raise ValueError("Expected one TP2 slot of recurrent state and four conv taps")
    if any(tuple(shape) != (1, 1, 5120) for shape in shapes[1:]):
        raise ValueError("Expected frozen TP2 convolution channel shape")
    return [384, 160, 160, 160, 160]


def copy_active(source, destination):
    import ttnn

    if len(source) != 5 or len(destination) != 5:
        raise ValueError("Expected complete state lists")
    source_shapes, destination_shapes = [tuple(tensor.shape) for tensor in source], [tuple(tensor.shape) for tensor in destination]
    compact = source if source_shapes[0][0] == 1 else destination
    counts = page_counts([tuple(tensor.shape) for tensor in compact])
    full_shapes = [(8, 24, 128, 128)] + [(1, 8, 5120)] * 4
    if source_shapes != full_shapes and destination_shapes != full_shapes:
        raise ValueError("One side must be the frozen eight-slot state")
    tensors = [tensor for pair in zip(source, destination, strict=True) for tensor in pair]
    if any(tensor.dtype != ttnn.bfloat16 or tensor.layout != ttnn.TILE_LAYOUT
           or tensor.memory_config() != ttnn.DRAM_MEMORY_CONFIG for tensor in tensors):
        raise ValueError("Only interleaved DRAM BF16 tiles supported")
    shards = [ttnn.get_device_tensors(tensor) for tensor in tensors]
    if any(len(local) != 2 for local in shards):
        raise ValueError("Both chips required")
    cores = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 5))])
    buffer = ttnn.CBDescriptor(total_size=2048, core_ranges=cores,
        format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=0, data_format=ttnn.bfloat16,
            page_size=2048, tile=ttnn.TileDescriptor(ttnn.Tile([32, 32])))])
    mesh_program = ttnn.MeshProgramDescriptor()
    for chip in range(2):
        local = [parts[chip] for parts in shards]
        if len({tensor.buffer_address() for tensor in local}) != 10:
            raise ValueError("State-copy buffers must not alias")
        compile_args = [value for tensor in local for value in ttnn.TensorAccessorArgs(tensor).get_compile_time_args()]
        runtime = []
        for index, count in enumerate(counts):
            runtime.extend([local[2 * index].buffer_address(), local[2 * index + 1].buffer_address(), count])
        kernel = ttnn.KernelDescriptor(kernel_source=str(Path(__file__).with_suffix(".cpp")),
            core_ranges=cores, compile_time_args=compile_args,
            config=ttnn.DataMovementConfigDescriptor(processor=ttnn.DataMovementProcessor.RISCV_0,
                                                     noc=ttnn.NOC.RISCV_0_default))
        arguments = ttnn.RuntimeArgs()
        for worker in range(48):
            arguments[worker % 8][worker // 8] = runtime + [worker]
        kernel.runtime_args = arguments
        coordinate = ttnn.MeshCoordinate(0, chip)
        mesh_program[ttnn.MeshCoordinateRange(coordinate, coordinate)] = ttnn.ProgramDescriptor(kernels=[kernel], cbs=[buffer])
    ttnn.generic_op(tensors, mesh_program)
