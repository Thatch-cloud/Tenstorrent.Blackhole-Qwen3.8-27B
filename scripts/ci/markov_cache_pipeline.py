"""Experimental device-owned bias cache plumbing; caller serializes ownership."""

from pathlib import Path


def build(mesh, state, decision, bias, cache, output, mask):
    import ttnn

    width = int(bias.shape[-1])
    if width not in (64, 248320):
        raise ValueError('Explicit Markov vocabulary geometry required')
    expected = ((state, (1, 1, 65, 8), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT),
        (decision, (1, 1, 1, 8), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT),
        (bias, (1, 1, 1, width), ttnn.float32, ttnn.TILE_LAYOUT),
        (cache, (1, 1, 64, width), ttnn.float32, ttnn.ROW_MAJOR_LAYOUT),
        (output, (1, 1, 1, width), ttnn.float32, ttnn.TILE_LAYOUT),
        (mask, (1, 1, 1, 1), ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT))
    for tensor, shape, dtype, layout in expected:
        if (tuple(tensor.shape) != shape or tensor.dtype != dtype or tensor.layout != layout
                or tensor.memory_config() != ttnn.DRAM_MEMORY_CONFIG):
            raise ValueError('Explicit owned DRAM cache operand required')
    parts = [ttnn.get_device_tensors(record[0]) for record in expected]
    if any(len(values) != 2 for values in parts):
        raise ValueError('Two chip-local copies required')
    for chip in range(2):
        if len({values[chip].buffer_address() for values in parts}) != len(parts):
            raise ValueError('Cache operands must not alias')

    def program(filename, indices, workers):
        cores = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(workers - 1, 0))])
        scratch = ttnn.CBDescriptor(total_size=4096, core_ranges=cores,
            format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=0, data_format=ttnn.uint32,
                page_size=4096, tile=ttnn.TileDescriptor(ttnn.Tile([32, 32])))])
        result = ttnn.MeshProgramDescriptor()
        for chip in range(2):
            tensors = [parts[index][chip] for index in indices]
            kernel = ttnn.KernelDescriptor(kernel_source=str(Path(__file__).with_name(filename)), core_ranges=cores,
                compile_time_args=[argument for tensor in tensors
                    for argument in ttnn.TensorAccessorArgs(tensor).get_compile_time_args()],
                config=ttnn.DataMovementConfigDescriptor(processor=ttnn.DataMovementProcessor.RISCV_0,
                    noc=ttnn.NOC.RISCV_0_default))
            runtime = ttnn.RuntimeArgs()
            for worker in range(workers):
                runtime[worker][0] = [tensor.buffer_address() for tensor in tensors] + (
                    [width, worker, workers] if len(indices) == 5 else [])
            kernel.runtime_args = runtime
            coordinate = ttnn.MeshCoordinate(0, chip)
            result[ttnn.MeshCoordinateRange(coordinate, coordinate)] = ttnn.ProgramDescriptor(
                kernels=[kernel], cbs=[scratch])
        return result

    return program('markov_cache_mask.cpp', (1, 5), 1), program(
        'markov_cache_payload.cpp', (0, 1, 2, 3, 4), 2 if width == 64 else 10)
