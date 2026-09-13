"""Metadata-only Markov cache controller; no cached bias or conditional matmul."""

from pathlib import Path


def build(mesh, state, request, decision, status):
    import ttnn

    tensors = (state, request, decision, status)
    shapes = ((1, 1, 65, 8),) + ((1, 1, 1, 8),) * 3
    for tensor, shape in zip(tensors, shapes, strict=True):
        if (tuple(tensor.shape) != shape or tensor.dtype != ttnn.uint32
                or tensor.layout != ttnn.ROW_MAJOR_LAYOUT
                or tensor.memory_config() != ttnn.DRAM_MEMORY_CONFIG):
            raise ValueError('Owned uint32 row-major DRAM metadata with 32-byte pages required')
    parts = [ttnn.get_device_tensors(tensor) for tensor in tensors]
    if any(len(values) != 2 for values in parts):
        raise ValueError('Two chip-local copies required')
    cores = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))])
    scratch = ttnn.CBDescriptor(total_size=4096, core_ranges=cores,
        format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=0, data_format=ttnn.uint32,
            page_size=4096, tile=ttnn.TileDescriptor(ttnn.Tile([32, 32])))])
    program = ttnn.MeshProgramDescriptor()
    for chip in range(2):
        local = [values[chip] for values in parts]
        addresses = [tensor.buffer_address() for tensor in local]
        if len(set(addresses)) != 4:
            raise ValueError('Controller buffers must not alias')
        kernel = ttnn.KernelDescriptor(
            kernel_source=str(Path(__file__).with_name('markov_cache_control.cpp')), core_ranges=cores,
            compile_time_args=[argument for tensor in local
                for argument in ttnn.TensorAccessorArgs(tensor).get_compile_time_args()],
            config=ttnn.DataMovementConfigDescriptor(processor=ttnn.DataMovementProcessor.RISCV_0,
                noc=ttnn.NOC.RISCV_0_default))
        runtime = ttnn.RuntimeArgs()
        runtime[0][0] = addresses
        kernel.runtime_args = runtime
        coordinate = ttnn.MeshCoordinate(0, chip)
        program[ttnn.MeshCoordinateRange(coordinate, coordinate)] = ttnn.ProgramDescriptor(
            kernels=[kernel], cbs=[scratch])
    return program
