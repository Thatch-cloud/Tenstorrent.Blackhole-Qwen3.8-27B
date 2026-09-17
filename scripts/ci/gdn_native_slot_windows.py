"""Experimental T16 windows reading row zero directly from native B8 history."""

from pathlib import Path


def build_windows(mesh, projected, history, *, experimental=False):
    import ttnn
    if experimental is not True:
        raise ValueError('Native history windows require an explicit experiment')
    validate_geometry(tuple(projected.shape), [tuple(value.shape) for value in history])
    rows = 16
    if tuple(mesh.shape) != (1, 2):
        raise ValueError('Two-chip native-history mesh required')
    inputs = [projected, *history]
    if any(value.dtype != ttnn.bfloat16 or value.layout != ttnn.TILE_LAYOUT or
           value.memory_config() not in (ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG) for value in inputs):
        raise ValueError('Interleaved BF16 tiled inputs required')
    grid = mesh.compute_with_storage_grid_size()
    if grid.x < 8 or grid.y < 6:
        raise ValueError('Window DMA requires the audited 8x6 worker grid')
    outputs = []
    try:
        for slot in range(4):
            outputs.append(ttnn.empty((1, rows, 5120), device=mesh, dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG))
        tensors = inputs + outputs
        shards = [ttnn.get_device_tensors(value) for value in tensors]
        if any(len(parts) != 2 for parts in shards):
            raise ValueError('Both chips required')
        cores = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 5))])
        buffer = ttnn.CBDescriptor(total_size=6 * 2048, core_ranges=cores,
            format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=0, data_format=ttnn.bfloat16,
                page_size=2048, tile=ttnn.TileDescriptor(ttnn.Tile([32, 32])))])
        program = ttnn.MeshProgramDescriptor()
        for chip in range(2):
            local = [parts[chip] for parts in shards]
            addresses = [value.buffer_address() for value in local]
            if len(set(addresses)) != len(addresses):
                raise ValueError('Immutable input and mutable windows must not alias')
            descriptor = ttnn.KernelDescriptor(kernel_source=str(Path(__file__).with_name('gdn_conv_windows.cpp')),
                core_ranges=cores,
                compile_time_args=[argument for value in local for argument in ttnn.TensorAccessorArgs(value).get_compile_time_args()],
                config=ttnn.DataMovementConfigDescriptor(processor=ttnn.DataMovementProcessor.RISCV_0,
                                                         noc=ttnn.NOC.RISCV_0_default))
            runtime = ttnn.RuntimeArgs()
            for worker in range(48):
                runtime[worker % 8][worker // 8] = addresses + [rows, worker]
            descriptor.runtime_args = runtime
            coordinate = ttnn.MeshCoordinate(0, chip)
            program[ttnn.MeshCoordinateRange(coordinate, coordinate)] = ttnn.ProgramDescriptor(kernels=[descriptor], cbs=[buffer])
        ttnn.generic_op(tensors, program)
        return outputs
    except BaseException:
        for value in outputs:
            ttnn.deallocate(value)
        raise



def validate_geometry(projected, history):
    if tuple(projected) not in ((1, 16, 8240), (1, 16, 8256)):
        raise ValueError('T16 projected TP2 geometry required')
    if len(history) != 4 or any(tuple(shape) != (1, 8, 5120) for shape in history):
        raise ValueError('Four full native B8 convolution histories required')
