"""Experimental direct BF16 tile permutation for bounded TP2 query groups."""

from pathlib import Path


def source_row(rows, output_index, *, inverse=False):
    if type(inverse) is not bool or type(rows) is not int or not 1 <= rows <= 8 or type(output_index) is not int or not 0 <= output_index < rows * 12:
        raise ValueError('One to eight complete query rows required')
    if inverse:
        token, head = divmod(output_index, 12)
        return (head // 6) * rows * 6 + token * 6 + head % 6
    key_head, remainder = divmod(output_index, rows * 6)
    token, head = divmod(remainder, 6)
    return token * 12 + key_head * 6 + head


def device_layout_dma(mesh, source, rows, owned, *, inverse=False, offset=0):
    import ttnn

    if type(rows) is not int or not 1 <= rows <= 8 or type(inverse) is not bool or type(offset) is not int or offset < 0:
        raise ValueError('Bounded explicit tile permutation required')
    expected = (1, 1, rows * 12, 256) if inverse else None
    shape = tuple(source.shape)
    if (inverse and (shape != expected or offset)) or (not inverse and
            (len(shape) != 4 or shape[0] != 1 or shape[2:] != (12, 256) or offset + rows > shape[1])):
        raise ValueError('Native tiled TP2 query geometry required')
    if source.dtype != ttnn.bfloat16 or source.layout != ttnn.TILE_LAYOUT or source.memory_config() not in (
            ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG):
        raise ValueError('Interleaved BF16 tile input required')
    output_shape = (1, rows, 12, 256) if inverse else (1, 1, rows * 12, 256)
    tiles = rows * 8 if inverse else ((rows * 12 + 31) // 32) * 8
    grid = mesh.compute_with_storage_grid_size()
    if grid.x < 8 or grid.y < tiles // 8:
        raise ValueError('Direct permutation requires the bounded 8-column worker grid')
    output = ttnn.empty(output_shape, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
        device=mesh, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    owned.append(output)
    source_shards, output_shards = ttnn.get_device_tensors(source), ttnn.get_device_tensors(output)
    if len(source_shards) != 2 or len(output_shards) != 2:
        raise ValueError('Both chips required')
    cores = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, tiles // 8 - 1))])
    buffer = ttnn.CBDescriptor(total_size=(max(4, rows) + 1) * 2048, core_ranges=cores,
        format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=0, data_format=ttnn.bfloat16,
            page_size=2048, tile=ttnn.TileDescriptor(ttnn.Tile([32, 32])))])
    program = ttnn.MeshProgramDescriptor()
    for chip in range(2):
        local_source, local_output = source_shards[chip], output_shards[chip]
        descriptor = ttnn.KernelDescriptor(kernel_source=str(Path(__file__).with_suffix('.cpp')), core_ranges=cores,
            compile_time_args=[argument for tensor in (local_source, local_output)
                for argument in ttnn.TensorAccessorArgs(tensor).get_compile_time_args()],
            config=ttnn.DataMovementConfigDescriptor(processor=ttnn.DataMovementProcessor.RISCV_0,
                noc=ttnn.NOC.RISCV_0_default))
        runtime = ttnn.RuntimeArgs()
        for worker in range(tiles):
            runtime[worker % 8][worker // 8] = [local_source.buffer_address(), local_output.buffer_address(),
                rows, int(inverse), offset, worker]
        descriptor.runtime_args = runtime
        coordinate = ttnn.MeshCoordinate(0, chip)
        program[ttnn.MeshCoordinateRange(coordinate, coordinate)] = ttnn.ProgramDescriptor(kernels=[descriptor], cbs=[buffer])
    ttnn.generic_op([source, output], program)
    return output
