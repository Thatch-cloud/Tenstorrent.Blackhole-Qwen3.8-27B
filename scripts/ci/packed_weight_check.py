"""Byte-exact device comparison of BF4 paired tiles against separate projections."""

from pathlib import Path


def comparison_geometry(packed_shape, separate_shape, offset):
    if type(offset) is not int or offset not in (0, 1):
        raise ValueError('Gate or up tile offset required')
    if (len(separate_shape) != 4 or tuple(separate_shape[:2]) != (1, 1)
            or any(type(value) is not int or value % 32 for value in separate_shape[2:])
            or not 32 <= separate_shape[2] <= 5120 or not 32 <= separate_shape[3] <= 8704
            or tuple(packed_shape) != (*separate_shape[:3], 2 * separate_shape[3])):
        raise ValueError('Tile-aligned bounded local BF4 projection geometry required')
    columns = separate_shape[3] // 32
    pages = separate_shape[2] // 32 * columns
    return min(64, pages), columns, pages


def compare_packed_weights(mesh, packed, separate, offset, owned):
    import ttnn

    workers, columns, pages = comparison_geometry(tuple(packed.shape), tuple(separate.shape), offset)
    for tensor in (packed, separate):
        if tensor.dtype != ttnn.bfloat4_b or tensor.layout != ttnn.TILE_LAYOUT or tensor.memory_config() != ttnn.DRAM_MEMORY_CONFIG:
            raise ValueError('Interleaved tiled BF4 weights required')
    if len(ttnn.get_device_tensors(packed)) != 2 or len(ttnn.get_device_tensors(separate)) != 2:
        raise ValueError('Both TP2 weight shards required')
    grid = mesh.compute_with_storage_grid_size()
    width = min(8, workers)
    if grid.x < width or grid.y < (workers + width - 1) // width:
        raise ValueError('Comparison workers exceed available grid')
    cores = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(worker % width, worker // width),
        ttnn.CoreCoord(worker % width, worker // width)) for worker in range(workers)])
    output = ttnn.empty((1, 1, workers * 32, 32), dtype=ttnn.uint32, layout=ttnn.TILE_LAYOUT,
        device=mesh, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    owned.append(output)
    buffer = ttnn.CBDescriptor(total_size=8192, core_ranges=cores,
        format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=0, data_format=ttnn.uint32,
            page_size=4096, tile=ttnn.TileDescriptor(ttnn.Tile([32, 32])))])
    program = ttnn.MeshProgramDescriptor()
    for chip, shards in enumerate(zip(*(ttnn.get_device_tensors(tensor) for tensor in (packed, separate, output)), strict=True)):
        runtime = ttnn.RuntimeArgs()
        for worker in range(workers):
            runtime[worker % width][worker // width] = [tensor.buffer_address() for tensor in shards] + [worker, workers, columns, pages, offset]
        reader = ttnn.KernelDescriptor(kernel_source=str(Path(__file__).with_name('packed_weight_check.cpp')),
            core_ranges=cores, runtime_args=runtime,
            compile_time_args=[argument for tensor in shards for argument in ttnn.TensorAccessorArgs(tensor).get_compile_time_args()],
            config=ttnn.DataMovementConfigDescriptor(processor=ttnn.DataMovementProcessor.RISCV_0,
                noc=ttnn.NOC.RISCV_0_default))
        coordinate = ttnn.MeshCoordinate(0, chip)
        program[ttnn.MeshCoordinateRange(coordinate, coordinate)] = ttnn.ProgramDescriptor(kernels=[reader], cbs=[buffer])
    ttnn.generic_op([packed, separate, output], program)
    return output


def read_comparison(operations, result, pages):
    import torch

    workers = min(64, pages)
    checks = []
    shards = operations.get_device_tensors(result)
    if len(shards) != 2:
        raise ValueError('Both comparison outputs required')
    for chip, shard in enumerate(shards):
        values = operations.to_torch(shard).long().reshape(workers, 32, 32)
        counts = values[:, 0, :6]
        for worker in range(workers):
            if int(counts[worker, 1]) != len(range(worker, pages, workers)) or int(counts[worker, 4]) != 0x514B5631:
                raise AssertionError('Incomplete device comparison coverage')
            if int(counts[worker, 5]) != (int(counts[worker, 0]) ^ 0xFFFFFFFF):
                raise AssertionError('Comparison counter readback integrity failed')
        padding = values.clone()
        padding[:, 0, :6] = 0
        if torch.count_nonzero(padding):
            raise AssertionError('Comparison output canary changed')
        mismatches = int(counts[:, 0].sum())
        checks.append(dict(chip=chip, pages=int(counts[:, 1].sum()), mismatched_words=mismatches,
            exact=mismatches == 0, workers=workers))
    return checks
