"""Experimental byte-preserving BF4 transport layout, not a qualified MLP kernel."""

from pathlib import Path


TILE_BYTES = 576
BLOCK_TILES = 48
BLOCK_BYTES = TILE_BYTES * BLOCK_TILES


def geometry(pairs=272, blocks=20):
    if type(pairs) is not int or not 1 <= pairs <= 272 or type(blocks) is not int or not 1 <= blocks <= 20:
        raise ValueError('Bounded integer gate/up pairs and K blocks required')
    workers = (pairs + 2) // 3
    return dict(pairs=pairs, blocks=blocks, workers=workers,
        source_pages=blocks * 8 * pairs * 2, stream_pages=blocks * workers,
        stream_bytes=blocks * workers * BLOCK_BYTES,
        padding_tiles=blocks * 8 * (workers * 3 - pairs) * 2)


def source_pages(worker, block, *, pairs=272, blocks=20):
    shape = geometry(pairs, blocks)
    if (type(worker) is not int or not 0 <= worker < shape['workers']
            or type(block) is not int or not 0 <= block < blocks):
        raise ValueError('Worker and K block outside transport geometry')
    return tuple((block * 8 + inner) * pairs * 2 + worker * 6 + column
        if worker * 6 + column < pairs * 2 else None
        for inner in range(8) for column in range(6))


def reader_source(original):
    start = '    for (uint32_t block = 0; block < 20; ++block) {\n'
    finish = '    for (uint32_t pair = 0; pair < pairs_per_worker; ++pair) {\n'
    accessor = '    const auto weights = TensorAccessor(weight_args, weight_address, 576);'
    if any(original.count(anchor) != 1 for anchor in (start, finish, accessor)):
        raise ValueError('Unique original fused reader anchors required')
    if original.index(start) >= original.index(finish):
        raise ValueError('Original weight-load loop must precede output publication')
    loop = '''    static_assert(pairs_per_worker == 3);
    for (uint32_t block = 0; block < 20; ++block) {
        cb_reserve_back(1, 48);
        const uint32_t destination = get_write_ptr(1);
        noc_async_read(weights.get_noc_addr(block * 91 + first_pair / 3), destination, 27648);
        noc_async_read_barrier();
        cb_push_back(1, 48);
    }
'''
    return (original[:original.index(start)] + loop + original[original.index(finish):]).replace(
        accessor, '    const auto weights = TensorAccessor(weight_args, weight_address, 27648);')


def pack(mesh, weights, *, pairs=272, blocks=20, raw_tile_fixture=False):
    import ttnn

    shape = geometry(pairs, blocks)
    if type(raw_tile_fixture) is not bool:
        raise ValueError('Explicit raw fixture selection required')
    if raw_tile_fixture:
        valid = (weights.dtype == ttnn.uint32 and weights.layout == ttnn.ROW_MAJOR_LAYOUT
            and tuple(weights.shape) == (1, 1, shape['source_pages'], TILE_BYTES // 4))
    else:
        valid = (weights.dtype == ttnn.bfloat4_b and weights.layout == ttnn.TILE_LAYOUT
            and tuple(weights.shape) in ((blocks * 256, pairs * 64), (1, 1, blocks * 256, pairs * 64)))
    if not valid or weights.memory_config() != ttnn.DRAM_MEMORY_CONFIG:
        raise ValueError('Exact interleaved packed BF4 tiles or explicit raw-word fixture required')
    source_shards = ttnn.get_device_tensors(weights)
    if len(source_shards) != 2:
        raise ValueError('Two-chip transport required')
    grid = mesh.compute_with_storage_grid_size()
    width = min(11, shape['workers'])
    if grid.x < width or grid.y < (shape['workers'] + width - 1) // width:
        raise ValueError('Transport workers exceed compute grid')
    cores = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(worker % width, worker // width),
        ttnn.CoreCoord(worker % width, worker // width)) for worker in range(shape['workers'])])
    output = ttnn.empty((1, 1, shape['stream_pages'], BLOCK_BYTES // 4), dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    try:
        scratch = ttnn.CBDescriptor(total_size=BLOCK_BYTES, core_ranges=cores,
            format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=0, data_format=ttnn.uint32,
                page_size=BLOCK_BYTES)])
        program = ttnn.MeshProgramDescriptor()
        destination_shards = ttnn.get_device_tensors(output)
        if len(destination_shards) != 2:
            raise ValueError('Both packed output shards required')
        for chip, (source, destination) in enumerate(zip(source_shards, destination_shards, strict=True)):
            if source.buffer_address() == destination.buffer_address():
                raise ValueError('Packing must not overwrite native weights')
            runtime = ttnn.RuntimeArgs()
            for worker in range(shape['workers']):
                runtime[worker % width][worker // width] = [source.buffer_address(), destination.buffer_address(),
                    worker, shape['workers'], pairs, blocks]
            kernel = ttnn.KernelDescriptor(kernel_source=str(Path(__file__).with_suffix('.cpp')),
                core_ranges=cores, runtime_args=runtime,
                compile_time_args=ttnn.TensorAccessorArgs(source).get_compile_time_args()
                    + ttnn.TensorAccessorArgs(destination).get_compile_time_args(),
                config=ttnn.DataMovementConfigDescriptor(processor=ttnn.DataMovementProcessor.RISCV_0,
                    noc=ttnn.NOC.RISCV_0_default))
            coordinate = ttnn.MeshCoordinate(0, chip)
            program[ttnn.MeshCoordinateRange(coordinate, coordinate)] = ttnn.ProgramDescriptor(kernels=[kernel], cbs=[scratch])
        ttnn.generic_op([weights, output], program)
        return output
    except BaseException:
        ttnn.deallocate(output)
        raise
