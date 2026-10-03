"""ordered_cache.validate_shapes / update (the audited ordered K/V write, one launch per 32-row tile) at any served width.

ordered_cache.py carries the pair's cache geometry: (N, 2, 64, 256), a head count of 2 in the upstream kernels' compile
arguments, and a two-chip program. At four cards a chip holds ONE KV head, so the cache is (N, 1, 64, 256) and the
kernels' num_heads arguments (reader compile argument 9, writer 10, compute 7 - upstream paged_cache
update_cache kernels at v0.77.0-rc1) are 1. The prepared input tile stays (1, T, 32, 256): a 32-row tile whose row h
holds head h, so at one head only row 0 is read. This twin is the same validation and launch with those numbers from
tp_shapes; the kernels are ordered_cache.load_kernels' hash-pinned sources, imported, never copied.
tp_addresses.install() rebinds ordered_cache.validate_shapes / update to these at QWEN_FAST_TP=4 only.

The admitted page-table widths are page_width_tp4.admitted's: the pinned answer, plus 4,096 (a 262,144-token window) once the
E1 record qualifies it. It is looked up through the module at every call, so a harness can scope it.
"""

import ordered_cache as pinned
import page_width_tp4
import tp_shapes


def cache_heads():
    """KV heads per chip: 2 at the pair, 1 at four cards."""
    return tp_shapes.active().attn_kv_heads


def validate_shapes(cache, packed, positions, pages):
    rows = packed[1] if len(packed) == 4 else 0
    if type(rows) is not int or rows not in (1, 2, 4, 8, 16, 32) or tuple(packed) != (1, rows, 32, 256):
        raise ValueError('Native prepared T=1/2/4/8/16/32 KV tiles required')
    if len(cache) != 4 or cache[0] < 1 or tuple(cache[1:]) != (cache_heads(), 64, 256):
        raise ValueError('Expected %s-head 64-row BF8 paged cache' % {1: 'one', 2: 'two'}[cache_heads()])
    if (tuple(positions) != (rows,) or len(pages) != 2 or pages[0] != rows
            or not page_width_tp4.admitted(pages[1]) or pages[1] > cache[0]):
        raise ValueError('Paired position vector and page-table rows required')
    return rows


def update(mesh, cache, packed, positions, pages, kernels):
    import ttnn

    heads = cache_heads()
    rows = validate_shapes(tuple(cache.shape), tuple(packed.shape), tuple(positions.shape), tuple(pages.shape))
    tensors = [cache, packed, positions, pages]
    if any(value.memory_config() != ttnn.DRAM_MEMORY_CONFIG for value in tensors):
        raise ValueError('Interleaved DRAM buffers required')
    if cache.dtype != ttnn.bfloat8_b or packed.dtype != ttnn.bfloat16 or any(
            value.layout != ttnn.TILE_LAYOUT for value in tensors[:2]):
        raise ValueError('Native BF8 cache and BF16 input tiles required')
    if any(value.dtype != ttnn.int32 or value.layout != ttnn.ROW_MAJOR_LAYOUT for value in tensors[2:]):
        raise ValueError('Int32 row-major metadata required')
    shards = [ttnn.get_device_tensors(value) for value in tensors]
    chips = tp_shapes.chip_count()
    if any(len(parts) != chips for parts in shards):
        raise ValueError('%s chips required' % tp_shapes.count_word())
    grid = mesh.compute_with_storage_grid_size()
    if grid.x < 8 or grid.y < 2:
        raise ValueError('Audited 8x2 worker subset required')
    coordinates = [ttnn.CoreCoord(index % 8, index // 8) for index in range(rows)]
    cores = ttnn.CoreRangeSet([ttnn.CoreRange(core, core) for core in coordinates])
    buffers = []

    def buffer(indices, count, dtype, page, tiled=True):
        formats = [ttnn.CBFormatDescriptor(buffer_index=index, data_format=dtype, page_size=page,
            **(dict(tile=ttnn.TileDescriptor(ttnn.Tile([32, 32]))) if tiled else {})) for index in indices]
        buffers.append(ttnn.CBDescriptor(total_size=count * page, core_ranges=cores, format_descriptors=formats))

    buffer([0], 16, ttnn.bfloat8_b, 1088)
    buffer([1], 8, ttnn.bfloat16, 2048)
    buffer([24, 25], 16, ttnn.bfloat16, 2048)
    buffer([26], 16, ttnn.bfloat16, 2048)
    buffer([16], rows * 8, ttnn.bfloat8_b, 1088)
    buffer([2], 1, ttnn.int32, 4096, False)
    page_bytes = pages.padded_shape[-1] * 4
    buffer([3], 1, ttnn.int32, page_bytes, False)
    semaphore = ttnn.SemaphoreDescriptor(id=0, core_ranges=cores, initial_value=0)
    program = ttnn.MeshProgramDescriptor()
    for chip in range(chips):
        local = [parts[chip] for parts in shards]
        addresses = [value.buffer_address() for value in local]
        if len(set(addresses)) != 4:
            raise ValueError('Cache, packed input and metadata must not alias')
        reader_args = [0, 1, 1, 2, 0, 8, 0, rows * 4, 1, heads, 64, 2, pages.padded_shape[-1], 0, page_bytes, 3, 2, 0, 0]
        for index in (0, 2, 3, 1):
            reader_args.extend(ttnn.TensorAccessorArgs(local[index]).get_compile_time_args())
        writer_args = [16, 24, 25, 26, 1, 2, 0, 8, 512, 1, heads, 64, 2, pages.padded_shape[-1], 3, 2, 0, 0]
        writer_args.extend(ttnn.TensorAccessorArgs(local[0]).get_compile_time_args())
        descriptors = []
        for role, args, config in (
            ('reader', reader_args, ttnn.DataMovementConfigDescriptor(processor=ttnn.DataMovementProcessor.RISCV_1,
                                                                     noc=ttnn.NOC.RISCV_1_default)),
            ('writer', writer_args, ttnn.DataMovementConfigDescriptor(processor=ttnn.DataMovementProcessor.RISCV_0,
                                                                     noc=ttnn.NOC.RISCV_0_default)),
            ('compute', [0, 1, 24, 25, 26, 16, 8, heads], ttnn.ComputeConfigDescriptor(fp32_dest_acc_en=False)),
        ):
            runtime = ttnn.RuntimeArgs()
            for index, core in enumerate(coordinates):
                next_core = local[0].device().worker_core_from_logical_core(coordinates[min(index + 1, rows - 1)])
                runtime[core.x][core.y] = ([addresses[0], 0, addresses[2], index, addresses[3], int(index > 0), addresses[1]]
                    if role == 'reader' else [addresses[0], 0, 0, index, int(index < rows - 1), next_core.x, next_core.y]
                    if role == 'writer' else [])
            descriptor = ttnn.KernelDescriptor(kernel_source=kernels[role],
                source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE, core_ranges=cores,
                compile_time_args=args, config=config)
            descriptor.runtime_args = runtime
            descriptors.append(descriptor)
        coordinate = ttnn.MeshCoordinate(0, chip)
        program[ttnn.MeshCoordinateRange(coordinate, coordinate)] = ttnn.ProgramDescriptor(
            kernels=descriptors, cbs=buffers, semaphores=[semaphore])
    ttnn.generic_op(tensors, program)
