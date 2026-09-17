"""Fixture-only ordered shared-page writes using audited native BF8 RMW math."""

import hashlib
from pathlib import Path


ROOT = Path('ttnn/cpp/ttnn/operations/experimental/paged_cache/device/kernels')
FILES = dict(reader='dataflow/reader_update_cache_interleaved_start_id.cpp',
             writer='dataflow/writer_update_cache_interleaved_start_id.cpp',
             compute='compute/update_cache.cpp')
HASHES = dict(reader='ed5c680a1ab781f44b09204128556c97b086b7c7176b954e54a2739f665217da',
              writer='05e1c7a93f5f1431e4c14a0e17904fc700a6b2e9e0a0cad8a396d6ee11d999b2',
              compute='6e71c7ecee25271a49ade605e0afa3b78ef68b87d5afb49d7346309a006a0cfb')


def replace_once(source, old, new):
    if source.count(old) != 1:
        raise ValueError('Audited cache reader anchor changed')
    return source.replace(old, new)


def load_kernels(root):
    sources = {}
    for role, relative in FILES.items():
        raw = (Path(root) / ROOT / relative).read_bytes()
        if hashlib.sha256(raw).hexdigest() != HASHES[role]:
            raise ValueError(f'Unaudited native cache {role}')
        sources[role] = raw.decode()
    reader = replace_once(sources['reader'],
        'constexpr auto page_table_args = TensorAccessorArgs<index_tensor_args.next_compile_time_args_offset()>();',
        'constexpr auto page_table_args = TensorAccessorArgs<index_tensor_args.next_compile_time_args_offset()>();\n'
        '    constexpr auto input_args = TensorAccessorArgs<page_table_args.next_compile_time_args_offset()>();')
    sources['reader'] = replace_once(reader,
        'cb_input.reserve_back(Wt);\n    cb_input.push_back(Wt);',
        'cb_input.reserve_back(Wt);\n'
        '    const auto input = TensorAccessor(input_args, get_arg_val<uint32_t>(6));\n'
        '    const uint32_t input_tile_bytes = get_tile_size(input_cb_id);\n'
        '    for (uint32_t tile = 0; tile < Wt; ++tile) {\n'
        '        noc.async_read(input, CoreLocalMem<uint32_t>(cb_input.get_write_ptr() + tile * input_tile_bytes),\n'
        '            input_tile_bytes, {.page_id = my_batch_idx * Wt + tile}, {});\n'
        '    }\n'
        '    noc.async_read_barrier();\n'
        '    cb_input.push_back(Wt);')
    return sources


def validate_shapes(cache, packed, positions, pages):
    rows = packed[1] if len(packed) == 4 else 0
    if type(rows) is not int or rows not in (1, 2, 4, 8, 16, 32) or tuple(packed) != (1, rows, 32, 256):
        raise ValueError('Native prepared T=1/2/4/8/16/32 KV tiles required')
    if len(cache) != 4 or cache[0] < 1 or tuple(cache[1:]) != (2, 64, 256):
        raise ValueError('Expected two-head 64-row BF8 paged cache')
    if tuple(positions) != (rows,) or len(pages) != 2 or pages[0] != rows or not 1 <= pages[1] <= min(1024, cache[0]):
        raise ValueError('Paired position vector and page-table rows required')
    return rows


def update(mesh, cache, packed, positions, pages, kernels):
    import ttnn

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
    if any(len(parts) != 2 for parts in shards):
        raise ValueError('Two chips required')
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
    for chip in range(2):
        local = [parts[chip] for parts in shards]
        addresses = [value.buffer_address() for value in local]
        if len(set(addresses)) != 4:
            raise ValueError('Cache, packed input and metadata must not alias')
        reader_args = [0, 1, 1, 2, 0, 8, 0, rows * 4, 1, 2, 64, 2, pages.padded_shape[-1], 0, page_bytes, 3, 2, 0, 0]
        for index in (0, 2, 3, 1):
            reader_args.extend(ttnn.TensorAccessorArgs(local[index]).get_compile_time_args())
        writer_args = [16, 24, 25, 26, 1, 2, 0, 8, 512, 1, 2, 64, 2, pages.padded_shape[-1], 3, 2, 0, 0]
        writer_args.extend(ttnn.TensorAccessorArgs(local[0]).get_compile_time_args())
        descriptors = []
        for role, args, config in (
            ('reader', reader_args, ttnn.DataMovementConfigDescriptor(processor=ttnn.DataMovementProcessor.RISCV_1,
                                                                     noc=ttnn.NOC.RISCV_1_default)),
            ('writer', writer_args, ttnn.DataMovementConfigDescriptor(processor=ttnn.DataMovementProcessor.RISCV_0,
                                                                     noc=ttnn.NOC.RISCV_0_default)),
            ('compute', [0, 1, 24, 25, 26, 16, 8, 2], ttnn.ComputeConfigDescriptor(fp32_dest_acc_en=False)),
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
