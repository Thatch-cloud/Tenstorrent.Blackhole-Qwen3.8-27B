"""Unqualified fixed-capacity BF16 sliding history transport; no serving integration."""

from pathlib import Path


def geometry(history_rows, prefix):
    if (type(history_rows) is not int or not 1 <= history_rows <= 2048
            or type(prefix) is not int or not 1 <= prefix <= 32):
        raise ValueError('One committed history and bounded accepted prefix required')
    rows = min(2048, history_rows + prefix)
    return dict(history_rows=history_rows, prefix=prefix, rows=rows,
        drop=history_rows + prefix - rows)


def row_source(history_rows, prefix, output_row):
    shape = geometry(history_rows, prefix)
    if type(output_row) is not int or not 0 <= output_row < 2048:
        raise ValueError('Output row outside fixed history capacity')
    if output_row >= shape['rows']:
        return 'zero', 0
    source = output_row + shape['drop']
    return ('active', source) if source < history_rows else ('delta', source - history_rows)


def prepare(mesh, active, delta, spare, *, history_rows, prefix):
    import ttnn

    shape = geometry(history_rows, prefix)
    tensors = [active, delta, spare]
    shards = [ttnn.get_device_tensors(value) for value in tensors]
    if any(len(parts) != 2 for parts in shards):
        raise ValueError('Two independent device shards required')
    for parts, expected in zip(shards, ((1, 4, 2048, 128), (1, 4, 32, 128), (1, 4, 2048, 128))):
        for value in parts:
            if (tuple(value.shape) != expected or value.dtype != ttnn.bfloat16
                    or value.layout != ttnn.TILE_LAYOUT or value.memory_config() != ttnn.DRAM_MEMORY_CONFIG
                    or tuple(value.tile.tile_shape) != (32, 32)
                    or value.tile.transpose_of_faces or value.tile.transpose_within_face):
                raise ValueError('Exact non-transposed interleaved BF16 cache tiles required')
    grid = mesh.compute_with_storage_grid_size()
    if grid.x * grid.y < 16:
        raise ValueError('Sixteen head/column transport workers required')
    coordinates = [ttnn.CoreCoord(worker % grid.x, worker // grid.x) for worker in range(16)]
    cores = ttnn.CoreRangeSet([ttnn.CoreRange(core, core) for core in coordinates])
    buffer = ttnn.CBDescriptor(total_size=8192, core_ranges=cores,
        format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=0, data_format=ttnn.bfloat16,
            page_size=2048, tile=ttnn.TileDescriptor(ttnn.Tile((32, 32))))])
    program = ttnn.MeshProgramDescriptor()
    for chip in range(2):
        local = [parts[chip] for parts in shards]
        addresses = [value.buffer_address() for value in local]
        if len(set(addresses)) != 3:
            raise ValueError('Active, delta and spare storage must not alias')
        kernel = ttnn.KernelDescriptor(kernel_source=str(Path(__file__).with_suffix('.cpp')), core_ranges=cores,
            compile_time_args=[argument for value in local
                for argument in ttnn.TensorAccessorArgs(value).get_compile_time_args()],
            config=ttnn.DataMovementConfigDescriptor(processor=ttnn.DataMovementProcessor.RISCV_0,
                noc=ttnn.NOC.RISCV_0_default))
        runtime = ttnn.RuntimeArgs()
        for worker, core in enumerate(coordinates):
            runtime[core.x][core.y] = addresses + [history_rows, prefix, shape['drop'], shape['rows'], worker]
        kernel.runtime_args = runtime
        coordinate = ttnn.MeshCoordinate(0, chip)
        program[ttnn.MeshCoordinateRange(coordinate, coordinate)] = ttnn.ProgramDescriptor(kernels=[kernel], cbs=[buffer])
    return lambda: ttnn.generic_op(tensors, program)
