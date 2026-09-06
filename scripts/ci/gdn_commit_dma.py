"""Single-launch publication of retained GDN prefixes; fixture-only and arithmetic-free."""

from pathlib import Path


def validate_shapes(layers, prefix):
    if not 1 <= len(layers) <= 48:
        raise ValueError('One to 48 complete layer records required')
    rows = layers[0][5][0] if len(layers[0]) == 20 and len(layers[0][5]) == 4 else 0
    if type(rows) is not int or rows not in (2, 4, 8, 16):
        raise ValueError('Multirow packed history required')
    compact = [(1, 24, 128, 128)] + [(1, 1, 5120)] * 4
    expected = compact + [(rows, 24, 128, 128)] + [(1, rows, 5120)] * 4
    expected += [(8, 24, 128, 128)] + [(1, 8, 5120)] * 4 + compact
    if any([tuple(shape) for shape in layer] != expected for layer in layers):
        raise ValueError('Complete entry/history/native/checkpoint geometry required')
    if type(prefix) is not int or not 0 <= prefix <= rows:
        raise ValueError('Commit prefix outside verified rows')
    return rows


def prepare(mesh, layers, prefix):
    import ttnn

    validate_shapes([[tuple(value.shape) for value in layer] for layer in layers], prefix)
    tensors = [value for layer in layers for value in layer]
    if any(value.dtype != ttnn.bfloat16 or value.layout != ttnn.TILE_LAYOUT or
           value.memory_config() != ttnn.DRAM_MEMORY_CONFIG for value in tensors):
        raise ValueError('Interleaved DRAM BF16 state required')
    shards = [ttnn.get_device_tensors(value) for value in tensors]
    if any(len(parts) != 2 for parts in shards):
        raise ValueError('Both chips required')
    grid = mesh.compute_with_storage_grid_size()
    workers = len(layers) * 2
    if grid.x * grid.y < workers:
        raise ValueError('Two independent workers per layer required')
    coordinates = [ttnn.CoreCoord(worker % grid.x, worker // grid.x) for worker in range(workers)]
    cores = ttnn.CoreRangeSet([ttnn.CoreRange(core, core) for core in coordinates])
    buffer = ttnn.CBDescriptor(total_size=4096, core_ranges=cores,
        format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=0, data_format=ttnn.bfloat16,
            page_size=2048, tile=ttnn.TileDescriptor(ttnn.Tile([32, 32])))])
    program = ttnn.MeshProgramDescriptor()
    for chip in range(2):
        local = [parts[chip] for parts in shards]
        addresses = [value.buffer_address() for value in local]
        if len(set(addresses)) != len(addresses):
            raise ValueError('Layer histories and publication destinations must not alias')
        descriptors = [ttnn.TensorAccessorArgs(value).get_compile_time_args() for value in local]
        reference = descriptors[:20]
        if any(descriptors[index:index + 20] != reference for index in range(20, len(local), 20)):
            raise ValueError('All layer accessors must have identical allocation geometry')
        for first in (1, 6, 11, 16):
            if any(reference[first + slot] != reference[first] for slot in range(4)):
                raise ValueError('Four identical convolution accessor layouts required')
        kernel = ttnn.KernelDescriptor(kernel_source=str(Path(__file__).with_suffix('.cpp')),
            core_ranges=cores,
            compile_time_args=[argument for index in (0, 1, 5, 6, 10, 11, 15, 16) for argument in reference[index]],
            config=ttnn.DataMovementConfigDescriptor(processor=ttnn.DataMovementProcessor.RISCV_0,
                                                     noc=ttnn.NOC.RISCV_0_default))
        runtime = ttnn.RuntimeArgs()
        for worker, core in enumerate(coordinates):
            offset = (worker // 2) * 20
            runtime[core.x][core.y] = addresses[offset:offset + 20] + [prefix, worker % 2]
        kernel.runtime_args = runtime
        coordinate = ttnn.MeshCoordinate(0, chip)
        program[ttnn.MeshCoordinateRange(coordinate, coordinate)] = ttnn.ProgramDescriptor(kernels=[kernel], cbs=[buffer])
    def execute():
        ttnn.generic_op(tensors, program)

    return execute


def publish(mesh, layers, prefix):
    prepare(mesh, layers, prefix)()
