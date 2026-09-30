"""gdn_commit_dma at any served width: single-launch publication of retained GDN prefixes.

The pinned module (gdn_commit_dma.py, held unedited by the K5 evidence) carries the pair's record shapes and page
counts and requires exactly two chips. This is its twin: the same launch, with the widths from tp_shapes, the chip
loop over the mesh's devices, and the sibling kernel gdn_commit_dma_tp.cpp whose page counts arrive as defines
(tp_kernels). tp_addresses.install() rebinds gdn_commit_dma.prepare / publish / validate_shapes to these at
QWEN_FAST_TP=4 only.
"""

from pathlib import Path

import tp_kernels
import tp_shapes


def validate_shapes(layers, prefix):
    found = tp_shapes.active()
    if not 1 <= len(layers) <= 48:
        raise ValueError('One to 48 complete layer records required')
    rows = layers[0][5][0] if len(layers[0]) == 20 and len(layers[0][5]) == 4 else 0
    if type(rows) is not int or rows not in (2, 4, 8, 16, 32):
        raise ValueError('Multirow packed history required')
    compact = [(1, found.gdn_nv, 128, 128)] + [(1, 1, found.gdn_qkv)] * 4
    expected = compact + [(rows, found.gdn_nv, 128, 128)] + [(1, rows, found.gdn_qkv)] * 4
    expected += [(8, found.gdn_nv, 128, 128)] + [(1, 8, found.gdn_qkv)] * 4 + compact
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
    chips = tp_shapes.chip_count()
    if any(len(parts) != chips for parts in shards):
        raise ValueError('%s chips required' % tp_shapes.all_chips())
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
    for chip in range(chips):
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
        kernel = ttnn.KernelDescriptor(kernel_source=tp_kernels.source(Path(__file__).with_name('gdn_commit_dma.cpp')),
            core_ranges=cores, defines=tp_kernels.defines(),
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
