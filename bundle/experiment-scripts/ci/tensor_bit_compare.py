"""Unqualified exact BF16 physical-storage comparator for simulator diagnostics."""

import math
from pathlib import Path


def prepare(operations, mesh, left, right, output):
    if (tuple(left.shape) != tuple(right.shape) or tuple(left.padded_shape) != tuple(right.padded_shape)
            or any(value.dtype != operations.bfloat16 or value.layout != operations.TILE_LAYOUT
                or value.memory_config() != operations.DRAM_MEMORY_CONFIG for value in (left, right))
            or output.dtype != operations.uint32 or output.layout != operations.ROW_MAJOR_LAYOUT
            or tuple(output.shape) != (1, 1, 32) or output.memory_config() != operations.DRAM_MEMORY_CONFIG
            or tuple(mesh.shape) != (1, 2)):
        raise ValueError('Matching BF16 tile inputs and a separate 32-word uint32 output on two chips required')
    tiles = math.prod(left.padded_shape) // 1024
    if not 1 <= tiles <= 8388607:
        raise ValueError('Whole tiled tensors with non-overflowing mismatch counts required')
    grid = mesh.compute_with_storage_grid_size()
    if grid.x * grid.y < 32:
        raise ValueError('32 comparison workers required')
    coordinates = [operations.CoreCoord(worker % grid.x, worker // grid.x) for worker in range(32)]
    cores = operations.CoreRangeSet([operations.CoreRange(core, core) for core in coordinates])
    buffer = operations.CBDescriptor(total_size=4096, core_ranges=cores,
        format_descriptors=[operations.CBFormatDescriptor(buffer_index=0, data_format=operations.bfloat16,
            page_size=2048, tile=operations.TileDescriptor(operations.Tile([32, 32])))])
    shards = [operations.get_device_tensors(value) for value in (left, right, output)]
    if any(len(parts) != 2 for parts in shards):
        raise ValueError('Two-chip operands required')
    program = operations.MeshProgramDescriptor()
    for chip in range(2):
        local = [parts[chip] for parts in shards]
        pointers = [value.buffer_address() for value in local]
        if pointers[2] in pointers[:2]:
            raise ValueError('Comparison must not overwrite either input')
        kernel = operations.KernelDescriptor(kernel_source=str(Path(__file__).with_suffix('.cpp')),
            core_ranges=cores, compile_time_args=[argument for value in local
                for argument in operations.TensorAccessorArgs(value).get_compile_time_args()],
            config=operations.DataMovementConfigDescriptor(processor=operations.DataMovementProcessor.RISCV_0,
                noc=operations.NOC.RISCV_0_default))
        runtime = operations.RuntimeArgs()
        for worker, core in enumerate(coordinates):
            runtime[core.x][core.y] = pointers + [tiles, worker]
        kernel.runtime_args = runtime
        coordinate = operations.MeshCoordinate(0, chip)
        program[operations.MeshCoordinateRange(coordinate, coordinate)] = operations.ProgramDescriptor(kernels=[kernel], cbs=[buffer])
    def execute():
        operations.generic_op([left, right, output], program)
    return execute
