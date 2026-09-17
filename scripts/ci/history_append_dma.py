"""Unqualified BF16 dirty-tile writer; no model or serving integration."""

from pathlib import Path

from history_append_plan import AppendPlan


def validate_plan(plan, capacity):
    if (not isinstance(plan, AppendPlan) or type(capacity) is not int or capacity % 32
            or any(type(value) is not int for value in (plan.position, plan.prefix, plan.first_row, plan.end_row))
            or not 1 <= plan.prefix <= 32 or not 0 <= plan.first_row <= plan.position
            or not plan.position + plan.prefix <= plan.end_row <= capacity
            or plan.first_row % 32 or plan.end_row % 32
            or not 32 <= plan.end_row - plan.first_row <= 96):
        raise ValueError('Bounded tile-aligned append plan required')


def prepare(mesh, active, delta, spare, plan):
    import ttnn

    tensors = [active, delta, spare]
    shards = [ttnn.get_device_tensors(value) for value in tensors]
    if any(len(parts) != 2 for parts in shards):
        raise ValueError('Two independent device shards required')
    capacity = shards[0][0].shape[2]
    validate_plan(plan, capacity)
    for parts, shape in zip(shards, ((1, 4, capacity, 128), (1, 4, 32, 128), (1, 4, capacity, 128))):
        for value in parts:
            if (tuple(value.shape) != shape or value.dtype != ttnn.bfloat16
                    or value.layout != ttnn.TILE_LAYOUT or value.memory_config() != ttnn.DRAM_MEMORY_CONFIG
                    or tuple(value.tile.tile_shape) != (32, 32)
                    or value.tile.transpose_of_faces or value.tile.transpose_within_face):
                raise ValueError('Exact non-transposed BF16 interleaved history tiles required')
    grid = mesh.compute_with_storage_grid_size()
    if grid.x * grid.y < 16:
        raise ValueError('Sixteen independent head/column workers required')
    coordinates = [ttnn.CoreCoord(worker % grid.x, worker // grid.x) for worker in range(16)]
    cores = ttnn.CoreRangeSet([ttnn.CoreRange(core, core) for core in coordinates])
    buffer = ttnn.CBDescriptor(total_size=6144, core_ranges=cores,
        format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=0, data_format=ttnn.bfloat16,
            page_size=2048, tile=ttnn.TileDescriptor(ttnn.Tile((32, 32))))])
    program = ttnn.MeshProgramDescriptor()
    for chip in range(2):
        local = [parts[chip] for parts in shards]
        addresses = [value.buffer_address() for value in local]
        if len(set(addresses)) != 3:
            raise ValueError('Active, delta and spare buffers must not alias')
        kernel = ttnn.KernelDescriptor(kernel_source=str(Path(__file__).with_suffix('.cpp')), core_ranges=cores,
            compile_time_args=[argument for value in local
                for argument in ttnn.TensorAccessorArgs(value).get_compile_time_args()],
            config=ttnn.DataMovementConfigDescriptor(processor=ttnn.DataMovementProcessor.RISCV_0,
                noc=ttnn.NOC.RISCV_0_default))
        runtime = ttnn.RuntimeArgs()
        for worker, core in enumerate(coordinates):
            runtime[core.x][core.y] = addresses + [capacity // 32, plan.position, plan.prefix,
                plan.first_row // 32, plan.end_row // 32, worker]
        kernel.runtime_args = runtime
        coordinate = ttnn.MeshCoordinate(0, chip)
        program[ttnn.MeshCoordinateRange(coordinate, coordinate)] = ttnn.ProgramDescriptor(kernels=[kernel], cbs=[buffer])
    return lambda: ttnn.generic_op(tensors, program)
