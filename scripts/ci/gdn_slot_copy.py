"""Unqualified per-user GDN snapshot DMA; the slot-zero production path is unchanged."""

from pathlib import Path

from gdn_state_copy import transfer_counts


def transfer_plan(source_shapes, destination_shapes, slot):
    if type(slot) is not int or not 0 <= slot < 8:
        raise ValueError('Explicit native GDN slot in [0, 8) required')
    counts = transfer_counts(source_shapes, destination_shapes)
    source_slot = slot if tuple(source_shapes[0])[0] == 8 else 0
    destination_slot = slot if tuple(destination_shapes[0])[0] == 8 else 0
    return [(count, source_slot, destination_slot) for count in counts]


def page_segments(operand, page, source_slot, destination_slot):
    if (type(operand) is not int or not 0 <= operand < 5
            or type(page) is not int or not 0 <= page < (384 if operand == 0 else 160)
            or any(type(slot) is not int or not 0 <= slot < 8 for slot in (source_slot, destination_slot))):
        raise ValueError('Bounded state operand, page and slots required')
    if operand == 0:
        return [(source_slot * 384 + page, 0, destination_slot * 384 + page, 0, 0, 2048)]
    return [(page, face * 512 + source_slot * 32, page,
             face * 512 + destination_slot * 32, face * 512 + source_slot * 32, 32)
            for face in (0, 1)]


def copy_slot(mesh, source, destination, *, slot):
    import ttnn

    plan = transfer_plan([tuple(value.shape) for value in source],
        [tuple(value.shape) for value in destination], slot)
    tensors = [value for pair in zip(source, destination, strict=True) for value in pair]
    if any(value.dtype != ttnn.bfloat16 or value.layout != ttnn.TILE_LAYOUT
            or value.memory_config() != ttnn.DRAM_MEMORY_CONFIG
            or tuple(value.tile.tile_shape) != (32, 32)
            or value.tile.transpose_of_faces or value.tile.transpose_within_face for value in tensors):
        raise ValueError('Non-transposed interleaved DRAM BF16 tiles required')
    shards = [ttnn.get_device_tensors(value) for value in tensors]
    if any(len(parts) != 2 for parts in shards):
        raise ValueError('Both chips required')
    grid = mesh.compute_with_storage_grid_size()
    if grid.x < 8 or grid.y < 6:
        raise ValueError('Audited 8x6 state-copy worker grid required')
    cores = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 5))])
    buffer = ttnn.CBDescriptor(total_size=4096, core_ranges=cores,
        format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=0, data_format=ttnn.bfloat16,
            page_size=2048, tile=ttnn.TileDescriptor(ttnn.Tile([32, 32])))])
    program = ttnn.MeshProgramDescriptor()
    for chip in range(2):
        local = [parts[chip] for parts in shards]
        if len({value.buffer_address() for value in local}) != 10:
            raise ValueError('State and snapshot buffers must not alias')
        runtime = []
        for index, (count, source_slot, destination_slot) in enumerate(plan):
            runtime.extend([local[2 * index].buffer_address(), local[2 * index + 1].buffer_address(),
                count, source_slot, destination_slot])
        kernel = ttnn.KernelDescriptor(kernel_source=str(Path(__file__).with_suffix('.cpp')),
            core_ranges=cores,
            compile_time_args=[argument for value in local for argument in ttnn.TensorAccessorArgs(value).get_compile_time_args()],
            config=ttnn.DataMovementConfigDescriptor(processor=ttnn.DataMovementProcessor.RISCV_0,
                noc=ttnn.NOC.RISCV_0_default))
        arguments = ttnn.RuntimeArgs()
        for worker in range(48):
            arguments[worker % 8][worker // 8] = runtime + [worker]
        kernel.runtime_args = arguments
        coordinate = ttnn.MeshCoordinate(0, chip)
        program[ttnn.MeshCoordinateRange(coordinate, coordinate)] = ttnn.ProgramDescriptor(kernels=[kernel], cbs=[buffer])
    ttnn.generic_op(tensors, program)
