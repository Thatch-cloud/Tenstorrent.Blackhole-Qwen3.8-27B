"""attention_mask_replay's mask refresh program at any served width.

attention_mask_replay.py / .cpp are frozen-recipe evidence and carry the pair's 12 folded query rows per token and a
two-chip check. This is the pinned mask_position and prepare with the head-row count from tp_shapes (6 at four cards, one
row per local query head on the one KV head), the chip loop over the mesh's devices, and the sibling kernel
attention_mask_replay_tp.cpp (the head-row count is a define, tp_kernels.fold_defines). validate_ticket, execute and
the rest stay the pinned module's own. At the pair prepare selects the pinned kernel file and no defines.
"""

from pathlib import Path

import attention_mask_replay as pinned
import tp_kernels
import tp_shapes


def mask_position(start, rows, batch, head, offset=0):
    head_rows = tp_shapes.active().attn_fold_rows
    if not 1 <= rows <= 8 or not 0 <= batch < 3 or not 0 <= head < rows * head_rows:
        raise ValueError('Bounded folded query head required')
    return start + offset + batch * rows + (head % (rows * 6)) // 6


def prepare(mesh, positions, mask, *, rows, batches, offset, capacity, short_context=False):
    import ttnn

    head_rows = tp_shapes.active().attn_fold_rows
    if any(type(value) is not int for value in (rows, batches, offset, capacity)):
        raise ValueError('Integer mask geometry required')
    if not 1 <= rows <= 8 or not 1 <= batches <= 3 or offset < 0 or offset + rows * batches > 32:
        raise ValueError('At most three bounded contiguous query groups required')
    first = max(128, capacity - 256) if short_context else capacity - 256
    pinned.validate_ticket(first, offset + rows * batches, capacity, short_context=short_context)
    if tuple(positions.shape) != (8,) or positions.dtype != ttnn.int32 or positions.layout != ttnn.ROW_MAJOR_LAYOUT:
        raise ValueError('Eight-word position input required; first word is block start')
    if (tuple(mask.shape) != (batches, 1, rows * head_rows, capacity) or mask.dtype != ttnn.bfloat16
            or mask.layout != ttnn.TILE_LAYOUT):
        raise ValueError('Fixed-shape BF16 folded attention mask required')
    if any(tensor.memory_config() != ttnn.DRAM_MEMORY_CONFIG for tensor in (positions, mask)):
        raise ValueError('Interleaved DRAM metadata required')
    head_tiles = (rows * head_rows + 31) // 32
    tasks = batches * head_tiles * 8
    grid = mesh.compute_with_storage_grid_size()
    if grid.x < 8 or grid.y < tasks // 8:
        raise ValueError('Bounded eight-column mask worker grid required')
    cores = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, tasks // 8 - 1))])
    buffer = ttnn.CBDescriptor(total_size=4096, core_ranges=cores,
        format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=0, data_format=ttnn.bfloat16,
            page_size=2048, tile=ttnn.TileDescriptor(ttnn.Tile([32, 32])))])
    parts = [ttnn.get_device_tensors(tensor) for tensor in (positions, mask)]
    chips = tp_shapes.chip_count()
    if any(len(values) != chips for values in parts):
        raise ValueError('%s chip-local metadata buffers required' % tp_shapes.count_word())
    program = ttnn.MeshProgramDescriptor()
    for chip in range(chips):
        local = [values[chip] for values in parts]
        if local[0].buffer_address() == local[1].buffer_address():
            raise ValueError('Mask and position input must not alias')
        descriptor = ttnn.KernelDescriptor(
            kernel_source=tp_kernels.source(Path(pinned.__file__).with_suffix('.cpp')), core_ranges=cores,
            defines=tp_kernels.fold_defines(),
            compile_time_args=[argument for tensor in local for argument in ttnn.TensorAccessorArgs(tensor).get_compile_time_args()],
            config=ttnn.DataMovementConfigDescriptor(processor=ttnn.DataMovementProcessor.RISCV_0,
                noc=ttnn.NOC.RISCV_0_default))
        runtime = ttnn.RuntimeArgs()
        for task in range(tasks):
            runtime[task % 8][task // 8] = [local[0].buffer_address(), local[1].buffer_address(),
                rows, capacity, offset, task]
        descriptor.runtime_args = runtime
        coordinate = ttnn.MeshCoordinate(0, chip)
        program[ttnn.MeshCoordinateRange(coordinate, coordinate)] = ttnn.ProgramDescriptor(kernels=[descriptor], cbs=[buffer])
    return program
