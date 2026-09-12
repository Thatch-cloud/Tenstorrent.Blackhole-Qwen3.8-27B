"""Experimental same-address causal mask refresh within one native chunk family."""

from pathlib import Path
import hashlib
import os


def source_hashes():
    return {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
            for name in ('attention_mask_replay.py', 'attention_mask_replay.cpp')}


def validate_ticket(start, rows, capacity, *, short_context=False):
    if type(short_context) is not bool:
        raise ValueError('Explicit short-context experimental selection required')
    if any(type(value) is not int for value in (start, rows, capacity)):
        raise ValueError('Integer replay geometry required')
    minimum, maximum = (256, 768) if short_context else (4096, 16640)
    if os.environ.get('QWEN_CONTEXT_LADDER_SIM') == '1':
        if (os.environ.get('QWEN_SIM_ONLY') != '1' or short_context
                or os.environ.get('QWEN_HARDWARE_TESTS') == '1' or os.environ.get('QWEN_CARDS_ALLOCATED') == '1'):
            raise ValueError('Experimental context ladder is simulator-only')
        if capacity not in tuple(context + 256 for context in (2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144)):
            raise ValueError('Explicit context-ladder capacity required')
        minimum, maximum = 2304, 262400
    if not 1 <= rows <= 32 or capacity < minimum or capacity > maximum or capacity % 256:
        raise ValueError('Bounded long-context native chunk family required')
    if short_context and rows != 8:
        raise ValueError('Short-context qualification is limited to T8')
    first = max(128, capacity - 256) if short_context else capacity - 256
    if start < first or start + rows > capacity:
        raise ValueError('Ticket crosses the captured native chunk family')


def mask_position(start, rows, batch, head, offset=0):
    if not 1 <= rows <= 8 or not 0 <= batch < 3 or not 0 <= head < rows * 12:
        raise ValueError('Bounded folded query head required')
    return start + offset + batch * rows + (head % (rows * 6)) // 6


def prepare(mesh, positions, mask, *, rows, batches, offset, capacity, short_context=False):
    import ttnn

    if any(type(value) is not int for value in (rows, batches, offset, capacity)):
        raise ValueError('Integer mask geometry required')
    if not 1 <= rows <= 8 or not 1 <= batches <= 3 or offset < 0 or offset + rows * batches > 32:
        raise ValueError('At most three bounded contiguous query groups required')
    first = max(128, capacity - 256) if short_context else capacity - 256
    validate_ticket(first, offset + rows * batches, capacity, short_context=short_context)
    if tuple(positions.shape) != (8,) or positions.dtype != ttnn.int32 or positions.layout != ttnn.ROW_MAJOR_LAYOUT:
        raise ValueError('Eight-word position input required; first word is block start')
    if tuple(mask.shape) != (batches, 1, rows * 12, capacity) or mask.dtype != ttnn.bfloat16 or mask.layout != ttnn.TILE_LAYOUT:
        raise ValueError('Fixed-shape BF16 folded attention mask required')
    if any(tensor.memory_config() != ttnn.DRAM_MEMORY_CONFIG for tensor in (positions, mask)):
        raise ValueError('Interleaved DRAM metadata required')
    head_tiles = (rows * 12 + 31) // 32
    tasks = batches * head_tiles * 8
    grid = mesh.compute_with_storage_grid_size()
    if grid.x < 8 or grid.y < tasks // 8:
        raise ValueError('Bounded eight-column mask worker grid required')
    cores = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, tasks // 8 - 1))])
    buffer = ttnn.CBDescriptor(total_size=4096, core_ranges=cores,
        format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=0, data_format=ttnn.bfloat16,
            page_size=2048, tile=ttnn.TileDescriptor(ttnn.Tile([32, 32])))])
    parts = [ttnn.get_device_tensors(tensor) for tensor in (positions, mask)]
    if any(len(values) != 2 for values in parts):
        raise ValueError('Two chip-local metadata buffers required')
    program = ttnn.MeshProgramDescriptor()
    for chip in range(2):
        local = [values[chip] for values in parts]
        if local[0].buffer_address() == local[1].buffer_address():
            raise ValueError('Mask and position input must not alias')
        descriptor = ttnn.KernelDescriptor(kernel_source=str(Path(__file__).with_suffix('.cpp')), core_ranges=cores,
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


def execute(positions, mask, program):
    """Caller validates the ticket and owns zero-initialized mask, metadata and trace."""
    import ttnn

    ttnn.generic_op([positions, mask], program)
