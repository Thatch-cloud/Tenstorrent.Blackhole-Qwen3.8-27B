"""Copy slot zero between interleaved BF16 state and compact snapshots, without arithmetic."""

import os
from pathlib import Path


FLAG = 'QWEN_FAST_GDN_STATE_COPY_BATCH'


def batch_enabled(environ=None):
    """The opt-in flag for carrying several transfers in one launch.

    Default OFF; anything but '0' or '1' is a configuration error rather than a
    silent fallback, matching gdn_user_batch.enabled(). The transfers themselves
    are identical either way - the same addresses, the same page counts, the same
    order - so this only decides how many dispatches pay for them.
    """
    value = (os.environ if environ is None else environ).get(FLAG, '0')
    if value not in ('0', '1'):
        raise ValueError('%s must be 0 or 1' % FLAG)
    return value == '1'


def page_counts(shapes):
    if len(shapes) != 5 or tuple(shapes[0]) != (1, 24, 128, 128):
        raise ValueError("Expected one TP2 slot of recurrent state and four conv taps")
    if any(tuple(shape) != (1, 1, 5120) for shape in shapes[1:]):
        raise ValueError("Expected frozen TP2 convolution channel shape")
    return [384, 160, 160, 160, 160]


def transfer_counts(source_shapes, destination_shapes, compact_only=False):
    compact_shapes = [(1, 24, 128, 128)] + [(1, 1, 5120)] * 4
    full_shapes = [(8, 24, 128, 128)] + [(1, 8, 5120)] * 4
    source_shapes = [tuple(shape) for shape in source_shapes]
    destination_shapes = [tuple(shape) for shape in destination_shapes]
    allowed = [(compact_shapes, compact_shapes)] if compact_only else [
        (compact_shapes, full_shapes), (full_shapes, compact_shapes)]
    if (source_shapes, destination_shapes) not in allowed:
        raise ValueError("Expected frozen compact-to-compact or active-slot transfer geometry")
    return page_counts(compact_shapes)


def copy_active(source, destination):
    _copy_state([(source, destination)], compact_only=False)


def copy_compact(source, destination):
    _copy_state([(source, destination)], compact_only=True)


def copy_compact_batch(transfers):
    """Several compact-to-compact transfers in ONE launch.

    Each entry is a (source, destination) pair of five-tensor state lists, exactly
    what `copy_compact` takes one at a time. Every pair moves the same bytes to the
    same place as the separate calls would, in the same order; only the dispatch
    count changes, from one launch per pair to one for all of them. The packed
    decode's per-user carried state is the case this exists for: the earlier
    segments' state moves cost one launch each per GDN layer, 48 layers deep, for
    transfers that already share a grid, a kernel and a page layout.
    """
    if not transfers:
        raise ValueError("At least one transfer required")
    _copy_state(list(transfers), compact_only=True)


def _copy_state(transfers, compact_only):
    import ttnn

    counts, tensors = None, []
    for source, destination in transfers:
        if len(source) != 5 or len(destination) != 5:
            raise ValueError("Expected complete state lists")
        source_shapes = [tuple(tensor.shape) for tensor in source]
        destination_shapes = [tuple(tensor.shape) for tensor in destination]
        group_counts = transfer_counts(source_shapes, destination_shapes, compact_only)
        if counts is not None and group_counts != counts:
            raise ValueError("Batched transfers must share one page layout")
        counts = group_counts
        # Plain zip, not strict=: both lists are checked to be exactly five above, so
        # there is nothing left for strict= to catch, and dropping it lets the CPU
        # test suite exercise this on the 3.7 host as well as the container's 3.10.
        tensors.extend(tensor for pair in zip(source, destination) for tensor in pair)
    if any(tensor.dtype != ttnn.bfloat16 or tensor.layout != ttnn.TILE_LAYOUT
           or tensor.memory_config() != ttnn.DRAM_MEMORY_CONFIG for tensor in tensors):
        raise ValueError("Only interleaved DRAM BF16 tiles supported")
    shards = [ttnn.get_device_tensors(tensor) for tensor in tensors]
    if any(len(local) != 2 for local in shards):
        raise ValueError("Both chips required")
    cores = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 5))])
    buffer = ttnn.CBDescriptor(total_size=2048, core_ranges=cores,
        format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=0, data_format=ttnn.bfloat16,
            page_size=2048, tile=ttnn.TileDescriptor(ttnn.Tile([32, 32])))])
    mesh_program = ttnn.MeshProgramDescriptor()
    for chip in range(2):
        local = [parts[chip] for parts in shards]
        if len({tensor.buffer_address() for tensor in local}) != len(local):
            raise ValueError("State-copy buffers must not alias")
        # The kernel resolves its accessor specs at COMPILE time and reuses that one
        # set for every group, so a group may only ride along if its tensors carry
        # identical accessor arguments. Same shapes, same dtype, same memory config,
        # so they do - checked here rather than assumed, because a mismatch would
        # read the right addresses through the wrong layout and corrupt state with
        # no error anywhere.
        groups = [[value for tensor in local[10 * index:10 * index + 10]
                   for value in ttnn.TensorAccessorArgs(tensor).get_compile_time_args()]
                  for index in range(len(transfers))]
        if any(group != groups[0] for group in groups[1:]):
            raise ValueError("Batched transfers must share one tensor accessor layout")
        runtime = []
        for index in range(len(transfers)):
            base = 10 * index
            for slot, count in enumerate(counts):
                runtime.extend([local[base + 2 * slot].buffer_address(),
                                local[base + 2 * slot + 1].buffer_address(), count])
        kernel = ttnn.KernelDescriptor(kernel_source=str(Path(__file__).with_suffix(".cpp")),
            core_ranges=cores, compile_time_args=groups[0],
            config=ttnn.DataMovementConfigDescriptor(processor=ttnn.DataMovementProcessor.RISCV_0,
                                                     noc=ttnn.NOC.RISCV_0_default))
        arguments = ttnn.RuntimeArgs()
        for worker in range(48):
            arguments[worker % 8][worker // 8] = [len(transfers), worker] + runtime
        kernel.runtime_args = arguments
        coordinate = ttnn.MeshCoordinate(0, chip)
        mesh_program[ttnn.MeshCoordinateRange(coordinate, coordinate)] = ttnn.ProgramDescriptor(kernels=[kernel], cbs=[buffer])
    ttnn.generic_op(tensors, mesh_program)
