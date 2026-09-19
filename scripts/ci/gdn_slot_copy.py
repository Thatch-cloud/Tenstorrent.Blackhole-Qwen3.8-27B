"""Simulator-only candidate for copying one explicit P150-pair GDN user slot."""

import os
from pathlib import Path

from gdn_state_copy import transfer_counts


def transfer_plan(source_shapes, destination_shapes, slot):
    if type(slot) is not int or not 0 <= slot < 8:
        raise ValueError("Explicit native user slot in [0, 7] required")
    counts = transfer_counts(source_shapes, destination_shapes)
    source_slot = slot if tuple(source_shapes[0])[0] == 8 else 0
    destination_slot = slot if tuple(destination_shapes[0])[0] == 8 else 0
    return [(count, source_slot * (384 if index == 0 else 32),
             destination_slot * (384 if index == 0 else 32))
            for index, count in enumerate(counts)]


def copy_slot(source, destination, slot):
    if (os.environ.get("QWEN_SIM_ONLY") != "1"
            or os.environ.get("QWEN_HARDWARE_TESTS") == "1"
            or os.environ.get("QWEN_CARDS_ALLOCATED") == "1"):
        raise ValueError("Multi-slot state copy requires simulator qualification")
    _copy_state(source, destination, slot)


def _copy_state(source, destination, slot):
    import ttnn

    if len(source) != 5 or len(destination) != 5:
        raise ValueError("Expected complete state lists")
    source_shapes, destination_shapes = [tuple(tensor.shape) for tensor in source], [tuple(tensor.shape) for tensor in destination]
    plan = transfer_plan(source_shapes, destination_shapes, slot)
    tensors = [tensor for pair in zip(source, destination, strict=True) for tensor in pair]
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
        if len({tensor.buffer_address() for tensor in local}) != 10:
            raise ValueError("State-copy buffers must not alias")
        compile_args = [value for tensor in local for value in ttnn.TensorAccessorArgs(tensor).get_compile_time_args()]
        runtime = []
        for index, (count, source_offset, destination_offset) in enumerate(plan):
            runtime.extend([local[2 * index].buffer_address(), local[2 * index + 1].buffer_address(), count,
                source_offset, destination_offset])
        kernel = ttnn.KernelDescriptor(kernel_source=str(Path(__file__).with_suffix(".cpp")),
            core_ranges=cores, compile_time_args=compile_args,
            config=ttnn.DataMovementConfigDescriptor(processor=ttnn.DataMovementProcessor.RISCV_0,
                                                     noc=ttnn.NOC.RISCV_0_default))
        arguments = ttnn.RuntimeArgs()
        for worker in range(48):
            arguments[worker % 8][worker // 8] = runtime + [worker]
        kernel.runtime_args = arguments
        coordinate = ttnn.MeshCoordinate(0, chip)
        mesh_program[ttnn.MeshCoordinateRange(coordinate, coordinate)] = ttnn.ProgramDescriptor(kernels=[kernel], cbs=[buffer])
    ttnn.generic_op(tensors, mesh_program)
