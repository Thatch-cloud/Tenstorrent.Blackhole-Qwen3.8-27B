"""Batched raw-word readback for projection audits, including all physical tile rows."""

import math
from pathlib import Path


def raw_shape(padded_shape, tile_bytes):
    if (tuple(padded_shape[:2]) != (1, 1) or len(padded_shape) != 4
            or any(type(value) is not int or value < 32 or value % 32 for value in padded_shape[2:])
            or type(tile_bytes) is not int or tile_bytes not in (576, 1088, 2048)):
        raise ValueError('Native BF4/BF8/BF16 single-matrix tile geometry required')
    return (1, 1, math.prod(padded_shape) // 1024, tile_bytes // 4)


def copy_raw(operations, source, destination, tile_bytes):
    shape = raw_shape(tuple(source.padded_shape), tile_bytes)
    dtype = {576: operations.bfloat4_b, 1088: operations.bfloat8_b, 2048: operations.bfloat16}[tile_bytes]
    if (source.dtype != dtype or source.layout != operations.TILE_LAYOUT
            or source.memory_config() not in (operations.L1_MEMORY_CONFIG, operations.DRAM_MEMORY_CONFIG)
            or tuple(destination.shape) != shape or destination.dtype != operations.uint32
            or destination.layout != operations.ROW_MAJOR_LAYOUT or destination.memory_config() != operations.DRAM_MEMORY_CONFIG):
        raise ValueError('Typed tiled input and correctly sized raw DRAM sink required')
    source_shards, destination_shards = [operations.get_device_tensors(value) for value in (source, destination)]
    if (len(source_shards) != 2 or len(destination_shards) != 2
            or any(left.buffer_address() == right.buffer_address()
                for left, right in zip(source_shards, destination_shards, strict=True))):
        raise ValueError('Two disjoint source and sink shards required')
    workers = operations.CoreRangeSet([operations.CoreRange(operations.CoreCoord(0, 8), operations.CoreCoord(7, 8))])
    scratch = operations.CBDescriptor(total_size=8 * tile_bytes, core_ranges=workers,
        format_descriptors=[operations.CBFormatDescriptor(buffer_index=0, data_format=dtype, page_size=tile_bytes)])
    program = operations.MeshProgramDescriptor()
    for chip, (source_shard, destination_shard) in enumerate(zip(source_shards, destination_shards, strict=True)):
        arguments = operations.RuntimeArgs()
        for worker in range(8):
            arguments[worker][8] = [source_shard.buffer_address(), destination_shard.buffer_address(), shape[2], worker]
        compile_args = [tile_bytes]
        for value in (source_shard, destination_shard):
            compile_args.extend(operations.TensorAccessorArgs(value).get_compile_time_args())
        kernel = operations.KernelDescriptor(kernel_source=str(Path(__file__).with_name('tensix_projection_raw.cpp')),
            core_ranges=workers, compile_time_args=compile_args, runtime_args=arguments,
            config=operations.DataMovementConfigDescriptor(processor=operations.DataMovementProcessor.RISCV_0,
                noc=operations.NOC.RISCV_0_default))
        coordinate = operations.MeshCoordinate(0, chip)
        program[operations.MeshCoordinateRange(coordinate, coordinate)] = operations.ProgramDescriptor(kernels=[kernel], cbs=[scratch])
    operations.generic_op([source, destination], program)
    return destination
