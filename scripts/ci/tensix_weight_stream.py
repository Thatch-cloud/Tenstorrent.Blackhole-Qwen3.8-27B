"""Tensix-producer compressed-weight transport; not a qualified matmul or runtime."""

from pathlib import Path
from types import SimpleNamespace

from tiny_tile_matmul import PROJECTIONS


def stream_geometry(projection, blocks, producers=8):
    if projection not in PROJECTIONS:
        raise ValueError('Explicit target MLP projection required')
    inner, width, unused_cores, dtype, unused_silu = PROJECTIONS[projection]
    if type(blocks) is not int or not 1 <= blocks <= inner // 256:
        raise ValueError('Positive bounded eight-tile K-block count required')
    if type(producers) is not int or producers not in (8, 16):
        raise ValueError('Only the explicit eight or sixteen producer mappings are supported')
    receivers = 80 if projection == 'down' else 68
    tile_bytes = 1088 if dtype == 'bfloat8_b' else 576
    per_receiver = width // 32 // receivers
    coordinates = [(index % 11, index // 11) for index in range(receivers)]
    mapping = [((producer % 8, 9 - producer // 8), list(range(producer, receivers, producers)))
        for producer in range(producers)]
    geometry = dict(projection=projection, blocks=blocks, rows=blocks * 256, width=width, dtype=dtype,
        receivers=receivers, tile_bytes=tile_bytes, per_receiver=per_receiver, key_block_tiles=8,
        page_bytes=8 * per_receiver * tile_bytes, coordinates=coordinates, mapping=mapping,
        full_projection=blocks == inner // 256)
    if producers == 16:
        geometry['producers'] = producers
    return geometry


def prepare_stream(operations, mesh, source, output, geometry):
    grid = mesh.compute_with_storage_grid_size()
    if (grid.x, grid.y) != (11, 10):
        raise ValueError('Qualified 110-worker P150 geometry required')
    expected = stream_geometry(geometry['projection'], geometry['blocks'], geometry.get('producers', 8))
    if geometry != expected:
        raise ValueError('Stream geometry changed')
    if (tuple(source.shape) != (1, 1, geometry['rows'], geometry['width'])
            or source.dtype != getattr(operations, geometry['dtype']) or source.layout != operations.TILE_LAYOUT
            or source.memory_config() != operations.DRAM_MEMORY_CONFIG):
        raise ValueError('Unchanged compressed tiled DRAM weights required')
    tiles = geometry['rows'] // 32 * (geometry['width'] // 32)
    if (tuple(output.shape) != (1, 1, tiles, geometry['tile_bytes'] // 4)
            or output.dtype != operations.uint32 or output.layout != operations.ROW_MAJOR_LAYOUT
            or output.memory_config() != operations.DRAM_MEMORY_CONFIG):
        raise ValueError('Preallocated raw packed-word DRAM sink required')
    source_shards, output_shards = [operations.get_device_tensors(value) for value in (source, output)]
    if len(source_shards) != 2 or len(output_shards) != 2:
        raise ValueError('Exactly two source and sink shards required')
    if any(left.buffer_address() == right.buffer_address() for left, right in zip(source_shards, output_shards, strict=True)):
        raise ValueError('Source and sink must not alias')
    def cores(coordinates):
        return operations.CoreRangeSet([operations.CoreRange(operations.CoreCoord(*coordinate),
            operations.CoreCoord(*coordinate)) for coordinate in coordinates])
    sender_coordinates = [coordinate for coordinate, unused_receivers in geometry['mapping']]
    senders, receivers = cores(sender_coordinates), cores(geometry['coordinates'])
    mapping = [(operations.CoreCoord(*sender), cores([geometry['coordinates'][index] for index in indices]))
        for sender, indices in geometry['mapping']]
    gcb = operations.create_global_circular_buffer(mesh, mapping, 2 * geometry['page_bytes'])
    if gcb.sender_core_type() != 'worker':
        raise ValueError('This experiment requires real Tensix senders, not DRISC capability spoofing')
    dtype = getattr(operations, geometry['dtype'])
    buffers = []
    for sender, indices in geometry['mapping']:
        count = len(indices) * 8 * geometry['per_receiver']
        buffers.append(operations.CBDescriptor(total_size=2 * count * geometry['tile_bytes'], core_ranges=cores([sender]),
            format_descriptors=[operations.CBFormatDescriptor(buffer_index=0, data_format=dtype,
                page_size=geometry['tile_bytes'])]))
    buffers.append(operations.CBDescriptor(total_size=32, core_ranges=senders,
        format_descriptors=[operations.CBFormatDescriptor(buffer_index=3, data_format=operations.bfloat16, page_size=32)]))
    remote = operations.CBDescriptor(total_size=2 * geometry['page_bytes'],
        core_ranges=cores([*sender_coordinates, *geometry['coordinates']]), format_descriptors=[])
    remote.remote_format_descriptors = [operations.CBFormatDescriptor(buffer_index=31, data_format=dtype, page_size=16)]
    remote.set_global_circular_buffer(gcb)
    buffers.append(remote)
    program = operations.MeshProgramDescriptor()
    common = [geometry['tile_bytes'], 8, geometry['per_receiver'], geometry['width'] // 32]
    for chip, (input_shard, output_shard) in enumerate(zip(source_shards, output_shards, strict=True)):
        reader_args, writer_args, consumer_args = [operations.RuntimeArgs() for unused in range(3)]
        for sender, indices in geometry['mapping']:
            reader_args[sender[0]][sender[1]] = [input_shard.buffer_address(), geometry['blocks'], len(indices), *indices]
            writer_args[sender[0]][sender[1]] = [geometry['blocks'], len(indices)]
        for index, coordinate in enumerate(geometry['coordinates']):
            consumer_args[coordinate[0]][coordinate[1]] = [output_shard.buffer_address(), geometry['blocks'], index]
        kernels = []
        for filename, core_set, arguments, accessor, processor, noc in (
                ('tensix_weight_stream_reader.cpp', senders, reader_args, input_shard,
                    operations.DataMovementProcessor.RISCV_1, operations.NOC.RISCV_0_default),
                ('tensix_weight_stream_writer.cpp', senders, writer_args, None,
                    operations.DataMovementProcessor.RISCV_0, operations.NOC.RISCV_1_default),
                ('tensix_weight_stream_sink.cpp', receivers, consumer_args, output_shard,
                    operations.DataMovementProcessor.RISCV_0, operations.NOC.RISCV_0_default)):
            compile_args = common + ([] if accessor is None else operations.TensorAccessorArgs(accessor).get_compile_time_args())
            kernels.append(operations.KernelDescriptor(kernel_source=str(Path(__file__).with_name(filename)),
                core_ranges=core_set, compile_time_args=compile_args, runtime_args=arguments,
                config=operations.DataMovementConfigDescriptor(processor=processor, noc=noc)))
        coordinate = operations.MeshCoordinate(0, chip)
        program[operations.MeshCoordinateRange(coordinate, coordinate)] = operations.ProgramDescriptor(kernels=kernels, cbs=buffers)
    return SimpleNamespace(gcb=gcb, program=program, source=source, output=output, geometry=geometry)


def execute_stream(operations, prepared):
    operations.generic_op([prepared.source, prepared.output], prepared.program)
    return prepared.output
