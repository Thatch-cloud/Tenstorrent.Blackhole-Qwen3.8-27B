"""Experimental worker-prefetched projection using the unchanged native compute kernel."""

from pathlib import Path
from types import SimpleNamespace

from tensix_weight_stream import stream_geometry


COMPUTE = 'ttnn/cpp/ttnn/operations/matmul/device/kernels/compute/bmm_large_block_zm_fused_bias_activation.cpp'


def compute_arguments(geometry):
    if geometry != stream_geometry(geometry['projection'], geometry['blocks'], geometry.get('producers', 8)):
        raise ValueError('Exact reviewed transport geometry required')
    columns = geometry['per_receiver']
    return [8, 1, 8, 8, 1, 8 * columns, columns, geometry['blocks'], 1, 1, 1, columns, columns, 1, columns, 0, 0, 0]


def validate_inputs(operations, mesh, activation, weight, output, geometry):
    compute_arguments(geometry)
    grid = mesh.compute_with_storage_grid_size()
    if (grid.x, grid.y) != (11, 10):
        raise ValueError('Explicit 110-worker P150 geometry required')
    for value, shape, dtype, memory in (
            (activation, (1, 1, 8, geometry['rows']), operations.bfloat16, operations.L1_MEMORY_CONFIG),
            (weight, (1, 1, geometry['rows'], geometry['width']), getattr(operations, geometry['dtype']),
                operations.DRAM_MEMORY_CONFIG),
            (output, (1, 1, 8, geometry['width']), operations.bfloat16, operations.L1_MEMORY_CONFIG)):
        if (tuple(value.shape) != shape or value.dtype != dtype or value.layout != operations.TILE_LAYOUT
                or value.memory_config() != memory):
            raise ValueError('Native T8 shapes, compressed DRAM weights and BF16 L1 activations required')
    shards = [operations.get_device_tensors(value) for value in (activation, weight, output)]
    if any(len(values) != 2 for values in shards):
        raise ValueError('Exactly two local tensor shards required')
    if any(len({values[chip].buffer_address() for values in shards}) != 3 for chip in range(2)):
        raise ValueError('Activation, weight and output must be disjoint')
    return shards


def prepare_projection(operations, mesh, activation, weight, output, geometry, native_root):
    activation_shards, weight_shards, output_shards = validate_inputs(
        operations, mesh, activation, weight, output, geometry)
    compute_source = Path(native_root) / COMPUTE
    if not compute_source.is_file():
        raise ValueError('Pinned native matmul compute source required')
    def cores(coordinates):
        return operations.CoreRangeSet([operations.CoreRange(operations.CoreCoord(*coordinate),
            operations.CoreCoord(*coordinate)) for coordinate in coordinates])
    sender_coordinates = [coordinate for coordinate, unused in geometry['mapping']]
    senders = cores(sender_coordinates)
    receivers = cores(geometry['coordinates'])
    grid_rows = (geometry['receivers'] + 10) // 11
    activation_coordinates = [(index % 11, index // 11) for index in range(11 * grid_rows)]
    activation_cores = cores(activation_coordinates)
    mapping = [(operations.CoreCoord(*sender), cores([geometry['coordinates'][index] for index in indices]))
        for sender, indices in geometry['mapping']]
    gcb = operations.create_global_circular_buffer(mesh, mapping, 2 * geometry['page_bytes'])
    if gcb.sender_core_type() != 'worker':
        raise ValueError('Real Tensix weight producers required')
    weight_dtype = getattr(operations, geometry['dtype'])
    def buffer(index, dtype, page_bytes, count, core_set):
        return operations.CBDescriptor(total_size=page_bytes * count, core_ranges=core_set,
            format_descriptors=[operations.CBFormatDescriptor(buffer_index=index, data_format=dtype,
                page_size=page_bytes)])
    buffers = [buffer(0, operations.bfloat16, 2048, 16, activation_cores),
        buffer(4, operations.bfloat16, 2048, geometry['per_receiver'], receivers),
        buffer(5, operations.float32, 4096, geometry['per_receiver'], receivers),
        buffer(3, operations.bfloat16, 32, 1, senders)]
    for sender, indices in geometry['mapping']:
        buffers.append(buffer(0, weight_dtype, geometry['tile_bytes'],
            2 * len(indices) * 8 * geometry['per_receiver'], cores([sender])))
    for core_set, formats in ((senders, []), (receivers, [operations.CBFormatDescriptor(buffer_index=1,
            data_format=weight_dtype, page_size=geometry['tile_bytes'],
            tile=operations.TileDescriptor(operations.Tile([32, 32])))])):
        remote = operations.CBDescriptor(total_size=2 * geometry['page_bytes'], core_ranges=core_set,
            format_descriptors=formats)
        remote.remote_format_descriptors = [operations.CBFormatDescriptor(buffer_index=31,
            data_format=weight_dtype, page_size=geometry['page_bytes'])]
        remote.set_global_circular_buffer(gcb)
        buffers.append(remote)
    program = operations.MeshProgramDescriptor()
    common = [geometry['tile_bytes'], 8, geometry['per_receiver'], geometry['width'] // 32]
    for chip, (activation_shard, weight_shard, output_shard) in enumerate(zip(
            activation_shards, weight_shards, output_shards, strict=True)):
        activation_args, reader_args, writer_args, consumer_args = [operations.RuntimeArgs() for unused in range(4)]
        device = activation_shard.device()
        first = device.worker_core_from_logical_core(operations.CoreCoord(0, 0))
        last = device.worker_core_from_logical_core(operations.CoreCoord(10, grid_rows - 1))
        for worker, coordinate in enumerate(activation_coordinates):
            activation_args[coordinate[0]][coordinate[1]] = [activation_shard.buffer_address(), worker,
                first.x, first.y, last.x, last.y, geometry['receivers'], len(activation_coordinates), geometry['blocks']]
        for sender, indices in geometry['mapping']:
            reader_args[sender[0]][sender[1]] = [weight_shard.buffer_address(), geometry['blocks'], len(indices), *indices]
            writer_args[sender[0]][sender[1]] = [geometry['blocks'], len(indices)]
        for index, coordinate in enumerate(geometry['coordinates']):
            consumer_args[coordinate[0]][coordinate[1]] = [output_shard.buffer_address(), geometry['blocks'], index]
        kernels = []
        for filename, core_set, arguments, compile_args, processor, noc in (
                ('tensix_stream_activation.cpp', activation_cores, activation_args,
                    operations.TensorAccessorArgs(activation_shard).get_compile_time_args(),
                    operations.DataMovementProcessor.RISCV_1, operations.NOC.RISCV_1_default),
                ('tensix_weight_stream_reader.cpp', senders, reader_args,
                    common + operations.TensorAccessorArgs(weight_shard).get_compile_time_args(),
                    operations.DataMovementProcessor.RISCV_1, operations.NOC.RISCV_0_default),
                ('tensix_weight_stream_writer.cpp', senders, writer_args, common,
                    operations.DataMovementProcessor.RISCV_0, operations.NOC.RISCV_1_default),
                ('tensix_stream_matmul_sink.cpp', receivers, consumer_args,
                    [8 * geometry['per_receiver'], geometry['per_receiver']]
                    + operations.TensorAccessorArgs(output_shard).get_compile_time_args(),
                    operations.DataMovementProcessor.RISCV_0, operations.NOC.RISCV_0_default)):
            kernels.append(operations.KernelDescriptor(kernel_source=str(Path(__file__).with_name(filename)),
                core_ranges=core_set, compile_time_args=compile_args, runtime_args=arguments,
                config=operations.DataMovementConfigDescriptor(processor=processor, noc=noc)))
        compute_config = operations.ComputeConfigDescriptor(math_fidelity=operations.MathFidelity.LoFi,
            fp32_dest_acc_en=True, math_approx_mode=True)
        unpack_modes = [operations.UnpackToDestMode.Default] * 64
        unpack_modes[5] = operations.UnpackToDestMode.UnpackToDestFp32
        compute_config.unpack_to_dest_mode.extend(unpack_modes)
        named_args = [('cb_in0', 0), ('cb_in1', 1), ('cb_bias', 3), ('cb_out', 4), ('cb_intermed0', 5),
            ('cb_in0_transposed', 10), ('bias_ntiles', geometry['per_receiver'])]
        defines = [('FP32_DEST_ACC_EN', '1')]
        if geometry['blocks'] > 1:
            defines.append(('PACKER_L1_ACC', '1'))
        if geometry['projection'] == 'gate':
            defines.append(('SFPU_ACTIVATION', '1'))
            named_args.extend([('activation_type', 4), ('activation_param0', 0),
                ('activation_param1', 0), ('activation_param2', 0)])
        kernels.append(operations.KernelDescriptor(kernel_source=str(compute_source), core_ranges=receivers,
            compile_time_args=compute_arguments(geometry), named_compile_time_args=named_args,
            defines=defines, config=compute_config))
        coordinate = operations.MeshCoordinate(0, chip)
        program[operations.MeshCoordinateRange(coordinate, coordinate)] = operations.ProgramDescriptor(
            kernels=kernels, cbs=buffers, semaphores=[operations.SemaphoreDescriptor(id=index,
                core_ranges=activation_cores, initial_value=0) for index in (0, 1)])
    return SimpleNamespace(program=program, gcb=gcb, activation=activation, weight=weight, output=output, geometry=geometry)


def execute_projection(operations, prepared):
    operations.generic_op([prepared.activation, prepared.weight, prepared.output], prepared.program)
    return prepared.output
