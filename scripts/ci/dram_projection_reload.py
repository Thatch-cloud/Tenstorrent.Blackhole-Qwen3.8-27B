"""Experimental FP32 partial reload on a native no-bias matmul descriptor."""


COMPUTE = 'ttnn/cpp/ttnn/operations/matmul/device/kernels/compute/bmm_large_block_zm_fused_bias_activation.cpp'


def preserve_fp32_partials(operations, descriptor):
    matches = [kernel for kernel in descriptor.kernels if str(kernel.kernel_source) == COMPUTE]
    if len(matches) != 1:
        raise ValueError('Exactly one known native matmul compute descriptor required')
    kernel = matches[0]
    config = kernel.config
    if not config.fp32_dest_acc_en:
        raise ValueError('FP32 destination accumulation required')
    buffers = [format for buffer in descriptor.cbs for format in buffer.format_descriptors
        if format.buffer_index == 5]
    expected_format = operations.CBFormatDescriptor(
        buffer_index=5, data_format=operations.float32, page_size=4096)
    if len(buffers) != 1 or buffers[0].data_format_as_uint8 != expected_format.data_format_as_uint8:
        raise ValueError('Native Float32 partials CB5 required')
    modes = list(config.unpack_to_dest_mode)
    if modes and (len(modes) != 64 or any(mode != operations.UnpackToDestMode.Default for mode in modes)):
        raise ValueError('Unexpected existing unpack policy')
    modes = [operations.UnpackToDestMode.Default] * 64
    modes[5] = operations.UnpackToDestMode.UnpackToDestFp32
    config.unpack_to_dest_mode = modes
    kernel.config = config
    return descriptor


def matmul_inputs(operations, source, weight):
    inputs = operations.MatmulInputs()
    inputs.input_tensors = [source, weight]
    inputs.optional_input_tensors = [None]
    return inputs


def execute(operations, source, weight, config, compute, retain):
    parameters = operations.MatmulParams()
    parameters.program_config = config['program']
    parameters.output_mem_config = config['outputs']
    parameters.output_dtype = operations.bfloat16
    parameters.compute_kernel_config = compute
    attributes = operations.create_matmul_attributes(source, weight, parameters, [])
    inputs = matmul_inputs(operations, source, weight)
    outputs = operations.MatmulDeviceOperation.create_output_tensors(attributes, inputs)
    if len(outputs) != 1:
        raise ValueError('One native output required')
    output = retain(outputs[0])
    shards = [operations.get_device_tensors(value) for value in (source, weight, output)]
    if any(len(values) != 2 for values in shards):
        raise ValueError('Two-chip descriptor correction required')
    factory = operations._ttnn.operations.matmul.MatmulMultiCoreReuseMultiCastDRAMShardedProgramFactory
    program = operations.MeshProgramDescriptor()
    for chip in range(2):
        local = matmul_inputs(operations, shards[0][chip], shards[1][chip])
        descriptor = factory.create_descriptor(attributes, local, [shards[2][chip]])
        preserve_fp32_partials(operations, descriptor)
        coordinate = operations.MeshCoordinate(0, chip)
        program[operations.MeshCoordinateRange(coordinate, coordinate)] = descriptor
    operations.generic_op([source, weight, output], program)
    return output
