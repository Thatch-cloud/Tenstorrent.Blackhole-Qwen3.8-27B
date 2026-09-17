"""Check the installed native descriptor bindings without opening devices."""

import ttnn

from dram_projection_reload import COMPUTE, preserve_fp32_partials


def main():
    config = ttnn.ComputeConfigDescriptor()
    config.fp32_dest_acc_en = True
    kernel = ttnn.KernelDescriptor()
    kernel.kernel_source = COMPUTE
    kernel.config = config
    buffer = ttnn.CBDescriptor()
    buffer.format_descriptors = [ttnn.CBFormatDescriptor(
        buffer_index=5, data_format=ttnn.float32, page_size=4096)]
    descriptor = ttnn.ProgramDescriptor()
    descriptor.kernels = [kernel]
    descriptor.cbs = [buffer]
    preserve_fp32_partials(ttnn, descriptor)
    expected = [ttnn.UnpackToDestMode.Default] * 64
    expected[5] = ttnn.UnpackToDestMode.UnpackToDestFp32
    if list(descriptor.kernels[0].config.unpack_to_dest_mode) != expected:
        raise AssertionError('Native descriptor did not retain the exact unpack policy')
    inputs = ttnn.MatmulInputs()
    inputs.optional_input_tensors = [None]
    if list(inputs.optional_input_tensors) != [None]:
        raise AssertionError('Native inputs did not retain the explicit no-bias slot')
    print('Native descriptor binding contract passed without opening devices', flush=True)


if __name__ == '__main__':
    main()
