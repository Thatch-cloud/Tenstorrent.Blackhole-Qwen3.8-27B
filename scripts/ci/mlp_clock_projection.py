"""Diagnostic source adapter; not admitted into serving or hardware recipes."""

from frozen_recipe_context import replace_once
from mlp_clock_samples import SCRATCH


def replacements():
    return (
        ("source_root=Path('/opt/tt-metal'), math_approx_mode=False):",
         "source_root=Path('/opt/tt-metal'), math_approx_mode=False, sample_buffers=None):"),
        ('        self.math_approx_mode = math_approx_mode\n',
         '        from mlp_clock_projection import validate_buffers\n'
         '        validate_buffers(mesh, sample_buffers)\n'
         '        self.sample_buffers = sample_buffers\n'
         '        self.math_approx_mode = math_approx_mode\n'),
        ('        import ttnn\n',
         '        import ttnn\n'
         '        from mlp_clock_samples import instrument\n'
         '        from mlp_clock_projection import validate_buffers\n'
         '        validate_buffers(self.mesh, self.sample_buffers)\n'),
        ('        mesh_program = ttnn.MeshProgramDescriptor()\n',
         '        for index in ' + repr(tuple(SCRATCH.values())) + ':\n'
         '            buffers.append(ttnn.CBDescriptor(total_size=128, core_ranges=all_cores,\n'
         '                format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=index,\n'
         '                    data_format=ttnn.uint32, page_size=128)]))\n'
         '        mesh_program = ttnn.MeshProgramDescriptor()\n'),
        ('            device = local_input.device()\n',
         '            input_samples = ttnn.get_device_tensors(self.sample_buffers[0])[chip]\n'
         '            weight_samples = ttnn.get_device_tensors(self.sample_buffers[1])[chip]\n'
         '            device = local_input.device()\n'),
        ('kernel_source=str(Path(__file__).with_name("fused_1d_input.cpp")), core_ranges=all_cores,',
         'kernel_source=instrument(Path(__file__).with_name("fused_1d_input.cpp").read_text(), "input"),\n'
         '                source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE, core_ranges=all_cores,'),
        ('compile_time_args=ttnn.TensorAccessorArgs(local_input).get_compile_time_args(),',
         'compile_time_args=ttnn.TensorAccessorArgs(local_input).get_compile_time_args()\n'
         '                    + ttnn.TensorAccessorArgs(input_samples).get_compile_time_args(),'),
        ('first.x, first.y, last.x, last.y, len(self.workers), 11 * self.rows]',
         'first.x, first.y, last.x, last.y, len(self.workers), 11 * self.rows, input_samples.buffer_address()]'),
        ('kernel_source=str(Path(__file__).with_name("fused_1d_weights.cpp")), core_ranges=workers,',
         'kernel_source=instrument(Path(__file__).with_name("fused_1d_weights.cpp").read_text(), "weights"),\n'
         '                source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE, core_ranges=workers,'),
        ('+ ttnn.TensorAccessorArgs(local_output).get_compile_time_args(),',
         '+ ttnn.TensorAccessorArgs(local_output).get_compile_time_args()\n'
         '                                  + ttnn.TensorAccessorArgs(weight_samples).get_compile_time_args(),'),
        ('begin, count, output_tiles]\n',
         'begin, count, output_tiles, weight_samples.buffer_address()]\n'),
        ('ttnn.generic_op([value, self.weights, output], mesh_program)',
         'ttnn.generic_op([value, self.weights, output, *self.sample_buffers], mesh_program)'),
    )


def instrument_projection(source):
    candidate = source
    for before, after in replacements():
        candidate = replace_once(candidate, before, after)
    if remove_projection(candidate) != source:
        raise ValueError('Diagnostic adapter changed unrelated projection code')
    compile(candidate, 'mlp_clock_projection_candidate.py', 'exec')
    return candidate


def remove_projection(source):
    for before, after in reversed(replacements()):
        source = replace_once(source, after, before)
    return source


def validate_buffers(mesh, buffers):
    import ttnn

    if not isinstance(buffers, tuple) or len(buffers) != 2:
        raise ValueError('Two caller-owned persistent sample tensors required')
    addresses = set()
    for tensor, shape in zip(buffers, ((2, 32), (1, 32))):
        if (tuple(tensor.shape) != shape or tensor.dtype != ttnn.uint32
                or tensor.layout != ttnn.ROW_MAJOR_LAYOUT
                or tensor.memory_config() != ttnn.DRAM_MEMORY_CONFIG
                or tensor.device() != mesh):
            raise ValueError('Replicated uint32 DRAM row-major sample pages required')
        shards = ttnn.get_device_tensors(tensor)
        if len(shards) != 2:
            raise ValueError('Two-chip sample storage required')
        for chip, shard in enumerate(shards):
            identity = (chip, shard.buffer_address())
            if identity in addresses:
                raise ValueError('Input and weight sample pages must not alias')
            addresses.add(identity)
