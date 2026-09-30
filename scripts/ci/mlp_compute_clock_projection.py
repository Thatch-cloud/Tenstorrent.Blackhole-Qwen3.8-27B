"""Diagnostic-only local L1 sample storage for the three compute processors."""

from frozen_recipe_context import replace_once


def sample_memory(operations):
    point = operations.CoreCoord(0, 0)
    cores = operations.CoreRangeSet([operations.CoreRange(point, point)])
    shard = operations.ShardSpec(cores, [3, 64], operations.ShardOrientation.ROW_MAJOR)
    return operations.MemoryConfig(operations.TensorMemoryLayout.HEIGHT_SHARDED,
        operations.BufferType.L1, shard)


def validate_buffer(operations, mesh, tensor):
    if (tuple(tensor.shape) != (3, 64) or tensor.dtype != operations.uint32
            or tensor.layout != operations.ROW_MAJOR_LAYOUT or tensor.device() != mesh
            or tensor.memory_config() != sample_memory(operations)
            or len(operations.get_device_tensors(tensor)) != 2):
        raise ValueError('Caller-owned two-chip uint32 L1 pages on logical core (0,0) required')


def replacements():
    return (
        ("source_root=Path('/opt/tt-metal'), math_approx_mode=False):",
         "source_root=Path('/opt/tt-metal'), math_approx_mode=False, compute_samples=None):"),
        ('        self.math_approx_mode = math_approx_mode\n',
         '        if token_rows != 16 or pairs_per_worker != 3 or intermediates:\n'
         "            raise ValueError('Compute diagnostic requires the exact T16 three-pair projection')\n"
         '        self.compute_samples = compute_samples\n'
         '        self.math_approx_mode = math_approx_mode\n'),
        ('        self.compute = fused_compute(original, intermediates=intermediates, pairs_per_worker=pairs_per_worker)',
         '        from mlp_compute_clock import instrument\n'
         '        self.compute = instrument(fused_compute(original, intermediates=intermediates, pairs_per_worker=pairs_per_worker))'),
        ('        import ttnn\n',
         '        import ttnn\n'
         '        from mlp_compute_clock_projection import validate_buffer\n'
         '        validate_buffer(ttnn, self.mesh, self.compute_samples)\n'),
        ('            device = local_input.device()\n',
         '            sample_shard = ttnn.get_device_tensors(self.compute_samples)[chip]\n'
         '            if sample_shard.buffer_address() in (local_input.buffer_address(), local_weight.buffer_address(), local_output.buffer_address()):\n'
         "                raise ValueError('Sample storage must not alias numerical operands')\n"
         '            device = local_input.device()\n'),
        ('            program = ttnn.ProgramDescriptor(kernels=[input_kernel, writer, compute], cbs=buffers,',
         '            compute_args = ttnn.RuntimeArgs()\n'
         '            for core_x, core_y, begin, count in self.workers:\n'
         '                compute_args[core_x][core_y] = [sample_shard.buffer_address(), int(core_x == 0 and core_y == 0)]\n'
         '            compute.runtime_args = compute_args\n'
         '            program = ttnn.ProgramDescriptor(kernels=[input_kernel, writer, compute], cbs=buffers,'),
        ('ttnn.generic_op([value, self.weights, output], mesh_program)',
         'ttnn.generic_op([value, self.weights, output, self.compute_samples], mesh_program)'),
    )


def instrument_projection(source):
    result = source
    for before, after in replacements():
        result = replace_once(result, before, after)
    compile(result, 'compute_clock_projection.py', 'exec')
    if remove_projection(result) != source:
        raise ValueError('Diagnostic adapter changed unrelated projection source')
    return result


def remove_projection(source):
    for before, after in reversed(replacements()):
        source = replace_once(source, after, before)
    return source
