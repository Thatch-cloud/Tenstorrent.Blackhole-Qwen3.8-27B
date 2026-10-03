"""T16 native gate/up/product with corrected DRAM-sharded down projection only."""

from dram_sharded_projection import execute as project
from gdn_multitoken_conv import release_owned


def execute(operations, source, weights, configurations, compute, retain):
    names = {'gate', 'up', 'down'}
    if set(weights) != names or set(configurations) != names:
        raise ValueError('Complete hybrid MLP weights and configurations required')
    if (tuple(source.shape) != (1, 1, 16, 5120) or source.dtype != operations.bfloat16
            or source.memory_config() != operations.L1_MEMORY_CONFIG):
        raise ValueError('T16 BF16 interleaved L1 input required')
    for name in ('gate', 'up'):
        weight = weights[name]
        if (tuple(weight.shape)[-2:] != (5120, 8704) or weight.dtype != operations.bfloat4_b
                or weight.memory_config() != operations.DRAM_MEMORY_CONFIG):
            raise ValueError('Unchanged native interleaved BF4 gate/up weights required')
    live = []
    def keep(value):
        live.append(value)
        return value
    try:
        gate = keep(operations.linear(source, weights['gate'], program_config=configurations['gate']['program'],
            compute_kernel_config=compute, memory_config=operations.L1_MEMORY_CONFIG))
        up = keep(operations.linear(source, weights['up'], program_config=configurations['up']['program'],
            compute_kernel_config=compute, memory_config=operations.L1_MEMORY_CONFIG))
        hidden = keep(operations.mul(gate, up, memory_config=operations.L1_MEMORY_CONFIG))
        release_owned(operations, [gate, up])
        live[:] = [value for value in live if value is not gate and value is not up]
        partial = project(operations, hidden, weights['down'], configurations['down'], compute,
            keep, preserve_partials=True)
        retain(partial)
        live[:] = [value for value in live if value is not partial]
        return partial
    finally:
        release_owned(operations, live)
