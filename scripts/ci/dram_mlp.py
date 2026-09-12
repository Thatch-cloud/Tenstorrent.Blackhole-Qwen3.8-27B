"""Experimental T16 DRAM-sharded MLP composition; no serving integration."""

from dram_sharded_projection import execute as project
from gdn_multitoken_conv import release_owned


def execute(operations, source, weights, configurations, compute, retain):
    names = {'gate', 'up', 'down'}
    if set(weights) != names or set(configurations) != names:
        raise ValueError('Complete gate/up/down weights and configurations required')
    if tuple(source.shape) != (1, 1, 16, 5120) or source.dtype != operations.bfloat16:
        raise ValueError('T16 target BF16 input required')
    live = []
    def projection(value, name):
        allocated = []
        def keep(tensor):
            allocated.append(tensor)
            return tensor
        result = None
        try:
            result = project(operations, value, weights[name], configurations[name],
                compute, keep, preserve_partials=True)
            live.append(result)
            return result
        finally:
            release_owned(operations, [tensor for tensor in allocated if tensor is not result])
    try:
        gate = projection(source, 'gate')
        up = projection(source, 'up')
        hidden = operations.mul(gate, up, memory_config=operations.L1_MEMORY_CONFIG)
        live.append(hidden)
        release_owned(operations, [gate, up])
        live[:] = [tensor for tensor in live if tensor is not gate and tensor is not up]
        partial = projection(hidden, 'down')
        retain(partial)
        live[:] = [tensor for tensor in live if tensor is not partial]
        return partial
    finally:
        release_owned(operations, live)
