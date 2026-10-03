"""T16 DRAM MLP with shared input staging and sharded gate/up product."""

from dram_projection_reload import execute as project
from gdn_multitoken_conv import release_owned
from tiny_tile_matmul import PROJECTIONS


def execute(operations, source, weights, configurations, compute, retain):
    if set(weights) != set(PROJECTIONS) or set(configurations) != set(PROJECTIONS):
        raise ValueError('Complete gate/up/down configuration required')
    if tuple(source.shape) != (1, 1, 16, 5120) or source.dtype != operations.bfloat16:
        raise ValueError('T16 BF16 source required')
    for name, (inner, width, unused, dtype, activated) in PROJECTIONS.items():
        if (tuple(weights[name].shape) != (1, 1, inner, width)
                or weights[name].dtype != getattr(operations, dtype)
                or weights[name].memory_config() != configurations[name]['weights']):
            raise ValueError('Unchanged precision and exact native TP2 weight layout required')
    if configurations['gate']['inputs'] != configurations['up']['inputs']:
        raise ValueError('Gate/up must share input staging geometry')
    live = []
    def keep(value):
        live.append(value)
        return value
    def release(values):
        release_owned(operations, values)
        live[:] = [value for value in live if all(value is not released for released in values)]
    try:
        staged = keep(operations.to_memory_config(source, configurations['gate']['inputs']))
        gate = project(operations, staged, weights['gate'], configurations['gate'], compute, keep)
        up = project(operations, staged, weights['up'], configurations['up'], compute, keep)
        release([staged])
        if gate.memory_config() != up.memory_config():
            raise ValueError('Gate/up output shard layouts must match')
        hidden = keep(operations.mul(gate, up, memory_config=gate.memory_config()))
        release([gate, up])
        down_input = keep(operations.to_memory_config(hidden, configurations['down']['inputs']))
        if down_input is not hidden:
            release([hidden])
        partial = project(operations, down_input, weights['down'], configurations['down'], compute, keep)
        output = keep(operations.to_memory_config(partial, operations.L1_MEMORY_CONFIG))
        retain(output)
        live[:] = [value for value in live if value is not output]
        return output
    finally:
        release_owned(operations, live)
