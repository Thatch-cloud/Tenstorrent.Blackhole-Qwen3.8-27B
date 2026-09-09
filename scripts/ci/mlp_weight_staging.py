"""Unchanged compressed MLP weights staged into caller-owned L1 buffers."""

import math

from gdn_multitoken_conv import addresses
from tiny_mlp import execute
from tiny_tile_matmul import PROJECTIONS


def weight_budget(banks=110):
    if type(banks) is not int or banks != 110:
        raise ValueError('Qualified P150 worker-dispatch bank count required')
    sizes = {}
    for name, (inner, width, unused_cores, dtype, unused_activation) in PROJECTIONS.items():
        pages = (inner // 32) * (width // 32)
        page_bytes = 576 if dtype == 'bfloat4_b' else 1088
        sizes[name] = dict(pages=pages, page_bytes=page_bytes, bytes=pages * page_bytes,
            bytes_per_bank_upper_bound=math.ceil(pages / banks) * page_bytes)
    return dict(weights=sizes, total_bytes=sum(value['bytes'] for value in sizes.values()),
        weight_bytes_per_bank_upper_bound=sum(value['bytes_per_bank_upper_bound'] for value in sizes.values()),
        scope='Weight data only; excludes allocator alignment, live activations, static CBs and runtime reservations')


def stage_weights(operations, sources, destinations):
    if set(sources) != set(PROJECTIONS) or set(destinations) != set(PROJECTIONS):
        raise ValueError('All three target MLP matrices required')
    bindings = []
    for name, (inner, width, unused_cores, dtype, unused_activation) in PROJECTIONS.items():
        source, destination = sources[name], destinations[name]
        if any(tuple(value.shape) != (1, 1, inner, width) or value.dtype != getattr(operations, dtype)
                or value.layout != operations.TILE_LAYOUT for value in (source, destination)):
            raise ValueError('Unchanged native compressed weight shape, dtype and tile layout required')
        if source.memory_config() != operations.DRAM_MEMORY_CONFIG or destination.memory_config() != operations.L1_MEMORY_CONFIG:
            raise ValueError('DRAM sources and interleaved L1 destinations required')
        for value in (source, destination):
            binding = addresses(operations, value)
            if len(binding) != 2:
                raise ValueError('Exactly two weight shards required')
            bindings.append(binding)
    if any(len({binding[chip] for binding in bindings}) != len(bindings) for chip in range(2)):
        raise ValueError('All weight buffers must be disjoint')
    for name in PROJECTIONS:
        result = operations.copy(sources[name], destinations[name])
        if addresses(operations, result) != addresses(operations, destinations[name]):
            raise ValueError('Weight copy replaced caller-owned L1 storage')


def forward(operations, source, dram_weights, l1_weights, programs, compute, retain, *, mode):
    if mode not in ('dram', 'resident', 'staged'):
        raise ValueError('Explicit DRAM control, resident compute or staged full-cost mode required')
    if mode == 'staged':
        stage_weights(operations, dram_weights, l1_weights)
    return execute(operations, source, dram_weights if mode == 'dram' else l1_weights,
        programs, compute, retain, tiny=False)
