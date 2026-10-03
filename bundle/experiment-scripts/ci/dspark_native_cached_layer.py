"""Experimental learned cached layer with native SDPA; not full-request qualified."""

from dspark_full_attention import geometry
from dspark_native_full_attention import execute as attend
from dspark_layer import SPECIFICATIONS, finish, linear, mlp_partial, norm
from dspark_mesh import gather_partials
from dspark_projection import require_tensor
from dspark_rotary_device import execute as rotate


def append_queries(operations, history, queries, retain, *, position, proposals):
    padded_keys = geometry(position, proposals)[-1][1]
    require_tensor(operations, history, (1, 4, position, 128), operations.bfloat16)
    require_tensor(operations, queries, (1, 4, 32, 128), operations.bfloat16)
    valid = retain(operations.slice(queries, (0, 0, 0, 0), (1, 4, proposals, 128)))
    joined = retain(operations.concat([history, valid], dim=2, memory_config=operations.DRAM_MEMORY_CONFIG))
    if padded_keys > position + proposals:
        joined = retain(operations.pad(joined, [(0, 0), (0, 0), (0, padded_keys - position - proposals), (0, 0)], 0.0))
    return joined


def execute(operations, mesh, collectives, noise, history, weights, tables, mask, live, retain, *,
        position, proposals, mask_validated=False):
    geometry(position, proposals)
    if (list(mesh.shape) != [1, 2] or mask_validated is not True or not callable(retain)
            or set(weights) != set(SPECIFICATIONS) or len(history) != 2 or len(tables) != 2):
        raise ValueError('Complete learned TP2 layer, cached history, absolute tables and prevalidated mask required')
    require_tensor(operations, noise, (1, 1, 32, 5120), operations.bfloat16)
    require_tensor(operations, live, (1, 1, 32, 1), operations.float32)
    for value in history:
        require_tensor(operations, value, (1, 4, position, 128), operations.bfloat16)
    memory = operations.DRAM_MEMORY_CONFIG
    normalized = norm(operations, noise, weights['input_layernorm.weight'], retain)
    heads = {}
    for name, count in (('q', 16), ('k', 4), ('v', 4)):
        projected = linear(operations, normalized, weights[f'self_attn.{name}_proj.weight'], retain)
        shaped = retain(operations.reshape(projected, (1, 32, count, 128)))
        value = retain(operations.transpose(shaped, 1, 2))
        if name != 'v':
            value = norm(operations, value, weights[f'self_attn.{name}_norm.weight'], retain)
            owned = []
            try:
                value = rotate(operations, value, *tables, owned, composed=True)
            finally:
                for tensor in owned:
                    retain(tensor)
        if name != 'q':
            value = append_queries(operations, history[0 if name == 'k' else 1], value, retain,
                position=position, proposals=proposals)
        heads[name] = value
    owned = []
    try:
        attended = attend(operations, mesh, heads['q'], heads['k'], heads['v'], mask, owned,
            context_rows=position, proposals=proposals, mask_validated=True)
    finally:
        for value in owned:
            retain(value)
    transposed = retain(operations.transpose(attended, 1, 2))
    merged = retain(operations.reshape(transposed, (1, 1, 32, 2048)))
    partial = linear(operations, merged, weights['self_attn.o_proj.weight'], retain, rounded=False)
    partial = retain(operations.multiply(partial, live, dtype=operations.float32, fast_and_approximate_mode=False, memory_config=memory))
    parts = gather_partials(operations, mesh, collectives, partial, retain)
    mlp = mlp_partial(operations, *parts, noise, weights, retain)
    parts = gather_partials(operations, mesh, collectives, mlp['down_partial'], retain)
    result = finish(operations, *parts, mlp['attention_residual'], retain)
    return dict(attention=attended, heads=heads, mlp=mlp, finish=result)
