"""Unqualified 31-query drafter attention using unchanged native SDPA arithmetic."""

from draft_attention import draft_sdpa, validate_attention
from dspark_projection import require_tensor
from dspark_t32_inputs import geometry


def append_queries(operations, history, queries, retain, *, position, proposals=31):
    padded = geometry(position, proposals)[-1][1]
    if not callable(retain):
        raise ValueError('Explicit output ownership required')
    require_tensor(operations, history, (1, 4, position, 128), operations.bfloat16)
    require_tensor(operations, queries, (1, 4, 32, 128), operations.bfloat16)
    valid = retain(operations.slice(queries, (0, 0, 0, 0), (1, 4, proposals, 128)))
    joined = retain(operations.concat([history, valid], dim=2, memory_config=operations.DRAM_MEMORY_CONFIG))
    if padded > position + proposals:
        joined = retain(operations.pad(joined, [(0, 0), (0, 0), (0, padded - position - proposals), (0, 0)], 0.0))
    return joined


def execute(operations, mesh, query, key, value, mask, owned, *, context_rows,
        proposals=31, mask_validated=False):
    padded = geometry(context_rows, proposals)[-1][1]
    if mesh is None or not isinstance(owned, list) or mask_validated is not True:
        raise ValueError('Explicit mesh, validated full-history mask and caller ownership required')
    validate_attention(operations, query, key, value, mask)
    if key.shape[2] != padded:
        raise ValueError('Complete padded history plus 31 query keys required')
    for tensor in (query, key, value, mask):
        require_tensor(operations, tensor, tuple(tensor.shape), operations.bfloat16)
    output = draft_sdpa(operations, query, key, value, mask, key_chunk_size=64)
    owned.append(output)
    return output
