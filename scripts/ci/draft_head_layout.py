"""Exact native DFlash2 head layout without generic sub-tile reshapes.

The drafter's heads are sharded over the chips: 16 query and four KV heads per chip at the pair, 8 and two at four cards
(tp_shapes.draft_heads, draft_kv_heads). Every count below is the served width's; at the pair each is the literal it was."""

import tp_shapes


def split_projected_heads(operations, query, key, value, retain):
    found = tp_shapes.active()
    query_count, kv_count = found.draft_heads, found.draft_kv_heads
    key_rows = key.shape[2] if len(key.shape) == 4 else 0
    if (tuple(query.shape) != (1, 1, 32, found.draft_query) or key_rows not in range(32, 2081, 32)
            or tuple(key.shape) != (1, 1, key_rows, kv_count * 128) or tuple(value.shape) != tuple(key.shape)
            or any(tensor.dtype != operations.bfloat16 or tensor.layout != operations.TILE_LAYOUT
                or tensor.memory_config() != operations.DRAM_MEMORY_CONFIG for tensor in (query, key, value))):
        raise ValueError('Padded BF16 DRAM projections for %d query and %s KV heads required'
                         % (query_count, tp_shapes.number_word(kv_count)))
    padded_query = query
    if key_rows != 32:
        padded_query = retain(operations.pad(query, [(0, 0), (0, 0), (0, key_rows - 32), (0, 0)], 0.0))
    combined_kv = retain(operations.concat([key, value], dim=3, memory_config=operations.DRAM_MEMORY_CONFIG))
    heads = operations.experimental.nlp_create_qkv_heads(padded_query, combined_kv,
        num_heads=query_count, num_kv_heads=kv_count, transpose_k_heads=False, memory_config=operations.DRAM_MEMORY_CONFIG)
    query_heads, key_heads, value_heads = (retain(tensor) for tensor in heads)
    if key_rows != 32:
        query_heads = retain(operations.slice(query_heads, (0, 0, 0, 0), (1, query_count, 32, 128)))
    return dict(q=query_heads, k=key_heads, v=value_heads)


def concatenate_query_heads(operations, value, retain):
    if (tuple(value.shape) != (1, tp_shapes.active().draft_heads, 32, 128) or value.dtype != operations.bfloat16
            or value.layout != operations.TILE_LAYOUT or value.memory_config() != operations.DRAM_MEMORY_CONFIG):
        raise ValueError('Padded BF16 DRAM query heads required')
    return retain(operations.experimental.nlp_concat_heads(value, memory_config=operations.DRAM_MEMORY_CONFIG))
