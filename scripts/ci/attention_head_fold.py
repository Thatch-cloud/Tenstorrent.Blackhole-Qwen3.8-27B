"""Host layout oracle for token-to-query-head folding; not a serving adapter.

The head layout is the width this process serves at (tp_shapes): 12 query heads on 2 KV heads per chip at the pair, 6 on
1 at four cards. At the pair every function is what it was."""

import tp_shapes


def fold_query(query, kv_heads=None):
    found = tp_shapes.active()
    heads = found.attn_fold_rows
    kv_heads = found.attn_kv_heads if kv_heads is None else kv_heads
    if query.ndim != 4 or query.shape[0] != 1 or query.shape[2:] != (heads, 256):
        raise ValueError('Expected native TP%d query shape [1,T,%d,256]' % (found.tp, heads))
    if not 1 <= query.shape[1] <= 32 or kv_heads != found.attn_kv_heads:
        raise ValueError('Bounded Qwen TP%d geometry required' % found.tp)
    rows = query.shape[1]
    return query.reshape(rows, kv_heads, 6, 256).permute(1, 0, 2, 3).reshape(1, 1, rows * heads, 256).contiguous()


def unfold_output(output, rows):
    found = tp_shapes.active()
    heads = found.attn_fold_rows
    if type(rows) is not int or not 1 <= rows <= 32 or tuple(output.shape) != (1, 1, rows * heads, 256):
        raise ValueError('Expected folded native TP%d output' % found.tp)
    return output.reshape(found.attn_kv_heads, rows, 6, 256).permute(1, 0, 2, 3).reshape(1, rows, heads, 256).contiguous()


def causal_mask(rows, start, capacity):
    import torch

    if type(rows) is not int or not 1 <= rows <= 32:
        raise ValueError('Supported token count required')
    if any(type(value) is not int for value in (start, capacity)) or start < 0 or start + rows > capacity:
        raise ValueError('Valid positions within cache required')
    found = tp_shapes.active()
    positions = torch.arange(start, start + rows).reshape(1, rows, 1).expand(found.attn_kv_heads, rows, 6).reshape(-1)
    mask = torch.zeros(rows * found.attn_fold_rows, capacity, dtype=torch.bfloat16)
    mask.masked_fill_(torch.arange(capacity).unsqueeze(0) > positions.unsqueeze(1), float('-inf'))
    return mask.reshape(1, 1, rows * found.attn_fold_rows, capacity)


def chunk_groups(start, rows, *, max_chunk_tiles=8, max_group_rows=4):
    if any(type(value) is not int for value in (start, rows, max_chunk_tiles, max_group_rows)):
        raise ValueError('Integer chunk geometry required')
    if start < 0 or not 1 <= rows <= 32 or max_chunk_tiles not in (4, 8) or not 1 <= max_group_rows <= 32:
        raise ValueError('Bounded native SDPA geometry required')
    groups = []
    for offset in range(rows):
        position = start + offset
        tiles = position // 32 + 1
        chunk_size = min(max_chunk_tiles, 1 << (tiles - 1).bit_length()) * 32
        capacity = ((position + 1 + chunk_size - 1) // chunk_size) * chunk_size
        signature = (chunk_size, capacity)
        if groups and groups[-1]['signature'] == signature and groups[-1]['rows'] < max_group_rows:
            groups[-1]['rows'] += 1
        else:
            groups.append(dict(offset=offset, rows=1, signature=signature))
    return groups


def parallel_groups(start, rows, *, max_batches=3, max_group_rows=4):
    if type(max_batches) is not int or not 1 <= max_batches <= 3:
        raise ValueError('At most three TP2 groups preserve sixteen workers per KV head on110 cores')
    if type(max_group_rows) is not int or max_group_rows not in (4, 8):
        raise ValueError('Parallel groups require a four or eight row limit')
    bundles = []
    for group in chunk_groups(start, rows, max_group_rows=max_group_rows):
        signature = (group['rows'], group['signature'])
        if bundles and len(bundles[-1]) < max_batches and (bundles[-1][0]['rows'], bundles[-1][0]['signature']) == signature:
            bundles[-1].append(group)
        else:
            bundles.append([group])
    return bundles


def device_layout(operations, tensor, rows, owned, *, inverse=False, offset=0):
    if type(inverse) is not bool or type(rows) is not int or not 1 <= rows <= 8:
        raise ValueError('Explicit direction and one to eight query rows required')
    if type(offset) is not int or offset < 0 or (inverse and offset):
        raise ValueError('Valid input-row offset required')
    shape = tuple(tensor.shape)
    found = tp_shapes.active()
    heads = found.attn_fold_rows
    if inverse:
        valid = shape == (1, 1, rows * heads, 256)
    else:
        valid = len(shape) == 4 and shape[0] == 1 and shape[2:] == (heads, 256) and offset + rows <= shape[1]
    if not valid or tensor.dtype != operations.bfloat16 or tensor.layout != operations.TILE_LAYOUT:
        raise ValueError('Native BF16 tiled attention geometry required')

    def keep(value):
        owned.append(value)
        return value

    selected = tensor if inverse else keep(operations.slice(tensor, (0, offset, 0, 0), (1, offset + rows, heads, 256),
        memory_config=operations.DRAM_MEMORY_CONFIG))
    linear = keep(operations.to_layout(selected, operations.ROW_MAJOR_LAYOUT, memory_config=operations.DRAM_MEMORY_CONFIG))
    shaped = keep(operations.reshape(linear, (found.attn_kv_heads, rows, 6, 256) if inverse else (rows, found.attn_kv_heads, 6, 256)))
    transposed = keep(operations.permute(shaped, (1, 0, 2, 3), memory_config=operations.DRAM_MEMORY_CONFIG))
    final = keep(operations.reshape(transposed, (1, rows, heads, 256) if inverse else (1, 1, rows * heads, 256)))
    return keep(operations.to_layout(final, operations.TILE_LAYOUT, memory_config=operations.DRAM_MEMORY_CONFIG))
