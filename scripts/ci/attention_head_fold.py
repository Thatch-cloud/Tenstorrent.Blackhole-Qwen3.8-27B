"""Host layout oracle for token-to-query-head folding; not a serving adapter."""


def fold_query(query, kv_heads=2):
    if query.ndim != 4 or query.shape[0] != 1 or query.shape[2:] != (12, 256):
        raise ValueError('Expected native TP2 query shape [1,T,12,256]')
    if not 1 <= query.shape[1] <= 32 or kv_heads != 2:
        raise ValueError('Bounded Qwen TP2 geometry required')
    rows = query.shape[1]
    return query.reshape(rows, kv_heads, 6, 256).permute(1, 0, 2, 3).reshape(1, 1, rows * 12, 256).contiguous()


def unfold_output(output, rows):
    if type(rows) is not int or not 1 <= rows <= 32 or tuple(output.shape) != (1, 1, rows * 12, 256):
        raise ValueError('Expected folded native TP2 output')
    return output.reshape(2, rows, 6, 256).permute(1, 0, 2, 3).reshape(1, rows, 12, 256).contiguous()


def causal_mask(rows, start, capacity):
    import torch

    if type(rows) is not int or not 1 <= rows <= 32:
        raise ValueError('Supported token count required')
    if any(type(value) is not int for value in (start, capacity)) or start < 0 or start + rows > capacity:
        raise ValueError('Valid positions within cache required')
    positions = torch.arange(start, start + rows).reshape(1, rows, 1).expand(2, rows, 6).reshape(-1)
    mask = torch.zeros(rows * 12, capacity, dtype=torch.bfloat16)
    mask.masked_fill_(torch.arange(capacity).unsqueeze(0) > positions.unsqueeze(1), float('-inf'))
    return mask.reshape(1, 1, rows * 12, capacity)


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
