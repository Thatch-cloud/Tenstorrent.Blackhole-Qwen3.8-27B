"""Host layout oracle for token-to-query-head folding; not a serving adapter."""


def fold_query(query, kv_heads=2):
    if query.ndim != 4 or query.shape[0] != 1 or query.shape[2:] != (12, 256):
        raise ValueError('Expected native TP2 query shape [1,T,12,256]')
    if query.shape[1] not in (1, 2, 4, 8, 16, 32) or kv_heads != 2:
        raise ValueError('Bounded Qwen TP2 geometry required')
    rows = query.shape[1]
    return query.reshape(rows, kv_heads, 6, 256).permute(1, 0, 2, 3).reshape(1, 1, rows * 12, 256).contiguous()


def unfold_output(output, rows):
    if type(rows) is not int or rows not in (1, 2, 4, 8, 16, 32) or tuple(output.shape) != (1, 1, rows * 12, 256):
        raise ValueError('Expected folded native TP2 output')
    return output.reshape(2, rows, 6, 256).permute(1, 0, 2, 3).reshape(1, rows, 12, 256).contiguous()


def causal_mask(rows, start, capacity):
    import torch

    if type(rows) is not int or rows not in (1, 2, 4, 8, 16, 32):
        raise ValueError('Supported token count required')
    if any(type(value) is not int for value in (start, capacity)) or start < 0 or start + rows > capacity:
        raise ValueError('Valid positions within cache required')
    positions = torch.arange(start, start + rows).reshape(1, rows, 1).expand(2, rows, 6).reshape(-1)
    mask = torch.zeros(rows * 12, capacity, dtype=torch.bfloat16)
    mask.masked_fill_(torch.arange(capacity).unsqueeze(0) > positions.unsqueeze(1), float('-inf'))
    return mask.reshape(1, 1, rows * 12, capacity)
