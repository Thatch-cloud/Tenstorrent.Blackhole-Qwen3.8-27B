"""Full-history FP32 proposal attention using bounded existing SFPU kernels, not a sliding window."""

from draft_attention import validate_attention
from draft_dot import fused_dot
from draft_row_sum import row_sum
from dspark_projection import require_tensor


POLICY = 'Full-history global FP32 softmax across <=2048-key SFPU chunks; BF16 inputs/output; no history truncation'
CHUNK_KEYS = 2048
PROPOSALS = (7, 15)
MAX_CONTEXT = 8192


def geometry(context_rows, proposals):
    if (type(context_rows) is not int or not 1 <= context_rows <= MAX_CONTEXT
            or type(proposals) is not int or proposals not in PROPOSALS):
        raise ValueError('Complete history up to8192 rows and explicit seven/15-query geometry required')
    padded_keys = ((context_rows + proposals + 63) // 64) * 64
    return tuple((start, min(start + CHUNK_KEYS, padded_keys)) for start in range(0, padded_keys, CHUNK_KEYS))


def full_mask(context_rows, proposals):
    import torch

    padded_keys = geometry(context_rows, proposals)[-1][1]
    mask = torch.full((1, 1, 32, padded_keys), float('-inf'), dtype=torch.bfloat16)
    mask[:, :, :proposals, :context_rows + proposals] = 0
    mask[:, :, proposals:, context_rows] = 0
    return mask


def validate_mask(mask, context_rows, proposals):
    import torch

    expected = full_mask(context_rows, proposals)
    if (not isinstance(mask, torch.Tensor) or mask.device.type != 'cpu'
            or mask.dtype != torch.bfloat16 or mask.shape != expected.shape or not torch.equal(mask, expected)):
        raise ValueError('Every live query must see the entire history and every proposal key; padding only is masked')


def validate_inputs(operations, query, key, value, mask, context_rows, proposals, mask_validated):
    chunks = geometry(context_rows, proposals)
    if mask_validated is not True:
        raise ValueError('Validate the complete host mask before upload or capture')
    validate_attention(operations, query, key, value, mask)
    if key.shape[2] != chunks[-1][1]:
        raise ValueError('Complete padded historical and proposal keys required')
    for tensor in (query, key, value, mask):
        require_tensor(operations, tensor, tuple(tensor.shape), operations.bfloat16)
    return chunks


def execute(operations, mesh, query, key, value, mask, owned, *, context_rows, proposals, mask_validated=False):
    if mesh is None or not isinstance(owned, list):
        raise ValueError('Explicit mesh and caller-owned attention intermediates required')
    chunks = validate_inputs(operations, query, key, value, mask, context_rows, proposals, mask_validated)
    memory = operations.DRAM_MEMORY_CONFIG

    def retain(tensor):
        owned.append(tensor)
        return tensor

    wide_query = retain(operations.typecast(query, operations.float32))
    parts = []
    maximum = None
    for start, end in chunks:
        if len(chunks) == 1:
            local_key, local_value, local_mask = key, value, mask
        else:
            local_key = retain(operations.slice(key, (0, 0, start, 0), (1, 4, end, 128)))
            local_value = retain(operations.slice(value, (0, 0, start, 0), (1, 4, end, 128)))
            local_mask = retain(operations.slice(mask, (0, 0, 0, start), (1, 1, 32, end)))
        keys = retain(operations.repeat_interleave(local_key, 4, dim=1, memory_config=memory))
        values = retain(operations.repeat_interleave(local_value, 4, dim=1, memory_config=memory))
        keys = retain(operations.typecast(keys, operations.float32))
        values = retain(operations.typecast(values, operations.float32))
        scores = fused_dot(mesh, wide_query, keys, owned, cache_tiles=True)
        scaled = retain(operations.multiply(scores, 128 ** -0.5, dtype=operations.float32))
        wide_mask = retain(operations.typecast(local_mask, operations.float32))
        masked = retain(operations.add(scaled, wide_mask, dtype=operations.float32))
        local_maximum = retain(operations.max(masked, dim=-1, keepdim=True))
        maximum = local_maximum if maximum is None else retain(operations.maximum(maximum, local_maximum))
        parts.append(dict(masked=masked, values=values))
    total = None
    for part in parts:
        centered = retain(operations.subtract(part['masked'], maximum, dtype=operations.float32))
        exponentials = retain(operations.exp(centered, fast_and_approximate_mode=False))
        local_total = row_sum(mesh, exponentials, owned)
        total = local_total if total is None else retain(operations.add(total, local_total, dtype=operations.float32))
        part['exponentials'] = exponentials
    inverse = retain(operations.reciprocal(total))
    output = None
    for part in parts:
        probabilities = retain(operations.multiply(part['exponentials'], inverse, dtype=operations.float32))
        transposed = retain(operations.transpose(part['values'], -1, -2))
        partial = fused_dot(mesh, probabilities, transposed, owned, cache_tiles=True)
        output = partial if output is None else retain(operations.add(output, partial, dtype=operations.float32))
    return retain(operations.typecast(output, operations.bfloat16))
