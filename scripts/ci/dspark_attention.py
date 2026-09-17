"""Separate full-context DSpark proposal attention; no sliding window or target-model replacement."""

from draft_attention import draft_sdpa, validate_attention


POLICY = 'Native BF16 full-context DSpark proposal attention; FP32 CPU rtol=atol=0.01'
CONTEXTS = (31, 4096)


def full_mask(context_rows, *, key_multiple=32):
    import torch

    if (type(context_rows) is not int or not 1 <= context_rows <= 262137
            or type(key_multiple) is not int or key_multiple not in (32, 64)):
        raise ValueError('Bounded full context with seven proposal keys required')
    live_keys = context_rows + 7
    padded_keys = (live_keys + key_multiple - 1) // key_multiple * key_multiple
    mask = torch.full((1, 1, 32, padded_keys), float('-inf'), dtype=torch.bfloat16)
    mask[..., :7, :live_keys] = 0
    mask[..., 7:, context_rows] = 0
    return mask


def validate_mask(mask, context_rows, *, key_multiple=32):
    import torch

    expected = full_mask(context_rows, key_multiple=key_multiple)
    if (not isinstance(mask, torch.Tensor) or mask.device.type != 'cpu' or mask.dtype != torch.bfloat16
            or mask.shape != expected.shape or not torch.equal(mask, expected)):
        raise ValueError('Every live query must see all historical and seven proposal keys; only padding is masked')


def execute(operations, query, key, value, mask, *, context_rows, mask_validated=False, key_chunk_size=32):
    if (mask_validated is not True or type(context_rows) is not int or not 1 <= context_rows <= 262137
            or type(key_chunk_size) is not int or key_chunk_size not in (32, 64)):
        raise ValueError('Validate the full-context host mask before upload or capture')
    validate_attention(operations, query, key, value, mask)
    expected_keys = (context_rows + 7 + key_chunk_size - 1) // key_chunk_size * key_chunk_size
    if (key.shape[2] != expected_keys or any(tensor.layout != operations.TILE_LAYOUT
            or tensor.memory_config() != operations.DRAM_MEMORY_CONFIG for tensor in (query, key, value, mask))):
        raise ValueError('Full-context padded DRAM tile operands required, not a truncated history')
    return draft_sdpa(operations, query, key, value, mask, key_chunk_size=key_chunk_size)


def fixtures(context_rows, *, key_multiple=32):
    import torch

    if (context_rows not in CONTEXTS or type(context_rows) is not int
            or type(key_multiple) is not int or key_multiple not in (32, 64)):
        raise ValueError('Declared short and 4K attention fixtures required')
    generator = torch.Generator().manual_seed(382700 + context_rows)
    mask = full_mask(context_rows)
    patterns = [[*(torch.randn(shape, generator=generator).bfloat16() for shape in
        ((2, 16, 32, 128), (2, 4, mask.shape[-1], 128), (2, 4, mask.shape[-1], 128))), mask.clone()]
        for _ in range(2)]
    padded, oldest, future = ([value.clone() for value in patterns[0]] for _ in range(3))
    for value in padded[1:3]:
        value[..., context_rows + 7:, :] = (value[..., context_rows + 7:, :].float() * -3 + 17).bfloat16()
    oldest[2][..., 0, :] += 64
    future[2][..., context_rows + 6, :] += 64
    result = [*patterns, padded, oldest, future]
    aligned_mask = full_mask(context_rows, key_multiple=key_multiple)
    extra_keys = aligned_mask.shape[-1] - mask.shape[-1]
    if extra_keys:
        for pattern, values in enumerate(result):
            for index in (1, 2):
                values[index] = torch.nn.functional.pad(values[index], (0, 0, 0, extra_keys),
                    value=17 if pattern == 2 else 0)
            values[3] = aligned_mask.clone()
    return result


def reference(values, chip):
    import torch

    query, key, value, mask = values
    return torch.nn.functional.scaled_dot_product_attention(query[chip:chip + 1].float(),
        key[chip:chip + 1].float().repeat_interleave(4, dim=1),
        value[chip:chip + 1].float().repeat_interleave(4, dim=1), attn_mask=mask.float(), is_causal=False)


def numerical_difference(actual, expected):
    import torch

    if (tuple(actual.shape) != (1, 16, 32, 128) or expected.shape != actual.shape
            or actual.dtype != torch.bfloat16 or expected.dtype != torch.float32
            or not torch.isfinite(actual).all() or not torch.isfinite(expected).all()):
        raise ValueError('Finite full padded BF16 output and matching FP32 CPU reference required')
    difference = (actual.float() - expected).abs()
    failed = int((~torch.isclose(actual.float(), expected, rtol=.01, atol=.01)).sum())
    return dict(full_padded_close=failed == 0, failed_elements=failed, max_abs=float(difference.max()),
        valid_max_abs=float(difference[..., :7, :].max()))
