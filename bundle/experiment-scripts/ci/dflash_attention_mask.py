"""Host-only T16 mask specialization; shared simulator-qualified attention stays unchanged."""

from draft_attention import draft_attention_mask as original_mask


def draft_attention_mask(context_rows, block_rows=8, **options):
    if block_rows != 16:
        return original_mask(context_rows, block_rows=block_rows, **options)
    if type(block_rows) is not int:
        raise ValueError('Integer draft block width required')
    mask = original_mask(context_rows, block_rows=32, **options)
    multiple = options.get('key_multiple', 32)
    key_rows = ((context_rows + block_rows + multiple - 1) // multiple) * multiple
    mask = mask[..., :key_rows].contiguous()
    mask[..., :16, context_rows + 16:] = float('-inf')
    mask[..., 16:, :] = float('-inf')
    mask[..., 16:, context_rows] = 0
    return mask
