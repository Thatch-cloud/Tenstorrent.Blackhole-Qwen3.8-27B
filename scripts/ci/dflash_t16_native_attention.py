"""Unqualified T16 proposal-only native attention candidate; never target attention."""

from draft_attention import draft_sdpa, validate_attention


POLICY = 'dflash-t16-native-proposal-only-unqualified'


def validate_mask(mask):
    import torch

    if (not isinstance(mask, torch.Tensor) or mask.device.type != 'cpu'
            or mask.dtype != torch.bfloat16 or mask.ndim != 4
            or tuple(mask.shape[:3]) != (1, 1, 32)
            or not 32 <= mask.shape[-1] <= 2080 or mask.shape[-1] % 32):
        raise ValueError('Host tiled BF16 T16 proposal mask required')
    if not ((mask == 0) | torch.isneginf(mask)).all():
        raise ValueError('Only zero and negative infinity mask values supported')
    if not (mask[..., :16, :] == 0).any(-1).all():
        raise ValueError('Every live T16 query needs a visible key')
    if not (mask[..., 16:, :] == 0).sum(-1).eq(1).all():
        raise ValueError('Every padded query must have exactly one visible key')


def attention(operations, query, key, value, mask, *, mask_validated=False):
    if mask_validated is not True:
        raise ValueError('Validate the actual T16 host mask before upload and replay')
    validate_attention(operations, query, key, value, mask)
    if not 32 <= key.shape[2] <= 2080:
        raise ValueError('Only bounded 2048-history T16 proposals supported')
    if any(tensor.layout != operations.TILE_LAYOUT
            or tensor.memory_config() != operations.DRAM_MEMORY_CONFIG
            for tensor in (query, key, value, mask)):
        raise ValueError('Interleaved tiled DRAM operands required')
    return draft_sdpa(operations, query, key, value, mask)


def numerical_difference(actual, reference):
    import torch

    if (tuple(actual.shape) != (1, 16, 32, 128)
            or tuple(reference.shape) != tuple(actual.shape)
            or actual.dtype != torch.bfloat16 or not torch.isfinite(actual).all()
            or not torch.isfinite(reference).all()):
        raise ValueError('Finite complete BF16 output and matching reference required')
    current, expected = actual[..., :16, :].float(), reference[..., :16, :].float()
    difference = (current - expected).abs()
    return dict(max_abs=float(difference.max()), mean_abs=float(difference.mean()),
        rms=float(difference.square().mean().sqrt()), reference_max_abs=float(expected.abs().max()),
        legacy_close=bool(torch.isclose(current, expected, rtol=.01, atol=.01).all()))
