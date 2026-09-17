"""Unmodified native BF16 attention for a separate approximate-proposal experiment."""

from draft_attention import draft_sdpa, validate_attention
from draft_live_qk import validate_live_qk_mask


POLICY = 'native-bf16-proposal-only'


def validate_mask(mask):
    validate_live_qk_mask(mask)


def attention(operations, query, key, value, mask, *, mask_validated=False):
    if mask_validated is not True:
        raise ValueError('Validate the actual host proposal mask before upload or capture')
    validate_attention(operations, query, key, value, mask)
    if not 32 <= key.shape[2] <= 2080:
        raise ValueError('Only the bounded T8 proposal history is supported')
    if any(tensor.layout != operations.TILE_LAYOUT or tensor.memory_config() != operations.DRAM_MEMORY_CONFIG
            for tensor in (query, key, value, mask)):
        raise ValueError('Interleaved tiled DRAM proposal operands required')
    return draft_sdpa(operations, query, key, value, mask)


def numerical_difference(actual, reference):
    import torch

    if (tuple(actual.shape) != (1, 16, 32, 128) or tuple(reference.shape) != tuple(actual.shape)
            or actual.dtype != torch.bfloat16 or not torch.isfinite(actual).all()
            or not torch.isfinite(reference).all()):
        raise ValueError('Finite complete BF16 proposal output and matching reference required')
    current, expected = actual[..., :8, :].float(), reference[..., :8, :].float()
    difference = (current - expected).abs()
    return dict(max_abs=float(difference.max()), mean_abs=float(difference.mean()),
        rms=float(difference.square().mean().sqrt()), reference_max_abs=float(expected.abs().max()),
        legacy_close=bool(torch.isclose(current, expected, rtol=.01, atol=.01).all()))
