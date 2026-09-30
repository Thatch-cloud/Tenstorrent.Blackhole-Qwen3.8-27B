"""Simulator-only T32 native proposal attention; no target or serving admission."""

import os

from dflash_t16_native_attention import attention as native_attention


POLICY = 'dflash-t32-native-proposal-simulator-only'


def validate_mask(mask):
    import torch

    if (not isinstance(mask, torch.Tensor) or mask.device.type != 'cpu'
            or mask.dtype != torch.bfloat16 or mask.ndim != 4
            or tuple(mask.shape[:3]) != (1, 1, 32)
            or not 32 <= mask.shape[-1] <= 2080 or mask.shape[-1] % 32
            or not ((mask == 0) | torch.isneginf(mask)).all()
            or not (mask == 0).any(-1).all()):
        raise ValueError('All 32 live proposal queries require bounded zero/-inf masks with visible keys')


def attention(operations, query, key, value, mask, *, mask_validated=False):
    if (os.environ.get('QWEN_SIM_ONLY') != '1' or not os.environ.get('TT_METAL_SIMULATOR')
            or os.environ.get('QWEN_HARDWARE_TESTS') or os.environ.get('QWEN_CARDS_ALLOCATED')):
        raise ValueError('T32 native proposals remain simulator-only')
    return native_attention(operations, query, key, value, mask, mask_validated=mask_validated)


def numerical_difference(actual, reference):
    import torch

    if (tuple(actual.shape) != (1, 16, 32, 128) or tuple(reference.shape) != tuple(actual.shape)
            or actual.dtype != torch.bfloat16 or not torch.isfinite(actual).all()
            or not torch.isfinite(reference).all()):
        raise ValueError('Finite complete T32 output and reference required')
    current, expected = actual.float(), reference.float()
    difference = (current - expected).abs()
    return dict(max_abs=float(difference.max()), mean_abs=float(difference.mean()),
        rms=float(difference.square().mean().sqrt()), reference_max_abs=float(expected.abs().max()),
        legacy_close=bool(torch.isclose(current, expected, rtol=.01, atol=.01).all()))
