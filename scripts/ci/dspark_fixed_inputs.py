"""Logical history masking within fixed-capacity DSpark proposal storage."""

from dspark_full_attention import geometry
from dspark_wide_target import query_inputs


def fixed_mask(position, capacity, proposals):
    import torch

    padded = geometry(capacity, proposals)[-1][1]
    if (type(position) is not int or not 1 <= position <= capacity
            or capacity % 32):
        raise ValueError('Valid logical frontier within tile-aligned fixed history required')
    mask = torch.full((1, 1, 32, padded), float('-inf'), dtype=torch.bfloat16)
    mask[:, :, :proposals, :position] = 0
    mask[:, :, :proposals, capacity:capacity + proposals] = 0
    mask[:, :, proposals:, capacity] = 0
    return mask


def validate_fixed_mask(mask, position, capacity, proposals):
    import torch

    expected = fixed_mask(position, capacity, proposals)
    if (not isinstance(mask, torch.Tensor) or mask.device.type != 'cpu' or mask.dtype != torch.bfloat16
            or tuple(mask.shape) != tuple(expected.shape) or not torch.equal(mask, expected)):
        raise ValueError('Only committed history and every proposal key may be visible; mask the storage gap')


def proposal_inputs(anchor, position, capacity, proposals, rotary):
    import torch

    mask = fixed_mask(position, capacity, proposals)
    validate_fixed_mask(mask, position, capacity, proposals)
    cosine = torch.ones(1, 1, 32, 128, dtype=torch.bfloat16)
    sine = torch.zeros_like(cosine)
    actual = rotary.tables(position, proposals)
    cosine[:, :, :proposals], sine[:, :, :proposals] = actual
    live = torch.zeros(1, 1, 32, 1, dtype=torch.float32)
    live[:, :, :proposals] = 1
    return dict(identifiers=query_inputs(anchor, position, proposals),
        anchor=torch.tensor([[[[anchor]]]], dtype=torch.int64), mask=mask, cosine=cosine, sine=sine, live=live)
