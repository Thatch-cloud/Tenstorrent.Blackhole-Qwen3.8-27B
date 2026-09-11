"""Host inputs for an unqualified T32 experiment; not admitted by the live drafter."""

from dspark_inputs import MASK_TOKEN, VOCABULARY


PROPOSALS = 31


def geometry(capacity, proposals=PROPOSALS):
    if (type(capacity) is not int or not 1 <= capacity <= 8192
            or type(proposals) is not int or proposals != PROPOSALS):
        raise ValueError('Explicit 31-query complete-history geometry required')
    padded = ((capacity + proposals + 63) // 64) * 64
    return tuple((start, min(start + 2048, padded)) for start in range(0, padded, 2048))


def fixed_mask(position, capacity):
    import torch

    if (type(position) is not int or type(capacity) is not int
            or not 1 <= position <= capacity <= 8192 or capacity % 32):
        raise ValueError('Committed frontier in tile-aligned full history required')
    padded = geometry(capacity)[-1][1]
    mask = torch.full((1, 1, 32, padded), float('-inf'), dtype=torch.bfloat16)
    mask[:, :, :PROPOSALS, :position] = 0
    mask[:, :, :PROPOSALS, capacity:capacity + PROPOSALS] = 0
    mask[:, :, PROPOSALS:, capacity] = 0
    return mask


def proposal_inputs(anchor, position, capacity, rotary):
    import torch

    if type(anchor) is not int or not 0 <= anchor < VOCABULARY:
        raise ValueError('Valid global anchor required')
    mask = fixed_mask(position, capacity)
    identifiers = torch.full((1, PROPOSALS), MASK_TOKEN, dtype=torch.int64)
    identifiers[0, 0] = anchor
    cosine = torch.ones(1, 1, 32, 128, dtype=torch.bfloat16)
    sine = torch.zeros_like(cosine)
    tables = rotary.tables(position, PROPOSALS)
    if len(tables) != 2 or any(not isinstance(table, torch.Tensor)
            or table.device.type != 'cpu' or table.dtype != torch.bfloat16
            or tuple(table.shape) != (1, 1, PROPOSALS, 128)
            or not torch.isfinite(table).all() for table in tables):
        raise ValueError('Finite BF16 rotary tables for every experimental query required')
    cosine[:, :, :PROPOSALS], sine[:, :, :PROPOSALS] = tables
    live = torch.zeros(1, 1, 32, 1, dtype=torch.float32)
    live[:, :, :PROPOSALS] = 1
    return dict(identifiers=identifiers, anchor=torch.tensor([[[[anchor]]]], dtype=torch.int64),
        mask=mask, cosine=cosine, sine=sine, live=live)
