"""Host inputs for an unqualified T32 experiment; not admitted by the live drafter."""

from dspark_inputs import MASK_TOKEN, VOCABULARY


PROPOSALS = 31


def proposal_inputs(anchor, position, capacity, rotary):
    import torch

    if (type(anchor) is not int or not 0 <= anchor < VOCABULARY
            or type(position) is not int or type(capacity) is not int
            or not 1 <= position <= capacity <= 8192 or capacity % 32):
        raise ValueError('Valid anchor and committed frontier in tile-aligned full history required')
    padded = ((capacity + PROPOSALS + 63) // 64) * 64
    mask = torch.full((1, 1, 32, padded), float('-inf'), dtype=torch.bfloat16)
    mask[:, :, :PROPOSALS, :position] = 0
    mask[:, :, :PROPOSALS, capacity:capacity + PROPOSALS] = 0
    mask[:, :, PROPOSALS:, capacity] = 0
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
