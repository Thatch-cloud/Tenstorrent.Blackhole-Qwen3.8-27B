"""draft_mlp at any served width.

The pair's module is pinned (recorded evidence hashes its bytes; test_tp2_pins) and stays as it was;
tp_addresses rebinds the names below to these at four cards only. Each is the pair's function with its literal chip and
head counts read from tp_shapes; at two chips it would be call for call the pinned one."""

import tp_shapes


def split_mlp_weights(gate, up, down):
    import torch

    chips = tp_shapes.chip_count()
    if (gate.ndim != 2 or up.shape != gate.shape or down.shape != (gate.shape[1], gate.shape[0])
            or gate.shape[0] == 0 or gate.shape[0] % chips or gate.shape[1] == 0
            or any(value.dtype != torch.bfloat16 or value.device.type != 'cpu' for value in (gate, up, down))):
        raise ValueError('Matching CPU BF16 MLP matrices with an even intermediate dimension required')
    return tuple((gate_part.T.contiguous(), up_part.T.contiguous(), down_part.T.contiguous())
        for gate_part, up_part, down_part in zip(gate.chunk(chips, dim=0), up.chunk(chips, dim=0), down.chunk(chips, dim=1), strict=True))
