"""TP2 column-parallel gate/up and row-parallel down packing for a SwiGLU draft MLP."""


def split_mlp_weights(gate, up, down):
    import torch

    if (gate.ndim != 2 or up.shape != gate.shape or down.shape != (gate.shape[1], gate.shape[0])
            or gate.shape[0] == 0 or gate.shape[0] % 2 or gate.shape[1] == 0
            or any(value.dtype != torch.bfloat16 or value.device.type != 'cpu' for value in (gate, up, down))):
        raise ValueError('Matching CPU BF16 MLP matrices with an even intermediate dimension required')
    return tuple((gate_part.T.contiguous(), up_part.T.contiguous(), down_part.T.contiguous())
        for gate_part, up_part, down_part in zip(gate.chunk(2, dim=0), up.chunk(2, dim=0), down.chunk(2, dim=1), strict=True))


def swiglu_reference(gate, up):
    import torch

    if gate.shape != up.shape:
        raise ValueError('Matching gate/up activations required')
    return (torch.nn.functional.silu(gate.bfloat16().float()).bfloat16() * up.bfloat16()).bfloat16()
