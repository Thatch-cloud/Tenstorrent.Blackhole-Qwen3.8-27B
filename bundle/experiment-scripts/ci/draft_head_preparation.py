"""Head layout and nontraditional RoPE references for pinned DFlash2 attention."""


def rope_tables(start, rows):
    import torch

    if type(start) is not int or type(rows) is not int or start < 0 or rows < 1 or start + rows > 262144:
        raise ValueError('Bounded absolute RoPE positions required')
    frequency = 1.0 / (10000000.0 ** (torch.arange(0, 128, 2, dtype=torch.float32) / 128))
    angles = torch.arange(start, start + rows, dtype=torch.float32)[:, None] * frequency[None, :]
    doubled = torch.cat((angles, angles), dim=-1).reshape(1, 1, rows, 128)
    return doubled.cos().bfloat16(), doubled.sin().bfloat16()


def head_norm_reference(value, weight):
    import torch

    if value.shape[-1] != 128 or tuple(weight.shape) != (128,):
        raise ValueError('128-wide attention heads and learned norm required')
    widened = value.float()
    return (widened * torch.rsqrt(widened.square().mean(-1, keepdim=True) + 1e-6) * weight.float()).bfloat16()


def rope_reference(value, cosine, sine):
    import torch

    if (value.ndim != 4 or value.shape[-1] != 128 or tuple(cosine.shape) != (1, 1, value.shape[2], 128)
            or cosine.shape != sine.shape):
        raise ValueError('Matching head and rotary-cache geometry required')
    rotated = torch.cat((-value[..., 64:].float(), value[..., :64].float()), dim=-1)
    return (value.float() * cosine.float() + rotated * sine.float()).bfloat16()
