"""BF16 grouped-product diagnostic; excludes destination accumulator rounding."""


def grouped_projection_reference(features, weight, *, phases=4, reverse_sources=False):
    import torch

    if (features.dtype != torch.bfloat16 or weight.dtype != torch.bfloat16
            or features.device.type != 'cpu' or weight.device.type != 'cpu'
            or features.ndim < 2 or weight.ndim != 2 or features.shape[-1] != weight.shape[0]
            or weight.shape[0] % 8 or type(phases) is not int or phases not in (1, 2, 3, 4)):
        raise ValueError('CPU BF16 matrices with matching group-eight input width required')
    if not torch.isfinite(features).all() or not torch.isfinite(weight).all():
        raise ValueError('Finite operands required')

    def components(value):
        bits = value.contiguous().view(torch.int16).to(torch.int64)
        exponent = (bits >> 7) & 255
        if torch.any((exponent == 0) & ((bits & 127) != 0)):
            raise ValueError('Subnormal operands are outside this diagnostic')
        return torch.where(bits < 0, -1, 1), exponent, ((bits & 127) + 128) * 8

    feature_sign, feature_exp, feature_man = components(features.reshape(-1, features.shape[-1]))
    weight_sign, weight_exp, weight_man = components(weight)
    result = torch.zeros((feature_sign.shape[0], weight.shape[1]), dtype=torch.float64)
    for start in range(0, weight.shape[0], 8):
        stop = start + 8
        signs = feature_sign[:, start:stop, None] * weight_sign[None, start:stop, :]
        nonzero = (feature_exp[:, start:stop, None] != 0) & (weight_exp[None, start:stop, :] != 0)
        exponent = torch.where(nonzero, feature_exp[:, start:stop, None] + weight_exp[None, start:stop, :], 0)
        shared = exponent.max(dim=1, keepdim=True).values
        shift = (shared - exponent).clamp(max=30)
        source_a = weight_man[None, start:stop, :]
        source_b = feature_man[:, start:stop, None]
        if reverse_sources:
            source_a, source_b = source_b, source_a
        for phase in range(phases):
            mantissa_a = ((source_a >> 1) & 31) if phase & 1 else source_a >> 6
            mantissa_b = ((source_b & 15) << 3) if phase & 2 else source_b >> 4
            products = torch.where(nonzero, mantissa_a * mantissa_b, 0)
            aligned = (products + torch.where(shift > 0, torch.ones_like(shift) << (shift - 1).clamp(min=0), 0)) >> shift
            subtotal = (signs * aligned).sum(dim=1).double()
            scale = shared.squeeze(1) - 264 - (5 if phase & 1 else 0) - (7 if phase & 2 else 0)
            result += torch.ldexp(subtotal, scale.to(torch.int32))
    return result.reshape(*features.shape[:-1], weight.shape[1])
