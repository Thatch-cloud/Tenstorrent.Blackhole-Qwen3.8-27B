"""BF16 grouped-product diagnostics with optional Blackhole FP32 accumulation."""


def accumulate_fp32(destination, terms):
    import torch

    destination_sign = destination >> 31
    destination_exp = (destination >> 23) & 255
    destination_man = torch.where(destination_exp != 0, (destination & 0x7fffff) | 0x800000, 0)
    common_exp = destination_exp.clone()
    for exponent, subtotal in terms:
        common_exp = torch.maximum(common_exp, exponent)
    total = torch.zeros_like(destination)
    for exponent, subtotal in terms:
        negative = (subtotal < 0).to(torch.int64)
        magnitude = torch.where(exponent > 0, subtotal.abs() << 13, 0)
        distance = common_exp - exponent
        rounding = (torch.ones_like(distance) << (distance - 1).clamp(min=0, max=30)) - negative
        aligned = (magnitude + torch.where(distance > 0, rounding, 0)) >> distance.clamp(max=30)
        aligned = torch.where(distance < 31, aligned, 0)
        total += torch.where(negative != 0, -aligned, aligned)
    distance = common_exp - destination_exp
    rounding = torch.ones_like(distance) << (distance - 1).clamp(min=0, max=30)
    aligned = (destination_man + torch.where(distance > 0, rounding, 0)) >> distance.clamp(max=30)
    aligned = torch.where(distance < 31, aligned, 0)
    total += torch.where(destination_sign != 0, -aligned, aligned)
    negative = (total < 0).to(torch.int64)
    magnitude = total.abs()
    left_shift = 23 - torch.floor(torch.log2(magnitude.clamp(min=1).double())).to(torch.int64)
    exponent = common_exp - left_shift
    right_shift = (-left_shift).clamp(min=0)
    rounding = torch.ones_like(right_shift) << (right_shift - 1).clamp(min=0)
    normalized = torch.where(left_shift >= 0, magnitude << left_shift.clamp(min=0),
        (magnitude + rounding) >> right_shift)
    exponent += ((normalized & 0x1000000) != 0).to(torch.int64)
    encoded = (negative << 31) | (exponent.clamp(max=255) << 23) | torch.where(exponent < 255, normalized & 0x7fffff, 0)
    return torch.where((magnitude != 0) & (exponent > 0) & (common_exp > 0), encoded, 0)


def grouped_projection_reference(features, weight, *, phases=4, reverse_sources=False, destination_rounding=False,
        fidelity_span=16):
    import torch

    if (features.dtype != torch.bfloat16 or weight.dtype != torch.bfloat16
            or features.device.type != 'cpu' or weight.device.type != 'cpu'
            or features.ndim < 2 or weight.ndim != 2 or features.shape[-1] != weight.shape[0]
            or type(fidelity_span) is not int or fidelity_span not in (16, 32)
            or weight.shape[0] % (fidelity_span if destination_rounding else 8)
            or type(phases) is not int or phases not in (1, 2, 3, 4)):
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
    destination = torch.zeros_like(result, dtype=torch.int64)
    pending = [[] for phase in range(phases)]
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
            subtotal = (signs * aligned).sum(dim=1)
            scale = shared.squeeze(1) - 264 - (5 if phase & 1 else 0) - (7 if phase & 2 else 0)
            if destination_rounding:
                pending[phase].append((scale + 137, subtotal))
            else:
                result += torch.ldexp(subtotal.double(), scale.to(torch.int32))
        if destination_rounding and start % fidelity_span == fidelity_span - 8:
            for terms in pending:
                for offset in range(0, len(terms), 2):
                    destination = accumulate_fp32(destination, terms[offset:offset + 2])
            pending = [[] for phase in range(phases)]
    if destination_rounding:
        result = destination.to(torch.int32).view(torch.float32).double()
    return result.reshape(*features.shape[:-1], weight.shape[1])
