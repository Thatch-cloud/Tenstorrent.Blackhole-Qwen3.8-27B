"""Host-reduced projection handoff and learned RMS normalization diagnostics."""


def projection_row(report):
    import torch

    checks = report.get('checks', [])
    if (report.get('passed') is not True or report.get('output_width') != 5120 or len(checks) != 2
            or {check.get('chip') for check in checks} != {0, 1}
            or any(check.get('passed') is not True or check.get('rows') != 1
                   or check.get('shape') != [1, 1, 1, 5120]
                   or len(check.get('actual_first_row', [])) != 5120 for check in checks)):
        raise ValueError('Complete passed single-row TP2 projection required')
    values = [torch.tensor(check['actual_first_row'], dtype=torch.float32) for check in sorted(checks, key=lambda check: check['chip'])]
    if any(not torch.isfinite(value).all() for value in values):
        raise ValueError('Finite partial projections required')
    return (values[0] + values[1]).bfloat16().reshape(1, 1, 1, 5120)


def rms_reference(value, weight, epsilon=1e-6):
    import torch

    if (value.dtype != torch.bfloat16 or weight.dtype != torch.bfloat16
            or value.shape[-1] != 5120 or tuple(weight.shape) != (5120,)
            or epsilon != 1e-6 or not torch.isfinite(value).all() or not torch.isfinite(weight).all()):
        raise ValueError('Finite 5120-wide BF16 inputs and configured epsilon required')
    widened = value.float()
    return (widened * torch.rsqrt(widened.square().mean(dim=-1, keepdim=True) + epsilon) * weight.float()).bfloat16()


def bf16_ulp_distance(actual, expected):
    import torch

    if (actual.dtype != torch.bfloat16 or expected.dtype != torch.bfloat16 or actual.shape != expected.shape
            or not torch.isfinite(actual).all() or not torch.isfinite(expected).all()):
        raise ValueError('Matching finite BF16 tensors required')

    def ordered(value):
        bits = value.contiguous().view(torch.int16).int()
        magnitude = bits & 0x7fff
        return torch.where(bits < 0, 0x8000 - magnitude, 0x8000 + magnitude)

    return (ordered(actual) - ordered(expected)).abs()
