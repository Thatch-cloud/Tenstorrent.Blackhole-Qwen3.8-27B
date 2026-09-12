"""Opt-in 16-row activation tiles around the unchanged quantized 1D projection."""


PROJECTIONS = {'gate': (5120, 8704, 44, 'bfloat4_b', True),
    'up': (5120, 8704, 44, 'bfloat4_b', False),
    'down': (8704, 5120, 33, 'bfloat8_b', False)}


def project(operations, value, weight, program, compute, retain, *, tiny, retile=None):
    if type(tiny) is not bool or tuple(value.shape)[:3] != (1, 1, 8) or len(value.shape) != 4:
        raise ValueError('Explicit T8 single-stream projection required')
    if value.dtype != operations.bfloat16 or weight.dtype not in (operations.bfloat4_b, operations.bfloat8_b):
        raise ValueError('Native BF16 activations and unchanged compressed weights required')
    if tiny and not callable(retile):
        raise ValueError('Prepared bounded-memory tile conversion required')
    source = value
    if tiny:
        source = retile(value, 16)
    output = retain(operations.linear(source, weight, program_config=program, compute_kernel_config=compute,
        memory_config=operations.L1_MEMORY_CONFIG))
    if tiny:
        output = retile(output, 32)
    return output
