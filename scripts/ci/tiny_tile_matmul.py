"""Opt-in 16-row activation tiles around the unchanged quantized 1D projection."""


PROJECTIONS = {'gate': (5120, 8704, 44, 'bfloat4_b', True),
    'up': (5120, 8704, 44, 'bfloat4_b', False),
    'down': (8704, 5120, 33, 'bfloat8_b', False)}


def project(operations, value, weight, program, compute, retain, *, tiny):
    if type(tiny) is not bool or tuple(value.shape)[:3] != (1, 1, 8) or len(value.shape) != 4:
        raise ValueError('Explicit T8 single-stream projection required')
    if value.dtype != operations.bfloat16 or weight.dtype not in (operations.bfloat4_b, operations.bfloat8_b):
        raise ValueError('Native BF16 activations and unchanged compressed weights required')
    source = value
    if tiny:
        source = retain(operations.tilize(value, tile=operations.Tile((16, 32)),
            memory_config=operations.L1_MEMORY_CONFIG))
    output = retain(operations.linear(source, weight, program_config=program, compute_kernel_config=compute,
        memory_config=operations.L1_MEMORY_CONFIG))
    if tiny:
        output = retain(operations.tilize(output, tile=operations.Tile((32, 32)),
            memory_config=operations.L1_MEMORY_CONFIG))
    return output
