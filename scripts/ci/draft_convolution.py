"""DFlash2 kernel-two, group-16 causal convolution arithmetic diagnostic."""

from gdn_multitoken_conv import addresses, release_owned


def validate_shapes(hidden, dynamic, base):
    shape = tuple(hidden.shape)
    if len(shape) != 4 or shape[:2] != (1, 1) or shape[2] not in (1, 8, 32) or shape[3] != 5120:
        raise ValueError('Explicit one, eight or 32 row hidden block required')
    if (len(dynamic) != 2 or len(base) != 2
            or any(tuple(value.shape) != (1, 1, shape[2], 320) for value in dynamic)
            or any(tuple(value.shape) != (1, 1, 1, 5120) for value in base)):
        raise ValueError('Two group-16 dynamic kernels and two full-width base kernels required')
    return shape[2]


def convolution_reference(hidden, dynamic, base):
    import torch

    rows = validate_shapes(hidden, dynamic, base)
    if any(value.dtype != torch.bfloat16 or not torch.isfinite(value).all() for value in (hidden, *dynamic, *base)):
        raise ValueError('Finite BF16 operands required')
    output = torch.zeros_like(hidden)
    for offset in range(2):
        values = hidden if offset == 0 else torch.cat((torch.zeros_like(hidden[..., :1, :]), hidden[..., :rows - 1, :]), dim=2)
        output = output + base[offset] * values
        output = output + dynamic[offset].repeat_interleave(16, dim=-1) * values
    return output


def grouped_causal_convolution(operations, mesh, hidden, dynamic, base, *, fp32_intermediates=False, inspect=None):
    rows = validate_shapes(hidden, dynamic, base)
    borrowed = (hidden, *dynamic, *base)
    if list(mesh.shape) != [1, 2] or any(value.dtype != operations.bfloat16 for value in borrowed):
        raise ValueError('TP2 BF16 operands required')
    protected = {addresses(operations, value) for value in borrowed}
    temporaries = []
    output = None

    def retain(value):
        if addresses(operations, value) not in protected:
            temporaries.append(value)
        return value

    def arithmetic(operation, left, right):
        if fp32_intermediates:
            left = retain(operations.typecast(left, operations.float32))
            right = retain(operations.typecast(right, operations.float32))
        result = retain(operation(left, right, dtype=operations.float32 if fp32_intermediates else operations.bfloat16))
        rounded = retain(operations.typecast(result, operations.bfloat16)) if fp32_intermediates else result
        if inspect is not None:
            inspect('multiply' if operation == operations.multiply else 'add', left, right, result, rounded)
        return rounded

    try:
        zero = retain(operations.zeros_like(hidden))
        output = zero
        for offset in range(2):
            values = hidden
            if offset:
                values = zero
                if rows > 1:
                    leading = retain(operations.slice(zero, (0, 0, 0, 0), (1, 1, 1, 5120)))
                    previous = retain(operations.slice(hidden, (0, 0, 0, 0), (1, 1, rows - 1, 5120)))
                    values = retain(operations.concat((leading, previous), dim=2, memory_config=operations.DRAM_MEMORY_CONFIG))
            expanded = retain(operations.repeat_interleave(dynamic[offset], 16, dim=3,
                memory_config=operations.DRAM_MEMORY_CONFIG))
            static_term = arithmetic(operations.multiply, base[offset], values)
            output = arithmetic(operations.add, output, static_term)
            dynamic_term = arithmetic(operations.multiply, expanded, values)
            output = arithmetic(operations.add, output, dynamic_term)
        operations.synchronize_device(mesh)
    except BaseException:
        release_owned(operations, temporaries)
        raise
    release_owned(operations, [value for value in temporaries if addresses(operations, value) != addresses(operations, output)])
    return output
