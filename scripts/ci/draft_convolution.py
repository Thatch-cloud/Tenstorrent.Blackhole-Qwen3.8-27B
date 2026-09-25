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


def convolution_reference(hidden, dynamic, base, *, boundaries=None):
    import torch

    rows = validate_shapes(hidden, dynamic, base)
    boundaries = validate_boundaries(boundaries, rows)
    if any(value.dtype != torch.bfloat16 or not torch.isfinite(value).all() for value in (hidden, *dynamic, *base)):
        raise ValueError('Finite BF16 operands required')
    output = torch.zeros_like(hidden)
    for offset in range(2):
        if offset == 0:
            values = hidden
        else:
            parts = []
            for start, stop in (boundaries or ((0, rows),)):
                parts.append(torch.zeros_like(hidden[..., :1, :]))
                if stop - start > 1:
                    parts.append(hidden[..., start:stop - 1, :])
            values = torch.cat(parts, dim=2)
        output = output + base[offset] * values
        output = output + dynamic[offset].repeat_interleave(16, dim=-1) * values
    return output


def validate_boundaries(boundaries, rows):
    """Segment spans of a packed block, or None for one continuous sequence.

    The convolution is causal over the ROW axis: row r reads row r-1. Packed, the
    rows of the block belong to different users, so without segment spans user B's
    first row would convolve against user A's last draft. Only that one row per
    segment is wrong, and it is the anchor, so it corrupts the whole block below it.
    """
    if boundaries is None:
        return None
    spans = tuple(tuple(span) for span in boundaries)
    if (not spans or any(len(span) != 2 for span in spans) or spans[0][0] != 0
            or spans[-1][1] != rows
            or any(type(value) is not int for span in spans for value in span)
            or any(start >= stop for start, stop in spans)
            or any(spans[index][1] != spans[index + 1][0] for index in range(len(spans) - 1))):
        raise ValueError('Ordered contiguous packed segment spans covering the block required')
    return spans


def grouped_causal_convolution(operations, mesh, hidden, dynamic, base, *, fp32_intermediates=False, inspect=None,
                               retain_temporaries=None, boundaries=None):
    rows = validate_shapes(hidden, dynamic, base)
    boundaries = validate_boundaries(boundaries, rows)
    borrowed = (hidden, *dynamic, *base)
    if list(mesh.shape) != [1, 2] or any(value.dtype != operations.bfloat16 for value in borrowed):
        raise ValueError('TP2 BF16 operands required')
    protected = {addresses(operations, value) for value in borrowed}
    temporaries = []
    output = None

    def retain(value):
        if addresses(operations, value) not in protected:
            temporaries.append(value)
            if retain_temporaries is not None:
                retain_temporaries(value)
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
                    spans = boundaries or ((0, rows),)
                    parts = []
                    for start, stop in spans:
                        # Each segment starts from zero, exactly as row 0 does for a
                        # single sequence, so no user reads the user packed above it.
                        parts.append(retain(operations.slice(zero, (0, 0, 0, 0), (1, 1, 1, 5120))))
                        if stop - start > 1:
                            parts.append(retain(operations.slice(hidden, (0, 0, start, 0), (1, 1, stop - 1, 5120))))
                    values = parts[0] if len(parts) == 1 else retain(operations.concat(tuple(parts), dim=2,
                        memory_config=operations.DRAM_MEMORY_CONFIG))
            expanded = retain(operations.repeat_interleave(dynamic[offset], 16, dim=3,
                memory_config=operations.DRAM_MEMORY_CONFIG))
            static_term = arithmetic(operations.multiply, base[offset], values)
            output = arithmetic(operations.add, output, static_term)
            dynamic_term = arithmetic(operations.multiply, expanded, values)
            output = arithmetic(operations.add, output, dynamic_term)
        if retain_temporaries is None:
            operations.synchronize_device(mesh)
    except BaseException:
        if retain_temporaries is None:
            release_owned(operations, temporaries)
        raise
    if retain_temporaries is None:
        release_owned(operations, [value for value in temporaries if addresses(operations, value) != addresses(operations, output)])
    return output
