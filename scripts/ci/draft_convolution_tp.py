"""draft_convolution at any served width.

The pair's module is pinned (recorded evidence hashes its bytes; test_tp2_pins) and stays as it was;
tp_addresses rebinds the names below to these at four cards only. Each is the pair's function with its literal chip and
head counts read from tp_shapes; at two chips it would be call for call the pinned one."""

from draft_convolution import validate_boundaries, validate_shapes
from gdn_multitoken_conv import addresses, release_owned
import tp_shapes


def grouped_causal_convolution(operations, mesh, hidden, dynamic, base, *, fp32_intermediates=False, inspect=None,
                               retain_temporaries=None, boundaries=None):
    rows = validate_shapes(hidden, dynamic, base)
    boundaries = validate_boundaries(boundaries, rows)
    borrowed = (hidden, *dynamic, *base)
    if list(mesh.shape) != [1, tp_shapes.chip_count()] or any(value.dtype != operations.bfloat16 for value in borrowed):
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
