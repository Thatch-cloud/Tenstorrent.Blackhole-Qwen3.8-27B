"""Narrow input-projection batching over the pinned native GDN decode path."""


def validate_rows(shape):
    if len(shape) != 3 or shape[0] != 1 or shape[2] != 5120 or shape[1] not in (1, 2, 4, 8, 16):
        raise ValueError("Expected [1, T, 5120], T=1/2/4/8/16")
    return shape[1]


def decode_projected(gdn, packed_input, token_inputs, checkpoint, operations):
    rows = validate_rows(tuple(packed_input.shape))
    if len(token_inputs) != rows or any(tuple(token.shape) != (1, 1, 5120) for token in token_inputs):
        raise ValueError("Expected one B1 input per verification row")
    original = gdn._project_qkvzab_raw
    packed = original(packed_input, rows, operations.L1_MEMORY_CONFIG)
    cursor = 0

    def projected_row(unused_input, batch, memory):
        nonlocal cursor
        if batch != 1 or cursor >= rows:
            raise ValueError("Native projection callback did not consume exactly one row")
        sliced = operations.slice(packed, (0, cursor, 0), (1, cursor + 1, packed.shape[-1]),
                                  memory_config=memory)
        output = operations.clone(sliced, memory_config=memory)
        cursor += 1
        return output

    outputs = []
    try:
        gdn._project_qkvzab_raw = projected_row
        for index, token in enumerate(token_inputs):
            outputs.append(gdn.forward_decode(token))
            checkpoint(index + 1)
        if cursor != rows:
            raise ValueError("Native direct-projection path was not engaged")
        return outputs
    finally:
        gdn._project_qkvzab_raw = original
        operations.deallocate(packed)
