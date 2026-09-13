"""Explicit full-history ladder fixtures; geometry planning grants no runtime admission."""


CONTEXTS = (128, 4096, 8192, 32768, 65536)


def geometry(context, output_tokens=1024):
    if type(context) is not int or context not in CONTEXTS:
        raise ValueError('Explicit full-ladder prompt size required')
    if type(output_tokens) is not int or output_tokens not in (256, 1024):
        raise ValueError('Explicit short comparison or sustained output allowance required')
    capacity = ((context + output_tokens + 31) // 32) * 32
    storage_keys = ((capacity + 15 + 63) // 64) * 64
    key_chunk = 512 if context >= 32768 else 256
    native_keys = ((storage_keys + key_chunk - 1) // key_chunk) * key_chunk
    return dict(context=context, output_tokens=output_tokens, capacity=capacity,
        proposals=15, storage_keys=storage_keys, native_keys=native_keys,
        extra_masked_keys=native_keys - storage_keys, key_chunk=key_chunk,
        probe_positions=(context, capacity - 15),
        runtime_admitted=False, numerical_qualified=False, performance_qualified=False)


def ladder(output_tokens=1024):
    return tuple(geometry(context, output_tokens) for context in CONTEXTS)
