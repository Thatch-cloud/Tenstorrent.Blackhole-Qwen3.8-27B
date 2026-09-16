"""Requested prompt sizes, not a claim of numerical or hardware admission."""

import os


CONTEXTS = (4096, 8192, 16384, 32768, 65536, 131072, 262144)


def geometry(context):
    if type(context) is not int or context not in CONTEXTS:
        raise ValueError('Explicit frozen-recipe ladder context required')
    capacity = context + 256
    target_pages = max(1024, (capacity + 63) // 64)
    return dict(context=context, capacity=capacity, proposals=15,
        positions=(context, capacity - 15), storage_keys=capacity + 64,
        padded_keys=((capacity + 64 + 255) // 256) * 256, key_chunk=256,
        target_sequence_capacity=max(65536, 1 << (capacity - 1).bit_length()),
        target_page_count=target_pages, target_cache_blocks=target_pages + 8,
        target_block_size=64)


def selected_geometry():
    value = os.environ.get('QWEN_DSPARK_REQUEST_CONTEXT', '8192')
    if value not in tuple(map(str, CONTEXTS)):
        raise ValueError('Unsupported QWEN_DSPARK_REQUEST_CONTEXT')
    return geometry(int(value))


def factory_selector():
    return '(' + ' || '.join(f"Skt == {geometry(context)['padded_keys'] // 32}"
        for context in CONTEXTS) + ')'
