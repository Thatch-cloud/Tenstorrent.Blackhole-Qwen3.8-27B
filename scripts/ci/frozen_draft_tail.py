"""Unqualified tile-aligned draft KV assembly; no runtime or serving hook."""

from dspark_full_attention import geometry
from dspark_projection import require_tensor


def append_queries(operations, history, queries, retain, *, position, proposals):
    if type(position) is not int or position % 32 or not callable(retain):
        raise ValueError('Tile-aligned fixed history and explicit ownership required')
    padded_keys = geometry(position, proposals)[-1][1]
    require_tensor(operations, history, (1, 4, position, 128), operations.bfloat16)
    require_tensor(operations, queries, (1, 4, 32, 128), operations.bfloat16)
    tail_rows = padded_keys - position
    if tail_rows not in (32, 64):
        raise ValueError('One or two tile rows of proposal tail required')
    valid = retain(operations.slice(queries, (0, 0, 0, 0), (1, 4, proposals, 128)))
    tail = retain(operations.pad(valid, [(0, 0), (0, 0), (0, tail_rows - proposals), (0, 0)], 0.0))
    return retain(operations.concat([history, tail], dim=2, memory_config=operations.DRAM_MEMORY_CONFIG))
