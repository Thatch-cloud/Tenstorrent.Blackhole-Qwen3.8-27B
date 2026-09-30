"""pair_row_exact at any served width.

The pair's module is pinned (recorded evidence hashes its bytes; test_tp2_pins) and stays as it was;
tp_addresses rebinds the names below to these at four cards only. Each is the pair's function with its literal chip and
head counts read from tp_shapes; at two chips it would be call for call the pinned one."""

import os

from pair_row_exact import BLOCK_ROWS, HEAD_DIM, MARKER, SPAN, _NOTED, _log_line
import tp_shapes


def heads():
    """(query heads, key heads, group, folded query heads, folded key heads) per chip at the width this process serves at."""
    found = tp_shapes.active()
    query, key = found.draft_heads, found.draft_kv_heads
    return query, key, query // key, 2 * query, 2 * key


def note(pair, contexts, *, log=None):
    """MARKER once per process, at the first bucket that folds. Returns whether it logged."""
    if _NOTED:
        return False
    _NOTED.append(tuple(pair))
    (log or _log_line)('%s pair=[%s,%s] context=%s,%s heads=%d/%d keys=%d' % (
        MARKER, pair[0], pair[1], contexts[0], contexts[1], heads()[3], heads()[4], SPAN))
    return True


def validate_fold(operations, query, key, value, mask):
    """The packed pair's operands as execute_attention_branch assembles them, and the single-user mask."""
    query_heads, key_heads = heads()[:2]
    if (tuple(query.shape) != (1, query_heads, 32, HEAD_DIM)
            or tuple(key.shape) != (1, key_heads, 2 * SPAN, HEAD_DIM) or tuple(value.shape) != tuple(key.shape)
            or tuple(mask.shape) != (1, 1, 32, SPAN)):
        raise ValueError('The folded pair SDPA takes the packed (1, %d, 32, 128) query, the two-segment '
                         '(1, %d, 4160, 128) keys and values and the single-user (1, 1, 32, 2080) mask'
                         % (query_heads, key_heads))
    if any(tensor.dtype != operations.bfloat16 for tensor in (query, key, value, mask)):
        raise ValueError('BF16 folded draft attention operands required')
    if any(tensor.layout != operations.TILE_LAYOUT or tensor.memory_config() != operations.DRAM_MEMORY_CONFIG
           for tensor in (query, key, value, mask)):
        raise ValueError('Interleaved tiled DRAM operands required')


def fold_query(operations, query, retain):
    """(1, 16, 32, 128) -> (1, 32, 32, 128): Q head 8h+4u+j is head 4h+j with user u's 16 rows at rows 0-15.

    The u=0 half is the query as it is; the u=1 half is the query with its row halves swapped, so user b's rows
    sit at rows 0-15 exactly as they do in b's single-user trace. The row moves are bf16 copies."""
    memory = operations.DRAM_MEMORY_CONFIG
    query_heads, key_heads, group, folded_query_heads, _ = heads()
    upper = retain(operations.slice(query, (0, 0, 0, 0), (1, query_heads, BLOCK_ROWS, HEAD_DIM)))
    lower = retain(operations.slice(query, (0, 0, BLOCK_ROWS, 0), (1, query_heads, 2 * BLOCK_ROWS, HEAD_DIM)))
    shifted = retain(operations.concat([lower, upper], dim=2, memory_config=memory))
    groups = [retain(operations.reshape(value, (key_heads, group, 32, HEAD_DIM))) for value in (query, shifted)]
    folded = retain(operations.concat(groups, dim=1, memory_config=memory))
    return retain(operations.reshape(folded, (1, folded_query_heads, 32, HEAD_DIM)))


def fold_keys(operations, tensor, retain):
    """(1, 4, 4160, 128) -> (1, 8, 2080, 128), a view: each head's 130 tiles are [segment a | segment b], so KV
    head 2h+u is user u's segment of head h."""
    return retain(operations.reshape(tensor, (1, heads()[4], SPAN, HEAD_DIM)))


def unfold_output(operations, output, retain):
    """(1, 32, 32, 128) -> (1, 16, 32, 128): head 4h+j's rows 0-15 from folded head 8h+j (user a) and its rows
    16-31 from folded head 8h+4+j's rows 0-15 (user b) - the packed layout the rest of the branch reads."""
    query_heads, key_heads, group, _, _ = heads()
    grouped = retain(operations.reshape(output, (key_heads, 2 * group, 32, HEAD_DIM)))
    halves = []
    for user in range(2):
        rows = retain(operations.slice(grouped, (0, user * group, 0, 0),
                                       (key_heads, (user + 1) * group, BLOCK_ROWS, HEAD_DIM)))
        halves.append(retain(operations.reshape(rows, (1, query_heads, BLOCK_ROWS, HEAD_DIM))))
    return retain(operations.concat(halves, dim=2, memory_config=operations.DRAM_MEMORY_CONFIG))
