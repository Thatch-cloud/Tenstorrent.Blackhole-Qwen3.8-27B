"""Block-diagonal draft attention mask packing several users into one 32-row block.

Probe 35436807668 measured TTScheduler putting every live request's decode in ONE
scheduled step (cached=['B','A'], 16 rows each). Alternating them instead is ruled
out by the weight-pass floor: 19.92 GB of dense projections per step against
512 GB/s is ~19.5 ms, spent once per user per round, so per-user rate becomes the
single-user rate divided by N. Only a batched verify amortises that pass.

The draft block is already 32 rows wide and T16 uses 16 of them:

    mask[..., 16:, :] = float('-inf')
    mask[..., 16:, context_rows] = 0

so half of every proposal pass is padding today. Two T16 users fit in the block
that is already being paid for, at no extra pass.

Packing is by KEY SEGMENT. Each user gets a 32-aligned span of the key axis holding
its own history followed by its own block keys, and its query rows may only see that
span. Segment u is [offset_u, offset_u + span_u) where
span_u = ceil((context_u + block_rows) / 32) * 32, so the concatenation stays tile
aligned and K stays a multiple of 32.

Within a segment the rule is exactly the single-user one from `draft_attention`:
a live query at absolute row context_u + i sees history keys within `window`, plus
its own block keys. Across segments nothing is visible, which is what makes the pack
equivalent to running the users separately.
"""


def segments(contexts, block_rows, *, key_multiple=32):
    """Key-axis spans, one per user, in packing order."""
    if (type(block_rows) is not int or block_rows not in (1, 8, 16, 32)
            or type(key_multiple) is not int or key_multiple not in (32, 128)):
        raise ValueError('Configured DFlash2 draft block geometry required')
    contexts = tuple(contexts)
    if (not contexts or len(contexts) * block_rows > 32
            or any(type(value) is not int or not 0 <= value <= 262144 for value in contexts)):
        raise ValueError('At least one user and block rows totalling at most the 32-row draft block required')
    spans, offset = [], 0
    for index, context in enumerate(contexts):
        span = ((context + block_rows + key_multiple - 1) // key_multiple) * key_multiple
        spans.append(dict(user=index, context=context, offset=offset, span=span,
                          rows=slice(index * block_rows, (index + 1) * block_rows),
                          keys=slice(offset, offset + span)))
        offset += span
    return spans, offset


def batched_attention_mask(contexts, block_rows=16, *, window=2048, key_multiple=32, padded_rows=32):
    """Mask of (1, 1, 32, K) admitting each user only to its own key segment."""
    import torch

    if padded_rows != 32 or type(window) is not int or window != 2048:
        raise ValueError('Bounded 2048-history 32-row draft geometry required')
    spans, key_rows = segments(contexts, block_rows, key_multiple=key_multiple)
    mask = torch.full((1, 1, padded_rows, key_rows), float('-inf'), dtype=torch.bfloat16)
    for span in spans:
        context, offset = span['context'], span['offset']
        # Local coordinates inside the segment are the single-user problem, so this
        # is `draft_attention.draft_attention_mask`'s rule verbatim.
        query = context + torch.arange(block_rows)[:, None]
        key = torch.arange(span['span'])[None, :]
        allowed = (((key < context) & (query - key < window))
                   | ((key >= context) & (key < context + block_rows)))
        mask[0, 0, span['rows'], offset:offset + span['span']] = torch.where(
            allowed, 0., float('-inf')).bfloat16()
    # Rows past the packed users are padding. The kernel needs exactly one visible
    # key per padded row or its softmax has no finite term; the first user's anchor
    # column is the one the single-user mask already uses.
    used = len(spans) * block_rows
    if used < padded_rows:
        mask[0, 0, used:, :] = float('-inf')
        mask[0, 0, used:, spans[0]['context']] = 0
    return mask


def validate_batched_mask(mask, contexts, block_rows=16):
    """Reject any mask that leaks across users or strands a row."""
    import torch

    spans, key_rows = segments(contexts, block_rows)
    if (not isinstance(mask, torch.Tensor) or mask.dtype != torch.bfloat16
            or mask.ndim != 4 or tuple(mask.shape[:3]) != (1, 1, 32)
            or mask.shape[-1] != key_rows or mask.shape[-1] % 32):
        raise ValueError('Host tiled BF16 packed draft mask matching the declared users required')
    if not ((mask == 0) | torch.isneginf(mask)).all():
        raise ValueError('Only zero and negative infinity mask values supported')
    visible = (mask[0, 0] == 0)
    for span in spans:
        rows = visible[span['rows']]
        if not rows.any(-1).all():
            raise ValueError('Every live draft query needs a visible key')
        outside = rows.clone()
        outside[:, span['keys']] = False
        if outside.any():
            raise ValueError('A draft query can see another user: the pack is not block diagonal')
    used = len(spans) * block_rows
    if used < 32 and not visible[used:].sum(-1).eq(1).all():
        raise ValueError('Every padded query must have exactly one visible key')
    return spans, key_rows


def user_contexts(users):
    """Validate the packed user list and return their history lengths."""
    users = tuple(users)
    for user in users:
        if (not isinstance(user, dict) or type(user.get('position')) is not int
                or type(user.get('history_rows')) is not int
                or not 1 <= user['history_rows'] <= 2048
                or user['position'] < user['history_rows']):
            raise ValueError('Each packed user needs an absolute position and its bounded history length')
    return users, [user['history_rows'] for user in users]


def packed_rope_tables(users, block_rows=16, *, key_multiple=32):
    """RoPE tables laid out to match the packed query rows and key segments.

    Absolute positions are per user. Query row i of user u is at position_u + i, and
    key j of segment u is at (position_u - history_rows_u) + j, which is the same
    correspondence the single-user path relies on: a local key index k there maps to
    absolute (position - history_rows) + k, and a query row i sits at local
    history_rows + i, hence absolute position + i.

    Getting this wrong is silent. The mask would still be block diagonal and the
    shapes would still line up, and every user after the first would simply attend
    at the wrong distances.
    """
    import torch

    from draft_head_preparation import rope_tables

    users, contexts = user_contexts(users)
    spans, key_rows = segments(contexts, block_rows, key_multiple=key_multiple)
    query_parts, key_parts = [], []
    for user, span in zip(users, spans):
        query_parts.append(rope_tables(user['position'], block_rows))
        key_parts.append(rope_tables(user['position'] - user['history_rows'], span['span']))
    used = len(spans) * block_rows
    if used < 32:
        # Padded rows see exactly one key and their angles cannot change any
        # committed token, but they must still be finite and in range. They CONTINUE
        # past the last user's block rather than restarting at it, because at one
        # user that is what makes the pack byte-identical to the single contiguous
        # rope_tables(position, 32) the device builds today.
        query_parts.append(rope_tables(users[-1]['position'] + block_rows, 32 - used))
    pack = lambda parts: tuple(torch.cat([part[index] for part in parts], dim=2) for index in (0, 1))
    query, key = pack(query_parts), pack(key_parts)
    if any(tuple(table.shape) != (1, 1, 32, 128) for table in query):
        raise AssertionError('Packed query RoPE must cover the whole 32-row block')
    if any(tuple(table.shape) != (1, 1, key_rows, 128) for table in key):
        raise AssertionError('Packed key RoPE must cover every key segment')
    return dict(q=query, k=key)


def key_value_plan(contexts, block_rows=16, *, key_multiple=32):
    """Ordered pieces to concatenate into the packed key axis.

    This mirrors what `execute_attention_branch` does for one user on the
    cache_history path that production runs:

        heads[name] = concat([cached_history[name], live[name]], dim=2)

    where `cached_history` is the committed K/V for the prompt and `live` is the
    K/V of the 32-row proposal block. At one user, `live` is 32 rows of which 16
    carry the T16 proposals and 16 are padding, and context + 32 lands exactly on
    the aligned span.

    Packed, each user contributes its own cached rows, then ITS OWN 16 live rows
    taken out of the shared block, then padding out to its aligned span. So the
    live block is split by user rather than appended whole - the piece that makes
    one proposal pass serve several users.
    """
    spans, key_rows = segments(contexts, block_rows, key_multiple=key_multiple)
    plan = []
    for span in spans:
        pad = span['span'] - span['context'] - block_rows
        if pad < 0:
            raise ValueError('A user history leaves no room for its own proposal block')
        plan.append(dict(kind='cached', user=span['user'], rows=span['context']))
        plan.append(dict(kind='live', user=span['user'], rows=block_rows,
                         source=slice(span['user'] * block_rows, (span['user'] + 1) * block_rows)))
        if pad:
            plan.append(dict(kind='pad', user=span['user'], rows=pad))
    if sum(piece['rows'] for piece in plan) != key_rows:
        raise AssertionError('The assembly plan must cover the packed key axis exactly')
    return plan, spans, key_rows


def live_key_rope(users, block_rows=16, *, key_multiple=32):
    """The 32 rows of key RoPE that rotate the proposal block, in packed order.

    For one user the branch slices `rope['k'][context:key_rows]`, the 32 rows whose
    absolute positions are the block's own. Packed, each user needs the 16 rows at
    its own segment offset, concatenated in block order.
    """
    import torch

    users, contexts = user_contexts(users)
    spans, _ = segments(contexts, block_rows, key_multiple=key_multiple)
    tables = packed_rope_tables(users, block_rows, key_multiple=key_multiple)['k']
    parts = []
    for span in spans:
        start = span['offset'] + span['context']
        parts.append(tuple(table[:, :, start:start + block_rows] for table in tables))
    used = len(spans) * block_rows
    if used < 32:
        last = spans[-1]
        start = last['offset'] + last['context'] + block_rows
        parts.append(tuple(table[:, :, start:start + 32 - used] for table in tables))
    packed = tuple(torch.cat([part[index] for part in parts], dim=2) for index in (0, 1))
    if any(tuple(table.shape) != (1, 1, 32, 128) for table in packed):
        raise AssertionError('The live key RoPE must cover the whole 32-row proposal block')
    return packed


def live_key_rope_from(tables, users, block_rows=16, *, key_multiple=32):
    """live_key_rope over key tables the caller already built with packed_rope_tables
    for the SAME users - QWEN_FAST_ROUND_B1 (C8), so a pair update builds its 4160-row
    key tables once rather than twice. The slicing is live_key_rope's own, row for row;
    test_dflash_round_b1.py pins the two equal bit for bit."""
    import torch

    users, contexts = user_contexts(users)
    spans, key_rows = segments(contexts, block_rows, key_multiple=key_multiple)
    tables = tuple(tables)
    if len(tables) != 2 or any(tuple(table.shape) != (1, 1, key_rows, 128) for table in tables):
        raise ValueError('The packed key RoPE tables of these users required')
    parts = []
    for span in spans:
        start = span['offset'] + span['context']
        parts.append(tuple(table[:, :, start:start + block_rows] for table in tables))
    used = len(spans) * block_rows
    if used < 32:
        last = spans[-1]
        start = last['offset'] + last['context'] + block_rows
        parts.append(tuple(table[:, :, start:start + 32 - used] for table in tables))
    packed = tuple(torch.cat([part[index] for part in parts], dim=2) for index in (0, 1))
    if any(tuple(table.shape) != (1, 1, 32, 128) for table in packed):
        raise AssertionError('The live key RoPE must cover the whole 32-row proposal block')
    return packed
