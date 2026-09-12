"""Fixed-shape proposal contexts with masked holes and absolute rotary positions."""

from draft_attention import draft_attention_mask
from draft_head_preparation import rope_tables


def proposal_contexts(position, max_new_tokens):
    if (type(position) is not int or not 1 <= position <= 262111
            or type(max_new_tokens) is not int or not 1 <= max_new_tokens <= 262144 - position - 32):
        raise ValueError('Bounded prefill position and complete request budget required')
    first = next(rows for rows in (256, 512, 1024, 2048) if rows >= min(position, 2048))
    last = next(rows for rows in (256, 512, 1024, 2048) if rows >= min(2048, position + max_new_tokens))
    return tuple(rows for rows in (256, 512, 1024, 2048) if first <= rows <= last)


def proposal_inputs(seed, position, history_rows, block_rows, context_rows):
    import torch

    if (type(seed) is not int or not 0 <= seed < 248320
            or type(position) is not int or not 1 <= position <= 262112
            or type(history_rows) is not int or not 1 <= history_rows <= min(position, 2048)
            or type(block_rows) is not int or block_rows not in (8, 32)
            or type(context_rows) is not int or context_rows not in (256, 512, 1024, 2048)
            or history_rows > context_rows):
        raise ValueError('Committed history, anchor and explicit fixed draft bucket required')
    key_rows = context_rows + 32
    original = draft_attention_mask(history_rows, block_rows=block_rows)
    mask = torch.full((1, 1, 32, key_rows), float('-inf'), dtype=torch.bfloat16)
    mask[..., :history_rows] = original[..., :history_rows]
    mask[..., context_rows:context_rows + block_rows] = original[..., history_rows:history_rows + block_rows]
    query_rope = rope_tables(position, 32)
    history_rope = rope_tables(position - history_rows, history_rows)
    key_rope = []
    for historical, proposal in zip(history_rope, query_rope, strict=True):
        table = torch.zeros((1, 1, key_rows, 128), dtype=torch.bfloat16)
        table[..., :history_rows, :] = historical
        table[..., context_rows:context_rows + block_rows, :] = proposal[..., :block_rows, :]
        key_rope.append(table)
    identifiers = torch.tensor([[seed, *([248070] * (block_rows - 1))]], dtype=torch.int64)
    return dict(identifiers=identifiers, mask=mask, rope=dict(q=query_rope, k=tuple(key_rope)))
