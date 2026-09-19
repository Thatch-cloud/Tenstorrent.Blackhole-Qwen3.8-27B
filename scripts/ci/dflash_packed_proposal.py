"""Packed proposal inputs and per-user selection for the shared 32-row block.

One proposal pass carries every packed user. The anchor/draft identifiers, the
selector features and the vocabulary-head candidates are all laid out in block
order, so each user owns a contiguous run of `block_rows` rows and its results are
a slice rather than a separate pass.

`merge_chunk_candidates` drops the anchor row and returns rows 1..block_rows-1, so
with a 32-row block the returned index of user u's first draft is u * block_rows:
user 0 holds block rows 1..15 at indices 0..14, and user 1 holds block rows 17..31
at indices 16..30. Getting that offset wrong silently attributes one user's
proposals to another, which is why it is pinned by a test rather than a comment.
"""

DRAFT_FILLER = 248070


def packed_identifiers(seeds, block_rows=16, *, filler=DRAFT_FILLER):
    """The (1, 32) anchor/draft token ids the embedding consumes, in block order."""
    import torch

    seeds = tuple(seeds)
    if (type(block_rows) is not int or block_rows not in (8, 16, 32)
            or not seeds or len(seeds) * block_rows > 32
            or any(type(seed) is not int or not 0 <= seed < 248320 for seed in seeds)):
        raise ValueError('One committed anchor per packed user and a fitting block required')
    rows = []
    for seed in seeds:
        rows.extend([seed, *([filler] * (block_rows - 1))])
    rows.extend([filler] * (32 - len(rows)))
    return torch.tensor([rows], dtype=torch.int64)


def user_slices(users, block_rows=16):
    """Where each user's drafts sit in the merged candidate and selector tensors."""
    if (type(block_rows) is not int or type(users) is not int
            or not (1 <= users and users * block_rows <= 32)):
        raise ValueError('A fitting number of packed users required')
    return [dict(user=index, block=slice(index * block_rows, (index + 1) * block_rows),
                 drafts=slice(index * block_rows, index * block_rows + block_rows - 1),
                 selector=slice(index * block_rows + 1, (index + 1) * block_rows))
            for index in range(users)]


def split_selection(projected_hidden, candidates, unary, users, block_rows=16):
    """Per-user (selector features, candidate ids, candidate scores).

    `projected_hidden` is the replicated selector projection over the whole block,
    shaped (1, 32, 256). `candidates` and `unary` come straight from
    merge_chunk_candidates(block_rows=32), shaped (1, 31, 16).
    """
    if (projected_hidden.ndim != 3 or projected_hidden.shape[:2] != (1, 32)
            or candidates.ndim != 3 or candidates.shape[:2] != (1, 31)
            or unary.shape != candidates.shape):
        raise ValueError('Whole-block selector features and merged 32-row candidates required')
    parts = []
    for part in user_slices(users, block_rows):
        parts.append(dict(user=part['user'],
                          hidden=projected_hidden[:, part['selector'], :],
                          candidates=candidates[:, part['drafts']],
                          unary=unary[:, part['drafts']]))
        if parts[-1]['hidden'].shape[1] != block_rows - 1:
            raise AssertionError('Each user contributes block_rows - 1 drafts')
        if parts[-1]['candidates'].shape[1] != block_rows - 1:
            raise AssertionError('Each user owns block_rows - 1 merged candidate rows')
    return parts


def select_packed(parts, seeds, counts, predecessors, successors):
    """Run the shared selector once per user over its own slice of the block."""
    import torch

    from draft_selector import select_active_candidates

    seeds, counts = tuple(seeds), tuple(counts)
    if len(parts) != len(seeds) or len(parts) != len(counts):
        raise ValueError('One anchor and one proposal count per packed user required')
    chosen = []
    for part, seed, count in zip(parts, seeds, counts):
        if type(count) is not int or not 1 <= count <= part['candidates'].shape[1]:
            raise ValueError('Bounded proposal count within this user block required')
        tokens, _ = select_active_candidates(part['hidden'], part['candidates'], part['unary'],
            predecessors, successors, torch.tensor([seed], dtype=torch.int64))
        chosen.append(tuple(int(token) for token in tokens[0, :count]))
    return tuple(chosen)
