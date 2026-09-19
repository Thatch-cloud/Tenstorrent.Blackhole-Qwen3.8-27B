"""Per-row positions and page tables for a target verify block holding several users.

The target verify is already per row. `ModelBatch` builds

    positions = torch.arange(start, start + rows)
    self.positions = upload(positions)
    self.pages = upload(pages.repeat(self.rows, 1))
    self.cos, self.sin = rot_mats_decode(..., positions)

so the model's `_forward_decode` takes one position and one page-table row per
query row already. The single-user assumptions are only that the positions are one
contiguous run and that one page table is repeated across every row.

This module replaces both with a per-user pack: user u contributes `rows_u`
consecutive positions from its own frontier and `rows_u` copies of its own page
table. At one user it reproduces today's tensors exactly.

WHAT THIS IS NOT SUFFICIENT FOR. The 16 full-attention layers are per row, so
per-row pages and positions are all they need. The other 48 layers are GDN, and
there the row axis is TIME for one sequence: `gdn_prefix.decode_projected` runs

    for index, token in enumerate(token_inputs):
        outputs.append(forward(token))
        checkpoint(index + 1)

one row at a time through a single recurrent state, and its projection callback
refuses any batch but one. Packed without handling that, user B's rows would
continue user A's recurrence - wrong for every row of B, and it would leave A's
state advanced by the whole block. The weight-heavy parts of GDN, the qkvzab and
output projections, already run once across all rows and are per-token
independent, so the fix is a state swap at each segment boundary rather than a
batch dimension in the kernels. Until that exists, `ModelBatch` refuses a pack.
"""


def validate_users(users):
    users = tuple(users)
    if not users:
        raise ValueError('At least one packed user required')
    total = 0
    for user in users:
        pages = user.get('pages') if isinstance(user, dict) else None
        if (not isinstance(user, dict) or type(user.get('start')) is not int
                or type(user.get('rows')) is not int or user['rows'] < 1
                or user['start'] < 0 or pages is None
                or getattr(pages, 'ndim', None) != 2 or pages.shape[0] != 1):
            raise ValueError('Each packed user needs an absolute start, a row count and its own page table')
        total += user['rows']
    if total not in (1, 2, 4, 8, 16, 32):
        raise ValueError('Packed rows must total a legal verifier block width')
    if len({user['pages'].shape[1] for user in users}) != 1:
        raise ValueError('Every packed page table must cover the same number of blocks')
    return users, total


def segments(users):
    """Row spans, one per user, in pack order."""
    users, total = validate_users(users)
    spans, cursor = [], 0
    for user in users:
        spans.append((cursor, cursor + user['rows']))
        cursor += user['rows']
    return tuple(spans), total


def packed_rows(users):
    """The per-row `positions` and `pages` a packed ModelBatch needs.

    Returns the same shapes ModelBatch builds today: positions (rows,) int32 and
    pages (rows, blocks) int32, plus the row spans the GDN seam will need.
    """
    import torch

    users, total = validate_users(users)
    spans, _ = segments(users)
    positions = torch.cat([torch.arange(user['start'], user['start'] + user['rows'], dtype=torch.int32)
                           for user in users])
    pages = torch.cat([user['pages'].repeat(user['rows'], 1) for user in users])
    if tuple(positions.shape) != (total,) or tuple(pages.shape) != (total, users[0]['pages'].shape[1]):
        raise AssertionError('Packed positions and pages must cover exactly the block')
    return dict(positions=positions, pages=pages, segments=spans, rows=total)
