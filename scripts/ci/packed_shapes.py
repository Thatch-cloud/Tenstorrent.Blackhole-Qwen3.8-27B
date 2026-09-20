"""The M3 packed verify geometry: four T16 users in one 64-row block.

`packed_verifier.PackedShape` keys every shape-dependent choice of the packed block on
(users, rows_per_user, block_rows, page_width, capacity); the engine's `validate_shape`
and `m1_shape` stop at the 32-row block. This module holds the 64-row shape and its
bounds apart from the engine, so M3's geometry is defined, tested and budgeted before
the engine takes it (docs/packed-device-step-plan-2026-09-20.md section 5; part 2 wires
`m3_shape` into `PackedVerifierEngine`, which should then take `PackedShape` from here).

Only the VERIFY block is 64 rows. The draft proposes in 32-row passes, two T16 users
each (dflash_packed_proposal.py, BLOCK WIDTH), and each user's commit is that user's
16-row histories (gdn_records.RetainedGDNBlock.commit_user).
"""

from typing import NamedTuple

PAGE_TOKENS = 64
# 4352 tokens in whole pages: serving_fast_policy.MINIMUM_MODEL_LEN / PAGE_TOKENS.
MINIMUM_PAGE_WIDTH = 68
# A user's share of the block is a captured verify width (verifier_engine.VERIFY_WIDTHS
# below the M3 block itself).
ROWS_PER_USER = (1, 2, 4, 8, 16, 32)
# 64 is M3; every narrower width is the 32-row block's (packed_verifier.validate_shape).
BLOCK_ROWS = (2, 4, 8, 16, 32, 64)
M3_USERS, M3_ROWS_PER_USER, M3_BLOCK_ROWS = 4, 16, 64


class PackedShape(NamedTuple):
    users: int
    rows_per_user: int
    block_rows: int
    page_width: int
    capacity: int


def validate_shape(shape):
    """A packed shape by its fields, so the engine's own `PackedShape` passes too."""
    if (getattr(shape, '_fields', None) != PackedShape._fields
            or any(type(value) is not int for value in shape)
            or shape.users < 1 or shape.rows_per_user not in ROWS_PER_USER
            or shape.block_rows not in BLOCK_ROWS or shape.users * shape.rows_per_user != shape.block_rows
            or shape.page_width < MINIMUM_PAGE_WIDTH or shape.capacity != shape.page_width * PAGE_TOKENS):
        raise ValueError('A packed shape needs users x rows_per_user == block_rows within the 64-row block, '
                         'and a page-table width of whole 64-token pages')
    return shape


def m3_shape(page_width):
    """Four T16 users in one 64-row block over the serving page-table width."""
    return validate_shape(PackedShape(M3_USERS, M3_ROWS_PER_USER, M3_BLOCK_ROWS,
                                      page_width, page_width * PAGE_TOKENS))


def segment_rows(shape, segment):
    """Rows [start, stop) of one user's segment: segment u is rows_per_user * u onward."""
    validate_shape(shape)
    if type(segment) is not int or not 0 <= segment < shape.users:
        raise ValueError('Segment outside the packed block')
    return segment * shape.rows_per_user, (segment + 1) * shape.rows_per_user


def commit_traces(shape):
    """Commit traces the block captures: one per user per nonzero prefix (prefix zero
    is a no-op on the carry), so users x rows_per_user - 64 for M3."""
    validate_shape(shape)
    return shape.users * shape.rows_per_user
