"""The packed verify geometries: two T16 users in one 32-row block (M1), four in one
64-row block (M3).

`PackedShape` keys every shape-dependent choice of the packed block
(packed_verifier.PackedVerifierEngine) on (users, rows_per_user, block_rows, page_width,
capacity): the segments bound to pool slots, the checkpoint sets, the taps, the per-user
replay readers, the staged words and tables, the commit traces. The engine takes its
shape and validation from here, so the geometry is defined, tested and budgeted in one
place (docs/packed-device-step-plan-2026-09-20.md section 5), and `serving_shape` is the
one rule the serving runtime applies: two scheduler requests take M1, four take M3, any
other count builds no block.

Only the VERIFY block is 64 rows. The draft proposes per user in 32-row passes (16 live
rows each, dflash_packed_proposal.py, BLOCK WIDTH), and each user's commit is that user's
16-row histories (gdn_records.RetainedGDNBlock.commit_user).
"""

from typing import NamedTuple

PAGE_TOKENS = 64
# 4352 tokens in whole pages: serving_fast_policy.MINIMUM_MODEL_LEN / PAGE_TOKENS.
MINIMUM_PAGE_WIDTH = 68
# A user's share of the block is a captured verify width (verifier_engine.VERIFY_WIDTHS
# below the M3 block itself).
ROWS_PER_USER = (1, 2, 4, 8, 16, 32)
# 64 is M3; every narrower width is the 32-row block's.
BLOCK_ROWS = (2, 4, 8, 16, 32, 64)
M1_USERS, M1_ROWS_PER_USER, M1_BLOCK_ROWS = 2, 16, 32
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


def m1_shape(page_width):
    """Two T16 users in one 32-row block over the serving page-table width."""
    return validate_shape(PackedShape(M1_USERS, M1_ROWS_PER_USER, M1_BLOCK_ROWS,
                                      page_width, page_width * PAGE_TOKENS))


def m3_shape(page_width):
    """Four T16 users in one 64-row block over the serving page-table width."""
    return validate_shape(PackedShape(M3_USERS, M3_ROWS_PER_USER, M3_BLOCK_ROWS,
                                      page_width, page_width * PAGE_TOKENS))


# The block the serving runtime builds for a scheduler request count: the T16 share is
# fixed by the draft (16 rows per user), so the count picks the block width.
SERVING_SHAPES = {M1_USERS: m1_shape, M3_USERS: m3_shape}


def serving_shape(users, page_width):
    """The packed block for this many scheduler requests, or None when no captured
    block serves that count: two requests take M1, four take M3, any other count
    stays on the sequential step (serving_runtime.py)."""
    if type(users) is not int or users < 1:
        raise ValueError('Positive integer scheduler request count required')
    builder = SERVING_SHAPES.get(users)
    return None if builder is None else builder(page_width)


# The widest verify width a per-request engine captures beside a packed block. The
# engines serve only the sequential fallback (a narrow last ticket, the survivors after a
# partner finished), and beside the 64-row M3 block their 8- and 16-row captures - about
# 2.4 GB per engine, four engines - do not fit: run 35509307389 (image v52) had 32.9 of
# 33.1 GB per chip allocated when the first engine was built (docs, M3 memory budget).
# Beside the 32-row M1 block they fit, and the two-user gate ran with them (image v47).
SEQUENTIAL_CAPTURE_ROWS = 16
M3_SEQUENTIAL_CAPTURE_ROWS = 4


def sequential_capture_rows(shape):
    """The widest verify width a per-request engine captures beside this block: the
    engine's full T16 set with no block or the M1 block, four rows beside the M3 block
    (widths 1, 2, 4: the survivors decode sequentially at four rows per round)."""
    if shape is None:
        return SEQUENTIAL_CAPTURE_ROWS
    validate_shape(shape)
    return M3_SEQUENTIAL_CAPTURE_ROWS if shape.block_rows == M3_BLOCK_ROWS else SEQUENTIAL_CAPTURE_ROWS


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
