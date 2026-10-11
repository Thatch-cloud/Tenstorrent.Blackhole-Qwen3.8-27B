"""Tile-aligned drafter window (QWEN_FAST_TAP_WINDOW_ALIGNED, default off): the prefill's draft-feature window built from tile-aligned pieces.

WHY. The drafter window is the last 2048 rows [P - 2048, P) of the five tap outputs. It spans two prefill chunks, so today
dflash_prefill_window snapshots a rows piece of each chunk at that chunk's time (slice + clone) and the bridge's first
capture.outputs() joins the two pieces with ttnn.concat. The split point r = (P - 2048) mod 2048 is set by the prompt length,
so a new prompt length brings new piece shapes: rows a = 2048 - r and r. A TILE concat on dim 2 whose inputs carry tile padding
takes ttnn's untilize, row-major concat, retilize path: three programs keyed by the piece shapes, so a new prompt length
compiles them on the request's path, and untilize_with_unpadding's block factory bakes the piece's logical row count into its
writer kernel's compile-time arguments (tt-metal at the image's commit), so a split never served under the image's kernel-cache
key also JIT-builds kernels. Measured on four card runs (217 rebinds; references/fusion-jobs/RB/ORDER.txt): all 91 slow rebinds
compiled the programs of a new unaligned split (+3) and spent 640-770 ms between the slot adoption and the rebind (where the
concat runs); the 68 that compiled the same three programs fast spent 106-176 ms there (binaries already in the persistent
kernel cache: inferred, the cache is not logged), and the 52 with no new program or a tile-aligned split (+1) 59-112 ms, none
slow. The prefill step that took the snapshots pays the same way (12 of 29 cold/hit twin pairs: +187 to +814 ms). The kernel
cache persists per graft key, so the slow mode fades only as served splits accumulate, and every new image key starts it again.

WHAT. With the flag on, a chunk whose piece is not the whole window snapshots the TILE-ALIGNED superset of its piece instead:
rows [floor32(start), ceil32(end)) relative to the chunk (the chunk output carries its whole padded bucket, so the rows after the
valid ones exist; they are cut away below). At the bridge the aligned pieces are contiguous, so the join is a concat of
tile-aligned inputs (ttnn's tiled concat: no untilize) of fixed total height 2048 + 32 when the window is unaligned, and ONE
slice cuts the window out of it at offset s = (P - 2048) mod 32. The shapes any of these programs see are bounded: piece heights
are multiples of 32 (64 values each), and the only unaligned op is that last slice, of a [2080] input at one of 31 offsets. So the
set of kernels the window can ever compile is small and fixed, and after it is cached no new prompt length compiles anything.
A window inside one chunk (P <= 2048, or P a multiple of 2048) and a window already on tile boundaries are snapshotted as today.

EXACTNESS. Every op on the new path is data movement (slice, clone, concat): the window it hands the drafter holds the same
rows of the same chunk outputs, bit for bit, as today's concat of exact pieces. The rows the aligned pieces add (before the
window start in the first piece, the bucket padding after P in the last) are removed by the final slice, which is checked to
leave exactly the window's row count. QWEN_FAST_TAP_WINDOW_ALIGNED_AUDIT=1 (gate only) builds today's composition beside it on
the device (each piece cut to its exact rows, then today's concat) and compares every tap on every chip byte for byte: exact=True
or a mismatch, after which the served composition is used.

FALLBACK. A capture whose aligned pieces are not contiguous (a chunk not starting on a tile boundary, a chunk output shorter than
its aligned end) cuts every piece to its exact rows and joins them as today (logged 'fell back'); the flag off, nothing here is
imported and dflash_prefill_window runs exactly as before.
"""

import os


FLAG = 'QWEN_FAST_TAP_WINDOW_ALIGNED'
AUDIT_FLAG = 'QWEN_FAST_TAP_WINDOW_ALIGNED_AUDIT'
TILE = 32
MARKER = '[PINDIAG] tap window aligned'
ENGAGED_MARKER = MARKER + ' engaged '
FELL_BACK_MARKER = MARKER + ' fell back '
AUDIT_MARKER = MARKER + ' audit '


def _strict_bool(name, environ):
    value = (os.environ if environ is None else environ).get(name, '0')
    if value not in ('0', '1'):
        raise ValueError('%s must be 0 or 1, got %r' % (name, value))
    return value == '1'


def enabled(environ=None):
    """QWEN_FAST_TAP_WINDOW_ALIGNED: unset or '0' off, '1' on, anything else refused."""
    return _strict_bool(FLAG, environ)


def audit_enabled(environ=None):
    """QWEN_FAST_TAP_WINDOW_ALIGNED_AUDIT (gate only): '1' needs the flag itself on; anything but 0/1 refused."""
    audit = _strict_bool(AUDIT_FLAG, environ)
    if audit and not enabled(environ):
        raise ValueError('%s=1 audits the aligned window and needs %s=1' % (AUDIT_FLAG, FLAG))
    return audit


def _floor(value):
    return value // TILE * TILE


def _ceil(value):
    return -(-value // TILE) * TILE


def wanted(window, chunk_start, valid_rows):
    """Whether a chunk's piece of the window is snapshotted aligned: the window does not lie inside this one chunk (a join
    follows) and it is not on tile boundaries already (today's pieces are aligned then, and their join compiles nothing new)."""
    if window['start'] % TILE == 0 and window['end'] % TILE == 0:
        return False
    return not (window['start'] >= chunk_start and window['end'] <= chunk_start + valid_rows)


def aligned_bounds(window, chunk_start, valid_rows, available_rows):
    """(begin, end) relative to the chunk output: the tile-aligned superset of the window rows this chunk holds, or None when
    the chunk cannot give one (no rows of the window, a chunk start off a tile boundary, or an aligned end past the rows the
    chunk output holds) - the caller then snapshots the exact piece as today."""
    start, end = max(window['start'], chunk_start), min(window['end'], chunk_start + valid_rows)
    if end <= start or chunk_start % TILE:
        return None
    begin, stop = _floor(start - chunk_start), _ceil(end - chunk_start)
    if stop > available_rows:
        return None
    return begin, stop


def covers(window, pieces, heights):
    """What each snapshotted part holds: for a piece (chunk_start, valid_rows) of the window, dict(begin, end) its absolute rows
    - the aligned superset when the part's height is the superset's, else the exact piece when the height is the piece's -
    dict(start, stop) the piece's exact rows, and aligned (begin and end on tile boundaries). Raises on any other height."""
    result = []
    for (chunk_start, valid_rows), height in zip(pieces, heights, strict=True):
        start, stop = max(window['start'], chunk_start), min(window['end'], chunk_start + valid_rows)
        if stop <= start:
            raise ValueError('A snapshotted part holds no row of the window')
        begin, end = start, stop
        if chunk_start % TILE == 0 and height != stop - start:
            begin, end = chunk_start + _floor(start - chunk_start), chunk_start + _ceil(stop - chunk_start)
        if height != end - begin:
            raise ValueError('A snapshotted part of %d rows is neither its piece (%d) nor its aligned superset' % (height, stop - start))
        result.append(dict(begin=begin, end=end, start=start, stop=stop, aligned=begin % TILE == 0 and end % TILE == 0))
    return result


def expanded(cover):
    """Whether any part holds rows beyond its exact piece (an aligned superset was snapshotted)."""
    return any((part['begin'], part['end']) != (part['start'], part['stop']) for part in cover)


def plan(window, cover):
    """dict(lead=, union=) when the parts can be joined as tile-aligned inputs and one slice cuts the window out: every part
    aligned, the parts contiguous, their union holding the window. None otherwise (the exact join is used)."""
    if not cover or not all(part['aligned'] for part in cover):
        return None
    if any(cover[index]['end'] != cover[index + 1]['begin'] for index in range(len(cover) - 1)):
        return None
    first, last = cover[0]['begin'], cover[-1]['end']
    if first > window['start'] or last < window['end']:
        return None
    return dict(lead=window['start'] - first, union=last - first)


def _slice_rows(operations, value, begin, end):
    shape = tuple(value.shape)
    return operations.slice(value, (0, 0, begin, 0), (shape[0], shape[1], end, shape[3]))


def _release(operations, addresses, value, keep):
    """Deallocate `value` unless it is (or shares storage with) one of `keep`."""
    own = tuple(addresses(operations, value))
    if any(own == tuple(addresses(operations, other)) for other in keep):
        return
    operations.deallocate(value)


def aligned_join(operations, parts, lead, rows, *, addresses):
    """The window from tile-aligned contiguous parts: one tiled concat (when there is more than one part) and one slice of
    `rows` rows at `lead`. Returns (window tensor, intermediates already released)."""
    joined = parts[0] if len(parts) == 1 else operations.concat(list(parts), dim=2, memory_config=operations.DRAM_MEMORY_CONFIG)
    if lead == 0 and joined.shape[2] == rows:
        return joined
    try:
        window = _slice_rows(operations, joined, lead, lead + rows)
    except BaseException:
        if len(parts) > 1:
            operations.deallocate(joined)
        raise
    if len(parts) > 1:
        _release(operations, addresses, joined, (window,))
    if window.shape[2] != rows:
        raise AssertionError('The aligned window holds %d rows, not %d' % (window.shape[2], rows))
    return window


def exact_join(operations, parts, cover, window, *, addresses):
    """Today's composition from the snapshotted parts: each aligned part cut to its exact piece of the window, then today's
    concat (dflash_prefill_window.PrefillWindowCapture.outputs). Returns the joined window (its temporaries released)."""
    cut, owned = [], []
    try:
        for value, part in zip(parts, cover, strict=True):
            if (part['begin'], part['end']) == (part['start'], part['stop']):
                cut.append(value)
                continue
            piece = _slice_rows(operations, value, part['start'] - part['begin'], part['stop'] - part['begin'])
            cut.append(piece)
            owned.append(piece)
        if len(cut) == 1:
            # The one cut piece IS the window: the caller owns it from here.
            merged, owned = cut[0], []
        else:
            merged = operations.concat(cut, dim=2, memory_config=operations.DRAM_MEMORY_CONFIG)
    finally:
        for value in owned:
            _release(operations, addresses, value, parts)
    return merged


def same_bytes(operations, left, right):
    """Every chip's shard of two device tensors equal byte for byte (bfloat16 compared as int16, so NaN payloads and signed zeros
    count)."""
    import torch

    lefts, rights = operations.get_device_tensors(left), operations.get_device_tensors(right)
    if len(lefts) != len(rights):
        return False
    for a, b in zip(lefts, rights):
        a, b = operations.to_torch(a).contiguous(), operations.to_torch(b).contiguous()
        if a.shape != b.shape or a.dtype != b.dtype:
            return False
        if a.dtype in (torch.bfloat16, torch.float16):
            a, b = a.view(torch.int16), b.view(torch.int16)
        if not torch.equal(a, b):
            return False
    return True
