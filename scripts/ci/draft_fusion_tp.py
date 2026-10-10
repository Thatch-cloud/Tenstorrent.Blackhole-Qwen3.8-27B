"""The shared parts of the drafter fusion levers of tp4/fx-wp6 (op-fusion programme, work package 6): flags, markers, core plans and the
eager audit.

THE FOUR LEVERS (all default off, strict 0 or 1, four cards only; each has an _AUDIT twin that needs it):

  QWEN_FAST_DRAFT_REDUCE   F-F1  draft_reduce_tp   the drafter's gather-add chain - the dim-0 all-gather stays, but its four slices and three
                                                    fp32 adds become ONE launch that adds ((p0 + p1) + p2) + p3 tile by tile.
  QWEN_FAST_DRAFT_TAIL     F-F3a draft_tail_tp      the MLP branch's SwiGLU (nine launches) and its residual tail (four launches) become
                                                    one launch each, with every bf16 rounding point of the served composition kept.
  QWEN_FAST_DRAFT_GATEUP1  F-F3c draft_gateup_tp    the gate and up projections as one matmul launch over a load-time concatenation of
                                                    the two weights (the output columns are independent, so each equals the separate
                                                    matmul's column).

EXACTNESS CLASS. All three are bit-identical by construction: the same SFPU fp32 add order, the same typecast and silu primitives at the
same rounding points, the same matmul K loop. A drafter change cannot change a served token (the target verifies every proposal), only
the proposals, so the gate is the eager audit below plus the draft-singles audit and the position-keyed accepted-prefix
compare. None of the three is tau-only, so none needs a tau A/B; if an audit ever says otherwise the lever is NO-GO, not a tau question.

THE AUDIT (QWEN_FAST_DRAFT_<LEVER>_AUDIT=1, needs its lever). The served composition runs beside the fused launch on the very same
operands and every chip's bytes are compared (fp32 and bf16 as integer bit patterns: -0 and +0 differ, NaN payloads count). It is EAGER:
it reads back, so it can only run outside a trace capture. The drafter buckets warm every pass eagerly before they capture it
(dflash_proposal_trace: execute, synchronize, then capture_operation), so the first calls of each (lever, site, rows) are the warm
pass's, on real activations and real weights, and only those are audited (AUDIT_CALLS per key: all five layers' chains); later calls and
every call inside a capture run the fused launch alone. A mismatch logs the *_MISMATCH marker and raises. An audit flag with no lever
raises at the first read, as tp4_sampdraft's do: it would pass having compared nothing.

MARKERS (every FELL BACK and MISMATCH line fails a gated arm, draft_wp6_smoke.problems; ENGAGED is the proof a lever ran):
  [PINDIAG] tp4 draft reduce engaged chains=<n> ...     [PINDIAG] tp4 draft reduce fell back ...     [PINDIAG] tp4 draft reduce audit <n> exact=True ...
  [PINDIAG] tp4 draft tail engaged ...                  [PINDIAG] tp4 draft tail fell back ...       [PINDIAG] tp4 draft tail audit <n> exact=True ...
  [PINDIAG] tp4 draft gateup1 engaged ...               [PINDIAG] tp4 draft gateup1 fell back ...    [PINDIAG] tp4 draft gateup1 audit <n> exact=True ...
  [PINDIAG] tp4 draft mmgrid engaged ...                [PINDIAG] tp4 draft mmgrid fell back ...     [PINDIAG] tp4 draft mmgrid audit <n> exact=True ...
  (and "... audit mismatch ..." for each).

THE GRID. Every launch reads the compute grid from the device (mesh.compute_with_storage_grid_size(): 11 x 10 until 2026-10-10 07:10Z,
13 x 10 since the firmware unlock) and spreads its tiles over min(tiles, grid cores, WORKER_CAP) cores in row-major order; nothing
here names 11 or 13. A grid that cannot be read is a logged fall-back to the served ops. What 13 x 10 changes: the reduce chain
(320 tiles at 64 rows) and the residual (320 tiles) run on 64 / 110 / 130 cores = 5 / 3 / 3 tiles each, so a wider grid helps up to
WORKER_CAP only; the fused gate|up matmul keeps its 8 x 10 program grid (68 cores) on both, because 272 output tiles split into whole
per-core columns only as 4 x 68 (2 x 136 would need 136 cores, more than either grid has).

Stdlib only at import (torch inside the audit), importable on py 3.7.
"""

import os
import sys

import tp_shapes

REDUCE = 'QWEN_FAST_DRAFT_REDUCE'
TAIL = 'QWEN_FAST_DRAFT_TAIL'
GATEUP1 = 'QWEN_FAST_DRAFT_GATEUP1'
MM_GRID = 'QWEN_FAST_DRAFT_MM_GRID'
REDUCE_AUDIT = REDUCE + '_AUDIT'
TAIL_AUDIT = TAIL + '_AUDIT'
GATEUP1_AUDIT = GATEUP1 + '_AUDIT'
MM_GRID_AUDIT = MM_GRID + '_AUDIT'

LEVERS = (REDUCE, TAIL, GATEUP1, MM_GRID)
AUDITS = {REDUCE_AUDIT: REDUCE, TAIL_AUDIT: TAIL, GATEUP1_AUDIT: GATEUP1, MM_GRID_AUDIT: MM_GRID}
ALL_FLAGS = LEVERS + tuple(AUDITS)

REDUCE_ENGAGED = '[PINDIAG] tp4 draft reduce engaged'
REDUCE_FALLBACK = '[PINDIAG] tp4 draft reduce fell back'
REDUCE_AUDIT_LINE = '[PINDIAG] tp4 draft reduce audit'
REDUCE_MISMATCH = '[PINDIAG] tp4 draft reduce audit mismatch'
TAIL_ENGAGED = '[PINDIAG] tp4 draft tail engaged'
TAIL_FALLBACK = '[PINDIAG] tp4 draft tail fell back'
TAIL_AUDIT_LINE = '[PINDIAG] tp4 draft tail audit'
TAIL_MISMATCH = '[PINDIAG] tp4 draft tail audit mismatch'
GATEUP1_ENGAGED = '[PINDIAG] tp4 draft gateup1 engaged'
GATEUP1_FALLBACK = '[PINDIAG] tp4 draft gateup1 fell back'
GATEUP1_AUDIT_LINE = '[PINDIAG] tp4 draft gateup1 audit'
GATEUP1_MISMATCH = '[PINDIAG] tp4 draft gateup1 audit mismatch'
MM_GRID_ENGAGED = '[PINDIAG] tp4 draft mmgrid engaged'
MM_GRID_FALLBACK = '[PINDIAG] tp4 draft mmgrid fell back'
MM_GRID_AUDIT_LINE = '[PINDIAG] tp4 draft mmgrid audit'
MM_GRID_MISMATCH = '[PINDIAG] tp4 draft mmgrid audit mismatch'

# (lever flag, audit flag, engaged, fell back, audit, mismatch, what it is): draft_wp6_smoke and the manifest read this one table.
TABLE = ((REDUCE, REDUCE_AUDIT, REDUCE_ENGAGED, REDUCE_FALLBACK, REDUCE_AUDIT_LINE, REDUCE_MISMATCH, 'drafter gather-add reduce'),
         (TAIL, TAIL_AUDIT, TAIL_ENGAGED, TAIL_FALLBACK, TAIL_AUDIT_LINE, TAIL_MISMATCH, 'drafter SwiGLU and residual kernels'),
         (GATEUP1, GATEUP1_AUDIT, GATEUP1_ENGAGED, GATEUP1_FALLBACK, GATEUP1_AUDIT_LINE, GATEUP1_MISMATCH, 'drafter fused gate|up matmul'),
         (MM_GRID, MM_GRID_AUDIT, MM_GRID_ENGAGED, MM_GRID_FALLBACK, MM_GRID_AUDIT_LINE, MM_GRID_MISMATCH, 'drafter matmul grids'))

# Everything the three levers need at run time, by basename: the image copy lists, the overlay and the CPU allowlist must name every entry.
RUNTIME_FILES = ('draft_fusion_tp.py', 'draft_reduce_tp.py', 'draft_reduce_tp_io.cpp', 'draft_reduce_tp_compute.cpp',
                 'draft_tail_tp.py', 'draft_tail_tp_io.cpp', 'draft_tail_tp_swiglu_compute.cpp', 'draft_tail_tp_residual_compute.cpp',
                 'draft_gateup_tp.py', 'draft_mmgrid_tp.py', 'draft_fuse_out.cpp')

TILE = 32
WORKER_CAP = 110                    # most cores any launch of these levers uses, whatever the grid offers
AUDIT_CALLS = 10                    # calls audited per (lever, site, rows): five layers x (attention, MLP) at the quad
_LOGGED = set()
_AUDITED = {}
STATS = {'reduce': 0, 'tail': 0, 'gateup1': 0, 'mmgrid': 0, 'fallback': 0}


# ---------------------------------------------------------------------------------------------
# Flags.
# ---------------------------------------------------------------------------------------------

def _read(name, environ):
    value = (os.environ if environ is None else environ).get(name)
    if value is None or value == '0':
        return False
    if value == '1':
        return True
    raise ValueError('%s must be 0 or 1, got %r' % (name, value))


def _check_width(environ):
    source = os.environ if environ is None else environ
    if tp_shapes.chip_count(source) != tp_shapes.PAIR:
        return
    on = [name for name in ALL_FLAGS if _read(name, source)]
    if on:
        raise ValueError('%s are TP4 levers: they need QWEN_FAST_TP=4, this process serves the pair' % ', '.join(on))


def raw_flag(name, environ=None):
    """The strict 0 / 1 reading of `name` and nothing else (no width check, no import of the lever): what a branch asks before it
    imports a lever module, so a flag-off process never touches the new files."""
    return _read(name, environ)


def enabled(name, environ=None):
    """Whether lever `name` (one of LEVERS) is on. Strict; any WP6 flag at two cards raises."""
    if name not in LEVERS:
        raise ValueError('Unknown drafter fusion lever %r' % (name,))
    _check_width(environ)
    return _read(name, environ)


def audit_enabled(audit, environ=None):
    """Whether audit flag `audit` (one of AUDITS) is on. An audit without its lever is a misconfigured arm and raises."""
    if audit not in AUDITS:
        raise ValueError('Unknown drafter fusion audit %r' % (audit,))
    _check_width(environ)
    if not _read(audit, environ):
        return False
    if not _read(AUDITS[audit], environ):
        raise ValueError('%s needs %s=1: the audit would compare nothing' % (audit, AUDITS[audit]))
    return True


def validate(environ=None):
    """Every flag strict and every audit paired with its lever, checked once at attach (a misconfigured audit arm must fail before the first
    capture, not at the first round's readback). Raises ValueError."""
    _check_width(environ)
    for name in ALL_FLAGS:
        _read(name, environ)
    for audit in AUDITS:
        audit_enabled(audit, environ)


# ---------------------------------------------------------------------------------------------
# Log lines.
# ---------------------------------------------------------------------------------------------

def log_line(message):
    """One line into the server log: loguru where it exists, stderr otherwise (packed_verifier.diagnostic's way). Never raises."""
    try:
        try:
            from loguru import logger
        except ImportError:
            print(message, file=sys.stderr, flush=True)
        else:
            logger.info('{}', message)
    except BaseException:
        pass


def note(kind, text):
    """One marker line per (kind, text), so a loop of ten chains prints its few distinct lines."""
    key = (kind, text)
    if key in _LOGGED:
        return
    _LOGGED.add(key)
    log_line('%s %s' % (kind, text))


def fell_back(kind, site, reason):
    """The served ops will run: say why, once per (site, reason)."""
    STATS['fallback'] += 1
    note(kind, 'site=%s reason=%s' % (site, reason))


# ---------------------------------------------------------------------------------------------
# Core plans.
# ---------------------------------------------------------------------------------------------

def grid_of(mesh):
    """(columns, rows) of the mesh's compute grid, read from the device, or None when it cannot be read."""
    try:
        size = mesh.compute_with_storage_grid_size()
        grid = int(size.x), int(size.y)
    except Exception:  # noqa: BLE001 - a diagnostic read: the caller falls back to the served ops
        return None
    return grid if grid[0] >= 1 and grid[1] >= 1 else None


def worker_count(tiles, grid, cap=WORKER_CAP):
    """How many cores a launch of `tiles` tiles uses: one tile each at most, never more than the grid or `cap`."""
    if type(tiles) is not int or tiles < 1:
        raise ValueError('tiles must be a positive integer, got %r' % (tiles,))
    return min(tiles, grid[0] * grid[1], cap)


def plan_runs(tiles, workers):
    """[(first tile, tile count)] per worker: contiguous runs of `tiles` over `workers` cores, the first `tiles % workers` cores one tile
    longer. Every tile is in exactly one run, in ascending order."""
    if type(tiles) is not int or type(workers) is not int or tiles < 1 or not 1 <= workers <= tiles:
        raise ValueError('%r tiles over %r workers' % (tiles, workers))
    base, extra = divmod(tiles, workers)
    runs, start = [], 0
    for worker in range(workers):
        count = base + (1 if worker < extra else 0)
        runs.append((start, count))
        start += count
    return runs


def coordinates(grid, workers):
    """[(x, y)] of the first `workers` cores of a grid, row-major with the grid's column count: worker w sits at (w % columns, w // columns)."""
    columns = grid[0]
    if workers > grid[0] * grid[1]:
        raise ValueError('%d workers exceed a %d x %d grid' % (workers, grid[0], grid[1]))
    return [(worker % columns, worker // columns) for worker in range(workers)]


def core_ranges(operations, grid, workers):
    """The CoreRangeSet of those cores: the full rows as one rectangle and the last partial row as another."""
    columns = grid[0]
    full, remainder = divmod(workers, columns)
    ranges = []
    if full:
        ranges.append(operations.CoreRange(operations.CoreCoord(0, 0), operations.CoreCoord(columns - 1, full - 1)))
    if remainder:
        ranges.append(operations.CoreRange(operations.CoreCoord(0, full), operations.CoreCoord(remainder - 1, full)))
    return operations.CoreRangeSet(ranges)


def circular_buffer(operations, cores, index, pages, data_format, page_bytes):
    return operations.CBDescriptor(total_size=page_bytes * pages, core_ranges=cores,
        format_descriptors=[operations.CBFormatDescriptor(buffer_index=index, data_format=data_format,
            page_size=page_bytes, tile=operations.TileDescriptor(operations.Tile([32, 32])))])


def fp32_compute_config(operations, fp32_inputs):
    """The compute config of the served eltwise programs for these kernels: HiFi4, fp32 destination, no approximation, and the fp32 input
    circular buffers unpacked straight to the destination (UnpackToDestFp32 at their CB indices)."""
    config = operations.ComputeConfigDescriptor(math_fidelity=operations.MathFidelity.HiFi4, fp32_dest_acc_en=True,
                                                math_approx_mode=False)
    modes = [operations.UnpackToDestMode.Default] * 64
    for index in fp32_inputs:
        modes[index] = operations.UnpackToDestMode.UnpackToDestFp32
    config.unpack_to_dest_mode.extend(modes)
    return config


# ---------------------------------------------------------------------------------------------
# The eager audit.
# ---------------------------------------------------------------------------------------------

def capturing():
    """Whether a drafter capture scope of the samp-draft audits is open (a readback inside a capture is not allowed). False when that
    machinery is not loaded; the warm-pass rule in the module text covers the rest."""
    module = sys.modules.get('tp4_draft_conv')
    if module is None:
        return False
    try:
        return bool(module.capturing())
    except Exception:  # noqa: BLE001
        return False


def audit_due(lever, site, rows):
    """Whether this call is one of the first AUDIT_CALLS of its (lever, site, rows) - and counts it. False inside a capture scope."""
    if capturing():
        return False
    key = (lever, site, rows)
    if _AUDITED.get(key, 0) >= AUDIT_CALLS:
        return False
    _AUDITED[key] = _AUDITED.get(key, 0) + 1
    return True


def audited_total(lever):
    """How many calls of `lever` have been audited so far (the n of the 'audit <n> exact=True' line)."""
    return sum(count for (name, _, _), count in _AUDITED.items() if name == lever)


def reset():
    """Forget the logged lines, the audit counters and the statistics (tests only)."""
    _LOGGED.clear()
    _AUDITED.clear()
    for key in STATS:
        STATS[key] = 0


def chip_bits(operations, tensor):
    """Every chip's tensor as an integer bit pattern (int16 for 16-bit data, int32 for 32-bit), on the host."""
    import torch

    out = []
    for shard in operations.get_device_tensors(tensor):
        host = operations.to_torch(shard).contiguous()
        out.append(host.view(torch.int16 if host.element_size() == 2 else torch.int32))
    return out


def differing_chips(operations, mine, served):
    """[chip] whose bytes differ between the two tensors (shape or bit pattern). Reads back: eager only."""
    import torch

    left, right = chip_bits(operations, mine), chip_bits(operations, served)
    if len(left) != len(right):
        raise AssertionError('The audit compares %d chips against %d' % (len(left), len(right)))
    return [chip for chip, (a, b) in enumerate(zip(left, right)) if a.shape != b.shape or not torch.equal(a, b)]


def audit_result(lever, line, mismatch, site, rows, bad, count, **fields):
    """Log the audit line (exact=True) or the mismatch line and raise. `bad` is the list of differing chips."""
    extra = ''.join(' %s=%s' % (key, value) for key, value in sorted(fields.items()))
    if bad:
        message = '%s site=%s rows=%d chips=%s%s' % (mismatch, site, rows, bad, extra)
        log_line(message)
        raise AssertionError(message)
    log_line('%s %d exact=True site=%s rows=%d%s' % (line, count, site, rows, extra))
