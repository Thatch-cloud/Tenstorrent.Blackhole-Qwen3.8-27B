"""Matmul program grids re-tuned for the 13 x 10 compute grid the four p150a cards expose since the 2026-10-10 firmware unlock (QWEN_FAST_GRID13, default off;
tp4/m1-grid13, work package M1 of the op-fusion programme).

WHY. On 11 x 10 every matmul program grid of the stack was sized for 11 columns (or fixed at 8). On 13 x 10 the decode profile (v676 against v678) moved by about
1 percent and the solo prefill profile (v701) shows its three fused all-gather matmuls (gdn_in, attn_in, the gate|up SwiGLU: 85.7 of the 266 ms of non-SDPA kernel time
of a 2,048-row chunk) still on the 8 x 9 grid the pinned tp_common helper fixes (`grid = (8, grid[1])`). docs/tp4-fusion-m1.md has the ranked table.

WHAT (three sub-switches; QWEN_FAST_GRID13=1 is all of them, or a comma list of verify, drafter, prefill):

  prefill   the prefill all-gather + matmul (ttnn.experimental.all_gather_minimal_matmul_async, called by tp_common.all_gather_matmul_prefill and
            all_gather_swiglu_prefill) on a 12-wide grid (2 links x 6 workers) with the M block re-chosen so that the busiest core owns fewer output rows:
            the op gives M blocks to the grid columns, ceil(M blocks / columns) rounds, so M = 64 tiles at M_block 4 or 8 is 8 rows a core on 8 columns and
            M_block 6 is 11 blocks on 12 columns, 6 rows a core. K_block_size, N_block_size, the subblocks, the fidelity, the fp32 destination and the packer are the
            served ones. Done WITHOUT copying the pinned body: the pinned function is called as it is, with the `ttnn` global of tp_common viewed through a shim
            that rewrites two keyword arguments (config, num_workers_per_link) of that one call, after a census of the rest (links, transpose, grid, blocks).
  verify    the packed 64-row verify block's output projections (attention wo and GDN out, K = 1,536, N = 5,120) on 3 output tiles a core (54 cores, the shape the
            MLP down projection's card-M sweep found 9 % faster than its 32-core one) instead of 5; the in-projections only when QWEN_FAST_GRID13_CFG names them.
            tp_common.matmul_1d_decode is wrapped: the model built its program configs at ModelArgs, before this lever could be installed, so the swap is at the call,
            once per served config object (the warm pass and the captured launch of a layer get the same lever config). The pinned builder makes the lever config, the
            fields the T1 #11 rule keeps (in0_block_w, per_core_M, fuse_batch, mcast_in0) are checked equal and anything else refuses.
  drafter   the fused commit's feature projection (dflash_device, 80 cores of a fixed (8, 10) grid: 92.8 -> 124.7 us on 13 x 10) on the device-wide grid, through
            commit_program() below. The hook in dflash_device.py is the unapplied patch fusion-wp/M1-dflash-device.patch (that file is not this package's).

EXACTNESS. Nothing here changes a K reduction. A 1D- or 2D-multicast matmul and the minimal matmul give every output tile one core, which accumulates K block by block in
a fixed order into an fp32 destination; the grid, the M and N partition and the subblock move only WHICH core owns a tile. The all-gather part is a copy. This is an
argument, not a proof: the audit (QWEN_FAST_GRID13_AUDIT=1) runs the served configuration beside the lever on the very same operands and compares every chip's bytes.
Eager only: an audit inside a trace capture is skipped (one logged line), and nothing here issues a program inside a capture that the warm forward did not run (the verify
wrapper hands the SAME program config to the warm and the captured launch).

THE GRID. Everything is read from the device. On a device narrower than 13 columns (or shorter than 10 rows, or unreadable) every site falls back to the served ops with
`[PINDIAG] tp4 grid13 fell back reason=...`; nothing here names 11 or 13 beyond MIN_COLUMNS, the first width the plans were derived for.

Markers (the smoke rule is grid13_smoke):
  [PINDIAG] tp4 grid13 engaged site=<site> ...             one per distinct (site, rows, grid)
  [PINDIAG] tp4 grid13 fell back reason=<why> site=<site>  the served ops ran because something was refused (a census, a narrow or unreadable device); one per (site, reason); fails a smoke
  [PINDIAG] tp4 grid13 unchanged reason=<why> site=<site>  the served ops ran because the plan has nothing to gain at this shape (too few rows, no fewer rows a core); information, not a failure
  [PINDIAG] tp4 grid13 audit n=<n> exact=True site=...     a passing audit
  [PINDIAG] tp4 grid13 audit n=<n> exact=mismatch site=... then  [PINDIAG] tp4 grid13 audit mismatch ... and an AssertionError

Stdlib only at import, py 3.7. The flags are strict; any of them at the pair (QWEN_FAST_TP unset) raises.
"""

import os
import sys
import threading
from collections import namedtuple

import tp_shapes

FLAG = 'QWEN_FAST_GRID13'
AUDIT_FLAG = 'QWEN_FAST_GRID13_AUDIT'
CFG_FLAG = 'QWEN_FAST_GRID13_CFG'
ALL_FLAGS = (FLAG, AUDIT_FLAG, CFG_FLAG)
SUBSWITCHES = ('verify', 'drafter', 'prefill')

PREFIX = '[PINDIAG] tp4 grid13'
ENGAGED = PREFIX + ' engaged'
FELL_BACK = PREFIX + ' fell back'
UNCHANGED = PREFIX + ' unchanged'
AUDIT_LINE = PREFIX + ' audit'
AUDIT_MISMATCH = PREFIX + ' audit mismatch'
SKIPPED = PREFIX + ' skipped its audit inside a trace capture'

# Everything this lever needs at run time, by basename (the overlay, the CPU allowlist and the manifest name every entry).
RUNTIME_FILES = ('grid13_tp.py',)

TILE = 32
MIN_COLUMNS = 13                # the first width these plans were derived for (the unlocked cards); a narrower device keeps the served ops
MIN_ROWS = 10
LINKS = 2                       # tp_common.all_gather_matmul_prefill: num_links
SERVED_AGMM_COLUMNS = 8         # ... and grid = (8, rows): 2 links x 4 workers
AGMM_MAX_COLUMNS = 12           # 2 links x 6 workers: the widest even grid of a 13-wide device
AGMM_M_CAP = 8                  # the largest M block any served call uses (the SwiGLU's); a bigger block would take L1 the served ones never did
PRESET_OUT_PER_CORE_N = 3       # output projections: 160 N tiles in 3-tile columns = 54 cores
MIN_M_TILES = 16                # prefill chunks of fewer rows (512) keep the served call: the gain is a fraction of a block there and the block would shrink to one tile
AUDIT_CALLS = 4                 # calls audited per (site, rows)
DECODE_ROWS = 64                # the packed verify block

_LOGGED = set()
_AUDITED = {}
_CAPTURE = {'depth': 0}
_LOCK = threading.RLock()
STATS = {'agmm': 0, 'verify': 0, 'drafter': 0, 'fallback': 0, 'unchanged': 0}
_CONFIGS = {}                   # (id(served config), K, N) -> (served config, (lever config, site) or None): decided once, kept alive here


# ---------------------------------------------------------------------------------------------
# Flags.
# ---------------------------------------------------------------------------------------------

Settings = namedtuple('Settings', ('subs', 'audit', 'cfg'))
Config = namedtuple('Config', ('agmm_cols', 'agmm_m', 'out', 'gdn_in', 'attn_in', 'width'))
DEFAULT_CONFIG = Config(agmm_cols=None, agmm_m=None, out=PRESET_OUT_PER_CORE_N, gdn_in=None, attn_in=None, width=None)


def _source(environ):
    return os.environ if environ is None else environ


def _strict01(name, environ):
    value = _source(environ).get(name)
    if value is None or value == '0':
        return False
    if value == '1':
        return True
    raise ValueError('%s must be 0 or 1, got %r' % (name, value))


def parse_config(text):
    """The Config a QWEN_FAST_GRID13_CFG text spells: comma-separated name=value, each name at most once. agmm=<cols> or agmm=<cols>x<m block> (cols even, at least
    the served 8); out=, gdn_in=, attn_in= the per-core output columns (positive) of the 64-row projections; width= the grid width of those (positive).
    An empty text is the default. ValueError on anything else."""
    if not isinstance(text, str):
        raise ValueError('%s must be text, got %r' % (CFG_FLAG, text))
    values = dict(DEFAULT_CONFIG._asdict())
    seen = set()
    for item in [part for part in text.split(',') if part != ''] if text.strip() else []:
        name, separator, value = item.partition('=')
        if not separator or name in seen or name not in ('agmm', 'out', 'gdn_in', 'attn_in', 'width'):
            raise ValueError('%s=%r: expected agmm=<cols>[x<m>], out=<n>, gdn_in=<n>, attn_in=<n> or width=<n>, each once (bad item %r)' % (CFG_FLAG, text, item))
        seen.add(name)
        if name == 'agmm':
            columns, separator, block = value.partition('x')
            if not columns.isdigit() or (separator and not block.isdigit()):
                raise ValueError('%s: agmm=%r is not <cols> or <cols>x<m>' % (CFG_FLAG, value))
            cols = int(columns)
            if cols < SERVED_AGMM_COLUMNS or cols % LINKS:
                raise ValueError('%s: agmm columns %d must be even and at least %d' % (CFG_FLAG, cols, SERVED_AGMM_COLUMNS))
            values['agmm_cols'] = cols
            if block != '':
                if int(block) < 1:
                    raise ValueError('%s: the M block must be positive' % CFG_FLAG)
                values['agmm_m'] = int(block)
        else:
            if not value.isdigit() or int(value) < 1:
                raise ValueError('%s: %s=%r is not a positive integer' % (CFG_FLAG, name, value))
            values[name] = int(value)
    return Config(**values)


def parse(environ=None):
    """None when the lever is off (QWEN_FAST_GRID13 unset or 0), else the Settings. 1 is every sub-switch; otherwise a comma list of SUBSWITCHES, each at most once.
    An audit or a CFG without the lever, a malformed value, or any of it at the pair, raises ValueError."""
    source = _source(environ)
    raw = source.get(FLAG)
    on = raw is not None and raw != '0'
    if not on:
        if _strict01(AUDIT_FLAG, source):
            raise ValueError('%s needs %s: it would audit nothing' % (AUDIT_FLAG, FLAG))
        if source.get(CFG_FLAG, '') not in ('', '0'):
            raise ValueError('%s needs %s: it would configure nothing' % (CFG_FLAG, FLAG))
        return None
    if tp_shapes.chip_count(source) == tp_shapes.PAIR:
        raise ValueError('%s is a TP4 lever: it needs QWEN_FAST_TP=4, this process serves the pair' % FLAG)
    if raw == '1':
        subs = frozenset(SUBSWITCHES)
    else:
        names = raw.split(',')
        if any(name not in SUBSWITCHES for name in names) or len(set(names)) != len(names):
            raise ValueError('%s must be 0, 1 or a comma list of %s (each once), got %r' % (FLAG, ', '.join(SUBSWITCHES), raw))
        subs = frozenset(names)
    audit = _strict01(AUDIT_FLAG, source)
    cfg = parse_config(source.get(CFG_FLAG, ''))
    return Settings(subs=subs, audit=audit, cfg=cfg)


def enabled(environ=None):
    return parse(environ) is not None


# ---------------------------------------------------------------------------------------------
# Log lines and the eager audit.
# ---------------------------------------------------------------------------------------------

def log_line(message):
    """One line into the server log: loguru where it exists, stderr otherwise. Never raises."""
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
    """One marker line per (kind, text)."""
    key = (kind, text)
    with _LOCK:
        if key in _LOGGED:
            return
        _LOGGED.add(key)
    log_line('%s %s' % (kind, text))


def fell_back(site, reason):
    """The served ops will run: say why, once per (site, reason)."""
    STATS['fallback'] += 1
    note(FELL_BACK, 'reason=%s site=%s' % (reason, site))


def unchanged(site, reason):
    """The served ops will run because the plan changes nothing here (not a refusal): say why, once per (site, reason)."""
    STATS['unchanged'] += 1
    note(UNCHANGED, 'reason=%s site=%s' % (reason, site))


def reset():
    """Forget the logged lines, the audit counters, the registry and the statistics (tests only)."""
    _LOGGED.clear()
    _AUDITED.clear()
    _CONFIGS.clear()
    _CAPTURE['depth'] = 0
    for key in STATS:
        STATS[key] = 0


def _base(operations):
    for _ in range(8):
        inner = getattr(operations, 'original', None)
        if inner is None or inner is operations:
            break
        operations = inner
    return operations


def track(operations):
    """Wrap begin_trace_capture / end_trace_capture on the ttnn module once, so that in_capture() knows when a capture is open. Idempotent; an object without the two
    functions (a fake that never captures) is left alone."""
    target = _base(operations)
    begin, end = getattr(target, 'begin_trace_capture', None), getattr(target, 'end_trace_capture', None)
    if begin is None or end is None or getattr(begin, 'tracked_by_grid13', False):
        return

    def begin_capture(*args, **options):
        result = begin(*args, **options)
        _CAPTURE['depth'] += 1
        return result

    def end_capture(*args, **options):
        try:
            return end(*args, **options)
        finally:
            _CAPTURE['depth'] = max(0, _CAPTURE['depth'] - 1)

    begin_capture.tracked_by_grid13 = True
    end_capture.tracked_by_grid13 = True
    target.begin_trace_capture = begin_capture
    target.end_trace_capture = end_capture


def in_capture(operations=None):
    """Whether a trace capture is open: a tracked begin without its end, or another fusion package's capture scope."""
    if operations is not None:
        track(operations)
    if _CAPTURE['depth'] > 0:
        return True
    other = sys.modules.get('draft_fusion_tp')
    try:
        return bool(other is not None and other.in_capture())
    except Exception:  # noqa: BLE001 - a diagnostic read
        return False


CAPTURE_WORDS = ('trace capture', 'capturing')


def audit_begin(operations, mesh, site, rows):
    """Whether the caller may run its eager audit NOW (the device synchronized when it says yes): never inside a capture, never past AUDIT_CALLS of this (site, rows),
    and a synchronize the device refuses for a capture is a skip, not a failure."""
    track(operations)
    if in_capture():
        note(SKIPPED, 'site=%s rows=%d' % (site, rows))
        return False
    key = (site, rows)
    with _LOCK:
        if _AUDITED.get(key, 0) >= AUDIT_CALLS:
            return False
    try:
        operations.synchronize_device(mesh)
    except Exception as failure:  # noqa: BLE001 - classified below
        if any(word in str(failure).lower() for word in CAPTURE_WORDS):
            note(SKIPPED, 'site=%s rows=%d' % (site, rows))
            return False
        raise
    with _LOCK:
        _AUDITED[key] = _AUDITED.get(key, 0) + 1
    return True


def audited_total():
    return sum(_AUDITED.values())


def _bits(operations, tensor):
    """Every chip's tensor as comparable integers: a torch int16 / int32 view where torch exists, else the host object's own list."""
    out = []
    for shard in operations.get_device_tensors(tensor):
        host = operations.to_torch(shard)
        if hasattr(host, 'view') and hasattr(host, 'element_size'):
            import torch

            host = host.contiguous()
            out.append(('torch', tuple(host.shape), host.view(torch.int16 if host.element_size() == 2 else torch.int32)))
        else:
            out.append(('list', tuple(getattr(host, 'shape', ())), list(host.tolist() if hasattr(host, 'tolist') else host)))
    return out


def differing_chips(operations, mine, served):
    """[chip] whose bytes differ between the two tensors (shape or bit pattern). Reads back: eager only."""
    left, right = _bits(operations, mine), _bits(operations, served)
    if len(left) != len(right):
        raise AssertionError('The audit compares %d chips against %d' % (len(left), len(right)))
    bad = []
    for chip, (a, b) in enumerate(zip(left, right)):
        if a[0] != b[0] or a[1] != b[1]:
            bad.append(chip)
        elif a[0] == 'torch':
            import torch

            if not torch.equal(a[2], b[2]):
                bad.append(chip)
        elif a[2] != b[2]:
            bad.append(chip)
    return bad


def audit_result(site, rows, bad, **fields):
    """Log the audit line (exact=True) or the mismatch lines and raise. `bad` is the list of differing chips."""
    extra = ''.join(' %s=%s' % (key, value) for key, value in sorted(fields.items()))
    count = audited_total()
    if bad:
        log_line('%s n=%d exact=mismatch site=%s rows=%d chips=%s%s' % (AUDIT_LINE, count, site, rows, bad, extra))
        message = '%s site=%s rows=%d chips=%s%s' % (AUDIT_MISMATCH, site, rows, bad, extra)
        log_line(message)
        raise AssertionError(message)
    log_line('%s n=%d exact=True site=%s rows=%d%s' % (AUDIT_LINE, count, site, rows, extra))


def grid_of(mesh):
    """(columns, rows) of the mesh's compute grid, read from the device, or None when it cannot be read."""
    try:
        size = mesh.compute_with_storage_grid_size()
        grid = int(size.x), int(size.y)
    except Exception:  # noqa: BLE001 - a diagnostic read: the caller falls back to the served ops
        return None
    return grid if grid[0] >= 1 and grid[1] >= 1 else None


def xy(grid):
    """(x, y) of a grid given as a tuple or as a CoreCoord-like object."""
    if hasattr(grid, 'x') and hasattr(grid, 'y'):
        return int(grid.x), int(grid.y)
    return int(grid[0]), int(grid[1])


def too_small(grid):
    """Why a device grid is not one these plans were derived for (a short reason), or None."""
    if grid is None:
        return 'the device compute grid cannot be read'
    if grid[0] < MIN_COLUMNS or grid[1] < MIN_ROWS:
        return 'the device grid is %dx%d, the plans need %dx%d or wider' % (grid[0], grid[1], MIN_COLUMNS, MIN_ROWS)
    return None


# ---------------------------------------------------------------------------------------------
# Plans (pure functions).
# ---------------------------------------------------------------------------------------------

def agmm_columns(device_columns):
    """The widest grid the all-gather matmul can take on a device `device_columns` wide: even (2 links x workers), at most AGMM_MAX_COLUMNS."""
    return min(AGMM_MAX_COLUMNS, (device_columns // LINKS) * LINKS)


def rows_per_core(m_tiles, columns, m_block):
    """Output rows (tiles) the busiest core owns: the M blocks go to the grid columns, ceil(M blocks / columns) rounds of m_block rows."""
    if m_tiles < 1 or columns < 1 or m_block < 1:
        raise ValueError('rows_per_core(%r, %r, %r)' % (m_tiles, columns, m_block))
    blocks = -(-m_tiles // m_block)
    return m_block * -(-blocks // columns)


def pick_m_block(m_tiles, columns, cap=AGMM_M_CAP):
    """The M block (1..cap) minimising the busiest core's rows on `columns` columns; of equal rows the LARGEST block (more reuse of each weight tile per read)."""
    best = None
    for block in range(1, cap + 1):
        key = (rows_per_core(m_tiles, columns, block), -block)
        if best is None or key < best[0]:
            best = (key, block)
    return best[1]


class Declined(str):
    """Why a plan found nothing to change (a str, so a caller that only tests for text still works); a plain str is a refusal."""


AgmmPlan = namedtuple('AgmmPlan', ('columns', 'rows', 'm_block', 'workers', 'served_columns', 'served_m_block', 'rows_per_core', 'served_rows_per_core'))


def agmm_plan(m_tiles, served_columns, served_rows, served_m_block, device_grid, cfg=DEFAULT_CONFIG, links=LINKS):
    """The lever's grid and M block for an all-gather matmul over `m_tiles` rows whose served call used `served_columns` x `served_rows` and M block `served_m_block`
    on `device_grid`; or a reason (a string) why the served call stays: a plain str is a refusal (a device this plan was not derived for), a Declined is a plan with
    nothing to gain. Strictly fewer rows a core than the served one, or no plan."""
    why = too_small(device_grid)
    if why is not None:
        return why
    if m_tiles < MIN_M_TILES:
        return Declined('only %d tile rows, below the %d the plan is derived for' % (m_tiles, MIN_M_TILES))
    columns = cfg.agmm_cols or agmm_columns(device_grid[0])
    if columns % links:
        return 'columns %d are not a multiple of the %d links' % (columns, links)
    if columns > device_grid[0]:
        return 'columns %d exceed the device width %d' % (columns, device_grid[0])
    if served_rows + 1 > device_grid[1]:
        return 'the served rows %d leave no row for the mux cores on a %d-row device' % (served_rows, device_grid[1])
    block = cfg.agmm_m or pick_m_block(m_tiles, columns)
    served = rows_per_core(m_tiles, served_columns, served_m_block)
    mine = rows_per_core(m_tiles, columns, block)
    if mine >= served:
        return Declined('no gain: %d rows a core on %d columns at M block %d against the served %d' % (mine, columns, block, served))
    return AgmmPlan(columns=columns, rows=served_rows, m_block=block, workers=columns // links, served_columns=served_columns,
                    served_m_block=served_m_block, rows_per_core=mine, served_rows_per_core=served)


def wide_grid(device_grid, cores):
    """(columns, rows) of a program grid for `cores` cores laid out row-major in rows as wide as the device; ValueError when it does not fit."""
    if type(cores) is not int or cores < 1:
        raise ValueError('cores must be a positive integer, got %r' % (cores,))
    width = min(device_grid[0], cores)
    rows = -(-cores // width)
    if rows > device_grid[1]:
        raise ValueError('%d cores need %d rows of %d, the grid has %d' % (cores, rows, width, device_grid[1]))
    return (width, rows)


# ---------------------------------------------------------------------------------------------
# prefill: the all-gather matmul behind a shim of the `ttnn` tp_common sees.
# ---------------------------------------------------------------------------------------------

class _View(object):
    """A module as a caller sees it with some attributes replaced: everything else is the module's own."""

    def __init__(self, real, **replaced):
        object.__setattr__(self, '_real', real)
        for name, value in replaced.items():
            object.__setattr__(self, name, value)

    def __getattr__(self, name):
        return getattr(object.__getattribute__(self, '_real'), name)


def agmm_site(kwargs):
    """The site name of an all-gather matmul call, from its weight width and the SwiGLU flag (a label for the log and the audit key)."""
    if kwargs.get('fuse_swiglu'):
        return 'gate_up'
    try:
        width = int(kwargs['weight_tensor'].shape[-1])
    except Exception:  # noqa: BLE001
        return 'agmm'
    geometry = tp_shapes.active()
    return {geometry.gdn_qkvzab_padded: 'gdn_in', geometry.attn_out * 2 + 2 * geometry.attn_kv_heads * tp_shapes.ATTENTION_HEAD_DIM: 'attn_in',
            geometry.mlp: 'mlp_gate_or_up'}.get(width, 'agmm')


def _census(kwargs, config):
    """Why this all-gather matmul call is not the shape the plan was derived for (a short reason), or None."""
    for name in ('input_tensor', 'weight_tensor', 'config', 'num_links', 'num_workers_per_link', 'force_transpose'):
        if name not in kwargs:
            return 'the call has no %s keyword' % name
    if kwargs['num_links'] != LINKS:
        return 'num_links %r is not %d' % (kwargs['num_links'], LINKS)
    if kwargs['force_transpose'] is not True:
        return 'force_transpose is not set (the mux cores sit in another row or column)'
    for field in ('M_block_size', 'K_block_size', 'N_block_size', 'subblock_h', 'subblock_w', 'compute_with_storage_grid_size'):
        if not hasattr(config, field):
            return 'the config has no %s' % field
    grid = config.compute_with_storage_grid_size
    if kwargs['num_workers_per_link'] * LINKS != grid.x:
        return 'grid width %d is not links x workers (%r)' % (grid.x, kwargs['num_workers_per_link'])
    if grid.x != SERVED_AGMM_COLUMNS:
        return 'the served grid width is %d, not %d' % (grid.x, SERVED_AGMM_COLUMNS)
    if config.subblock_h != 1:
        return 'subblock_h %r is not 1' % (config.subblock_h,)
    return None


class _Shim(object):
    """ttnn.experimental.all_gather_minimal_matmul_async as tp_common sees it inside one wrapped call."""

    def __init__(self, real, settings):
        self.real = real                                  # the real ttnn module
        self.settings = settings
        self.gather = real.experimental.all_gather_minimal_matmul_async
        self.engaged = False

    def __call__(self, *args, **kwargs):
        if args:
            fell_back('agmm', '%d positional arguments' % len(args))
            return self.gather(*args, **kwargs)
        config = kwargs.get('config')
        site = agmm_site(kwargs)
        reason = _census(kwargs, config) if config is not None else 'the call has no config'
        if reason is not None:
            fell_back(site, reason)
            return self.gather(**kwargs)
        try:
            device = kwargs['input_tensor'].device()
        except Exception:  # noqa: BLE001
            device = None
        grid = grid_of(device) if device is not None else None
        rows = int(kwargs['input_tensor'].shape[-2])
        plan = agmm_plan(-(-rows // TILE), config.compute_with_storage_grid_size.x, config.compute_with_storage_grid_size.y, config.M_block_size,
                         grid, self.settings.cfg)
        if isinstance(plan, str):
            (unchanged if isinstance(plan, Declined) else fell_back)(site, plan)
            return self.gather(**kwargs)
        coordinate = type(config.compute_with_storage_grid_size)
        lever = type(config)(M_block_size=plan.m_block, K_block_size=config.K_block_size, N_block_size=config.N_block_size, subblock_h=config.subblock_h,
                             subblock_w=config.subblock_w, compute_with_storage_grid_size=coordinate(plan.columns, plan.rows))
        call = dict(kwargs)
        call['config'] = lever
        call['num_workers_per_link'] = plan.workers
        self.engaged = True
        STATS['agmm'] += 1
        note(ENGAGED, 'site=%s rows=%d grid=%dx%d served_grid=%dx%d m_block=%d served_m_block=%d workers=%d rows_per_core=%d served_rows_per_core=%d device=%dx%d' % (
            site, rows, plan.columns, plan.rows, plan.served_columns, plan.rows, plan.m_block, plan.served_m_block, plan.workers,
            plan.rows_per_core, plan.served_rows_per_core, grid[0], grid[1]))
        return self.gather(**call)


def wrap_agmm(original, settings):
    """The pinned all-gather matmul function with the lever's shim around its one device call. `original` is called unchanged."""
    namespace = original.__globals__

    def twin(*args, **kwargs):
        real = namespace['ttnn']
        shim = _Shim(real, settings)
        view = _View(real, experimental=_View(real.experimental, all_gather_minimal_matmul_async=shim))
        with _LOCK:
            namespace['ttnn'] = view
            try:
                lever = original(*args, **kwargs)
            finally:
                namespace['ttnn'] = real
        if settings.audit and shim.engaged:
            _audit_agmm(real, original, lever, args, kwargs)
        return lever

    twin.grid13_of = original
    twin.__name__ = getattr(original, '__name__', 'agmm')
    twin.__doc__ = getattr(original, '__doc__', None)
    return twin


def _audit_agmm(operations, original, lever, args, kwargs):
    """Run the pinned function again, unshimmed, on the same operands and compare every chip's bytes with the lever's output. Only after audit_begin said yes."""
    x = args[0] if args else kwargs['x']
    rows = int(x.shape[-2])
    try:
        mesh = x.device()
    except Exception:  # noqa: BLE001
        return
    weight = args[1] if len(args) > 1 else kwargs.get('weight')
    site = 'gate_up' if original.__name__.endswith('swiglu_prefill') else agmm_site(dict(weight_tensor=weight))
    if not audit_begin(operations, mesh, site, rows):
        return
    served = original(*args, **kwargs)
    try:
        operations.synchronize_device(mesh)
        bad = differing_chips(operations, lever, served)
    finally:
        try:
            operations.deallocate(served)
        except Exception:  # noqa: BLE001 - a temporary of an audit: never mask the verdict
            pass
    audit_result(site, rows, bad, kind='agmm')


# ---------------------------------------------------------------------------------------------
# verify: the packed 64-row block's projections through the pinned config builder.
# ---------------------------------------------------------------------------------------------

KEPT_FIELDS = ('in0_block_w', 'per_core_M', 'fuse_batch', 'mcast_in0')
# Every constructor field of the 1D multicast matmul program config but the grid (ttnn's MatmulMultiCoreReuseMultiCast1DProgramConfig): commit_program carries them all over.
PROGRAM_FIELDS = ('in0_block_w', 'out_subblock_h', 'out_subblock_w', 'out_block_h', 'out_block_w', 'per_core_M', 'per_core_N', 'fuse_batch', 'fused_activation',
                  'mcast_in0', 'gather_in0', 'hop_cores', 'num_global_cb_receivers', 'untilize_out', 'allowed_worker_cores', 'stream_in1')


def decode_sites():
    """{(K, N): site} of the 64-row projections this lever may re-partition, from the geometry of the width the process serves at."""
    geometry = tp_shapes.active()
    return {(geometry.attn_out, tp_shapes.HIDDEN): 'out',
            (tp_shapes.HIDDEN, geometry.gdn_qkvzab_padded): 'gdn_in',
            (tp_shapes.HIDDEN, 2 * geometry.attn_out + 2 * geometry.attn_kv_heads * tp_shapes.ATTENTION_HEAD_DIM): 'attn_in'}


def site_columns(cfg, site):
    return {'out': cfg.out, 'gdn_in': cfg.gdn_in, 'attn_in': cfg.attn_in}[site]


def _lever_config(builder, settings, served, k, n, mesh):
    """-> (lever config, site) for the served 64-row projection config `served` of a (k, n) weight, or None (served stays). Decided once per served config object and kept:
    the warm pass and the captured launch of one layer get the SAME lever object, so a capture never meets a program the warm pass did not run. The pinned builder makes the
    lever config (the same field set as the served one); the fields the T1 #11 rule keeps are checked equal and anything else refuses."""
    key = (id(served), k, n)
    with _LOCK:
        known = _CONFIGS.get(key)
    if known is not None and known[0] is served:
        return known[1]
    result = None
    site = decode_sites().get((k, n))
    columns = None if site is None else site_columns(settings.cfg, site)
    if columns is not None:
        result = _build_lever(builder, settings, served, k, n, mesh, site, columns)
    with _LOCK:
        _CONFIGS[key] = (served, result)
    return result


def _build_lever(builder, settings, served, k, n, mesh, site, columns):
    name = 'verify.' + site
    why = too_small(grid_of(mesh))
    if why is not None:
        fell_back(name, why)
        return None
    served_grid = xy(served.compute_with_storage_grid_size)
    width = settings.cfg.width or served_grid[0]
    n_tiles = -(-n // TILE)
    cores = -(-n_tiles // columns)
    try:
        wide_grid((width, MIN_ROWS), cores)
        lever = builder(DECODE_ROWS, k, n, cores, fused_activation=getattr(served, 'fused_activation', None), grid_w=width)
    except (ValueError, TypeError) as failure:
        fell_back(name, 'the lever config cannot be built: %s' % failure)
        return None
    for field in KEPT_FIELDS + ('fused_activation',):
        if getattr(lever, field, None) != getattr(served, field, None):
            fell_back(name, 'the lever config would change %s (%r -> %r): only the N partition may move' % (field, getattr(served, field, None), getattr(lever, field, None)))
            return None
    if getattr(lever, 'per_core_N', None) != columns:
        fell_back(name, 'the builder gave per_core_N %r for the %d asked' % (getattr(lever, 'per_core_N', None), columns))
        return None
    if xy(lever.compute_with_storage_grid_size) == served_grid and lever.per_core_N == served.per_core_N:
        unchanged(name, 'the served config is already %d output columns a core on that grid' % columns)
        return None
    STATS['verify'] += 1
    note(ENGAGED, 'site=%s rows=%d k=%d n=%d per_core_N=%d served_per_core_N=%d cores=%d served_cores=%d grid=%s served_grid=%s' % (
        name, DECODE_ROWS, k, n, lever.per_core_N, served.per_core_N, -(-n_tiles // lever.per_core_N), -(-n_tiles // served.per_core_N),
        'x'.join(map(str, xy(lever.compute_with_storage_grid_size))), 'x'.join(map(str, served_grid))))
    return lever, site


def wrap_matmul_1d_decode(original, settings):
    """tp_common.matmul_1d_decode with the lever's partition for the 64-row projections: called as it is, on the lever's config instead of the served one the model passes
    (the configs were built at ModelArgs, before the lever could be installed, so the swap is here, at the call). Under the audit flag a call that took a lever config is
    run again on the served config (eager only) and the bytes compared."""
    namespace = original.__globals__

    def matmul_1d_decode(x, weight, decode_1d_progcfg, *args, **kwargs):
        found = None
        if 'verify' in settings.subs:
            try:
                rows, shape = int(x.shape[-2]), tuple(int(size) for size in weight.shape[-2:])
            except Exception:  # noqa: BLE001 - not a tensor we know: the served call
                rows, shape = 0, ()
            if rows == DECODE_ROWS and len(shape) == 2:
                try:
                    mesh = x.device()
                except Exception:  # noqa: BLE001
                    mesh = None
                if mesh is not None:
                    found = _lever_config(namespace['create_matmul_1d_decode_progcfg'], settings, decode_1d_progcfg, shape[0], shape[1], mesh)
        if found is None:
            return original(x, weight, decode_1d_progcfg, *args, **kwargs)
        lever_config, site = found
        lever = original(x, weight, lever_config, *args, **kwargs)
        if not settings.audit:
            return lever
        operations = namespace['ttnn']
        if not audit_begin(operations, x.device(), 'verify.' + site, DECODE_ROWS):
            return lever
        served = original(x, weight, decode_1d_progcfg, *args, **kwargs)
        try:
            operations.synchronize_device(x.device())
            bad = differing_chips(operations, lever, served)
        finally:
            try:
                operations.deallocate(served)
            except Exception:  # noqa: BLE001
                pass
        audit_result('verify.' + site, DECODE_ROWS, bad, kind='matmul1d')
        return lever
    matmul_1d_decode.grid13_of = original
    return matmul_1d_decode


# ---------------------------------------------------------------------------------------------
# drafter: the fused commit's feature projection (the hook is fusion-wp/M1-dflash-device.patch).
# ---------------------------------------------------------------------------------------------

def commit_program(operations, mesh, served, value=None, weight=None, kernel=None, environ=None):
    """The fused commit's feature projection program (a 1D multicast matmul on a fixed (8, 10) grid, per_core_N 2) on the device-wide grid: the same cores, the same per-core
    columns, in0_block_w, subblocks and per_core_M; only compute_with_storage_grid_size moves. `served` itself when the drafter sub-switch is off or the plan does not fit
    (one logged line says why); under the audit flag, also runs the matmul on both programs once per call (eager only, the first AUDIT_CALLS)."""
    settings = parse(environ)
    if settings is None or 'drafter' not in settings.subs:
        return served
    site = 'drafter.commit'
    why = too_small(grid_of(mesh))
    if why is not None:
        fell_back(site, why)
        return served
    grid = grid_of(mesh)
    n_tiles = None
    if weight is not None:
        n_tiles = int(weight.shape[-1]) // TILE
    served_grid = xy(served.compute_with_storage_grid_size)
    cores = served_grid[0] * served_grid[1] if n_tiles is None else -(-n_tiles // served.per_core_N)
    try:
        columns, rows = wide_grid(grid, cores)
    except ValueError as failure:
        fell_back(site, str(failure))
        return served
    if (columns, rows) == served_grid:
        unchanged(site, 'the served grid is already %dx%d' % (columns, rows))
        return served
    fields = dict((name, getattr(served, name)) for name in PROGRAM_FIELDS if hasattr(served, name))
    fields['compute_with_storage_grid_size'] = (columns, rows)
    try:
        wide = type(served)(**fields)
    except (TypeError, ValueError) as failure:
        fell_back(site, 'the program config cannot be rebuilt: %s' % failure)
        return served
    if settings.audit and value is not None and weight is not None and audit_begin(operations, mesh, site, int(value.shape[-2])):
        held = []
        try:
            mine = operations.matmul(value, weight, dtype=operations.float32, program_config=wide, compute_kernel_config=kernel,
                                     memory_config=operations.DRAM_MEMORY_CONFIG)
            held.append(mine)
            reference = operations.matmul(value, weight, dtype=operations.float32, program_config=served, compute_kernel_config=kernel,
                                          memory_config=operations.DRAM_MEMORY_CONFIG)
            held.append(reference)
            operations.synchronize_device(mesh)
            bad = differing_chips(operations, mine, reference)
        finally:
            for tensor in reversed(held):
                try:
                    operations.deallocate(tensor)
                except Exception:  # noqa: BLE001
                    pass
        audit_result(site, int(value.shape[-2]), bad, kind='commit')
    STATS['drafter'] += 1
    note(ENGAGED, 'site=%s cores=%d grid=%dx%d served_grid=%dx%d per_core_N=%d in0_block_w=%d device=%dx%d' % (
        site, cores, columns, rows, served_grid[0], served_grid[1], served.per_core_N,
        served.in0_block_w, grid[0], grid[1]))
    return wide


# ---------------------------------------------------------------------------------------------
# install: bound once at startup (tp_addresses.install, QWEN_FAST_TP=4 only). That is after the model is loaded and before the fast path attaches; every wrapper acts at the call.
# ---------------------------------------------------------------------------------------------

TP_COMMON = 'models.demos.blackhole.qwen36.tt.tp_common'
PREFILL_FUNCTIONS = ('all_gather_matmul_prefill', 'all_gather_swiglu_prefill')


def _rebind(module, name, wrapper, changed):
    """Put `wrapper` in place of module.<name> and in every loaded module whose global of that name IS the original (`from m import name` binds a copy)."""
    original = getattr(module, name, None)
    if original is None or getattr(original, 'grid13_of', None) is not None:
        return
    for candidate in list(sys.modules.values()):
        namespace = getattr(candidate, '__dict__', None)
        if isinstance(namespace, dict) and namespace.get(name) is original:
            namespace[name] = wrapper
            changed.append((namespace, name, original))
    if getattr(module, '__dict__', {}).get(name) is original:
        module.__dict__[name] = wrapper
        changed.append((module.__dict__, name, original))


def install(environ=None, module=None):
    """Wrap the pinned tp_common functions the lever's sub-switches need. -> [(namespace, name, original)] for tp_addresses to put back; [] when QWEN_FAST_GRID13 is
    unset, when the model tree is not importable here (the CPU suite) or when the functions are already wrapped. A malformed flag raises (a misconfigured arm must fail at
    startup, not at its first prefill)."""
    settings = parse(environ)
    if settings is None:
        return []
    if module is None:
        import importlib

        try:
            module = importlib.import_module(TP_COMMON)
        except ImportError as error:
            if getattr(error, 'name', None) in (TP_COMMON.split('.')[0], TP_COMMON, 'models.demos', 'models.demos.blackhole', 'models.demos.blackhole.qwen36',
                                                'models.demos.blackhole.qwen36.tt'):
                return []
            raise
    changed = []
    if 'prefill' in settings.subs:
        for name in PREFILL_FUNCTIONS:
            original = getattr(module, name, None)
            if original is not None and getattr(original, 'grid13_of', None) is None:
                _rebind(module, name, wrap_agmm(original, settings), changed)
    if 'verify' in settings.subs:
        original = getattr(module, 'matmul_1d_decode', None)
        if original is not None and getattr(original, 'grid13_of', None) is None:
            _rebind(module, 'matmul_1d_decode', wrap_matmul_1d_decode(original, settings), changed)
    return changed
