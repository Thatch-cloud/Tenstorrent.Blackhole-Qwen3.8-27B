"""The long-context SDPA decode configuration at four cards (QWEN_FAST_TP4_SDPA, default off).

WHY. At a 262,144-token window the packed verify's attention is the largest term: every attention layer makes one K64j decode
launch per user (G8B2, flags 0x23: two 8-token groups of six query heads on one KV head, 16 cores per entry, 32 of 110 cores
active) and reads the user's KV at about 150 GB/s against roughly 405 GB/s of DRAM. The one-card sweep
(optimisation/ttnn-op/sdpa_tp4_long) times every candidate configuration against the served one at 32k / 64k / 131k / 262k and
byte-compares each with the served call; this module is the serving side of its answer: ONE flag whose value is the NAME of a
configuration the sweep measured.

WHAT. Unset, empty, '0' or 'off': nothing here runs and the pinned reader is untouched (byte for byte). A name from CONFIGS that
is servable rebuilds each segment reader's SDPAProgramConfig after the pinned reader has built and qualified it (the same
sentinel q_chunk_size, the same 256-key chunks, exp_approx_mode off), changing only the worker grid the program may place on.
A grid moves where the 16 leaders and twins sit relative to the DRAM banks; it does not touch the accumulation order, so
the result is byte-identical by construction, and the sweep proves it per row before a name is offered. grid11x4 is the CONTROL: it
is expected to put the 32 active cores where the served grid does (rows 0-3 of an 11-wide grid), so it differs from served only in
the idle-core and dispatch set; a win by it is noise or dispatch, never placement. 'served' is the
explicit no-op (the plumbing control).

Names the sweep measures that are NOT servable here (they need a reader, mask kernel or factory change that is a later lane's:
the one-launch multi-user entry, the row split, the KV read-ahead flag) are recognised and refused with their reason: a typo and a
configuration that cannot be served are different mistakes, and neither silently runs the served call.

extent_attention_replay_tp.py is held unedited by the four-card evidence (its sha256 is pinned), so the hook lives in
extent_attention_fold_tp.PackedExtentReplayReader (an unpinned subclass), which tp_addresses binds whenever this flag is set,
whether or not the V3a fold is on.

Stdlib only, py 3.7. Strict (an unknown value raises) and refused at the pair: this exists for four cards.
"""

import os

import tp_shapes

FLAG = 'QWEN_FAST_TP4_SDPA'
OFF_VALUES = ('', '0', 'off')

ENGAGED = '[PINDIAG] tp4 sdpa engaged'
RUNTIME_FILES = ('sdpa_long_tp.py',)

# The qualified served flags (tail 0x1 | share 0x2 | extent 0x20) and the bundle width the twin-band check uses (G8B2: two entries).
SERVED_FLAGS = 0x23
ENTRIES = 2
KV_CORES_PER_ENTRY = 16
MESH_GRID_MAX = (11, 10)

# name -> (grid (x, y) or None for the mesh's own grid, servable, why not)
CONFIGS = {
    'served': dict(grid=None, servable=True, why=''),
    'grid8x4': dict(grid=(8, 4), servable=True, why=''),
    'grid8x10': dict(grid=(8, 10), servable=True, why=''),
    'grid11x4': dict(grid=(11, 4), servable=True, why=''),   # the control: expected to place the active cores as served does
    'grid4x8': dict(grid=(4, 8), servable=True, why=''),
    'multi': dict(grid=None, servable=False,
                  why='one G16 launch for every live user needs a new reader, a per-entry mask kernel and a pool-lent table '
                      '(design K2)'),
    'rowsplit': dict(grid=None, servable=False,
                     why='G4B4 share per user is slower than served until the reader sizes its KV barrier to the real reader '
                         'count (design K1) and the mask and reader bounds lift (K3)'),
    'ra': dict(grid=None, servable=False,
               why='flag 0x8 (KV read-ahead) makes the flag set 0x2B, which the pinned extent reader refuses (it admits 0x23 only)'),
}


def names():
    return tuple(CONFIGS)


def servable_names():
    return tuple(name for name, entry in CONFIGS.items() if entry['servable'])


def _chip_count(environ):
    return tp_shapes.chip_count(os.environ if environ is None else environ)


def selected(environ=None):
    """The configuration name QWEN_FAST_TP4_SDPA picks, or None when the flag is off. Strict: an unknown or non-servable value
    raises, and any value but off raises at the pair."""
    source = os.environ if environ is None else environ
    value = source.get(FLAG)
    if value is None or value.strip().lower() in OFF_VALUES:
        return None
    name = value.strip()
    if _chip_count(source) == tp_shapes.PAIR:
        raise ValueError('%s is a TP4 lever: it needs QWEN_FAST_TP=4, this process serves the pair' % FLAG)
    if name not in CONFIGS:
        raise ValueError('%s=%r is not a configuration; the names are %s (off: unset, 0 or off)' % (FLAG, value, ', '.join(names())))
    if not CONFIGS[name]['servable']:
        raise ValueError('%s=%s is measured by the sweep but cannot be served yet: %s. Servable: %s'
                         % (FLAG, name, CONFIGS[name]['why'], ', '.join(servable_names())))
    return name


def enabled(environ=None):
    return selected(environ) is not None


def grid_problem(grid, mesh_grid, entries=ENTRIES):
    """Why `grid` (x, y) cannot hold the served program on a mesh whose worker grid is `mesh_grid`, or None. The program needs
    KV_CORES_PER_ENTRY cores per entry and, under KV share, a twin core beside every leader in bands of ceil(16 / x) rows per
    entry, so ceil(16 / x) * entries rows must fit."""
    x, y = grid
    if x < 1 or y < 1:
        return 'grid %dx%d is empty' % (x, y)
    if x > mesh_grid[0] or y > mesh_grid[1]:
        return 'grid %dx%d does not fit the %dx%d worker grid' % (x, y, mesh_grid[0], mesh_grid[1])
    if x * y < KV_CORES_PER_ENTRY * entries:
        return 'grid %dx%d has %d cores, the program needs %d' % (x, y, x * y, KV_CORES_PER_ENTRY * entries)
    bands = -(-KV_CORES_PER_ENTRY // x) * entries
    if bands > y:
        return 'grid %dx%d cannot hold the twin bands (%d rows needed)' % (x, y, bands)
    return None


def plan(name, mesh_grid, entries=ENTRIES):
    """-> the (x, y) grid configuration `name` runs on, or raises ValueError when it does not fit `mesh_grid`."""
    entry = CONFIGS[name]
    if entry['grid'] is None:
        return tuple(mesh_grid)
    problem = grid_problem(entry['grid'], mesh_grid, entries)
    if problem:
        raise ValueError('%s=%s: %s' % (FLAG, name, problem))
    return entry['grid']


def marker(name, grid, entries):
    return '%s config=%s grid=%dx%d entries=%d' % (ENGAGED, name, grid[0], grid[1], entries)


def _log(message):
    try:
        try:
            from loguru import logger
        except ImportError:
            print(message, flush=True)
        else:
            logger.info('{}', message)
    except Exception:  # noqa: BLE001 - a log line never fails a serve
        pass


def apply(reader, environ=None):
    """Called by the reader twin after the pinned constructor. Flag off: returns None and touches nothing. On: rebuilds every
    segment entry's program config on the configuration's grid and logs the ENGAGED line. -> the (x, y) grid, or None."""
    name = selected(environ)
    if name is None:
        return None
    from pooled_attention_replay import QWEN_DECODE_MAGIC

    mesh = reader.mesh.compute_with_storage_grid_size()
    grid = plan(name, (mesh.x, mesh.y))
    entries = 0
    for segment in reader.readers:
        applied = tuple(getattr(segment, 'sdpa_modes_applied', None) or ())
        if len(applied) != len(segment.metadata) or any(value != SERVED_FLAGS for value in applied):
            raise ValueError('%s=%s is qualified at flags 0x%x only, the reader runs %s'
                             % (FLAG, name, SERVED_FLAGS, ['0x%x' % value for value in applied]))
        if CONFIGS[name]['grid'] is not None:
            replaced = []
            for (bundle, pages, mask, config), flags in zip(segment.metadata, applied):
                config = reader.operations.SDPAProgramConfig(compute_with_storage_grid_size=grid, exp_approx_mode=False,
                                                             q_chunk_size=QWEN_DECODE_MAGIC | flags, k_chunk_size=256)
                replaced.append((bundle, pages, mask, config))
            segment.metadata[:] = replaced
        entries += len(segment.metadata)
    _log(marker(name, grid, entries))
    return grid
