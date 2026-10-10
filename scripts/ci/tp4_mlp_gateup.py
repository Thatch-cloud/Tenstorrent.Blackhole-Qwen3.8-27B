"""The TP4 MLP gate/up levers (op-fusion programme WP4: F-D2 and the streaming config now, F-D1 the fused op behind them).

The packed 64-row verify block runs each of the 64 layers' MLP as five launches: the gate matmul (SiLU fused, bfloat4_b, 68 cores),
the up matmul (bfloat4_b, 34 to 46 cores), their multiply (written to DRAM at 64 rows), the down matmul (bfloat8_b, 32 cores) and the
reduce-scatter. The device profile (v676, v678) has the three weight matmuls at 236, 284 and 357 GB/s of a 512 GB/s part, and
the multiply one more launch plus a DRAM round trip. The graft's mlp.py, tp_common.py and fused_1d.py are pinned and never edited;
this module is their TWIN for the 64-row decode arm only, bound per layer around the block's forward (model_batch.two_tile_bindings,
the way the two-tile binders are) and absent from the process unless a flag asks.

  QWEN_FAST_MLP_CFG=<name>    F-D2 and the streaming config: the served five launches with the multiply written to L1 (F-D2), and
                              the gate, up and down program configs re-partitioned as the name says. The name is `l1` (only the
                              multiply moves) or tokens in the order g u d p w, each at most once: g<n> / u<n> / d<n> the
                              per_core_N of the gate / up / down 1D-mcast matmul at 64 rows (the core count follows: ceil(N tiles /
                              n) cores laid out `w` wide), w<n> the grid width (default: the device's compute grid width), p<n> the
                              fused op's tile pairs per worker (only with QWEN_FAST_MLP_GATEUP=1). `g2u3d5` is the 13x10 served
                              partition spelled out; the card-M sweep (optimisation/ttnn-op/mlp_gateup) prints the best name.
  QWEN_FAST_MLP_GATEUP=1      F-D1: gate and up as ONE launch (tp4_mlp_fused: a port of the TP2 fused_1d.FusedProjection) with the SwiGLU
                              epilogue, the product left in L1 for the down matmul. Reads the served w1 and w3 tensors (no packed copy).
  QWEN_FAST_MLP_GATEUP_AUDIT / QWEN_FAST_MLP_CFG_AUDIT=1   (gate arms only) beside the lever, every QWEN_FAST_MLP_AUDIT_STRIDE-th
                              layer (default 4) also runs the served composition and compares the SwiGLU product and the down
                              projection's output on every chip as int16 bit patterns after the replay; the SERVED partial is the one
                              reduced and returned, and the layer still issues ONE all-reduce (the block counts them, and the ring's
                              reduce-scatter parity is a hang factor).

EXACTNESS. Every launch keeps the served arithmetic. The three matmuls keep in0_block_w, per_core_M, fuse_batch, mcast_in0, the fused
activation and the compute kernel config: a config that would move any of them is refused (logged, the served ops run). What moves is
which core computes which output tile and the output subblock shape, never how an output tile is reduced over K (the T1 #11 rule,
tests hold it with the emulation of test_verify_trace_t1_graft); the multiply is the same ttnn.multiply with an L1 output. The fused op
reproduces the matmul_1d K loop in the native compute kernel and rounds gate and up to bf16 before a bf16 product, as the served pair
and their multiply do. Byte equality against the served composition is the card-M gate and the in-trace audit; nothing is exact here
until a card says so.

Markers (the `engaged` line carries the route; a `fell back` line fails any smoke):
  [PINDIAG] tp4 mlp gateup engaged route=cfg|fused name=<name> rows=64 layers=<n> ...
  [PINDIAG] tp4 mlp gateup fell back reason=<why>
  [PINDIAG] tp4 mlp gateup audit <n> exact=True route=<route> ...        [PINDIAG] tp4 mlp gateup audit mismatch ...

All flags are strict and read when the block is bound (never at import); any of them set while the process serves the pair raises.
Stdlib only at import (py 3.7); ttnn and torch are reached through the `operations` handle.
"""

from collections import namedtuple
import math
import os
import re

import tp_shapes

TILE = 32
ROWS = 64
FP32_SUBBLOCK_CAP = 4
BLOCK_CAP = 8
MIN_ACTIVE_CORES = 8
PAIRS_PER_WORKER = (2, 3, 4, 5, 7)
DEFAULT_PAIRS = 3
DEFAULT_STRIDE = 4

GATEUP = 'QWEN_FAST_MLP_GATEUP'
GATEUP_AUDIT = 'QWEN_FAST_MLP_GATEUP_AUDIT'
CFG = 'QWEN_FAST_MLP_CFG'
CFG_AUDIT = 'QWEN_FAST_MLP_CFG_AUDIT'
AUDIT_STRIDE = 'QWEN_FAST_MLP_AUDIT_STRIDE'
ALL_FLAGS = (GATEUP, GATEUP_AUDIT, CFG, CFG_AUDIT, AUDIT_STRIDE)

ENGAGED = '[PINDIAG] tp4 mlp gateup engaged'
FALLBACK = '[PINDIAG] tp4 mlp gateup fell back'
AUDIT_MARKER = '[PINDIAG] tp4 mlp gateup audit'
AUDIT_MISMATCH = '[PINDIAG] tp4 mlp gateup audit mismatch'

# What each lever needs at run time, by basename. The overlay list and the CPU allowlist must name every entry (test_tp4_mlp_gateup).
RUNTIME_FILES = ('tp4_mlp_gateup.py', 'tp4_mlp_fused.py', 'tp4_mlp_fused_input.cpp', 'tp4_mlp_fused_weights.cpp')

# bfloat4_b: 512 bytes of mantissa plus 64 of exponents per tile; bfloat8_b: 1,024 plus 64; bfloat16: 2,048.
TILE_BYTES = {'bfp4': 576, 'bfp8': 1088, 'bf16': 2048}
PEAK_GBPS = 512.0

Shape = namedtuple('Shape', ('key', 'k', 'n', 'dtype', 'silu', 'requested_cores'))
Config = namedtuple('Config', ('grid', 'in0_block_w', 'per_core_M', 'per_core_N', 'out_subblock_h', 'out_subblock_w'))
Cfg = namedtuple('Cfg', ('name', 'g', 'u', 'd', 'p', 'w'))

# The fields the T1 #11 rule keeps: a named config that changes one of them changes how an output tile is reduced over K.
KEPT_FIELDS = ('in0_block_w', 'per_core_M', 'fuse_batch', 'mcast_in0')

_TOKEN = re.compile(r'([gudpw])([1-9][0-9]*)')
_ORDER = 'gudpw'


# ---------------------------------------------------------------------------------------------------------------------------
# The flags.
# ---------------------------------------------------------------------------------------------------------------------------

def _source(environ):
    return os.environ if environ is None else environ


def _read(name, environ):
    value = _source(environ).get(name)
    if value is None or value == '0':
        return False
    if value == '1':
        return True
    raise ValueError('%s must be 0 or 1, got %r' % (name, value))


def parse_cfg(name):
    """The Cfg a QWEN_FAST_MLP_CFG name spells, or ValueError. `l1` is the multiply alone; otherwise tokens g u d p w in that order,
    each at most once and at least one, every number a positive integer without a leading zero. One spelling per configuration."""
    if not isinstance(name, str) or not name:
        raise ValueError('%s must be `l1` or tokens in the order g u d p w (g2u3d5), got %r' % (CFG, name))
    if name == 'l1':
        return Cfg('l1', None, None, None, None, None)
    values, position, previous = {}, 0, -1
    while position < len(name):
        found = _TOKEN.match(name, position)
        if found is None:
            raise ValueError('%s=%r: expected `l1` or tokens in the order g u d p w (g2u3d5); stuck at %r' % (CFG, name, name[position:]))
        order = _ORDER.index(found.group(1))
        if order <= previous:
            raise ValueError('%s=%r: tokens go in the order g u d p w, each at most once (%r is out of place)' % (CFG, name, found.group(0)))
        previous = order
        values[found.group(1)] = int(found.group(2))
        position = found.end()
    if values.get('p') is not None and values['p'] not in PAIRS_PER_WORKER:
        raise ValueError('%s=%r: p is the fused op\'s tile pairs per worker, one of %s' % (CFG, name, ', '.join(map(str, PAIRS_PER_WORKER))))
    return Cfg(name, values.get('g'), values.get('u'), values.get('d'), values.get('p'), values.get('w'))


def canonical_name(g=None, u=None, d=None, p=None, w=None):
    """The one spelling of a configuration: `l1` when nothing is named."""
    tokens = [letter + str(value) for letter, value in zip(_ORDER, (g, u, d, p, w)) if value is not None]
    return ''.join(tokens) or 'l1'


class Selection(namedtuple('Selection', ('route', 'name', 'cfg', 'audit', 'stride', 'pairs'))):
    """What the environment asks: route 'fused' (QWEN_FAST_MLP_GATEUP=1), 'cfg' (QWEN_FAST_MLP_CFG), or None."""


def resolve(environ=None):
    """The Selection of this environment (route None when no lever is on), after every cross-flag rule: an audit flag needs its lever,
    the pair serves none of this, `g` and `u` cannot be named when the fused op replaces those matmuls, `p` needs the fused op."""
    source = _source(environ)
    named = [name for name in ALL_FLAGS if source.get(name) not in (None, '0')]
    if named and tp_shapes.chip_count(source) == tp_shapes.PAIR:
        raise ValueError('%s are TP4 levers: they need QWEN_FAST_TP=4, this process serves the pair' % ', '.join(named))
    fused = _read(GATEUP, source)
    fused_audit = _read(GATEUP_AUDIT, source)
    raw = source.get(CFG)
    cfg = None if raw in (None, '0') else parse_cfg(raw)
    cfg_audit = _read(CFG_AUDIT, source)
    if fused_audit and not fused:
        raise ValueError('%s=1 needs %s=1 (an audit of a lever that is off)' % (GATEUP_AUDIT, GATEUP))
    if cfg_audit and cfg is None:
        raise ValueError('%s=1 needs %s (an audit of a lever that is off)' % (CFG_AUDIT, CFG))
    stride = DEFAULT_STRIDE
    if source.get(AUDIT_STRIDE) is not None:
        text = source[AUDIT_STRIDE]
        if not re.match(r'^[1-9][0-9]{0,2}$', text):
            raise ValueError('%s must be a positive integer (audit every n-th layer), got %r' % (AUDIT_STRIDE, text))
        if not fused and cfg is None:
            raise ValueError('%s is set and no lever is on (%s or %s)' % (AUDIT_STRIDE, GATEUP, CFG))
        stride = int(text)
    if cfg is not None and fused and (cfg.g is not None or cfg.u is not None):
        raise ValueError('%s=%s names g or u, but %s=1 replaces those two matmuls with one launch' % (CFG, cfg.name, GATEUP))
    if cfg is not None and not fused and cfg.p is not None:
        raise ValueError('%s=%s names p, the fused op\'s pairs per worker, without %s=1' % (CFG, cfg.name, GATEUP))
    route = 'fused' if fused else ('cfg' if cfg is not None else None)
    pairs = (cfg.p if cfg is not None and cfg.p is not None else DEFAULT_PAIRS) if fused else None
    if fused:
        # the fused route's name always spells its pairs per worker: one spelling per configuration
        name = canonical_name(d=None if cfg is None else cfg.d, p=pairs, w=None if cfg is None else cfg.w)
    else:
        name = cfg.name if cfg is not None else None
    return Selection(route, name, cfg, fused_audit or cfg_audit, stride, pairs)


def requested(environ=None):
    """Whether any lever is on (the cheap test model_batch makes before it imports anything else). Raises on a bad flag."""
    return resolve(environ).route is not None


# ---------------------------------------------------------------------------------------------------------------------------
# Shapes, the builder, named configs.
# ---------------------------------------------------------------------------------------------------------------------------

def shapes(tp=4):
    """The per-chip decode matmuls at `tp` chips, by key: the MLP gate (mlp_w1), up (mlp_w3), down (mlp_w2) the lever acts on, and the four
    in/out projections the same sweep covers (R3). `requested_cores` is the num_cores the graft's model_config passes the builder at
    64 rows with QWEN_FAST_VERIFY_T1=1 (gate 88, attn_qkv and the others as listed)."""
    found = tp_shapes.geometry(tp)
    hidden = tp_shapes.HIDDEN
    kv_dim = found.attn_kv_heads * tp_shapes.ATTENTION_HEAD_DIM
    table = (Shape('mlp_w1', hidden, found.mlp, 'bfp4', True, 88),
             Shape('mlp_w3', hidden, found.mlp, 'bfp4', False, 44),
             Shape('mlp_w2', found.mlp, hidden, 'bfp8', False, 33),
             Shape('gdn_in', hidden, found.gdn_qkvzab, 'bfp8', False, 44),
             Shape('attn_in', hidden, found.attn_heads * tp_shapes.ATTENTION_HEAD_DIM * 2 + 2 * kv_dim, 'bfp8', False, 44),
             Shape('attn_wo', found.attn_out, hidden, 'bfp8', False, 33),
             Shape('gdn_out', found.gdn_value, hidden, 'bfp8', False, 33))
    return dict((shape.key, shape) for shape in table)


def tiles(count):
    return int(math.ceil(count / float(TILE)))


def largest_divisor(count, cap=BLOCK_CAP):
    """tp_common._find_largest_divisor: the model's in0_block_w."""
    for divisor in range(cap, 0, -1):
        if count % divisor == 0:
            return divisor
    return 1


def subblock(per_core_n, m_tiles=ROWS // TILE, cap=FP32_SUBBLOCK_CAP):
    """(h, w) as tp_common's builder picks them at fp32 accumulation: the widest w up to the cap that divides per_core_N, then the tallest h
    that divides the row tiles within the cap."""
    sub_w = max(i for i in range(1, cap + 1) if per_core_n % i == 0)
    sub_h = max(i for i in range(1, cap + 1) if m_tiles % i == 0 and i * sub_w <= cap)
    return sub_h, sub_w


def builder_config(shape, grid_w, num_cores=None, m=ROWS):
    """tp_common.create_matmul_1d_decode_progcfg(m, K, N, num_cores, grid_w) as a Config (the served config of `shape`: num_cores defaults
    to the model's request). Transcribed from the image's builder; the tests hold it against the graft's own transcription."""
    num_cores = shape.requested_cores if num_cores is None else num_cores
    cols = min(grid_w, num_cores)
    rows = int(math.ceil(num_cores / float(cols)))
    m_tiles = tiles(m)
    per_core_n = int(math.ceil(tiles(shape.n) / float(cols * rows)))
    sub_h, sub_w = subblock(per_core_n, m_tiles)
    return Config((cols, rows), largest_divisor(tiles(shape.k)), m_tiles, per_core_n, sub_h, sub_w)


def active_cores(config, shape):
    """The cores that carry output columns: ceil(N tiles / per_core_N); the rest of the grid idles."""
    return int(math.ceil(tiles(shape.n) / float(config.per_core_N)))


def minimal_grid(active, width):
    """The (cols, rows) rectangle `width` wide that holds `active` cores in row-major order."""
    cols = min(width, active)
    return cols, int(math.ceil(active / float(cols)))


def named_config(shape, per_core_n, width, m=ROWS):
    """The Config of `shape` at an explicit per_core_N on the minimal rectangle `width` wide, with the served in0_block_w and the builder's
    subblock rule. ValueError when per_core_N is outside 1..N tiles."""
    n_tiles = tiles(shape.n)
    if not 1 <= per_core_n <= n_tiles:
        raise ValueError('per_core_N %d is outside 1..%d for %s' % (per_core_n, n_tiles, shape.key))
    sub_h, sub_w = subblock(per_core_n, tiles(m))
    grid = minimal_grid(int(math.ceil(n_tiles / float(per_core_n))), width)
    return Config(grid, largest_divisor(tiles(shape.k)), tiles(m), per_core_n, sub_h, sub_w)


def per_core_n_choices(shape, grid_x, grid_y, min_cores=None):
    """Every per_core_N whose active core count fits the worker grid (`min_cores`, default MIN_ACTIVE_CORES, to grid_x x grid_y), the smallest
    per_core_N for each distinct core count (a larger one at the same count only pads more)."""
    min_cores = MIN_ACTIVE_CORES if min_cores is None else min_cores
    n_tiles = tiles(shape.n)
    seen, out = set(), []
    for per_core_n in range(1, n_tiles + 1):
        cores = int(math.ceil(n_tiles / float(per_core_n)))
        if cores in seen or not min_cores <= cores <= grid_x * grid_y:
            continue
        cols, rows = minimal_grid(cores, grid_x)
        if rows > grid_y:
            continue
        seen.add(cores)
        out.append(per_core_n)
    return out


def same_k_loop(served, other):
    """The KEPT_FIELDS of two program configs (objects or Configs) that differ, as [(field, served, other)]; empty = the same K loop."""
    moved = []
    for field in KEPT_FIELDS:
        a, b = getattr(served, field, None), getattr(other, field, None)
        if a != b:
            moved.append((field, a, b))
    return moved


def grid_of(config):
    """(cols, rows) of a Config or of a ttnn program config (whose grid is a CoreCoord with .x and .y, or a tuple)."""
    if hasattr(config, 'grid'):
        return tuple(config.grid)
    grid = config.compute_with_storage_grid_size
    return (int(grid.x), int(grid.y)) if hasattr(grid, 'x') else tuple(grid)


def config_label(config):
    """`13x6/pcn2/blk8/sub2x2`: a Config's or a program-config object's grid, per_core_N, K block and subblock."""
    grid = grid_of(config)
    return '%dx%d/pcn%d/blk%d/sub%dx%d' % (grid[0], grid[1], config.per_core_N, config.in0_block_w, config.out_subblock_h, config.out_subblock_w)


def program_config(operations, config, silu):
    """The ttnn 1D-mcast program config of a Config, the way tp_common builds it (fp32 destination, fuse_batch, mcast_in0)."""
    return operations.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=tuple(config.grid), in0_block_w=config.in0_block_w, out_subblock_h=config.out_subblock_h,
        out_subblock_w=config.out_subblock_w, per_core_M=config.per_core_M, per_core_N=config.per_core_N, fuse_batch=True,
        fused_activation=operations.UnaryOpType.SILU if silu else None, mcast_in0=True)


# ---------------------------------------------------------------------------------------------------------------------------
# Bytes and bandwidth.
# ---------------------------------------------------------------------------------------------------------------------------

def weight_tiles(shape):
    return tiles(shape.k) * tiles(shape.n)


def weight_bytes(shape):
    """The bytes one launch streams from DRAM: the padded weight once (bfloat4_b 576 B a tile, bfloat8_b 1,088 B)."""
    return weight_tiles(shape) * TILE_BYTES[shape.dtype]


def gbps(shape, microseconds):
    """Gigabytes per second a launch of `microseconds` achieved on `shape`'s weight."""
    return weight_bytes(shape) / float(microseconds) / 1e3


# ---------------------------------------------------------------------------------------------------------------------------
# Logging and counts.
# ---------------------------------------------------------------------------------------------------------------------------

def log_line(message):
    """One line into the server log: loguru where it exists, stdout otherwise. Never raises."""
    try:
        try:
            from loguru import logger
        except ImportError:
            print(message, flush=True)
        else:
            logger.info('{}', message)
    except BaseException:  # noqa: BLE001
        pass


_LOGGED = set()


def log_once(message, key):
    if key in _LOGGED:
        return False
    _LOGGED.add(key)
    log_line(message)
    return True


def forget_logged():
    """Tests only: let every once-per-process line be logged again, and drop the audit's state."""
    _LOGGED.clear()
    _AUDIT.update(rounds=0)
    del _HELD[:]
    _STATE.clear()


# ---------------------------------------------------------------------------------------------------------------------------
# The plan: what a bound block runs.
# ---------------------------------------------------------------------------------------------------------------------------

class Plan(object):
    """The program configs and route of one bound block, built once and shared by its 64 layers."""

    def __init__(self, selection, gate, up, down, grid, shape_table, fused_pairs=None):
        self.selection, self.gate, self.up, self.down = selection, gate, up, down
        self.grid, self.shapes = grid, shape_table
        self.route, self.name = selection.route, selection.name
        self.fused_pairs = fused_pairs

    def describe(self):
        parts = []
        for label, config in (('gate', self.gate), ('up', self.up), ('down', self.down)):
            if config is None:
                parts.append('%s=fused' % label)
            else:
                parts.append('%s=%s' % (label, config_label(config)))
        return ' '.join(parts)


def fallback_line(reason):
    return '%s reason=%s' % (FALLBACK, reason)


def served_configs(args):
    """The graft's own 64-row configs (the objects the served `_forward_tp` passes), or None when the model lacks them."""
    names = ('mlp_w1_decode_1d_progcfg_64', 'mlp_w3_decode_1d_progcfg_64', 'mlp_w2_decode_1d_progcfg_64')
    found = [getattr(args, name, None) for name in names]
    return None if any(value is None for value in found) else tuple(found)


def build_plan(selection, args, operations, grid_x, grid_y):
    """The Plan for `selection` over a model whose args carry the graft's 64-row configs, or ValueError naming why not.

    A tensor-partition token (g, u, d) builds its config at an explicit per_core_N on the minimal rectangle `w` (default grid_x) wide, with
    the served in0_block_w, and refuses it unless the K loop (KEPT_FIELDS) equals the served object's."""
    served = served_configs(args)
    if served is None:
        raise ValueError('the model carries no 64-row MLP configs (it needs the Lever N native graft)')
    table = shapes(4)
    cfg = selection.cfg
    width = cfg.w if cfg is not None and cfg.w is not None else grid_x
    if not 1 <= width <= grid_x:
        raise ValueError('w%d is outside the device grid width %d' % (width, grid_x))
    tokens = dict(mlp_w1=None if cfg is None else cfg.g, mlp_w3=None if cfg is None else cfg.u, mlp_w2=None if cfg is None else cfg.d)
    chosen = []
    for position, key in enumerate(('mlp_w1', 'mlp_w3', 'mlp_w2')):
        if selection.route == 'fused' and key != 'mlp_w2':
            chosen.append(None)                       # one launch replaces the gate and the up
            continue
        token = tokens[key]
        if token is None and (cfg is None or cfg.w is None):
            chosen.append(served[position])            # nothing named for this matmul: the graft's own config object
            continue
        shape = table[key]
        if token is None:
            token = served[position].per_core_N       # only the width moved: the served partition on that rectangle
        config = named_config(shape, token, width)
        if config.grid[0] * config.grid[1] > grid_x * grid_y or config.grid[1] > grid_y:
            raise ValueError('%s per_core_N %d needs a %dx%d grid; the device has %dx%d' % (key, token, config.grid[0], config.grid[1],
                                                                                         grid_x, grid_y))
        built = program_config(operations, config, shape.silu)
        moved = same_k_loop(served[position], built)
        if moved:
            raise ValueError('%s: the named config would change the K loop (%s)' % (
                key, ', '.join('%s %r -> %r' % item for item in moved)))
        chosen.append(built)
    gate, up, down = chosen
    return Plan(selection, gate, up, down, (grid_x, grid_y), table, fused_pairs=selection.pairs)


# ---------------------------------------------------------------------------------------------------------------------------
# The twin forward.
# ---------------------------------------------------------------------------------------------------------------------------

_STATE = {}
_HELD = []
_AUDIT = dict(rounds=0)


def refusal(mlp, x, rows):
    """Why this call cannot take the lever (a reason string), or None: the lever is the 64-row full-K decode arm of the 1D-decode MLP."""
    shape = tuple(x.shape)
    if len(shape) != 4:
        return 'activation rank %d' % len(shape)
    if shape[-2] != rows:
        return 'activation rows %d, bound for %d' % (shape[-2], rows)
    dim = getattr(getattr(mlp, 'args', None), 'dim', None)
    if dim is not None and shape[-1] != dim:
        return 'activation width %d is not the full K %d (the K-sharded prefill input)' % (shape[-1], dim)
    if not getattr(mlp, '_mlp_1d_decode', False) or getattr(mlp, '_dram_sharded', False):
        return 'the MLP is not on the 1D decode arm'
    return None


class Tp4MlpForward(object):
    """One layer's MLP forward at the block's 64 rows: the served `_forward_tp` arm with the lever's launches. `calls` counts every call
    (engaged or fallen back), the way the two-tile binders count, because the block checks one call per layer per forward."""

    def __init__(self, mlp, index, rows, operations, plan, fused=None, all_reduce=None):
        self.mlp, self.index, self.rows, self.operations, self.plan = mlp, index, rows, operations, plan
        self.fused = fused
        self.all_reduce = all_reduce
        self.calls = 0
        self.engaged = 0
        self.fell_back = 0

    # -- the call --------------------------------------------------------------------------------------------------------
    def __call__(self, x):
        self.calls += 1
        why = refusal(self.mlp, x, self.rows)
        if why is not None:
            self.fell_back += 1
            log_once(fallback_line(why), ('fallback', why))
            return self.mlp._forward_tp(x)
        selection = self.plan.selection
        audited = selection.audit and self.index % selection.stride == 0
        result = self.audited(x) if audited else self.lever(x)
        self.engaged += 1
        self.mark_engaged(audited)
        return result

    def mark_engaged(self, audited):
        """The engaged line, once per process per configuration. It is a log line: whatever goes wrong building it must not take the round with it."""
        plan = self.plan
        try:
            log_once('%s route=%s name=%s rows=%d layers=%d %s l1_multiply=1 audit=%d stride=%d' % (
                ENGAGED, plan.route, plan.name, self.rows, _STATE.get('layers', 0), plan.describe(), int(plan.selection.audit),
                plan.selection.stride), ('engaged', plan.route, plan.name))
        except Exception as error:  # noqa: BLE001
            log_once('%s route=%s name=%s rows=%d layers=%d l1_multiply=1 audit=%d stride=%d describe=%s' % (
                ENGAGED, plan.route, plan.name, self.rows, _STATE.get('layers', 0), int(plan.selection.audit), plan.selection.stride,
                type(error).__name__), ('engaged', plan.route, plan.name))

    # -- the arms --------------------------------------------------------------------------------------------------------
    def kernel_config(self, x):
        """The served arm's compute kernel config: decode (packer L1 accumulation) when the activation is 2-D or has one in dim 1."""
        mlp = self.mlp
        height = x.shape[1] if len(x.shape) >= 3 else 1
        return mlp.compute_kernel_config_decode if height <= 1 else mlp.compute_kernel_config

    def hidden(self, x_il, ckc):
        """The SwiGLU product: silu(x w1) * (x w3) left in L1. Frees x_il (as the served arm does) and the two projections."""
        operations, weights, plan = self.operations, self.mlp.weights, self.plan
        l1 = operations.L1_MEMORY_CONFIG
        if plan.route == 'fused':
            product = self.fused(x_il)
            operations.deallocate(x_il)
            return product
        gate = operations.linear(x_il, weights.w1, compute_kernel_config=ckc, program_config=plan.gate, memory_config=l1)
        up = operations.linear(x_il, weights.w3, compute_kernel_config=ckc, program_config=plan.up, memory_config=l1)
        operations.deallocate(x_il)
        product = operations.mul(gate, up, memory_config=l1)
        operations.deallocate(gate)
        operations.deallocate(up)
        return product

    def served_hidden(self, x_il, ckc):
        """The served SwiGLU product (the graft's configs, the multiply written to DRAM) from x_il, which it frees."""
        operations, weights, args = self.operations, self.mlp.weights, self.mlp.args
        l1 = operations.L1_MEMORY_CONFIG
        gate = operations.linear(x_il, weights.w1, compute_kernel_config=ckc, program_config=args.mlp_w1_decode_1d_progcfg_64,
                                 memory_config=l1)
        up = operations.linear(x_il, weights.w3, compute_kernel_config=ckc, program_config=args.mlp_w3_decode_1d_progcfg_64,
                               memory_config=l1)
        operations.deallocate(x_il)
        product = operations.mul(gate, up, memory_config=operations.DRAM_MEMORY_CONFIG)
        operations.deallocate(gate)
        operations.deallocate(up)
        return product

    def down(self, product, ckc, config):
        """The down projection (L1 output) with `config`: consumes the product."""
        operations, mlp = self.operations, self.mlp
        partial = operations.linear(product, mlp.weights.w2, compute_kernel_config=ckc, memory_config=operations.L1_MEMORY_CONFIG,
                                    program_config=config)
        operations.deallocate(product)
        return partial

    def reduce(self, partial):
        """The all-reduce of the down projection, as the served arm: the process's own tt_all_reduce, looked up now so a wrapper installed after the
        binding (tile_collective_tp) is the one reached. Consumes the partial. One per layer, audited or not: the block counts them."""
        operations, mlp = self.operations, self.mlp
        reduce = self.all_reduce
        if reduce is None:
            from models.tt_transformers.tt.ccl import tt_all_reduce as reduce
        return reduce(partial, mlp.device, mlp.tt_ccl, cluster_axis=0, dim=3, topology=mlp.args.ccl_topology(),
                      memory_config=operations.DRAM_MEMORY_CONFIG)

    def output(self, product, ckc):
        return self.reduce(self.down(product, ckc, self.plan.down))

    def lever(self, x):
        operations = self.operations
        x_il = operations.to_memory_config(x, operations.L1_MEMORY_CONFIG)
        ckc = self.kernel_config(x)
        return self.output(self.hidden(x_il, ckc), ckc)

    def audited(self, x):
        """The lever and the served composition side by side on a clone of x: the SwiGLU product and the down projection's output of each are cloned into DRAM
        and held for audit_round (after the replay), and the SERVED partial is the one reduced and returned. One all-reduce per layer, as unaudited: the
        block checks their count (tile_collective_tp.block_scope) and the ring's reduce-scatter parity."""
        operations = self.operations
        l1, dram = operations.L1_MEMORY_CONFIG, operations.DRAM_MEMORY_CONFIG
        x_il = operations.to_memory_config(x, l1)
        ckc = self.kernel_config(x)
        for_served = operations.clone(x_il, memory_config=l1)
        held = []
        try:
            mine_hidden = self.hidden(x_il, ckc)
            held.append(operations.clone(mine_hidden, memory_config=dram))
            mine_partial = self.down(mine_hidden, ckc, self.plan.down)
            held.append(operations.clone(mine_partial, memory_config=dram))
            operations.deallocate(mine_partial)
            served_hidden = self.served_hidden(for_served, ckc)
            held.append(operations.clone(served_hidden, memory_config=dram))
            served_partial = self.down(served_hidden, ckc, self.mlp.args.mlp_w2_decode_1d_progcfg_64)
            held.append(operations.clone(served_partial, memory_config=dram))
        except BaseException:
            for tensor in held:
                operations.deallocate(tensor)
            raise
        _HELD.append(dict(owner=None, label=None, layer=self.index, route=self.plan.route, name=self.plan.name, kind='hidden',
                          mine=held[0], served=held[2]))
        _HELD.append(dict(owner=None, label=None, layer=self.index, route=self.plan.route, name=self.plan.name, kind='down',
                          mine=held[1], served=held[3]))
        return self.reduce(served_partial)


class Tp4MlpBinding(object):
    """The block's MLP binding for the lever: one Tp4MlpForward per layer bound on `feed_forward.forward` around the block's forward; the
    block checks `calls` against `expected_calls` each forward (model_batch.run)."""

    label = 'MLP gate/up lever'

    def __init__(self, model, rows, operations, plan, fused=None):
        layers = list(getattr(model, 'layers', ()))
        if not layers or any(getattr(layer, 'feed_forward', None) is None for layer in layers):
            raise ValueError('A model whose every layer carries a feed_forward MLP is required')
        self.rows, self.plan = rows, plan
        _STATE['layers'] = len(layers)
        self.forwards = [Tp4MlpForward(layer.feed_forward, index, rows, operations, plan,
                                       fused=None if fused is None else fused[index]) for index, layer in enumerate(layers)]
        self.bindings = [(layer.feed_forward, 'forward', forward) for layer, forward in zip(layers, self.forwards)]
        self.expected_calls = len(self.forwards)

    @property
    def calls(self):
        return sum(forward.calls for forward in self.forwards)

    @property
    def fell_back(self):
        return sum(forward.fell_back for forward in self.forwards)


def bindings(model, rows, operations, native_m3=True, environ=None):
    """The binders model_batch.two_tile_bindings appends for the lever: () when no lever is on, otherwise one Tp4MlpBinding. A flag combination
    that cannot run raises here (at the block's warm, never mid-round); a model or grid the lever cannot serve raises with the reason, after
    one fell-back line, so a profile that asks for the lever never silently times the served path."""
    selection = resolve(environ)
    if selection.route is None:
        return ()
    try:
        if rows != ROWS:
            raise ValueError('the lever serves the %d-row block, not %d rows' % (ROWS, rows))
        if not native_m3:
            raise ValueError('the lever needs the Lever N native graft (the 64-row MLP configs)')
        args = model.args
        grid = model.mesh_device.compute_with_storage_grid_size()
        plan = build_plan(selection, args, operations, int(grid.x), int(grid.y))
        fused = None
        if plan.route == 'fused':
            import tp4_mlp_fused

            width = selection.cfg.w if selection.cfg is not None and selection.cfg.w is not None else None
            fused = [tp4_mlp_fused.FusedGateUp(operations, layer.feed_forward.device, layer.feed_forward.weights.w1,
                                               layer.feed_forward.weights.w3, pairs_per_worker=plan.fused_pairs,
                                               grid=(int(grid.x), int(grid.y)), width=width,
                                               math_approx_mode=bool(getattr(layer.feed_forward.compute_kernel_config_decode,
                                                                              'math_approx_mode', True)))
                     for layer in model.layers]
    except (ValueError, AttributeError, TypeError) as error:
        log_line(fallback_line(str(error).splitlines()[0][:200]))
        raise
    return (Tp4MlpBinding(model, rows, operations, plan, fused=fused),)


# ---------------------------------------------------------------------------------------------------------------------------
# The audit: compared after the replay (packed_verifier calls audit_claim after the warm forward and the capture, audit_round after a
# replay, audit_release before the fixture closes - the three hooks tile_collective_tp has).
# ---------------------------------------------------------------------------------------------------------------------------

def audit_claim(owner, kind='audit'):
    """Give every audited pair not yet owned to `owner` (the fixture whose forward just ran). Returns how many."""
    claimed, label = 0, None
    for pair in _HELD:
        if pair['owner'] is None:
            if label is None:
                _STATE['owners'] = _STATE.get('owners', 0) + 1
                label = '%s%d' % (kind, _STATE['owners'])
            pair['owner'], pair['label'] = owner, label
            claimed += 1
    return claimed


def audit_replayed(owner):
    """Note that `owner`'s trace replayed just now (the clones are valid only right after their own block's replay)."""
    if _HELD:
        _STATE['replayed'] = owner


def audit_pairs(owner):
    return [pair for pair in _HELD if pair['owner'] is owner]


def audit_held_of(owner):
    return [pair[name] for pair in audit_pairs(owner) for name in ('mine', 'served') if pair.get(name) is not None]


def audit_round(operations, owner, round_number, log=None):
    """Compare every pair `owner` holds on every chip as int16 bit patterns (-0 and +0 differ). Returns the pairs compared (0: audit off).
    One audit line on rounds 0 to 3 and every 50th; any differing element, shape or chip count logs AUDIT_MISMATCH and raises."""
    pairs = audit_pairs(owner)
    if not pairs:
        return 0
    if round_number > 0 and _STATE.get('replayed') is not owner:
        raise AssertionError('%s round=%d: the audit was read after another block replayed, which may have overwritten these clones'
                             % (AUDIT_MISMATCH, round_number))
    import torch

    log = log or log_line
    mismatches, elements, layers = [], 0, set()
    for pair in pairs:
        label = 'layer=%d %s' % (pair['layer'], pair['kind'])
        lefts, rights = operations.get_device_tensors(pair['mine']), operations.get_device_tensors(pair['served'])
        if len(lefts) != len(rights):
            mismatches.append('%s chips %d against %d' % (label, len(lefts), len(rights)))
            continue
        layers.add(pair['layer'])
        for chip, (left, right) in enumerate(zip(lefts, rights)):
            a = operations.to_torch(left).contiguous().view(torch.int16)
            b = operations.to_torch(right).contiguous().view(torch.int16)
            if a.shape != b.shape:
                mismatches.append('%s chip %d: shape %r against %r' % (label, chip, tuple(a.shape), tuple(b.shape)))
            elif not torch.equal(a, b):
                mismatches.append('%s chip %d: %d of %d elements differ' % (label, chip, int((a != b).sum()), a.numel()))
            else:
                elements += a.numel()
    first = pairs[0]
    if mismatches:
        message = '%s round=%d route=%s name=%s exact=False %s' % (AUDIT_MISMATCH, round_number, first['route'], first['name'],
                                                                  '; '.join(mismatches[:4]))
        log(message)
        raise AssertionError(message)
    _AUDIT['rounds'] += 1
    if round_number <= 3 or round_number % 50 == 0:
        log('%s %d exact=True route=%s name=%s owner=%s round=%d layers=%d pairs=%d chips=%d elements=%d' % (
            AUDIT_MARKER, _AUDIT['rounds'], first['route'], first['name'], first['label'], round_number, len(layers), len(pairs),
            len(operations.get_device_tensors(first['mine'])), elements))
    return len(pairs)


def audit_release(operations, owner):
    """Free every clone `owner` holds (before its fixture closes). Returns how many pairs."""
    pairs = audit_pairs(owner)
    if _STATE.get('replayed') is owner:
        _STATE['replayed'] = None
    for pair in pairs:
        _HELD.remove(pair)
        for name in ('mine', 'served'):
            if pair.get(name) is not None:
                operations.deallocate(pair[name])
    return len(pairs)
