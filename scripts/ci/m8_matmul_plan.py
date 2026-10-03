"""The 128-row matmul plan for the one-block verify (tp4/m8-phase1): which seven configs, at which shapes, and what must hold of them.

block128-design.md stage B1: the model's seven 1D decode matmuls (attn_qkv, attn_wo, gdn_qkvzab, gdn_out, mlp w1, w3, w2) and the LM head run at
M = 128 as one call instead of two 64-row calls. That is exact only if every 32-row tile of the 128-row output has the bits the 64-row halves
(and the sequential engine's 32-row tiles) give. The configs are built by the SAME builder at a new M (tp_common.create_matmul_1d_decode_progcfg:
per_core_M = ceil(m / 32), in0_block_w from K alone), so the arithmetic below is held by CPU tests; the bytes are the card's (m8_matmul_bytes.py).

What is in the table is read from the graft (docker/qwen-c2-graft/graft/model_config.py: the seven `*_progcfg_64`, and verify-trace T1 #11's wider
attn_qkv and gate), the per-chip widths from tp_shapes (the one table), and the weight dtypes from the graft's loaders (gate and up bfloat4_b, the
rest bfloat8_b). The compute kernel configs are the model's: the MLP's `compute_kernel_config_decode` (LoFi, fp32 dest accumulation, packer L1
accumulation) and the attention / GDN `COMPUTE_HIFI2`. The LM head is `ttnn.linear(x, lm_head_weight)` with NO program config (model.py): ttnn
picks its config from M, and that is the one real unknown of the design.

Stdlib only, py 3.7 (matmul_tp4_sweep is imported for its pure builder arithmetic; it imports ttnn only inside its device functions).
"""

import math

import matmul_tp4_sweep as sweep
import tp_shapes

TP = 4
TILE = 32
HALF = 64
BLOCK = 128
DECODE_GRID_W = 11
# The three row counts a projection is timed at: the sequential engine's tile, the packed 64-row block, the one 128-row block.
TIMED_ROWS = (32, 64, 128)
# The LM head's logits: one row tile at a time through the sampler, so the byte unit is the 32-row tile.
UNIT = TILE
# in0_block_w, fuse_batch and mcast_in0 are the K order; per_core_M is M itself (the T1 #11 rule: _QWEN_VERIFY_T1_KEPT).
K_FIELDS = ('in0_block_w',)
CONSTANT_FIELDS = (('fuse_batch', True), ('mcast_in0', True))


def geometry():
    return tp_shapes.geometry(TP)


def _attn_qkv_columns(found):
    # [q | k | v | gate]: NH * HD + 2 * NKV * HD + NH * HD (attention/tp.py prepare_attn_qkv_deint)
    return 2 * found.attn_heads * tp_shapes.ATTENTION_HEAD_DIM + 2 * found.attn_kv_heads * tp_shapes.ATTENTION_HEAD_DIM


def projections():
    """The eight projections of a verify block, in model order. Each: name, K, N (per chip, tile padded), the weight dtype, whether SiLU is fused,
    the compute kernel config key, and `variants` {name: builder kwargs (num_cores, grid_w)} - 'base' is model_config's `_64` build, 'served' the
    build the serving recipe runs (verify-trace T1 #11 widens attn_qkv to 44 cores and the gate to 88). The LM head has no variants: auto."""
    found = geometry()
    hidden = tp_shapes.HIDDEN
    pad = tp_shapes.ceil_tile
    return [
        dict(name='attn_qkv', k=hidden, n=pad(_attn_qkv_columns(found)), dtype='bfp8', silu=False, compute='hifi2',
             variants=dict(base=dict(num_cores=64, grid_w=8), served=dict(num_cores=44, grid_w=DECODE_GRID_W))),
        dict(name='attn_wo', k=found.attn_out, n=hidden, dtype='bfp8', silu=False, compute='hifi2',
             variants=dict(base=dict(num_cores=33, grid_w=DECODE_GRID_W), served=dict(num_cores=33, grid_w=DECODE_GRID_W))),
        dict(name='gdn_qkvzab', k=hidden, n=found.gdn_qkvzab_padded, dtype='bfp8', silu=False, compute='hifi2',
             variants=dict(base=dict(num_cores=44, grid_w=DECODE_GRID_W), served=dict(num_cores=44, grid_w=DECODE_GRID_W))),
        dict(name='gdn_out', k=found.gdn_value, n=hidden, dtype='bfp8', silu=False, compute='hifi2',
             variants=dict(base=dict(num_cores=33, grid_w=DECODE_GRID_W), served=dict(num_cores=33, grid_w=DECODE_GRID_W))),
        dict(name='mlp_w1', k=hidden, n=found.mlp, dtype='bfp4', silu=True, compute='lofi',
             variants=dict(base=dict(num_cores=44, grid_w=DECODE_GRID_W), served=dict(num_cores=88, grid_w=DECODE_GRID_W))),
        dict(name='mlp_w3', k=hidden, n=found.mlp, dtype='bfp4', silu=False, compute='lofi',
             variants=dict(base=dict(num_cores=44, grid_w=DECODE_GRID_W), served=dict(num_cores=44, grid_w=DECODE_GRID_W))),
        dict(name='mlp_w2', k=found.mlp, n=hidden, dtype='bfp8', silu=False, compute='lofi',
             variants=dict(base=dict(num_cores=33, grid_w=DECODE_GRID_W), served=dict(num_cores=33, grid_w=DECODE_GRID_W))),
        dict(name='lm_head', k=hidden, n=found.vocab, dtype='bfp8', silu=False, compute='default', variants={}),
    ]


def seven():
    """The seven 1D-mcast projections (the LM head is the eighth, auto)."""
    return [entry for entry in projections() if entry['variants']]


def named(name):
    for entry in projections():
        if entry['name'] == name:
            return entry
    raise KeyError(name)


def config_at(entry, variant, m):
    """The 1D program config of `entry`'s `variant` at M = m, as the model's builder makes it: sweep.builder_config is held equal to the graft's own
    transcription of tp_common.create_matmul_1d_decode_progcfg by test_matmul_tp4_sweep. fused SiLU is carried as a flag, not a field."""
    args = entry['variants'][variant]
    config = sweep.builder_config(m, entry['k'], entry['n'], args['num_cores'], args['grid_w'])
    config['fused_silu'] = entry['silu']
    config['fuse_batch'] = True
    config['mcast_in0'] = True
    return config


def k_order_problem(first, second):
    """Why two configs may not share a K order (None when they do): in0_block_w, fuse_batch and mcast_in0 must be equal; per_core_M, per_core_N,
    the grid and the output subblock are the free partition. This is `_QWEN_VERIFY_T1_KEPT` of the graft, applied to M = 64 against M = 128."""
    for field in K_FIELDS:
        if first[field] != second[field]:
            return '%s %r -> %r' % (field, first[field], second[field])
    for field, value in CONSTANT_FIELDS:
        if first.get(field) is not value or second.get(field) is not value:
            return '%s is not %r' % (field, value)
    return None


def sub_block_changes(first, second):
    """The output-subblock fields that moved: (h, w) of each. They are not the K order (the packer accumulates each output tile over K blocks in
    the same order whatever the subblock; docs and review section 3), but B1 reports them because the per-tile bytes are what proves that."""
    return dict(first=(first['out_subblock_h'], first['out_subblock_w']), second=(second['out_subblock_h'], second['out_subblock_w']))


def plan_for(entry, variant):
    """Everything B1 needs about one (projection, variant): its configs at 32, 64 and 128 rows, the K-order verdict, the L1 estimate at 128 rows
    and the cores the output columns use."""
    configs = {m: config_at(entry, variant, m) for m in TIMED_ROWS}
    n_tiles = math.ceil(entry['n'] / TILE)
    return dict(
        name=entry['name'], variant=variant, k=entry['k'], n=entry['n'], dtype=entry['dtype'], configs=configs,
        k_order_problem=k_order_problem(configs[HALF], configs[BLOCK]),
        k_order_problem_tile=k_order_problem(configs[TILE], configs[BLOCK]),
        subblocks=sub_block_changes(configs[HALF], configs[BLOCK]),
        l1_bytes_128=sweep.l1_bytes(configs[BLOCK], entry['dtype']),
        active_cores_128=sweep.active_cores(configs[BLOCK], n_tiles))


def regrid_candidates(entry, m=BLOCK):
    """The re-grid candidates at per_core_M = m / 32: matmul_tp4_sweep.stage1 (one config per per_core_N on the model's grid width, the model's
    in0_block_w, the best subblock, within the L1 budget). Every one keeps the K order, so every one is expected exact: the sweep says so."""
    shape = dict(name=entry['name'], K=entry['k'], N=entry['n'], dtype=entry['dtype'], silu=entry['silu'])
    out = []
    for config in sweep.stage1(shape, m=m):
        config = dict(config)
        config['fused_silu'] = entry['silu']
        config['fuse_batch'] = True
        config['mcast_in0'] = True
        out.append(config)
    return out


def builder_arguments(config, entry, m=BLOCK):
    """The (num_cores, grid_w) a builder call needs to reproduce `config`'s partition at M = m (matmul_tp4_sweep.builder_arguments is the M = 64 form)."""
    cols, rows = config['grid']
    built = sweep.builder_config(m, entry['k'], entry['n'], cols * rows, cols)
    return dict(num_cores=cols * rows, grid_w=cols,
                reproduces=all(built[field] == config[field] for field in ('per_core_N', 'out_subblock_h', 'out_subblock_w', 'in0_block_w')))


def rows_of_tile(tile, unit=UNIT):
    return tile * unit, (tile + 1) * unit


def halves():
    """The two 64-row halves of the 128-row block: (first row, last row) each."""
    return ((0, HALF), (HALF, BLOCK))


def timing_verdict(t32, t64, t128):
    """The review's two timing rules for one projection: t128 <= t64 + 1.5 * (t64 - t32) (a linear extrapolation with half again of room), and
    the design's t128 <= 1.1 * t64. Either holds -> `ok`. A missing time is None."""
    if None in (t32, t64, t128):
        return dict(ok=None, extrapolated=None, ratio=None)
    extrapolated = t64 + 1.5 * max(t64 - t32, 0.0)
    return dict(ok=t128 <= extrapolated or t128 <= 1.1 * t64, extrapolated=extrapolated, ratio=t128 / t64)


def total_timing_verdict(times):
    """The review's second rule over the projections together: sum of t128 <= 1.1 x sum of t64 (`times` {name: (t32, t64, t128)})."""
    rows = [value for value in times.values() if None not in value]
    if not rows:
        return None
    return sum(value[2] for value in rows) <= 1.1 * sum(value[1] for value in rows)
