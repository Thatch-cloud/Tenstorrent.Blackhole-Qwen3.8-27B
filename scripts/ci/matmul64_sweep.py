"""M=64 decode matmul grid sweep for the seven Lever N M3native 1D-mcast configs.

Device profile of run 35567165791 (chip 0, the four-user 64-row packed decode round)
shows the seven M64 decode matmuls (Qwen3.5-27B, TP2, bf8 weights) costing ~1 ms each,
about ten times their DRAM-bandwidth floor, and summing to 291 ms of the 1034 ms round:

  MatmulDeviceOperation 39 cores: 124 calls x 1.081 ms  (mlp_w1/w3_decode_1d_progcfg_64)
  MatmulDeviceOperation 32 cores: 124 calls x 0.909 ms  (mlp_w2 / attn_wo / gdn_out _64)
  MatmulDeviceOperation 43 cores:  48 calls x 0.916 ms  (gdn_qkvz_decode_1d_progcfg_64)

Every one of those seven configs is built by
models.demos.blackhole.qwen36.tt.tp_common.create_matmul_1d_decode_progcfg(M, K, N, ...)
in Qwen36ModelArgs._init_tp_config (model_config.py), with per_core_M = ceil(M / 32) and
a core grid sized for M = 32 (see the "_64" siblings' num_cores/grid_w arguments in
model_config.py, byte-identical to their M = 1 counterparts apart from M itself). At
M = 64 the same grid does twice the per-core work, which is the likely cause of the
~10x-floor cost. This sweep measures the real per-chip shapes at M in (32, 64) over a
range of core grids and in0_block_w choices so the seven configs can be widened.

Four measurement arms per shape, per M, per activation placement:
  (a) model_current   - the model's own tp_common.create_matmul_1d_decode_progcfg(...)
                         config (the exact call model_config.py makes), rebuilt here by
                         calling that same builder - not reimplemented.
  (b) grid_bxh_blkw    - a swept ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig over
                         core grids (up to the Blackhole p150a 11x10 worker grid) and
                         in0_block_w choices that divide K in tiles.
  (c) auto             - ttnn.linear with program_config=None (ttnn's own auto config),
                         as a no-tuning reference point.
  (d) dram_sharded     - tp_common's DRAM-width-sharded weight + matmul program config
                         (the model's M=1 path for some projections), extended to M=64.

Weights are bf8 (bfloat8_b), activations bf16, matching the model's decode precision
(model_config.py: self.weight_dtype = ttnn.bfloat8_b, self.act_dtype = ttnn.bfloat16).
Activation memory is swept over L1 (the model's actual 1D-decode placement - mlp.py's
`x_il = ttnn.to_memory_config(x, ttnn.L1_MEMORY_CONFIG)`) and DRAM-interleaved (a
secondary comparison point; mcast_in0 only requires an interleaved in0, not L1
specifically).

Runs entirely inside the serving image on ONE device - no mesh, no CCL, no weights or
fixtures. Import of ttnn/torch/the model package is deferred to the functions that need
it, so this module (and every pure function in it) imports and unit-tests cleanly under
plain CPython with no ttnn installed; see test_matmul64_sweep.py.
"""

import argparse
import json
import math
import statistics
import time
from pathlib import Path

TILE = 32

# Blackhole p150a physical worker grid, as MEASURED on the rig card this sweep ran on
# (device.compute_with_storage_grid_size() == (11, 10) - run against image v65, card
# M; 12x/13x candidates only produced TT_FATAL rows there). The model's own
# decode_grid_w (mesh_device.compute_with_storage_grid_size().x, 11 on BH per
# model_config.py's comments) already matches this width - the whole point of this
# sweep is to look past its narrow HEIGHT (grid_w=11 but only ~4 rows used).
WORKER_GRID_X = 11
WORKER_GRID_Y = 10

DRAM_BANDWIDTH_GBPS = 400.0

# bfloat8_b storage: tt-metal's bfp8_b tile packs 1024 (32x32) 8-bit mantissas plus a
# shared 8-bit exponent per 16x16 sub-block (4 sub-blocks/tile = 64 exponent bytes) ->
# 1088 bytes / 1024 elements = 1.0625 bytes/element. This is only used for the
# DRAM-floor reference ratio below, never for the actual on-device tensor (ttnn picks
# its own exact tile layout when the tensor is created).
BFP8_BYTES_PER_TILE = TILE * TILE + 64
BFP8_BYTES_PER_ELEMENT = BFP8_BYTES_PER_TILE / (TILE * TILE)

# The serving config this sweep targets (docs/lever-N-prefill-decode-interleave.md,
# goal-200tps-concurrent): TP2 on two Blackhole p150a.
MODEL_TP = 2

# Six corners/edges of the real 11x10 worker grid this sweep measured against
# (12x/13x candidates were dropped after they produced TT_FATAL on the rig - the
# device's actual compute_with_storage_grid_size() tops out at (11, 10)).
NAMED_GRIDS = ((8, 8), (10, 8), (11, 8), (8, 10), (10, 10), (11, 10))

# The seven decode matmuls, in model_config.py's own NATIVE_64_BLOCK order. Each entry
# is (name, K-from, N-from, num_cores, grid_w_pinned, silu) where K/N-from select which
# per-chip dim (see shape_table) and grid_w_pinned says whether the model call passes
# `grid_w=self.decode_grid_w` (True) or omits grid_w entirely, letting tp_common's own
# default apply (False, attn_qkv_fused only).
_SHAPE_SPECS = (
    ("mlp_w1", "dim", "hidden_tp", 44, True, True),
    ("mlp_w3", "dim", "hidden_tp", 44, True, False),
    ("mlp_w2", "hidden_tp", "dim", 33, True, False),
    ("attn_qkv_fused", "dim", "attn_qkv_fused_dim_tp", 64, False, False),
    ("gdn_qkvz", "dim", "gdn_qkvzab_dim_tp", 44, True, False),
    ("attn_wo", "attn_out_dim_tp", "dim", 33, True, False),
    ("gdn_out", "gdn_value_dim_tp", "dim", 33, True, False),
)


def tiles(n):
    """n in 32-row/col tiles, rounding UP (ceiling) for a non-tile-aligned dim. Never
    raises: real per-chip GDN widths are NOT always tile-aligned (gdn_qkvz's real N was
    8240 on the rig, not a multiple of 32 - head-count arithmetic, not tile alignment,
    drives that width), and ttnn pads a non-aligned tensor to the next full tile
    itself, so this mirrors that rather than rejecting the shape outright."""
    if n <= 0:
        raise ValueError("tiles() needs a positive size, got %d" % n)
    return math.ceil(n / TILE)


def pad_to_tile(n):
    """n rounded UP to the next multiple of TILE (32) - the padded width ttnn actually
    allocates on device for a non-tile-aligned dim (pad_to_tile(8240) == 8256)."""
    return tiles(n) * TILE


def divisors(n):
    if n <= 0:
        raise ValueError("divisors() needs a positive integer, got %r" % (n,))
    return [d for d in range(1, n + 1) if n % d == 0]


def gdn_tp_dims(nk, nv, dk, dv, tp):
    """Mirror Qwen36ModelArgs._init_tp_config's GDN arithmetic (model_config.py):
    gdn_key_dim = nk*dk (== gdn_value's q/k dim), gdn_value_dim = nv*dv, qkv_dim =
    2*key + value, qkvz_dim = qkv_dim + value (z has the same width as v), qkvzab folds
    in the per-chip (a, b) decay/beta projection (2*nv_tp additional columns)."""
    if tp <= 0 or nk % tp or nv % tp:
        raise ValueError("GDN head counts (nk=%d, nv=%d) must divide TP=%d" % (nk, nv, tp))
    key_dim = nk * dk
    value_dim = nv * dv
    qkv_dim = 2 * key_dim + value_dim
    z_dim = value_dim
    nv_tp = nv // tp
    qkvz_dim_tp = (qkv_dim + z_dim) // tp
    return dict(gdn_qkvzab_dim_tp=qkvz_dim_tp + 2 * nv_tp, gdn_value_dim_tp=value_dim // tp)


def attn_tp_dims(n_heads, n_kv_heads, head_dim, tp):
    """Mirror _init_tp_config's attention arithmetic: fused [q+gate|k|v] in-projection
    width (attn_qkv_fused_dim_tp) and the per-chip attention output width
    (attn_out_dim_tp), both column-parallel over TP."""
    if tp <= 0 or n_heads % tp:
        raise ValueError("n_heads=%d must divide TP=%d" % (n_heads, tp))
    n_local_heads = n_heads // tp
    n_local_kv_heads = max(1, n_kv_heads // tp)
    kv_dim_per_device = n_local_kv_heads * head_dim
    return dict(
        attn_qkv_fused_dim_tp=n_local_heads * head_dim * 2 + 2 * kv_dim_per_device,
        attn_out_dim_tp=(n_heads * head_dim) // tp,
    )


REQUIRED_DIMS = ("dim", "hidden_dim", "n_heads", "n_kv_heads", "head_dim",
                 "gdn_nk", "gdn_nv", "gdn_dk", "gdn_dv")


def shape_table(dims, tp=MODEL_TP):
    """The seven (K, N) shapes tp_common.create_matmul_1d_decode_progcfg is called
    with in Qwen36ModelArgs._init_tp_config, from real model dims (see
    load_model_dims: dim/hidden_dim/n_heads/n_kv_heads/head_dim plus the GDN raw head
    counts). Never hardcodes a guessed head count - `dims` should come from a live
    Qwen36ModelArgs(mesh_device=None) at runtime, or an equivalent fixture in tests."""
    missing = [key for key in REQUIRED_DIMS if key not in dims]
    if missing:
        raise ValueError("shape_table missing dims: %s" % ", ".join(sorted(missing)))
    derived = dict(
        dim=dims["dim"],
        hidden_tp=dims["hidden_dim"] // tp,
    )
    derived.update(attn_tp_dims(dims["n_heads"], dims["n_kv_heads"], dims["head_dim"], tp))
    derived.update(gdn_tp_dims(dims["gdn_nk"], dims["gdn_nv"], dims["gdn_dk"], dims["gdn_dv"], tp))
    table = []
    for name, k_key, n_key, num_cores, grid_w_pinned, silu in _SHAPE_SPECS:
        K, N = derived[k_key], derived[n_key]
        table.append(dict(
            name=name, K=K, N=N, K_padded=pad_to_tile(K), N_padded=pad_to_tile(N),
            num_cores=num_cores, grid_w_pinned=grid_w_pinned, silu=silu,
        ))
    return table


def weight_bytes(K, N):
    """Bytes for the weight ttnn actually allocates: K and N are padded up to the next
    full tile first (real DRAM traffic reads whole tiles, never a partial one - e.g.
    gdn_qkvz's real N of 8240 is read as its 8256-wide padded tile allocation)."""
    return pad_to_tile(K) * pad_to_tile(N) * BFP8_BYTES_PER_ELEMENT


def dram_floor_us(K, N, bandwidth_gbps=None):
    """Microseconds to stream this weight once out of DRAM at bandwidth_gbps GB/s -
    the floor every decode matmul is compared against (device profile shows ~10x this).
    bandwidth_gbps defaults to the CURRENT module-level DRAM_BANDWIDTH_GBPS (read at
    call time, not bound at import time), so --bandwidth-gbps in main() takes effect
    for every later call even though it reassigns the global after this module loads."""
    if bandwidth_gbps is None:
        bandwidth_gbps = DRAM_BANDWIDTH_GBPS
    return weight_bytes(K, N) / (bandwidth_gbps * 1e9) * 1e6


def legal_grids(n_tiles, max_x=WORKER_GRID_X, max_y=WORKER_GRID_Y):
    """Every (x, y) with x<=max_x, y<=max_y, cores<=n_tiles (every core gets at least
    one output tile column - a core with none assigned is not a meaningful config).
    per_core_N is computed elsewhere by CEILING division (per_core_N = ceil(n_tiles /
    cores)), the same convention tp_common uses for per_core_M = ceil(M / TILE), so an
    uneven split pads the last core rather than being excluded.

    An earlier version of this function required cores to divide n_tiles exactly. That
    is provably too strict: the model's own mlp_w1/w3 config (num_cores=44 over
    hidden_tp=8704 -> 272 tiles) is not exact either (272 / 44 = 6.18), and
    gdn_qkvz's N (6176, 193 tiles - 193 is prime) has NO divisor between 1 and
    itself, which made the exact-division sweep degenerate to a single 1x1 grid for
    that shape alone. Ceiling-based per_core_N matches what tp_common already does and
    keeps the full 11x10 worker grid in play for every shape."""
    out = set()
    for x in range(1, max_x + 1):
        for y in range(1, max_y + 1):
            cores = x * y
            if cores <= n_tiles:
                out.add((x, y))
    return sorted(out, key=lambda xy: (xy[0] * xy[1], xy))


def select_grids(n_tiles, quick=True, extra=(), budget=20):
    """quick keeps the six named grids (8x8, 10x8, 11x8, 8x10, 10x10, 11x10) that are legal for
    this N (cores <= n_tiles - see legal_grids), any `extra` grids (e.g. the model's
    own), and a core-count-spread sample of the rest up to `budget` grids total (the
    named/extra grids count against the same budget). build_configs sizes `budget`
    to leave room for its own in0_block_w probes, so the whole shape stays near ~24
    configs without either axis silently truncating the other's candidates (a fixed
    20-grid budget plus up to 6 block probes could overflow a flat 24-config cap and
    lose whichever block width iterated last - see build_configs). full returns every
    legal grid up to the 11x10 worker grid, uncapped."""
    legal = legal_grids(n_tiles)
    legal_set = set(legal)
    keep = [grid for grid in NAMED_GRIDS if grid in legal_set]
    for grid in extra:
        if grid in legal_set and grid not in keep:
            keep.append(grid)
    if not quick:
        return sorted(set(legal) | set(keep), key=lambda xy: (xy[0] * xy[1], xy))
    keep = keep[:budget]
    remaining = [grid for grid in legal if grid not in keep]
    remaining_budget = max(0, budget - len(keep))
    if remaining and remaining_budget:
        step = max(1, len(remaining) // remaining_budget)
        keep.extend(remaining[::step][:remaining_budget])
    return sorted(set(keep), key=lambda xy: (xy[0] * xy[1], xy))


def block_choices(k_tiles, quick=True):
    """in0_block_w candidates (units of 32-element tiles of K), all exact divisors of
    K in tiles. quick keeps a canonical small set (1/2/4/8/16 tiles, matching
    mlp-sweep.py's existing 4/8/16-tile blocking convention, plus the full-K block)."""
    divs = divisors(k_tiles)
    if not quick:
        return divs
    preferred = sorted(d for d in (1, 2, 4, 8, 16, k_tiles) if d in divs)
    return preferred or [k_tiles]


def out_subblocks(per_core_m, per_core_n, cap=4):
    """(h, w) pairs with h | per_core_m, w | per_core_n, h*w <= cap - the same
    dest-register-limited convention mlp-sweep.py's geometry() uses (cap=4), sorted
    largest-product first so callers can just take the first entry."""
    pairs = [(h, w) for h in divisors(per_core_m) for w in divisors(per_core_n) if h * w <= cap]
    if not pairs:
        raise ValueError("no (out_subblock_h, out_subblock_w) fits cap=%d for "
                         "per_core_M=%d, per_core_N=%d" % (cap, per_core_m, per_core_n))
    return sorted(set(pairs), key=lambda hw: (-(hw[0] * hw[1]), hw))


def build_configs(K, N, M, model_grid=None, quick=True):
    """Every (grid, in0_block_w) combination to try for one shape at one M, as plain
    kwargs for ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig (mlp.py's mcast_in0 1D
    decode branch; field names match mlp-sweep.py's own geometry() builder). Grid is
    the primary swept axis; in0_block_w is only varied at one grid (the model's own
    grid, or the smallest selected grid if none given). The grid budget is sized to
    leave room for every block probe up front (rather than truncating a flat
    grids-then-blocks list at ~24), so quick mode never silently drops a specific
    in0_block_w value - e.g. the full-K block, which iterates last and was the value
    an earlier version of this function lost to a trailing [:24] slice."""
    n_tiles = tiles(N)
    k_tiles = tiles(K)
    per_core_m = math.ceil(M / TILE)
    block_candidates = block_choices(k_tiles, quick=quick)
    # quick targets ~24 configs total: reserve one slot per block probe (fewer if that
    # would leave too few grids to be a useful sweep) and spend the rest on grids.
    grid_budget = max(12, 24 - len(block_candidates)) if quick else None
    grids = select_grids(n_tiles, quick=quick, extra=(model_grid,) if model_grid else (),
                        **(dict(budget=grid_budget) if quick else {}))
    default_block = max(d for d in divisors(k_tiles) if d <= min(8, k_tiles))
    configs = []
    seen = set()

    def add(grid, block_w):
        key = (grid, block_w)
        if key in seen:
            return
        seen.add(key)
        cores = grid[0] * grid[1]
        if cores > n_tiles or cores == 0:
            return
        # Ceiling, not floor: matches tp_common's own per_core_M = ceil(M / TILE) and
        # its real (non-exact) num_cores=44-over-272-tiles mlp_w1/w3 config - a config
        # this produces may pad the last core's shard rather than split evenly.
        per_core_n = math.ceil(n_tiles / cores)
        out_h, out_w = out_subblocks(per_core_m, per_core_n)[0]
        configs.append(dict(grid=grid, in0_block_w=block_w, per_core_M=per_core_m,
                             per_core_N=per_core_n, out_subblock_h=out_h, out_subblock_w=out_w))

    for grid in grids:
        add(grid, default_block)
    block_probe_grid = model_grid if (model_grid and model_grid in grids) else (grids[0] if grids else (1, 1))
    for block_w in block_candidates:
        add(block_probe_grid, block_w)
    return configs


def config_label(cfg):
    return "grid_%dx%d_blk%d" % (cfg["grid"][0], cfg["grid"][1], cfg["in0_block_w"])


# ---------------------------------------------------------------------------------
# Report assembly (pure - operates on plain dicts, no ttnn dependency).
# ---------------------------------------------------------------------------------

def summarize(runs, shapes_by_name):
    """Group flat per-run records (one per shape/M/activation/arm/config) into one
    candidate per (shape, arm label, activation), each carrying its m32 and m64
    sub-results, sorted per shape by m64 min_us ascending (errored/missing candidates
    sort last), with the model's current config flagged and a DRAM-floor ratio
    attached to every m64 timing."""
    candidates = {}
    order = []
    for run in runs:
        key = (run["shape"], run["arm"], run["activation"])
        if key not in candidates:
            candidates[key] = dict(shape=run["shape"], arm=run["arm"], activation=run["activation"],
                                   is_model_current=run.get("is_model_current", False), config=run.get("config"))
            order.append(key)
        slot = "m%d" % run["M"]
        payload = {k: v for k, v in run.items()
                  if k not in ("shape", "arm", "activation", "M", "is_model_current", "config")}
        candidates[key][slot] = payload
        candidates[key]["is_model_current"] = candidates[key]["is_model_current"] or run.get("is_model_current", False)
        if run.get("config") is not None:
            candidates[key]["config"] = run["config"]

    def sort_key(key):
        m64 = candidates[key].get("m64", {})
        if "error" in m64 or "min_us" not in m64:
            return (1, 0.0)
        return (0, m64["min_us"])

    by_shape = {}
    for key in order:
        by_shape.setdefault(key[0], []).append(key)
    result = {}
    for shape_name, keys in by_shape.items():
        keys.sort(key=sort_key)
        shape = shapes_by_name[shape_name]
        floor = dram_floor_us(shape["K"], shape["N"])
        rows = []
        for key in keys:
            candidate = dict(candidates[key])
            m64 = candidate.get("m64", {})
            if "min_us" in m64:
                candidate["dram_floor_us"] = floor
                candidate["ratio_vs_dram_floor"] = m64["min_us"] / floor
            rows.append(candidate)
        result[shape_name] = rows
    return result


def format_table(summary, shapes_by_name):
    lines = []
    for shape_name, rows in summary.items():
        shape = shapes_by_name[shape_name]
        lines.append("== %s  K=%d N=%d  DRAM floor=%.2f us ==" %
                     (shape_name, shape["K"], shape["N"], dram_floor_us(shape["K"], shape["N"])))
        header = "%-24s %-6s %-10s %10s %10s %10s %10s %8s" % (
            "arm", "act", "current", "m32_mean", "m32_min", "m64_mean", "m64_min", "ratio")
        lines.append(header)
        for row in rows:
            m32, m64 = row.get("m32", {}), row.get("m64", {})

            def cell(payload, field):
                if "error" in payload:
                    return "ERR"
                if field not in payload:
                    return "-"
                return "%.1f" % payload[field]

            lines.append("%-24s %-6s %-10s %10s %10s %10s %10s %8s" % (
                row["arm"], row["activation"], "yes" if row["is_model_current"] else "",
                cell(m32, "mean_us"), cell(m32, "min_us"), cell(m64, "mean_us"), cell(m64, "min_us"),
                ("%.2fx" % row["ratio_vs_dram_floor"]) if "ratio_vs_dram_floor" in row else "-"))
        lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------------
# Device-dependent parts. Every ttnn/torch/model import is local to these functions
# so the module (and every function above) stays importable and unit-testable on
# plain CPython with no ttnn installed.
# ---------------------------------------------------------------------------------

def load_model_dims():
    """Read real per-chip shape inputs from the pinned Qwen36ModelArgs WITHOUT a mesh
    device. The base tt_transformers ModelArgs sets dim/n_heads/n_kv_heads/head_dim/
    hidden_dim, and Qwen36ModelArgs.__init__ sets the GDN raw counts
    (linear_num_key_heads etc.) whenever mesh_device is None - both run well before
    _init_tp_config (which needs a >1-device mesh and is skipped here). This is how
    the sweep gets the exact numbers model_config.py's _init_tp_config would use,
    without needing a second device."""
    from models.demos.blackhole.qwen36.tt.model_config import Qwen36ModelArgs

    args = Qwen36ModelArgs(mesh_device=None)
    return dict(
        dim=args.dim, hidden_dim=args.hidden_dim, n_heads=args.n_heads,
        n_kv_heads=args.n_kv_heads, head_dim=args.head_dim,
        gdn_nk=args.linear_num_key_heads, gdn_nv=args.linear_num_value_heads,
        gdn_dk=args.linear_key_head_dim, gdn_dv=args.linear_value_head_dim,
    )


def compute_kernel_config(ttnn):
    """Best-effort match to the model's decode compute kernel config (LoFi, packer L1
    accumulation - mlp.py / attention/tp.py / gdn/tp.py all build theirs the same way
    upstream in tt_transformers). A mismatch here biases every arm's absolute timing
    evenly (same config shared by every candidate), so it does not change which config
    wins - only the reported microsecond scale."""
    ctor = getattr(ttnn, "WormholeComputeKernelConfig", None) or getattr(ttnn, "BlackholeComputeKernelConfig", None)
    if ctor is None:
        return None
    return ctor(math_fidelity=ttnn.MathFidelity.LoFi, math_approx_mode=True,
               fp32_dest_acc_en=False, packer_l1_acc=True)


def model_progcfg(ttnn, name, K, N, M, num_cores, grid_w, silu):
    """The model's own program config for this shape/M: calls tp_common's real
    builder with the exact arguments model_config.py uses (grid_w omitted for
    attn_qkv_fused, matching _init_tp_config) - not a reimplementation."""
    from models.demos.blackhole.qwen36.tt import tp_common as tpc

    kwargs = dict(num_cores=num_cores)
    if grid_w is not None:
        kwargs["grid_w"] = grid_w
    if silu:
        kwargs["fused_activation"] = ttnn.UnaryOpType.SILU
    return tpc.create_matmul_1d_decode_progcfg(M, K, N, **kwargs)


def dram_sharded_progcfg(K, N, M):
    """tp_common's DRAM-width-sharded weight memory config + matmul program config -
    the model's M=1 path for w1/w3/w2/qkv/qkvz (model_config.py's DRAM-sharded matmul
    progcfgs section), extended here to M=64."""
    from models.demos.blackhole.qwen36.tt import tp_common as tpc

    return tpc.create_dram_sharded_mem_config(K, N), tpc.create_dram_sharded_matmul_program_config(M, K, N)


def make_weight(ttnn, torch, device, K, N, memory_config, seed=0):
    """Allocates at the PADDED (K, N) - the program configs this sweep builds size
    per_core_M/N off ceil(dim / TILE) tile counts, so the tensor handed to ttnn.linear
    must already be the tile-rounded width (gdn_qkvz's real N=8240 pads to 8256) rather
    than relying on from_torch's own padding to agree with a progcfg built separately."""
    torch.manual_seed(seed)
    weight = torch.randn(pad_to_tile(K), pad_to_tile(N), dtype=torch.bfloat16) * 0.02
    return ttnn.from_torch(weight, device=device, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT,
                           memory_config=memory_config)


def make_activation(ttnn, torch, device, M, K, memory_config, seed=0):
    """M is always 32 or 64 (already tile-aligned by construction); K is padded for
    the same reason make_weight pads - it must match the progcfg's tile-rounded K."""
    torch.manual_seed(seed + 1000)
    value = torch.randn(1, 1, M, pad_to_tile(K), dtype=torch.bfloat16) * 0.02
    return ttnn.from_torch(value, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                           memory_config=memory_config)


def time_matmul(ttnn, device, run_once, warmup=3, iters=20):
    """Warm up `warmup` iterations, then time `iters` individually (perf_counter +
    ttnn.synchronize_device around each call) and report mean/min microseconds, per
    the task's measurement method. Every output tensor is deallocated so repeated
    calls do not exhaust L1/DRAM across the sweep."""
    for _ in range(warmup):
        out = run_once()
        ttnn.deallocate(out)
    ttnn.synchronize_device(device)
    samples = []
    for _ in range(iters):
        started = time.perf_counter()
        out = run_once()
        ttnn.synchronize_device(device)
        samples.append((time.perf_counter() - started) * 1e6)
        ttnn.deallocate(out)
    return dict(mean_us=statistics.mean(samples), min_us=min(samples), samples_us=samples)


def run_shape(ttnn, torch, device, shape, quick, iters, warmup, activations, runs):
    """Run every arm ((a) model_current, (b) grid sweep, (c) auto, (d) dram_sharded)
    for one shape at M in (32, 64) x the requested activation placements. A failing
    CONFIG is caught and recorded as an error row rather than aborting the sweep (see
    safe_time below). Appends into the caller-owned `runs` list IN PLACE, rather than
    building and returning its own list, so a failure that escapes this function
    entirely (e.g. build_configs raising before any config for a shape has even been
    tried) still leaves every row measured so far in `runs` for main() to save -
    main() wraps the call in its own try/except per shape and saves after every one."""
    K, N, name = shape["K"], shape["N"], shape["name"]
    ckc = compute_kernel_config(ttnn)
    act_memcfgs = []
    if "l1" in activations:
        act_memcfgs.append(("L1", ttnn.L1_MEMORY_CONFIG))
    if "dram" in activations:
        act_memcfgs.append(("DRAM", ttnn.DRAM_MEMORY_CONFIG))

    weight = make_weight(ttnn, torch, device, K, N, ttnn.DRAM_MEMORY_CONFIG)
    model_grid_w = None
    if shape["grid_w_pinned"]:
        model_grid_w = device.compute_with_storage_grid_size().x

    def record(M, act_name, arm, is_model_current, config, timing_or_error):
        row = dict(shape=name, M=M, activation=act_name, arm=arm,
                  is_model_current=is_model_current, config=config)
        row.update(timing_or_error)
        runs.append(row)

    def safe_time(x, weight_tensor, pc, out_mc, activation_kw=None):
        def once():
            kwargs = dict(compute_kernel_config=ckc, memory_config=out_mc)
            if pc is not None:
                kwargs["program_config"] = pc
            elif activation_kw:
                kwargs["activation"] = activation_kw
            return ttnn.linear(x, weight_tensor, **kwargs)
        try:
            return time_matmul(ttnn, device, once, warmup=warmup, iters=iters)
        except Exception as error:  # noqa: BLE001 - a bad config must not kill the sweep
            return dict(error="%s: %s" % (type(error).__name__, error))

    for M in (32, 64):
        for act_name, act_mc in act_memcfgs:
            x = make_activation(ttnn, torch, device, M, K, act_mc)

            # (a) model's current config
            model_grid = None
            try:
                pc = model_progcfg(ttnn, name, K, N, M, shape["num_cores"], model_grid_w, shape["silu"])
                grid_size = pc.compute_with_storage_grid_size
                model_grid = (getattr(grid_size, "x", None), getattr(grid_size, "y", None))
                timing = safe_time(x, weight, pc, ttnn.L1_MEMORY_CONFIG)
                record(M, act_name, "model_current", True,
                      dict(grid=model_grid, in0_block_w=pc.in0_block_w,
                           per_core_M=pc.per_core_M, per_core_N=pc.per_core_N), timing)
            except Exception as error:  # noqa: BLE001
                record(M, act_name, "model_current", True, None,
                      dict(error="%s: %s" % (type(error).__name__, error)))

            # (b) grid / in0_block_w sweep
            for cfg in build_configs(K, N, M, model_grid=model_grid, quick=quick):
                try:
                    pc = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
                        compute_with_storage_grid_size=cfg["grid"], in0_block_w=cfg["in0_block_w"],
                        out_subblock_h=cfg["out_subblock_h"], out_subblock_w=cfg["out_subblock_w"],
                        per_core_M=cfg["per_core_M"], per_core_N=cfg["per_core_N"],
                        fuse_batch=True, mcast_in0=True,
                        fused_activation=ttnn.UnaryOpType.SILU if shape["silu"] else None,
                    )
                    timing = safe_time(x, weight, pc, ttnn.L1_MEMORY_CONFIG)
                except Exception as error:  # noqa: BLE001
                    timing = dict(error="%s: %s" % (type(error).__name__, error))
                record(M, act_name, config_label(cfg), False, cfg, timing)

            # (c) ttnn auto (no program_config)
            timing = safe_time(x, weight, None, ttnn.L1_MEMORY_CONFIG,
                              activation_kw="silu" if shape["silu"] else None)
            record(M, act_name, "auto", False, None, timing)

            # (d) DRAM-width-sharded weight + matmul program config (model's M=1 path
            # for some projections; extended here to M=64 - guarded, since tp_common
            # may not support every shape/M combination this way).
            try:
                sharded_mc, sharded_pc = dram_sharded_progcfg(K, N, M)
                sharded_weight = make_weight(ttnn, torch, device, K, N, sharded_mc, seed=1)
                timing = safe_time(x, sharded_weight, sharded_pc, ttnn.L1_MEMORY_CONFIG)
                ttnn.deallocate(sharded_weight)
            except Exception as error:  # noqa: BLE001
                timing = dict(error="%s: %s" % (type(error).__name__, error))
            record(M, act_name, "dram_sharded", False, None, timing)

            ttnn.deallocate(x)
    ttnn.deallocate(weight)


def main():
    global DRAM_BANDWIDTH_GBPS
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, required=True, help="JSON report path")
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--shapes", default=None,
                       help="comma-separated shape names to run (default: all seven)")
    parser.add_argument("--activations", default="l1,dram",
                       help="comma-separated activation placements: l1, dram")
    parser.add_argument("--bandwidth-gbps", type=float, default=DRAM_BANDWIDTH_GBPS)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--quick", action="store_true", default=True,
                      help="cap grids to ~24 configs per shape (default)")
    group.add_argument("--full", dest="quick", action="store_false",
                      help="every legal grid up to the 11x10 worker grid")
    args = parser.parse_args()
    DRAM_BANDWIDTH_GBPS = args.bandwidth_gbps

    import torch
    import ttnn

    dims = load_model_dims()
    shapes = shape_table(dims)
    if args.shapes:
        wanted = set(args.shapes.split(","))
        shapes = [shape for shape in shapes if shape["name"] in wanted]
        if not shapes:
            parser.error("no shape matched --shapes=%r" % args.shapes)
    shapes_by_name = {shape["name"]: shape for shape in shapes}
    activations = set(args.activations.split(","))

    report = dict(passed=False, complete=False, dims=dims, shapes=shapes, quick=args.quick,
                 bandwidth_gbps=DRAM_BANDWIDTH_GBPS, runs=[], summary={}, shape_errors={},
                 scope="Single-device, single-layer decode matmul timing; not a full-model or "
                       "TP2-collective measurement")

    def save():
        # Called after EVERY shape (success or failure) so a later crash - device-level,
        # or a bug in a shape this sweep has not hit yet - never loses shapes already
        # measured (run 35... on the rig lost four completed shapes this way: gdn_qkvz's
        # N=8240 crashed build_configs, and the only write_text call was in a single
        # top-level `finally`, which then itself failed with an unwritable /results).
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2, default=str))

    device = None
    try:
        device = ttnn.open_device(device_id=args.device_id, l1_small_size=24576)
        for shape in shapes:
            print("== sweeping %s (K=%d, N=%d) ==" % (shape["name"], shape["K"], shape["N"]), flush=True)
            try:
                run_shape(ttnn, torch, device, shape, args.quick, args.iters, args.warmup,
                         activations, report["runs"])
            except Exception as error:  # noqa: BLE001 - one shape's bug must not lose the rest
                message = "%s: %s" % (type(error).__name__, error)
                report["shape_errors"][shape["name"]] = message
                print("!! %s failed: %s" % (shape["name"], message), flush=True)
            report["summary"] = summarize(report["runs"], shapes_by_name)
            save()
        report["complete"] = True
        report["passed"] = not report["shape_errors"]
    except Exception as error:  # noqa: BLE001 - e.g. device open itself failing
        report["fatal_error"] = "%s: %s" % (type(error).__name__, error)
    finally:
        save()
        if device is not None:
            ttnn.close_device(device)

    print(format_table(report["summary"], shapes_by_name))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
