"""The M = 64 decode matmul grid sweep at four cards: the MLP gate and up (and optionally down) at their TP4 per-chip shapes (lever V7).

WHY. matmul64_sweep.py (TP2, MODEL_TP = 2) picked the packed 64-row verify block's seven 1D-mcast configs for two cards. At four
cards the same builder (tp_common.create_matmul_1d_decode_progcfg, num_cores 44 for up, 88 for gate under verify-trace T1 #11, grid_w
11) lands on half the columns: per-chip gate and up are K = 5,120, N = 17,408 / 4 = 4,352 (136 tiles), so the builder's per_core_N is
ceil(136 / 88) = 2 on 68 cores for the gate and ceil(136 / 44) = 4 on 34 cores for the up. The device profile of the packed trace
(v170) has the 68-core gate at 3.39 ms per 64 calls against the 34-core up's 2.82 ms: 8.8 us per call slower with twice the cores, and
neither is at the DRAM floor (gate 58%, up 70% of the measured 405 GB/s). This sweep times both over per_core_N, grid shape,
in0_block_w and output subblock, on one card, eager, and emits the best configs.

HOW. Single device, no mesh, no collectives: the per-chip matmul is the same program at any width (activations (64, 5,120) bfloat16 in
L1, the weight bfloat4_b for gate and up as served, bfloat8_b for down, in DRAM, the model's own LoFi compute kernel). Stage 1
tries every per_core_N that fits the worker grid on the model's own grid shape (11 wide) and block; stage 2 takes the best few
per_core_N and varies the grid width, in0_block_w and the output subblock. Each config is timed as a batch of back-to-back calls
(host dispatch is not the thing measured) and its output is compared byte for byte against the model's current config on the same
inputs: a partition-only change (same in0_block_w, any grid or per_core_N) is expected exact, an in0_block_w change may move the
accumulation and is reported exact or not as it measures. The best EXACT config per shape is the answer; the best overall is listed
beside it. The report line TP4_SWEEP names the builder arguments (num_cores, grid_w) that reproduce each best config.

Every pure function here (shape table, builder arithmetic, candidate generation, L1 estimate, ranking) imports and tests under plain
CPython (test_matmul_tp4_sweep); ttnn, torch and the model import only inside the device functions. matmul64_sweep supplies the
timing, weight and activation helpers it already proved on the rig.
"""

import argparse
import json
import math
import statistics
import time
from pathlib import Path

import matmul64_sweep as base
import tp_shapes

TP = 4
M = 64
TILE = 32
WORKER_GRID = (11, 10)
# bfloat4_b: 512 bytes of mantissa plus 64 of exponents per tile; bfloat8_b: 1,024 plus 64; bfloat16: 2,048.
TILE_BYTES = {'bfp4': 576, 'bfp8': 1088, 'bf16': 2048}
# One core's circular-buffer budget for one matmul: the allocatable L1 is about 1.43 MB; the sweep runs alone, so no other op's buffers
# share the core, but the model's op runs beside resident activations: stay at 1.2 MB, the prefetch audit's own budget.
L1_BUDGET = 1200000
# The model's builder arguments (model_config.py, T1 off for up and w2, T1 on for the gate): (name, num_cores).
MODEL_CORES = {'mlp_w1': 88, 'mlp_w3': 44, 'mlp_w2': 33}
STAGE1_BLOCK_CAP = 8
STAGE2_PER_CORE_N = 4
STAGE2_WIDTHS = (8, 10, 11)
SUBBLOCK_CAP = 4


def shapes(tp=TP):
    """The sweep's shapes at `tp` chips: (name, K, N, weight dtype key, fused silu). Gate and up are K = dim, N = mlp / tp, bfloat4_b as served;
    down is K = mlp / tp, N = dim, bfloat8_b. Read from tp_shapes (the one table), not repeated here."""
    found = tp_shapes.geometry(tp)
    return [dict(name='mlp_w1', K=tp_shapes.HIDDEN, N=found.mlp, dtype='bfp4', silu=True),
            dict(name='mlp_w3', K=tp_shapes.HIDDEN, N=found.mlp, dtype='bfp4', silu=False),
            dict(name='mlp_w2', K=found.mlp, N=tp_shapes.HIDDEN, dtype='bfp8', silu=False)]


def largest_divisor(n, cap=STAGE1_BLOCK_CAP):
    """tp_common._find_largest_divisor: the model's in0_block_w."""
    for d in range(cap, 0, -1):
        if n % d == 0:
            return d
    return 1


def builder_config(m, k, n, num_cores, grid_w):
    """tp_common.create_matmul_1d_decode_progcfg(m, k, n, num_cores=num_cores, grid_w=grid_w) as a plain dict (transcribed from the
    image's builder; test_matmul_tp4_sweep holds it against the graft's own transcription)."""
    cols = min(grid_w, num_cores)
    rows = math.ceil(num_cores / cols)
    m_tiles, k_tiles, n_tiles = math.ceil(m / TILE), math.ceil(k / TILE), math.ceil(n / TILE)
    per_core_n = math.ceil(n_tiles / (cols * rows))
    sub_w = max(i for i in range(1, SUBBLOCK_CAP + 1) if per_core_n % i == 0)
    sub_h = max(i for i in range(1, SUBBLOCK_CAP + 1) if m_tiles % i == 0 and i * sub_w <= SUBBLOCK_CAP)
    return dict(grid=(cols, rows), in0_block_w=largest_divisor(k_tiles), per_core_M=m_tiles, per_core_N=per_core_n,
                out_subblock_h=sub_h, out_subblock_w=sub_w)


def active_cores(config, n_tiles):
    """How many cores of the grid carry output columns: ceil(N tiles / per_core_N) (the rest of the grid idles)."""
    return math.ceil(n_tiles / config['per_core_N'])


def l1_bytes(config, dtype):
    """One core's circular buffers for the 1D mcast matmul (an estimate: double-buffered in0 and in1 blocks, the output, and the fp32
    partials the packer accumulates): in0 per_core_M x block bf16, in1 block x per_core_N weight tiles, out per_core_M x per_core_N."""
    pm, pn, blk = config['per_core_M'], config['per_core_N'], config['in0_block_w']
    in0 = pm * blk * 2 * TILE_BYTES['bf16']
    in1 = blk * pn * 2 * TILE_BYTES[dtype]
    out = pm * pn * TILE_BYTES['bf16']
    partials = pm * pn * 4096
    return in0 + in1 + out + partials


def subblocks(per_core_m, per_core_n, keep=2):
    """The best `keep` (h, w) output subblocks, largest product first, distinct in h (the packer's order differs between 1 x 4 and 2 x 2)."""
    out, seen_h = [], set()
    for h, w in base.out_subblocks(per_core_m, per_core_n, cap=SUBBLOCK_CAP):
        if h not in seen_h:
            seen_h.add(h)
            out.append((h, w))
        if len(out) == keep:
            break
    return out


def grid_for(cores, width):
    """The (width, rows) grid that holds `cores` cores `width` wide, or None when it does not fit the worker grid."""
    width = min(width, cores)
    rows = math.ceil(cores / width)
    return (width, rows) if width <= WORKER_GRID[0] and rows <= WORKER_GRID[1] else None


def per_core_n_values(n_tiles, max_cores=WORKER_GRID[0] * WORKER_GRID[1]):
    """Every per_core_N that fits the worker grid: ceil(n_tiles / per_core_N) cores, at most max_cores, at least 8 (below that a core
    holds a third of the output and the sweep has nothing to learn)."""
    values = []
    for pcn in range(1, n_tiles + 1):
        cores = math.ceil(n_tiles / pcn)
        if 8 <= cores <= max_cores:
            values.append(pcn)
    return values


def stage1(shape, m=M):
    """One config per per_core_N, on the model's grid shape (11 wide) with the model's in0_block_w and the best subblock."""
    n_tiles, k_tiles = base.tiles(shape['N']), base.tiles(shape['K'])
    block = largest_divisor(k_tiles)
    per_core_m = math.ceil(m / TILE)
    out = []
    for pcn in per_core_n_values(n_tiles):
        grid = grid_for(math.ceil(n_tiles / pcn), WORKER_GRID[0])
        if grid is None:
            continue
        h, w = subblocks(per_core_m, pcn, keep=1)[0]
        config = dict(grid=grid, in0_block_w=block, per_core_M=per_core_m, per_core_N=pcn, out_subblock_h=h, out_subblock_w=w)
        if l1_bytes(config, shape['dtype']) <= L1_BUDGET:
            out.append(config)
    return out


def block_choices(k_tiles, per_core_n, dtype, per_core_m=2):
    """in0_block_w values that divide K in tiles and fit one core's L1 budget at this per_core_N."""
    out = []
    for block in base.divisors(k_tiles):
        probe = dict(per_core_M=per_core_m, per_core_N=per_core_n, in0_block_w=block)
        if block <= 32 and l1_bytes(probe, dtype) <= L1_BUDGET:
            out.append(block)
    return out


def stage2(shape, best_per_core_n, m=M):
    """For each per_core_N in `best_per_core_n`: every grid width in STAGE2_WIDTHS x every fitting in0_block_w x the best two subblocks."""
    n_tiles, k_tiles = base.tiles(shape['N']), base.tiles(shape['K'])
    per_core_m = math.ceil(m / TILE)
    seen, out = set(), []
    for pcn in best_per_core_n:
        cores = math.ceil(n_tiles / pcn)
        for width in STAGE2_WIDTHS:
            grid = grid_for(cores, width)
            if grid is None:
                continue
            for block in block_choices(k_tiles, pcn, shape['dtype'], per_core_m):
                for h, w in subblocks(per_core_m, pcn):
                    config = dict(grid=grid, in0_block_w=block, per_core_M=per_core_m, per_core_N=pcn,
                                  out_subblock_h=h, out_subblock_w=w)
                    key = config_key(config)
                    if key not in seen:
                        seen.add(key)
                        out.append(config)
    return out


def config_key(config):
    return (tuple(config['grid']), config['in0_block_w'], config['per_core_N'], config['out_subblock_h'], config['out_subblock_w'])


def label(config):
    return 'grid_%dx%d_pcn%d_blk%d_sub%dx%d' % (config['grid'][0], config['grid'][1], config['per_core_N'], config['in0_block_w'],
                                              config['out_subblock_h'], config['out_subblock_w'])


def builder_arguments(config, k_tiles, n_tiles):
    """The (num_cores, grid_w) a tp_common.create_matmul_1d_decode_progcfg call needs to reproduce `config`'s partition, or None when the
    builder cannot (it takes the whole grid it names: per_core_N = ceil(N tiles / (cols x rows)), its own block and subblock)."""
    cols, rows = config['grid']
    built = builder_config(M, k_tiles * TILE, n_tiles * TILE, cols * rows, cols)
    return dict(num_cores=cols * rows, grid_w=cols, reproduces=built['per_core_N'] == config['per_core_N']
                and built['out_subblock_w'] == config['out_subblock_w'] and built['out_subblock_h'] == config['out_subblock_h']
                and built['in0_block_w'] == config['in0_block_w'])


def rank(rows):
    """Timed rows (dicts with 'config', 'us' or 'error', 'exact') sorted fastest first, errors last."""
    return sorted(rows, key=lambda row: (1, 0.0) if 'us' not in row else (0, row['us']))


def best(rows, current_us=None):
    """(best exact row, best row of any exactness) from a ranked list; None where there is none. `exact` must be True, not just present."""
    ranked = [row for row in rank(rows) if 'us' in row]
    exact = next((row for row in ranked if row.get('exact') is True), None)
    return exact, (ranked[0] if ranked else None)


def speedup(current_us, row):
    return None if row is None or not current_us else current_us / row['us']


def dram_floor_us(shape, bandwidth_gbps=None):
    """Microseconds to read the shape's padded weight once at the measured per-card rate (base.DRAM_BANDWIDTH_GBPS: 400 unless overridden)."""
    rate = base.DRAM_BANDWIDTH_GBPS if bandwidth_gbps is None else bandwidth_gbps
    tiles = base.tiles(shape['K']) * base.tiles(shape['N'])
    return tiles * TILE_BYTES[shape['dtype']] / (rate * 1e9) * 1e6


# ---------------------------------------------------------------------------------------------------------------------------
# Device parts (ttnn and torch imported here only).
# ---------------------------------------------------------------------------------------------------------------------------

def weight_dtype(ttnn, key):
    return {'bfp4': ttnn.bfloat4_b, 'bfp8': ttnn.bfloat8_b, 'bf16': ttnn.bfloat16}[key]


def make_weight(ttnn, torch, device, shape):
    torch.manual_seed(0)
    value = torch.randn(base.pad_to_tile(shape['K']), base.pad_to_tile(shape['N']), dtype=torch.bfloat16) * 0.02
    return ttnn.from_torch(value, device=device, dtype=weight_dtype(ttnn, shape['dtype']), layout=ttnn.TILE_LAYOUT,
                           memory_config=ttnn.DRAM_MEMORY_CONFIG)


def program(ttnn, config, silu):
    return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=tuple(config['grid']), in0_block_w=config['in0_block_w'],
        out_subblock_h=config['out_subblock_h'], out_subblock_w=config['out_subblock_w'], per_core_M=config['per_core_M'],
        per_core_N=config['per_core_N'], fuse_batch=True, fused_activation=ttnn.UnaryOpType.SILU if silu else None, mcast_in0=True)


def time_batch(ttnn, device, once, calls, rounds):
    """Per-call microseconds: `calls` back-to-back launches then one synchronize, `rounds` times; the median and the best round."""
    for _ in range(3):
        ttnn.deallocate(once())
    ttnn.synchronize_device(device)
    samples = []
    for _ in range(rounds):
        started = time.perf_counter()
        for _ in range(calls):
            ttnn.deallocate(once())
        ttnn.synchronize_device(device)
        samples.append((time.perf_counter() - started) * 1e6 / calls)
    return dict(us=statistics.median(samples), min_us=min(samples))


def run_shape(ttnn, torch, device, shape, calls, rounds, rows):
    """Time and byte-check every candidate of one shape; appends to `rows` in place (a failure keeps the rows measured so far)."""
    ckc = base.compute_kernel_config(ttnn)
    n_tiles = base.tiles(shape['N'])
    weight = make_weight(ttnn, torch, device, shape)
    x = base.make_activation(ttnn, torch, device, M, shape['K'], ttnn.L1_MEMORY_CONFIG)

    def run(config):
        pc = program(ttnn, config, shape['silu'])
        return lambda: ttnn.linear(x, weight, compute_kernel_config=ckc, program_config=pc, memory_config=ttnn.L1_MEMORY_CONFIG)

    def readback(config):
        out = run(config)()
        try:
            return ttnn.to_torch(out).clone()
        finally:
            ttnn.deallocate(out)

    current = builder_config(M, shape['K'], shape['N'], MODEL_CORES[shape['name']], device.compute_with_storage_grid_size().x)
    reference = readback(current)

    def measure(arm, config, is_current=False):
        row = dict(shape=shape['name'], arm=arm, config=config, is_model_current=is_current,
                   active_cores=active_cores(config, n_tiles), l1_bytes=l1_bytes(config, shape['dtype']))
        try:
            row.update(time_batch(ttnn, device, run(config), calls, rounds))
            row['exact'] = bool(torch.equal(readback(config), reference))
        except Exception as error:  # noqa: BLE001 - a config the program refuses is a row, not the end of the sweep
            row['error'] = '%s: %s' % (type(error).__name__, error)
        rows.append(row)
        return row

    measure('model_current', current, True)
    seen = {config_key(current)}
    first = []
    for config in stage1(shape):
        if config_key(config) not in seen:
            seen.add(config_key(config))
            first.append(measure(label(config), config))
    ranked = [row for row in rank(first) if 'us' in row]
    top = []
    for row in ranked:
        if row['config']['per_core_N'] not in top:
            top.append(row['config']['per_core_N'])
        if len(top) == STAGE2_PER_CORE_N:
            break
    for config in stage2(shape, top):
        if config_key(config) not in seen:
            seen.add(config_key(config))
            measure(label(config), config)
    ttnn.deallocate(x)
    ttnn.deallocate(weight)


def summary(rows, shape_list):
    """Per shape: the model's current row, the best exact row, the best row of any exactness, the speed-ups, and the DRAM floor."""
    out = {}
    for shape in shape_list:
        mine = [row for row in rows if row['shape'] == shape['name']]
        current = next((row for row in mine if row.get('is_model_current')), None)
        current_us = current.get('us') if current else None
        exact, anyone = best(mine)
        n_tiles, k_tiles = base.tiles(shape['N']), base.tiles(shape['K'])
        out[shape['name']] = dict(
            K=shape['K'], N=shape['N'], dtype=shape['dtype'], dram_floor_us=dram_floor_us(shape), current=current,
            best_exact=exact, best_any=anyone, speedup_exact=speedup(current_us, exact), speedup_any=speedup(current_us, anyone),
            builder_exact=builder_arguments(exact['config'], k_tiles, n_tiles) if exact else None,
            top=[dict(arm=row['arm'], us=row['us'], exact=row.get('exact'), cores=row['active_cores']) for row in rank(mine)[:8]
                 if 'us' in row])
    return out


def verdict_lines(result):
    lines = []
    for name, entry in result.items():
        current, exact = entry['current'], entry['best_exact']
        if not current or 'us' not in current or not exact:
            lines.append('TP4_SWEEP %s no result (current=%s best_exact=%s)' % (name, bool(current), bool(exact)))
            continue
        lines.append('TP4_SWEEP %s current %s %.1f us (%d cores) best_exact %s %.1f us (%d cores) speedup %.3fx floor %.1f us builder %s' % (
            name, label(current['config']), current['us'], current['active_cores'], label(exact['config']), exact['us'],
            exact['active_cores'], entry['speedup_exact'] or 0.0, entry['dram_floor_us'], json.dumps(entry['builder_exact'], sort_keys=True)))
    return lines


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--device-id', type=int, default=0)
    parser.add_argument('--calls', type=int, default=40, help='launches per timed round')
    parser.add_argument('--rounds', type=int, default=7)
    parser.add_argument('--shapes', default='mlp_w1,mlp_w3', help='comma-separated: mlp_w1 (gate), mlp_w3 (up), mlp_w2 (down)')
    parser.add_argument('--bandwidth-gbps', type=float, default=base.DRAM_BANDWIDTH_GBPS)
    args = parser.parse_args()
    base.DRAM_BANDWIDTH_GBPS = args.bandwidth_gbps
    wanted = args.shapes.split(',')
    shape_list = [shape for shape in shapes() if shape['name'] in wanted]
    if not shape_list:
        parser.error('no shape matched --shapes=%r' % args.shapes)

    import torch
    import ttnn

    report = dict(passed=False, complete=False, tp=TP, m=M, calls=args.calls, rounds=args.rounds, shapes=shape_list, runs=[], summary={},
                  shape_errors={}, scope='Single-device eager matmul timing at the TP4 per-chip shapes; not a model or collective measurement')

    def save():
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2, default=str))

    device = None
    try:
        device = ttnn.open_device(device_id=args.device_id, l1_small_size=24576)
        for shape in shape_list:
            print('== sweeping %s (K=%d, N=%d, %s) ==' % (shape['name'], shape['K'], shape['N'], shape['dtype']), flush=True)
            try:
                run_shape(ttnn, torch, device, shape, args.calls, args.rounds, report['runs'])
            except Exception as error:  # noqa: BLE001 - one shape's failure must not lose the others
                report['shape_errors'][shape['name']] = '%s: %s' % (type(error).__name__, error)
                print('!! %s failed: %s' % (shape['name'], report['shape_errors'][shape['name']]), flush=True)
            report['summary'] = summary(report['runs'], shape_list)
            save()
        report['complete'] = True
        report['passed'] = not report['shape_errors'] and all(entry['best_exact'] for entry in report['summary'].values())
    except Exception as error:  # noqa: BLE001 - the device failing to open
        report['fatal_error'] = '%s: %s' % (type(error).__name__, error)
    finally:
        save()
        if device is not None:
            ttnn.close_device(device)
    for line in verdict_lines(report['summary']):
        print(line, flush=True)
    if not report['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
