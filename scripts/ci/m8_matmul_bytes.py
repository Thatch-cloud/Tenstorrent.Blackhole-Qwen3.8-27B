"""B1 of the one-block verify (tp4/m8-phase1): the 128-row matmul byte compare on one card (card M, the cardm step), every projection at its FOUR-CARD
per-chip shape, eager, no mesh and no collectives.

WHAT IT PROVES. At 128 rows the seven 1D-mcast decode matmuls (attn_qkv, attn_wo, gdn_qkvzab, gdn_out, mlp w1, w3, w2) and the LM head run as ONE
call instead of two 64-row calls. That is exact only if each 32-row tile of the 128-row output carries the bits it has at 64 rows and at 32. For
every projection and every config variant ('served': the build the serving recipe runs, with verify-trace T1 #11's wider attn_qkv and gate; 'base':
model_config's plain `_64` build) the same random activation is run

  * once at 128 rows with the config built at M = 128 (per_core_M 4, the same builder, the same in0_block_w: m8_matmul_plan holds that on the CPU),
  * as the two 64-row halves with the config built at M = 64 (what the two-block verify runs today), and
  * as its four 32-row tiles with the config built at M = 32 (the sequential engine's tile),

and every 32-row tile of the 128-row output is compared BIT FOR BIT (int16 view, so a NaN or a -0 counts) with the half and the tile that cover the same
rows. The LM head is `ttnn.linear(x, w)` with no program config (the model's call: ttnn picks the config from M, the design's one real unknown), so
there the two sides may differ in K blocking and the answer is whatever ttnn does; a mismatch is the finding that makes phase 2 pin a config.

TIMING. Each (projection, variant) is also timed at 32, 64 and 128 rows (back-to-back launches, one synchronise: host dispatch is not measured). The
review's pass rule per projection is t128 <= t64 + 1.5 x (t64 - t32), or the design's t128 <= 1.1 x t64; and sum t128 <= 1.1 x sum t64 over the seven.
The timing verdicts are reported, never gating: exactness gates `passed`.

RE-GRID. For --regrid projections (default the MLP's three) every per_core_N candidate at per_core_M 4 (matmul_tp4_sweep.stage1: the model's in0_block_w, so
the K order is kept) is byte-compared against the 64-row halves' bits and timed: the best EXACT one is the answer, with the builder arguments that
reproduce it. NOTHING IS APPLIED by this job: the result is the input of the graft's M = 128 section (phase 2).

The orchestration (run_projection) is written against a Backend and holds on the CPU against a fake one (test_m8_matmul_plan); DeviceBackend is the ttnn
and torch layer, imported only on the card. Stdlib only at import, py 3.7.
"""

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import m8_matmul_plan as plan

SEED = 0
WEIGHT_SCALE = 0.02
INPUT_SCALE = 1.0


# ---------------------------------------------------------------------------------------------------------------------------
# Orchestration (pure: against a Backend).
# ---------------------------------------------------------------------------------------------------------------------------

def run_projection(backend, entry, variant, calls, rounds, timing=True):
    """One (projection, variant): the 128-row output against the 64-row halves and the 32-row tiles, tile by tile, and the three timings.

    `variant` None is the LM head (auto config). Returns a dict: tiles_vs_halves / tiles_vs_tiles (mismatching elements per 32-row tile, 0 = exact),
    exact_halves / exact_tiles, the timings and the plan's own verdicts. Never raises for a refused config: the caller records the exception text."""
    lm = variant is None
    configs = {} if lm else {m: plan.config_at(entry, variant, m) for m in plan.TIMED_ROWS}
    result = dict(name=entry['name'], variant='auto' if lm else variant, k=entry['k'], n=entry['n'], dtype=entry['dtype'],
                  configs=configs)
    if not lm:
        result['k_order_problem'] = plan.k_order_problem(configs[plan.HALF], configs[plan.BLOCK])
        result['k_order_problem_tile'] = plan.k_order_problem(configs[plan.TILE], configs[plan.BLOCK])
        result['subblocks'] = plan.sub_block_changes(configs[plan.HALF], configs[plan.BLOCK])
    weight = backend.weight(entry)
    handles = []
    try:
        def run(first, last, m):
            x = backend.activation(entry, first, last)
            handles.append(x)
            out = backend.linear(entry, x, weight, None if lm else configs[m])
            host = backend.read(out)
            backend.free(out)
            return host

        whole = run(0, plan.BLOCK, plan.BLOCK)
        halves = [run(first, last, plan.HALF) for first, last in plan.halves()]
        tiles = [run(first, first + plan.TILE, plan.TILE) for first in range(0, plan.BLOCK, plan.TILE)]
        vs_halves, vs_tiles = [], []
        for tile in range(plan.BLOCK // plan.UNIT):
            first, last = plan.rows_of_tile(tile)
            half = 0 if first < plan.HALF else 1
            offset = first - half * plan.HALF
            vs_halves.append(backend.mismatches(whole, first, last, halves[half], offset, 'half %d' % half))
            vs_tiles.append(backend.mismatches(whole, first, last, tiles[tile], 0, 'tile %d' % tile))
        result.update(tiles_vs_halves=vs_halves, tiles_vs_tiles=vs_tiles, exact_halves=not any(vs_halves), exact_tiles=not any(vs_tiles))
        if timing:
            times = {}
            for m in plan.TIMED_ROWS:
                x = backend.activation(entry, 0, m)
                handles.append(x)
                times[m] = backend.time(entry, x, weight, None if lm else configs[m], calls, rounds)
            result.update(t32_us=times[plan.TILE], t64_us=times[plan.HALF], t128_us=times[plan.BLOCK])
            result['timing'] = plan.timing_verdict(times[plan.TILE], times[plan.HALF], times[plan.BLOCK])
        return result
    finally:
        for handle in handles:
            backend.free(handle)
        backend.free(weight)


def run_regrid(backend, entry, calls, rounds, limit):
    """Every re-grid candidate of `entry` at 128 rows, byte-compared with the 64-row halves of the model's served config and timed. Returns the rows
    (a refused config is a row with an `error`) and the best exact one."""
    served = {m: plan.config_at(entry, 'served', m) for m in (plan.HALF, plan.BLOCK)}
    weight = backend.weight(entry)
    handles = []
    rows = []
    try:
        x_whole = backend.activation(entry, 0, plan.BLOCK)
        handles.append(x_whole)
        oracle = []
        for first, last in plan.halves():
            x = backend.activation(entry, first, last)
            handles.append(x)
            out = backend.linear(entry, x, weight, served[plan.HALF])
            oracle.append(backend.read(out))
            backend.free(out)
        for config in plan.regrid_candidates(entry)[:limit]:
            row = dict(entry=entry['name'], config=config, arguments=plan.builder_arguments(config, entry), l1_bytes=None)
            try:
                out = backend.linear(entry, x_whole, weight, config)
                host = backend.read(out)
                backend.free(out)
                counts = []
                for tile in range(plan.BLOCK // plan.UNIT):
                    first, last = plan.rows_of_tile(tile)
                    half = 0 if first < plan.HALF else 1
                    counts.append(backend.mismatches(host, first, last, oracle[half], first - half * plan.HALF, 'half %d' % half))
                row.update(tiles_vs_halves=counts, exact=not any(counts), us=backend.time(entry, x_whole, weight, config, calls, rounds))
            except Exception as error:  # noqa: BLE001 - a config the program refuses is a row, not the end of the sweep
                row['error'] = '%s: %s' % (type(error).__name__, error)
            rows.append(row)
    finally:
        for handle in handles:
            backend.free(handle)
        backend.free(weight)
    timed = sorted((row for row in rows if 'us' in row), key=lambda row: row['us'])
    best_exact = next((row for row in timed if row.get('exact') is True), None)
    return rows, best_exact


def passed(report):
    """True when the run completed, nothing errored, and every (projection, variant) is exact against the halves AND the tiles."""
    if not report.get('complete') or report.get('errors') or report.get('fatal_error'):
        return False
    results = report.get('results', [])
    return bool(results) and all(row.get('exact_halves') is True and row.get('exact_tiles') is True for row in results)


def verdict_lines(report):
    """One M8_MATMUL line per (projection, variant), then the regrid winners and the overall verdict."""
    lines = []
    for row in report.get('results', []):
        timing = row.get('timing') or {}
        lines.append('M8_MATMUL %s/%s exact_halves=%s exact_tiles=%s mismatches_vs_halves=%s mismatches_vs_tiles=%s t32=%s t64=%s t128=%s timing_ok=%s ratio=%s' % (
            row['name'], row['variant'], row.get('exact_halves'), row.get('exact_tiles'), row.get('tiles_vs_halves'), row.get('tiles_vs_tiles'),
            _us(row.get('t32_us')), _us(row.get('t64_us')), _us(row.get('t128_us')), timing.get('ok'),
            '%.3f' % timing['ratio'] if timing.get('ratio') else None))
    for name, entry in sorted(report.get('regrid', {}).items()):
        best = entry.get('best_exact')
        lines.append('M8_REGRID %s candidates=%d exact=%d best_exact=%s' % (
            name, len(entry.get('rows', [])), sum(1 for row in entry.get('rows', []) if row.get('exact') is True),
            None if best is None else '%s %.1f us builder %s' % (_label(best['config']), best['us'], json.dumps(best['arguments'], sort_keys=True))))
    lines.append('M8_MATMUL timing_total_ok=%s sum_t64=%s sum_t128=%s' % (report.get('timing_total_ok'), _us(report.get('sum_t64_us')),
                                                                         _us(report.get('sum_t128_us'))))
    lines.append('M8_MATMUL verdict=%s' % ('PASS' if passed(report) else 'FAIL'))
    return lines


def _us(value):
    return None if value is None else '%.1f' % value


def _label(config):
    return 'grid_%dx%d_pcn%d_blk%d_sub%dx%d' % (config['grid'][0], config['grid'][1], config['per_core_N'], config['in0_block_w'],
                                              config['out_subblock_h'], config['out_subblock_w'])


def summarize(report):
    """Fill the report's totals from its results."""
    times = {}
    for row in report.get('results', []):
        if row['variant'] in ('served', 'auto') and row['name'] != 'lm_head':
            times[row['name']] = (row.get('t32_us'), row.get('t64_us'), row.get('t128_us'))
    report['timing_total_ok'] = plan.total_timing_verdict(times)
    complete = [value for value in times.values() if None not in value]
    report['sum_t64_us'] = sum(value[1] for value in complete) if complete else None
    report['sum_t128_us'] = sum(value[2] for value in complete) if complete else None
    report['passed'] = passed(report)
    return report


# ---------------------------------------------------------------------------------------------------------------------------
# The device layer (ttnn and torch imported here only).
# ---------------------------------------------------------------------------------------------------------------------------

class DeviceBackend(object):
    """One card, eager. Activations are seeded per projection and uploaded per row range (so every call sees exact input bits); outputs are read back to
    host torch tensors; mismatches compares int16 views."""

    def __init__(self, ttnn, torch, device):
        self.ttnn, self.torch, self.device = ttnn, torch, device
        self.hosts = {}
        self.compute = {}

    def weight_dtype(self, key):
        return {'bfp4': self.ttnn.bfloat4_b, 'bfp8': self.ttnn.bfloat8_b, 'bf16': self.ttnn.bfloat16}[key]

    def host_input(self, entry):
        if entry['name'] not in self.hosts:
            self.torch.manual_seed(SEED + 1000 + sum(map(ord, entry['name'])))
            self.hosts[entry['name']] = (self.torch.randn(1, 1, plan.BLOCK, entry['k'], dtype=self.torch.bfloat16) * INPUT_SCALE)
        return self.hosts[entry['name']]

    def weight(self, entry):
        self.torch.manual_seed(SEED + sum(map(ord, entry['name'])))
        value = self.torch.randn(entry['k'], entry['n'], dtype=self.torch.bfloat16) * WEIGHT_SCALE
        return self.ttnn.from_torch(value, device=self.device, dtype=self.weight_dtype(entry['dtype']), layout=self.ttnn.TILE_LAYOUT,
                                    memory_config=self.ttnn.DRAM_MEMORY_CONFIG)

    def activation(self, entry, first, last):
        memory = self.ttnn.DRAM_MEMORY_CONFIG if entry['name'] == 'lm_head' else self.ttnn.L1_MEMORY_CONFIG
        return self.ttnn.from_torch(self.host_input(entry)[:, :, first:last, :], device=self.device, dtype=self.ttnn.bfloat16,
                                    layout=self.ttnn.TILE_LAYOUT, memory_config=memory)

    def compute_config(self, key):
        if key in self.compute:
            return self.compute[key]
        ttnn = self.ttnn
        config = None
        if key == 'lofi':
            config = ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.LoFi, fp32_dest_acc_en=True, packer_l1_acc=True)
        elif key == 'hifi2':
            try:
                from models.demos.blackhole.qwen36.tt import tp_common
                config = tp_common.COMPUTE_HIFI2
            except Exception:  # noqa: BLE001 - the image's own constant when it imports, the same fields otherwise
                config = ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.HiFi2, math_approx_mode=False,
                                                          fp32_dest_acc_en=True, packer_l1_acc=True)
        self.compute[key] = config
        return config

    def program(self, entry, config):
        ttnn = self.ttnn
        return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=tuple(config['grid']), in0_block_w=config['in0_block_w'], out_subblock_h=config['out_subblock_h'],
            out_subblock_w=config['out_subblock_w'], per_core_M=config['per_core_M'], per_core_N=config['per_core_N'], fuse_batch=True,
            fused_activation=ttnn.UnaryOpType.SILU if entry['silu'] else None, mcast_in0=True)

    def linear(self, entry, x, weight, config):
        ttnn = self.ttnn
        if config is None:     # the LM head: the model's call, no program config, no compute config
            return ttnn.linear(x, weight, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        return ttnn.linear(x, weight, compute_kernel_config=self.compute_config(entry['compute']), program_config=self.program(entry, config),
                           memory_config=ttnn.DRAM_MEMORY_CONFIG)

    def read(self, out):
        return self.ttnn.to_torch(out).clone()

    def mismatches(self, whole, first, last, other, other_first, label):
        torch = self.torch
        mine = whole[..., first:last, :].contiguous().view(torch.int16)
        theirs = other[..., other_first:other_first + (last - first), :].contiguous().view(torch.int16)
        if mine.shape != theirs.shape:
            raise ValueError('%s: shapes %r and %r differ' % (label, tuple(mine.shape), tuple(theirs.shape)))
        return int((mine != theirs).sum())

    def time(self, entry, x, weight, config, calls, rounds):
        ttnn = self.ttnn

        def once():
            return self.linear(entry, x, weight, config)

        for _ in range(3):
            ttnn.deallocate(once())
        ttnn.synchronize_device(self.device)
        samples = []
        for _ in range(rounds):
            started = time.perf_counter()
            for _ in range(calls):
                ttnn.deallocate(once())
            ttnn.synchronize_device(self.device)
            samples.append((time.perf_counter() - started) * 1e6 / calls)
        return statistics.median(samples)

    def free(self, handle):
        try:
            self.ttnn.deallocate(handle)
        except Exception:  # noqa: BLE001 - a host tensor or an already freed one
            pass


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--device-id', type=int, default=0)
    parser.add_argument('--calls', type=int, default=40, help='launches per timed round')
    parser.add_argument('--rounds', type=int, default=7)
    parser.add_argument('--projections', default=','.join(entry['name'] for entry in plan.projections()),
                        help='comma-separated: attn_qkv attn_wo gdn_qkvzab gdn_out mlp_w1 mlp_w3 mlp_w2 lm_head')
    parser.add_argument('--variants', default='served,base', help='config variants of the seven: served (T1 #11 attn_qkv and gate), base (the _64 build)')
    parser.add_argument('--regrid', default='mlp_w1,mlp_w3,mlp_w2', help='projections whose per_core_N candidates at per_core_M 4 are swept ("" for none)')
    parser.add_argument('--regrid-max', type=int, default=16)
    parser.add_argument('--no-timing', action='store_true')
    parser.add_argument('--l1-small-size', type=int, default=24576)
    args = parser.parse_args(argv)
    known = {entry['name'] for entry in plan.projections()}
    args.projection_list = [name for name in args.projections.split(',') if name]
    args.variant_list = [name for name in args.variants.split(',') if name]
    args.regrid_list = [name for name in args.regrid.split(',') if name]
    for name in args.projection_list + args.regrid_list:
        if name not in known:
            parser.error('unknown projection %r' % name)
    for name in args.variant_list:
        if name not in ('served', 'base'):
            parser.error('unknown variant %r' % name)
    return args


def main(argv=None):
    args = parse_arguments(argv)
    import torch
    import ttnn

    report = dict(passed=False, complete=False, tp=plan.TP, rows=list(plan.TIMED_ROWS), calls=args.calls, rounds=args.rounds, results=[], regrid={},
                  errors={}, scope='Single-device eager matmul at the TP4 per-chip shapes: the 128-row output against the 64-row halves and the '
                                   '32-row tiles, bit for bit; not a model or collective measurement')

    def save():
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(summarize(report), indent=2, default=str))

    device = None
    try:
        device = ttnn.open_device(device_id=args.device_id, l1_small_size=args.l1_small_size)
        backend = DeviceBackend(ttnn, torch, device)
        for name in args.projection_list:
            entry = plan.named(name)
            variants = [None] if name == 'lm_head' else args.variant_list
            for variant in variants:
                key = '%s/%s' % (name, variant or 'auto')
                print('== %s ==' % key, flush=True)
                try:
                    report['results'].append(run_projection(backend, entry, variant, args.calls, args.rounds, timing=not args.no_timing))
                except Exception as error:  # noqa: BLE001 - one projection's failure must not lose the others
                    report['errors'][key] = '%s: %s' % (type(error).__name__, error)
                    print('!! %s failed: %s' % (key, report['errors'][key]), flush=True)
                save()
        for name in args.regrid_list:
            print('== regrid %s ==' % name, flush=True)
            try:
                rows, best = run_regrid(backend, plan.named(name), args.calls, args.rounds, args.regrid_max)
                report['regrid'][name] = dict(rows=rows, best_exact=best)
            except Exception as error:  # noqa: BLE001
                report['errors']['regrid/%s' % name] = '%s: %s' % (type(error).__name__, error)
            save()
        report['complete'] = True
    except Exception as error:  # noqa: BLE001 - the device failing to open
        report['fatal_error'] = '%s: %s' % (type(error).__name__, error)
    finally:
        save()
        if device is not None:
            ttnn.close_device(device)
    for line in verdict_lines(report):
        print(line, flush=True)
    if not report['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
