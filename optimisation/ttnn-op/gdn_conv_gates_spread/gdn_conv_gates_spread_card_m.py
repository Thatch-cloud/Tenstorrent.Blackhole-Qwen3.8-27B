"""F1 on ONE card: the conv-gates spread launch against the served gdn_decode_conv_gates, bit for bit, at 16, 64 and 128 rows, and timed.

Run with QWEN_FAST_TP=4 in the environment (the geometry is the four-card shard's: 2,560 channels, 12 value heads, a at column 4,096 and b at
4,108 of a 4,120-wide projection). One p150a, a 1x1 mesh, no model and no collective: the launch's program is per chip, so one chip proves the
kernels compile, run and move the same bytes; whether four chips of a mesh agree is the audited attach's job (A1, every chip).

For each case (rows = state rows = x rows = batch) and regime (random finite bf16; edge values: +-0, the smallest normal, denormals, the largest
finite, +-1) the SAME host data is uploaded twice. The served op runs on one copy, the spread launch (gdn_conv_gates_spread.launch, chips=1) on the
other; conv, beta, g and the four advanced windows are read back and compared as int16 bit patterns, the in-trace audit's rule (the runtime's bf16
upload and readback may flush denormals and turn NaN into Inf, but they do so for both arms alike). Timing: per case, eager, 24 launches then one
synchronise, 25 rounds in serpentine order (served, spread, spread, served, ...); the per-launch median and IQR of each arm. The timing also settles
how long a conv core takes: the spread launch's floor is the slower of its conv cores and its gate cores.

Exit: 0 PASS (every compare exact); 1 FAIL (any differing element, a launch that fell back); 3 the watchdog; 4 NOT-RUN (a section raised). The
last stdout line is one JSON object (kind f1-card-m) and the line above it 'GDN_CG_SPREAD verdict=...'.
"""

import argparse
import json
import os
import statistics
import sys
import threading
import time

KIND = 'f1-card-m'
CASES = (16, 64, 128)
REGIMES = ('random', 'edge')
LAUNCHES = 24
ROUNDS = 25
WATCHDOG_S = 2400
EDGE_BITS = (0x0000, 0x8000, 0x0080, 0x8080, 0x0001, 0x8001, 0x007F, 0x7F7F, 0xFF7F, 0x3F80, 0xBF80)


def host_data(torch, rows, regime, seed, qkv, width, heads):
    """The seeded host tensors of one case: x (1, rows, width), four windows (1, rows, qkv), four taps (1, 1, qkv), dt_bias and neg_exp_A
    (1, 1, heads), as bfloat16 built from int16 bit patterns (finite only: an infinity or NaN would only test the runtime's own flush)."""
    generator = torch.Generator().manual_seed(seed)

    def bits(shape):
        raw = torch.randint(0, 65536, shape, generator=generator, dtype=torch.int32)
        if regime == 'edge':
            # half the elements are one of the edge patterns, half random
            pick = torch.randint(0, len(EDGE_BITS), shape, generator=generator, dtype=torch.int32)
            edge = torch.tensor(EDGE_BITS, dtype=torch.int32)[pick]
            raw = torch.where(torch.rand(shape, generator=generator) < 0.5, edge, raw)
        raw = torch.where((raw & 0x7F80) == 0x7F80, (raw & 0x8000) | 0x3F80, raw)     # no infinity, no NaN
        return (raw - ((raw & 0x8000) << 1)).to(torch.int16)                            # the unsigned pattern as a signed int16

    def bf16(shape):
        return bits(shape).contiguous().view(torch.bfloat16)

    return dict(x=bf16((1, rows, width)), windows=[bf16((1, rows, qkv)) for _ in range(4)], taps=[bf16((1, 1, qkv)) for _ in range(4)],
                dt_bias=bf16((1, 1, heads)), neg_exp_A=bf16((1, 1, heads)))


def differing(torch, left, right):
    """Elements that differ between two int16 views of the same shape (a shape difference counts every element of the larger)."""
    if tuple(left.shape) != tuple(right.shape):
        return max(left.numel(), right.numel())
    return int((left != right).sum())


def summarize(values):
    ordered = sorted(values)
    quarter = lambda fraction: ordered[min(len(ordered) - 1, int(fraction * len(ordered)))]
    return dict(n=len(ordered), median_us=round(statistics.median(ordered), 2), q1_us=round(quarter(0.25), 2), q3_us=round(quarter(0.75), 2),
                min_us=round(ordered[0], 2))


def verdict(sections):
    """(PASS | FAIL | NOT-RUN, exit status) from the compare sections: every one exact and none fell back is a PASS."""
    if not sections or any(section.get('error') for section in sections):
        return 'NOT-RUN', 4
    if any(section.get('differing', 1) or section.get('fell_back') for section in sections):
        return 'FAIL', 1
    return 'PASS', 0


class Rig(object):
    """The device side of one case: upload, run each arm, read back. `ttnn` is the module (a fake in the tests)."""

    def __init__(self, ttnn, mesh, torch, found, spread):
        self.ttnn, self.mesh, self.torch, self.found, self.spread = ttnn, mesh, torch, found, spread

    def upload(self, host):
        ttnn, mesh = self.ttnn, self.mesh

        def up(tensor):
            return ttnn.from_torch(tensor, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh, memory_config=ttnn.DRAM_MEMORY_CONFIG,
                                   mesh_mapper=ttnn.ReplicateTensorToMesh(mesh))

        return dict(x=up(host['x']), windows=[up(value) for value in host['windows']], taps=[up(value) for value in host['taps']],
                    dt_bias=up(host['dt_bias']), neg_exp_A=up(host['neg_exp_A']))

    def served(self, tensors, rows):
        ttnn, found = self.ttnn, self.found
        return ttnn.transformer.gdn_decode_conv_gates(tensors['x'], tensors['windows'], tensors['taps'], tensors['x'], tensors['x'],
                                                      tensors['dt_bias'], tensors['neg_exp_A'], batch=rows, memory_config=ttnn.DRAM_MEMORY_CONFIG,
                                                      channels=found.gdn_qkv, a_col=found.gdn_a_col, b_col=found.gdn_b_col)

    def spreaded(self, tensors, rows):
        found = self.found
        return self.spread.launch(self.ttnn, self.mesh, tensors['x'], tensors['windows'], tensors['taps'], tensors['dt_bias'],
                                  tensors['neg_exp_A'], rows, found.gdn_qkv, found.gdn_a_col, found.gdn_b_col, chips=1)

    def read(self, tensor):
        ttnn, torch = self.ttnn, self.torch
        return ttnn.to_torch(ttnn.get_device_tensors(tensor)[0]).contiguous().view(torch.int16)


def compare_case(rig, host, rows):
    """One compare section: the served op on one upload, the spread launch on another, seven tensors each, bit for bit."""
    served_in, spread_in = rig.upload(host), rig.upload(host)
    served = rig.served(served_in, rows)
    spread = rig.spreaded(spread_in, rows)
    if spread is None:
        return dict(rows=rows, differing=-1, fell_back=True)
    labels = ('conv', 'beta', 'g', 'window0', 'window1', 'window2', 'window3')
    pairs = list(zip(labels, [*served, *served_in['windows']], [*spread, *spread_in['windows']]))
    counts = {label: differing(rig.torch, rig.read(left), rig.read(right)) for label, left, right in pairs}
    return dict(rows=rows, differing=sum(counts.values()), by_tensor=counts, fell_back=False)


def time_case(rig, host, rows, launches=LAUNCHES, rounds=ROUNDS, clock=time.perf_counter):
    """Per-launch microseconds of the served op and the spread launch over serpentine rounds."""
    served_in, spread_in = rig.upload(host), rig.upload(host)
    arms = dict(served=lambda: rig.served(served_in, rows), spread=lambda: rig.spreaded(spread_in, rows))
    samples = dict(served=[], spread=[])
    for name in arms:
        arms[name]()
    rig.ttnn.synchronize_device(rig.mesh)
    for index in range(rounds):
        order = ('served', 'spread') if index % 2 == 0 else ('spread', 'served')
        for name in order:
            started = clock()
            for _ in range(launches):
                arms[name]()
            rig.ttnn.synchronize_device(rig.mesh)
            samples[name].append((clock() - started) / launches * 1e6)
    served, spread = summarize(samples['served']), summarize(samples['spread'])
    return dict(rows=rows, served=served, spread=spread, spread_minus_served_us=round(spread['median_us'] - served['median_us'], 2))


def main(argv=None, torch=None, ttnn=None, spread=None, tp_shapes=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--out', required=True)
    parser.add_argument('--cases', default=','.join(str(value) for value in CASES))
    parser.add_argument('--regimes', default=','.join(REGIMES))
    parser.add_argument('--seeds', default='17,23')
    parser.add_argument('--timing', choices=('on', 'off'), default='on')
    options = parser.parse_args(argv)
    cases = [int(value) for value in options.cases.split(',')]
    regimes = options.regimes.split(',')
    seeds = [int(value) for value in options.seeds.split(',')]
    if any(value not in (16, 32, 64, 96, 128) for value in cases) or any(value not in REGIMES for value in regimes):
        print('refusing: cases are 16 32 64 96 128, regimes random edge', file=sys.stderr)
        return 2
    timer = threading.Timer(WATCHDOG_S, lambda: (print('GDN_CG_SPREAD watchdog', flush=True), os._exit(3)))
    timer.daemon = True
    timer.start()
    if torch is None:
        import torch
        import ttnn
        import gdn_conv_gates_spread as spread
        import tp_shapes
    report = dict(kind=KIND, cases=cases, regimes=regimes, seeds=seeds, environment=dict(QWEN_FAST_TP=os.environ.get('QWEN_FAST_TP')))
    mesh, status, sections = None, 4, []
    try:
        if os.environ.get('QWEN_FAST_TP') != '4':
            raise RuntimeError('QWEN_FAST_TP=4 required (the geometry is the four-card shard\'s)')
        found = tp_shapes.active()
        report['geometry'] = dict(channels=found.gdn_qkv, heads=found.gdn_nv, a_col=found.gdn_a_col, b_col=found.gdn_b_col, width=found.gdn_qkvzab)
        grid = None
        mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1))
        grid = mesh.compute_with_storage_grid_size()
        report['grid'] = [grid.x, grid.y]
        report['plans'] = {str(rows): {key: value for key, value in spread.plan(grid.x, grid.y, rows, rows, found.gdn_qkv, found.gdn_nv).items()
                                       if key != 'work'} for rows in cases}
        report['new_kernels_sha256'] = spread.source_sha256()
        rig = Rig(ttnn, mesh, torch, found, spread)
        for rows in cases:
            for regime in regimes:
                for seed in seeds:
                    host = host_data(torch, rows, regime, seed, found.gdn_qkv, found.gdn_qkvzab, found.gdn_nv)
                    try:
                        section = compare_case(rig, host, rows)
                    except BaseException as error:  # noqa: BLE001
                        section = dict(rows=rows, error='%s: %s' % (error.__class__.__name__, str(error)[:400]))
                    section.update(regime=regime, seed=seed)
                    sections.append(section)
                    print('GDN_CG_SPREAD compare rows=%d regime=%s seed=%d differing=%s fell_back=%s' % (
                        rows, regime, seed, section.get('differing', section.get('error')), section.get('fell_back')), flush=True)
        report['compare'] = sections
        text, status = verdict(sections)
        if options.timing == 'on' and text == 'PASS':
            report['timing'] = []
            for rows in cases:
                host = host_data(torch, rows, 'random', seeds[0], found.gdn_qkv, found.gdn_qkvzab, found.gdn_nv)
                result = time_case(rig, host, rows)
                report['timing'].append(result)
                print('GDN_CG_SPREAD timing rows=%d served_us=%s spread_us=%s delta_us=%s' % (
                    rows, result['served']['median_us'], result['spread']['median_us'], result['spread_minus_served_us']), flush=True)
        report['verdict'] = text
        print('GDN_CG_SPREAD verdict=%s sections=%d differing=%d' % (text, len(sections), sum(max(section.get('differing', 0), 0) for section in sections)),
              flush=True)
    except BaseException as error:  # noqa: BLE001
        report['verdict'] = 'NOT-RUN'
        report['error'] = '%s: %s' % (error.__class__.__name__, str(error)[:500])
        print('GDN_CG_SPREAD verdict=NOT-RUN error=%s' % report['error'], flush=True)
        status = 4
    finally:
        if mesh is not None:
            try:
                ttnn.close_mesh_device(mesh)
            except BaseException:  # noqa: BLE001
                pass
    with open(options.out, 'w') as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
    print(json.dumps(report, sort_keys=True), flush=True)
    timer.cancel()
    return status


if __name__ == '__main__':
    sys.exit(main())
