"""Measure achieved DRAM bandwidth per p150a card. Measurement only, no claim about TG.

docs/200tps-reachability-2026-09-19.md derives every latency floor from a 512 GB/s
spec figure that has never been measured on this rig, and says so. Every floor scales
inversely with it, so this is the single number that decides whether the 4-user target
is 17 ms short or something else entirely.

Two regimes, because they answer different questions:

  stream  a large DRAM tensor copied DRAM->DRAM. Traffic is read + write, so this is
          total memory bandwidth, and it is the easy case for the memory system.
  weights a decode-shaped matvec [1, K] x [K, N]. Arithmetic intensity is about two
          FLOP per weight byte, so it is read-dominated and bandwidth-bound: the
          closest proxy for a decode weight pass, which is what the budget is made of.

Weights are replicated across the mesh, so each device does identical work and the
per-device figure is the per-card bandwidth. Sizes are swept because a single point
cannot distinguish a saturated memory system from a launch-bound one.

Every measurement is guarded: an API that is not present in this build is recorded as
unavailable rather than failing the run, since one hardware slot should return as many
numbers as it can.
"""

import argparse
import json
import os
import time
from pathlib import Path

BEGIN = '<<<DRAM_BW_JSON_BEGIN>>>'
END = '<<<DRAM_BW_JSON_END>>>'
SPEC_GB_S = 512.0


def timed(fn, iters, sync):
    """Median-of-three wall time for `iters` calls, with the device drained each time."""
    samples = []
    for _ in range(3):
        sync()
        started = time.perf_counter()
        for _ in range(iters):
            fn()
        sync()
        samples.append(time.perf_counter() - started)
    return sorted(samples)[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--iters', type=int, default=30)
    parser.add_argument('--output', type=Path)
    options = parser.parse_args()

    report = dict(scope=__doc__, spec_gb_s=SPEC_GB_S, iters=options.iters, results=[])
    import torch

    from feature_projection import require_projection_environment
    require_projection_environment(os.environ, True)
    import ttnn

    ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 2), l1_small_size=24576)
    try:
        rep = ttnn.ReplicateTensorToMesh(mesh)

        def sync():
            ttnn.synchronize_device(mesh)

        def dram(t):
            return ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                                   device=mesh, memory_config=ttnn.DRAM_MEMORY_CONFIG,
                                   mesh_mapper=rep)

        for side in (4096, 8192, 16384):
            entry = dict(regime='stream', side=side, dtype='bfloat16')
            try:
                elems = side * side
                src = dram(torch.zeros(1, 1, side, side, dtype=torch.bfloat16))
                out = ttnn.clone(src)
                sync()
                elapsed = timed(lambda: ttnn.copy(src, out), options.iters, sync)
                moved = 2 * elems * 2 * options.iters
                entry.update(bytes_per_iter=2 * elems * 2, seconds=round(elapsed, 6),
                             gb_s=round(moved / elapsed / 1e9, 1))
                entry['percent_of_spec'] = round(100 * entry['gb_s'] / SPEC_GB_S, 1)
                ttnn.deallocate(out)
                ttnn.deallocate(src)
            except BaseException as error:
                entry['error'] = '%s: %s' % (type(error).__name__, str(error)[:200])
            report['results'].append(entry)

        for k, n in ((4096, 4096), (8192, 8192), (5120, 8704), (8704, 5120)):
            entry = dict(regime='weights', k=k, n=n, dtype='bfloat16')
            try:
                w = dram(torch.zeros(1, 1, k, n, dtype=torch.bfloat16))
                x = dram(torch.zeros(1, 1, 32, k, dtype=torch.bfloat16))
                y = ttnn.matmul(x, w)
                sync()
                elapsed = timed(lambda: ttnn.matmul(x, w), options.iters, sync)
                weight_bytes = k * n * 2
                entry.update(bytes_per_iter=weight_bytes, seconds=round(elapsed, 6),
                             gb_s=round(weight_bytes * options.iters / elapsed / 1e9, 1))
                entry['percent_of_spec'] = round(100 * entry['gb_s'] / SPEC_GB_S, 1)
                ttnn.deallocate(y)
                ttnn.deallocate(x)
                ttnn.deallocate(w)
            except BaseException as error:
                entry['error'] = '%s: %s' % (type(error).__name__, str(error)[:200])
            report['results'].append(entry)

        good = [r for r in report['results'] if r.get('gb_s')]
        if good:
            best = max(r['gb_s'] for r in good)
            weights = [r['gb_s'] for r in report['results']
                       if r.get('regime') == 'weights' and r.get('gb_s')]
            report['best_gb_s'] = best
            report['best_percent_of_spec'] = round(100 * best / SPEC_GB_S, 1)
            if weights:
                report['weight_stream_gb_s'] = max(weights)
                report['weight_stream_percent_of_spec'] = round(
                    100 * max(weights) / SPEC_GB_S, 1)
    except BaseException as error:
        report['fatal'] = '%s: %s' % (type(error).__name__, str(error)[:400])
    finally:
        try:
            ttnn.close_mesh_device(mesh)
        except BaseException:
            pass

    print(BEGIN)
    print(json.dumps(report, indent=2))
    print(END)
    if options.output:
        try:
            options.output.write_text(json.dumps(report, indent=2))
        except OSError:
            pass
    return 0 if report.get('best_gb_s') else 1


if __name__ == '__main__':
    raise SystemExit(main())
