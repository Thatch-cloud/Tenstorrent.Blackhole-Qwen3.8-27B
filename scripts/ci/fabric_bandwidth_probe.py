"""Measure achieved inter-card collective bandwidth. Measurement only, no TG claim.

The prefill device profile puts collectives at 37% of device time, against 17% for
arithmetic, and the traffic implied by the all-gathers works out near 9.3 GB/s effective.
DRAM on the same cards measures 405 GB/s, so the question is whether 9.3 is close to what
the link can do or far below it. The answer decides the remedy:

  near the link's limit  -> collectives are link-bound, and only a sharding change helps,
                            which costs decode because TP2 exists to halve its weight pass
  far below it           -> collectives are overhead-bound, and fewer or larger ones win
                            with no sharding change and no decode penalty

Sweeps size because one point cannot separate a saturated link from a latency-bound one:
a fixed per-collective cost shows up as rising GB/s with size, while a saturated link is
flat. Each measurement is guarded so a missing API is recorded rather than costing the
whole hardware slot.
"""

import argparse
import json
import os
import time
from pathlib import Path

BEGIN = '<<<FABRIC_BW_JSON_BEGIN>>>'
END = '<<<FABRIC_BW_JSON_END>>>'


def timed(fn, iters, sync):
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
    parser.add_argument('--iters', type=int, default=20)
    parser.add_argument('--output', type=Path)
    options = parser.parse_args()

    report = dict(scope=__doc__, iters=options.iters, results=[])
    import torch

    from feature_projection import require_projection_environment
    require_projection_environment(os.environ, True)
    import ttnn

    # Collectives need the fabric up, unlike the DRAM probe which disables it.
    for name in ('FABRIC_1D', 'FABRIC_1D_RING'):
        config = getattr(ttnn.FabricConfig, name, None)
        if config is not None:
            ttnn.set_fabric_config(config)
            report['fabric'] = name
            break
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 2), l1_small_size=24576)
    try:
        def sync():
            ttnn.synchronize_device(mesh)

        shard = ttnn.ShardTensor2dMesh(mesh, dims=(None, -1), mesh_shape=(1, 2))
        gather = (getattr(ttnn, 'all_gather', None)
                  or getattr(getattr(ttnn, 'experimental', None), 'all_gather_async', None))
        report['all_gather_api'] = gather.__name__ if gather else None

        # Shapes chosen to bracket a prefill activation: 2048 tokens x 5120 hidden bf16.
        for rows, cols in ((512, 5120), (2048, 5120), (8192, 5120)):
            entry = dict(rows=rows, cols=cols, dtype='bfloat16')
            try:
                host = torch.zeros(1, 1, rows, cols, dtype=torch.bfloat16)
                sharded = ttnn.from_torch(host, dtype=ttnn.bfloat16,
                                          layout=ttnn.TILE_LAYOUT, device=mesh,
                                          memory_config=ttnn.DRAM_MEMORY_CONFIG,
                                          mesh_mapper=shard)
                out = gather(sharded, dim=3)
                sync()
                elapsed = timed(lambda: gather(sharded, dim=3), options.iters, sync)
                # each device receives the other's half
                moved = rows * (cols // 2) * 2 * options.iters
                entry.update(bytes_per_call=rows * (cols // 2) * 2,
                             seconds=round(elapsed, 6),
                             gb_s=round(moved / elapsed / 1e9, 2),
                             us_per_call=round(1e6 * elapsed / options.iters, 1))
                ttnn.deallocate(out)
                ttnn.deallocate(sharded)
            except BaseException as error:
                entry['error'] = '%s: %s' % (type(error).__name__, str(error)[:200])
            report['results'].append(entry)
            print('all_gather %5dx%d -> %s GB/s' % (rows, cols, entry.get('gb_s')), flush=True)

        good = [r for r in report['results'] if r.get('gb_s')]
        if good:
            report['best_gb_s'] = max(r['gb_s'] for r in good)
            report['implied_in_prefill_gb_s'] = 9.3
            # 800 Gb/s of link is 100 GB/s. The implied 9.3 GB/s in prefill is about 9%
            # of that, so the expectation going in is overhead-bound rather than
            # link-bound; this measurement is what settles it.
            report['link_spec_gb_s'] = 100.0
            report['percent_of_link_spec'] = round(100 * report['best_gb_s'] / 100.0, 1)
            report['prefill_percent_of_link_spec'] = round(100 * 9.3 / 100.0, 1)
            if len(good) > 1:
                small, large = good[0], good[-1]
                report['scales_with_size'] = round(large['gb_s'] / small['gb_s'], 2)
                report['reading'] = ('latency-bound, fewer or larger collectives should win'
                                     if large['gb_s'] > small['gb_s'] * 1.5
                                     else 'close to saturated, only a sharding change helps')
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
