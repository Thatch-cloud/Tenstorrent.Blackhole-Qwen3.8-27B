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

        # Shapes bracket a prefill activation (2048 tokens x 5120 hidden bf16), and the
        # link count is swept because the runtime uses num_links=4 throughout while
        # ttnn.all_gather defaults to one. A first pass at one link measured 83 GB/s,
        # which is 83% of a single 100 GB/s port, so the question is whether four links
        # scale it and whether production is already getting that.
        for rows, cols, links in ((512, 5120, 1), (2048, 5120, 1), (8192, 5120, 1),
                                  (2048, 5120, 2), (2048, 5120, 4),
                                  (8192, 5120, 4)):
            entry = dict(rows=rows, cols=cols, links=links, dtype='bfloat16')
            try:
                host = torch.zeros(1, 1, rows, cols, dtype=torch.bfloat16)
                sharded = ttnn.from_torch(host, dtype=ttnn.bfloat16,
                                          layout=ttnn.TILE_LAYOUT, device=mesh,
                                          memory_config=ttnn.DRAM_MEMORY_CONFIG,
                                          mesh_mapper=shard)
                def call(t=sharded, n=links):
                    try:
                        return gather(t, dim=3, num_links=n)
                    except TypeError:
                        return gather(t, dim=3)
                out = call()
                sync()
                elapsed = timed(call, options.iters, sync)
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
            print('all_gather %5dx%d links=%d -> %s GB/s'
                  % (rows, cols, links, entry.get('gb_s')), flush=True)

        good = [r for r in report['results'] if r.get('gb_s')]
        if good:
            report['best_gb_s'] = max(r['gb_s'] for r in good)
            report['implied_in_prefill_gb_s'] = 9.3
            # QSFP-DD is 800 Gb/s per port, which is 100 GB/s, and this pair runs four
            # links (num_links=4 throughout the runtime), so 400 GB/s aggregate in
            # theory. The 9.3 GB/s implied by the prefill profile is about 2% of that,
            # which is very hard to explain as a saturated link. Note the 9.3 rests on
            # an estimate of all-gather traffic rather than a measurement; this probe
            # measures the achievable rate directly, and the gap between the two is the
            # point of running it.
            report['link_spec_gb_s_per_port'] = 100.0
            report['links'] = 4
            report['link_spec_gb_s_aggregate'] = 400.0
            report['percent_of_link_aggregate'] = round(100 * report['best_gb_s'] / 400.0, 2)
            report['prefill_percent_of_link_aggregate'] = round(100 * 9.3 / 400.0, 2)
            one = [r for r in good if r.get('links') == 1]
            if len(one) > 1:
                report['scales_with_size'] = round(one[-1]['gb_s'] / one[0]['gb_s'], 2)
            # Does adding links add bandwidth? Compare the same shape across link counts.
            same = {r['links']: r['gb_s'] for r in good if r.get('rows') == 2048}
            if len(same) > 1:
                report['by_links_2048x5120'] = same
                base = same.get(1)
                if base:
                    report['link_scaling'] = {k: round(v / base, 2) for k, v in sorted(same.items())}
            report['reading'] = (
                'links scale: production num_links=4 should already get the multiple'
                if report.get('link_scaling', {}).get(4, 1) > 1.5 else
                'extra links do not add bandwidth for this shape')
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
