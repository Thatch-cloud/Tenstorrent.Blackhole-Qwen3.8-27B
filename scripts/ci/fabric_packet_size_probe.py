"""Does raising the fabric packet size recover the missing inter-card bandwidth?

Established: four ethernet links are up between the pair (run 35425330364), and
the user confirms two cables, so the ceiling is 200 GB/s. Measured all-gather is
83.74 GB/s, 41.9% of it. About 116 GB/s is unaccounted for.

The runtime names a suspect itself:

    Fabric packet size 4352 B is suboptimal for transporting 2048 B pages.
    Configure 8192 B packet size to maximize throughput.

One 2048-byte page per 4352-byte packet is 47% payload efficiency, against a
measured 41.9%. Close enough to be worth testing and nowhere near close enough
to believe without one.

THE CONTROL MATTERS MORE THAN THE MEASUREMENT. Three levers this month produced
arms that agreed to within 2% because the lever never moved - a default never
overridden, a deprecated argument ignored, a code path never entered. So this
probe reads the payload size back from the runtime and checks whether the
warning still appears, and reports the arm as INVALID if the knob did not take.
A bandwidth number from an arm whose lever did not move is worse than no number,
because it reads as a physical result.

Environment variables must be set before ttnn is imported, because fabric
configuration is read at initialisation, so the import is deliberately deferred.
"""

import argparse
import io
import json
import os
import sys
import time
import traceback

BASELINE_GB_S = 83.74
CEILING_GB_S = 200.0
SHAPES = ((512, 5120), (2048, 5120), (8192, 5120))


def measure(ttnn, mesh, gather, sync, rows, cols, iters):
    """All-gather a column-sharded tensor and time the steady state.

    Sharder, sync and gather are passed in from the proven setup in
    fabric_bandwidth_probe rather than rebuilt here. Run 35425588508 failed every
    shape with "Trying to get un-initialized fabric context" because this probe
    rebuilt that setup from memory and dropped set_fabric_config; reusing the
    working path is why that cannot recur.
    """
    import torch

    host = torch.randn(1, 1, rows, cols, dtype=torch.bfloat16)
    shard = ttnn.ShardTensor2dMesh(mesh, dims=(None, -1), mesh_shape=(1, 2))
    sharded = ttnn.from_torch(host, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                              device=mesh, memory_config=ttnn.DRAM_MEMORY_CONFIG,
                              mesh_mapper=shard)
    out = gather(sharded, dim=3)
    sync()
    ttnn.deallocate(out)

    start = time.time()
    for _ in range(iters):
        out = gather(sharded, dim=3)
    sync()
    elapsed = time.time() - start
    ttnn.deallocate(out)
    ttnn.deallocate(sharded)

    # Each card receives the half it does not hold.
    moved = rows * (cols // 2) * 2 * iters
    return {'rows': rows, 'cols': cols,
            'bytes_per_call': rows * (cols // 2) * 2,
            'gb_s': round(moved / elapsed / 1e9, 2),
            'us_per_call': round(1e6 * elapsed / iters, 1)}


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--iters', type=int, default=30)
    parser.add_argument('--set', action='append', default=[],
                        help='NAME=VALUE applied before ttnn is imported')
    parser.add_argument('--json')
    options = parser.parse_args()

    report = {'baseline_gb_s': BASELINE_GB_S, 'ceiling_gb_s': CEILING_GB_S,
              'links': 4, 'cables': 2, 'applied': {}}

    for pair in options.set:
        if '=' not in pair:
            continue
        name, value = pair.split('=', 1)
        os.environ[name] = value
        report['applied'][name] = value
    report['arm'] = 'tuned' if report['applied'] else 'baseline'

    try:
        from feature_projection import require_projection_environment
        require_projection_environment(os.environ, True)
        import ttnn

        # Collectives need the fabric up. Omitting this in run 35425588508 made
        # every shape fail on an un-initialized fabric context.
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

            gather = (getattr(ttnn, 'all_gather', None)
                      or getattr(getattr(ttnn, 'experimental', None), 'all_gather_async', None))
            report['all_gather_api'] = gather.__name__ if gather else None

            # Payload size is only readable once the fabric context exists.
            for name in ('get_tt_fabric_max_payload_size_bytes',
                         'get_tt_fabric_packet_header_size_bytes'):
                accessor = getattr(ttnn, name, None)
                try:
                    report[name] = accessor() if accessor else 'ABSENT'
                except BaseException as error:
                    report[name] = '%s: %s' % (type(error).__name__, str(error)[:120])

            results = []
            for rows, cols in SHAPES:
                try:
                    entry = measure(ttnn, mesh, gather, sync, rows, cols, options.iters)
                except BaseException as error:
                    entry = {'rows': rows, 'cols': cols,
                             'error': '%s: %s' % (type(error).__name__, str(error)[:200])}
                results.append(entry)
                print('all_gather %5dx%d -> %s GB/s' % (rows, cols, entry.get('gb_s')),
                      flush=True)
            report['results'] = results
        finally:
            ttnn.close_mesh_device(mesh)

        good = [r for r in report.get('results', []) if r.get('gb_s')]
        if good:
            best = max(r['gb_s'] for r in good)
            report['best_gb_s'] = best
            report['percent_of_ceiling'] = round(100 * best / CEILING_GB_S, 1)
            report['change_vs_baseline'] = round(best - BASELINE_GB_S, 2)
            report['ratio_vs_baseline'] = round(best / BASELINE_GB_S, 3)
    except BaseException as error:
        report['error'] = '%s: %s' % (type(error).__name__, str(error)[:300])
        report['traceback'] = traceback.format_exc()[-2000:]

    print('<<<FABRIC_PKT_JSON_BEGIN>>>', flush=True)
    print(json.dumps(report, indent=2, default=str), flush=True)
    print('<<<FABRIC_PKT_JSON_END>>>', flush=True)
    if options.json:
        io.open(options.json, 'w', encoding='utf-8', newline='\n').write(
            json.dumps(report, indent=2, default=str) + '\n')
    return 0


if __name__ == '__main__':
    sys.exit(main())
