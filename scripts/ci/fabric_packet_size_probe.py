"""Does raising the fabric packet payload size recover the missing bandwidth?

Four ethernet links are up between the pair (run 35425330364) and the user
confirms two QSFP-DD800 cables, so the ceiling is 200 GB/s. Measured all-gather
is 83.74 GB/s, 41.9% of it.

ccl_common.cpp says why that might be. Blackhole allows a 15232 B packet payload
and scatter-write caps a packet at 4 chunks, so for the 2048 B page of a bf16
32x32 tile the ideal packet is 4 x 2048 = 8192 B. The configured 4352 B fits only
two pages and wastes 256 B on top. Two pages per packet where four should fit is
a factor of two, and a factor of two is what the measurement is missing.

The knob is `FabricContext::validate_and_apply_packet_size`, fed from
`router_config_.max_packet_payload_size_bytes`, and ttnn exposes it:

    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D,
                           router_config=ttnn.FabricRouterConfig(...))

THE CONTROL IS THE POINT. Three levers this month were inert - a default never
overridden, a deprecated argument accepted and ignored, a code path never
entered - and each produced arms agreeing to within 2%, which reads as a
physical result. Here `get_tt_fabric_max_payload_size_bytes()` reports the value
the runtime actually adopted, so the arm can state whether the lever moved
instead of inviting the reader to assume it did.

Prediction, recorded before the run: if packing is the whole story, four pages
per packet against two should take 83.74 GB/s towards 160-170 GB/s. If the
readback changes to 8192 and bandwidth does not move, packing was not the
constraint and the remaining suspect is how many of the four links the
collective uses.
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


def configure_fabric(ttnn, report, packet_size, num_planes=0):
    """Bring the fabric up, optionally requesting a packet payload size."""
    config = None
    for name in ('FABRIC_1D', 'FABRIC_1D_RING'):
        config = getattr(ttnn.FabricConfig, name, None)
        if config is not None:
            report['fabric'] = name
            break
    if config is None:
        report['fabric'] = 'ABSENT'
        return

    extra = {}
    if num_planes:
        # set_fabric_config takes num_planes: int | None. With four links up and
        # only about one cable of bandwidth arriving, how many parallel routing
        # planes the fabric builds is the leading suspect.
        extra['num_planes'] = num_planes
        report['requested_num_planes'] = num_planes

    if not packet_size:
        ttnn.set_fabric_config(config, **extra)
        return

    # FabricRouterConfig carries exactly one field. Try the constructor first,
    # then attribute assignment, and record which worked rather than assuming.
    builder = getattr(ttnn, 'FabricRouterConfig', None)
    if builder is None:
        report['router_config'] = 'ABSENT'
        ttnn.set_fabric_config(config)
        return
    router = None
    try:
        router = builder(max_packet_payload_size_bytes=packet_size)
        report['router_config'] = 'constructor kwarg'
    except BaseException:
        try:
            router = builder()
            router.max_packet_payload_size_bytes = packet_size
            report['router_config'] = 'attribute assignment'
        except BaseException as error:
            report['router_config'] = '%s: %s' % (type(error).__name__, str(error)[:200])
    if router is None:
        ttnn.set_fabric_config(config, **extra)
        return
    ttnn.set_fabric_config(config, router_config=router, **extra)
    report['requested_packet_size'] = packet_size


def measure(ttnn, mesh, gather, sync, rows, cols, iters):
    """All-gather a column-sharded tensor and time the steady state."""
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

    moved = rows * (cols // 2) * 2 * iters
    return {'rows': rows, 'cols': cols,
            'bytes_per_call': rows * (cols // 2) * 2,
            'gb_s': round(moved / elapsed / 1e9, 2),
            'us_per_call': round(1e6 * elapsed / iters, 1)}


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--iters', type=int, default=30)
    parser.add_argument('--packet-size', type=int, default=0,
                        help='request this payload size; 0 leaves the default')
    parser.add_argument('--num-planes', type=int, default=0,
                        help='request this many parallel routing planes; 0 leaves the default')
    parser.add_argument('--json')
    options = parser.parse_args()

    report = {'baseline_gb_s': BASELINE_GB_S, 'ceiling_gb_s': CEILING_GB_S,
              'links': 4, 'cables': 2,
              'arm': ('planes' if options.num_planes else
                      'tuned' if options.packet_size else 'baseline'),
              'page_size_bytes': 2048, 'ideal_packet_size_bytes': 8192,
              'hw_max_payload_blackhole': 15232}
    try:
        from feature_projection import require_projection_environment
        require_projection_environment(os.environ, True)
        import ttnn

        configure_fabric(ttnn, report, options.packet_size, options.num_planes)

        mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 2), l1_small_size=24576)
        try:
            def sync():
                ttnn.synchronize_device(mesh)

            gather = (getattr(ttnn, 'all_gather', None)
                      or getattr(getattr(ttnn, 'experimental', None), 'all_gather_async', None))
            report['all_gather_api'] = gather.__name__ if gather else None

            # The control. Readable only once the fabric context exists.
            for key, name in (('packet_size_actual', 'get_tt_fabric_max_payload_size_bytes'),
                              ('packet_header_bytes', 'get_tt_fabric_packet_header_size_bytes')):
                accessor = getattr(ttnn, name, None)
                try:
                    report[key] = accessor() if accessor else 'ABSENT'
                except BaseException as error:
                    report[key] = '%s: %s' % (type(error).__name__, str(error)[:120])

            actual = report.get('packet_size_actual')
            if options.packet_size and isinstance(actual, int):
                report['lever_moved'] = (actual == options.packet_size)
                if not report['lever_moved']:
                    report['INVALID'] = (
                        'requested %d B but the runtime reports %s B, so this arm says '
                        'nothing about packing' % (options.packet_size, actual))
            if isinstance(actual, int) and actual:
                report['pages_per_packet'] = actual // 2048

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
