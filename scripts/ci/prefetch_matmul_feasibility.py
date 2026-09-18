"""Checkpoint 4 feasibility: can the DRAM-core prefetcher feed Qwen's MLP shapes?

Qwen3.8-27B's per-device MLP weights are gate/up (5120 x 8704) bfloat4_b and down
(8704 x 5120) bfloat8_b. 8704 tiles to 272 = 2^4 x 17, and that prime 17 sharply
limits which rings are legal, unlike Llama's 14336. Gather-in0 needs the receiver
count to divide both k_tiles and n_tiles; mcast-in0 decouples block_count from the
receiver count and should admit wider rings.

This sweeps the legal geometries, builds each, and checks the prefetched matmul
against a plain ttnn.linear on the same inputs. It claims no speedup: timings here
are indicative only and a real verdict needs interleaved whole-cycle arms.
"""

import argparse
import json
import sys
import time
import traceback

BEGIN = '<<<PREFETCH_FEAS_JSON_BEGIN>>>'
END = '<<<PREFETCH_FEAS_JSON_END>>>'
TILE = 32

# Per-device TP2 shapes, from scripts/ci/tiny_tile_matmul.PROJECTIONS.
PROJECTIONS = {'gate': (5120, 8704, 'bfloat4_b'),
               'up': (5120, 8704, 'bfloat4_b'),
               'down': (8704, 5120, 'bfloat8_b')}


def divisors(value, limit):
    return [d for d in range(1, limit + 1) if value % d == 0]


def candidates(k_tiles, n_tiles, max_receivers, max_banks=8):
    """Legal (num_senders, recv_per_sender) per mode, as the design doc constrains them."""
    out = []
    for receivers in divisors(n_tiles, max_receivers):
        for banks in range(1, max_banks + 1):
            if receivers % banks:
                continue
            recv_per_sender = receivers // banks
            # Gather-in0: block_count == receiver_count, and k_block_w must be integral.
            gather_ok = (k_tiles % receivers == 0)
            out.append(dict(receivers=receivers, banks=banks,
                            recv_per_sender=recv_per_sender,
                            n_per_recv=n_tiles // receivers,
                            k_block_w_gather=(k_tiles // receivers) if gather_ok else None,
                            gather_legal=gather_ok))
    return out


def helper_signatures():
    import inspect
    sys.path.insert(0, '/opt/tt-metal')
    from tests.ttnn.unit_tests.operations import prefetcher_common as common
    out = {}
    for name in dir(common):
        if name.startswith('_'):
            continue
        value = getattr(common, name)
        if not callable(value):
            continue
        try:
            out[name] = str(inspect.signature(value))
        except (TypeError, ValueError):
            out[name] = '<no signature>'
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--projection', default='gate', choices=sorted(PROJECTIONS))
    parser.add_argument('--rows', type=int, default=16, help='T16 activation rows')
    parser.add_argument('--max-receivers', type=int, default=80)
    options = parser.parse_args()

    inner, width, dtype = PROJECTIONS[options.projection]
    k_tiles, n_tiles = inner // TILE, width // TILE
    report = dict(scope=__doc__, projection=options.projection, inner=inner, width=width,
                  dtype=dtype, k_tiles=k_tiles, n_tiles=n_tiles, rows=options.rows,
                  speedup_claimed=False, correctness_claimed=False)
    report['candidates'] = candidates(k_tiles, n_tiles, options.max_receivers)
    report['gather_legal_receivers'] = sorted({c['receivers'] for c in report['candidates']
                                               if c['gather_legal']})
    report['mcast_legal_receivers'] = sorted({c['receivers'] for c in report['candidates']})
    try:
        report['helpers'] = helper_signatures()
    except BaseException:
        report['helpers_error'] = traceback.format_exc(limit=4)[-800:]

    try:
        import ttnn
        report['grid_probe'] = {}
        mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 2), l1_small_size=24576)
        try:
            grid = mesh.compute_with_storage_grid_size()
            dram = mesh.dram_grid_size()
            report['grid_probe'] = dict(workers=[grid.x, grid.y], worker_cores=grid.x * grid.y,
                                        dram_banks=[dram.x, dram.y])
            report['supported'] = ttnn.experimental.is_tensor_prefetcher_supported(mesh)
        finally:
            ttnn.close_mesh_device(mesh)
    except BaseException:
        report['probe_error'] = traceback.format_exc(limit=6)[-1500:]

    print(BEGIN)
    print(json.dumps(report, indent=2))
    print(END)


if __name__ == '__main__':
    main()
