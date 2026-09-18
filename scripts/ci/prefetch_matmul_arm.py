"""Checkpoint 4 arm: run prefetch_and_linear on Qwen MLP shapes and check it against ttnn.linear.

Correctness first. Timings printed here are indicative only — a verdict needs
interleaved whole-cycle arms with host load recorded, not a microbenchmark.

Geometry is forced by three independent constraints: the receiver count must divide
n_tiles (integer per_core_N), must equal the area of a rectangle fitting the 11x10
worker grid, and must be divisible by the sender bank count. For gate/up
(n_tiles=272=2^4x17) that caps receivers at 16; for down (n_tiles=160) it reaches 80.
"""

import argparse
import json
import sys
import time
import traceback

BEGIN = '<<<PREFETCH_ARM_JSON_BEGIN>>>'
END = '<<<PREFETCH_ARM_JSON_END>>>'
TILE = 32
GRID_X, GRID_Y = 11, 10
PROJECTIONS = {'gate': (5120, 8704, 'bfloat4_b'),
               'up': (5120, 8704, 'bfloat4_b'),
               'down': (8704, 5120, 'bfloat8_b')}


def rectangles(area):
    return [(x, y) for x in range(1, GRID_X + 1) for y in range(1, GRID_Y + 1) if x * y == area]


def best_geometry(k_tiles, n_tiles, max_banks=8):
    best = None
    for receivers in range(1, GRID_X * GRID_Y + 1):
        if n_tiles % receivers:
            continue
        rects = rectangles(receivers)
        if not rects:
            continue
        banks = max([b for b in range(1, max_banks + 1) if receivers % b == 0])
        best = dict(receivers=receivers, grid=rects[0], banks=banks,
                    recv_per_bank=receivers // banks, per_core_N=n_tiles // receivers)
    return best


def run_one(ttnn, common, torch, mesh, name, rows, dtype_name, report):
    inner, width, native_dtype = PROJECTIONS[name]
    k_tiles, n_tiles = inner // TILE, width // TILE
    geom = best_geometry(k_tiles, n_tiles)
    entry = dict(projection=name, dtype=dtype_name, native_dtype=native_dtype,
                 inner=inner, width=width, geometry=geom)
    report['arms'].append(entry)
    if geom is None:
        entry['stopped_at'] = 'no legal geometry'
        return
    dtype = getattr(ttnn, dtype_name)
    # in0_block_w must divide k_tiles; mcast-in0 uses block_count = k_tiles / in0_block_w.
    in0_block_w = next((b for b in (8, 5, 4, 2, 1) if k_tiles % b == 0), 1)
    entry['in0_block_w'] = in0_block_w
    entry['block_count'] = k_tiles // in0_block_w
    try:
        torch.manual_seed(0)
        pt_weight = torch.randn(1, 1, inner, width, dtype=torch.bfloat16)
        pt_in0 = torch.randn(1, 1, rows, inner, dtype=torch.bfloat16)
        weight = common.make_recv_contig_weight(
            mesh, pt_weight, geom['banks'], geom['receivers'], dtype)
        entry['weight_built'] = True
        ring_cols = common.ring_grid_cols(geom['banks'], geom['receivers'])
        bank_to_receivers = [(b, common.bank_receivers_strided(
            b, geom['recv_per_bank'], geom['banks'], ring_cols)) for b in range(geom['banks'])]
        page_bytes = (k_tiles // entry['block_count']) * geom['per_core_N'] * common.bytes_per_tile(dtype)
        entry['page_bytes'] = page_bytes
        global_cb = ttnn.experimental.create_global_circular_buffer_for_tensor_prefetcher(
            mesh, bank_to_receivers, page_bytes * 4, ttnn.BufferType.L1)
        entry['gcb_built'] = True
        program_config = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(geom['grid'][0], geom['grid'][1]),
            in0_block_w=in0_block_w, out_subblock_h=1, out_subblock_w=1,
            per_core_M=max(1, rows // TILE), per_core_N=geom['per_core_N'],
            fuse_batch=True, fused_activation=None, mcast_in0=True)
        entry['program_config_built'] = True
        in0 = ttnn.from_torch(pt_in0, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh)
        # Control: the same 1D matmul on the same grid, weights read from DRAM the
        # ordinary way. Isolates what the prefetcher itself contributes.
        plain_weight = ttnn.from_torch(pt_weight, dtype=dtype, layout=ttnn.TILE_LAYOUT,
                                       device=mesh)
        expected = (pt_in0.float() @ pt_weight.float())

        def timed(call, iterations):
            call()
            ttnn.synchronize_device(mesh)
            samples = []
            for _ in range(iterations):
                start = time.perf_counter()
                result = call()
                ttnn.synchronize_device(mesh)
                samples.append(time.perf_counter() - start)
                ttnn.deallocate(result)
            samples.sort()
            return samples[len(samples) // 2]

        with common.tensor_prefetcher_session(mesh):
            out = ttnn.experimental.tensor_prefetcher_matmul.prefetch_and_linear(
                in0, weight, global_cb=global_cb, program_config=program_config)
            entry['matmul_ran'] = True
            got = ttnn.to_torch(out)
            passed, message = common.comp_pcc(expected, got.float(), 0.97)
            entry['pcc_passed'] = bool(passed)
            entry['pcc_message'] = str(message)[:200]
            ttnn.deallocate(out)
            if entry['pcc_passed']:
                # Interleave the arms rather than blocking them: host contention on
                # this rig inflates host-dispatch-bound phases far more than
                # device-bound ones, so blocked arms can mislead badly.
                prefetch_ms, plain_ms = [], []
                for _ in range(3):
                    prefetch_ms.append(1000.0 * timed(
                        lambda: ttnn.experimental.tensor_prefetcher_matmul.prefetch_and_linear(
                            in0, weight, global_cb=global_cb,
                            program_config=program_config), 10))
                    plain_ms.append(1000.0 * timed(
                        lambda: ttnn.linear(in0, plain_weight,
                                            program_config=program_config), 10))
                prefetch_ms.sort()
                plain_ms.sort()
                entry['prefetch_ms_median'] = round(prefetch_ms[1], 4)
                entry['plain_ms_median'] = round(plain_ms[1], 4)
                entry['prefetch_ms_all'] = [round(v, 4) for v in prefetch_ms]
                entry['plain_ms_all'] = [round(v, 4) for v in plain_ms]
                entry['ratio_prefetch_over_plain'] = round(
                    prefetch_ms[1] / plain_ms[1], 4) if plain_ms[1] else None
                entry['timing_is_indicative_only'] = True
    except BaseException:
        entry['error'] = traceback.format_exc(limit=8)[-2200:]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rows', type=int, default=32)
    parser.add_argument('--projections', default='gate,down')
    parser.add_argument('--dtypes', default='bfloat4_b,bfloat8_b')
    options = parser.parse_args()
    report = dict(scope=__doc__, arms=[], speedup_claimed=False)
    try:
        sys.path.insert(0, '/opt/tt-metal')
        import time
        import torch
        import ttnn
        from tests.ttnn.unit_tests.operations import prefetcher_common as common
        # A cluster open straight after another process released the cards can hit
        # "Setting power state failed ... Input/output error" from the ARC. Give the
        # device a moment and retry rather than reporting a false negative.
        mesh = None
        report['open_attempts'] = []
        for attempt in range(1, 4):
            try:
                mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 2), l1_small_size=24576)
                report['open_attempts'].append(dict(attempt=attempt, ok=True))
                break
            except BaseException as error:
                report['open_attempts'].append(dict(attempt=attempt, ok=False,
                                                    error=str(error)[:300]))
                if attempt == 3:
                    raise
                time.sleep(15)
        try:
            report['supported'] = ttnn.experimental.is_tensor_prefetcher_supported(mesh)
            for name in options.projections.split(','):
                native = PROJECTIONS[name][2]
                for dtype_name in options.dtypes.split(','):
                    if dtype_name != native and dtype_name != 'bfloat8_b':
                        continue
                    run_one(ttnn, common, torch, mesh, name, options.rows, dtype_name, report)
        finally:
            ttnn.close_mesh_device(mesh)
    except BaseException:
        report['fatal'] = traceback.format_exc(limit=8)[-2000:]
    print(BEGIN)
    print(json.dumps(report, indent=2))
    print(END)


if __name__ == '__main__':
    main()
