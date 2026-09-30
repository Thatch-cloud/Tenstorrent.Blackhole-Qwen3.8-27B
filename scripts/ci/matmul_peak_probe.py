"""Measure achieved matmul FLOPS per card, so MFU stops resting on a spec sheet.

Three derived percentages on this rig have now been wrong because their denominator
was assumed rather than measured: the 400 GB/s fabric aggregate (really one cable at
84), the FLOP requirement taken from the model's name (25.36 B counted, not 27), and
prefill MFU at 62-67%, which implies 127% per-core efficiency once the profiler's own
occupancy is applied and is therefore impossible. That last one closed arithmetic as
an optimisation target this morning and has had to be withdrawn.

This measures the denominator directly.

WHAT PEAK MEANS HERE. Not a marketing number: the best sustained rate this device
reaches on a large matmul, at each math fidelity, for the dtype pair the model runs
(bfloat8_b weights, bfloat16 activations, per model_config.py). Fidelity is swept
because model_config.py carries no explicit MathFidelity and the rate depends on it -
LoFi, HiFi2 and HiFi4 differ by the number of passes.

BEING COMPUTE-BOUND IS THE WHOLE POINT. A matmul that is memory-bound measures DRAM,
not arithmetic. Arithmetic intensity is 2MNK / bytes-moved; at M=N=K=8192 bfloat16
that is ~2700 FLOP/byte against a measured 405 GB/s, so DRAM could only supply about
1.1 PFLOP/s - comfortably above any plausible answer, meaning the matmul is compute
bound and the number is real. The probe reports the intensity so that stays checkable
rather than asserted.

Both cards run the same replicated work, so the figure is per card.
"""

import argparse
import io
import json
import os
import time
import traceback

BEGIN = '<<<MATMUL_PEAK_JSON_BEGIN>>>'
END = '<<<MATMUL_PEAK_JSON_END>>>'
DRAM_GB_S = 405.0


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--iters', type=int, default=12)
    parser.add_argument('--json')
    options = parser.parse_args()

    report = {'iters': options.iters, 'dram_gb_s': DRAM_GB_S, 'results': []}
    try:
        import torch

        from feature_projection import require_projection_environment
        require_projection_environment(os.environ, True)
        import ttnn

        mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 2), l1_small_size=24576)
        try:
            def sync():
                ttnn.synchronize_device(mesh)

            replicate = ttnn.ReplicateTensorToMesh(mesh)

            def dev(t, dtype):
                return ttnn.from_torch(t.to(torch.bfloat16), dtype=dtype,
                                       layout=ttnn.TILE_LAYOUT, device=mesh,
                                       memory_config=ttnn.DRAM_MEMORY_CONFIG,
                                       mesh_mapper=replicate)

            fidelities = []
            for name in ('LoFi', 'HiFi2', 'HiFi3', 'HiFi4'):
                value = getattr(getattr(ttnn, 'MathFidelity', None), name, None)
                if value is not None:
                    fidelities.append((name, value))
            report['fidelities_available'] = [n for n, _ in fidelities]

            weight_dtypes = []
            for name in ('bfloat8_b', 'bfloat16'):
                value = getattr(ttnn, name, None)
                if value is not None:
                    weight_dtypes.append((name, value))

            # square shapes for a clean peak, plus the model's own MLP shape so the
            # answer can be applied to prefill without a second leap
            shapes = [(4096, 4096, 4096), (8192, 8192, 8192), (2048, 5120, 17408)]

            torch.manual_seed(0)
            for M, K, N in shapes:
                a_host = torch.randn(1, 1, M, K)
                b_host = torch.randn(1, 1, K, N)
                flops = 2.0 * M * K * N
                # bytes a perfectly cache-less implementation must move
                moved = 2.0 * (M * K + K * N + M * N)
                intensity = flops / moved
                for wname, wdtype in weight_dtypes:
                    try:
                        a = dev(a_host, ttnn.bfloat16)
                        b = dev(b_host, wdtype)
                    except BaseException as error:
                        report['results'].append(
                            {'M': M, 'K': K, 'N': N, 'weights': wname,
                             'error': '%s: %s' % (type(error).__name__, str(error)[:180])})
                        continue
                    for fname, fid in fidelities:
                        entry = {'M': M, 'K': K, 'N': N, 'weights': wname,
                                 'activations': 'bfloat16', 'fidelity': fname,
                                 'flop_per_byte': round(intensity, 1),
                                 'dram_bound_pflops': round(
                                     intensity * DRAM_GB_S * 1e9 / 1e15, 2)}
                        try:
                            cfg = ttnn.init_device_compute_kernel_config(
                                mesh.arch(), math_fidelity=fid, fp32_dest_acc_en=False,
                                packer_l1_acc=True)

                            def once():
                                return ttnn.matmul(a, b, compute_kernel_config=cfg,
                                                   memory_config=ttnn.DRAM_MEMORY_CONFIG)

                            out = once()
                            sync()
                            ttnn.deallocate(out)
                            start = time.time()
                            for _ in range(options.iters):
                                out = once()
                            sync()
                            elapsed = (time.time() - start) / options.iters
                            ttnn.deallocate(out)
                            entry['ms'] = round(1e3 * elapsed, 3)
                            entry['tflops'] = round(flops / elapsed / 1e12, 1)
                            # If this exceeds what DRAM could feed, the run was not
                            # compute bound and the number measures the wrong thing.
                            entry['compute_bound'] = (
                                entry['tflops'] / 1e3 < entry['dram_bound_pflops'])
                        except BaseException as error:
                            entry['error'] = '%s: %s' % (type(error).__name__,
                                                         str(error)[:180])
                        report['results'].append(entry)
                        print('%5dx%5dx%5d %-9s %-5s -> %s TFLOPS'
                              % (M, K, N, wname, fname, entry.get('tflops')), flush=True)
                    ttnn.deallocate(a)
                    ttnn.deallocate(b)
        finally:
            ttnn.close_mesh_device(mesh)

        good = [r for r in report['results']
                if r.get('tflops') and r.get('compute_bound')]
        if good:
            best = max(good, key=lambda r: r['tflops'])
            report['peak_tflops_per_card'] = best['tflops']
            report['peak_at'] = '%dx%dx%d %s %s' % (best['M'], best['K'], best['N'],
                                                    best['weights'], best['fidelity'])
            report['assumed_previously'] = 774.0
            report['ratio_to_assumed'] = round(best['tflops'] / 774.0, 3)
            # what prefill's arithmetic window implies against a MEASURED peak
            ideal_ms = 1e3 * (1.52e15 / 2.0) / (best['tflops'] * 1e12)
            report['prefill_ideal_ms_at_measured_peak'] = round(ideal_ms, 1)
            report['prefill_arithmetic_ms'] = 1719.0
            report['prefill_mfu_pct'] = round(100 * ideal_ms / 1719.0, 1)
            report['grid_occupancy_pct'] = 50.7
            report['implied_per_core_efficiency_pct'] = round(
                100 * (ideal_ms / 1719.0) / 0.507, 1)
    except BaseException as error:
        report['error'] = '%s: %s' % (type(error).__name__, str(error)[:300])
        report['traceback'] = traceback.format_exc()[-1800:]

    print(BEGIN, flush=True)
    print(json.dumps(report, indent=2, default=str), flush=True)
    print(END, flush=True)
    if options.json:
        io.open(options.json, 'w', encoding='utf-8', newline='\n').write(
            json.dumps(report, indent=2, default=str) + '\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
