"""Measure the fused gate/up recipe across its reviewed worker mappings.

fused_1d.mapping supports 39/55/68/91 workers via pairs_per_worker 7/5/4/3 over 272
N-tile pairs. Only 68 divides 272 exactly; the others leave a ragged last worker
(91 -> last does 2 of 3, 39 -> last does 6 of 7, 55 -> last does 2 of 5). If that
raggedness costs real time, padding N to 280 would buy exact mappings at 70/56/40 -
and that is worth knowing before changing a qualified kernel.

Also times fused_1d.native_gate_up_control as the unfused reference, and checks the
fused output matches it exactly, which is the same invariant fused-batch-probe
asserts.

No serving path is touched: weights here are synthetic.
"""

import argparse
import json
import os
import sys
import time
import traceback

BEGIN = '<<<FUSED_SWEEP_JSON_BEGIN>>>'
END = '<<<FUSED_SWEEP_JSON_END>>>'
INNER = 5120
PAIRS = 272          # N-tile pairs per chip (8704 / 32)
TOTAL_WIDTH = 17408  # both chips' gate halves stacked before pair_pack


def emit(report):
    print(BEGIN)
    print(json.dumps(report, indent=2))
    print(END)
    sys.stdout.flush()


def timed(ttnn, device, call, iterations):
    warm = call()
    ttnn.synchronize_device(device)
    ttnn.deallocate(warm)
    samples = []
    for _ in range(iterations):
        start = time.perf_counter()
        result = call()
        ttnn.synchronize_device(device)
        samples.append(time.perf_counter() - start)
        ttnn.deallocate(result)
    samples.sort()
    return 1000.0 * samples[len(samples) // 2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--token-rows', type=int, default=32)
    parser.add_argument('--iterations', type=int, default=10)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--pairs', default='3,4,5,7')
    parser.add_argument('--warmup', type=int, default=5)
    options = parser.parse_args()
    report = dict(scope=__doc__, token_rows=options.token_rows, pairs_total=PAIRS,
                  arms=[], speedup_claimed=False)
    device = None
    try:
        sys.path.insert(0, '/opt/tt-metal')
        sys.path.insert(0, '/source/scripts/ci')
        import torch
        import ttnn
        from fused_1d import FusedProjection, native_gate_up_control, mapping

        device = ttnn.open_mesh_device(ttnn.MeshShape(1, 2), l1_small_size=24576)
        rows = options.token_rows
        torch.manual_seed(0)

        # Same packing the production probe uses: (17408, 5120) BF16 gate/up, split
        # per chip, transposed, then gate/up interleaved in 32-column groups.
        gate = torch.randn(TOTAL_WIDTH, INNER, dtype=torch.bfloat16)
        up = torch.randn(TOTAL_WIDTH, INNER, dtype=torch.bfloat16)
        gate_parts = [part.T.contiguous() for part in gate.chunk(2, dim=0)]
        up_parts = [part.T.contiguous() for part in up.chunk(2, dim=0)]
        packed = [torch.stack((first.reshape(INNER, PAIRS, 32),
                               second.reshape(INNER, PAIRS, 32)), dim=2).reshape(INNER, TOTAL_WIDTH)
                  for first, second in zip(gate_parts, up_parts)]
        stacked = [torch.stack(parts).unsqueeze(1)
                   for parts in (gate_parts, up_parts, packed)]

        def upload(value, sharded=True):
            return ttnn.from_torch(
                value, device=device, dtype=ttnn.bfloat4_b, layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG if sharded else ttnn.L1_MEMORY_CONFIG,
                mesh_mapper=ttnn.ShardTensorToMesh(device, dim=0) if sharded
                else ttnn.ReplicateTensorToMesh(device))

        device_gate, device_up, device_packed = [upload(v) for v in stacked]
        pt_act = torch.randn(1, 1, rows, INNER, dtype=torch.bfloat16)
        inputs = ttnn.from_torch(pt_act, device=device, dtype=ttnn.bfloat16,
                                 layout=ttnn.TILE_LAYOUT,
                                 memory_config=ttnn.L1_MEMORY_CONFIG,
                                 mesh_mapper=ttnn.ReplicateTensorToMesh(device))
        kernel = ttnn.init_device_compute_kernel_config(
            device.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
            fp32_dest_acc_en=True, packer_l1_acc=True, dst_full_sync_en=True)

        # ---- Unfused reference ----
        # Held in a live list for the whole run. The previous version passed a
        # throwaway [] and compared after the native timing loop had churned L1,
        # so the reference had been overwritten and every bit-exact check failed.
        reference_owned = []
        reference = None
        try:
            reference = native_gate_up_control(ttnn, inputs, device_gate, device_up,
                                               kernel, reference_owned)
            reference_host = [ttnn.to_torch(ttnn.get_device_tensors(reference)[chip]).clone()
                              for chip in range(2)]
            report['reference_captured'] = True
        except BaseException as error:
            report['reference_error'] = dict(kind=type(error).__name__,
                                             message=str(error)[:600])
            reference_host = None

        # Self-test: native against itself. If this fails the harness is at fault and
        # no bit-exact verdict below means anything.
        if reference_host is not None:
            try:
                owned = []
                again = native_gate_up_control(ttnn, inputs, device_gate, device_up,
                                               kernel, owned)
                report['native_self_consistent'] = all(
                    torch.equal(ttnn.to_torch(ttnn.get_device_tensors(again)[chip]),
                                reference_host[chip]) for chip in range(2))
            except BaseException as error:
                report['native_self_test_error'] = str(error)[:300]

        try:
            native_ms = []
            for _ in range(options.repeats):
                native_ms.append(timed(ttnn, device,
                                       lambda: native_gate_up_control(
                                           ttnn, inputs, device_gate, device_up,
                                           kernel, []),
                                       options.iterations))
            native_ms.sort()
            report['native'] = dict(workers=39, grid=[11, 4],
                                    ms_median=round(native_ms[len(native_ms) // 2], 4),
                                    ms_all=[round(v, 4) for v in native_ms])
        except BaseException as error:
            report['native_error'] = dict(kind=type(error).__name__,
                                          message=str(error)[:600])

        # ---- Fused, across the reviewed mappings ----
        for pairs_per_worker in [int(v) for v in options.pairs.split(',')]:
            arm = dict(pairs_per_worker=pairs_per_worker)
            try:
                layout = mapping(pairs_per_worker)
                tail = PAIRS - (len(layout) - 1) * pairs_per_worker
                arm.update(workers=len(layout), grid=[11, (len(layout) + 10) // 11],
                           tail_pairs=tail, exact=tail == pairs_per_worker)
                operation = FusedProjection(device, device_packed,
                                            pairs_per_worker=pairs_per_worker,
                                            token_rows=rows,
                                            source_root=os.environ.get('TT_METAL_HOME',
                                                                       '/opt/tt-metal'))
                # Warm until the patched program is built and cached; the first call
                # carries JIT and program construction, which is not what we are timing.
                warm_ms = []
                for _ in range(options.warmup):
                    start = time.perf_counter()
                    produced = operation(inputs)
                    ttnn.synchronize_device(device)
                    warm_ms.append(round(1000.0 * (time.perf_counter() - start), 4))
                    if produced is not None:
                        last = produced
                arm['warmup_ms'] = warm_ms
                arm['ran'] = True
                if reference_host is not None:
                    arm['bit_exact_vs_native'] = all(
                        torch.equal(ttnn.to_torch(ttnn.get_device_tensors(last)[chip]),
                                    reference_host[chip]) for chip in range(2))
                ttnn.deallocate(last)
                samples = []
                for _ in range(options.repeats):
                    samples.append(timed(ttnn, device, lambda: operation(inputs),
                                         options.iterations))
                samples.sort()
                arm['ms_median'] = round(samples[len(samples) // 2], 4)
                arm['ms_all'] = [round(v, 4) for v in samples]
                arm['spread_pct'] = round(100.0 * (samples[-1] - samples[0]) / samples[0], 1)
            except BaseException as error:
                arm['error'] = dict(kind=type(error).__name__, message=str(error)[:700])
            report['arms'].append(arm)

        best = [a for a in report['arms'] if a.get('ms_median')]
        if best:
            fastest = min(best, key=lambda a: a['ms_median'])
            report['fastest_mapping'] = dict(workers=fastest['workers'],
                                             pairs_per_worker=fastest['pairs_per_worker'],
                                             ms_median=fastest['ms_median'])
    except BaseException:
        report['fatal'] = traceback.format_exc(limit=8)[-1800:]
    emit(report)
    if device is not None:
        try:
            import ttnn
            ttnn.close_mesh_device(device)
        except BaseException:
            pass
    sys.stdout.flush()
    os._exit(0)


if __name__ == '__main__':
    main()
