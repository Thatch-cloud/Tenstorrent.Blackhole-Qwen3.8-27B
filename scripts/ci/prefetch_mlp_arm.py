"""Prefetched gate+up+multiply on Qwen shapes against the repo's native 1D control.

The earlier arm timed a single gate projection against dram_sharded_projection, which
is an experimental path (its own module says "no serving integration") running four
workers. This compares a comparable unit of work instead: gate, up and their product.

Control is fused_1d.native_gate_up_control - two mcast_in0 1D matmuls on an (11,4)
grid plus the multiply. With per_core_N=7 and 272 N-tiles it occupies 39 of the 44
cores, which is the repo's 39-worker native mapping.

Prefetched arm: both weights padded 8704 -> 8960 so a ring of 40 exists that keeps
the GCB inside L1, receiver-contiguous layout, strided topology.

Padding is charged to the prefetched arm; the control runs native width. Timings are
indicative - a verdict still needs whole-cycle TG.
"""

import argparse
import json
import math
import os
import sys
import time
import traceback

BEGIN = '<<<PREFETCH_MLP_JSON_BEGIN>>>'
END = '<<<PREFETCH_MLP_JSON_END>>>'
TILE = 32
INNER = 5120
NATIVE_WIDTH = 8704
PADDED_WIDTH = 8960
TILE_BYTES = {'bfloat16': 2048, 'bfloat8_b': 1088, 'bfloat4_b': 576}
L1_CB_BUDGET = 1_200_000


def legal_rings(k_tiles, n_tiles, banks, max_rows=10):
    out = []
    for rows in range(1, max_rows + 1):
        ring = banks * rows
        if k_tiles % ring or n_tiles % ring:
            continue
        gcb = (k_tiles // ring) * (n_tiles // ring) * TILE_BYTES['bfloat4_b'] * ring
        out.append(dict(rows=rows, ring=ring, k_tiles_per_shard=k_tiles // ring,
                        n_tiles_per_receiver=n_tiles // ring, gcb_bytes=gcb,
                        fits_l1=gcb <= L1_CB_BUDGET))
    return out


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
    parser.add_argument('--rows', type=int, default=32)
    parser.add_argument('--iterations', type=int, default=10)
    parser.add_argument('--repeats', type=int, default=3)
    options = parser.parse_args()
    report = dict(scope=__doc__, rows=options.rows, speedup_claimed=False,
                  padding_charged_to='prefetched arm')
    device = None
    try:
        sys.path.insert(0, '/opt/tt-metal')
        sys.path.insert(0, '/source/scripts/ci')
        import torch
        import ttnn
        from tests.ttnn.unit_tests.operations import prefetcher_common as common
        from fused_1d import native_gate_up_control

        device = ttnn.open_mesh_device(ttnn.MeshShape(1, 2), l1_small_size=24576)
        report['supported'] = ttnn.experimental.is_tensor_prefetcher_supported(device)
        banks = device.dram_grid_size().x
        rows = options.rows

        torch.manual_seed(0)
        pt_act = torch.randn(1, 1, rows, INNER, dtype=torch.bfloat16)
        act_native = ttnn.from_torch(pt_act, device=device, dtype=ttnn.bfloat16,
                                     layout=ttnn.TILE_LAYOUT,
                                     memory_config=ttnn.L1_MEMORY_CONFIG)
        kernel = ttnn.init_device_compute_kernel_config(
            device.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
            fp32_dest_acc_en=True, packer_l1_acc=True, dst_full_sync_en=True)

        # ---- Control: native 1D gate/up + multiply, unpadded ----
        control_ms = []
        try:
            pt_gate = torch.randn(1, 1, INNER, NATIVE_WIDTH)
            pt_up = torch.randn(1, 1, INNER, NATIVE_WIDTH)
            gate_native = ttnn.from_torch(pt_gate, device=device, dtype=ttnn.bfloat4_b,
                                          layout=ttnn.TILE_LAYOUT,
                                          memory_config=ttnn.DRAM_MEMORY_CONFIG)
            up_native = ttnn.from_torch(pt_up, device=device, dtype=ttnn.bfloat4_b,
                                        layout=ttnn.TILE_LAYOUT,
                                        memory_config=ttnn.DRAM_MEMORY_CONFIG)

            def control():
                owned = []
                return native_gate_up_control(ttnn, act_native, gate_native, up_native,
                                              kernel, owned)

            report['control'] = dict(path='fused_1d.native_gate_up_control',
                                     width=NATIVE_WIDTH, grid=[11, 4],
                                     effective_workers=math.ceil((NATIVE_WIDTH // TILE) / 7))
            for _ in range(options.repeats):
                control_ms.append(timed(ttnn, device, control, options.iterations))
        except BaseException as error:
            report['control_error'] = dict(kind=type(error).__name__,
                                           message=str(error)[:900])

        # ---- Prefetched arm: padded gate+up+multiply ----
        n_tiles = PADDED_WIDTH // TILE
        rings = [o for o in legal_rings(INNER // TILE, n_tiles, banks) if o['fits_l1']]
        report['rings_fitting_l1'] = [o['ring'] for o in rings]
        if not rings:
            report['prefetch_error'] = 'no ring fits L1'
        else:
            chosen = rings[-1]
            report['prefetch'] = dict(width=PADDED_WIDTH, native_width=NATIVE_WIDTH,
                                      pad_pct=round(100.0 * (PADDED_WIDTH - NATIVE_WIDTH)
                                                    / NATIVE_WIDTH, 2), **chosen)
            ring_cols, ring_rows = banks, chosen['rows']
            ring_size = chosen['ring']
            receivers = ttnn.CoreRangeSet({ttnn.CoreRange(
                ttnn.CoreCoord(0, 0), ttnn.CoreCoord(ring_cols - 1, ring_rows - 1))})

            weights = []
            for _ in range(2):
                pt_w = torch.randn(1, 1, INNER, PADDED_WIDTH)
                pt_w[:, :, :, NATIVE_WIDTH:] = 0.0  # padded columns contribute nothing
                weights.append(common.make_recv_contig_weight(
                    device, pt_w, banks, ring_size, ttnn.bfloat4_b))

            k_per_shard = common.round_up(math.ceil(INNER / ring_size), TILE)
            act_mem = ttnn.create_sharded_memory_config(
                shape=(rows, k_per_shard), core_grid=receivers,
                strategy=ttnn.ShardStrategy.WIDTH,
                orientation=ttnn.ShardOrientation.ROW_MAJOR,
                use_height_and_width_as_shard_shape=True)
            act_ring = ttnn.from_torch(pt_act, device=device, dtype=ttnn.bfloat16,
                                       layout=ttnn.TILE_LAYOUT, memory_config=act_mem)

            out_block_w = chosen['n_tiles_per_receiver']
            out_subblock_w = min(out_block_w, 8)
            while out_subblock_w > 1 and out_block_w % out_subblock_w:
                out_subblock_w -= 1
            configs = []
            for activation in (ttnn.UnaryOpType.SILU, None):
                configs.append(ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
                    compute_with_storage_grid_size=(ring_cols, ring_rows),
                    in0_block_w=1, out_subblock_h=1, out_subblock_w=out_subblock_w,
                    per_core_M=max(1, rows // TILE), per_core_N=out_block_w,
                    fuse_batch=True, fused_activation=activation, mcast_in0=False,
                    gather_in0=True, hop_cores=ttnn.CoreRangeSet([]),
                    num_global_cb_receivers=ring_rows, untilize_out=False))

            in1_block = chosen['k_tiles_per_shard'] * out_block_w * TILE_BYTES['bfloat4_b']
            gcb = ttnn.experimental.create_global_circular_buffer_for_matmul_1d(
                device, configs, weights,
                bank_to_receivers=[(b, common.bank_receivers_strided(
                    b, ring_rows, banks, ring_cols)) for b in range(banks)],
                size=ring_size * in1_block)
            out_mem = ttnn.create_sharded_memory_config(
                shape=(rows, PADDED_WIDTH // ring_size), core_grid=receivers,
                strategy=ttnn.ShardStrategy.WIDTH,
                orientation=ttnn.ShardOrientation.ROW_MAJOR,
                use_height_and_width_as_shard_shape=True)

            def prefetched():
                parts = []
                for weight, config in zip(weights, configs):
                    parts.append(ttnn.experimental.tensor_prefetcher_matmul.prefetch_and_linear(
                        act_ring, weight, global_cb=gcb, program_config=config,
                        memory_config=out_mem, compute_kernel_config=kernel,
                        dtype=ttnn.bfloat16))
                product = ttnn.multiply(parts[0], parts[1])
                for part in parts:
                    ttnn.deallocate(part)
                return product

            prefetch_ms = []
            try:
                with common.tensor_prefetcher_session(device):
                    warm = prefetched()
                    ttnn.deallocate(warm)
                    report['prefetch_ran'] = True
                    for _ in range(options.repeats):
                        prefetch_ms.append(timed(ttnn, device, prefetched,
                                                 options.iterations))
            except BaseException as error:
                report['prefetch_error'] = dict(kind=type(error).__name__,
                                                message=str(error)[:900])
            if prefetch_ms:
                prefetch_ms.sort()
                report['prefetch_ms_median'] = round(prefetch_ms[len(prefetch_ms) // 2], 4)
                report['prefetch_ms_all'] = [round(v, 4) for v in prefetch_ms]
        if control_ms:
            control_ms.sort()
            report['control_ms_median'] = round(control_ms[len(control_ms) // 2], 4)
            report['control_ms_all'] = [round(v, 4) for v in control_ms]
        if report.get('prefetch_ms_median') and report.get('control_ms_median'):
            report['ratio_prefetch_over_control'] = round(
                report['prefetch_ms_median'] / report['control_ms_median'], 4)
        report['arms_interleaved'] = False  # the GCB holds receiver L1 while it lives
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
