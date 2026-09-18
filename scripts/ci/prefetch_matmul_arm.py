"""Checkpoint 4 arm: prefetched 1D matmul on Qwen MLP shapes vs an unprefetched control.

Construction is ported from upstream's test_prefetcher_BH_tensor_large so the GCB
receiver set and the matmul output workers cannot drift: the ring is
ring_cols = num_dram_banks wide by recv_per_bank tall, and that same core range set
carries the GCB receivers, the activation shards and the output shards. An earlier
hand-rolled attempt transposed the grid and hit "mcast_in0 global_cb receivers must
exactly match output worker cores".

Correctness gates timing. Timings are indicative only; a verdict needs whole-cycle
TG against the real recipe, not a microbenchmark.
"""

import argparse
import json
import math
import os
import sys
import time
import traceback

BEGIN = '<<<PREFETCH_ARM_JSON_BEGIN>>>'
END = '<<<PREFETCH_ARM_JSON_END>>>'
TILE = 32
# Per-device TP2 shapes from scripts/ci/tiny_tile_matmul.PROJECTIONS.
PROJECTIONS = {'gate': (5120, 8704, 'bfloat4_b'),
               'up': (5120, 8704, 'bfloat4_b'),
               'down': (8704, 5120, 'bfloat8_b')}


# Upstream's prefetcher_common.bytes_per_tile maps only bfloat16 and bfloat8_b, so
# their harness never exercises bfloat4_b - which is exactly what Qwen's gate and up
# weights use. Blackhole tiles are 32x32: bf8_b is 1024 mantissa + 64 exponent bytes,
# bf4_b is 512 + 64. Supplying it here lets the arm run; whether the DRISC path is
# happy with bf4_b is what the run then tells us.
TILE_BYTES = {'bfloat16': 2048, 'bfloat8_b': 1088, 'bfloat4_b': 576}


def tile_bytes(common, ttnn, dtype_name, dtype):
    try:
        return common.bytes_per_tile(dtype)
    except KeyError:
        return TILE_BYTES[dtype_name]


def legal_rings(k_tiles, n_tiles, banks, max_rows):
    """Rings are banks x rows; K and N must both divide the ring for integral shards."""
    out = []
    for rows in range(1, max_rows + 1):
        ring = banks * rows
        if k_tiles % ring == 0 and n_tiles % ring == 0:
            out.append(dict(rows=rows, ring=ring, k_tiles_per_shard=k_tiles // ring,
                            n_tiles_per_receiver=n_tiles // ring))
    return out


def emit(report):
    print(BEGIN)
    print(json.dumps(report, indent=2))
    print(END)
    sys.stdout.flush()


def build_and_run(ttnn, torch, common, device, name, rows_choice, report,
                  dtype_override=None, stream=True):
    inner, width, dtype_name = PROJECTIONS[name]
    if dtype_override:
        dtype_name = dtype_override
    k_tiles, n_tiles = inner // TILE, width // TILE
    banks = device.dram_grid_size().x
    options = legal_rings(k_tiles, n_tiles, banks, 10)
    entry = dict(projection=name, dtype=dtype_name, inner=inner, width=width,
                 k_tiles=k_tiles, n_tiles=n_tiles, banks=banks,
                 legal_rings=[o['ring'] for o in options])
    report['arms'].append(entry)
    if not options:
        entry['stopped_at'] = 'no ring divides both K and N'
        return
    chosen = options[-1]
    if rows_choice is not None:
        chosen = next((o for o in options if o['rows'] == rows_choice), options[-1])
    entry.update(chosen)

    dtype = getattr(ttnn, dtype_name)
    ring_cols, ring_rows = banks, chosen['rows']
    ring_size = chosen['ring']
    entry['grid'] = [ring_cols, ring_rows]
    entry['receiver_cores'] = ring_size

    receivers = ttnn.CoreRangeSet(
        {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(ring_cols - 1, ring_rows - 1))})
    rows = 32
    torch.manual_seed(0)
    pt_weight = torch.randn(1, 1, inner, width)
    pt_act = torch.randn(1, 1, rows, inner)

    dram_cores = ttnn.CoreRangeSet(
        {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(banks - 1, 0))})
    weight_mem = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.DRAM,
        ttnn.ShardSpec(dram_cores, [inner, width // banks], ttnn.ShardOrientation.ROW_MAJOR))
    weight = ttnn.as_tensor(pt_weight, device=device, dtype=dtype,
                            memory_config=weight_mem, layout=ttnn.TILE_LAYOUT)
    entry['weight_built'] = True

    k_per_shard = common.round_up(math.ceil(inner / ring_size), TILE)
    act_mem = ttnn.create_sharded_memory_config(
        shape=(rows, k_per_shard), core_grid=receivers, strategy=ttnn.ShardStrategy.WIDTH,
        orientation=ttnn.ShardOrientation.ROW_MAJOR, use_height_and_width_as_shard_shape=True)
    act = ttnn.from_torch(pt_act, device=device, dtype=ttnn.bfloat16,
                          memory_config=act_mem, layout=ttnn.TILE_LAYOUT)
    entry['act_built'] = True

    # Batched gather-in0 needs ring_size pages resident per receiver. At this ring
    # a page is k_tiles_per_shard x n_tiles_per_receiver x tile_bytes, so the whole
    # fifo exceeds Blackhole's ~1.5 MB L1. Streaming consumes from a shallow window.
    stream_kwargs = {}
    if stream:
        try:
            ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
                compute_with_storage_grid_size=ttnn.CoreCoord(1, 1), in0_block_w=1,
                out_subblock_h=1, out_subblock_w=1, per_core_M=1, per_core_N=1,
                fuse_batch=True, fused_activation=None, mcast_in0=False,
                gather_in0=True, stream_in1=True)
            stream_kwargs = dict(stream_in1=True)
        except BaseException as error:
            entry['stream_in1_unavailable'] = str(error)[:200]
    program_config = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(ring_cols, ring_rows),
        in0_block_w=chosen['k_tiles_per_shard'], out_subblock_h=1, out_subblock_w=1,
        per_core_M=1, per_core_N=chosen['n_tiles_per_receiver'],
        fuse_batch=True, fused_activation=None, mcast_in0=False, gather_in0=True,
        **stream_kwargs)
    entry['program_config_built'] = True
    entry['stream_in1'] = stream_kwargs.get('stream_in1', False)

    per_tile = tile_bytes(common, ttnn, dtype_name, dtype)
    entry['tile_bytes'] = per_tile
    in1_block = chosen['k_tiles_per_shard'] * chosen['n_tiles_per_receiver'] * per_tile
    depth = 2 if stream_kwargs else ring_size
    gcb_size = depth * in1_block
    entry['in1_block_bytes'] = in1_block
    entry['gcb_depth_pages'] = depth
    entry['gcb_size'] = gcb_size
    entry['gcb_size_mb'] = round(gcb_size / (1024 * 1024), 3)
    bank_to_receivers = [(b, common.bank_receivers_strided(b, ring_rows, banks, ring_cols))
                         for b in range(banks)]
    global_cb = ttnn.experimental.create_global_circular_buffer_for_matmul_1d(
        device, [program_config], [weight], bank_to_receivers=bank_to_receivers, size=gcb_size)
    entry['gcb_built'] = True

    out_mem = ttnn.create_sharded_memory_config(
        shape=(rows, width // ring_size), core_grid=receivers,
        strategy=ttnn.ShardStrategy.WIDTH, orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True)
    compute_kernel_config = ttnn.init_device_compute_kernel_config(
        device.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
        fp32_dest_acc_en=True, packer_l1_acc=True, dst_full_sync_en=True)

    expected = pt_act.float() @ pt_weight.float()

    def prefetched():
        return ttnn.experimental.tensor_prefetcher_matmul.prefetch_and_linear(
            act, weight, global_cb=global_cb, program_config=program_config,
            memory_config=out_mem, compute_kernel_config=compute_kernel_config, dtype=dtype)

    def control():
        return ttnn.linear(act, weight, program_config=program_config, memory_config=out_mem,
                           compute_kernel_config=compute_kernel_config, dtype=dtype)

    with common.tensor_prefetcher_session(device):
        out = prefetched()
        entry['matmul_ran'] = True
        got = ttnn.to_torch(out)
        ttnn.deallocate(out)
        passed, message = common.comp_pcc(expected, got.float(), 0.96)
        entry['pcc_passed'] = bool(passed)
        entry['pcc_message'] = str(message)[:200]
        if not entry['pcc_passed']:
            return

        def timed(call, iterations):
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

        # Interleave the arms: host contention on this rig inflates
        # host-dispatch-bound phases far more than device-bound ones.
        prefetch_ms, control_ms = [], []
        for _ in range(3):
            prefetch_ms.append(timed(prefetched, 10))
            control_ms.append(timed(control, 10))
        prefetch_ms.sort()
        control_ms.sort()
        entry['prefetch_ms_median'] = round(prefetch_ms[1], 4)
        entry['control_ms_median'] = round(control_ms[1], 4)
        entry['prefetch_ms_all'] = [round(v, 4) for v in prefetch_ms]
        entry['control_ms_all'] = [round(v, 4) for v in control_ms]
        entry['ratio_prefetch_over_control'] = round(prefetch_ms[1] / control_ms[1], 4)
        entry['timing_is_indicative_only'] = True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--projections', default='gate')
    parser.add_argument('--no-stream', action='store_true',
                        help='use batched gather-in0 instead of a streaming window')
    parser.add_argument('--dtype', default=None,
                        help='override the weight dtype, e.g. bfloat8_b as a control')
    parser.add_argument('--rows', type=int, default=None,
                        help='receivers per bank; default is the largest legal ring')
    options = parser.parse_args()
    report = dict(scope=__doc__, arms=[], speedup_claimed=False)
    device = None
    try:
        sys.path.insert(0, '/opt/tt-metal')
        import torch
        import ttnn
        from tests.ttnn.unit_tests.operations import prefetcher_common as common
        device = ttnn.open_mesh_device(ttnn.MeshShape(1, 2), l1_small_size=24576)
        report['supported'] = ttnn.experimental.is_tensor_prefetcher_supported(device)
        for name in options.projections.split(','):
            try:
                build_and_run(ttnn, torch, common, device, name, options.rows, report,
                              options.dtype, not options.no_stream)
            except BaseException as error:
                detail = dict(kind=type(error).__name__,
                              message=str(error)[:1600],
                              stack_tail=traceback.format_exc(limit=4)[-700:])
                if report['arms']:
                    report['arms'][-1]['error'] = detail
                else:
                    report['fatal'] = detail
    except BaseException:
        report['fatal'] = traceback.format_exc(limit=8)[-2000:]
    # Emit before teardown: a TT_FATAL can leave close_mesh_device hanging, and an
    # earlier run lost its entire report that way.
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
