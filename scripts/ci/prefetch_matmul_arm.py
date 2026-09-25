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

# Padding N changes the factorisation of n_tiles, which is what caps the ring.
# 8704 -> 8960 takes n_tiles from 272 = 2^4 x 17 to 280 = 2^3 x 5 x 7, admitting
# ring=40 instead of 16. The padded columns are zeroed here, so the padded output
# columns are zero and the unpadded result is unchanged.
PADDED_WIDTH = {'gate': 8960, 'up': 8960}

# Worker L1 CB space, per the prefetcher design doc. Held below the nominal ~1.5 MB
# because the activation and output CBs also have to live there.
L1_CB_BUDGET = 1_200_000


def gcb_bytes_per_receiver(k_tiles, n_tiles, tile, ring):
    """The matmul does wait_front(ring), so every page must be resident."""
    return (k_tiles // ring) * (n_tiles // ring) * tile * ring


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


def bank_receivers_row_major(ttnn, bank_idx, recv_per_bank, ring_cols):
    """Verbatim from upstream's test_prefetcher_BH_tensor_large.

    Bank b owns ring positions [b*recv_per_bank, (b+1)*recv_per_bank), each as its
    own single-core CoreRange. Laying them out column-major instead yields a set the
    GCB factory counts as one receiver per bank.
    """
    cores = []
    for k in range(recv_per_bank):
        ring_pos = bank_idx * recv_per_bank + k
        col = ring_pos % ring_cols
        row = ring_pos // ring_cols
        cores.append(ttnn.CoreRange(ttnn.CoreCoord(col, row), ttnn.CoreCoord(col, row)))
    return ttnn.CoreRangeSet(cores)


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
                  dtype_override=None, stream=True, pad=True):
    inner, native_width, dtype_name = PROJECTIONS[name]
    if dtype_override:
        dtype_name = dtype_override
    width = PADDED_WIDTH.get(name, native_width) if pad else native_width
    k_tiles, n_tiles = inner // TILE, width // TILE
    banks = device.dram_grid_size().x
    per_tile_probe = TILE_BYTES[dtype_name]
    options = legal_rings(k_tiles, n_tiles, banks, 10)
    for option in options:
        option['gcb_bytes'] = gcb_bytes_per_receiver(k_tiles, n_tiles, per_tile_probe,
                                                     option['ring'])
        option['fits_l1'] = option['gcb_bytes'] <= L1_CB_BUDGET
    fitting = [o for o in options if o['fits_l1']]
    entry = dict(projection=name, dtype=dtype_name, inner=inner,
                 native_width=native_width, width=width, padded=width != native_width,
                 pad_overhead_pct=round(100.0 * (width - native_width) / native_width, 2),
                 k_tiles=k_tiles, n_tiles=n_tiles, banks=banks,
                 legal_rings=[o['ring'] for o in options],
                 rings_fitting_l1=[o['ring'] for o in fitting])
    report['arms'].append(entry)
    if not options:
        entry['stopped_at'] = 'no ring divides both K and N'
        return
    if not fitting:
        entry['stopped_at'] = 'no legal ring keeps the GCB inside L1'
        return
    chosen = fitting[-1]
    if rows_choice is not None:
        chosen = next((o for o in fitting if o['rows'] == rows_choice), fitting[-1])
    entry.update(chosen)

    dtype = getattr(ttnn, dtype_name)
    ring_cols, ring_rows = banks, chosen['rows']
    ring_size = chosen['ring']
    entry['grid'] = [ring_cols, ring_rows]
    entry['receiver_cores'] = ring_size

    receivers = ttnn.CoreRangeSet(
        {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(ring_cols - 1, ring_rows - 1))})
    rows = 32  # activation rows (M)
    torch.manual_seed(0)
    pt_weight = torch.randn(1, 1, inner, width)
    if width != native_width:
        # Zeroing the padding is what keeps the real output columns untouched.
        pt_weight[:, :, :, native_width:] = 0.0
    pt_act = torch.randn(1, 1, rows, inner)

    # One shard per receiver, not per bank: with ring > num_banks the factory requires
    # a receiver-contiguous layout ("num_shards must equal receiver_count"). A plain
    # width-sharded DRAM weight gives num_shards = banks and is rejected.
    weight = common.make_recv_contig_weight(device, pt_weight, banks, ring_size, dtype)
    entry['weight_built'] = True
    entry['weight_layout'] = 'receiver_contiguous'

    k_per_shard = common.round_up(math.ceil(inner / ring_size), TILE)
    act_mem = ttnn.create_sharded_memory_config(
        shape=(rows, k_per_shard), core_grid=receivers, strategy=ttnn.ShardStrategy.WIDTH,
        orientation=ttnn.ShardOrientation.ROW_MAJOR, use_height_and_width_as_shard_shape=True)
    act = ttnn.from_torch(pt_act, device=device, dtype=ttnn.bfloat16,
                          memory_config=act_mem, layout=ttnn.TILE_LAYOUT)
    entry['act_built'] = True

    # Parameters taken from upstream's test_tensor_prefetcher_BH_param. The one that
    # mattered: num_global_cb_receivers defaults to 1, and the GCB factory reads
    # receivers-per-bank from the program config rather than from bank_to_receivers,
    # which is why it kept reporting "8 senders * 1 receivers/bank".
    out_block_w = chosen['n_tiles_per_receiver']
    out_subblock_w = min(out_block_w, 8)
    while out_subblock_w > 1 and out_block_w % out_subblock_w != 0:
        out_subblock_w -= 1
    stream_kwargs = {}
    if stream:
        try:
            ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
                compute_with_storage_grid_size=(1, 1), in0_block_w=1, out_subblock_h=1,
                out_subblock_w=1, per_core_M=1, per_core_N=1, fuse_batch=True,
                fused_activation=None, mcast_in0=False, gather_in0=True,
                hop_cores=ttnn.CoreRangeSet([]), num_global_cb_receivers=1,
                untilize_out=False, stream_in1=True)
            stream_kwargs = dict(stream_in1=True)
        except BaseException as error:
            entry['stream_in1_unavailable'] = str(error)[:200]
    program_config = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=(ring_cols, ring_rows),
        in0_block_w=1,  # the DRISC factory's kbw defaults to 1
        out_subblock_h=1, out_subblock_w=out_subblock_w,
        per_core_M=rows // TILE, per_core_N=out_block_w,
        fuse_batch=True, fused_activation=None, mcast_in0=False, gather_in0=True,
        hop_cores=ttnn.CoreRangeSet([]), num_global_cb_receivers=ring_rows,
        untilize_out=False, **stream_kwargs)
    entry['program_config_built'] = True
    entry['out_subblock_w'] = out_subblock_w
    entry['stream_in1'] = stream_kwargs.get('stream_in1', False)


    out_mem = ttnn.create_sharded_memory_config(
        shape=(rows, width // ring_size), core_grid=receivers,
        strategy=ttnn.ShardStrategy.WIDTH, orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True)
    compute_kernel_config = ttnn.init_device_compute_kernel_config(
        device.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
        fp32_dest_acc_en=True, packer_l1_acc=True, dst_full_sync_en=True)

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

    expected = pt_act.float() @ pt_weight.float()

    # The control is the path this workload actually uses today:
    # dram_sharded_projection at its native T16 and native (unpadded) width. Charging
    # the padding to the prefetched arm is the point - that is the real trade.
    control = None
    try:
        import dram_sharded_projection as projection
        control_config = projection.configurations(ttnn, device, name)
        control_plan = control_config['plan']
        entry['control'] = dict(path='dram_sharded_projection',
                                width=control_plan['width'],
                                workers=control_plan['workers'],
                                per_core_N=control_plan['per_core_N'],
                                in0_block_w=control_plan['in0_block_w'])
        pt_control_weight = pt_weight[:, :, :, :native_width].contiguous()
        control_weight = ttnn.as_tensor(
            pt_control_weight, device=device, dtype=dtype, layout=ttnn.TILE_LAYOUT,
            memory_config=control_config['weights'])
        # execute() validates a T16 BF16 source.
        pt_control_act = pt_act[:, :16, :].contiguous() if pt_act.shape[1] >= 16 else pt_act
        pt_control_act = torch.randn(1, 1, 16, inner, dtype=torch.bfloat16)
        control_act = ttnn.from_torch(pt_control_act, device=device, dtype=ttnn.bfloat16,
                                      layout=ttnn.TILE_LAYOUT,
                                      memory_config=ttnn.DRAM_MEMORY_CONFIG)

        def control():
            kept = []
            return projection.execute(ttnn, control_act, control_weight, control_config,
                                      compute_kernel_config, lambda t: (kept.append(t), t)[1])
    except BaseException as error:
        entry['control_unavailable'] = '%s: %s' % (type(error).__name__, str(error)[:400])

    control_ms = []
    if control is not None:
        try:
            for _ in range(3):
                control_ms.append(timed(control, 10))
            entry['control_measured_before_gcb'] = True
        except BaseException as error:
            entry['control_timing_error'] = '%s: %s' % (type(error).__name__, str(error)[:300])
            control_ms = []

    per_tile = tile_bytes(common, ttnn, dtype_name, dtype)
    entry['tile_bytes'] = per_tile
    in1_block = chosen['k_tiles_per_shard'] * chosen['n_tiles_per_receiver'] * per_tile
    depth = ring_size
    gcb_size = depth * in1_block
    entry['in1_block_bytes'] = in1_block
    entry['gcb_depth_pages'] = depth
    entry['gcb_size'] = gcb_size
    entry['gcb_size_mb'] = round(gcb_size / (1024 * 1024), 3)
    # Receiver-contiguous weights require the STRIDED topology, not row-major:
    # design doc section 6 says BDS round-robin puts shard m at bank m % num_senders,
    # slab m // num_senders, and the caller pairs that with bank b -> ring positions
    # [b, b+num_senders, ...] so shard index == ring position with no host permutation.
    # Row-major pairs with a width-sharded weight; mixing them delivers each receiver
    # the wrong shard, which is what produced PCC 0.025.
    bank_to_receivers = [(b, common.bank_receivers_strided(b, ring_rows, banks, ring_cols))
                         for b in range(banks)]
    entry['topology'] = 'strided'
    entry['receivers_per_bank'] = ring_rows
    entry['bank0_receivers'] = str(bank_to_receivers[0][1])[:120]
    global_cb = ttnn.experimental.create_global_circular_buffer_for_matmul_1d(
        device, [program_config], [weight], bank_to_receivers=bank_to_receivers, size=gcb_size)
    entry['gcb_built'] = True

    def prefetched():
        return ttnn.experimental.tensor_prefetcher_matmul.prefetch_and_linear(
            act, weight, global_cb=global_cb, program_config=program_config,
            memory_config=out_mem, compute_kernel_config=compute_kernel_config, dtype=dtype)



    with common.tensor_prefetcher_session(device):
        out = prefetched()
        entry['matmul_ran'] = True
        # The weight is replicated across the 1x2 mesh, so every device computes the
        # same result; read one shard rather than composing duplicates together.
        shards = ttnn.get_device_tensors(out)
        entry['device_shards'] = len(shards)
        got = ttnn.to_torch(shards[0])
        ttnn.deallocate(out)
        passed, message = common.comp_pcc(expected, got.float(), 0.96)
        entry['pcc_passed'] = bool(passed)
        entry['pcc_message'] = str(message)[:200]
        if not entry['pcc_passed']:
            return

        # Arms are blocked, not interleaved: the GCB holds L1 on every receiver core,
        # so the control cannot run while it exists. Host pressure was low and steady
        # across this run, which is the condition that makes blocked arms acceptable.
        prefetch_ms = [timed(prefetched, 10) for _ in range(3)]
        entry['arms_interleaved'] = False
        if not control_ms:
            entry['prefetch_ms_median'] = round(sorted(prefetch_ms)[1], 4)
            entry['prefetch_ms_all'] = [round(v, 4) for v in sorted(prefetch_ms)]
            return
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
    parser.add_argument('--no-pad', action='store_true',
                        help='use the native width instead of the padded one')
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
        sys.path.insert(0, '/source/scripts/ci')
        import torch
        import ttnn
        from tests.ttnn.unit_tests.operations import prefetcher_common as common
        device = ttnn.open_mesh_device(ttnn.MeshShape(1, 2), l1_small_size=24576)
        report['supported'] = ttnn.experimental.is_tensor_prefetcher_supported(device)
        for name in options.projections.split(','):
            try:
                build_and_run(ttnn, torch, common, device, name, options.rows, report,
                              options.dtype, not options.no_stream,
                              not options.no_pad)
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
