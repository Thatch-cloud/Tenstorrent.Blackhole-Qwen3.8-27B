"""C1e card-B harness: is the rebuilt-per-layer packed gate/up byte-identical to the served one?

C1e (lever_n_m3native_patch section J, QWEN_FAST_C1_EXACT=1 under QWEN_FAST_SINGLE_GATEUP=1) keeps
the served prefill op, all_gather_minimal_matmul_async(fuse_swiglu=True), and only rebuilds its
packed weight per layer into one scratch (scripts/ci/mlp_c1e_pack.py). This harness qualifies that
copy on one card before any model run: run_card_b.sh beside it mounts the op from THIS checkout, the
files named by lever_n_m3native_patch.C1E_FILES (the table the graft stages and the arm mounts).

Single chip, the serving image, random weights of the real TP2 per-chip shapes and dtypes
(w1 / w3 [5120, 8704] bfloat4_b, the packed [5120, 17408] bfloat4_b built by the image's own
prepare_for_fused_swiglu exactly as mlp.py _build_gate_up does, then one TP2 shard of it).
Nothing is read from the model and nothing is written outside --out.

Sections (all on by default):

  host      the image's prepare_for_fused_swiglu shard == the per-chip tile-pair interleave of
            the w1 / w3 shards (the map mlp_c1e_pack.cpp writes), for every emulated chip.
  bytes     on device: the served packed tensor vs w1 (even pages) and w3 (odd pages), and the
            C1e scratch after mlp_c1e_pack vs the same, word for word (packed_weight_check.cpp);
            the scratch's tensor spec equals the served packed tensor's.
  equality  per rows in --rows and per x seed: the served formulation (minimal_matmul with the
            served AGMM block config, compute kernel config and fuse_swiglu=True on the served
            packed weight) vs C1e (the same call on the scratch), compared as raw bf16 bits:
            -0 vs +0 and NaN payloads count as differences and are classified; per 1024-row
            slice counts localise any difference (C1c's slice boundaries, its last partial
            slice at 1056 / 1536 rows, the fused op's partial M block at 128 / 1056). Also a
            rerun of the served call (the op must be deterministic for any of this to mean
            anything), and two CONTROLS that are expected to differ: C1c (the grafted path:
            per-1024-row slices, w1 + SiLU and w3 2D matmuls with the decode compute config,
            bf16 mul, concat) and, with --explore, E3 (separate fp32 minimal_matmul on w1 and
            w3, fp32 silu and mul, one bf16 pack). C1c agreeing everywhere would mean this
            single-chip harness cannot see the divergence v178 saw, and fails the run.
  negative  weights B packed through the SAME cached program: the scratch equals B's packed
            bytes and B's served output, and differs from A's (runtime args are refreshed per
            call, as the model's 64 layers need); gate and up swapped must differ; a second pack
            adds no program-cache entry.
  timing    at 2048 rows: the pack (per call with a sync, and pipelined), the served fused call,
            and C1c - the C1e and C1c penalties per 2048-row chunk over 64 layers.

It runs minimal_matmul, not the TP2 all_gather_minimal_matmul_async: card B has no Ethernet. The
claim under test is that the scratch the fused op reads is the served weight, byte for byte and
spec for spec; the gathered TP2 run (a v178-style arm with QWEN_FAST_C1_EXACT=1 on M + A) is the
decisive gate for the model.
"""

import argparse
import hashlib
import json
import os
import statistics
import sys
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

DIM = 5120           # K: the model width (every chip holds all of K for w1 / w3)
HIDDEN_TP = 8704     # N of one projection per chip at TP2 (17408 / 2)
TP = 2
ROWS = (128, 256, 512, 1024, 1056, 1536, 2048)
SEEDS = (0, 1)
CHIPS = (0, 1)
SLICE = 1024         # C1c's _QWEN_C1_SLICE_ROWS
WEIGHT_STD = 0.02
LAYERS_PER_CHUNK = 64
# tp_common.all_gather_swiglu_prefill at TP2 (K_local 2560 -> agmm_k_block_size 8), grid (8, 9).
SERVED_BLOCKS = dict(M_block_size=8, K_block_size=8, N_block_size=16, subblock_h=1, subblock_w=4)
SERVED_GRID = (8, 9)
E3_BLOCKS = dict(M_block_size=8, K_block_size=8, N_block_size=8, subblock_h=1, subblock_w=4)
IMAGE_SOURCES = ('models/demos/blackhole/qwen36/tt/tp_common.py', 'models/demos/blackhole/qwen36/tt/mlp.py',
                 'models/tt_dit/utils/tensor.py')
SECTIONS = ('host', 'bytes', 'equality', 'negative', 'timing')
OP_DIR = '/bench/c1e'        # where run_card_b.sh mounts lever_n_m3native_patch.C1E_FILES

WATCHDOG = None


# ---------------------------------------------------------------------------------------------
# Pure helpers (CPU-tested by test_mlp_c1e_pack.py).
# ---------------------------------------------------------------------------------------------

def interleave(torch, w1, w3):
    """Per-chip tile-pair interleave: [K, N] x 2 -> [K, 2N], column tile 2t = w1 tile t, 2t + 1 = w3 tile t."""
    k, n = w1.shape
    return torch.stack((w1.reshape(k, n // 32, 32), w3.reshape(k, n // 32, 32)), dim=-2).reshape(k, 2 * n)


def ordered(torch, bits):
    """bf16 bit patterns (int16) -> integers whose difference is the ulp distance (+0 and -0 -> 0)."""
    bits = bits.to(torch.int32)
    magnitude = bits & 0x7FFF
    return torch.where(bits < 0, -magnitude, magnitude)


def bits_compare(torch, left, right, slice_rows=SLICE):
    """Raw-bit comparison of two bf16 [rows, cols] tensors: exact only when every 16-bit pattern is
    equal. Differences are classified: zero_sign (+0 vs -0), nan_payload (both NaN, different bits),
    nan_one_side, numeric (the rest, with the largest ulp distance among finite pairs)."""
    if left.shape != right.shape or left.dtype != torch.bfloat16 or right.dtype != torch.bfloat16:
        raise ValueError('two bf16 tensors of one shape required: %s %s vs %s %s'
                         % (left.dtype, tuple(left.shape), right.dtype, tuple(right.shape)))
    left, right = left.reshape(-1, left.shape[-1]), right.reshape(-1, right.shape[-1])
    a, b = left.view(torch.int16), right.view(torch.int16)
    differ = a != b
    nan_a, nan_b = torch.isnan(left), torch.isnan(right)
    zero_sign = differ & (left == 0) & (right == 0)
    nan_payload = differ & nan_a & nan_b
    nan_one_side = nan_a ^ nan_b
    numeric = differ & ~zero_sign & ~nan_payload & ~nan_one_side
    finite = numeric & torch.isfinite(left) & torch.isfinite(right)
    ulps = (ordered(torch, a) - ordered(torch, b)).abs()
    max_ulp = int(ulps[finite].max()) if bool(finite.any()) else 0
    rows = left.shape[0]
    per_slice = [int(differ[start:start + slice_rows].sum()) for start in range(0, rows, slice_rows)]
    first = None
    if bool(differ.any()):
        index = int(differ.reshape(-1).nonzero()[0])
        first = [index // left.shape[1], index % left.shape[1]]
    count = int(differ.sum())
    return dict(elements=int(differ.numel()), mismatches=count, exact=count == 0,
                zero_sign=int(zero_sign.sum()), nan_payload=int(nan_payload.sum()),
                nan_one_side=int(nan_one_side.sum()), numeric=int(numeric.sum()), max_ulp=max_ulp,
                per_slice=per_slice, first=first,
                nan_left=int(nan_a.sum()), nan_right=int(nan_b.sum()))


def make_x(torch, rows, seed, specials=True):
    """[1, 1, rows, DIM] bf16 ~ N(0, 1) (post-RMSNorm scale). With specials, four rows that probe the
    edge handling: all +0, all -0, one NaN element, and 256x magnitude."""
    generator = torch.Generator().manual_seed(1000 + seed)
    x = torch.randn((rows, DIM), generator=generator).to(torch.bfloat16)
    if specials and rows >= 4:
        x[0] = 0.0
        x[1] = -0.0
        x[2, 17] = float('nan')
        x[3] = x[3] * 256
    return x.reshape(1, 1, rows, DIM)


def make_weights(torch, prepare, seed, chips):
    """The served host path for a random layer: HF-layout gate / up [HIDDEN, DIM], transposed to [K, N]
    in bf16 as _build_gate_up and tp_common.shard_w do, packed by the image's prepare_for_fused_swiglu
    (ndev=TP, gate first), then chip c's column shards. Returns {chip: (w1, w3, packed)} and the
    host-identity result per chip."""
    generator = torch.Generator().manual_seed(2000 + seed)
    hidden = HIDDEN_TP * TP
    gate = (torch.randn((hidden, DIM), generator=generator) * WEIGHT_STD).to(torch.bfloat16)
    up = (torch.randn((hidden, DIM), generator=generator) * WEIGHT_STD).to(torch.bfloat16)
    gk, uk = gate.T.contiguous(), up.T.contiguous()
    packed = prepare(torch.cat([gk, uk], dim=-1), ndev=TP, gate_is_first=True)
    shards, identity = {}, {}
    for chip in chips:
        w1 = gk[:, chip * HIDDEN_TP:(chip + 1) * HIDDEN_TP].contiguous()
        w3 = uk[:, chip * HIDDEN_TP:(chip + 1) * HIDDEN_TP].contiguous()
        shard = packed[:, chip * 2 * HIDDEN_TP:(chip + 1) * 2 * HIDDEN_TP].contiguous()
        identity[chip] = bool(torch.equal(shard.view(torch.int16), interleave(torch, w1, w3).view(torch.int16)))
        shards[chip] = (w1, w3, shard)
    return shards, identity


def median_ms(samples):
    return round(statistics.median(samples) * 1e3, 4) if samples else None


def projections(timing):
    """C1e and C1c penalties per 2048-row prefill chunk (every layer has one MLP), from the single-chip
    timings. The gathered TP2 op is the served one in C1e, so its only extra work is the pack."""
    out = {}
    if timing.get('pack_ms') is not None:
        out['c1e_penalty_ms_per_chunk'] = round(timing['pack_ms'] * LAYERS_PER_CHUNK, 2)
    if timing.get('c1c_ms') is not None and timing.get('fused_ms') is not None:
        out['c1c_penalty_ms_per_chunk_gate_up_only'] = round((timing['c1c_ms'] - timing['fused_ms'])
                                                             * LAYERS_PER_CHUNK, 2)
    return out


def verdict(report):
    return not report['failures'] and report.get('cases_run', 0) > 0


# ---------------------------------------------------------------------------------------------
# Device harness.
# ---------------------------------------------------------------------------------------------

class Watchdog:
    """A per-device-call deadline: a hung NoC handshake cannot be interrupted from Python, so the
    poller prints WATCHDOG, writes the partial report and os._exit(3)s."""

    def __init__(self, seconds, on_fire=None):
        self.seconds, self.on_fire = seconds, on_fire
        self.label, self.deadline = None, None
        self.lock = threading.Lock()

    def start(self):
        if self.seconds:
            threading.Thread(target=self.poll, name='c1e-watchdog', daemon=True).start()
        return self

    @contextmanager
    def op(self, label):
        if not self.seconds:
            yield
            return
        with self.lock:
            outer = (self.label, self.deadline)
            self.label, self.deadline = label, time.monotonic() + self.seconds
        try:
            yield
        finally:
            with self.lock:
                self.label, self.deadline = outer

    def poll(self):
        while True:
            time.sleep(1.0)
            with self.lock:
                label, deadline = self.label, self.deadline
            if label is not None and time.monotonic() >= deadline:
                sys.stdout.write('WATCHDOG: %r did not return within %ss; exiting 3 (docker rm -f, then reset '
                                 'this card only, by the runner\'s printed reset command)\n' % (label, self.seconds))
                sys.stdout.flush()
                try:
                    if self.on_fire is not None:
                        self.on_fire(label)
                finally:
                    os._exit(3)


class Bench:
    def __init__(self, ttnn, torch, pack, tpc, device, op_dir):
        self.ttnn, self.torch, self.pack, self.tpc, self.device, self.op_dir = ttnn, torch, pack, tpc, device, op_dir
        self.dram = ttnn.DRAM_MEMORY_CONFIG
        # mlp.py: compute_kernel_config_agmm (served fused) and compute_kernel_config_decode (what C1c
        # passes: prefill x is 4D, so the model's T = x.shape[1] = 1 selects the decode config).
        self.ckc_agmm = ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.LoFi,
                                                         fp32_dest_acc_en=True, packer_l1_acc=False)
        self.ckc_decode = ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.LoFi,
                                                           fp32_dest_acc_en=True, packer_l1_acc=True)
        grid = ttnn.CoreCoord(*SERVED_GRID)
        self.served_cfg = ttnn.MinimalMatmulConfig(compute_with_storage_grid_size=grid, **SERVED_BLOCKS)
        self.e3_cfg = ttnn.MinimalMatmulConfig(compute_with_storage_grid_size=grid, **E3_BLOCKS)
        self.grid_w = device.compute_with_storage_grid_size().x   # args.decode_grid_w
        self.tuning = tpc.prefill_tuning(TP)                         # args.prefill_tuning
        # The model makes w1 / w3 (tp_common.shard_w) and w_gate_up (_build_gate_up) with
        # ShardTensorToMesh(dim=-1); on one chip that is one shard, but the placement metadata is the
        # served one. A device that refuses a mesh mapper falls back to a plain upload (recorded).
        try:
            self.mapper = ttnn.ShardTensorToMesh(device, dim=-1)
        except Exception as error:  # noqa: BLE001 - recorded in the report
            self.mapper, self.mapper_error = None, repr(error)
        else:
            self.mapper_error = None

    def weight(self, value):
        if self.mapper is not None:
            try:
                return self.ttnn.from_torch(value, dtype=self.ttnn.bfloat4_b, layout=self.ttnn.TILE_LAYOUT,
                                            device=self.device, memory_config=self.dram, mesh_mapper=self.mapper)
            except Exception as error:  # noqa: BLE001 - recorded in the report; the plain upload follows
                self.mapper, self.mapper_error = None, 'from_torch with the mapper failed: %r' % (error,)
        return self.ttnn.from_torch(value, dtype=self.ttnn.bfloat4_b, layout=self.ttnn.TILE_LAYOUT,
                                    device=self.device, memory_config=self.dram)

    def activation(self, value):
        return self.ttnn.from_torch(value, dtype=self.ttnn.bfloat16, layout=self.ttnn.TILE_LAYOUT,
                                    device=self.device, memory_config=self.dram)

    def fused(self, x, packed):
        with WATCHDOG.op('minimal_matmul fuse_swiglu'):
            return self.ttnn.experimental.minimal_matmul(x, packed, config=self.served_cfg,
                                                         compute_kernel_config=self.ckc_agmm,
                                                         dtype=self.ttnn.bfloat16, memory_config=self.dram,
                                                         fuse_swiglu=True)

    def pack_into(self, w1, w3, scratch):
        with WATCHDOG.op('mlp_c1e_pack'):
            return self.pack.pack_gate_up(self.device, w1, w3, scratch, operations=self.ttnn, directory=self.op_dir)

    def c1c(self, x, w1, w3):
        """mlp.py _qwen_c1_swiglu as grafted (lever_n_m3native_patch C1C_SWIGLU_HELPER), verbatim in effect."""
        ttnn, tpc = self.ttnn, self.tpc
        seq, rank = x.shape[-2], len(x.shape)
        parts = []
        with WATCHDOG.op('c1c'):
            for start in range(0, seq, SLICE):
                rows = min(SLICE, seq - start)
                if rows == seq:
                    part = x
                else:
                    begins = [0] * rank
                    ends = [x.shape[i] for i in range(rank)]
                    begins[rank - 2], ends[rank - 2] = start, start + rows
                    part = ttnn.slice(x, begins, ends)
                gate_config = tpc.create_prefill_mlp_matmul_program_config(
                    rows, DIM, w1.shape[-1], max_cols=self.grid_w, tuning=self.tuning,
                    fused_activation=ttnn.UnaryOpType.SILU)
                up_config = tpc.create_prefill_mlp_matmul_program_config(
                    rows, DIM, w3.shape[-1], max_cols=self.grid_w, tuning=self.tuning)
                gate = ttnn.linear(part, w1, compute_kernel_config=self.ckc_decode, program_config=gate_config,
                                   memory_config=self.dram)
                up = ttnn.linear(part, w3, compute_kernel_config=self.ckc_decode, program_config=up_config,
                                 memory_config=self.dram)
                if part is not x:
                    ttnn.deallocate(part)
                parts.append(ttnn.mul(gate, up, memory_config=self.dram))
                ttnn.deallocate(gate)
                ttnn.deallocate(up)
            if len(parts) == 1:
                return parts[0]
            joined = ttnn.concat(parts, dim=rank - 2, memory_config=self.dram)
            for part in parts:
                ttnn.deallocate(part)
            return joined

    def e3(self, x, w1, w3):
        """Separate fp32 minimal_matmul (N block 8 keeps each tile's N-block index), fp32 SiLU and
        multiply, one bf16 pack. Exploratory: exact only if the fused epilogue is these ops."""
        ttnn = self.ttnn
        with WATCHDOG.op('e3'):
            gate = ttnn.experimental.minimal_matmul(x, w1, config=self.e3_cfg, compute_kernel_config=self.ckc_agmm,
                                                    dtype=ttnn.float32, memory_config=self.dram)
            up = ttnn.experimental.minimal_matmul(x, w3, config=self.e3_cfg, compute_kernel_config=self.ckc_agmm,
                                                  dtype=ttnn.float32, memory_config=self.dram)
            act = ttnn.silu(gate, memory_config=self.dram)
            out = ttnn.mul(act, up, dtype=ttnn.bfloat16, memory_config=self.dram)
            for value in (gate, up, act):
                ttnn.deallocate(value)
            return out

    def host(self, tensor, free=False):
        with WATCHDOG.op('readback'):
            value = self.ttnn.to_torch(tensor)
        if free:
            self.ttnn.deallocate(tensor)
        return value.reshape(-1, value.shape[-1]).to(self.torch.bfloat16)

    def check(self, packed, separate, offset):
        owned = []
        try:
            with WATCHDOG.op('packed_weight_check'):
                result = self.pack.check_pairs(self.device, packed, separate, offset, owned, operations=self.ttnn,
                                               directory=self.op_dir)
                _, _, pages = self.pack.geometry(separate.shape, packed.shape)
                return self.pack.read_check(self.ttnn, result, pages)
        finally:
            for value in owned:
                self.ttnn.deallocate(value)

    def sync(self):
        with WATCHDOG.op('synchronize'):
            self.ttnn.synchronize_device(self.device)


def record(report, label, result, expect_exact):
    report['cases'].append(dict(label=label, expect_exact=expect_exact, **result))
    ok = result['exact'] if expect_exact else True
    print('%-58s exact=%s mismatches=%d zero_sign=%d nan_payload=%d numeric=%d max_ulp=%d slices=%s'
          % (label, result['exact'], result['mismatches'], result['zero_sign'], result['nan_payload'],
             result['numeric'], result['max_ulp'], result['per_slice']), flush=True)
    if not ok:
        report['failures'].append('%s: %d mismatched elements (first %s)' % (label, result['mismatches'], result['first']))


def bytes_section(bench, report, tensors, scratch, chip, tag):
    w1, w3, packed = tensors
    checks = {}
    for name, target in (('served', packed), ('scratch', scratch)):
        for offset, separate in ((0, w1), (1, w3)):
            result = bench.check(target, separate, offset)
            checks['%s_vs_%s' % (name, ('w1', 'w3')[offset])] = result
            if not all(item['exact'] for item in result):
                report['failures'].append('%s chip %d: %s pages differ from %s: %s'
                                          % (tag, chip, name, ('w1', 'w3')[offset], result))
    spec_served, spec_scratch = bench.pack.spec(packed), bench.pack.spec(scratch)
    checks['spec_equal'] = spec_served == spec_scratch
    checks['spec'] = spec_served
    if not checks['spec_equal']:
        report['failures'].append('%s chip %d: scratch spec %s != served %s' % (tag, chip, spec_scratch, spec_served))
    report['bytes'].append(dict(tag=tag, chip=chip, **checks))
    print('bytes %s chip %d: %s' % (tag, chip, {k: v for k, v in checks.items() if k != 'spec'}), flush=True)


def run(args, report):
    import torch
    import ttnn

    sys.path.insert(0, str(args.op_dir))
    sys.path.insert(0, os.environ.get('TT_METAL_HOME', '/opt/tt-metal'))   # the image's models package
    import mlp_c1e_pack as pack
    from models.demos.blackhole.qwen36.tt import tp_common as tpc
    from models.tt_dit.utils.tensor import prepare_for_fused_swiglu

    root = Path(os.environ.get('TT_METAL_HOME', '/opt/tt-metal'))
    report['image_sources'] = {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
                               for name in IMAGE_SOURCES if (root / name).is_file()}
    report['op_files'] = {name: hashlib.sha256((Path(args.op_dir) / name).read_bytes()).hexdigest()
                          for name in pack.RUNTIME_FILES}
    print('image sources %s' % {k: v[:8] for k, v in report['image_sources'].items()}, flush=True)

    shards_a, identity_a = make_weights(torch, prepare_for_fused_swiglu, args.weight_seed, args.chips)
    shards_b, identity_b = make_weights(torch, prepare_for_fused_swiglu, args.weight_seed + 1, args.chips[:1])
    report['host_identity'] = dict(a={str(k): v for k, v in identity_a.items()},
                                   b={str(k): v for k, v in identity_b.items()})
    if 'host' in args.sections and (not all(identity_a.values()) or not all(identity_b.values())):
        report['failures'].append('the image prepare_for_fused_swiglu shard is not the per-chip tile-pair '
                                  'interleave of w1 / w3: %s' % report['host_identity'])
    print('host identity %s' % report['host_identity'], flush=True)

    device = ttnn.open_device(device_id=args.device_id)
    try:
        grid = device.compute_with_storage_grid_size()
        report['grid'] = [grid.x, grid.y]
        bench = Bench(ttnn, torch, pack, tpc, device, args.op_dir)
        report['mesh_mapper'] = 'ShardTensorToMesh(dim=-1)' if bench.mapper is not None else bench.mapper_error
        like = SimpleNamespace(shape=(DIM, HIDDEN_TP))
        try:
            scratch = pack.allocate_scratch(ttnn, device, like, served_topology=not args.plain_scratch)
            report['scratch_allocation'] = 'plain empty' if args.plain_scratch else 'served topology'
        except Exception as error:  # noqa: BLE001 - recorded; the plain allocation still tests the copy
            report['scratch_allocation'] = 'plain empty (served topology failed: %r)' % (error,)
            report['failures'].append('the served-topology scratch allocation (the model path) failed: %r' % (error,))
            scratch = pack.allocate_scratch(ttnn, device, like, served_topology=False)
        print('scratch: %s, weights: %s' % (report['scratch_allocation'], report['mesh_mapper']), flush=True)
        read, written = pack.traffic_bytes((DIM, HIDDEN_TP))
        report['copy'] = dict(read_bytes=read, written_bytes=written, scratch_bytes=written,
                              pairs=pack.geometry((DIM, HIDDEN_TP))[2])
        device_b = {chip: tuple(bench.weight(value) for value in values) for chip, values in shards_b.items()}
        for chip in args.chips:
            tensors = tuple(bench.weight(value) for value in shards_a[chip])
            w1, w3, packed = tensors
            bench.pack_into(w1, w3, scratch)
            if 'bytes' in args.sections:
                bytes_section(bench, report, tensors, scratch, chip, 'A')
            if 'equality' in args.sections:
                for rows in args.rows:
                    for seed in args.seeds:
                        x = bench.activation(make_x(torch, rows, seed, specials=not args.no_specials))
                        tag = 'chip%d rows%d seed%d' % (chip, rows, seed)
                        served = bench.fused(x, packed)
                        reference = bench.host(served)
                        bench.pack_into(w1, w3, scratch)
                        candidate = bench.fused(x, scratch)
                        record(report, tag + ' C1e vs served', bits_compare(torch, bench.host(candidate), reference), True)
                        rerun = bench.fused(x, packed)
                        record(report, tag + ' served rerun', bits_compare(torch, bench.host(rerun), reference), True)
                        control = bench.c1c(x, w1, w3)
                        result = bits_compare(torch, bench.host(control), reference)
                        record(report, tag + ' C1c vs served (control)', result, False)
                        report['c1c_mismatch_cases'] += int(not result['exact'])
                        values = [served, candidate, rerun, control]
                        if args.explore:
                            try:
                                explored = bench.e3(x, w1, w3)
                                record(report, tag + ' E3 vs served (explore)',
                                       bits_compare(torch, bench.host(explored), reference), False)
                                values.append(explored)
                            except Exception as error:  # noqa: BLE001 - exploratory arm: recorded, not fatal
                                report['e3_errors'].append('%s: %r' % (tag, error))
                                print('E3 %s: %r' % (tag, error), flush=True)
                        for value in values + [x]:
                            ttnn.deallocate(value)
                        report['cases_run'] += 1
            if 'negative' in args.sections and chip == args.chips[0]:
                negative_section(bench, report, tensors, device_b[chip], scratch, chip, args)
            if 'timing' in args.sections and not args.no_timing and chip == args.chips[0]:
                timing_section(bench, report, tensors, scratch, args)
            for value in tensors:
                ttnn.deallocate(value)
        if 'equality' in args.sections and report['cases_run'] and report['c1c_mismatch_cases'] == 0:
            report['failures'].append('C1c matched the served formulation in every case: this harness cannot '
                                      'see the divergence v178 saw, so its C1e verdict proves nothing')
        report['descriptor_cache'] = pack.cache_size()
        report['mesh_mapper'] = 'ShardTensorToMesh(dim=-1)' if bench.mapper is not None else bench.mapper_error
        pack.release_scratch(ttnn, device)
    finally:
        with WATCHDOG.op('close device'):
            ttnn.close_device(device)


def negative_section(bench, report, tensors_a, tensors_b, scratch, chip, args):
    import torch

    ttnn = bench.ttnn
    w1a, w3a, packed_a = tensors_a
    w1b, w3b, packed_b = tensors_b
    rows = max(args.rows)
    x = bench.activation(make_x(torch, rows, 7, specials=False))
    served_a = bench.host(bench.fused(x, packed_a), free=True)
    served_b = bench.host(bench.fused(x, packed_b), free=True)
    bench.pack_into(w1a, w3a, scratch)
    bench.sync()
    entries = bench.device.num_program_cache_entries()
    bench.pack_into(w1b, w3b, scratch)          # the same cached program, B's addresses
    bench.sync()
    new_entries = bench.device.num_program_cache_entries() - entries
    negative = dict(program_cache_new_entries=new_entries)
    if new_entries:
        report['failures'].append('a second pack added %d program-cache entries' % new_entries)
    for offset, separate in ((0, w1b), (1, w3b)):
        result = bench.check(scratch, separate, offset)
        negative['refresh_vs_%s' % ('w1', 'w3')[offset]] = result
        if not all(item['exact'] for item in result):
            report['failures'].append('scratch after packing B differs from B (stale runtime args?): %s' % result)
    candidate = bench.host(bench.fused(x, scratch), free=True)
    record(report, 'negative chip%d rows%d packed B vs served B' % (chip, rows), bits_compare(torch, candidate, served_b), True)
    stale = bits_compare(torch, candidate, served_a)
    negative['packed_b_vs_served_a_mismatches'] = stale['mismatches']
    if stale['exact']:
        report['failures'].append('packed B equals served A: the comparison cannot tell two layers apart')
    bench.pack_into(w3a, w1a, scratch)          # gate and up swapped
    swapped = bits_compare(torch, bench.host(bench.fused(x, scratch), free=True), served_a)
    negative['swapped_mismatches'] = swapped['mismatches']
    if swapped['exact']:
        report['failures'].append('gate / up swapped still equals served: the comparison is blind to the packing')
    swapped_bytes = bench.check(scratch, w1a, 0)
    negative['swapped_bytes_vs_w1_exact'] = [item['exact'] for item in swapped_bytes]
    if any(item['exact'] for item in swapped_bytes):
        report['failures'].append('the byte check did not see gate / up swapped: %s' % swapped_bytes)
    report['negative'] = negative
    print('negative %s' % negative, flush=True)
    ttnn.deallocate(x)


def timing_section(bench, report, tensors, scratch, args):
    import torch

    ttnn = bench.ttnn
    w1, w3, packed = tensors
    x = bench.activation(make_x(torch, 2048, 11, specials=False))

    def per_call(fn):
        samples = []
        for index in range(args.warmup + args.iters):
            bench.sync()
            start = time.perf_counter()
            value = fn()
            bench.sync()
            if index >= args.warmup:
                samples.append(time.perf_counter() - start)
            if value is not None and value is not scratch:
                ttnn.deallocate(value)
        return samples

    pack_samples = per_call(lambda: bench.pack_into(w1, w3, scratch))
    bench.sync()
    start = time.perf_counter()
    for _ in range(args.iters):
        bench.pack_into(w1, w3, scratch)
    bench.sync()
    pipelined = (time.perf_counter() - start) / args.iters
    timing = dict(rows=2048, iters=args.iters, pack_ms=median_ms(pack_samples),
                  pack_pipelined_ms=round(pipelined * 1e3, 4),
                  fused_ms=median_ms(per_call(lambda: bench.fused(x, packed))),
                  c1c_ms=median_ms(per_call(lambda: bench.c1c(x, w1, w3))))
    read, written = bench.pack.traffic_bytes((DIM, HIDDEN_TP))
    if timing['pack_pipelined_ms']:
        timing['pack_gb_per_s'] = round((read + written) / (timing['pack_pipelined_ms'] * 1e-3) / 1e9, 1)
    timing.update(projections(dict(timing, pack_ms=timing['pack_pipelined_ms'])))
    report['timing'] = timing
    print('timing %s' % timing, flush=True)
    ttnn.deallocate(x)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split(chr(10))[0])
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--op-dir', type=Path, default=Path(OP_DIR))
    parser.add_argument('--device-id', type=int, default=0)
    parser.add_argument('--rows', default=','.join(map(str, ROWS)))
    parser.add_argument('--seeds', default=','.join(map(str, SEEDS)))
    parser.add_argument('--chips', default=','.join(map(str, CHIPS)), help='which TP2 shards of the random layer to emulate')
    parser.add_argument('--weight-seed', type=int, default=0)
    parser.add_argument('--sections', default=','.join(SECTIONS))
    parser.add_argument('--explore', action='store_true', help='also run E3 (separate fp32 matmuls, fp32 SwiGLU)')
    parser.add_argument('--no-specials', action='store_true', help='no +0 / -0 / NaN / 256x rows in x')
    parser.add_argument('--plain-scratch', action='store_true',
                        help='allocate the scratch with ttnn.empty instead of the served ShardTensorToMesh path')
    parser.add_argument('--quick', action='store_true', help='rows 2048 and 1056, seed 0, chip 0 (the watcher pass)')
    parser.add_argument('--warmup', type=int, default=3)
    parser.add_argument('--iters', type=int, default=20)
    parser.add_argument('--no-timing', action='store_true')
    parser.add_argument('--watchdog', type=float, default=0)
    args = parser.parse_args(argv)
    if args.quick:
        args.rows, args.seeds, args.chips = '2048,1056', '0', '0'
    args.rows = [int(value) for value in args.rows.split(',') if value]
    args.seeds = [int(value) for value in args.seeds.split(',') if value]
    args.chips = [int(value) for value in args.chips.split(',') if value]
    args.sections = [value for value in args.sections.split(',') if value]
    unknown = set(args.sections) - set(SECTIONS)
    if unknown or not args.rows or any(rows % 32 or rows <= 32 for rows in args.rows) \
            or not args.chips or any(chip not in range(TP) for chip in args.chips):
        parser.error('unknown section %s, a row count that is not a multiple of 32 above 32 (the fused '
                     'path takes rows > 32 only), or a chip outside 0..%d' % (sorted(unknown), TP - 1))
    return args


def main(argv=None):
    global WATCHDOG
    args = parse_args(argv)
    report = dict(passed=False, failures=[], cases=[], bytes=[], negative={}, timing={}, e3_errors=[],
                  cases_run=0, c1c_mismatch_cases=0, args={k: str(v) for k, v in vars(args).items()})
    args.out.parent.mkdir(parents=True, exist_ok=True)

    def write(extra=None):
        payload = dict(report)
        if extra:
            payload.update(extra)
        args.out.write_text(json.dumps(payload, indent=2, default=str))

    WATCHDOG = Watchdog(args.watchdog, on_fire=lambda label: write(dict(error='watchdog: %r' % label))).start()
    try:
        run(args, report)
        report['passed'] = verdict(report)
    except Exception as error:  # noqa: BLE001 - recorded, then re-raised for the exit status
        report['error'] = repr(error)
        write()
        raise
    write()
    print('PASSED' if report['passed'] else 'FAILED: %d failures' % len(report['failures']), flush=True)
    for failure in report['failures'][:40]:
        print('  ' + failure, flush=True)
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    sys.exit(main())
