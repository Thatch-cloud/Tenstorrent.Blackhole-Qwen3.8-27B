"""CPU checks for Q0, the Q4 card-B probe (probe_quad_draft_card_b.py, quad_candidates.py, quad_conv_io.cpp,
run_card_b.sh); no device, no ttnn.

  - the pure helpers: fixtures and regimes, the quad K/V plan (its pads are the pair assemblies' bytes, R2), the
    fold's head arithmetic, the dense control mask, the conv seam words and page split (E1b: 100 workers x 3, 10 x 2,
    every page once), the bitwise comparison, the device-log parser, the saving model and every step's verdict table
    on synthetic reports, the verdict lines and the JSON summary;
  - the contract: the pinned SDPA sources are the T16 admission's ORIGINAL; the embedded qual_card block is the
    canonical scripts/ci/qual_card.sh byte for byte; every file is LF;
  - a full run of every step against a torch stand-in for ttnn, and one fault per control, each giving the verdict
    the control exists for:
      Q0a  exact -> PASS; a tile-row-1 matmul residue at per_core_M=2 -> FAIL and kill=STOP-Q4; a norm residue ->
           FAIL without a kill; a lossless K accumulation (in0_block_w changes nothing) -> INCONCLUSIVE under the
           plan's control, PASS under --matmul-control either (the K-split control); duplicated halves plus an op
           that reads tile row 0 twice -> INCONCLUSIVE (the perturbation control); partial coverage -> INCONCLUSIVE;
           the cores pass -> norm cores 2 / 32 / 8, and 1 when the norm runs on one core;
      Q0b  an SDPA with the M1 residue -> PASS (the dense control fires); exact arithmetic -> INCONCLUSIVE (it
           cannot fire); a quad plan with the branch's row-0 pads (R2) -> FAIL on C5; a faulty 12-piece concat ->
           FAIL on C5; a wrong unfold -> FAIL; the FREED-VIEW hazard (a retainer that owns views of its inputs)
           -> case errors, never a pass;
      Q0c  exact -> PASS; a corrupting uint16 concat -> PASS on the uint32 fallback; a duplicating concat -> FAIL;
           identical halves -> INCONCLUSIVE (the order control);
      Q0d  exact -> PASS (conv=110); a kernel that reads tile row 0's seam word -> INCONCLUSIVE (the seam control);
           tile row 0's dynamic kernel -> FAIL; a compute kernel expecting the wrong page count -> the stand-in's
           hang, case errors; a grid too small for E1b -> INCONCLUSIVE with conv=80; an E1b kernel that writes
           nothing, on a first-fit DRAM that hands it E1's freed outputs -> FAIL (it PASSED before the poison);
      Q0e  every arm timed and the model computed; a short run is INCONCLUSIVE; the model counts only what the
           run proved (a failed E1b is not credited with E1b's saving; a failed op leaves its item);
      Q0b  a non-finite single-user reference is a failure and case errors, never a pass; a recorded failure is
           never a go;
  - the runner (needs bash): the dry run launches on card B by board id with the K64i graft mounts and sha, image
    P6, every module at /bench and nothing over /experiment-scripts/ci, the cores pass first; the watcher pass;
    KOPGRAFT64=none; the pre-launch graft checks; bash -n;
  - the kernel (needs a g++): quad_conv_io.cpp and the served I/O kernel compile against the same API stub.

    py -3.11 -B -m unittest discover -s optimisation/ttnn-op/quad_draft_probe -p 'test_*.py'
"""

import hashlib
import itertools
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest import mock

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
CI = ROOT / 'scripts' / 'ci'
for path in (str(CI), str(HERE)):
    if path not in sys.path:
        sys.path.insert(0, path)

import torch  # noqa: E402

import probe_quad_draft_card_b as probe  # noqa: E402
import quad_candidates as quad  # noqa: E402

RUNNER = HERE / 'run_card_b.sh'
CARD_B = 'blackhole-F36F768B9A5CAFA0'
NL = chr(10)
# K spans at least four in0 blocks at these widths, so the K-split control (two halves added in fp32) changes the
# stand-in's association as the device's does at K = 5120 (40 blocks); at two blocks the two would coincide.
SMALL = dict(HIDDEN=512, INTERMEDIATE=768, Q_WIDTH=512, KV_WIDTH=128, CONV_COLUMNS=128, SELECTOR_COLUMNS=64,
             EMBED_WIDTH=128)
ALL_A = ','.join(probe.REGIMES_A)
ALL_B = ','.join(probe.REGIMES_B)


def bits16(tensor):
    return tensor.contiguous().to(torch.bfloat16).view(torch.int16)


# ---------------------------------------------------------------------------------------------
# A torch stand-in for the ttnn surface the harness, the candidates and the served modules use.
# ---------------------------------------------------------------------------------------------

class FakeTensor:
    """A device tensor. `base` is its buffer: a reshape is a view that shares its source's buffer, as in ttnn, so
    deallocating the view frees the source (the pair probe's card-B watcher pass lost its K/V that way)."""

    buffers = itertools.count(1)

    def __init__(self, value, dtype, layout='tile', base=None):
        self.value, self.dtype, self.layout = value, dtype, layout
        self.shape = tuple(value.shape)
        self.base = next(FakeTensor.buffers) if base is None else base

    def memory_config(self):
        return 'dram'

    def buffer_address(self):
        return self.base


class CoreSet:
    def __init__(self, ranges):
        self.cores = set()
        for start, end in ranges:
            for x in range(start.x, end.x + 1):
                for y in range(start.y, end.y + 1):
                    self.cores.add((x, y))


class Runtime:
    """ttnn.RuntimeArgs: runtime[x][y] = [...]."""

    def __init__(self):
        self.cores = {}

    def __getitem__(self, x):
        owner = self

        class Column:
            def __setitem__(self, y, value):
                owner.cores[(x, y)] = list(value)
        return Column()


class Kernel:
    def __init__(self, kernel_source, core_ranges, compile_time_args=None, runtime_args=None, config=None):
        self.kernel_source, self.core_ranges = kernel_source, core_ranges
        self.compile_time_args, self.runtime_args, self.config = list(compile_time_args or []), runtime_args, config


class DataMovement(SimpleNamespace):
    pass


class Compute(SimpleNamespace):
    pass


class MeshProgram(dict):
    pass


class FakeTtnn:
    """ttnn on torch. Row-local ops run per 32-row tile, so a row's result never depends on its block's position;
    matmul accumulates K in blocks of in0_block_w tiles (so in0_block_w changes fp32 bits); the SDPA is a GQA
    reference in float64 per (batch, head); generic_op interprets the two conv I/O kernels from their runtime args,
    checks that every core's compute kernel expects exactly the pages its reader walks (else the real kernel would
    hang) and that the pages cover the block once. `modes` injects the faults the controls guard against:
      m1             SDPA rows whose first visible key lies past the first 32-key chunk come out scaled
      lossless       matmul accumulates K in float64: in0_block_w changes nothing
      row1           matmul at per_core_M=2 scales tile row 1
      norm-row1      rms_norm at 64 rows computes tile row 1 with another epsilon
      norm-1core     the device log shows every rms_norm on one core
      row0-twice     silu at 64 rows returns tile row 0's result for both tile rows
      dm             a 12-piece concat (the quad K/V assembly) flips one element
      concat-u16     a uint16 concat corrupts one index
      concat-dup     a candidate concat (last dim 16) repeats its first part
      conv-seam-low  the quad conv kernel reads tile row 0's seam word for every tile row
      conv-dyn-tile  the quad conv kernel reads tile row 0's dynamic kernel for every tile row
      c0-nan         the single-user SDPA (16 heads over 2080 keys) returns NaN
      stale-dram     DRAM as a first-fit allocator leaves it: an upload or ttnn.empty takes the oldest freed buffer
                     of its shape, and ttnn.empty reads back that buffer's old bytes (an upload overwrites them)
      e1b-no-write   the quad conv kernel on 110 workers writes no output page
      profiler-at-close  ReadDeviceProfiler writes nothing; the device log appears at close_device (P6's runtime),
                     each launch at its wall-clock time in cycles"""

    bfloat16, bfloat8_b, float32, uint32, uint16 = 'bf16', 'bf8', 'fp32', 'u32', 'u16'
    TILE_LAYOUT, ROW_MAJOR_LAYOUT, DRAM_MEMORY_CONFIG = 'tile', 'row', 'dram'
    MathFidelity = SimpleNamespace(HiFi4='hifi4')
    DataMovementProcessor = SimpleNamespace(RISCV_0='riscv0')
    NOC = SimpleNamespace(RISCV_0_default='noc0')

    def __init__(self, modes=(), grid=(13, 10)):
        self.modes, self.grid = set(modes), grid
        self.calls, self.freed, self.captures = [], 0, 0
        self.freed_buffers = set()
        self.stale = []
        self.launches, self.flushed = [], 0
        self.host_ids = itertools.count(1)
        self.transformer = SimpleNamespace(scaled_dot_product_attention=self.sdpa)
        self.experimental = SimpleNamespace(rotary_embedding_hf=self.rotary, nlp_create_qkv_heads=self.create_heads,
                                            nlp_concat_heads=self.concat_heads)
        self.device = None

    # device, trace and profiler
    def open_device(self, **options):
        self.device = SimpleNamespace(options=options, enable_program_cache=lambda: None,
                                      compute_with_storage_grid_size=lambda: SimpleNamespace(x=self.grid[0],
                                                                                              y=self.grid[1]))
        return self.device

    def close_device(self, device):
        self.calls.append('close_device')
        if 'profiler-at-close' in self.modes:
            self.write_profile()

    def synchronize_device(self, device):
        pass

    def begin_trace_capture(self, device, cq_id=0):
        self.captures += 1
        return SimpleNamespace(kind='trace')

    def end_trace_capture(self, device, trace, cq_id=0):
        pass

    def execute_trace(self, device, trace, cq_id=0, blocking=True):
        pass

    def release_trace(self, device, trace):
        pass

    def launch(self, cores):
        self.launches.append((next(self.host_ids), max(1, int(cores)), time.perf_counter()))

    def ReadDeviceProfiler(self, device):
        if 'profiler-at-close' not in self.modes:
            self.write_profile()

    def write_profile(self):
        root = os.environ.get('TT_METAL_PROFILER_DIR')
        if not root:
            return
        path = Path(root) / '.logs' / 'profile_log_device.csv'
        path.parent.mkdir(parents=True, exist_ok=True)
        fresh = not path.exists()
        with path.open('a') as handle:
            if fresh:
                handle.write('ARCH: blackhole, CHIP_FREQ[MHz]: 1350' + NL)
                handle.write('PCIe slot, core_x, core_y, RISC processor type, timer_id, time[cycles since reset], '
                             'data, run host ID, zone name, type, source line, source file' + NL)
            for host_id, cores, when in self.launches[self.flushed:]:
                for index in range(cores):
                    for risc in ('BRISC', 'TRISC_0'):
                        handle.write('0, %d, %d, %s, 1, %d, 0, %d, %s-FW, ZONE_START, 1, fw.cpp%s'
                                     % (index % 13, index // 13, risc, int(when * 1350e6), host_id, risc, NL))
        self.flushed = len(self.launches)

    # configuration objects
    def WormholeComputeKernelConfig(self, **options):
        return ('kernel', tuple(sorted(options.items())))

    def SDPAProgramConfig(self, **options):
        return ('program', tuple(sorted(options.items())))

    def MatmulMultiCoreReuseMultiCast1DProgramConfig(self, **options):
        return ('matmul', tuple(sorted(options.items())))

    def CoreCoord(self, x, y):
        return SimpleNamespace(x=x, y=y)

    def CoreRange(self, start, end):
        return (start, end)

    def CoreRangeSet(self, ranges):
        return CoreSet(ranges)

    def CBDescriptor(self, **options):
        return SimpleNamespace(**options)

    def CBFormatDescriptor(self, **options):
        return SimpleNamespace(**options)

    def TileDescriptor(self, tile):
        return tile

    def Tile(self, shape):
        return tuple(shape)

    def KernelDescriptor(self, **options):
        return Kernel(**options)

    def ComputeConfigDescriptor(self, **options):
        return Compute(**options)

    def DataMovementConfigDescriptor(self, **options):
        return DataMovement(**options)

    def RuntimeArgs(self):
        return Runtime()

    def TensorAccessorArgs(self, value):
        return SimpleNamespace(get_compile_time_args=lambda: [0])

    def MeshProgramDescriptor(self):
        return MeshProgram()

    def MeshCoordinate(self, *coordinate):
        return tuple(coordinate)

    def MeshCoordinateRange(self, start, end):
        return (start, end)

    def ProgramDescriptor(self, kernels, cbs):
        return SimpleNamespace(kernels=kernels, cbs=cbs)

    # data
    def cast(self, value, dtype):
        if dtype == self.float32:
            return value.float()
        if dtype in (self.bfloat16, self.bfloat8_b):
            return value.to(torch.bfloat16) if value.is_floating_point() else value
        return value.to(torch.int64)

    def reuse(self, shape):
        """stale-dram: the oldest freed buffer of this shape's old bytes, taken off the free list (or None)."""
        if 'stale-dram' not in self.modes:
            return None
        for index, (freed_shape, value) in enumerate(self.stale):
            if freed_shape == tuple(shape):
                del self.stale[index]
                return value
        return None

    def from_torch(self, value, dtype=None, layout=None, device=None, memory_config=None):
        self.reuse(tuple(value.shape))
        return FakeTensor(self.cast(value.clone(), dtype or self.bfloat16), dtype or self.bfloat16, layout)

    def empty(self, shape, dtype=None, layout=None, device=None, memory_config=None):
        old = self.reuse(tuple(shape))
        return FakeTensor(torch.zeros(shape, dtype=torch.bfloat16) if old is None else old.clone(), dtype, layout)

    def get_device_tensors(self, tensor):
        return [tensor]

    def live(self, *tensors):
        for tensor in tensors:
            if tensor.base in self.freed_buffers:
                raise RuntimeError('TT_THROW: Tensor is not allocated')

    def to_torch(self, tensor):
        self.live(tensor)
        return tensor.value.clone()

    def deallocate(self, tensor):
        self.freed += 1
        if 'stale-dram' in self.modes and tensor.base not in self.freed_buffers:
            self.stale.append((tuple(tensor.shape), tensor.value.clone()))
        self.freed_buffers.add(tensor.base)

    def slice(self, tensor, start, end, steps=None):
        self.live(tensor)
        index = tuple(slice(low, high) for low, high in zip(start, end))
        self.launch(1)
        return FakeTensor(tensor.value[index].clone(), tensor.dtype)

    def concat(self, parts, dim, memory_config=None):
        self.live(*parts)
        values = [part.value for part in parts]
        if 'concat-dup' in self.modes and dim == 2 and parts[0].shape[-1] == 16:
            values = [values[0]] * len(values)
        value = torch.cat(values, dim=dim).clone()
        if 'dm' in self.modes and len(parts) == 12:
            value.view(-1)[12345] = value.view(-1)[12345] + 1
        if 'concat-u16' in self.modes and parts[0].dtype == self.uint16 and dim == 2:
            value[..., 40, 3] = value[..., 40, 3] + 1
        self.launch(1)
        return FakeTensor(value, parts[0].dtype)

    def reshape(self, tensor, shape):
        # ttnn's rule: a tile-layout reshape is a view (the same buffer) when the last dim is kept and the
        # second-last dims are equal or both tile-aligned; otherwise it is a copy.
        self.live(tensor)
        old, new = tuple(tensor.shape), tuple(shape)
        view = (tensor.layout == 'tile' and old[-1] == new[-1]
                and (old[-2] == new[-2] or (old[-2] % 32 == 0 and new[-2] % 32 == 0)))
        return FakeTensor(tensor.value.reshape(shape), tensor.dtype, tensor.layout, base=tensor.base if view else None)

    def pad(self, tensor, padding, value):
        pads = []
        for low, high in reversed(padding):
            pads.extend([low, high])
        return FakeTensor(torch.nn.functional.pad(tensor.value.float(), pads, value=value).to(tensor.value.dtype),
                          tensor.dtype)

    def typecast(self, tensor, dtype):
        self.live(tensor)
        self.launch(1)
        return FakeTensor(self.cast(tensor.value, dtype), dtype)

    # compute
    @staticmethod
    def tiles(value, fn):
        rows = value.shape[-2]
        return torch.cat([fn(value[..., start:start + 32, :]) for start in range(0, rows, 32)], dim=-2)

    def matmul(self, left, right, dtype=None, compute_kernel_config=None, program_config=None, memory_config=None):
        self.live(left, right)
        config = dict(program_config[1]) if program_config else {}
        block = config.get('in0_block_w', 4) * 32
        weight = right.value.float()

        def tile(part):
            part = part.float()
            if 'lossless' in self.modes:
                return (part.double() @ weight.double()).float()
            total = torch.zeros(part.shape[:-1] + (weight.shape[-1],))
            for start in range(0, weight.shape[0], block):
                total = total + part[..., start:start + block] @ weight[start:start + block]
            return total
        out = self.tiles(left.value, tile)
        if 'row1' in self.modes and config.get('per_core_M') == 2:
            out[..., 32:, :] = out[..., 32:, :] * (1 + 2 ** -10)
        grid = config.get('compute_with_storage_grid_size', (1, 1))
        self.launch(grid[0] * grid[1])
        return FakeTensor(out, self.float32)

    def linear(self, left, right):
        return FakeTensor(self.tiles(left.value, lambda part: part.float() @ right.value.float()).bfloat16(),
                          self.bfloat16)

    def topk(self, tensor, k, dim=-1, largest=True, sorted=True):
        values, indices = torch.topk(tensor.value.float(), k, dim=dim, largest=largest, sorted=sorted)
        self.launch(1)
        return FakeTensor(values.bfloat16(), self.bfloat16), FakeTensor(indices.to(torch.int64), self.uint16)

    def rms_norm(self, tensor, epsilon, weight, compute_kernel_config=None, memory_config=None):
        self.live(tensor, weight)
        scale = weight.value.float().reshape(-1)
        rows = tensor.shape[-2]

        def norm(part, eps=epsilon):
            part = part.float()
            return part * torch.rsqrt(part.pow(2).mean(dim=-1, keepdim=True) + eps) * scale
        out = self.tiles(tensor.value, norm)
        if 'norm-row1' in self.modes and rows == 64:
            out[..., 32:, :] = norm(tensor.value[..., 32:, :], 2 * epsilon)
        heads = 1
        for size in tensor.shape[:-2]:
            heads *= size
        self.launch(1 if 'norm-1core' in self.modes else heads * ((rows + 31) // 32))
        return FakeTensor(self.cast(out, tensor.dtype), tensor.dtype)

    def rotary(self, value, cosine, sine, is_decode_mode=False, compute_kernel_config=None, memory_config=None):
        x = value.value.float()
        half = x.shape[-1] // 2
        rotated = torch.cat([-x[..., half:], x[..., :half]], dim=-1)
        self.launch(1)
        return FakeTensor(x * cosine.value.float() + rotated * sine.value.float(), self.float32)

    def create_heads(self, query, combined, num_heads, num_kv_heads, transpose_k_heads=False, memory_config=None):
        self.live(query, combined)
        rows = query.shape[2]
        q = query.value.reshape(1, rows, num_heads, 128).transpose(1, 2).contiguous()
        kv = combined.value.reshape(1, rows, 2 * num_kv_heads, 128).transpose(1, 2).contiguous()
        self.launch(1)
        return [FakeTensor(q, query.dtype), FakeTensor(kv[:, :num_kv_heads].contiguous(), query.dtype),
                FakeTensor(kv[:, num_kv_heads:].contiguous(), query.dtype)]

    def concat_heads(self, value, memory_config=None):
        self.live(value)
        _, heads, rows, width = value.shape
        self.launch(1)
        return FakeTensor(value.value.transpose(1, 2).reshape(1, 1, rows, heads * width).contiguous(), value.dtype)

    def embedding(self, identifiers, table, layout=None, memory_config=None):
        self.live(identifiers, table)
        self.launch(1)
        return FakeTensor(table.value[identifiers.value.long()].clone(), self.bfloat16)

    def silu(self, tensor, memory_config=None):
        out = self.tiles(tensor.value, lambda part: torch.nn.functional.silu(part.float()))
        if 'row0-twice' in self.modes and tensor.shape[-2] == 64:
            out[..., 32:, :] = out[..., :32, :]
        self.launch(1)
        return FakeTensor(out, self.float32)

    def multiply(self, left, right, dtype=None):
        self.launch(1)
        return FakeTensor(left.value.float() * right.value.float(), self.float32)

    def add(self, left, right, dtype=None, memory_config=None):
        self.live(left, right)
        self.launch(1)
        return FakeTensor(left.value.float() + right.value.float(), self.float32)

    def sdpa(self, query, key, value, *, attn_mask, is_causal, scale, program_config, compute_kernel_config,
             memory_config):
        self.live(query, key, value, attn_mask)
        q, k, v, mask = query.value, key.value, value.value, attn_mask.value
        batches, heads, rows = q.shape[0], q.shape[1], q.shape[2]
        kv_heads = k.shape[1]
        out = torch.empty((batches, heads, rows, v.shape[-1]))
        for batch in range(batches):
            for head in range(heads):
                kv = head // (heads // kv_heads)
                head_mask = mask[batch if mask.shape[0] > 1 else 0, head if mask.shape[1] > 1 else 0].float()
                scores = q[batch, head].double() @ k[batch, kv].double().T * scale + head_mask.double()
                result = (torch.softmax(scores, dim=-1) @ v[batch, kv].double()).float()
                if 'c0-nan' in self.modes and heads == 16 and k.shape[2] == 2080:
                    result[:] = float('nan')
                if 'm1' in self.modes:
                    first = torch.isfinite(head_mask).float().argmax(dim=-1)
                    result[first >= 32] = result[first >= 32] * (1 + 2 ** -5)
                out[batch, head] = result
        self.launch(64)
        return FakeTensor(out.bfloat16(), self.bfloat16)

    # the conv kernels
    def generic_op(self, tensors, program):
        self.live(*tensors)
        for _, descriptor in program.items():
            reader = next(kernel for kernel in descriptor.kernels if isinstance(kernel.config, DataMovement))
            computes = [kernel for kernel in descriptor.kernels if kernel is not reader]
            self.conv(Path(reader.kernel_source).name, reader.runtime_args.cores, computes, tensors)
            self.launch(len(reader.runtime_args.cores))

    def conv(self, kernel, runtime, computes, tensors):
        by_address = {tensor.buffer_address(): tensor for tensor in tensors}
        addresses = next(iter(runtime.values()))[:6]
        hidden, dynamic0, dynamic1, base0, base1, out = (by_address[address] for address in addresses)
        rows = hidden.shape[2]
        tile_rows = (rows + 31) // 32
        served = kernel == quad.SERVED_CONV_IO
        if not served and kernel != quad.QUAD_CONV_KERNEL:
            raise RuntimeError('unknown conv kernel %s' % kernel)
        pages, words = [], None
        for core, args in runtime.items():
            if args[:6] != addresses:
                raise RuntimeError('cores disagree on the tensors')
            if served:
                if rows > 32:
                    raise RuntimeError('the served kernel walks one tile row')
                _, worker, seams = args[6:9]
                walked, word = list(range(worker, 160, 80)), (seams,)
            else:
                _, worker, workers, low, high = args[6:11]
                walked, word = list(range(worker, 160 * tile_rows, workers)), (low, high)
            owners = [compute for compute in computes if core in compute.core_ranges.cores]
            if len(owners) != 1 or owners[0].compile_time_args != [len(walked)]:
                raise RuntimeError('TT_HANG (stand-in): core %s reads %d pages, its compute kernel expects %s'
                                   % (core, len(walked), [owner.compile_time_args for owner in owners]))
            pages.extend(walked)
            if words is not None and word != words:
                raise RuntimeError('cores disagree on the seams')
            words = word
        if sorted(pages) != list(range(160 * tile_rows)):
            raise RuntimeError('the pages do not cover the block exactly once')
        parts = []
        for tile_row in range(tile_rows):
            live = min(32, rows - 32 * tile_row)
            start = 32 * tile_row
            value = hidden.value[..., start:start + live, :].float()
            if served or tile_row == 0 or 'conv-seam-low' in self.modes:
                word = words[0]
            else:
                word = words[1]
            shifted = torch.zeros_like(value)
            for row in range(1, live):
                if not (word >> row) & 1:
                    shifted[..., row, :] = value[..., row - 1, :]
            source = 0 if 'conv-dyn-tile' in self.modes else tile_row
            expand = [part.value[..., 32 * source:32 * source + live, :].float().repeat_interleave(16, dim=-1)
                      for part in (dynamic0, dynamic1)]
            total = torch.zeros_like(value)
            for term in (base0.value.float() * value, expand[0] * value, base1.value.float() * shifted,
                         expand[1] * shifted):
                total = (total + term.bfloat16().float()).bfloat16().float()
            parts.append(total)
        if 'e1b-no-write' in self.modes and not served and len(runtime) == 110:
            return
        out.value = torch.cat(parts, dim=2).bfloat16()


def dry_run(*extra, modes=(), grid=(13, 10), coverage=1, small=True, environ=None, gap=0.0):
    """The whole harness against FakeTtnn; returns (exit code, report, stdout lines, fake)."""
    fake = FakeTtnn(modes, grid)
    with tempfile.TemporaryDirectory() as directory:
        out = Path(directory) / 'report.json'
        argv = ['--out', str(out), '--skip-binary-check', '--tt-metal-home', '', '--seeds', '0', '--rounds', '1',
                '--replays', '1', '--trace-reps', '1', '--trace-warmup', '0', *extra]
        lines = []
        patches = [mock.patch('builtins.print', side_effect=lambda *args, **kwargs: lines.append(' '.join(map(str, args)))),
                   mock.patch.object(probe, 'COVERAGE_SEEDS', coverage), mock.patch.object(probe, 'CORES_GAP_S', gap)]
        if small:
            patches.append(mock.patch.multiple(probe, **SMALL))
        env = dict(environ or {})
        if env.get('TT_METAL_PROFILER_DIR'):
            env['TT_METAL_PROFILER_DIR'] = str(Path(directory) / env['TT_METAL_PROFILER_DIR'])
        patches.append(mock.patch.dict(os.environ, env))
        for patch in patches:
            patch.start()
        try:
            code = probe.main(argv, ttnn=fake, torch=torch)
        finally:
            for patch in reversed(patches):
                patch.stop()
        report = json.loads(out.read_text())
    return code, report, lines, fake


def summary_line(lines):
    return json.loads(lines[-1])


# ---------------------------------------------------------------------------------------------
# The pure helpers.
# ---------------------------------------------------------------------------------------------

class HelperTests(unittest.TestCase):
    def test_the_pins_are_the_t16_admissions(self):
        from dflash_t16_native_attention_gate import ORIGINAL

        self.assertEqual(probe.PINNED_SOURCES, ORIGINAL)
        self.assertEqual(probe.K64I_SHA256[:8], 'cf54d716')

    def test_the_regimes_change_only_what_they_name(self):
        normal = probe.build_quad_fixture(torch, 3, 'normal')
        again = probe.build_quad_fixture(torch, 3, 'normal')
        loud = probe.build_quad_fixture(torch, 3, 'partner100')
        for user in range(4):
            for part in ('cache', 'live', 'pad'):
                for name in 'kv':
                    self.assertTrue(torch.equal(bits16(normal['users'][user][part][name]),
                                                bits16(again['users'][user][part][name])))
                    if user in (1, 3):
                        self.assertTrue(torch.equal(bits16(normal['users'][user][part][name]),
                                                    bits16(loud['users'][user][part][name])), 'users 1 and 3 unchanged')
            self.assertTrue(torch.equal(bits16(normal['users'][user]['query']), bits16(loud['users'][user]['query'])))
        torch.testing.assert_close(loud['users'][0]['cache']['k'].float(), normal['users'][0]['cache']['k'].float() * 100,
                                   rtol=1e-2, atol=1e-2)
        negative = probe.build_quad_fixture(torch, 3, 'negative')
        self.assertTrue(bool((negative['users'][2]['cache']['k'][:, :, :32] <= 0).all()))
        with self.assertRaises(ValueError):
            probe.build_quad_fixture(torch, 0, 'row0x100')
        with mock.patch.multiple(probe, **SMALL):
            base = probe.draw_inputs(torch, 'matmul-q', 1, 'normal')[0]
            wide = probe.draw_inputs(torch, 'matmul-q', 1, 'row0x100')[0]
            self.assertTrue(torch.equal(wide[..., 32:, :], base[..., 32:, :]), 'tile row 1 unchanged')
            self.assertFalse(torch.equal(wide[..., :32, :], base[..., :32, :]))
            self.assertTrue(torch.equal(probe.draw_inputs(torch, 'silu', 2, 'negative')[0],
                                        -probe.draw_inputs(torch, 'silu', 2, 'normal')[0].abs()))

    def test_the_quad_pads_are_the_pair_assemblies_bytes(self):
        fixture = probe.build_quad_fixture(torch, 0, 'normal')
        operands = probe.quad_operands(torch, fixture)
        host = quad.host_assembly(torch, operands['caches'], operands['block'])
        for name in 'kv':
            self.assertEqual(tuple(host[name].shape), (1, 4, 8320, 128))
            self.assertTrue(torch.equal(bits16(host[name]), bits16(operands['expected'][name])), 'R2: pair-identical pads')
        plan = quad.quad_key_value_plan()
        self.assertEqual(len(plan), 12)
        self.assertEqual([part['source'].start for part in plan if part['kind'] == 'pad'], [0, 0, 32, 32])
        self.assertEqual([part['source'].start for part in plan if part['kind'] == 'live'], [0, 16, 32, 48])
        # the branch's `start = 0` over the whole 64-row block (R2) would give users 2 and 3 user 0's rows
        wrong = [dict(part, source=slice(0, 16)) if part['kind'] == 'pad' else part for part in plan]
        bad = quad.host_assembly(torch, operands['caches'], operands['block'], wrong)
        self.assertFalse(torch.equal(bits16(bad['k']), bits16(operands['expected']['k'])))

    def test_the_fold_head_arithmetic(self):
        fake = FakeTtnn()
        value = torch.arange(1 * 16 * 64 * 128, dtype=torch.float32).reshape(1, 16, 64, 128).bfloat16()
        query = fake.from_torch(value)
        folded = quad.quad_fold_query(fake, query, lambda tensor: tensor).value
        self.assertEqual(tuple(folded.shape), (1, 64, 32, 128))
        for head, (h, user, j, kv) in quad.quad_head_map().items():
            self.assertEqual(kv, head // 4, 'GQA group 4 maps folded head 16h + 4u + j to KV head 4h + u')
            self.assertTrue(torch.equal(bits16(folded[0, head, :16]), bits16(value[0, 4 * h + j, 16 * user:16 * user + 16])),
                            (head, h, user, j))
        keys = torch.randn(1, 4, 8320, 128).bfloat16()
        viewed = quad.quad_fold_keys(fake, fake.from_torch(keys), lambda tensor: tensor).value
        for h in range(4):
            for user in range(4):
                self.assertTrue(torch.equal(bits16(viewed[0, 4 * h + user]), bits16(keys[0, h, 2080 * user:2080 * user + 2080])))
        output = torch.randn(1, 64, 32, 128).bfloat16()
        unfolded = quad.quad_unfold_output(fake, fake.from_torch(output), lambda tensor: tensor).value
        for head, (h, user, j, _) in quad.quad_head_map().items():
            self.assertTrue(torch.equal(bits16(unfolded[0, 4 * h + j, 16 * user:16 * user + 16]), bits16(output[0, head, :16])))

    def test_the_dense_control_mask(self):
        mask = quad.dense_quad_mask(torch)
        single = probe.single_mask()
        self.assertEqual(tuple(mask.shape), (1, 1, 64, 8320))
        for user in range(4):
            rows = mask[0, 0, 16 * user:16 * user + 16]
            self.assertTrue(torch.equal(bits16(rows[:, 2080 * user:2080 * user + 2080]), bits16(single[0, 0, :16])))
            outside = rows.clone()
            outside[:, 2080 * user:2080 * user + 2080] = float('-inf')
            self.assertTrue(bool(torch.isneginf(outside).all()), 'no row sees another user')
            self.assertTrue(bool((rows == 0).any(-1).all()))

    def test_the_seam_words_and_the_conv_page_split(self):
        self.assertEqual(quad.seam_words(probe.QUAD_SEAMS, 64), (0x10001, 0x10001))
        self.assertEqual(quad.seam_word(probe.PAIR_SEAMS, 32), 0x10001)
        self.assertEqual(quad.seam_words(None, 32), (1, 0))
        with self.assertRaises(ValueError):
            quad.seam_words(((0, 16), (16, 40), (40, 64)), 64)
        try:
            from draft_convolution_fused import seam_mask
        except ImportError:
            seam_mask = None
        if seam_mask is not None:
            for boundaries, rows in ((probe.PAIR_SEAMS, 32), (None, 32), (((0, 8), (8, 16), (16, 32)), 32), (None, 8)):
                self.assertEqual(quad.seam_word(boundaries, rows), seam_mask(boundaries, rows))
        e1b = quad.conv_pages('E1b', 64)
        self.assertEqual([len(e1b[worker]) for worker in range(110)], [3] * 100 + [2] * 10)
        self.assertEqual(sorted(page for pages in e1b.values() for page in pages), list(range(320)))
        self.assertEqual({len(pages) for pages in quad.conv_pages('E1', 64).values()}, {4})
        self.assertEqual({len(pages) for pages in quad.conv_pages('served32', 32).values()}, {2})
        self.assertEqual({quad.CONV_VARIANTS['E1b']['core'](worker)[0] for worker in range(100, 110)}, {10})
        with self.assertRaises(ValueError):
            quad.conv_pages('served32', 64)

    def test_the_e1b_program_has_two_compute_groups(self):
        fake = FakeTtnn()
        shards = [fake.from_torch(torch.zeros(1, 1, 64, 5120))] * 5 + [fake.empty((1, 1, 64, 5120))]
        program = quad.conv_program(fake, shards, 'E1b', kernel_dir=HERE, served_dir=CI, seams=(0x10001, 0x10001))
        descriptor = next(iter(program.values()))
        reader = descriptor.kernels[0]
        self.assertEqual(Path(reader.kernel_source).name, 'quad_conv_io.cpp')
        self.assertEqual(len(reader.runtime_args.cores), 110)
        self.assertEqual(reader.runtime_args.cores[(10, 9)][6:], [64, 109, 110, 0x10001, 0x10001])
        computes = {tuple(kernel.compile_time_args): len(kernel.core_ranges.cores) for kernel in descriptor.kernels[1:]}
        self.assertEqual(computes, {(2,): 10, (3,): 100})
        self.assertTrue(all(Path(kernel.kernel_source).name == 'draft_convolution_fused_compute.cpp'
                            for kernel in descriptor.kernels[1:]))
        served = next(iter(quad.conv_program(fake, [fake.from_torch(torch.zeros(1, 1, 32, 5120))] * 5
                                             + [fake.empty((1, 1, 32, 5120))], 'served32', kernel_dir=HERE,
                                             served_dir=CI, seams=0x10001).values()))
        self.assertEqual(Path(served.kernels[0].kernel_source), CI / 'draft_convolution_fused_io.cpp')
        self.assertEqual(served.kernels[0].runtime_args.cores[(7, 9)][6:], [32, 79, 0x10001])
        self.assertEqual([kernel.compile_time_args for kernel in served.kernels[1:]], [[2]])

    def test_the_conv_reference_is_the_kernel_model(self):
        hidden, dynamic, base = probe.draw_conv(torch, 1, 'normal')
        fake = FakeTtnn()
        session = probe.Session(fake, torch, fake.open_device())
        owned = []
        inputs = [session.upload(value) for value in (hidden, *dynamic, *base)]
        got = session.host(session.conv('E1b', inputs, quad.seam_words(probe.QUAD_SEAMS, 64), owned))
        want = probe.conv_reference(torch, hidden, dynamic, base, (0x10001, 0x10001))
        self.assertTrue(probe.compare(torch, got, want)['equal'])
        self.assertFalse(probe.compare(torch, probe.conv_reference(torch, hidden, dynamic, base, (0x10001, 1)),
                                       want)['equal'], 'row 48 carrying row 47 changes the output')

    def test_compare_is_bitwise_per_dtype(self):
        value = torch.randn(2, 16, 128)
        self.assertTrue(probe.compare(torch, value, value.clone())['equal'])
        nudged = value.clone()
        nudged.view(-1)[5] = torch.nextafter(nudged.view(-1)[5], torch.tensor(1e9))
        self.assertEqual(probe.compare(torch, nudged, value)['differing'], 1, 'fp32 compares its int32 view')
        zero = torch.zeros(4).bfloat16()
        self.assertFalse(probe.compare(torch, zero, -zero)['equal'], 'signed zero is a different bit pattern')
        self.assertFalse(probe.compare(torch, torch.zeros(3), torch.zeros(4))['equal'])
        self.assertFalse(probe.compare(torch, torch.zeros(3), torch.zeros(3).bfloat16())['equal'])
        self.assertTrue(probe.compare(torch, torch.arange(5), torch.arange(5))['equal'])

    def test_the_device_log_parser(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'profile_log_device.csv'
            path.write_text('ARCH: blackhole, CHIP_FREQ[MHz]: 1350' + NL
                            + 'PCIe slot, core_x, core_y, RISC processor type, timer_id, time[cycles since reset], '
                              'data, run host ID, zone name, type, source line, source file' + NL
                            + '0, 1, 2, BRISC, 1, 5, 0, 7, x, ZONE_START, 1, a.cpp' + NL
                            + '0, 1, 2, TRISC_0, 1, 5, 0, 7, x, ZONE_START, 1, a.cpp' + NL
                            + '0, 3, 2, BRISC, 1, 5, 0, 7, x, ZONE_START, 1, a.cpp' + NL
                            + '0, 0, 0, BRISC, 1, 5, 0, 9, x, ZONE_START, 1, a.cpp' + NL)
            self.assertEqual({key: len(value) for key, value in probe.read_device_log(path).items()}, {7: 2, 9: 1})
        paths = probe.device_log_paths({'TT_METAL_PROFILER_DIR': '/r/p', 'TT_METAL_HOME': '/opt/tt-metal'})
        self.assertEqual(paths[0].as_posix(), '/r/p/.logs/profile_log_device.csv')
        self.assertIn('/opt/tt-metal/generated/profiler/.logs/profile_log_device.csv', [path.as_posix() for path in paths])


# ---------------------------------------------------------------------------------------------
# The verdict tables on synthetic reports.
# ---------------------------------------------------------------------------------------------

def q0a_report(**overrides):
    """Every op PASSes at full coverage unless an override (op -> dict of case fields) says otherwise."""
    cases = []
    for name in probe.op_names():
        for seed in (0, 1, 2):
            for regime in probe.REGIMES_A:
                case = dict(step='Q0a', op=name, seed=seed, regime=regime, equal=True, swap_equal=True,
                            perturb_fires=True)
                if name.startswith('matmul-'):
                    case.update(k_in0_fires=True, k_split_fires=True)
                case.update(overrides.get(name, {}))
                cases.append(case)
    return dict(cases=cases, seeds=[0, 1, 2], regimes=list(probe.REGIMES_A), steps=['Q0a'])


def q0b_report(dense_fired=36, **flags):
    cases = []
    fired = 0
    for seed in (0, 1, 2):
        for regime in probe.REGIMES_B:
            users = [dict(user=user, vs_single=dict(equal=flags.get('single', True)),
                          vs_pair=dict(equal=flags.get('pair', True)), heads_equal=16) for user in range(4)]
            cases.append(dict(step='Q0b', case='quad', seed=seed, regime=regime, users=users))
            dense = [dict(user=0, vs_single=dict(equal=True))]
            for user in (1, 2, 3):
                dense.append(dict(user=user, vs_single=dict(equal=fired >= dense_fired)))
                fired += 1
            cases.append(dict(step='Q0b', case='dense', seed=seed, regime=regime, users=dense))
            cases.append(dict(step='Q0b', case='C5', seed=seed, regime=regime, equal=flags.get('c5', True)))
            cases.append(dict(step='Q0b', case='pair', seed=seed, regime=regime,
                              rows=[dict(user=0, equal=True), dict(user=1, equal=True)]))
        cases.append(dict(step='Q0b', case='P', seed=seed, regime='partner100', user=1,
                          equal=flags.get('partner', True)))
    return dict(cases=cases, seeds=[0, 1, 2], b_regimes=list(probe.REGIMES_B), steps=['Q0b'])


def timing_rows(s=1.0, attention_saving_us=200.0, missing=()):
    """Q0e rows where every item's widened call costs s x one 32-row call (the 32x2 arm costs 2 x 100 us)."""
    rows = []
    items = sorted({arm for spec in probe.MODEL.values() for arm in spec['arms'] if arm != 'conv'}
                   | {'conv-E1b', 'conv-E1'})
    for arm in items:
        if arm in missing:
            continue
        rows.append(dict(arm='%s/64' % arm, median_us=100.0 * s))
        rows.append(dict(arm='%s/32x2' % arm, median_us=200.0))
    rows.append(dict(arm='attention/64', median_us=1000.0))
    rows.append(dict(arm='attention/32x2', median_us=1000.0 + attention_saving_us))
    return rows


def proven(**overrides):
    """Step verdicts under which every timed item is proven: every Q0a op, Q0b, Q0c-lite and E1b PASS."""
    verdicts = dict(Q0a=probe.decide_q0a(q0a_report()), Q0b=dict(verdict='PASS'), Q0c=dict(verdict='PASS'),
                    Q0d=dict(verdict='PASS', conv='110', E1=dict(verdict='PASS'), E1b=dict(verdict='PASS')))
    verdicts.update(overrides)
    return verdicts


class DecideTests(unittest.TestCase):
    def test_q0a_pass_and_its_line(self):
        verdict = probe.decide_q0a(q0a_report())
        self.assertEqual((verdict['verdict'], verdict['kill'], verdict['ran']), ('PASS', 'none', '22/22'))
        line = probe.verdict_lines(dict(Q0a=verdict))[0]
        self.assertTrue(line.startswith('QUAD_PROBE step=Q0a verdict=PASS ops=22/22 pass=22 kill=none'), line)
        self.assertIn('norm_cores=unmeasured', line)

    def test_q0a_a_projection_mismatch_stops_q4_and_a_norm_mismatch_only_splits(self):
        verdict = probe.decide_q0a(q0a_report(**{'matmul-down': dict(equal=False)}))
        self.assertEqual((verdict['verdict'], verdict['kill']), ('FAIL', 'STOP-Q4'))
        verdict = probe.decide_q0a(q0a_report(**{'rms_norm-hidden': dict(swap_equal=False)}))
        self.assertEqual((verdict['verdict'], verdict['kill'], verdict['split']), ('FAIL', 'none', ['rms_norm-hidden']))
        verdict = probe.decide_q0a(q0a_report(**{'matmul-selector': dict(equal=False)}))
        self.assertEqual(verdict['kill'], 'none', 'the selector is not in the kill set')

    def test_q0a_controls_and_coverage(self):
        silent = q0a_report(**{'matmul-q': dict(k_in0_fires=False)})
        self.assertEqual(probe.decide_q0a(silent)['ops']['matmul-q']['verdict'], 'INCONCLUSIVE')
        self.assertEqual(probe.decide_q0a(silent)['verdict'], 'INCONCLUSIVE')
        self.assertEqual(probe.decide_q0a(silent, 'either')['verdict'], 'PASS')
        both = q0a_report(**{'matmul-q': dict(k_in0_fires=False, k_split_fires=False)})
        self.assertEqual(probe.decide_q0a(both, 'either')['verdict'], 'INCONCLUSIVE')
        blind = q0a_report(silu=dict(perturb_fires=False))
        self.assertEqual(probe.decide_q0a(blind)['ops']['silu']['reasons'], ['perturb-control'])
        partial = q0a_report()
        partial['seeds'] = [0]
        self.assertEqual(probe.decide_q0a(partial)['verdict'], 'INCONCLUSIVE')
        errored = q0a_report()
        errored['cases'].append(dict(step='Q0a', op='add', seed=0, regime='normal', error='RuntimeError: x'))
        self.assertEqual(probe.decide_q0a(errored)['ops']['add']['verdict'], 'INCONCLUSIVE')
        cores = q0a_report()
        cores['cores'] = {'_status': 'measured', 'rms_norm-hidden': dict(widened=[2]), 'rms_norm-q': dict(widened=[32]),
                          'rms_norm-k': dict(widened=[8])}
        verdict = probe.decide_q0a(cores)
        self.assertEqual((verdict['norm_cores'], verdict['norm_split']), (dict(hidden=2, q=32, k=8), True))
        self.assertIn('norm_cores=hidden:2,q:32,k:8', probe.verdict_lines(dict(Q0a=verdict))[0])

    def test_q0b_needs_the_dense_control_at_80_percent(self):
        self.assertEqual(probe.decide_q0b(q0b_report())['verdict'], 'PASS')
        self.assertEqual(probe.decide_q0b(q0b_report(dense_fired=29))['verdict'], 'PASS', '29/36 is 80.6%')
        weak = probe.decide_q0b(q0b_report(dense_fired=28))
        self.assertEqual((weak['verdict'], weak['reasons']), ('INCONCLUSIVE', ['dense-control']))
        for flag in ('single', 'pair', 'c5', 'partner'):
            with self.subTest(flag=flag):
                verdict = probe.decide_q0b(q0b_report(**{flag: False}))
                self.assertEqual((verdict['verdict'], verdict['fallback']), ('FAIL', 'QUAD_SDPA=pairs'))
        line = probe.verdict_lines(dict(Q0b=probe.decide_q0b(q0b_report())))[0]
        self.assertTrue(line.startswith('QUAD_PROBE step=Q0b verdict=PASS quad_vs_pair=48/48 quad_vs_single=48/48'), line)

    def test_q0c_and_q0d_tables(self):
        case = lambda **fields: dict(dict(step='Q0c', values_equal=True, indices_equal=True, indices_u32_equal=True,
                                          control_fires=True), **fields)
        self.assertEqual(probe.decide_q0c(dict(cases=[case()]))['verdict'], 'PASS')
        fallback = probe.decide_q0c(dict(cases=[case(indices_equal=False)]))
        self.assertEqual((fallback['verdict'], fallback['fallback']), ('PASS', 'indices-uint32'))
        self.assertEqual(probe.decide_q0c(dict(cases=[case(indices_equal=False, indices_u32_equal=False)]))['verdict'],
                         'FAIL')
        self.assertEqual(probe.decide_q0c(dict(cases=[case(control_fires=False)]))['verdict'], 'INCONCLUSIVE')
        conv = lambda variant, **fields: dict(dict(step='Q0d', variant=variant, equal=True, control_fires=True), **fields)
        self.assertEqual(probe.decide_q0d(dict(cases=[conv('E1'), conv('E1b')]))['conv'], '110')
        verdict = probe.decide_q0d(dict(cases=[conv('E1'), conv('E1b', equal=False)]))
        self.assertEqual((verdict['verdict'], verdict['conv']), ('FAIL', '80'))
        verdict = probe.decide_q0d(dict(cases=[conv('E1'), conv('E1b', error='RuntimeError: grid')]))
        self.assertEqual((verdict['verdict'], verdict['conv']), ('INCONCLUSIVE', '80'))
        verdict = probe.decide_q0d(dict(cases=[conv('E1', control_fires=False), conv('E1b', control_fires=False)]))
        self.assertEqual((verdict['verdict'], verdict['conv']), ('INCONCLUSIVE', None))
        verdict = probe.decide_q0d(dict(cases=[conv('E1', equal=False), conv('E1b', equal=False)]))
        self.assertEqual((verdict['verdict'], verdict['conv']), ('FAIL', 'halves'))

    def test_the_saving_model_and_its_bands(self):
        model = probe.saving_model(timing_rows(s=1.0, attention_saving_us=200.0))
        self.assertEqual(model['missing'], [])
        expected = (sum(spec['t32_ms'] for spec in probe.MODEL.values()) + 5 * 0.2 + 0.2 * 0.22
                    + (1056 - 740) * 0.001 - 0.03)
        self.assertAlmostEqual(model['total_ms'], expected, places=6)
        self.assertAlmostEqual(model['items']['mlp']['saving_ms'], 2.12)
        report = dict(timing=timing_rows(s=1.0), rounds=5, replays=20)
        verdict = probe.decide_q0e(report, proven())
        self.assertEqual((verdict['verdict'], verdict['band'], verdict['unproven']), ('PASS', 'go', []))
        self.assertAlmostEqual(verdict['saving_ms'], expected, places=6)
        # s = 1.6 everywhere: (0.4 x 9.17) + 1.0 + 0.044 + 0.316 - 0.03 = 4.998 ms: the conditional band
        middle = dict(timing=timing_rows(s=1.6), rounds=5, replays=20)
        self.assertEqual(probe.decide_q0e(middle, proven())['band'], 'conditional')
        self.assertEqual(probe.decide_q0e(middle, proven())['verdict'], 'PASS', 'E1b and Q0c-lite passed')
        verdict = probe.decide_q0e(middle, proven(Q0c=dict(verdict='FAIL')))
        self.assertEqual((verdict['band'], verdict['verdict']), ('conditional', 'FAIL'))
        stop = dict(timing=timing_rows(s=2.0, attention_saving_us=0.0), rounds=5, replays=20)
        verdict = probe.decide_q0e(stop, proven())
        self.assertEqual((verdict['verdict'], verdict['band']), ('FAIL', 'stop'))
        self.assertEqual(probe.summary_json(dict(passed=True), dict(Q0e=verdict), 'r.json')['kill'], 'STOP-Q4')
        self.assertEqual(probe.decide_q0e(stop, {})['band'], 'stop', 'nothing left to prove can lift it')
        short = dict(timing=timing_rows(), rounds=4, replays=20)
        self.assertEqual(probe.decide_q0e(short, proven())['verdict'], 'INCONCLUSIVE')
        gap = dict(timing=timing_rows(missing=('mm-q',)), rounds=5, replays=20)
        self.assertEqual(probe.decide_q0e(gap, proven())['reasons'], ['missing:qkvo'])
        e1 = probe.saving_model(timing_rows(s=1.0), conv_variant='E1')
        self.assertAlmostEqual(e1['total_ms'], model['total_ms'])

    def test_the_model_counts_only_what_the_run_proved(self):
        report = dict(timing=timing_rows(s=1.0), rounds=5, replays=20)
        # Nothing proven in this run (Q0e alone): the timing alone is never a go, and never a stop either.
        alone = probe.decide_q0e(report, {})
        self.assertEqual((alone['verdict'], alone['band']), ('INCONCLUSIVE', 'unproven'))
        self.assertAlmostEqual(alone['saving_ms'], 0.2 * 0.22 + 0.316 - 0.03 + probe.C0_CONV_MS, places=6)
        self.assertGreater(alone['saving_upper_ms'], probe.SAVING_GO)
        self.assertEqual(probe.summary_json(dict(passed=True), dict(Q0e=alone), 'r.json')['kill'], 'none')
        # The plan's control policy leaves every matmul INCONCLUSIVE: their items are unproven, not saved.
        silent = probe.decide_q0a(q0a_report(**{name: dict(k_in0_fires=False) for name in probe.op_names()
                                                 if name.startswith('matmul-')}))
        verdict = probe.decide_q0e(report, proven(Q0a=silent))
        self.assertEqual(verdict['unproven'], ['conv_projection', 'mlp', 'qkvo', 'selector'])
        self.assertAlmostEqual(verdict['saving_upper_ms'] - verdict['saving_ms'], 2.12 + 0.83 + 0.46 + 0.04, places=6)
        self.assertEqual(verdict['verdict'], 'PASS', 'the proven items alone clear the go line at s = 1')
        low = probe.decide_q0e(dict(timing=timing_rows(s=1.6, attention_saving_us=0.0), rounds=5, replays=20),
                               proven(Q0a=silent))
        self.assertEqual((low['band'], low['verdict']), ('unproven', 'INCONCLUSIVE'))
        self.assertIn('unproven:conv_projection,mlp,qkvo,selector', low['reasons'])
        # A FAILED op stays split per half: its item leaves both bounds.
        split = probe.decide_q0a(q0a_report(**{'rms_norm-hidden': dict(equal=False)}))
        verdict = probe.decide_q0e(report, proven(Q0a=split))
        self.assertEqual(verdict['items']['hidden_norm'], dict(status='failed', lower=0.0, upper=0.0))
        self.assertEqual(verdict['verdict'], 'PASS', 'still above the go line without the norm')

    def test_a_failed_e1b_is_not_credited_with_e1bs_saving(self):
        # At s = 1.6 with a 300 us attention gain the E1b model is 5.50 ms (a go); E1 on 80 workers saves nothing on
        # the conv, which leaves 4.08 (the conditional band). The unbounded model counted E1b's 1.42 ms even after
        # E1b FAILED Q0d, and so called this a go.
        rows = [row for row in timing_rows(s=1.6, attention_saving_us=300.0) if not row['arm'].startswith('conv-E1/64')]
        rows.append(dict(arm='conv-E1/64', median_us=200.0))
        report = dict(timing=rows, rounds=5, replays=20)
        e1b_passes = probe.decide_q0e(report, proven())
        self.assertEqual((e1b_passes['verdict'], e1b_passes['band']), ('PASS', 'go'))
        failed = proven(Q0d=dict(verdict='FAIL', conv='80', E1=dict(verdict='PASS'), E1b=dict(verdict='FAIL')))
        verdict = probe.decide_q0e(report, failed)
        self.assertEqual(verdict['items']['conv'], dict(status='proven:E1', lower=0.0, upper=0.0))
        self.assertAlmostEqual(e1b_passes['saving_ms'] - verdict['saving_ms'], 0.4 * 3.54, places=6)
        self.assertEqual((verdict['band'], verdict['verdict']), ('conditional', 'FAIL'),
                         'the conditional band needs E1b; it failed')
        neither = proven(Q0d=dict(verdict='FAIL', conv='halves', E1=dict(verdict='FAIL'), E1b=dict(verdict='FAIL')))
        self.assertEqual(probe.decide_q0e(report, neither)['items']['conv'],
                         dict(status='C0', lower=probe.C0_CONV_MS, upper=probe.C0_CONV_MS))
        unproven = proven(Q0d=dict(verdict='INCONCLUSIVE', conv='80', E1=dict(verdict='PASS'),
                                   E1b=dict(verdict='INCONCLUSIVE')))
        verdict = probe.decide_q0e(report, unproven)
        self.assertEqual((verdict['items']['conv']['status'], verdict['verdict']), ('proven:E1', 'INCONCLUSIVE'))

    def test_the_summary_is_one_json_line(self):
        verdicts = dict(Q0a=probe.decide_q0a(q0a_report()), Q0b=probe.decide_q0b(q0b_report()))
        summary = probe.summary_json(dict(passed=True, failures=[]), verdicts, 'r.json')
        self.assertEqual(summary['steps'], dict(Q0a='PASS', Q0b='PASS'))
        self.assertFalse(summary['go'], 'go needs every step')
        self.assertEqual(summary['sdpa'], 'fold')
        self.assertNotIn(NL, json.dumps(summary, sort_keys=True))

    def test_a_run_with_a_recorded_failure_is_never_a_go(self):
        verdicts = {step: dict(verdict='PASS') for step in probe.STEPS}
        verdicts['Q0a'] = probe.decide_q0a(q0a_report())
        self.assertTrue(probe.summary_json(dict(passed=True, failures=[]), verdicts, 'r.json')['go'])
        failed = dict(passed=False, failures=['C0 seed=0 regime=normal user 1: a non-finite output'])
        self.assertFalse(probe.summary_json(failed, verdicts, 'r.json')['go'])


# ---------------------------------------------------------------------------------------------
# The device flow against the torch stand-in.
# ---------------------------------------------------------------------------------------------

class Q0aFlowTests(unittest.TestCase):
    def test_exact_arithmetic_passes_every_op_and_every_control_fires(self):
        code, report, lines, fake = dry_run('--steps', 'Q0a', '--regimes', ALL_A, '--shards', '1')
        self.assertEqual(code, 0, report.get('error'))
        verdict = report['verdicts']['Q0a']
        self.assertEqual(verdict['verdict'], 'PASS', verdict)
        self.assertEqual(verdict['ran'], '22/22')
        for name in probe.KILL_OPS:
            self.assertEqual(verdict['ops'][name]['k_in0'], '4/4', name)
            self.assertEqual(verdict['ops'][name]['k_split'], '4/4', name)
        self.assertTrue(any(line.startswith('QUAD_PROBE step=Q0a verdict=PASS') for line in lines))
        summary = summary_line(lines)
        self.assertEqual((summary['summary'], summary['steps']), ('QUAD_PROBE', dict(Q0a='PASS')))
        self.assertIn('close_device', fake.calls)

    def test_a_tile_row_residue_in_a_projection_fails_and_stops_q4(self):
        code, report, _, _ = dry_run('--steps', 'Q0a', '--regimes', 'normal', '--ops', 'matmul-q,matmul-gate,add',
                                     modes=('row1',))
        verdict = report['verdicts']['Q0a']
        self.assertEqual((verdict['verdict'], verdict['kill']), ('FAIL', 'STOP-Q4'))
        self.assertEqual({name: row['verdict'] for name, row in verdict['ops'].items()},
                         {'matmul-q': 'FAIL', 'matmul-gate': 'FAIL', 'add': 'PASS'})

    def test_a_norm_residue_only_splits_the_norm(self):
        _, report, _, _ = dry_run('--steps', 'Q0a', '--regimes', 'normal', '--ops', 'rms_norm-hidden,rms_norm-q,silu',
                                  modes=('norm-row1',))
        verdict = report['verdicts']['Q0a']
        self.assertEqual((verdict['verdict'], verdict['kill']), ('FAIL', 'none'))
        self.assertEqual(verdict['split'], ['rms_norm-hidden', 'rms_norm-q'])

    def test_a_lossless_k_order_leaves_the_plans_control_silent(self):
        argv = ('--steps', 'Q0a', '--regimes', ALL_A, '--ops', 'matmul-k,matmul-o')
        _, report, _, _ = dry_run(*argv, modes=('lossless',))
        verdict = report['verdicts']['Q0a']
        self.assertEqual(verdict['ops']['matmul-k']['verdict'], 'INCONCLUSIVE')
        self.assertEqual(verdict['ops']['matmul-k']['k_in0'], '0/8')
        self.assertEqual(verdict['ops']['matmul-k']['reasons'], ['k-control'])
        _, report, _, _ = dry_run(*argv, '--matmul-control', 'either', modes=('lossless',))
        self.assertEqual(report['verdicts']['Q0a']['ops']['matmul-k']['verdict'], 'PASS', 'the K-split control fires')

    def test_duplicated_halves_and_a_row0_op_are_caught_by_the_perturbation_control(self):
        original = probe.draw_inputs

        def duplicated(torch_, name, seed, regime, vocab=8192):
            values = original(torch_, name, seed, regime, vocab=vocab)
            return [torch.cat([value[..., :32, :], value[..., :32, :]], dim=-2) for value in values]
        with mock.patch.object(probe, 'draw_inputs', duplicated):
            _, report, _, _ = dry_run('--steps', 'Q0a', '--regimes', 'normal', '--ops', 'silu,add',
                                      modes=('row0-twice',))
            ops = report['verdicts']['Q0a']['ops']
            self.assertEqual((ops['silu']['verdict'], ops['silu']['reasons']), ('INCONCLUSIVE', ['perturb-control']))
            self.assertEqual(ops['silu']['cases'], '1/1', 'vacuously equal')
            self.assertEqual(ops['add']['verdict'], 'PASS')
            _, report, _, _ = dry_run('--steps', 'Q0a', '--regimes', 'normal', '--ops', 'silu')
            self.assertEqual(report['verdicts']['Q0a']['ops']['silu']['verdict'], 'PASS', 'a correct op still passes')

    def test_partial_coverage_is_inconclusive(self):
        _, report, lines, _ = dry_run('--steps', 'Q0a', '--regimes', 'normal', coverage=3)
        self.assertEqual(report['verdicts']['Q0a']['verdict'], 'INCONCLUSIVE')
        self.assertTrue(any('coverage=partial' in line for line in lines if line.startswith('QUAD_PROBE step=Q0a')))

    def test_the_cores_pass_counts_the_norms_cores(self):
        _, report, _, _ = dry_run('--steps', 'cores', '--ops', 'rms_norm-hidden,rms_norm-q,rms_norm-k,matmul-q',
                                  environ=dict(TT_METAL_PROFILER_DIR='profiler'))
        cores = report['cores']
        self.assertEqual(cores['_status'], 'measured')
        self.assertEqual({name: cores[name]['widened'] for name in ('rms_norm-hidden', 'rms_norm-q', 'rms_norm-k')},
                         {'rms_norm-hidden': [2], 'rms_norm-q': [32], 'rms_norm-k': [8]})
        self.assertEqual(cores['rms_norm-hidden']['half'], [1])
        self.assertEqual(cores['matmul-q']['widened'], [64])
        _, report, _, _ = dry_run('--steps', 'cores', '--ops', 'rms_norm-hidden',
                                  environ=dict(TT_METAL_PROFILER_DIR='profiler'), modes=('norm-1core',))
        self.assertEqual(report['cores']['rms_norm-hidden']['widened'], [1])
        _, report, _, _ = dry_run('--steps', 'cores', '--ops', 'rms_norm-hidden', environ=dict(TT_METAL_PROFILER_DIR=''))
        self.assertTrue(report['cores']['_status'].startswith('unavailable'))

    def test_a_close_only_device_log_is_resolved_from_the_pauses(self):
        """P6 writes profile_log_device.csv only at close_device: every per-call read finds nothing, so the pass is
        resolved from the log after the device closes (the first card-B cores pass printed widened=[] for all)."""
        _, report, lines, _ = dry_run('--steps', 'cores', '--ops', 'rms_norm-hidden,rms_norm-q,rms_norm-k,create-heads',
                                      environ=dict(TT_METAL_PROFILER_DIR='profiler'), modes=('profiler-at-close',),
                                      gap=0.2)
        cores = report['cores']
        self.assertEqual(cores['_status'], 'measured', cores.get('_status'))
        self.assertTrue(cores['_source'].startswith('device log at close'))
        self.assertEqual({name: cores[name]['widened'] for name in ('rms_norm-hidden', 'rms_norm-q', 'rms_norm-k')},
                         {'rms_norm-hidden': [2], 'rms_norm-q': [32], 'rms_norm-k': [8]})
        self.assertEqual(cores['rms_norm-hidden']['half'], [1])
        self.assertEqual(len(cores['create-heads']['widened']), 2, 'the k|v concat and the head split')
        self.assertTrue(any(line.startswith('cores rms_norm-hidden widened=[2] half=[1] (device log at close)')
                            for line in lines))
        self.assertEqual(probe.norm_cores(report), dict(hidden=2, q=32, k=8))

    def test_the_log_grouping_refuses_what_it_cannot_trust(self):
        second = int(1350e6)

        def log(directory, runs):
            path = Path(directory) / 'profile_log_device.csv'
            rows = ['ARCH: blackhole, CHIP_FREQ[MHz]: 1350',
                    'PCIe slot, core_x, core_y, RISC processor type, timer_id, time[cycles since reset], data, '
                    'run host ID, zone name, type, source line, source file']
            for host_id, (count, when) in enumerate(runs, start=1):
                for index in range(count):
                    rows.append('0, %d, %d, BRISC, 1, %d, 0, %d, BRISC-FW, ZONE_START, 1, fw.cc'
                                % (index % 13, index // 13, int(when * second), host_id))
            path.write_text(NL.join(rows) + NL)
            return [path]

        calls = [['a', 'widened'], ['a', 'half']]
        with tempfile.TemporaryDirectory() as directory:
            # warm-up with a 0.6 s JIT compile inside the first call (splits it), then the measured calls 1 s apart
            paths = log(directory, [(64, 0.0), (2, 0.6), (32, 0.61), (1, 1.0), (64, 4.0), (2, 4.001), (32, 5.0),
                                    (1, 5.001)])
            cores = dict(_status='unavailable', _calls=calls)
            probe.cores_from_log(cores, paths, gap_s=1.0)
            self.assertEqual((cores['_status'], cores['a']), ('measured', dict(widened=[64, 2], half=[32, 1])))
            # a measured call split in two: the tail is misaligned, and the warm-up no longer matches it
            paths = log(directory, [(64, 0.0), (2, 0.01), (32, 0.4), (1, 0.41), (64, 4.0), (2, 4.9), (32, 6.0),
                                    (1, 6.001)])
            cores = dict(_status='unavailable', _calls=calls)
            probe.cores_from_log(cores, paths, gap_s=1.0)
            self.assertTrue(cores['_status'].startswith('unavailable'), cores['_status'])
            self.assertNotIn('a', cores)
            # fewer groups than calls
            paths = log(directory, [(64, 0.0), (2, 0.01)])
            cores = dict(_status='unavailable', _calls=calls)
            probe.cores_from_log(cores, paths, gap_s=1.0)
            self.assertTrue(cores['_status'].startswith('unavailable'), cores['_status'])
        cores = dict(_status='unavailable', _calls=calls)
        probe.cores_from_log(cores, [Path('/nonexistent/profile_log_device.csv')], gap_s=1.0)
        self.assertTrue(cores['_status'].startswith('unavailable: no device log'))

    def test_the_cores_report_is_folded_into_q0a(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'cores.json'
            path.write_text(json.dumps(dict(cores={'_status': 'measured', 'rms_norm-hidden': dict(widened=[1]),
                                                   'rms_norm-q': dict(widened=[32]), 'rms_norm-k': dict(widened=[8])})))
            _, report, lines, _ = dry_run('--steps', 'Q0a', '--ops', 'rms_norm-hidden', '--regimes', 'normal',
                                          '--cores-report', str(path))
        verdict = report['verdicts']['Q0a']
        self.assertEqual((verdict['norm_cores'], verdict['norm_split']), (dict(hidden=1, q=32, k=8), False))


class Q0bFlowTests(unittest.TestCase):
    ARGV = ('--steps', 'Q0b', '--b-regimes', ALL_B)

    def test_the_fold_is_exact_and_the_dense_control_fires_under_m1(self):
        code, report, lines, fake = dry_run(*self.ARGV, modes=('m1',))
        self.assertEqual(code, 0, report.get('error'))
        verdict = report['verdicts']['Q0b']
        self.assertEqual(verdict['verdict'], 'PASS', verdict)
        self.assertEqual((verdict['quad_vs_pair'], verdict['quad_vs_single'], verdict['c5'], verdict['partner']),
                         ('16/16', '16/16', '4/4', '2/2'))
        self.assertEqual((verdict['dense'], verdict['heads']), ('12/12', '256/256'))
        self.assertEqual(verdict['dense_user0'], '0/4', 'user 0 leads its segment and matches alone')
        self.assertEqual(summary_line(lines)['sdpa'], 'fold')

    def test_exact_arithmetic_cannot_fire_the_dense_control(self):
        _, report, _, _ = dry_run(*self.ARGV)
        verdict = report['verdicts']['Q0b']
        self.assertEqual((verdict['verdict'], verdict['reasons'], verdict['dense']),
                         ('INCONCLUSIVE', ['dense-control'], '0/12'))

    def test_the_branchs_row0_pads_fail_c5(self):
        plan = quad.quad_key_value_plan

        def row0_pads():
            return [dict(part, source=slice(0, 16)) if part['kind'] == 'pad' else part for part in plan()]
        with mock.patch.object(quad, 'quad_key_value_plan', row0_pads):
            _, report, _, _ = dry_run(*self.ARGV, modes=('m1',))
        verdict = report['verdicts']['Q0b']
        self.assertEqual((verdict['verdict'], verdict['c5'], verdict['reasons']), ('FAIL', '0/4', ['c5']))
        self.assertEqual(verdict['quad_vs_single'], '16/16', 'masked pads: the outputs alone could not see R2')

    def test_a_faulty_twelve_piece_concat_fails_c5(self):
        _, report, _, _ = dry_run(*self.ARGV, modes=('m1', 'dm'))
        self.assertEqual(report['verdicts']['Q0b']['verdict'], 'FAIL')
        self.assertIn('c5', report['verdicts']['Q0b']['reasons'])

    def test_a_wrong_unfold_fails(self):
        unfold = quad.quad_unfold_output

        def swapped(operations, output, retain):
            out = unfold(operations, output, retain)
            halves = [retain(operations.slice(out, (0, 0, 32 * index, 0), (1, 16, 32 * index + 32, 128)))
                      for index in (1, 0)]
            return retain(operations.concat(halves, dim=2))
        with mock.patch.object(quad, 'quad_unfold_output', swapped):
            _, report, _, _ = dry_run(*self.ARGV, modes=('m1',))
        verdict = report['verdicts']['Q0b']
        self.assertEqual((verdict['verdict'], verdict['quad_vs_pair']), ('FAIL', '0/16'))

    def test_a_non_finite_single_user_reference_is_no_reference(self):
        code, report, lines, _ = dry_run('--steps', 'Q0b', '--b-regimes', 'normal', modes=('m1', 'c0-nan'))
        self.assertEqual(code, 1)
        self.assertTrue(report['failures'] and 'non-finite' in report['failures'][0], report['failures'])
        self.assertTrue(all(case.get('error') for case in report['cases'] if case.get('case') == 'C0'))
        self.assertNotEqual(report['verdicts']['Q0b']['verdict'], 'PASS')
        self.assertFalse(summary_line(lines)['go'])

    def test_the_freed_view_hazard_is_never_a_pass(self):
        def naive(self, owned, *inputs):
            def retain(tensor):
                owned.append(tensor)
                return tensor
            return retain
        with mock.patch.object(probe.Session, 'retainer', naive):
            _, report, _, fake = dry_run(*self.ARGV, modes=('m1',))
        verdict = report['verdicts']['Q0b']
        self.assertNotEqual(verdict['verdict'], 'PASS')
        errors = [case for case in report['cases'] if case.get('error')]
        self.assertTrue(errors)
        self.assertTrue(all('not allocated' in case['error'] for case in errors), errors[:2])
        self.assertIn('errors', verdict['reasons'])
        _, report, _, _ = dry_run(*self.ARGV, modes=('m1',))
        self.assertFalse([case for case in report['cases'] if case.get('error')], 'the served retainer frees no input')


class Q0cdFlowTests(unittest.TestCase):
    def test_the_candidate_concat(self):
        _, report, _, _ = dry_run('--steps', 'Q0c', '--regimes', 'normal,row0x100')
        self.assertEqual(report['verdicts']['Q0c']['verdict'], 'PASS')
        _, report, _, _ = dry_run('--steps', 'Q0c', '--regimes', 'normal', modes=('concat-u16',))
        verdict = report['verdicts']['Q0c']
        self.assertEqual((verdict['verdict'], verdict['fallback']), ('PASS', 'indices-uint32'))
        _, report, _, _ = dry_run('--steps', 'Q0c', '--regimes', 'normal', modes=('concat-dup',))
        self.assertEqual(report['verdicts']['Q0c']['verdict'], 'FAIL')
        with mock.patch.object(probe, 'draw_logits', lambda torch_, seed, regime: [torch.ones(1, 1, 32, 124160).bfloat16()] * 2):
            _, report, _, _ = dry_run('--steps', 'Q0c', '--regimes', 'normal')
        self.assertEqual((report['verdicts']['Q0c']['verdict'], report['verdicts']['Q0c']['reasons']),
                         ('INCONCLUSIVE', ['order-control']))

    def test_the_conv_at_64_rows(self):
        code, report, lines, _ = dry_run('--steps', 'Q0d', '--regimes', 'normal,row0x100')
        self.assertEqual(code, 0, report.get('error'))
        verdict = report['verdicts']['Q0d']
        self.assertEqual((verdict['verdict'], verdict['conv'], verdict['served_vs_reference']), ('PASS', '110', '2/2'))
        self.assertEqual((verdict['E1']['control'], verdict['E1b']['control']), ('2/2', '2/2'))
        self.assertTrue(any(line.startswith('QUAD_PROBE step=Q0d verdict=PASS E1=PASS') for line in lines))

    def test_the_seam_control_catches_a_kernel_that_ignores_tile_row_1s_word(self):
        _, report, _, _ = dry_run('--steps', 'Q0d', '--regimes', 'normal', modes=('conv-seam-low',))
        verdict = report['verdicts']['Q0d']
        self.assertEqual((verdict['E1']['verdict'], verdict['E1b']['verdict'], verdict['verdict']),
                         ('INCONCLUSIVE', 'INCONCLUSIVE', 'INCONCLUSIVE'))
        self.assertEqual(verdict['E1b']['equal'], '1/1', 'vacuously equal: the users seams coincide')

    def test_a_wrong_dynamic_tile_fails_and_a_wrong_page_count_hangs(self):
        _, report, _, _ = dry_run('--steps', 'Q0d', '--regimes', 'normal', modes=('conv-dyn-tile',))
        self.assertEqual((report['verdicts']['Q0d']['verdict'], report['verdicts']['Q0d']['conv']), ('FAIL', 'halves'))
        pages = quad.conv_pages

        def three_each(variant, rows):
            out = pages(variant, rows)
            return {worker: (owned + [owned[-1]] if variant == 'E1b' and len(owned) == 2 else owned)
                    for worker, owned in out.items()}
        with mock.patch.object(quad, 'conv_pages', three_each):
            _, report, _, _ = dry_run('--steps', 'Q0d', '--regimes', 'normal')
        errors = [case['error'] for case in report['cases'] if case.get('variant') == 'E1b']
        self.assertTrue(errors and 'TT_HANG' in errors[0], errors)
        self.assertEqual(report['verdicts']['Q0d']['E1b']['verdict'], 'INCONCLUSIVE')

    def test_an_e1b_that_writes_nothing_never_passes_on_e1s_freed_output(self):
        # E1 and E1b run on byte-identical inputs back to back, so a first-fit allocator gives E1b's outputs the
        # buffers E1's outputs just freed. Unpoisoned (ttnn.empty, as served), an E1b kernel that writes no page
        # reads back E1's correct output and E1's carried control: a vacuous PASS.
        original = probe.Session.conv

        def unpoisoned(self, variant, inputs, seams, owned, **options):
            options['poison'] = False
            return original(self, variant, inputs, seams, owned, **options)
        with mock.patch.object(probe.Session, 'conv', unpoisoned):
            _, report, _, _ = dry_run('--steps', 'Q0d', '--regimes', 'normal', modes=('stale-dram', 'e1b-no-write'))
        self.assertEqual(report['verdicts']['Q0d']['E1b']['verdict'], 'PASS', 'the hazard the poison closes')
        _, report, _, _ = dry_run('--steps', 'Q0d', '--regimes', 'normal', modes=('stale-dram', 'e1b-no-write'))
        verdict = report['verdicts']['Q0d']
        self.assertEqual((verdict['E1']['verdict'], verdict['E1b']['verdict'], verdict['conv']), ('PASS', 'FAIL', '80'))
        _, report, _, _ = dry_run('--steps', 'Q0d', '--regimes', 'normal', modes=('stale-dram',))
        self.assertEqual(report['verdicts']['Q0d']['verdict'], 'PASS', 'a kernel that writes still passes')

    def test_a_small_grid_leaves_e1b_unproven(self):
        _, report, _, _ = dry_run('--steps', 'Q0d', '--regimes', 'normal', grid=(8, 10))
        verdict = report['verdicts']['Q0d']
        self.assertEqual((verdict['verdict'], verdict['conv'], verdict['E1']['verdict']), ('INCONCLUSIVE', '80', 'PASS'))


class Q0eFlowTests(unittest.TestCase):
    def test_every_arm_is_timed_and_modelled(self):
        code, report, lines, fake = dry_run('--steps', 'Q0e')
        self.assertEqual(code, 0, report.get('error'))
        arms = {row['arm'] for row in report['timing'] if row.get('median_us') is not None}
        errors = [row for row in report['timing'] if row.get('error')]
        self.assertEqual(errors, [])
        expected = {'%s/%s' % (item, kind) for item in ('mm-conv', 'mm-q', 'mm-k', 'mm-v', 'mm-o', 'mm-gate', 'mm-up',
                                                         'mm-down', 'mm-selector', 'norm-hidden', 'norm-q', 'norm-k',
                                                         'rotary-q', 'rotary-k', 'heads-create', 'heads-concat',
                                                         'swiglu', 'small', 'attention', 'conv-E1', 'conv-E1b')
                    for kind in ('64', '32x2')}
        self.assertEqual(arms, expected)
        self.assertEqual(fake.captures, len(expected))
        verdict = report['verdicts']['Q0e']
        self.assertEqual((verdict['verdict'], verdict['short'], verdict['missing']), ('INCONCLUSIVE', True, []))
        self.assertTrue(any(line.startswith('QUAD_PROBE step=Q0e verdict=INCONCLUSIVE saving_ms=') for line in lines))

    def test_decide_only_rereads_a_report(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'saved.json'
            report = q0a_report(**{'matmul-q': dict(k_in0_fires=False)})
            path.write_text(json.dumps(report))
            lines = []
            with mock.patch('builtins.print', side_effect=lambda *args, **kwargs: lines.append(' '.join(map(str, args)))):
                self.assertEqual(probe.main(['--decide-only', str(path)]), 0)
                self.assertTrue(lines[0].startswith('QUAD_PROBE step=Q0a verdict=INCONCLUSIVE'), lines[0])
                lines.clear()
                probe.main(['--decide-only', str(path), '--matmul-control', 'either'])
                self.assertTrue(lines[0].startswith('QUAD_PROBE step=Q0a verdict=PASS'), lines[0])


class HarnessTests(unittest.TestCase):
    def test_a_wrong_binary_unpinned_sources_and_a_matmul_pin_fail(self):
        with tempfile.TemporaryDirectory() as directory:
            binary = Path(directory) / '_ttnncpp.so'
            binary.write_bytes(b'not the served binary')
            report = dict(failures=[], warnings=[])
            args = probe.parse_args(['--out', str(Path(directory) / 'r.json'), '--expect-binary-sha256', '0' * 64])
            with mock.patch.object(probe, 'loaded_binary', return_value=str(binary)), mock.patch('builtins.print'):
                self.assertFalse(probe.check_binary(args, report))
            self.assertIn('not the expected', report['failures'][0])
            report = dict(failures=[], warnings=[])
            self.assertFalse(probe.check_sources(directory, report, expect_matmul_kernel='1' * 64))
            self.assertEqual(len(report['failures']), 5)
            self.assertIn('retained BMM export', report['warnings'][0])
            self.assertIn('matmul', report['op_trees'])

    def test_the_watchdog_writes_and_exits_3(self):
        fired, exits = [], []
        watchdog = probe.Watchdog(5, on_fire=fired.append, backstop=False, exit=exits.append)
        watchdog.label, watchdog.deadline = 'hung call', 0.0
        with mock.patch('sys.stdout'):
            self.assertTrue(watchdog.check())
        self.assertEqual((fired, exits), (['hung call'], [3]))

    def test_the_arguments(self):
        args = probe.parse_args(['--out', 'x.json'])
        self.assertEqual((args.steps, args.seeds, args.shards, args.matmul_control),
                         (list(probe.STEPS), [0, 1, 2], 2, 'in0'))
        self.assertEqual((args.rounds, args.replays), (5, 20))
        for argv in (['--steps', 'Q9'], ['--steps', 'cores,Q0a'], ['--regimes', 'partner100'], ['--b-regimes', 'row0x100'],
                     ['--ops', 'matmul-x'], ['--expect-binary-sha256', 'abc'], ['--rounds', '0'], ['--shards', '0']):
            with self.subTest(argv=argv), mock.patch('sys.stderr'), self.assertRaises(SystemExit):
                probe.parse_args(['--out', 'x.json', *argv])


# ---------------------------------------------------------------------------------------------
# The runner and the kernel.
# ---------------------------------------------------------------------------------------------

def find_bash():
    candidates = []
    if os.name == 'nt':
        for root in (os.environ.get('ProgramW6432'), os.environ.get('ProgramFiles'), 'C:/Program Files'):
            if root:
                candidates.append(Path(root) / 'Git' / 'bin' / 'bash.exe')
    found = shutil.which('bash')
    if found and not (os.name == 'nt' and ('system32' in found.lower() or 'windowsapps' in found.lower())):
        candidates.append(Path(found))
    return next((str(candidate) for candidate in candidates if candidate.is_file()), None)


BASH = find_bash()
SCRUB = ('QUAL_CARD', 'ALLOW_SERVING_CARD', 'KOPGRAFT64', 'IMAGE', 'RESULTS', 'CARD_B_ARGS', 'EXPECT_TTNNCPP_SHA256',
         'WATCHER', 'WATCHDOG_S', 'QUAD_DRY_RUN', 'QUAD_CORES')


class FileTests(unittest.TestCase):
    def test_the_qual_card_block_is_canonical_and_every_file_is_lf(self):
        text = RUNNER.read_text(encoding='utf-8')
        begin, end = '# >>> qual_card.sh', '# <<< qual_card.sh'
        starts = [m.start() for m in re.finditer('^' + re.escape(begin), text, flags=re.M)]
        ends = [m.start() for m in re.finditer('^' + re.escape(end), text, flags=re.M)]
        self.assertEqual((len(starts), len(ends)), (1, 1))
        block = text[starts[0]:text.index(NL, ends[0]) + 1]
        self.assertEqual(block, (CI / 'qual_card.sh').read_text(encoding='utf-8'))
        for path in HERE.iterdir():
            if path.suffix in ('.py', '.sh', '.cpp'):
                self.assertNotIn(b'\r', path.read_bytes(), path.name)

    def test_the_kernel_pins_the_models_addressing(self):
        text = (HERE / 'quad_conv_io.cpp').read_text(encoding='utf-8')
        for expression in ('page < 160 * tile_rows; page += workers', 'tile_row * 10 + col / 16',
                           'noc_async_read_tile(col, base0', 'tile_row == 0 ? seams_low : seams_high',
                           'lane(row, 2 * (col % 16) + column / 16)', 'noc_async_write_tile(page, output'):
            self.assertIn(expression, text)


@unittest.skipUnless(BASH, 'bash not found')
class RunnerTests(unittest.TestCase):
    def run_runner(self, directory, **env):
        base = {name: value for name, value in os.environ.items() if name not in SCRUB}
        base.update(HOME=Path(directory).as_posix(), QUAD_DRY_RUN='1', MSYS_NO_PATHCONV='1')
        base.update(env)
        return subprocess.run([BASH, RUNNER.as_posix()], capture_output=True, text=True, timeout=120, env=base)

    def argv(self, result):
        lines = [line for line in result.stdout.splitlines() if line.startswith('### argv: ')]
        self.assertEqual(len(lines), 1, result.stdout + result.stderr)
        return shlex.split(lines[0][len('### argv: '):])

    def graft(self, directory, *, verify=True):
        graft = Path(directory) / 'graft'
        for part in ('attn_prep', 'nlp_concat_heads_decode', 'sdpa_decode', 'sdpa'):
            (graft / part).mkdir(parents=True)
        (graft / '_ttnn.so').write_bytes(b'ttnn')
        (graft / '_ttnncpp.so').write_bytes(b'ttnncpp')
        manifest = ''.join('%s  %s%s' % (hashlib.sha256((graft / name).read_bytes()).hexdigest(), name, NL)
                           for name in ('_ttnn.so', '_ttnncpp.so'))
        (graft / 'MANIFEST.sha256').write_bytes((manifest if verify else manifest.replace('0', '1', 1)).encode())
        return graft

    def test_the_dry_run_launches_on_card_b_with_the_served_graft_and_the_checkouts_modules(self):
        with tempfile.TemporaryDirectory() as directory:
            result = self.run_runner(directory)
            self.assertEqual(result.returncode, 0, result.stderr)
            argv = self.argv(result)
        self.assertEqual(argv[argv.index('--device') + 1], '/dev/tenstorrent/by-id/' + CARD_B)
        self.assertEqual(argv[argv.index('--name') + 1], 'qwen-quaddraft-card-b')
        mounts = [argv[index + 1] for index, word in enumerate(argv) if word == '--mount']
        for name in ('probe_quad_draft_card_b.py', 'quad_candidates.py', 'quad_conv_io.cpp', 'pair_row_exact.py',
                     'draft_attention.py', 'dflash_batched_mask.py', 'draft_shared_head.py', 'draft_head_preparation.py',
                     'dflash_t16_native_attention.py', 'draft_mlp.py', 'draft_convolution_fused_io.cpp',
                     'draft_convolution_fused_compute.cpp'):
            self.assertEqual(len([mount for mount in mounts if mount.endswith('dst=/bench/%s,readonly' % name)]), 1, name)
        self.assertFalse([mount for mount in mounts if 'experiment-scripts' in mount])
        ops = '/opt/tt-metal/ttnn/cpp/ttnn/operations/'
        for target in ('/opt/tt-metal/ttnn/ttnn/_ttnn.so', '/opt/tt-metal/build_Release/ttnn/_ttnncpp.so',
                       '/opt/tt-metal/build_Release/lib/_ttnncpp.so', ops + 'transformer/attn_prep',
                       ops + 'experimental/transformer/nlp_concat_heads_decode', ops + 'transformer/sdpa_decode',
                       ops + 'transformer/sdpa'):
            self.assertEqual(len([mount for mount in mounts if ',dst=%s,' % target in mount]), 1, target)
        self.assertTrue(all('opgraft-K64i' in mount for mount in mounts if '/opt/tt-metal/' in mount))
        self.assertIn('sha256:c9a585ef3eebb775c8de1883b0e0032ec3b16305feb1e23968e0661f401363e9', argv)
        self.assertEqual(argv[argv.index('--expect-binary-sha256') + 1], probe.K64I_SHA256)
        self.assertIn('TT_METAL_CACHE=/kcache', argv)
        self.assertIn('QWEN_SDPA_TREE_SCRATCH_ROUNDS=1', argv)
        self.assertTrue(any(word.startswith('QUAD_CORES_OUT=/results/cores-') for word in argv))
        inner = argv[argv.index('-c') + 1]
        self.assertIn('TT_METAL_DEVICE_PROFILER=1', inner)
        self.assertIn('--steps cores', inner)
        self.assertIn('--cores-report', inner)
        self.assertIn('exec python3 -B /bench/probe_quad_draft_card_b.py', inner)
        self.assertEqual(argv[argv.index('probe') + 1:argv.index('probe') + 3], ['--out', argv[argv.index('--out') + 1]])
        self.assertIn('--watchdog', argv)
        self.assertIn('dry run: ', result.stdout)

    def test_the_watcher_pass_and_no_graft(self):
        with tempfile.TemporaryDirectory() as directory:
            watcher = self.argv(self.run_runner(directory, WATCHER='1'))
            bare = self.argv(self.run_runner(directory, KOPGRAFT64='none'))
        self.assertIn('TT_METAL_WATCHER=5', watcher)
        self.assertEqual(watcher[watcher.index('--steps') + 1], 'Q0a,Q0b,Q0c,Q0d')
        self.assertEqual(watcher[watcher.index('--shards') + 1], '1')
        self.assertFalse(any(word.startswith('QUAD_CORES_OUT=') for word in watcher), 'no profiler under the watcher')
        self.assertEqual([word for word in bare if word.startswith('type=bind') and ',dst=/opt/tt-metal/' in word], [])
        self.assertEqual(bare[bare.index('--expect-binary-sha256') + 1], '')
        self.assertTrue([word for word in watcher if word.startswith('type=bind') and ',dst=/opt/tt-metal/' in word])

    def test_the_graft_is_checked_before_launch(self):
        with tempfile.TemporaryDirectory() as directory:
            graft = self.graft(directory)
            expected = hashlib.sha256(b'ttnncpp').hexdigest()
            ok = self.run_runner(directory, KOPGRAFT64=graft.as_posix(), EXPECT_TTNNCPP_SHA256=expected)
            self.assertEqual(ok.returncode, 0, ok.stderr)
            self.assertIn('manifest verified', ok.stdout)
            default = self.run_runner(directory, KOPGRAFT64=graft.as_posix())
            self.assertEqual(default.returncode, 1, 'the default expectation is K64i')
            self.assertIn('not cf54d716669be6b7', default.stderr)
            self.assertEqual(self.run_runner(directory, KOPGRAFT64=graft.as_posix(), EXPECT_TTNNCPP_SHA256='').returncode, 0)
            (graft / 'sdpa').rmdir()
            missing = self.run_runner(directory, KOPGRAFT64=graft.as_posix(), EXPECT_TTNNCPP_SHA256='')
            self.assertEqual(missing.returncode, 1)
            self.assertIn('sdpa missing', missing.stderr)
        with tempfile.TemporaryDirectory() as directory:
            bad = self.run_runner(directory, KOPGRAFT64=self.graft(directory, verify=False).as_posix(),
                                  EXPECT_TTNNCPP_SHA256='')
            self.assertEqual(bad.returncode, 1)
            self.assertIn('does not verify', bad.stderr)

    def test_the_runner_parses(self):
        result = subprocess.run([BASH, '-n', RUNNER.as_posix()], capture_output=True, text=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stderr)


STUB = r'''#pragma once
#include <cstdint>
template <uint32_t Base> struct TensorAccessorArgs {
    constexpr uint32_t next_compile_time_args_offset() const { return Base + 1; }
};
template <typename A> struct TensorAccessor { TensorAccessor(A, uint32_t, uint32_t) {} };
template <typename A> TensorAccessor(A, uint32_t, uint32_t) -> TensorAccessor<A>;
template <typename T> T get_arg_val(int index);
uint32_t get_write_ptr(uint32_t cb);
uint32_t get_read_ptr(uint32_t cb);
void cb_reserve_back(uint32_t cb, uint32_t pages);
void cb_push_back(uint32_t cb, uint32_t pages);
void cb_wait_front(uint32_t cb, uint32_t pages);
void cb_pop_front(uint32_t cb, uint32_t pages);
template <typename A> void noc_async_read_tile(uint32_t page, const TensorAccessor<A>& source, uint32_t address);
template <typename A> void noc_async_write_tile(uint32_t page, const TensorAccessor<A>& target, uint32_t address);
void noc_async_read_barrier();
void noc_async_write_barrier();
'''


def find_gxx():
    for candidate in (os.environ.get('QUAD_GXX'), shutil.which('g++'), shutil.which('clang++'),
                      'C:/Users/liamb/AppData/Local/stm32cube/bundles/gnu-tools-for-stm32/13.3.1+st.9/bin/'
                      'arm-none-eabi-g++.exe'):
        if candidate and Path(candidate).is_file():
            return candidate
    return None


GXX = find_gxx()


@unittest.skipUnless(GXX, 'no g++ (QUAD_GXX)')
class KernelTests(unittest.TestCase):
    def test_the_quad_kernel_compiles_where_the_served_one_does(self):
        with tempfile.TemporaryDirectory() as directory:
            include = Path(directory) / 'api' / 'dataflow'
            include.mkdir(parents=True)
            (include / 'dataflow_api.h').write_text(STUB)
            for source in (CI / 'draft_convolution_fused_io.cpp', HERE / 'quad_conv_io.cpp'):
                with self.subTest(source=source.name):
                    result = subprocess.run([GXX, '-std=c++17', '-fsyntax-only', '-Wall', '-Wextra', '-Werror',
                                             '-I', directory, '-x', 'c++', str(source)],
                                            capture_output=True, text=True, timeout=120)
                    self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == '__main__':
    unittest.main()
