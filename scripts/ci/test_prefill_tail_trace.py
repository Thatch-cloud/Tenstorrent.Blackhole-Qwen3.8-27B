"""PF3-T, the traced masked tail (prefill_tail_trace, levern_route), executed on a recording fake ttnn with an address-level DRAM allocator and a toy hybrid model.

The harness is small on purpose and models exactly what the lever depends on:
  * a first-fit DRAM allocator with addresses, so that a trace's intermediates are HOLES (freed during the capture, baked into the replay) and a persistent buffer allocated after the capture
    that overlaps a hole is SEEN when the trace replays (docs/tp4-traced-publish.md section 5);
  * trace capture that records the ops instead of running them, refuses a compile or a host write inside the capture, and bakes every runtime argument the ops were built with;
  * a toy GDN layer whose conv op takes valid_len as a baked runtime argument and returns the FIR's state (the last three real inputs), and whose chunk op masks pad rows inside the op when
    valid_len < T (so the stock tail masks by valid_len and the traced tail by gates, which is exactly the claim a card has to confirm on the real ops: C1 and C2 of the module docstring);
  * a toy full-attention layer that fills K/V through the chunk page table and reads the earlier context through the full page table and the chunk start tensor;
  * the real target_features.LayerOutputCapture around the tap layers, as dflash_prefill_window wraps them.

What is held:
  settings   the flags are strict and need Lever N (and NP1's scratch page at '1'); the audit, the buckets and the arena need '1';
  plan       the bucket is the stock one; the chunk page table names the real blocks then the sink; the pad vector and the conv-state selectors are exactly the stock tail's;
  exact      a traced tail equals the stock tail byte for byte (logits, every GDN state, the K/V of every real row, the drafter taps) at every valid length and chunk start tried, and the pad
             blocks reach only the sink;
  replay     after the warm a tail compiles nothing, replays one trace and stages a constant handful of buffers whatever the valid length;
  holes      the arena covers the regions the captures used (a later persistent allocation is never overwritten by a replay); without it the harness SEES the overwrite;
  fallbacks  vision tokens, an untraced bucket, another page-table width and another tap set run the stock pass; a staging failure latches the lever off and the request is served;
  audit      the audited tail compares the scratch, the output rows and the taps with the stock pass, serves the stock result, runs the first 8 and then every 24th, and a toy kernel whose
             masked arithmetic differs is caught (and goes unnoticed without the audit);
  warm       refuses a model whose conv op is not engaged; the hooks are off again after every pass; uninstall puts every shadow back and frees every buffer;
  route      the Lever N route installs it behind the flags only, imports nothing when they are unset, and warms it with the route."""

import sys
import types
import unittest
from collections import namedtuple
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import levern_route as route  # noqa: E402
import prefill_tail_trace as tail  # noqa: E402
import qwen_prefix_model_fixture as F  # noqa: E402
from target_features import LayerOutputCapture  # noqa: E402
from test_levern_route import Rig as RouteRig, assert_same_state  # noqa: E402

BLOCK = 64
D = 8            # hidden
QKV = 6          # conv channels: q 2, k 2, v 2
NV = 3           # value heads per chip
KH = 2           # key/value width
SINK = 90
WIDTH = 96       # page-table width (a multiple of 32)
POOL = 91        # blocks 0..SINK
DEVICE_BYTES = 1 << 22

ENV = {'QWEN_FAST_LEVER_N': '1', 'QWEN_FAST_SOLO_PACKED': '1', 'QWEN_FAST_PREFILL_TAIL_TRACE': '1'}
CONTROL = {'QWEN_FAST_LEVER_N': '1', 'QWEN_FAST_PREFILL_TAIL_TRACE': '0'}
AUDITED = dict(ENV, QWEN_FAST_PREFILL_TAIL_TRACE_AUDIT='1')


class HoleCorruption(AssertionError):
    pass


class DType(object):
    def __init__(self, name, torch_dtype, size):
        self.name, self.torch, self.size = name, torch_dtype, size

    def __repr__(self):
        return 'DataType.' + self.name


class Tensor(object):
    def __init__(self, ops, data, dtype, layout, on_device):
        self.ops, self.data, self.dtype, self.layout, self.on_device = ops, data, dtype, layout, on_device
        self.deallocated = False
        self.addr = None
        self.size = int(data.numel()) * dtype.size
        self.seq = None

    @property
    def shape(self):
        return tuple(self.data.shape)

    def memory_config(self):
        return 'DRAM'


class Allocator(object):
    """First fit over a flat address space; every allocation has a sequence number."""

    def __init__(self, total):
        self.total, self.live, self.seq = total, [], 0

    def alloc(self, size, owner):
        size = max(1, int(size))
        cursor = 0
        for tensor in sorted(self.live, key=lambda t: t.addr):
            if tensor.addr - cursor >= size:
                break
            cursor = max(cursor, tensor.addr + tensor.size)
        if cursor + size > self.total:
            raise MemoryError('out of device memory (%d bytes asked)' % size)
        owner.addr, self.seq = cursor, self.seq + 1
        owner.seq = self.seq
        self.live.append(owner)

    def free(self, owner):
        self.live.remove(owner)


class Device(object):
    def arch(self):
        return 'blackhole'


class Ops(object):
    """The ttnn stand-in: eager ops run at once, ops inside a capture are recorded, execute_trace replays them."""

    ROW_MAJOR_LAYOUT, TILE_LAYOUT = 'RM', 'TILE'
    DRAM_MEMORY_CONFIG, L1_MEMORY_CONFIG = 'DRAM', 'L1'
    MathFidelity = SimpleNamespace(HiFi4='HiFi4')

    def __init__(self):
        self.bfloat16 = DType('bfloat16', torch.bfloat16, 2)
        self.float32 = DType('float32', torch.float32, 4)
        self.uint32 = DType('uint32', torch.int64, 4)
        self.int32 = DType('int32', torch.int64, 4)
        self.alloc = Allocator(DEVICE_BYTES)
        self.programs = set()
        self.log = []
        self.capturing = None
        self.traces = {}
        self.fail_h2d = False
        self.executed = 0

    # -- tensors ---------------------------------------------------------------------------------
    def _make(self, data, dtype, layout, on_device=True):
        tensor = Tensor(self, data.to(dtype.torch).clone(), dtype, layout, on_device)
        if on_device:
            self.alloc.alloc(tensor.size, tensor)
            if self.capturing is not None:
                self.capturing['made'].append(tensor)
        return tensor

    def from_torch(self, tensor, dtype=None, layout=None, device=None, mesh_mapper=None, memory_config=None):
        dtype = dtype or self.bfloat16
        self.log.append(('from_torch', device is not None, tuple(tensor.shape)))
        return self._make(tensor, dtype, layout, device is not None)

    def zeros(self, shape, device=None, dtype=None, layout=None, memory_config=None):
        self.log.append(('zeros', tuple(shape)))
        tensor = self._make(torch.zeros(tuple(shape)), dtype or self.bfloat16, layout, True)
        tensor.ballast = tuple(shape)[-1] == tail.ARENA_WIDTH         # the arena: a buffer nobody reads, whose content a replay may overwrite
        return tensor

    def to_torch(self, tensor, mesh_composer=None):
        if tensor.deallocated:
            raise RuntimeError('to_torch of a deallocated tensor')
        self.log.append(('to_torch', tensor.shape))
        return tensor.data.clone()

    def deallocate(self, tensor):
        base = getattr(tensor, 'alias', tensor)
        while getattr(base, 'alias', None) is not None:
            base = base.alias
        if tensor.deallocated or base.deallocated:
            raise RuntimeError('deallocate of a deallocated tensor')
        tensor.deallocated = base.deallocated = True
        if base.on_device:
            self.alloc.free(base)
        self.log.append(('deallocate', tensor.shape))

    def ConcatMeshToTensor(self, mesh, dim):
        return SimpleNamespace(kind='concat', dim=dim)

    def ReplicateTensorToMesh(self, mesh):
        return SimpleNamespace(kind='replicate')

    def init_device_compute_kernel_config(self, arch, **keywords):
        return SimpleNamespace(arch=arch, **keywords)

    def synchronize_device(self, device):
        self.log.append(('synchronize',))

    def copy_host_to_device_tensor(self, host, device_tensor, cq_id=0):
        self.log.append(('h2d', host.shape))
        if self.fail_h2d:
            raise RuntimeError('copy_host_to_device_tensor: refused')
        if self.capturing is not None:
            raise AssertionError('a host write inside a trace capture')
        if host.on_device or device_tensor.deallocated or host.shape != device_tensor.shape or host.layout != device_tensor.layout:
            raise RuntimeError('copy_host_to_device_tensor: spec %s/%s into %s/%s' % (host.shape, host.layout, device_tensor.shape, device_tensor.layout))
        device_tensor.data.copy_(host.data.to(device_tensor.data.dtype))

    # -- ops -------------------------------------------------------------------------------------------
    def _program(self, name, inputs):
        key = (name,) + tuple(tuple(t.shape) for t in inputs)
        if key not in self.programs:
            if self.capturing is not None:
                raise AssertionError('a program compiled inside a trace capture: %r' % (key,))
            self.programs.add(key)
            self.log.append(('compile',) + key)

    def run(self, name, function, inputs, shape=None, dtype=None, layout='TILE', into=None):
        """One op: allocate the output (unless `into`), run `function(*input data)` now, or record it inside a capture. The output of a recorded op is a placeholder."""
        self._program(name, inputs)
        out = into if into is not None else self._make(torch.zeros(tuple(shape)), dtype or inputs[0].dtype, layout)

        def execute():
            result = function(*[t.data for t in inputs])
            if result is not None:
                out.data.copy_(result.to(out.data.dtype))
        self.log.append(('op', name))
        if self.capturing is not None:
            self.capturing['ops'].append(execute)
        else:
            execute()
        return out

    def add(self, a, b, memory_config=None):
        return self.run('add', lambda x, y: x + y, [a, b], a.shape, a.dtype)

    def matmul(self, a, b, memory_config=None, compute_kernel_config=None):
        return self.run('matmul', lambda x, y: torch.matmul(x, y), [a, b], (a.shape[0], a.shape[1], b.shape[2]), a.dtype)

    def copy(self, src, dst):
        if src.shape != dst.shape:
            raise RuntimeError('copy shape %s into %s' % (src.shape, dst.shape))
        return self.run('copy', lambda x: x, [src], None, None, into=dst)

    def reshape(self, tensor, shape):
        """A view: it shares the buffer (and the address) of `tensor`, and freeing either frees the buffer once."""
        self._program('reshape', [tensor])
        self.log.append(('op', 'reshape'))
        view = Tensor.__new__(Tensor)
        view.__dict__.update(tensor.__dict__)
        view.data = tensor.data.reshape(tuple(shape))
        view.alias = tensor
        return view

    def to_memory_config(self, tensor, config):
        return tensor

    # -- traces ----------------------------------------------------------------------------------------
    def begin_trace_capture(self, device, cq_id=0):
        if self.capturing is not None:
            raise AssertionError('nested trace capture')
        self.capturing = {'ops': [], 'made': []}
        return 'trace%d' % (len(self.traces) + 1)

    def end_trace_capture(self, device, trace_id, cq_id=0):
        capture, self.capturing = self.capturing, None
        holes = [(t.addr, t.size) for t in capture['made'] if t.deallocated]
        self.traces[trace_id] = dict(ops=capture['ops'], holes=holes, after=self.alloc.seq)

    def execute_trace(self, device, trace_id, cq_id=0, blocking=False):
        trace = self.traces[trace_id]
        self.log.append(('execute_trace', trace_id))
        self.executed += 1
        for start, size in trace['holes']:
            for tensor in self.alloc.live:
                if tensor.seq > trace['after'] and not getattr(tensor, 'ballast', False) and tensor.addr < start + size and start < tensor.addr + tensor.size:
                    raise HoleCorruption('trace %s writes [%d, %d), where the live buffer [%d, %d) sits' % (trace_id, start, start + size, tensor.addr, tensor.addr + tensor.size))
        for execute in trace['ops']:
            execute()

    def release_trace(self, device, trace_id):
        self.traces.pop(trace_id, None)

    def fill(self, cache, kv, page_table):
        """paged_fill_cache: row r of kv goes to block page_table[r // 64], offset r % 64."""
        def function(c, rows, table):
            rows = rows.reshape(-1, rows.shape[-1])
            for r in range(rows.shape[0]):
                c[int(table.reshape(-1)[r // BLOCK]), r % BLOCK] = rows[r].to(c.dtype)
        self._program('fill', [kv, page_table])
        self.log.append(('op', 'fill'))
        if self.capturing is not None:
            self.capturing['ops'].append(lambda: function(cache.data, kv.data, page_table.data))
        else:
            function(cache.data, kv.data, page_table.data)


# -----------------------------------------------------------------------------------------------------------------------------------------
# The toy model
# ---------------------------------------------------------------------------------------------------------------------------------------

GDN_MODULE = types.ModuleType('fake_gdn_tp')


def _prefill_conv(layer, qkv, cstate, valid_len, T):
    """The graft conv op: causal conv of [carry; qkv] and the FIR's state, the last three REAL inputs (valid_len is a baked runtime argument)."""
    ops = layer.ops

    def conv(x, c):
        padded = torch.cat([c, x], dim=1)
        return sum(w * padded[:, j:j + T, :] for j, w in enumerate((0.125, 0.25, 0.5, 1.0)))

    def state(x, c):
        padded = torch.cat([c, x], dim=1)
        return padded[:, valid_len:valid_len + 3, :] + 0.0

    out = ops.run('conv', conv, [qkv, cstate], (1, T, QKV), qkv.dtype)
    q = ops.run('slice_q', lambda o: o[..., 0:KH], [out], (1, T, KH), qkv.dtype)
    k = ops.run('slice_k', lambda o: o[..., KH:2 * KH], [out], (1, T, KH), qkv.dtype)
    v = ops.run('slice_v', lambda o: o[..., 2 * KH:3 * KH], [out], (1, T, KH), qkv.dtype)
    new_state = ops.run('conv_state_v%s' % ('' if layer.same_programs else valid_len), state, [qkv, cstate], (1, 3, QKV), qkv.dtype)
    ops.deallocate(out)
    return q, k, v, new_state


def _fir(layer, qkv, cstate, valid_len, T):
    """The FIR path (the graft conv op not engaged): the same arithmetic, no hook point."""
    return _prefill_conv(layer, qkv, cstate, valid_len, T)


GDN_MODULE._qwen_prefill_conv = _prefill_conv
sys.modules['fake_gdn_tp'] = GDN_MODULE


class ToyGDN(object):
    __module__ = 'fake_gdn_tp'

    def __init__(self, ops, index, device, break_mask=False, engaged=True):
        self.ops, self.index, self.mesh, self.same_programs = ops, index, device, True
        self.Nv = NV
        self.break_mask = break_mask
        self.engaged = engaged
        self.w = torch.arange(D * (QKV + NV + NV + 2), dtype=torch.float32).reshape(D, -1).sin() * 0.3
        zeros = lambda shape: ops.from_torch(torch.zeros(shape), dtype=ops.bfloat16, layout='TILE', device=device)
        self.rec_state = ops.from_torch(torch.zeros(1, KH, KH), dtype=ops.float32, layout='TILE', device=device)
        self.conv_carry = zeros((1, 3, QKV))
        self.conv_states = [zeros((1, 1, QKV)) for _ in range(4)]

    def _project_qkvzab(self, x, S, out_mc=None):
        ops, w = self.ops, self.w
        pieces = {}
        for name, lo, hi in (('qkv', 0, QKV), ('z', QKV, QKV + 2), ('a', QKV + 2, QKV + 2 + NV), ('b', QKV + 2 + NV, QKV + 2 + 2 * NV)):
            pieces[name] = ops.run('proj_' + name, lambda t, lo=lo, hi=hi: torch.matmul(t.reshape(1, S, D), w[:, lo:hi].to(t.dtype)),
                                   [x], (1, S, hi - lo), x.dtype)
        return pieces['qkv'], pieces['z'], pieces['a'], pieces['b']

    def forward_prefill(self, x, chunk_size=128, valid_len=None, capture_state=False):
        ops = self.ops
        T = x.shape[-2]
        qkv, z, a, b = self._project_qkvzab(x, T, out_mc='L1')
        conv_op = GDN_MODULE._qwen_prefill_conv if self.engaged else _fir
        q, k, v, new_state = conv_op(self, qkv, self.conv_carry, valid_len, T)
        ops.deallocate(qkv)
        dt = torch.linspace(-0.5, 0.5, NV)
        g = ops.run('gate_g', lambda t: -torch.nn.functional.softplus(t.float() + dt), [a], (1, T, NV), ops.float32)
        beta = ops.run('gate_beta', lambda t: torch.sigmoid(t.float()), [b], (1, T, NV), ops.float32)
        ops.deallocate(a)
        ops.deallocate(b)
        rec = ops.run('chunk_rec', lambda qq, kk, vv, bb, gg, s: self._scan(qq, kk, vv, bb, gg, s, valid_len, T, want='state'),
                      [q, k, v, beta, g, self.rec_state], (1, KH, KH), ops.float32)
        out = ops.run('chunk_out', lambda qq, kk, vv, bb, gg, s: self._scan(qq, kk, vv, bb, gg, s, valid_len, T, want='out'),
                      [q, k, v, beta, g, self.rec_state], (1, T, KH), ops.float32)
        for tensor in (q, k, v, beta, g, z):
            ops.deallocate(tensor)
        ops.copy(rec, self.rec_state)
        ops.deallocate(rec)
        ops.copy(new_state, self.conv_carry)
        if capture_state:
            for j in range(3):
                piece = ops.run('window_%d' % j, lambda s, j=j: s[:, j:j + 1, :], [new_state], (1, 1, QKV), new_state.dtype)
                ops.copy(piece, self.conv_states[j + 1])
                ops.deallocate(piece)
        ops.deallocate(new_state)
        res = ops.run('out_proj', lambda o: torch.matmul(o, torch.ones(KH, D) * 0.01), [out], (1, 1, T, D), ops.bfloat16)
        ops.deallocate(out)
        return res

    def _scan(self, q, k, v, beta, g, s0, valid_len, T, want):
        """The toy fused chunk op: the gated delta recurrence over the first valid_len rows; a pad row (t >= valid_len) is masked INSIDE the op."""
        state = s0.clone().reshape(KH, KH).float()
        outputs = []
        for t in range(T):
            if t < valid_len or self.break_mask:
                decay = torch.exp(g[0, t].mean())
                state = decay * state + beta[0, t].mean() * torch.outer(k[0, t].float(), v[0, t].float())
            outputs.append(torch.matmul(state.t(), q[0, t].float()))
        if want == 'state':
            return state.reshape(1, KH, KH)
        return torch.stack(outputs).reshape(1, T, KH)

    def reset(self):
        ops = self.ops
        for tensor, shape in [(self.rec_state, (1, KH, KH)), (self.conv_carry, (1, 3, QKV))] + [(c, (1, 1, QKV)) for c in self.conv_states]:
            zero = ops.from_torch(torch.zeros(shape), dtype=tensor.dtype, layout='TILE', device=ops.dev)
            ops.copy(zero, tensor)
            ops.deallocate(zero)


class ToyAttention(object):
    def __init__(self, ops, index, cache):
        self.ops, self.index, self.cache = ops, index, cache

    def forward_prefill_paged(self, x, cos, sin, page_table, chunk_page_table=None, chunk_start_idx=0, chunk_start_idx_tensor=None):
        ops = self.ops
        rows = x.shape[-2]
        kv = ops.run('kv', lambda t: t * 2.0 + float(self.index), [x], (1, 1, rows, D), x.dtype)
        # a fill as wide as the chunk page table: the stock pass slices the K/V down to the real blocks first
        width = chunk_page_table.shape[-1] * BLOCK
        piece = kv if width >= rows else ops.run('slice', lambda t: t[..., :width, :], [kv], (1, 1, width, D), kv.dtype)
        ops.fill(self.cache, piece, chunk_page_table)
        if piece is not kv:
            ops.deallocate(piece)
        ops.deallocate(kv)

        def context(c, table, csi, xx):
            start = int(csi.reshape(-1)[0])
            if start == 0:
                return torch.zeros(1, 1, rows, D, dtype=xx.dtype)
            used = table.reshape(-1)[:start // BLOCK].long()
            return (c[used].float().mean() * 0.001 * torch.ones(1, 1, rows, D)).to(xx.dtype)

        prior = ops.run('context', lambda table, csi, xx: context(self.cache.data, table, csi, xx), [page_table, chunk_start_idx_tensor, x], (1, 1, rows, D), x.dtype)
        res = ops.add(x, prior)
        ops.deallocate(prior)
        return res


class ToyLayer(object):
    def __init__(self, ops, index, full, device, cache=None, **keywords):
        self.is_full_attention = full
        self.attention = ToyAttention(ops, index, cache) if full else ToyGDN(ops, index, device, **keywords)
        self.ops = ops

    def forward(self, x, cos=None, sin=None, mode='decode', chunk_size=128, page_table=None, chunk_page_table=None, chunk_start_idx=None,
                chunk_start_idx_tensor=None, valid_len=None, **rest):
        if self.is_full_attention:
            return self.attention.forward_prefill_paged(x, cos, sin, page_table, chunk_page_table=chunk_page_table, chunk_start_idx_tensor=chunk_start_idx_tensor)
        return self.attention.forward_prefill(x, chunk_size=chunk_size, valid_len=valid_len, capture_state=True)


class ToyModel(object):
    num_devices = 4
    _PREFILL_MASK_BUCKETS = (128, 256, 512, 1024, 2048)

    def __init__(self, ops, layout='GAGAGA', **keywords):
        self.ops = ops
        ops.dev = self.device = self.mesh_device = Device()
        self.args = SimpleNamespace(gdn_chunk_size=128)
        self.kv = torch.zeros(POOL, BLOCK, D)
        self.cache = SimpleNamespace(data=self.kv)
        self.layers = [ToyLayer(ops, i, kind == 'A', self.device, cache=self.cache, **keywords) for i, kind in enumerate(layout)]
        self.vocab = 11
        self.table = (torch.arange(64 * D, dtype=torch.float32).reshape(64, D).sin())
        self.head = torch.arange(D * self.vocab, dtype=torch.float32).reshape(D, self.vocab).cos()
        self.scratch_bound = False

    # -- scratch ---------------------------------------------------------------------------------------
    def _ensure_gdn_prefill_scratch(self):
        return None

    def _bind_gdn_prefill_scratch(self):
        self.scratch_bound = True
        return 'previous'

    def _unbind_gdn_prefill_scratch(self, previous):
        self.scratch_bound = False

    def _reset_gdn_state_for_new_sequence(self):
        for layer in self.layers:
            if not layer.is_full_attention:
                layer.attention.reset()

    def _qwen_prefix_read_scratch(self):
        gdn = [layer.attention for layer in self.layers if not layer.is_full_attention]
        return [d.rec_state.data.clone() for d in gdn], [d.conv_carry.data.clone() for d in gdn], 0

    def _qwen_prefix_restore(self, rec, carry):
        gdn = [layer.attention for layer in self.layers if not layer.is_full_attention]
        for d, r, c in zip(gdn, rec, carry):
            d.rec_state.data.copy_(r)
            d.conv_carry.data.copy_(c)

    # -- the leaves ------------------------------------------------------------------------------------
    def embd(self, tok):
        return self.ops.run('embd', lambda t: self.table[t.long() % 64].reshape(1, -1, D), [tok], (1, tok.shape[-1], D), self.ops.bfloat16)

    def _apply_vision_merge(self, x, length):
        return x

    def _rope_tp_cos_sin_torch(self, start, length):
        cos = torch.cos(torch.arange(start, start + length, dtype=torch.float32)).reshape(1, 1, length, 1).expand(1, 1, length, 2).contiguous()
        return cos, cos * 0.5

    def _set_vision_merge(self, ids, vision_tokens, offset=0):
        return None

    def _forward_prefill_chunk_masked_tp(self, token_buf, valid_len, chunk_start, page_table, bucket, flex_sdpa=True, vision_tokens=None):
        ops = self.ops
        rep = ops.ReplicateTensorToMesh(self.device)
        tok = ops.from_torch(token_buf.to(torch.int32), dtype=ops.uint32, layout='RM', device=self.device, mesh_mapper=rep)
        x = self.embd(tok)
        x = ops.reshape(x, (1, 1, bucket, x.shape[-1]))
        x = ops.to_memory_config(x, 'DRAM')
        ops.deallocate(tok)
        x = self._apply_vision_merge(x, length=bucket)
        cos_t, sin_t = self._rope_tp_cos_sin_torch(chunk_start, bucket)
        cos = ops.from_torch(cos_t, dtype=ops.bfloat16, layout='TILE', device=self.device, mesh_mapper=rep)
        sin = ops.from_torch(sin_t, dtype=ops.bfloat16, layout='TILE', device=self.device, mesh_mapper=rep)
        full_pt = ops.from_torch(page_table, dtype=ops.int32, layout='RM', device=self.device)
        blk0 = chunk_start // BLOCK
        blkN = -(-(chunk_start + valid_len) // BLOCK)
        chunk_pt = ops.from_torch(page_table[:, blk0:blkN].contiguous(), dtype=ops.int32, layout='RM', device=self.device)
        csi = ops.from_torch(torch.tensor([chunk_start], dtype=torch.int32), dtype=ops.int32, layout='RM', device=self.device)
        for layer in self.layers:
            if layer.is_full_attention:
                x_new = layer.forward(x, cos=cos, sin=sin, mode='prefill', page_table=full_pt, chunk_page_table=chunk_pt, chunk_start_idx_tensor=csi)
            else:
                x_new = layer.forward(x, mode='prefill', chunk_size=self.args.gdn_chunk_size, valid_len=valid_len)
            ops.deallocate(x)
            x = x_new
        for tensor in (cos, sin, full_pt, chunk_pt, csi):
            ops.deallocate(tensor)
        return x

    def _masked_bucket_logits_tp(self, hidden, actual_len, bucket):
        ops = self.ops
        sel = torch.zeros(1, 1, 1, bucket)
        sel[0, 0, 0, actual_len - 1] = 1.0
        sel_tt = ops.from_torch(sel, dtype=ops.bfloat16, layout='TILE', device=self.device)
        x_last = ops.run('select', lambda s, h: torch.matmul(s, h), [sel_tt, hidden], (1, 1, 1, hidden.shape[-1]), hidden.dtype)
        ops.deallocate(sel_tt)
        logits = ops.run('lm_head', lambda xx: torch.matmul(xx.float(), self.head).reshape(1, 1, -1), [x_last], (1, 1, self.vocab), ops.float32)
        ops.deallocate(x_last)
        return logits

    def prefill_masked_bucket(self, token_ids, page_table, actual_len, chunk_start=0, bucket=None, flex_sdpa=True, vision_tokens=None, vis_row_offset=0):
        ops = self.ops
        if bucket is None:
            bucket = next(b for b in self._PREFILL_MASK_BUCKETS if actual_len <= b)
        if chunk_start == 0:
            self._reset_gdn_state_for_new_sequence()
        real = token_ids[:, :actual_len].to(torch.int32)
        token_buf = torch.cat([real, torch.zeros(1, bucket - actual_len, dtype=torch.int32)], dim=1) if bucket > actual_len else real
        self._set_vision_merge(token_buf, vision_tokens, vis_row_offset)
        hidden = self._forward_prefill_chunk_masked_tp(token_buf, actual_len, chunk_start, page_table, bucket, flex_sdpa=flex_sdpa, vision_tokens=vision_tokens)
        ops.synchronize_device(self.device)
        return self._masked_bucket_logits_tp(hidden, actual_len, bucket)

    # -- views -----------------------------------------------------------------------------------------
    def state(self):
        gdn = [layer.attention for layer in self.layers if not layer.is_full_attention]
        return [(d.rec_state.data.clone(), d.conv_carry.data.clone(), [c.data.clone() for c in d.conv_states]) for d in gdn]


Rig = namedtuple('Rig', 'ops model tail logs')


def build(environ=None, install=True, warm=True, taps=(1, 3), arena_mb=None, **keywords):
    """A toy model with the lever installed and warmed (environ None: the stock model)."""
    ops = Ops()
    model = ToyModel(ops, **keywords)
    logs = []
    config = None
    instance = None
    stock_warm(model)
    if environ is not None:
        environ = dict(environ)
        if environ.get(tail.FLAG) == '1':
            environ['QWEN_FAST_PREFILL_TAIL_TRACE_ARENA_MB'] = str(1 if arena_mb is None else arena_mb)         # the toy device holds 4 MiB
        config = tail.settings(environ)
        instance = tail.TailTrace(config, environ, ops=ops, addresses=lambda t: (t.addr,), extent=lambda t: t.size)
        if install:
            instance.install(model, lambda message, *values: logs.append(message.format(*values) if values else message), taps=taps, sink=SINK)
        if warm and config.traced:
            instance.warm(model, WIDTH)
    return Rig(ops, model, instance, logs)


def stock_warm(model):
    """The part of the model's own bucket warmup (warmup_prefill_masked_buckets) a traced tail still needs after the replay: the logits programs of every bucket."""
    ops = model.ops
    for bucket in ToyModel._PREFILL_MASK_BUCKETS:
        hidden = ops.from_torch(torch.zeros(1, 1, bucket, D), dtype=ops.bfloat16, layout='TILE', device=model.device)
        ops.deallocate(model._masked_bucket_logits_tp(hidden, bucket // 2, bucket))
        ops.deallocate(hidden)


def page_row(blocks):
    row = torch.zeros(1, WIDTH, dtype=torch.int32)
    row[0, :len(blocks)] = torch.tensor(blocks, dtype=torch.int32)
    return row


def tokens(n, seed=1):
    return torch.randint(1, 60, (1, n), generator=torch.Generator().manual_seed(seed), dtype=torch.int64)


def run_tail(rig, row, valid, start, seed, capture=None):
    """One request's tail through prefill_masked_bucket: first the stock 2,048-row chunks of [0, start) (bucket 2,048, which the lever never traces), then the tail.
    Returns (logits, state, kv)."""
    model = rig.model
    ids = tokens(start + valid, seed)
    model._bind_gdn_prefill_scratch()
    for lo in range(0, start, 2048):
        model.prefill_masked_bucket(ids[:, lo:lo + 2048], row, 2048, chunk_start=lo, bucket=2048)
    if capture is not None:
        with capture:
            logits = model.prefill_masked_bucket(ids[:, start:], row, valid, chunk_start=start)
    else:
        logits = model.prefill_masked_bucket(ids[:, start:], row, valid, chunk_start=start)
    return logits.data.clone(), model.state(), model.kv.clone()


def kv_rows(kv, row, length):
    blocks = row[0, :-(-length // BLOCK)].long()
    return kv[blocks].reshape(-1, D)[:length]


def same_state(case, a, b, message=''):
    for li, (x, y) in enumerate(zip(a, b)):
        case.assertTrue(torch.equal(x[0], y[0]), '%s layer %d recurrent' % (message, li))
        case.assertTrue(torch.equal(x[1], y[1]), '%s layer %d conv carry' % (message, li))
        for tap, (p, q) in enumerate(zip(x[2], y[2])):
            case.assertTrue(torch.equal(p, q), '%s layer %d window %d' % (message, li, tap))


class SettingsTests(unittest.TestCase):
    def test_unset_is_off_and_not_requested(self):
        self.assertEqual(tail.settings({}), tail.Settings(False, False, False, tail.DEFAULT_BUCKETS, tail.DEFAULT_ARENA_MB))
        self.assertFalse(tail.requested({}))

    def test_zero_is_the_timed_control_and_one_is_the_lever(self):
        self.assertEqual(tail.settings(CONTROL)[:3], (True, False, False))
        self.assertEqual(tail.settings(ENV)[:3], (True, True, False))
        self.assertEqual(tail.settings(AUDITED)[:3], (True, True, True))
        self.assertTrue(tail.requested(CONTROL))

    def test_anything_else_raises(self):
        for value in ('', '2', 'true', ' 1', 'on'):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    tail.settings(dict(ENV, QWEN_FAST_PREFILL_TAIL_TRACE=value))
        with self.assertRaises(ValueError):
            tail.settings(dict(ENV, QWEN_FAST_PREFILL_TAIL_TRACE_AUDIT='yes'))

    def test_the_lever_needs_lever_n_the_audit_needs_one_and_one_needs_the_sink_page(self):
        with self.assertRaises(ValueError):
            tail.settings({'QWEN_FAST_PREFILL_TAIL_TRACE': '0'})
        with self.assertRaises(ValueError):
            tail.settings(dict(CONTROL, QWEN_FAST_PREFILL_TAIL_TRACE_AUDIT='1'))
        with self.assertRaises(ValueError):
            tail.settings({'QWEN_FAST_LEVER_N': '1', 'QWEN_FAST_PREFILL_TAIL_TRACE': '1'})

    def test_buckets_and_the_arena(self):
        self.assertEqual(tail.settings(dict(ENV, QWEN_FAST_PREFILL_TAIL_TRACE_BUCKETS='128,1024')).buckets, (128, 1024))
        for text in ('', '2048', '256,128', '128,128', 'a', '100'):
            with self.subTest(text=text):
                with self.assertRaises(ValueError):
                    tail.settings(dict(ENV, QWEN_FAST_PREFILL_TAIL_TRACE_BUCKETS=text))
        with self.assertRaises(ValueError):
            tail.settings(dict(CONTROL, QWEN_FAST_PREFILL_TAIL_TRACE_BUCKETS='128'))
        self.assertEqual(tail.settings(dict(ENV, QWEN_FAST_PREFILL_TAIL_TRACE_ARENA_MB='0')).arena_mb, 0)
        for text in ('-1', '1.5', 'x', '4096'):
            with self.subTest(text=text):
                with self.assertRaises(ValueError):
                    tail.settings(dict(ENV, QWEN_FAST_PREFILL_TAIL_TRACE_ARENA_MB=text))

    def test_audit_due_is_the_first_eight_then_every_twenty_fourth(self):
        self.assertEqual([i for i in range(60) if tail.audit_due(i)], list(range(8)) + [31, 55])


class PlanTests(unittest.TestCase):
    def test_the_bucket_is_the_stock_one(self):
        for valid, bucket in ((1, 128), (128, 128), (129, 256), (256, 256), (257, 512), (512, 512), (513, 1024), (1024, 1024), (1025, 2048), (2048, 2048)):
            self.assertEqual(tail.bucket_for(valid), bucket)
        self.assertIsNone(tail.bucket_for(2049))
        for valid in range(1, 2049, 37):
            self.assertEqual(tail.bucket_for(valid), next(b for b in ToyModel._PREFILL_MASK_BUCKETS if valid <= b))

    def test_the_chunk_page_table_names_the_real_blocks_then_the_sink(self):
        row = list(range(100, 100 + WIDTH))
        plan, why = tail.plan_tail(300, 2048, WIDTH, row, SINK)
        self.assertIsNone(why)
        self.assertEqual((plan.bucket, plan.real_blocks), (512, 5))
        self.assertEqual(plan.chunk_row, row[32:37] + [SINK] * 3)
        plan, _ = tail.plan_tail(64, 0, WIDTH, row, SINK)
        self.assertEqual((plan.bucket, plan.chunk_row), (128, [100, SINK]))
        plan, _ = tail.plan_tail(1024, 4096, WIDTH, row, SINK)
        self.assertEqual(plan.chunk_row, row[64:80])

    def test_the_refusals(self):
        row = list(range(WIDTH))
        self.assertEqual(tail.plan_tail(0, 0, WIDTH, row, SINK)[1], 'valid_len')
        self.assertEqual(tail.plan_tail(1500, 2048, WIDTH, row, SINK)[1], 'bucket')
        self.assertEqual(tail.plan_tail(100, 2050, WIDTH, row, SINK)[1], 'unaligned')
        self.assertEqual(tail.plan_tail(1000, 5000 * 64 // 64 * 64, WIDTH, row, SINK)[1], 'width')
        self.assertEqual(tail.plan_tail(100, 0, WIDTH, row[:-1], SINK)[1], 'width')
        self.assertEqual(tail.plan_tail(100, 0, WIDTH, row, SINK, buckets=(256,))[1], 'bucket')

    def test_the_pad_vector_and_the_conv_state_selectors(self):
        self.assertEqual(tail.pad_vector(2, 5), [0.0, 0.0] + [tail.PAD_GATE] * 3)
        for valid in (1, 2, 3, 4, 100, 128):
            padded = list(range(-3, 0)) + list(range(200))          # the carry rows -3..-1, then the inputs 0..
            chosen = []
            for j, (kind, index) in tail.select_indices(valid):
                chosen.append(padded[index] if kind == 'c' else padded[3 + index])
            self.assertEqual(chosen, [valid - 3, valid - 2, valid - 1], valid)


def collect(model, slots):
    """The tap records of a LayerOutputCapture over a tail, the way dflash_prefill_window builds it."""
    return LayerOutputCapture(model, slots, snapshot=lambda t: t.data.clone(), release=lambda t: None, storage_ids=lambda t: (id(t),))


class ExactnessTests(unittest.TestCase):
    CASES = ((1, 0), (3, 0), (4, 0), (100, 0), (128, 0), (129, 0), (300, 2048), (512, 2048), (700, 2048), (1024, 4096), (1000, 4096), (64, 2048), (65, 2048), (5, 4096))

    def test_a_traced_tail_equals_the_stock_tail_byte_for_byte(self):
        for valid, start in self.CASES:
            with self.subTest(valid=valid, start=start):
                row = page_row(list(range(1, 1 + -(-(start + valid) // BLOCK))))
                stock, traced = build(None), build(ENV)
                a = run_tail(stock, row, valid, start, seed=valid + start)
                b = run_tail(traced, row, valid, start, seed=valid + start)
                self.assertTrue(torch.equal(a[0], b[0]), 'logits')
                same_state(self, a[1], b[1])
                self.assertTrue(torch.equal(kv_rows(a[2], row, start + valid), kv_rows(b[2], row, start + valid)), 'the K/V of every real row')
                self.assertGreaterEqual(traced.ops.executed, 1)

    def test_the_pad_blocks_reach_only_the_sink(self):
        valid, start = 300, 2048
        row = page_row(list(range(1, 1 + -(-(start + valid) // BLOCK))))
        stock, traced = build(None), build(ENV)
        a = run_tail(stock, row, valid, start, seed=5)
        b = run_tail(traced, row, valid, start, seed=5)
        real = set(row[0, :-(-(start + valid) // BLOCK)].tolist())
        for block in range(POOL):
            if block == SINK:
                continue
            if block in real:
                continue
            self.assertTrue(torch.equal(a[2][block], b[2][block]), 'block %d (no request owns it) changed' % block)
        self.assertFalse(torch.equal(b[2][SINK], torch.zeros(BLOCK, D)), 'the pad rows were written somewhere other than the sink, or nowhere')
        # the rows past the last real one inside the last real block are the pad rows' own K/V: equal here because the toy's pad rows are identical in both paths
        self.assertTrue(torch.equal(a[2][list(real)], b[2][list(real)]))

    def test_the_drafter_taps_are_answered_from_the_retained_outputs(self):
        for valid, start in ((100, 2048), (700, 2048)):
            with self.subTest(valid=valid):
                row = page_row(list(range(1, 1 + -(-(start + valid) // BLOCK))))
                stock, traced = build(None), build(ENV)
                records = []
                for rig in (stock, traced):
                    capture = collect(rig.model, (1, 3))
                    run_tail(rig, row, valid, start, seed=9, capture=capture.capture())
                    records.append(capture.records)
                self.assertEqual(sorted(records[0]), sorted(records[1]))
                for index in (1, 3):
                    self.assertTrue(torch.equal(records[0][index][..., :valid, :], records[1][index][..., :valid, :]), 'tap %d' % index)

    def test_a_break_in_the_masked_arithmetic_is_visible_to_the_comparison(self):
        """The toy's pad-row masking can be switched off (a kernel that does not mask): the stock tail then differs from the traced one, so the comparison above has teeth."""
        row = page_row(list(range(1, 4)))
        stock, traced = build(None, break_mask=True), build(ENV)
        a = run_tail(stock, row, 100, 0, seed=3)
        b = run_tail(traced, row, 100, 0, seed=3)
        with self.assertRaises(AssertionError):
            same_state(self, a[1], b[1])


class ReplayTests(unittest.TestCase):
    def test_nothing_compiles_after_the_warm_and_a_tail_stages_a_constant_handful(self):
        traced = build(ENV)
        programs = len(traced.ops.programs)
        row = page_row(list(range(1, 40)))
        counts = []
        for valid in (1, 100, 128, 129, 400, 900):
            mark = len(traced.ops.log)
            traced.model._bind_gdn_prefill_scratch()
            traced.model.prefill_masked_bucket(tokens(valid), row, valid, chunk_start=1024)
            kinds = {}
            for entry in traced.ops.log[mark:]:
                kinds[entry[0]] = kinds.get(entry[0], 0) + 1
            counts.append(kinds)
            self.assertEqual(kinds.get('execute_trace'), 1)
            self.assertEqual(kinds.get('h2d'), 9)
            self.assertNotIn('compile', kinds)
        self.assertEqual(len(traced.ops.programs), programs)
        # the replay is a few ops whatever the bucket; the stock pass is the whole forward every time
        stock = build(None)
        mark = len(stock.ops.log)
        stock.model.prefill_masked_bucket(tokens(100), row, 100, chunk_start=1024)
        eager_ops = sum(1 for entry in stock.ops.log[mark:] if entry[0] == 'op')
        mark = len(traced.ops.log)
        traced.model.prefill_masked_bucket(tokens(100), row, 100, chunk_start=1024)
        replay_ops = sum(1 for entry in traced.ops.log[mark:] if entry[0] == 'op')
        self.assertGreater(eager_ops, 40)
        self.assertLess(replay_ops, eager_ops // 4)

    def test_the_timing_lines(self):
        traced, control = build(ENV), build(CONTROL)
        row = page_row(list(range(1, 40)))
        for rig in (traced, control):
            rig.model._bind_gdn_prefill_scratch()
            rig.model.prefill_masked_bucket(tokens(100), row, 100, chunk_start=1024)
        self.assertTrue(any(line.startswith(tail.TIMING_MARKER + ' mode=traced bucket=128 valid=100 start=1024') for line in traced.logs))
        engaged = [line for line in traced.logs if line.startswith(tail.ENGAGED_MARKER)]
        self.assertEqual(len(engaged), 1)
        self.assertIn('freed=', engaged[0])
        self.assertIn('uncovered=0', engaged[0])
        self.assertTrue(any(line.startswith(tail.TIMING_MARKER + ' mode=eager bucket=128 valid=100 start=1024') for line in control.logs))
        self.assertFalse(any(tail.ENGAGED_MARKER in line for line in control.logs))
        self.assertEqual(sum(tail.ENGAGED_MARKER in line for line in traced.logs), 1)

    def test_the_stock_chunks_are_not_tails_and_are_not_timed(self):
        traced = build(ENV)
        row = page_row(list(range(1, 40)))
        traced.model._bind_gdn_prefill_scratch()
        traced.model.prefill_masked_bucket(tokens(1024), row, 1024, chunk_start=0, bucket=1024)
        self.assertEqual(traced.ops.executed, 1)            # a 1,024 tail IS a tail: traced
        mark = traced.ops.executed
        traced.model._forward_prefill_chunk_masked_tp(tokens(2048).to(torch.int32), 2048, 2048, row, 2048)
        self.assertEqual(traced.ops.executed, mark)         # the 2,048-row bucket is a chunk: stock
        self.assertFalse(any('bucket=2048' in line for line in traced.logs))


class HoleTests(unittest.TestCase):
    def test_the_arena_is_freed_for_the_captures_and_back_at_the_same_address(self):
        traced = build(ENV, arena_mb=1)
        self.assertIsNotNone(traced.tail.arena)
        self.assertEqual(traced.tail.arena_address, (traced.tail.arena.addr,))
        self.assertIn(traced.tail.arena, traced.ops.alloc.live)

    def test_a_buffer_allocated_after_the_captures_is_never_overwritten_with_the_arena(self):
        traced = build(ENV, arena_mb=1)
        late = traced.ops.from_torch(torch.zeros(1, 64, 4096), dtype=traced.ops.bfloat16, layout='TILE', device=traced.model.device)      # a persistent buffer the attach allocates later
        self.assertFalse(any(start < late.addr + late.size and late.addr < start + size for t in traced.ops.traces.values() for start, size in t['holes']))
        row = page_row(list(range(1, 40)))
        traced.model._bind_gdn_prefill_scratch()
        traced.model.prefill_masked_bucket(tokens(900), row, 900, chunk_start=1024)         # replays without HoleCorruption

    def test_without_the_arena_the_harness_sees_the_overwrite(self):
        traced = build(ENV, arena_mb=0)
        late = traced.ops.from_torch(torch.zeros(1, 64, 4096), dtype=traced.ops.bfloat16, layout='TILE', device=traced.model.device)
        self.assertTrue(any(start < late.addr + late.size and late.addr < start + size for t in traced.ops.traces.values() for start, size in t['holes']))
        row = page_row(list(range(1, 40)))
        traced.model._bind_gdn_prefill_scratch()
        with self.assertRaises(HoleCorruption):
            traced.model.prefill_masked_bucket(tokens(900), row, 900, chunk_start=1024)

    def test_an_arena_that_does_not_come_back_where_it_was_is_refused(self):
        ops = Ops()
        model = ToyModel(ops)
        instance = tail.TailTrace(tail.settings(dict(ENV, QWEN_FAST_PREFILL_TAIL_TRACE_ARENA_MB='1')), dict(ENV), ops=ops, addresses=lambda t: (t.addr + (1 if getattr(t, 'second', False) else 0),), extent=lambda t: t.size)
        instance.install(model, lambda *a: None, taps=(1, 3), sink=SINK)
        original = instance._arena_tensor
        calls = []

        def moved(rows):
            tensor = original(rows)
            if calls:
                tensor.second = True
            calls.append(rows)
            return tensor
        instance._arena_tensor = moved
        with self.assertRaises(ValueError):
            instance.warm(model, WIDTH)

    def test_a_free_gap_below_the_arena_that_a_capture_used_is_found_and_refused_at_the_warm(self):
        """The hole census: the toy allocator is first fit, so a gap left below the arena by an earlier free is used by the capture's small intermediates. That is a hole no ballast covers,
        and the warm fails by name instead of serving a trace that can overwrite whatever lands in the gap later."""
        ops = Ops()
        model = ToyModel(ops)
        small = ops.from_torch(torch.zeros(1, 1, 128, 128), dtype=ops.bfloat16, layout='TILE', device=model.device)
        keeper = ops.from_torch(torch.zeros(1, 1, 8, 8), dtype=ops.bfloat16, layout='TILE', device=model.device)
        ops.deallocate(small)                  # a gap of 32 KB below every later allocation
        env = dict(ENV, QWEN_FAST_PREFILL_TAIL_TRACE_ARENA_MB='1')
        instance = tail.TailTrace(tail.settings(env), env, ops=ops, addresses=lambda t: (t.addr,), extent=lambda t: t.size)
        instance.install(model, lambda *a: None, taps=(1, 3), sink=SINK)
        with self.assertRaises(ValueError) as caught:
            instance.warm(model, WIDTH)
        self.assertIn('outside the arena', str(caught.exception))
        self.assertGreater(instance.holes['uncovered'], 0)
        self.assertGreater(keeper.addr, 0)

    def test_a_clean_allocator_leaves_every_hole_inside_the_arena(self):
        traced = build(ENV)
        self.assertGreater(traced.tail.holes['recorded'], 100)
        self.assertEqual(traced.tail.holes['uncovered'], 0)

    def test_every_persistent_buffer_is_allocated_before_the_first_capture(self):
        traced = build(ENV)
        first_capture = min(trace['after'] for trace in traced.ops.traces.values())
        for trace in traced.tail.traces.values():
            for tensor in trace.buffers():
                self.assertLessEqual(tensor.seq, first_capture - 1, 'a persistent buffer was allocated after a capture')


class FallbackTests(unittest.TestCase):
    def setUp(self):
        self.row = page_row(list(range(1, 40)))

    def stock_result(self, valid, start, **keywords):
        rig = build(None)
        rig.model._bind_gdn_prefill_scratch()
        logits = rig.model.prefill_masked_bucket(tokens(valid), self.row, valid, chunk_start=start, **keywords)
        return rig, logits

    def test_vision_tokens_a_cold_bucket_and_another_width_run_the_stock_pass(self):
        traced = build(ENV)
        traced.model._bind_gdn_prefill_scratch()
        base = traced.ops.executed
        traced.model.prefill_masked_bucket(tokens(100), self.row, 100, chunk_start=1024, vision_tokens=object())
        self.assertEqual(traced.ops.executed, base)
        self.assertIn('vision', traced.tail.skips)
        wide = torch.zeros(1, WIDTH + 32, dtype=torch.int32)
        wide[0, :39] = self.row[0, :39]
        traced.model.prefill_masked_bucket(tokens(100), wide, 100, chunk_start=1024)
        self.assertEqual(traced.ops.executed, base)
        self.assertIn('width', traced.tail.skips)
        subset = build(dict(ENV, QWEN_FAST_PREFILL_TAIL_TRACE_BUCKETS='1024'))
        subset.model._bind_gdn_prefill_scratch()
        subset.model.prefill_masked_bucket(tokens(100), self.row, 100, chunk_start=1024)
        self.assertEqual(subset.ops.executed, 0)
        self.assertIn('bucket', subset.tail.skips)
        subset.model.prefill_masked_bucket(tokens(900), self.row, 900, chunk_start=1024)
        self.assertEqual(subset.ops.executed, 1)

    def test_a_warmup_call_with_an_explicit_bucket_larger_than_the_stock_one_is_stock(self):
        traced = build(ENV)
        traced.model._bind_gdn_prefill_scratch()
        traced.model.prefill_masked_bucket(tokens(64), self.row, 64, chunk_start=0, bucket=1024)
        self.assertEqual(traced.ops.executed, 0)

    def test_another_tap_set_runs_the_stock_pass(self):
        traced = build(ENV)
        capture = collect(traced.model, (0, 2))
        traced.model._bind_gdn_prefill_scratch()
        with capture.capture():
            traced.model.prefill_masked_bucket(tokens(100), self.row, 100, chunk_start=1024)
        self.assertEqual(traced.ops.executed, 0)
        self.assertIn('taps', traced.tail.skips)
        self.assertEqual(sorted(capture.records), [0, 2])

    def test_a_staging_failure_latches_the_lever_off_and_the_request_is_served(self):
        traced = build(ENV)
        traced.model._bind_gdn_prefill_scratch()
        traced.ops.fail_h2d = True
        logits = traced.model.prefill_masked_bucket(tokens(100), self.row, 100, chunk_start=1024)
        stock, expected = self.stock_result(100, 1024)
        self.assertTrue(torch.equal(logits.data, expected.data))
        self.assertIsNotNone(traced.tail.latched)
        self.assertEqual(traced.tail.fallbacks, 1)
        self.assertEqual(sum(tail.FELL_BACK_MARKER in line for line in traced.logs), 1)
        traced.ops.fail_h2d = False
        traced.model.prefill_masked_bucket(tokens(100), self.row, 100, chunk_start=1024)
        self.assertEqual(traced.ops.executed, 0)
        self.assertEqual(sum(tail.FELL_BACK_MARKER in line for line in traced.logs), 1)

    def test_the_control_arm_never_replays(self):
        control = build(CONTROL)
        control.model._bind_gdn_prefill_scratch()
        control.model.prefill_masked_bucket(tokens(100), self.row, 100, chunk_start=1024)
        self.assertEqual(control.ops.executed, 0)
        self.assertEqual(control.ops.traces, {})
        self.assertFalse(control.tail.warm(control.model, WIDTH))


class AuditTests(unittest.TestCase):
    def setUp(self):
        self.row = page_row(list(range(1, 40)))

    def run_tails(self, rig, count, valid=100, start=1024):
        for index in range(count):
            rig.model._bind_gdn_prefill_scratch()
            rig.model._reset_gdn_state_for_new_sequence()
            rig.model.prefill_masked_bucket(tokens(valid, index), self.row, valid, chunk_start=start)

    def test_an_exact_tail_passes_and_the_stock_result_is_served(self):
        audited = build(AUDITED)
        stock = build(None)
        stock.model._bind_gdn_prefill_scratch()
        expected = stock.model.prefill_masked_bucket(tokens(100, 0), self.row, 100, chunk_start=1024)
        audited.model._bind_gdn_prefill_scratch()
        logits = audited.model.prefill_masked_bucket(tokens(100, 0), self.row, 100, chunk_start=1024)
        self.assertTrue(torch.equal(logits.data, expected.data))
        self.assertTrue(any(line.startswith(tail.AUDIT_MARKER + ' n=1 exact=True bucket=128 valid=100') for line in audited.logs))
        same_state(self, audited.model.state(), stock.model.state())

    def test_the_first_eight_and_then_every_twenty_fourth_are_audited(self):
        audited = build(AUDITED)
        self.run_tails(audited, 60)
        self.assertEqual(audited.tail.audited, 10)
        self.assertEqual(sum(tail.AUDIT_MARKER + ' n=' in line for line in audited.logs), 10)

    def test_the_first_two_tails_of_every_bucket_are_audited_besides(self):
        audited = build(AUDITED)
        audited.model._bind_gdn_prefill_scratch()
        for valid in [100] * 9 + [200, 200, 200, 300, 300, 300, 900, 900, 900]:
            audited.model._reset_gdn_state_for_new_sequence()
            audited.model.prefill_masked_bucket(tokens(valid), self.row, valid, chunk_start=1024)
        # nine tails of bucket 128 (the schedule audits the first eight), then two each of 256, 512 and 1024 by the per-bucket rule alone
        self.assertEqual(audited.tail.audited, 8 + 2 + 2 + 2)
        self.assertEqual(sorted({line.split('bucket=')[1].split()[0] for line in audited.logs if line.startswith(tail.AUDIT_MARKER + ' n=')}), ['1024', '128', '256', '512'])

    def test_a_kernel_whose_masked_arithmetic_differs_is_caught_and_unnoticed_without_the_audit(self):
        audited = build(AUDITED, break_mask=True)
        audited.model._bind_gdn_prefill_scratch()
        with self.assertRaises(AssertionError) as caught:
            audited.model.prefill_masked_bucket(tokens(100), self.row, 100, chunk_start=1024)
        self.assertIn('audit mismatch', str(caught.exception))
        self.assertEqual(audited.tail.mismatches, 1)
        self.assertTrue(any(tail.AUDIT_MISMATCH_MARKER in line and 'exact=False' in line for line in audited.logs))
        unaudited = build(ENV, break_mask=True)
        unaudited.model._bind_gdn_prefill_scratch()
        unaudited.model.prefill_masked_bucket(tokens(100), self.row, 100, chunk_start=1024)          # break_mask models C1 being false: the traced tail is served and nothing complains
        self.assertEqual(unaudited.ops.executed, 1)

    def test_the_audit_compares_the_taps(self):
        audited = build(AUDITED)
        original = audited.ops.execute_trace

        def corrupting(*args, **keywords):
            original(*args, **keywords)
            audited.tail.traces[128].taps[3].data.add_(1.0)          # a tap buffer that is not what the layer produced
        audited.ops.execute_trace = corrupting
        audited.model._bind_gdn_prefill_scratch()
        with self.assertRaises(AssertionError) as caught:
            audited.model.prefill_masked_bucket(tokens(100), self.row, 100, chunk_start=1024)
        self.assertIn('audit mismatch', str(caught.exception))
        self.assertTrue(any('tap layer 3 rows' in line for line in audited.logs))


class WarmTests(unittest.TestCase):
    def test_a_model_whose_conv_op_is_not_engaged_is_refused(self):
        with self.assertRaises(ValueError) as caught:
            build(ENV, engaged=False)
        self.assertIn('conv op engaged', str(caught.exception))
        for layer in build(None).model.layers:
            self.assertNotIn('_project_qkvzab', vars(getattr(layer, 'attention')))
        self.assertIs(GDN_MODULE._qwen_prefill_conv, _prefill_conv)

    def test_the_hooks_are_off_again_after_the_warm_and_after_every_pass(self):
        traced = build(ENV)
        self.assertIs(GDN_MODULE._qwen_prefill_conv, _prefill_conv)
        for layer in traced.model.layers:
            if not layer.is_full_attention:
                self.assertNotIn('_project_qkvzab', vars(layer.attention))
        self.assertEqual(traced.tail.hook_calls['conv'], 3 * 4 * 2)       # three GDN layers, four buckets, a compile pass and a capture pass each
        self.assertEqual(traced.tail.hook_calls['project'], 3 * 4 * 2)

    def test_an_unfinished_conv_audit_would_run_inside_a_capture_and_is_refused(self):
        GDN_MODULE._QWEN_PREFILL_CONV = {'audited': 1}
        self.addCleanup(lambda: vars(GDN_MODULE).pop('_QWEN_PREFILL_CONV', None))
        env = dict(ENV, QWEN_FAST_GDN_PREFILL_CONV_AUDIT='4', QWEN_FAST_PREFILL_TAIL_TRACE_ARENA_MB='1')
        ops = Ops()
        model = ToyModel(ops)
        instance = tail.TailTrace(tail.settings(env), env, ops=ops, addresses=lambda t: (t.addr,), extent=lambda t: t.size)
        instance.install(model, lambda *a: None, taps=(1, 3), sink=SINK)
        with self.assertRaises(ValueError) as caught:
            instance.warm(model, WIDTH)
        self.assertIn('audited 1 of its 4', str(caught.exception))
        GDN_MODULE._QWEN_PREFILL_CONV = {'audited': 4}
        again = tail.TailTrace(tail.settings(env), env, ops=Ops(), addresses=lambda t: (t.addr,), extent=lambda t: t.size)
        ops2 = again._ops()
        model2 = ToyModel(ops2)
        again.install(model2, lambda *a: None, taps=(1, 3), sink=SINK)
        self.assertTrue(again.warm(model2, WIDTH))

    def test_the_warm_needs_a_page_table_width_in_multiples_of_32(self):
        ops = Ops()
        model = ToyModel(ops)
        instance = tail.TailTrace(tail.settings(ENV), dict(ENV), ops=ops, addresses=lambda t: (t.addr,), extent=lambda t: t.size)
        instance.install(model, lambda *a: None, taps=(1, 3), sink=SINK)
        with self.assertRaises(ValueError):
            instance.warm(model, 100)

    def test_the_warm_is_idempotent(self):
        traced = build(ENV)
        self.assertFalse(traced.tail.warm(traced.model, WIDTH))

    def test_uninstall_puts_every_shadow_back_and_frees_every_buffer(self):
        traced = build(ENV)
        model = traced.model
        before_live = len(traced.ops.alloc.live)
        self.assertIn('_forward_prefill_chunk_masked_tp', vars(model))
        traced.tail.uninstall()
        self.assertNotIn('_forward_prefill_chunk_masked_tp', vars(model))
        for index in (1, 3):
            self.assertNotIn('forward', vars(model.layers[index]))
        self.assertEqual(traced.ops.traces, {})
        self.assertLess(len(traced.ops.alloc.live), before_live)
        persistent = [t for t in traced.ops.alloc.live if t.size and t is not None]
        # what remains is the model's own state (the KV is host-side in the toy; the GDN states and nothing else)
        self.assertEqual(len(persistent), 3 * (1 + 1 + 4))

    def test_install_refuses_a_model_without_what_it_needs(self):
        ops = Ops()

        def attempt(model, **keywords):
            keywords.setdefault('sink', SINK)
            return tail.TailTrace(tail.settings(ENV), dict(ENV), ops=ops).install(model, lambda *a: None, taps=keywords.pop('taps', (1, 3)), **keywords)

        single = ToyModel(ops)
        single.num_devices = 1
        with self.assertRaises(ValueError):
            attempt(single)
        bare = ToyModel(ops)
        bare._rope_tp_cos_sin_torch = None
        with self.assertRaises(ValueError):
            attempt(bare)
        with self.assertRaises(ValueError):
            attempt(ToyModel(ops), sink=1)
        with self.assertRaises(ValueError):
            attempt(ToyModel(ops), taps=(1, 99))

        class Odd(ToyModel):
            _PREFILL_MASK_BUCKETS = (128, 256)
        with self.assertRaises(ValueError):
            attempt(Odd(ops))
        audited = tail.TailTrace(tail.settings(AUDITED), dict(AUDITED), ops=ops)
        model = ToyModel(ops)
        model._qwen_prefix_restore = None
        with self.assertRaises(ValueError):
            audited.install(model, lambda *a: None, taps=(1, 3), sink=SINK)

    def test_install_without_a_scratch_page_is_refused_by_name(self):
        import idle_scratch

        idle_scratch.forget()
        ops = Ops()
        with self.assertRaises(ValueError) as caught:
            tail.TailTrace(tail.settings(ENV), dict(ENV), ops=ops).install(ToyModel(ops), lambda *a: None, taps=(1, 3))
        self.assertIn('scratch page', str(caught.exception))
        idle_scratch.record(SINK, 1)
        self.addCleanup(idle_scratch.forget)
        installed = tail.TailTrace(tail.settings(ENV), dict(ENV), ops=ops).install(ToyModel(ops), lambda *a: None, taps=(1, 3))
        self.assertEqual(installed.sink, SINK)


class CensusTests(unittest.TestCase):
    def test_the_module_makes_no_ttnn_call_the_tests_have_not_seen(self):
        import ast

        seen = set()
        source = (HERE / 'prefill_tail_trace.py').read_text(encoding='utf-8')
        for node in ast.walk(ast.parse(source)):
            if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == 'ops':
                seen.add(node.attr)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Call):
                inner = node.func.value
                if isinstance(inner.func, ast.Attribute) and inner.func.attr == '_ops':
                    seen.add(node.func.attr)
        known = {'ConcatMeshToTensor', 'ReplicateTensorToMesh', 'add', 'begin_trace_capture', 'bfloat16', 'copy', 'copy_host_to_device_tensor', 'deallocate', 'end_trace_capture',
                 'execute_trace', 'from_torch', 'init_device_compute_kernel_config', 'int32', 'matmul', 'release_trace', 'reshape', 'synchronize_device', 'to_memory_config', 'to_torch',
                 'uint32', 'zeros', 'MathFidelity', 'ROW_MAJOR_LAYOUT', 'TILE_LAYOUT', 'DRAM_MEMORY_CONFIG'}
        self.assertEqual(seen - known, set(), 'the module calls ttnn ops the harness does not model')

    def test_the_stock_chunks_callers_never_free_a_traced_output(self):
        """The shadow returns a PERSISTENT output buffer: nothing in the repo may deallocate the return value of _forward_prefill_chunk_masked_tp for a bucket below 2,048 rows
        (the full chunks, which do, are bucket 2,048 and never traced)."""
        callers = []
        for path in sorted(HERE.glob('*.py')):
            if path.name.startswith('test_') or path.name == 'prefill_tail_trace.py':
                continue
            text = path.read_text(encoding='utf-8', errors='replace')
            if '_forward_prefill_chunk_masked_tp(' in text and path.name not in ('qwen_prefix_model_fixture.py',):
                callers.append(path.name)
        # prefill_prefix_resume is the TP2 eager resume (it refuses num_devices != 2 and runs 2,048-row chunks) and the probe is a measurement script: neither is on the four-card path.
        self.assertEqual(callers, ['dflash-prefill-window-probe.py', 'prefill_prefix_resume.py'],
                         'a new caller of _forward_prefill_chunk_masked_tp: check that it does not free the output of a traced tail, then name it here')


class RouteTests(unittest.TestCase):
    """levern_route.install / warm / uninstall on the REAL staged model.py with the toy of qwen_prefix_model_fixture (whose leaves have no GDN layers to trace: the traced path itself is the tests above)."""

    ON = {'QWEN_FAST_LEVER_N': '1'}

    def test_with_every_flag_unset_the_route_has_no_tail_trace_and_imports_nothing(self):
        rig = RouteRig(self.ON)
        self.assertIsNone(rig.handle.tailtrace)
        self.assertFalse(any('tailtrace' in line for line in rig.marker(route.levern_policy.ROUTE_INSTALLED_LINE.split('{')[0])))
        self.assertEqual(rig.model._forward_prefill_chunk_masked_tp, rig.toy.chunk_masked)

    def test_the_timed_control_installs_a_shadow_that_changes_nothing(self):
        plain, timed = RouteRig(self.ON), RouteRig(dict(self.ON, QWEN_FAST_PREFILL_TAIL_TRACE='0'))
        self.assertIsNotNone(timed.handle.tailtrace)
        self.assertFalse(timed.handle.tailtrace.settings.traced)
        self.assertTrue(any('tailtrace=timed' in line for line in timed.lines))
        self.assertEqual(timed.model._forward_prefill_chunk_masked_tp, timed.handle.tailtrace.shadow)
        for total in (4097, 6208, 8193):
            tokens_ = F.prompt(total, seed=total)
            a, b = plain.pool.row(total), timed.pool.row(total)
            logits_a, logits_b = plain.split('r', tokens_, a, 1), timed.split('r', tokens_, b, 1)
            self.assertTrue(torch.equal(logits_a, logits_b))
            assert_same_state(self, plain.state(a, total, 1), timed.state(b, total, 1), 'P=%d' % total)

    def test_uninstall_puts_the_model_back(self):
        timed = RouteRig(dict(self.ON, QWEN_FAST_PREFILL_TAIL_TRACE='0'))
        timed.handle.uninstall()
        self.assertEqual(timed.model._forward_prefill_chunk_masked_tp, timed.toy.chunk_masked)
        self.assertIsNone(timed.handle.tailtrace)

    def test_a_flag_without_the_route_or_with_a_bad_value_fails_the_attach(self):
        for environ in ({'QWEN_FAST_LEVERN_AUDIT': '1', 'QWEN_FAST_PREFILL_TAIL_TRACE': '0'}, dict(self.ON, QWEN_FAST_PREFILL_TAIL_TRACE='maybe'),
                        dict(self.ON, QWEN_FAST_PREFILL_TAIL_TRACE='1'), dict(self.ON, QWEN_FAST_PREFILL_TAIL_TRACE_ARENA_MB='5')):
            with self.subTest(environ=environ):
                with self.assertRaises(ValueError):
                    RouteRig(environ)

    def test_the_traced_flag_without_a_scratch_page_fails_the_attach_by_name(self):
        import idle_scratch

        idle_scratch.forget()
        with self.assertRaises(ValueError) as caught:
            RouteRig(dict(self.ON, QWEN_FAST_SOLO_PACKED='1', QWEN_FAST_PREFILL_TAIL_TRACE='1'))
        self.assertIn('scratch page', str(caught.exception))

    def test_the_warm_hook_and_the_route_source_name_the_lever_only_behind_its_flags(self):
        source = (HERE / 'levern_route.py').read_text(encoding='utf-8')
        self.assertIn("TAILTRACE_FLAGS = ('QWEN_FAST_PREFILL_TAIL_TRACE', 'QWEN_FAST_PREFILL_TAIL_TRACE_AUDIT', 'QWEN_FAST_PREFILL_TAIL_TRACE_BUCKETS', 'QWEN_FAST_PREFILL_TAIL_TRACE_ARENA_MB')", source)
        self.assertEqual(route.TAILTRACE_FLAGS, tail.FLAGS)
        self.assertEqual(source.count('import prefill_tail_trace'), 1)
        self.assertIn('if handle is not None and handle.tailtrace is not None:', source)


if __name__ == '__main__':
    unittest.main()
