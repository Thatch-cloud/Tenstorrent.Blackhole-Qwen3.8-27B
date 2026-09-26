"""A host stand-in for the image's qwen36 model tree, for the prefix-reuse model graft tests.

It executes the REAL model.py and qwen36_vllm.py (fixtures/qwen36_model.py and
fixtures/qwen36_vllm.py, the IMG bytes qwen_prefix_model_patch pins), stock or staged, against:

  * FakeTTNN, a recording ttnn: every op the model code calls is logged, device ops are checked
    the way the device would refuse them (a deallocated tensor, a shape or spec mismatch, and
    ttnn.deallocate(None), which raises here - design section 2.2's host-test requirement), and
    ttnn.copy "compiles" one program per (shape, dtypes), so a first compile after warmup shows;
  * a toy hybrid model, built with object.__new__ on the real Qwen36Model class so every method
    the graft touches runs as written. Only the leaf forwards are replaced (the chunk program, the
    masked bucket, the logits, RoPE and vision staging). The toy chunk program has the dependency
    structure section 2.0.4 relies on: a chunk's KV and GDN state depend on its tokens, positions,
    the KV of every earlier position (read through the page table) and the GDN recurrent state
    AND conv carry it starts from, in deterministic fp32. A resume from the wrong state, without
    the carry, or over the wrong blocks changes the bytes; an exact resume reproduces them.

The registry is qwen_prefix_registry.PrefixRegistry, driven the way the scheduler graft
(qwen_prefix_scheduler_patch) drives it: begin_step, stage, commit, with the capture plan computed
by the graft's own SchedulerGraft.plan (which plans a boundary below the loop's drain only once the
model has declared mid-loop captures, as its warmup does) and Q checked by the trim's token check.

What the fixture does NOT model (review finding 10): mesh mappers are recorded but a device tensor
is its logical host view (dim 0 = the chips), so how copy_host_to_device_tensor distributes a
sharded host tensor into a replicated scratch is only modelled through the h2d_mode failure cases;
bfloat8_b is stored as fp32 (bf8 exactness is a hardware property the design marks UNVERIFIED);
tile padding does not exist here.
"""

import contextlib
import hashlib
import os
import sys
import types
from array import array
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch

import qwen_prefix_model_patch as patcher
import qwen_prefix_registry as prefix_registry
import qwen_prefix_scheduler_patch as scheduler_patch

HERE = Path(__file__).resolve().parent
MODEL_FIXTURE = HERE / 'fixtures' / 'qwen36_model.py'
VLLM_FIXTURE = HERE / 'fixtures' / 'qwen36_vllm.py'
BLOCK = 64
CHUNK = 2048


# ---------------------------------------------------------------------------------------------
# The recording fake ttnn
# ---------------------------------------------------------------------------------------------

class FakeDType(object):
    def __init__(self, name, torch_dtype):
        self.name = name
        self.torch = torch_dtype

    def __repr__(self):
        return 'DataType.' + self.name.upper()


class FakeTensor(object):
    def __init__(self, data, dtype, layout, on_device):
        self.data = data
        self.dtype = dtype
        self.layout = layout
        self.on_device = on_device
        self.deallocated = False

    @property
    def shape(self):
        return tuple(self.data.shape)


class FakeMesh(object):
    shape = (1, 2)

    def __init__(self, fake):
        self.fake = fake
        self.traces = {}

    def get_num_devices(self):
        return 2

    def num_program_cache_entries(self):
        if not self.fake.program_count_known:
            raise AttributeError("'MeshDevice' object has no attribute 'num_program_cache_entries'")
        return len(self.fake.programs)


class Unfaked(object):
    def __init__(self, name):
        self.name = name

    def __call__(self, *args, **kwargs):
        raise AssertionError('ttnn.%s is not faked: the code under test reached an op the fixture '
                             'does not model' % self.name)


class FakeTTNN(object):
    """A ttnn stand-in that records every call. h2d_refuse makes copy_host_to_device_tensor refuse
    the spec (the design's UNVERIFIED case); h2d_mode makes it accept the spec and then write
    wrongly (review finding 2: 'noop' writes nothing, 'chip0' writes chip 0's shard only,
    'chip0_to_all' writes chip 0's shard to every chip - the host view's dim 0 is the mesh);
    copy_corrupt makes ttnn.copy write wrong bytes; to_torch_hook(tensor) may raise (MemoryError
    during a capture); program_count_known=False makes the mesh lack num_program_cache_entries.
    get_device_tensors / get_memory_view / BufferType model the DRAM allocator view G2's reading uses
    (one shard per chip of the 1x2 mesh, a view per chip from dram_views or DRAM_VIEW); every view read
    is recorded in memory_views, not in log, and memory_view_refuse makes the view refuse."""

    # One chip's DRAM as the allocator reports it, per bank (8 banks of 4,138,123,648 B).
    DRAM_VIEW = dict(num_banks=8, total_bytes_per_bank=4138123648, total_bytes_allocated_per_bank=3237000000,
                     total_bytes_free_per_bank=901123648, largest_contiguous_bytes_free_per_bank=880000000)

    H2D_MODES = ('exact', 'noop', 'chip0', 'chip0_to_all')

    def __init__(self):
        self.log = []
        self.programs = set()
        self.h2d_refuse = False
        self.h2d_mode = 'exact'
        self.copy_corrupt = False
        self.to_torch_hook = None
        self.program_count_known = True
        self.memory_view_refuse = False
        self.memory_views = []
        self.dram_views = {}
        self.bfloat16 = FakeDType('bfloat16', torch.bfloat16)
        self.float32 = FakeDType('float32', torch.float32)
        self.bfloat8_b = FakeDType('bfloat8_b', torch.float32)
        self.uint32 = FakeDType('uint32', torch.int64)
        self.int32 = FakeDType('int32', torch.int64)
        self.TILE_LAYOUT = 'TILE'
        self.ROW_MAJOR_LAYOUT = 'ROW_MAJOR'
        self.DRAM_MEMORY_CONFIG = 'DRAM'
        self.L1_MEMORY_CONFIG = 'L1'
        self.BufferType = SimpleNamespace(DRAM='DRAM', L1='L1')

    def module(self):
        module = types.ModuleType('ttnn')
        for name in ('bfloat16', 'float32', 'bfloat8_b', 'uint32', 'int32', 'TILE_LAYOUT',
                     'ROW_MAJOR_LAYOUT', 'DRAM_MEMORY_CONFIG', 'L1_MEMORY_CONFIG', 'BufferType'):
            setattr(module, name, getattr(self, name))
        for name in ('from_torch', 'as_tensor', 'to_torch', 'copy', 'copy_host_to_device_tensor',
                     'deallocate', 'synchronize_device', 'execute_trace', 'ConcatMeshToTensor',
                     'ShardTensorToMesh', 'ReplicateTensorToMesh', 'get_device_tensors', 'get_memory_view'):
            setattr(module, name, getattr(self, name))
        module.__getattr__ = Unfaked
        return module

    # -- tensors ---------------------------------------------------------------------------------
    def device_tensor(self, data, dtype):
        return FakeTensor(data.to(dtype.torch).clone(), dtype, self.TILE_LAYOUT, True)

    def dtype_of(self, data):
        return {torch.float32: self.float32, torch.bfloat16: self.bfloat16}.get(data.dtype, self.int32)

    def from_torch(self, tensor, dtype=None, layout=None, device=None, mesh_mapper=None, memory_config=None):
        dtype = dtype or self.dtype_of(tensor)
        self.log.append(('from_torch', device is not None, tuple(tensor.shape), dtype.name,
                         getattr(mesh_mapper, 'kind', None)))
        return FakeTensor(tensor.detach().to(dtype.torch).clone(), dtype, layout, device is not None)

    as_tensor = from_torch

    def to_torch(self, tensor, mesh_composer=None):
        if tensor is None or tensor.deallocated:
            raise RuntimeError('ttnn.to_torch of a missing or deallocated tensor')
        self.log.append(('to_torch', tensor.shape, getattr(mesh_composer, 'dim', None)))
        if self.to_torch_hook is not None:
            self.to_torch_hook(tensor)
        return tensor.data.clone()

    def copy(self, src, dst):
        for tensor in (src, dst):
            if tensor is None or tensor.deallocated or not tensor.on_device:
                raise RuntimeError('ttnn.copy needs two live device tensors')
        if src.shape != dst.shape:
            raise RuntimeError('ttnn.copy shape %s into %s' % (src.shape, dst.shape))
        key = ('copy', src.shape, src.dtype.name, dst.dtype.name)
        if key not in self.programs:
            self.programs.add(key)
            self.log.append(('compile',) + key)
        self.log.append(key)
        dst.data.copy_(src.data + 1 if self.copy_corrupt else src.data)

    def copy_host_to_device_tensor(self, host, device_tensor, cq_id=0):
        self.log.append(('h2d', host.shape, host.dtype.name))
        if self.h2d_refuse:
            raise RuntimeError('copy_host_to_device_tensor: tensor spec mismatch')
        if host.on_device or not device_tensor.on_device or device_tensor.deallocated:
            raise RuntimeError('copy_host_to_device_tensor needs a host source and a live device target')
        if (host.shape != device_tensor.shape or host.dtype.name != device_tensor.dtype.name
                or host.layout != device_tensor.layout):
            raise RuntimeError('copy_host_to_device_tensor: spec %s/%s/%s into %s/%s/%s' % (
                host.shape, host.dtype.name, host.layout, device_tensor.shape,
                device_tensor.dtype.name, device_tensor.layout))
        if self.h2d_mode not in self.H2D_MODES:
            raise ValueError('unknown h2d_mode %r' % self.h2d_mode)
        chip = host.data.shape[0] // 2
        if self.h2d_mode == 'exact':
            device_tensor.data.copy_(host.data)
        elif self.h2d_mode == 'chip0':
            device_tensor.data[:chip].copy_(host.data[:chip])
        elif self.h2d_mode == 'chip0_to_all':
            device_tensor.data[:chip].copy_(host.data[:chip])
            device_tensor.data[chip:].copy_(host.data[:chip])

    def deallocate(self, tensor):
        if tensor is None:
            raise TypeError('ttnn.deallocate(None)')
        if tensor.deallocated:
            raise RuntimeError('ttnn.deallocate of a deallocated tensor')
        tensor.deallocated = True
        self.log.append(('deallocate', tensor.shape))

    def synchronize_device(self, device):
        self.log.append(('synchronize',))

    def execute_trace(self, device, trace_id, cq_id=0, blocking=False):
        self.log.append(('execute_trace', trace_id))
        device.traces[trace_id]()

    def ConcatMeshToTensor(self, mesh, dim):
        return SimpleNamespace(kind='concat', dim=dim)

    def ShardTensorToMesh(self, mesh, dim):
        return SimpleNamespace(kind='shard', dim=dim)

    def ReplicateTensorToMesh(self, mesh):
        return SimpleNamespace(kind='replicate', dim=None)

    # -- the DRAM allocator view (G2's reading) ----------------------------------------------------
    def get_device_tensors(self, tensor):
        if tensor is None or tensor.deallocated or not tensor.on_device:
            raise RuntimeError('ttnn.get_device_tensors needs a live device tensor')
        return [SimpleNamespace(device=lambda chip=chip: 'chip%d' % chip) for chip in range(2)]

    def get_memory_view(self, device, buffer_type):
        self.memory_views.append((device, buffer_type))
        if self.memory_view_refuse:
            raise RuntimeError('no allocator on this device')
        return SimpleNamespace(**self.dram_views.get(device, self.DRAM_VIEW))


class FakeLogger(object):
    def __init__(self):
        self.records = []

    def _record(self, level):
        def emit(message, *args, **kwargs):
            self.records.append((level, message.format(*args) if args else message))
        return emit

    def __getattr__(self, level):
        return self._record(level)

    def lines(self, marker):
        return [message for _, message in self.records if marker in message]


# ---------------------------------------------------------------------------------------------
# Loading the real files
# ---------------------------------------------------------------------------------------------

def _packages(names):
    modules = {}
    for name in names:
        parts = name.split('.')
        for index in range(1, len(parts) + 1):
            dotted = '.'.join(parts[:index])
            modules.setdefault(dotted, types.ModuleType(dotted))
    return modules


def model_stubs(fake, logger):
    modules = _packages(['models.common.rmsnorm', 'models.demos.blackhole.qwen36.tt.layer',
                         'models.demos.blackhole.qwen36.tt.model_config', 'models.demos.blackhole.qwen36.tt.rope',
                         'models.tt_transformers.tt.common', 'loguru', 'tqdm'])
    modules['ttnn'] = fake.module()
    modules['loguru'].logger = logger
    modules['tqdm'].tqdm = lambda iterable, **kwargs: iterable
    modules['models.common.rmsnorm'].RMSNorm = type('RMSNorm', (), {})
    modules['models.demos.blackhole.qwen36.tt.layer'].Qwen36DecoderLayer = type('Qwen36DecoderLayer', (), {})
    modules['models.demos.blackhole.qwen36.tt.model_config'].Qwen36ModelArgs = type('Qwen36ModelArgs', (), {})
    modules['models.demos.blackhole.qwen36.tt.rope'].Qwen36RoPESetup = type('Qwen36RoPESetup', (), {})
    common = modules['models.tt_transformers.tt.common']
    common.Mode = SimpleNamespace(PREFILL='prefill', DECODE='decode')
    common.get_block_size = lambda caches: BLOCK
    common.num_blocks_in_seq = lambda seq, block: -(-seq // block)
    return modules


class StubGenerator(object):
    def warmup_model_decode(self, *args, **kwargs):
        return None


def vllm_stubs(fake, logger):
    modules = _packages(['vllm.model_executor.models.interfaces', 'vllm.model_executor.models.qwen3_5',
                         'vllm.multimodal', 'models.demos.blackhole.qwen36.tt.common',
                         'models.demos.blackhole.qwen36.tt.generator_interface',
                         'models.tt_transformers.tt.generator', 'loguru'])
    modules['ttnn'] = fake.module()
    modules['loguru'].logger = logger
    modules['vllm.model_executor.models.interfaces'].SupportsMultiModal = type('SupportsMultiModal', (), {})
    qwen3_5 = modules['vllm.model_executor.models.qwen3_5']
    for name in ('Qwen3_5ProcessingInfo', 'Qwen3VLDummyInputsBuilder', 'Qwen3VLMultiModalProcessor'):
        setattr(qwen3_5, name, type(name, (), {}))
    modules['vllm.multimodal'].MULTIMODAL_REGISTRY = SimpleNamespace(
        register_processor=lambda *args, **kwargs: (lambda cls: cls))
    modules['models.demos.blackhole.qwen36.tt.common'].create_tt_model = None
    interface = modules['models.demos.blackhole.qwen36.tt.generator_interface']
    interface.prefill_dispatch = None
    interface.warmup_decode_buckets = lambda wrapper, base, *args, **kwargs: None
    modules['models.tt_transformers.tt.generator'].Generator = StubGenerator
    return modules


_CODE = {}


def load_source(source, name, stubs, environ=None):
    """Execute source as a fresh module under the stubs. The code object is compiled once per
    source text (the model file is 175 KB); every load still runs its module body afresh."""
    module = types.ModuleType(name)
    module.__file__ = name + '.py'
    env = {} if environ is None else environ
    code = _CODE.get(source)
    if code is None:
        code = _CODE[source] = compile(source, 'qwen36_under_test.py', 'exec')
    with mock.patch.dict(sys.modules, stubs), mock.patch.dict(os.environ, env):
        if 'QWEN_PREFIX_REUSE' not in env:
            os.environ.pop('QWEN_PREFIX_REUSE', None)
        exec(code, module.__dict__)
    return module


def stock_sources():
    return MODEL_FIXTURE.read_text(encoding='utf-8'), VLLM_FIXTURE.read_text(encoding='utf-8')


_STAGED = []


def staged_sources():
    """The stage's output for the fixtures (computed once: each edit re-parses the 175 KB file)."""
    if not _STAGED:
        model, vllm = stock_sources()
        _STAGED.append((patcher.patch_model_source(model), patcher.patch_vllm_source(vllm)))
    return _STAGED[0]


# ---------------------------------------------------------------------------------------------
# The toy hybrid model
# ---------------------------------------------------------------------------------------------

class ToyGDN(object):
    """TPGatedDeltaNet's state handling, host-view shapes (mesh dim 0 = the two chips):
    rec_state [2*B, Nv, Dk, Dv] fp32, conv_states K x [2, B, D] and conv_carry [2, K-1, D] bf16."""

    def __init__(self, fake, B, index):
        self.fake = fake
        self.B = B
        self.index = index
        self.K = 4
        self.Nv, self.Dk, self.Dv, self.D = 2, 2, 2, 4
        self.conv_states = self.rec_state = self.conv_carry = None
        self._zero_conv0 = self._zero_conv_carry = self._zero_rec = None
        self._stable_state = False
        self.slots = {}
        self.reset_state()
        self._stable_state = True

    def reset_state(self):
        fake, B = self.fake, self.B
        z = lambda *shape: fake.device_tensor(torch.zeros(*shape), fake.bfloat16)  # noqa: E731
        self.conv_states = [z(2, B, self.D) for _ in range(self.K)]
        self.rec_state = fake.device_tensor(torch.zeros(2 * B, self.Nv, self.Dk, self.Dv), fake.float32)
        self.conv_carry = z(2, self.K - 1, self.D)
        self._zero_conv0 = z(2, B, self.D)
        self._zero_conv_carry = z(2, self.K - 1, self.D)
        self._zero_rec = z(2 * B, self.Nv, self.Dk, self.Dv)

    def reset_state_inplace(self):
        for cs in self.conv_states:
            self.fake.copy(self._zero_conv0, cs)
        self.fake.copy(self._zero_rec, self.rec_state)
        self.fake.copy(self._zero_conv_carry, self.conv_carry)

    def write_slot(self, slot, rec, convs):
        self.slots[slot] = (rec.data.clone(), [c.data.clone() for c in convs])


class Toy(object):
    """The leaf forwards, installed on a real Qwen36Model instance."""

    def __init__(self, module, fake, traced=False, num_blocks=4096, vocab=16, layout='GGAGA'):
        self.fake = fake
        self.segments = []
        self.ropes = []
        model = object.__new__(module.Qwen36Model)
        self.model = model
        mesh = FakeMesh(fake)
        model.device = model.mesh_device = mesh
        model.num_devices = 2
        model.args = SimpleNamespace(vocab_size=vocab, max_batch_size=4)
        model.layers = []
        model._paged_kv_caches = []
        for index, kind in enumerate(layout):
            if kind == 'G':
                model.layers.append(SimpleNamespace(is_full_attention=False, attention=ToyGDN(fake, 4, index)))
            else:
                caches = [fake.device_tensor(torch.zeros(num_blocks, 2, BLOCK, 2), fake.bfloat8_b) for _ in range(2)]
                model._paged_kv_caches.append(caches)
                model.layers.append(SimpleNamespace(is_full_attention=True, attention=None, kv=len(model._paged_kv_caches) - 1))
        model._gdn_prefill_scratch = None
        model._chunked_chunk_size = None
        model._chunked_trace_id = None
        model._forward_prefill_chunk_masked_tp = self.chunk_masked
        model.prefill_masked_bucket = self.masked_bucket
        model._masked_bucket_logits_tp = self.logits_from_hidden
        model._build_request_rope = lambda token_ids, vision_tokens: self.ropes.append(tuple(token_ids.shape))
        model._rope_tp_cos_sin_torch = lambda start, length: (torch.zeros(1, 1, length, 2), torch.zeros(1, 1, length, 2))
        model._set_vision_merge = lambda ids, vision_tokens, offset=0: None
        model._vis_row_offset_for = lambda token_ids, chunk_start: 0
        model.capture_prefill_trace_chunked = lambda *args, **kwargs: self.segments.append(('capture-trace',))
        self.vocab = vocab
        if traced:
            self.enable_trace(buf_blocks=num_blocks)

    def enable_trace(self, buf_blocks):
        model, fake = self.model, self.fake
        model._chunked_chunk_size = CHUNK
        model._chunked_trace_id = 'chunk-trace'
        model._chunk_token_buf = fake.device_tensor(torch.zeros(1, CHUNK, dtype=torch.int64), fake.uint32)
        model._chunk_token_buf.layout = fake.ROW_MAJOR_LAYOUT
        model._chunk_start_idx_tensor = fake.device_tensor(torch.zeros(1, dtype=torch.int64), fake.int32)
        model._chunk_start_idx_tensor.layout = fake.ROW_MAJOR_LAYOUT
        model._chunk_full_page_table_buf = fake.device_tensor(torch.zeros(1, buf_blocks, dtype=torch.int64), fake.int32)
        model._chunk_full_page_table_buf.layout = fake.ROW_MAJOR_LAYOUT
        model._chunk_page_table_buf = fake.device_tensor(torch.zeros(1, CHUNK // BLOCK, dtype=torch.int64), fake.int32)
        model._chunk_page_table_buf.layout = fake.ROW_MAJOR_LAYOUT
        model._chunk_cos_buf = fake.device_tensor(torch.zeros(1, 1, CHUNK, 2), fake.bfloat16)
        model._chunk_sin_buf = fake.device_tensor(torch.zeros(1, 1, CHUNK, 2), fake.bfloat16)
        model._chunked_trace_output = fake.device_tensor(torch.zeros(1, 2), fake.float32)
        model.mesh_device.traces['chunk-trace'] = self.trace_body

    # -- the chunk program -----------------------------------------------------------------------
    def segment(self, tokens, start, page_row, path):
        """One forward over tokens at absolute positions [start, start+n): writes KV, advances the
        bound GDN scratch, returns the segment's hidden summary."""
        model = self.model
        n = int(tokens.numel())
        self.segments.append((path, int(start), n))
        toks = tokens.reshape(-1).to(torch.float32)
        positions = torch.arange(start, start + n, dtype=torch.float32)
        rows = page_row.reshape(-1).to(torch.long)
        h = torch.zeros((), dtype=torch.float32)
        for layer in model.layers:
            if layer.is_full_attention:
                k_cache, v_cache = model._paged_kv_caches[layer.kv]
                prior = self.gather(k_cache.data, rows, start).sum() if start else torch.zeros(())
                vals = toks * 0.001 + positions * 1e-5 + prior * 1e-7 + h * 1e-3
                where = torch.arange(start, start + n)
                blocks, offsets = rows[where // BLOCK], where % BLOCK
                shape = (n, k_cache.data.shape[1], k_cache.data.shape[3])
                k_cache.data[blocks, :, offsets, :] = vals.reshape(n, 1, 1).expand(shape)
                v_cache.data[blocks, :, offsets, :] = (vals * 2).reshape(n, 1, 1).expand(shape)
                h = h + vals.sum() * 1e-6
            else:
                dn = layer.attention
                carry = dn.conv_carry.data.to(torch.float32)
                # Both chips' carries feed the next chunk (each chip holds its own shard on the
                # device), so a restore that loses or corrupts either chip's carry changes the bytes.
                window = torch.cat([carry[0, :, 0] + 0.25 * carry[1, :, -1], toks])
                conv = window[3:] + 0.5 * window[2:-1] + 0.25 * window[1:-2] + 0.125 * window[:-3]
                update = conv.sum() * 1e-4 + h * 1e-3 + dn.index
                dn.rec_state.data.copy_(dn.rec_state.data * 0.99 + update)
                tail = window[-3:]
                chip = torch.tensor([0.0, 0.5]).reshape(2, 1, 1)
                new_carry = (tail.reshape(1, 3, 1) + chip).expand(2, 3, dn.D)
                dn.conv_carry.data.copy_(new_carry.to(dn.conv_carry.data.dtype))
                dn.conv_states[0].data.zero_()
                for j in range(dn.K - 1):
                    dn.conv_states[j + 1].data.copy_(new_carry[:, j:j + 1, :].to(torch.bfloat16))
                h = h + dn.rec_state.data.sum() * 1e-6
        return h

    @staticmethod
    def gather(cache, rows, length):
        blocks = rows[:-(-length // BLOCK)]
        seq = cache.index_select(0, blocks).permute(1, 0, 2, 3).reshape(cache.shape[1], -1, cache.shape[3])
        return seq[:, :length]

    def hidden(self, h, last):
        return self.fake.device_tensor(torch.stack([h, last.to(torch.float32)]).reshape(1, 2), self.fake.float32)

    def logits(self, h):
        gdn = sum(layer.attention.rec_state.data.sum() for layer in self.model.layers if not layer.is_full_attention)
        row = torch.arange(self.vocab, dtype=torch.float32) * (h + gdn * 1e-3)
        return FakeTensor(row.reshape(1, 1, -1).expand(2, 1, -1).clone(), self.fake.float32, 'TILE', False)

    def chunk_masked(self, token_buf, valid_len, chunk_start, page_table, bucket, flex_sdpa=True, vision_tokens=None):
        tokens = token_buf[0, :valid_len]
        h = self.segment(tokens, chunk_start, page_table[0], 'eager')
        return self.hidden(h, tokens[-1])

    def masked_bucket(self, token_ids, page_table, actual_len, chunk_start=0, bucket=None, flex_sdpa=True,
                      vision_tokens=None, vis_row_offset=0):
        if chunk_start == 0:
            self.model._reset_gdn_state_for_new_sequence()
            self.model._build_request_rope(token_ids[:, :actual_len], vision_tokens)
        h = self.segment(token_ids[0, :actual_len], chunk_start, page_table[0], 'tail')
        return self.logits(h)

    def logits_from_hidden(self, hidden, actual_len, bucket):
        return self.logits(hidden.data[0, 0])

    def trace_body(self):
        model = self.model
        tokens = model._chunk_token_buf.data[0]
        start = int(model._chunk_start_idx_tensor.data[0])
        h = self.segment(tokens, start, model._chunk_full_page_table_buf.data[0], 'traced')
        model._chunked_trace_output.data.copy_(torch.stack([h, tokens[-1].to(torch.float32)]).reshape(1, 2))

    # -- views for the comparisons ---------------------------------------------------------------
    def kv(self, page_row, length):
        rows = torch.as_tensor(page_row).reshape(-1).to(torch.long)
        return [self.gather(cache.data, rows, length).clone()
                for pair in self.model._paged_kv_caches for cache in pair]

    def slot(self, slot):
        return [layer.attention.slots.get(slot) for layer in self.model.layers if not layer.is_full_attention]


def build_wrapper(vllm_module, toy):
    wrapper = object.__new__(vllm_module.Qwen36ForCausalLM)
    wrapper.model = [toy.model]
    wrapper.mesh_device = toy.model.mesh_device
    return wrapper


# ---------------------------------------------------------------------------------------------
# The scheduler side, as the scheduler graft drives the registry
# ---------------------------------------------------------------------------------------------

def as_ids(tokens):
    return tokens.tolist() if hasattr(tokens, 'tolist') else list(tokens)


def key_at(tokens, pos):
    """A stand-in for vLLM's chained block hash at pos: a function of every token before it."""
    return hashlib.sha256(array('q', as_ids(tokens)[:pos]).tobytes()).hexdigest()


class Pool(object):
    """vLLM's block ids: fresh blocks per request, the first Q/64 shared on a hit."""

    def __init__(self, width=4096, first=1):
        self.width = width
        self.next = first

    def row(self, length, shared=()):
        shared = list(shared)
        need = -(-length // BLOCK) - len(shared)
        fresh = list(range(self.next, self.next + need))
        self.next += need
        blocks = shared + fresh
        if self.next > self.width:
            raise ValueError('toy pool exhausted')
        return torch.tensor([blocks + [0] * (self.width - len(blocks))], dtype=torch.int32)


class BlockHashes(object):
    """request.block_hashes as vLLM keeps it (one per full 64-token block), computed on demand:
    entry i is key_at(tokens, (i + 1) * 64)."""

    def __init__(self, tokens):
        self.tokens = tokens

    def __len__(self):
        return len(self.tokens) // BLOCK

    def __getitem__(self, index):
        if not 0 <= index < len(self):
            raise IndexError(index)
        return key_at(self.tokens, (index + 1) * BLOCK)


def admit(registry, req_id, tokens, q, plan=None, h=None, step=True, force_plan=False):
    """Stage and commit one grant as SchedulerGraft.trim + commit do; returns start_pos.

    The plan is the scheduler graft's own (SchedulerGraft.plan: the prompt's last boundary above Q
    plus the gap boundary floor2048(h), the latter only once the model declared mid-loop captures on
    this registry); a test's explicit plan must equal it unless force_plan,
    which hands the model a plan the scheduler would not make. Q's checkpoint must pass the trim's
    token check (Checkpoint.matches), or the scheduler would have lowered Q."""
    if step:
        registry.begin_step()
    tokens = as_ids(tokens)
    h = q if h is None else h
    request = SimpleNamespace(request_id=req_id, all_token_ids=tokens, num_prompt_tokens=len(tokens),
                              num_tokens=len(tokens), block_hashes=BlockHashes(tokens))
    key = key_at(tokens, q) if q else None
    checkpoint = registry.get(key) if q else None
    if q and checkpoint is None:
        raise AssertionError('the test asked for a hit at %d with no checkpoint' % q)
    if q and (checkpoint.pos != q or not checkpoint.matches(tokens[0:q])):
        raise AssertionError('the trim would refuse the checkpoint at %d (token mismatch)' % q)
    planned, unplanned, drain = scheduler_patch.SchedulerGraft.plan(SimpleNamespace(registry=registry), request, h, q)
    if plan is None:
        pairs = planned
    elif force_plan:
        pairs = [(pos, key_at(tokens, pos)) for pos in plan]
    else:
        if list(plan) != [pos for pos, _ in planned]:
            raise AssertionError('the test plan %r is not the scheduler graft\'s %r (L=%d h=%d Q=%d)' % (
                list(plan), [pos for pos, _ in planned], len(tokens), h, q))
        pairs = planned
    registry.stage(prefix_registry.Grant(req_id, q, h, key, checkpoint, pairs, request, drain, unplanned))
    registry.commit({req_id: q})
    return q


@contextlib.contextmanager
def registry_installed(registry):
    holder = types.ModuleType(patcher.REGISTRY_KEY)
    holder.registry = registry
    with mock.patch.dict(sys.modules, {patcher.REGISTRY_KEY: holder}):
        yield registry


def prompt(length, seed):
    generator = torch.Generator().manual_seed(seed)
    return torch.randint(1, 5000, (length,), generator=generator, dtype=torch.int64)
