"""Stage E, E0: the census (serving_parked_engines' docstring, design section 3.5).

CensusOps is a fake ttnn that keeps no values but logs every device allocation (with the captures
completed so far), every read, every write, every capture and every replay. A capture records, per
tensor it touches, whether its FIRST access was a read (the trace reads what it was given: an input)
or a write (written in-trace before it is read); it counts the tensors it allocated and freed again
(its holes). A replay then:
  - fails on an input that has been freed, or that is EXPOSED (below);
  - marks every tensor it writes, and its own outputs, as freshly written;
  - EXPOSES every live tensor allocated after its capture ended, when it has holes: such a tensor may
    sit where the capture's freed intermediates were, and the replay writes over it. An exposed tensor
    is safe again once written in full; reading it first is an R1 (or, for a trace output read after
    another trace replayed, R2) violation.
Eager reads (copy, slice, readbacks...) are checked the same way.

The world runs the REAL serving_buffer_pool.ServingBufferPool, serving_request_factory.from_prefill,
dflash_device.DFlashDevice, draft_kv_history.DraftKVHistory, dflash_proposal_trace.PreparedDFlashProposal,
verifier_engine.VerifierEngine, greedy_session.GreedySession, dflash_request_runtime and
serving_fast_request.FastRequest over it. Faked, because they are device programs this host cannot run:
the verifier's ModelBatch forward (CensusFixture, which reads and writes what design section 3.2 says the
verify trace does), its commit DMAs, its sampler and target taps, the draft pass itself
(execute_proposal and select_proposal), the draft K/V projection and the projection's collective, and a
packed-block trace captured at attach that reads the pool's carries.

Tests here: today's per-request churn is census-clean, with every live tensor owned by a known holder
and nothing left behind; the census's own negative controls; R2's replay ledger
(QWEN_FAST_PARKED_AUDIT); the lifetime-invariant list; the source-pin rules. test_parked_engine_set
reuses the world for the parked cycles.
"""

from contextlib import ExitStack, contextmanager
from itertools import count
import os
from pathlib import Path
import re
import subprocess
import sys
from types import FunctionType, MethodType, ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
for _path in (HERE, ROOT / 'speculative-decoding' / 'harness'):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import torch  # noqa: E402

import serving_parked_engines as parked  # noqa: E402
import verifier_engine  # noqa: E402

# The phase-1 commit Stage E was written on (s2/sticky-sessions): the NEVER_EDITED files must equal it byte
# for byte, and the flag-off parity runs its verifier_engine and serving_buffer_pool as the reference.
BASE_COMMIT = '927e1fd9'
ITEMSIZE = {'bf16': 2, 'bf8': 1, 'f32': 4, 'u32': 4, 'i32': 4, 'u16': 2}
TORCH_DTYPES = {'bf16': torch.bfloat16, 'bf8': torch.bfloat16, 'f32': torch.float32, 'u32': torch.int32,
                'i32': torch.int32, 'u16': torch.int16}
MESH_CHIPS = 2
# A small but real geometry: 68 pages of 64 tokens (the serving floor), four scheduler slots.
PAGE_WIDTH = 68
USERS = 4
GDN_LAYERS = 48
LIVE_PER_LAYER = 1


def payload_digest(value):
    import hashlib

    flat = value.detach().contiguous().reshape(-1)
    try:
        raw = flat.view(torch.uint8).numpy().tobytes()
    except (RuntimeError, TypeError):
        raw = repr(flat.tolist()).encode()
    return (tuple(value.shape), str(value.dtype), hashlib.sha256(raw).hexdigest()[:16])


class CensusShard:
    def __init__(self, tensor, chip, address):
        self.tensor, self.chip, self.address = tensor, chip, address

    def buffer_address(self):
        return self.address

    def device(self):
        return self.tensor.ops.chips[self.chip]


class CensusTensor:
    """A device tensor (two chip shards, an address each) or a host payload. `shape` is per chip, as ttnn
    reports a sharded tensor's."""

    def __init__(self, ops, shape, dtype, layout, *, device, label, value=None):
        self.ops, self.shape, self.dtype, self.layout = ops, tuple(int(size) for size in shape), dtype, layout
        self.label, self.value, self.on_device = label, value, device
        self.serial = next(ops.serials)
        self.alive = device
        self.born_trace = ops.capturing if device else None
        self.exposed = None
        self.shards = [CensusShard(self, chip, next(ops.addresses)) for chip in range(MESH_CHIPS)] if device else None
        if device:
            ops.live[self.serial] = self
            ops.allocated_count += 1

    def memory_config(self):
        return self.ops.DRAM_MEMORY_CONFIG

    @property
    def nbytes(self):
        total = 1
        for size in self.shape:
            total *= size
        return total * ITEMSIZE.get(self.dtype, 2)

    def __repr__(self):
        return 'CensusTensor(%s #%d %s)' % (self.label, self.serial, 'x'.join(map(str, self.shape)))


class CensusTrace:
    def __init__(self, label, begun_serial):
        self.label = label
        self.begun_serial, self.end_serial = begun_serial, None
        self.first_access = {}   # serial -> ('read' | 'write', tensor), pre-capture tensors only
        self.writes = {}         # serial -> tensor: every pre-capture tensor the capture wrote
        self.outputs = []        # allocated by the capture and alive when it ended
        self.holes = 0           # allocated by the capture and freed before it ended
        self.released = False

    def inputs(self):
        return [tensor for kind, tensor in self.first_access.values() if kind == 'read']


class CensusOps:
    """The ttnn surface the serving classes use, over CensusTensor."""

    bfloat16, bfloat8_b, float32, uint32, int32, uint16 = 'bf16', 'bf8', 'f32', 'u32', 'i32', 'u16'
    TILE_LAYOUT, ROW_MAJOR_LAYOUT = 'tile', 'row'
    DRAM_MEMORY_CONFIG = 'dram'
    MathFidelity = SimpleNamespace(HiFi4='hifi4')
    BufferType = SimpleNamespace(DRAM='dram', TRACE='trace')
    Topology = SimpleNamespace(Linear='linear')

    def __init__(self, *, capacity=34 * 10 ** 9, trace_capacity=268 * 10 ** 6):
        self.serials, self.addresses = count(1), count(0x100000, 0x1000)
        self.live = {}
        self.allocated_count = 0
        self.capturing = None
        self.traces = []
        self.captures = self.replays = self.syncs = 0
        self.violations = []
        self.log = []
        self.logging = False
        # The parity trail (tracking): every allocation, read, write, free, upload (by the payload's bytes),
        # capture, replay and fence, by serial number, so two runs of the same scenario compare exactly.
        self.trail = []
        self.tracking = False
        self.chips = [SimpleNamespace(chip=chip) for chip in range(MESH_CHIPS)]
        self.capacity, self.trace_capacity = capacity, trace_capacity
        self.trace_bytes = 0
        self.stolen = 0

    # -- bookkeeping ------------------------------------------------------------------------------------
    def allocate(self, shape, dtype='bf16', layout='tile', label='tensor'):
        tensor = CensusTensor(self, shape, dtype, layout, device=True, label=label)
        if self.logging:
            self.log.append(('allocate', label, tensor.shape))
        if self.tracking:
            self.trail.append(('alloc', tensor.serial, label, tensor.shape, dtype, layout))
        return tensor

    def track(self, *event):
        if self.tracking:
            self.trail.append(event)

    def violation(self, kind, tensor, detail):
        self.violations.append((kind, getattr(tensor, 'label', tensor), detail))

    def read(self, tensor, why='read'):
        if not isinstance(tensor, CensusTensor) or not tensor.on_device:
            return
        self.track('read', tensor.serial, why)
        if not tensor.alive:
            self.violation('read-freed', tensor, why)
            return
        trace = self.capturing
        if trace is not None:
            if tensor.born_trace is not trace:
                trace.first_access.setdefault(tensor.serial, ('read', tensor))
            return
        if tensor.exposed is not None:
            self.violation('exposed-read', tensor, '%s after a replay of %s' % (why, tensor.exposed))

    def write(self, tensor, why='write'):
        if not isinstance(tensor, CensusTensor) or not tensor.on_device:
            return
        self.track('write', tensor.serial, why)
        if not tensor.alive:
            self.violation('write-freed', tensor, why)
            return
        trace = self.capturing
        if trace is not None:
            if tensor.born_trace is not trace:
                trace.first_access.setdefault(tensor.serial, ('write', tensor))
                trace.writes[tensor.serial] = tensor
            return
        tensor.exposed = None

    def free(self, tensor):
        if not isinstance(tensor, CensusTensor) or not tensor.on_device or not tensor.alive:
            return
        self.track('free', tensor.serial)
        tensor.alive = False
        del self.live[tensor.serial]
        if self.capturing is not None and tensor.born_trace is self.capturing:
            self.capturing.holes += 1

    # -- ttnn --------------------------------------------------------------------------------------------
    def ReplicateTensorToMesh(self, mesh):
        return ('replicate',)

    def ShardTensorToMesh(self, mesh, dim):
        return ('shard', dim)

    def WormholeComputeKernelConfig(self, **options):
        return SimpleNamespace(kind='kernel', **options)

    def MatmulMultiCoreReuseMultiCast1DProgramConfig(self, **options):
        return SimpleNamespace(kind='program', **options)

    def from_torch(self, value, device=None, dtype=None, layout=None, memory_config=None, mesh_mapper=None):
        shape = list(value.shape)
        if mesh_mapper is not None and mesh_mapper[0] == 'shard':
            shape[mesh_mapper[1]] //= MESH_CHIPS
        dtype = self.bfloat16 if dtype is None else dtype
        layout = self.ROW_MAJOR_LAYOUT if layout is None else layout
        if device is None:
            return CensusTensor(self, shape, dtype, layout, device=False, label='host', value=value)
        return self.allocate(shape, dtype, layout, label='from_torch')

    def get_device_tensors(self, tensor):
        if not isinstance(tensor, CensusTensor) or not tensor.on_device:
            raise ValueError('A device tensor is required')
        return list(tensor.shards)

    def to_torch(self, value):
        tensor = value.tensor if isinstance(value, CensusShard) else value
        if isinstance(tensor, CensusTensor) and not tensor.on_device:
            return tensor.value
        self.read(tensor, 'readback')
        return torch.zeros(tensor.shape, dtype=TORCH_DTYPES.get(tensor.dtype, torch.bfloat16))

    def deallocate(self, tensor):
        self.free(tensor)

    def synchronize_device(self, mesh):
        self.syncs += 1
        self.track('sync')
        if self.logging:
            self.log.append(('synchronize_device',))

    def copy(self, source, destination):
        self.read(source, 'copy source')
        self.write(destination, 'copy')
        if self.logging:
            self.log.append(('copy', source.shape, destination.shape))

    def copy_host_to_device_tensor(self, source, destination):
        value = getattr(source, 'value', None)
        self.track('upload', getattr(destination, 'serial', None), tuple(source.shape),
                   None if value is None else payload_digest(value))
        self.write(destination, 'host upload')
        if self.logging:
            self.log.append(('copy_host_to_device_tensor', tuple(source.shape), destination.shape))

    def full_like(self, value, fill, optional_tensor=None):
        if optional_tensor is not None:
            self.write(optional_tensor, 'full_like')
            if self.logging:
                self.log.append(('full_like', optional_tensor.serial, fill))
            return optional_tensor
        return self.allocate(value.shape, value.dtype, value.layout, 'full_like')

    def zeros_like(self, value):
        return self.allocate(value.shape, value.dtype, value.layout, 'zeros_like')

    def clone(self, value, memory_config=None):
        self.read(value, 'clone')
        return self.allocate(value.shape, value.dtype, value.layout, 'clone')

    def slice(self, value, start, end):
        self.read(value, 'slice')
        if self.logging:
            self.log.append(('slice', value.shape, tuple(start), tuple(end)))
        return self.allocate([stop - begin for begin, stop in zip(start, end)], value.dtype, value.layout, 'slice')

    def pad(self, value, padding, fill):
        self.read(value, 'pad')
        if self.logging:
            self.log.append(('pad', value.shape, tuple(tuple(pair) for pair in padding)))
        return self.allocate([size + before + after for size, (before, after) in zip(value.shape, padding)],
                             value.dtype, value.layout, 'pad')

    def concat(self, values, dim, memory_config=None):
        values = list(values)
        for value in values:
            self.read(value, 'concat')
        dim = dim % len(values[0].shape)
        shape = list(values[0].shape)
        shape[dim] = sum(value.shape[dim] for value in values)
        if self.logging:
            self.log.append(('concat', tuple(value.shape for value in values), dim))
        return self.allocate(shape, values[0].dtype, values[0].layout, 'concat')

    def matmul(self, left, right, dtype=None, compute_kernel_config=None, program_config=None, memory_config=None):
        self.read(left, 'matmul')
        self.read(right, 'matmul weight')
        if self.logging:
            self.log.append(('matmul', left.shape, right.shape, dtype, vars(program_config) if program_config else None))
        return self.allocate(tuple(left.shape[:-1]) + (right.shape[-1],), dtype or left.dtype, left.layout, 'matmul')

    def typecast(self, value, dtype):
        self.read(value, 'typecast')
        if self.logging:
            self.log.append(('typecast', value.shape, dtype))
        return self.allocate(value.shape, dtype, value.layout, 'typecast')

    def rms_norm(self, value, epsilon=None, weight=None, compute_kernel_config=None, memory_config=None):
        self.read(value, 'rms_norm')
        self.read(weight, 'rms_norm weight')
        if self.logging:
            self.log.append(('rms_norm', value.shape))
        return self.allocate(value.shape, value.dtype, value.layout, 'rms_norm')

    def begin_trace_capture(self, mesh, cq_id=0):
        if self.capturing is not None:
            raise RuntimeError('Nested trace capture')
        trace = CensusTrace('trace%d' % len(self.traces), next(self.serials))
        self.traces.append(trace)
        self.capturing = trace
        self.track('begin', trace.label)
        return trace

    def end_trace_capture(self, mesh, trace, cq_id=0):
        if self.capturing is not trace:
            raise RuntimeError('Ending a capture that is not open')
        trace.end_serial = next(self.serials)
        self.track('end', trace.label)
        trace.outputs = [tensor for tensor in self.live.values() if tensor.born_trace is trace]
        self.capturing = None
        self.captures += 1
        self.trace_bytes += 4 * 10 ** 6

    def execute_trace(self, mesh, trace, cq_id=0, blocking=True):
        if self.capturing is not None:
            raise RuntimeError('A replay inside a capture')
        if trace.released:
            self.violation('replay-released', trace.label, 'a released trace was replayed')
            return None
        self.replays += 1
        self.track('replay', trace.label, blocking)
        for tensor in trace.inputs():
            if not tensor.alive:
                self.violation('trace-read-freed', tensor, 'input of %s' % trace.label)
            elif tensor.exposed is not None:
                self.violation('exposed-read', tensor, 'input of %s after a replay of %s' % (trace.label, tensor.exposed))
        for tensor in list(trace.writes.values()) + trace.outputs:
            if tensor.alive:
                tensor.exposed = None
        if trace.holes:
            for tensor in self.live.values():
                if (tensor.serial > trace.end_serial and tensor.born_trace is not trace and tensor.exposed is None):
                    tensor.exposed = trace.label
        return None

    def release_trace(self, mesh, trace):
        self.track('release', trace.label)
        trace.released = True
        self.trace_bytes -= 4 * 10 ** 6

    def get_memory_view(self, device, buffer_type):
        if buffer_type == self.BufferType.TRACE:
            total, used = self.trace_capacity, self.trace_bytes
        else:
            total = self.capacity
            used = sum(tensor.nbytes for tensor in self.live.values()) + self.stolen
        return SimpleNamespace(num_banks=1, total_bytes_allocated_per_bank=used,
                               total_bytes_free_per_bank=total - used,
                               largest_contiguous_bytes_free_per_bank=total - used, total_bytes_per_bank=total)


# -- the world's fakes of what the host cannot run ----------------------------------------------------------

class CensusHelper:
    """gdn_snapshot.ActiveSnapshot(direct=True) over one GDN layer: the native eight-slot state, slot 0
    saved into and restored from snapshots (copy_active), a prefill slot adopted into slot 0."""

    def __init__(self, ops, mesh, layer):
        self.operations = ops
        live = [ops.allocate((8, 1, 32, 128), label='native gdn L%d.%d' % (layer, index))
                for index in range(LIVE_PER_LAYER)]
        self.live = live
        self.gdn = SimpleNamespace(B=8, _stable_state=True, rec_state=live[0], conv_states=live[1:], mesh=mesh)

    def allocate(self):
        return [self.operations.allocate((1, 1, 32, 128), label='gdn snapshot') for _ in self.live]

    def save(self, destination):
        if len(destination) != len(self.live):
            raise ValueError('Incomplete active snapshot')
        for source, target in zip(self.live, destination):
            self.operations.copy(source, target)

    def restore(self, source):
        if len(source) != len(self.live):
            raise ValueError('Incomplete active snapshot')
        for value, target in zip(source, self.live):
            self.operations.copy(value, target)

    def adopt_slot(self, index, *, layer=None):
        for value in self.live:
            self.operations.read(value, 'adopt')
            self.operations.write(value, 'adopt')
        return MESH_CHIPS


class CensusRetained(object):
    """RetainedGDNBlock's host protocol for a CensusFixture: the same flags (from the real class, so a
    reset that misses one fails the attribute comparisons), one record, and commit/replay by those
    flags' rules."""

    def __new__(cls, rows, operations, fixture):
        from gdn_records import RetainedGDNBlock

        class Retained(RetainedGDNBlock):
            def __init__(self):
                super().__init__(rows, operations)
                self.fixture = fixture

            def validate_bindings(self):
                pass

            def bound_mesh(self):
                return self.fixture.model.mesh_device

            def commit(self, prefix, *, dma=False, publication=None, synchronize=False):
                if self.closed or self.selected_prefix is not None or not self.records:
                    raise ValueError('Exactly one decision on a complete live block required')
                self.selected_prefix = prefix
                self.replay_ready = False
                publication(prefix)
                if synchronize:
                    self.operations.synchronize_device(self.bound_mesh())
                    self.replay_ready = True

            def replay(self, operation):
                if self.closed or not self.replay_ready or self.selected_prefix is None:
                    raise ValueError('A successfully synchronized commit is required before replay')
                self.replay_ready = False
                if operation() is not None:
                    raise RuntimeError('Replay operation must return None after enqueueing the bound trace')
                self.operations.synchronize_device(self.bound_mesh())
                self.selected_prefix = None
                self.decisions = {}
                self.replay_epoch += 1

            def close(self):
                if not self.closed:
                    for state, result, checkpoint in self.records:
                        for value in result['owned']:
                            self.operations.deallocate(value)
                    self.records.clear()
                    self.closed = True

        return Retained()


class CensusFixture:
    """The verifier's ModelBatch fixture, as design section 3.2 describes what its trace reads and writes:
    it reads its staged inputs (pooled, restaged before every verify), the native GDN state and the KV
    cache; it writes its entry state, the checkpoints and the target taps before anything reads them;
    its intermediates are freed inside the capture (holes); its logits and retained histories are trace
    outputs."""

    def __init__(self, engine, rows, checkpoints, *, retain, position=None, pack=None, storage=None):
        ops, model = engine.operations, engine.model
        self.operations, self.model, self.rows, self.helpers = ops, model, rows, engine.helpers
        self.checkpoints = checkpoints
        self.owned = []
        width = engine.pages.shape[1]
        integers = dict(dtype=ops.int32, layout=ops.ROW_MAJOR_LAYOUT)
        if storage is None:
            def own(shape, label, **options):
                value = ops.allocate(shape, options.get('dtype', 'bf16'), options.get('layout', 'tile'), label)
                self.owned.append(value)
                return value
            self.tokens = own((rows, 1), 'fixture tokens', dtype=ops.uint32, layout=ops.ROW_MAJOR_LAYOUT)
            self.positions = own((rows,), 'fixture positions', **integers)
            self.pages = own((rows, width), 'fixture pages', **integers)
            self.singleton_pages = own((1, width), 'fixture singleton pages', **integers)
            self.singleton_positions = [own((1,), 'fixture singleton position', **integers) for _ in range(rows)]
            self.cos = own((1, rows, 1, 128), 'fixture cos')
            self.sin = own((1, rows, 1, 128), 'fixture sin')
        else:
            self.tokens, self.positions, self.pages = storage.tokens, storage.positions, storage.pages
            self.singleton_pages, self.cos, self.sin = storage.singleton_pages, storage.cos, storage.sin
            self.singleton_positions = list(storage.singleton_positions)
            for value in (self.tokens, self.positions, self.pages, self.singleton_pages, *self.singleton_positions):
                ops.copy_host_to_device_tensor(SimpleNamespace(shape=value.shape), value)
            for value in (self.cos, self.sin):
                native = ops.allocate(value.shape, label='native rope')
                ops.copy(native, value)
                ops.deallocate(native)
        self.entry = ops.allocate((1, 1, 32, 128), label='fixture entry state')
        self.owned.append(self.entry)
        self.retained = CensusRetained(rows, ops, self) if retain else None
        self.replay_reader = None
        self.readers, self.writers, self.grouped_readers = [], [], []
        self.last_singleton_uploads = None
        self.history = None

    def inputs(self):
        return (self.tokens, self.positions, self.cos, self.sin, self.pages, self.singleton_pages,
                *self.singleton_positions)

    def run(self, sharded_logits=False):
        ops = self.operations
        scratch = [ops.allocate((1, 1, 32, 5120), label='verify intermediate') for _ in range(2)]
        for value in self.inputs():
            ops.read(value, 'verify input')
        ops.copy(self.helpers[0].live[0], self.entry)
        for snapshot in self.checkpoints:
            for value in snapshot:
                ops.copy(scratch[0], value)
        ops.read(self.model.kv_cache, 'kv cache')
        ops.write(self.model.kv_cache, 'kv append')
        ops.read(self.entry, 'entry state')
        features = getattr(self.model, '_census_features', None)
        if features is not None:
            for destination in features.destinations:
                ops.copy(scratch[1], destination)
        for helper in self.helpers:
            for value in helper.live:
                ops.copy(scratch[0], value)
        if self.retained is not None:
            if self.history is None:
                self.history = ops.allocate((self.rows, 1, 32, 128), label='retained history')
                state = SimpleNamespace(entry=[self.entry], gdn=self.helpers[0].gdn)
                self.retained.records.append((state, dict(states=self.history, packed_conv_states=[],
                                                          owned=[self.history]), self.checkpoints[0]))
            else:
                ops.copy(scratch[0], self.history)
        logits = ops.allocate((1, 1, self.rows, 32), label='verify logits')
        ops.copy(scratch[1], logits)
        for value in scratch:
            ops.deallocate(value)
        return logits

    def close(self):
        if self.retained is not None:
            self.retained.close()
        for value in self.owned:
            self.operations.deallocate(value)
        self.owned.clear()


def census_prepare(mesh, layers, prefix):
    """gdn_commit_dma.prepare: the publication reads the entry state and the retained histories and writes the
    native state and the checkpoints."""
    ops = sys.modules['ttnn']

    def publication():
        for layer in layers:
            for value in layer[:2]:
                ops.read(value, 'commit source')
            for value in layer[2:]:
                ops.write(value, 'commit')
    return publication


def census_sample_rows(sampler, logits, rows, operations, native_rows=False):
    operations.read(logits, 'sampler')
    ids = operations.allocate((1, 1, 1, 32), operations.uint32, operations.ROW_MAJOR_LAYOUT, 'verify ids')
    operations.write(ids, 'sampler')
    return ids


class CensusFeatures:
    """prepared_target_features.PreparedTargetFeatures: the fixture's forward copies the taps into the
    pooled destinations while its capture is open."""

    def __init__(self, model, tap_ids, destinations, *, copy, storage_ids):
        self.model, self.destinations = model, tuple(destinations)
        self.closed = False

    @contextmanager
    def capture(self):
        self.model._census_features = self
        try:
            yield self
        finally:
            del self.model._census_features

    def outputs(self):
        return self.destinations

    def close(self):
        self.closed = True


def census_execute_proposal(self, identifiers, history, mask, rope, *, context, owned, retain, stage, audit=True,
                            audit_convolution=False, cached_history=None, pack=None, observe=None, row_exact=False,
                            quad=None):
    """DFlashDevice.execute_proposal's reads and outputs: its restaged inputs, the historical K/V (the cached
    path never reads `history`), the lent weights; its outputs are retained for the capture's life."""
    from dflash_packed_proposal_coordinator import PASS_ROWS, SELECTOR_WIDTH, TOP_CANDIDATES
    from draft_shared_head import candidate_chunks

    ops = self.operations
    for value in (identifiers, mask, *rope['q'], *rope['k']):
        ops.read(value, 'draft input')
    if cached_history is None:
        ops.read(history, 'draft history')
    else:
        for layer in cached_history:
            for value in layer.values():
                ops.read(value, 'draft cached K/V')
    for value in self.borrowed:
        ops.read(value, 'draft weight')
    scratch = ops.allocate((1, 1, 32, 5120), label='draft intermediate')
    retain(scratch)
    projected = retain(ops.allocate((1, 1, max(self.block_rows, PASS_ROWS), SELECTOR_WIDTH), label='draft projected'))
    ops.copy(scratch, projected)
    chunks = []
    for start, stop in candidate_chunks():
        values = retain(ops.allocate((1, 1, self.block_rows, TOP_CANDIDATES), label='draft values'))
        indices = retain(ops.allocate((1, 1, self.block_rows, TOP_CANDIDATES), ops.uint16, ops.TILE_LAYOUT,
                                      'draft indices'))
        ops.copy(scratch, values)
        ops.copy(scratch, indices)
        chunks.append(dict(start=start, stop=stop, values=values, indices=indices))
    return SimpleNamespace(projected=projected, chunks=chunks)


def census_select_proposal(self, outputs, seed, count):
    ops = self.operations
    for tensor in (outputs.projected, *(chunk[name] for chunk in outputs.chunks for name in ('values', 'indices'))):
        ops.to_torch(ops.get_device_tensors(tensor)[0])
    return tuple(0 for _ in range(count))


def census_gather(operations, mesh, collectives, value, *, retain_temporaries=None, observe=None):
    operations.read(value, 'projection gather')
    return operations.allocate(value.shape, operations.float32, value.layout, 'gathered projection')


def census_project_key_value(operations, inputs, query, tables, retain, *, parameters):
    for value in (inputs, query, *tables, parameters['k'], parameters['v']):
        operations.read(value, 'draft K/V projection')
    rows = inputs.shape[2]
    return dict(k=retain(operations.allocate((1, 4, rows, 128), label='projected k')),
                v=retain(operations.allocate((1, 4, rows, 128), label='projected v')))


class CensusWeights:
    """PreparedDraftWeights as lent: every tensor uploaded at attach, lend() recording the borrower."""

    def __init__(self, ops, mesh):
        self.operations, self.mesh = ops, mesh
        self.tensors, self.borrowers = [], []
        self.closed = False

        def upload(shape, label, **options):
            value = ops.allocate(shape, options.get('dtype', 'bf16'), options.get('layout', 'tile'), label)
            self.tensors.append(value)
            return value
        self.layers = [(dict(k=upload((1, 1, 128, 128), 'draft attention k'),
                             v=upload((1, 1, 128, 128), 'draft attention v')),
                        dict(mlp=upload((1, 1, 128, 128), 'draft mlp')), 'weights', 'convolution')
                       for _ in range(5)]
        self.projection = upload((1, 1, 12800, 5120), 'draft feature projection')
        self.feature_norm = upload((1, 1, 160, 32), 'draft feature norm', layout='row')
        self.final_norm = upload((1, 1, 160, 32), 'draft final norm', layout='row')
        self.selector_projection = upload((5120, 256), 'draft selector projection')
        self.predecessors = self.successors = torch.zeros(4, 4, dtype=torch.float64)

    def lend(self, borrower, **geometry):
        if self.closed:
            raise ValueError('Closed shared draft weights cannot be lent')
        self.borrowers.append(borrower)
        return self

    def release(self, borrower):
        self.borrowers = [other for other in self.borrowers if other is not borrower]

    def close(self):
        self.closed = True
        for value in self.tensors:
            self.operations.deallocate(value)
        self.tensors = []
        if self.borrowers:
            raise ValueError('Shared draft weights closed while lent')


class CensusCapture:
    """dflash_prefill_window.PrefillWindowCapture after its prefill: the window's five taps (sharded, 2560
    per chip), the native slot the prefill wrote, and close()."""

    def __init__(self, ops, mesh, position, *, prefill_slot=None):
        from dflash_prefill_window import prefill_window

        rows = prefill_window(position)['rows']
        self.operations = ops
        self.taps = tuple(ops.allocate((1, 1, rows, 2560), label='prefill tap') for _ in range(5))
        self.prefill_slot = prefill_slot
        self.closed = False

    def outputs(self):
        return self.taps

    def close(self):
        if not self.closed:
            for value in self.taps:
                self.operations.deallocate(value)
            self.closed = True


def sampling(max_tokens, ignore_eos=True):
    return SimpleNamespace(temperature=0, n=1, max_tokens=max_tokens, min_tokens=0, ignore_eos=ignore_eos,
                           logprobs=None, prompt_logprobs=None, presence_penalty=0, frequency_penalty=0,
                           repetition_penalty=1, stop=[], stop_token_ids=[])


def census_environment(**extra):
    """The S2 C2-packed-any request path's flags, and nothing else of the QWEN_FAST family."""
    environment = {name: value for name, value in os.environ.items()
                   if not name.startswith('QWEN_FAST_') and not name.startswith('QWEN_SKIP_')}
    environment.update(QWEN_FAST_ANY_REQUEST='1', QWEN_FAST_EXTENT_REPLAY='1', QWEN_FAST_PUBLISH_PREWARM='1')
    environment.update(extra)
    return patch.dict(os.environ, environment, clear=True)


def quiet_log():
    lines = []
    stub = ModuleType('loguru')
    stub.logger = SimpleNamespace(info=lambda template, *values, **named: lines.append(
        template.format(*values, **named)))
    return lines, stub


class World:
    """One attach's worth of the S2 serving stack on CensusOps: the native GDN state and KV cache, the pool
    (four slots, sequential widths 1/2/4, the drafts' output sets), the shared draft weights, a packed-block
    trace over the pool's carries, and every patch the fake needs. Use as a context manager."""

    def __init__(self, *, users=USERS, environment=None, capacity=34 * 10 ** 9, modules=None, tracking=False):
        self.users = users
        self.environment = environment or {}
        # {module name: module}: a module to run in place of today's (the parity tests' base-commit copies).
        self.modules = dict(modules or {})
        self.ops = CensusOps(capacity=capacity)
        self.ops.tracking = tracking
        self.mesh = SimpleNamespace(shape=[1, 2], kind='mesh')
        self.stack = ExitStack()
        self.lines = []

    def __enter__(self):
        ops = self.ops
        stack = self.stack
        self.lines, loguru = quiet_log()
        stack.enter_context(census_environment(**self.environment))
        stack.enter_context(patch.dict(sys.modules, {
            'ttnn': ops, 'loguru': loguru,
            'models.tt_transformers.tt.ccl': SimpleNamespace(TT_CCL=lambda mesh: SimpleNamespace(kind='ccl')),
            **self.modules}))
        engine = sys.modules['verifier_engine']
        self.engine_module = engine
        import dflash_device
        import draft_kv_history
        import dflash_t16_native_scope
        import prepared_target_features
        import publish_prewarm
        import serving_request_factory

        stack.enter_context(patch.object(engine.VerifierEngine, 'fixture',
                                         lambda built, rows, checkpoints, **options:
                                         CensusFixture(built, rows, checkpoints, **options)))
        stack.enter_context(patch.object(engine, 'prepare', census_prepare))
        stack.enter_context(patch.object(engine, 'sample_rows', census_sample_rows))
        stack.enter_context(patch.object(prepared_target_features, 'PreparedTargetFeatures', CensusFeatures))
        stack.enter_context(patch.object(dflash_device.DFlashDevice, 'execute_proposal', census_execute_proposal))
        stack.enter_context(patch.object(dflash_device.DFlashDevice, 'select_proposal', census_select_proposal))
        stack.enter_context(patch.object(dflash_device, 'gather_add_projection', census_gather))
        stack.enter_context(patch.object(draft_kv_history, 'project_key_value', census_project_key_value))
        stack.enter_context(patch.object(dflash_t16_native_scope, 'require_active', lambda: {}))
        stack.enter_context(patch.object(publish_prewarm, '_WARMED', set()))
        # The pool's pairwise independence check is quadratic in its tensors; CensusOps hands out unique
        # addresses by construction, so the answer is known and only the time is saved.
        import serving_buffer_pool

        stack.enter_context(patch.object(serving_buffer_pool, 'overlaps', lambda left, right: False))
        qualification = dict(serving_request_factory._ATTACH_QUALIFICATION)
        serving_request_factory._ATTACH_QUALIFICATION['/census'] = dict(report_sha256='census')
        stack.callback(lambda: (serving_request_factory._ATTACH_QUALIFICATION.clear(),
                                serving_request_factory._ATTACH_QUALIFICATION.update(qualification)))
        engine.note_prefill()
        stack.callback(engine.note_prefill)
        try:
            self.build()
        except BaseException:
            stack.close()
            raise
        return self

    def build(self):
        from dflash_packed_proposal_coordinator import pooled_draft_output_shapes
        from serving_buffer_pool import ServingBufferPool

        ops, mesh = self.ops, self.mesh
        self.model = SimpleNamespace(mesh_device=mesh, num_devices=2, vocab_size=248320, _lmhead_vocab_sharded=True,
            layers=[SimpleNamespace(forward=None) for _ in range(64)],
            args=SimpleNamespace(vocab_size=248320, rope_head_dim=128, rope_theta=1e7, max_seq_len=131072),
            kv_cache=ops.allocate((1, 1, 32, 128), label='vllm kv cache'))
        self.helpers = [CensusHelper(ops, mesh, layer) for layer in range(GDN_LAYERS)]

        def rope(positions):
            rows = len(positions)
            return (ops.allocate((1, rows, 1, 128), label='rope cos'), ops.allocate((1, rows, 1, 128), label='rope sin'))
        self.pool = ServingBufferPool(ops, mesh, users=self.users, helpers=self.helpers, page_width=PAGE_WIDTH,
            bucket_rows=(1, 2, 4), feature_taps=5, rope=rope, packed_shapes=(),
            draft_outputs=pooled_draft_output_shapes(self.users, 16))
        self.stack.callback(self.pool.close)
        self.weights = CensusWeights(ops, mesh)
        self.stack.callback(self.weights.close)
        self.collectives, self.sampler = SimpleNamespace(kind='ccl'), SimpleNamespace(kind='sampler')
        self.fixtures = ('manifests', [('attention', 'convolution', 'mlp')] * 5, {'fc.weight': 'projection'},
                         {'norm.weight': 'selector'})
        # The packed block: captured after the pool and the weights, reading every slot's carry in place.
        self.block_scratch = []
        self.block_trace = ops.begin_trace_capture(mesh)
        scratch = ops.allocate((1, 1, 64, 5120), label='block intermediate')
        for slot in self.pool.slots:
            for snapshot in slot.verifier.carry:
                for value in snapshot:
                    ops.read(value, 'block carry')
        output = ops.allocate((1, 1, 64, 32), label='block logits')
        ops.copy(scratch, output)
        ops.deallocate(scratch)
        ops.end_trace_capture(mesh, self.block_trace)
        self.block_output = output
        self.stack.callback(lambda: ops.deallocate(output))
        self.attached = set(ops.live)

    def __exit__(self, kind, failure, traceback):
        if kind is None:
            self.stack.close()
            return False
        # The body failed: its error is the one to report, not a teardown that found the world half-used.
        try:
            self.stack.close()
        except Exception:
            pass
        return False

    def replay_block(self):
        """A packed round: the block's trace replays, and its output is read back."""
        self.ops.execute_trace(self.mesh, self.block_trace)
        self.ops.to_torch(self.ops.get_device_tensors(self.block_output)[0])

    def state(self, request_id, prompt, max_tokens, *, blocks=None):
        blocks = list(range(1, -(-(prompt + max_tokens) // 64) + 2)) if blocks is None else blocks
        return SimpleNamespace(req_id=request_id, prompt_token_ids=[1] * prompt, output_token_ids=[7],
                               num_computed_tokens=0, sampling_params=sampling(max_tokens), block_ids=(blocks,))

    def pages(self, state):
        blocks = state.block_ids[0]
        pages = torch.full((1, PAGE_WIDTH), blocks[0], dtype=torch.int32)
        pages[0, :len(blocks)] = torch.tensor(blocks, dtype=torch.int32)
        return pages

    def admit(self, request_id, prompt, max_tokens):
        """Today's per-request build: serving_request_factory.from_prefill, as serving_runtime's bridge
        factory calls it beside the four-user block."""
        from serving_request_factory import from_prefill

        state = self.state(request_id, prompt, max_tokens)
        capture = CensusCapture(self.ops, self.mesh, prompt)
        return from_prefill(self.ops, self.model, self.sampler, self.pages(state), self.helpers, state=state,
                            capture=capture, fixtures=self.fixtures, eos_ids=(99,), collectives=self.collectives,
                            buffer_pool=self.pool, shared_weights=self.weights, capture_rows=4)

    def step(self, request):
        request_id = request.session.request_id
        if request.session.finished:
            return False
        request.step(request_id, cancelled=lambda: False)
        return not request.session.finished

    # -- the census ----------------------------------------------------------------------------------------
    def owners(self, *roots):
        """serial -> the first holder path that reaches the live tensor, walking attributes, lists, tuples,
        sets and dicts from the named roots (memory_ledger's walk, without its budget)."""
        found, seen = {}, set()
        stack = [(name, root) for name, root in roots]
        while stack:
            path, value = stack.pop()
            if id(value) in seen or value is None or isinstance(value, (str, bytes, int, float, bool, torch.Tensor)):
                continue
            seen.add(id(value))
            if isinstance(value, CensusTensor):
                if value.on_device and value.alive:
                    found.setdefault(value.serial, path)
                continue
            if isinstance(value, MethodType):
                stack.append((path + '.__self__', value.__self__))
                continue
            if isinstance(value, (CensusOps, CensusShard, ModuleType, type, FunctionType)):
                continue
            if isinstance(value, dict):
                items = list(value.items())
            elif isinstance(value, (list, tuple, set, frozenset)):
                items = list(enumerate(value))
            elif hasattr(value, '__dict__'):
                items = list(vars(value).items())
            else:
                continue
            for key, item in items:
                stack.append(('%s.%s' % (path, key), item))
        return found

    def roots(self, requests=()):
        return [('model', self.model), ('helpers', self.helpers), ('pool', self.pool), ('weights', self.weights),
                ('block', self.block_output), *(('request[%d]' % index, request) for index, request in enumerate(requests))]

    def unowned(self, requests=(), extra=()):
        owners = self.owners(*self.roots(requests), *extra)
        return [tensor for serial, tensor in self.ops.live.items() if serial not in owners]


class TodayChurnCensusTests(unittest.TestCase):
    """Today's per-request path (from_prefill beside the four-user block) under the census: the property
    the pool exists for, checked mechanically against the real classes."""

    def churn(self, world):
        a = world.admit('request-a', 4096, 48)
        b = world.admit('request-b', 300, 40)
        live = [a, b]
        for round_number in range(6):
            world.replay_block()
            for request in live:
                world.step(request)
            self.assertEqual(world.unowned(live), [], 'every live tensor has a holder')
        a.close('request-a')
        c = world.admit('request-c', 2053, 32)
        live = [b, c]
        for round_number in range(6):
            for request in live:
                world.step(request)
            world.replay_block()
            self.assertEqual(world.unowned(live), [])
        for request in live:
            request.close(request.session.request_id)
        return world.ops

    def test_the_per_request_churn_reads_nothing_a_replay_may_have_overwritten(self):
        with World() as world:
            ops = self.churn(world)
            self.assertEqual(ops.violations, [])
            self.assertGreater(ops.replays, 60)
            self.assertGreater(ops.captures, 30, 'three engines, their commits and their proposals were captured')
            # everything a request built is gone with it: the attach's own tensors are all that is left
            self.assertEqual(set(ops.live), world.attached)

    def test_every_live_tensor_of_a_request_is_owned_by_the_request_or_the_attach(self):
        with World() as world:
            request = world.admit('request-a', 4096, 16)
            owners = world.owners(*world.roots([request]))
            paths = {serial: path for serial, path in owners.items() if path.startswith('request[0]')}
            self.assertTrue(paths)
            labels = {world.ops.live[serial].label for serial in paths}
            # the per-request allocations that outlive the build: the engine's traces' outputs and its
            # fixtures' own state, and the proposal capture's inputs, temporaries and outputs
            self.assertTrue({'verify logits', 'verify ids', 'fixture entry state', 'retained history'} <= labels)
            self.assertIn('from_torch', labels)
            request.close('request-a')


class FlagOffParityTests(unittest.TestCase):
    """Flag off, the census trail - every allocation, read, write, free, upload by its bytes, capture, replay
    and fence - of today's churn is the base commit's, with the Stage E modules at the base commit swapped in."""

    def trail(self, modules):
        with World(modules=modules, tracking=True) as world:
            TodayChurnCensusTests().churn(world)
            return world.ops.trail

    def test_the_churn_trail_is_the_base_commits(self):
        base = {name: base_module(name) for name in ('verifier_engine', 'serving_buffer_pool')}
        before = self.trail(base)
        self.assertGreater(len(before), 10000)
        self.assertEqual(self.trail({}), before)

    def test_one_extra_fence_in_a_verify_changes_the_trail(self):
        """The negative control: the trail sees a single added synchronize."""
        base = base_module('verifier_engine')
        before = self.trail({'verifier_engine': base})
        mutated = base_module('verifier_engine')
        original = mutated.VerifierEngine.verify

        def verify(engine, ticket):
            engine.operations.synchronize_device(engine.mesh)
            return original(engine, ticket)
        mutated.VerifierEngine.verify = verify
        self.assertNotEqual(self.trail({'verifier_engine': mutated}), before)


class CensusNegativeControlTests(unittest.TestCase):
    """The census fails where it should."""

    def test_a_buffer_allocated_after_a_trace_and_read_after_its_replay_is_a_violation(self):
        ops, mesh = CensusOps(), object()
        trace = ops.begin_trace_capture(mesh)
        hole = ops.allocate((1, 32), label='hole')
        ops.deallocate(hole)
        ops.end_trace_capture(mesh, trace)
        late = ops.allocate((1, 32), label='late persistent')
        ops.execute_trace(mesh, trace)
        ops.copy(late, ops.allocate((1, 32), label='reader'))
        self.assertEqual([kind for kind, label, detail in ops.violations], ['exposed-read'])
        self.assertEqual(ops.violations[0][1], 'late persistent')

    def test_a_rewrite_before_the_read_clears_it_and_a_trace_without_holes_exposes_nothing(self):
        ops, mesh = CensusOps(), object()
        trace = ops.begin_trace_capture(mesh)
        hole = ops.allocate((1, 32), label='hole')
        ops.deallocate(hole)
        ops.end_trace_capture(mesh, trace)
        late = ops.allocate((1, 32), label='restaged')
        ops.execute_trace(mesh, trace)
        ops.copy_host_to_device_tensor(SimpleNamespace(shape=(1, 32)), late)
        ops.read(late)
        kept = ops.begin_trace_capture(mesh)
        ops.allocate((1, 32), label='kept temporary')
        ops.end_trace_capture(mesh, kept)
        other = ops.allocate((1, 32), label='after the kept trace')
        ops.execute_trace(mesh, kept)
        ops.read(other)
        self.assertEqual(ops.violations, [])

    def test_a_trace_input_freed_or_exposed_before_its_replay_is_a_violation(self):
        ops, mesh = CensusOps(), object()
        first = ops.begin_trace_capture(mesh)
        ops.deallocate(ops.allocate((1, 32), label='hole'))
        ops.end_trace_capture(mesh, first)
        late = ops.allocate((1, 32), label='late input')
        second = ops.begin_trace_capture(mesh)
        ops.read(late)
        ops.end_trace_capture(mesh, second)
        ops.execute_trace(mesh, first)
        ops.execute_trace(mesh, second)
        self.assertEqual(ops.violations[0][:2], ('exposed-read', 'late input'))
        ops.deallocate(late)
        ops.execute_trace(mesh, second)
        self.assertEqual(ops.violations[-1][:2], ('trace-read-freed', 'late input'))

    def test_an_unowned_live_tensor_is_reported(self):
        with World() as world:
            stray = world.ops.allocate((1, 32), label='stray')
            self.assertEqual(world.unowned(), [stray])
            world.ops.deallocate(stray)


class ReplayLedgerTests(unittest.TestCase):
    """R2's ledger (QWEN_FAST_PARKED_AUDIT): the verify notes the replay count after its own replay, and the
    publication refuses if any other trace replayed since."""

    def build(self, world, audit):
        request = world.admit('request-a', 4096, 32)
        ledger = parked.ReplayLedger(world.ops).install() if audit else None
        return request, ledger

    def test_back_to_back_verify_and_commit_pass_and_a_foreign_replay_between_them_fails(self):
        with World() as world:
            request, ledger = self.build(world, True)
            try:
                world.step(request)
                self.assertGreater(ledger.count, 0)
                session, engine = request.session, request.engine
                ticket = session.propose(session.request_id, max_rows=engine.proposal_rows(),
                                         selected=request.runtime.drafter_name)
                predictions, _ = engine.verify(ticket)
                self.assertEqual(engine.replay_mark, ledger.count)
                world.replay_block()
                with self.assertRaisesRegex(AssertionError, 'R2: 1 other trace replay'):
                    session.commit(session.request_id, ticket, predictions, request.runtime.publish)
            finally:
                ledger.uninstall()
                request.close('request-a')
            self.assertIsNone(verifier_engine._replay_count)

    def test_without_the_ledger_nothing_is_noted_or_checked(self):
        with World() as world:
            request, _ = self.build(world, False)
            world.step(request)
            self.assertIsNone(getattr(request.engine, 'replay_mark', None))
            session, engine = request.session, request.engine
            ticket = session.propose(session.request_id, max_rows=engine.proposal_rows(),
                                     selected=request.runtime.drafter_name)
            predictions, _ = engine.verify(ticket)
            world.replay_block()
            session.commit(session.request_id, ticket, predictions, request.runtime.publish)
            request.close('request-a')

    def test_the_ledger_counts_every_replay_and_uninstalls_cleanly(self):
        ops = CensusOps()
        original = ops.execute_trace
        ledger = parked.ReplayLedger(ops).install()
        with self.assertRaisesRegex(ValueError, 'already installed'):
            ledger.install()
        trace = ops.begin_trace_capture(object())
        ops.end_trace_capture(object(), trace)
        ops.execute_trace(object(), trace)
        ops.execute_trace(object(), trace, cq_id=0, blocking=False)
        self.assertEqual(ledger.count, 2)
        self.assertEqual(verifier_engine._replay_count(), 2)
        ledger.uninstall()
        self.assertEqual(ops.execute_trace, original)
        self.assertIsNone(verifier_engine._replay_count)
        with self.assertRaisesRegex(ValueError, 'callable'):
            verifier_engine.set_replay_count(3)


class LifetimeInvariantTests(unittest.TestCase):
    def test_every_lifetime_claim_in_the_drafter_sources_is_listed_with_its_stage_e_status(self):
        listed = {}
        for invariant in parked.LIFETIME_INVARIANTS:
            listed.setdefault(invariant['file'], []).append(invariant['text'])
            self.assertTrue(invariant['status'].startswith(('changed', 'unchanged')), invariant)
            if invariant['status'].startswith('changed'):
                self.assertTrue(invariant['guard'], 'a changed claim names what guards it: %r' % invariant)
        for name in parked.LIFETIME_FILES:
            lines = (HERE / name).read_text(encoding='utf-8').splitlines()
            for text in listed.get(name, ()):
                self.assertTrue(any(text in line for line in lines), '%s no longer says %r' % (name, text))
            for number, line in enumerate(lines, 1):
                if any(phrase in line for phrase in parked.LIFETIME_PHRASES):
                    self.assertTrue(any(text in line for text in listed.get(name, ())),
                                    '%s:%d makes a lifetime claim the list does not cover: %s'
                                    % (name, number, line.strip()))


def base_module(name, relative=None):
    """scripts/ci/<relative> at BASE_COMMIT, loaded as a module named `name` (SkipTest without git)."""
    relative = relative or name + '.py'
    source = git('show', '%s:scripts/ci/%s' % (BASE_COMMIT, relative)).decode('utf-8')
    module = ModuleType(name)
    module.__file__ = str(HERE / relative)
    exec(compile(source, '%s@%s' % (relative, BASE_COMMIT), 'exec'), module.__dict__)
    return module


def git(*arguments):
    try:
        result = subprocess.run(['git', *arguments], capture_output=True, cwd=str(ROOT), timeout=60)
    except (OSError, subprocess.SubprocessError):
        raise unittest.SkipTest('no git')
    if result.returncode != 0:
        raise unittest.SkipTest('no git history for %s' % BASE_COMMIT)
    return result.stdout


class SourcePinTests(unittest.TestCase):
    def test_no_stage_e_edit_is_a_pinned_source_and_each_reaches_the_image(self):
        import c2_overlay

        path = ROOT / 'docker' / 'qwen-c2-overlay.txt'
        if not path.is_file():
            self.skipTest('no checkout')
        sources = {entry.source for entry in c2_overlay.parse_manifest(path.read_text(encoding='utf-8'))}
        for name in parked.STAGE_E_EDITS:
            self.assertNotIn(name, c2_overlay.FORBIDDEN)
            self.assertNotIn(name, parked.NEVER_EDITED)
            self.assertIn('scripts/ci/' + name, sources, '%s must be overlaid to reach the C2 image' % name)

    def test_the_never_edited_sources_are_the_branch_base_bytes(self):
        for name in parked.NEVER_EDITED:
            base = git('show', '%s:scripts/ci/%s' % (BASE_COMMIT, name))
            self.assertEqual((HERE / name).read_bytes().replace(b'\r\n', b'\n'), base.replace(b'\r\n', b'\n'),
                             '%s must stay byte-identical' % name)

    def test_the_kv_slide_adapter_still_matches_draft_kv_history_prepare(self):
        text = (HERE / 'draft_kv_slide_scope.py').read_text(encoding='utf-8')
        self.assertIn('DraftKVHistory', text)


if __name__ == '__main__':
    unittest.main()
