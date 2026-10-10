"""A fake ttnn + torch for the CPU tests of the sub-device harnesses (test support, never mounted into a container).

It models just enough of tt-metal's sub-device and command-queue semantics, as read from the sources of the image the harnesses run in, for the
harnesses' protocol to be tested on a CPU:

  - the call signatures of every ttnn function the harnesses use are the real bindings' (an unknown keyword is a TypeError);
  - a sub-device manager must be loaded before sub-device 1 exists, an op names its sub-device (matmul: sub_device_id; eltwise: sub_core_grids; the
    gather: subdevice_id + sub_core_grids) and a matmul grid must fit in it;
  - CQOwnerState: a program or trace takes ownership of its sub-device for the queue it is enqueued on and raises while another queue owns it;
    synchronize_device(cq, sub_devices) releases the queue's sub-devices; record_event / wait_for_event transfer ownership (cq_shared_state.cpp);
  - a trace replays only on the queue it was captured on, a manager cannot be cleared while a trace is alive (traces are stored per manager), and an
    op captured in a trace must have run eagerly before (the program binaries must already be on the device: a capture of a never-run program is
    a TT_FATAL, and without the program cache every call builds a fresh program that has never run);
  - a timeline: every queue has a free-at time, a replay occupies its queue for the trace's op count x the op cost, `overlap` says how the queues
    interact ('ideal': fully concurrent; 'serial': one dispatcher for all queues; a float: concurrent traces are stretched by that factor), and
    `corrupt_on_overlap` makes a trace replayed while another queue is busy produce different bytes (to prove the harness notices),
    `drop_replays_on=(1,)` makes a queue's replays do nothing at all (to prove that stale outputs cannot pass), and
    overlap='link' serialises two replays only while their fabric links intersect (the gather's link set is range(offset, offset + num_links), the
    offset being the QWEN_AG_LINK_OFFSET_SD1 of `environ` read when the gather is built, for sub-device 1 only, as the prepared graft does).

Tensors carry a fingerprint instead of data: an op's fingerprint is a hash of its kind and its inputs', so "bitwise equal" means "same computation".
"""

import zlib
from contextlib import contextmanager


# ---------------------------------------------------------------------------------------------------------------------
# torch
# ---------------------------------------------------------------------------------------------------------------------

class FakeScalar(object):
    def __init__(self, positive=True):
        self.positive = positive

    def __gt__(self, other):
        return self.positive


class FakeBool(object):
    def __init__(self, value=True):
        self.value = value

    def all(self):
        return self

    def __bool__(self):
        return bool(self.value)


class FakeArray(object):
    """shape + canonical data; `parts` is kept by cat so that a sharded upload can split it again."""

    def __init__(self, shape, data, dtype='float32', parts=None):
        self.shape, self.data, self.dtype, self.parts = tuple(shape), data, dtype, parts

    def contiguous(self):
        return self

    def clone(self):
        return self

    def float(self):
        return self

    def abs(self):
        return self

    def sum(self):
        return FakeScalar(self.data != ('zeros',))

    def view(self, dtype):
        return FakeArray(self.shape, self.data, dtype, self.parts)

    def to(self, dtype):
        return FakeArray(self.shape, self.data, dtype, self.parts)

    def reshape(self, *shape):
        shape = shape[0] if len(shape) == 1 and isinstance(shape[0], (tuple, list)) else shape
        return FakeArray(shape, self.data, self.dtype, self.parts)

    def __mul__(self, other):
        return self

    __rmul__ = __add__ = __radd__ = __sub__ = __rsub__ = __truediv__ = __mul__

    def __getitem__(self, key):
        stop_slice = key[-1]
        width = self.shape[-1]
        start, stop = stop_slice.start or 0, stop_slice.stop if stop_slice.stop is not None else width
        return FakeArray(self.shape[:-1] + (stop - start,), ('slice', self.data, start, stop), self.dtype)


def canon(data):
    """A gather of consecutive slices of one array is that array."""
    if isinstance(data, tuple) and data and data[0] == 'gathered':
        pieces = data[1]
        if pieces and all(isinstance(p, tuple) and p[0] == 'slice' and p[1] == pieces[0][1] for p in pieces):
            position = 0
            for piece in pieces:
                if piece[2] != position:
                    return data
                position = piece[3]
            return pieces[0][1]
    return data


class FakeGenerator(object):
    def __init__(self):
        self.seed, self.calls = 0, 0

    def manual_seed(self, seed):
        self.seed = seed
        return self


class FakeTorch(object):
    bfloat16, int16 = 'bfloat16', 'int16'

    def Generator(self):  # noqa: N802
        return FakeGenerator()

    def randn(self, *shape, generator=None):
        generator.calls += 1
        return FakeArray(shape, ('rand', generator.seed, generator.calls))

    rand = randn

    def cat(self, arrays, dim=0):
        shape = list(arrays[0].shape)
        shape[dim] = sum(array.shape[dim] for array in arrays)
        return FakeArray(shape, ('cat', tuple(array.data for array in arrays)), arrays[0].dtype, parts=list(arrays))

    def equal(self, first, second):
        return canon(first.data) == canon(second.data)

    def isfinite(self, value):
        return FakeBool(value.data != ('poison',))

    nan = float('nan')

    def full(self, shape, value, dtype=None):
        return FakeArray(shape, ('poison',), dtype or 'float32')


# ---------------------------------------------------------------------------------------------------------------------
# ttnn
# ---------------------------------------------------------------------------------------------------------------------

class SubDeviceId(object):
    def __init__(self, value):
        self.value = int(value)

    def __int__(self):
        return self.value

    def __eq__(self, other):
        return isinstance(other, SubDeviceId) and other.value == self.value

    def __hash__(self):
        return hash(self.value)


class CoreCoord(object):
    def __init__(self, x, y):
        self.x, self.y = x, y


class CoreRange(object):
    def __init__(self, start, end):
        self.rect = (start.x, start.y, end.x, end.y)


class CoreRangeSet(object):
    def __init__(self, ranges):
        self.rects = tuple(sorted(r.rect for r in ranges))


class CoreGrid(object):
    def __init__(self, x, y):
        self.x, self.y = x, y


class SubDevice(object):
    def __init__(self, sets):
        self.rects = sets[0].rects


class FakeTensor(object):
    def __init__(self, shape, parts, fp, owner=None):
        self.shape, self.parts, self.fp, self.owner = tuple(shape), parts, fp, owner
        self.poisoned = False


class FakeEvent(object):
    def __init__(self, ident, cq, time_ns):
        self.ident, self.cq, self.time_ns = ident, cq, time_ns


class FakeSemaphore(object):
    pass


class Namespace(object):
    def __init__(self, **values):
        self.__dict__.update(values)


def fingerprint(*items):
    return zlib.crc32(repr(items).encode())


class FakeMesh(object):
    def __init__(self, ttnn, chips):
        self.ttnn, self.chips = ttnn, chips

    def get_num_devices(self):
        return self.chips

    def enable_program_cache(self):
        self.ttnn.program_cache = True
        self.ttnn.calls.append('program_cache')

    def compute_with_storage_grid_size(self):
        return Namespace(x=self.ttnn.grid[0], y=self.ttnn.grid[1])

    def arch(self):
        return 'blackhole'

    def create_sub_device_manager(self, sub_devices, local_l1_size):
        self.ttnn.managers.append(list(sub_devices))
        return len(self.ttnn.managers) - 1

    def load_sub_device_manager(self, manager_id):
        self.ttnn.loaded = manager_id
        self.ttnn.owner.clear()
        self.ttnn.advance(300000)
        self.ttnn.calls.append('load')

    def clear_loaded_sub_device_manager(self):
        if any(not trace.released for trace in self.ttnn.traces.values()):
            raise RuntimeError('Cannot switch sub device managers while traces are alive (traces are stored per manager)')
        self.ttnn.loaded = None
        self.ttnn.owner.clear()
        self.ttnn.advance(200000)
        self.ttnn.calls.append('clear')

    def remove_sub_device_manager(self, manager_id):
        if manager_id == self.ttnn.loaded:
            raise RuntimeError('Cannot remove active sub device manager')
        self.ttnn.calls.append('remove')


class Trace(object):
    def __init__(self, ident, cq):
        self.ident, self.cq, self.ops, self.sds, self.released, self.corrupt = ident, cq, [], set(), False, False
        self.links = set()
        self.outputs = []


class FakeTTNN(object):
    """`overlap`: 'ideal' | 'serial' | float stretch; `grid`: the compute grid; `chips`: 1 (H1) or 4 (H2)."""
    bfloat16, bfloat8_b = 'bfloat16', 'bfloat8_b'
    TILE_LAYOUT = 'tile'
    DRAM_MEMORY_CONFIG = 'dram'
    OP_COST_NS = 20000

    def __init__(self, overlap='ideal', grid=(11, 10), chips=1, corrupt_on_overlap=False, op_costs=None, environ=None, drop_replays_on=()):
        self.overlap, self.grid, self.chips, self.corrupt_on_overlap = overlap, grid, chips, corrupt_on_overlap
        self.environ = {} if environ is None else environ
        self.drop_replays_on = tuple(drop_replays_on)
        self.busy_links = {0: set(), 1: set()}
        self.gather_env = []
        self.compiled = set()
        self.program_cache = False
        self.op_costs = op_costs or {}
        self.managers, self.loaded = [], None
        self.now = 0
        self.free_at = {0: 0, 1: 0}
        self.owner, self.owner_event = {}, {}
        self.cq_stack = [0]
        self.capture = None
        self.traces, self.events = {}, 0
        self.calls = []
        self.opened = None
        self.SubDeviceId, self.SubDevice, self.CoreCoord, self.CoreRange = SubDeviceId, SubDevice, CoreCoord, CoreRange
        self.CoreRangeSet, self.CoreGrid = CoreRangeSet, CoreGrid
        self.MathFidelity = Namespace(HiFi2='HiFi2')
        self.Topology = Namespace(Ring='Ring', Linear='Linear')
        self.FabricConfig = Namespace(FABRIC_1D='FABRIC_1D', FABRIC_1D_RING='FABRIC_1D_RING')
        self.DispatchCoreType = Namespace(WORKER='worker', ETH='eth')
        self.experimental = Namespace(all_gather_async=self.all_gather_async)

    # ----------------------------------------------------------------- time and ownership

    def clock(self):
        return self.now

    def advance(self, ns):
        self.now += ns

    def op_cost(self, kind):
        return self.op_costs.get(kind, self.OP_COST_NS)

    def other_busy(self, cq):
        return any(free > self.now for other, free in self.free_at.items() if other != cq)

    def submit(self, cq, duration, links=()):
        start = max(self.now, self.free_at[cq])
        busy_elsewhere = self.other_busy(cq)
        if self.overlap == 'serial':
            start = max(start, max(self.free_at.values()))
        elif self.overlap == 'link':
            for other, free in self.free_at.items():
                if other != cq and free > self.now and self.busy_links[other] & set(links):
                    start = max(start, free)
        elif isinstance(self.overlap, float) and busy_elsewhere:
            duration = int(duration * (1.0 + self.overlap))
        self.free_at[cq] = start + duration
        self.busy_links[cq] = set(links)
        return busy_elsewhere

    def take(self, sd, cq):
        current = self.owner.get(sd)
        if current is not None and current != cq:
            raise RuntimeError('TT_FATAL: Sub device id %d currently in use by cq %d. Can\'t enqueue program from cq %d. Finish or wait '
                               'for an event to transfer ownership.' % (sd, current, cq))
        self.owner[sd] = cq
        self.owner_event.pop(sd, None)

    def sub_device_count(self):
        return len(self.managers[self.loaded]) if self.loaded is not None else 1

    def sd_of_rects(self, rects):
        if self.loaded is None:
            return 0
        for index, sub_device in enumerate(self.managers[self.loaded]):
            if tuple(sub_device.rects) == tuple(rects):
                return index
        raise RuntimeError('TT_FATAL: Programs must be executed on a single sub-device (cores %s)' % (rects,))

    def check_sd(self, sd):
        if int(sd) >= self.sub_device_count():
            raise RuntimeError('TT_FATAL: SubDevice index %d out of bounds %d' % (int(sd), self.sub_device_count()))
        return int(sd)

    # ----------------------------------------------------------------- devices

    def open_mesh_device(self, mesh_shape=None, *, l1_small_size=0, trace_region_size=0, num_command_queues=1, dispatch_core_config=None,
                         offset=None, physical_device_ids=(), worker_l1_size=0):
        self.opened = dict(shape=mesh_shape, queues=num_command_queues, trace_region_size=trace_region_size)
        return FakeMesh(self, self.chips)

    def close_mesh_device(self, mesh):
        self.calls.append('close')

    def MeshShape(self, *dims):  # noqa: N802
        return tuple(dims)

    def DispatchCoreConfig(self, *args):  # noqa: N802
        return args

    def set_fabric_config(self, config):
        self.calls.append(('fabric', config))

    def init_device_compute_kernel_config(self, arch, **kwargs):
        return kwargs

    def ReplicateTensorToMesh(self, mesh):  # noqa: N802
        return ('replicate',)

    def ShardTensorToMesh(self, mesh, dim=0):  # noqa: N802
        return ('shard', dim)

    @contextmanager
    def command_queue(self, cq_id):
        self.cq_stack.append(cq_id)
        try:
            yield
        finally:
            self.cq_stack.pop()

    # ----------------------------------------------------------------- tensors

    def from_torch(self, values, *, dtype=None, layout=None, device=None, memory_config=None, mesh_mapper=None):
        if mesh_mapper[0] == 'replicate':
            parts = [('rep', values.data)] * self.chips
        else:
            if values.parts is None or len(values.parts) != self.chips:
                raise RuntimeError('a sharded upload needs one slice per chip')
            parts = [chunk.data for chunk in values.parts]
        return FakeTensor(values.shape, parts, fingerprint('upload', repr(values.data)))

    def get_device_tensors(self, tensor):
        parts = [FakeTensor(tensor.shape, [part], fingerprint(tensor.fp, index), tensor.owner) for index, part in enumerate(tensor.parts)]
        for part in parts:
            part.poisoned = tensor.poisoned
        return parts

    def copy_host_to_device_tensor(self, host, device_tensor, cq_id=None):
        device_tensor.poisoned = True
        self.advance(1000)

    def to_torch(self, tensor, **kwargs):
        if tensor.poisoned:
            return FakeArray(tensor.shape, ('poison',), 'bfloat16')
        corrupt = tensor.owner is not None and self.traces[tensor.owner].corrupt
        part = tensor.parts[0]
        data = ('corrupt', tensor.fp) if corrupt else (part if isinstance(part, tuple) and part[0] == 'gathered' else ('dev', tensor.fp, part))
        return FakeArray(tensor.shape, data, 'bfloat16')

    # ----------------------------------------------------------------- ops

    def current_cq(self):
        return self.cq_stack[-1]

    def run_op(self, kind, shape, inputs, sd, parts_fn=None, links=()):
        cq = self.current_cq()
        fp = fingerprint(kind, tuple(t.fp for t in inputs))
        parts = parts_fn(inputs) if parts_fn else [('op', fp)] * self.chips
        program = (kind, tuple(shape), sd, tuple(links))
        if self.capture is not None:
            trace = self.capture
            if cq != trace.cq:
                raise RuntimeError('an op enqueued on queue %d while a trace is captured on queue %d' % (cq, trace.cq))
            if not self.program_cache or program not in self.compiled:
                raise RuntimeError('TT_FATAL: Expected program binaries to be written to the MeshDevice (%s was never run eagerly before the capture)'
                                   % (program,))
            trace.ops.append(kind)
            trace.sds.add(sd)
            trace.links.update(links)
            tensor = FakeTensor(shape, parts, fp, owner=trace.ident)
            trace.outputs.append(tensor)
            return tensor
        self.take(sd, cq)
        self.compiled.add(program)
        self.submit(cq, self.op_cost(kind), links)
        self.advance(1000)
        return FakeTensor(shape, parts, fp)

    def matmul(self, a, b, *, memory_config=None, dtype=None, compute_kernel_config=None, core_grid=None, sub_device_id=None, program_config=None):
        sd = self.check_sd(sub_device_id if sub_device_id is not None else SubDeviceId(0))
        if self.loaded is not None and core_grid is not None:
            x0, y0, x1, y1 = self.managers[self.loaded][sd].rects[0]
            if core_grid.x > x1 - x0 + 1 or core_grid.y > y1 - y0 + 1:
                raise RuntimeError('TT_FATAL: matmul grid %dx%d extends past the sub-device %s' % (core_grid.x, core_grid.y, (x0, y0, x1, y1)))
        return self.run_op('mm', a.shape[:-1] + (b.shape[-1],), [a, b], sd)

    def eltwise(self, kind, inputs, sub_core_grids):
        if sub_core_grids is None:
            raise RuntimeError('TT_FATAL: Programs must be executed on a single sub-device (no sub_core_grids given)')
        return self.run_op(kind, inputs[0].shape, inputs, self.sd_of_rects(sub_core_grids.rects))

    def silu(self, a, *, memory_config=None, output_tensor=None, sub_core_grids=None):
        return self.eltwise('silu', [a], sub_core_grids)

    def multiply(self, a, b, *, memory_config=None, sub_core_grids=None, sub_device_id=None):
        return self.eltwise('mul', [a, b], sub_core_grids)

    def add(self, a, b, *, memory_config=None, sub_core_grids=None, sub_device_id=None):
        return self.eltwise('add', [a, b], sub_core_grids)

    def create_global_semaphore(self, mesh, cores, initial_value, buffer_type=None):
        return FakeSemaphore()

    def all_gather_async(self, input_tensor, *, persistent_output_buffer=None, dim, multi_device_global_semaphore, num_links=None,
                         memory_config=None, topology='Ring', subdevice_id=None, cluster_axis=None, use_optimal_ccl_for_llama=False,
                         use_broadcast=False, barrier_semaphore=None, chunks_per_sync=None, num_workers_per_link=None,
                         num_buffers_per_channel=None, sub_core_grids=None):
        if not isinstance(multi_device_global_semaphore, list) or barrier_semaphore is None:
            raise RuntimeError('the gather needs its semaphore list and a barrier semaphore')
        sd = self.sd_of_rects(sub_core_grids.rects) if sub_core_grids is not None else 0
        if subdevice_id is not None and int(subdevice_id) != sd:
            raise RuntimeError('TT_FATAL: subdevice_id and sub_core_grids name different sub-devices')
        shape = input_tensor.shape[:-1] + (input_tensor.shape[-1] * self.chips,)

        def gathered(inputs):
            pieces = tuple(part for part in inputs[0].parts)
            return [('gathered', pieces)] * self.chips

        offset = int(self.environ.get('QWEN_AG_LINK_OFFSET_SD1', 0) or 0) if sd == 1 else 0
        self.gather_env.append((sd, offset))
        links = tuple(range(offset, offset + (num_links or 1)))
        return self.run_op('ag%d' % (num_links or 1), shape, [input_tensor], sd, parts_fn=gathered, links=links)

    # ----------------------------------------------------------------- queues, events, traces

    def synchronize_device(self, mesh, cq_id=None, sub_device_ids=()):
        queues = [cq_id] if cq_id is not None else sorted(self.free_at)
        sds = [int(s) for s in sub_device_ids] if sub_device_ids else list(range(self.sub_device_count()))
        for queue in queues:
            self.now = max(self.now, self.free_at[queue])
            for sd in sds:
                if self.owner.get(sd) == queue:
                    del self.owner[sd]
                    self.owner_event.pop(sd, None)
        self.advance(2000)

    def record_event(self, mesh, cq_id=None, sub_device_ids=(), device_range=None):
        self.events += 1
        event = FakeEvent(self.events, cq_id or 0, self.free_at[cq_id or 0])
        for sd in ([int(s) for s in sub_device_ids] or list(range(self.sub_device_count()))):
            if self.owner.get(sd) == event.cq and sd not in self.owner_event:
                self.owner_event[sd] = event.ident
        self.advance(1500)
        return event

    def wait_for_event(self, cq_id, mesh_event):
        for sd, event_id in list(self.owner_event.items()):
            if self.owner.get(sd) == mesh_event.cq and mesh_event.cq != cq_id and event_id <= mesh_event.ident:
                del self.owner[sd]
                del self.owner_event[sd]
        self.free_at[cq_id] = max(self.free_at[cq_id], mesh_event.time_ns)
        self.advance(1500)

    def begin_trace_capture(self, mesh, *, cq_id=None):
        if self.capture is not None:
            raise RuntimeError('a trace capture is already open')
        trace = Trace(len(self.traces) + 1, 0 if cq_id is None else cq_id)
        self.traces[trace.ident] = trace
        self.capture = trace
        return trace.ident

    def end_trace_capture(self, mesh, trace_id, *, cq_id=None):
        if self.capture is None or self.capture.ident != trace_id:
            raise RuntimeError('no such open capture')
        self.capture = None

    def execute_trace(self, mesh, trace_id, *, cq_id=None, blocking=True):
        trace = self.traces[trace_id]
        cq = 0 if cq_id is None else cq_id
        if trace.released:
            raise RuntimeError('trace %d was released' % trace_id)
        if cq != trace.cq:
            raise RuntimeError('trace %d was captured on queue %d and cannot replay on queue %d' % (trace_id, trace.cq, cq))
        if cq in self.drop_replays_on:        # a queue that silently runs nothing: its outputs keep whatever they held
            self.advance(3000)
            return
        for sd in sorted(trace.sds):
            self.take(sd, cq)
        for tensor in trace.outputs:
            tensor.poisoned = False
        busy = self.submit(cq, sum(self.op_cost(kind) for kind in trace.ops), tuple(trace.links))
        trace.corrupt = bool(busy and self.corrupt_on_overlap)
        self.advance(3000)
        if blocking:
            self.now = max(self.now, self.free_at[cq])

    def release_trace(self, mesh, trace_id):
        self.traces[trace_id].released = True
        self.calls.append('release')

    def deallocate(self, tensor, force=True):
        pass
