"""A shape-checking stand-in for the ttnn surface the drafter's publication and proposal capture use, on four chips.

TEST SUPPORT ONLY: no served module imports it and it is not in the image copy lists. The torch fakes the pair's tests use
(test_publication_warm, test_publish_prewarm) replace the matmul, the K/V projection and the shard mapper with stand-ins that
do not care about widths, which is exactly why they could not see a pair literal left on the four-card path. This fake holds
only shapes, dtypes, layouts and per-chip addresses, and every operation refuses what the device would refuse: a slice past
an extent, a copy between different shapes, a matmul whose inner widths differ, a concat whose other dimensions differ, a
gather over the wrong chip count, a tensor used after it was freed, a tensor freed twice.

Nothing here computes a value; the drafter's arithmetic is a card's job (HW-B). What it proves is that every tensor the
served code builds, slices, copies and gathers at QWEN_FAST_TP=4 has the shape the next operation takes."""

from itertools import count
from types import SimpleNamespace

TILE_LAYOUT, ROW_MAJOR_LAYOUT, DRAM = 'tile', 'row', 'dram'


class ShapeError(AssertionError):
    """An operation was handed shapes the device would refuse."""


class Shard:
    def __init__(self, address):
        self.address = address

    def buffer_address(self):
        return self.address


class Tensor:
    def __init__(self, ops, shape, dtype, layout=TILE_LAYOUT, *, on_device=True):
        self.shape, self.dtype, self.layout = tuple(int(size) for size in shape), dtype, layout
        self.ops, self.freed = ops, False
        self.shards = [Shard(next(ops.addresses)) for _ in range(ops.chips)] if on_device else None
        if on_device:
            ops.live.add(id(self))

    def memory_config(self):
        return DRAM

    def __repr__(self):
        return 'Tensor%r' % (self.shape,)


class Mapper:
    def __init__(self, dim=None):
        self.dim = dim


class Namespace(SimpleNamespace):
    pass


class ShapeOps:
    """`chips` devices; `events` lists every operation by name for the tests that count them."""

    bfloat16, float32, uint32 = 'bf16', 'f32', 'u32'
    TILE_LAYOUT, ROW_MAJOR_LAYOUT, DRAM_MEMORY_CONFIG = TILE_LAYOUT, ROW_MAJOR_LAYOUT, DRAM
    MathFidelity = SimpleNamespace(HiFi4='hifi4')
    Topology = SimpleNamespace(Linear='linear', Ring='ring')

    def __init__(self, chips=4):
        self.chips, self.addresses, self.events, self.live = chips, count(0x10000, 0x1000), [], set()
        self.experimental = SimpleNamespace(
            all_gather_async=self.all_gather_async, nlp_create_qkv_heads=self.nlp_create_qkv_heads,
            nlp_concat_heads=self.nlp_concat_heads, rotary_embedding_hf=self.rotary_embedding_hf)

    # -- helpers -------------------------------------------------------------------------------------------
    def new(self, like, shape, dtype=None, layout=None):
        return Tensor(self, shape, like.dtype if dtype is None else dtype, like.layout if layout is None else layout)

    def use(self, *tensors):
        for tensor in tensors:
            if isinstance(tensor, Tensor) and tensor.freed:
                raise ShapeError('input_tensor.is_allocated(): a freed tensor %r was used' % (tensor,))

    def note(self, name, *shapes):
        self.events.append((name,) + tuple(shapes))

    # -- construction and lifetime --------------------------------------------------------------------------
    def ReplicateTensorToMesh(self, mesh):
        return Mapper()

    def ShardTensorToMesh(self, mesh, dim):
        return Mapper(dim)

    def WormholeComputeKernelConfig(self, **fields):
        return ('kernel-config',) + tuple(sorted(fields.items()))

    def MatmulMultiCoreReuseMultiCast1DProgramConfig(self, **fields):
        return SimpleNamespace(**fields)

    def from_torch(self, value, device=None, dtype=None, layout=None, memory_config=None, mesh_mapper=None):
        shape = tuple(value.shape)
        if isinstance(mesh_mapper, Mapper) and mesh_mapper.dim is not None:
            dim = mesh_mapper.dim
            if shape[dim] % self.chips:
                raise ShapeError('a %r host tensor does not shard over %d chips on dim %d' % (shape, self.chips, dim))
            shape = shape[:dim] + (shape[dim] // self.chips,) + shape[dim + 1:]
        tensor = Tensor(self, shape, dtype, layout, on_device=device is not None)
        self.note('from_torch', shape)
        return tensor

    def copy_host_to_device_tensor(self, host, device):
        self.use(device)
        if tuple(host.shape) != tuple(device.shape):
            raise ShapeError('host payload %r into device tensor %r' % (host.shape, device.shape))

    def deallocate(self, tensor):
        if tensor.freed:
            raise ShapeError('tensor %r freed twice' % (tensor,))
        tensor.freed = True
        self.live.discard(id(tensor))

    def get_device_tensors(self, tensor):
        self.use(tensor)
        return list(tensor.shards)

    def synchronize_device(self, mesh):
        self.note('sync')

    def num_program_cache_entries(self):
        return 0

    # -- data movement -------------------------------------------------------------------------------------
    def slice(self, value, start, end, *args, **keywords):
        self.use(value)
        if len(start) != len(value.shape) or len(end) != len(value.shape):
            raise ShapeError('slice rank %r %r on %r' % (start, end, value.shape))
        if any(not 0 <= first <= last <= size for first, last, size in zip(start, end, value.shape)):
            raise ShapeError('slice %r:%r is outside %r' % (tuple(start), tuple(end), value.shape))
        return self.new(value, [last - first for first, last in zip(start, end)])

    def pad(self, value, padding, fill, *args, **keywords):
        self.use(value)
        if len(padding) != len(value.shape) or any(min(pair) < 0 for pair in padding):
            raise ShapeError('pad %r on %r' % (padding, value.shape))
        return self.new(value, [size + first + last for size, (first, last) in zip(value.shape, padding)])

    def concat(self, values, dim=0, **keywords):
        self.use(*values)
        rank = len(values[0].shape)
        dim = dim % rank
        for value in values:
            if len(value.shape) != rank or any(a != b for axis, (a, b) in enumerate(zip(value.shape, values[0].shape))
                                               if axis != dim):
                raise ShapeError('concat on dim %d of %r' % (dim, [tuple(value.shape) for value in values]))
        shape = list(values[0].shape)
        shape[dim] = sum(value.shape[dim] for value in values)
        return self.new(values[0], shape)

    def copy(self, source, destination):
        self.use(source, destination)
        if tuple(source.shape) != tuple(destination.shape):
            raise ShapeError('copy %r into %r' % (source.shape, destination.shape))
        return destination

    def zeros_like(self, value, **keywords):
        self.use(value)
        return self.new(value, value.shape)

    # -- arithmetic ----------------------------------------------------------------------------------------
    def matmul(self, left, right, dtype=None, **keywords):
        self.use(left, right)
        if len(left.shape) < 2 or len(right.shape) < 2 or left.shape[-1] != right.shape[-2]:
            raise ShapeError('matmul %r x %r' % (left.shape, right.shape))
        return self.new(left, left.shape[:-1] + (right.shape[-1],), dtype)

    def typecast(self, value, dtype, **keywords):
        self.use(value)
        return self.new(value, value.shape, dtype)

    def rms_norm(self, value, epsilon=None, weight=None, **keywords):
        self.use(value)
        return self.new(value, value.shape)

    def add(self, left, right, dtype=None, **keywords):
        self.use(left, right)
        if tuple(left.shape) != tuple(right.shape):
            raise ShapeError('add %r + %r' % (left.shape, right.shape))
        return self.new(left, left.shape, dtype)

    # -- collectives and head layout -----------------------------------------------------------------------
    def all_gather_async(self, value, *, dim, **keywords):
        self.use(value)
        shape = list(value.shape)
        shape[dim] *= self.chips
        self.note('all_gather', dim)
        return self.new(value, shape)

    def nlp_create_qkv_heads(self, query, key_value, *, num_heads, num_kv_heads, transpose_k_heads=False, **keywords):
        self.use(query, key_value)
        rows = query.shape[2]
        if (len(query.shape) != 4 or query.shape[3] != num_heads * 128 or key_value.shape != (1, 1, rows, 2 * num_kv_heads * 128)):
            raise ShapeError('qkv heads: query %r, key/value %r for %d/%d heads' % (
                query.shape, key_value.shape, num_heads, num_kv_heads))
        return (self.new(query, (1, num_heads, rows, 128)), self.new(query, (1, num_kv_heads, rows, 128)),
                self.new(query, (1, num_kv_heads, rows, 128)))

    def nlp_concat_heads(self, value, **keywords):
        self.use(value)
        return self.new(value, (1, 1, value.shape[2], value.shape[1] * value.shape[3]))

    def rotary_embedding_hf(self, value, cosine, sine, **keywords):
        self.use(value, cosine, sine)
        if value.shape[2] != cosine.shape[2] or cosine.shape != sine.shape or cosine.shape[3] != value.shape[3]:
            raise ShapeError('rotary %r with tables %r %r' % (value.shape, cosine.shape, sine.shape))
        return self.new(value, value.shape)

    # -- traces ---------------------------------------------------------------------------------------------
    def begin_trace_capture(self, mesh, cq_id=0):
        return object()

    def end_trace_capture(self, mesh, trace, cq_id=0):
        pass

    def release_trace(self, mesh, trace):
        pass

    def execute_trace(self, mesh, trace, cq_id=0, blocking=True):
        self.note('execute_trace')


class Collectives:
    """The semaphore handles the fast path's collectives ask for."""

    def get_and_cycle_ag_semaphore_handles(self):
        return 'ag'

    def get_and_cycle_rs_semaphore_handles(self):
        return 'rs'

    def get_and_cycle_barrier_semaphore_handle(self):
        return 'barrier'
