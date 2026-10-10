"""P0 of the upload plan (docs/tp4-fabric-upload.md, section 5): the big zero buffers of an engine start are filled ON THE DEVICE.

WHAT IT REPLACES. At engine start every card takes about 14 GB of zeros over its own PCIe, all built on the host:
  * the paged KV pool (Qwen36Model._allocate_kv_caches_tp: 32 tensors, `as_tensor(torch.zeros(shape), Replicate)`, bf8 under QWEN_SDPA_BF8=1,
    about 11.1 GB per card) - a 650 MB host torch tensor, a host bf8 pack and a PCIe write, 32 times;
  * the replicated tiled bf16 zeros of ServingBufferPool.allocate (the draft history pairs, the draft K/V banks, ...).
With QWEN_FAST_DEVICE_ZEROS=1 each of those is allocated on the device (`ttnn.empty`) and filled in place by the SFPU fill
(`ttnn.full_like(t, 0.0, optional_tensor=t)`): no host tensor, no host pack, no PCIe byte.

THE PINNED RUNTIME (tt-metal 9f9cd4fd590f4b606bd0981a4fe0b6403eb38ec9; paths are into it):
  * `ttnn.zeros(shape, device=...)` does NOT help: ttnn/cpp/ttnn/operations/creation/creation.cpp:52-72 (creation_detail::full_impl: std::fill over a host
    vector, then `host_tensor.to_device`) and :203-211 (bf8/bf4: Tensor::from_vector, then to_device) build a HOST tensor and upload it. Same PCIe bytes, plus a host fill.
  * `ttnn.full_like(device_tensor, 0.0, optional_tensor=device_tensor)` (and zeros_like, creation.cpp:362-370, the same call) does: creation.cpp:222-245
    (full_like_impl): for a TILE bf8/bf16/fp32 tensor on the device whose output dtype equals its own it calls
    `ttnn::fill(tensor, fill_value, memory_config, optional_output_tensor)` (line 244), the on-device unary FILL op
    (ttnn/cpp/ttnn/operations/eltwise/unary/unary.cpp:262, DEFINE_UNARY_OP_SCALAR_VARIANT(fill, FILL)). Given `optional_tensor=` it writes that tensor
    (creation.cpp:230-234 take layout and dtype from it; unary.cpp:35-37 the memory config), so no second allocation exists and the DRAM allocator sees the
    same sequence of sizes as the host path: the addresses are the host path's. THIS CALL IS ALREADY CARD-PROVEN for bf16 TILE tensors: ServingBufferPool.acquire and
    .rezero (serving_buffer_pool.py, `operations.full_like(value, 0.0, optional_tensor=value)`) re-zero the history buffers of every loan on the cards; bf8 is new.
  * `ttnn.empty` allocates without a write (creation.cpp:305-311 -> create_device_tensor, ttnn/core/tensor/tensor_ops.cpp:75-111), with the default topology
    Replicate on every mesh dimension and every coordinate (tensor_ops.cpp:96-107): the topology `ReplicateTensorToMesh` gives the host path. It is already used on
    the mesh by attention_block_fold_tp.py and draft_convolution_fused_tp.py.
  The binding names are `ttnn.empty(shape, dtype=, layout=, device=, memory_config=)` and `ttnn.full_like(tensor, fill_value, ..., optional_tensor=)`
  (creation_nanobind.cpp: bind_empty, bind_full_like).
Only a replicated, interleaved-DRAM, TILE bf8/bf16/fp32 tensor is built this way. Everything else (row-major integers, sharded mappers, sharded memory
configs, tensors under QWEN_FAST_DEVICE_ZEROS_MIN_BYTES per card, whose fill program would cost a JIT build for no saving) keeps its host path, unchanged.

IS THE RESULT THE HOST PATH'S?
  * bf16 / fp32 TILE: the fill writes the value 0.0 into every datum, i.e. the all-zero bit pattern; the host path packs +0.0 to the same pattern.
  * bf8 TILE (1088 bytes a tile: 64 bytes of shared exponents, one per 16-datum row segment, then 1024 bytes of sign-magnitude mantissas): the HOST packer writes
    all zeros for zero input - tt_metal/impl/data_format/blockfloat_common.cpp:26-45 (get_max_exp: the shared exponent is the max exponent of the 16 datums, 0 for
    +-0.0) and :261-265 (convert_u32_to_bfp returns 0 for +-0.0 and denormals: no sign, no mantissa) - so the bytes the stock path leaves are 1088 zero bytes a tile.
    The DEVICE packer packs the SFPU's zeros and its shared exponent is the same max over zero datums, so it should write the same zeros. "Should" is a statement
    about hardware, and nothing on a CPU can prove it: the bytes are written by the hardware packer.
  * WHAT IF THEY DIFFER? Only the 64 exponent bytes of a tile could (the mantissas of a zero datum are zero in any packer). A zero mantissa unpacks to zero whatever
    the shared exponent - the reference unpack, blockfloat_common.cpp:204-207 (Bfp8_b: `if (man == 0) { man = 0; exp = 0; }`), mirrors the unpacker's rule of
    normalising by the leading one and has no value for a tile without a one - so a compute op reading the tile would see the same values. That is the format's
    definition: an argument, not a measurement. But a byte-level consumer sees the difference: the prefix store's spill digests, the region read (qwen_read_blocks)
    and any restore compare RAW pages, and a partially written KV block keeps its zero tail. So this lever does not rest on the argument: it requires the bytes to
    be equal and REFUSES on any difference, an exponent-only one included. QWEN_FAST_DEVICE_ZEROS_AUDIT=1 is that check, on the card, once per run: for the first
    QWEN_FAST_DEVICE_ZEROS_AUDIT_TENSORS tensors of each user (default 2) it reads back a sample of pages (the whole tensor when it is small; else
    QWEN_FAST_DEVICE_ZEROS_AUDIT_BLOCKS blocks spread along dim 0, which include the first and the LAST block, so a fill that stopped short is caught) from the
    device-filled tensor AND from a host-built twin of the same spec, as the packed bytes the device holds (Tensor.host_buffer().get_shard: ttnn/cpp/ttnn-nanobind/
    tensor.cpp HostBuffer, __array__ / __dlpack__), per chip, and compares them. exact=True is logged only when every chip's bytes of every sampled page equal the
    twin's and the chips equal each other. Anything else - a differing byte, a shard it cannot read as bytes, an op that raises - logs a mismatch line and LATCHES the
    lever off for the process: the tensor under audit is freed and rebuilt on the host path, every later tensor takes the host path, and the engine start goes on
    exactly as it does with the flag off.
  * An uninitialised tensor is what a fill that did not run leaves behind: the sample includes the last block for that reason. The audit is the gate for
    a fill that covers less than the tensor; the flag alone (no audit) trusts the fill, as every unaudited lever does once its audit has passed on cards.

OFF: QWEN_FAST_DEVICE_ZEROS unset or 0 -> nothing here runs: the model hook is not installed (arm), the pool takes no branch (filler() is None) and the
op sequence is the served one (test_qwen_device_zeros holds both). The flags are strict ('0'/'1'); anything else refuses the boot (flag_problems).
"""

import hashlib
import inspect
import os
import sys
import textwrap
import time

FLAG = 'QWEN_FAST_DEVICE_ZEROS'
AUDIT_FLAG = 'QWEN_FAST_DEVICE_ZEROS_AUDIT'
MIN_BYTES_FLAG = 'QWEN_FAST_DEVICE_ZEROS_MIN_BYTES'
AUDIT_TENSORS_FLAG = 'QWEN_FAST_DEVICE_ZEROS_AUDIT_TENSORS'
AUDIT_BLOCKS_FLAG = 'QWEN_FAST_DEVICE_ZEROS_AUDIT_BLOCKS'
NAMES = (FLAG, AUDIT_FLAG, MIN_BYTES_FLAG, AUDIT_TENSORS_FLAG, AUDIT_BLOCKS_FLAG)

DEFAULT_MIN_BYTES = 4 * 1024 * 1024
DEFAULT_AUDIT_TENSORS = 2
DEFAULT_AUDIT_BLOCKS = 8
WHOLE_READ_LIMIT = 32 * 1024 * 1024        # a tensor at most this big per card is audited whole
TILE = 32
BF8_TILE_BYTES = 1088
BF8_EXPONENT_BYTES = 64                    # a bf8 tile: 64 bytes of shared exponents, then 1024 of mantissas

# The lines the smoke check reads (c2_smoke_check.upload_p0_problems).
ENGAGED = '[PINDIAG] tp4 device zeros engaged'
REFUSED = '[PINDIAG] tp4 device zeros refused'
AUDIT_LINE = '[PINDIAG] tp4 device zeros audit'
AUDIT_MISMATCH = '[PINDIAG] tp4 device zeros audit mismatch'

MODEL_MODULE = 'models.demos.blackhole.qwen36.tt.model'
MODEL_CLASS = 'Qwen36Model'
KV_METHOD = '_allocate_kv_caches_tp'
# sha256 of textwrap.dedent(inspect.getsource(Qwen36Model._allocate_kv_caches_tp)) in the image's model.py (c977f380..., the fixtures/qwen36_model.py
# bytes qwen_prefix_stage pins; the prefix stage adds methods and leaves this one alone). A model.py whose method differs is not replaced.
KV_SOURCE_SHA256 = '161621b67c19c99503df23582f7d10d907b58ddf05077a9a2d9c222fd71ef5f1'


def _log(template, *values):
    """One loguru INFO line, brace-formatted (dflash_device.pindiag's contract); plain print where loguru is absent (host tests)."""
    try:
        from loguru import logger
    except ImportError:
        print(template.format(*values), flush=True)
        return
    logger.info(template, *values)


# ---------------------------------------------------------------------------------------------------------------------------------------------
# The flags (strict)
# ---------------------------------------------------------------------------------------------------------------------------------------------

def _switch(environ, name):
    value = environ.get(name)
    if value is None or value == '0':
        return False
    if value == '1':
        return True
    raise ValueError('%s must be 0 or 1, got %r' % (name, value))


def _count(environ, name, default, low, high):
    value = environ.get(name)
    if value is None or value == '':
        return default
    try:
        number = int(value)
    except ValueError:
        raise ValueError('%s must be an integer in %d..%d, got %r' % (name, low, high, value))
    if str(number) != value or not low <= number <= high:
        raise ValueError('%s must be an integer in %d..%d, got %r' % (name, low, high, value))
    return number


def enabled(environ=None):
    return _switch(os.environ if environ is None else environ, FLAG)


def audited(environ=None):
    return _switch(os.environ if environ is None else environ, AUDIT_FLAG)


def flag_problems(env):
    """[problem] for a profile env (or a process environment) that names these flags wrongly: a value that is not 0/1, a count out of range, an audit
    without the lever it audits."""
    problems = []
    for name in (FLAG, AUDIT_FLAG):
        try:
            _switch(env, name)
        except ValueError as error:
            problems.append(str(error))
    for name, default, low, high in ((MIN_BYTES_FLAG, DEFAULT_MIN_BYTES, 0, 1 << 40), (AUDIT_TENSORS_FLAG, DEFAULT_AUDIT_TENSORS, 1, 64),
                                     (AUDIT_BLOCKS_FLAG, DEFAULT_AUDIT_BLOCKS, 1, 64)):
        try:
            _count(env, name, default, low, high)
        except ValueError as error:
            problems.append(str(error))
    if not problems:
        if env.get(AUDIT_FLAG) == '1' and env.get(FLAG) != '1':
            problems.append('%s=1 without %s=1: there is no device fill to audit' % (AUDIT_FLAG, FLAG))
        for name in (MIN_BYTES_FLAG, AUDIT_TENSORS_FLAG, AUDIT_BLOCKS_FLAG):
            if name in env and env.get(FLAG) != '1':
                problems.append('%s is set without %s=1: it would do nothing' % (name, FLAG))
    return problems


# ---------------------------------------------------------------------------------------------------------------------------------------------
# Packed bytes of a host tensor (the audit's eyes)
# ---------------------------------------------------------------------------------------------------------------------------------------------

class RawBytesUnavailable(Exception):
    pass


def mesh_coordinates(ttnn, mesh):
    """The mesh's coordinates in row-major order, as ttnn.MeshCoordinate objects."""
    shape = tuple(int(extent) for extent in mesh.shape)
    if len(shape) == 1:
        shape = (1,) + shape
    coordinates = []
    for row in range(shape[0]):
        for column in range(shape[1]):
            try:
                coordinates.append(ttnn.MeshCoordinate(row, column))
            except TypeError:
                coordinates.append(ttnn.MeshCoordinate([row, column]))
    return coordinates


def buffer_bytes(buffer):
    """-> (bytes, how). The packed bytes of one HostBuffer shard: its __array__ / __dlpack__ views (tensor.cpp) or, last, its byte iterator."""
    attempts = []
    try:
        import numpy
    except ImportError:
        numpy = None
    if numpy is not None:
        try:
            return numpy.asarray(buffer).astype(numpy.uint8, copy=False).tobytes(), 'numpy'
        except Exception as error:  # noqa: BLE001 - the next view
            attempts.append('numpy: %s' % type(error).__name__)
    try:
        import torch
        return torch.from_dlpack(buffer).to(torch.uint8).numpy().tobytes(), 'dlpack'
    except Exception as error:  # noqa: BLE001
        attempts.append('dlpack: %s' % type(error).__name__)
    try:
        return bytes(bytearray(buffer)), 'iter'
    except Exception as error:  # noqa: BLE001
        attempts.append('iter: %s' % type(error).__name__)
    raise RawBytesUnavailable('a host buffer shard has no byte view (%s)' % '; '.join(attempts))


def shard_bytes(ttnn, mesh, host_tensor):
    """-> ([bytes per chip, in mesh order], how). `host_tensor` is what ttnn.from_device returned (a mesh tensor on the host)."""
    try:
        distributed = host_tensor.host_buffer()
    except Exception as error:  # noqa: BLE001
        raise RawBytesUnavailable('Tensor.host_buffer() raised %s: %s' % (type(error).__name__, str(error)[:120]))
    shards, how = [], None
    for coordinate in mesh_coordinates(ttnn, mesh):
        try:
            buffer = distributed.get_shard(coordinate)
        except Exception as error:  # noqa: BLE001
            raise RawBytesUnavailable('get_shard(%s) raised %s: %s' % (coordinate, type(error).__name__, str(error)[:120]))
        if buffer is None:
            raise RawBytesUnavailable('the shard at %s is not populated' % (coordinate,))
        data, how = buffer_bytes(buffer)
        shards.append(data)
    return shards, how


def differing_offsets(left, right, limit=64):
    """The first `limit` offsets at which two byte strings differ, and the count of all of them (a length difference counts every extra byte)."""
    common = min(len(left), len(right))
    extra = abs(len(left) - len(right))
    try:
        import numpy
        a, b = numpy.frombuffer(left, dtype=numpy.uint8, count=common), numpy.frombuffer(right, dtype=numpy.uint8, count=common)
        where = numpy.flatnonzero(a != b)
        return [int(offset) for offset in where[:limit]], int(where.size) + extra
    except ImportError:
        offsets = [offset for offset in range(common) if left[offset] != right[offset]]
        return offsets[:limit], len(offsets) + extra


def nonzero_bytes(data):
    try:
        import numpy
        return int(numpy.count_nonzero(numpy.frombuffer(data, dtype=numpy.uint8)))
    except ImportError:
        return sum(1 for byte in data if byte)


# ---------------------------------------------------------------------------------------------------------------------------------------------
# The filler
# ---------------------------------------------------------------------------------------------------------------------------------------------

def tile_count(shape):
    """Tiles in a TILE-layout tensor of this logical shape (the last two dimensions padded to 32)."""
    if len(shape) < 2:
        return 0
    batch = 1
    for extent in shape[:-2]:
        batch *= int(extent)
    return batch * -(-int(shape[-2]) // TILE) * -(-int(shape[-1]) // TILE)


class DeviceZeros(object):
    """Builds replicated TILE zero tensors on the device; audits the first few; latches off on any doubt. One per process (filler())."""

    def __init__(self, ttnn, torch_module, environ, log=_log):
        self.ttnn, self.torch, self.log = ttnn, torch_module, log
        self.min_bytes = _count(environ, MIN_BYTES_FLAG, DEFAULT_MIN_BYTES, 0, 1 << 40)
        self.audit = _switch(environ, AUDIT_FLAG)
        self.audit_tensors = _count(environ, AUDIT_TENSORS_FLAG, DEFAULT_AUDIT_TENSORS, 1, 64)
        self.audit_blocks = _count(environ, AUDIT_BLOCKS_FLAG, DEFAULT_AUDIT_BLOCKS, 1, 64)
        self.latched = None             # why the lever is off for this process, else None
        self.tags = {}                  # tag -> dict(built, host, bytes, seconds, audited)

    # -- bookkeeping ------------------------------------------------------------------------------------------------------------------------
    def _tag(self, tag):
        return self.tags.setdefault(tag, dict(built=0, host=0, bytes=0, seconds=0.0, audited=0, first=True))

    def _dtype_info(self, dtype):
        ttnn = self.ttnn
        if dtype == getattr(ttnn, 'bfloat8_b', object()):
            return 'bf8', BF8_TILE_BYTES, self.torch.bfloat16
        if dtype == getattr(ttnn, 'bfloat16', object()):
            return 'bf16', 2 * TILE * TILE, self.torch.bfloat16
        if dtype == getattr(ttnn, 'float32', object()):
            return 'fp32', 4 * TILE * TILE, self.torch.float32
        return None

    def per_card_bytes(self, shape, dtype):
        info = self._dtype_info(dtype)
        return None if info is None else tile_count(shape) * info[1]

    def eligible(self, shape, dtype, layout, memory_config, replicated):
        """-> None when this tensor may be built on the device, else the reason it keeps the host path."""
        ttnn = self.ttnn
        if self.latched is not None:
            return 'latched off: %s' % self.latched
        if not replicated:
            return 'not replicated'
        if layout != ttnn.TILE_LAYOUT:
            return 'not TILE'
        if self._dtype_info(dtype) is None:
            return 'dtype is not bf8, bf16 or fp32'
        if memory_config != ttnn.DRAM_MEMORY_CONFIG:
            return 'not interleaved DRAM'
        if self.per_card_bytes(shape, dtype) < self.min_bytes:
            return 'under %s' % MIN_BYTES_FLAG
        return None

    # -- the build --------------------------------------------------------------------------------------------------------------------------
    def build(self, shape, dtype, layout, mesh, memory_config, tag, replicated=True):
        """-> a device tensor of zeros built on the device, or None when the caller must run its own host path (ineligible, latched, refused, or its
        audit found a difference). The caller's host code is unchanged either way."""
        reason = self.eligible(shape, dtype, layout, memory_config, replicated)
        state = self._tag(tag)
        if reason is not None:
            state['host'] += 1
            return None
        ttnn = self.ttnn
        started = time.time()
        tensor = None
        try:
            tensor = ttnn.empty(list(shape), dtype=dtype, layout=layout, device=mesh, memory_config=memory_config)
            ttnn.full_like(tensor, 0.0, optional_tensor=tensor)
        except Exception as error:  # noqa: BLE001 - any refusal is the host path's
            self._free(tensor)
            self._latch('the device fill raised %s: %s' % (type(error).__name__, str(error).strip().splitlines()[0][:160] if str(error).strip() else ''),
                        refused=True)
            state['host'] += 1
            return None
        state['seconds'] += time.time() - started
        if self.audit and state['audited'] < self.audit_tensors:
            state['audited'] += 1
            if not self._audit(tag, tensor, shape, dtype, layout, mesh, memory_config, state['audited']):
                self._free(tensor)
                state['host'] += 1
                return None
        state['built'] += 1
        state['bytes'] += self.per_card_bytes(shape, dtype)
        if state['first']:
            state['first'] = False
            info = self._dtype_info(dtype)
            self.log('[PINDIAG] tp4 device zeros engaged tag={} shape={} dtype={} bytes_per_card={} audit={}', tag, tuple(int(extent) for extent in shape),
                     info[0], self.per_card_bytes(shape, dtype), 'on' if self.audit else 'off')
        return tensor

    def _free(self, tensor):
        if tensor is None:
            return
        try:
            self.ttnn.deallocate(tensor)
        except Exception:  # noqa: BLE001 - a tensor that cannot be freed is not worse than the raise that got us here
            pass

    def _latch(self, reason, refused=False):
        if self.latched is None:
            self.latched = reason
            self.log('{} reason={}: every later zero buffer takes the host path', REFUSED if refused else AUDIT_MISMATCH, reason)

    def report(self, tag, synchronize=None):
        """The tag's totals, once its build is over (the caller synchronises the device first when it wants the fill's own time in `seconds`)."""
        state = self._tag(tag)
        if synchronize is not None:
            started = time.time()
            synchronize()
            state['seconds'] += time.time() - started
        self.log('[PINDIAG] tp4 device zeros engaged summary tag={} device={} host={} bytes_per_card={} seconds={:.3f} latched={}', tag, state['built'],
                 state['host'], state['bytes'], state['seconds'], self.latched or 'no')
        return dict(state, latched=self.latched)

    # -- the audit --------------------------------------------------------------------------------------------------------------------------
    def sample_ranges(self, shape, dtype):
        """[(lo, hi) along dim 0] or [None] for the whole tensor: a small tensor is read whole; a large one by `audit_blocks` blocks spread evenly over dim 0
        that always include the first and the LAST block (a fill that stopped short leaves the last block unwritten)."""
        blocks = int(shape[0]) if len(shape) >= 3 else 1
        if blocks <= 2 or self.per_card_bytes(shape, dtype) <= WHOLE_READ_LIMIT:
            return [None]
        count = max(2, min(self.audit_blocks, blocks))
        picks = sorted({(index * (blocks - 1)) // (count - 1) for index in range(count)})
        return [(pick, pick + 1) for pick in picks]

    def _audit(self, tag, tensor, shape, dtype, layout, mesh, memory_config, index):
        """True when the device-filled tensor's sampled pages equal a host-built twin's, byte for byte on every chip."""
        ttnn, torch = self.ttnn, self.torch
        name = self._dtype_info(dtype)[0]
        freed = []
        try:
            ranges = self.sample_ranges(shape, dtype)
            twin_shape = tuple(int(extent) for extent in shape) if ranges == [None] else (1,) + tuple(int(extent) for extent in shape[1:])
            host = torch.zeros(twin_shape, dtype=self._dtype_info(dtype)[2])
            twin = ttnn.from_torch(host, dtype=dtype, layout=layout, device=mesh, memory_config=memory_config,
                                   mesh_mapper=ttnn.ReplicateTensorToMesh(mesh))
            freed.append(twin)
            reference, how = shard_bytes(ttnn, mesh, ttnn.from_device(twin))
            compared, differing, nonzero, exponent_only, first = 0, 0, 0, True, None
            for window in ranges:
                if window is None:
                    part = tensor
                else:
                    ends = [window[1]] + [int(extent) for extent in shape[1:]]
                    part = ttnn.slice(tensor, [window[0]] + [0] * (len(shape) - 1), ends)
                    freed.append(part)
                shards, how = shard_bytes(ttnn, mesh, ttnn.from_device(part))
                if len(shards) != len(reference):
                    raise RawBytesUnavailable('%d shards read back, %d expected' % (len(shards), len(reference)))
                for chip, (got, want) in enumerate(zip(shards, reference)):
                    offsets, count = differing_offsets(got, want)
                    compared += len(got)
                    nonzero += nonzero_bytes(got)
                    if count:
                        differing += count
                        if first is None:
                            first = (chip, window, offsets[0] if offsets else None)
                        if name == 'bf8' and any(offset % BF8_TILE_BYTES >= BF8_EXPONENT_BYTES for offset in offsets):
                            exponent_only = False
                        elif name != 'bf8':
                            exponent_only = False
                if any(shard != shards[0] for shard in shards):
                    differing += 1
                    exponent_only = False
                    first = first or (None, window, None)
        except RawBytesUnavailable as error:
            self._latch('audit of %s tensor %d: %s' % (tag, index, error))
            self._release(freed, tensor)
            return False
        except Exception as error:  # noqa: BLE001 - an audit that cannot run is not a pass
            self._latch('audit of %s tensor %d raised %s: %s' % (tag, index, type(error).__name__, str(error).strip()[:160]))
            self._release(freed, tensor)
            return False
        self._release(freed, tensor)
        if differing or nonzero:
            self.log('{} tag={} tensor={} exact=False dtype={} differing_bytes={} nonzero_bytes={} exponent_only={} first={} sampled_bytes_per_chip={}',
                     AUDIT_MISMATCH, tag, index, name, differing, nonzero, bool(differing and exponent_only), first, compared // max(1, len(reference)))
            self._latch('audit of %s tensor %d: %d bytes differ from the host-built zeros (%d nonzero)' % (tag, index, differing, nonzero))
            return False
        self.log('{} exact=True tag={} tensor={} shape={} dtype={} windows={} sampled_bytes_per_chip={} chips={} raw={}', AUDIT_LINE, tag, index,
                 tuple(int(extent) for extent in shape), name, len(ranges), compared // max(1, len(reference)), len(reference), how)
        return True

    def _release(self, freed, tensor):
        for item in freed:
            if item is not tensor:
                self._free(item)


# ---------------------------------------------------------------------------------------------------------------------------------------------
# The process-wide filler and the two call sites
# ---------------------------------------------------------------------------------------------------------------------------------------------

_FILLERS = {}


def filler(operations, torch_module, environ=None, log=_log):
    """-> the process's DeviceZeros for `operations` (the ttnn module), or None when QWEN_FAST_DEVICE_ZEROS is not 1 (no state is created)."""
    environ = os.environ if environ is None else environ
    if not _switch(environ, FLAG):
        return None
    key = id(operations)
    found = _FILLERS.get(key)
    if found is None or found.ttnn is not operations:
        found = _FILLERS[key] = DeviceZeros(operations, torch_module, environ, log=log)
    return found


def forget():
    """Drop every filler (tests)."""
    _FILLERS.clear()


def source_digest(function):
    return hashlib.sha256(textwrap.dedent(inspect.getsource(function)).encode('utf-8')).hexdigest()


def kv_twin(module, original, zeros_for):
    """A replacement for Qwen36Model._allocate_kv_caches_tp that differs from `original` (pinned by KV_SOURCE_SHA256) in `_mk` alone: each zero tensor is
    built on the device when the filler allows, else by the original expression."""
    ttnn, torch = module.ttnn, module.torch

    def _allocate_kv_caches_tp(self, kv_cache_shape, dtype, batch_size):
        """TP paged KV allocation (B=1). Replicated per device; GDN self-manages state. [device zeros twin: see qwen_device_zeros]"""
        zeros = zeros_for()

        def _mk():
            if zeros is not None:
                built = zeros.build(tuple(kv_cache_shape), dtype, ttnn.TILE_LAYOUT, self.device, ttnn.DRAM_MEMORY_CONFIG, 'kv_cache')
                if built is not None:
                    return built
            return ttnn.as_tensor(
                torch.zeros(kv_cache_shape, dtype=torch.bfloat16),
                device=self.device,
                dtype=dtype,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.device),
            )

        kv_caches = [[_mk(), _mk()] for _ in self._attention_layer_indices]
        if zeros is not None:
            zeros.report('kv_cache', synchronize=lambda: ttnn.synchronize_device(self.device))
        self.set_paged_kv_caches(kv_caches)  # binds via TPAttention.set_paged_kv_cache
        for layer in self.layers:
            if not layer.is_full_attention:
                layer.attention.B = batch_size
                layer.attention.reset_state()
                # Fixed-address GDN state for decode trace compatibility.
                layer.attention._stable_state = True
        # Marker for re-entry assert; TP GDN state lives in module, not external buffers.
        self._deltanet_external_states = []
        return kv_caches

    _allocate_kv_caches_tp._qwen_device_zeros = True
    return _allocate_kv_caches_tp


def install_kv(module, environ=None, log=_log):
    """Replace Qwen36Model._allocate_kv_caches_tp with the twin once the method's source is the pinned one. -> whether the twin is installed."""
    environ = os.environ if environ is None else environ
    if not _switch(environ, FLAG):
        return False
    cls = getattr(module, MODEL_CLASS, None)
    original = getattr(cls, KV_METHOD, None)
    if original is None:
        log('{} reason={}', REFUSED, '%s.%s is not there' % (MODEL_CLASS, KV_METHOD))
        return False
    if getattr(original, '_qwen_device_zeros', False):
        return True
    try:
        digest = source_digest(original)
    except (OSError, TypeError) as error:
        log('{} reason={}', REFUSED, 'the model method has no readable source (%s): the KV pool keeps the host path' % type(error).__name__)
        return False
    if digest != KV_SOURCE_SHA256:
        log('{} reason={}', REFUSED, 'the model method is not the pinned one (sha256 %s): the KV pool keeps the host path' % digest[:16])
        return False
    ttnn, torch = module.ttnn, module.torch
    setattr(cls, KV_METHOD, kv_twin(module, original, lambda: filler(ttnn, torch, environ, log=log)))
    return True


def arm(environ=None, on_import=None, log=_log):
    """In every process of a profile that sets QWEN_FAST_DEVICE_ZEROS=1: install the KV twin when the model module imports. -> whether it was armed.
    `on_import(name, callback)` runs the callback right after that module executes (serving_c2_contract.PostImportHook), at once when it already has."""
    environ = os.environ if environ is None else environ
    if not _switch(environ, FLAG):
        return False
    on_import(MODEL_MODULE, lambda module: install_kv(module, environ, log=log))
    return True
