"""The host KV tier's host side of an upload: threaded, pipelined staging copies and parallel digest checks (docs/tp4-upload1.md).

WHY. A returning session's blocks come back from host RAM in two steps per device call: the host copies a chunk of blocks out of the tier's slab into a
transfer buffer (single-threaded numpy, ~10 GB/s on a busy host), then ttnn.qwen_write_blocks_raw writes the buffer to every chip (22.8 GB/s in Q1). They run
one after the other on one thread, so a 254k restore (8.8 GB) costs the sum of the two plus the sampled digest checks (sha256, 1.5 to 2.5 GB/s, one block in
sixteen) instead of the slowest of them. The spill is the same in reverse: a device read, then a copy out into the slab. Both raw ops release the GIL
(qwen_kv_read.cpp nb::gil_scoped_release) and so do numpy's large copies and hashlib's sha256, so threads help.

WHAT (default off; each switch strict; every one needs the host tier, QWEN_PREFIX_HOST_TIER_GIB > 0, else the registry refuses to start):
  QWEN_PREFIX_TIER_IO_THREADS=<n>        1..16 staging threads. ThreadedTierIO wraps the model graft's _QwenKvTierIO (registry.attach_tier_io does it) and
                                         answers read_blocks / write_blocks with the same arguments, the same payload layout, the same errors and the same bytes,
                                         the device calls in the same order and strictly one at a time (never two in flight), but with the staging copy of the
                                         NEXT device call running on the workers while the device call of the CURRENT one runs. 0 or unset: the wrapper does not exist.
  QWEN_PREFIX_TIER_IO_CHUNK_BLOCKS=<n>   16..4096 blocks per device call (default: the adapter's CHUNK_BLOCKS, 512 = 8.9 MB per chip). A gate's sweep knob for the
                                         host cap on a four-card upload (docs/tp4-fabric-upload.md 6.5: calls over 32 MiB per card fall off a cliff). Inert without the threads.
  QWEN_PREFIX_TIER_VERIFY_THREADS=<n>    1..16 threads for the digest checks of a restore's records (TierHooks.check_digests). The sha256 of a record is a pure read of its
                                         payload against the digest taken at spill, so the verdicts, the counters and the record dropped on a failure are the sequential
                                         loop's. 0 or unset: the sequential loop, untouched.

NEVER WORSE THAN THE STOCK PATH. A failure inside the threaded path (a worker's error, a device refusal) is not the caller's first sight of it: the wrapper latches itself off for the process
(one line, `[PINDIAG] prefix: host tier upload latched off: <why>`, and the counter tier_io_latched), repeats that one call through the adapter's own serial code and goes on with the adapter for
every later call. A device that really refuses fails the retry too, and the caller sees that error exactly as without the lever. The adapter's argument refusals (a wrong-sized payload, a count
mismatch) are raised before any thread or device is touched and never latch.

NOT IN IT. No device program, no new ttnn call, no change to a payload byte: the wrapper calls ttnn.qwen_read_blocks_raw / qwen_write_blocks_raw exactly as the
adapter does, with buffers of the same shape ([chips, blocks, slice_bytes], uint8, C-contiguous). The model graft's pinned source is untouched.

Stdlib at import (numpy and concurrent.futures are imported where the bytes move)."""

import threading
import time

ENV_IO_THREADS = 'QWEN_PREFIX_TIER_IO_THREADS'
ENV_IO_CHUNK = 'QWEN_PREFIX_TIER_IO_CHUNK_BLOCKS'
ENV_VERIFY_THREADS = 'QWEN_PREFIX_TIER_VERIFY_THREADS'
NAMES = (ENV_IO_THREADS, ENV_IO_CHUNK, ENV_VERIFY_THREADS)
MAX_THREADS = 16
MIN_CHUNK_BLOCKS = 16
MAX_CHUNK_BLOCKS = 4096
# Transfer buffers in flight: one the device call reads or writes, the others being staged or released.
BUFFERS = 3

# The registry counters this module adds (only when a flag is on; registry.UPLOAD_STAT_NAMES is pinned equal by test_qwen_tier_upload).
STAT_NAMES = ('tier_io_write_calls', 'tier_io_write_blocks', 'tier_io_write_ms', 'tier_io_write_device_ms', 'tier_io_write_stage_ms',
              'tier_io_read_calls', 'tier_io_read_blocks', 'tier_io_read_ms', 'tier_io_read_device_ms', 'tier_io_read_stage_ms',
              'tier_io_latched', 'tier_verify_batches', 'tier_verify_blocks', 'tier_verify_ms')

MARKER = '[PINDIAG] prefix: host tier upload'
ON_LINE = MARKER + ' on io_threads={} io_chunk_blocks={} verify_threads={}'
LATCHED_LINE = MARKER + ' latched off: {}'      # (the form the log line takes once the registry's log has put its head on)


# ---------------------------------------------------------------------------------------------------------------------------------------------
# The flags (strict)
# ---------------------------------------------------------------------------------------------------------------------------------------------

def _whole(environ, name, low, high, zero_ok):
    raw = environ.get(name)
    if raw is None or raw == '':
        return 0
    if not (raw.isascii() and raw.isdigit()):
        raise ValueError('%s=%r is not a whole number (digits only)' % (name, raw))
    value = int(raw)
    if value == 0 and zero_ok:
        return 0
    if not low <= value <= high:
        raise ValueError('%s=%d is outside %d..%d%s' % (name, value, low, high, ' (0 turns it off)' if zero_ok else ''))
    return value


class UploadPlan(object):
    """io_threads, io_chunk (blocks per device call; 0: the adapter's own) and verify_threads of the tier's host side; at least one of the threads is non-zero."""

    __slots__ = ('io_threads', 'io_chunk', 'verify_threads')

    def __init__(self, io_threads, io_chunk, verify_threads):
        self.io_threads = io_threads
        self.io_chunk = io_chunk
        self.verify_threads = verify_threads


def upload_plan(environ, tier_config):
    """The plan from the environment, None when no switch is on (the registry then builds nothing and adds no counter). Strict: digits only, in range, a chunk
    size without the IO threads (it would be inert), and any of them without the host tier (there is nothing to move)."""
    io_threads = _whole(environ, ENV_IO_THREADS, 1, MAX_THREADS, True)
    verify_threads = _whole(environ, ENV_VERIFY_THREADS, 1, MAX_THREADS, True)
    chunk = _whole(environ, ENV_IO_CHUNK, MIN_CHUNK_BLOCKS, MAX_CHUNK_BLOCKS, False)
    if chunk and not io_threads:
        raise ValueError('%s is set and %s is not: the call size would be inert' % (ENV_IO_CHUNK, ENV_IO_THREADS))
    if not (io_threads or verify_threads):
        return None
    if tier_config is None:
        raise ValueError('%s need the host KV tier (QWEN_PREFIX_HOST_TIER_GIB > 0): without it there is nothing to move' % ' / '.join(
            name for name, value in ((ENV_IO_THREADS, io_threads), (ENV_VERIFY_THREADS, verify_threads)) if value))
    return UploadPlan(io_threads, chunk, verify_threads)


def flag_problems(env):
    """Every way a profile's env names these switches wrongly, [] when none (the contract's check; the registry's own refusal is upload_plan)."""
    problems = []
    for name in NAMES:
        if name in env:
            try:
                _whole(dict(env), name, MIN_CHUNK_BLOCKS if name == ENV_IO_CHUNK else 1, MAX_CHUNK_BLOCKS if name == ENV_IO_CHUNK else MAX_THREADS,
                       name != ENV_IO_CHUNK)
            except ValueError as error:
                problems.append(str(error))
    if env.get(ENV_IO_CHUNK) not in (None, '') and env.get(ENV_IO_THREADS) in (None, '', '0'):
        problems.append('%s is set and %s is not: the call size would be inert' % (ENV_IO_CHUNK, ENV_IO_THREADS))
    return problems


# ---------------------------------------------------------------------------------------------------------------------------------------------
# The threaded, pipelined tier IO
# ---------------------------------------------------------------------------------------------------------------------------------------------

def device_ops(io):
    """(read_raw, write_raw): the ttnn raw block ops the adapter calls, taken from the namespace the adapter's own methods run in (the model module's `ttnn`: the real
    module in the image, the fake in a test), else from `import ttnn`. Raises when there are none."""
    namespace = getattr(getattr(type(io), 'write_blocks', None), '__globals__', None) or {}
    module = namespace.get('ttnn')
    if module is None:
        import ttnn as module
    return module.qwen_read_blocks_raw, module.qwen_write_blocks_raw


def split(count, parts):
    """[(low, high)] covering range(count) in at most `parts` nearly equal consecutive pieces (no empty piece)."""
    parts = max(1, min(int(parts), int(count)))
    step, extra = divmod(count, parts)
    out, low = [], 0
    for index in range(parts):
        high = low + step + (1 if index < extra else 0)
        if high > low:
            out.append((low, high))
        low = high
    return out


class ThreadedTierIO(object):
    """The adapter's read_blocks / write_blocks with the staging copies on `threads` workers, overlapped with the device calls. Everything else (block_bytes,
    fingerprint, caches, chips, slice_bytes, ...) is the adapter's, by delegation. The device calls are made by the CALLING thread, one at a time, in the adapter's order
    (chunk by chunk, cache by cache); a worker only copies host bytes between a payload and a transfer buffer that no device call is using."""

    def __init__(self, io, threads, chunk_blocks=0, stats=None, ops=None, clock=time.perf_counter, log=None):
        if int(threads) < 1:
            raise ValueError('ThreadedTierIO needs at least one thread')
        self.io = io
        self.threads = int(threads)
        self.chunk_blocks = int(chunk_blocks or 0)
        self.stats = stats if stats is not None else {}
        self.clock = clock
        self.log = log if log is not None else (lambda message, *values: None)
        self.latched = None            # why the threaded path stopped, once it has
        # The ops are the adapter's own (always resolvable in the image, where the adapter has just asked ttnn for them); a stand-in IO without them fails at its first call.
        try:
            self._ops = ops if ops is not None else device_ops(io)
        except (ImportError, AttributeError):
            self._ops = None
        self._pool = None
        self._buffers = []
        self._lock = threading.Lock()

    def __getattr__(self, name):
        # Only called for what this class does not define: the adapter's geometry and fingerprint.
        if name in ('io', '_ops', 'latched'):
            raise AttributeError(name)
        return getattr(self.io, name)

    def _device(self):
        """(read_raw, write_raw), resolved once."""
        if self._ops is None:
            self._ops = device_ops(self.io)
        return self._ops

    # -- plumbing --------------------------------------------------------------------------------
    def _chunk(self):
        return self.chunk_blocks or int(self.io.CHUNK_BLOCKS)

    def _executor(self):
        if self._pool is None:
            from concurrent.futures import ThreadPoolExecutor

            self._pool = ThreadPoolExecutor(max_workers=self.threads, thread_name_prefix='qwen-tier-io')
        return self._pool

    def _buffer(self, number, count):
        """A C-contiguous uint8 [chips, count, slice_bytes] view of transfer buffer `number`, kept between calls (no page faults after the first)."""
        import numpy as np

        need = int(self.io.chips) * count * int(self.io.slice_bytes)
        while len(self._buffers) <= number:
            self._buffers.append(None)
        if self._buffers[number] is None or self._buffers[number].size < need:
            self._buffers[number] = np.empty(need, dtype=np.uint8)
        return self._buffers[number][:need].reshape(int(self.io.chips), count, int(self.io.slice_bytes))

    def _count(self, name, amount):
        if name in self.stats:
            self.stats[name] += amount

    def close(self):
        pool, self._pool = self._pool, None
        if pool is not None:
            pool.shutdown(wait=True)

    def _drain(self, pending):
        """Wait for every submitted task, whatever happened (no worker may still be writing into a buffer or a payload when the caller sees an error), and re-raise the first error."""
        error = None
        for future in pending:
            try:
                future.result()
            except BaseException as caught:  # noqa: BLE001  (re-raised below)
                error = error or caught
        return error

    def _latch(self, call, error):
        self.latched = '%s: %s: %s' % (call, type(error).__name__, str(error)[:160])
        self._count('tier_io_latched', 1)
        self.log('host tier upload latched off: %s', self.latched)       # (the registry's log adds the '[PINDIAG] prefix: ' head)
        pool, self._pool = self._pool, None
        if pool is not None:
            pool.shutdown(wait=True)

    # -- the two calls ---------------------------------------------------------------------------
    def read_blocks(self, block_ids, into=None):
        """The adapter's read_blocks (QwenKvTierIO.read_blocks): one payload per block id, in order, bit for bit what the adapter returns. The adapter's argument checks come
        first and never latch; a failure after them latches the threaded path off and repeats the call through the adapter."""
        ids = [int(block) for block in block_ids]
        if into is not None and (len(into) != len(ids) or any(len(view) != self.io.block_bytes for view in into)):
            raise ValueError(f'{len(into)} buffers for {len(ids)} blocks of {self.io.block_bytes} bytes')
        if self.latched is not None:
            return self.io.read_blocks(ids, into=into)
        try:
            return self._read(ids, into)
        except Exception as error:
            self._latch('read_blocks', error)
            return self.io.read_blocks(ids, into=into)

    def write_blocks(self, block_ids, payloads):
        """The adapter's write_blocks: payload i into block id i of every cache on every chip, returning when the device has it (same refusals, same latch and retry as read_blocks)."""
        ids = [int(block) for block in block_ids]
        if len(payloads) != len(ids):
            raise ValueError(f'{len(payloads)} payloads for {len(ids)} blocks')
        for payload in payloads:
            if len(payload) != self.io.block_bytes:
                raise ValueError(f'a tier payload of {len(payload)} bytes, the KV block is {self.io.block_bytes}')
        if self.latched is not None:
            return self.io.write_blocks(ids, payloads)
        try:
            return self._write(ids, payloads)
        except Exception as error:
            self._latch('write_blocks', error)
            return self.io.write_blocks(ids, payloads)

    def _read(self, ids, into):
        import numpy as np

        began = self.clock()
        io = self.io
        if into is None:
            payloads = [np.empty(io.block_bytes, dtype=np.uint8) for _ in ids]
        else:
            payloads = [np.frombuffer(view, dtype=np.uint8) for view in into]
        shaped = [payload.reshape(len(io.caches), io.chips, io.slice_bytes) for payload in payloads]
        step = self._chunk()
        jobs = [(first, ids[first:first + step], index) for first in range(0, len(ids), step) for index in range(len(io.caches))]
        pool = self._executor()
        stage_ms = [0.0]
        device_ms = 0.0
        waiting = {}
        everything = []

        def copy_out(part, first, low, high, index):
            mark = self.clock()
            for offset in range(low, high):
                shaped[first + offset][index] = part[:, offset, :]
            with self._lock:
                stage_ms[0] += (self.clock() - mark) * 1000.0

        try:
            for number, (first, chunk, index) in enumerate(jobs):
                slot = number % BUFFERS
                if slot in waiting:                    # this buffer's previous contents are still being copied out
                    error = self._drain(waiting.pop(slot))
                    if error is not None:
                        raise error
                part = self._buffer(slot, len(chunk))
                mark = self.clock()
                self._device()[0](io.caches[index], part, chunk)
                device_ms += (self.clock() - mark) * 1000.0
                futures = [pool.submit(copy_out, part, first, low, high, index) for low, high in split(len(chunk), self.threads)]
                waiting[slot] = futures
                everything.extend(futures)
        except BaseException:
            self._drain(everything)
            raise
        error = self._drain(everything)
        if error is not None:
            raise error
        self._count('tier_io_read_calls', 1)
        self._count('tier_io_read_blocks', len(ids))
        self._count('tier_io_read_ms', (self.clock() - began) * 1000.0)
        self._count('tier_io_read_device_ms', device_ms)
        self._count('tier_io_read_stage_ms', stage_ms[0])
        return payloads

    def _write(self, ids, payloads):
        import numpy as np

        began = self.clock()
        io = self.io
        shaped = [np.frombuffer(payload, dtype=np.uint8).reshape(len(io.caches), io.chips, io.slice_bytes) for payload in payloads]
        step = self._chunk()
        jobs = [(first, ids[first:first + step], index) for first in range(0, len(ids), step) for index in range(len(io.caches))]
        pool = self._executor()
        stage_ms = [0.0]
        device_ms = 0.0
        staged = {}
        everything = []

        def copy_in(part, first, low, high, index):
            mark = self.clock()
            for offset in range(low, high):
                part[:, offset, :] = shaped[first + offset][index]
            with self._lock:
                stage_ms[0] += (self.clock() - mark) * 1000.0

        def stage(number):
            first, chunk, index = jobs[number]
            part = self._buffer(number % BUFFERS, len(chunk))
            futures = [pool.submit(copy_in, part, first, low, high, index) for low, high in split(len(chunk), self.threads)]
            staged[number] = (part, futures)
            everything.extend(futures)

        try:
            for number in range(min(BUFFERS, len(jobs))):
                stage(number)
            for number, (first, chunk, index) in enumerate(jobs):
                part, futures = staged.pop(number)
                error = self._drain(futures)
                if error is not None:
                    raise error
                mark = self.clock()
                self._device()[1](io.caches[index], part, chunk)
                device_ms += (self.clock() - mark) * 1000.0
                if number + BUFFERS < len(jobs):       # the buffer the device call just finished with
                    stage(number + BUFFERS)
        except BaseException:
            self._drain(everything)
            raise
        self._count('tier_io_write_calls', 1)
        self._count('tier_io_write_blocks', len(ids))
        self._count('tier_io_write_ms', (self.clock() - began) * 1000.0)
        self._count('tier_io_write_device_ms', device_ms)
        self._count('tier_io_write_stage_ms', stage_ms[0])


# ---------------------------------------------------------------------------------------------------------------------------------------------
# Parallel digest checks
# ---------------------------------------------------------------------------------------------------------------------------------------------

class Verifier(object):
    """verdicts(tier, records) -> [True | False | None] for the records, in order: TierStore.verify of each (a pure read of the payload against the digest taken at spill)
    on `threads` workers. hashlib releases the GIL for a payload this size, so the hashes overlap."""

    def __init__(self, threads, stats=None, clock=time.perf_counter):
        self.threads = int(threads)
        self.stats = stats if stats is not None else {}
        self.clock = clock
        self._pool = None

    def verdicts(self, tier, records):
        began = self.clock()
        if len(records) < 2 or self.threads < 2:
            out = [tier.verify(record) for record in records]
        else:
            if self._pool is None:
                from concurrent.futures import ThreadPoolExecutor

                self._pool = ThreadPoolExecutor(max_workers=self.threads, thread_name_prefix='qwen-tier-verify')
            futures = [self._pool.submit(tier.verify, record) for record in records]
            out, error = [], None
            for future in futures:
                try:
                    out.append(future.result())
                except BaseException as caught:  # noqa: BLE001
                    error = error or caught
            if error is not None:
                raise error
        if records:
            for name, amount in (('tier_verify_batches', 1), ('tier_verify_blocks', len(records)), ('tier_verify_ms', (self.clock() - began) * 1000.0)):
                if name in self.stats:
                    self.stats[name] += amount
        return out

    def close(self):
        pool, self._pool = self._pool, None
        if pool is not None:
            pool.shutdown(wait=True)


# ---------------------------------------------------------------------------------------------------------------------------------------------
# What a checkpoint costs in the store (docs/tp4-upload1.md: the memory budget of the preconverted checkpoints)
# ---------------------------------------------------------------------------------------------------------------------------------------------

LAYERS = 48               # GatedDeltaNet layers of the 27B
VALUE_HEADS = 48
HEAD_DIM = 128
CONV_ROWS = 3             # the conv carry [3, 10240] (bf16)
CONV_WIDTH = 10240
TILE = 32
WINDOW_ROWS = 2048
WINDOW_TAPS = 5
HIDDEN = 5120


def checkpoint_bytes(state_bf16=True, preconverted=False, window=False):
    """Bytes one checkpoint is charged in the checkpoint store: 48 layers of recurrent state [48, 128, 128] (bf16, or fp32 without QWEN35_GDN_STATE_BF16) and a bf16 conv carry
    [3, 10240]; preconverted, the carry is held tile-padded (3 rows become 32); with `window`, the exact-resume drafter window (2048 rows of 5 taps of 5120 bf16) beside it."""
    state = LAYERS * VALUE_HEADS * HEAD_DIM * HEAD_DIM * (2 if state_bf16 else 4)
    rows = TILE if preconverted else CONV_ROWS
    carry = LAYERS * rows * CONV_WIDTH * 2
    return state + carry + (WINDOW_ROWS * WINDOW_TAPS * HIDDEN * 2 if window else 0)


def checkpoints_that_fit(store_bytes, **kwargs):
    return int(store_bytes) // checkpoint_bytes(**kwargs)
