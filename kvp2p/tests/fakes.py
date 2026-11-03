"""Fakes for the ttkvp2p tests: a numpy "card", a file "drive", a thread-pool P2PIO, and dict-backed stand-ins for the
runtime and the NVMe manager that WP3a delivers.

The card is a FIFO command queue in front of numpy arrays: an op's effect is applied when it is enqueued (so program order
is queue order) and it COMPLETES only when the test says so (``card.hold``), which is what lets a test put a forward between
an enqueue and its completion. The geometry is tiny (rows of 128 B) but has the real shape: 4 chips x 10 tensors, an object
is one row of every tensor on every chip, and the staging layout is [tensor][row] per object per chip.

``FakeRuntime`` and ``FakeNvmeKVManager`` implement the CONTRACT THE WORKER AND THE CONNECTOR NEED (see ``ttkvp2p.worker.RuntimePort``
and the README): they are the in-repo stand-in so these tests do not wait for WP3a. The manager is dict-backed and takes its store
verdict and digests from ``complete_store(keys, ctx, success=, digests=)``, exactly what the connector's scheduler role hands it.
"""

import collections
import errno
import hashlib
import os
import sys
import threading
import types
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PACKAGE_ROOT = os.environ.get('TTKVP2P_PACKAGE_ROOT') or os.path.dirname(HERE)      # the override lets a mutated copy of the package be tested
STUB_DIR = os.path.join(HERE, 'nvml_stub')


def prepare_environment() -> None:
    """Idempotent: the package on sys.path, the no-GPU probe for this process and for the subprocess vLLM starts, the v1 model runner off."""
    if PACKAGE_ROOT not in sys.path:
        sys.path.insert(0, PACKAGE_ROOT)
    if STUB_DIR not in sys.path:
        sys.path.insert(0, STUB_DIR)
    parts = [p for p in os.environ.get('PYTHONPATH', '').split(os.pathsep) if p]
    for entry in (STUB_DIR, PACKAGE_ROOT):
        if entry not in parts:
            parts.insert(0, entry)
    os.environ['PYTHONPATH'] = os.pathsep.join(parts)
    os.environ.setdefault('VLLM_USE_V2_MODEL_RUNNER', '0')
    import hide_nvml
    hide_nvml.install()


prepare_environment()

from vllm.v1.kv_offload.base import (  # noqa: E402
    LoadStoreSpec,
    LookupResult,
    OffloadingManager,
    PrepareStoreOutput,
    RequestOffloadingContext,
)

from ttkvp2p.spec import (  # noqa: E402
    STATE_AVAILABLE,
    STATE_CLOSING,
    STATE_LATCHED,
    STATE_UNAVAILABLE,
    LayoutGeometry,
    NvmeLoadStoreSpec,
    ReqTag,
)
from ttkvp2p.worker import Backend, StoreResult  # noqa: E402

# --------------------------------------------------------------------------------------------------------------------
# Geometry
# --------------------------------------------------------------------------------------------------------------------

CHIPS = 4
TENSORS = 10
ROW = 128                    # bytes of one block row of one tensor on one chip
PPB = 4                      # pages per block
SHARD_OBJ = TENSORS * ROW    # 1280: an object on one chip
OBJ = CHIPS * SHARD_OBJ      # 5120: an object on the drive
TINY = LayoutGeometry('test-tiny-v1', chips=CHIPS, tensors=TENSORS, row_bytes=ROW, pages_per_block=PPB)
LEAD = 64                    # staging_info aligned_4k_offset: the bank range does not start 4 KiB aligned
BANKS = 8
DATA_OFF = 8192              # where slot 0 lives on the drive


def bank_bytes(slot_objs: int) -> int:
    return LEAD + 4096 + slot_objs * SHARD_OBJ + 3 * 64


# --------------------------------------------------------------------------------------------------------------------
# The card
# --------------------------------------------------------------------------------------------------------------------

class FakeTensor:
    def __init__(self, idx: int):
        self.idx = idx

    def __repr__(self):
        return 'FakeTensor(%d)' % self.idx


class FakeStaging:
    pass


class FakeEvent:
    def __init__(self, card: 'FakeCard', seq: int):
        self.card, self.seq = card, seq

    def query(self) -> bool:
        if self.card.query_raises:
            raise RuntimeError('event query failed')
        return self.card.is_done(self.seq)


class FakeDigest:
    def __init__(self, card: 'FakeCard', seq: int, values: List[List[int]]):
        self.card, self.seq, self.values = card, seq, values

    def ready(self) -> bool:
        return self.card.is_done(self.seq)

    def result(self) -> List[List[int]]:
        return self.values


def row_words_weights(k: int) -> np.ndarray:
    idx = np.arange(k, dtype=np.uint64)
    return (idx * np.uint64(2654435761) + np.uint64(1)) & np.uint64(0xFFFFFFFF)


WEIGHTS = row_words_weights(TENSORS * ROW // 4)


def digest_bytes(data: np.ndarray) -> int:
    """The v3 arithmetic: sum_k ((k * 2654435761 + 1) mod 2^32) * x_k mod 2^64 over the little-endian u32 words."""
    words = np.ascontiguousarray(data).view('<u4').astype(np.uint64)
    return int((words * WEIGHTS[:len(words)]).sum(dtype=np.uint64))


class FakeCard:
    def __init__(self, num_blocks: int = 600, slot_objs: int = 8):
        self.num_blocks = num_blocks
        self.kv = [np.zeros((CHIPS, num_blocks, ROW), dtype=np.uint8) for _ in range(TENSORS)]
        self.bank_bytes = bank_bytes(slot_objs)
        self.stg = np.zeros((BANKS, CHIPS, self.bank_bytes), dtype=np.uint8)
        self.seq = 0
        self.done_upto = 0
        self.hold = False
        self.query_raises = False
        self.log: List[tuple] = []                        # (seq, name, info) in queue order, forwards included
        self.fail_plan: Dict[str, int] = {}               # op name -> raise on the nth call from now (1 = the next)
        self.trace_active = False

    # --- the command queue
    def _enqueue(self, name: str, **info) -> int:
        remaining = self.fail_plan.get(name)
        if remaining is not None:
            if remaining <= 1:
                del self.fail_plan[name]
                raise RuntimeError('injected failure of %s' % name)
            self.fail_plan[name] = remaining - 1
        self.seq += 1
        self.log.append((self.seq, name, info))
        if not self.hold:
            self.done_upto = self.seq
        return self.seq

    def is_done(self, seq: int) -> bool:
        return seq <= self.done_upto

    def release(self) -> None:
        self.hold = False
        self.done_upto = self.seq

    def names(self, name: str) -> List[tuple]:
        return [entry for entry in self.log if entry[1] == name]

    def mark(self, name: str, **info) -> int:
        """A non-op entry in the queue (a forward, a test marker), so tests can order ops against it."""
        self.seq += 1
        self.log.append((self.seq, name, info))
        if not self.hold:
            self.done_upto = self.seq
        return self.seq

    # --- model side: the forward writes KV
    def write_rows(self, tensor_idx: int, block: int, rows: np.ndarray) -> None:
        self.kv[tensor_idx][:, block, :] = rows

    def row(self, tensor_idx: int, block: int) -> np.ndarray:
        return self.kv[tensor_idx][:, block, :].copy()

    def block_bytes(self, block: int) -> np.ndarray:
        """The object of a block as the drive holds it: [chip][tensor][row]."""
        return np.stack([np.concatenate([self.kv[t][d, block] for t in range(TENSORS)]) for d in range(CHIPS)]).reshape(-1)


class FakeOps:
    """The qwen_kv_read-shaped ops over the fake card. Arguments are validated before anything is enqueued (as the real bindings do)."""

    def __init__(self, card: FakeCard, slot_objs: int):
        self.card = card
        self.slot_objs = slot_objs

    def _check(self, tensor, staging, bank, byte_offset, blocks, row_stride, pages_per_block, cq_id):
        card = self.card
        assert isinstance(staging, FakeStaging), 'staging handle'
        assert cq_id == 0, 'CQ0 only'
        assert not card.trace_active, 'no op while a trace is being captured'
        assert 0 <= bank < BANKS
        assert pages_per_block == PPB
        assert row_stride == SHARD_OBJ, 'row stride is the object size on one chip'
        assert 1 <= len(blocks) <= 512
        assert all(1 <= b < card.num_blocks for b in blocks), 'block 0 is the null block'
        first = LEAD + 4096                                               # never the canary page
        assert byte_offset >= first and (byte_offset - first) % ROW == 0, 'offset %d is inside the canary or unaligned' % byte_offset
        end = byte_offset + (len(blocks) - 1) * row_stride + ROW
        assert end <= first + self.slot_objs * SHARD_OBJ, 'offset %d x %d blocks leaves the slot' % (byte_offset, len(blocks))

    def qwen_blocks_to_staging(self, tensor, staging, bank, byte_offset, blocks, row_stride, pages_per_block, cq_id=0):
        self._check(tensor, staging, bank, byte_offset, blocks, row_stride, pages_per_block, cq_id)
        self.card._enqueue('gather', tensor=tensor.idx, bank=bank, offset=byte_offset, blocks=list(blocks))
        for i, block in enumerate(blocks):
            at = byte_offset + i * row_stride
            self.card.stg[bank, :, at:at + ROW] = self.card.kv[tensor.idx][:, block, :]
        return None

    def qwen_write_blocks_from_staging(self, tensor, staging, bank, byte_offset, blocks, row_stride, pages_per_block, cq_id=0):
        self._check(tensor, staging, bank, byte_offset, blocks, row_stride, pages_per_block, cq_id)
        self.card._enqueue('scatter', tensor=tensor.idx, bank=bank, offset=byte_offset, blocks=list(blocks))
        for i, block in enumerate(blocks):
            at = byte_offset + i * row_stride
            self.card.kv[tensor.idx][:, block, :] = self.card.stg[bank, :, at:at + ROW]
        return None

    def qwen_blocks_digest_async(self, tensors, blocks, pages_per_block, cq_id=0):
        card = self.card
        assert cq_id == 0 and pages_per_block == PPB and len(tensors) <= 16 and 1 <= len(blocks) <= 128
        assert all(1 <= b < card.num_blocks for b in blocks)
        seq = card._enqueue('digest', blocks=list(blocks))
        values = [[digest_bytes(np.concatenate([card.kv[t.idx][d, b] for t in tensors])) for b in blocks] for d in range(CHIPS)]
        return FakeDigest(card, seq, values)

    def qwen_p2p_record_event(self, mesh, cq_id=0):
        assert cq_id == 0
        return FakeEvent(self.card, self.card._enqueue('event'))


# --------------------------------------------------------------------------------------------------------------------
# The drive and P2PIO
# --------------------------------------------------------------------------------------------------------------------

class FakeDrive:
    def __init__(self, path: str, num_slots: int):
        self.path = path
        self.num_slots = num_slots
        self.fd = os.open(path, os.O_RDWR | os.O_CREAT)
        os.ftruncate(self.fd, DATA_OFF + num_slots * OBJ)
        self.eio_slots = set()

    def close(self):
        try:
            os.close(self.fd)
        except OSError:
            pass

    def read(self, off: int, length: int) -> np.ndarray:
        return np.frombuffer(os.pread(self.fd, length, off), dtype=np.uint8).copy()

    def write(self, off: int, data: np.ndarray) -> None:
        assert os.pwrite(self.fd, bytes(data), off) == len(data)

    def slot_object(self, slot: int) -> np.ndarray:
        return self.read(DATA_OFF + slot * OBJ, OBJ)

    def corrupt(self, slot: int, at: int = 17) -> None:
        data = self.slot_object(slot)
        data[at] ^= 0x5A
        self.write(DATA_OFF + slot * OBJ, data)


class FakeIO:
    """P2PIO over the fake drive and card staging, on a thread pool. Completions arrive on other threads, as in production."""

    def __init__(self, card: FakeCard, drive: FakeDrive, workers: int = 1):
        self.card, self.drive = card, drive
        self.pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix='fake-p2pio')
        self.gate = threading.Event()
        self.gate.set()                                  # clear it to stall every call (a stalled drive)
        self.calls: List[tuple] = []                     # (kind, drive_off, bank, n_obj, first_obj)
        self.read_errors: collections.deque = collections.deque()   # exceptions to raise on the next reads
        self.write_errors: collections.deque = collections.deque()
        self.short_next = 0
        self.landed = 0
        self.tear_next_reads = 0                                    # the next reads land with one byte wrong in staging (a torn or late DMA)
        self.held: collections.deque = collections.deque()          # events the next calls wait on, one per call (see hold_next)
        self.every_hold: List[threading.Event] = []                 # all of them, so that close() can always let go

    def hold_next(self, count: int = 1) -> List[threading.Event]:
        """The next ``count`` calls block until the returned event is set (use workers >= 2 so later calls can pass)."""
        events = [threading.Event() for _ in range(count)]
        self.held.extend(events)
        self.every_hold.extend(events)
        return events

    def slot_of(self, drive_off: int) -> int:
        return (drive_off - DATA_OFF) // OBJ

    def read_piece(self, drive_off: int, bank: int, n_obj: int, first_obj: int = 0):
        self.calls.append(('read', drive_off, bank, n_obj, first_obj))
        hold = self.held.popleft() if self.held else None
        return self.pool.submit(self._read, drive_off, bank, n_obj, first_obj, hold)

    def write_subrun(self, drive_off: int, bank: int, n_obj: int, first_obj: int = 0):
        assert n_obj <= 64
        self.calls.append(('write', drive_off, bank, n_obj, first_obj))
        hold = self.held.popleft() if self.held else None
        return self.pool.submit(self._write, drive_off, bank, n_obj, first_obj, hold)

    def _read(self, drive_off, bank, n_obj, first_obj, hold=None):
        if hold is not None:
            hold.wait()
        self.gate.wait()
        if self.read_errors:
            raise self.read_errors.popleft()
        first_slot = self.slot_of(drive_off)
        if any(first_slot + i in self.drive.eio_slots for i in range(n_obj)):
            raise OSError(errno.EIO, 'media error')
        base = LEAD + 4096
        for i in range(n_obj):
            data = self.drive.read(drive_off + i * OBJ, OBJ)
            at = base + (first_obj + i) * SHARD_OBJ
            for d in range(CHIPS):
                self.card.stg[bank, d, at:at + SHARD_OBJ] = data[d * SHARD_OBJ:(d + 1) * SHARD_OBJ]
        self.landed += n_obj
        if self.tear_next_reads:
            self.tear_next_reads -= 1
            self.card.stg[bank, 0, base + first_obj * SHARD_OBJ + 3] ^= 0x40
        if self.short_next:
            self.short_next -= 1
            return n_obj * OBJ - 1
        return n_obj * OBJ

    def _write(self, drive_off, bank, n_obj, first_obj, hold=None):
        if hold is not None:
            hold.wait()
        self.gate.wait()
        if self.write_errors:
            raise self.write_errors.popleft()
        base = LEAD + 4096
        for i in range(n_obj):
            at = base + (first_obj + i) * SHARD_OBJ
            data = np.concatenate([self.card.stg[bank, d, at:at + SHARD_OBJ] for d in range(CHIPS)])
            self.drive.write(drive_off + i * OBJ, data)
        return n_obj * OBJ

    def release_all(self):
        """Let every stalled and held call go (a test that failed half way must not hang its own teardown)."""
        self.gate.set()
        for event in self.every_hold:
            event.set()

    def close(self):
        self.release_all()
        self.pool.shutdown(wait=True)


class FakeBackendFactory:
    """worker.BACKEND_FACTORY for the tests; also the record of what the start-up built."""

    def __init__(self, card: FakeCard, drive: FakeDrive, slot_objs: int, workers: int = 1):
        self.card, self.drive, self.slot_objs, self.workers = card, drive, slot_objs, workers
        self.builds = 0
        self.flushes = 0
        self.closed = 0
        self.io: Optional[FakeIO] = None

    def __call__(self, *, cfg, registry, runtime, geometry):
        self.builds += 1
        self.io = FakeIO(self.card, self.drive, self.workers)
        info = {
            'address': 0x80000000, 'size': self.card.bank_bytes, 'page_bytes': 1088,
            'aligned_4k_offset': LEAD, 'aligned_4k_address': 0x80000000 + LEAD,
            'aligned_4k_size': self.card.bank_bytes - LEAD,
            'banks': [{'bank': b, 'noc_x': 1, 'noc_y': b} for b in range(BANKS)],
            'devices': [{'mesh_coord': d, 'device_id': d, 'pci_bus_id': registry.shard_bdf[d]} for d in range(CHIPS)],
        }

        def flush():
            self.flushes += 1

        def close():
            self.closed += 1
            self.io.close()

        return Backend(ops=FakeOps(self.card, self.slot_objs), staging=FakeStaging(), staging_info=info, io=self.io, flush=flush, close=close)


# --------------------------------------------------------------------------------------------------------------------
# The runtime and the allocator (the contract the worker needs)
# --------------------------------------------------------------------------------------------------------------------

class FakeAllocator:
    def __init__(self, num_slots: int):
        self.n = num_slots
        self.gen = [0] * num_slots
        self.inflight = [0] * num_slots
        self.free = collections.deque(range(num_slots))
        self.pending_free = set()
        self.handed_out_in_flight = []          # violations of I5 (must stay empty)
        self.begin_refused = 0

    def alloc(self, count: int) -> Optional[List[int]]:
        if len(self.free) < count:
            return None
        slots = [self.free.popleft() for _ in range(count)]
        for slot in slots:
            if self.inflight[slot]:
                self.handed_out_in_flight.append(slot)
            self.gen[slot] += 1
        return slots

    def release(self, slot: int) -> None:
        """The index no longer names the slot. It is allocatable once no bio is in flight on it (TRIM is instant here)."""
        if self.inflight[slot]:
            self.pending_free.add(slot)
        else:
            self.free.append(slot)

    def io_begin(self, slots, gens) -> bool:
        if any(self.gen[s] != g for s, g in zip(slots, gens)):
            self.begin_refused += 1
            return False
        for s in slots:
            self.inflight[s] += 1
        return True

    def io_end(self, slots) -> None:
        for s in slots:
            assert self.inflight[s] > 0, 'io_end without io_begin on slot %d' % s
            self.inflight[s] -= 1
            if not self.inflight[s] and s in self.pending_free:
                self.pending_free.discard(s)
                self.free.append(s)


class FakeRuntime:
    """One per drive path per process (invariant I10). ``acquire`` returns the same object to both roles."""

    instances: Dict[str, 'FakeRuntime'] = {}
    num_slots = 400
    acquire_log: List[tuple] = []
    host_ok_default = True

    @classmethod
    def reset(cls, num_slots: int = 400) -> None:
        cls.instances = {}
        cls.num_slots = num_slots
        cls.acquire_log = []
        cls.host_ok_default = True

    @classmethod
    def acquire(cls, cfg, role: str) -> 'FakeRuntime':
        cls.acquire_log.append((cfg.nvme_device, role))
        runtime = cls.instances.get(cfg.nvme_device)
        if runtime is None:
            runtime = cls(cfg)
            cls.instances[cfg.nvme_device] = runtime
        runtime.roles.append(role)
        return runtime

    def __init__(self, cfg):
        self.cfg = cfg
        self.state = STATE_UNAVAILABLE
        self.host_ok = self.host_ok_default
        self.host_reason = '' if self.host_ok else 'the status file is stale'
        self.latch_reason: Optional[str] = None
        self.roles: List[str] = []
        self.released: List[str] = []
        self.closed = False
        self.allocator = FakeAllocator(self.num_slots)
        self.store_results: Dict[int, StoreResult] = {}
        self.bad = set()
        self.counters: collections.Counter = collections.Counter()
        self.restore_seconds: List[float] = []
        self.faults: collections.Counter = collections.Counter()

    # --- what the worker needs
    def latch(self, reason: str) -> None:
        if self.state != STATE_LATCHED:
            self.state = STATE_LATCHED
            self.latch_reason = reason

    def set_available(self) -> None:
        if self.state == STATE_UNAVAILABLE:
            self.state = STATE_AVAILABLE

    def mark_bad(self, slot: int, generation: int) -> None:
        self.bad.add((slot, generation))

    def slot_offset(self, slot: int) -> int:
        return DATA_OFF + slot * OBJ

    def count(self, name: str, n: int = 1) -> None:
        self.counters[name] += n

    def observe_restore(self, seconds: float) -> None:
        self.restore_seconds.append(seconds)

    def take_fault(self, kind: str) -> bool:
        if self.faults[kind] > 0:
            self.faults[kind] -= 1
            return True
        return False

    def release(self, role: str) -> None:
        self.released.append(role)
        self.roles.remove(role)
        if not self.roles:
            self.state = STATE_CLOSING
            self.closed = True


# --------------------------------------------------------------------------------------------------------------------
# The manager (a dict-backed NvmeKVManager)
# --------------------------------------------------------------------------------------------------------------------

class Entry:
    __slots__ = ('slot', 'gen', 'state', 'ref', 'digest')

    def __init__(self, slot: int, gen: int):
        self.slot, self.gen = slot, gen
        self.state = 'ALLOCATED'
        self.ref = 0
        self.digest: Optional[np.ndarray] = None


class FakeNvmeKVManager(OffloadingManager):
    """Same interface as ttkvp2p.manager.NvmeKVManager, minimal: no TTL, no cap, LRU eviction only when the slots run out."""

    verdict_calls = 0

    def __init__(self, cfg, runtime):
        self.cfg, self.runtime = cfg, runtime
        self.index: 'collections.OrderedDict[bytes, Entry]' = collections.OrderedDict()
        self.shuffle_keys = False               # a buggy allocator: keys_to_store comes back in another order than it was asked
        self.hits = self.misses = 0
        self.lookups_refused_no_retry = 0
        self.prepare_loads = 0
        self.schedule_ends = 0
        self.complete_store_calls: List[tuple] = []     # (number of keys, success, digests shape) of every complete_store

    # --- helpers
    def _tag(self, ctx) -> Optional[ReqTag]:
        return ctx.get_state(ReqTag)

    def _is_bad(self, entry: Entry) -> bool:
        return entry.state == 'BAD' or (entry.slot, entry.gen) in self.runtime.bad

    def _drop(self, key: bytes) -> None:
        entry = self.index.pop(key)
        self.runtime.allocator.release(entry.slot)

    # --- OffloadingManager
    def on_new_request(self, req_context) -> RequestOffloadingContext:
        return RequestOffloadingContext()

    def lookup(self, key, req_context) -> LookupResult:
        tag = self._tag(req_context)
        if self.runtime.state != STATE_AVAILABLE:
            return LookupResult.MISS
        if tag is not None and tag.no_retry:
            self.lookups_refused_no_retry += 1
            return LookupResult.MISS
        entry = self.index.get(key)
        if entry is None or self._is_bad(entry):
            self.misses += 1
            return LookupResult.MISS
        if entry.state == 'ALLOCATED':
            return LookupResult.HIT_PENDING
        self.hits += 1
        return LookupResult.HIT

    def touch(self, keys, req_context) -> None:
        for key in keys:
            if key in self.index:
                self.index.move_to_end(key)

    def prepare_store(self, keys, req_context) -> Optional[PrepareStoreOutput]:
        empty = PrepareStoreOutput(keys_to_store=[], store_spec=NvmeLoadStoreSpec([], []), evicted_keys=[])
        if self.runtime.state != STATE_AVAILABLE:
            return empty
        tag = self._tag(req_context)
        if tag is not None and (tag.has_mm or tag.admitted is False):
            return empty
        todo = [k for k in keys if k not in self.index]
        if not todo:
            return empty
        allocator = self.runtime.allocator
        slots = allocator.alloc(len(todo))
        if slots is None:
            for key in [k for k, e in self.index.items() if e.state == 'WRITTEN' and e.ref == 0][:len(todo)]:
                self._drop(key)
            slots = allocator.alloc(len(todo))
            if slots is None:
                return empty
        if tag is not None:
            tag.admitted = True
        gens = [allocator.gen[s] for s in slots]
        for key, slot, gen in zip(todo, slots, gens):
            self.index[key] = Entry(slot, gen)
        out_keys, out_slots, out_gens = list(todo), list(slots), list(gens)
        if self.shuffle_keys:
            out_keys, out_slots, out_gens = out_keys[::-1], out_slots[::-1], out_gens[::-1]
        return PrepareStoreOutput(keys_to_store=out_keys, store_spec=NvmeLoadStoreSpec(out_slots, out_gens), evicted_keys=[])

    def prepare_load(self, keys, req_context) -> LoadStoreSpec:
        self.prepare_loads += 1
        slots, gens, expected = [], [], []
        for key in keys:
            entry = self.index[key]
            assert entry.state == 'WRITTEN' and not self._is_bad(entry), 'a HIT entry vanished before prepare_load (I7)'
            entry.ref += 1
            slots.append(entry.slot)
            gens.append(entry.gen)
            expected.append(entry.digest)
        return NvmeLoadStoreSpec(slots, gens, np.array(expected, dtype=np.uint64).reshape(len(keys), 4))

    def complete_store(self, keys, req_context, success: bool = True, digests=None) -> None:
        """The contract of the connector's scheduler role (INTERFACE 3.2): ``keys`` in the order ``prepare_store`` returned them, the
        job's verdict, and on success one digest row per key in that same order. The manager looks nothing up in ``store_results``."""
        keys = list(keys)
        self.complete_store_calls.append((len(keys), success, None if digests is None else tuple(np.shape(digests))))
        if success:
            assert digests is not None and len(digests) == len(keys), 'a successful store hands over one digest row per key'
        for position, key in enumerate(keys):
            entry = self.index.get(key)
            if entry is None or entry.state != 'ALLOCATED':
                continue
            if not success:
                self._drop(key)
                continue
            entry.digest = np.array(digests[position], dtype=np.uint64)
            entry.state = 'WRITTEN'

    def complete_load(self, keys, req_context) -> None:
        for key in keys:
            entry = self.index.get(key)
            if entry is None:
                continue
            entry.ref -= 1
            if self._is_bad(entry):
                entry.state = 'BAD'
                if entry.ref <= 0:
                    self._drop(key)

    def on_schedule_end(self, context) -> None:
        self.schedule_ends += 1

    def reset_cache(self) -> None:
        for key in list(self.index):
            self._drop(key)

    def shutdown(self) -> None:
        self.runtime.release('scheduler')

    # --- test helpers
    def written(self) -> int:
        return sum(1 for e in self.index.values() if e.state == 'WRITTEN')


def install_fake_modules() -> Dict[str, Any]:
    """Put the stand-ins where ``TTP2PNvmeSpec`` looks for ``ttkvp2p.runtime`` and ``ttkvp2p.manager``. Returns what to restore."""
    saved = {name: sys.modules.get(name) for name in ('ttkvp2p.runtime', 'ttkvp2p.manager')}
    runtime_module = types.ModuleType('ttkvp2p.runtime')
    runtime_module.KVP2PRuntime = FakeRuntime
    manager_module = types.ModuleType('ttkvp2p.manager')
    manager_module.NvmeKVManager = FakeNvmeKVManager
    sys.modules['ttkvp2p.runtime'] = runtime_module
    sys.modules['ttkvp2p.manager'] = manager_module
    return saved


def restore_modules(saved: Dict[str, Any]) -> None:
    for name, module in saved.items():
        if module is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = module


# --------------------------------------------------------------------------------------------------------------------
# KV content: a deterministic function of the block hash, the group, the tensor and the chip
# --------------------------------------------------------------------------------------------------------------------

def kv_row(block_hash: bytes, group: int, tensor: int, chip: int) -> np.ndarray:
    seed = bytes(block_hash) + bytes([group, tensor, chip])
    return np.frombuffer(hashlib.shake_128(seed).digest(ROW), dtype=np.uint8)


def kv_rows(block_hash: bytes, group: int, tensor: int) -> np.ndarray:
    return np.stack([kv_row(block_hash, group, tensor, d) for d in range(CHIPS)])
