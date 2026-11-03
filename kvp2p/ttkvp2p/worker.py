"""TTP2PWorker: the job state machine that moves KV objects between card DRAM and the NVMe drive.

Everything here runs on the engine thread. The I/O pool (P2PIO) completes futures on other threads; the worker only
ever polls them, so no state is shared. Card operations and drive I/O are reached ONLY through the injected
collaborators (``Backend``: the ``qwen_kv_read`` ops, a staging handle and a P2PIO), and the process-wide runtime
only through ``RuntimePort``.

Invariants (design 2.4) and where they are enforced:

  I1  exactly one terminal report per job: ``_finish`` is the only place a result is queued, it refuses a job that
      is already terminal, and ``get_finished`` tombstones what it reports (a bounded set of the last 64k ids), so a
      late completion, a retry or a second failure for the id reports nothing.
  I2  no device op after the terminal report: every gather, scatter and digest enqueue goes through ``_enqueue_chunk``,
      which checks the job state and the runtime state at the moment of the enqueue. A failure is terminal.
  I3  quarantine: a staging slot whose bios (or device ops) are still outstanding when its job ends is held in
      ``_holds`` and never reused until the last one returns; a half-enqueued sequence gets a trailing event.
  I4  verification uses the expected digests copied into the job at prepare_load, never a slot table.
  I5  a drive slot is entered into the allocator's in-flight count before a bio touches it
      (``RuntimePort.io_begin`` also checks the slot generation) and leaves it only when the bio returned.
  I8  ``wait()`` accepts unknown, terminal and load ids; a store id gets its gather and source digest enqueued before
      it returns (or the job fails without reading its source); a load id fails the job and quarantines its slots.
  I11 any device-op throw fails the job, quarantines the slot and latches the tier.
  I13 BAD only on proof (a digest mismatch after one re-read, or EIO / EILSEQ on a read); a deadline or a stall fails
      the job and flags the request no-retry, and the keys stay valid.

The worker holds no reference to requests or keys: it sees job ids, drive slots with their generations, device block
ids and the expected digests. The connector worker maps load jobs to request ids.
"""

import collections
import dataclasses
import errno
import importlib
import logging
import threading
import time
from typing import Any, Callable, Deque, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np

from vllm.v1.kv_offload.base import GPULoadStoreSpec, OffloadingWorker, TransferResult

from ttkvp2p.spec import (
    STATE_AVAILABLE,
    NvmeLoadStoreSpec,
    P2PConfig,
    TierRefused,
)

logger = logging.getLogger(__name__)

CANARY_BYTES = 4096                 # the first 4 KiB of a bank, at aligned_4k_offset, is the canary page; slot data follows it
WINDOW_TOP = 0xFF000000             # the window's top 16 MiB is not DRAM
TOMBSTONE_LIMIT = 65536             # I1: the last 64k reported job ids
STORE_RESULT_LIMIT = 1024           # reported store results kept in the runtime for the connector's scheduler role to consume; a result
                                    # pruned before it was consumed reads as a FAILED store there (a missing result means failure)

# Failure reasons.
R_EIO = 'eio'                       # EIO / EILSEQ on a read: the medium is bad (BAD keys)
R_IO = 'io'                         # any other I/O error
R_SHORT = 'short'                   # an I/O call moved fewer bytes than asked
R_DIGEST = 'digest'                 # a digest mismatch after one re-read (BAD keys)
R_DEADLINE = 'deadline'
R_STALL = 'stall'
R_THROW = 'throw'                   # a device op threw (I11): latches the tier
R_LATCHED = 'latched'               # the tier is not AVAILABLE
R_RESET = 'reset'                   # a load flushed by reset_cache (I8)
R_NO_SLOT = 'no_slot'               # a flushed store with no free spill slot (I8)
R_STALE = 'stale'                   # a slot's generation moved on before the bio started (I5)
R_BAD_SPEC = 'bad_spec'
R_SHUTDOWN = 'shutdown'
R_INTERNAL = 'internal'             # a bug in the state machine hit this job at submit (the tier latches; the job is still reported, I1)

# Consecutive failures of these causes (a success resets the count) latch the tier. The others are the tier working as
# designed under pressure or a reset, so they neither count nor reset.
STRIKE_REASONS = frozenset({R_EIO, R_IO, R_SHORT, R_DIGEST, R_DEADLINE, R_STALL, R_THROW})
# Only these mark the keys BAD. Every other failure leaves them valid and flags the request no-retry.
BAD_KEY_REASONS = frozenset({R_EIO, R_DIGEST})
MEDIA_ERRNOS = frozenset({errno.EIO, errno.EILSEQ})
DEVICE_LOST_ERRNOS = frozenset({errno.ENODEV, errno.ENXIO})

# Job and piece states.
ACTIVE, DONE, FAILED = 'active', 'done', 'failed'
P_WAIT, P_READ, P_SCATTER, P_GATHER, P_WRITE, P_DONE = 'wait', 'read', 'scatter', 'gather', 'write', 'done'


# --------------------------------------------------------------------------------------------------------------------
# Results
# --------------------------------------------------------------------------------------------------------------------

@dataclasses.dataclass
class TTTransferResult(TransferResult):
    """A TransferResult that also says why a job failed (the connector worker forwards it to the scheduler role)."""

    reason: str = ''


@dataclasses.dataclass
class StoreResult:
    """What a store job left for the manager (``runtime.store_results[job_id]``): the per-object digests of the SOURCE rows."""

    job_id: int
    success: bool
    slot_ids: Tuple[int, ...] = ()
    generations: Tuple[int, ...] = ()
    digests: Optional[np.ndarray] = None            # [n, 4] uint64, in slot order; None on failure
    reason: str = ''


# --------------------------------------------------------------------------------------------------------------------
# The two ports
# --------------------------------------------------------------------------------------------------------------------

class RuntimePort:
    """Everything the worker needs from the process-wide ``KVP2PRuntime`` (design 3.2), in one place.

    The frozen interface names ``state``, ``allocator``, ``store_results`` and ``latch``; the rest is what the worker
    needs on top of that. A runtime that lacks a member is refused up front (the tier stays UNAVAILABLE with a log line),
    never half-used, because a missing I5 count would be silent unsafety.
    """

    REQUIRED = ('state', 'host_ok', 'latch', 'set_available', 'store_results', 'allocator', 'mark_bad', 'slot_offset')
    ALLOCATOR_REQUIRED = ('io_begin', 'io_end')

    def __init__(self, runtime: Any):
        missing = [name for name in self.REQUIRED if not hasattr(runtime, name)]
        allocator = getattr(runtime, 'allocator', None)
        missing += ['allocator.' + name for name in self.ALLOCATOR_REQUIRED if not hasattr(allocator, name)]
        if missing:
            raise TierRefused('the runtime lacks what the worker needs: ' + ', '.join(sorted(missing)))
        self.runtime = runtime

    # --- state
    @property
    def state(self) -> str:
        return self.runtime.state

    @property
    def host_ok(self) -> bool:
        return bool(self.runtime.host_ok)

    @property
    def host_reason(self) -> str:
        return str(getattr(self.runtime, 'host_reason', '') or '')

    def latch(self, reason: str) -> None:
        self.runtime.latch(reason)

    def set_available(self) -> None:
        self.runtime.set_available()

    # --- I5
    def io_begin(self, slots: Sequence[int], generations: Sequence[int]) -> bool:
        """Count a bio in flight on each slot, unless a slot's generation moved on (then count nothing and return False)."""
        return bool(self.runtime.allocator.io_begin(slots, generations))

    def io_end(self, slots: Sequence[int]) -> None:
        self.runtime.allocator.io_end(slots)

    def slot_offset(self, slot: int) -> int:
        return int(self.runtime.slot_offset(slot))

    # --- results for the manager
    @property
    def store_results(self) -> Dict[int, Any]:
        return self.runtime.store_results

    def mark_bad(self, slot: int, generation: int) -> None:
        self.runtime.mark_bad(slot, generation)

    # --- optional telemetry and the test-only fault commands
    def count(self, name: str, n: int = 1) -> None:
        hook = getattr(self.runtime, 'count', None)
        if hook is not None:
            try:
                hook(name, n)
            except Exception:                   # telemetry must never fail a job
                logger.debug('ttkvp2p: runtime.count failed', exc_info=True)

    def observe_restore(self, seconds: float) -> None:
        hook = getattr(self.runtime, 'observe_restore', None)
        if hook is not None:
            try:
                hook(seconds)
            except Exception:
                logger.debug('ttkvp2p: runtime.observe_restore failed', exc_info=True)

    def take_fault(self, kind: str) -> bool:
        hook = getattr(self.runtime, 'take_fault', None)
        return bool(hook(kind)) if hook is not None else False

    def release(self, role: str) -> None:
        hook = getattr(self.runtime, 'release', None)
        if hook is not None:
            hook(role)


@dataclasses.dataclass
class Backend:
    """The device and drive side, injected (tests pass numpy and file fakes; production wraps ``qwen_kv_read``, CardLink and P2PIO)."""

    ops: Any                                  # qwen_kv_read-like: qwen_blocks_to_staging, qwen_write_blocks_from_staging,
                                              # qwen_blocks_digest_async, qwen_p2p_record_event
    staging: Any                              # the P2PStaging handle
    staging_info: Dict[str, Any]              # qwen_p2p_staging_info(staging)
    io: Any                                   # P2PIO-like: read_piece(drive_off, bank, n_obj[, first_obj]) / write_subrun(...)
    flush: Callable[[], None] = lambda: None  # advisory 8-byte non-posted read through the UC windows after a load's bios returned
    close: Callable[[], None] = lambda: None  # unmap, FREE_TLB, close the drive (section 2.9)


# --------------------------------------------------------------------------------------------------------------------
# Internal records
# --------------------------------------------------------------------------------------------------------------------

class _Never:
    """An event that never completes: the slot behind a sequence whose trailing event could not be enqueued is lost for good."""

    def query(self) -> bool:
        return False


class _Run:
    """One contiguous drive run inside a piece: one P2PIO call."""

    __slots__ = ('first', 'n', 'drive_off', 'slots', 'gens', 'fut', 'done', 'error', 'nbytes', 'began')

    def __init__(self, first: int, n: int, drive_off: int, slots: List[int], gens: List[int]):
        self.first, self.n, self.drive_off, self.slots, self.gens = first, n, drive_off, slots, gens
        self.fut = None
        self.done = False
        self.error: Optional[BaseException] = None
        self.nbytes: Optional[int] = None
        self.began = False


class _Group:
    """One device sequence (10 single-tensor copies, a row digest and an event) over objects [lo, hi) of a piece."""

    __slots__ = ('lo', 'hi', 'handle', 'event', 'complete', 'checked')

    def __init__(self, lo: int, hi: int, handle: Any, event: Any):
        self.lo, self.hi, self.handle, self.event = lo, hi, handle, event
        self.complete = False
        self.checked = False


class _Piece:
    """Up to one staging slot of consecutive objects of a job."""

    __slots__ = ('job', 'lo', 'hi', 'bank', 'state', 'attempt', 'runs', 'groups', 'dev_next', 'retry', 'stalled')

    def __init__(self, job: '_Job', lo: int, hi: int):
        self.job, self.lo, self.hi = job, lo, hi
        self.bank: Optional[int] = None
        self.state = P_WAIT
        self.attempt = 0
        self.runs: List[_Run] = []
        self.groups: List[_Group] = []
        self.dev_next = 0                 # next piece-relative object not yet put through a device sequence
        self.retry = False                # a digest mismatch: re-read once when the outstanding sequences have drained
        self.stalled = False              # test-only fault: this piece's reads are never issued

    @property
    def n(self) -> int:
        return self.hi - self.lo


class _Job:
    __slots__ = ('id', 'is_load', 'n', 'blocks', 'slots', 'gens', 'expected', 'digests', 'state', 'reason', 'pieces',
                 'deadline', 't_submit', 'bad', 'reported', 'starved_since')

    def __init__(self, job_id: int, is_load: bool, blocks: np.ndarray, slots: List[int], gens: List[int],
                 expected: Optional[np.ndarray], now: float, chips: int):
        self.id, self.is_load = job_id, is_load
        self.n = len(slots)
        self.blocks, self.slots, self.gens, self.expected = blocks, slots, gens, expected
        self.digests: Optional[np.ndarray] = None if is_load else np.zeros((self.n, chips), dtype=np.uint64)
        self.state = ACTIVE
        self.reason = ''
        self.pieces: List[_Piece] = []
        self.deadline = float('inf')
        self.t_submit = now
        self.bad: List[Tuple[int, int]] = []
        self.reported = False
        self.starved_since: Optional[float] = None      # stores: since when a piece of the job has waited for a spill bank in vain


class _Hold:
    """A staging slot held back from reuse (I3) until its outstanding bios and device events are done."""

    __slots__ = ('bank', 'runs', 'groups', 'events', 'why', 'since', 'warned')

    def __init__(self, bank: int, runs: List[_Run], groups: List[_Group], events: List[Any], why: str, since: float):
        self.bank, self.runs, self.groups, self.events, self.why, self.since = bank, runs, groups, events, why, since
        self.warned = False


def contiguous_runs(slots: Sequence[int], limit: int) -> List[Tuple[int, int]]:
    """Split a slot list into (start index, count) runs of consecutive slot numbers, each at most ``limit`` long."""
    runs: List[Tuple[int, int]] = []
    i, n = 0, len(slots)
    while i < n:
        j = i + 1
        while j < n and slots[j] == slots[j - 1] + 1 and j - i < limit:
            j += 1
        runs.append((i, j - i))
        i = j
    return runs


# --------------------------------------------------------------------------------------------------------------------
# The worker
# --------------------------------------------------------------------------------------------------------------------

class TTP2PWorker(OffloadingWorker):
    def __init__(self, *, cfg: P2PConfig, geometry, registry, runtime: RuntimePort, backend: Backend,
                 clock: Callable[[], float] = time.monotonic):
        self.cfg = cfg
        self.geometry = geometry
        self.registry = registry
        self.rt = runtime
        self.be = backend
        self.ops = backend.ops
        self._clock = clock

        self._tensors = list(registry.tensors)
        self._mesh = registry.mesh_device
        self._ppb = int(registry.pages_per_block)
        self._row = int(registry.row_bytes_per_chip)
        self._chips = int(registry.num_shards)
        self._shard_obj = len(self._tensors) * self._row            # an object's bytes on one chip: the stride between objects
        self.object_bytes = self._chips * self._shard_obj
        self._num_blocks = int(registry.num_blocks)

        info = backend.staging_info
        lead = int(info['aligned_4k_offset'])
        avail = int(info['aligned_4k_size'])
        self._slot_base = lead + CANARY_BYTES                       # bank-relative offset of object 0 of a slot
        capacity = (avail - CANARY_BYTES) // self._shard_obj
        self.slot_objs = max(0, min(int(cfg.slot_objects), int(capacity)))
        if self.slot_objs < 1:
            raise TierRefused('a staging bank (%d usable bytes) cannot hold one object (%d bytes per chip)' % (avail - CANARY_BYTES, self._shard_obj))
        bank_entries = {int(entry['bank']): entry for entry in info.get('banks', ())}
        wanted = set(cfg.spill_banks) | set(cfg.restore_banks)
        if bank_entries and not wanted <= set(bank_entries):
            raise TierRefused('banks %s are not in the staging allocation (%s)' % (sorted(wanted - set(bank_entries)), sorted(bank_entries)))
        # Every bank has its own address (the allocator's address plus the bank's offset): the highest slot of ANY configured bank
        # must stay below the top 16 MiB of the window. A staging record without per-bank addresses gives bank 0's for all.
        for bank in sorted(wanted):
            address = int(bank_entries.get(bank, {}).get('address', info.get('address', 0)))
            if address + self._slot_base + self.slot_objs * self._shard_obj > WINDOW_TOP:
                raise TierRefused('the staging slot of bank %d would reach the top 16 MiB of the window, which is not DRAM' % bank)

        self._spill_free: List[int] = list(cfg.spill_banks)
        self._restore_free: List[int] = list(cfg.restore_banks)
        self._bank_kind = {bank: 'spill' for bank in cfg.spill_banks}
        self._bank_kind.update({bank: 'restore' for bank in cfg.restore_banks})

        self._jobs: Dict[int, _Job] = {}
        self._order: List[_Job] = []                                  # active jobs, oldest first
        self._outbox: Deque[TTTransferResult] = collections.deque()
        self._tombstones: 'collections.OrderedDict[int, None]' = collections.OrderedDict()
        self._open_runs: List[_Run] = []
        self._holds: List[_Hold] = []
        self._reported_stores: Deque[int] = collections.deque()

        self._strikes = 0
        self._last_progress = self._clock()
        self._spill_released_at = self._clock()                       # when a spill bank last came back (a store waiting for one is starved only while none does)
        self._closing = False
        self._latched = False
        self._decode_active = True
        self._idle = False
        self._cv = threading.Condition()
        self._woken = False
        self.stats: Dict[str, int] = collections.defaultdict(int)

    # ------------------------------------------------------------------------------------------------------------
    # submit
    # ------------------------------------------------------------------------------------------------------------

    def _known(self, job_id: int) -> bool:
        return job_id in self._jobs or job_id in self._tombstones

    def submit_store(self, job_id: int, src: GPULoadStoreSpec, dst: NvmeLoadStoreSpec) -> bool:
        """Register a store job. No device op here: the gather runs at a service point, or at a flush (``wait``). Always True."""
        if self._known(job_id):
            logger.warning('ttkvp2p: store job %s submitted twice; ignored (I1)', job_id)
            return True
        try:
            self._admit(self._make_job(job_id, False, src, dst))
        except Exception as error:
            self._internal_error('submit_store', error)
            self._report_lost(job_id, is_load=False)
        return True

    def submit_load(self, job_id: int, src: NvmeLoadStoreSpec, dst: GPULoadStoreSpec) -> bool:
        """Register a load job. Always True; a job that cannot run is failed and reported like any other (I1)."""
        if self._known(job_id):
            logger.warning('ttkvp2p: load job %s submitted twice; ignored (I1)', job_id)
            return True
        try:
            job = self._make_job(job_id, True, dst, src)
            if job.state == ACTIVE:
                nbytes = job.n * self.object_bytes
                job.deadline = job.t_submit + self.cfg.load_deadline_base_s + nbytes / (self.cfg.load_deadline_gbps * 1e9)
            self._admit(job)
        except Exception as error:
            self._internal_error('submit_load', error)
            self._report_lost(job_id, is_load=True)
        return True

    def _make_job(self, job_id: int, is_load: bool, gpu: GPULoadStoreSpec, nvme: NvmeLoadStoreSpec) -> _Job:
        now = self._clock()
        problem = None
        expected = None
        try:
            blocks = np.asarray(gpu.block_ids, dtype=np.int64)
            slots = [int(s) for s in nvme.slot_ids]
            gens = [int(g) for g in nvme.generations]
            if is_load:
                expected = getattr(nvme, 'expected', None)
            problem = getattr(nvme, 'poison', None)
        except Exception as error:
            blocks, slots, gens = np.zeros(0, dtype=np.int64), [], []
            problem = 'unreadable spec: %s' % (error,)
        n = len(slots)
        if problem is None:
            if n == 0:
                problem = 'empty job'
            elif len(blocks) != n or len(gens) != n:
                problem = 'slots %d, generations %d, blocks %d disagree (I6)' % (n, len(gens), len(blocks))
            elif int(blocks.min()) < 1 or int(blocks.max()) >= self._num_blocks:
                problem = 'a block id is outside 1..%d (block 0 is the null block)' % (self._num_blocks - 1)
            elif len(set(slots)) != n:
                problem = 'a drive slot appears twice in one job'
            elif min(slots) < 0:
                problem = 'a negative drive slot'
            elif is_load and len(np.unique(blocks)) != n:
                problem = 'a destination block appears twice in one load'          # the scatter refuses it, and a refusal is an I11 latch
            elif is_load:
                expected = None if expected is None else np.asarray(expected, dtype=np.uint64)
                if expected is None or expected.shape != (n, self._chips):
                    problem = 'a load needs expected digests of shape (%d, %d) (I4)' % (n, self._chips)
        job = _Job(job_id, is_load, blocks, slots, gens, expected if problem is None else None, now, self._chips)
        if problem is not None:
            logger.error('ttkvp2p: job %s (%s) refused: %s', job_id, 'load' if is_load else 'store', problem)
            job.state = FAILED
            job.reason = R_BAD_SPEC
            return job
        for lo in range(0, n, self.slot_objs):
            job.pieces.append(_Piece(job, lo, min(n, lo + self.slot_objs)))
        return job

    def _admit(self, job: _Job) -> None:
        self._jobs[job.id] = job
        if job.state == FAILED:                       # refused at submit
            self._queue_result(job)
            self._count('fail_' + job.reason)
            return
        state = self.rt.state
        if self._closing or self._latched or state != STATE_AVAILABLE:
            job.state = FAILED
            job.reason = R_LATCHED
            self._queue_result(job)
            self._count('fail_' + R_LATCHED)
            return
        if not self._order:
            # Work starts now. The stall clock measures the time since the last completion WHILE work is in flight, so after an idle
            # period (no P2P job at all, e.g. a long decode with offload_prompt_only) it must not carry the idle time into this job.
            self._note_progress()
        self._order.append(job)
        self._count('submit_load' if job.is_load else 'submit_store')

    # ------------------------------------------------------------------------------------------------------------
    # results (I1)
    # ------------------------------------------------------------------------------------------------------------

    def _queue_result(self, job: _Job) -> None:
        """Queue the one terminal report of a job.

        ``job.reported`` is set only once the report is in the outbox, and the store result is left for the manager only after that:
        if anything raises before the append, the job is still unreported (``_report_lost`` and ``_fail_all`` then queue it) and no
        store result has been written twice. Nothing between the append and the flag can raise."""
        if job.reported:
            return
        ok = job.state == DONE
        if ok:
            now = self._clock()
            report = TTTransferResult(job.id, True, transfer_size=job.n * self.object_bytes, transfer_time=max(0.0, now - job.t_submit))
        else:
            report = TTTransferResult(job.id, False, reason=job.reason)
        self._outbox.append(report)
        job.reported = True
        if job in self._order:
            self._order.remove(job)
        if not job.is_load:
            try:
                results = self.rt.store_results
                results[job.id] = StoreResult(
                    job_id=job.id, success=ok,
                    slot_ids=tuple(job.slots), generations=tuple(job.gens),
                    digests=job.digests.copy() if ok else None, reason=job.reason)
                self._reported_stores.append(job.id)
                while len(self._reported_stores) > STORE_RESULT_LIMIT:
                    if results.pop(self._reported_stores.popleft(), None) is not None:
                        self._count('store_result_pruned')           # never consumed: the scheduler role reads a missing result as a failed store
            except Exception:
                logger.exception('ttkvp2p: cannot leave the store result of job %s for the manager', job.id)

    def _finish_ok(self, job: _Job) -> None:
        if job.state != ACTIVE:
            return
        job.state = DONE
        if job.is_load:
            self.rt.observe_restore(self._clock() - job.t_submit)
        self._strikes = 0
        self._count('done_load' if job.is_load else 'done_store')
        self._queue_result(job)

    def _fail(self, job: _Job, reason: str, *, bad: Sequence[Tuple[int, int]] = (), count: bool = True) -> None:
        """Fail a job (terminal), retire its pieces, report it once. ``bad`` are (slot, generation) pairs proven bad (I13)."""
        if job.state != ACTIVE:
            return
        job.state = FAILED
        job.reason = reason
        if reason in BAD_KEY_REASONS and job.is_load:
            for slot, generation in bad:
                try:
                    self.rt.mark_bad(slot, generation)
                except Exception:
                    logger.exception('ttkvp2p: mark_bad(%s, %s) failed', slot, generation)
        for piece in job.pieces:
            self._retire_piece(piece)
        self._queue_result(job)
        self._count('fail_' + reason)
        if count:
            self._strike(reason)

    def _strike(self, reason: str) -> None:
        if reason == R_THROW:
            self._latch('a device op threw')
            return
        if reason in STRIKE_REASONS:
            self._strikes += 1
            if self._strikes >= self.cfg.strikes_to_latch:
                self._latch('%d consecutive job failures (last: %s)' % (self._strikes, reason))

    def _latch(self, reason: str) -> None:
        if not self._latched:
            self._latched = True
            logger.error('ttkvp2p: latching the tier: %s', reason)
            try:
                self.rt.latch(reason)
            except Exception:
                logger.exception('ttkvp2p: runtime.latch failed')
        self._fail_all(R_LATCHED)

    def _fail_all(self, reason: str) -> None:
        for job in list(self._order):
            if job.state == ACTIVE:
                self._fail(job, reason, count=False)
            elif not job.reported:
                # Terminal but never reported: an error inside _queue_result or _fail came before the report was queued. It keeps its
                # verdict (a finished job is still a finished job) and goes out now, so it cannot sit in the order for ever (I1).
                try:
                    self._queue_result(job)
                except Exception:
                    logger.exception('ttkvp2p: could not report job %s', job.id)

    def _count(self, name: str, n: int = 1) -> None:
        self.stats[name] += n
        self.rt.count(name, n)

    def _internal_error(self, where: str, error: BaseException) -> None:
        """A bug in the state machine must become a miss, never a crash of the engine: latch the tier and fail every job."""
        logger.error('ttkvp2p: internal error in %s: %s: %s', where, type(error).__name__, error, exc_info=True)
        self.stats['internal_error'] += 1
        try:
            self._latch('internal error in %s: %s' % (where, type(error).__name__))
        except Exception:
            logger.exception('ttkvp2p: latching after an internal error failed; failing the jobs by hand')
            self._latched = True
            for job in list(self._order):
                job.state = FAILED
                job.reason = R_LATCHED
                try:
                    self._queue_result(job)
                except Exception:
                    logger.exception('ttkvp2p: could not report job %s', job.id)

    def _report_lost(self, job_id: int, is_load: bool) -> None:
        """I1 after an internal error in a submit: the job must still reach exactly one terminal report.

        Without it the scheduler would keep the job in its table for good (the engine would loop on empty steps while a store is
        pending, and a lost load would park its request). The job may not exist yet (the error came before it was built), may be in
        the table but not in the order (the error came inside ``_admit``), or may already have been failed and reported by the
        latch that ``_internal_error`` ran; the last case reports nothing more."""
        try:
            if job_id in self._tombstones:
                return
            job = self._jobs.get(job_id)
            if job is None:
                job = _Job(job_id, is_load, np.zeros(0, dtype=np.int64), [], [], None, self._clock(), self._chips)
                self._jobs[job_id] = job
            if job.reported:
                return
            job.state = FAILED
            job.reason = R_INTERNAL
            for piece in job.pieces:
                self._retire_piece(piece)
            self._queue_result(job)
            self._count('fail_' + R_INTERNAL)
        except Exception:
            logger.exception('ttkvp2p: job %s could not be reported after an internal error', job_id)

    # ------------------------------------------------------------------------------------------------------------
    # the ring: banks, quarantine (I3)
    # ------------------------------------------------------------------------------------------------------------

    def _take_bank(self, kind: str) -> Optional[int]:
        free = self._spill_free if kind == 'spill' else self._restore_free
        return free.pop(0) if free else None

    def _release_bank(self, bank: int) -> None:
        spill = self._bank_kind[bank] == 'spill'
        free = self._spill_free if spill else self._restore_free
        if bank not in free:
            free.append(bank)
            free.sort(key=lambda b: (self.cfg.spill_banks + self.cfg.restore_banks).index(b))
            if spill:
                self._spill_released_at = self._clock()

    def _group_complete(self, group: _Group) -> bool:
        if not group.complete:
            if group.event.query() and group.handle.ready():
                group.complete = True
        return group.complete

    def _retire_piece(self, piece: _Piece) -> None:
        """The piece ends (done or its job failed): free its slot now, or quarantine it until the outstanding bios and events return."""
        bank = piece.bank
        piece.bank = None
        if bank is None:
            return
        runs = [run for run in piece.runs if not run.done]
        groups = []
        for group in piece.groups:
            try:
                if not self._group_complete(group):
                    groups.append(group)
            except Exception:
                groups.append(group)           # an event that cannot be queried is not known complete
        piece.runs, piece.groups = [], []
        if runs or groups:
            self._holds.append(_Hold(bank, runs, groups, [], 'job ended with work outstanding', self._clock()))
            self._count('quarantined')
        else:
            self._release_bank(bank)

    def _quarantine_permanently(self, bank: int, why: str) -> None:
        logger.error('ttkvp2p: staging bank %d is lost (%s)', bank, why)
        self._holds.append(_Hold(bank, [], [], [_Never()], why, self._clock()))
        self._count('banks_lost')

    def _poll_holds(self) -> None:
        if not self._holds:
            return
        kept = []
        now = self._clock()
        for hold in self._holds:
            try:
                runs_done = all(run.done for run in hold.runs)
                groups_done = all(self._group_complete(group) for group in hold.groups)
                events_done = all(event.query() for event in hold.events)
            except Exception:
                runs_done = groups_done = events_done = False
                if not hold.warned:
                    hold.warned = True
                    logger.error('ttkvp2p: staging bank %d cannot be polled; it stays quarantined', hold.bank, exc_info=True)
            if runs_done and groups_done and events_done:
                self._release_bank(hold.bank)
                self._count('quarantine_released')
            else:
                kept.append(hold)
                if not hold.warned and now - hold.since > 5.0:
                    hold.warned = True
                    logger.warning('ttkvp2p: staging bank %d has been quarantined for %.1f s (%s)', hold.bank, now - hold.since, hold.why)
        self._holds = kept

    # ------------------------------------------------------------------------------------------------------------
    # completions from the I/O threads (polled, never shared)
    # ------------------------------------------------------------------------------------------------------------

    def _wake(self, _future=None) -> None:
        with self._cv:
            self._woken = True
            self._cv.notify_all()

    def _note_progress(self) -> None:
        self._last_progress = self._clock()

    def _reap_runs(self) -> None:
        if not self._open_runs:
            return
        still = []
        for run in self._open_runs:
            fut = run.fut
            if fut.done():
                run.done = True
                try:
                    run.nbytes = fut.result()
                except Exception as error:         # includes CancelledError
                    run.error = error
                if run.began:
                    try:
                        self.rt.io_end(run.slots)
                    except Exception:
                        logger.exception('ttkvp2p: io_end failed for slots %s', run.slots[:4])
                self._note_progress()
            else:
                still.append(run)
        self._open_runs = still

    def _issue_run(self, piece: _Piece, run: _Run, reading: bool) -> bool:
        """Count the run's slots in flight (I5) and hand it to the I/O pool. False: the slots moved on (stale)."""
        if not self.rt.io_begin(run.slots, run.gens):
            return False
        run.began = True
        piece.runs.append(run)
        method = self.be.io.read_piece if reading else self.be.io.write_subrun
        try:
            if run.first == 0:
                fut = method(run.drive_off, piece.bank, run.n)
            else:
                fut = method(run.drive_off, piece.bank, run.n, first_obj=run.first)
        except Exception as error:                 # the pool refused the call: no bio is in flight on these slots
            run.began = False
            self.rt.io_end(run.slots)
            run.done, run.error = True, error
            return True
        run.fut = fut
        self._open_runs.append(run)
        try:
            fut.add_done_callback(self._wake)
        except Exception:
            pass
        return True

    def _build_runs(self, piece: _Piece, limit: int) -> List[_Run]:
        job = piece.job
        slots = job.slots[piece.lo:piece.hi]
        gens = job.gens[piece.lo:piece.hi]
        runs = []
        for start, count in contiguous_runs(slots, limit):
            drive_off = self.rt.slot_offset(slots[start])
            runs.append(_Run(start, count, drive_off, slots[start:start + count], gens[start:start + count]))
        return runs

    # ------------------------------------------------------------------------------------------------------------
    # device sequences (I2, I11)
    # ------------------------------------------------------------------------------------------------------------

    def _enqueue_chunk(self, piece: _Piece, count: int) -> bool:
        """Put the next ``count`` objects of the piece through one device sequence.

        A store gathers the source rows into the slot and digests the SOURCE rows; a load scatters the slot into the
        destination rows and digests the DESTINATION rows. The state of the job and of the tier is checked here, at the
        moment of the enqueue (I2). Returns False when nothing was enqueued."""
        job = piece.job
        if job.state != ACTIVE or piece.bank is None or self._closing or self._latched or self.rt.state != STATE_AVAILABLE:
            return False
        lo = piece.dev_next
        hi = min(piece.n, lo + count)
        if hi <= lo:
            return False
        blocks = [int(b) for b in job.blocks[piece.lo + lo:piece.lo + hi]]
        loading = job.is_load
        op = self.ops.qwen_write_blocks_from_staging if loading else self.ops.qwen_blocks_to_staging
        slot_base = self._slot_base + lo * self._shard_obj
        try:
            if self.rt.take_fault('throw'):
                raise RuntimeError('injected device-op throw')
            for t, tensor in enumerate(self._tensors):
                op(tensor, self.be.staging, bank=piece.bank, byte_offset=slot_base + t * self._row, blocks=blocks,
                   row_stride=self._shard_obj, pages_per_block=self._ppb, cq_id=0)
            handle = self.ops.qwen_blocks_digest_async(list(self._tensors), blocks, self._ppb, cq_id=0)
            event = self.ops.qwen_p2p_record_event(self._mesh, cq_id=0)
        except Exception as error:
            self._device_throw(piece, error)
            return False
        piece.dev_next = hi
        piece.groups.append(_Group(lo, hi, handle, event))
        self._count('objects_scattered' if loading else 'objects_gathered', hi - lo)
        self._note_progress()               # a sequence went onto the queue: the stall clock counts from here, not from before the work existed
        return True

    def _device_throw(self, piece: _Piece, error: BaseException) -> None:
        """I11: the job fails, the slot stays quarantined behind its outstanding work and a trailing event, the tier latches.

        Part of a sequence may be on the queue already, so the slot is not free until a trailing record_event enqueued now
        completes (CQ0 is FIFO). If even that cannot be enqueued the slot is lost for the life of the process."""
        logger.error('ttkvp2p: a device op threw for job %s: %s: %s', piece.job.id, type(error).__name__, error)
        bank, piece.bank = piece.bank, None             # keep _retire_piece from freeing it
        if bank is not None:
            try:
                trailing = self.ops.qwen_p2p_record_event(self._mesh, cq_id=0)
            except Exception:
                logger.error('ttkvp2p: the trailing event could not be enqueued', exc_info=True)
                trailing = _Never()
                self._count('banks_lost')
            runs = [run for run in piece.runs if not run.done]
            groups = []
            for group in piece.groups:
                try:
                    if not self._group_complete(group):
                        groups.append(group)
                except Exception:
                    groups.append(group)
            piece.runs, piece.groups = [], []
            self._holds.append(_Hold(bank, runs, groups, [trailing], 'device op threw', self._clock()))
            self._count('quarantined')
        self._fail(piece.job, R_THROW)

    # ------------------------------------------------------------------------------------------------------------
    # per-piece progress
    # ------------------------------------------------------------------------------------------------------------

    def _check_group(self, piece: _Piece, group: _Group) -> bool:
        """A completed load group is verified against the job's expected digests (I4). False: the job failed or the piece must re-read."""
        job = piece.job
        try:
            result = np.array(group.handle.result(), dtype=np.uint64)
        except Exception as error:
            self._device_throw(piece, error)
            return False
        count = group.hi - group.lo
        if result.shape != (self._chips, count):
            self._device_throw(piece, RuntimeError('digest result has shape %s, wanted %s' % (result.shape, (self._chips, count))))
            return False
        digests = result.T.copy()                                       # [objects, chips]
        base = piece.lo + group.lo
        if not job.is_load:
            job.digests[base:base + count] = digests
            return True
        if self.rt.take_fault('digest'):
            digests[0, 0] ^= np.uint64(1)
        bad = np.nonzero((digests != job.expected[base:base + count]).any(axis=1))[0]
        if len(bad) == 0:
            return True
        self._count('digest_mismatch')
        if piece.attempt == 0:
            piece.retry = True                                          # re-read the piece once, after the sequences in flight drain
            return True
        # The second mismatch is proof: the objects that still disagree go BAD (I13).
        proof = [(job.slots[base + int(i)], job.gens[base + int(i)]) for i in bad]
        self._fail(job, R_DIGEST, bad=proof)
        return False

    def _advance(self) -> None:
        for job in list(self._order):
            if job.state != ACTIVE:
                continue
            for piece in job.pieces:
                if job.state != ACTIVE:
                    break
                if job.is_load:
                    self._advance_load(piece)
                else:
                    self._advance_store(piece)
            if job.state == ACTIVE and all(piece.state == P_DONE for piece in job.pieces):
                self._finish_ok(job)

    def _classify(self, runs: Iterable[_Run]) -> Tuple[Optional[str], List[_Run]]:
        """The reason a set of finished runs failed (None: all good) and the runs behind a media error."""
        reason = None
        media = []
        for run in runs:
            if run.error is not None:
                code = getattr(run.error, 'errno', None)
                if code in MEDIA_ERRNOS:
                    media.append(run)
                    reason = R_EIO
                elif code in DEVICE_LOST_ERRNOS:
                    return R_THROW, []
                elif reason is None:
                    reason = R_IO
            elif run.nbytes is not None and run.nbytes != run.n * self.object_bytes:
                reason = reason or R_SHORT
        return reason, media

    def _advance_load(self, piece: _Piece) -> None:
        job = piece.job
        if piece.state == P_READ:
            if piece.stalled or not all(run.done for run in piece.runs):
                return
            if piece.runs and self.rt.take_fault('eio'):
                piece.runs[0].error = OSError(errno.EIO, 'injected read error')
            reason, media = self._classify(piece.runs)
            if reason == R_THROW:
                self._latch('the drive reported the device lost')
                return
            if reason is not None:
                proof = [(slot, gen) for run in media for slot, gen in zip(run.slots, run.gens)]
                self._fail(job, reason, bad=proof)
                return
            try:
                self.be.flush()                                         # advisory only: the digest decides
            except Exception:
                logger.debug('ttkvp2p: the advisory flush read failed', exc_info=True)
            piece.state = P_SCATTER
            piece.dev_next = 0
            piece.groups = []
            return
        if piece.state == P_SCATTER:
            for group in piece.groups:
                if job.state != ACTIVE:
                    return
                if group.checked:
                    continue
                try:
                    done = self._group_complete(group)
                except Exception as error:
                    self._device_throw(piece, error)
                    return
                if not done:
                    continue
                self._note_progress()
                group.checked = True
                if not self._check_group(piece, group):
                    return
            if job.state != ACTIVE:
                return
            outstanding = [group for group in piece.groups if not group.checked]
            if piece.retry:
                if outstanding:
                    return
                self._reread(piece)
                return
            if piece.dev_next >= piece.n and not outstanding:
                piece.state = P_DONE
                bank, piece.bank = piece.bank, None
                piece.groups, piece.runs = [], []
                self._release_bank(bank)

    def _reread(self, piece: _Piece) -> None:
        """A digest mismatch: read the piece from the drive once more (I13). The slot is still held, nothing is outstanding."""
        piece.attempt += 1
        piece.retry = False
        piece.stalled = False
        piece.groups, piece.runs = [], []
        piece.dev_next = 0
        piece.state = P_READ
        self._count('reread')
        self._start_reads(piece)

    def _start_reads(self, piece: _Piece) -> None:
        job = piece.job
        if self.rt.take_fault('stall'):
            piece.stalled = True                     # test-only: the reads are never issued, so the deadline decides
            self._note_progress()
            return
        runs = self._build_runs(piece, piece.n)
        for run in runs:
            if not self._issue_run(piece, run, reading=True):
                self._fail(job, R_STALE)
                return
        self._note_progress()

    def _advance_store(self, piece: _Piece) -> None:
        job = piece.job
        if piece.state == P_GATHER:
            for group in piece.groups:
                if job.state != ACTIVE:
                    return
                if group.checked:
                    continue
                try:
                    done = self._group_complete(group)
                except Exception as error:
                    self._device_throw(piece, error)
                    return
                if not done:
                    continue
                self._note_progress()
                group.checked = True
                if not self._check_group(piece, group):
                    return
            if job.state != ACTIVE:
                return
            if piece.dev_next >= piece.n and all(group.checked for group in piece.groups):
                self._start_writes(piece)
            return
        if piece.state == P_WRITE:
            if not all(run.done for run in piece.runs):
                return
            reason, _ = self._classify(piece.runs)
            if reason == R_THROW:
                self._latch('the drive reported the device lost')
                return
            if reason is not None:
                self._fail(job, R_IO if reason == R_EIO else reason)
                return
            piece.state = P_DONE
            bank, piece.bank = piece.bank, None
            piece.groups, piece.runs = [], []
            self._release_bank(bank)

    def _start_writes(self, piece: _Piece) -> None:
        job = piece.job
        piece.state = P_WRITE
        piece.groups = []
        for run in self._build_runs(piece, self.cfg.spill_subrun_objects):
            if not self._issue_run(piece, run, reading=False):
                self._fail(job, R_STALE)
                return
        self._note_progress()

    # ------------------------------------------------------------------------------------------------------------
    # service: one pass on the engine thread
    # ------------------------------------------------------------------------------------------------------------

    def begin_step(self, *, decode_active: bool = True, idle: bool = False) -> None:
        """Optional hint from the model runner for the next ``get_finished``: whether a decode batch runs (else the budget is
        unlimited) and whether this is a zero-token step (then it may wait ``idle_wait_ms`` for a completion instead of spinning)."""
        self._decode_active = decode_active
        self._idle = idle

    def get_finished(self) -> List[TransferResult]:
        budget = self.cfg.device_budget_objects_per_step if self._decode_active else None
        idle_ms = self.cfg.idle_wait_ms if self._idle else 0.0
        self._decode_active, self._idle = True, False
        try:
            self.service(budget_objects=budget, idle_wait_ms=idle_ms)
        except Exception as error:
            self._internal_error('service', error)
        results = []
        while self._outbox:
            result = self._outbox.popleft()
            self._jobs.pop(result.job_id, None)
            self._tombstones[result.job_id] = None
            while len(self._tombstones) > TOMBSTONE_LIMIT:
                self._tombstones.popitem(last=False)
            results.append(result)
        return results

    def service(self, *, budget_objects: Optional[int], idle_wait_ms: float) -> None:
        now = self._clock()
        self._reap_runs()
        self._poll_holds()
        state = self.rt.state
        if (self._closing or self._latched or state != STATE_AVAILABLE) and self._order:
            self._fail_all(R_SHUTDOWN if self._closing else R_LATCHED)
        progressed_before = self._last_progress
        if self._order:
            self._advance()
            self._check_deadlines(now)
            self._check_store_starvation(now)
            self._check_stall(now)
            self._dispatch(budget_objects)
            self._advance()                                     # a sequence that completed at once (and a re-read) moves on
        if idle_wait_ms > 0 and not self._outbox and self._order and self._last_progress == progressed_before:
            with self._cv:
                if not self._woken:
                    self._cv.wait(idle_wait_ms / 1000.0)
                self._woken = False
            self._reap_runs()
        else:
            with self._cv:
                self._woken = False

    def _dispatch(self, budget: Optional[int]) -> None:
        if self._latched or self._closing or self.rt.state != STATE_AVAILABLE:
            return
        left = budget if budget is not None else float('inf')
        # 1. restore: reads for pieces waiting on a slot (no device time), then the device stage for landed pieces.
        for job in list(self._order):
            if not job.is_load or job.state != ACTIVE:
                continue
            for piece in job.pieces:
                if job.state != ACTIVE:
                    break
                if piece.state == P_WAIT:
                    bank = self._take_bank('restore')
                    if bank is None:
                        break
                    piece.bank = bank
                    piece.state = P_READ
                    self._start_reads(piece)
        for job in list(self._order):
            if not job.is_load or job.state != ACTIVE:
                continue
            for piece in job.pieces:
                if job.state != ACTIVE or left <= 0:
                    break
                if piece.state == P_SCATTER and not piece.retry and piece.dev_next < piece.n:
                    want = int(min(piece.n - piece.dev_next, left))
                    if self._enqueue_chunk(piece, want):
                        left -= want
        # 2. spill: a slot for a waiting piece, then its gathers.
        for job in list(self._order):
            if job.is_load or job.state != ACTIVE:
                continue
            for piece in job.pieces:
                if job.state != ACTIVE or left <= 0:
                    break
                if piece.state == P_WAIT:
                    bank = self._take_bank('spill')
                    if bank is None:
                        if job.starved_since is None:
                            job.starved_since = self._clock()
                        break
                    job.starved_since = None
                    piece.bank = bank
                    piece.state = P_GATHER
                    piece.dev_next = 0
                if piece.state == P_GATHER and piece.dev_next < piece.n:
                    want = int(min(piece.n - piece.dev_next, left))
                    if self._enqueue_chunk(piece, want):
                        left -= want

    def _check_deadlines(self, now: float) -> None:
        """I13: a load past its deadline fails (recompute); its keys stay valid and its request is flagged no-retry. Decided only here."""
        for job in list(self._order):
            if job.is_load and job.state == ACTIVE and now > job.deadline:
                self._fail(job, R_DEADLINE)

    def _check_store_starvation(self, now: float) -> None:
        """A store that cannot get a spill bank is dropped, never left waiting (design 2.8): a later miss, not a stall.

        A store is starved while a piece of it waits for a bank and no spill bank has come back for ``store_wait_s``. Banks
        that keep rotating, however busy, starve nobody. Banks that are held by stuck I/O, quarantined or lost starve every
        waiting store, which would otherwise keep its keys ALLOCATED (lookups defer on them) for good. The failure is neutral."""
        limit = self.cfg.store_wait_s
        for job in list(self._order):
            if job.is_load or job.state != ACTIVE or job.starved_since is None:
                continue
            if now - max(job.starved_since, self._spill_released_at) > limit:
                self._fail(job, R_NO_SLOT, count=False)

    def _check_stall(self, now: float) -> None:
        """No completion for ``stall_latch_s`` while work is in flight latches the tier (section 2.9)."""
        inflight = bool(self._open_runs) or any(
            piece.state in (P_READ, P_WRITE) or any(not group.checked for group in piece.groups)
            for job in self._order for piece in job.pieces)
        if not inflight:
            self._last_progress = now
            return
        if now - self._last_progress > self.cfg.stall_latch_s:
            self._latch('no completion for %.1f s with work in flight' % (now - self._last_progress))

    # ------------------------------------------------------------------------------------------------------------
    # wait (I8)
    # ------------------------------------------------------------------------------------------------------------

    def wait(self, job_ids: Set[int]) -> None:
        """Called from ``handle_preemptions`` before the forward that may reallocate blocks (the stock worker submits first).

        Accepts unknown, already-terminal and load ids. Never blocks on the drive."""
        try:
            self._reap_runs()
            self._poll_holds()
            for job_id in sorted(job_ids):
                job = self._jobs.get(job_id)
                if job is None or job.state != ACTIVE:
                    continue
                if job.is_load:
                    self._fail(job, R_RESET, count=False)      # fail it (reported once) and quarantine its slots
                else:
                    self._flush_store(job)
        except Exception as error:
            self._internal_error('wait', error)

    def _flush_store(self, job: _Job) -> None:
        """Enqueue every remaining gather and source digest of the job NOW, ignoring the step budget, or fail the job
        without reading its source (no spill slot free)."""
        if self._latched or self._closing or self.rt.state != STATE_AVAILABLE:
            self._fail(job, R_LATCHED, count=False)
            return
        flushed = False
        for piece in job.pieces:
            if job.state != ACTIVE:
                return
            if piece.state == P_WAIT:
                bank = self._take_bank('spill')
                if bank is None:
                    self._fail(job, R_NO_SLOT, count=False)
                    return
                job.starved_since = None
                piece.bank = bank
                piece.state = P_GATHER
                piece.dev_next = 0
            if piece.state == P_GATHER:
                while piece.dev_next < piece.n and job.state == ACTIVE:
                    if not self._enqueue_chunk(piece, piece.n - piece.dev_next):
                        break
                    flushed = True
                if job.state == ACTIVE and piece.dev_next < piece.n:
                    self._fail(job, R_LATCHED, count=False)
                    return
        if flushed:
            self._count('flush_gathered')

    # ------------------------------------------------------------------------------------------------------------
    # shutdown (section 2.9)
    # ------------------------------------------------------------------------------------------------------------

    def shutdown(self) -> None:
        """Fail every job, wait (bounded) for bios in flight, then unmap and close. Never exits with DMA in flight on purpose:
        if the bios do not return inside the drain budget the mappings are left alone and a line is logged."""
        if self._closing:
            return
        self._closing = True
        try:
            self._fail_all(R_SHUTDOWN)
        except Exception as error:
            self._internal_error('shutdown', error)
        deadline = time.monotonic() + float(self.cfg.shutdown_drain_s)         # real time: the budget must hold whatever clock was injected
        while True:
            try:
                self._reap_runs()
                self._poll_holds()
            except Exception as error:
                self._internal_error('shutdown', error)
            if not self._open_runs or time.monotonic() >= deadline:
                break
            with self._cv:
                self._cv.wait(0.005)
        drained = not self._open_runs
        if drained:
            try:
                self.be.close()
            except Exception:
                logger.exception('ttkvp2p: closing the backend failed')
        else:
            logger.error('ttkvp2p: %d bios did not return within the drain budget; the windows and the drive stay open', len(self._open_runs))
        try:
            self.rt.release('worker')
        except Exception:
            logger.exception('ttkvp2p: releasing the runtime failed')

    # ------------------------------------------------------------------------------------------------------------
    # introspection (tests and the state file)
    # ------------------------------------------------------------------------------------------------------------

    @property
    def free_banks(self) -> Dict[str, List[int]]:
        return {'spill': list(self._spill_free), 'restore': list(self._restore_free)}

    @property
    def quarantined_banks(self) -> List[int]:
        return [hold.bank for hold in self._holds]

    @property
    def active_job_ids(self) -> List[int]:
        return [job.id for job in self._order]

    @property
    def strikes(self) -> int:
        return self._strikes

    @property
    def tombstones(self) -> Set[int]:
        return set(self._tombstones)


# --------------------------------------------------------------------------------------------------------------------
# Building a worker
# --------------------------------------------------------------------------------------------------------------------

# Replaced by tests (and by integration glue): called as factory(cfg=, registry=, runtime=, geometry=) -> Backend.
BACKEND_FACTORY: Optional[Callable[..., Backend]] = None


def build_worker(spec, registry) -> TTP2PWorker:
    """The worker role's start-up (design 2.9): attach to the runtime (it runs the host checks), refuse with zero device
    footprint if they failed, then set up the device side and go AVAILABLE. Raises TierRefused; the connector worker
    turns that into a tier that stays off."""
    cfg = spec.p2p_config
    runtime_module = importlib.import_module('ttkvp2p.runtime')
    runtime = runtime_module.KVP2PRuntime.acquire(cfg, 'worker')
    try:
        port = RuntimePort(runtime)
        if not port.host_ok:
            raise TierRefused('host checks failed: %s' % (port.host_reason or 'see the runtime log'))
        factory = BACKEND_FACTORY or default_backend_factory
        backend = factory(cfg=cfg, registry=registry, runtime=runtime, geometry=spec.layout_geometry)
        try:
            worker = TTP2PWorker(cfg=cfg, geometry=spec.layout_geometry, registry=registry, runtime=port, backend=backend)
        except Exception:
            try:
                backend.close()
            except Exception:
                logger.exception('ttkvp2p: closing a backend that was never used failed')
            raise
        port.set_available()
        return worker
    except BaseException:
        try:
            runtime.release('worker')
        except Exception:
            logger.exception('ttkvp2p: releasing the runtime after a failed start failed')
        raise


def default_backend_factory(*, cfg: P2PConfig, registry, runtime, geometry) -> Backend:
    """Production wiring of the device side (design 2.9 steps 3 and 4).

    UNVERIFIED until cardlink.py, p2pio.py and the v4-lite extension exist. It uses only what the frozen interface names,
    except for ASSUMPTIONS: the constructor of P2PIO, ``P2PIO.dma_canary()`` / ``P2PIO.close()`` (the DMA canary needs a 4 KiB
    write and read that read_piece/write_subrun cannot express), and the chardev path ``/dev/tenstorrent/<N>`` with N the tt-kmd
    index the staging record carries as ``pcie_device``. Nothing is imported at module import (I14), and any failure here leaves
    the tier UNAVAILABLE.

    The sequence: the host-only p2pmem file match (zero device footprint when it fails), staging alloc, the shard-map check, a
    CardLink per shard (open, canary write, aim and verify every bank), the DMA canary, the warm-up of every program to
    completion, the freeze. Only then does the caller flip the runtime AVAILABLE.
    """
    import glob

    # Host checks that need no device: each card's p2pmem allocate file, found by its PCI address in the registry's shard map.
    allocate_paths = sorted(glob.glob(cfg.p2pmem_glob))
    allocate_of = {}
    for bdf in registry.shard_bdf:
        matches = [path for path in allocate_paths if str(bdf).lower() in path.lower()]
        if len(matches) != 1:
            raise TierRefused('expected one p2pmem allocate file for %s, found %d' % (bdf, len(matches)))
        allocate_of[str(bdf).lower()] = matches[0]

    ops = importlib.import_module('qwen_kv_read')
    staging = ops.qwen_p2p_staging_alloc(registry.mesh_device, cfg.staging_bytes_per_bank)
    info = ops.qwen_p2p_staging_info(staging)
    cardlink = importlib.import_module('ttkvp2p.cardlink')
    p2pio = importlib.import_module('ttkvp2p.p2pio')
    devices = info['devices']
    bdfs = [str(entry['pci_bus_id']).lower() for entry in devices]
    if [str(b).lower() for b in registry.shard_bdf] != bdfs:
        raise TierRefused('the registry shard map %s disagrees with the staging devices %s' % (list(registry.shard_bdf), bdfs))
    # /dev/tenstorrent/N is the tt-kmd index. tt-metal's logical device_id is NOT that index; pcie_device is (None when the host could
    # not resolve it, and then the card cannot be found by number). Settled for every card before any window is opened.
    chardevs = []
    for position, entry in enumerate(devices):
        index = entry.get('pcie_device')
        if index is None:
            raise TierRefused('the staging record of device %d carries no tt-kmd index (pcie_device)' % position)
        chardevs.append('/dev/tenstorrent/%d' % int(index))
    links = []
    io = None
    try:
        for chardev, entry in zip(chardevs, devices):
            bdf = str(entry['pci_bus_id'])
            link = cardlink.CardLink(chardev, bdf, allocate_of[bdf.lower()], cfg.p2p_window_id)
            links.append(link)
            link.open(info)
        for bank in tuple(cfg.spill_banks) + tuple(cfg.restore_banks):
            for link in links:
                link.aim(bank)                                     # CONFIGURE_TLB plus the canary read-back; no bio is in flight yet
        io = p2pio.P2PIO(links=links, runtime=runtime, config=cfg, staging_info=info)
        io.dma_canary()
        ops.qwen_p2p_warm(staging, list(registry.tensors), null_block=0)
        ops.qwen_p2p_freeze(staging)
    except BaseException:
        for closer in ([io.close] if io is not None else []) + [link.close for link in links]:
            try:
                closer()
            except Exception:
                logger.exception('ttkvp2p: closing after a failed start failed')
        raise

    def flush() -> None:
        for link in links:
            link.flush()

    def close() -> None:
        io.close()
        for link in links:
            link.close()

    return Backend(ops=ops, staging=staging, staging_info=info, io=io, flush=flush, close=close)
