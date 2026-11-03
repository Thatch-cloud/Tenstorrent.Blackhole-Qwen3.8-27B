"""KVP2PRuntime: the process-wide health, slot and drive state both connector roles share (invariant I10), and the SlotAllocator (I5).

One runtime per drive path per process. Under the ``uni`` executor both connector roles live in the EngineCore process and the
worker initialises first: the worker role ``acquire``s with role "worker" and OPENS the runtime (the host checks of design 2.9 step 1,
the drive claim, the superblock); the scheduler role ``acquire``s with role "scheduler" and only ATTACHES. Release is reference
counted: the last release runs the shutdown (state CLOSING, a final state file, the drive closed unless a bio is still in flight).

The ENGINE THREAD owns every piece of state (I7). The drive-side I/O (the open-time discard of the data region and the TRIM of
evicted slots) runs on one daemon thread that only posts completions to a queue; the engine thread applies them in ``service()``,
which the manager calls from ``on_schedule_end``. Nothing here ever blocks the engine thread on the drive.

Invariant I5 (slot reuse). A drive slot becomes allocatable only when all three hold: no index entry names it, no bio of any job
(current or stale) is in flight on it, and its TRIM has completed. Each reallocation bumps the slot generation. The slot life::

    TRIM_WAIT --(TRIM issued)--> TRIMMING --(TRIM ok)--> FREE --(alloc, generation + 1)--> USED --(release)--> TRIM_WAIT
                                     \\--(TRIM failed)--> QUARANTINED (never reused)         \\--(release with a bio in flight)--> DRAINING
    DRAINING --(last bio returns)--> TRIM_WAIT

Every slot starts in TRIM_WAIT: that IS the open-time whole-data-region discard of mode "process" ("start empty"), done in runs of at
most 1 GiB, and a slot is allocatable as its run completes.

This module does not import vLLM, ttnn or the extension (stdlib and numpy only).
"""

import bisect
import collections
import logging
import queue
import threading
import time
from typing import Any, Callable, Deque, Dict, List, Optional, Set, Tuple

import numpy as np

from ttkvp2p import health
from ttkvp2p import store as store_module
from ttkvp2p.health import HostEnv, HostRefusal

logger = logging.getLogger(__name__)

STATE_UNAVAILABLE = 'UNAVAILABLE'
STATE_AVAILABLE = 'AVAILABLE'
STATE_LATCHED = 'LATCHED'
STATE_CLOSING = 'CLOSING'

# slot states
S_FREE, S_USED, S_TRIM_WAIT, S_TRIMMING, S_DRAINING, S_QUARANTINED = range(6)
SLOT_STATE_NAMES = ('free', 'used', 'trim_wait', 'trimming', 'draining', 'quarantined')

CONTROL_POLL_S = 1.0
STATE_WRITE_S = 10.0
STATUS_POLL_S = 5.0
TRIMS_IN_FLIGHT = 2
RESTORE_SAMPLES = 1024
FIT_SCAN_LIMIT = 256


# --------------------------------------------------------------------------------------------------------------------
# Sorted runs of slot numbers
# --------------------------------------------------------------------------------------------------------------------

class _Runs:
    """Disjoint runs (start, length) of slot numbers in two sorted parallel lists, coalescing on insert."""

    def __init__(self) -> None:
        self.starts: List[int] = []
        self.lens: List[int] = []
        self.total = 0

    def clear(self) -> None:
        self.starts.clear()
        self.lens.clear()
        self.total = 0

    def add(self, start: int, length: int) -> None:
        if length <= 0:
            return
        index = bisect.bisect_left(self.starts, start)
        merge_prev = index > 0 and self.starts[index - 1] + self.lens[index - 1] == start
        merge_next = index < len(self.starts) and start + length == self.starts[index]
        if merge_prev and merge_next:
            self.lens[index - 1] += length + self.lens[index]
            del self.starts[index]
            del self.lens[index]
        elif merge_prev:
            self.lens[index - 1] += length
        elif merge_next:
            self.starts[index] = start
            self.lens[index] += length
        else:
            self.starts.insert(index, start)
            self.lens.insert(index, length)
        self.total += length

    def add_slots(self, slots) -> None:
        """Insert many slot numbers (any order, no duplicates, none already present)."""
        array = np.unique(np.asarray(slots, dtype=np.int64)) if len(slots) else np.zeros(0, dtype=np.int64)
        if array.size == 0:
            return
        breaks = (np.nonzero(np.diff(array) != 1)[0] + 1).tolist()
        bounds = [0] + breaks + [int(array.size)]
        for lo, hi in zip(bounds[:-1], bounds[1:]):
            self.add(int(array[lo]), hi - lo)

    def take(self, count: int) -> Optional[List[int]]:
        """Remove and return ``count`` slots, preferring one contiguous run (first fit over a bounded scan), else the first runs in order."""
        if count <= 0:
            return []
        if count > self.total:
            return None
        scanned = 0
        for index, length in enumerate(self.lens):
            if length >= count:
                start = self.starts[index]
                self._shrink_front(index, count)
                return list(range(start, start + count))
            scanned += 1
            if scanned >= FIT_SCAN_LIMIT:
                break
        slots: List[int] = []
        while len(slots) < count:
            start, length = self.starts[0], self.lens[0]
            use = min(length, count - len(slots))
            slots.extend(range(start, start + use))
            self._shrink_front(0, use)
        return slots

    def pop_front(self, max_length: int) -> Optional[Tuple[int, int]]:
        if not self.starts:
            return None
        start = self.starts[0]
        use = min(self.lens[0], max_length)
        self._shrink_front(0, use)
        return start, use

    def _shrink_front(self, index: int, count: int) -> None:
        if self.lens[index] == count:
            del self.starts[index]
            del self.lens[index]
        else:
            self.starts[index] += count
            self.lens[index] -= count
        self.total -= count

    def contains(self, slot: int) -> bool:
        index = bisect.bisect_right(self.starts, slot) - 1
        return index >= 0 and slot < self.starts[index] + self.lens[index]

    def slots(self) -> List[int]:
        out: List[int] = []
        for start, length in zip(self.starts, self.lens):
            out.extend(range(start, start + length))
        return out


# --------------------------------------------------------------------------------------------------------------------
# The slot allocator (I5)
# --------------------------------------------------------------------------------------------------------------------

class SlotAllocator:
    """Free list, generations, in-flight counts, quarantine and the TRIM queue of the drive's slots. Engine thread only."""

    def __init__(self, num_slots: int = 0):
        self.num_slots = 0
        self.state = np.zeros(0, dtype=np.uint8)
        self.gen = np.zeros(0, dtype=np.uint32)
        self.inflight = np.zeros(0, dtype=np.int32)
        self._free = _Runs()
        self._trim_wait = _Runs()
        self._counts = [0] * len(SLOT_STATE_NAMES)
        self.trim_failed = 0
        if num_slots:
            self.reset(num_slots)

    # --- set-up
    def reset(self, num_slots: int, *, all_need_trim: bool = True) -> None:
        """Size the allocator. Every slot starts in TRIM_WAIT (the open-time discard) unless ``all_need_trim`` is False (then FREE)."""
        self.num_slots = int(num_slots)
        self.state = np.full(self.num_slots, S_TRIM_WAIT if all_need_trim else S_FREE, dtype=np.uint8)
        self.gen = np.zeros(self.num_slots, dtype=np.uint32)
        self.inflight = np.zeros(self.num_slots, dtype=np.int32)
        self._free.clear()
        self._trim_wait.clear()
        (self._trim_wait if all_need_trim else self._free).add(0, self.num_slots)
        self._recount()

    def _recount(self) -> None:
        self._counts = np.bincount(self.state, minlength=len(SLOT_STATE_NAMES)).tolist() if self.num_slots else [0] * len(SLOT_STATE_NAMES)

    # --- views
    @property
    def n_free(self) -> int:
        return self._counts[S_FREE]

    @property
    def n_used(self) -> int:
        return self._counts[S_USED]

    def count(self, state: int) -> int:
        return self._counts[state]

    def total_inflight(self) -> int:
        return int(self.inflight.sum()) if self.num_slots else 0

    def generation(self, slot: int) -> int:
        return int(self.gen[slot])

    def is_free(self, slot: int) -> bool:
        return int(self.state[slot]) == S_FREE

    def state_name(self, slot: int) -> str:
        return SLOT_STATE_NAMES[int(self.state[slot])]

    def stats(self) -> Dict[str, int]:
        out = {name: self._counts[i] for i, name in enumerate(SLOT_STATE_NAMES)}
        out['slots'] = self.num_slots
        out['inflight'] = self.total_inflight()
        out['trim_failed'] = self.trim_failed
        return out

    # --- allocation
    def alloc(self, count: int) -> Optional[List[int]]:
        """``count`` free slots in allocation order (contiguous where possible), their generations bumped; None when fewer are free."""
        if count <= 0:
            return []
        slots = self._free.take(count)
        if slots is None:
            return None
        index = np.asarray(slots, dtype=np.int64)
        self.state[index] = S_USED
        self.gen[index] += 1
        self._counts[S_FREE] -= count
        self._counts[S_USED] += count
        return slots

    # --- release (the index no longer names the slot)
    def release(self, slot: int) -> None:
        self.release_many([slot])

    def release_many(self, slots) -> None:
        """The index stopped naming these slots. A slot with a bio in flight drains first (I5); the others wait for their TRIM."""
        if not len(slots):
            return
        index = np.unique(np.asarray(slots, dtype=np.int64))
        index = index[self.state[index] == S_USED]          # a second drop of the same entry is harmless
        if index.size == 0:
            return
        busy = self.inflight[index] > 0
        self.state[index[busy]] = S_DRAINING
        quiet = index[~busy]
        self.state[quiet] = S_TRIM_WAIT
        self._trim_wait.add_slots(quiet)
        self._recount()

    def invalidate_all(self) -> None:
        """Every slot goes back through TRIM (the ``wipe`` control verb): USED and FREE slots to TRIM_WAIT (a USED one with a bio in
        flight drains first); TRIMMING and QUARANTINED slots stay as they are."""
        used = self.state == S_USED
        draining = used & (self.inflight > 0)
        wait = (used & ~draining) | (self.state == S_FREE)
        self.state[draining] = S_DRAINING
        self.state[wait] = S_TRIM_WAIT
        self._free.clear()
        self._trim_wait.add_slots(np.nonzero(wait)[0])
        self._recount()

    # --- bios in flight (the worker)
    def io_begin(self, slots, generations) -> bool:
        """Count a bio in flight on each slot. Refuses, counting nothing, when any slot is not USED or its generation moved on."""
        slots = [int(s) for s in slots]
        for slot, generation in zip(slots, generations):
            if not 0 <= slot < self.num_slots or int(self.state[slot]) != S_USED or int(self.gen[slot]) != int(generation):
                return False
        for slot in slots:
            self.inflight[slot] += 1
        return True

    def io_end(self, slots) -> None:
        released: List[int] = []
        for slot in slots:
            slot = int(slot)
            if self.inflight[slot] <= 0:
                logger.error('ttkvp2p: io_end on slot %d with no bio in flight', slot)
                continue
            self.inflight[slot] -= 1
            if self.inflight[slot] == 0 and int(self.state[slot]) == S_DRAINING:
                self.state[slot] = S_TRIM_WAIT
                released.append(slot)
        if released:
            self._trim_wait.add_slots(released)
            self._recount()

    # --- the TRIM queue
    def take_trim_runs(self, max_runs: int, max_slots: int) -> List[Tuple[int, int]]:
        """Up to ``max_runs`` runs of slots (first, count) waiting for TRIM, each at most ``max_slots`` long; they move to TRIMMING."""
        runs: List[Tuple[int, int]] = []
        for _ in range(max_runs):
            run = self._trim_wait.pop_front(max_slots)
            if run is None:
                break
            first, count = run
            self.state[first:first + count] = S_TRIMMING
            self._counts[S_TRIM_WAIT] -= count
            self._counts[S_TRIMMING] += count
            runs.append(run)
        return runs

    def trim_done(self, first: int, count: int, ok: bool) -> None:
        """A TRIM run finished. On success its slots become FREE (allocatable); on failure they are quarantined for good (their old KV may
        still be readable, and the TRIM is mandatory)."""
        window = self.state[first:first + count]
        mine = np.nonzero(window == S_TRIMMING)[0] + first
        if mine.size == 0:
            return
        if ok:
            self.state[mine] = S_FREE
            self._free.add_slots(mine)
        else:
            self.state[mine] = S_QUARANTINED
            self.trim_failed += int(mine.size)
        self._recount()

    def quarantine(self, slot: int) -> None:
        """Take a slot out of circulation for good (a USED or DRAINING one; the others are already on their way through TRIM)."""
        if int(self.state[slot]) in (S_USED, S_DRAINING):
            self.state[slot] = S_QUARANTINED
            self._recount()

    def trim_backlog(self) -> int:
        return self._counts[S_TRIM_WAIT] + self._counts[S_TRIMMING]


# --------------------------------------------------------------------------------------------------------------------
# Store results: what the worker leaves for the manager
# --------------------------------------------------------------------------------------------------------------------

class StoreResultsMap(dict):
    """``job_id -> StoreResult`` (worker.StoreResult: ``success``, ``slot_ids``, ``generations``, ``digests`` [n, 4], ``reason``), with an
    index from (slot, generation) to (job id, position). The manager pairs an entry with its digest through the index, in O(1) per
    key: no scan over the results (a 262k session is about 2,000 keys against up to 1,024 kept results). A result the worker pruned is
    gone for good, and the manager treats a missing result as a failed store."""

    def __init__(self) -> None:
        super().__init__()
        self._by_slot: Dict[Tuple[int, int], Tuple[int, int]] = {}

    def __setitem__(self, job_id, result) -> None:
        if job_id in self:
            self._unindex(job_id)
        super().__setitem__(job_id, result)
        slots = getattr(result, 'slot_ids', ()) or ()
        generations = getattr(result, 'generations', ()) or ()
        for position, (slot, generation) in enumerate(zip(slots, generations)):
            self._by_slot[(int(slot), int(generation))] = (job_id, position)

    def _unindex(self, job_id) -> None:
        result = dict.get(self, job_id)
        if result is None:
            return
        slots = getattr(result, 'slot_ids', ()) or ()
        generations = getattr(result, 'generations', ()) or ()
        for slot, generation in zip(slots, generations):
            key = (int(slot), int(generation))
            owner = self._by_slot.get(key)
            if owner is not None and owner[0] == job_id:
                del self._by_slot[key]

    def __delitem__(self, job_id) -> None:
        self._unindex(job_id)
        super().__delitem__(job_id)

    def pop(self, job_id, *default):
        if job_id in self:
            self._unindex(job_id)
        return super().pop(job_id, *default)

    def clear(self) -> None:
        super().clear()
        self._by_slot.clear()

    def popitem(self):
        job_id, result = super().popitem()
        self._by_slot = {k: v for k, v in self._by_slot.items() if v[0] != job_id}
        return job_id, result

    def setdefault(self, job_id, default=None):
        if job_id not in self:
            self[job_id] = default
        return self[job_id]

    def update(self, *args, **kwargs) -> None:
        for job_id, result in dict(*args, **kwargs).items():
            self[job_id] = result

    def lookup(self, slot: int, generation: int):
        """(result, position) of the store job that wrote (slot, generation), or None."""
        owner = self._by_slot.get((int(slot), int(generation)))
        if owner is None:
            return None
        result = dict.get(self, owner[0])
        return None if result is None else (result, owner[1])


# --------------------------------------------------------------------------------------------------------------------
# The drive I/O thread: TRIM only. It posts completions to a queue and touches no other state.
# --------------------------------------------------------------------------------------------------------------------

class _IoThread:
    def __init__(self, done: 'queue.SimpleQueue', sync: bool):
        self._jobs: 'queue.Queue' = queue.Queue()
        self._done = done
        self._sync = sync
        self._thread: Optional[threading.Thread] = None
        self._closed = False

    def submit(self, token: Any, fn: Callable[[], Any]) -> None:
        if self._sync:
            self._run(token, fn)
            return
        if self._thread is None:
            self._thread = threading.Thread(target=self._loop, name='kvp2p-trim', daemon=True)
            self._thread.start()
        self._jobs.put((token, fn))

    def _run(self, token: Any, fn: Callable[[], Any]) -> None:
        try:
            fn()
            self._done.put((token, None))
        except BaseException as error:           # a failed TRIM is reported, never raised into the engine
            self._done.put((token, error))

    def _loop(self) -> None:
        while True:
            item = self._jobs.get()
            if item is None:
                return
            self._run(*item)

    def close(self, timeout: float) -> bool:
        """Stop after the queued work; True when the thread is gone within ``timeout`` seconds."""
        if self._thread is None:
            return True
        self._jobs.put(None)
        self._thread.join(timeout)
        return not self._thread.is_alive()


# --------------------------------------------------------------------------------------------------------------------
# The outside world, replaceable by tests
# --------------------------------------------------------------------------------------------------------------------

_HOST_ENV = HostEnv()


def set_host_env(env: Optional[HostEnv]) -> HostEnv:
    """Replace the outside inputs (tests). ``None`` restores the real ones. Returns the previous object."""
    global _HOST_ENV
    previous = _HOST_ENV
    _HOST_ENV = env if env is not None else HostEnv()
    return previous


def get_host_env() -> HostEnv:
    return _HOST_ENV


# --------------------------------------------------------------------------------------------------------------------
# The runtime
# --------------------------------------------------------------------------------------------------------------------

class KVP2PRuntime:
    """See the module docstring. ``acquire`` / ``release`` are the only constructors and destructor callers use."""

    _instances: Dict[str, 'KVP2PRuntime'] = {}
    _guard = threading.Lock()

    # --- construction
    @classmethod
    def acquire(cls, cfg, role: str) -> 'KVP2PRuntime':
        """The process's runtime for this drive path. Role "worker" opens it (the host checks, the drive); "scheduler" attaches."""
        if role not in ('worker', 'scheduler'):
            raise ValueError('role must be "worker" or "scheduler", not %r' % (role,))
        with cls._guard:
            runtime = cls._instances.get(cfg.nvme_device)
            if runtime is None:
                runtime = cls(cfg, get_host_env())
                cls._instances[cfg.nvme_device] = runtime
            runtime.roles.append(role)
        if role == 'worker':
            runtime._open_once()
        return runtime

    @classmethod
    def reset_registry(cls) -> None:
        """Forget every runtime without closing it (tests that leave the process state clean)."""
        with cls._guard:
            cls._instances.clear()

    def __init__(self, cfg, env: HostEnv):
        self.cfg = cfg
        self.env = env
        self.roles: List[str] = []
        self.state = STATE_UNAVAILABLE
        self.latch_reason: Optional[str] = None
        self.latch_code: Optional[str] = None
        self.latched_total: Dict[str, int] = collections.Counter()
        self.host_ok = False
        self.host_code: Optional[str] = None
        self.host_reason = 'the worker role has not opened the runtime yet'
        self.allocator = SlotAllocator()
        self.store_results = StoreResultsMap()
        self.bad: Set[Tuple[int, int]] = set()
        self.counters: 'collections.Counter[str]' = collections.Counter()
        self.faults: 'collections.Counter[str]' = collections.Counter()
        self.restore_seconds: Deque[float] = collections.deque(maxlen=RESTORE_SAMPLES)
        self.store: Optional[store_module.Store] = None
        self.status: Optional[health.StatusFacts] = None
        self.p2pmem_paths: List[str] = []
        self.layout = None
        self.closed = False
        self.opened_at: Optional[float] = None
        self.last_schedule_wall: Optional[float] = None
        self.selftest_requests = 0
        self._opened = False
        self._device_ready = False
        self._pre_latch_state = STATE_UNAVAILABLE
        self._manager = None
        self._done: 'queue.SimpleQueue' = queue.SimpleQueue()
        self._io = _IoThread(self._done, env.sync_io)
        self._trims_out = 0
        self._edge_seq = -1
        self._level_seq_acked = -1
        self._acks: Deque[Dict[str, Any]] = collections.deque(maxlen=16)
        self._next_control = 0.0
        self._next_state = 0.0
        self._next_status = 0.0
        self._aer_baseline: Optional[int] = None
        self._drive_state = 'unknown'

    # --- opening: the host checks of design 2.9 step 1 (no device work)
    def _open_once(self) -> None:
        with self._guard:
            if self._opened:
                return
            self._opened = True
        self.opened_at = self.env.wall()
        try:
            self._run_host_checks()
            self.host_ok = True
            self.host_code = None
            self.host_reason = ''
            logger.info('ttkvp2p: host checks passed; %d slots of %d bytes, data region discard queued', self.allocator.num_slots,
                        self.store.object_bytes if self.store else 0)
        except HostRefusal as refusal:
            self._refuse(refusal.code, refusal.detail)
        except store_module.StoreError as error:
            self._refuse(error.code, str(error))
        except Exception as error:           # the tier must never stop the engine: it stays off with the cause on record
            logger.exception('ttkvp2p: the host checks failed unexpectedly')
            self._refuse('superblock_unknown', 'host checks failed unexpectedly: %s: %s' % (type(error).__name__, error))
        self.write_state(self.env.wall(), force=True)

    def _refuse(self, code: str, detail: str) -> None:
        self.host_ok = False
        self.host_code = code
        self.host_reason = '%s: %s' % (code, detail)
        logger.warning('ttkvp2p: the tier cannot start (%s)', self.host_reason)
        if self.store is not None:
            if self.allocator.total_inflight() == 0:
                self.store.close()
            self.store = None

    def _run_host_checks(self) -> None:
        from ttkvp2p.spec import LAYOUTS          # the layout table; imported here so this module itself stays free of vLLM until it opens

        cfg, env = self.cfg, self.env
        layout = LAYOUTS.get(cfg.layout)
        if layout is None:
            raise HostRefusal('schema_invalid', 'unknown layout %r' % (cfg.layout,))
        self.layout = layout
        # the kill switch is re-read at every open: an `off` survives restarts by design
        commands = health.list_control(cfg.control_dir)
        if health.switch_is_off(commands):
            self._set_latched('control off (a control directory switch from before this start)', 'control_off')
            raise HostRefusal('control_off', 'the control directory says off (clear it with an `on` file, then restart the engine)')
        health.check_policy(cfg, env)
        # require_sed false is a gate-only value (check_policy refused it outside test mode): then the status file is optional
        facts = health.check_status(cfg, env) if cfg.require_sed else self._lenient_status()
        self.status = facts
        self._aer_baseline = facts.aer_uncorrectable if facts is not None else 0

        def expected_claim(size: int) -> Optional[bytes]:
            # the superblock's claim covers the drive's by-id name (from the status file) and its size: a cloned or swapped drive fails here
            return store_module.claim_digest(facts.by_id, size) if facts is not None and facts.by_id else None

        self.store = store_module.open_store(
            cfg.nvme_device, layout_id=cfg.layout, object_bytes=layout.object_bytes, persistence=cfg.persistence,
            allow_regular=env.test_mode, expect_claim_for=expected_claim)
        sb = self.store.superblock
        floor = -(-sb.num_slots * cfg.free_floor_pct // 100)
        reserve = -(-cfg.reserve_bytes // sb.object_bytes) if cfg.reserve_bytes else 0
        need = store_module.min_session_objects(cfg.layout)
        if sb.num_slots - floor - reserve < need:
            raise store_module.CapacityShort('the drive holds %d objects; after the %d%% floor and the reserve %d are left, one maximum-context session needs %d'
                                             % (sb.num_slots, cfg.free_floor_pct, sb.num_slots - floor - reserve, need))
        self.p2pmem_paths = health.check_p2pmem(cfg, layout.chips, env)
        # all checks passed: size the allocator. Every slot is queued for TRIM: that is the open-time discard of the data region.
        self.allocator.reset(sb.num_slots, all_need_trim=True)

    def _lenient_status(self) -> Optional[health.StatusFacts]:
        """Test mode with require_sed false: the status file is optional; read it if there is one."""
        try:
            return health.read_status(self.cfg.nvme_status_dir, self.env)
        except HostRefusal:
            return None

    # --- the frozen interface
    @property
    def drive_path(self) -> str:
        return self.cfg.nvme_device

    def latch(self, reason: str, code: Optional[str] = None) -> None:
        """Turn the tier off until the engine restarts (or a control ``on`` clears a control ``off``). Idempotent: the first reason is kept."""
        self._set_latched(reason, code or health.latch_code(reason))

    def _set_latched(self, reason: str, code: str) -> None:
        if self.state == STATE_CLOSING:
            return
        if self.state != STATE_LATCHED:
            self._pre_latch_state = self.state
            self.state = STATE_LATCHED
            self.latch_reason = reason
            self.latch_code = code
            self.latched_total[code] += 1
            logger.error('ttkvp2p: LATCHED (%s): %s', code, reason)

    def set_available(self) -> None:
        """The worker's last start-up step. Only a runtime whose host checks passed goes AVAILABLE; a latched one stays latched."""
        if self.state == STATE_UNAVAILABLE and self.host_ok:
            self.state = STATE_AVAILABLE
            self._device_ready = True
        elif self.state == STATE_LATCHED and self.host_ok:
            self._device_ready = True
            self._pre_latch_state = STATE_AVAILABLE

    def mark_bad(self, slot: int, generation: int) -> None:
        """Keys go BAD only on proof (I13). The manager sees the pair in ``bad`` and removes the entry when its last load completes."""
        self.bad.add((int(slot), int(generation)))
        self.counters['bad_marked'] += 1
        manager = self._manager
        if manager is not None:
            manager.on_slot_bad(int(slot), int(generation))

    def slot_offset(self, slot: int) -> int:
        if self.store is None:
            raise RuntimeError('the drive is not open')
        return self.store.slot_offset(int(slot))

    @property
    def canary_offset(self) -> int:
        return store_module.CANARY_OFF

    @property
    def canary_bytes(self) -> int:
        return store_module.CANARY_BYTES

    @property
    def drive(self) -> Optional[store_module.DriveFile]:
        return self.store.drive if self.store is not None else None

    # --- telemetry hooks the worker uses
    def count(self, name: str, n: int = 1) -> None:
        self.counters[name] += n

    def observe_restore(self, seconds: float) -> None:
        self.restore_seconds.append(float(seconds))

    def take_fault(self, kind: str) -> bool:
        """A test-only injected fault (control verb ``fault``): consumed one at a time. Always False outside test mode."""
        if self.faults[kind] > 0 and self.env.test_mode:
            self.faults[kind] -= 1
            return True
        return False

    def pop_selftest(self) -> bool:
        """True once per control ``selftest`` (test mode); the connector runs the round-trip self-test when it sees it."""
        if self.selftest_requests > 0:
            self.selftest_requests -= 1
            return True
        return False

    # --- the manager
    def attach_manager(self, manager) -> None:
        self._manager = manager

    def detach_manager(self, manager) -> None:
        if self._manager is manager:
            self._manager = None

    # --- the engine thread's pump
    def heartbeat(self, now: Optional[float] = None) -> None:
        """The time of the last ``schedule()`` (the decode-stall alert reads it from the state file)."""
        self.last_schedule_wall = self.env.wall() if now is None else now

    def service(self, now: Optional[float] = None) -> None:
        """Apply completed drive I/O, issue the next TRIMs, poll the control directory and the status file, write the state file.
        Called by the manager from ``on_schedule_end`` and by tests; cheap when idle; never raises."""
        try:
            now = self.env.wall() if now is None else now
            self._drain_done()
            self._pump_trims()
            self.poll_control(now)
            self._poll_status(now)
            self.write_state(now)
        except Exception:
            logger.exception('ttkvp2p: the runtime service pass failed')

    def _drain_done(self) -> None:
        while True:
            try:
                token, error = self._done.get_nowait()
            except queue.Empty:
                return
            kind = token[0]
            if kind == 'trim':
                _kind, first, count = token
                self._trims_out -= 1
                ok = error is None
                if not ok:
                    self.counters['trim_failed'] += count
                    logger.error('ttkvp2p: TRIM of slots %d..%d failed: %s: %s; the slots are quarantined', first, first + count - 1,
                                 type(error).__name__, error)
                    if getattr(error, 'errno', None) in (19, 6):        # ENODEV, ENXIO: the drive is gone
                        self.latch('device lost: TRIM failed with %s' % (error,), 'device_lost')
                    elif self.allocator.trim_failed >= max(3, getattr(self.cfg, 'strikes_to_latch', 3)):
                        self.latch('%d slots failed their TRIM' % self.allocator.trim_failed, 'strikes')
                else:
                    self.counters['trim_runs'] += 1
                    self.counters['trim_slots'] += count
                self.allocator.trim_done(first, count, ok)

    def _pump_trims(self) -> None:
        if self.store is None or self.closed or self.state == STATE_CLOSING:
            return
        per_run = max(1, store_module.DISCARD_RUN_BYTES // self.store.object_bytes)
        while self._trims_out < TRIMS_IN_FLIGHT:
            runs = self.allocator.take_trim_runs(1, per_run)
            if not runs:
                return
            first, count = runs[0]
            drive, offset, length = self.store.drive, self.store.slot_offset(first), count * self.store.object_bytes
            self._trims_out += 1
            self._io.submit(('trim', first, count), lambda drive=drive, offset=offset, length=length: drive.discard(offset, length))

    def settle(self, timeout: float = 10.0, *, poll: float = 0.002) -> bool:
        """Run ``service`` until no TRIM is waiting or in flight (the open-time discard done). Tests and boot logs use it; the engine does not."""
        deadline = self.env.monotonic() + timeout
        while True:
            self._drain_done()
            self._pump_trims()
            if self.allocator.trim_backlog() == 0 and self._trims_out == 0:
                return True
            if self.env.monotonic() >= deadline:
                return False
            time.sleep(poll)

    # --- control directory (3.6)
    def poll_control(self, now: float) -> None:
        if now < self._next_control:
            return
        self._next_control = now + CONTROL_POLL_S
        commands = health.list_control(self.cfg.control_dir)
        off = health.switch_is_off(commands)
        if off and self.state != STATE_LATCHED and self.state != STATE_CLOSING:
            self._set_latched('control off', 'control_off')
            self._ack_level(commands, 'off', 'latched')
        elif not off and self.state == STATE_LATCHED and self.latch_code == 'control_off':
            self.state = self._pre_latch_state if self._device_ready else STATE_UNAVAILABLE
            self.latch_reason = self.latch_code = None
            self._ack_level(commands, 'on', 'cleared' if self._device_ready else 'cleared (the engine must restart to start the tier)')
        for command in commands:
            if command.verb in health.EDGE_VERBS and command.seq > self._edge_seq:
                self._edge_seq = command.seq
                self._apply_edge(command)

    def _ack_level(self, commands, verb: str, result: str) -> None:
        last = [c for c in commands if c.verb in health.LEVEL_VERBS]
        seq = last[-1].seq if last else -1
        if seq != self._level_seq_acked:
            self._level_seq_acked = seq
            self._acks.append({'seq': seq, 'verb': verb, 'result': result})

    def _apply_edge(self, command) -> None:
        verb, result = command.verb, 'ok'
        if verb in health.TEST_ONLY_VERBS and not self.env.test_mode:
            result = 'refused: test-only (needs THATCH_KVP2P_TEST=1)'
        elif verb == 'wipe':
            self.wipe()
        elif verb == 'delete_tenant':
            result = 'unsupported: tenant delete is v1b'
        elif verb == 'fault':
            parsed = health.parse_fault(command.body)
            if parsed is None:
                result = 'refused: bad fault'
            else:
                self.faults[parsed[0]] += parsed[1]
        elif verb == 'selftest':
            self.selftest_requests += 1
        self._acks.append({'seq': command.seq, 'verb': verb, 'result': result})
        self.counters['control_' + verb] += 1

    def wipe(self) -> None:
        """Control ``wipe``: invalidate the index, bump the epoch, and send every slot back through TRIM (the whole data region)."""
        manager = self._manager
        if manager is not None:
            manager.invalidate_all('wipe')
        self.allocator.invalidate_all()
        self.store_results.clear()
        self.bad.clear()
        self.counters['wipes'] += 1

    # --- the status file (AER and liveness while running)
    def _poll_status(self, now: float) -> None:
        if now < self._next_status or self.state in (STATE_CLOSING,) or not self.host_ok:
            return
        self._next_status = now + STATUS_POLL_S
        try:
            facts = health.read_status(self.cfg.nvme_status_dir, self.env)
        except HostRefusal:
            return                                  # an absent file is the monitor's alert, not a latch
        self.status = facts
        self._drive_state = facts.state
        if facts.state != 'ok' and self.state == STATE_AVAILABLE:
            self.latch('device lost: the cache drive reports %r' % (facts.state,), 'device_lost')
        baseline = self._aer_baseline or 0
        if self.cfg.aer_latch == 'uncorrectable' and facts.aer_uncorrectable > baseline and self.state != STATE_LATCHED:
            self.latch('uncorrectable AER on the drive path (%d -> %d)' % (baseline, facts.aer_uncorrectable), 'aer_uncorrectable')

    # --- the state file (3.6)
    def snapshot(self, now: float) -> Dict[str, Any]:
        sb = self.store.superblock if self.store is not None else None
        manager = self._manager
        manager_stats = manager.snapshot() if manager is not None else {}
        status = self.status
        restore = list(self.restore_seconds)
        counters = dict(self.counters)
        failures = {k[len('fail_'):]: v for k, v in counters.items() if k.startswith('fail_')}
        out: Dict[str, Any] = {
            'schema': health.STATE_SCHEMA,
            'spawn_id': self.env.spawn_id,
            'written_at': health.iso_utc(now),
            'state': self.state,
            'latch_reason': self.latch_reason,
            'latch_code': self.latch_code,
            'host_ok': self.host_ok,
            'host_reason': None if self.host_ok else self.host_reason,
            'persistence': self.cfg.persistence,
            'layout': self.cfg.layout,
            'kv_epoch': self.cfg.kv_epoch,
            'engine_heartbeat': self.last_schedule_wall,
            'counters': {
                'hits': manager_stats.get('hits', 0), 'misses': manager_stats.get('misses', 0),
                'lookups': manager_stats.get('lookups', 0),
                'loads': counters.get('done_load', 0), 'stores': counters.get('done_store', 0),
                'store_admitted_objects': manager_stats.get('admitted_objects', 0),
                'store_dropped': manager_stats.get('dropped', {}),
                'evictions': manager_stats.get('evictions', {}),
                'failures': failures,
                'latched': dict(self.latched_total),
                'trim_runs': counters.get('trim_runs', 0), 'trim_failed': counters.get('trim_failed', 0),
                'bad_marked': counters.get('bad_marked', 0),
            },
            'bytes_written_today': manager_stats.get('bytes_written_today', 0),
            'write_cap_bytes': self.cfg.daily_write_cap_bytes,
            'capacity': {
                'slots': self.allocator.num_slots,
                'object_bytes': sb.object_bytes if sb else 0,
                'entries': manager_stats.get('entries', 0),
                'capacity_slots': manager_stats.get('capacity_slots', 0),
                'slot_states': self.allocator.stats(),
            },
            'restore': {'count': len(restore), 'p50_s': health.quantile(restore, 0.5), 'p99_s': health.quantile(restore, 0.99)},
            'drive': {
                'state': self._drive_state if status is None else status.state,
                'link_gts': None if status is None else status.link_gts,
                'aer_uncorrectable': None if status is None else status.aer_uncorrectable,
                'aer_correctable': None if status is None else status.aer_correctable,
                'status_age_s': None if status is None or status.timestamp is None else round(now - status.timestamp, 1),
            },
            'control': {'acks': list(self._acks), 'last_edge_seq': self._edge_seq},
        }
        return out

    def write_state(self, now: float, *, force: bool = False) -> None:
        """Rewrite the state file (every 10 s from the engine's pass). Never raises: telemetry must not stop the tier."""
        if not force and now < self._next_state:
            return
        self._next_state = now + STATE_WRITE_S
        path = self.cfg.state_file
        if not path:
            return
        try:
            health.write_json_atomic(path, self.snapshot(now))
        except Exception:
            logger.warning('ttkvp2p: cannot write the state file', exc_info=True)

    # --- shutdown (design 2.9): reference counted
    def release(self, role: str) -> None:
        with self._guard:
            if role in self.roles:
                self.roles.remove(role)
            else:
                logger.warning('ttkvp2p: release of role %r that is not held', role)
            last = not self.roles
        if last:
            self._shutdown()

    def _shutdown(self) -> None:
        if self.closed:
            return
        self.closed = True
        self.state = STATE_CLOSING
        drain = float(getattr(self.cfg, 'shutdown_drain_s', 15.0))
        self._drain_done()
        stopped = self._io.close(min(drain, 5.0))
        self._drain_done()
        self.write_state(self.env.wall(), force=True)
        if self.store is not None:
            if self.allocator.total_inflight() == 0 and stopped:
                self.store.close()
            else:
                logger.error('ttkvp2p: leaving the drive open: %d bios in flight, TRIM thread %s',
                             self.allocator.total_inflight(), 'stopped' if stopped else 'still running')
            self.store = None
        with self._guard:
            if self._instances.get(self.cfg.nvme_device) is self:
                del self._instances[self.cfg.nvme_device]
