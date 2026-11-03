"""P2PIO: the drive side of the peer-to-peer path, both directions (design 2.8; INTERFACE 3.2).

Adapted from the Qwen ``P2PEngine`` / ``Stream`` (``qwen_p2p_restore.py``): a pool of I/O threads that issue O_DIRECT reads of the
drive straight into the cards' p2pmem mappings. The Gemma version adds the write direction (``pwritev`` from p2pmem to the drive:
the spill) and, because ONE TLB window per card serves all eight staging banks, a scheduler that decides which bank the window is
aimed at.

  * One **object** is one KV block of one group: ``[shard 0][shard 1][shard 2][shard 3]`` on the drive, each shard
    ``shard_object_bytes`` (696,320) long and in a different card's staging bank. One ``preadv`` / ``pwritev`` covers one contiguous
    run of objects, and its iovecs are, per object, **shard 0..3 in order** (the cards' ``slot_view`` buffers), so the drive offset
    of every byte follows from the iovec order alone.
  * An **aim epoch** points window 202 of all four cards at the same bank. Calls for the bank of the current epoch run (up to the
    queue depth); a call for another bank waits until every call in flight has returned (**re-aim only when idle**), then all four
    cards are re-aimed (``CardLink.aim``: a canary read-back on every re-aim) and the epoch advances. A canary mismatch kills the I/O
    layer and latches the tier.
  * **Restore first.** Spill (``write_subrun``, at most ``spill_subrun_objects`` objects, so the drain is bounded) and restore
    (``read_piece``) have separate queues. While a restore is queued no new spill call starts; calls in flight drain, then the epoch
    moves to the restore bank. Spill banks and restore banks are disjoint, so a late restore DMA can never land in a spill slot.
  * **Hard bounds.** Before every syscall each iovec is checked against the plan: the card is the shard's mapped card, the buffer
    sits exactly where ``CardLink.expected_slice_offset(bank, object)`` says (inside the slot, never the canary page), the window of
    every card is aimed at the call's bank in the call's epoch, and the drive range lies inside the data region. A violation is an
    assertion (``BoundsViolation``): no syscall is issued, the I/O layer stops, and the tier is latched.
  * Short reads and writes are continued; an EOF before the end of a call is an error. The drive's ``errno`` is kept on the future's
    exception (EIO / EILSEQ are what the worker counts as media errors).

``read_piece`` and ``write_subrun`` return a ``Future[int]`` (bytes moved, which is ``n_obj * object_bytes`` or the future carries an
exception). A future resolves only when **no call of its request is still in flight** (the worker releases the bank and the slots'
in-flight counts when it sees the future done). They never touch the device command queue. Importing this module imports neither ttnn
nor the extension and starts nothing (invariant I14). Stdlib only; Python 3.10.

**What it asks of the runtime** (the names WP3a's ``KVP2PRuntime`` has; INTERFACE.md, amendment A2): ``runtime.drive.fd`` (the open
O_DIRECT descriptor of the cache drive, which the runtime opened exclusively), ``runtime.store.data_off`` / ``num_slots`` /
``object_bytes`` (the data region, which is the fence of every request), optionally ``runtime.canary_offset``, and ``runtime.latch(reason)``.
It never opens the device itself, never guesses a data region (a runtime that states none is refused and the tier stays UNAVAILABLE), and refuses a
descriptor that is not O_DIRECT (buffered I/O would make the CPU copy the cards' BAR pages).

**Threads (I7).** The layer's own threads (``kvp2p-sched``, ``kvp2p-io_N``) never call into the runtime. A fatal fault found there stops the layer
and is recorded in ``dead`` (a reason that begins ``P2PIO bounds violation``, which the runtime files under that latch code); the worker latches the
tier from the engine thread at its next ``service``. A fault found on the caller's thread (a bad request, the start-up DMA canary) is latched here, on
that thread.

Additions to the frozen INTERFACE.md text, all backwards compatible (INTERFACE.md, amendment A2): the optional keyword ``first_obj`` of
``read_piece`` / ``write_subrun`` (the WP4 worker already passes it), and ``dma_canary()``, ``close()``, ``stats()``, ``wait_idle()`` and ``dead``.
"""

import collections
import concurrent.futures
import errno
import fcntl
import hashlib
import logging
import operator
import os
import threading
import time
from typing import Any, Callable, Deque, Dict, List, Optional, Sequence, Tuple

from ttkvp2p.cardlink import (
    PAGE,
    BoundsViolation,
    CanaryMismatch,
    CardLinkError,
    as_index,
    normalise_bdf,
)

logger = logging.getLogger(__name__)

DMA_CANARY_OFFSET = 2 * PAGE          # the drive's DMA canary block (INTERFACE 3.4: LBA 2-3, 8 KiB)
DMA_CANARY_BYTES = 2 * PAGE
DEFAULT_DRAIN_S = 15.0                # design 2.9: the drain waits 15 s or less for in-flight bios
FATAL_PREFIX = 'P2PIO bounds violation'   # begins every reason of a fatal fault found by this layer; the runtime's latch classifier files it as bounds_violation


def _iov_max() -> int:
    try:
        value = os.sysconf('SC_IOV_MAX')
        return int(value) if value and value > 0 else 1024
    except (ValueError, OSError, AttributeError):
        return 1024


IOV_MAX = _iov_max()


# --------------------------------------------------------------------------------------------------------------------
# Errors
# --------------------------------------------------------------------------------------------------------------------

class P2PIOError(RuntimeError):
    """Base of what this module raises on purpose."""


class P2PClosed(P2PIOError):
    """The request was refused or failed because the I/O layer is closing."""


class P2PDead(P2PIOError):
    """The I/O layer stopped itself after a canary or bounds violation (the tier is latched); nothing more is issued, not even a call that was
    already decided."""


class DmaCanaryError(P2PIOError):
    """The start-up DMA canary did not round-trip: the drive's DMA does not land where the window points (design 2.9 step 3)."""


class P2PShortIO(OSError):
    """The drive returned 0 bytes before the call was complete (EOF). Deliberately has no errno: it is not a media error."""


# --------------------------------------------------------------------------------------------------------------------
# The drive, behind one small object
# --------------------------------------------------------------------------------------------------------------------

class SystemDriveOps(object):
    """The real syscalls. The tests wrap this to count calls, hold them in flight and inject short I/O and errors."""

    def preadv(self, fd: int, buffers: Sequence[Any], offset: int) -> int:
        return os.preadv(fd, buffers, offset)

    def pwritev(self, fd: int, buffers: Sequence[Any], offset: int) -> int:
        return os.pwritev(fd, buffers, offset)


def fd_is_direct(fd: int) -> bool:
    """Whether the descriptor is open with O_DIRECT (False for a bad descriptor and where the platform has no O_DIRECT)."""
    flag = getattr(os, 'O_DIRECT', 0)
    if not flag:
        return False
    try:
        return bool(fcntl.fcntl(fd, fcntl.F_GETFL) & flag)
    except (OSError, ValueError, OverflowError):
        return False


def _index(value: Any) -> Optional[int]:
    """``value`` as an int if it is one (anything with ``__index__``, never a bool), else None."""
    if isinstance(value, bool):
        return None
    try:
        return operator.index(value)
    except TypeError:
        return None


def vector_io(function: Callable[[int, Sequence[Any], int], int], fd: int, views: Sequence[Any], offset: int, total: int) -> int:
    """One logical vectored transfer of ``total`` bytes at ``offset``, continued after every short count. Fully consumed iovecs are dropped
    and the partly consumed one is sliced (the slices are released here). A zero count is EOF (``P2PShortIO``); a count beyond what was asked is
    a bug of the layer below (``P2PIOError``). Returns ``total``."""
    done = 0
    pending = list(views)
    temporaries: List[memoryview] = []
    try:
        while True:
            count = function(fd, pending, offset + done)
            if count <= 0:
                raise P2PShortIO('the drive returned %d bytes at offset %d with %d of %d bytes done' % (count, offset + done, done, total))
            if count > total - done:
                raise P2PIOError('the drive reported %d bytes for a call with %d left' % (count, total - done))
            done += count
            if done == total:
                return done
            skip = count
            rest: List[Any] = []
            for view in pending:
                size = len(view) if not isinstance(view, memoryview) else view.nbytes
                if skip >= size:
                    skip -= size
                    continue
                if skip:
                    view = view[skip:]
                    temporaries.append(view)
                rest.append(view)
                skip = 0
            pending = rest
    finally:
        for piece in temporaries:
            piece.release()


# --------------------------------------------------------------------------------------------------------------------
# Requests and calls
# --------------------------------------------------------------------------------------------------------------------

class _Call(object):
    """One preadv / pwritev: a contiguous run of objects of one request."""

    __slots__ = ('req', 'first_obj', 'n', 'drive_off', 'tokens')

    def __init__(self, req: '_Request', first_obj: int, n: int, drive_off: int):
        self.req, self.first_obj, self.n, self.drive_off = req, first_obj, n, drive_off
        self.tokens: Tuple[int, ...] = ()

    @property
    def kind(self) -> str:
        return self.req.kind

    @property
    def bank(self) -> int:
        return self.req.bank


class _Request(object):
    """What ``read_piece`` / ``write_subrun`` was asked for, split into calls; the future resolves when every issued call has returned."""

    __slots__ = ('kind', 'bank', 'calls', 'next_call', 'outstanding', 'nbytes', 'error', 'future', 't_enqueue', 'resolved', 'behind_spill')

    def __init__(self, kind: str, bank: int, now: float):
        self.kind, self.bank = kind, bank
        self.calls: List[_Call] = []
        self.next_call = 0
        self.outstanding = 0
        self.nbytes = 0
        self.error: Optional[BaseException] = None
        self.future: 'concurrent.futures.Future[int]' = concurrent.futures.Future()
        self.future.set_running_or_notify_cancel()          # nobody can cancel it: the DMA of a started call cannot be taken back
        self.t_enqueue = now
        self.resolved = False
        self.behind_spill = False                           # a restore that found spill calls in flight when it arrived

    @property
    def issued_all(self) -> bool:
        return self.next_call >= len(self.calls)


# --------------------------------------------------------------------------------------------------------------------
# P2PIO
# --------------------------------------------------------------------------------------------------------------------

class P2PIO(object):
    def __init__(self, links: Sequence[Any], runtime: Any, config: Any, staging_info: dict, *,
                 drive_fd: Optional[int] = None, drive_ops: Any = None, drive_range: Optional[Tuple[int, int]] = None,
                 restore_run_objects: Optional[int] = None, canary_offset: Optional[int] = None, allow_buffered: bool = False,
                 clock: Callable[[], float] = time.monotonic, nonce: Optional[bytes] = None):
        """``links`` are the four opened ``CardLink`` s (any order: the shard map comes from each link's ``shard``), ``runtime`` is the process-wide runtime (see the
        module docstring for what is read from it), ``config`` is the tier's ``P2PConfig`` (read by attribute name with the design's defaults), ``staging_info`` is
        ``qwen_p2p_staging_info``.

        The drive descriptor is ``drive_fd`` if given, else ``runtime.drive.fd``; there is no other source, and it must be open O_DIRECT unless ``allow_buffered`` (a
        test on a file system without O_DIRECT; the production wiring never passes it). ``drive_range`` is the byte range of the drive that may be read or written
        as DATA: ``drive_range`` if given, else ``[runtime.store.data_off, + num_slots * object_bytes)``; a runtime that states neither is refused. The superblock and
        the DMA canary block are never data. ``canary_offset`` is where the start-up DMA canary reads and writes (default ``runtime.canary_offset``, else the frozen
        8192). ``restore_run_objects`` caps the objects of one ``preadv`` (default: the whole run, as the design says; the W1 sweeps may lower it)."""
        self._links_given = list(links)
        self._runtime = runtime
        self._cfg = config
        self._clock = clock
        self._nonce = nonce if nonce is not None else os.urandom(16)
        self._drive = drive_ops or SystemDriveOps()
        self._fd: Optional[int] = None
        self._allow_buffered = bool(allow_buffered)
        self._own_threads: set = set()          # idents of the layer's own threads (the scheduler and the pool): they never call into the runtime
        self._pool: Optional[concurrent.futures.ThreadPoolExecutor] = None
        self._thread: Optional[threading.Thread] = None
        self._cv = threading.Condition()
        # state, all guarded by self._cv
        self._restore_q: Deque[_Request] = collections.deque()
        self._spill_q: Deque[_Request] = collections.deque()
        self._inflight_calls = 0
        self._inflight_objects = 0
        self._epoch_bank: Optional[int] = None
        self._epoch = 0
        self._tokens: Tuple[int, ...] = ()
        self._closing = False
        self._closed = False
        self._stop = False
        self._exclusive = False
        self._dead: Optional[str] = None
        self._to_resolve: List[_Request] = []
        self._drained = True
        self._stats: Dict[str, Any] = dict(
            aims=0, aim_s_max=0.0, aim_s_total=0.0, calls_read=0, calls_write=0, objects_read=0, objects_written=0, bytes_read=0, bytes_written=0,
            short_ios=0, io_errors=0, bounds_violations=0, canary_mismatches=0, restore_wait_s_max=0.0, restore_behind_spill=0, restore_behind_spill_wait_s_max=0.0,
            spill_held_for_restore=0, inflight_calls_max=0, io_thread_cpu_s=0.0, call_s_max=0.0, dead=None)
        try:
            self._setup(staging_info, drive_fd, drive_range, restore_run_objects, canary_offset)
        except BaseException:
            self._teardown_failed_start()
            raise

    # -- construction ----------------------------------------------------------------------------------------------
    def _setting(self, name: str, default: Any) -> Any:
        value = getattr(self._cfg, name, None)
        return default if value is None else value

    def _setup(self, staging_info: dict, drive_fd: Optional[int], drive_range: Optional[Tuple[int, int]],
               restore_run_objects: Optional[int], canary_offset: Optional[int]) -> None:
        links = self._links_given
        if not links:
            raise P2PIOError('no card links')
        self.chips = len(links)
        for link in links:
            if not getattr(link, 'proved', False):
                raise P2PIOError('the link to %s has not proved every bank canary: no data may move yet' % link.bdf)
        shards = [getattr(link, 'shard', None) for link in links]
        if sorted(s for s in shards if s is not None) != list(range(self.chips)) or None in shards:
            raise P2PIOError('the links\' shards are %s, they must be exactly 0..%d' % (shards, self.chips - 1))
        self._by_shard = sorted(links, key=lambda link: link.shard)
        # the shard map against the staging record (design 2.2: shard d is mesh coordinate d)
        devices = list((staging_info or {}).get('devices') or ())
        if devices:
            if len(devices) != self.chips:
                raise P2PIOError('the staging record lists %d devices, there are %d links' % (len(devices), self.chips))
            for d, entry in enumerate(devices):
                bus = entry.get('pci_bus_id')
                if bus is None:
                    continue
                try:
                    same = normalise_bdf(bus) == self._by_shard[d].bdf
                except ValueError:
                    same = False
                if not same:
                    raise P2PIOError('shard %d is %s in the staging record, the link for shard %d is %s' % (d, bus, d, self._by_shard[d].bdf))
        sizes = {link.shard_object_bytes for link in self._by_shard}
        if len(sizes) != 1:
            raise P2PIOError('the links disagree about the size of an object on one card: %s' % sorted(sizes))
        self.shard_object_bytes = sizes.pop()
        self.object_bytes = self.chips * self.shard_object_bytes
        self.slot_objects = min([int(self._setting('slot_objects', 128))] + [int(link.slot_objects) for link in self._by_shard])
        if self.slot_objects < 1:
            raise P2PIOError('a staging slot holds no object')
        self.spill_banks = tuple(int(b) for b in self._setting('spill_banks', (0, 1)))
        self.restore_banks = tuple(int(b) for b in self._setting('restore_banks', (2, 3, 4, 5, 6, 7)))
        if not self.spill_banks or not self.restore_banks or set(self.spill_banks) & set(self.restore_banks):
            raise P2PIOError('spill banks %s and restore banks %s must both be non-empty and disjoint' % (self.spill_banks, self.restore_banks))
        for link in self._by_shard:
            missing = (set(self.spill_banks) | set(self.restore_banks)) - set(link.banks)
            if missing:
                raise P2PIOError('the link to %s has no bank(s) %s' % (link.bdf, sorted(missing)))
        self.spill_subrun_objects = int(self._setting('spill_subrun_objects', 8))
        self._max_call_objects = max(1, IOV_MAX // self.chips)
        if self.spill_subrun_objects < 1 or self.spill_subrun_objects > self._max_call_objects:
            raise P2PIOError('spill_subrun_objects %d must be 1..%d (IOV_MAX %d over %d cards)' % (self.spill_subrun_objects, self._max_call_objects, IOV_MAX, self.chips))
        self.restore_run_objects = min(int(restore_run_objects), self._max_call_objects) if restore_run_objects else self._max_call_objects
        if self.restore_run_objects < 1:
            raise P2PIOError('restore_run_objects must be at least 1')
        self._threads = max(1, int(self._setting('io_threads', 8)))
        self._qd_objects = max(1, int(self._setting('io_queue_depth_objects', 8)))
        self._drain_s = float(self._setting('shutdown_drain_s', DEFAULT_DRAIN_S))
        # the drive
        self._resolve_drive(drive_fd)
        self._drive_range = self._resolve_range(drive_range)
        self._canary_offset = self._resolve_canary_offset(canary_offset)
        # threads last: nothing can fail after this
        self._pool = concurrent.futures.ThreadPoolExecutor(max_workers=self._threads, thread_name_prefix='kvp2p-io', initializer=self._mark_own_thread)
        self._thread = threading.Thread(target=self._scheduler, name='kvp2p-sched', daemon=True)
        self._thread.start()

    def _resolve_drive(self, drive_fd: Optional[int]) -> None:
        if drive_fd is None:
            drive_fd = getattr(getattr(self._runtime, 'drive', None), 'fd', None)
        fd = _index(drive_fd)
        if fd is None or fd < 0:
            raise P2PIOError('no drive: the runtime holds no open drive (runtime.drive.fd) and no drive_fd was passed; this layer never opens the device itself')
        if not self._allow_buffered and not fd_is_direct(fd):
            raise P2PIOError('the drive descriptor %d is not open O_DIRECT: buffered I/O would make the CPU copy the cards\' BAR pages, so it is refused' % fd)
        self._fd = fd

    def _resolve_range(self, drive_range: Optional[Tuple[int, int]]) -> Tuple[int, int]:
        if drive_range is None:
            store = getattr(self._runtime, 'store', None)
            data_off, slots, stride = (_index(getattr(store, name, None)) for name in ('data_off', 'num_slots', 'object_bytes'))
            if data_off is None or slots is None or slots <= 0:
                raise P2PIOError('the runtime states no data region (runtime.store.data_off / num_slots): there is no fence to enforce, and this layer does not guess one')
            if stride is not None and stride != self.object_bytes:
                raise P2PIOError('the runtime\'s objects are %d bytes, the cards\' are %d (%d cards x %d)' % (stride, self.object_bytes, self.chips, self.shard_object_bytes))
            drive_range = (data_off, data_off + slots * self.object_bytes)
        lo, hi = int(drive_range[0]), int(drive_range[1])
        if lo < 0 or hi <= lo or lo % PAGE or hi % PAGE:
            raise P2PIOError('the data range [%d, %d) must be non-empty and 4 KiB aligned' % (lo, hi))
        if lo < DMA_CANARY_OFFSET + DMA_CANARY_BYTES:
            raise P2PIOError('the data range starts at %d, inside the superblock / DMA canary block (below %d)' % (lo, DMA_CANARY_OFFSET + DMA_CANARY_BYTES))
        return lo, hi

    def _resolve_canary_offset(self, canary_offset: Optional[int]) -> int:
        """An explicit offset is taken as given (``dma_canary`` checks it against the block before every use); the runtime's must be inside the frozen block now."""
        if canary_offset is not None:
            return int(canary_offset)
        offered = getattr(self._runtime, 'canary_offset', None)
        if offered is None:
            return DMA_CANARY_OFFSET
        offset = _index(offered)
        if offset is None or offset < DMA_CANARY_OFFSET or offset + PAGE > DMA_CANARY_OFFSET + DMA_CANARY_BYTES or offset % PAGE:
            raise P2PIOError('the runtime puts the DMA canary at %r, outside the canary block [%d, +%d)' % (offered, DMA_CANARY_OFFSET, DMA_CANARY_BYTES))
        return offset

    def _teardown_failed_start(self) -> None:
        if self._pool is not None:
            self._pool.shutdown(wait=False)
        self._fd = None

    # -- the public surface ----------------------------------------------------------------------------------------
    def read_piece(self, drive_off: int, bank: int, n_obj: int, first_obj: int = 0) -> 'concurrent.futures.Future[int]':
        """Restore: read ``n_obj`` consecutive objects of the drive from ``drive_off`` into objects ``first_obj`` .. of ``bank``'s slot on the four cards. ``bank`` must be a
        restore bank. The future's result is the bytes read; a failure is its exception (an ``OSError`` keeps the drive's errno)."""
        return self._submit('read', drive_off, bank, n_obj, first_obj)

    def write_subrun(self, drive_off: int, bank: int, n_obj: int, first_obj: int = 0) -> 'concurrent.futures.Future[int]':
        """Spill: write objects ``first_obj`` .. of ``bank``'s slot on the four cards to the drive at ``drive_off``. ``bank`` must be a spill bank and ``n_obj`` at most
        ``spill_subrun_objects`` (the drain bound). The future's result is the bytes written."""
        return self._submit('write', drive_off, bank, n_obj, first_obj)

    def _violation(self, message: str) -> BoundsViolation:
        """Count and latch a bounds violation; the caller raises the returned exception."""
        with self._cv:
            self._stats['bounds_violations'] += 1
        logger.error('ttkvp2p: %s: %s', FATAL_PREFIX, message)
        self._die('%s: %s' % (FATAL_PREFIX, message))               # stops issuing, fails what is queued, records why, latches the tier (from a caller's thread)
        return BoundsViolation(message)

    def _mark_own_thread(self) -> None:
        self._own_threads.add(threading.get_ident())

    def _on_own_thread(self) -> bool:
        return threading.get_ident() in self._own_threads

    def _latch(self, reason: str) -> None:
        hook = getattr(self._runtime, 'latch', None)
        if hook is None:
            return
        try:
            hook(reason)
        except Exception:
            logger.exception('ttkvp2p: latching the tier failed')

    def _check_request(self, kind: str, drive_off: Any, bank: Any, n_obj: Any, first_obj: Any) -> Tuple[int, int, int, int]:
        """Every refusal that needs no hardware: the arguments of a call, against the slot, the banks of its direction and the data region. Returns the arguments
        as Python ints."""
        try:
            drive_off, bank, n_obj, first_obj = (as_index(drive_off, 'drive_off'), as_index(bank, 'bank'), as_index(n_obj, 'n_obj'), as_index(first_obj, 'first_obj'))
        except BoundsViolation as error:
            raise self._violation(str(error))
        banks = self.restore_banks if kind == 'read' else self.spill_banks
        if bank not in banks:
            raise self._violation('bank %d is not a %s bank (%s)' % (bank, 'restore' if kind == 'read' else 'spill', list(banks)))
        if n_obj < 1:
            raise self._violation('%d objects' % n_obj)
        if first_obj < 0 or first_obj + n_obj > self.slot_objects:
            raise self._violation('objects [%d, %d) are outside the %d-object slot' % (first_obj, first_obj + n_obj, self.slot_objects))
        if kind == 'write' and n_obj > self.spill_subrun_objects:
            raise self._violation('a spill sub-run of %d objects exceeds spill_subrun_objects %d' % (n_obj, self.spill_subrun_objects))
        self._check_drive_range(drive_off, n_obj * self.object_bytes, kind)
        return drive_off, bank, n_obj, first_obj

    def _check_drive_range(self, drive_off: int, nbytes: int, kind: str) -> None:
        lo, hi = self._drive_range
        if drive_off % PAGE:
            raise self._violation('drive offset %d is not 4 KiB aligned' % drive_off)
        if drive_off < lo or drive_off + nbytes > hi:
            raise self._violation('the %s range [%d, +%d) is outside the data region [%d, %d)' % (kind, drive_off, nbytes, lo, hi))

    def _submit(self, kind: str, drive_off: int, bank: int, n_obj: int, first_obj: int) -> 'concurrent.futures.Future[int]':
        drive_off, bank, n_obj, first_obj = self._check_request(kind, drive_off, bank, n_obj, first_obj)
        per_call = self.restore_run_objects if kind == 'read' else self._max_call_objects
        req = _Request(kind, bank, self._clock())
        done = 0
        while done < n_obj:
            count = min(per_call, n_obj - done)
            req.calls.append(_Call(req, first_obj + done, count, drive_off + done * self.object_bytes))
            done += count
        refusal: Optional[BaseException] = None
        with self._cv:
            if self._closing or self._closed:
                refusal = P2PClosed('the I/O layer is closing')
            elif self._dead:
                refusal = P2PDead(self._dead)
            else:
                if kind == 'read':
                    req.behind_spill = bool(self._inflight_calls and self._epoch_bank in self.spill_banks) or bool(self._spill_q)
                    self._stats['spill_held_for_restore'] += len(self._spill_q)
                    self._restore_q.append(req)
                else:
                    self._spill_q.append(req)
                self._cv.notify_all()
        if refusal is not None:
            req.error = refusal
            self._resolve(req)
        return req.future

    # -- the scheduler thread --------------------------------------------------------------------------------------
    def _decide(self) -> Optional[tuple]:
        """With the lock held: the next thing the scheduler thread should do, or None to wait. Restore first; the head of a queue only; re-aim only when idle."""
        if self._exclusive or self._closing or self._dead:
            return None
        queue = self._restore_q or self._spill_q
        if not queue:
            return None
        req = queue[0]
        if req.error is not None:               # a call of it failed: no more calls of this request
            queue.popleft()
            if req.outstanding == 0:
                self._to_resolve.append(req)
            return ('again',)
        if self._epoch_bank != req.bank:
            if self._inflight_calls:
                return None                      # drain first: the window never moves under a bio
            return ('aim', req.bank)
        call = req.calls[req.next_call]
        if self._inflight_calls >= self._threads:
            return None
        if self._inflight_calls and self._inflight_objects + call.n > self._qd_objects:
            return None
        req.next_call += 1
        if req.issued_all:
            queue.popleft()
        req.outstanding += 1
        self._inflight_calls += 1
        self._inflight_objects += call.n
        self._stats['inflight_calls_max'] = max(self._stats['inflight_calls_max'], self._inflight_calls)
        return ('issue', req, call)

    def _scheduler(self) -> None:
        self._mark_own_thread()
        try:
            while True:
                with self._cv:
                    while True:
                        if self._stop:
                            return
                        action = self._decide()
                        pending, self._to_resolve = self._to_resolve, []
                        if action is not None or pending:
                            break
                        self._cv.wait()
                for req in pending:
                    self._resolve(req)
                if action is None or action[0] == 'again':
                    continue
                if action[0] == 'aim':
                    self._do_aim(action[1])
                else:
                    self._do_issue(action[1], action[2])
        except BaseException as error:           # the layer must never die silently with work queued
            logger.exception('ttkvp2p: the P2PIO scheduler thread failed')
            self._die('P2PIO internal error: the I/O scheduler thread failed: %s: %s' % (type(error).__name__, error))

    def _do_aim(self, bank: int) -> None:
        """Every card's window to ``bank`` (no bio in flight: the scheduler only gets here idle), each with its canary read-back."""
        began = self._clock()
        try:
            for link in self._by_shard:
                link.aim(bank)
        except BaseException as error:
            with self._cv:
                if isinstance(error, CanaryMismatch):
                    self._stats['canary_mismatches'] += 1
            self._die('%s: re-aim to bank %d failed: %s: %s' % (FATAL_PREFIX, bank, type(error).__name__, error))
            return
        tokens = tuple(link.aim_epoch for link in self._by_shard)
        took = self._clock() - began
        with self._cv:
            self._epoch_bank = bank
            self._epoch += 1
            self._tokens = tokens
            self._stats['aims'] += 1
            self._stats['aim_s_total'] += took
            self._stats['aim_s_max'] = max(self._stats['aim_s_max'], took)
            self._cv.notify_all()

    def _do_issue(self, req: _Request, call: _Call) -> None:
        """Count the call in flight on every card, at the epoch it was planned for, and hand it to the pool. The layer may have died since the call was decided: it
        is checked again here, before the cards count the call, and again on the pool thread right before the syscall (``_execute``); a dead layer completes the call
        with ``P2PDead`` instead."""
        with self._cv:
            tokens = self._tokens
            dead = self._dead
            if req.kind == 'read' and req.next_call == 1:
                waited = self._clock() - req.t_enqueue
                self._stats['restore_wait_s_max'] = max(self._stats['restore_wait_s_max'], waited)
                if req.behind_spill:
                    self._stats['restore_behind_spill'] += 1
                    self._stats['restore_behind_spill_wait_s_max'] = max(self._stats['restore_behind_spill_wait_s_max'], waited)
        if dead:
            self._complete(call, 0, P2PDead(dead), 0.0, 0.0)
            return
        call.tokens = tokens
        started = 0
        try:
            for d, link in enumerate(self._by_shard):
                link.begin_io(call.bank, tokens[d])
                started += 1
        except BaseException as error:
            for link in self._by_shard[:started]:
                link.end_io()
            self._complete(call, 0, error, 0.0, 0.0)
            return
        try:
            self._pool.submit(self._execute, call)       # type: ignore[union-attr]
        except RuntimeError as error:                    # the pool was shut down under us
            for link in self._by_shard:
                link.end_io()
            self._complete(call, 0, error, 0.0, 0.0)

    # -- one call, on a pool thread --------------------------------------------------------------------------------
    def _build_iovecs(self, call: _Call) -> List[memoryview]:
        """Per object, shard 0..3: the cards' slot views in the order of the bytes on the drive."""
        per_card = [link.slot_view(call.bank, call.first_obj, call.n) for link in self._by_shard]
        return [per_card[d][j] for j in range(call.n) for d in range(self.chips)]

    def _check_iovecs(self, call: _Call, iovs: Sequence[memoryview]) -> None:
        """The hard check, right before the syscall (design 2.8, Bounds). Everything the plan says is compared with what the buffers really are."""
        if len(iovs) != call.n * self.chips or len(iovs) > IOV_MAX:
            raise self._violation('%d iovecs for %d objects over %d cards' % (len(iovs), call.n, self.chips))
        for d, link in enumerate(self._by_shard):
            if link.shard != d:
                raise self._violation('link %d is the card of shard %s' % (d, link.shard))
            if link.aimed_bank != call.bank:
                raise self._violation('the window of shard %d is aimed at bank %s, the call is for bank %d' % (d, link.aimed_bank, call.bank))
            if link.aim_epoch != call.tokens[d]:
                raise self._violation('the window of shard %d is in aim epoch %d, the call was planned in epoch %d' % (d, link.aim_epoch, call.tokens[d]))
        for j in range(call.n):
            for d, link in enumerate(self._by_shard):
                view = iovs[j * self.chips + d]
                if view.nbytes != self.shard_object_bytes:
                    raise self._violation('iovec %d (object %d, shard %d) is %d bytes, not %d' % (j * self.chips + d, call.first_obj + j, d, view.nbytes, self.shard_object_bytes))
                want = link.expected_slice_offset(call.bank, call.first_obj + j)
                got = link.locate(view)
                if got != want:
                    raise self._violation('iovec %d (object %d, shard %d) points at %s in the mapping of card %d, the slot of bank %d has it at %d'
                                          % (j * self.chips + d, call.first_obj + j, d, got, d, call.bank, want))
        self._check_drive_range(call.drive_off, call.n * self.object_bytes, call.kind)

    def _transfer(self, call: _Call, iovs: Sequence[memoryview]) -> int:
        total = call.n * self.object_bytes
        function = self._drive.preadv if call.kind == 'read' else self._drive.pwritev
        return vector_io(function, self._fd, iovs, call.drive_off, total)      # type: ignore[arg-type]

    def _execute(self, call: _Call) -> None:
        iovs: List[memoryview] = []
        error: Optional[BaseException] = None
        nbytes = 0
        began = self._clock()
        cpu_began = time.thread_time()
        try:
            iovs = self._build_iovecs(call)
            self._check_iovecs(call, iovs)
            dead = self._dead
            if dead:                                  # the layer died after this call was decided: nothing more is issued
                raise P2PDead(dead)
            nbytes = self._transfer(call, iovs)
        except BaseException as caught:            # the request records it; the scheduler decides what it means
            error = caught
        finally:
            for view in iovs:
                try:
                    view.release()
                except BufferError:
                    pass
            for link in self._by_shard:
                try:
                    link.end_io()
                except CardLinkError:
                    logger.exception('ttkvp2p: end_io failed')
        self._complete(call, nbytes, error, self._clock() - began, time.thread_time() - cpu_began)

    # -- completion ------------------------------------------------------------------------------------------------
    def _complete(self, call: _Call, nbytes: int, error: Optional[BaseException], seconds: float, cpu_s: float) -> None:
        req = call.req
        fatal = isinstance(error, (BoundsViolation, CardLinkError))
        finished = False
        with self._cv:
            self._inflight_calls -= 1
            self._inflight_objects -= call.n
            stats = self._stats
            stats['io_thread_cpu_s'] += cpu_s
            stats['call_s_max'] = max(stats['call_s_max'], seconds)
            if error is None:
                req.nbytes += nbytes
                if call.kind == 'read':
                    stats['calls_read'] += 1
                    stats['objects_read'] += call.n
                    stats['bytes_read'] += nbytes
                else:
                    stats['calls_write'] += 1
                    stats['objects_written'] += call.n
                    stats['bytes_written'] += nbytes
            else:
                if req.error is None:
                    req.error = error
                stats['io_errors'] += 1
                if isinstance(error, P2PShortIO):
                    stats['short_ios'] += 1
            req.outstanding -= 1
            if req.outstanding == 0 and (req.error is not None or req.issued_all) and not req.resolved:
                finished = True
            self._cv.notify_all()
        if fatal:
            self._die('%s: %s: %s' % (FATAL_PREFIX, type(error).__name__, error))
        if finished:
            self._resolve(req)

    def _resolve(self, req: _Request) -> None:
        """Set the future, once, outside the lock (its callbacks run user code)."""
        with self._cv:
            if req.resolved:
                return
            req.resolved = True
        if req.error is not None:
            req.future.set_exception(req.error)
        else:
            req.future.set_result(req.nbytes)

    def _drain_queues(self, error: BaseException) -> List[_Request]:
        """With the lock held: fail every queued request. A request with calls in flight resolves when the last of them returns."""
        out: List[_Request] = []
        for queue in (self._restore_q, self._spill_q):
            while queue:
                req = queue.popleft()
                if req.error is None:
                    req.error = error
                if req.outstanding == 0:
                    out.append(req)
        return out

    def _die(self, reason: str) -> None:
        """A canary, bounds or scheduler fault: stop issuing, fail what is queued, record why (``dead``). The tier is latched from here only when this runs on a
        caller's thread (a bad request or the start-up canary: the engine thread). On the layer's own threads it only records: the runtime belongs to the engine
        thread (I7), and the worker latches from there when it sees ``dead``."""
        with self._cv:
            if self._dead:
                return
            self._dead = reason
            self._stats['dead'] = reason
            pending = self._drain_queues(P2PDead(reason))
            self._cv.notify_all()
        logger.error('ttkvp2p: the I/O layer stopped: %s', reason)
        if not self._on_own_thread():
            self._latch(reason)
        for req in pending:
            self._resolve(req)

    # -- start-up: the DMA canary (design 2.9 step 3) --------------------------------------------------------------
    def dma_canary(self) -> None:
        """For each card: write a page of its own pattern into the first page of a spill-bank slot through the window, DMA it to the drive's canary block (the drive reads
        the card), aim at a restore bank, zero a page there, DMA the block back into it (the drive writes the card), flush, and compare through the window. It proves both
        directions of the drive's DMA, the bank map and the shard's card, with only a few KiB, before any KV is stored. Needs an idle layer; raises ``DmaCanaryError``."""
        with self._cv:
            if self._closing or self._dead or self._inflight_calls or self._restore_q or self._spill_q or self._exclusive:
                raise P2PIOError('the DMA canary needs an idle I/O layer')
            self._exclusive = True
        try:
            for d, link in enumerate(self._by_shard):
                self._canary_card(d, link)
        except BaseException as error:
            self._die('%s: DMA canary failed: %s: %s' % (FATAL_PREFIX, type(error).__name__, error))
            raise
        finally:
            with self._cv:
                self._exclusive = False
                self._epoch_bank = None            # the windows are somewhere the epoch does not know: the next call re-aims all four
                self._cv.notify_all()

    def _canary_pattern(self, shard: int) -> bytes:
        return hashlib.sha256(b'ttkvp2p-dma|' + self._nonce + b'|%d' % shard).digest() * (PAGE // 32)

    def _canary_io(self, kind: str, link: Any, bank: int) -> None:
        """One page between the drive's canary block and the first page of ``bank``'s slot on one card, with the same plan and bounds checks as a data call."""
        if self._canary_offset < DMA_CANARY_OFFSET or self._canary_offset + PAGE > DMA_CANARY_OFFSET + DMA_CANARY_BYTES or self._canary_offset % PAGE:
            raise self._violation('the DMA canary offset %d is outside the canary block' % self._canary_offset)
        link.begin_io(bank, link.aim_epoch)
        views: List[memoryview] = []
        try:
            whole = link.slot_view(bank, 0, 1)[0]
            views.append(whole)
            page = whole[:PAGE]
            views.append(page)
            if link.locate(page) != link.expected_slice_offset(bank, 0):
                raise self._violation('the canary page buffer is not at object 0 of bank %d' % bank)
            function = self._drive.preadv if kind == 'read' else self._drive.pwritev
            vector_io(function, self._fd, [page], self._canary_offset, PAGE)    # type: ignore[arg-type]
        finally:
            for view in views:
                view.release()
            link.end_io()

    def _canary_card(self, shard: int, link: Any) -> None:
        pattern = self._canary_pattern(shard)
        spill, restore = self.spill_banks[0], self.restore_banks[0]
        link.aim(spill)
        link.scratch_write(spill, pattern)
        link.scratch_read(spill, 8)                                   # a non-posted read after the CPU writes: they have landed before the drive reads
        self._canary_io('write', link, spill)
        link.aim(restore)
        link.scratch_write(restore, bytes(PAGE))
        link.scratch_read(restore, 8)
        self._canary_io('read', link, restore)
        link.flush()
        back = link.scratch_read(restore, PAGE)
        if back != pattern:
            where = next((i for i in range(PAGE) if back[i] != pattern[i]), None)
            raise DmaCanaryError('shard %d (%s): the page DMA-ed from the card to the drive and back into bank %d differs at byte %s (zeros: %s)'
                                 % (shard, link.bdf, restore, where, back == bytes(PAGE)))

    # -- observation and shutdown ----------------------------------------------------------------------------------
    @property
    def epoch(self) -> int:
        return self._epoch

    @property
    def epoch_bank(self) -> Optional[int]:
        return self._epoch_bank

    @property
    def dead(self) -> Optional[str]:
        return self._dead

    def stats(self) -> Dict[str, Any]:
        with self._cv:
            snapshot = dict(self._stats)
            snapshot['queued_restore'] = len(self._restore_q)
            snapshot['queued_spill'] = len(self._spill_q)
            snapshot['inflight_calls'] = self._inflight_calls
            snapshot['epoch'] = self._epoch
            snapshot['epoch_bank'] = self._epoch_bank
            return snapshot

    def wait_idle(self, timeout: float = 5.0) -> bool:
        """True once nothing is queued or in flight (a test and shutdown helper)."""
        end = time.monotonic() + timeout                       # (wall time: the injectable clock is for the statistics, and may stand still in a test)
        with self._cv:
            while self._inflight_calls or self._restore_q or self._spill_q or self._to_resolve:
                left = end - time.monotonic()
                if left <= 0:
                    return False
                self._cv.wait(min(left, 0.05))
            return True

    def close(self, timeout: Optional[float] = None) -> bool:
        """Fail what is queued, wait up to ``timeout`` (``shutdown_drain_s``) for the calls in flight to return and stop the threads (the drive descriptor is the
        runtime's and is left alone). Returns True when every call returned. If not, the cards' links are marked abandoned: a bio may still be in flight, DMA cannot be cancelled, and
        nothing may be unmapped under it (design 2.9)."""
        with self._cv:
            if self._closed:
                return self._drained
            self._closing = True
            pending = self._drain_queues(P2PClosed('the I/O layer is closing'))
            self._cv.notify_all()
        for req in pending:
            self._resolve(req)
        end = time.monotonic() + (self._drain_s if timeout is None else timeout)
        with self._cv:
            while self._inflight_calls:
                left = end - time.monotonic()
                if left <= 0:
                    break
                self._cv.wait(min(left, 0.05))
            drained = self._inflight_calls == 0
            self._drained = drained
            self._closed = True
            self._stop = True
            self._cv.notify_all()
        if self._thread is not None and self._thread is not threading.current_thread():
            self._thread.join(timeout=2.0 if drained else 0.2)
        with self._cv:
            leftover, self._to_resolve = self._to_resolve, []
        for req in leftover:
            self._resolve(req)
        if self._pool is not None:
            self._pool.shutdown(wait=False)
        if not drained:
            logger.error('ttkvp2p: P2PIO closed with %d call(s) still in flight; the card links are left mapped', self._inflight_calls)
            for link in self._by_shard:
                abandon = getattr(link, 'abandon', None)
                if abandon is not None:
                    abandon()
        return drained
