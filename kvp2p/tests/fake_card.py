"""A fake tt-kmd, four fake cards and a fake-able drive, for the CardLink / P2PIO tests. Stdlib only (no numpy, no vLLM, no ttnn).

What is modelled, because the safety of the real path rests on it:

  * Each card has eight DRAM banks. TLB window 202 is aimed at ONE bank at a time (``CONFIGURE_TLB`` with that bank's NOC x, y and
    address 0). The CPU's window mapping and the p2pmem slice are two views of THE SAME bytes: whichever bank the window is aimed at.
    So the p2pmem mapping is one anonymous mapping (the "aperture", real memory, O_DIRECT can DMA into it), and a re-aim saves the
    aperture's bytes into the old bank's store and loads the new bank's store into it. A DMA that runs while the window is somewhere
    else than the test thinks lands in the wrong bank, and the tests can see it.
  * ``ALLOCATE_TLB`` hands out the lowest free 4 GiB window id (202 and up); ``EXPORT_TLB_DMABUF`` pins; the p2pmem file can be
    mapped once (a second mapping fails with ENOMEM, the lease). The lease ends when the MAPPING goes away (munmap: ``close()`` of the
    mapping, which fails while a view of it exists, or its deallocation), not when the file descriptor closes.
  * The CPU reaches the window only through ``ops.words``: aligned 8-byte loads and stores, counted per card (``FakeBoard.word_loads`` /
    ``word_stores``); nothing narrower exists in this fake.
  * Faults: a ``CONFIGURE_TLB`` the card ignores (the window stays where it was), a bank that lands on another bank, a window held by
    another process, a pin that fails, and a sysfs record that disagrees.
  * ``Rig.inflight`` counts the bios the drive wrapper has in flight; every ``CONFIGURE_TLB`` records it, so a re-aim with a bio in
    flight is visible (``Rig.reaim_while_busy``).

``DriveOps`` wraps the real ``os.preadv`` / ``os.pwritev`` on a real (O_DIRECT where the file system allows it) file and adds call logs, holds (to
keep a call in flight), short counts and injected errors.
"""

import errno
import mmap
import os
import struct
import sys
import tempfile
import threading
import time
import types
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from ttkvp2p import cardlink as cl

PAGE = 4096
CLASS_ROOT = '/fake/sys/class/tenstorrent'
PCI_ROOT = '/fake/sys/bus/pci/devices'
ALLOCATE_ROOT = '/fake/run/p2pmem'
BDFS = ('0000:01:00.0', '0000:02:00.0', '0000:03:00.0', '0000:04:00.0')     # synthetic: no real card has these
NOC = {bank: (17 + bank // 4, 12 + 3 * (bank % 4)) for bank in range(8)}   # a bank's translated NOC point (any distinct pairs would do)

SMALL_SHARD = 3 * PAGE        # an object on one card in the small geometry (12,288 B): every size a whole number of pages
SMALL_SLOT = 6
GEMMA_SHARD = cl.GEMMA_SHARD_OBJECT_BYTES      # 696,320


def drive_file(directory: str, size: int) -> Tuple[int, bool]:
    """A zero-filled file of ``size`` bytes opened for O_DIRECT I/O where the file system allows it. Returns (fd, direct)."""
    path = os.path.join(directory, 'drive.img')
    with open(path, 'wb') as handle:
        handle.truncate(size)
    flags = os.O_RDWR | getattr(os, 'O_CLOEXEC', 0)
    direct = hasattr(os, 'O_DIRECT')
    if direct:
        try:
            fd = os.open(path, flags | os.O_DIRECT)
            probe = mmap.mmap(-1, PAGE)
            try:
                os.pwritev(fd, [memoryview(probe)], 0)
            finally:
                probe.close()
            return fd, True
        except OSError:
            try:
                os.close(fd)
            except (OSError, UnboundLocalError):
                pass
    return os.open(path, flags), False


def anonymous_file(name: str) -> int:
    """A descriptor of memory that is not a file anybody can open: ``memfd_create`` where there is one, else an unlinked file on tmpfs, else any unlinked file."""
    if hasattr(os, 'memfd_create'):
        return os.memfd_create(name)
    for directory in ('/dev/shm', None):
        try:
            handle = tempfile.NamedTemporaryFile(dir=directory, prefix=name, delete=True)
            fd = os.dup(handle.fileno())
            handle.close()                 # (deletes the name; the duplicate keeps the file)
            return fd
        except OSError:
            continue
    raise OSError('no place for anonymous shared memory')


class FakeRuntime(object):
    """What P2PIO asks of the runtime. ``latch`` records the reason and the name of the thread that called it. The other members are WP3a's names (``drive.fd``,
    ``store.data_off`` / ``num_slots`` / ``object_bytes``, ``canary_offset``) and are absent unless given (see ``wp3a_runtime``)."""

    def __init__(self, **attributes):
        self.latched: List[str] = []
        self.latch_threads: List[str] = []
        self.__dict__.update(attributes)

    def latch(self, reason: str) -> None:
        self.latched.append(reason)
        self.latch_threads.append(threading.current_thread().name)


def wp3a_runtime(fd: int, data_off: int, num_slots: int, object_bytes: int, canary_offset: Optional[int] = 2 * PAGE, **attributes) -> FakeRuntime:
    """A runtime with the shape of WP3a's ``KVP2PRuntime``: ``drive`` (a DriveFile: ``fd``), ``store`` (``data_off``, ``num_slots``, ``object_bytes``) and ``canary_offset``."""
    members = dict(drive=types.SimpleNamespace(fd=fd), store=types.SimpleNamespace(data_off=data_off, num_slots=num_slots, object_bytes=object_bytes))
    if canary_offset is not None:
        members['canary_offset'] = canary_offset
    members.update(attributes)
    return FakeRuntime(**members)


class Config(object):
    """A P2PConfig stand-in with the design's defaults (P2PIO reads attributes by name)."""

    def __init__(self, **overrides):
        self.spill_banks = (0, 1)
        self.restore_banks = (2, 3, 4, 5, 6, 7)
        self.spill_subrun_objects = 8
        self.io_threads = 8
        self.io_queue_depth_objects = 8
        self.slot_objects = 128
        self.shutdown_drain_s = 5.0
        self.nvme_device = None
        self.__dict__.update(overrides)


class FakeBoard(object):
    def __init__(self, rig: 'Rig', shard: int):
        self.rig, self.shard, self.index, self.bdf = rig, shard, 10 + shard, BDFS[shard]
        self.chardev = '/dev/tenstorrent/%d' % self.index
        self.allocate_path = '%s/%s/allocate' % (rig.allocate_root, self.bdf)
        size = rig.region_size
        self.memfd = anonymous_file('ttkvp2p-aperture-%d' % shard)      # the card's DRAM behind the aperture: it outlives any one mapping, as the real one does
        os.ftruncate(self.memfd, size)
        self.aperture = mmap.mmap(self.memfd, size)                 # the test's own mapping of it: what the p2pmem mapping and the CPU window both show
        self.banks = {bank: bytearray(size) for bank in range(rig.banks)}     # the DRAM behind each bank (the part that matters)
        self.aimed: Optional[int] = None
        self.taken: Set[int] = set()
        self.held: Set[int] = set(rig.held_windows)                 # windows another process holds
        self.window_id: Optional[int] = None
        self.pinned: Set[int] = set()
        self.slice_mapped = False
        self.available = size
        self.powered = False
        self.ignore_configure = False                               # CONFIGURE_TLB has no effect (the window stays where it was)
        self.ignore_banks: Set[int] = set()                         # ... or only when it asks for one of these banks
        self.alias: Dict[int, int] = {}                             # a bank that actually lands on another bank
        self.configures: List[Tuple[float, int, int]] = []          # (time, bank or -1, bios in flight)
        self.window_kind: Optional[str] = None
        self.bad_config = 0
        self.lock = threading.Lock()
        self.swaps = 0
        self.word_loads = 0
        self.word_stores = 0

    def release_lease(self) -> None:
        """The p2pmem mapping went away (munmap): the slice can be mapped again."""
        self.slice_mapped = False
        self.available = self.rig.region_size
        self.rig.kmd.files['%s/%s/p2pmem/available' % (PCI_ROOT, self.bdf)] = str(self.rig.region_size)

    def point(self, x: int, y: int) -> Optional[int]:
        for bank, xy in NOC.items():
            if xy == (x, y):
                return self.alias.get(bank, bank)
        return None

    def configure(self, window_id: int, addr: int, x: int, y: int) -> None:
        with self.lock:
            self.configures.append((time.monotonic(), -1 if self.point(x, y) is None else self.point(x, y), self.rig.inflight))
            if self.rig.inflight:
                self.rig.reaim_while_busy.append((self.shard, self.rig.inflight))
            if window_id != self.window_id or addr != 0:
                self.bad_config += 1
            if self.ignore_configure:
                return
            target = self.point(x, y)
            if target in self.ignore_banks:
                return
            if target == self.aimed:
                return
            if self.aimed is not None:
                self.banks[self.aimed][:] = self.aperture[:]
            if target is not None:
                self.aperture[:] = bytes(self.banks[target])
            self.aimed = target
            self.swaps += 1

    def content(self, bank: int) -> bytes:
        """The bytes of a bank's region, wherever they currently are (in the aperture if the window is aimed there)."""
        with self.lock:
            if bank == self.aimed:
                return bytes(self.aperture[:])
            return bytes(self.banks[bank])

    def poke(self, bank: int, offset: int, data: bytes) -> None:
        """Test-side write into a bank's region, bypassing the window (the model runner's gather would have put it there)."""
        with self.lock:
            target = self.aperture if bank == self.aimed else self.banks[bank]
            target[offset:offset + len(data)] = data

    def close_mappings(self) -> None:
        try:
            self.aperture.close()
        except (BufferError, ValueError):
            pass
        try:
            os.close(self.memfd)
        except OSError:
            pass


class SliceMap(mmap.mmap):
    """The mapping of the published p2pmem slice. Closing it (munmap) ends the lease; ``close`` fails with BufferError while a view of the mapping exists, and then
    the lease stays until the last view is gone and the mapping is deallocated."""

    lease_held = False

    def _release(self) -> None:
        if self.lease_held:
            self.lease_held = False
            self.board.release_lease()

    def close(self):
        super().close()                       # BufferError while a view exists: the lease is kept
        self._release()

    def __del__(self):
        try:
            self._release()
        except Exception:                     # (interpreter shutdown)
            pass


class FakeWords(object):
    """The 64-bit word view of a window (``ops.words``): ``words[i]`` is one aligned 8-byte load of window byte ``8 * i``, ``words[i] = v`` one 8-byte store."""

    def __init__(self, window: 'FakeWindow'):
        self.window = window
        self.board = window.board
        self.released = False

    def _at(self, index: int) -> int:
        if self.released:
            raise ValueError('the word view was released')
        lo = index * 8 - self.board.rig.region_base
        if index < 0 or lo < 0 or lo + 8 > self.board.rig.region_size:
            raise IndexError('window word %d is outside what the fake card backs' % index)
        return lo

    def __getitem__(self, index: int) -> int:
        lo = self._at(index)
        self.board.word_loads += 1
        return int.from_bytes(self.board.aperture[lo:lo + 8], sys.byteorder)

    def __setitem__(self, index: int, value: int) -> None:
        lo = self._at(index)
        self.board.word_stores += 1
        self.board.aperture[lo:lo + 8] = int(value).to_bytes(8, sys.byteorder)

    def release(self) -> None:
        self.released = True


class FakeWindow(object):
    """The CPU's mapping of the TLB window: byte w of the window is byte w - region_base of the aperture (the aimed bank)."""

    def __init__(self, board: FakeBoard, kind: str):
        self.board, self.kind, self.closed = board, kind, False

    def _span(self, key: slice) -> Tuple[int, int]:
        start, stop, step = key.indices(cl.WINDOW_4G)
        if step != 1:
            raise IndexError('the window is not strided')
        lo, hi = start - self.board.rig.region_base, stop - self.board.rig.region_base
        if lo < 0 or hi > self.board.rig.region_size:
            raise IndexError('window offset [%#x, %#x) is outside what the fake card backs' % (start, stop))
        return lo, hi

    def __getitem__(self, key: slice) -> bytes:
        lo, hi = self._span(key)
        with self.board.lock:
            return bytes(self.board.aperture[lo:hi])

    def __setitem__(self, key: slice, value: bytes) -> None:
        lo, hi = self._span(key)
        with self.board.lock:
            self.board.aperture[lo:hi] = bytes(value)

    def close(self) -> None:
        self.closed = True


class FakeKmd(object):
    """The ``ops`` object a CardLink calls. One per rig; it serves all four cards (the fd says which)."""

    def __init__(self, rig: 'Rig'):
        self.rig = rig
        self.next_fd = 1000
        self.fds: Dict[int, Tuple[str, FakeBoard]] = {}
        self.fail: Dict[str, int] = {}                       # 'power' | 'pin' | 'configure' | 'map_window' | 'open' -> errno
        self.log: List[Tuple[str, int]] = []
        self.files: Dict[str, str] = {}
        self.links: Dict[str, str] = {}
        self.lock = threading.Lock()
        for board in rig.boards:
            self._sysfs(board)

    def _sysfs(self, board: FakeBoard) -> None:
        rig = self.rig
        base = '%s/tenstorrent!%d' % (CLASS_ROOT, board.index)
        self.links[base + '/device'] = '%s/%s' % (PCI_ROOT, board.bdf)
        window_base = (rig.window_id - cl.FIRST_4G_TLB) * cl.WINDOW_4G
        for name, value in (('bar', 4), ('offset', window_base + rig.region_base), ('size', rig.region_size),
                            ('tlb_first', rig.window_id), ('tlb_last', rig.window_id)):
            self.files['%s/p2pmem/%s' % (base, name)] = str(value)
        self.files['%s/%s/p2pmem/size' % (PCI_ROOT, board.bdf)] = str(rig.region_size)
        self.files['%s/%s/p2pmem/available' % (PCI_ROOT, board.bdf)] = str(rig.region_size)

    # -- sysfs
    def read_text(self, path: str) -> Optional[str]:
        return self.files.get(path)

    def realpath(self, path: str) -> str:
        return self.links.get(path, path)

    def listdir(self, path: str) -> List[str]:
        if path != CLASS_ROOT:
            raise FileNotFoundError(path)
        return ['tenstorrent!%d' % board.index for board in self.rig.boards]

    # -- descriptors
    def _new_fd(self, kind: str, board: FakeBoard) -> int:
        with self.lock:
            self.next_fd += 1
            self.fds[self.next_fd] = (kind, board)
            return self.next_fd

    def _board(self, fd: int, kind: Optional[str] = None) -> FakeBoard:
        entry = self.fds.get(fd)
        if entry is None or (kind is not None and entry[0] != kind):
            raise OSError(errno.EBADF, 'bad fake descriptor %r' % (fd,))
        return entry[1]

    def open_chardev(self, path: str) -> int:
        if 'open' in self.fail:
            raise OSError(self.fail['open'], 'injected')
        for board in self.rig.boards:
            if board.chardev == path:
                return self._new_fd('chardev', board)
        raise FileNotFoundError(path)

    def open_allocate(self, path: str) -> int:
        for board in self.rig.boards:
            if board.allocate_path == path:
                return self._new_fd('allocate', board)
        raise FileNotFoundError(path)

    def close_fd(self, fd: int) -> None:
        with self.lock:
            if fd not in self.fds:
                raise OSError(errno.EBADF, 'not open')
            kind, board = self.fds.pop(fd)
        # (the p2pmem lease does NOT end here: it ends when the mapping goes away, see SliceMap)

    @property
    def open_fds(self) -> int:
        return len(self.fds)

    # -- ioctls
    def ioctl(self, fd: int, request: int, buffer: bytearray) -> None:
        board = self._board(fd, 'chardev')
        self.log.append((_NAMES.get(request, str(request)), board.shard))
        if request == cl.IOCTL_SET_POWER_STATE:
            if 'power' in self.fail:
                raise OSError(self.fail['power'], 'injected')
            argsz, flags, _, validity, power_flags = cl.POWER_STATE.unpack_from(buffer, 0)[:5]
            assert argsz == cl.POWER_STATE.size and validity == cl.BUSY_VALIDITY and power_flags == cl.BUSY_FLAGS, 'not the busy state'
            board.powered = True
        elif request == cl.IOCTL_ALLOCATE_TLB:
            size, _ = cl.ALLOCATE_IN.unpack_from(buffer, 0)
            assert size == cl.WINDOW_4G
            window = cl.FIRST_4G_TLB
            while window in board.taken or window in board.held:
                window += 1
            if window >= cl.FIRST_4G_TLB + 8:
                raise OSError(errno.ENOMEM, 'no free 4 GiB window')
            board.taken.add(window)
            cl.ALLOCATE_OUT.pack_into(buffer, cl.ALLOCATE_IN.size, window, 0, 0x100000000 * window + 1, 0x100000000 * window + 2, 0)
        elif request == cl.IOCTL_FREE_TLB:
            (window,) = struct.unpack_from('<I', buffer, 0)
            board.taken.discard(window)
            if window == board.window_id:
                board.window_id = None
        elif request == cl.IOCTL_CONFIGURE_TLB:
            if 'configure' in self.fail:
                raise OSError(self.fail['configure'], 'injected')
            fields = cl.CONFIGURE_IN.unpack_from(buffer, 0)
            window, _, addr, x, y = fields[0], fields[1], fields[2], fields[3], fields[4]
            if board.window_id is None:
                board.window_id = window
            board.configure(window, addr, x, y)
        elif request == cl.IOCTL_EXPORT_TLB_DMABUF:
            if 'pin' in self.fail:
                raise OSError(self.fail['pin'], 'injected')
            argsz, flags, window, _, _, _ = cl.EXPORT_ARG.unpack_from(buffer, 0)
            assert window in board.taken
            board.pinned.add(window)
            fd2 = self._new_fd('dmabuf', board)
            cl.EXPORT_ARG.pack_into(buffer, 0, argsz, flags, window, fd2, 0, 0)
        else:
            raise OSError(errno.ENOTTY, 'unknown ioctl %d' % request)

    # -- mappings
    def mmap_window(self, fd: int, size: int, offset: int) -> FakeWindow:
        board = self._board(fd, 'chardev')
        if 'map_window' in self.fail:
            raise OSError(self.fail['map_window'], 'injected')
        window = board.window_id
        if offset == 0x100000000 * window + 1:
            kind = 'uc'
        elif offset == 0x100000000 * window + 2:
            kind = 'wc'
        else:
            raise OSError(errno.EINVAL, 'not the mmap offset of window %s' % window)
        board.window_kind = kind
        return FakeWindow(board, kind)

    def words(self, window: 'FakeWindow') -> FakeWords:
        return FakeWords(window)

    def mmap_slice(self, fd: int, size: int):
        board = self._board(fd, 'allocate')
        if board.slice_mapped:
            raise OSError(errno.ENOMEM, 'the slice is already mapped')
        assert size == board.rig.region_size, 'the whole slice, or none'
        board.slice_mapped = True
        board.available = 0
        self.files['%s/%s/p2pmem/available' % (PCI_ROOT, board.bdf)] = '0'
        mapping = SliceMap(board.memfd, size)                  # a mapping of its own (the link unmaps it on close; the DRAM stays)
        mapping.board = board
        mapping.lease_held = True
        return mapping


_NAMES = {cl.IOCTL_SET_POWER_STATE: 'power', cl.IOCTL_ALLOCATE_TLB: 'allocate', cl.IOCTL_FREE_TLB: 'free',
          cl.IOCTL_CONFIGURE_TLB: 'configure', cl.IOCTL_EXPORT_TLB_DMABUF: 'pin'}


class Rig(object):
    """Four fake cards with their staging record, a fake kmd, and the geometry arithmetic the tests need."""

    def __init__(self, *, shard_object_bytes: int = SMALL_SHARD, slot_objects: int = SMALL_SLOT, banks: int = 8, chips: int = 4,
                 window_id: int = cl.FIRST_4G_TLB, held_windows: Sequence[int] = (), bank_address: int = 0x200040, lead: int = 4032,
                 region_base: int = 0x200000, per_bank_offset: int = 0, cardlink_module: Any = None, allocate_root: str = ALLOCATE_ROOT):
        self.cl = cardlink_module or cl
        self.allocate_root = allocate_root
        self.shard_object_bytes, self.slot_objects, self.banks, self.chips = shard_object_bytes, slot_objects, banks, chips
        self.window_id, self.held_windows = window_id, tuple(held_windows)
        self.bank_address, self.lead, self.region_base, self.per_bank_offset = bank_address, lead, region_base, per_bank_offset
        canary_window = bank_address + lead
        assert canary_window % PAGE == 0
        last = max(self.bank_window(bank) for bank in range(banks)) + lead
        used = last + PAGE + slot_objects * shard_object_bytes - region_base
        self.region_size = (used + PAGE + PAGE - 1) // PAGE * PAGE         # one spare page of slack at the end
        self.inflight = 0
        self.reaim_while_busy: List[Tuple[int, int]] = []
        self.lock = threading.Lock()
        self.boards = [FakeBoard(self, shard) for shard in range(chips)]
        self.kmd = FakeKmd(self)
        self.links: List[cl.CardLink] = []

    @property
    def object_bytes(self) -> int:
        return self.chips * self.shard_object_bytes

    def bank_window(self, bank: int) -> int:
        """The byte address of bank ``bank``'s staging range (and, the window base being 0, its window offset)."""
        return self.bank_address + bank * self.per_bank_offset

    def canary_offset(self, bank: int) -> int:
        """Offset of a bank's canary page inside the aperture."""
        return self.bank_window(bank) + self.lead - self.region_base

    def data_offset(self, bank: int, obj: int, shard_bytes: Optional[int] = None) -> int:
        """Offset of object ``obj`` of a bank's slot inside the aperture."""
        return self.canary_offset(bank) + PAGE + obj * (shard_bytes or self.shard_object_bytes)

    def staging_info(self) -> dict:
        banks = [{'bank': bank, 'noc_x': NOC[bank][0], 'noc_y': NOC[bank][1], 'address': self.bank_window(bank), 'bank_offset': bank * self.per_bank_offset,
                  'noc_endpoints': [list(NOC[bank])]} for bank in range(self.banks)]
        size = self.region_size + self.region_base - self.bank_address
        return {'address': self.bank_address, 'size': size, 'page_bytes': 1088, 'aligned_4k_offset': self.lead,
                'aligned_4k_address': self.bank_address + self.lead, 'aligned_4k_size': (size - self.lead) // PAGE * PAGE,
                'banks': banks,
                'devices': [{'mesh_coord': [d, 0], 'mesh_index': d, 'device_id': b.shard, 'mmio_chip_id': b.index, 'pcie_device': b.index, 'pci_bus_id': b.bdf}
                            for d, b in enumerate(self.boards)]}

    def make_link(self, shard: int, **overrides) -> cl.CardLink:
        board = self.boards[shard]
        kwargs = dict(shard_object_bytes=self.shard_object_bytes, slot_objects=self.slot_objects, sysfs_class=CLASS_ROOT, sysfs_pci=PCI_ROOT, ops=self.kmd, settle_s=0.0)
        kwargs.update(overrides)
        link = self.cl.CardLink(board.chardev, board.bdf, board.allocate_path, self.window_id, **kwargs)
        self.links.append(link)
        return link

    def open_links(self, **overrides) -> List[cl.CardLink]:
        info = self.staging_info()
        links = [self.make_link(shard, **overrides) for shard in range(self.chips)]
        for link in links:
            link.open(info)
        return links

    def close(self) -> None:
        for link in self.links:
            link.close()
        for board in self.boards:
            board.close_mappings()


class DriveOps(object):
    """The drive syscalls of P2PIO, around the real ones: logged, holdable, with short counts and injected errors."""

    def __init__(self, rig: Rig):
        self.rig = rig
        self.calls: List[Dict[str, Any]] = []
        self.lock = threading.Lock()
        self.holds: List[threading.Event] = []            # the next calls block on these, one each
        self.entered: List[threading.Event] = []          # set when the matching held call is in flight
        self.every_hold: List[threading.Event] = []
        self.shorts: List[int] = []                       # the next calls move at most this many bytes
        self.errors: List[BaseException] = []             # the next calls raise these
        self.zero_next = 0                                # the next calls return 0 (EOF)
        self.fail_at: Dict[int, BaseException] = {}       # call number (0-based) -> the error that call raises
        self.swallow_next = 0                             # the next calls report success and move nothing
        self.sequence = 0
        self.over_next = 0                                # the next calls report one byte more than they were asked for
        self.on_call = None                               # a hook(kind, call) run inside the call, after it counted as in flight

    def hold_next(self, count: int = 1) -> List[Tuple[threading.Event, threading.Event]]:
        pairs = []
        for _ in range(count):
            release, entered = threading.Event(), threading.Event()
            self.holds.append(release)
            self.entered.append(entered)
            self.every_hold.append(release)
            pairs.append((release, entered))
        return pairs

    def release_all(self) -> None:
        for event in self.every_hold:
            event.set()

    # the interface P2PIO uses
    def preadv(self, fd: int, buffers: Sequence[Any], offset: int) -> int:
        return self._call('read', os.preadv, fd, buffers, offset)

    def pwritev(self, fd: int, buffers: Sequence[Any], offset: int) -> int:
        return self._call('write', os.pwritev, fd, buffers, offset)

    def _call(self, kind: str, function, fd: int, buffers: Sequence[Any], offset: int) -> int:
        lengths = [memoryview(b).nbytes for b in buffers]
        with self.lock:
            hold = self.holds.pop(0) if self.holds else None
            entered = self.entered.pop(0) if self.entered else None
            record = dict(kind=kind, offset=offset, lengths=lengths, total=sum(lengths), aimed=[board.aimed for board in self.rig.boards],
                          t=time.monotonic(), thread=threading.current_thread().name, returned=None, number=self.sequence)
            self.sequence += 1
            self.calls.append(record)
        with self.rig.lock:
            self.rig.inflight += 1
        try:
            if entered is not None:
                entered.set()
            if self.on_call is not None:
                self.on_call(kind, record)
            if hold is not None:
                hold.wait(30)
            with self.lock:
                error = self.errors.pop(0) if self.errors else self.fail_at.pop(record['number'], None)
                cap = self.shorts.pop(0) if self.shorts else None
                swallow = self.swallow_next > 0
                if swallow:
                    self.swallow_next -= 1
                zero = self.zero_next > 0
                over = self.over_next > 0
                if zero:
                    self.zero_next -= 1
                if over:
                    self.over_next -= 1
            if error is not None:
                raise error
            if zero:
                record['returned'] = 0
                return 0
            if swallow:
                record['returned'] = sum(lengths)
                return sum(lengths)
            use = list(buffers)
            temporaries: List[memoryview] = []
            try:
                if cap is not None:
                    use, remaining = [], cap
                    for buffer in buffers:
                        view = memoryview(buffer)
                        if remaining <= 0:
                            break
                        if view.nbytes > remaining:
                            view = view[:remaining]
                            temporaries.append(view)
                        use.append(view)
                        remaining -= view.nbytes
                count = function(fd, use, offset)
            finally:
                for view in temporaries:
                    view.release()
            record['returned'] = count + (1 if over else 0)
            return count + (1 if over else 0)
        finally:
            with self.rig.lock:
                self.rig.inflight -= 1


import random
import shutil
import sys

from ttkvp2p import p2pio as pio


def pattern(shard: int, tag: Any, nbytes: int) -> bytes:
    """Deterministic bytes that differ per shard and tag."""
    return random.Random('%s|%s' % (shard, tag)).randbytes(nbytes)


class Env(object):
    """Four opened links, a drive file, a wrapped drive and a P2PIO over them. ``modules`` swaps in mutated copies of cardlink / p2pio."""

    DATA_OFF = 1 << 20

    def __init__(self, *, rig_kwargs: Optional[dict] = None, config_kwargs: Optional[dict] = None, io_kwargs: Optional[dict] = None,
                 modules: Optional[Tuple[Any, Any]] = None, data_slots: int = 64, runtime_kwargs: Optional[dict] = None, start: bool = True):
        self.cl, self.io_module = modules if modules else (cl, pio)
        self.rig = Rig(cardlink_module=self.cl, **(rig_kwargs or {}))
        self.dir = tempfile.mkdtemp(prefix='ttkvp2p-test-')
        self.data_slots = data_slots
        self.object_bytes = self.rig.object_bytes
        self.data_hi = self.DATA_OFF + data_slots * self.object_bytes
        self.fd, self.direct = drive_file(self.dir, self.data_hi)
        self.path = os.path.join(self.dir, 'drive.img')
        self.ops = DriveOps(self.rig)
        self.runtime = FakeRuntime(**(runtime_kwargs or {}))
        self.config = Config(**(config_kwargs or {}))
        self.links: List[Any] = []
        self.io: Any = None
        self.io_kwargs = dict(io_kwargs or {})
        self.kept: List[Any] = []
        if start:
            self.start()

    def keep(self, *views: Any) -> Any:
        """Register slot views (or lists of them) a test made, so that ``close`` releases them before the links unmap: a view that outlives its link stops the unmap
        and costs a garbage collection. Returns the first argument."""
        for view in views:
            self.kept.extend(view if isinstance(view, (list, tuple)) else [view])
        return views[0] if views else None

    def start(self) -> None:
        self.links = self.rig.open_links()
        self.build_io()

    def wp3a_runtime(self, **overrides) -> FakeRuntime:
        """A runtime shaped like WP3a's, over this drive (the descriptor, the data region and the canary block of INTERFACE 3.4)."""
        members = dict(fd=self.fd, data_off=self.DATA_OFF, num_slots=self.data_slots, object_bytes=self.object_bytes)
        members.update(overrides)
        return wp3a_runtime(**members)

    def build_io(self, **overrides) -> Any:
        kwargs = dict(drive_fd=self.fd, drive_ops=self.ops, drive_range=(self.DATA_OFF, self.data_hi), allow_buffered=not self.direct)
        kwargs.update(self.io_kwargs)
        kwargs.update(overrides)
        self.io = self.io_module.P2PIO(self.links, self.runtime, self.config, self.rig.staging_info(), **kwargs)
        return self.io

    # the drive, seen from the test (buffered, so it does not depend on O_DIRECT alignment)
    def drive_read(self, offset: int, size: int) -> bytes:
        fd = os.open(self.path, os.O_RDONLY)
        try:
            return os.pread(fd, size, offset)
        finally:
            os.close(fd)

    def drive_write(self, offset: int, data: bytes) -> None:
        fd = os.open(self.path, os.O_RDWR)
        try:
            os.pwrite(fd, data, offset)
            os.fsync(fd)
        finally:
            os.close(fd)

    def drive_off(self, slot: int) -> int:
        return self.DATA_OFF + slot * self.object_bytes

    # the cards, seen from the test
    def put(self, shard: int, bank: int, obj: int, data: bytes) -> None:
        self.rig.boards[shard].poke(bank, self.rig.data_offset(bank, obj), data)

    def get(self, shard: int, bank: int, obj: int) -> bytes:
        offset = self.rig.data_offset(bank, obj)
        return self.rig.boards[shard].content(bank)[offset:offset + self.rig.shard_object_bytes]

    def card_object(self, shard: int, tag: Any) -> bytes:
        return pattern(shard, tag, self.rig.shard_object_bytes)

    def drive_object(self, tag: Any) -> bytes:
        """What the drive holds for an object whose shard d is ``card_object(d, tag)``."""
        return b''.join(self.card_object(d, tag) for d in range(self.rig.chips))

    def close(self) -> None:
        self.ops.release_all()
        if self.io is not None:
            self.io.close(timeout=5)
        for view in self.kept:
            try:
                view.release()
            except (BufferError, ValueError):
                pass
        self.kept = []
        for link in self.links:
            link.close()
        self.rig.close()
        try:
            os.close(self.fd)
        except OSError:
            pass
        shutil.rmtree(self.dir, ignore_errors=True)
