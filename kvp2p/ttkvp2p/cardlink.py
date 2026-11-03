"""CardLink: one card's end of the peer-to-peer path (design 2.2, 2.8, 2.9; INTERFACE 3.2).

A port of the Qwen ``TtLink`` (``qwen_p2p_restore.py``) with the Gemma changes:

  * the window is the fixed TLB window 202, taken once and **re-aimed per staging bank** instead of aimed once;
  * the CPU reaches the window through the driver's **uncached** (UC) mapping, so a canary read cannot be served from a write
    buffer;
  * every bank carries a **canary page** (the first 4 KiB of the bank's 4 KiB-aligned range). All of them are written, with a
    per-open random nonce, and read back **before any data goes into the card** (the order of the WP2a probe, which has not yet run on a card). Every later
    ``aim`` ends with an 8-byte non-posted read of the canary of the bank it aimed at, which cannot complete before the three
    posted ``CONFIGURE_TLB`` register writes landed; a different value means the window is somewhere else (``CanaryMismatch``);
  * the published slice is mapped once and never handed out whole: ``slot_view`` returns per-object views inside one bank's
    slot, bounds-checked, and ``locate`` lets the I/O layer prove where any iovec points (INTERFACE: DMA-only views);
  * ``begin_io`` / ``end_io`` count the bios in flight on this card; ``aim`` refuses while any is (the window must never move
    under a DMA), and ``begin_io`` refuses an I/O planned in an earlier aim epoch.

Everything the kernel is asked for goes through one small ``ops`` object (``SystemOps`` by default), so the CPU tests run the
whole class against a fake driver (``tests/fake_card.py``). Importing this module imports neither ttnn nor the extension and
touches nothing (invariant I14). Stdlib only; Python 3.10.

The geometry (all from ``qwen_p2p_staging_info`` plus the layout)::

    bank k, byte address a_k of the staging range in the bank (= the window offset, because the window base is 0)
    canary page            window offset  a_k + lead                      lead = aligned_4k_offset
    slot object j          window offset  a_k + lead + 4096 + j * shard    shard = bytes of one object on ONE card (696,320)
    slice offset           window offset - (slice BAR4 offset - (window id - 202) * 4 GiB)
"""

import ctypes
import fcntl
import gc
import hashlib
import logging
import mmap
import operator
import os
import re
import struct
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

PAGE = 4096
GIB = 1 << 30
WINDOW_4G = 4 * GIB
FIRST_4G_TLB = 202                    # Blackhole: 202 windows of 2 MiB, then 8 of 4 GiB
WINDOW_TOP = 0xFF000000               # the top 16 MiB of a 4 GiB window is not DRAM
NOC_COORDINATE_LIMIT = 64             # the TLB register's x and y fields are 6 bits wide
ORDERINGS = {'relaxed': 0, 'strict': 1, 'posted': 2}
CANARY_BYTES = PAGE                   # the canary page at the start of every bank's range; slot data follows it
CANARY_COMPARE_BYTES = 8              # what an aim reads back (one non-posted 8-byte read)
GEMMA_SHARD_OBJECT_BYTES = 10 * 69632  # 696,320: one object on one card = 10 tensors x one 69,632-byte block row
MAX_SLOT_OBJECTS = 128                # the device digest takes at most 128 blocks (INTERFACE 3.3)
SYSFS_CLASS = '/sys/class/tenstorrent'
SYSFS_PCI = '/sys/bus/pci/devices'

# tt-kmd ioctl ABI (ioctl.h): _IO(0xFA, n), the argument is a struct passed by pointer. The same numbers as
# scripts/ci/qwen_p2p_restore.py and scripts/p2p/p2p_probe.py (a test pins them equal).
TT_MAGIC = 0xFA
IOCTL_ALLOCATE_TLB = (TT_MAGIC << 8) | 11
IOCTL_FREE_TLB = (TT_MAGIC << 8) | 12
IOCTL_CONFIGURE_TLB = (TT_MAGIC << 8) | 13
IOCTL_SET_POWER_STATE = (TT_MAGIC << 8) | 15
IOCTL_EXPORT_TLB_DMABUF = (TT_MAGIC << 8) | 16
# struct tenstorrent_allocate_tlb { in { u64 size; u64 reserved; } out { u32 id; u32 reserved0; u64 mmap_offset_uc; u64 mmap_offset_wc; u64 reserved1; } }
ALLOCATE_IN = struct.Struct('<QQ')
ALLOCATE_OUT = struct.Struct('<IIQQQ')
# struct tenstorrent_configure_tlb { in { u32 id; u32 reserved; noc_tlb_config { u64 addr; u16 x_end, y_end, x_start, y_start;
#   u8 noc, mcast, ordering, linked, static_vc; u8 reserved0[3]; u32 reserved1[2]; } } out { u64 reserved; } }
CONFIGURE_IN = struct.Struct('<IIQHHHHBBBBB3x2I')
# struct tenstorrent_power_state { u32 argsz; u32 flags; u8 reserved0; u8 validity; u16 power_flags; u16 power_settings[14]; }
POWER_STATE = struct.Struct('<IIBBH14H')
POWER_FLAG_MRISC_PHY_WAKEUP = 1 << 1
POWER_FLAG_TENSIX_ENABLE = 1 << 2
POWER_FLAG_L2CPU_ENABLE = 1 << 3
BUSY_FLAGS = POWER_FLAG_MRISC_PHY_WAKEUP | POWER_FLAG_TENSIX_ENABLE | POWER_FLAG_L2CPU_ENABLE
BUSY_VALIDITY = 4
# struct tenstorrent_export_tlb_dmabuf { u32 argsz; u32 flags; u32 tlb_id; s32 fd; u64 offset; u64 size; }
EXPORT_ARG = struct.Struct('<IIIiQQ')


# --------------------------------------------------------------------------------------------------------------------
# Errors
# --------------------------------------------------------------------------------------------------------------------

class CardLinkError(RuntimeError):
    """Base of everything this module raises on purpose."""


class CardLinkUnavailable(CardLinkError):
    """The path cannot be set up on this card (a missing sysfs entry, a held window, a slice that does not fit). Raised by ``open``."""


class CanaryMismatch(CardLinkError):
    """After aiming the window, the canary read back is not the aimed bank's: the window is somewhere else. No bio may be issued; the
    tier latches (design 2.8 step 3)."""


class AimWhileBusy(CardLinkError):
    """``aim`` was called with a bio in flight on this card: a re-aim happens only when the card is idle."""


class StaleAim(CardLinkError):
    """An I/O was planned for an earlier aim of the window (another bank, or an older epoch of the same bank)."""


class LinkClosed(CardLinkError):
    """The link is closed (or poisoned by a canary mismatch)."""


class BoundsViolation(AssertionError):
    """A request reached outside what it is allowed to touch: a bank or object that is not in the slot, an iovec that does not point
    where the plan says, a drive range outside the data region. It is an assertion because it is a bug, never a condition; the tier
    latches (design 2.8, Bounds)."""


# --------------------------------------------------------------------------------------------------------------------
# The kernel, behind one small object
# --------------------------------------------------------------------------------------------------------------------

class SystemOps(object):
    """The real kernel: the chardev, its ioctls and mappings, the p2pmem file and sysfs. The tests pass a fake with the same methods."""

    def open_chardev(self, path: str) -> int:
        return os.open(path, os.O_RDWR | getattr(os, 'O_CLOEXEC', 0))

    def ioctl(self, fd: int, request: int, buffer: bytearray) -> None:
        fcntl.ioctl(fd, request, buffer, True)

    def mmap_window(self, fd: int, size: int, offset: int):
        return mmap.mmap(fd, size, flags=mmap.MAP_SHARED, prot=mmap.PROT_READ | mmap.PROT_WRITE, offset=offset)

    def open_allocate(self, path: str) -> int:
        return os.open(path, os.O_RDWR | getattr(os, 'O_CLOEXEC', 0))

    def mmap_slice(self, fd: int, size: int):
        return mmap.mmap(fd, size, flags=mmap.MAP_SHARED, prot=mmap.PROT_READ | mmap.PROT_WRITE)

    def close_fd(self, fd: int) -> None:
        os.close(fd)

    def read_text(self, path: str) -> Optional[str]:
        try:
            with open(path) as handle:
                return handle.read().strip()
        except OSError:
            return None

    def realpath(self, path: str) -> str:
        return os.path.realpath(path)

    def listdir(self, path: str) -> List[str]:
        return sorted(os.listdir(path))


def default_ops():
    """The ops a ``CardLink`` built without ``ops=`` uses (the production wiring constructs links without them; tests replace this)."""
    return SystemOps()


# --------------------------------------------------------------------------------------------------------------------
# Small pure helpers
# --------------------------------------------------------------------------------------------------------------------

def normalise_bdf(text: Any) -> str:
    """A PCI address as ``dddd:bb:dd.f`` in lower case. A short form gets the domain 0000. Anything else is returned lower-cased and
    unchanged so that opaque stand-ins keep working, but only if it is a safe path component (no separators)."""
    value = str(text).strip().lower()
    if re.fullmatch(r'[0-9a-f]{2}:[0-9a-f]{2}\.[0-7]', value):
        value = '0000:' + value
    if not value or not re.fullmatch(r'[0-9a-z][0-9a-z:._-]*', value) or '..' in value:
        raise ValueError('%r is not a usable PCI address' % (text,))
    return value


def chardev_index(chardev: str) -> int:
    """N of ``/dev/tenstorrent/N``."""
    found = re.fullmatch(r'(?:.*/)?(\d+)', str(chardev))
    if not found:
        raise CardLinkUnavailable('the card device %r does not end in the driver\'s index' % (chardev,))
    return int(found.group(1))


def chardev_of_bdf(bdf: str, sysfs_class: Optional[str] = None, ops: Any = None, dev_dir: str = '/dev/tenstorrent') -> str:
    """The ``/dev/tenstorrent/N`` of the card at ``bdf``, from the driver's own table (the nodes renumber on every reset, and the staging
    record's ``device_id`` is tt-metal's logical id, not N)."""
    sysfs_class = sysfs_class or SYSFS_CLASS
    ops = ops or default_ops()
    want = normalise_bdf(bdf)
    try:
        names = ops.listdir(sysfs_class)
    except OSError as error:
        raise CardLinkUnavailable('cannot list %s: %s' % (sysfs_class, error))
    for name in names:
        found = re.fullmatch(r'tenstorrent!(\d+)', name)
        if not found:
            continue
        device = ops.realpath(os.path.join(sysfs_class, name, 'device'))
        if os.path.basename(device).lower() == want:
            return os.path.join(dev_dir, found.group(1))
    raise CardLinkUnavailable('the driver lists no card at %s' % want)


def as_index(value: Any, what: str) -> int:
    """``value`` as a Python int: an int or anything with ``__index__`` (a numpy integer is what the worker's slot arithmetic may hand over), never a bool, a float or a
    string. Raises ``BoundsViolation``: a request that is not made of integers is a bug in the caller."""
    if isinstance(value, bool):
        raise BoundsViolation('%s must be an integer, got %r' % (what, value))
    try:
        return operator.index(value)
    except TypeError:
        raise BoundsViolation('%s must be an integer, got %r' % (what, value))


def _address_of(buffer: Any) -> int:
    """The virtual address of a writable buffer's first byte. The ctypes export is dropped again before returning, so the buffer's owner can
    still be closed."""
    return ctypes.addressof(ctypes.c_char.from_buffer(buffer))


def canary_page(nonce: bytes, bdf: str, bank: int) -> bytes:
    """One bank's canary page: a value no other bank (or card, or open of the link) has, repeated over the page."""
    digest = hashlib.sha256(b'ttkvp2p-canary|' + nonce + b'|' + bdf.encode() + b'|%d' % bank).digest()
    return digest * (CANARY_BYTES // len(digest))


@dataclass(frozen=True)
class BankPlan:
    """Where one staging bank is, in the three address spaces."""

    bank: int
    noc_x: int
    noc_y: int
    canary_window_off: int        # window offset of the canary page (= the bank byte address, the window base being 0)
    canary_slice_off: int         # the same byte in the p2pmem mapping
    data_slice_off: int           # the mapping offset of slot object 0 (one page after the canary)


# --------------------------------------------------------------------------------------------------------------------
# The link
# --------------------------------------------------------------------------------------------------------------------

class CardLink(object):
    """One card: the fd, TLB window 202 aimed at one staging bank at a time, its UC mapping for the canary reads, and the mapping of the
    published slice. See the module docstring. ``slot_view`` and ``locate`` are the DMA side; ``aim`` and ``flush`` are the CPU side."""

    cpu_window_bytes = WINDOW_4G          # what the CPU mapping of the window covers (a test shrinks it)

    def __init__(self, chardev: str, bdf: str, allocate_path: str, window_id: int = FIRST_4G_TLB, *,
                 shard_object_bytes: int = GEMMA_SHARD_OBJECT_BYTES, slot_objects: Optional[int] = None,
                 ordering: str = 'relaxed', cpu_mapping: str = 'uc', settle_s: float = 0.0,
                 sysfs_class: Optional[str] = None, sysfs_pci: Optional[str] = None,
                 ops: Any = None, nonce: Optional[bytes] = None):
        if ordering not in ORDERINGS:
            raise ValueError('ordering %r is not one of %s' % (ordering, sorted(ORDERINGS)))
        if cpu_mapping not in ('uc', 'wc'):
            raise ValueError('cpu_mapping is "uc" or "wc"')
        if shard_object_bytes <= 0 or shard_object_bytes % PAGE:
            raise ValueError('an object on one card (%d bytes) must be a whole number of 4 KiB pages (O_DIRECT)' % shard_object_bytes)
        self.chardev = chardev
        self.bdf = normalise_bdf(bdf)
        self.allocate_path = allocate_path
        self.window_id = int(window_id)
        self.shard_object_bytes = int(shard_object_bytes)
        self._requested_slot_objects = slot_objects
        self.slot_objects = 0
        self.ordering = ordering
        self.cpu_mapping = cpu_mapping
        self.settle_s = settle_s
        self.sysfs_class = sysfs_class or SYSFS_CLASS
        self.sysfs_pci = sysfs_pci or SYSFS_PCI
        self.ops = ops or default_ops()
        self._nonce = nonce if nonce is not None else os.urandom(16)
        self.index: Optional[int] = None          # N of /dev/tenstorrent/N
        self.shard: Optional[int] = None          # the mesh coordinate: shard d of every object (design 2.2)
        self.banks: Tuple[int, ...] = ()
        # state
        self._lock = threading.RLock()
        self._state = 'new'                       # new -> opening -> open -> closed
        self._fd: Optional[int] = None
        self._pin_fd: Optional[int] = None
        self._p2p_fd: Optional[int] = None
        self._cpu = None
        self._slice_map = None
        self._view: Optional[memoryview] = None
        self._base_address = 0
        self._slice_size = 0
        self._slice_window_off = 0
        self._mmap_offset_uc = 0
        self._mmap_offset_wc = 0
        self._plan: Dict[int, BankPlan] = {}
        self._canaries: Dict[int, bytes] = {}
        self._aimed: Optional[int] = None
        self._epoch = 0
        self._inflight = 0
        self._poisoned: Optional[str] = None
        self._abandoned = False
        self._window_held = False
        self.proved = False                       # every bank's canary page was written, then read back, before any data
        self.info: Dict[str, Any] = {}
        self.stats: Dict[str, int] = dict(aims=0, flushes=0, flush_mismatches=0, flush_errors=0, canary_mismatches=0)

    # -- state -----------------------------------------------------------------------------------------------------
    @property
    def aimed_bank(self) -> Optional[int]:
        return self._aimed

    @property
    def aim_epoch(self) -> int:
        """Counts successful aims (the same bank again counts). An I/O planned in epoch e is stale in any other epoch."""
        return self._epoch

    @property
    def inflight(self) -> int:
        return self._inflight

    @property
    def closed(self) -> bool:
        return self._state == 'closed'

    @property
    def poisoned(self) -> Optional[str]:
        return self._poisoned

    def _require_open(self) -> None:
        if self._state != 'open':
            raise LinkClosed('the link to %s is %s' % (self.bdf, self._state))
        if self._poisoned:
            raise LinkClosed('the link to %s is unusable: %s' % (self.bdf, self._poisoned))

    # -- open ------------------------------------------------------------------------------------------------------
    def open(self, staging_info: dict) -> None:
        """ALLOCATE_TLB 202, SET_POWER_STATE, EXPORT_TLB_DMABUF, the UC mapping of the window, the mapping of the published slice, then a
        canary page per bank, all proved before any data. Every host-side check (sysfs, geometry) runs before the first device call; a
        failure closes whatever was opened and re-raises."""
        with self._lock:
            if self._state != 'new':
                raise CardLinkError('open() was already called on the link to %s' % self.bdf)
            self._state = 'opening'
        try:
            self._open(staging_info)
        except BaseException:
            self.close()
            raise
        with self._lock:
            self._state = 'open'

    def _sysfs_int(self, path: str, what: str) -> int:
        text = self.ops.read_text(path)
        if text is None:
            raise CardLinkUnavailable('%s is missing (%s): the patched tt-kmd is not loaded, or its p2pmem_devices does not name %s' % (path, what, self.bdf))
        try:
            return int(text, 0)
        except ValueError:
            raise CardLinkUnavailable('%s says %r, not a number' % (path, text))

    def _host_checks(self, staging_info: dict) -> None:
        """Everything that needs neither a device call nor a mapping: which card this is, what it published, where the banks are."""
        self.index = chardev_index(self.chardev)
        class_dir = os.path.join(self.sysfs_class, 'tenstorrent!%d' % self.index)
        seen = os.path.basename(self.ops.realpath(os.path.join(class_dir, 'device'))).lower()
        if seen != self.bdf:
            raise CardLinkUnavailable('%s is the card at %s, not %s: the staging record names a card this process would not open' % (self.chardev, seen, self.bdf))
        # the shard: the card's position in the mesh, from the staging record (design 2.2; never the /dev order)
        self.shard = self._resolve_shard(staging_info)
        # what the driver published
        info = {name: self._sysfs_int(os.path.join(class_dir, 'p2pmem', name), name) for name in ('bar', 'offset', 'size', 'tlb_first', 'tlb_last')}
        info['core_size'] = self._sysfs_int(os.path.join(self.sysfs_pci, self.bdf, 'p2pmem', 'size'), 'size')
        info['core_available'] = self._sysfs_int(os.path.join(self.sysfs_pci, self.bdf, 'p2pmem', 'available'), 'available')
        self.info = dict(info)
        if info['bar'] != 4:
            raise CardLinkUnavailable('the published slice of %s is in BAR%d; only BAR4 (the 4 GiB windows) is a DRAM aperture' % (self.bdf, info['bar']))
        if info['tlb_first'] != self.window_id or info['tlb_last'] != self.window_id:
            raise CardLinkUnavailable('the published slice of %s lies in TLB windows %d..%d, the tier uses window %d only' % (self.bdf, info['tlb_first'], info['tlb_last'], self.window_id))
        if self.window_id < FIRST_4G_TLB:
            raise CardLinkUnavailable('TLB window %d is not one of the 4 GiB windows' % self.window_id)
        if info['core_size'] != info['size']:
            raise CardLinkUnavailable('the PCI core\'s p2pmem size %d of %s differs from the driver\'s %d' % (info['core_size'], self.bdf, info['size']))
        if info['core_available'] != info['core_size']:
            raise CardLinkUnavailable('the published slice of %s is already allocated (available %d of %d): another holder has it' % (self.bdf, info['core_available'], info['core_size']))
        if info['size'] <= 0 or info['size'] % PAGE:
            raise CardLinkUnavailable('the published slice of %s is %d bytes, not a whole number of pages' % (self.bdf, info['size']))
        window_base = (self.window_id - FIRST_4G_TLB) * WINDOW_4G
        self._slice_window_off = info['offset'] - window_base
        self._slice_size = info['size']
        if self._slice_window_off < 0 or self._slice_window_off + self._slice_size > WINDOW_4G:
            raise CardLinkUnavailable('the published slice [%d, +%d) of %s is not inside window %d' % (info['offset'], info['size'], self.bdf, self.window_id))
        self._plan_banks(staging_info)

    def _resolve_shard(self, staging_info: dict) -> int:
        devices = list(staging_info.get('devices') or ())
        if not devices:
            raise CardLinkUnavailable('the staging record lists no devices: the shard of %s cannot be found' % self.bdf)
        found = []
        for position, entry in enumerate(devices):
            bus = entry.get('pci_bus_id')
            if bus is not None:
                try:
                    match = normalise_bdf(bus) == self.bdf
                except ValueError:
                    match = False
                if match:
                    found.append(position)
            elif entry.get('pcie_device') is not None and int(entry['pcie_device']) == self.index:
                found.append(position)       # the extension could not read sysfs: fall back to the kmd index it reports
        if len(found) != 1:
            raise CardLinkUnavailable('the card at %s is %d of the %d mesh devices in the staging record (it must be exactly one)' % (self.bdf, len(found), len(devices)))
        entry = devices[found[0]]
        if entry.get('pcie_device') is not None and entry.get('pci_bus_id') is not None and int(entry['pcie_device']) != self.index:
            raise CardLinkUnavailable('the staging record says %s is /dev/tenstorrent/%s, the card opened is /dev/tenstorrent/%d' % (self.bdf, entry['pcie_device'], self.index))
        return found[0]

    def _plan_banks(self, staging_info: dict) -> None:
        shard = self.shard_object_bytes
        lead = int(staging_info['aligned_4k_offset'])
        usable = int(staging_info['aligned_4k_size'])
        if lead < 0 or usable < CANARY_BYTES + shard:
            raise CardLinkUnavailable('a staging bank (%d usable bytes from its first 4 KiB-aligned byte) cannot hold a canary page and one object (%d bytes per card)' % (usable, shard))
        capacity = (usable - CANARY_BYTES) // shard
        wanted = self._requested_slot_objects if self._requested_slot_objects is not None else MAX_SLOT_OBJECTS
        self.slot_objects = max(0, min(int(wanted), int(capacity), MAX_SLOT_OBJECTS))
        if self.slot_objects < 1:
            raise CardLinkUnavailable('a staging slot cannot hold one object')
        banks = list(staging_info.get('banks') or ())
        if not banks:
            raise CardLinkUnavailable('the staging record lists no banks')
        base_address = staging_info.get('address')
        plan: Dict[int, BankPlan] = {}
        for entry in banks:
            bank = int(entry['bank'])
            x, y = entry.get('noc_x'), entry.get('noc_y')
            if x is None or y is None:
                raise CardLinkUnavailable('bank %d has no NOC coordinates (%s)' % (bank, entry.get('endpoint_error', 'unknown reason')))
            if not (0 <= int(x) < NOC_COORDINATE_LIMIT and 0 <= int(y) < NOC_COORDINATE_LIMIT):
                raise CardLinkUnavailable('bank %d: NOC (%s, %s) does not fit the 6-bit coordinate fields of a TLB register' % (bank, x, y))
            address = entry.get('address', base_address)
            if address is None:
                raise CardLinkUnavailable('bank %d has no address' % bank)
            canary_window = int(address) + lead
            if canary_window < 0 or canary_window % PAGE:
                raise CardLinkUnavailable('bank %d: the canary page would be at window offset %#x, not 4 KiB aligned' % (bank, canary_window))
            end = canary_window + CANARY_BYTES + self.slot_objects * shard
            if end > WINDOW_TOP:
                raise CardLinkUnavailable('bank %d: the slot ends at window offset %#x, in the top 16 MiB of the window (from %#x), which is not DRAM' % (bank, end, WINDOW_TOP))
            canary_slice = canary_window - self._slice_window_off
            if canary_slice < 0 or canary_slice + CANARY_BYTES + self.slot_objects * shard > self._slice_size:
                raise CardLinkUnavailable('bank %d: the canary page and slot [window offset %#x, +%d) are outside the published slice (window offset %#x, +%d)'
                                          % (bank, canary_window, CANARY_BYTES + self.slot_objects * shard, self._slice_window_off, self._slice_size))
            if bank in plan:
                raise CardLinkUnavailable('bank %d is listed twice' % bank)
            plan[bank] = BankPlan(bank, int(x), int(y), canary_window, canary_slice, canary_slice + CANARY_BYTES)
        if len({(p.noc_x, p.noc_y) for p in plan.values()}) != len(plan):
            raise CardLinkUnavailable('two banks share NOC coordinates: a window aimed at one cannot be told from the other')
        self._plan = plan
        self.banks = tuple(sorted(plan))

    def _open(self, staging_info: dict) -> None:
        self._host_checks(staging_info)
        ops = self.ops
        try:
            self._fd = ops.open_chardev(self.chardev)
        except OSError as error:
            raise CardLinkUnavailable('cannot open %s: %s' % (self.chardev, error))
        self._power_up()
        self._acquire_window()
        first = self._plan[self.banks[0]]
        self._configure(first)
        self._pin()
        offset = self._mmap_offset_uc if self.cpu_mapping == 'uc' else self._mmap_offset_wc
        if self.cpu_mapping == 'uc' and not offset:
            raise CardLinkUnavailable('the driver did not report an uncached mapping offset for window %d of %s' % (self.window_id, self.bdf))
        try:
            self._cpu = ops.mmap_window(self._fd, self.cpu_window_bytes, offset)
        except (OSError, ValueError) as error:
            raise CardLinkUnavailable('cannot map TLB window %d of %s (%s): %s' % (self.window_id, self.bdf, self.cpu_mapping, error))
        self._map_slice()
        self._write_canaries()
        self._prove_canaries()
        self.proved = True

    # -- the driver's ioctls ---------------------------------------------------------------------------------------
    def _ioctl(self, request: int, buffer: bytearray, what: str) -> None:
        try:
            self.ops.ioctl(self._fd, request, buffer)
        except OSError as error:
            raise CardLinkUnavailable('%s failed on %s: %s' % (what, self.bdf, error))

    def _power_up(self) -> None:
        """The busy power state (DRAM PHY awake), as the UMD asks when it opens a card. The state is this fd's and is withdrawn when the fd closes."""
        buffer = bytearray(POWER_STATE.size)
        POWER_STATE.pack_into(buffer, 0, POWER_STATE.size, 0, 0, BUSY_VALIDITY, BUSY_FLAGS, *([0] * 14))
        self._ioctl(IOCTL_SET_POWER_STATE, buffer, 'SET_POWER_STATE')
        if self.settle_s:
            time.sleep(self.settle_s)

    def _acquire_window(self) -> None:
        """Own TLB window ``window_id``: ALLOCATE_TLB hands out the lowest free 4 GiB window, so take windows until it comes up and give the others back."""
        taken: List[int] = []
        try:
            while True:
                buffer = bytearray(ALLOCATE_IN.size + ALLOCATE_OUT.size)
                ALLOCATE_IN.pack_into(buffer, 0, WINDOW_4G, 0)
                self._ioctl(IOCTL_ALLOCATE_TLB, buffer, 'ALLOCATE_TLB (window %d)' % self.window_id)
                tlb_id, _, mmap_uc, mmap_wc, _ = ALLOCATE_OUT.unpack_from(buffer, ALLOCATE_IN.size)
                if tlb_id == self.window_id:
                    self._mmap_offset_uc, self._mmap_offset_wc = mmap_uc, mmap_wc
                    self._window_held = True
                    return
                taken.append(tlb_id)
                if tlb_id > self.window_id:
                    raise CardLinkUnavailable('TLB window %d of %s is held by another process (the next free one is %d)' % (self.window_id, self.bdf, tlb_id))
        finally:
            for tlb_id in taken:
                self._free(tlb_id)

    def _free(self, tlb_id: int) -> None:
        try:
            self.ops.ioctl(self._fd, IOCTL_FREE_TLB, bytearray(struct.pack('<I', tlb_id)))
        except OSError:
            pass

    def _configure(self, bank: BankPlan) -> None:
        """CONFIGURE_TLB: three posted register writes with no read-back (the proof is the canary read that follows)."""
        buffer = bytearray(CONFIGURE_IN.size + 8)
        CONFIGURE_IN.pack_into(buffer, 0, self.window_id, 0, 0, bank.noc_x, bank.noc_y, 0, 0, 0, 0, ORDERINGS[self.ordering], 0, 0, 0, 0)
        self._ioctl(IOCTL_CONFIGURE_TLB, buffer, 'CONFIGURE_TLB')

    def _pin(self) -> None:
        """Export the window as a dma-buf nobody imports: the export keeps the window allocated past FREE_TLB and close(), and while it lives the driver refuses a device reset."""
        buffer = bytearray(EXPORT_ARG.size)
        EXPORT_ARG.pack_into(buffer, 0, EXPORT_ARG.size, 0, self.window_id, -1, 0, 0)
        try:
            self.ops.ioctl(self._fd, IOCTL_EXPORT_TLB_DMABUF, buffer)
        except OSError as error:
            raise CardLinkUnavailable('the window of %s cannot be pinned (EXPORT_TLB_DMABUF: %s): without the pin a close or a reset under a live mapping is not stopped' % (self.bdf, error))
        self._pin_fd = EXPORT_ARG.unpack_from(buffer, 0)[3]

    def _map_slice(self) -> None:
        """Map the WHOLE published slice (the pool is one allocation: a second mapping fails with ENOMEM, which is the lease). Byte x of the mapping is byte x of the slice."""
        try:
            self._p2p_fd = self.ops.open_allocate(self.allocate_path)
            self._slice_map = self.ops.mmap_slice(self._p2p_fd, self._slice_size)
        except (OSError, ValueError) as error:
            raise CardLinkUnavailable('cannot map %s (%s): it needs a writable p2pmem directory in this container and a free slice' % (self.allocate_path, error))
        self._view = memoryview(self._slice_map)
        self._base_address = _address_of(self._slice_map)
        left = self.ops.read_text(os.path.join(self.sysfs_pci, self.bdf, 'p2pmem', 'available'))
        self.info['available_after_map'] = int(left, 0) if left is not None and left.strip().isdigit() else None
        if self.info['available_after_map'] not in (0, None):
            logger.warning('ttkvp2p: after mapping the whole slice of %s, p2pmem reports %s bytes still available', self.bdf, self.info['available_after_map'])

    # -- the canaries ----------------------------------------------------------------------------------------------
    def _cpu_read(self, window_off: int, size: int) -> bytes:
        return bytes(self._cpu[window_off:window_off + size])

    def _cpu_write(self, window_off: int, data: bytes) -> None:
        self._cpu[window_off:window_off + len(data)] = data

    def _write_canaries(self) -> None:
        """A fresh canary page in every bank. The nonce is new for every open, so a canary left in the DRAM by an earlier process (or a write that
        went to the wrong bank) cannot satisfy the read-back."""
        prefixes = set()
        for bank in self.banks:
            page = canary_page(self._nonce, self.bdf, bank)
            self._canaries[bank] = page
            prefixes.add(page[:CANARY_COMPARE_BYTES])
        if len(prefixes) != len(self.banks):
            raise CardLinkError('two banks got the same canary prefix')
        for bank in self.banks:
            plan = self._plan[bank]
            self._configure(plan)
            self._cpu_read(plan.canary_window_off, CANARY_COMPARE_BYTES)     # a non-posted read after the posted register writes, before the data write
            self._cpu_write(plan.canary_window_off, self._canaries[bank])

    def _prove_canaries(self) -> None:
        """Before any data goes into the card: aim at every bank again and read its canary page back in full. A window that does not move when
        told to (the same canary at every bank, or the wrong one) is found here, while only a page has been written."""
        for bank in self.banks:
            plan = self._plan[bank]
            self._configure(plan)
            got = self._cpu_read(plan.canary_window_off, CANARY_BYTES)
            if got != self._canaries[bank]:
                others = [other for other in self.banks if other != bank and got[:CANARY_COMPARE_BYTES] == self._canaries[other][:CANARY_COMPARE_BYTES]]
                self.stats['canary_mismatches'] += 1
                raise CanaryMismatch('bank %d of %s: the canary page read back after aiming the window is %s: the window is not where it was aimed'
                                     % (bank, self.bdf, "bank %s's" % others[0] if others else 'nobody\'s'))
        self._aimed = self.banks[-1]
        self._epoch += 1

    # -- aim -------------------------------------------------------------------------------------------------------
    def aim(self, bank: int) -> None:
        """Point window 202 at ``bank`` and prove it: CONFIGURE_TLB, then an 8-byte non-posted CPU read of the bank's canary through the UC mapping. The caller
        guarantees no bio is in flight on this card (and this refuses if one is counted). A different value raises ``CanaryMismatch`` and poisons the link."""
        with self._lock:
            self._require_open()
            if self._inflight:
                raise AimWhileBusy('re-aim of %s to bank %s with %d bio(s) in flight' % (self.bdf, bank, self._inflight))
            bank = as_index(bank, 'bank')
            plan = self._plan.get(bank)
            if plan is None:
                raise BoundsViolation('bank %r is not a staging bank of %s (banks %s)' % (bank, self.bdf, list(self.banks)))
            self._aimed = None
            self._configure(plan)
            got = self._cpu_read(plan.canary_window_off, CANARY_COMPARE_BYTES)
            if got != self._canaries[bank][:CANARY_COMPARE_BYTES]:
                self.stats['canary_mismatches'] += 1
                self._poisoned = 'the canary read after aiming at bank %d was %s' % (bank, got.hex())
                raise CanaryMismatch('after aiming the window of %s at bank %d the canary read %s: the window is not where it was aimed' % (self.bdf, bank, got.hex()))
            self._aimed = bank
            self._epoch += 1
            self.stats['aims'] += 1

    def begin_io(self, bank: int, epoch: int) -> None:
        """Count a bio (or a batch of them) in flight on this card, for an I/O planned at ``(bank, epoch)``. ``StaleAim`` if the window is not aimed there now."""
        with self._lock:
            self._require_open()
            if self._aimed != bank or self._epoch != epoch:
                raise StaleAim('I/O planned for bank %s epoch %s, but the window of %s is at bank %s epoch %s' % (bank, epoch, self.bdf, self._aimed, self._epoch))
            self._inflight += 1

    def end_io(self) -> None:
        with self._lock:
            if self._inflight <= 0:
                raise CardLinkError('end_io without begin_io on %s' % self.bdf)
            self._inflight -= 1

    # -- the DMA side ----------------------------------------------------------------------------------------------
    def expected_slice_offset(self, bank: int, obj: int) -> int:
        """Where object ``obj`` of ``bank``'s slot is in the p2pmem mapping, bounds-checked."""
        bank = as_index(bank, 'bank')
        plan = self._plan.get(bank)
        if plan is None:
            raise BoundsViolation('bank %r is not a staging bank of %s (banks %s)' % (bank, self.bdf, list(self.banks)))
        obj = as_index(obj, 'object')
        if obj < 0 or obj >= self.slot_objects:
            raise BoundsViolation('object %r is outside the %d-object slot of bank %d' % (obj, self.slot_objects, bank))
        return plan.data_slice_off + obj * self.shard_object_bytes

    def slot_view(self, bank: int, first_obj: int, n_obj: int) -> List[memoryview]:
        """``n_obj`` views, one per object (``shard_object_bytes`` each), of objects ``first_obj`` .. in bank ``bank``'s slot of the p2pmem mapping. They are for the
        drive's DMA only: the CPU never reads or writes them. Anything outside the slot (a bank that is not staging, an object index below 0 or at or past the slot,
        a count below 1) raises ``BoundsViolation``. The caller releases the views."""
        with self._lock:
            self._require_open()
        first_obj, n_obj = as_index(first_obj, 'first_obj'), as_index(n_obj, 'n_obj')
        if n_obj < 1:
            raise BoundsViolation('slot_view of %d objects' % n_obj)
        if first_obj < 0 or first_obj + n_obj > self.slot_objects:
            raise BoundsViolation('objects [%d, %d) are outside the %d-object slot of bank %s' % (first_obj, first_obj + n_obj, self.slot_objects, bank))
        first = self.expected_slice_offset(bank, first_obj)
        size = self.shard_object_bytes
        view = self._view
        return [view[first + j * size:first + (j + 1) * size] for j in range(n_obj)]

    def locate(self, view: Any) -> Optional[int]:
        """The offset of a buffer's first byte in this card's p2pmem mapping, or None if the buffer does not lie wholly inside it. Lets the I/O layer prove
        where an iovec really points, whatever view code produced it."""
        try:
            address = _address_of(view)
            nbytes = view.nbytes if isinstance(view, memoryview) else memoryview(view).nbytes
        except (TypeError, ValueError, BufferError):
            return None
        offset = address - self._base_address
        if offset < 0 or offset + nbytes > self._slice_size:
            return None
        return offset

    # -- the CPU side: flush and the scratch page the DMA canary uses --------------------------------------------
    def flush(self) -> None:
        """Advisory: one 8-byte non-posted read through the UC window, after a load's bios returned and before the first device op over the data. A posted
        write finishing at the drive is not an ordering point on the card's NOC, and a read that comes back is the usual flush; it does not strictly order
        another requester's writes, so the digest stays the arbiter (design 2.3). Never raises."""
        with self._lock:
            if self._state != 'open' or self._poisoned or self._cpu is None:
                return
            bank = self._aimed if self._aimed is not None else self.banks[0]
            plan = self._plan[bank]
            try:
                got = self._cpu_read(plan.canary_window_off, CANARY_COMPARE_BYTES)
            except (OSError, ValueError):
                self.stats['flush_errors'] += 1
                return
            self.stats['flushes'] += 1
            if got != self._canaries[bank][:CANARY_COMPARE_BYTES]:
                self.stats['flush_mismatches'] += 1

    def scratch_write(self, bank: int, data: bytes) -> None:
        """CPU write of up to one page through the UC window at the start of ``bank``'s slot (object 0). For the DMA canary only; the bank must be the aimed one."""
        with self._lock:
            self._require_open()
            plan = self._scratch_plan(bank, len(data))
            self._cpu_write(plan.canary_window_off + CANARY_BYTES, bytes(data))

    def scratch_read(self, bank: int, size: int) -> bytes:
        with self._lock:
            self._require_open()
            plan = self._scratch_plan(bank, size)
            return self._cpu_read(plan.canary_window_off + CANARY_BYTES, size)

    def _scratch_plan(self, bank: int, size: int) -> BankPlan:
        bank = as_index(bank, 'bank')
        if self._aimed != bank:
            raise StaleAim('scratch access to bank %s while the window of %s is at bank %s' % (bank, self.bdf, self._aimed))
        if self._inflight:
            raise AimWhileBusy('scratch access on %s with %d bio(s) in flight' % (self.bdf, self._inflight))
        if not 0 < size <= PAGE:
            raise BoundsViolation('scratch access of %d bytes (at most one page)' % size)
        return self._plan[bank]

    # -- close -----------------------------------------------------------------------------------------------------
    def abandon(self) -> None:
        """A bio of this card may still be in flight and cannot be cancelled: ``close`` will release nothing (the process is going away; the export keeps the
        window pinned and the kernel tears the mappings down at exit)."""
        self._abandoned = True

    def close(self) -> None:
        with self._lock:
            if self._state == 'closed':
                return
            self._state = 'closed'
            abandoned = self._abandoned
        if abandoned:
            logger.warning('ttkvp2p: the link to %s is left open: a bio may still be in flight', self.bdf)
            return
        if self._view is not None:
            try:
                self._view.release()
            except BufferError:
                pass
            self._view = None
        for name in ('_slice_map', '_cpu'):
            mapped = getattr(self, name)
            if mapped is None:
                continue
            for _attempt in (0, 1):
                try:
                    mapped.close()
                    break
                except BufferError:
                    gc.collect()                    # (a view of a finished I/O that has not been collected yet still exports the mapping)
                except (OSError, ValueError):
                    break
            setattr(self, name, None)
        for name in ('_p2p_fd', '_pin_fd'):
            descriptor = getattr(self, name)
            if descriptor is not None:
                try:
                    self.ops.close_fd(descriptor)
                except OSError:
                    pass
                setattr(self, name, None)
        if self._window_held and self._fd is not None:
            self._free(self.window_id)
            self._window_held = False
        if self._fd is not None:
            try:
                self.ops.close_fd(self._fd)
            except OSError:
                pass
            self._fd = None
