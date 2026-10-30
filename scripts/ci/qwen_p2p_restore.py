"""Peer-to-peer KV restore, host side (docs/tp4-p2p-restore.md): NVMe raw namespace -> O_DIRECT read into PCI p2pmem -> the card's BAR4 TLB window -> a staging buffer in card DRAM ->
a device op copies it into the paged KV cache. No host bounce. The device ops (ttnn.qwen_p2p_staging_alloc / _info, qwen_staging_checksum, qwen_write_blocks_from_staging) are the
device-op branch's; the kernel side is the tt-kmd p2pmem patch (patches/tt-kmd-p2pmem); this module is everything the host does between them. Stdlib + numpy + ctypes.

  * THE OBJECT. The NVMe tier stores one object per KV block. Under the host tier's own layout ('packed-v1') it is [cache tensor][chip][slice], all four chips interleaved, a chip's
    slice 17,408 bytes (4.25 pages): no chip's bytes are 4 KiB-aligned, so no card could read its own share. The P2P layout ('chipmajor-v1', QWEN_PREFIX_NVME_P2P=1) stores the same
    bytes as [chip][cache tensor][slice]: chip d's SHARD of the block is 32 x 17,408 = 557,056 = 136 x 4,096 bytes at d x 557,056 inside the object, aligned for O_DIRECT, one
    contiguous read per card per block. ShardLayout converts both ways.
  * THE CHECKSUM. The device cannot run sha256; it runs a position-weighted sum (INTERFACE section 4): p2p_checksum(). The spill thread records one per shard beside the object's
    sha256 (the store entry's `aux`), the card computes the same over the staging bytes after the read, and a difference is a miss (nothing is written).
  * THE WINDOW. TtLink opens /dev/tenstorrent/<n>, takes the published 4 GiB TLB window (ALLOCATE_TLB until the id the p2pmem sysfs names comes up), aims it at the staging buffer's
    NOC tile and address (CONFIGURE_TLB), pins it (EXPORT_TLB_DMABUF: the driver then refuses a reset), maps it for the CPU and proves with a canary that it stores, and only then maps
    the published slice (p2pmem/allocate, the WHOLE slice, so mapping offset == slice offset == window offset). The fd lives as long as the engine.
  * THE READ. Stream: chunks of up to `chunk_blocks` blocks; for every block and every card one pread of that card's shard straight into the card's p2pmem mapping, `threads` in
    flight over all cards; two halves of the staging buffer, so chunk k+1 is on the wire while chunk k is checked and copied. Completion of a chunk = all its preads returned.
  * THE CHECK AND THE COPY. Per chunk: one checksum op over the chunk's shards (compared with the stored values; a mismatch is retried once after a device sync, because a posted
    write may be a moment behind the drive's completion), then one write op per cache tensor with `row_stride` = the shard size. Nothing is written for a chunk that failed.

Nothing here decides policy: qwen_kv_nvme.P2PBackend does (flags, fallbacks, counters).
"""

import ctypes
import fcntl
import gc
import glob
import mmap
import os
import re
import struct
import threading
import time

import numpy as np

PAGE = 4096
GIB = 1 << 30
WINDOW_4G = 4 * GIB
FIRST_4G_TLB = 202                    # Blackhole: 202 windows of 2 MiB, then 8 of 4 GiB
NOC_COORDINATE_LIMIT = 64             # the TLB register's x and y fields are 6 bits wide
ORDERINGS = {'relaxed': 0, 'strict': 1, 'posted': 2}
SYSFS_CLASS = '/sys/class/tenstorrent'
SYSFS_PCI = '/sys/bus/pci/devices'
DEV_DIR = '/dev/tenstorrent'

# tt-kmd ioctl ABI (ioctl.h): _IO(0xFA, n), the argument is a struct passed by pointer. (scripts/p2p/p2p_probe.py holds the same numbers; test_qwen_p2p_restore pins them equal.)
TT_MAGIC = 0xFA
IOCTL_ALLOCATE_TLB = (TT_MAGIC << 8) | 11
IOCTL_FREE_TLB = (TT_MAGIC << 8) | 12
IOCTL_CONFIGURE_TLB = (TT_MAGIC << 8) | 13
IOCTL_SET_POWER_STATE = (TT_MAGIC << 8) | 15
IOCTL_EXPORT_TLB_DMABUF = (TT_MAGIC << 8) | 16
ALLOCATE_IN = struct.Struct('<QQ')
ALLOCATE_OUT = struct.Struct('<IIQQQ')
CONFIGURE_IN = struct.Struct('<IIQHHHHBBBBB3x2I')
POWER_STATE = struct.Struct('<IIBBH14H')
POWER_FLAG_MRISC_PHY_WAKEUP = 1 << 1
POWER_FLAG_TENSIX_ENABLE = 1 << 2
POWER_FLAG_L2CPU_ENABLE = 1 << 3
BUSY_FLAGS = POWER_FLAG_MRISC_PHY_WAKEUP | POWER_FLAG_TENSIX_ENABLE | POWER_FLAG_L2CPU_ENABLE
BUSY_VALIDITY = 4
EXPORT_ARG = struct.Struct('<IIIiQQ')

CHECKSUM_MULTIPLIER = 2654435761
CHECKSUM_BLOCK_WORDS = 1 << 20        # words weighed per numpy pass
LAYOUT_CHIPMAJOR = 'chipmajor-v1'
DEFAULT_THREADS = 8
DEFAULT_STAGING_BYTES = 128 << 20     # two halves of 64 MiB per card
CANARY_BYTES = PAGE


class P2PError(RuntimeError):
    """The peer-to-peer path failed; the tier falls back to the host backend (never worse than a miss)."""


class P2PUnavailable(P2PError):
    """The path cannot be set up here (a missing op, sysfs entry, window, mapping, or a failed canary). Raised at construction; the tier serves from the host backend."""


class P2PReadError(P2PError):
    """A read from the drive into a card failed."""


class P2PDeviceError(P2PError):
    """A device op raised, or answered something that is not a checksum."""


class P2PMismatch(P2PError):
    """The card's checksum over what the drive delivered differs from the one stored at spill. `bad` lists (position, chip) pairs. The restore is a miss; nothing was written for it."""

    miss_only = True               # the restore is a miss, and the host tier's failure count (three latch it) is not the path's to move

    def __init__(self, bad, entries=()):
        P2PError.__init__(self, 'device checksum differs from the stored one for %d shard(s) (first: block position %d, chip %d)' % (len(bad), bad[0][0], bad[0][1]))
        self.bad = list(bad)
        self.entries = list(entries)


# ================================================================================================
# The checksum (INTERFACE section 4)
# ================================================================================================
_weights = {}


def _weights_for(start, count):
    """w_k = (k * 2654435761 + 1) mod 2^32 for k in [start, start + count), as uint64. (k * multiplier may wrap uint64: the low 32 bits are the same.)"""
    if start == 0 and count in _weights:
        return _weights[count]
    k = np.arange(start, start + count, dtype=np.uint64)
    w = (k * np.uint64(CHECKSUM_MULTIPLIER) + np.uint64(1)) & np.uint64(0xFFFFFFFF)
    if start == 0 and count <= CHECKSUM_BLOCK_WORDS and len(_weights) < 8:
        _weights[count] = w
    return w


def p2p_checksum(data):
    """The device checksum, exactly: words x_k = little-endian uint32 at byte 4k; w_k = (k * 2654435761 + 1) mod 2^32; S = sum(w_k * x_k) mod 2^64, k counted from the first byte of `data`.
    `data` is bytes-like (bytes, bytearray, memoryview, numpy array) with a length that is a multiple of 4 (the device op takes multiples of 4096). Returns an int."""
    view = memoryview(data)
    if view.format != 'B' or view.ndim != 1:
        view = view.cast('B')
    if len(view) % 4:
        raise ValueError('a checksum is over whole 32-bit words: %d bytes' % len(view))
    words = np.frombuffer(view, dtype='<u4')
    total = np.uint64(0)
    with np.errstate(over='ignore'):
        for start in range(0, words.size, CHECKSUM_BLOCK_WORDS):
            part = words[start:start + CHECKSUM_BLOCK_WORDS].astype(np.uint64)
            part *= _weights_for(start, part.size)
            total = total + part.sum(dtype=np.uint64)
    return int(total)


def checksum_vectors():
    """The three vectors both sides test against (INTERFACE section 4): (name, bytes) pairs. The expected values are in scripts/ci/p2p_checksum_vectors.json."""
    return [('zero_4096', bytes(4096)),
            ('mod251_4096', bytes(index % 251 for index in range(4096))),
            ('prng_1mib', np.random.default_rng(20261030).integers(0, 256, 1 << 20, dtype=np.uint8).tobytes())]


# ================================================================================================
# The layout: [cache][chip][slice] (the host tier's block) <-> [chip][cache][slice] (the P2P object)
# ================================================================================================
class ShardLayout(object):
    """The geometry of one KV block: `caches` cache tensors, `chips` cards, `slice_bytes` per (cache, chip). The P2P object is the block with the chip axis first; chip d's SHARD is
    `caches * slice_bytes` contiguous bytes at d * shard_bytes. For the production geometry a shard is 557,056 bytes = 136 x 4,096."""

    def __init__(self, caches, chips, slice_bytes):
        self.caches, self.chips, self.slice_bytes = int(caches), int(chips), int(slice_bytes)
        if min(self.caches, self.chips, self.slice_bytes) < 1:
            raise ValueError('a KV block has at least one cache tensor, one chip and one byte per slice')
        self.shard_bytes = self.caches * self.slice_bytes
        self.block_bytes = self.chips * self.shard_bytes
        self.aux_bytes = 8 * self.chips

    @classmethod
    def of_io(cls, io):
        """The layout of the model's tier IO (_QwenKvTierIO: caches, chips, slice_bytes); None when the IO does not say (a test double without a geometry)."""
        inner = getattr(io, 'io', None)                # (a wrapper such as the threaded tier IO holds the model's IO as .io)
        found = [source for source in (io, inner) if source is not None and getattr(source, 'caches', None) is not None and getattr(source, 'chips', None)
                 and getattr(source, 'slice_bytes', None)]
        if not found:
            return None
        source = found[0]
        caches = source.caches
        layout = cls(len(caches) if not isinstance(caches, int) else caches, source.chips, source.slice_bytes)
        if getattr(io, 'block_bytes', layout.block_bytes) != layout.block_bytes:
            raise ValueError('the IO says a block is %d bytes, its geometry makes %d' % (io.block_bytes, layout.block_bytes))
        return layout

    def aligned(self, align=PAGE):
        return self.shard_bytes % align == 0

    def _shape(self, buffer, chip_first):
        array = np.frombuffer(buffer, dtype=np.uint8)
        if array.size != self.block_bytes:
            raise ValueError('a block of %d bytes, the layout holds %d' % (array.size, self.block_bytes))
        return array.reshape((self.chips, self.caches, self.slice_bytes) if chip_first else (self.caches, self.chips, self.slice_bytes))

    def to_chipmajor(self, payload, out):
        """The host tier's block `payload` ([cache][chip][slice]) into `out` (a writable buffer of block_bytes) as [chip][cache][slice]."""
        target = np.frombuffer(out, dtype=np.uint8)
        target = target.reshape((self.chips, self.caches, self.slice_bytes))
        target[...] = self._shape(payload, False).transpose(1, 0, 2)
        return out

    def from_chipmajor(self, obj, out):
        """A stored object ([chip][cache][slice]) into `out` (a writable buffer of block_bytes) as the host tier's block ([cache][chip][slice])."""
        target = np.frombuffer(out, dtype=np.uint8).reshape((self.caches, self.chips, self.slice_bytes))
        target[...] = self._shape(obj, True).transpose(1, 0, 2)
        return out

    def shard(self, obj, chip):
        """Chip's shard of a stored object, as a view."""
        view = memoryview(obj)
        if view.format != 'B' or view.ndim != 1:
            view = view.cast('B')
        return view[chip * self.shard_bytes:(chip + 1) * self.shard_bytes]

    def checksums(self, obj):
        """The per-chip checksums of a stored object (a tuple of `chips` ints)."""
        return tuple(p2p_checksum(self.shard(obj, chip)) for chip in range(self.chips))

    def pack_checksums(self, sums):
        if len(sums) != self.chips:
            raise ValueError('%d checksums for %d chips' % (len(sums), self.chips))
        return struct.pack('<%dQ' % self.chips, *sums)

    def unpack_checksums(self, aux):
        if aux is None or len(aux) != self.aux_bytes:
            raise ValueError('an entry\'s checksums are %d bytes, this one has %s' % (self.aux_bytes, 'none' if not aux else len(aux)))
        return struct.unpack('<%dQ' % self.chips, bytes(aux))


# ================================================================================================
# Where the window goes: pure arithmetic
# ================================================================================================
def plan_aim(info, staging, ordering='relaxed'):
    """Where a card's staging buffer is in its published slice, and how its TLB window must be aimed. `info` is the p2pmem sysfs record (bar, offset, size, tlb_first, tlb_last);
    `staging` is the device op's answer for this card (noc_x, noc_y, address: the byte address inside the DRAM bank, size). A 4 GiB window starts on a 4 GiB boundary of the NOC address
    space, so the window is aimed at the staging address rounded down to 4 GiB; byte x of the slice is at BAR4 offset info.offset + x, at window offset (that - the window's base), at NOC
    address base + that. Raises P2PUnavailable for anything that does not hold together."""
    if info['bar'] != 4:
        raise P2PUnavailable('the published slice is in BAR%d; only BAR4 (the 4 GiB windows) is a DRAM aperture' % info['bar'])
    if info['tlb_first'] < 0 or info['tlb_first'] != info['tlb_last']:
        raise P2PUnavailable('the published slice must lie in one 4 GiB window, the driver says windows %d..%d' % (info['tlb_first'], info['tlb_last']))
    window = info['tlb_first']
    if window < FIRST_4G_TLB:
        raise P2PUnavailable('TLB window %d is not one of the 4 GiB windows' % window)
    x, y, address, size = (int(staging[name]) for name in ('noc_x', 'noc_y', 'address', 'size'))
    for name, value in (('noc_x', x), ('noc_y', y)):
        if not 0 <= value < NOC_COORDINATE_LIMIT:
            raise P2PUnavailable('%s %d does not fit the 6-bit coordinate field of a TLB register' % (name, value))
    if address < 0 or size <= 0 or size % PAGE:
        raise P2PUnavailable('the staging buffer (%#x, %d bytes) is not a whole number of pages' % (address, size))
    noc_base = address - address % WINDOW_4G
    in_window = address - noc_base
    if in_window + size > WINDOW_4G:
        raise P2PUnavailable('the staging buffer [%#x, +%d) crosses a 4 GiB boundary: one window cannot cover it' % (address, size))
    window_base = (window - FIRST_4G_TLB) * WINDOW_4G
    slice_offset = in_window - (info['offset'] - window_base)
    if slice_offset < 0 or slice_offset + size > info['size']:
        raise P2PUnavailable('the staging buffer [%#x, +%d) is outside the published slice (window offset %d, slice [%d, +%d))'
                             % (address, size, in_window, info['offset'] - window_base, info['size']))
    if slice_offset % PAGE:
        raise P2PUnavailable('the staging buffer is %d bytes into the published slice; O_DIRECT needs it page-aligned' % slice_offset)
    return {'window': window, 'noc': {'x': x, 'y': y, 'addr': noc_base, 'ordering': ORDERINGS[ordering]}, 'slice_offset': slice_offset, 'window_offset': in_window, 'size': size}


def _sysfs_text(path):
    try:
        with open(path) as handle:
            return handle.read().strip()
    except OSError:
        return None


def normalise_bdf(text):
    """'0000:41:00.0' from '41:00.0' or an upper-case form; ValueError otherwise."""
    value = str(text).strip().lower()
    if re.fullmatch(r'[0-9a-f]{2}:[0-9a-f]{2}\.[0-7]', value):
        value = '0000:' + value
    if not re.fullmatch(r'[0-9a-f]{4}:[0-9a-f]{2}:[0-9a-f]{2}\.[0-7]', value):
        raise ValueError('%r is not a PCI address' % (text,))
    return value


def card_table(sysfs_class=SYSFS_CLASS):
    """{PCI address: /dev/tenstorrent index} of the cards the driver exposes, found at run time: the nodes renumber on every reset."""
    table = {}
    for path in glob.glob(os.path.join(sysfs_class, 'tenstorrent!*')):
        found = re.fullmatch(r'tenstorrent!(\d+)', os.path.basename(path))
        if not found:
            continue
        device = os.path.realpath(os.path.join(path, 'device'))
        try:
            table[normalise_bdf(os.path.basename(device))] = int(found.group(1))
        except ValueError:
            continue
    return table


def locate_cards(infos, sysfs_class=SYSFS_CLASS):
    """[(device index, PCI address)] in mesh order, from the staging buffer's per-device records: `pci_bus_id` (a PCI address) or `pcie_device_id` / `tt_device_index` (the N of
    /dev/tenstorrent/N), each resolved against the driver's own table. P2PUnavailable when a record names neither, or names a card the driver does not know."""
    table = card_table(sysfs_class)
    by_index = {index: bdf for bdf, index in table.items()}
    out = []
    for position, info in enumerate(infos):
        bdf = info.get('pci_bus_id')
        index = info.get('pcie_device_id', info.get('tt_device_index'))
        if bdf:
            try:
                bdf = normalise_bdf(bdf)
            except ValueError as error:
                raise P2PUnavailable('device %d: %s' % (position, error))
            if bdf not in table:
                raise P2PUnavailable('device %d is %s, which the driver does not list under %s' % (position, bdf, sysfs_class))
            out.append((table[bdf], bdf))
        elif index is not None:
            if int(index) not in by_index:
                raise P2PUnavailable('device %d is /dev/tenstorrent/%s, which the driver does not list' % (position, index))
            out.append((int(index), by_index[int(index)]))
        else:
            raise P2PUnavailable('the staging record of device %d names neither pci_bus_id nor pcie_device_id: the card cannot be found' % position)
    if len(set(index for index, _ in out)) != len(out):
        raise P2PUnavailable('two mesh devices resolve to the same card')
    return out


def read_p2pmem(index, bdf, sysfs_class=SYSFS_CLASS, sysfs_pci=SYSFS_PCI):
    """The published slice of a card: the driver's record (bar, offset, size, bus_addr, tlb_first, tlb_last) and the PCI core's size and available. P2PUnavailable when the patched
    module is not loaded for this card."""
    base = os.path.join(sysfs_class, 'tenstorrent!%d' % index, 'p2pmem')
    info = {'bdf': bdf, 'index': index}
    for name in ('bar', 'offset', 'size', 'tlb_first', 'tlb_last'):
        text = _sysfs_text(os.path.join(base, name))
        if text is None:
            raise P2PUnavailable('%s/%s is missing: the patched tt-kmd is not loaded, or its p2pmem_devices does not name %s' % (base, name, bdf))
        info[name] = int(text)
    text = _sysfs_text(os.path.join(base, 'bus_addr'))
    info['bus_addr'] = int(text, 0) if text else 0
    for name in ('size', 'available'):
        text = _sysfs_text(os.path.join(sysfs_pci, bdf, 'p2pmem', name))
        if text is None:
            raise P2PUnavailable('%s/%s/p2pmem/%s is unreadable: the kernel has no peer-to-peer memory for the card' % (sysfs_pci, bdf, name))
        info['core_' + name] = int(text)
    return info


def _pagemap_physical(address, pagemap='/proc/self/pagemap'):
    """The CPU physical address behind a mapped virtual address, 0 when the kernel will not say (it needs CAP_SYS_ADMIN)."""
    try:
        with open(pagemap, 'rb') as handle:
            handle.seek(address // PAGE * 8)
            entry = struct.unpack('<Q', handle.read(8))[0]
    except (OSError, struct.error):
        return 0
    if not entry & (1 << 63):
        return 0
    return (entry & ((1 << 55) - 1)) * PAGE + address % PAGE


# ================================================================================================
# One card
# ================================================================================================
class TtLink(object):
    """A real card's end of the path: the fd, the TLB window aimed at the staging buffer, and the mapping of the published slice. Construct with open(). Everything a caller needs
    afterwards is `view(staging_offset, length)` (a window onto the card's staging bytes in the p2pmem mapping), `selftest()` and `close()`."""

    cpu_window_bytes = WINDOW_4G          # what the CPU mapping of the window covers (a test shrinks it)

    def __init__(self, index, bdf, logger=None, sysfs_class=SYSFS_CLASS, sysfs_pci=SYSFS_PCI, dev_dir=DEV_DIR):
        self.index, self.bdf = index, bdf
        self.log = logger or (lambda message, *values: None)
        self.sysfs_class, self.sysfs_pci, self.dev_dir = sysfs_class, sysfs_pci, dev_dir
        self.fd = None
        self.window_id = None
        self.pin_fd = None
        self.cpu = None
        self.p2p_fd = None
        self.p2p_map = None
        self.buffer = None
        self.aim = None
        self.info = None
        self.map_verified = False
        self.closed = False

    @classmethod
    def open(cls, index, bdf, staging, ordering='relaxed', settle=0.5, logger=None, sysfs_class=SYSFS_CLASS, sysfs_pci=SYSFS_PCI, dev_dir=DEV_DIR, power=True):
        link = cls(index, bdf, logger, sysfs_class, sysfs_pci, dev_dir)
        try:
            link._open(staging, ordering, settle, power)
        except BaseException:
            link.close()
            raise
        return link

    def _open(self, staging, ordering, settle, power):
        try:
            self.fd = os.open(os.path.join(self.dev_dir, str(self.index)), os.O_RDWR | getattr(os, 'O_CLOEXEC', 0))
        except OSError as error:
            raise P2PUnavailable('cannot open %s/%d: %s' % (self.dev_dir, self.index, error))
        self.info = read_p2pmem(self.index, self.bdf, self.sysfs_class, self.sysfs_pci)
        if self.info['core_available'] != self.info['core_size']:
            raise P2PUnavailable('the published slice of %s is already allocated (available %d of %d): another holder has it' % (self.bdf, self.info['core_available'], self.info['core_size']))
        if self.info['core_size'] != self.info['size']:
            raise P2PUnavailable('the PCI core\'s p2pmem size %d differs from the driver\'s %d' % (self.info['core_size'], self.info['size']))
        self.aim = plan_aim(self.info, staging, ordering)
        if power:
            self._power_up(settle)
        self._acquire_window(self.aim['window'])
        noc = self.aim['noc']
        self._configure(noc['x'], noc['y'], noc['addr'], noc['ordering'])
        self._pin()
        self._map_cpu_window()
        self.selftest()
        self._map_p2pmem()

    # -- the driver's ioctls -----------------------------------------------------------------------
    def _ioctl(self, request, buffer, what):
        try:
            fcntl.ioctl(self.fd, request, buffer, True)
        except OSError as error:
            raise P2PUnavailable('%s failed on %s: %s' % (what, self.bdf, error))

    def _power_up(self, settle):
        """The busy power state (the DRAM PHY awake), as the UMD asks when it opens a card: the state is this fd's and is withdrawn when the fd closes, so the PHY stays up for as long
        as the engine holds the link even if the UMD's fd goes."""
        buffer = bytearray(POWER_STATE.size)
        POWER_STATE.pack_into(buffer, 0, POWER_STATE.size, 0, 0, BUSY_VALIDITY, BUSY_FLAGS, *([0] * 14))
        self._ioctl(IOCTL_SET_POWER_STATE, buffer, 'SET_POWER_STATE')
        if settle:
            time.sleep(settle)

    def _acquire_window(self, wanted):
        """Own TLB window `wanted`: ALLOCATE_TLB hands out the lowest free 4 GiB window, so take windows until it comes up and give the others back."""
        taken = []
        try:
            while True:
                buffer = bytearray(ALLOCATE_IN.size + ALLOCATE_OUT.size)
                ALLOCATE_IN.pack_into(buffer, 0, WINDOW_4G, 0)
                self._ioctl(IOCTL_ALLOCATE_TLB, buffer, 'ALLOCATE_TLB (window %d)' % wanted)
                tlb_id, _, _, mmap_wc, _ = ALLOCATE_OUT.unpack_from(buffer, ALLOCATE_IN.size)
                if tlb_id == wanted:
                    self.window_id, self.mmap_offset_wc = tlb_id, mmap_wc
                    return
                taken.append(tlb_id)
                if tlb_id > wanted:
                    raise P2PUnavailable('TLB window %d of %s is held by another process (the next free one is %d)' % (wanted, self.bdf, tlb_id))
        finally:
            for tlb_id in taken:
                self._free(tlb_id)

    def _free(self, tlb_id):
        try:
            fcntl.ioctl(self.fd, IOCTL_FREE_TLB, bytearray(struct.pack('<I', tlb_id)), True)
        except OSError:
            pass

    def _configure(self, x, y, addr, ordering):
        buffer = bytearray(CONFIGURE_IN.size + 8)
        CONFIGURE_IN.pack_into(buffer, 0, self.window_id, 0, addr, x, y, 0, 0, 0, 0, ordering, 0, 0, 0, 0)
        self._ioctl(IOCTL_CONFIGURE_TLB, buffer, 'CONFIGURE_TLB')

    def _pin(self):
        """Export the window as a dma-buf nobody imports: the export keeps the window allocated past FREE_TLB and close(), and while it lives the driver refuses a device reset."""
        buffer = bytearray(EXPORT_ARG.size)
        EXPORT_ARG.pack_into(buffer, 0, EXPORT_ARG.size, 0, self.window_id, -1, 0, 0)
        try:
            fcntl.ioctl(self.fd, IOCTL_EXPORT_TLB_DMABUF, buffer, True)
        except OSError as error:
            raise P2PUnavailable('the window of %s cannot be pinned (EXPORT_TLB_DMABUF: %s): without the pin a close or a reset under a live mapping is not stopped' % (self.bdf, error))
        self.pin_fd = EXPORT_ARG.unpack_from(buffer, 0)[3]

    def _map_cpu_window(self):
        try:
            self.cpu = mmap.mmap(self.fd, self.cpu_window_bytes, flags=mmap.MAP_SHARED, prot=mmap.PROT_READ | mmap.PROT_WRITE, offset=self.mmap_offset_wc)
        except (OSError, ValueError) as error:
            raise P2PUnavailable('cannot map TLB window %d of %s: %s' % (self.window_id, self.bdf, error))

    # -- the canary --------------------------------------------------------------------------------
    def selftest(self):
        """Before any DMA: a pattern written through the window reads back from it at both ends of the staging buffer, and the two ends do not alias. It shows the window is aimed at
        something that stores, not at what. It writes only inside the staging buffer (nothing is staged between restores). Never call it while a read is in flight."""
        first = bytes(range(256)) * (CANARY_BYTES // 256)
        second = bytes(reversed(range(256))) * (CANARY_BYTES // 256)
        base, size = self.aim['window_offset'], self.aim['size']
        far = base + size - CANARY_BYTES
        self.cpu[base:base + CANARY_BYTES] = first
        self.cpu[far:far + CANARY_BYTES] = second
        if bytes(self.cpu[base:base + CANARY_BYTES]) != first or bytes(self.cpu[far:far + CANARY_BYTES]) != second:
            raise P2PUnavailable('the window of %s does not read back what was written through it: the TLB is not pointing at memory' % self.bdf)

    # -- the published slice -----------------------------------------------------------------------
    def _map_p2pmem(self):
        """Map the WHOLE published slice (the pool is one allocation: a second mapping fails with ENOMEM, which is the lease). Pages come from the pool lowest first, so a mapping
        that starts at the pool's start puts byte x of the mapping at byte x of the slice, at window offset x: nothing else makes the offset arithmetic safe, and the DMA writes
        wherever the pages are."""
        size = self.info['size']
        path = os.path.join(self.sysfs_pci, self.bdf, 'p2pmem', 'allocate')
        try:
            self.p2p_fd = os.open(path, os.O_RDWR | getattr(os, 'O_CLOEXEC', 0))
            self.p2p_map = mmap.mmap(self.p2p_fd, size, flags=mmap.MAP_SHARED, prot=mmap.PROT_READ | mmap.PROT_WRITE)
        except (OSError, ValueError) as error:
            raise P2PUnavailable('cannot map %s (%s): it needs root, a writable p2pmem directory in this container, and a free slice' % (path, error))
        self.buffer = memoryview(self.p2p_map)
        self.map_verified = self._verify_mapping()

    def _verify_mapping(self):
        """True when /proc/self/pagemap shows the mapping's first page, the staging buffer's first and last page at the physical addresses the BAR arithmetic says; False when the kernel
        will not say (no CAP_SYS_ADMIN); P2PUnavailable when it says and they differ (the DMA would land elsewhere in the bank)."""
        resource = _sysfs_text(os.path.join(self.sysfs_pci, self.bdf, 'resource'))
        if not resource:
            return False
        try:
            lines = resource.splitlines()
            bar4 = int(lines[4].split()[0], 16)
        except (IndexError, ValueError):
            return False
        base = ctypes.addressof(ctypes.c_char.from_buffer(self.p2p_map))
        checked = False
        for offset in (0, self.aim['slice_offset'], self.aim['slice_offset'] + self.aim['size'] - PAGE):
            found = _pagemap_physical(base + offset)
            if not found:
                return False
            expected = bar4 + self.info['offset'] + offset
            if found != expected:
                raise P2PUnavailable('the p2pmem mapping of %s is at physical %#x at offset %d, the slice is at %#x: the DMA would not land where the window points' % (self.bdf, found, offset, expected))
            checked = True
        return checked

    def view(self, staging_offset, length):
        """The bytes [staging_offset, +length) of the staging buffer as a writable view of the p2pmem mapping (what O_DIRECT reads into). The caller releases it."""
        start = self.aim['slice_offset'] + staging_offset
        return self.buffer[start:start + length]

    def close(self):
        if self.closed:
            return
        self.closed = True
        if self.buffer is not None:
            try:
                self.buffer.release()
            except BufferError:
                pass
        for what in ('p2p_map', 'cpu'):
            mapped = getattr(self, what)
            if mapped is not None:
                for _attempt in (0, 1):
                    try:
                        mapped.close()
                        break
                    except BufferError:
                        gc.collect()                    # (a view of a finished read that has not been collected yet still exports the mapping)
                    except (OSError, ValueError):
                        break
        if self.p2p_fd is not None:
            os.close(self.p2p_fd)
        if self.pin_fd is not None:
            os.close(self.pin_fd)
        if self.window_id is not None and self.fd is not None:
            self._free(self.window_id)
        if self.fd is not None:
            os.close(self.fd)
        self.fd = self.p2p_fd = self.pin_fd = None


# ================================================================================================
# The device ops
# ================================================================================================
class StagingApi(object):
    """The device-op branch's ops behind one adapter (a ttnn module or a fake):
         qwen_p2p_staging_alloc(mesh, size) -> handle           qwen_p2p_staging_info(handle) -> [{device_id, pci_bus_id | pcie_device_id, dram_bank, noc_x, noc_y, address, size}] per device
         qwen_staging_checksum(handle, byte_offset, length, count) -> per device `count` uint64: the checksums of the consecutive ranges [byte_offset + i * length, + length)
         qwen_write_blocks_from_staging(cache, handle, byte_offset, blocks, row_stride) -> row i of device d's [len(blocks), slice_bytes] = staging_d[byte_offset + i * row_stride : + slice_bytes]
       and ttnn.synchronize_device(mesh)."""

    REQUIRED = ('qwen_p2p_staging_alloc', 'qwen_p2p_staging_info', 'qwen_staging_checksum', 'qwen_write_blocks_from_staging')

    def __init__(self, ttnn, mesh, tensors):
        missing = [name for name in self.REQUIRED if not callable(getattr(ttnn, name, None))]
        if missing:
            raise P2PUnavailable('this ttnn lacks %s (the device-op extension is older than the peer-to-peer restore)' % ', '.join(missing))
        self.ttnn, self.mesh, self.tensors = ttnn, mesh, list(tensors)
        self.handle = None

    def alloc(self, size):
        try:
            self.handle = self.ttnn.qwen_p2p_staging_alloc(self.mesh, int(size))
        except Exception as error:
            raise P2PUnavailable('qwen_p2p_staging_alloc(%d) failed: %s: %s' % (size, type(error).__name__, error))
        return self.handle

    def info(self):
        try:
            return [dict(record) for record in self.ttnn.qwen_p2p_staging_info(self.handle)]
        except Exception as error:
            raise P2PUnavailable('qwen_p2p_staging_info failed: %s: %s' % (type(error).__name__, error))

    def checksums(self, byte_offset, length, count, chips):
        """[[int] * count] per device. Whatever the op returns (lists, a numpy array, a torch tensor) is read as a (chips, count) matrix of unsigned 64-bit integers."""
        try:
            raw = self.ttnn.qwen_staging_checksum(self.handle, int(byte_offset), int(length), int(count))
        except Exception as error:
            raise P2PDeviceError('qwen_staging_checksum failed: %s: %s' % (type(error).__name__, error))
        try:
            rows = raw.tolist() if hasattr(raw, 'tolist') else list(raw)
            matrix = []
            for row in rows:
                row = row.tolist() if hasattr(row, 'tolist') else row
                matrix.append([int(item) & 0xFFFFFFFFFFFFFFFF for item in (row if isinstance(row, (list, tuple)) else [row])])
        except (TypeError, ValueError) as error:
            raise P2PDeviceError('qwen_staging_checksum answered something that is not %d x %d integers (%s)' % (chips, count, error))
        if len(matrix) != chips or any(len(row) != count for row in matrix):
            raise P2PDeviceError('qwen_staging_checksum answered %s, wanted %d devices x %d checksums' % ([len(row) for row in matrix], chips, count))
        return matrix

    def write(self, tensor_index, byte_offset, block_ids, row_stride):
        try:
            self.ttnn.qwen_write_blocks_from_staging(self.tensors[tensor_index], self.handle, int(byte_offset), [int(block) for block in block_ids], row_stride=int(row_stride))
        except Exception as error:
            raise P2PDeviceError('qwen_write_blocks_from_staging (tensor %d) failed: %s: %s' % (tensor_index, type(error).__name__, error))

    def sync(self):
        try:
            self.ttnn.synchronize_device(self.mesh)
        except Exception as error:
            raise P2PDeviceError('synchronize_device failed: %s: %s' % (type(error).__name__, error))


# ================================================================================================
# The engine and its stream
# ================================================================================================
class Chunk(object):
    """Up to chunk_blocks consecutive positions of a restore: the ones with a stored object (`pending`: [(position, entry)]) are read into one half of the staging buffer, slot j
    at half + j * shard_bytes on every card."""

    __slots__ = ('index', 'first', 'count', 'pending', 'half', 'offset', 'todo', 'error', 'done', 'lock', 'began', 'ended', 'ops')

    def __init__(self, index, first, count, pending):
        self.index, self.first, self.count, self.pending = index, first, count, pending
        self.half = None
        self.offset = 0
        self.todo = 0
        self.error = None
        self.done = threading.Event()
        self.lock = threading.Lock()
        self.began = self.ended = None
        self.ops = False                       # a device op has been issued over this chunk's half

    @property
    def positions(self):
        return range(self.first, self.first + self.count)

    def finish_task(self, error=None):
        with self.lock:
            if error is not None and self.error is None:
                self.error = error
            self.todo -= 1
            last = self.todo <= 0
            if last:
                self.ended = time.perf_counter()
        if last:
            self.done.set()


STAT_NAMES = ('restores', 'chunks', 'blocks', 'bytes', 'read_ms', 'check_ms', 'write_ms', 'mismatch_retries', 'mismatches', 'read_errors', 'device_errors')


class Stream(object):
    """One restore in progress: its chunks in position order, the reads in flight, the halves of the staging buffer. Used from one thread (the scheduler's); the reads run on the engine's
    pool. `take` hands out the next chunk whose reads all returned; `verify` and `write` run the device ops for it; `release` gives its half back; `close` waits for the reads still in
    flight (they cannot be cancelled and they write into the staging buffer)."""

    def __init__(self, engine, items, chunk_blocks=None, chips=None):
        self.engine = engine
        self.items = list(items)
        self.chips = tuple(range(engine.layout.chips)) if chips is None else tuple(sorted(set(chips)))
        if not self.chips or not set(self.chips) <= set(range(engine.layout.chips)):
            raise ValueError('chips %s are not among the engine\'s %d' % (list(self.chips), engine.layout.chips))
        step = max(1, min(engine.chunk_blocks, int(chunk_blocks or engine.chunk_blocks)))
        self.chunks = []
        for first in range(0, len(self.items), step):
            part = self.items[first:first + step]
            self.chunks.append(Chunk(len(self.chunks), first, len(part), [(first + offset, item) for offset, item in enumerate(part) if item is not None]))
        self.submitted = 0                     # chunks whose reads were started
        self.taken = 0                         # chunks handed out
        self.closed = False
        engine.stats['restores'] += 1
        self.pump()

    def __len__(self):
        return len(self.chunks)

    @property
    def finished(self):
        return self.taken >= len(self.chunks)

    def pump(self):
        """Start the reads of the next chunks while a half of the staging buffer is free. A half a device op was reading is synced first (the drive's DMA ignores the device's queue)."""
        engine = self.engine
        while self.submitted < len(self.chunks) and not self.closed:
            chunk = self.chunks[self.submitted]
            if not chunk.pending:
                self.submitted += 1
                chunk.done.set()
                continue
            half = engine.claim_half()
            if half is None:
                return
            self.submitted += 1
            self._submit(chunk, half)

    def _submit(self, chunk, half):
        engine = self.engine
        chunk.half = half
        chunk.offset = half * engine.half_bytes
        chunk.todo = len(chunk.pending) * len(self.chips)
        chunk.began = time.perf_counter()
        with engine.inflight_lock:
            engine.inflight += chunk.todo
        for slot, (_position, entry) in enumerate(chunk.pending):
            for chip in self.chips:
                engine.pool.submit(engine.read_task, chunk, slot, chip, entry)

    def ready(self):
        """Whether the next chunk's reads have all returned (no wait)."""
        return self.taken < len(self.chunks) and self.chunks[self.taken].done.is_set()

    def take(self, wait=False, timeout=None):
        """The next chunk, once its reads returned: None when it is not (and `wait` is false, or the wait ran out), or when there are no more. P2PReadError when a read failed."""
        if self.taken >= len(self.chunks):
            return None
        self.pump()
        chunk = self.chunks[self.taken]
        if chunk.index >= self.submitted:
            if not wait:
                return None
            raise P2PError('the stream has no free half to read the next chunk into')       # (cannot happen: a taken chunk is released before the next take)
        if not chunk.done.is_set():
            if not wait or not chunk.done.wait(timeout):
                return None
        self.taken += 1
        if chunk.error is not None:
            self.engine.stats['read_errors'] += 1
            error = chunk.error
            self.release(chunk)
            raise P2PReadError('a read of chunk %d (blocks %d..%d) failed: %s: %s' % (chunk.index, chunk.first, chunk.first + chunk.count - 1, type(error).__name__, error))
        if chunk.began is not None and chunk.ended is not None:
            self.engine.stats['read_ms'] += (chunk.ended - chunk.began) * 1000.0
        return chunk

    def verify(self, chunk):
        """The cards' checksums over the chunk's staged shards against the stored ones. A difference is retried once after a device sync and a moment (a posted write can trail the
        drive's completion); a second difference raises P2PMismatch and nothing is written for the chunk."""
        if not chunk.pending:
            return
        engine = self.engine
        layout, stats = engine.layout, engine.stats
        wanted = []
        for position, entry in chunk.pending:
            try:
                wanted.append(layout.unpack_checksums(entry.aux))
            except ValueError as error:
                raise P2PError('block %d: %s' % (position, error))
        began = time.perf_counter()
        chunk.ops = True
        bad = self._compare(chunk, wanted)
        if bad:
            stats['mismatch_retries'] += 1
            engine.api.sync()
            engine.sleep(engine.recheck_s)
            bad = self._compare(chunk, wanted)
        stats['check_ms'] += (time.perf_counter() - began) * 1000.0
        if bad:
            stats['mismatches'] += len(bad)
            raise P2PMismatch([(chunk.pending[slot][0], chip) for slot, chip in bad], [chunk.pending[slot][1] for slot in sorted(set(slot for slot, _ in bad))])

    def _compare(self, chunk, wanted):
        engine = self.engine
        got = engine.api.checksums(chunk.offset, engine.layout.shard_bytes, len(chunk.pending), engine.layout.chips)
        return [(slot, chip) for slot in range(len(chunk.pending)) for chip in self.chips if got[chip][slot] != wanted[slot][chip]]

    def write(self, chunk, block_ids):
        """Enqueue the copy of the chunk's staged shards into the KV cache: one op per cache tensor, `block_ids` the device block of each pending position in order."""
        if not chunk.pending:
            return
        engine = self.engine
        if len(block_ids) != len(chunk.pending):
            raise ValueError('%d device blocks for %d staged blocks' % (len(block_ids), len(chunk.pending)))
        if len(self.chips) != engine.layout.chips:
            raise P2PError('a stream over chips %s staged only part of each block: the copy op writes every chip' % list(self.chips))
        layout = engine.layout
        began = time.perf_counter()
        chunk.ops = True
        for tensor in range(layout.caches):
            engine.api.write(tensor, chunk.offset + tensor * layout.slice_bytes, block_ids, layout.shard_bytes)
        engine.stats['write_ms'] += (time.perf_counter() - began) * 1000.0
        engine.stats['chunks'] += 1
        engine.stats['blocks'] += len(chunk.pending)
        engine.stats['bytes'] += len(chunk.pending) * layout.block_bytes

    def release(self, chunk):
        """The chunk is done with (its writes enqueued, or abandoned): its half can take the next read once the device has finished with it."""
        if chunk.half is not None:
            self.engine.return_half(chunk.half, dirty=chunk.ops)
            chunk.half = None

    def close(self, timeout=60.0):
        """Wait for the reads in flight (the drive's DMA into the staging buffer cannot be taken back), then stop. True when they all returned in time."""
        self.closed = True
        ok = True
        end = time.monotonic() + timeout
        for chunk in self.chunks[:self.submitted]:
            if chunk.pending and not chunk.done.wait(max(0.0, end - time.monotonic())):
                ok = False
        for chunk in self.chunks[:self.submitted]:
            if chunk.half is not None and chunk.done.is_set():
                self.release(chunk)             # (a chunk read but never taken, or taken and not released: its half goes back)
        return ok


class P2PEngine(object):
    """The path for one mesh: the staging buffer, a TtLink per card, the reader pool. Raises P2PUnavailable from the constructor when anything is missing (the tier then serves from
    the host backend). `reader(offset, view)` fills a writable view from the drive at an aligned offset (qwen_nvme_store's BlockFile.pread_into)."""

    def __init__(self, api, layout, reader, threads=DEFAULT_THREADS, staging_bytes=DEFAULT_STAGING_BYTES, ordering='relaxed', link_factory=None, expected=None, logger=None,
                 sleep=time.sleep, recheck_s=0.002, io_align=PAGE, sysfs_class=SYSFS_CLASS, settle=0.5):
        self.api, self.layout, self.reader = api, layout, reader
        self.log = logger or (lambda message, *values: None)
        self.sleep, self.recheck_s, self.io_align = sleep, recheck_s, io_align
        self.stats = dict.fromkeys(STAT_NAMES, 0)
        self.close_wait_s = 3.0
        self.inflight = 0                      # shard reads submitted and not yet returned
        self.inflight_lock = threading.Lock()
        self.links = []
        self.pool = None
        self.closed = False
        self.threads = max(1, int(threads))
        if layout.shard_bytes % io_align:
            raise P2PUnavailable('a shard is %d bytes, not a multiple of the %d-byte direct-I/O alignment' % (layout.shard_bytes, io_align))
        self.halves = 2
        self.free_halves = list(range(self.halves))
        self.dirty = set()                     # halves a device op may still be reading (the drive's DMA must not overwrite them before a sync)
        self.chunk_blocks = (int(staging_bytes) // self.halves) // layout.shard_bytes
        if self.chunk_blocks < 1:
            raise P2PUnavailable('a staging buffer of %d bytes cannot hold two shards of %d' % (staging_bytes, layout.shard_bytes))
        self.half_bytes = self.chunk_blocks * layout.shard_bytes
        self.staging_bytes = self.half_bytes * self.halves
        try:
            api.alloc(self.staging_bytes)
            infos = api.info()
            if len(infos) != layout.chips:
                raise P2PUnavailable('the staging buffer answers for %d devices, the KV block is split over %d' % (len(infos), layout.chips))
            for position, info in enumerate(infos):
                for name in ('noc_x', 'noc_y', 'address', 'size'):
                    if info.get(name) is None:
                        raise P2PUnavailable('the staging record of device %d lacks %s' % (position, name))
                if int(info['size']) < self.staging_bytes:
                    raise P2PUnavailable('device %d staged %d bytes, %d were asked for' % (position, int(info['size']), self.staging_bytes))
            self.infos = infos
            self.cards = locate_cards(infos, sysfs_class) if link_factory is None else [(position, info.get('pci_bus_id')) for position, info in enumerate(infos)]
            if expected:
                have = set(bdf for _, bdf in self.cards)
                if have != set(expected):
                    raise P2PUnavailable('the mesh runs on %s, the configuration names %s' % (sorted(have), sorted(expected)))
            for position, (index, bdf) in enumerate(self.cards):
                staging = dict(infos[position], size=self.staging_bytes)
                if link_factory is not None:
                    link = link_factory(position, index, bdf, staging, ordering)
                else:
                    link = TtLink.open(index, bdf, staging, ordering, settle=settle, logger=self.log, sysfs_class=sysfs_class)
                self.links.append(link)
            self.pool = _make_pool(self.threads)
        except BaseException:
            self.close()
            raise

    # -- the reads ---------------------------------------------------------------------------------
    def read_task(self, chunk, slot, chip, entry):
        """One shard: chip's part of one stored object into that card's staging slot. Runs on the pool."""
        error = None
        view = None
        try:
            if chunk.error is None and not self.closed:
                view = self.links[chip].view(chunk.offset + slot * self.layout.shard_bytes, self.layout.shard_bytes)
                self.reader(entry.offset + chip * self.layout.shard_bytes, view)
        except BaseException as caught:         # (the chunk records it; the scheduler thread raises)
            error = caught
        finally:
            if view is not None:
                try:
                    view.release()
                except BufferError:
                    pass
        with self.inflight_lock:
            self.inflight -= 1
        chunk.finish_task(error)

    # -- the halves of the staging buffer ----------------------------------------------------------
    def claim_half(self):
        """A free half for the next chunk's reads, or None. A half a device op was reading is synced first (the drive's DMA ignores the device's queue)."""
        if not self.free_halves:
            return None
        half = self.free_halves.pop(0)
        if half in self.dirty:
            self.api.sync()
            self.dirty.clear()
        return half

    def return_half(self, half, dirty):
        if dirty:
            self.dirty.add(half)
        if half not in self.free_halves:
            self.free_halves.append(half)

    # -- use ---------------------------------------------------------------------------------------
    def start(self, items, chunk_blocks=None, chips=None):
        """A Stream over `items`: one stored entry (or None, for a position that needs no read) per position of the restore, in position order. `chunk_blocks` shrinks the chunks
        (the sliced restore's block cap); it cannot enlarge them past what a half of the staging buffer holds. `chips` reads and checks only those cards' shares (the first DMA of
        a bring-up on one card; such a stream cannot write)."""
        if self.closed:
            raise P2PError('the peer-to-peer engine is closed')
        return Stream(self, items, chunk_blocks, chips)

    def restore(self, entries, block_ids):
        """The whole restore, synchronously: every entry read, checked and copied into block_ids[i]. Raises P2PMismatch (nothing was written for the failing chunk), P2PReadError or
        P2PDeviceError. The device has the writes enqueued on return; sync() waits for them."""
        if len(entries) != len(block_ids):
            raise ValueError('%d device blocks for %d stored blocks' % (len(block_ids), len(entries)))
        stream = self.start(entries)
        try:
            while True:
                chunk = stream.take(wait=True)
                if chunk is None:
                    break
                try:
                    stream.verify(chunk)
                    stream.write(chunk, [block_ids[position] for position, _entry in chunk.pending])
                finally:
                    stream.release(chunk)
        finally:
            stream.close()
        self.api.sync()
        self.dirty.clear()

    def selftest(self):
        for link in self.links:
            link.selftest()

    def describe(self):
        link = self.links[0] if self.links else None
        return dict(cards=len(self.links), threads=self.threads, staging_bytes=self.staging_bytes, chunk_blocks=self.chunk_blocks, shard_bytes=self.layout.shard_bytes,
                    mapping_verified=int(all(getattr(item, 'map_verified', False) for item in self.links)) if self.links else 0, window=getattr(link, 'window_id', None))

    def close(self, wait_s=None):
        """Stop the pool, wait (at most `wait_s`, close_wait_s by default) for the shard reads in flight to return, then give the cards back. A read that does not return in that time
        is left: the process is going away, and a bio in flight holds its pages until the drive answers."""
        if self.closed:
            return
        self.closed = True
        if self.pool is not None:
            self.pool.shutdown(wait=False)                  # (tasks still queued run as no-ops once `closed` is set: read_task checks it)
            end = time.monotonic() + (self.close_wait_s if wait_s is None else wait_s)
            while self.inflight > 0 and time.monotonic() < end:
                time.sleep(0.005)
            if self.inflight > 0:
                self.log('p2p engine closed with %d shard read(s) still in flight', self.inflight)
        for link in self.links:
            try:
                if link is not None:
                    link.close()
            except Exception:
                pass


def _make_pool(threads):
    from concurrent.futures import ThreadPoolExecutor

    return ThreadPoolExecutor(max_workers=threads, thread_name_prefix='qwen-p2p-read')
