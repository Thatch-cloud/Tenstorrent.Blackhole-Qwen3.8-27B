#!/usr/bin/env python3
"""p2p_probe: can an NVMe drive DMA straight into a Blackhole card's DRAM, and how fast?

The probe reads a source (a file, or a range of a block device) with O_DIRECT into peer-to-peer memory published by a
patched tt-kmd (patches/tt-kmd-p2pmem), so the drive writes into the card over PCIe and the data never lands in host RAM.
It then checks every byte it can reach, and times the same read through host RAM (the drive into a host buffer, then CPU
stores into the card) as an indicative baseline.

What it needs, all of it checked before a single DMA is issued:
  * the patched tt-kmd loaded with p2pmem_devices naming this card, so /sys/class/tenstorrent/tenstorrent!N/p2pmem exists
    (Linux 6.2 or later, for the p2pmem/allocate mmap and O_DIRECT into it), and nothing else holding the card open;
  * the card awake: the driver puts every card in its low-power state (DRAM PHY down) when no client is open, so the probe sends the
    same busy power state the UMD does, and keeps its fd open for the whole run;
  * the 4 GiB TLB window that covers the published slice: the probe allocates it, points it at a DRAM bank (CONFIGURE_TLB), pins it
    (EXPORT_TLB_DMABUF, so a reset is refused and FREE_TLB or close cannot pull it from under a live mapping) and keeps it;
  * a source whose bytes are all written to the drive: a hole or an unwritten extent would be zero-filled by the CPU into the
    peer-to-peer pages, which the kernel forbids, so a regular file is checked with FIEMAP first.

Nothing is touched when --dry-run is given, and --mock runs the whole flow against in-memory fakes (no card, no drive, no root), which
is what the CPU tests do.

The card's DRAM is overwritten from --dest-offset on. Run it only in a window in which nothing serves from the card, and reset the card
afterwards. The published range is mapped uncached by the kernel, so CPU access through the window (the verification, the baseline's
stores) is slow; the baseline is indicative of "through host memory", not of the UMD's DMA path.

Exit status: 0 PASS, 1 FAIL (a difference, or below --min-gbps), 2 could not run, 3 UNVERIFIED (--verify none: nothing was checked).

Stdlib only.
"""

import argparse
import concurrent.futures
import ctypes
import fcntl
import json
import mmap
import os
import random
import re
import stat
import struct
import sys
import time

PAGE = 4096
GIB = 1 << 30
WINDOW_4G = 4 * GIB
SENTINEL = 0xA5
MIN_SIZE = 2 * PAGE            # the window self-test writes at the start and at the end of the destination

# tt-kmd ioctl ABI (ioctl.h): _IO(0xFA, n), the argument is a struct passed by pointer.
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
POWER_FLAG_MRISC_PHY_WAKEUP = 1 << 1      # the DRAM PHY: asleep unless a client asks for it
POWER_FLAG_TENSIX_ENABLE = 1 << 2
POWER_FLAG_L2CPU_ENABLE = 1 << 3
BUSY_FLAGS = POWER_FLAG_MRISC_PHY_WAKEUP | POWER_FLAG_TENSIX_ENABLE | POWER_FLAG_L2CPU_ENABLE      # what the UMD asks for when it opens a card
BUSY_VALIDITY = 4                                                                                  # four flags specified, no settings
# struct tenstorrent_export_tlb_dmabuf { u32 argsz; u32 flags; u32 tlb_id; s32 fd; u64 offset; u64 size; }
EXPORT_ARG = struct.Struct('<IIIiQQ')

FIRST_4G_TLB = 202       # Blackhole: 202 windows of 2 MiB, then 8 of 4 GiB
ORDERINGS = {'relaxed': 0, 'strict': 1, 'posted': 2}
NOC_COORDINATE_LIMIT = 64        # the TLB register's x and y fields are 6 bits wide

FS_IOC_FIEMAP = 0xC020660B
FIEMAP_FLAG_SYNC = 1
FIEMAP_HEADER = struct.Struct('<QQIIII')
FIEMAP_EXTENT = struct.Struct('<QQQQQIIII')
FIEMAP_EXTENT_LAST = 0x1
FIEMAP_MAX_EXTENTS = 32768
# Extents the kernel would not read from the drive into our buffer: unknown or delayed allocation, encoded or encrypted by
# the filesystem, inlined in the inode, a tail, or allocated but never written.
FIEMAP_HAZARDS = 0x2 | 0x4 | 0x8 | 0x80 | 0x200 | 0x400 | 0x800

PASS, FAIL, UNVERIFIED = 'PASS', 'FAIL', 'UNVERIFIED'
EXIT_CODES = {PASS: 0, FAIL: 1, UNVERIFIED: 3}
NOTE_UNCACHED = ('CPU access through the window is uncached once the range is p2pmem (the kernel maps it uncached and reserves the range): the '
                 'verification is slow and the host-bounce baseline is indicative only, not the UMD DMA path')


class ProbeError(RuntimeError):
    """A precondition failed. The message says which."""


def parse_size(text):
    """'4096', '64K', '16M', '2G' -> bytes."""
    text = str(text).strip().upper()
    scale = {'K': 1 << 10, 'M': 1 << 20, 'G': 1 << 30, 'T': 1 << 40}
    if text and text[-1] in scale:
        return int(float(text[:-1]) * scale[text[-1]])
    return int(text, 0)


def dram_bank_xy(bank, port=0):
    """Translated NOC coordinates of a Blackhole DRAM bank's port (the translation is on by default): banks 0-3 are in column 17,
    banks 4-7 in column 18, three ports per bank going down from row 12. From the UMD's Blackhole coordinate manager."""
    if not 0 <= bank < 8 or not 0 <= port < 3:
        raise ValueError('a Blackhole has DRAM banks 0-7 with ports 0-2')
    return 17 + bank // 4, 12 + 3 * (bank % 4) + port


# -- the plan: pure arithmetic, no I/O ---------------------------------------------------------------------------------

def make_plan(info, size, dest_offset, chunk, source_offset, noc_x, noc_y, noc_addr, ordering, sector=PAGE):
    """Where the bytes will go, and every refusal that does not need the hardware. `info` is the p2pmem sysfs record
    (bar, offset, size, bus_addr, tlb_first, tlb_last)."""
    if info['bar'] != 4:
        raise ProbeError('the published slice is in BAR%d; only BAR4 (the 4 GiB windows) is a DRAM aperture' % info['bar'])
    if info['tlb_first'] < 0 or info['tlb_first'] != info['tlb_last']:
        raise ProbeError('the published slice [%d, +%d) must lie in one 4 GiB TLB window, the kernel reports windows %d..%d; '
                         'publish a smaller slice' % (info['offset'], info['size'], info['tlb_first'], info['tlb_last']))
    window = info['tlb_first']
    if window < FIRST_4G_TLB:
        raise ProbeError('TLB window %d is not one of the 4 GiB windows' % window)
    window_base = (window - FIRST_4G_TLB) * WINDOW_4G
    if noc_addr < 0 or noc_addr % WINDOW_4G:
        raise ProbeError('--noc-addr %#x is not a multiple of 4 GiB: a 4 GiB window can only start on a 4 GiB boundary' % noc_addr)
    for name, value in (('--noc-x', noc_x), ('--noc-y', noc_y)):
        if not 0 <= value < NOC_COORDINATE_LIMIT:
            raise ProbeError('%s %d does not fit the 6-bit coordinate field of a TLB register' % (name, value))
    for name, value in (('--size', size), ('--chunk', chunk), ('--dest-offset', dest_offset), ('--source-offset', source_offset)):
        if value < 0:
            raise ProbeError('%s %d is negative' % (name, value))
        if value % sector:
            raise ProbeError('%s %d is not a multiple of the %d-byte sector O_DIRECT needs' % (name, value, sector))
    if size < MIN_SIZE or chunk <= 0:
        raise ProbeError('--size must be at least %d (the window self-test needs room at both ends) and --chunk positive' % MIN_SIZE)
    chunk = min(chunk, size)
    if dest_offset + size > info['size']:
        raise ProbeError('the destination [%d, +%d) is outside the published slice (%d bytes)' % (dest_offset, size, info['size']))
    # Byte x of the slice is at BAR4 offset info['offset'] + x, hence at window offset (that - window_base), hence at NOC address
    # noc_addr + that: the window is configured with `noc_addr` as its base.
    window_offset = info['offset'] - window_base
    return {
        'window': window,
        'window_bar4_offset': window_base,
        'noc': {'x': noc_x, 'y': noc_y, 'addr': noc_addr, 'ordering': ordering},
        'slice': {'bar4_offset': info['offset'], 'size': info['size'], 'bus_addr': info['bus_addr']},
        'dest': {'slice_offset': dest_offset, 'window_offset': window_offset + dest_offset,
                 'noc_addr': noc_addr + window_offset + dest_offset, 'size': size},
        'chunk': chunk,
        'chunks': (size + chunk - 1) // chunk,
        'source_offset': source_offset,
    }


# -- the source --------------------------------------------------------------------------------------------------------

def fiemap_extents(fd, start, length, max_extents=FIEMAP_MAX_EXTENTS):
    """The extents of an open regular file over [start, start+length): a list of (logical, physical, length, flags). A file with more extents
    than fit is refused rather than reported short."""
    buf = bytearray(FIEMAP_HEADER.size + max_extents * FIEMAP_EXTENT.size)
    FIEMAP_HEADER.pack_into(buf, 0, start, length, FIEMAP_FLAG_SYNC, 0, max_extents, 0)
    fcntl.ioctl(fd, FS_IOC_FIEMAP, buf, True)
    _, _, _, mapped, _, _ = FIEMAP_HEADER.unpack_from(buf, 0)
    extents = []
    for index in range(mapped):
        logical, physical, extent_length, _, _, flags, _, _, _ = FIEMAP_EXTENT.unpack_from(
            buf, FIEMAP_HEADER.size + index * FIEMAP_EXTENT.size)
        extents.append((logical, physical, extent_length, flags))
    if mapped >= max_extents and not extents[-1][3] & FIEMAP_EXTENT_LAST:
        raise ProbeError('the file has more than %d extents: too fragmented to prove it has no hole' % max_extents)
    return extents


def check_extents(extents, start, length):
    """Raise unless the extents map every byte of [start, start+length) to written blocks on the drive. A hole, delayed allocation,
    an unwritten extent, inline data or filesystem-side encoding would make the kernel fill our buffer by CPU, which it must never
    do into peer-to-peer pages."""
    cursor = start
    end = start + length
    for logical, _, extent_length, flags in extents:
        if logical + extent_length <= cursor:
            continue
        if logical > cursor:
            raise ProbeError('hole in the source at %d..%d: the kernel would zero it by CPU into the peer-to-peer pages' % (cursor, logical))
        if flags & FIEMAP_HAZARDS:
            raise ProbeError('source extent at %d has flags %#x (unwritten, delayed, inline or encoded): rewrite the file with plain '
                             'writes and fsync, no fallocate-only or sparse ranges' % (logical, flags))
        cursor = logical + extent_length
        if cursor >= end:
            return
    if cursor < end:
        raise ProbeError('the source ends at %d, short of the %d bytes asked for' % (cursor, end))


class Source(object):
    """The thing the drive reads: a file or a block device, opened O_DIRECT (direct=False for the mock)."""

    def __init__(self, path, direct=True, extents=None):
        self.path = path
        flags = os.O_RDONLY | (getattr(os, 'O_DIRECT', 0) if direct else 0)
        self.fd = os.open(path, flags)
        try:
            mode = os.fstat(self.fd).st_mode
            self.is_block = stat.S_ISBLK(mode)
            if self.is_block:
                self.length = os.lseek(self.fd, 0, os.SEEK_END)
            else:
                self.length = os.fstat(self.fd).st_size
            self.buffered = os.open(path, os.O_RDONLY)
        except OSError:
            os.close(self.fd)
            raise
        self.extents = extents          # a callable (fd, start, length) -> extents, for tests; None means FIEMAP on a regular file

    def check(self, offset, size):
        if offset + size > self.length:
            raise ProbeError('the source holds %d bytes, %d are needed from %d' % (self.length, size, offset))
        if self.is_block:
            return
        provider = self.extents or fiemap_extents
        try:
            found = provider(self.fd, offset, size)
        except OSError as error:
            raise ProbeError('FIEMAP failed on %s (%s): cannot prove the source has no holes; use --allow-unchecked to go on'
                             % (self.path, error)) from error
        check_extents(found, offset, size)

    def read_into(self, view, file_offset):
        """O_DIRECT read exactly len(view) bytes at file_offset into the buffer `view`."""
        done = 0
        total = len(view)
        while done < total:
            rest = view[done:]
            try:
                count = os.preadv(self.fd, [rest], file_offset + done)
            except OSError as error:
                raise ProbeError('read of %d bytes at source offset %d failed: %s%s' % (
                    total - done, file_offset + done, error,
                    ' (the kernel refused peer-to-peer DMA to this memory: see dmesg and p2p_topology.py)' if error.errno == 121 else '')) from error
            finally:
                rest.release()
            if count <= 0:
                raise ProbeError('short read from the source at %d (%d of %d bytes)' % (file_offset + done, done, total))
            done += count
        return done

    def read_buffered(self, file_offset, size):
        data = os.pread(self.buffered, size, file_offset)
        if len(data) != size:
            raise ProbeError('short buffered read at %d' % file_offset)
        return data

    def close(self):
        os.close(self.fd)
        os.close(self.buffered)


def block_device_in_use(path, proc_mounts='/proc/mounts', proc_mdstat='/proc/mdstat', proc_swaps='/proc/swaps',
                        sys_class_block='/sys/class/block'):
    """Why a block device (or a partition of it) must not be overwritten, or None: it, or a partition of it, is mounted, used as swap, is held
    by another block device (device-mapper, md, LVM), or is named in /proc/mdstat."""
    real = os.path.realpath(path)
    name = os.path.basename(real)
    for table, what in ((proc_mounts, 'mounted'), (proc_swaps, 'in use as swap')):
        try:
            with open(table) as handle:
                for line in handle:
                    device = line.split()[0] if line.strip() else ''
                    if device.startswith('/dev/') and os.path.realpath(device).startswith(real):
                        return 'it, or a partition of it, is %s (%s)' % (what, device)
        except OSError:
            pass
    try:
        with open(proc_mdstat) as handle:
            if re.search(r'\b%s\w*\[' % re.escape(name), handle.read()):
                return 'it is part of a software RAID array'
    except OSError:
        pass
    # holders of the device and of each of its partitions
    base = os.path.join(sys_class_block, name)
    candidates = [base]
    try:
        candidates += [os.path.join(base, entry) for entry in os.listdir(base) if entry.startswith(name)]
    except OSError:
        pass
    for candidate in candidates:
        try:
            if os.listdir(os.path.join(candidate, 'holders')):
                return 'it is held by another block device (md, device-mapper or LVM)'
        except OSError:
            continue
    return None


def first_nonzero_byte(path, length=1 << 20):
    """Offset of the first byte in the first `length` bytes of `path` that is not zero, or None. A drive that carries a partition table, a
    filesystem, an array or a volume label has one; so does any drive somebody has used."""
    with open(path, 'rb') as handle:
        data = handle.read(length)
    for index in range(0, len(data), 4096):
        if data[index:index + 4096].count(0) != min(4096, len(data) - index):
            block = data[index:index + 4096]
            return index + next(i for i, byte in enumerate(block) if byte)
    return None


def prepare_source(path, size, seed=None, overwrite_device=False, overwrite_file=False):
    """Write `size` bytes of random data to `path` with plain writes and fsync, so every extent is written. A block device is written only when
    `overwrite_device` says so, nothing mounts or holds it, and its first MiB is zero (a drive with anything in it is somebody's: blkdiscard or
    wipefs it first, on the operator's decision). A regular file that already has content is replaced only with `overwrite_file`. Returns the path."""
    if os.path.exists(path) and stat.S_ISBLK(os.stat(path).st_mode):
        if not overwrite_device:
            raise ProbeError('%s is a block device: --prepare would overwrite its first %d bytes; pass --overwrite-device if that is meant' % (path, size))
        reason = block_device_in_use(path)
        if reason:
            raise ProbeError('refusing to overwrite %s: %s' % (path, reason))
        found = first_nonzero_byte(path)
        if found is not None:
            raise ProbeError('refusing to overwrite %s: it is not blank (first non-zero byte at %d: a partition table, a filesystem or data). '
                             'If it is the spare drive and its operator agrees, wipe it first (wipefs -a, blkdiscard)' % (path, found))
    elif os.path.exists(path) and os.path.getsize(path) > 0 and not overwrite_file:
        raise ProbeError('%s exists and is not empty: pass --overwrite-file to replace it' % path)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        rng = random.Random(seed) if seed is not None else None
        left = size
        while left:
            piece = min(left, 8 << 20)
            os.write(fd, rng.randbytes(piece) if rng else os.urandom(piece))
            left -= piece
        os.fsync(fd)
    finally:
        os.close(fd)
    return path


def other_openers(index, proc='/proc'):
    """Pids (other than ours) that have /dev/tenstorrent/<index> open. Needs root to see other users' descriptors."""
    target = '/dev/tenstorrent/%d' % index
    pids = []
    for entry in os.listdir(proc):
        if not entry.isdigit() or int(entry) == os.getpid():
            continue
        fd_dir = os.path.join(proc, entry, 'fd')
        try:
            names = os.listdir(fd_dir)
        except OSError:
            continue
        for name in names:
            try:
                if os.readlink(os.path.join(fd_dir, name)) == target:
                    pids.append(int(entry))
                    break
            except OSError:
                continue
    return pids


# -- the card ----------------------------------------------------------------------------------------------------------

class CpuWindow(object):
    """CPU access to the DRAM behind the TLB window, through a mapping of the window: a path to the bytes that does not go through the
    peer-to-peer memory, so it can tell whether the drive put them there."""

    def __init__(self, mapping):
        self.mapping = mapping

    def read(self, offset, size):
        return bytes(self.mapping[offset:offset + size])

    def write(self, offset, data):
        self.mapping[offset:offset + len(data)] = data

    def close(self):
        self.mapping.close()


class TtCard(object):
    """A real card: /dev/tenstorrent/N for the TLB ioctls and the window mapping, sysfs for the published slice."""

    def __init__(self, index):
        self.index = index
        self.fd = os.open('/dev/tenstorrent/%d' % index, os.O_RDWR)
        self.class_dir = '/sys/class/tenstorrent/tenstorrent!%d' % index
        self.pci_dir = os.path.realpath(os.path.join(self.class_dir, 'device'))
        self.bdf = os.path.basename(self.pci_dir)
        self.windows = {}

    def other_openers(self):
        return other_openers(self.index)

    def _sysfs(self, name):
        try:
            with open(os.path.join(self.class_dir, 'p2pmem', name)) as handle:
                return handle.read().strip()
        except OSError as error:
            raise ProbeError('%s/p2pmem/%s is missing (%s): the patched tt-kmd is not loaded, or p2pmem_devices does not name %s'
                             % (self.class_dir, name, error, self.bdf)) from error

    def _core(self, name):
        """The PCI core's p2pmem attributes (size, available), which sit on the PCI device, not on the card's class device."""
        try:
            with open(os.path.join(self.pci_dir, 'p2pmem', name)) as handle:
                return int(handle.read().strip())
        except (OSError, ValueError) as error:
            raise ProbeError('%s/p2pmem/%s is unreadable (%s)' % (self.pci_dir, name, error)) from error

    def p2pmem_info(self):
        info = {'bdf': self.bdf}
        for name in ('bar', 'offset', 'size', 'tlb_first', 'tlb_last'):
            info[name] = int(self._sysfs(name))
        info['bus_addr'] = int(self._sysfs('bus_addr'), 0)
        # in use = some of the pool is allocated: the PCI core's available is below its size
        info['in_use'] = int(self._core('available') != self._core('size'))
        return info

    def bar4_start(self):
        with open(os.path.join(self.pci_dir, 'resource')) as handle:
            lines = handle.read().splitlines()
        start, _, _ = (int(field, 16) for field in lines[4].split())
        return start

    def expected_phys(self, info):
        """The CPU physical address of the first byte of the slice, for the pagemap check."""
        return self.bar4_start() + info['offset']

    def power_up(self, settle=1.0):
        """Ask for the busy power state (the DRAM PHY awake, Tensix and the L2 CPUs enabled), as the UMD does when it opens a card, and give the
        firmware a moment. The driver puts every card in its low-power state at load and again when the last client closes; with the PHY asleep a
        write into DRAM has nowhere to go. The state is this fd's and is withdrawn when the fd closes."""
        buf = bytearray(POWER_STATE.size)
        POWER_STATE.pack_into(buf, 0, POWER_STATE.size, 0, 0, BUSY_VALIDITY, BUSY_FLAGS, *([0] * 14))
        try:
            fcntl.ioctl(self.fd, IOCTL_SET_POWER_STATE, buf, True)
        except OSError as error:
            raise ProbeError('SET_POWER_STATE failed (%s): cannot make sure the card\'s DRAM is awake; needs tt-kmd 2.6.0 or later' % error) from error
        time.sleep(settle)

    def acquire_window(self, wanted):
        """Own TLB window `wanted` (a Blackhole 4 GiB window id): ALLOCATE_TLB hands out the lowest free one, so take windows until it
        comes up and give the others back."""
        taken = []
        try:
            while True:
                buf = bytearray(ALLOCATE_IN.size + ALLOCATE_OUT.size)
                ALLOCATE_IN.pack_into(buf, 0, WINDOW_4G, 0)
                try:
                    fcntl.ioctl(self.fd, IOCTL_ALLOCATE_TLB, buf, True)
                except OSError as error:
                    raise ProbeError('ALLOCATE_TLB failed before window %d came up (%s): another process holds it'
                                     % (wanted, error)) from error
                tlb_id, _, _, mmap_wc, _ = ALLOCATE_OUT.unpack_from(buf, ALLOCATE_IN.size)
                if tlb_id == wanted:
                    self.windows[tlb_id] = mmap_wc
                    return TtWindow(self, tlb_id, mmap_wc)
                if tlb_id > wanted:
                    self._free(tlb_id)
                    raise ProbeError('window %d is held by another process (the next free one is %d)' % (wanted, tlb_id))
                taken.append(tlb_id)
        finally:
            for tlb_id in taken:
                self._free(tlb_id)

    def _free(self, tlb_id):
        fcntl.ioctl(self.fd, IOCTL_FREE_TLB, bytearray(struct.pack('<I', tlb_id)), True)

    def p2p_buffer(self, size):
        return P2PBuffer(self, size)

    def close(self):
        os.close(self.fd)


class TtWindow(object):
    def __init__(self, card, tlb_id, mmap_offset_wc):
        self.card = card
        self.id = tlb_id
        self.mmap_offset_wc = mmap_offset_wc
        self.pin_fd = None
        self.cpu = None

    def configure(self, x, y, addr, ordering):
        buf = bytearray(CONFIGURE_IN.size + 8)
        CONFIGURE_IN.pack_into(buf, 0, self.id, 0, addr, x, y, 0, 0, 0, 0, ordering, 0, 0, 0, 0)
        try:
            fcntl.ioctl(self.card.fd, IOCTL_CONFIGURE_TLB, buf, True)
        except OSError as error:
            raise ProbeError('CONFIGURE_TLB failed (%s)' % error) from error

    def pin(self):
        """Export the window as a dma-buf nobody imports: the export keeps the window allocated past FREE_TLB and close(), and while any
        export lives the driver refuses a device reset. False when the driver has no such ioctl."""
        buf = bytearray(EXPORT_ARG.size)
        EXPORT_ARG.pack_into(buf, 0, EXPORT_ARG.size, 0, self.id, -1, 0, 0)
        try:
            fcntl.ioctl(self.card.fd, IOCTL_EXPORT_TLB_DMABUF, buf, True)
        except OSError:
            return False
        self.pin_fd = EXPORT_ARG.unpack_from(buf, 0)[3]
        return True

    def cpu_window(self, size):
        try:
            mapping = mmap.mmap(self.card.fd, size, flags=mmap.MAP_SHARED, prot=mmap.PROT_READ | mmap.PROT_WRITE,
                                offset=self.mmap_offset_wc)
        except (OSError, ValueError) as error:
            raise ProbeError('cannot map TLB window %d (%s)' % (self.id, error)) from error
        self.cpu = CpuWindow(mapping)
        return self.cpu

    def close(self):
        if self.cpu is not None:
            try:
                self.cpu.close()
            except (BufferError, OSError, ValueError):
                pass
        if self.pin_fd is not None:
            os.close(self.pin_fd)
        try:
            self.card._free(self.id)
        except OSError:
            pass


class P2PBuffer(object):
    """The published slice, allocated whole: mmap of p2pmem/allocate. Offsets into it are offsets into the slice, and so into BAR4."""

    def __init__(self, card, size):
        path = os.path.join(card.pci_dir, 'p2pmem', 'allocate')
        try:
            self.fd = os.open(path, os.O_RDWR)
        except OSError as error:
            raise ProbeError('cannot open %s (%s): it needs root, Linux 6.2 or later, and the patched tt-kmd' % (path, error)) from error
        try:
            self.mapping = mmap.mmap(self.fd, size, flags=mmap.MAP_SHARED, prot=mmap.PROT_READ | mmap.PROT_WRITE)
        except (OSError, ValueError) as error:
            os.close(self.fd)
            raise ProbeError('cannot map %s (%s): the slice must be free (available equal to size) and the kernel 6.2 or later'
                             % (path, error)) from error
        self.view = memoryview(self.mapping)
        self.size = size

    def bus_addr_of(self, offset):
        """The CPU physical address behind byte `offset` of the mapping, from /proc/self/pagemap (root only; 0 when it cannot say)."""
        address = ctypes.addressof(ctypes.c_char.from_buffer(self.mapping, offset))
        try:
            with open('/proc/self/pagemap', 'rb') as handle:
                handle.seek(address // PAGE * 8)
                entry = struct.unpack('<Q', handle.read(8))[0]
        except OSError:
            return 0
        if not entry & (1 << 63):
            return 0
        return (entry & ((1 << 55) - 1)) * PAGE + offset % PAGE

    def close(self):
        try:
            self.view.release()
            self.mapping.close()
        finally:
            os.close(self.fd)


# -- the run -----------------------------------------------------------------------------------------------------------

def timed(function):
    began = time.perf_counter()
    result = function()
    return result, time.perf_counter() - began


def read_all(source, view, plan, threads, offset_in_view):
    """Read plan['dest']['size'] bytes from the source into view[offset_in_view:], chunk by chunk, `threads` at a time. Returns the
    per-chunk seconds."""
    size = plan['dest']['size']
    chunk = plan['chunk']
    jobs = [(done, min(chunk, size - done)) for done in range(0, size, chunk)]

    def one(job):
        done, count = job
        began = time.perf_counter()
        piece = view[offset_in_view + done:offset_in_view + done + count]
        try:
            source.read_into(piece, plan['source_offset'] + done)
        finally:
            piece.release()
        return time.perf_counter() - began

    if threads <= 1:
        return [one(job) for job in jobs]
    with concurrent.futures.ThreadPoolExecutor(max_workers=threads) as pool:
        return list(pool.map(one, jobs))


def gbps(size, seconds):
    return (size / seconds / 1e9) if seconds > 0 else float('inf')


def percentile(values, fraction):
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(len(ordered) * fraction))]


def sample_offsets(size, count, span, rng):
    """First, last and `count` random span-sized windows of [0, size), as sorted, non-overlapping (offset, length) pairs."""
    span = min(span, size)
    starts = {0, max(0, size - span)}
    slots = max(1, size // span)
    for _ in range(count):
        starts.add(rng.randrange(slots) * span)
    return [(start, min(span, size - start)) for start in sorted(starts)]


def verify(source, cpu, plan, mode, samples, span, seed):
    """Compare what the drive put in the card (read back through the CPU window, not through the peer-to-peer mapping) with the
    source. 'sample' reads the first and last span and `samples` random spans; 'full' reads everything (slow: a CPU read of uncached
    memory); 'none' skips. Returns a report dict; 'ok' is None when nothing was checked."""
    if mode == 'none':
        return {'mode': 'none', 'bytes_compared': 0, 'ok': None, 'mismatches': []}
    size = plan['dest']['size']
    base = plan['dest']['window_offset']
    spans = [(0, size)] if mode == 'full' else sample_offsets(size, samples, span, random.Random(seed))
    mismatches = []
    compared = 0
    for start, length in spans:
        # Read in pieces so a failure is located to a megabyte rather than to the span.
        for piece in range(0, length, 1 << 20):
            count = min(1 << 20, length - piece)
            expected = source.read_buffered(plan['source_offset'] + start + piece, count)
            found = cpu.read(base + start + piece, count)
            compared += count
            if found != expected:
                first = next(i for i in range(count) if found[i] != expected[i])
                mismatches.append({'source_offset': plan['source_offset'] + start + piece + first,
                                   'expected': expected[first], 'found': found[first]})
                if len(mismatches) >= 8:
                    break
        if len(mismatches) >= 8:
            break
    return {'mode': mode, 'bytes_compared': compared, 'ok': not mismatches, 'mismatches': mismatches}


def paint_sentinel(cpu, plan, mode, samples, span, seed):
    """Fill what verify will read with a sentinel first, so a DMA that wrote nothing cannot pass on leftovers."""
    size = plan['dest']['size']
    base = plan['dest']['window_offset']
    spans = [(0, size)] if mode == 'full' else sample_offsets(size, samples, span, random.Random(seed))
    for start, length in spans:
        for piece in range(0, length, 1 << 20):
            count = min(1 << 20, length - piece)
            cpu.write(base + start + piece, bytes([SENTINEL]) * count)


def window_selftest(cpu, plan):
    """Before any DMA: a pattern written through the window reads back from it, at both ends of the destination, and the two do not alias.
    It shows the window is configured to something that stores, not what it is configured to. It writes only inside the destination."""
    base = plan['dest']['window_offset']
    size = plan['dest']['size']
    first, second = bytes(range(256)) * 16, bytes(reversed(range(256))) * 16
    far = base + size - len(second)
    cpu.write(base, first)
    cpu.write(far, second)
    if cpu.read(base, len(first)) != first or cpu.read(far, len(second)) != second:
        raise ProbeError('the window does not read back what was written through it: the TLB is not pointing at memory')


def run_p2p(card, source, cpu, plan, args, info, report):
    """The timed peer-to-peer runs, each verified. Fills report['p2p'] and report['verify']."""
    buffer = card.p2p_buffer(info['size'])
    try:
        first_bus = buffer.bus_addr_of(0)
        expected = card.expected_phys(info)
        report['bus_addr_check'] = None if not first_bus or not expected else first_bus == expected
        if report['bus_addr_check'] is False:
            raise ProbeError('the mapping starts at physical address %#x, the slice at %#x: the DMA would not land where the TLB points'
                             % (first_bus, expected))
        runs = []
        checks = []
        for _ in range(max(1, args.repeat)):
            # A sentinel before every run: a DMA that stops writing must not pass on what an earlier run left.
            paint_sentinel(cpu, plan, args.verify, args.samples, args.span, args.seed)
            latencies, seconds = timed(lambda: read_all(source, buffer.view, plan, args.threads, plan['dest']['slice_offset']))
            runs.append((seconds, latencies))
            checks.append(verify(source, cpu, plan, args.verify, args.samples, args.span, args.seed))
    finally:
        buffer.close()
    seconds_all = sorted(seconds for seconds, _ in runs)
    median = seconds_all[len(seconds_all) // 2]
    all_latencies = [value for _, latencies in runs for value in latencies]
    report['p2p'] = {'bytes': args.size, 'seconds': median, 'gbps': gbps(args.size, median),
                     'best_gbps': gbps(args.size, seconds_all[0]), 'runs': len(runs), 'threads': args.threads,
                     'chunk': plan['chunk'], 'chunk_p50_ms': percentile(all_latencies, 0.5) * 1e3,
                     'chunk_p99_ms': percentile(all_latencies, 0.99) * 1e3}
    merged = dict(checks[-1])
    merged['mismatches'] = [dict(item, run=number) for number, check in enumerate(checks, 1) for item in check['mismatches']][:8]
    merged['ok'] = None if checks[-1]['ok'] is None else all(check['ok'] for check in checks)
    report['verify'] = merged


def run_probe(card, source, args, out=None):
    """The flow, against a card and a source (real or mock). Returns the report; raises ProbeError on a failed precondition."""
    info = card.p2pmem_info()
    if info.get('in_use'):
        raise ProbeError('the published slice is already allocated (p2pmem available is below its size): another process holds it')
    plan = make_plan(info, args.size, args.dest_offset, args.chunk, args.source_offset, args.noc_x, args.noc_y, args.noc_addr,
                     ORDERINGS[args.ordering])
    if not args.allow_unchecked:
        source.check(args.source_offset, args.size)
    elif args.source_offset + args.size > source.length:
        raise ProbeError('the source is shorter than the range asked for')
    holders = [] if args.allow_shared else card.other_openers()
    if holders:
        raise ProbeError('the card is open in process(es) %s: nothing may hold it during the test (--allow-shared overrides)'
                         % ', '.join(str(pid) for pid in holders))
    report = {'plan': plan, 'info': info, 'notes': [NOTE_UNCACHED]}

    card.power_up(args.settle)
    window = card.acquire_window(plan['window'])
    try:
        noc = plan['noc']
        window.configure(noc['x'], noc['y'], noc['addr'], noc['ordering'])
        report['pinned'] = window.pin()
        if not report['pinned'] and not args.allow_unpinned:
            raise ProbeError('the window could not be pinned with a dma-buf export (tt-kmd older than 2.10.0-rc1?): without the pin a close or FREE_TLB '
                             'under a live mapping, or a reset, would not be stopped; --allow-unpinned goes on regardless')
        cpu = window.cpu_window(WINDOW_4G)
        window_selftest(cpu, plan)
        if not args.baseline_only:
            run_p2p(card, source, cpu, plan, args, info, report)
        if args.baseline or args.baseline_only:
            report['baseline'] = run_baseline(source, cpu, plan, args.threads)
            report['baseline']['verify'] = verify(source, cpu, plan, args.verify, args.samples, args.span, args.seed)
    finally:
        window.close()
    return finish(report, args)


def finish(report, args):
    """The verdict: FAIL on any difference or a rate under the gate; UNVERIFIED when something that was run was not compared; else PASS."""
    verdicts = []
    if 'verify' in report:
        verdicts.append(report['verify']['ok'])
    if 'baseline' in report:
        verdicts.append(report['baseline']['verify']['ok'])
    gate = bool(args.min_gbps) and 'p2p' in report and report['p2p']['gbps'] < args.min_gbps
    report['below_min_gbps'] = gate
    if gate or any(value is False for value in verdicts):
        report['result'] = FAIL
    elif not verdicts or any(value is None for value in verdicts):
        report['result'] = UNVERIFIED
    else:
        report['result'] = PASS
    report['ok'] = report['result'] == PASS
    return report


def run_baseline(source, cpu, plan, threads):
    """The same bytes the ordinary way: drive -> host buffer (O_DIRECT), then CPU stores into the card through the window. Also the drive on
    its own, to host RAM only, which is the most any path can do."""
    size = plan['dest']['size']
    chunk = plan['chunk']
    host = mmap.mmap(-1, size)
    view = memoryview(host)
    try:
        latencies, seconds_read = timed(lambda: read_all(source, view, plan, threads, 0))
        base = plan['dest']['window_offset']

        def copy():
            for done in range(0, size, chunk):
                count = min(chunk, size - done)
                cpu.write(base + done, bytes(view[done:done + count]))

        _, seconds_copy = timed(copy)
    finally:
        view.release()
        host.close()
    total = seconds_read + seconds_copy
    return {'bytes': size,
            'drive_to_host_gbps': gbps(size, seconds_read),
            'host_to_card_cpu_gbps': gbps(size, seconds_copy),
            'bounce_gbps': gbps(size, total),
            'drive_to_host_p99_chunk_ms': percentile(latencies, 0.99) * 1e3}


# -- the mock: the same flow, in memory ---------------------------------------------------------------------------------

class MockCard(object):
    """A card that is a bytearray of DRAM. The published slice and the window see the same bytes, as on the real thing, so a read into the
    slice shows up through the window. `fault` names a defect the verify must catch: 'flip' (one byte of the drive's data lands wrong),
    'misroute' (the data never lands, leaving the destination as it was) or 'stall' (the first run lands, later ones do not). `events`
    records the order of the calls that matter."""

    def __init__(self, slice_size=16 << 20, slice_offset=0, window=FIRST_4G_TLB, fault=None, in_use=0, held_windows=(), pinnable=True,
                 power_fails=False, openers=(), selftest_fails=False, phys_offset=0, stall_after=8 << 20):
        self.dram = bytearray(slice_offset + slice_size + (4 << 20))
        self.slice_size = slice_size
        self.slice_offset = slice_offset
        self.window = window
        self.fault = fault
        self.in_use = in_use
        self.held_windows = set(held_windows)
        self.pinnable = pinnable
        self.power_fails = power_fails
        self.openers = list(openers)
        self.selftest_fails = selftest_fails
        self.phys_offset = phys_offset
        self.stall_after = stall_after
        self.bdf = '0000:00:00.0'
        self.configured = None
        self.closed = []
        self.events = []

    def p2pmem_info(self):
        return {'bdf': self.bdf, 'bar': 4, 'offset': self.slice_offset, 'size': self.slice_size, 'tlb_first': self.window,
                'tlb_last': self.window, 'in_use': self.in_use, 'bus_addr': 0x38000000000 + self.slice_offset}

    def other_openers(self):
        return list(self.openers)

    def expected_phys(self, info):
        return 0x38000000000 + info['offset']

    def power_up(self, settle=0):
        self.events.append('power_up')
        if self.power_fails:
            raise ProbeError('SET_POWER_STATE failed (mock)')

    def acquire_window(self, wanted):
        self.events.append('acquire_window')
        if wanted in self.held_windows:
            raise ProbeError('window %d is held by another process' % wanted)
        return MockWindow(self)

    def p2p_buffer(self, size):
        self.events.append('p2p_buffer')
        return MockBuffer(self, size)

    def close(self):
        pass


class MockWindow(object):
    def __init__(self, card):
        self.card = card
        self.id = card.window

    def configure(self, x, y, addr, ordering):
        self.card.events.append('configure')
        self.card.configured = (x, y, addr, ordering)

    def pin(self):
        self.card.events.append('pin')
        return self.card.pinnable

    def cpu_window(self, size):
        return MockCpu(self.card.dram, dead=self.card.selftest_fails)

    def close(self):
        self.card.closed.append(self.id)


class MockCpu(object):
    def __init__(self, dram, dead=False):
        self.dram = dram
        self.dead = dead          # a window that is not memory: writes vanish, reads give zeros

    def read(self, offset, size):
        return bytes(size) if self.dead else bytes(self.dram[offset:offset + size])

    def write(self, offset, data):
        if not self.dead:
            self.dram[offset:offset + len(data)] = data

    def close(self):
        pass


class MockBuffer(object):
    """The slice as a memoryview of the mock DRAM: what is read into it is what the window shows."""

    def __init__(self, card, size):
        self.view = memoryview(card.dram)[card.slice_offset:card.slice_offset + size]
        self.size = size
        self.phys = 0x38000000000 + card.slice_offset + card.phys_offset      # where the "kernel" says the mapping is

    def bus_addr_of(self, offset):
        return self.phys + offset

    def close(self):
        self.view.release()


class MockSource(Source):
    """A Source over a regular file, buffered (not every filesystem the tests run on takes O_DIRECT), that applies the card's fault."""

    def __init__(self, path, card=None, extents=None):
        Source.__init__(self, path, direct=False, extents=extents)
        self.card = card
        self.damaged = False
        self.delivered = 0

    def read_into(self, view, file_offset):
        fault = self.card.fault if self.card is not None else None
        if fault == 'misroute':
            return len(view)                # the bytes never arrive
        if fault == 'stall' and self.delivered >= self.card.stall_after:
            return len(view)                # the first run landed; the later ones do not
        done = Source.read_into(self, view, file_offset)
        self.delivered += done
        if fault == 'flip' and not self.damaged and done:
            view[done // 2] = view[done // 2] ^ 0xFF
            self.damaged = True
        return done


# -- command line ------------------------------------------------------------------------------------------------------

def build_parser():
    parser = argparse.ArgumentParser(prog='p2p_probe', description=__doc__.split('\n\n')[0],
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--device', type=int, default=0, help='/dev/tenstorrent/N (default 0)')
    parser.add_argument('--source', help='file, or block device, to read with O_DIRECT')
    parser.add_argument('--source-offset', type=parse_size, default=0, help='where to start reading (multiple of 4096)')
    parser.add_argument('--size', type=parse_size, default=256 << 20, help='bytes to move (default 256M, at least 8K)')
    parser.add_argument('--chunk', type=parse_size, default=16 << 20, help='bytes per read (default 16M)')
    parser.add_argument('--threads', type=int, default=1, help='reads in flight (default 1)')
    parser.add_argument('--repeat', type=int, default=1, help='timed runs; the report carries the median (default 1)')
    parser.add_argument('--bank', type=int, default=0, help='DRAM bank to aim the window at (default 0); --noc-x/--noc-y override')
    parser.add_argument('--noc-x', type=int, help='translated NOC x of the target (default: bank 0 port 0)')
    parser.add_argument('--noc-y', type=int, help='translated NOC y of the target')
    parser.add_argument('--noc-addr', type=parse_size, default=0, help='NOC address the window starts at (multiple of 4 GiB, default 0)')
    parser.add_argument('--dest-offset', type=parse_size, default=1 << 30, help='offset inside the published slice to write at (default 1G)')
    parser.add_argument('--ordering', choices=sorted(ORDERINGS), default='relaxed', help='TLB transaction ordering (default relaxed)')
    parser.add_argument('--verify', choices=('sample', 'full', 'none'), default='sample',
                        help='none checks nothing and ends UNVERIFIED (exit 3): for CPU-time measurements only')
    parser.add_argument('--samples', type=int, default=16, help='random spans verified in sample mode (default 16)')
    parser.add_argument('--span', type=parse_size, default=1 << 20, help='bytes per verified span (default 1M)')
    parser.add_argument('--seed', type=int, default=1, help='seed for the sampling')
    parser.add_argument('--baseline', action='store_true', help='also time drive -> host RAM -> card (CPU stores; indicative)')
    parser.add_argument('--baseline-only', action='store_true', help='time only the host-bounce path (no peer-to-peer DMA)')
    parser.add_argument('--min-gbps', type=float, default=0.0, help='FAIL if the peer-to-peer rate is below this')
    parser.add_argument('--settle', type=float, default=1.0, help='seconds to wait after asking for the busy power state (default 1)')
    parser.add_argument('--allow-unchecked', action='store_true', help='skip the FIEMAP check of the source (do not)')
    parser.add_argument('--allow-unpinned', action='store_true', help='go on when the window cannot be pinned with a dma-buf export (do not)')
    parser.add_argument('--allow-shared', action='store_true', help='go on although another process has the card open (do not)')
    parser.add_argument('--prepare', metavar='FILE', help='write --size bytes of random data to FILE (plain writes, fsync) and exit')
    parser.add_argument('--overwrite-device', action='store_true',
                        help='allow --prepare to write onto a blank block device (a spare; never one that is mounted, in an array, or has data)')
    parser.add_argument('--overwrite-file', action='store_true', help='allow --prepare to replace a regular file that has content')
    parser.add_argument('--extents', metavar='FILE', help='print FILE\'s FIEMAP extents and whether they are safe to read, and exit')
    parser.add_argument('--dry-run', action='store_true', help='print the plan; touch no device')
    parser.add_argument('--mock', action='store_true', help='run the flow against in-memory fakes')
    parser.add_argument('--mock-fault', choices=('flip', 'misroute', 'stall'), help='with --mock: inject a defect the verify must catch')
    parser.add_argument('--json', action='store_true', help='print the report as JSON only')
    return parser


def resolve_target(args):
    if args.noc_x is None or args.noc_y is None:
        args.noc_x, args.noc_y = dram_bank_xy(args.bank)


def emit(report, as_json):
    if as_json:
        print(json.dumps(report, indent=2, sort_keys=True))
        return
    plan = report.get('plan', {})
    print('window %s -> NOC (%s,%s) addr %#x, slice offset %s, %d chunks of %d bytes, pinned %s' % (
        plan.get('window'), plan['noc']['x'], plan['noc']['y'], plan['noc']['addr'], plan['dest']['slice_offset'],
        plan['chunks'], plan['chunk'], report.get('pinned')))
    if 'p2p' in report:
        p2p = report['p2p']
        print('p2p      %10.3f GB/s   %.3f s for %d bytes (%d thread(s), chunk p50 %.1f ms p99 %.1f ms)' % (
            p2p['gbps'], p2p['seconds'], p2p['bytes'], p2p['threads'], p2p['chunk_p50_ms'], p2p['chunk_p99_ms']))
    if 'baseline' in report:
        base = report['baseline']
        print('bounce   %10.3f GB/s   (drive to host %.3f, CPU stores to card %.3f)' % (
            base['bounce_gbps'], base['drive_to_host_gbps'], base['host_to_card_cpu_gbps']))
    for label, check in (('verify  ', report.get('verify')), ('bounce  ', report.get('baseline', {}).get('verify'))):
        if check:
            print('%s %s: %s, %d bytes compared' % (label, check['mode'], {True: 'OK', False: 'MISMATCH', None: 'skipped'}[check['ok']],
                                                    check['bytes_compared']))
            for item in check.get('mismatches', []):
                print('  first difference at source offset %d: expected %#04x found %#04x' % (
                    item['source_offset'], item['expected'], item['found']))
    for note in report.get('notes', []):
        print('note: %s' % note)
    if report.get('below_min_gbps'):
        print('below --min-gbps')
    print('RESULT', report['result'])


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        if args.prepare:
            prepare_source(args.prepare, args.size, args.seed, args.overwrite_device, args.overwrite_file)
            print('wrote %d bytes to %s' % (args.size, args.prepare))
            return 0
        if args.extents:
            fd = os.open(args.extents, os.O_RDONLY)
            try:
                length = os.fstat(fd).st_size
                found = fiemap_extents(fd, 0, length)
                for logical, physical, extent_length, flags in found:
                    print('logical %d physical %d length %d flags %#x%s' % (
                        logical, physical, extent_length, flags, '  <- HAZARD' if flags & FIEMAP_HAZARDS else ''))
                check_extents(found, 0, length)
                print('all %d bytes are written extents' % length)
            finally:
                os.close(fd)
            return 0
        if not args.source and not args.dry_run:
            raise ProbeError('--source is required')
        resolve_target(args)
        if args.dry_run:
            info = {'bar': 4, 'offset': 0, 'size': 4 * GIB, 'bus_addr': 0, 'tlb_first': FIRST_4G_TLB, 'tlb_last': FIRST_4G_TLB}
            plan = make_plan(info, args.size, args.dest_offset, args.chunk, args.source_offset, args.noc_x, args.noc_y,
                             args.noc_addr, ORDERINGS[args.ordering])
            print(json.dumps({'dry_run': True, 'plan': plan}, indent=2, sort_keys=True))
            return 0
        if args.mock:
            args.settle = 0
            card = MockCard(fault=args.mock_fault)
            source = MockSource(args.source, card=card)
        else:
            if os.geteuid() != 0:
                raise ProbeError('run as root: the p2pmem allocation, the TLB ioctls and the pagemap check all need it')
            source = Source(args.source)
            try:
                card = TtCard(args.device)
            except OSError:
                source.close()
                raise
        try:
            report = run_probe(card, source, args)
        finally:
            source.close()
            card.close()
        emit(report, args.json)
        return EXIT_CODES[report['result']]
    except ProbeError as error:
        print('p2p_probe: %s' % error, file=sys.stderr)
        return 2
    except OSError as error:
        print('p2p_probe: %s' % error, file=sys.stderr)
        return 2


if __name__ == '__main__':
    sys.exit(main())
