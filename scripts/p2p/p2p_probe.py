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

--mode write is the other direction: the drive DMA-reads the card's DRAM through the same p2pmem mapping (an O_DIRECT pwritev whose buffers
are the mapping) into a scratch range of a drive that the operator names and owns, and the probe reads the range back with O_DIRECT to
compare it with what it put into the card. It sweeps write sizes and queue depths, re-aims the window across the DRAM banks with a canary
read after every re-aim, can soak for minutes, and reads the PCIe AER counters of the card, the drive and every bridge between them before
and after. It refuses a range that is not blank unless --i-own-this-scratch-range is given with the range, refuses a PCIe link at 2.5 GT/s,
and never writes outside the range it was given. See --help for the flags and the README for a worked example.

Stdlib only.
"""

import argparse
import concurrent.futures
import contextlib
import ctypes
import fcntl
import hashlib
import json
import mmap
import os
import random
import re
import stat
import struct
import sys
import tempfile
import threading
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
WINDOW_DRAM_LIMIT = 0xFF000000   # the top 16 MiB of a 4 GiB window is not DRAM (a pattern written there does not read back)

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
NOTE_WRITE_FILL = ('the card is filled and its canaries are read through the CPU mapping of the window (--cpu-map), which is slow; the DMA itself goes '
                   'through the p2pmem mapping and the drive')
NOTE_WRITE_CPU = ('the cpu figures of the write arm are the process CPU while the pwritev batches run (not the sentinel paint, the read-back compare '
                  'or the AER sampling, which are reported apart as non_write_process_seconds); "cores" is CPU seconds per second of write time')
NOTE_WRITE_QD = ('queue depth is batch-synchronous: QD writes are issued together and all are waited for before the next sentinel paint and verify, so '
                 'the queue drains to zero between batches; the rates understate a steady queue of the same depth')
NOTE_WRITE_AER = ('a root port\'s correctable total and its devices\' correctable counters count the same error: the correctable figures are not a count of '
                  'distinct errors (uncorrectable detection is not affected)')
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
    if window_offset + dest_offset + size > WINDOW_DRAM_LIMIT:
        raise ProbeError('the destination ends at window offset %#x: the top 16 MiB of a 4 GiB window (from %#x) is not DRAM; keep every '
                         'address below it (a 4 GiB bank leaves about 3 GiB above a 1 GiB offset)' % (window_offset + dest_offset + size, WINDOW_DRAM_LIMIT))
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

    def pci_chain(self):
        """The PCI functions from the root port down to this card (for the link speeds and the AER counters)."""
        return pci_chain(self.pci_dir)

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
                tlb_id, _, mmap_uc, mmap_wc, _ = ALLOCATE_OUT.unpack_from(buf, ALLOCATE_IN.size)
                if tlb_id == wanted:
                    self.windows[tlb_id] = mmap_wc
                    return TtWindow(self, tlb_id, mmap_wc, mmap_uc)
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
    def __init__(self, card, tlb_id, mmap_offset_wc, mmap_offset_uc=None):
        self.card = card
        self.id = tlb_id
        self.mmap_offset_wc = mmap_offset_wc
        self.mmap_offset_uc = mmap_offset_uc
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

    def cpu_window(self, size, uc=False):
        """CPU mapping of the window: write-combining (the default, what the read arm uses) or uncached (what the tier's canary reads use)."""
        offset = self.mmap_offset_uc if uc else self.mmap_offset_wc
        if offset is None:
            raise ProbeError('the driver did not report an uncached mapping offset for window %d' % self.id)
        try:
            mapping = mmap.mmap(self.card.fd, size, flags=mmap.MAP_SHARED, prot=mmap.PROT_READ | mmap.PROT_WRITE, offset=offset)
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


def read_all(source, view, plan, threads, offset_in_view, cpu=None):
    """Read plan['dest']['size'] bytes from the source into view[offset_in_view:], chunk by chunk, `threads` at a time. Returns the
    per-chunk seconds. When `cpu` is a dict, the CPU seconds each thread spent in its reads (user + system, not the time it slept waiting
    for the drive) are added to it, keyed by thread."""
    size = plan['dest']['size']
    chunk = plan['chunk']
    jobs = [(done, min(chunk, size - done)) for done in range(0, size, chunk)]
    lock = threading.Lock()

    def one(job):
        done, count = job
        began_cpu, began = time.thread_time(), time.perf_counter()
        piece = view[offset_in_view + done:offset_in_view + done + count]
        try:
            source.read_into(piece, plan['source_offset'] + done)
        finally:
            piece.release()
        elapsed = time.perf_counter() - began
        if cpu is not None:
            used = time.thread_time() - began_cpu
            with lock:
                ident = threading.get_ident()
                cpu[ident] = cpu.get(ident, 0.0) + used
        return elapsed

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
        cpu_seconds = 0.0
        worker_runs = []
        for _ in range(max(1, args.repeat)):
            # A sentinel before every run: a DMA that stops writing must not pass on what an earlier run left.
            paint_sentinel(cpu, plan, args.verify, args.samples, args.span, args.seed)
            workers = {}
            cpu_before = time.process_time()
            latencies, seconds = timed(lambda: read_all(source, buffer.view, plan, args.threads, plan['dest']['slice_offset'], workers))
            cpu_seconds += time.process_time() - cpu_before
            worker_runs.append(list(workers.values()))
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
                     'chunk_p99_ms': percentile(all_latencies, 0.99) * 1e3,
                     'cpu_seconds': cpu_seconds, 'cpu_fraction': cpu_seconds / max(1e-9, sum(seconds_all)),
                     # per-thread CPU of the reads (the timed part only, not the sentinel or the verify): the sum over all runs, the busiest
                     # single thread of any one run, and CPU seconds per second of read time
                     'worker_seconds': sum(sum(run) for run in worker_runs),
                     'busiest_worker_seconds': max([max(run) for run in worker_runs if run] or [0.0]),
                     'worker_threads': max([len(run) for run in worker_runs] or [0]),
                     'worker_cores': sum(sum(run) for run in worker_runs) / max(1e-9, sum(seconds_all))}
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

    def cpu_window(self, size, uc=False):
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


# -- the write arm: the card's DRAM -> the drive -----------------------------------------------------------------------
#
# The drive DMA-reads the card's DRAM through the p2pmem mapping (an O_DIRECT pwritev whose buffers are the mapping), into a scratch
# range of a drive the operator named and owns. The bytes in the card were put there by the CPU through the TLB window from a
# deterministic pattern; the probe reads the range back (O_DIRECT, into host memory) and compares. Every refusal that needs no hardware
# is arithmetic or a sysfs read, and is made before the first DMA.

SHARD_BYTES = 696320                 # one object's share on one card: 10 tensors x 69,632 B (170 pages of 4 KiB)
OBJECT_BYTES = 4 * SHARD_BYTES       # one object over four cards: 2,785,280 B
SUBRUN_BYTES = 8 * OBJECT_BYTES      # a spill sub-run of eight objects: 22,282,240 B
WRITE_SIZES = (SHARD_BYTES, OBJECT_BYTES, SUBRUN_BYTES)
WRITE_QDS = (1, 4, 8, 16, 32)       # QD 1 is the drain-bound measurement (one sub-run on its own); the sweep proper is 4 to 32
NUM_BANKS = 8
MIN_SCRATCH_START = 1 << 20          # the first MiB of a drive holds a partition table, a superblock, the DMA canary block
GEN1_GTS = 2.5
SOAK_BUCKET_SECONDS = 30.0
PCI_FUNCTION = re.compile(r'^[0-9a-f]{4}:[0-9a-f]{2}:[0-9a-f]{2}\.[0-7]$')
AER_DEV_FILES = ('aer_dev_correctable', 'aer_dev_nonfatal', 'aer_dev_fatal')
AER_ROOT_FILES = ('aer_rootport_total_err_cor', 'aer_rootport_total_err_nonfatal', 'aer_rootport_total_err_fatal')
AER_CORRECTABLE = ('aer_dev_correctable', 'aer_rootport_total_err_cor')


class StopRun(ProbeError):
    """A condition that ends the DMA at once (a canary that does not match, bytes that differ, an uncorrectable AER error, a drive that is no
    longer live, a spent write budget): the probe stops sending, records what happened and does not try again to see."""

    def __init__(self, kind, message):
        ProbeError.__init__(self, message)
        self.kind = kind


def parse_int_list(text):
    """'1,4 8' -> [1, 4, 8] (sizes may carry K/M/G)."""
    items = [parse_size(part) for part in str(text).replace(',', ' ').split()]
    if not items:
        raise ValueError('an empty list')
    return items


def read_text(path):
    try:
        with open(path) as handle:
            return handle.read().strip()
    except (OSError, UnicodeDecodeError):
        return None


# -- PCIe: the chain of functions between a device and the root, their links and their AER counters ---------------------

def pci_chain(path):
    """The PCI functions from the root port down to the one `path` is under, as directories (the sysfs real path of a device lists every
    bridge above it as a path component named like a PCI address). [] when the path is not under a PCI function."""
    parts = os.path.realpath(path).split(os.sep)
    return [os.sep.join(parts[:index + 1]) for index, part in enumerate(parts) if PCI_FUNCTION.match(part)]


def parse_gts(text):
    """'16.0 GT/s PCIe' -> 16.0; None for 'Unknown speed' or nothing."""
    found = re.match(r'\s*([0-9]+(?:\.[0-9]+)?)\s*GT/s', text or '')
    return float(found.group(1)) if found else None


def link_info(directory):
    return {'bdf': os.path.basename(directory),
            'speed_gts': parse_gts(read_text(os.path.join(directory, 'current_link_speed'))),
            'max_gts': parse_gts(read_text(os.path.join(directory, 'max_link_speed'))),
            'width': read_text(os.path.join(directory, 'current_link_width'))}


def check_links(label, chain, endpoint_min_gts=0.0, allow_unknown=False):
    """The links along `chain` (root port first, the device last). A link at 2.5 GT/s (Gen1) is refused whatever the flags: a drive that
    trained there moves 0.8 GB/s and every figure the probe would report is worthless. A function whose speed cannot be read is refused
    unless allow_unknown. The last function must also be at least endpoint_min_gts when that is set (the drive: 16 GT/s)."""
    found = [link_info(directory) for directory in chain]
    for index, link in enumerate(found):
        speed = link['speed_gts']
        if speed is not None and speed <= GEN1_GTS:
            raise ProbeError('%s: %s trained at %.1f GT/s (Gen1): retrain the link (a drive behind a switch port needs the port\'s target speed '
                             'set and a retrain) before any DMA; every rate measured on it would be 0.8 GB/s' % (label, link['bdf'], speed))
        if speed is None and not allow_unknown:
            raise ProbeError('%s: the speed of %s cannot be read (current_link_speed missing or unknown): cannot prove the link is not Gen1; '
                             '--allow-unknown-link goes on' % (label, link['bdf']))
        if index == len(found) - 1 and endpoint_min_gts and speed is not None and speed < endpoint_min_gts:
            raise ProbeError('%s: %s runs at %.1f GT/s, below the %.1f GT/s the cache drive must hold (--min-drive-gts)' % (
                label, link['bdf'], speed, endpoint_min_gts))
    return found


def parse_aer_counters(text):
    counters = {}
    for line in (text or '').splitlines():
        fields = line.split()
        if len(fields) == 2 and fields[1].isdigit():
            counters[fields[0]] = int(fields[1])
    return counters


def read_aer(directory):
    """Every AER counter file a function has: {file name: {counter: value}}; {} for a function without AER."""
    found = {}
    for name in AER_DEV_FILES:
        text = read_text(os.path.join(directory, name))
        if text is not None:
            found[name] = parse_aer_counters(text)
    for name in AER_ROOT_FILES:
        text = read_text(os.path.join(directory, name))
        if text is not None and text.split() and text.split()[0].isdigit():
            found[name] = {'total': int(text.split()[0])}
    return found


def snapshot_aer(functions):
    return {os.path.basename(directory): read_aer(directory) for directory in functions}


def aer_delta(before, after):
    """What grew between two snapshots. Correctable = aer_dev_correctable and the root port's correctable total; everything else
    (non-fatal, which holds the completion timeout, and fatal) is uncorrectable. TOTAL_* lines repeat the counters above them and are not
    summed. `measured` is False when no function had any AER counter in both snapshots (the kernel without AER, or no files), which is not a
    clean bill."""
    per_function = {}
    correctable = uncorrectable = 0
    measured = False
    for bdf, files in after.items():
        was = before.get(bdf, {})
        grew = {}
        for name, counters in files.items():
            if name not in was:
                continue
            measured = True
            for counter, value in counters.items():
                if counter.startswith('TOTAL_'):
                    continue
                step = value - was[name].get(counter, value)
                if step > 0:
                    grew['%s.%s' % (name, counter)] = step
                    if name in AER_CORRECTABLE:
                        correctable += step
                    else:
                        uncorrectable += step
        if grew:
            per_function[bdf] = grew
    return {'measured': measured, 'correctable': correctable, 'uncorrectable': uncorrectable, 'by_function': per_function}


def nvme_states(function_directory):
    """The state of every NVMe controller under a PCI function ('live' is the only good one)."""
    states = []
    base = os.path.join(function_directory, 'nvme')
    try:
        names = sorted(os.listdir(base))
    except OSError:
        return states
    for name in names:
        state = read_text(os.path.join(base, name, 'state'))
        if state is not None:
            states.append(state)
    return states


# -- the drive end -----------------------------------------------------------------------------------------------------

class Sink(object):
    """The drive (or a file standing in for it), opened O_DIRECT for writing and reading. A block device is opened O_EXCL as well, which
    the kernel refuses (EBUSY) while anything mounts or holds it. `allow(start, end)` fences every write: nothing outside [start, end) is
    ever written, whatever the caller computes."""

    def __init__(self, path, direct=True):
        self.path = path
        self.direct = direct
        flags = os.O_RDWR | (getattr(os, 'O_DIRECT', 0) if direct else 0)
        if stat.S_ISBLK(os.stat(path).st_mode):
            flags |= os.O_EXCL
        self.fd = os.open(path, flags)
        try:
            info = os.fstat(self.fd)
            self.is_block = stat.S_ISBLK(info.st_mode)
            self.length = os.lseek(self.fd, 0, os.SEEK_END) if self.is_block else info.st_size
            self.rdev = info.st_rdev if self.is_block else None
        except OSError:
            os.close(self.fd)
            raise
        self.fence = None
        self.written = 0
        self.lock = threading.Lock()

    def allow(self, start, end):
        self.fence = (start, end)

    def sysfs_dir(self, sys_dev_block='/sys/dev/block'):
        """The sysfs directory of the block device, or None (a regular file has none)."""
        if not self.is_block:
            return None
        path = os.path.join(sys_dev_block, '%d:%d' % (os.major(self.rdev), os.minor(self.rdev)))
        return os.path.realpath(path) if os.path.exists(path) else None

    def pci_chain(self):
        """The PCI functions above the drive. With NVMe multipath the visible namespace hangs off the subsystem, not off a PCI function, so
        the controllers beside it are followed to theirs."""
        block = self.sysfs_dir()
        if not block:
            return []
        chain = pci_chain(block)
        if chain:
            return chain
        subsystem = os.path.dirname(block)
        try:
            names = sorted(name for name in os.listdir(subsystem) if re.match(r'^nvme[0-9]+$', name))
        except OSError:
            return []
        for name in names:
            chain = pci_chain(os.path.join(subsystem, name))
            if chain:
                return chain
        return []

    def _check_fence(self, offset, nbytes):
        if self.fence is None:
            raise ProbeError('internal error: a write was attempted before the scratch range was fenced')
        if offset < self.fence[0] or offset + nbytes > self.fence[1]:
            raise ProbeError('internal error: a write of %d bytes at %d falls outside the scratch range [%d, %d): refused' % (
                nbytes, offset, self.fence[0], self.fence[1]))

    def write_iov(self, views, offset):
        """pwritev of `views` (buffers of the p2pmem mapping, in order) at `offset`; a short write is continued."""
        total = sum(len(view) for view in views)
        self._check_fence(offset, total)
        done = 0
        pending = list(views)
        temporary = []
        try:
            while True:
                try:
                    count = os.pwritev(self.fd, pending, offset + done)
                except OSError as error:
                    raise ProbeError('write of %d bytes at drive offset %d failed: %s%s' % (
                        total - done, offset + done, error,
                        ' (the kernel refused peer-to-peer DMA from this memory: see dmesg and p2p_topology.py)' if error.errno == 121 else '')) from error
                if count <= 0:
                    raise ProbeError('short write to the drive at %d (%d of %d bytes)' % (offset + done, done, total))
                done += count
                if done >= total:
                    break
                skip, rest = count, []
                for view in pending:
                    if skip >= len(view):
                        skip -= len(view)
                        continue
                    if skip:
                        view = view[skip:]
                        temporary.append(view)
                    rest.append(view)
                    skip = 0
                pending = rest
        finally:
            for piece in temporary:
                piece.release()
        with self.lock:
            self.written += total
        return total

    def write_buf(self, view, offset):
        return self.write_iov([view], offset)

    def read_into(self, view, offset):
        """O_DIRECT read exactly len(view) bytes at `offset` into `view` (page-aligned host memory)."""
        total = len(view)
        done = 0
        while done < total:
            rest = view[done:]
            try:
                count = os.preadv(self.fd, [rest], offset + done)
            except OSError as error:
                raise ProbeError('read of %d bytes at drive offset %d failed: %s' % (total - done, offset + done, error)) from error
            finally:
                rest.release()
            if count <= 0:
                raise ProbeError('short read from the drive at %d (%d of %d bytes)' % (offset + done, done, total))
            done += count
        return done

    def close(self):
        os.close(self.fd)


def first_nonzero_in_range(sink, start, size, chunk=8 << 20):
    """Offset (from `start`) of the first byte of the range that is not zero, or None. Reads the whole range with O_DIRECT."""
    buffer = mmap.mmap(-1, chunk)
    zeros = bytes(chunk)
    try:
        for done in range(0, size, chunk):
            count = min(chunk, size - done)
            with memoryview(buffer)[:count] as view:
                sink.read_into(view, start + done)
            if buffer[:count] != zeros[:count]:
                block = next(index for index in range(0, count, PAGE) if buffer[index:index + PAGE] != zeros[:min(PAGE, count - index)])
                return done + block + next(i for i, byte in enumerate(buffer[block:block + PAGE]) if byte)
    finally:
        buffer.close()
    return None


class Scratch(object):
    """The range [start, start+size) the probe may write, handed out in order and wrapping, so a long run spreads over all of it."""

    def __init__(self, start, size):
        self.start, self.size, self.cursor = start, size, 0

    def take(self, nbytes):
        if nbytes > self.size:
            raise ProbeError('the scratch range (%d bytes) is smaller than one batch (%d bytes)' % (self.size, nbytes))
        if self.cursor + nbytes > self.size:
            self.cursor = 0
        offset = self.start + self.cursor
        self.cursor += nbytes
        return offset


# -- what is in the card, and what the drive must therefore hold ---------------------------------------------------------

def canary_page(seed, bank):
    """The first page of a bank's region: a value no other bank has, repeated over the page."""
    digest = hashlib.sha256(b'p2p-canary:%d:%d' % (seed, bank)).digest()
    return digest * (PAGE // len(digest))


def bank_pattern(seed, bank, size):
    return random.Random('p2p-bank:%d:%d' % (seed, bank)).randbytes(size)


def required_scratch(args):
    """The smallest scratch range that holds the largest batch any arm writes."""
    need = 0
    if not args.no_sweep:
        need = max(need, max(args.sizes) * max(args.qds))
    if args.reaim:
        need = max(need, args.reaim_ops * args.reaim_op_bytes)
    if args.soak > 0:
        need = max(need, args.soak_size * args.soak_qd)
    return need


def check_write_args(args):
    """Every refusal that needs neither the card nor the drive."""
    for name, values in (('--sizes', args.sizes), ('--qd', args.qds), ('--banks', args.banks)):
        if not values:
            raise ProbeError('%s is empty' % name)
    for size in list(args.sizes) + [args.reaim_op_bytes, args.soak_size, args.iov_bytes]:
        if size < 0 or size % PAGE:
            raise ProbeError('size %d is not a multiple of the %d-byte sector O_DIRECT needs' % (size, PAGE))
    if (not args.no_sweep and any(size <= 0 for size in args.sizes)) or (args.soak > 0 and args.soak_size <= 0):
        raise ProbeError('--sizes and --soak-size must be positive')
    if any(qd < 1 for qd in list(args.qds) + [args.soak_qd]):
        raise ProbeError('a queue depth is at least 1')
    if any(not 0 <= bank < NUM_BANKS for bank in list(args.banks) + [args.bank]):
        raise ProbeError('a Blackhole has DRAM banks 0-7')
    iov_max = 1024
    for size in (list(args.sizes) if not args.no_sweep else []) + ([args.reaim_op_bytes] if args.reaim else []) + ([args.soak_size] if args.soak > 0 else []):
        pieces = 1 if not args.iov_bytes else -(-size // args.iov_bytes)
        if pieces > iov_max:
            raise ProbeError('a %d-byte write in pieces of %d bytes is %d iovecs; the kernel takes %d' % (size, args.iov_bytes, pieces, iov_max))
    if args.region % PAGE or args.region <= 0:
        raise ProbeError('--region %d must be a positive multiple of %d' % (args.region, PAGE))
    largest = max(([] if args.no_sweep else list(args.sizes)) + ([args.soak_size] if args.soak > 0 else []) or [0])
    if args.region < largest:
        raise ProbeError('--region %d must hold at least the largest write (%d)' % (args.region, largest))
    if args.soak < 0 or args.arm_seconds < 0 or args.soak_max_gbps < 0 or args.max_write_gib <= 0:
        raise ProbeError('--soak, --arm-seconds and --soak-max-gbps must not be negative, --max-write-gib must be positive')
    if args.reaim and (args.reaim_rounds < 1 or args.reaim_ops < 1 or args.reaim_op_bytes < PAGE):
        raise ProbeError('--reaim needs at least one round of at least one write of at least a page')
    if args.no_sweep and not args.reaim and args.soak <= 0:
        raise ProbeError('nothing to do: --no-sweep with neither --reaim nor --soak')
    if args.own_scratch and (args.scratch_start is None or args.scratch_size is None):
        raise ProbeError('--i-own-this-scratch-range needs the range itself: --scratch-start and --scratch-size')
    if args.scratch_start is None or args.scratch_size is None:
        raise ProbeError('--scratch-start and --scratch-size are required: the probe writes only to a range that was named')
    if args.scratch_start % PAGE or args.scratch_size % PAGE or args.scratch_size <= 0:
        raise ProbeError('the scratch range must start and be sized in multiples of %d bytes' % PAGE)
    if args.scratch_start < MIN_SCRATCH_START:
        raise ProbeError('the scratch range starts at %d: the first MiB of the drive holds its partition table or superblock and is never written' % args.scratch_start)
    need = required_scratch(args)
    if args.scratch_size < need:
        raise ProbeError('the scratch range (%d bytes) is smaller than the largest batch (%d bytes = queue depth x write size): widen it or lower --qd/--sizes'
                         % (args.scratch_size, need))
    if args.cpu_map not in ('uc', 'wc'):
        raise ProbeError('--cpu-map is uc or wc')


def write_plan(args, info):
    """The geometry of the card side: one region per bank, [canary page][data], at the same window offset in every bank. The bank the sweep
    and the soak write from holds --region bytes of data (or what the re-aim arm needs, if more); the other banks hold what the re-aim arm
    writes from them."""
    reaim_data = args.reaim_ops * args.reaim_op_bytes if args.reaim else 0
    primary = args.region if (not args.no_sweep or args.soak > 0) else 0
    data_bytes = {bank: reaim_data for bank in (args.banks if args.reaim else [])}
    data_bytes[args.bank] = max(primary, reaim_data, PAGE)
    region = PAGE + max(data_bytes.values())
    plan = make_plan(info, region, args.dest_offset, PAGE, 0, args.noc_x, args.noc_y, args.noc_addr, ORDERINGS[args.ordering])
    plan['banks'] = {str(bank): {'noc': list(dram_bank_xy(bank)), 'data_bytes': data_bytes[bank]} for bank in sorted(data_bytes)}
    plan['region_bytes'] = region
    return plan, data_bytes


# -- the write run -----------------------------------------------------------------------------------------------------

class AerMonitor(object):
    """AER counters of the card, the drive and every bridge between them and the root, read before the run, sampled during it and read
    after it. An uncorrectable increment (non-fatal, which holds the completion timeout, or fatal) stops the DMA: the arms call `check`
    after every batch and between the segments of a cycle (each sentinel chunk, each read-back), which samples when `interval` seconds
    have passed since the last sample (and always after a cell, with force), so an error is acted on within the interval plus the
    segment in flight. The longest time between two samples is kept (`max_gap`) and reported, so the report shows how quickly the probe
    could have acted."""

    MAX_SAMPLES = 4000          # the report keeps this many samples; the counts and the longest gap cover all of them

    def __init__(self, functions, drive_function=None):
        self.functions = list(dict.fromkeys(functions))
        self.drive_function = drive_function
        self.before = snapshot_aer(self.functions)
        self.last = None
        self.samples = []
        self.count = 0
        self.max_gap = 0.0
        self.cost = 0.0
        self.stamp = time.monotonic()

    def delta(self):
        return aer_delta(self.before, snapshot_aer(self.functions))

    def start(self):
        """The DMA begins now: the gaps are counted from here (the card fill before it moves no data over the drive's link)."""
        self.stamp = time.monotonic()
        self.max_gap = 0.0

    def summary(self):
        return {'sample_count': self.count, 'max_gap_seconds': self.max_gap, 'mean_check_ms': (self.cost / self.count * 1e3) if self.count else None}

    def check(self, force=False, interval=0.0):
        now = time.monotonic()
        if not force and now - self.stamp < interval:
            return None
        self.max_gap = max(self.max_gap, now - self.stamp)
        self.stamp = now
        began = time.perf_counter()
        try:
            delta = self.delta()
            self.last = delta
            self.count += 1
            if len(self.samples) < self.MAX_SAMPLES:
                self.samples.append({'t': time.time(), 'correctable': delta['correctable'], 'uncorrectable': delta['uncorrectable']})
            if delta['uncorrectable']:
                raise StopRun('aer', 'uncorrectable PCIe AER error(s) appeared: %s' % json.dumps(delta['by_function'], sort_keys=True))
            if self.drive_function:
                states = nvme_states(self.drive_function)
                if any(state != 'live' for state in states):
                    raise StopRun('drive', 'the NVMe controller is no longer live: %s' % states)
            return delta
        finally:
            self.cost += time.perf_counter() - began


class Budget(object):
    def __init__(self, limit_bytes):
        self.limit = limit_bytes
        self.spent = 0

    def spend(self, nbytes):
        if self.spent + nbytes > self.limit:
            raise StopRun('budget', 'the write budget (%.0f GiB, --max-write-gib) is spent: stopping before it is exceeded' % (self.limit / GIB))
        self.spent += nbytes


class Aimer(object):
    """Aims the window at a DRAM bank and proves it: CONFIGURE_TLB is three posted register writes, so the proof is a CPU read of the
    bank's canary through the window, a non-posted read from the same requester that cannot complete before the writes landed. The value must
    be that bank's canary; if not, the window is somewhere else and no bio may be issued."""

    def __init__(self, window, cpu, plan, canaries):
        self.window, self.cpu, self.plan, self.canaries = window, cpu, plan, canaries
        self.base = plan['dest']['window_offset']
        self.bank = None
        self.records = []

    def aim(self, bank):
        x, y = dram_bank_xy(bank)
        noc = self.plan['noc']
        began = time.perf_counter()
        self.window.configure(x, y, noc['addr'], noc['ordering'])
        configured = time.perf_counter()
        got = self.cpu.read(self.base, 8)
        done = time.perf_counter()
        record = {'bank': bank, 'configure_ms': (configured - began) * 1e3, 'canary_ms': (done - configured) * 1e3, 'ok': got == self.canaries[bank][:8]}
        self.records.append(record)
        if not record['ok']:
            record['matches_bank'] = [other for other, page in sorted(self.canaries.items()) if page[:8] == got]
            raise StopRun('canary', 'after aiming the window at bank %d the canary read %s, which is %s: the window is not where it was aimed' % (
                bank, got.hex(), 'bank %s\'s' % record['matches_bank'] if record['matches_bank'] else 'no bank\'s'))
        self.bank = bank
        return record


class WriteRun(object):
    """The state the arms share: the drive, the scratch range, the p2pmem mapping, the card-side patterns and the monitors."""

    def __init__(self, args, sink, scratch, buffer, plan, patterns, aimer, aer, budget):
        self.args, self.sink, self.scratch, self.buffer, self.plan = args, sink, scratch, buffer, plan
        self.patterns, self.aimer, self.aer, self.budget = patterns, aimer, aer, budget
        self.data_base = plan['dest']['slice_offset'] + PAGE
        self.iov = args.iov_bytes
        self.verify_on = args.write_verify == 'full'
        self.sentinel = mmap.mmap(-1, 8 << 20)
        self.sentinel[:] = bytes([SENTINEL]) * len(self.sentinel)
        self.readback = None
        self.p2w_bytes = 0
        self.sentinel_bytes = 0
        self.last_mismatch = None
        # CPU of the P2W path: `batch_process` is the process CPU seconds spent inside run.batch() and nowhere else (not the sentinel paint,
        # the read-back compare or the AER sampling); `threads` is each worker's own CPU inside its pwritev.
        self.cpu = {'threads': {}, 'batch_process': 0.0}
        self.lock = threading.Lock()

    def close(self):
        for name in ('sentinel', 'readback'):
            mapped = getattr(self, name)
            if mapped is not None:
                mapped.close()
                setattr(self, name, None)

    # one write: pieces of the mapping, in order, as the iovecs of one pwritev
    def _one(self, op):
        slice_offset, nbytes, drive_offset = op
        began_cpu, began = time.thread_time(), time.perf_counter()
        with contextlib.ExitStack() as stack:
            piece = stack.enter_context(self.buffer.view[slice_offset:slice_offset + nbytes])
            step = self.iov or nbytes
            views = [stack.enter_context(piece[start:start + step]) for start in range(0, nbytes, step)]
            self.sink.write_iov(views, drive_offset)
        elapsed = time.perf_counter() - began
        used = time.thread_time() - began_cpu
        with self.lock:
            ident = threading.get_ident()
            self.cpu['threads'][ident] = self.cpu['threads'].get(ident, 0.0) + used
        return elapsed

    def batch(self, pool, ops):
        """Run the writes concurrently; returns (wall seconds, per-write seconds). The bytes are charged to the budget first. The process
        CPU spent while the writes run is added to cpu['batch_process']: that is the CPU of P2W, and only that."""
        total = sum(op[1] for op in ops)
        self.budget.spend(total)
        began_cpu, began = time.process_time(), time.perf_counter()
        latencies = list(pool.map(self._one, ops))
        wall = time.perf_counter() - began
        self.cpu['batch_process'] += time.process_time() - began_cpu
        self.p2w_bytes += total
        return wall, latencies

    def watch(self):
        """Sample the AER counters and the drive's state if --aer-interval has passed. Called between the segments of a cycle (every sentinel
        chunk, every read-back, after every batch), so no segment is followed by more than one interval of silence."""
        self.aer.check(force=False, interval=self.args.aer_interval)

    def paint(self, offset, nbytes):
        """Overwrite [offset, +nbytes) of the scratch range with a sentinel, so a write that never happened cannot pass on what was there."""
        done = 0
        while done < nbytes:
            count = min(nbytes - done, len(self.sentinel))
            self.budget.spend(count)
            with memoryview(self.sentinel)[:count] as view:
                self.sink.write_buf(view, offset + done)
            done += count
            self.sentinel_bytes += count
            self.watch()

    def verify(self, label, bank, drive_offset, nbytes, count, slots):
        """Read `count` consecutive writes of `nbytes` back from `drive_offset` and compare each with the pattern of the card slot it came from
        (write j came from slot j mod slots). Raises StopRun at the first difference, after recording where."""
        if self.readback is None or len(self.readback) < nbytes:
            if self.readback is not None:
                self.readback.close()
            self.readback = mmap.mmap(-1, max(nbytes, 1 << 20))
        compared = 0
        for index in range(count):
            self.watch()
            slot = index % slots
            expected = memoryview(self.patterns[bank])[slot * nbytes:(slot + 1) * nbytes]
            with memoryview(self.readback)[:nbytes] as view:
                self.sink.read_into(view, drive_offset + index * nbytes)
                if view != expected:
                    raise StopRun('mismatch', self._mismatch(label, bank, index, slot, drive_offset + index * nbytes, view, expected))
            expected.release()
            compared += nbytes
        return {'ops': count, 'bytes': compared, 'ok': True}

    def _mismatch(self, label, bank, index, slot, drive_offset, view, expected):
        nbytes = len(expected)
        bad = [start for start in range(0, nbytes, PAGE) if view[start:start + PAGE] != expected[start:start + PAGE]]
        pages = len(bad)
        first = bad[0] + next(i for i in range(PAGE) if view[bad[0] + i] != expected[bad[0] + i])
        record = {'arm': label, 'bank': bank, 'write': index, 'slot': slot, 'drive_offset': drive_offset + first, 'write_offset': first,
                  'expected': expected[first], 'found': view[first], 'bad_pages': pages, 'pages': nbytes // PAGE,
                  'found_is_sentinel': all(view[start] == SENTINEL for start in range(first, min(nbytes, first + 64)))}
        self.last_mismatch = record
        expected.release()
        return 'bytes differ in %s (bank %d, write %d, slot %d): first at drive offset %d, expected %#04x found %#04x, %d of %d pages wrong%s' % (
            label, bank, index, slot, record['drive_offset'], record['expected'], record['found'], pages, record['pages'],
            ' (the sentinel: the drive wrote nothing there)' if record['found_is_sentinel'] else '')


def summarize_latencies(latencies):
    if not latencies:
        return {'p50_ms': None, 'p99_ms': None, 'max_ms': None}
    return {'p50_ms': percentile(latencies, 0.5) * 1e3, 'p99_ms': percentile(latencies, 0.99) * 1e3, 'max_ms': max(latencies) * 1e3}


def run_arm(run, size, qd, seconds, min_batches):
    """One cell of the sweep: batches of `qd` concurrent writes of `size` bytes from bank `args.bank`, for at least `seconds` of write time."""
    args = run.args
    bank = args.bank
    slots = max(1, len(run.patterns[bank]) // size)
    label = 'size %d qd %d' % (size, qd)
    batches, wall_total, latencies = 0, 0.0, []
    verified, compared = 0, 0
    process_before = time.process_time()
    write_cpu_before = run.cpu['batch_process']
    threads_before = dict(run.cpu['threads'])
    with concurrent.futures.ThreadPoolExecutor(max_workers=qd) as pool:
        while True:
            offset = run.scratch.take(qd * size)
            if run.verify_on:
                run.paint(offset, qd * size)
            ops = [(run.data_base + (index % slots) * size, size, offset + index * size) for index in range(qd)]
            wall, each = run.batch(pool, ops)
            wall_total += wall
            latencies += each
            batches += 1
            run.aer.check(force=False, interval=args.aer_interval)
            if run.verify_on:
                result = run.verify(label, bank, offset, size, qd, slots)
                verified += result['ops']
                compared += result['bytes']
            if wall_total >= seconds and batches >= min_batches:
                break
    total = batches * qd * size
    threads = [value - threads_before.get(ident, 0.0) for ident, value in run.cpu['threads'].items()]
    write_cpu = run.cpu['batch_process'] - write_cpu_before
    return dict({'size': size, 'iovecs': 1 if not run.iov else -(-size // run.iov), 'qd': qd, 'bank': bank, 'card_slots': slots,
                 'batches': batches, 'writes': batches * qd, 'bytes': total, 'seconds': wall_total, 'gbps': gbps(total, wall_total),
                 'cpu': {'process_seconds': write_cpu, 'worker_seconds': sum(threads), 'busiest_worker_seconds': max(threads) if threads else 0.0,
                         'process_fraction_of_write_time': write_cpu / max(1e-9, wall_total),
                         'non_write_process_seconds': max(0.0, time.process_time() - process_before - write_cpu)},
                 'verify': {'mode': args.write_verify, 'writes': verified, 'bytes': compared, 'ok': True if run.verify_on else None}},
                **{'write_' + key: value for key, value in summarize_latencies(latencies).items()})


def bank_sequence(banks, rounds, seed):
    """`rounds` banks drawn by shuffled passes over `banks`, never the same bank twice in a row when there is a choice."""
    rng = random.Random('p2p-reaim:%d' % seed)
    order = []
    while len(order) < rounds:
        passes = list(banks)
        rng.shuffle(passes)
        if order and len(passes) > 1 and passes[0] == order[-1]:
            passes[0], passes[-1] = passes[-1], passes[0]
        order += passes
    return order[:rounds]


def run_reaim(run):
    """Re-aim the window to another bank, prove it with the canary, write a few objects from it, and read them back; many times over. The
    writes of one round finish before the next re-aim, as the tier's I/O scheduler guarantees."""
    args = run.args
    order = bank_sequence(args.banks, args.reaim_rounds, args.seed)
    size, count = args.reaim_op_bytes, args.reaim_ops
    configure, canary, latencies = [], [], []
    verified = 0
    per_bank = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=count) as pool:
        for bank in order:
            record = run.aimer.aim(bank)
            offset = run.scratch.take(count * size)
            if run.verify_on:
                run.paint(offset, count * size)
            ops = [(run.data_base + index * size, size, offset + index * size) for index in range(count)]
            _, each = run.batch(pool, ops)
            latencies += each
            if run.verify_on:
                verified += run.verify('re-aim round bank %d' % bank, bank, offset, size, count, count)['bytes']
            configure.append(record['configure_ms'])
            canary.append(record['canary_ms'])
            per_bank[str(bank)] = per_bank.get(str(bank), 0) + 1
            run.aer.check(force=False, interval=args.aer_interval)
    return dict({'rounds': len(order), 'banks': sorted(set(order)), 'rounds_per_bank': per_bank, 'canary_ok': len(order),
                 'writes_per_round': count, 'write_size': size, 'bytes': len(order) * count * size,
                 'configure_ms': {'p50': percentile(configure, 0.5), 'p99': percentile(configure, 0.99), 'max': max(configure)},
                 'canary_read_ms': {'p50': percentile(canary, 0.5), 'p99': percentile(canary, 0.99), 'max': max(canary)},
                 'verify': {'mode': args.write_verify, 'bytes': verified, 'ok': True if run.verify_on else None}},
                **{'write_' + key: value for key, value in summarize_latencies(latencies).items()})


def run_soak(run, minutes):
    """Sustained writes for `minutes`: batches back to back, every Nth one painted first and verified after (the last one always), AER sampled
    every --aer-interval seconds and the drive's state with it. Reports the write-only rate per 30 s bucket as well as overall, so a throttle
    shows as a falling rate."""
    args = run.args
    size, qd, every = args.soak_size, args.soak_qd, max(1, args.soak_verify_every)
    bank = args.bank
    slots = max(1, len(run.patterns[bank]) // size)
    end = time.monotonic() + minutes * 60.0
    started = time.monotonic()
    cycles, write_seconds, latencies, written = 0, 0.0, [], 0
    buckets = {}
    verified_cycles, compared = 0, 0
    process_before = time.process_time()
    write_cpu_before = run.cpu['batch_process']
    threads_before = dict(run.cpu['threads'])
    with concurrent.futures.ThreadPoolExecutor(max_workers=qd) as pool:
        while True:
            last = time.monotonic() >= end
            check = run.verify_on and (last or cycles % every == every - 1)
            offset = run.scratch.take(qd * size)
            if check:
                run.paint(offset, qd * size)
            ops = [(run.data_base + (index % slots) * size, size, offset + index * size) for index in range(qd)]
            wall, each = run.batch(pool, ops)
            cycles += 1
            write_seconds += wall
            latencies += each
            written += qd * size
            slot = int((time.monotonic() - started) // SOAK_BUCKET_SECONDS)
            bucket = buckets.setdefault(slot, [0, 0.0])
            bucket[0] += qd * size
            bucket[1] += wall
            run.aer.check(force=False, interval=args.aer_interval)
            if check:
                compared += run.verify('soak cycle %d' % cycles, bank, offset, size, qd, slots)['bytes']
                verified_cycles += 1
            if last:
                break
            if args.soak_max_gbps > 0:
                ahead = written / (args.soak_max_gbps * 1e9) - (time.monotonic() - started)
                if ahead > 0:
                    time.sleep(min(ahead, 5.0))
    threads = [value - threads_before.get(ident, 0.0) for ident, value in run.cpu['threads'].items()]
    write_cpu = run.cpu['batch_process'] - write_cpu_before
    wall = time.monotonic() - started
    rates = [gbps(count, seconds) for _, (count, seconds) in sorted(buckets.items()) if seconds > 0]
    return dict({'minutes': minutes, 'wall_seconds': wall, 'cycles': cycles, 'write_size': size, 'qd': qd, 'bank': bank, 'bytes': written,
                 'write_seconds': write_seconds, 'gbps': gbps(written, write_seconds), 'effective_gbps': gbps(written, wall), 'paced_to_gbps': args.soak_max_gbps or None,
                 'bucket_gbps': rates, 'min_bucket_gbps': min(rates) if rates else None,
                 'cpu': {'process_seconds': write_cpu, 'worker_seconds': sum(threads), 'busiest_worker_seconds': max(threads) if threads else 0.0,
                         'process_fraction_of_write_time': write_cpu / max(1e-9, write_seconds),
                         'non_write_process_seconds': max(0.0, time.process_time() - process_before - write_cpu)},
                 'verify': {'mode': args.write_verify, 'cycles_verified': verified_cycles, 'every': every, 'bytes': compared,
                            'ok': True if verified_cycles else None}},
                **{'write_' + key: value for key, value in summarize_latencies(latencies).items()})


def prove_banks(window, aimer, cpu, plan, canaries):
    """Before any data goes into the card: write each bank's canary page, then aim at every bank again and read its canary back. A window
    that does not move when told to (the same canary at every bank, or the wrong one) is found here, while only a page has been written."""
    base = plan['dest']['window_offset']
    noc = plan['noc']
    for bank in sorted(canaries):
        x, y = dram_bank_xy(bank)
        window.configure(x, y, noc['addr'], noc['ordering'])
        cpu.write(base, canaries[bank])
    for bank in sorted(canaries):
        aimer.aim(bank)


def fill_banks(aimer, cpu, plan, patterns):
    """Put each bank's data pattern into the card through the window (aim and prove the aim, write, read the ends back), after its canary
    page. The region sits at the same window offset in every bank."""
    base = plan['dest']['window_offset']
    checked = {}
    for bank in sorted(patterns):
        aimer.aim(bank)
        data = memoryview(patterns[bank])
        try:
            for start in range(0, len(data), 8 << 20):
                cpu.write(base + PAGE + start, data[start:start + (8 << 20)])
            tail = max(0, len(data) - PAGE)
            first = cpu.read(base + PAGE, min(PAGE, len(data))) == bytes(data[:PAGE])
            last = cpu.read(base + PAGE + tail, len(data) - tail) == bytes(data[tail:])
        finally:
            data.release()
        checked[bank] = bool(first and last)
        if not checked[bank]:
            raise StopRun('fill', 'bank %d does not read back what was written into it through the window (first page %s, last page %s): '
                                  'not writing it to the drive' % (bank, first, last))
    return checked


def run_write_probe(card, sink, args):
    """The write flow, against a card and a drive (real or mock). Returns the report; raises ProbeError on a failed precondition (before
    any DMA). From the first DMA on, a stop condition is recorded in the report (`aborted`) and the result is FAIL."""
    check_write_args(args)
    scratch_end = args.scratch_start + args.scratch_size
    if scratch_end > sink.length:
        raise ProbeError('the scratch range ends at %d, beyond the %d bytes of %s' % (scratch_end, sink.length, sink.path))
    report = {'mode': 'write', 'notes': [NOTE_WRITE_FILL, NOTE_WRITE_CPU, NOTE_WRITE_QD, NOTE_WRITE_AER], 'sweep': [], 'reaim': None, 'soak': None,
              'aborted': None, 'mismatch': None}
    if sink.is_block:
        reason = block_device_in_use(sink.path)
        if reason:
            raise ProbeError('refusing to write to %s: %s' % (sink.path, reason))
    # the range must be blank, or the operator must say it is theirs
    if args.own_scratch:
        blank = None
        report['notes'].append('--i-own-this-scratch-range: the range was not checked for content')
    else:
        found = first_nonzero_in_range(sink, args.scratch_start, args.scratch_size)
        if found is not None:
            raise ProbeError('the scratch range [%d, +%d) of %s is not blank (first non-zero byte at %d): the probe overwrites it. If it is yours to '
                             'overwrite, pass --i-own-this-scratch-range with the same range' % (
                                 args.scratch_start, args.scratch_size, sink.path, args.scratch_start + found))
        blank = True
    sink.allow(args.scratch_start, scratch_end)
    report['scratch'] = {'target': sink.path, 'kind': 'block' if sink.is_block else 'file', 'start': args.scratch_start, 'size': args.scratch_size,
                         'end': scratch_end, 'start_lba512': args.scratch_start // 512, 'blank': blank, 'owner_flag': bool(args.own_scratch),
                         'device_bytes': sink.length}
    # the drive's links, before the card is touched
    drive_chain = sink.pci_chain()
    if sink.is_block and not drive_chain and not args.allow_unknown_link:
        raise ProbeError('no PCI function was found above %s: its link speed and AER counters cannot be read; --allow-unknown-link goes on' % sink.path)
    if not drive_chain:
        report['notes'].append('no PCI function was found above the target (a regular file?): the drive\'s link, state and AER counters were not checked')
    drive_links = check_links('drive', drive_chain, args.min_drive_gts, args.allow_unknown_link) if drive_chain else []
    drive_function = drive_chain[-1] if drive_chain else None
    if drive_function and any(state != 'live' for state in nvme_states(drive_function)):
        raise ProbeError('the NVMe controller of %s is not live: %s' % (sink.path, nvme_states(drive_function)))
    info = card.p2pmem_info()
    if info.get('in_use'):
        raise ProbeError('the published slice is already allocated (p2pmem available is below its size): another process holds it')
    plan, data_bytes = write_plan(args, info)
    holders = [] if args.allow_shared else card.other_openers()
    if holders:
        raise ProbeError('the card is open in process(es) %s: nothing may hold it during the test (--allow-shared overrides)'
                         % ', '.join(str(pid) for pid in holders))
    patterns = {bank: bank_pattern(args.seed, bank, size) for bank, size in sorted(data_bytes.items())}
    canaries = {bank: canary_page(args.seed, bank) for bank in patterns}
    card_chain = card.pci_chain()
    monitor = AerMonitor(list(card_chain) + list(drive_chain), drive_function)
    report.update({'plan': plan, 'info': info, 'cpu_map': args.cpu_map, 'aer': {'before': monitor.before}})

    card.power_up(args.settle)
    report['links'] = {'drive': drive_links, 'card': check_links('card', card_chain, 0.0, args.allow_unknown_link) if card_chain else []}
    window = card.acquire_window(plan['window'])
    stack = contextlib.ExitStack()
    stack.callback(window.close)
    run = aimer = None
    began = time.time()
    try:
        noc = plan['noc']
        window.configure(noc['x'], noc['y'], noc['addr'], noc['ordering'])
        report['pinned'] = window.pin()
        if not report['pinned'] and not args.allow_unpinned:
            raise ProbeError('the window could not be pinned with a dma-buf export (tt-kmd older than 2.10.0-rc1?): without the pin a close or FREE_TLB '
                             'under a live mapping, or a reset, would not be stopped; --allow-unpinned goes on regardless')
        cpu = window.cpu_window(WINDOW_4G, uc=args.cpu_map == 'uc')
        window_selftest(cpu, dict(plan, dest=dict(plan['dest'], size=plan['region_bytes'])))
        aimer = Aimer(window, cpu, plan, canaries)
        try:
            fill_began = time.perf_counter()
            prove_banks(window, aimer, cpu, plan, canaries)
            report['fill'] = fill_banks(aimer, cpu, plan, patterns)
            fill_seconds = time.perf_counter() - fill_began
            fill_bytes = sum(len(pattern) for pattern in patterns.values()) + PAGE * len(canaries)
            # what the CPU filled through the window (uncached by default, a speed nobody has measured on the rig yet) and how long it took
            report['fill_stats'] = {'bytes': fill_bytes, 'seconds': fill_seconds, 'mbps': fill_bytes / 1e6 / max(1e-9, fill_seconds), 'cpu_map': args.cpu_map}
            buffer = card.p2p_buffer(info['size'])
            stack.callback(buffer.close)
            first_bus = buffer.bus_addr_of(0)
            expected = card.expected_phys(info)
            report['bus_addr_check'] = None if not first_bus or not expected else first_bus == expected
            if report['bus_addr_check'] is False:
                raise ProbeError('the mapping starts at physical address %#x, the slice at %#x: the DMA would not land where the TLB points' % (first_bus, expected))
            run = WriteRun(args, sink, Scratch(args.scratch_start, args.scratch_size), buffer, plan, patterns, aimer, monitor,
                           Budget(int(args.max_write_gib * GIB)))
            stack.callback(run.close)
            aimer.aim(args.bank)
            monitor.start()
            if not args.no_sweep:
                for size in args.sizes:
                    for qd in args.qds:
                        report['sweep'].append(run_arm(run, size, qd, args.arm_seconds, args.min_batches))
                        monitor.check(force=True)
            if args.reaim:
                report['reaim'] = run_reaim(run)
                monitor.check(force=True)
            if args.soak > 0:
                aimer.aim(args.bank)
                report['soak'] = run_soak(run, args.soak)
            monitor.check(force=True)
        except StopRun as error:
            report['aborted'] = {'kind': error.kind, 'message': str(error)}
            report['mismatch'] = run.last_mismatch if run is not None and error.kind == 'mismatch' else None
        except ProbeError as error:
            report['aborted'] = {'kind': 'error', 'message': str(error)}
        finally:
            report['aer']['after'] = snapshot_aer(monitor.functions)
            report['aer']['delta'] = monitor.delta()
            report['aer']['samples'] = monitor.samples
            report['aer'].update(monitor.summary(), interval_seconds=args.aer_interval)
    finally:
        report['bytes'] = {'p2w': run.p2w_bytes if run else 0, 'sentinel': run.sentinel_bytes if run else 0, 'written_to_drive': sink.written}
        report['aims'] = aimer.records[-200:] if aimer else []
        stack.close()
    report['seconds'] = time.time() - began
    return finish_write(report, args)


def finish_write(report, args):
    """The verdict. PASS needs every applicable condition True (the go/no-go of the design: byte-exact, no uncorrectable AER, a sustained rate,
    a sub-run that drains inside the bound, a canary that matched after every re-aim); FAIL if any is False or the run stopped; UNVERIFIED if
    one could not be measured (nothing compared, no AER counters)."""
    verdict = {}
    stopped = report['aborted']
    verifies = [arm['verify'] for arm in report['sweep']] + [part['verify'] for part in (report['reaim'], report['soak']) if part]
    if stopped and stopped['kind'] == 'mismatch':
        verdict['byte_exact'] = False
    elif args.write_verify == 'none' or not verifies:
        verdict['byte_exact'] = None
    else:
        verdict['byte_exact'] = all(item['ok'] for item in verifies)
    delta = report.get('aer', {}).get('delta')
    verdict['no_uncorrectable_aer'] = None if delta is None or not delta['measured'] else delta['uncorrectable'] == 0
    soak = report['soak']
    deep = [arm['gbps'] for arm in report['sweep'] if arm['qd'] >= 4]          # queue depth 1 is the drain measurement, not a sustained load
    rates = deep or [arm['gbps'] for arm in report['sweep']]
    if soak or rates:
        verdict['p2w_sustained_gbps'] = soak['gbps'] if soak else min(rates)
        verdict['p2w_gbps_ok'] = verdict['p2w_sustained_gbps'] >= args.min_write_gbps
    drain = [arm for arm in report['sweep'] if arm['size'] == SUBRUN_BYTES and arm['qd'] == 1]
    if drain:
        # Design 2.8: a sub-run drains in --drain-ms "at the measured P2W rate", that is SUBRUN_BYTES divided by the rate of the QD 1 cell
        # (all its writes, the first and the slowest included). The slowest single write and the 99th percentile are reported beside it,
        # not gated on: one slow write out of hundreds is information for the owner, not the bound the scheduler is sized by.
        cell = drain[0]
        verdict['subrun_drain_ms'] = SUBRUN_BYTES / (max(cell['gbps'], 1e-12) * 1e9) * 1e3
        verdict['subrun_drain_ok'] = verdict['subrun_drain_ms'] <= args.drain_ms
        verdict['subrun_drain_p99_ms'] = cell['write_p99_ms']
        verdict['subrun_drain_max_ms'] = cell['write_max_ms']
    if args.reaim:
        reaim = report['reaim']
        if stopped and stopped['kind'] == 'canary':
            verdict['reaim_canary_ok'] = False
        else:
            verdict['reaim_canary_ok'] = (reaim['canary_ok'] == reaim['rounds']) if reaim else None
    flags = [value for key, value in verdict.items() if key.endswith('_ok') or key in ('byte_exact', 'no_uncorrectable_aer')]
    verdict['go'] = all(value is True for value in flags) and not stopped
    report['verdict'] = verdict
    report['below_min_gbps'] = verdict.get('p2w_gbps_ok') is False
    if stopped or any(value is False for value in flags):
        report['result'] = FAIL
    elif any(value is None for value in flags):
        report['result'] = UNVERIFIED
    else:
        report['result'] = PASS
    report['ok'] = report['result'] == PASS
    return report


VERIFY_WORDS = {True: 'OK', False: 'MISMATCH', None: 'skipped'}


def emit_write(report, as_json):
    if as_json:
        print(json.dumps(report, indent=2, sort_keys=True))
        return
    plan = report.get('plan', {})
    scratch = report.get('scratch', {})
    print('write arm: window %s, cpu map %s, %d bytes per bank at window offset %#x, banks %s' % (
        plan.get('window'), report.get('cpu_map'), plan.get('region_bytes', 0), plan['dest']['window_offset'], ','.join(sorted(plan.get('banks', {})))))
    print('scratch: %s [%d, +%d) lba512 %d, %s' % (scratch.get('target'), scratch.get('start', 0), scratch.get('size', 0), scratch.get('start_lba512', 0),
                                                    'blank' if scratch.get('blank') else 'owner flag'))
    for label, links in sorted(report.get('links', {}).items()):
        if links:
            print('links %-5s %s' % (label, ', '.join('%s %s GT/s x%s' % (link['bdf'], link['speed_gts'], link['width']) for link in links)))
    for arm in report['sweep']:
        print('write %9d B x%-2d qd %-2d %8.3f GB/s  op p50 %7.2f p99 %7.2f max %7.2f ms  cpu %5.2f cores (writes only)  verify %s' % (
            arm['size'], arm['iovecs'], arm['qd'], arm['gbps'], arm['write_p50_ms'], arm['write_p99_ms'], arm['write_max_ms'],
            arm['cpu']['process_fraction_of_write_time'], VERIFY_WORDS[arm['verify']['ok']]))
    if report['reaim']:
        part = report['reaim']
        print('re-aim %d rounds over banks %s: canary %d/%d, configure p50 %.3f max %.3f ms, canary read p50 %.3f max %.3f ms, verify %s' % (
            part['rounds'], part['banks'], part['canary_ok'], part['rounds'], part['configure_ms']['p50'], part['configure_ms']['max'],
            part['canary_read_ms']['p50'], part['canary_read_ms']['max'], VERIFY_WORDS[part['verify']['ok']]))
    if report['soak']:
        part = report['soak']
        print('soak %.1f min: %d cycles, %.1f GB written, %.3f GB/s while writing (slowest 30 s bucket %s), %.3f GB/s overall, verify %s' % (
            part['minutes'], part['cycles'], part['bytes'] / 1e9, part['gbps'],
            '%.3f' % part['min_bucket_gbps'] if part['min_bucket_gbps'] is not None else 'n/a', part['effective_gbps'], VERIFY_WORDS[part['verify']['ok']]))
    verdict = report.get('verdict', {})
    if 'subrun_drain_ms' in verdict:
        print('sub-run drain: %.2f ms for %d B at the QD 1 cell\'s rate (gate %s); slowest single write %.2f ms, p99 %.2f ms' % (
            verdict['subrun_drain_ms'], SUBRUN_BYTES, 'held' if verdict['subrun_drain_ok'] else 'MISSED', verdict['subrun_drain_max_ms'], verdict['subrun_drain_p99_ms']))
    fill = report.get('fill_stats')
    if fill:
        print('card fill: %.1f MB through the %s CPU window in %.1f s (%.1f MB/s)' % (fill['bytes'] / 1e6, fill['cpu_map'], fill['seconds'], fill['mbps']))
    delta = report.get('aer', {}).get('delta')
    if delta:
        print('aer: %s, correctable +%d, uncorrectable +%d %s' % ('measured' if delta['measured'] else 'NOT MEASURED', delta['correctable'], delta['uncorrectable'],
                                                                 json.dumps(delta['by_function'], sort_keys=True) if delta['by_function'] else ''))
        if 'max_gap_seconds' in report['aer']:
            print('aer sampling: every %.2f s asked, %d samples, longest gap %.2f s, %.2f ms per sample' % (
                report['aer']['interval_seconds'], report['aer']['sample_count'], report['aer']['max_gap_seconds'], report['aer']['mean_check_ms'] or 0.0))
    if report['aborted']:
        print('STOPPED (%s): %s' % (report['aborted']['kind'], report['aborted']['message']))
    print('bytes: %.2f GB from the card to the drive, %.2f GB of sentinel' % (report['bytes']['p2w'] / 1e9, report['bytes']['sentinel'] / 1e9))
    print('verdict: %s' % json.dumps(report['verdict'], sort_keys=True))
    for note in report.get('notes', []):
        print('note: %s' % note)
    print('RESULT', report['result'])


def estimate_writes(args, drive_gbps=7.0):
    """An upper bound on what a run writes to the drive, assuming it can take `drive_gbps`: the sweep, the re-aim arm and the soak, the
    sentinel included (so the operator can weigh it against the drive's endurance)."""
    total = 0.0
    if not args.no_sweep:
        total += len(args.sizes) * len(args.qds) * args.arm_seconds * drive_gbps * 1e9 * 2
    if args.reaim:
        total += args.reaim_rounds * args.reaim_ops * args.reaim_op_bytes * 2
    if args.soak > 0:
        pace = args.soak_max_gbps if args.soak_max_gbps > 0 else drive_gbps
        total += args.soak * 60 * min(pace, drive_gbps) * 1e9 * (1 + 1.0 / max(1, args.soak_verify_every))
    return min(total, args.max_write_gib * GIB)


def dry_run_write(args):
    """The plan of a write run, from the arguments alone: no file is opened, no device read."""
    info = {'bar': 4, 'offset': 0, 'size': 4 * GIB, 'bus_addr': 0, 'tlb_first': FIRST_4G_TLB, 'tlb_last': FIRST_4G_TLB}
    if args.scratch_start is not None and args.scratch_size is not None:
        check_write_args(args)
    plan, data_bytes = write_plan(args, info)
    arms = [] if args.no_sweep else [{'size': size, 'qd': qd, 'iovecs': 1 if not args.iov_bytes else -(-size // args.iov_bytes)}
                                     for size in args.sizes for qd in args.qds]
    return {'dry_run': True, 'mode': 'write', 'plan': plan, 'arms': arms,
            'reaim': {'banks': args.banks, 'rounds': args.reaim_rounds, 'ops': args.reaim_ops, 'op_bytes': args.reaim_op_bytes} if args.reaim else None,
            'soak': {'minutes': args.soak, 'size': args.soak_size, 'qd': args.soak_qd, 'max_gbps': args.soak_max_gbps or None} if args.soak > 0 else None,
            'scratch': {'start': args.scratch_start, 'size': args.scratch_size, 'largest_batch': required_scratch(args), 'owner_flag': bool(args.own_scratch)},
            'write_budget_gib': args.max_write_gib, 'estimated_writes_gib_at_most': estimate_writes(args) / GIB, 'cpu_map': args.cpu_map,
            'note': 'a real run also refuses a range that is not blank (without --i-own-this-scratch-range), a link at 2.5 GT/s, a drive that is '
                    'mounted or held, and a slice somebody else holds; none of that is looked at here'}


# -- the mock banked card and a mock drive: the write flow, in memory -----------------------------------------------------

def bank_of_xy(x, y):
    for bank in range(NUM_BANKS):
        if dram_bank_xy(bank)[0] == x and 0 <= y - dram_bank_xy(bank)[1] < 3:
            return bank
    raise ValueError('(%d, %d) is not a DRAM bank endpoint' % (x, y))


FAKE_AER_CORRECTABLE = ('RxErr', 'BadTLP', 'BadDLLP', 'Rollover', 'Timeout', 'NonFatalErr', 'CorrIntErr', 'HeaderOF', 'TOTAL_ERR_COR')
FAKE_AER_UNCORRECTABLE = ('Undefined', 'DLP', 'SDES', 'TLP', 'FCP', 'CmpltTO', 'CmpltAbrt', 'UnxCmplt', 'RxOF', 'MalfTLP', 'ECRC', 'UnsupReq',
                          'ACSViol', 'UncorrIntErr', 'BlockedTLP', 'AtomicOpBlocked', 'TLPBlockedErr', 'PoisonTLPBlocked')


def make_fake_sysfs(root, card_gts=32.0, drive_gts=16.0, state='live'):
    """A tree shaped like sysfs for a card (root port, endpoint) and a drive (root port, switch up and down ports, endpoint), with link
    speeds, zeroed AER counters and an NVMe controller. Returns the directories the probe needs."""
    base = os.path.join(root, 'devices', 'pci0000:00')
    card = [os.path.join(base, '0000:00:01.0'), None]
    card[1] = os.path.join(card[0], '0000:01:00.0')
    drive = [os.path.join(base, '0000:00:02.0')]
    for name in ('0000:02:00.0', '0000:03:01.0', '0000:04:00.0'):
        drive.append(os.path.join(drive[-1], name))
    for chain, speed in ((card, card_gts), (drive, drive_gts)):
        for index, directory in enumerate(chain):
            os.makedirs(directory, exist_ok=True)
            for name, text in (('current_link_speed', '%.1f GT/s PCIe' % speed), ('max_link_speed', '%.1f GT/s PCIe' % speed), ('current_link_width', '4')):
                with open(os.path.join(directory, name), 'w') as handle:
                    handle.write(text + '\n')
            for name, counters in (('aer_dev_correctable', FAKE_AER_CORRECTABLE), ('aer_dev_nonfatal', FAKE_AER_UNCORRECTABLE + ('TOTAL_ERR_NONFATAL',)),
                                   ('aer_dev_fatal', FAKE_AER_UNCORRECTABLE + ('TOTAL_ERR_FATAL',))):
                with open(os.path.join(directory, name), 'w') as handle:
                    handle.write(''.join('%s 0\n' % counter for counter in counters))
            if index == 0:
                for name in AER_ROOT_FILES:
                    with open(os.path.join(directory, name), 'w') as handle:
                        handle.write('0\n')
    controller = os.path.join(drive[-1], 'nvme', 'nvme0')
    block = os.path.join(controller, 'nvme0n1')
    os.makedirs(block, exist_ok=True)
    with open(os.path.join(controller, 'state'), 'w') as handle:
        handle.write(state + '\n')
    return {'card_chain': card, 'drive_chain': drive, 'block_dir': block}


def anonymous_file(name, size):
    """A descriptor for `size` bytes of memory nothing else can open: memfd_create where the C library has it, else an unlinked temporary file."""
    if hasattr(os, 'memfd_create'):
        fd = os.memfd_create(name)
    else:
        with tempfile.TemporaryFile() as handle:
            fd = os.dup(handle.fileno())
    os.ftruncate(fd, size)
    return fd


class MockBankedCard(MockCard):
    """A card with eight DRAM banks, each a window onto one memfd, and a window that selects between them: what the CPU window and the
    p2pmem mapping see is the bank the window was last aimed at (offset x of the slice is offset x of that bank). `fault`: 'stale-aim' (a
    CONFIGURE_TLB after the first does not move the window), 'wrong-bank' (the CPU window follows the aim but the drive's DMA always reads
    bank 0)."""

    def __init__(self, bank_bytes=64 << 20, fault=None, window=FIRST_4G_TLB, sysfs=None, in_use=0, openers=(), slice_offset=0):
        MockCard.__init__(self, slice_size=bank_bytes, slice_offset=slice_offset, window=window, fault=fault, in_use=in_use, openers=openers)
        self.bank_bytes = bank_bytes
        self.memfd = anonymous_file('p2p-fake-card', NUM_BANKS * bank_bytes)
        self.banks = [mmap.mmap(self.memfd, bank_bytes, offset=bank * bank_bytes) for bank in range(NUM_BANKS)]
        self.aimed = 0
        self.configures = 0
        self.stale_after = 1                 # the 'stale-aim' fault: CONFIGURE_TLB calls after this many are ignored
        self.write_limit = {}                # bank -> offset beyond which CPU writes through the window are lost (a bank that does not store)
        self.cpu_maps = []                   # the CPU mapping each window was asked for: 'uc' or 'wc'
        self.tmp = None
        if sysfs is None:
            self.tmp = tempfile.TemporaryDirectory()
            sysfs = make_fake_sysfs(self.tmp.name)
        self.sysfs = sysfs
        self.pci_dir = sysfs['card_chain'][-1]

    def pci_chain(self):
        return list(self.sysfs['card_chain'])

    def acquire_window(self, wanted):
        self.events.append('acquire_window')
        if wanted in self.held_windows:
            raise ProbeError('window %d is held by another process' % wanted)
        return MockBankedWindow(self)

    def p2p_buffer(self, size):
        self.events.append('p2p_buffer')
        return MockBankedBuffer(self)

    def close(self):
        for bank in self.banks:
            try:
                bank.close()
            except BufferError:
                pass
        os.close(self.memfd)
        if self.tmp is not None:
            self.tmp.cleanup()


class MockBankedWindow(MockWindow):
    def configure(self, x, y, addr, ordering):
        self.card.events.append('configure')
        self.card.configured = (x, y, addr, ordering)
        self.card.configures += 1
        if self.card.fault == 'stale-aim' and self.card.configures > self.card.stale_after:
            return
        self.card.aimed = bank_of_xy(x, y)

    def cpu_window(self, size, uc=False):
        self.card.cpu_maps.append('uc' if uc else 'wc')
        return MockBankedCpu(self.card, dead=self.card.selftest_fails)


class MockBankedCpu(MockCpu):
    def __init__(self, card, dead=False):
        MockCpu.__init__(self, None, dead=dead)
        self.card = card

    def read(self, offset, size):
        return bytes(size) if self.dead else bytes(self.card.banks[self.card.aimed][offset:offset + size])

    def write(self, offset, data):
        if self.dead:
            return
        limit = self.card.write_limit.get(self.card.aimed)
        if limit is not None:
            data = data[:max(0, limit - offset)]
        self.card.banks[self.card.aimed][offset:offset + len(data)] = data


class MockBankedBuffer(object):
    """The p2pmem mapping: `view` shows the bank the window is aimed at (bank 0 under the 'wrong-bank' fault)."""

    def __init__(self, card):
        self.card = card
        self.views = [memoryview(bank) for bank in card.banks]
        self.size = card.slice_size
        self.phys = 0x38000000000 + card.slice_offset

    @property
    def view(self):
        return self.views[0 if self.card.fault == 'wrong-bank' else self.card.aimed]

    def bus_addr_of(self, offset):
        return self.phys + offset

    def close(self):
        for view in self.views:
            view.release()


class MockSink(Sink):
    """A regular file as the drive (buffered: not every filesystem the tests run on takes O_DIRECT), with the drive's sysfs tree supplied by
    the mock card. `fault`: 'flip' (one byte of the data lands wrong), 'drop' (a write is acknowledged and nothing is written)."""

    def __init__(self, path, card=None, fault=None):
        Sink.__init__(self, path, direct=False)
        self.card = card
        self.fault = fault
        self.calls = 0

    def sysfs_dir(self, sys_dev_block='/sys/dev/block'):
        return self.card.sysfs['block_dir'] if self.card is not None else None

    def pci_chain(self):
        return list(self.card.sysfs['drive_chain']) if self.card is not None else []

    def write_iov(self, views, offset):
        with self.lock:
            self.calls += 1
            number = self.calls
        if self.fault == 'drop' and number == 2:
            self._check_fence(offset, sum(len(view) for view in views))
            return sum(len(view) for view in views)
        if self.fault == 'flip' and number == 2:
            data = bytearray(b''.join(bytes(view) for view in views))
            data[len(data) // 2] ^= 0xFF
            with memoryview(data) as whole:
                return Sink.write_iov(self, [whole], offset)
        return Sink.write_iov(self, views, offset)


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
    parser.add_argument('--mock-fault', choices=('flip', 'misroute', 'stall', 'stale-aim', 'wrong-bank', 'drop'),
                        help='with --mock: inject a defect the verify must catch (the last three are for --mode write)')
    parser.add_argument('--json', action='store_true', help='print the report as JSON only')
    parser.add_argument('--json-out', metavar='FILE', help='also write the report as JSON to FILE')
    write = parser.add_argument_group('write arm (--mode write): the drive DMA-reads the card\'s DRAM into a scratch range of a drive you own')
    write.add_argument('--mode', choices=('read', 'write'), default='read', help='read: the drive into the card (the default). write: the card into the drive')
    write.add_argument('--target', help='the drive (or a file standing in for it) the card\'s bytes are written to, opened O_DIRECT')
    write.add_argument('--scratch-start', type=parse_size, help='first byte of the scratch range (multiple of 4096, at least 1M): the probe writes nowhere else')
    write.add_argument('--scratch-size', type=parse_size, help='size of the scratch range; at least queue depth x write size of the largest arm')
    write.add_argument('--i-own-this-scratch-range', dest='own_scratch', action='store_true',
                       help='the range is not blank and the operator says it may be overwritten; needs --scratch-start and --scratch-size. '
                            'Without it a range that is not all zeros is refused')
    write.add_argument('--sizes', type=parse_int_list, default=list(WRITE_SIZES),
                       help='bytes per write (one pwritev), comma separated (default 696320,2785280,22282240: a shard, an object, an 8-object sub-run)')
    write.add_argument('--qd', dest='qds', type=parse_int_list, default=list(WRITE_QDS),
                       help='writes in flight, comma separated (default 1,4,8,16,32; 1 is the drain-bound measurement)')
    write.add_argument('--iov-bytes', type=parse_size, default=SHARD_BYTES, help='bytes per iovec inside a write (default 696320, one shard; 0 = one iovec)')
    write.add_argument('--arm-seconds', type=float, default=2.0, help='write time per size x queue-depth cell (default 2)')
    write.add_argument('--min-batches', type=int, default=2, help='batches per cell at least (default 2)')
    write.add_argument('--region', type=parse_size, default=128 << 20, help='bytes of pattern put in the sweep bank of the card (default 128M)')
    write.add_argument('--no-sweep', action='store_true', help='skip the size x queue-depth sweep (to run only --reaim and/or --soak)')
    write.add_argument('--reaim', action='store_true', help='the re-aim arm: aim the window at each bank in turn, canary read after every aim, write, verify')
    write.add_argument('--banks', type=parse_int_list, default=list(range(NUM_BANKS)), help='banks the re-aim arm visits (default 0-7)')
    write.add_argument('--reaim-rounds', type=int, default=64, help='re-aims in the re-aim arm (default 64)')
    write.add_argument('--reaim-ops', type=int, default=4, help='concurrent writes per aim epoch (default 4)')
    write.add_argument('--reaim-op-bytes', type=parse_size, default=OBJECT_BYTES, help='bytes per write in the re-aim arm (default 2785280)')
    write.add_argument('--soak', type=float, default=0.0, metavar='MINUTES', help='sustained writes for this many minutes, AER sampled throughout')
    write.add_argument('--soak-size', type=parse_size, default=OBJECT_BYTES, help='bytes per write in the soak (default 2785280)')
    write.add_argument('--soak-qd', type=int, default=8, help='writes in flight in the soak (default 8)')
    write.add_argument('--soak-verify-every', type=int, default=4, help='paint and verify every Nth soak cycle, and the last (default 4)')
    write.add_argument('--soak-max-gbps', type=float, default=2.0, help='pace the soak to this average rate, to spare the drive\'s endurance (default 2; 0 = unpaced)')
    write.add_argument('--aer-interval', type=float, default=0.5,
                       help='seconds between AER/drive-state samples inside an arm (default 0.5: an uncorrectable error stops the DMA within this plus '
                            'the segment in flight (a batch, a sentinel chunk, a read-back), about a second at most; a sample is a few dozen small '
                            'sysfs reads, about 0.2 ms on the test tree, and the report carries the mean measured on the rig)')
    write.add_argument('--write-verify', choices=('full', 'none'), default='full',
                       help='full: paint a sentinel, write, read back and compare every byte; none: write only, the run ends UNVERIFIED')
    write.add_argument('--max-write-gib', type=float, default=1024.0, help='stop before this many GiB have been written to the drive (default 1024)')
    write.add_argument('--min-write-gbps', type=float, default=0.25,
                       help='gate: sustained write rate at least this (default 0.25); a miss ends FAIL (exit 1). The W1 kit passes 0 and applies the gate '
                            'in its summary, so a slow drive costs no data')
    write.add_argument('--drain-ms', type=float, default=50.0,
                       help='gate: one 22282240-byte sub-run on its own drains within this many ms at the measured rate, that is 22282240 B divided by '
                            'the rate of the queue-depth-1 cell (default 50); the slowest write and the p99 are reported, not gated. A miss ends FAIL')
    write.add_argument('--cpu-map', choices=('uc', 'wc'), default='uc', help='CPU mapping of the window: uc (what the tier reads canaries through) or wc')
    write.add_argument('--min-drive-gts', type=float, default=16.0, help='refuse a drive link slower than this many GT/s (default 16; 0 = only Gen1 is refused)')
    write.add_argument('--allow-unknown-link', action='store_true', help='go on when a link speed or the drive\'s PCI function cannot be found (never for Gen1)')
    return parser


def resolve_target(args):
    if args.noc_x is None or args.noc_y is None:
        try:
            args.noc_x, args.noc_y = dram_bank_xy(args.bank)
        except ValueError as error:
            raise ProbeError(str(error)) from error


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


def write_json_report(path, report):
    """The JSON report file. Run through sudo, the probe is root, and a file root creates in the invoking user's home would be root's:
    the user who ran sudo (SUDO_UID, SUDO_GID) gets it instead, so the report can be read, moved and deleted without root. Best effort: a
    failed chown leaves the file with root, which is still readable."""
    with open(path, 'w') as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write('\n')
        handle.flush()
        owner = (os.environ.get('SUDO_UID', ''), os.environ.get('SUDO_GID', ''))
        if os.geteuid() == 0 and all(item.isdigit() for item in owner):
            try:
                os.fchown(handle.fileno(), int(owner[0]), int(owner[1]))
            except OSError:
                pass


def main_write(args):
    """--mode write: dry-run plan, or the flow against a real card and drive, or against the mocks."""
    resolve_target(args)
    if args.dry_run:
        print(json.dumps(dry_run_write(args), indent=2, sort_keys=True))
        return 0
    if not args.target:
        raise ProbeError('--target is required (the drive, or a file standing in for it, that the card\'s bytes are written to)')
    if args.mock:
        args.settle = 0
        need = args.dest_offset + 2 * PAGE + max(args.region, args.reaim_ops * args.reaim_op_bytes)
        card = MockBankedCard(bank_bytes=max(64 << 20, -(-need // PAGE) * PAGE), fault=args.mock_fault)
        sink = MockSink(args.target, card=card, fault=args.mock_fault)
    else:
        if os.geteuid() != 0:
            raise ProbeError('run as root: the p2pmem allocation, the TLB ioctls and the raw drive all need it')
        sink = Sink(args.target)
        try:
            card = TtCard(args.device)
        except OSError:
            sink.close()
            raise
    try:
        report = run_write_probe(card, sink, args)
    finally:
        sink.close()
        card.close()
    if args.json_out:
        write_json_report(args.json_out, report)
    emit_write(report, args.json)
    return EXIT_CODES[report['result']]


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        if args.mode == 'write':
            return main_write(args)
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
