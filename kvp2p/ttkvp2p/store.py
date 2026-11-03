"""The drive store, v1a (``gemma-kv`` format v2, mode "process"): superblocks, geometry, the drive file, discard.

LAYOUT (design 3.4)::

    byte 0        superblock slot A  (4 KiB)       generation flip: the valid slot with the higher generation wins
    byte 4096     superblock slot B  (4 KiB)
    byte 8192     DMA canary block   (8 KiB)       the start-up DMA canary (design 2.9) writes and reads this, never a slot
    data_off      slot 0 ...                       slot i at data_off + i * object_bytes; data_off is MiB aligned on a real drive

A superblock names: uuid, layout id, object_bytes, num_slots, data_off, the claim (sha256 of the by-id name and the size), the mode
("process" | "durable"), the creation time and the device size it was claimed at. Magic ``TTKVGEM\\0``, VERSION 2, so the Qwen store (v1,
magic ``QNVMSB01``) refuses a drive claimed here and this store refuses one that is not.

THE TIER NEVER FORMATS. ``open_store`` verifies; the claim tool (``ttkvp2p.claim``, owner-run) is the only writer of a superblock.
Mode "process" keeps nothing across a restart: the open verifies the superblock and the runtime queues a discard of the whole data
region; this module never writes a superblock again.

Standard library only (the claim tool runs on a host without vLLM). Python 3.10.
"""

import ctypes
import errno
import fcntl
import hashlib
import mmap
import os
import stat
import struct
import threading
import uuid as uuid_module
import zlib
from dataclasses import dataclass, replace
from typing import Callable, List, Optional, Tuple

ALIGN = 4096
SECTOR = 512                                  # the smallest object size a superblock may name (a CPU test's tiny layout); a block device needs ALIGN
SB_MAGIC = b'TTKVGEM\x00'
QWEN_MAGIC = b'QNVMSB01'                      # the retired Qwen store's superblock magic (the claim tool names it when it sees it)
VERSION = 2
SB_SLOT_A = 0
SB_SLOT_B = ALIGN
CANARY_OFF = 2 * ALIGN                        # LBA 2-3
CANARY_BYTES = 2 * ALIGN
HEADER_BYTES = CANARY_OFF + CANARY_BYTES      # the data region starts at the next alignment boundary at or after this
MIN_DATA_ALIGN = 1 << 20                      # a real drive's data region starts on a MiB
BIG_DEVICE = 64 << 20                         # below this the data region only needs ALIGN (the CPU tests use small files)
DISCARD_RUN_BYTES = 1 << 30                   # a whole-region discard goes in runs of at most 1 GiB
BLKDISCARD = 0x1277
MODES = ('process', 'durable')
PROBE_BYTES = 1 << 20                         # the claim tool looks at the first MiB of a device

# little endian: magic, version, flags, generation, uuid, layout id (32 bytes, NUL padded), object_bytes, num_slots, data_off,
# device_bytes, claim, mode (8 bytes, NUL padded), created (unix seconds), crc32 of everything before it
_SB = struct.Struct('<8sIIQ16s32sQQQQ32s8sQI')


class StoreError(Exception):
    """Base of this module's errors. ``code`` is one of the tier's closed UNAVAILABLE reasons."""

    code = 'superblock_unknown'


class DeviceAbsent(StoreError):
    code = 'device_absent'


class DeviceGone(StoreError):
    code = 'device_dead'


class DeviceBusy(StoreError):
    code = 'claim_busy'


class UnknownSuperblock(StoreError):
    code = 'superblock_unknown'


class LayoutMismatch(StoreError):
    code = 'layout_mismatch'


class CapacityShort(StoreError):
    code = 'capacity_short'


# --------------------------------------------------------------------------------------------------------------------
# Aligned memory (O_DIRECT needs aligned buffers)
# --------------------------------------------------------------------------------------------------------------------

def roundup(count: int, align: int = ALIGN) -> int:
    return (int(count) + align - 1) // align * align


class AlignedBuffer:
    """A page-aligned anonymous buffer: what O_DIRECT reads into and writes from."""

    def __init__(self, size: int):
        self.size = roundup(max(int(size), ALIGN))
        self._map = mmap.mmap(-1, self.size)
        self.view = memoryview(self._map)

    def close(self) -> None:
        self.view = None
        try:
            self._map.close()
        except BufferError:
            pass


# --------------------------------------------------------------------------------------------------------------------
# Superblock
# --------------------------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Superblock:
    generation: int
    uuid: bytes
    layout_id: str
    object_bytes: int
    num_slots: int
    data_off: int
    device_bytes: int
    claim: bytes
    mode: str
    created: int

    @property
    def data_bytes(self) -> int:
        return self.num_slots * self.object_bytes

    @property
    def end(self) -> int:
        """The first byte after the last slot: the smallest device the superblock fits."""
        return self.data_off + self.data_bytes


def claim_digest(by_id_name: str, size_bytes: int) -> bytes:
    """The claim: sha256 of the drive's by-id name and its size. The claim tool computes it from what the operator typed back;
    the tier recomputes it from the status file's by-id name and the device's size, so a cloned or swapped drive does not verify."""
    return hashlib.sha256(by_id_name.encode('utf-8') + b'\x00' + str(int(size_bytes)).encode('ascii')).digest()


def encode_superblock(sb: Superblock) -> bytes:
    layout = sb.layout_id.encode('utf-8')
    mode = sb.mode.encode('ascii')
    if len(layout) > 32 or len(mode) > 8 or len(sb.uuid) != 16 or len(sb.claim) != 32:
        raise ValueError('a superblock field does not fit')
    body = _SB.pack(SB_MAGIC, VERSION, 0, sb.generation, sb.uuid, layout, sb.object_bytes, sb.num_slots, sb.data_off,
                    sb.device_bytes, sb.claim, mode, sb.created, 0)
    crc = zlib.crc32(body[:-4]) & 0xFFFFFFFF
    body = body[:-4] + struct.pack('<I', crc)
    return body + bytes(ALIGN - len(body))


def decode_superblock(block: bytes) -> Optional[Superblock]:
    """The superblock a 4 KiB block holds, or None when the magic, the version or the checksum is wrong."""
    if len(block) < _SB.size:
        return None
    fields = _SB.unpack_from(block, 0)
    magic, version, _flags, generation, ident, layout, object_bytes, num_slots, data_off, device_bytes, claim, mode, created, crc = fields
    if magic != SB_MAGIC or version != VERSION:
        return None
    if zlib.crc32(bytes(block[:_SB.size - 4])) & 0xFFFFFFFF != crc:
        return None
    try:
        layout_id = layout.rstrip(b'\x00').decode('utf-8')
        mode_text = mode.rstrip(b'\x00').decode('ascii')
    except UnicodeDecodeError:
        return None
    if mode_text not in MODES or object_bytes <= 0 or num_slots < 0 or data_off % ALIGN or object_bytes % SECTOR:
        return None
    return Superblock(generation=generation, uuid=bytes(ident), layout_id=layout_id, object_bytes=object_bytes, num_slots=num_slots,
                      data_off=data_off, device_bytes=device_bytes, claim=bytes(claim), mode=mode_text, created=created)


# --------------------------------------------------------------------------------------------------------------------
# Geometry
# --------------------------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Geometry:
    device_bytes: int
    object_bytes: int
    data_off: int
    num_slots: int
    canary_off: int = CANARY_OFF
    canary_bytes: int = CANARY_BYTES


def plan_geometry(device_bytes: int, object_bytes: int, *, object_align: int = ALIGN) -> Geometry:
    """Where the slots go on a device of ``device_bytes``: the data region starts on the next MiB after the header (4 KiB on a small
    file), every slot is ``object_bytes`` (a multiple of 4 KiB on a real drive, so each is O_DIRECT aligned; ``object_align`` is lowered
    to 512 only for the CPU tests' tiny layout on a regular file), and a partial slot at the end is unused."""
    device_bytes, object_bytes = int(device_bytes), int(object_bytes)
    if object_bytes <= 0 or object_bytes % object_align:
        raise ValueError('object_bytes %d is not a positive multiple of %d' % (object_bytes, object_align))
    align = MIN_DATA_ALIGN if device_bytes >= BIG_DEVICE else ALIGN
    data_off = roundup(HEADER_BYTES, align)
    slots = (device_bytes - data_off) // object_bytes
    if slots < 1:
        raise CapacityShort('a device of %d bytes holds no %d-byte object after %d bytes of header' % (device_bytes, object_bytes, data_off))
    return Geometry(device_bytes=device_bytes, object_bytes=object_bytes, data_off=data_off, num_slots=slots)


# --------------------------------------------------------------------------------------------------------------------
# The drive file
# --------------------------------------------------------------------------------------------------------------------

_LIBC = None


def _fallocate_punch(fd: int, offset: int, length: int) -> bool:
    """Deallocate a range of a regular file (the CPU tests' stand-in for TRIM). False when the filesystem cannot."""
    global _LIBC
    try:
        if _LIBC is None:
            _LIBC = ctypes.CDLL(None, use_errno=True)
        FALLOC_FL_KEEP_SIZE, FALLOC_FL_PUNCH_HOLE = 1, 2
        done = _LIBC.fallocate(ctypes.c_int(fd), ctypes.c_int(FALLOC_FL_PUNCH_HOLE | FALLOC_FL_KEEP_SIZE),
                               ctypes.c_longlong(offset), ctypes.c_longlong(length))
        return done == 0
    except Exception:
        return False


class DriveFile:
    """A raw block device (O_DIRECT) or, for the CPU tests only, a regular file. Reads and writes are aligned.

    The open takes the kernel's exclusive claim (``O_EXCL`` on a block device fails with EBUSY while anything else holds or mounts it)
    and an advisory ``flock``, so a second engine in another container cannot use the same drive.
    """

    def __init__(self, path: str, *, allow_regular: bool = False, exclusive: bool = True, readonly: bool = False):
        self.path = path
        try:
            info = os.stat(path)
        except FileNotFoundError:
            raise DeviceAbsent('%s does not exist' % path)
        except OSError as error:
            raise DeviceAbsent('%s: %s' % (path, error.strerror or error))
        self.is_block = stat.S_ISBLK(info.st_mode)
        if not self.is_block and not (allow_regular and stat.S_ISREG(info.st_mode)):
            raise DeviceAbsent('%s is not a block device' % path)
        flags = os.O_RDONLY if readonly else os.O_RDWR
        flags |= getattr(os, 'O_CLOEXEC', 0)
        if exclusive and self.is_block:
            flags |= os.O_EXCL
        self.direct = True
        try:
            self.fd = os.open(path, flags | getattr(os, 'O_DIRECT', 0))
        except OSError as error:
            if error.errno == errno.EINVAL and not self.is_block:     # a filesystem without O_DIRECT (a test on tmpfs)
                self.direct = False
                self.fd = os.open(path, flags)
            elif error.errno in (errno.EBUSY, errno.EACCES, errno.EPERM) and exclusive and self.is_block:
                raise DeviceBusy('%s is in use or not ours to open exclusively (%s)' % (path, error.strerror))
            elif error.errno in (errno.ENODEV, errno.ENXIO, errno.ENOENT):
                raise DeviceGone('%s: %s' % (path, error.strerror))
            else:
                raise StoreError('cannot open %s: %s' % (path, error.strerror or error))
        self.readonly = readonly
        self._lock = threading.Lock()
        self._local = threading.local()
        if exclusive and not readonly:
            try:
                fcntl.flock(self.fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as error:
                self.close()
                raise DeviceBusy('%s is locked by another process (%s)' % (path, error.strerror or error))
        self.size = self._device_size()
        self.fault: Optional[Callable[[str, int, int], None]] = None      # a test hook, called as fault(op, offset, length); it may raise

    def _device_size(self) -> int:
        if self.is_block:
            return os.lseek(self.fd, 0, os.SEEK_END)
        return os.fstat(self.fd).st_size

    def refresh_size(self) -> int:
        self.size = self._device_size()
        return self.size

    def _scratch(self, size: int) -> AlignedBuffer:
        buffer = getattr(self._local, 'buffer', None)
        if buffer is None or buffer.size < size:
            if buffer is not None:
                buffer.close()
            buffer = self._local.buffer = AlignedBuffer(size)
        return buffer

    def _check(self, offset: int, length: int) -> None:
        if offset < 0 or offset % ALIGN or length % ALIGN or length <= 0:
            raise ValueError('unaligned access: offset %d length %d' % (offset, length))
        if offset + length > self.size:
            raise ValueError('access past the end: %d + %d > %d' % (offset, length, self.size))

    def read(self, offset: int, length: int) -> bytes:
        self._check(offset, length)
        if self.fault is not None:
            self.fault('read', offset, length)
        buffer = self._scratch(length)
        view = buffer.view[:length]
        got = os.preadv(self.fd, [view], offset)
        if got != length:
            raise StoreError('short read at %d: %d of %d bytes' % (offset, got, length))
        return bytes(view)

    def write(self, offset: int, data: bytes) -> None:
        if self.readonly:
            raise StoreError('%s is open read-only' % self.path)
        length = len(data)
        self._check(offset, length)
        if self.fault is not None:
            self.fault('write', offset, length)
        buffer = self._scratch(length)
        buffer.view[:length] = data
        done = os.pwritev(self.fd, [buffer.view[:length]], offset)
        if done != length:
            raise StoreError('short write at %d: %d of %d bytes' % (offset, done, length))

    def sync(self) -> None:
        os.fdatasync(self.fd)

    def discard(self, offset: int, length: int) -> bool:
        """TRIM a range (BLKDISCARD on a block device, a punched hole on a test file). Raises OSError when the call fails;
        returns False only when nothing could be done on a regular file that cannot punch holes (it then zeroes the range)."""
        if self.readonly:
            raise StoreError('%s is open read-only' % self.path)
        if offset < 0 or length <= 0 or offset + length > self.size:
            raise ValueError('discard outside the device: %d + %d > %d' % (offset, length, self.size))
        if self.fault is not None:
            self.fault('discard', offset, length)
        if self.is_block:
            fcntl.ioctl(self.fd, BLKDISCARD, struct.pack('QQ', offset, length))
            return True
        if _fallocate_punch(self.fd, offset, length):
            return True
        zero = bytes(min(length, 1 << 20))
        position, left = offset, length
        while left:
            take = min(left, len(zero))
            os.pwrite(self.fd, zero[:take], position)
            position += take
            left -= take
        return False

    def close(self) -> None:
        fd, self.fd = getattr(self, 'fd', None), None
        if fd is not None:
            try:
                os.close(fd)
            except OSError:
                pass


# --------------------------------------------------------------------------------------------------------------------
# Reading and verifying
# --------------------------------------------------------------------------------------------------------------------

def read_superblocks(drive: DriveFile) -> List[Superblock]:
    """The valid superblocks of the two slots, highest generation first."""
    found: List[Superblock] = []
    for offset in (SB_SLOT_A, SB_SLOT_B):
        if offset + ALIGN > drive.size:
            continue
        try:
            block = drive.read(offset, ALIGN)
        except (OSError, StoreError, ValueError):
            continue
        decoded = decode_superblock(block)
        if decoded is not None:
            found.append(decoded)
    found.sort(key=lambda sb: sb.generation, reverse=True)
    return found


@dataclass
class Store:
    """An opened, verified drive: the file, the superblock it carries, and where each slot lives."""

    drive: DriveFile
    superblock: Superblock

    @property
    def num_slots(self) -> int:
        return self.superblock.num_slots

    @property
    def object_bytes(self) -> int:
        return self.superblock.object_bytes

    @property
    def data_off(self) -> int:
        return self.superblock.data_off

    @property
    def canary_off(self) -> int:
        return CANARY_OFF

    def slot_offset(self, slot: int) -> int:
        if not 0 <= slot < self.num_slots:
            raise IndexError('slot %d is outside 0..%d' % (slot, self.num_slots - 1))
        return self.superblock.data_off + slot * self.superblock.object_bytes

    def close(self) -> None:
        self.drive.close()


def verify_superblock(drive: DriveFile, *, layout_id: str, object_bytes: int, persistence: str,
                      expect_claim: Optional[bytes] = None) -> Superblock:
    """The superblock this drive carries, after every check of design 2.9 step 1 that needs only the drive. Raises a StoreError
    whose ``code`` is the closed UNAVAILABLE reason."""
    if drive.size <= 0:
        raise DeviceGone('%s reads 0 bytes: the namespace is gone or its controller is not live' % drive.path)
    found = read_superblocks(drive)
    if not found:
        raise UnknownSuperblock('%s carries no gemma-kv v%d superblock: the tier never formats; run the claim tool' % (drive.path, VERSION))
    sb = found[0]
    if sb.layout_id != layout_id:
        raise LayoutMismatch('the drive was claimed for layout %r, this engine runs %r' % (sb.layout_id, layout_id))
    if sb.object_bytes != object_bytes:
        raise LayoutMismatch('the drive\'s objects are %d bytes, this engine\'s are %d' % (sb.object_bytes, object_bytes))
    if sb.mode != persistence:
        raise LayoutMismatch('the drive was claimed in mode %r, this engine runs %r' % (sb.mode, persistence))
    if drive.size < sb.end:
        raise CapacityShort('the drive was claimed for %d bytes and now reads %d' % (sb.end, drive.size))
    if sb.device_bytes and sb.device_bytes != drive.size:
        # the namespace was resized or this is another device: the claim covers the size
        raise UnknownSuperblock('the drive was claimed at %d bytes and now reads %d' % (sb.device_bytes, drive.size))
    if sb.data_off % ALIGN or sb.data_off < HEADER_BYTES or sb.object_bytes % (ALIGN if drive.is_block else SECTOR):
        raise UnknownSuperblock('the superblock\'s geometry is not aligned')
    if expect_claim is not None and sb.claim != expect_claim:
        raise UnknownSuperblock('the superblock\'s claim does not match this drive (another drive, or the status file names another)')
    return sb


def open_store(path: str, *, layout_id: str, object_bytes: int, persistence: str = 'process', allow_regular: bool = False,
               expect_claim: Optional[bytes] = None, expect_claim_for: Optional[Callable[[int], Optional[bytes]]] = None) -> Store:
    """Open the drive exclusively and verify its superblock. ``expect_claim_for(size)`` computes the claim to expect from the device
    size (the status file supplies the by-id name); ``expect_claim`` is the same when it is already known."""
    drive = DriveFile(path, allow_regular=allow_regular, exclusive=True)
    try:
        if expect_claim is None and expect_claim_for is not None:
            expect_claim = expect_claim_for(drive.size)
        sb = verify_superblock(drive, layout_id=layout_id, object_bytes=object_bytes, persistence=persistence, expect_claim=expect_claim)
        return Store(drive, sb)
    except BaseException:
        drive.close()
        raise


# --------------------------------------------------------------------------------------------------------------------
# Writing (the claim tool only)
# --------------------------------------------------------------------------------------------------------------------

def new_superblock(*, geometry: Geometry, layout_id: str, claim: bytes, mode: str = 'process', created: int = 0,
                   generation: int = 1, ident: Optional[bytes] = None) -> Superblock:
    return Superblock(generation=generation, uuid=ident or uuid_module.uuid4().bytes, layout_id=layout_id,
                      object_bytes=geometry.object_bytes, num_slots=geometry.num_slots, data_off=geometry.data_off,
                      device_bytes=geometry.device_bytes, claim=claim, mode=mode, created=int(created))


def write_superblocks(drive: DriveFile, sb: Superblock) -> None:
    """Both slots, the older generation first (a crash between the two writes leaves one valid superblock): generation g in slot A
    and g + 1 in slot B. Used by the claim tool, never by the tier."""
    first = encode_superblock(sb)
    second = encode_superblock(replace(sb, generation=sb.generation + 1))
    drive.write(SB_SLOT_A, first)
    drive.sync()
    drive.write(SB_SLOT_B, second)
    drive.sync()


def region_runs(first_slot: int, count: int, slots_per_run: int) -> List[Tuple[int, int]]:
    """Split ``count`` consecutive slots from ``first_slot`` into (first, count) runs of at most ``slots_per_run``."""
    runs = []
    done = 0
    while done < count:
        take = min(slots_per_run, count - done)
        runs.append((first_slot + done, take))
        done += take
    return runs


def discard_slots(store: Store, first_slot: int, count: int) -> None:
    store.drive.discard(store.slot_offset(first_slot), count * store.object_bytes)


# the number of objects one maximum-context session stores, per layout (design 1.4: 2,048 full + 5 x 8 sliding at 262,144 tokens); a
# layout that is not listed needs room for one object. Start-up refuses a drive with less room than a session (capacity_short).
MIN_SESSION_OBJECTS = {'gemma-chipmajor-v1': 2088}


def min_session_objects(layout_id: str) -> int:
    return MIN_SESSION_OBJECTS.get(layout_id, 1)


# The object size of each layout the claim tool knows (chips x tensors x bytes of one block row of one tensor on one chip). The tier takes
# the same number from ``spec.LAYOUTS``; a test keeps the two tables equal. Kept here so the claim tool needs nothing but the standard library.
KNOWN_LAYOUTS = {'gemma-chipmajor-v1': 4 * 10 * 69632}
