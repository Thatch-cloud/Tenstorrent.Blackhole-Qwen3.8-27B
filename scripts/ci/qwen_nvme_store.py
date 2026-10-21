"""A crash-safe object store on a raw block device: the NVMe cold tier's storage (docs/tp4-kv-nvme.md).

WHAT. A flat key -> bytes store on a RAW block device or a regular file standing in for one (the CPU tests): no filesystem, no md, no LVM, no dm. The data
region holds each object at a 4 KiB-aligned LBA range, so a later peer-to-peer reader (NVMe -> card window, no host copy) can name an object by its namespace
LBAs. Objects are immutable and content-keyed (the caller's keys are hashes of what the bytes stand for), of two kinds (KV = a tier block, CKPT = a GDN
checkpoint), each with a tenant tag, a chain id (the first block hash of its prompt), an index, a layout fingerprint and a sha256 of its bytes.

LAYOUT (all offsets multiples of ALIGN = 4096; the geometry is written into the superblocks at format time):
    [0, 4 KiB)          superblock slot 0           magic, version, generation, geometry, the active snapshot, the journal epoch; crc32
    [4 KiB, 8 KiB)      superblock slot 1           the older generation survives a torn write of the newer
    snapshot slot 0/1   the whole index, written to the slot that is NOT active, then the superblock flips to it
    journal             records appended since the snapshot: PUT (batches of entries), DEL, DEL_TENANT, WIPE, TOUCH; each carries (epoch, seq, crc32)
    data region         the objects, allocated by an extent allocator (first fit, coalescing free list, rebuilt from the index at open)

CRASH SAFETY. A put is: (1) write the data, (2) fdatasync, (3) append the journal record that names it, (4) fdatasync. Until (4) the object is not in any
state a reader of the device can reach: a crash before it leaves the old state (the extent is free again at the next open, the index never named it); a crash
after it leaves the new state. A journal group torn mid-write is cut at the first record whose crc, epoch or sequence number does not check, so every record
that survives is whole, and the data every surviving PUT names was synced before the record was written. A delete is the other way round: the DEL record is
durable (fdatasync) BEFORE the extent is allocatable again, so an object is never listed whose bytes have been overwritten; an extent freed while a reader holds
a pin on it waits for the unpin. A snapshot writes the inactive slot, syncs, then writes the superblock (generation + 1, into the slot the older generation does
not occupy) and syncs: a crash before the superblock leaves the old snapshot and the old epoch's journal; after it, the new pair. The journal is reset by the
epoch (a record of another epoch ends a replay), never by erasing it. A multi-record batch can be cut after any record: what survives is a prefix of the batch
(the blocks of a session in index order, so a prefix is a shorter restorable session), never a mixture.

FORMAT SAFETY. open() writes to a device only if it already carries this store's superblock, or its first MiB is all zero (a blank namespace), or format='force'
says to destroy whatever is there. A device with a store's magic but no valid superblock is CORRUPT, not blank, and also needs 'force'.

THREADS. One writer at a time (_wlock) for everything durable: put_batch, delete*, wipe, sweep, snapshot. Lookups, pins, touches and the allocator take the short
in-memory lock (_mutex) only, and never wait for device IO: the scheduler thread of the serving engine calls them. Reads of object bytes (read_many) take no lock
at all: they work on pinned entries. Pure stdlib; Python 3.7 syntax.
"""

import bisect
import collections
import ctypes
import hashlib
import mmap
import os
import stat
import struct
import threading
import time
import uuid
import zlib

ALIGN = 4096
KV, CKPT = 1, 2
KINDS = (KV, CKPT)
MAX_KEY = 255
HEADER_BYTES = 2 * ALIGN
MIN_DATA_ALIGN = 1 << 20                  # the data region starts on a MiB
SB_MAGIC = b'QNVMSB01'
SNAP_MAGIC = b'QNVMSN01'
JREC_MAGIC = 0x4A524543
VERSION = 1
NO_SNAPSHOT = 0xFFFFFFFF
MAX_RUN_BYTES = 32 << 20                  # one read call; larger runs are split
SCRATCH_BYTES = 4 << 20
IOV_CHUNK = 256
PUT_RECORD_ENTRIES = 96                   # entries per journal PUT record
DEFAULT_ENTRY_BUDGET = 200                # bytes of snapshot space per entry the capacity check assumes
TOUCH_JOURNAL_S = 60.0
TOUCH_RECORD_KEYS = 2048
PIN_SCAN_LIMIT = 4096
SWEEP_LIMIT = 20000
BLKDISCARD = 0x1277

# journal record kinds
R_PUT, R_DEL, R_DEL_TENANT, R_WIPE, R_TOUCH = 1, 2, 3, 4, 5


class StoreError(Exception):
    """Base of the store's errors."""


class NotOurs(StoreError):
    """The device holds something that is not this store and the format policy does not allow overwriting it."""


class Corrupt(StoreError):
    """The device is this store's but its metadata cannot be read (both superblocks or the snapshot failed their checks)."""


class StoreFull(StoreError):
    """Nothing could be evicted to make room."""


def roundup(count, align=ALIGN):
    return (int(count) + align - 1) // align * align


def digest_of(buffers):
    """sha256 of the concatenation of buffers (bytes-like objects)."""
    sha = hashlib.sha256()
    for buffer in buffers:
        sha.update(buffer)
    return sha.digest()


def tenant_hash(tag):
    """The 8 bytes a tenant tag is stored as (the tag itself never reaches the device)."""
    if isinstance(tag, str):
        tag = tag.encode('utf-8', 'surrogatepass')
    return hashlib.blake2b(bytes(tag or b''), digest_size=8, person=b'qnvm-tenant').digest()


# ------------------------------------------------------------------------------------------------
# Aligned memory and the device
# ------------------------------------------------------------------------------------------------
def buffer_address(view):
    """The address of a writable buffer, or None when it cannot be read from the object (read-only buffers)."""
    try:
        return ctypes.addressof(ctypes.c_char.from_buffer(view))
    except (TypeError, ValueError, AttributeError):
        return None


def is_aligned(view, align=ALIGN):
    """Whether a buffer starts on an `align` boundary and is a whole number of them long (so O_DIRECT may use it as it is)."""
    if len(view) % align:
        return False
    address = buffer_address(view)
    return address is not None and address % align == 0


def _flat(buffer):
    """A byte-format memoryview of any bytes-like object."""
    view = memoryview(buffer)
    return view if view.format == 'B' else view.cast('B')


class Scratch(object):
    """A page-aligned anonymous buffer for O_DIRECT bounces; grows on demand."""

    def __init__(self, size=SCRATCH_BYTES):
        self.size = 0
        self._map = None
        self.view = None
        self.ensure(size)

    def ensure(self, size):
        size = roundup(max(int(size), ALIGN))
        if size > self.size:
            self.view = None
            if self._map is not None:
                try:
                    self._map.close()
                except BufferError:
                    pass                    # a slice of the old map is still alive somewhere: the garbage collector will unmap it
            self._map = mmap.mmap(-1, size)
            self.view = memoryview(self._map)
            self.size = size
        return self.view

    def close(self):
        self.view = None
        if self._map is not None:
            try:
                self._map.close()
            except BufferError:
                pass
            self._map = None
        self.size = 0


class BlockFile(object):
    """A raw block device (O_DIRECT) or a regular file (buffered, for the CPU tests), read and written at aligned offsets in aligned lengths. The alignment is
    enforced for both, so a test on a file finds the bug a device would."""

    def __init__(self, path, direct=None, size=None, create=False, readonly=False, align=ALIGN):
        self.path = path
        self.align = int(align)
        flags = os.O_RDONLY if readonly else os.O_RDWR
        if create:
            flags |= os.O_CREAT
        try:
            info = os.stat(path)
            self.is_block = stat.S_ISBLK(info.st_mode)
        except FileNotFoundError:
            if not create:
                raise
            self.is_block = False
        if direct is None:
            direct = self.is_block
        self.direct = bool(direct)
        if self.direct:
            flags |= getattr(os, 'O_DIRECT', 0)
        flags |= getattr(os, 'O_CLOEXEC', 0)
        self.fd = os.open(path, flags, stat.S_IRUSR | stat.S_IWUSR)
        self.readonly = readonly
        self.local = threading.local()
        if not self.is_block and size is not None and os.fstat(self.fd).st_size < size:
            os.ftruncate(self.fd, int(size))
        self.size = self._device_size()
        self.fail = None             # a test hook: called as fail('read'|'write'|'sync', offset, length) before the call; it may raise (a crash at that call)

    def _device_size(self):
        if self.is_block:
            return os.lseek(self.fd, 0, os.SEEK_END)
        return os.fstat(self.fd).st_size

    def scratch(self, size=SCRATCH_BYTES):
        scratch = getattr(self.local, 'scratch', None)
        if scratch is None:
            scratch = self.local.scratch = Scratch(size)
        scratch.ensure(size)
        return scratch

    def _check(self, offset, length):
        if offset % self.align or length % self.align or offset < 0:
            raise ValueError('unaligned device access: offset %d length %d (align %d)' % (offset, length, self.align))
        if offset + length > self.size:
            raise ValueError('device access past the end: %d + %d > %d' % (offset, length, self.size))

    # -- aligned primitives (every view a whole number of aligned bytes, at an aligned address when direct) -----------------
    def pread_into(self, offset, views):
        """Fill the views, back to back, from `offset`. A short read (end of a file) is an error."""
        views = [_flat(view) for view in views]
        total = sum(len(view) for view in views)
        self._check(offset, total)
        if self.fail is not None:
            self.fail('read', offset, total)
        position = offset
        pending = list(views)
        while pending:
            batch = pending[:IOV_CHUNK]
            wanted = sum(len(view) for view in batch)
            got = os.preadv(self.fd, batch, position)
            if got <= 0:
                raise IOError('short read at %d: wanted %d' % (position, wanted))
            position += got
            # drop what was filled, keep the rest of a partly filled view
            remaining = got
            while remaining and pending:
                head = pending[0]
                if len(head) <= remaining:
                    remaining -= len(head)
                    pending.pop(0)
                else:
                    pending[0] = head[remaining:]
                    remaining = 0

    def pwrite_from(self, offset, views):
        """Write the views, back to back, at `offset`."""
        views = [_flat(view) for view in views]
        total = sum(len(view) for view in views)
        self._check(offset, total)
        if self.fail is not None:
            self.fail('write', offset, total)
        position = offset
        pending = list(views)
        while pending:
            got = os.pwritev(self.fd, pending[:IOV_CHUNK], position)
            if got <= 0:
                raise IOError('short write at %d' % position)
            position += got
            remaining = got
            while remaining and pending:
                head = pending[0]
                if len(head) <= remaining:
                    remaining -= len(head)
                    pending.pop(0)
                else:
                    pending[0] = head[remaining:]
                    remaining = 0

    def sync(self):
        if self.fail is not None:
            self.fail('sync', 0, 0)
        os.fdatasync(self.fd)

    def lock(self):
        """Take the device for this process (an advisory exclusive lock, held while the descriptor is open): two writers on one store would corrupt it."""
        try:
            import fcntl

            fcntl.flock(self.fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except ImportError:
            return
        except OSError as error:
            raise StoreError('%s is in use by another process (%s)' % (self.path, error))

    def discard(self, offset, length):
        """Tell the drive these bytes are free (TRIM). Best effort: True when the call was made."""
        if not self.is_block or self.readonly:
            return False
        try:
            import fcntl

            fcntl.ioctl(self.fd, BLKDISCARD, struct.pack('QQ', offset, length))
            return True
        except Exception:
            return False

    def close(self):
        fd, self.fd = self.fd, None
        if fd is not None:
            os.close(fd)

    # -- logical helpers: any length, bounced through aligned scratch when the caller's memory is not usable as it is ---------------------
    def read_bytes(self, offset, count):
        """`count` bytes at an aligned `offset` as a bytes object."""
        out = bytearray(count)
        self.read_run(offset, [count], [memoryview(out)])
        return bytes(out)

    def read_run(self, offset, lengths, views):
        """Entries laid back to back at `offset`, entry i occupying roundup(lengths[i]) bytes on the device; fill views[i] (len == lengths[i]) from each. The fast path
        (every view aligned in address and a whole number of blocks) reads straight into the views; otherwise each entry goes through the aligned scratch."""
        views = [_flat(view) for view in views]
        if all(is_aligned(view, self.align) and len(view) == length for view, length in zip(views, lengths)):
            self.pread_into(offset, views)
            return
        position = offset
        for length, view in zip(lengths, views):
            span = roundup(length, self.align)
            scratch = self.scratch()
            done = 0
            while done < span:
                piece = min(span - done, scratch.size)
                self.pread_into(position + done, [scratch.view[:piece]])
                take = max(0, min(piece, length - done))
                if take:
                    view[done:done + take] = scratch.view[:take]
                done += piece
            position += span

    def write_buffers(self, offset, buffers, nbytes=None):
        """Write the concatenation of `buffers` at an aligned `offset`, zero-padded to a whole block. Straight from the caller's memory when it is one aligned buffer."""
        buffers = [_flat(buffer) for buffer in buffers]
        total = sum(len(buffer) for buffer in buffers) if nbytes is None else nbytes
        if len(buffers) == 1 and is_aligned(buffers[0], self.align) and len(buffers[0]) == total:
            self.pwrite_from(offset, [buffers[0]])
            return roundup(total, self.align)
        position = offset
        scratch = self.scratch()
        fill = 0
        view = scratch.view
        for buffer in buffers:
            index = 0
            while index < len(buffer):
                take = min(len(buffer) - index, scratch.size - fill)
                view[fill:fill + take] = buffer[index:index + take]
                fill += take
                index += take
                if fill == scratch.size:
                    self.pwrite_from(position, [view[:fill]])
                    position += fill
                    fill = 0
        if fill:
            tail = roundup(fill, self.align)
            view[fill:tail] = bytes(tail - fill)
            self.pwrite_from(position, [view[:tail]])
            position += tail
        return position - offset


# ------------------------------------------------------------------------------------------------
# Entries and their encodings
# ------------------------------------------------------------------------------------------------
class Entry(object):
    """One object: where it is (offset, alloc bytes reserved, nbytes of data) and what it is."""

    __slots__ = ('kind', 'key', 'offset', 'nbytes', 'alloc', 'tenant', 'chain', 'index', 'extra', 'created', 'last_used', 'digest', 'fp', 'pins', 'dead', 'durable')

    def __init__(self, kind, key, offset, nbytes, tenant, chain, index, extra, created, last_used, digest, fp):
        self.kind = kind
        self.key = key
        self.offset = offset
        self.nbytes = nbytes
        self.alloc = roundup(nbytes)
        self.tenant = tenant
        self.chain = chain
        self.index = index
        self.extra = extra
        self.created = created
        self.last_used = last_used
        self.digest = digest
        self.fp = fp
        self.pins = 0
        self.dead = False
        self.durable = False           # (dead only) the removal is in the journal: the extent may be reused as soon as no reader pins it


_ENTRY = struct.Struct('<BBBBIQQQdd')
_SB = struct.Struct('<8sIIQ16sIQQQQQQQQIQIQQI')
_SNAP_HEAD = struct.Struct('<8sQQQQ')
_JHEAD = struct.Struct('<IIQQHHI')
_TOUCH = struct.Struct('<d')


def pack_entry(entry):
    return b''.join((_ENTRY.pack(entry.kind, len(entry.key), len(entry.chain), 0, entry.index & 0xFFFFFFFF, entry.offset, entry.nbytes, entry.extra,
                                 entry.created, entry.last_used), entry.key, entry.chain, entry.tenant, entry.digest, entry.fp))


def unpack_entry(buffer, position):
    kind, klen, clen, _flags, index, offset, nbytes, extra, created, last_used = _ENTRY.unpack_from(buffer, position)
    position += _ENTRY.size
    key = bytes(buffer[position:position + klen])
    position += klen
    chain = bytes(buffer[position:position + clen])
    position += clen
    tenant = bytes(buffer[position:position + 8])
    position += 8
    digest = bytes(buffer[position:position + 32])
    position += 32
    fp = bytes(buffer[position:position + 8])
    position += 8
    if len(fp) != 8 or len(digest) != 32 or len(key) != klen:
        raise ValueError('truncated entry')
    return Entry(kind, key, offset, nbytes, tenant, chain, index, extra, created, last_used, digest, fp), position


class Geometry(object):
    """Where everything is on the device."""

    __slots__ = ('align', 'device_bytes', 'snap_off', 'snap_bytes', 'journal_off', 'journal_bytes', 'data_off', 'data_bytes')

    def describe(self):
        return dict(device_bytes=self.device_bytes, snap_bytes=self.snap_bytes, journal_bytes=self.journal_bytes, data_off=self.data_off, data_bytes=self.data_bytes)


def plan_geometry(device_bytes, snap_bytes=None, journal_bytes=None, align=ALIGN):
    """The layout of a device of device_bytes. Defaults: each snapshot slot 1/2048 of the device (a 1 TB device: about 488 MB, room for 2.4 million entries at
    200 bytes), the journal 1/4096, both within sane bounds; the data region from the next MiB."""
    device_bytes = int(device_bytes)
    floor = max(16 * align, min(1 << 20, device_bytes // 32))        # (a MiB for any real device; a small one keeps most of its space for data)
    if snap_bytes is None:
        snap_bytes = min(max(device_bytes // 2048, floor), 1 << 30)
    if journal_bytes is None:
        journal_bytes = min(max(device_bytes // 4096, floor), 512 << 20)
    snap_bytes = roundup(snap_bytes, align)
    journal_bytes = roundup(journal_bytes, align)
    geometry = Geometry()
    geometry.align = align
    geometry.device_bytes = device_bytes
    geometry.snap_bytes = snap_bytes
    geometry.snap_off = (HEADER_BYTES, HEADER_BYTES + snap_bytes)
    geometry.journal_off = HEADER_BYTES + 2 * snap_bytes
    geometry.journal_bytes = journal_bytes
    align_data = MIN_DATA_ALIGN if device_bytes >= (64 << 20) else align
    geometry.data_off = roundup(geometry.journal_off + journal_bytes, align_data)
    geometry.data_bytes = (device_bytes - geometry.data_off) // align * align
    if geometry.data_bytes < 16 * align:
        raise ValueError('a device of %d bytes leaves no room for data after %d bytes of metadata' % (device_bytes, geometry.data_off))
    return geometry


# ------------------------------------------------------------------------------------------------
# The extent allocator
# ------------------------------------------------------------------------------------------------
class Allocator(object):
    """Free extents of the data region as two sorted parallel lists (offset, length); first fit, coalescing on free. The caller holds the lock."""

    def __init__(self, start, length):
        self.starts = [start] if length else []
        self.lengths = [length] if length else []
        self.free = length
        self.total = length

    @classmethod
    def from_used(cls, start, length, used):
        """The allocator for a region with the (offset, length) extents of `used` taken (sorted by offset, non-overlapping)."""
        allocator = cls(start, 0)
        allocator.total = length
        cursor = start
        end = start + length
        for offset, size in used:
            if offset > cursor:
                allocator.starts.append(cursor)
                allocator.lengths.append(offset - cursor)
            cursor = offset + size
        if cursor < end:
            allocator.starts.append(cursor)
            allocator.lengths.append(end - cursor)
        allocator.free = sum(allocator.lengths)
        return allocator

    def alloc(self, size):
        """The offset of `size` contiguous free bytes (first fit), or None."""
        for index, length in enumerate(self.lengths):
            if length >= size:
                offset = self.starts[index]
                if length == size:
                    del self.starts[index]
                    del self.lengths[index]
                else:
                    self.starts[index] = offset + size
                    self.lengths[index] = length - size
                self.free -= size
                return offset
        return None

    def release(self, offset, size):
        position = bisect.bisect_left(self.starts, offset)
        merge_prev = position > 0 and self.starts[position - 1] + self.lengths[position - 1] == offset
        merge_next = position < len(self.starts) and offset + size == self.starts[position]
        if merge_prev and merge_next:
            self.lengths[position - 1] += size + self.lengths[position]
            del self.starts[position]
            del self.lengths[position]
        elif merge_prev:
            self.lengths[position - 1] += size
        elif merge_next:
            self.starts[position] = offset
            self.lengths[position] += size
        else:
            self.starts.insert(position, offset)
            self.lengths.insert(position, size)
        self.free += size

    def largest(self):
        return max(self.lengths) if self.lengths else 0

    def fragments(self):
        return len(self.starts)


class PutItem(object):
    """One object to store: buffers (bytes-like, concatenated = the object), its identity and, optionally, its digest (computed here when absent)."""

    __slots__ = ('kind', 'key', 'buffers', 'tenant', 'chain', 'index', 'extra', 'fp', 'digest', 'nbytes')

    def __init__(self, kind, key, buffers, tenant, chain, index, fp, extra=0, digest=None):
        self.kind = kind
        self.key = bytes(key)
        self.buffers = list(buffers)
        self.nbytes = sum(len(buffer) for buffer in self.buffers)
        self.tenant = tenant if isinstance(tenant, bytes) and len(tenant) == 8 else tenant_hash(tenant)      # (a tag is a str; 8 bytes are already a hash)
        self.chain = bytes(chain or b'')
        self.index = int(index)
        self.fp = fp
        self.extra = int(extra)
        self.digest = digest


STAT_NAMES = ('puts', 'put_bytes', 'put_duplicates', 'put_refused', 'put_evictions', 'gets', 'get_hits', 'get_misses', 'reads', 'read_bytes', 'read_errors',
              'digest_failures', 'evicted_lru', 'expired_ttl', 'expired_age', 'deleted_keys', 'deleted_tenant', 'wipes', 'journal_records', 'journal_syncs',
              'snapshots', 'discarded_bytes', 'fp_drops', 'recovered_entries', 'recovered_records', 'recover_dropped', 'touch_records')


class NvmeStore(object):
    """See the module docstring. Construct with open()."""

    def __init__(self, bf, geometry, superblock, clock=time.time, ttl_s=0, max_age_s=0, max_bytes=0, free_floor_bytes=0, discard=False, threads=4):
        self.bf = bf
        self.geometry = geometry
        self.clock = clock
        self.ttl_s = float(ttl_s or 0)
        self.max_age_s = float(max_age_s or 0)
        self.max_bytes = int(max_bytes or 0)
        self.free_floor_bytes = int(free_floor_bytes or 0)
        self.discard_on_free = bool(discard)
        self.threads = max(1, int(threads))
        self._mutex = threading.RLock()
        self._wlock = threading.RLock()
        self.entries = collections.OrderedDict()           # (kind, key) -> Entry, oldest use first
        self.by_tenant = {}                                # tenant hash -> set of (kind, key)
        self.chains = {}                                   # chain -> {pos: key} of its CKPT entries
        self.interned = {}
        self.logical_bytes = 0                             # the allocs of live entries
        self.kind_counts = {KV: 0, CKPT: 0}
        self._dead = []                                    # entries taken out of the index in memory whose DEL record is not journaled yet (mark_bad)
        self.allocator = Allocator(geometry.data_off, geometry.data_bytes)
        self.created_order = collections.deque()           # (created, kind, key, serial)
        self.serial = 0
        self.stats = dict.fromkeys(STAT_NAMES, 0)
        self.generation = superblock['generation']
        self.uuid = superblock['uuid']
        self.epoch = superblock['epoch']
        self.seq = superblock['seq']
        self.active_snap = superblock['active_snap']
        self.jpos = 0
        self.touched = {}
        self.touch_journaled = clock()
        self._pool = None
        self._closed = False
        self.sweep_cursor_time = 0.0

    # -- opening -----------------------------------------------------------------------------------------------
    @classmethod
    def open(cls, path, format='auto', file_factory=BlockFile, direct=None, size=None, snap_bytes=None, journal_bytes=None, clock=time.time, **options):
        """Open (and, per the format policy, initialize) the device at `path`. format: 'auto' = initialize a blank device and refuse anything else that is not this
        store; 'force' = initialize whatever is there; 'never' = refuse a device that is not this store's. -> the store, recovered. `size` creates a regular file."""
        create = size is not None
        bf = file_factory(path, direct=direct, size=size, create=create)
        try:
            bf.lock()
            return cls._open_on(bf, format, snap_bytes, journal_bytes, clock, options)
        except BaseException:
            bf.close()
            raise

    @classmethod
    def _open_on(cls, bf, format, snap_bytes, journal_bytes, clock, options):
        if format not in ('auto', 'force', 'never'):
            raise ValueError('format must be auto, force or never, not %r' % (format,))
        found = cls._read_superblocks(bf)
        superblock = None
        if found:
            superblock = max(found, key=lambda item: item['generation'])
        else:
            probe = bf.read_bytes(0, min(MIN_DATA_ALIGN, bf.size // ALIGN * ALIGN))
            magic_present = SB_MAGIC in (probe[0:8], probe[ALIGN:ALIGN + 8])
            if magic_present and format != 'force':
                raise Corrupt('%s carries this store\'s magic but no valid superblock: format=force to reinitialize' % bf.path)
            blank = not any(probe)
            if not blank and not magic_present and format != 'force':
                raise NotOurs('%s is not blank and not this store\'s: refusing to write to it (format=force to overwrite)' % bf.path)
            if format == 'never' and not magic_present:
                raise NotOurs('%s holds no store and format=never' % bf.path)
            superblock = cls._format(bf, plan_geometry(bf.size, snap_bytes, journal_bytes), clock)
        geometry = Geometry()
        geometry.align = ALIGN
        geometry.device_bytes = superblock['device_bytes']
        geometry.snap_bytes = superblock['snap_bytes']
        geometry.snap_off = (superblock['snap_off0'], superblock['snap_off1'])
        geometry.journal_off = superblock['journal_off']
        geometry.journal_bytes = superblock['journal_bytes']
        geometry.data_off = superblock['data_off']
        geometry.data_bytes = superblock['data_bytes']
        if geometry.device_bytes > bf.size:
            raise Corrupt('the store was formatted for %d bytes, the device is %d' % (geometry.device_bytes, bf.size))
        store = cls(bf, geometry, superblock, clock=clock, **options)
        store._recover(superblock)
        return store

    @staticmethod
    def _decode_superblock(block):
        if len(block) < _SB.size:
            return None
        fields = _SB.unpack_from(block, 0)
        (magic, version, _flags, generation, ident, align, device_bytes, snap_off0, snap_off1, snap_bytes, journal_off, journal_bytes, data_off, data_bytes,
         active_snap, snap_len, snap_crc, epoch, seq, crc) = fields
        if magic != SB_MAGIC or version != VERSION or align != ALIGN:
            return None
        if zlib.crc32(block[:_SB.size - 4]) & 0xFFFFFFFF != crc:
            return None
        return dict(generation=generation, uuid=ident, device_bytes=device_bytes, snap_off0=snap_off0, snap_off1=snap_off1, snap_bytes=snap_bytes,
                    journal_off=journal_off, journal_bytes=journal_bytes, data_off=data_off, data_bytes=data_bytes, active_snap=active_snap,
                    snap_len=snap_len, snap_crc=snap_crc, epoch=epoch, seq=seq)

    @classmethod
    def _read_superblocks(cls, bf):
        found = []
        for slot in (0, 1):
            try:
                block = bf.read_bytes(slot * ALIGN, ALIGN)
            except (IOError, ValueError, OSError):
                continue
            decoded = cls._decode_superblock(block)
            if decoded is not None:
                decoded['slot'] = slot
                found.append(decoded)
        return found

    @staticmethod
    def _encode_superblock(sb):
        body = _SB.pack(SB_MAGIC, VERSION, 0, sb['generation'], sb['uuid'], ALIGN, sb['device_bytes'], sb['snap_off0'], sb['snap_off1'], sb['snap_bytes'],
                        sb['journal_off'], sb['journal_bytes'], sb['data_off'], sb['data_bytes'], sb['active_snap'], sb['snap_len'], sb['snap_crc'],
                        sb['epoch'], sb['seq'], 0)
        crc = zlib.crc32(body[:-4]) & 0xFFFFFFFF
        body = body[:-4] + struct.pack('<I', crc)
        return body + bytes(ALIGN - len(body))

    @classmethod
    def _format(cls, bf, geometry, clock):
        zero = bytes(ALIGN)
        bf.write_buffers(0, [zero])
        bf.write_buffers(ALIGN, [zero])
        bf.sync()
        # a journal block of zeros first: nothing stale can look like a record of this store's first epoch
        bf.write_buffers(geometry.journal_off, [bytes(ALIGN)])
        snapshot = cls._encode_snapshot([], 1, 0)
        bf.write_buffers(geometry.snap_off[0], [snapshot])
        bf.sync()
        superblock = dict(generation=1, uuid=uuid.uuid4().bytes, device_bytes=geometry.device_bytes, snap_off0=geometry.snap_off[0], snap_off1=geometry.snap_off[1],
                          snap_bytes=geometry.snap_bytes, journal_off=geometry.journal_off, journal_bytes=geometry.journal_bytes, data_off=geometry.data_off,
                          data_bytes=geometry.data_bytes, active_snap=0, snap_len=len(snapshot), snap_crc=zlib.crc32(snapshot) & 0xFFFFFFFF, epoch=1, seq=0)
        bf.write_buffers(ALIGN, [cls._encode_superblock(superblock)])      # generation 1 lives in slot 1
        bf.sync()
        return superblock

    # -- snapshots -------------------------------------------------------------------------------------------------
    @staticmethod
    def _encode_snapshot(entries, epoch, seq):
        body = b''.join(pack_entry(entry) for entry in entries)
        head = _SNAP_HEAD.pack(SNAP_MAGIC, epoch, seq, len(entries), len(body))
        return head + body

    def _decode_snapshot(self, data, epoch):
        magic, snap_epoch, seq, count, length = _SNAP_HEAD.unpack_from(data, 0)
        if magic != SNAP_MAGIC or snap_epoch != epoch:
            raise Corrupt('the snapshot header does not match the superblock (epoch %d vs %d)' % (snap_epoch, epoch))
        position = _SNAP_HEAD.size
        if position + length > len(data):
            raise Corrupt('the snapshot is cut short')
        entries = []
        end = position + length
        for _ in range(count):
            entry, position = unpack_entry(data, position)
            entries.append(entry)
        if position != end:
            raise Corrupt('the snapshot\'s entries do not fill its length')
        return entries, seq

    def _recover(self, superblock):
        geometry = self.geometry
        stats = self.stats
        now = self.clock()
        entries = []
        if superblock['active_snap'] != NO_SNAPSHOT:
            slot = superblock['active_snap']
            span = roundup(superblock['snap_len'])
            data = self.bf.read_bytes(geometry.snap_off[slot], span)[:superblock['snap_len']]
            if zlib.crc32(data) & 0xFFFFFFFF != superblock['snap_crc']:
                raise Corrupt('the snapshot of slot %d failed its crc32' % slot)
            entries, snap_seq = self._decode_snapshot(data, superblock['epoch'])
            self.seq = snap_seq
        for entry in entries:
            self._index_add(entry, journal_replay=True)
        records = self._replay_journal()
        stats['recovered_records'] = records
        dropped = self._validate_extents()
        stats['recover_dropped'] = dropped
        self._rebuild_allocator()
        stats['recovered_entries'] = len(self.entries)
        self.touch_journaled = now
        # A journal with anything in it, or an index that lost entries to validation, is compacted into a fresh snapshot (a new epoch): the next boot reads one
        # snapshot, and no record of this epoch's torn tail can ever be confused with a later one.
        if records or dropped:
            with self._wlock:
                self._snapshot_locked()

    def _read_journal_span(self, position, length):
        geometry = self.geometry
        start = position // ALIGN * ALIGN
        end = min(roundup(position + length), geometry.journal_bytes)
        if end <= start:
            return b''
        data = self.bf.read_bytes(geometry.journal_off + start, end - start)
        return data[position - start:position - start + length]

    def _replay_journal(self):
        """Apply the journal records of the current epoch in sequence; stop at the first that does not check. -> records applied."""
        geometry = self.geometry
        position = 0
        expected = self.seq + 1
        applied = 0
        cache = {'base': -1, 'data': b''}

        def span(offset, length):
            if offset + length > geometry.journal_bytes:
                return None
            base = cache['base']
            data = cache['data']
            if base < 0 or offset < base or offset + length > base + len(data):
                base = offset // ALIGN * ALIGN
                window = min(max(length + ALIGN, 1 << 20), geometry.journal_bytes - base)
                data = self.bf.read_bytes(geometry.journal_off + base, roundup(window))
                cache['base'], cache['data'] = base, data
            return data[offset - base:offset - base + length]

        def parse(offset):
            head = span(offset, _JHEAD.size)
            if head is None or len(head) < _JHEAD.size:
                return None
            magic, crc, epoch, seq, kind, _flags, length = _JHEAD.unpack(head)
            if magic != JREC_MAGIC or epoch != self.epoch or seq != expected:
                return None
            payload = span(offset + _JHEAD.size, length)
            if payload is None or len(payload) < length:
                return None
            if zlib.crc32(head[8:] + payload) & 0xFFFFFFFF != crc:
                return None
            return kind, payload

        while True:
            record = parse(position)
            if record is None and position % ALIGN:
                position = roundup(position)
                record = parse(position)
            if record is None:
                break
            kind, payload = record
            try:
                self._apply_record(kind, payload)
            except (ValueError, struct.error):
                break
            position += _JHEAD.size + len(payload)
            expected += 1
            applied += 1
        self.seq = expected - 1
        self.jpos = roundup(position)
        return applied

    def _apply_record(self, kind, payload):
        if kind == R_PUT:
            count = struct.unpack_from('<I', payload, 0)[0]
            position = 4
            items = []
            for _ in range(count):
                entry, position = unpack_entry(payload, position)
                items.append(entry)
            for entry in items:
                self._index_add(entry, journal_replay=True)
        elif kind == R_DEL:
            count = struct.unpack_from('<I', payload, 0)[0]
            position = 4
            for _ in range(count):
                kind_byte, klen = struct.unpack_from('<BB', payload, position)
                position += 2
                key = bytes(payload[position:position + klen])
                position += klen
                self._index_remove((kind_byte, key), free=False)
        elif kind == R_DEL_TENANT:
            tenant = bytes(payload[:8])
            for pair in list(self.by_tenant.get(tenant, ())):
                self._index_remove(pair, free=False)
        elif kind == R_WIPE:
            for pair in list(self.entries):
                self._index_remove(pair, free=False)
        elif kind == R_TOUCH:
            count = struct.unpack_from('<I', payload, 0)[0]
            position = 4
            for _ in range(count):
                kind_byte, klen = struct.unpack_from('<BB', payload, position)
                position += 2
                key = bytes(payload[position:position + klen])
                position += klen
                stamp = _TOUCH.unpack_from(payload, position)[0]
                position += _TOUCH.size
                entry = self.entries.get((kind_byte, key))
                if entry is not None:
                    entry.last_used = max(entry.last_used, stamp)
                    self.entries.move_to_end((kind_byte, key))
        else:
            raise ValueError('unknown record kind %d' % kind)

    def _validate_extents(self):
        """Drop entries whose extents leave the data region or overlap an earlier one (a corrupt or inconsistent index must cost entries, never serve bytes twice)."""
        geometry = self.geometry
        low, high = geometry.data_off, geometry.data_off + geometry.data_bytes
        ordered = sorted(self.entries.values(), key=lambda entry: entry.offset)
        cursor = low
        dropped = 0
        for entry in ordered:
            if entry.offset < cursor or entry.offset + entry.alloc > high or entry.offset % ALIGN:
                self._index_remove((entry.kind, entry.key), free=False)
                dropped += 1
            else:
                cursor = entry.offset + entry.alloc
        return dropped

    def _rebuild_allocator(self):
        geometry = self.geometry
        used = sorted((entry.offset, entry.alloc) for entry in self.entries.values())
        self.allocator = Allocator.from_used(geometry.data_off, geometry.data_bytes, used)
        self.logical_bytes = sum(entry.alloc for entry in self.entries.values())

    def _snapshot_locked(self):
        """Write the whole index to the inactive snapshot slot, then flip the superblock to it. Holds _wlock."""
        geometry = self.geometry
        with self._mutex:
            entries = list(self.entries.values())
            epoch = self.epoch + 1
            seq = self.seq
        data = self._encode_snapshot(entries, epoch, seq)
        if len(data) > geometry.snap_bytes:
            raise StoreFull('the index (%d bytes) does not fit its snapshot slot (%d)' % (len(data), geometry.snap_bytes))
        slot = 1 if self.active_snap == 0 else 0
        self.bf.write_buffers(geometry.snap_off[slot], [data])
        self.bf.sync()
        generation = self.generation + 1
        superblock = dict(generation=generation, uuid=self.uuid, device_bytes=geometry.device_bytes, snap_off0=geometry.snap_off[0], snap_off1=geometry.snap_off[1],
                          snap_bytes=geometry.snap_bytes, journal_off=geometry.journal_off, journal_bytes=geometry.journal_bytes, data_off=geometry.data_off,
                          data_bytes=geometry.data_bytes, active_snap=slot, snap_len=len(data), snap_crc=zlib.crc32(data) & 0xFFFFFFFF, epoch=epoch, seq=seq)
        self.bf.write_buffers((generation % 2) * ALIGN, [self._encode_superblock(superblock)])
        self.bf.sync()
        self.generation = generation
        self.active_snap = slot
        self.epoch = epoch
        self.jpos = 0
        self.stats['snapshots'] += 1

    def snapshot(self):
        with self._wlock:
            self._snapshot_locked()

    # -- the index ---------------------------------------------------------------------------------------------
    def _intern(self, value):
        known = self.interned.get(value)
        if known is None:
            if len(self.interned) > 1 << 20:
                self.interned.clear()
            self.interned[value] = known = value
        return known

    def _index_add(self, entry, journal_replay=False):
        pair = (entry.kind, entry.key)
        old = self.entries.get(pair)
        if old is not None:
            self._index_remove(pair, free=not journal_replay)
        entry.chain = self._intern(entry.chain)
        entry.tenant = self._intern(entry.tenant)
        entry.fp = self._intern(entry.fp)
        with self._mutex:
            self.entries[pair] = entry
            self.by_tenant.setdefault(entry.tenant, set()).add(pair)
            if entry.kind == CKPT:
                self.chains.setdefault(entry.chain, {})[entry.index] = entry.key
            self.logical_bytes += entry.alloc
            self.kind_counts[entry.kind] = self.kind_counts.get(entry.kind, 0) + 1
            if self.max_age_s:
                self.serial += 1
                self.created_order.append((entry.created, pair, self.serial, entry))

    def _index_remove(self, pair, free=True):
        """Take an entry out of the index. free: give its extent back (now, or at the last unpin when a reader holds it). Returns the entry."""
        with self._mutex:
            entry = self.entries.pop(pair, None)
            if entry is None:
                return None
            group = self.by_tenant.get(entry.tenant)
            if group is not None:
                group.discard(pair)
                if not group:
                    del self.by_tenant[entry.tenant]
            if entry.kind == CKPT:
                positions = self.chains.get(entry.chain)
                if positions is not None and positions.get(entry.index) == entry.key:
                    del positions[entry.index]
                    if not positions:
                        del self.chains[entry.chain]
            self.logical_bytes -= entry.alloc
            self.kind_counts[entry.kind] -= 1
            entry.dead = True
            entry.durable = bool(free)
            if free and entry.pins == 0:
                self._release_extent(entry)
            return entry

    def _release_extent(self, entry):
        self.allocator.release(entry.offset, entry.alloc)
        if self.discard_on_free:
            if self.bf.discard(entry.offset, entry.alloc):
                self.stats['discarded_bytes'] += entry.alloc

    # -- lookups (scheduler thread; short lock, no device IO) -----------------------------------------------------------
    def _expired(self, entry, now):
        if self.ttl_s and now - entry.last_used > self.ttl_s:
            return 'ttl'
        if self.max_age_s and now - entry.created > self.max_age_s:
            return 'age'
        return None

    def lookup(self, kind, key, now=None):
        """The live, unexpired entry, or None. Does not touch it."""
        self.stats['gets'] += 1
        with self._mutex:
            entry = self.entries.get((kind, key))
            if entry is None:
                self.stats['get_misses'] += 1
                return None
            if self._expired(entry, self.clock() if now is None else now):
                self.stats['get_misses'] += 1
                return None
            self.stats['get_hits'] += 1
            return entry

    def has(self, kind, key):
        return self.lookup(kind, key) is not None

    def touch(self, kind, key):
        """Mark an entry used now (its TTL restarts, it becomes the youngest in the LRU order)."""
        with self._mutex:
            entry = self.entries.get((kind, key))
            if entry is None:
                return False
            entry.last_used = self.clock()
            self.entries.move_to_end((kind, key))
            self.touched[(kind, key)] = entry.last_used
            return True

    def chain_reaches(self, chain, need_tokens):
        """Whether a CKPT entry of `chain` sits at a position of at least need_tokens."""
        with self._mutex:
            positions = self.chains.get(chain)
            if not positions:
                return False
            now = self.clock()
            for pos, key in positions.items():
                if pos >= need_tokens:
                    entry = self.entries.get((CKPT, key))
                    if entry is not None and not self._expired(entry, now):
                        return True
            return False

    def pin(self, pairs):
        """Pin the live entries named (kind, key) so their extents survive any eviction or deletion until unpin. -> a list parallel to `pairs`: the Entry or None."""
        out = []
        with self._mutex:
            now = self.clock()
            for pair in pairs:
                entry = self.entries.get(pair)
                if entry is None or self._expired(entry, now):
                    out.append(None)
                else:
                    entry.pins += 1
                    out.append(entry)
        return out

    def unpin(self, entries):
        with self._mutex:
            for entry in entries:
                if entry is None:
                    continue
                entry.pins -= 1
                if entry.pins == 0 and entry.dead and entry.durable:
                    self._release_extent(entry)

    # -- reads ---------------------------------------------------------------------------------------------------
    def _executor(self):
        if self._pool is None:
            from concurrent.futures import ThreadPoolExecutor

            self._pool = ThreadPoolExecutor(max_workers=self.threads, thread_name_prefix='qwen-nvme-read')
        return self._pool

    def read_many(self, entries, views, verify=None, threads=None):
        """Read each (pinned) entry's bytes into its view (len(view) == entry.nbytes). Runs of entries that sit back to back on the device are one device call
        (at most MAX_RUN_BYTES), the runs on up to `threads` workers. `verify(entry, view)` -> bool is called on the worker after a run is read. -> a list parallel to
        `entries`: None (read and verified), 'io' (the device call failed) or 'digest' (verify said no)."""
        count = len(entries)
        results = [None] * count
        order = sorted(range(count), key=lambda index: entries[index].offset)
        runs = []
        for index in order:
            entry = entries[index]
            if runs:
                last = entries[runs[-1][-1]]
                size = sum(entries[member].alloc for member in runs[-1])
                if last.offset + last.alloc == entry.offset and size + entry.alloc <= MAX_RUN_BYTES:
                    runs[-1].append(index)
                    continue
            runs.append([index])

        def do_run(run):
            first = entries[run[0]]
            try:
                self.bf.read_run(first.offset, [entries[member].nbytes for member in run], [views[member] for member in run])
            except Exception:
                for member in run:
                    results[member] = 'io'
                return
            if verify is not None:
                for member in run:
                    if not verify(entries[member], views[member]):
                        results[member] = 'digest'

        threads = self.threads if threads is None else max(1, int(threads))
        if threads > 1 and len(runs) > 1:
            futures = [self._executor().submit(do_run, run) for run in runs]
            for future in futures:
                future.result()
        else:
            for run in runs:
                do_run(run)
        self.stats['reads'] += count
        self.stats['read_bytes'] += sum(entry.nbytes for entry in entries)
        self.stats['read_errors'] += sum(1 for item in results if item == 'io')
        self.stats['digest_failures'] += sum(1 for item in results if item == 'digest')
        return results

    def read_bytes(self, entry):
        """One entry's bytes as a memoryview over an aligned buffer (one device call, no bounce): checkpoints, audits, tests."""
        scratch = Scratch(entry.alloc)
        self.bf.pread_into(entry.offset, [scratch.view[:entry.alloc]])
        return scratch.view[:entry.nbytes]

    # -- the journal ---------------------------------------------------------------------------------------------
    def _journal_append(self, records):
        """Append (kind, payload) records and make them durable. Holds _wlock. The records go out in groups of at most a quarter of the journal (one write and one sync
        each), so a crash between groups leaves a prefix of the records, and a group that does not fit the rest of the journal snapshots first."""
        limit = max(ALIGN, self.geometry.journal_bytes // 4)
        group, size = [], 0
        for record in records:
            span = _JHEAD.size + len(record[1])
            if span > limit:
                raise ValueError('a journal record of %d bytes is over a quarter of the journal' % span)
            if group and roundup(size + span) > limit:
                self._journal_group(group, size)
                group, size = [], 0
            group.append(record)
            size += span
        if group:
            self._journal_group(group, size)

    def _journal_group(self, records, total):
        geometry = self.geometry
        encoded = []
        if self.jpos + roundup(total) > geometry.journal_bytes:
            self._snapshot_locked()
        seq = self.seq
        for kind, payload in records:
            seq += 1
            body = _JHEAD.pack(JREC_MAGIC, 0, self.epoch, seq, kind, 0, len(payload))
            crc = zlib.crc32(body[8:] + payload) & 0xFFFFFFFF
            encoded.append(_JHEAD.pack(JREC_MAGIC, crc, self.epoch, seq, kind, 0, len(payload)))
            encoded.append(payload)
        self.bf.write_buffers(geometry.journal_off + self.jpos, [b''.join(encoded)])
        self.bf.sync()
        self.jpos += roundup(total)
        self.seq = seq
        self.stats['journal_records'] += len(records)
        self.stats['journal_syncs'] += 1

    @staticmethod
    def _put_payloads(entries):
        out = []
        for low in range(0, len(entries), PUT_RECORD_ENTRIES):
            part = entries[low:low + PUT_RECORD_ENTRIES]
            out.append((R_PUT, struct.pack('<I', len(part)) + b''.join(pack_entry(entry) for entry in part)))
        return out

    @staticmethod
    def _del_payload(pairs):
        body = [struct.pack('<I', len(pairs))]
        for kind, key in pairs:
            body.append(struct.pack('<BB', kind, len(key)) + key)
        return (R_DEL, b''.join(body))

    def _touch_records(self, force=False):
        """A TOUCH record for the entries used since the last one, at most every TOUCH_JOURNAL_S (their TTL clocks survive a restart)."""
        with self._mutex:
            now = self.clock()
            if not self.touched or (not force and now - self.touch_journaled < TOUCH_JOURNAL_S):
                return []
            pending = list(self.touched.items())[:TOUCH_RECORD_KEYS]
            for pair, _ in pending:
                del self.touched[pair]
            self.touch_journaled = now
        body = [struct.pack('<I', len(pending))]
        for (kind, key), stamp in pending:
            body.append(struct.pack('<BB', kind, len(key)) + key + _TOUCH.pack(stamp))
        self.stats['touch_records'] += 1
        return [(R_TOUCH, b''.join(body))]

    # -- capacity ---------------------------------------------------------------------------------------------
    def capacity_bytes(self):
        """Bytes of objects the store may hold: the data region less the free-space floor, and at most max_bytes when that is set."""
        limit = max(0, self.geometry.data_bytes - self.free_floor_bytes)
        if self.max_bytes:
            limit = min(limit, self.max_bytes)
        return limit

    def max_entries(self):
        return max(16, (self.geometry.snap_bytes - 4096) // DEFAULT_ENTRY_BUDGET)

    def _remove_durably(self, entries, counter):
        """Journal the deletion of `entries`, then take them out of the index and give their extents back. Holds _wlock."""
        if not entries:
            return
        for low in range(0, len(entries), 4096):
            part = entries[low:low + 4096]
            self._journal_append([self._del_payload([(entry.kind, entry.key) for entry in part])])
            for entry in part:
                self._index_remove((entry.kind, entry.key), free=True)
        self.stats[counter] += len(entries)

    # -- writes ----------------------------------------------------------------------------------------------
    def put_batch(self, items):
        """Store the items (PutItem). -> a list parallel to `items`: True stored (or already held), False refused (no room). Blocks for the device; the caller is the
        tier's writer thread, never the scheduler's."""
        if self._closed:
            raise StoreError('the store is closed')
        for item in items:
            if item.digest is None:
                item.digest = digest_of(item.buffers)
        results = [False] * len(items)
        with self._wlock:
            now = self.clock()
            fresh = []
            for position, item in enumerate(items):
                pair = (item.kind, item.key)
                with self._mutex:
                    old = self.entries.get(pair)
                    if old is not None and not self._expired(old, now):
                        old.last_used = now
                        self.entries.move_to_end(pair)
                        self.touched[pair] = now
                        self.stats['put_duplicates'] += 1
                        results[position] = True
                        continue
                fresh.append(position)
            if not fresh:
                return results
            # an expired entry under the same key is replaced
            stale = []
            for position in fresh:
                old = self.entries.get((items[position].kind, items[position].key))
                if old is not None:
                    stale.append(old)
            self._remove_durably([entry for entry in stale if not entry.pins], 'expired_ttl')
            need = sum(roundup(items[position].nbytes) for position in fresh)
            ok = self._make_room(need, len(fresh))
            if not ok:
                self.stats['put_refused'] += len(fresh)
                return results
            placed = self._place(fresh, items)
            if placed is None:
                self.stats['put_refused'] += len(fresh)
                return results
            entries = []
            try:
                self._write_data(placed, items)
                self.bf.sync()
                created = self.clock()
                for position, offset in placed:
                    item = items[position]
                    entries.append(Entry(item.kind, item.key, offset, item.nbytes, item.tenant, item.chain, item.index, item.extra, created, created, item.digest,
                                         item.fp))
                records = self._touch_records() + self._put_payloads(entries)
                self._journal_append(records)
            except BaseException:
                # nothing named these extents durably: they are free again
                with self._mutex:
                    for position, offset in placed:
                        self.allocator.release(offset, roundup(items[position].nbytes))
                raise
            for entry in entries:
                self._index_add(entry)
            for position, _offset in placed:
                results[position] = True
            self.stats['puts'] += len(entries)
            self.stats['put_bytes'] += sum(entry.nbytes for entry in entries)
        return results

    def _make_room(self, need, count):
        """Evict until `need` more bytes (and `count` more entries) fit under the capacity. Holds _wlock."""
        limit = self.capacity_bytes()
        if need > limit:
            return False
        with self._mutex:
            fits = self.logical_bytes + need <= limit and len(self.entries) + count <= self.max_entries()
        if fits:
            return True
        entries_over = len(self.entries) + count - self.max_entries()
        victims = []
        with self._mutex:
            over = self.logical_bytes + need - limit
            for pair, entry in self.entries.items():
                if over <= 0 and entries_over <= 0:
                    break
                if entry.pins:
                    continue
                victims.append(entry)
                over -= entry.alloc
                entries_over -= 1
            if over > 0 or entries_over > 0:
                return False
        self._remove_durably(victims, 'evicted_lru')
        self.stats['put_evictions'] += len(victims)
        return True

    def _place(self, positions, items):
        """Extents for the items at `positions`: one contiguous run when it fits, else one per item, evicting the least recently used until it does. -> [(position,
        offset)] or None. Holds _wlock."""
        sizes = [roundup(items[position].nbytes) for position in positions]
        total = sum(sizes)
        for _attempt in range(64):
            with self._mutex:
                base = self.allocator.alloc(total)
                if base is not None:
                    out, cursor = [], base
                    for position, size in zip(positions, sizes):
                        out.append((position, cursor))
                        cursor += size
                    return out
                # no single run: one extent per item, all or nothing
                taken = []
                for position, size in zip(positions, sizes):
                    offset = self.allocator.alloc(size)
                    if offset is None:
                        for _p, taken_offset, taken_size in taken:
                            self.allocator.release(taken_offset, taken_size)
                        taken = None
                        break
                    taken.append((position, offset, size))
                if taken is not None:
                    return [(position, offset) for position, offset, _size in taken]
                # fragmentation: evict the least recently used and try again
                victim = next((entry for entry in self.entries.values() if not entry.pins), None)
            if victim is None:
                return None
            self._remove_durably([victim], 'evicted_lru')
        return None

    def _write_data(self, placed, items):
        """Write each placed item at its offset; items that sit back to back and whose single buffer is aligned go out as one call."""
        bf = self.bf
        run_offset = None
        run_views = []
        run_end = None

        def flush():
            nonlocal run_offset, run_views, run_end
            if run_views:
                bf.pwrite_from(run_offset, run_views)
            run_offset, run_views, run_end = None, [], None

        for position, offset in sorted(placed, key=lambda pair: pair[1]):
            item = items[position]
            direct = len(item.buffers) == 1 and is_aligned(memoryview(item.buffers[0]).cast('B') if memoryview(item.buffers[0]).format != 'B' else
                                                           memoryview(item.buffers[0]), bf.align) and item.nbytes % bf.align == 0
            if direct:
                view = memoryview(item.buffers[0])
                if view.format != 'B':
                    view = view.cast('B')
                if run_views and offset == run_end:
                    run_views.append(view)
                else:
                    flush()
                    run_offset, run_views = offset, [view]
                run_end = offset + item.nbytes
            else:
                flush()
                bf.write_buffers(offset, item.buffers, item.nbytes)
        flush()

    # -- deletion ----------------------------------------------------------------------------------------------
    def delete(self, pairs):
        """Remove the entries named (kind, key), durably. -> how many were held."""
        with self._wlock:
            entries = []
            with self._mutex:
                for pair in pairs:
                    entry = self.entries.get(pair)
                    if entry is not None:
                        entries.append(entry)
            self._remove_durably(entries, 'deleted_keys')
            return len(entries)

    def delete_tenant(self, tenant):
        """Remove every entry of a tenant (the tag or its 8-byte hash), durably, with ONE journal record. -> how many."""
        tenant = tenant if isinstance(tenant, bytes) and len(tenant) == 8 else tenant_hash(tenant)
        with self._wlock:
            with self._mutex:
                pairs = list(self.by_tenant.get(tenant, ()))
            if not pairs:
                return 0
            self._journal_append([(R_DEL_TENANT, tenant)])
            for pair in pairs:
                self._index_remove(pair, free=True)
            self.stats['deleted_tenant'] += len(pairs)
            return len(pairs)

    def wipe(self):
        """Remove everything, durably (one journal record) and at once: the index is cleared wholesale and the allocator rebuilt around the extents readers still pin.
        -> how many entries were held."""
        with self._wlock:
            self._journal_append([(R_WIPE, b'')])
            with self._mutex:
                count = len(self.entries)
                held = [entry for entry in self.entries.values() if entry.pins] + [entry for entry in self._dead if entry.pins]
                for entry in list(self.entries.values()) + self._dead:
                    entry.dead = True
                    entry.durable = True
                self.entries.clear()
                self.by_tenant.clear()
                self.chains.clear()
                self.created_order.clear()
                self.touched.clear()
                self.kind_counts = {KV: 0, CKPT: 0}
                self.logical_bytes = 0
                self._dead = []
                geometry = self.geometry
                self.allocator = Allocator.from_used(geometry.data_off, geometry.data_bytes, sorted((entry.offset, entry.alloc) for entry in held))
            self.stats['wipes'] += 1
            if self.discard_on_free:
                self.bf.discard(self.geometry.data_off, self.geometry.data_bytes)
            return count

    def mark_bad(self, entries):
        """Take entries out of the index now (any thread: a block that failed its digest must not be offered again) and leave their DEL record and their extents to
        flush_dead on the writer thread."""
        with self._mutex:
            for entry in entries:
                if self._index_remove((entry.kind, entry.key), free=False) is not None:
                    self._dead.append(entry)

    def flush_dead(self):
        """Journal the deletion of the entries mark_bad took out, then give their extents back (once no reader pins them). -> how many."""
        with self._wlock:
            with self._mutex:
                dead, self._dead = self._dead, []
            if not dead:
                return 0
            for low in range(0, len(dead), 4096):
                part = dead[low:low + 4096]
                self._journal_append([self._del_payload([(entry.kind, entry.key) for entry in part])])
            with self._mutex:
                for entry in dead:
                    entry.durable = True
                    if entry.pins == 0:
                        self._release_extent(entry)
            self.stats['deleted_keys'] += len(dead)
            return len(dead)

    def drop_where(self, predicate, counter='fp_drops'):
        """Remove every entry for which predicate(entry) is true (a layout fingerprint that no longer matches), durably. -> how many."""
        with self._wlock:
            with self._mutex:
                entries = [entry for entry in self.entries.values() if predicate(entry)]
            entries = [entry for entry in entries if not entry.pins]
            self._remove_durably(entries, counter)
            return len(entries)

    def sweep(self, now=None):
        """Remove expired entries (the TTL since last use, the age since written), durably. -> how many."""
        if self._closed:
            return 0
        with self._wlock:
            now = self.clock() if now is None else now
            expired_ttl, expired_age = [], []
            with self._mutex:
                scanned = 0
                for pair, entry in self.entries.items():
                    scanned += 1
                    if scanned > SWEEP_LIMIT:
                        break
                    if self.ttl_s and now - entry.last_used > self.ttl_s:
                        if not entry.pins:
                            expired_ttl.append(entry)
                    else:
                        break                              # the LRU order is the order of last use: the rest is younger
                if self.max_age_s:
                    seen = set(id(entry) for entry in expired_ttl)
                    while self.created_order and now - self.created_order[0][0] > self.max_age_s:
                        _created, pair, _serial, entry = self.created_order[0]
                        live = self.entries.get(pair)
                        if live is not entry:
                            self.created_order.popleft()
                            continue
                        if entry.pins:
                            break
                        self.created_order.popleft()
                        if id(entry) not in seen:
                            expired_age.append(entry)
            self._remove_durably(expired_ttl, 'expired_ttl')
            self._remove_durably(expired_age, 'expired_age')
            records = self._touch_records()
            if records:
                self._journal_append(records)
            return len(expired_ttl) + len(expired_age)

    # -- reporting and closing -------------------------------------------------------------------------------
    def snapshot_stats(self):
        with self._mutex:
            values = dict(self.stats)
            values.update(entries=len(self.entries), kv_entries=self.kind_counts.get(KV, 0), ckpt_entries=self.kind_counts.get(CKPT, 0), logical_bytes=self.logical_bytes,
                          capacity_bytes=self.capacity_bytes(), data_bytes=self.geometry.data_bytes, free_bytes=self.allocator.free,
                          fragments=self.allocator.fragments(), largest_free=self.allocator.largest(), journal_used=self.jpos,
                          journal_bytes=self.geometry.journal_bytes, epoch=self.epoch, generation=self.generation)
        return values

    def tenant_entries(self, tenant):
        tenant = tenant if isinstance(tenant, bytes) and len(tenant) == 8 else tenant_hash(tenant)
        with self._mutex:
            return len(self.by_tenant.get(tenant, ()))

    def close(self, snapshot=True):
        if self._closed:
            return
        with self._wlock:
            if snapshot and not self.bf.readonly:
                try:
                    records = self._touch_records(force=True)
                    if records:
                        self._journal_append(records)
                except Exception:
                    pass
            self._closed = True
        pool, self._pool = self._pool, None
        if pool is not None:
            pool.shutdown(wait=True)
        self.bf.close()
