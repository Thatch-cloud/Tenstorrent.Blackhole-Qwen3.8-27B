"""P3a: the peer-to-peer restore on the cards, standalone (qwen-c2-serving.yml 'fabric' with C2_FABRIC_PROBE=p2p, C2_CARDS=quad; docs/tp4-p2p-restore.md section 8).

    python3 -B /c2/scripts/ci/tp4_p2p_probe.py --fabric FABRIC_1D --output /probe-results/p2p-p3a.json

What it does, in this order, and what it refuses before it does anything:
  0. THE DRIVE. Identity first, before a byte is written or a card opened: the drive named by --drive (P2P_DRIVE_BY_ID) must be the one with the model and serial the job config gives
     (P2P_DRIVE_MODEL, P2P_DRIVE_SERIAL), proved through /dev/disk/by-id (a link of the right name that resolves to the node) and, independently, through the node's sysfs model and
     serial. Then R1's two readings, printed: the controller state and the namespace size. Anything but `live` (and a non-empty namespace) is a refusal. A drive that carries the NVMe
     tier's superblock is refused too (--allow-store-overwrite says it is a gate's drive to destroy). The values come from the job's environment, never from a tracked file, and are
     never printed (the log of a public repository is public): the report carries a short hash of them.
  1. THE MESH AND THE OPS. The (1, 4) mesh as the kv-read probe opens it; the device-op extension's ops must exist (staging alloc and info, the checksum, the copy from staging) with the
     raw block ops of the host tier; the scratch KV caches are allocated as production allocates its own (bfloat8_b tiles, one head per chip).
  2. THE KNOWN OBJECT. N blocks of random bytes in the host tier's layout, laid out chip-major (the NVMe tier's P2P layout), written O_DIRECT to the drive at --scratch-offset, with the
     per-chip checksums the spill thread would record.
  3. THE WINDOWS. Every published card's TLB window is taken, aimed at its staging buffer, pinned and proved with the canary (qwen_p2p_restore.TtLink); the staging buffer's description is
     printed (without PCI addresses).
  4. ONE CARD FIRST. The first published card, the widest link first (the x16 card), alone: its shards of the first blocks are read from the drive into its staging buffer and its device
     checksums compared with the host's. Only then
  5. EVERY PUBLISHED CARD. The whole object: read (timed on its own, GB/s drive -> cards), checked (the cards' checksums against the host's), copied by qwen_write_blocks_from_staging into
     the scratch caches, read back with qwen_read_blocks_raw, and compared byte for byte with what was written to the drive. Timings: read GB/s, the checksum and copy ops in ms.
  6. THE NEGATIVE ARM. One byte of one chip's shard flipped on the drive: the restore must be refused by the checksum and the block it was for must still be zeros in the cache (nothing
     written); then the byte is put back and the same restore must pass.
  7. THE END. The windows released and every `p2pmem/available` equal to `size` again.

Last stdout lines: 'P3A verdict=PASS|FAIL|NOT-MEASURED ...' then one JSON object (kind p2p-p3a-quad). Exit 0 PASS, 1 FAIL, 2 refused or could not run, 3 the watchdog.
"""

import argparse
import gc
import hashlib
import json
import mmap
import os
import re
import stat
import sys
import threading
import time
import traceback
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import qwen_p2p_restore as p2p  # noqa: E402

KIND = 'p2p-p3a-quad'
WATCHDOG_S = 1800
PAGE = 4096
BLOCK = 64
HEAD_DIM = 256
PACKED_HEAD_BLOCK_BYTES = 17408          # one head's 64 x 256 block packed as bfloat8_b: 16 tiles of 1,088 bytes
STORE_MAGIC = b'QNVMSB01'                # the NVMe tier's superblock (qwen_nvme_store.SB_MAGIC), at byte 0 or 4096
DEFAULT_BLOCKS = 96
DEFAULT_SCRATCH_OFFSET = 1 << 30
ENV_NAMES = ('P2P_DRIVE_BY_ID', 'P2P_DRIVE_MODEL', 'P2P_DRIVE_SERIAL', 'P2P_CARDS')


class Refusal(Exception):
    """The probe will not go on, and nothing was written. The message says why (it never carries the drive's model, serial or a PCI address)."""


PCI_ADDRESS = re.compile(r'(?:[0-9a-fA-F]{4}:)?[0-9a-fA-F]{2}:[0-9a-fA-F]{2}\.[0-7]')


def scrubber(options):
    """A function that takes the drive's path, by-id name, model and serial and every PCI address out of a text: the Actions log of a public repository is public, and an exception message
    (a path in an OSError, a card in a refusal) is how they would get there."""
    secrets = []
    for value in (options.drive, os.path.basename(options.drive or ''), options.drive_model, options.drive_serial,
                  model_token(options.drive_model or ''), model_token(options.drive_serial or '')):
        if value and len(str(value)) >= 3:
            secrets.append(str(value))
    secrets.sort(key=len, reverse=True)

    def scrub(text):
        text = str(text)
        for secret in secrets:
            text = text.replace(secret, '<redacted>')
        text = re.sub(r'/dev/disk/by-id/\S+', '/dev/disk/by-id/<redacted>', text)
        text = re.sub(r'nvme-[A-Za-z0-9_.:+-]{6,}', 'nvme-<redacted>', text)
        return PCI_ADDRESS.sub('<pci>', text)

    return scrub


def log_line(log, message, *values):
    log('P3A ' + (message % values if values else message))


# ================================================================================================
# The drive: identity first, then R1's readings
# ================================================================================================
def model_token(text):
    """The way udev writes a model into a /dev/disk/by-id name: runs of anything but letters and digits become one underscore."""
    return re.sub(r'[^A-Za-z0-9]+', '_', str(text).strip()).strip('_')


def accepted_names(model, serial):
    """A regular expression for the by-id names of the NVMe namespaces of this model and serial: nvme-<model>_<serial> and nvme-<model>_<serial>_<namespace id>."""
    return re.compile(r'nvme-%s_%s(?:_[0-9]+)?' % (re.escape(model_token(model)), re.escape(model_token(serial))))


def _text(path):
    try:
        with open(path) as handle:
            return handle.read().strip()
    except OSError:
        return None


def block_sysfs(device, sysfs='/sys'):
    """The sysfs directory of a block device node, found by its device NUMBER (inside a container the node can be called anything); None for anything that is not a block device."""
    try:
        info = os.stat(device)
    except OSError:
        return None
    if not stat.S_ISBLK(info.st_mode):
        return None
    base = os.path.realpath(os.path.join(sysfs, 'dev', 'block', '%d:%d' % (os.major(info.st_rdev), os.minor(info.st_rdev))))
    return base if os.path.isdir(base) else None


def sysfs_identity(device, sysfs='/sys'):
    """(model, serial) the kernel reports for the namespace's controller, either None when unreadable."""
    base = block_sysfs(device, sysfs)
    if base is None:
        return None, None
    holders = [os.path.join(base, 'device')]
    parent = base
    for _level in range(3):
        parent = os.path.dirname(parent)
        if os.path.basename(parent).startswith('nvme'):
            holders.append(parent)
    model = serial = None
    for holder in holders:
        model = model or _text(os.path.join(holder, 'model'))
        serial = serial or _text(os.path.join(holder, 'serial'))
    return model, serial


def identity_check(device, model, serial, by_id_dir='/dev/disk/by-id', sysfs='/sys'):
    """Prove that `device` is the drive with this model and serial, or raise Refusal. Two independent looks, at least one positive and none contradicting:
      by-id  a link in /dev/disk/by-id whose name is nvme-<model>_<serial>[_<nsid>] and that resolves to the same node as `device`; a link of that name pointing elsewhere refuses;
      sysfs  the model and serial the kernel reports for the node's controller (found by device number).
    A device given AS a by-id path that is a plain node (a container that was handed the node under that name) proves nothing by its name alone: only the sysfs look can then."""
    if not model or not serial:
        raise Refusal('the drive\'s model and serial are not configured (P2P_DRIVE_MODEL, P2P_DRIVE_SERIAL): not a byte is written to a drive that was not identified')
    if not device or not os.path.exists(device):
        raise Refusal('the drive node does not exist')
    real = os.path.realpath(device)
    pattern = accepted_names(model, serial)
    positive, why = [], []
    try:
        names = sorted(os.listdir(by_id_dir))
    except OSError:
        names = []
    for name in names:
        if not pattern.fullmatch(name):
            continue
        link = os.path.join(by_id_dir, name)
        if os.path.islink(link):
            if os.path.realpath(link) == real:
                positive.append('by-id')
            else:
                why.append('the by-id link of that model and serial names another device')
    seen_model, seen_serial = sysfs_identity(device, sysfs)
    if seen_model is not None and seen_serial is not None:
        if model_token(seen_model) == model_token(model) and model_token(seen_serial) == model_token(serial):
            positive.append('sysfs')
        else:
            why.append('the kernel reports another model or serial for the device')
    if why:
        raise Refusal('the drive is not the one configured: %s' % '; '.join(sorted(set(why))))
    if not positive:
        raise Refusal('the drive cannot be shown to be the one configured: no /dev/disk/by-id link of its model and serial resolves to the node, and the kernel\'s identity for it is unreadable')
    return {'proof': sorted(set(positive)), 'fingerprint': hashlib.sha256(('%s|%s' % (model_token(model), model_token(serial))).encode()).hexdigest()[:6]}


def r1_readings(device, sysfs='/sys'):
    """R1's two readings of the cache drive, never an I/O to it: the controller's state and the namespace's size in bytes (docs/tp4-kv-nvme.md section 12). state is 'live' only when the
    kernel says so; 'unreadable' when it cannot be asked."""
    base = block_sysfs(device, sysfs)
    if base is None:
        try:
            size = os.stat(device).st_size
        except OSError:
            size = 0
        return {'state': 'unreadable', 'namespace_bytes': size, 'node': os.path.basename(device)}
    sectors = _text(os.path.join(base, 'size'))
    nbytes = int(sectors) * 512 if sectors and sectors.isdigit() else 0
    holders = [os.path.join(base, 'device')]
    parent = base
    for _level in range(3):
        parent = os.path.dirname(parent)
        if os.path.basename(parent).startswith('nvme'):
            holders.append(parent)
    state = None
    for holder in holders:
        state = _text(os.path.join(holder, 'state'))
        if state:
            break
    return {'state': state or 'unreadable', 'namespace_bytes': nbytes, 'node': os.path.basename(base)}


def check_drive(options, log):
    """Steps 0: identity, then R1's readings, then the tier's superblock. -> the report's drive section; Refusal."""
    identity = identity_check(options.drive, options.drive_model, options.drive_serial, options.by_id_dir, options.sysfs)
    readings = r1_readings(options.drive, options.sysfs)
    log_line(log, 'drive identity=verified proof=%s fingerprint=%s', '+'.join(identity['proof']), identity['fingerprint'])
    log_line(log, 'R1 controller_state=%s namespace_bytes=%d', readings['state'], readings['namespace_bytes'])
    if readings['state'] != 'live':
        raise Refusal('R1: the controller state is %r, not live: re-probe the drive (docs/tp4-kv-nvme.md R1) before any tier job' % readings['state'])
    if readings['namespace_bytes'] <= 0:
        raise Refusal('R1: the namespace reads 0 bytes')
    need = options.scratch_offset + options.blocks * options.caches * 4 * PACKED_HEAD_BLOCK_BYTES * options.heads_per_chip
    if options.scratch_offset % PAGE or options.scratch_offset < 1 << 20:
        raise Refusal('--scratch-offset %d must be a multiple of %d and at least 1 MiB (the tier\'s superblocks and journal are in the first MiB)' % (options.scratch_offset, PAGE))
    if need > readings['namespace_bytes']:
        raise Refusal('the object (%d bytes from offset %d) does not fit the %d-byte namespace' % (need - options.scratch_offset, options.scratch_offset, readings['namespace_bytes']))
    head = read_head(options.drive)
    if head is not None and (STORE_MAGIC in head[:8] or STORE_MAGIC in head[PAGE:PAGE + 8]) and not options.allow_store_overwrite:
        raise Refusal('the drive carries the NVMe tier\'s superblock: P3a would overwrite a region of a live store (--allow-store-overwrite if this is a gate\'s drive to destroy)')
    return {'identity': 'verified', 'proof': identity['proof'], 'fingerprint': identity['fingerprint'], 'r1': readings}


def read_head(device, count=2 * PAGE):
    """The first two pages of the device with a plain buffered read (a read changes nothing); None when it cannot be read."""
    try:
        with open(device, 'rb') as handle:
            return handle.read(count)
    except OSError:
        return None


# ================================================================================================
# The drive as a file to read and write with O_DIRECT
# ================================================================================================
class Drive(object):
    """The block device open O_DIRECT for the probe's own reads and writes of its scratch region. (Opening it is what writes become possible with: check_drive has already run.)"""

    def __init__(self, path, direct=True):
        flags = os.O_RDWR | (getattr(os, 'O_DIRECT', 0) if direct else 0)
        self.fd = os.open(path, flags)
        self.direct = direct
        self.reads = 0

    def write_at(self, offset, data):
        view = memoryview(data)
        done = 0
        while done < len(view):
            done += os.pwritev(self.fd, [view[done:]], offset + done)
        os.fsync(self.fd)

    def read_at(self, offset, count):
        arena = mmap.mmap(-1, count)
        view = memoryview(arena)
        done = 0
        while done < count:
            got = os.preadv(self.fd, [view[done:]], offset + done)
            if got <= 0:
                raise IOError('short read at %d' % (offset + done))
            done += got
        data = bytes(view)
        view.release()
        arena.close()
        return data

    def reader(self, offset, view):
        """The engine's read: fill `view` from the drive at `offset`."""
        if offset % PAGE or len(view) % PAGE:
            raise ValueError('unaligned read: offset %d length %d' % (offset, len(view)))
        done = 0
        while done < len(view):
            got = os.preadv(self.fd, [view[done:]], offset + done)
            if got <= 0:
                raise IOError('short read at %d' % (offset + done))
            done += got

    def flip(self, offset):
        """Flip every bit of the byte at `offset` (a read-modify-write of its page); -> the old byte."""
        page = offset - offset % PAGE
        data = bytearray(self.read_at(page, PAGE))
        old = data[offset - page]
        data[offset - page] = old ^ 0xFF
        arena = mmap.mmap(-1, PAGE)
        arena.write(bytes(data))
        self.write_at(page, arena)
        arena.close()
        return old

    def put_byte(self, offset, value):
        page = offset - offset % PAGE
        data = bytearray(self.read_at(page, PAGE))
        data[offset - page] = value
        arena = mmap.mmap(-1, PAGE)
        arena.write(bytes(data))
        self.write_at(page, arena)
        arena.close()

    def close(self):
        if self.fd is not None:
            os.close(self.fd)
            self.fd = None


# ================================================================================================
# The known object
# ================================================================================================
def make_blocks(layout, count, seed=20261030):
    """`count` random blocks in the host tier's layout, their chip-major objects in one page-aligned arena, and the per-chip checksums of each. -> (payloads, arena, sums)"""
    rng = np.random.default_rng(seed)
    payloads = [rng.integers(0, 256, layout.block_bytes, dtype=np.uint8) for _ in range(count)]
    arena = mmap.mmap(-1, max(PAGE, count * layout.block_bytes))
    view = memoryview(arena)
    sums = []
    for index, payload in enumerate(payloads):
        obj = view[index * layout.block_bytes:(index + 1) * layout.block_bytes]
        layout.to_chipmajor(payload, obj)
        sums.append(layout.checksums(obj))
        obj.release()
    return payloads, arena, sums


class Entry(object):
    """What the engine needs of a stored block (the store entry's offset, size and checksums)."""

    def __init__(self, offset, nbytes, aux):
        self.offset, self.nbytes, self.aux = offset, nbytes, aux


# ================================================================================================
# The cards
# ================================================================================================
def link_width(bdf, sysfs_pci=p2p.SYSFS_PCI):
    text = _text(os.path.join(sysfs_pci, bdf, 'current_link_width'))
    return int(text) if text and text.isdigit() else 0


def alloc_cache(ttnn, mesh, blocks, heads_per_chip):
    """One scratch KV cache tensor as production allocates its own: zeros of the per-chip shape, bfloat8_b tiles in DRAM, replicated to the mesh."""
    import torch

    return ttnn.as_tensor(torch.zeros(blocks, heads_per_chip, BLOCK, HEAD_DIM, dtype=torch.bfloat16), device=mesh, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT,
                          memory_config=ttnn.DRAM_MEMORY_CONFIG, mesh_mapper=ttnn.ReplicateTensorToMesh(mesh))


def expected_rows(layout, payloads, tensor, chip, ids_count):
    """The bytes chip's slice of cache tensor `tensor` must hold for blocks 0..ids_count-1: [ids_count, slice_bytes]."""
    return np.stack([payloads[j].reshape(layout.caches, layout.chips, layout.slice_bytes)[tensor, chip] for j in range(ids_count)])


def read_back(ttnn, cache, chips, ids, slice_bytes):
    out = np.empty((chips, len(ids), slice_bytes), dtype=np.uint8)
    ttnn.qwen_read_blocks_raw(cache, out, ids)
    return out


# ================================================================================================
# The run
# ================================================================================================
def run_data_plane(ttnn, mesh, options, drive_info, report, log, alloc=alloc_cache, drive_factory=Drive, engine_factory=None):
    chips = int(mesh.get_num_devices())
    report['chips'] = chips
    heads = options.heads_per_chip
    blocks = options.blocks
    cache_blocks = blocks + 2                         # (the last id is the negative arm's: never written by the good restores)
    caches = [alloc(ttnn, mesh, cache_blocks, heads) for _ in range(options.caches)]
    slice_bytes = int(ttnn.qwen_block_bytes(caches[0]))
    layout = p2p.ShardLayout(options.caches, chips, slice_bytes)
    report['layout'] = dict(caches=layout.caches, chips=layout.chips, slice_bytes=slice_bytes, shard_bytes=layout.shard_bytes, block_bytes=layout.block_bytes)
    if not layout.aligned():
        raise Refusal('a shard of %d x %d bytes is not a whole number of 4 KiB pages: pick --caches as a multiple of 4' % (layout.caches, slice_bytes))
    payloads, arena, sums = make_blocks(layout, blocks)
    total = blocks * layout.block_bytes
    drive = drive_factory(options.drive)
    report['bytes'] = total
    try:
        # 3. the windows, BEFORE a byte of the object is written: the cards named, the staging buffer allocated, every window taken, aimed, pinned and proved, the mappings checked
        api = p2p.StagingApi(ttnn, mesh, caches)
        make_engine = engine_factory or (lambda api, layout, drive, options: p2p.P2PEngine(api, layout, drive.reader, threads=options.threads, staging_bytes=options.staging_mib << 20,
                                                                                            expected=options.cards or None, allow_unverified_map=options.allow_unverified_map,
                                                                                            logger=lambda m, *v: log_line(log, '%s', scrubber(options)(m % v if v else m))))
        engine = make_engine(api, layout, drive, options)
        try:
            describe_engine(engine, report, log, options.sysfs_pci)
            # 2. the known object, written after the identity was proved and the windows were
            began = time.perf_counter()
            drive.write_at(options.scratch_offset, arena)
            report['write'] = dict(seconds=time.perf_counter() - began, gbps=total / max(time.perf_counter() - began, 1e-9) / 1e9)
            log_line(log, 'wrote %d blocks (%d bytes) chip-major at offset %d: %.2f GB/s', blocks, total, options.scratch_offset, report['write']['gbps'])
            entries = [Entry(options.scratch_offset + index * layout.block_bytes, layout.block_bytes, layout.pack_checksums(sums[index])) for index in range(blocks)]
            ids = list(range(blocks))
            # 4. one card first: the widest link among the published cards
            widths = [(link_width(bdf, options.sysfs_pci), position) for position, (_index, bdf) in enumerate(engine.cards)]
            first = max(widths, key=lambda item: (item[0], -item[1]))[1]
            report['first_card'] = dict(position=first, link_width=widths[first][0])
            few = entries[:min(blocks, 2 * engine.chunk_blocks)]
            began = time.perf_counter()
            stream = engine.start(few, chips=[first])
            try:
                while True:
                    chunk = stream.take(wait=True)
                    if chunk is None:
                        break
                    try:
                        stream.verify(chunk)
                    finally:
                        stream.release(chunk)
            finally:
                stream.close()
            engine.api.sync()
            engine.dirty.clear()
            report['one_card'] = dict(position=first, blocks=len(few), verified=True, seconds=time.perf_counter() - began)
            log_line(log, 'one card (position %d, link width x%d): %d blocks read, checksums match', first, widths[first][0], len(few))
            # 5. every published card: the read on its own, then the whole restore
            began = time.perf_counter()
            stream = engine.start(entries)
            try:
                while True:
                    chunk = stream.take(wait=True)
                    if chunk is None:
                        break
                    stream.release(chunk)
            finally:
                stream.close()
            seconds = time.perf_counter() - began
            report['read'] = dict(seconds=seconds, gbps=total / max(seconds, 1e-9) / 1e9, chunk_blocks=engine.chunk_blocks, threads=engine.threads)
            log_line(log, 'read %d bytes drive -> cards in %.3f s: %.2f GB/s (%d threads, chunks of %d blocks)', total, seconds, report['read']['gbps'], engine.threads, engine.chunk_blocks)
            for key in ('check_ms', 'write_ms', 'read_ms'):
                engine.stats[key] = 0
            began = time.perf_counter()
            engine.restore(entries, ids)
            seconds = time.perf_counter() - began
            stats = dict(engine.stats)
            report['restore'] = dict(seconds=seconds, gbps=total / max(seconds, 1e-9) / 1e9, check_ms=stats['check_ms'], write_ms=stats['write_ms'],
                                     check_ms_per_chunk=stats['check_ms'] / max(1, stats['chunks']), write_ms_per_chunk=stats['write_ms'] / max(1, stats['chunks']),
                                     late_data=stats['mismatch_retries'])
            log_line(log, 'restore %d blocks in %.3f s (%.2f GB/s end to end): checksum ops %.1f ms, copy ops %.1f ms (%d chunks)', blocks, seconds, report['restore']['gbps'],
                     stats['check_ms'], stats['write_ms'], stats['chunks'])
            mismatched = compare_cache(ttnn, caches, layout, payloads, ids, chips, slice_bytes)
            report['exact'] = dict(tensors=len(caches), blocks=blocks, mismatched_rows=mismatched)
            log_line(log, 'byte-exact against what was written to the drive: %s (%d mismatched rows)', 'YES' if not mismatched else 'NO', mismatched)
            # 6. the negative arm
            report['negative'] = negative_arm(ttnn, engine, drive, entries, caches, layout, payloads, chips, slice_bytes, options, blocks, log)
        finally:
            engine.close()
        gc.collect()
        report['released'] = check_released(engine, options)
        log_line(log, 'windows released: %s; p2pmem available == size: %s', 'yes' if report['released']['links_closed'] else 'NO', 'yes' if report['released']['available_equals_size'] else 'NO')
    finally:
        drive.close()
        try:
            arena.close()
        except BufferError:
            pass
    return report


def describe_engine(engine, report, log, sysfs_pci=p2p.SYSFS_PCI):
    info = engine.describe()
    report['engine'] = dict(info, cards=[dict(position=position, link_width=link_width(bdf, sysfs_pci), window=getattr(link, 'window_id', None), mapping_verified=bool(getattr(link, 'map_verified', False)),
                                              noc=dict(getattr(link, 'aim', {}).get('noc', {})), slice_offset=getattr(link, 'aim', {}).get('slice_offset'))
                                         for position, ((_index, bdf), link) in enumerate(zip(engine.cards, engine.links))])
    log_line(log, 'engine cards=%d threads=%d staging_bytes=%d chunk_blocks=%d shard_bytes=%d mapping_verified=%d', info['cards'], info['threads'], info['staging_bytes'],
             info['chunk_blocks'], info['shard_bytes'], info['mapping_verified'])
    for position, card in enumerate(report['engine']['cards']):
        log_line(log, 'card %d: link x%d window %s noc (%s,%s) staging at slice offset %s mapping_verified=%s', position, card['link_width'], card['window'],
                 card['noc'].get('x'), card['noc'].get('y'), card['slice_offset'], card['mapping_verified'])


def compare_cache(ttnn, caches, layout, payloads, ids, chips, slice_bytes):
    bad = 0
    for tensor, cache in enumerate(caches):
        back = read_back(ttnn, cache, chips, ids, slice_bytes)
        for chip in range(chips):
            want = expected_rows(layout, payloads, tensor, chip, len(ids))
            bad += int((back[chip] != want).any(axis=1).sum())
    return bad


def negative_arm(ttnn, engine, drive, entries, caches, layout, payloads, chips, slice_bytes, options, blocks, log):
    """One byte of one chip's shard of one block flipped on the drive: the checksum must refuse it and the cache block it was for stays zeros; then the byte is put back and it passes."""
    from qwen_p2p_restore import P2PMismatch

    target = blocks // 2
    chip = chips - 1
    offset = entries[target].offset + chip * layout.shard_bytes + 12345
    spare = blocks + 1                                 # a cache block no good restore wrote
    result = dict(block=target, chip=chip)
    old = drive.flip(offset)
    try:
        refused = False
        try:
            engine.restore([entries[target]], [spare])
        except P2PMismatch as error:
            refused = (error.bad == [(0, chip)])
        engine.api.sync()
        untouched = all(not read_back(ttnn, cache, chips, [spare], slice_bytes).any() for cache in caches)
        result.update(refused=refused, nothing_written=untouched)
    finally:
        drive.put_byte(offset, old)
    engine.restore([entries[target]], [spare])
    engine.api.sync()
    again = all(bool((read_back(ttnn, cache, chips, [spare], slice_bytes)[c] == expected_rows(layout, payloads, tensor, c, blocks)[target]).all())
                for tensor, cache in enumerate(caches) for c in range(chips))
    result['passes_after_the_byte_is_back'] = again
    log_line(log, 'negative arm: refused=%s nothing_written=%s passes_after_the_byte_is_back=%s', result.get('refused'), result.get('nothing_written'), again)
    return result


def check_released(engine, options):
    """Step 7: every link closed, and every card's p2pmem pool whole again (available == size)."""
    links = [link for link in engine.links if link is not None]
    pools = []
    for _index, bdf in engine.cards:
        size = p2p._sysfs_text(os.path.join(options.sysfs_pci, bdf, 'p2pmem', 'size'))
        available = p2p._sysfs_text(os.path.join(options.sysfs_pci, bdf, 'p2pmem', 'available'))
        pools.append(size is not None and size == available)
    return dict(links_closed=all(getattr(link, 'closed', False) for link in links), available_equals_size=bool(pools) and all(pools))


def verdict_of(report):
    """(verdict, reasons) of a finished report."""
    reasons = []
    if report.get('refusal'):
        return 'NOT-MEASURED', [report['refusal']]
    if report.get('error'):
        return 'NOT-MEASURED', [report['error']]
    if not (report.get('one_card') or {}).get('verified'):
        reasons.append('the one-card stage did not verify')
    exact = report.get('exact') or {}
    if exact.get('mismatched_rows', 1) != 0:
        reasons.append('the restored cache differs from what was written to the drive (%s rows)' % exact.get('mismatched_rows'))
    negative = report.get('negative') or {}
    if not (negative.get('refused') and negative.get('nothing_written') and negative.get('passes_after_the_byte_is_back')):
        reasons.append('the negative arm: %s' % {key: negative.get(key) for key in ('refused', 'nothing_written', 'passes_after_the_byte_is_back')})
    released = report.get('released') or {}
    if not (released.get('links_closed') and released.get('available_equals_size')):
        reasons.append('the windows or the p2pmem pools were not released: %s' % released)
    if not report.get('restore'):
        reasons.append('no restore was timed')
    if not (report.get('engine') or {}).get('mapping_verified') and not report.get('allow_unverified_map'):
        reasons.append('the p2pmem mappings were not confirmed against the BAR (/proc/self/pagemap needs CAP_SYS_ADMIN)')
    return ('FAIL' if reasons else 'PASS'), reasons


def open_mesh(ttnn, options, report):
    import tp4_mesh

    ttnn.set_fabric_config(getattr(ttnn.FabricConfig, options.fabric))
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(*tp4_mesh.MESH_SHAPE), l1_small_size=24576, trace_region_size=0)
    report['opened'] = True
    return mesh


def required_ops(ttnn):
    needed = ('qwen_block_bytes', 'qwen_read_blocks_raw', 'qwen_write_blocks_raw') + p2p.StagingApi.REQUIRED
    return [name for name in needed if not callable(getattr(ttnn, name, None))]


def run(options, ttnn=None, environ=None, log=print, alloc=alloc_cache, drive_factory=Drive, engine_factory=None, mesh_opener=None):
    environ = os.environ if environ is None else environ
    report = dict(kind=KIND, fabric=options.fabric, blocks=options.blocks, opened=False, serving_contract=environ.get('QWEN_C2_SERVING'),
                  allow_unverified_map=bool(options.allow_unverified_map))
    try:
        report['drive'] = check_drive(options, log)
    except Refusal as refusal:
        report['refusal'] = str(refusal)
        log_line(log, 'REFUSED (nothing was written): %s', refusal)
        return finish(report, log)
    if ttnn is None:
        import tp4_fabric_probe
        import tp4_mesh

        descriptor = environ.get('TT_MESH_GRAPH_DESC_PATH')
        report['descriptor'] = descriptor
        problems = None
        if descriptor and os.path.isfile(descriptor):
            with open(descriptor, encoding='utf-8') as handle:
                problems = tp4_mesh.descriptor_problems(handle.read())
        refusal = tp4_fabric_probe.descriptor_refusal(descriptor, problems)
        if refusal:
            report['refusal'] = 'refused to open the mesh: ' + refusal
            log_line(log, 'REFUSED (nothing was written): %s', report['refusal'])
            return finish(report, log)
        import ttnn
    missing = required_ops(ttnn)
    if missing:
        report['refusal'] = 'this ttnn lacks %s (the device-op extension, or the host tier\'s raw ops)' % ', '.join(missing)
        log_line(log, 'REFUSED (nothing was written): %s', report['refusal'])
        return finish(report, log)
    mesh = None
    try:
        mesh = (mesh_opener or open_mesh)(ttnn, options, report)
        run_data_plane(ttnn, mesh, options, report['drive'], report, log, alloc, drive_factory, engine_factory)
    except Refusal as refusal:
        report['refusal'] = str(refusal)
        log_line(log, 'REFUSED: %s', refusal)
    except Exception as error:  # noqa: BLE001
        scrub = scrubber(options)
        report['error'] = '%s: %s' % (type(error).__name__, scrub(error)[:500])
        report['traceback'] = scrub(traceback.format_exc())[-3000:]
        log_line(log, 'error: %s', report['error'])
    finally:
        if mesh is not None:
            try:
                ttnn.close_mesh_device(mesh)
                report['closed'] = True
            except Exception as error:  # noqa: BLE001
                report['closed'] = '%s: %s' % (type(error).__name__, error)
    return finish(report, log)


def finish(report, log):
    verdict, reasons = verdict_of(report)
    report['verdict'], report['reasons'] = verdict, reasons
    read = report.get('read') or {}
    restore = report.get('restore') or {}
    log_line(log, 'verdict=%s read_gbps=%s restore_gbps=%s check_ms_per_chunk=%s write_ms_per_chunk=%s reasons=%s', verdict,
             '%.2f' % read['gbps'] if read else None, '%.2f' % restore['gbps'] if restore else None,
             '%.2f' % restore['check_ms_per_chunk'] if restore else None, '%.2f' % restore['write_ms_per_chunk'] if restore else None, '; '.join(reasons) or '-')
    return report


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--fabric', choices=('FABRIC_1D', 'FABRIC_1D_RING'), default='FABRIC_1D')
    parser.add_argument('--output', required=True)
    parser.add_argument('--drive', default=os.environ.get('P2P_DRIVE_BY_ID'), help='the drive: its /dev/disk/by-id path (default: P2P_DRIVE_BY_ID)')
    parser.add_argument('--drive-model', default=os.environ.get('P2P_DRIVE_MODEL'), help='the drive\'s model (default: P2P_DRIVE_MODEL; job config, never a tracked file)')
    parser.add_argument('--drive-serial', default=os.environ.get('P2P_DRIVE_SERIAL'), help='the drive\'s serial (default: P2P_DRIVE_SERIAL)')
    parser.add_argument('--cards', default=os.environ.get('P2P_CARDS', ''), help='comma-separated PCI addresses of the cards whose p2pmem is published (default: P2P_CARDS); the mesh must run on exactly these')
    parser.add_argument('--blocks', type=int, default=DEFAULT_BLOCKS, help='KV blocks of the known object (default %d, about 214 MB)' % DEFAULT_BLOCKS)
    parser.add_argument('--caches', type=int, default=32, help='scratch cache tensors (default 32 = the production count; a multiple of 4 so a shard is whole pages)')
    parser.add_argument('--heads-per-chip', type=int, default=1)
    parser.add_argument('--threads', type=int, default=p2p.DEFAULT_THREADS)
    parser.add_argument('--staging-mib', type=int, default=32, help='staging buffer per card (default 32: halves of 29 blocks, so the default object is several chunks and the halves rotate)')
    parser.add_argument('--allow-unverified-map', action='store_true', help='go on although /proc/self/pagemap cannot confirm where the p2pmem mapping is (no CAP_SYS_ADMIN): the DMA then trusts the pool\'s order')
    parser.add_argument('--scratch-offset', type=int, default=DEFAULT_SCRATCH_OFFSET, help='where on the drive the object is written (default 1 GiB)')
    parser.add_argument('--allow-store-overwrite', action='store_true', help='go on although the drive carries the NVMe tier\'s superblock (a gate\'s drive to destroy)')
    parser.add_argument('--by-id-dir', default='/dev/disk/by-id')
    parser.add_argument('--sysfs', default='/sys')
    parser.add_argument('--sysfs-pci', default=p2p.SYSFS_PCI)
    return parser


def parse_options(argv):
    options = build_parser().parse_args(argv)
    options.cards = [name.strip() for name in options.cards.split(',') if name.strip()]
    return options


def main(argv=None, ttnn=None, environ=None, log=print, **kwargs):
    options = parse_options(argv)
    if options.blocks < 8 or options.caches < 4 or options.caches % 4 or options.heads_per_chip < 1 or options.threads < 1 or options.staging_mib < 2:
        print('refusing: blocks >= 8, caches a multiple of 4 and at least 4, heads-per-chip >= 1, threads >= 1, staging-mib >= 2', file=sys.stderr)
        return 2
    timer = threading.Timer(WATCHDOG_S, lambda: (print('P3A watchdog', flush=True), os._exit(3)))
    timer.daemon = True
    timer.start()
    report = run(options, ttnn=ttnn, environ=environ, log=log, **kwargs)
    with open(options.output, 'w') as handle:
        json.dump(report, handle, indent=2, sort_keys=True, default=str)
    log(json.dumps(report, sort_keys=True, default=str))
    timer.cancel()
    return {'PASS': 0, 'FAIL': 1}.get(report.get('verdict'), 2)


if __name__ == '__main__':
    sys.exit(main())
