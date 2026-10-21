"""qwen_nvme_store on a file-backed fake block device: the storage layer of the NVMe cold tier (docs/tp4-kv-nvme.md).

Pure python, stdlib only, no card, no real drive: every device here is a sparse file in a temp directory, with the real store's alignment rules enforced on it
(BlockFile rejects an unaligned access whether or not the file is opened O_DIRECT), so a test that passes would not trip a raw namespace on alignment.

Held:
  * the format policy: a blank device is initialized, a foreign one is never written to without 'force', a store whose superblocks are both bad is corrupt;
  * a round trip through reopen (snapshot + journal replay) returns every entry, byte for byte, in LRU order;
  * TTL (since last use), the age cap, LRU eviction, the free-space floor, max_bytes and the index capacity behave and persist;
  * per-tenant deletion and a whole wipe are durable and cost one journal record;
  * a pinned entry's extent survives a delete or an eviction until it is unpinned;
  * CRASH SAFETY on a simulated disk with a volatile write cache: the process is killed at every device call of a put, a delete, a tenant delete, a wipe and a
    snapshot, with a random subset (and torn pieces) of the writes since the last sync reaching the media; the device then opens, every entry it lists reads back to
    its digest, no extents overlap, and the state is the old one, the new one, or (for a multi-record batch) a prefix of the new one.
"""

import hashlib
import os
import random
import shutil
import sys
import tempfile
import threading
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import qwen_nvme_store as nvme  # noqa: E402
from qwen_nvme_store import ALIGN, CKPT, KV, BlockFile, Corrupt, NotOurs, NvmeStore, PutItem  # noqa: E402

FP = b'fingerpr'
SMALL = 8 * ALIGN                  # an aligned payload
ODD = 5000                         # an unaligned payload (bounces through the scratch)


class Clock(object):
    def __init__(self, now=1000000.0):
        self.now = now

    def __call__(self):
        return self.now


def key_of(name):
    return hashlib.sha256(('k%s' % (name,)).encode()).digest()


def payload_of(name, size=SMALL):
    seed = hashlib.sha256(('p%s' % (name,)).encode()).digest()
    return (seed * (size // len(seed) + 1))[:size]


def item(name, size=SMALL, kind=KV, tenant='alpha', chain=b'chain-0', index=0, fp=FP, extra=0):
    return PutItem(kind, key_of(name), [payload_of(name, size)], tenant, chain, index, fp, extra=extra)


def aligned_item(name, size=SMALL, **kwargs):
    """An item whose buffer is page-aligned (the zero-copy write path)."""
    scratch = nvme.Scratch(size)
    scratch.view[:size] = payload_of(name, size)
    made = PutItem(kwargs.pop('kind', KV), key_of(name), [scratch.view[:size]], kwargs.pop('tenant', 'alpha'), kwargs.pop('chain', b'chain-0'),
                   kwargs.pop('index', 0), kwargs.pop('fp', FP), **kwargs)
    made.keepalive = scratch if False else None
    return made, scratch


class SimDisk(BlockFile):
    """A file-backed device with a volatile write cache: a write is held until sync() and a crash() lets any subset of the held writes (and torn pieces of them,
    at 512-byte sectors) reach the file. Reads see the process's own pending writes, as a drive's cache would serve them."""

    def __init__(self, *args, **kwargs):
        BlockFile.__init__(self, *args, **kwargs)
        self.pending = []
        self.calls = 0
        self.crash_at = None            # the 1-based device call (write or sync) that raises Crash instead of running

    class Crash(BaseException):
        pass

    def _tick(self):
        self.calls += 1
        if self.crash_at is not None and self.calls == self.crash_at:
            raise SimDisk.Crash()

    def pwrite_from(self, offset, views):
        views = [nvme._flat(view) for view in views]
        total = sum(len(view) for view in views)
        self._check(offset, total)
        self._tick()
        self.pending.append((offset, b''.join(bytes(view) for view in views)))

    def sync(self):
        self._tick()
        for offset, data in self.pending:
            os.pwrite(self.fd, data, offset)
        self.pending = []

    def pread_into(self, offset, views):
        views = [nvme._flat(view) for view in views]
        total = sum(len(view) for view in views)
        self._check(offset, total)
        data = bytearray(os.pread(self.fd, total, offset))
        data.extend(bytes(total - len(data)))
        for woffset, wdata in self.pending:
            low, high = max(offset, woffset), min(offset + total, woffset + len(wdata))
            if low < high:
                data[low - offset:high - offset] = wdata[low - woffset:high - woffset]
        position = 0
        for view in views:
            view[:] = data[position:position + len(view)]
            position += len(view)

    def crash(self, rng):
        """The power goes: each pending write reaches the media whole, in 512-byte pieces, or not at all."""
        for offset, data in self.pending:
            mode = rng.random()
            if mode < 0.4:
                os.pwrite(self.fd, data, offset)
            elif mode < 0.8:
                for piece in range(0, len(data), 512):
                    if rng.random() < 0.5:
                        os.pwrite(self.fd, data[piece:piece + 512], offset + piece)
        self.pending = []


class StoreCase(unittest.TestCase):
    def setUp(self):
        # (a memory-backed directory when there is one: the tests model durability themselves, SimDisk, and an fdatasync per put on a busy disk only costs time)
        shm = '/dev/shm' if os.path.isdir('/dev/shm') and os.access('/dev/shm', os.W_OK) else None
        self.directory = tempfile.mkdtemp(prefix='qnvm-test-', dir=shm)
        self.addCleanup(shutil.rmtree, self.directory, True)
        self.path = os.path.join(self.directory, 'dev.img')
        self.clock = Clock()
        self.opened = []

    def open(self, size=8 << 20, **kwargs):
        kwargs.setdefault('clock', self.clock)
        kwargs.setdefault('snap_bytes', 64 * ALIGN)
        kwargs.setdefault('journal_bytes', 64 * ALIGN)
        kwargs.setdefault('threads', 2)
        store = NvmeStore.open(self.path, size=size, **kwargs)
        self.opened.append(store)
        return store

    def tearDown(self):
        for store in self.opened:
            try:
                store.close()
            except Exception:
                pass

    def reopen(self, store, **kwargs):
        store.close()
        return self.open(**kwargs)

    def put(self, store, *names, **kwargs):
        result = store.put_batch([item(name, **kwargs) for name in names])
        self.assertTrue(all(result), result)
        return result

    def read(self, store, name, kind=KV):
        entry = store.lookup(kind, key_of(name))
        if entry is None:
            return None
        out = bytearray(entry.nbytes)
        pinned = store.pin([(kind, key_of(name))])[0]
        try:
            result = store.read_many([pinned], [memoryview(out)])
        finally:
            store.unpin([pinned])
        self.assertEqual(result, [None])
        return bytes(out)


class AlignmentTests(StoreCase):
    def test_unaligned_device_access_is_refused_on_a_plain_file_too(self):
        bf = BlockFile(self.path, size=1 << 20, create=True)
        self.addCleanup(bf.close)
        buffer = nvme.Scratch(2 * ALIGN)
        for offset, length in ((1, ALIGN), (ALIGN, 100), (0, ALIGN + 1)):
            with self.assertRaises(ValueError):
                bf.pread_into(offset, [buffer.view[:length]]) if length % 2 == 0 or True else None
        with self.assertRaises(ValueError):
            bf.pwrite_from(0, [b'x' * 100])
        with self.assertRaises(ValueError):
            bf.pread_into(1 << 20, [buffer.view[:ALIGN]])

    def test_a_logical_read_and_write_of_any_length_bounces_through_aligned_scratch(self):
        bf = BlockFile(self.path, size=1 << 20, create=True)
        self.addCleanup(bf.close)
        data = os.urandom(ODD)
        written = bf.write_buffers(ALIGN, [data[:1000], data[1000:]])
        self.assertEqual(written, 2 * ALIGN)
        out = bytearray(ODD)
        bf.read_run(ALIGN, [ODD], [memoryview(out)])
        self.assertEqual(bytes(out), data)
        # two entries back to back: the second starts at the next block
        second = os.urandom(SMALL)
        bf.write_buffers(ALIGN + 2 * ALIGN, [second])
        out2 = bytearray(SMALL)
        both = [bytearray(ODD), out2]
        bf.read_run(ALIGN, [ODD, SMALL], [memoryview(both[0]), memoryview(both[1])])
        self.assertEqual((bytes(both[0]), bytes(both[1])), (data, second))

    def test_the_zero_copy_path_takes_an_aligned_buffer_as_it_is(self):
        bf = BlockFile(self.path, size=1 << 20, create=True)
        self.addCleanup(bf.close)
        scratch = nvme.Scratch(SMALL)
        scratch.view[:SMALL] = payload_of('z')
        self.assertTrue(nvme.is_aligned(scratch.view[:SMALL]))
        self.assertFalse(nvme.is_aligned(scratch.view[1:1 + SMALL - 1]))
        bf.write_buffers(0, [scratch.view[:SMALL]])
        back = nvme.Scratch(SMALL)
        bf.read_run(0, [SMALL], [back.view[:SMALL]])
        self.assertEqual(bytes(back.view[:SMALL]), payload_of('z'))


class AllocatorTests(unittest.TestCase):
    def test_first_fit_and_coalescing(self):
        allocator = nvme.Allocator(0, 100 * ALIGN)
        a = allocator.alloc(10 * ALIGN)
        b = allocator.alloc(10 * ALIGN)
        c = allocator.alloc(10 * ALIGN)
        self.assertEqual((a, b, c), (0, 10 * ALIGN, 20 * ALIGN))
        allocator.release(b, 10 * ALIGN)
        self.assertEqual(allocator.fragments(), 2)
        self.assertEqual(allocator.alloc(5 * ALIGN), b)                    # the hole is used first
        allocator.release(a, 10 * ALIGN)
        allocator.release(b, 5 * ALIGN)
        allocator.release(c, 10 * ALIGN)
        self.assertEqual(allocator.fragments(), 1, 'everything released: one extent again')
        self.assertEqual(allocator.free, 100 * ALIGN)
        self.assertEqual(allocator.starts, [0])

    def test_from_used_leaves_the_gaps(self):
        allocator = nvme.Allocator.from_used(0, 100 * ALIGN, [(10 * ALIGN, 5 * ALIGN), (20 * ALIGN, 10 * ALIGN)])
        self.assertEqual(allocator.starts, [0, 15 * ALIGN, 30 * ALIGN])
        self.assertEqual(allocator.lengths, [10 * ALIGN, 5 * ALIGN, 70 * ALIGN])
        self.assertIsNone(allocator.alloc(200 * ALIGN))
        self.assertEqual(allocator.alloc(70 * ALIGN), 30 * ALIGN)


class FormatTests(StoreCase):
    def test_a_blank_device_is_initialized_and_reopens(self):
        store = self.open()
        self.put(store, 'a', 'b')
        again = self.reopen(store)
        self.assertEqual(self.read(again, 'a'), payload_of('a'))
        self.assertEqual(again.snapshot_stats()['entries'], 2)

    def test_a_foreign_device_is_never_written_without_force(self):
        with open(self.path, 'wb') as handle:
            handle.write(b'EXT4-or-whatever' * 1000)
            handle.truncate(8 << 20)
        before = Path(self.path).read_bytes()
        with self.assertRaises(NotOurs):
            self.open()
        self.assertEqual(Path(self.path).read_bytes(), before, 'refusing must not write a byte')
        store = self.open(format='force')
        self.put(store, 'a')
        self.assertEqual(self.read(store, 'a'), payload_of('a'))

    def test_format_never_refuses_a_blank_device(self):
        with self.assertRaises(NotOurs):
            self.open(format='never')

    def test_a_store_whose_superblocks_are_both_bad_is_corrupt_not_blank(self):
        store = self.open()
        self.put(store, 'a')
        store.close()
        with open(self.path, 'r+b') as handle:
            for slot in (0, 1):
                handle.seek(slot * ALIGN + 100)
                handle.write(b'\xff\xff\xff\xff')
        with self.assertRaises(Corrupt):
            self.open()
        store = self.open(format='force')
        self.assertEqual(store.snapshot_stats()['entries'], 0)

    def test_the_newer_superblock_wins_and_a_torn_newer_one_falls_back(self):
        store = self.open()
        self.put(store, 'a')
        store.snapshot()
        generation = store.generation
        store.close()
        with open(self.path, 'r+b') as handle:                  # tear the newest generation's slot
            handle.seek((generation % 2) * ALIGN + 20)
            handle.write(b'\x00\x01\x02\x03')
        store = self.open()
        self.assertLess(store.generation, generation + 1)
        self.assertEqual(self.read(store, 'a'), payload_of('a'))


class LockTests(StoreCase):
    def test_a_second_process_cannot_open_a_store_that_is_open(self):
        store = self.open()
        with self.assertRaises(nvme.StoreError):
            NvmeStore.open(self.path, clock=self.clock)
        store.close()
        self.open().close()


class RoundTripTests(StoreCase):
    def test_aligned_and_odd_payloads_round_trip_through_reopen(self):
        store = self.open()
        items = [item('s%d' % n) for n in range(5)] + [item('o%d' % n, size=ODD) for n in range(5)]
        self.assertTrue(all(store.put_batch(items)))
        for name, size in [('s%d' % n, SMALL) for n in range(5)] + [('o%d' % n, ODD) for n in range(5)]:
            self.assertEqual(self.read(store, name), payload_of(name, size))
        again = self.reopen(store)
        for name, size in [('s%d' % n, SMALL) for n in range(5)] + [('o%d' % n, ODD) for n in range(5)]:
            self.assertEqual(self.read(again, name), payload_of(name, size))
        self.assertEqual(list(again.entries), [(KV, key_of(n)) for n in ['s0', 's1', 's2', 's3', 's4', 'o0', 'o1', 'o2', 'o3', 'o4']], 'LRU order survives')

    def test_the_same_key_is_one_entry_and_a_duplicate_only_refreshes_it(self):
        store = self.open()
        self.put(store, 'a', 'b')
        used = store.logical_bytes
        self.clock.now += 100
        self.assertEqual(store.put_batch([item('a')]), [True])
        self.assertEqual(store.logical_bytes, used)
        self.assertEqual(store.stats['put_duplicates'], 1)
        self.assertEqual(list(store.entries)[-1], (KV, key_of('a')), 'a duplicate is a use')

    def test_kv_and_checkpoint_entries_of_one_hash_are_two_entries(self):
        store = self.open()
        self.assertTrue(all(store.put_batch([item('same'), item('same', kind=CKPT, index=2048, extra=7)])))
        self.assertTrue(store.has(KV, key_of('same')) and store.has(CKPT, key_of('same')))
        self.assertTrue(store.chain_reaches(b'chain-0', 2048))
        self.assertFalse(store.chain_reaches(b'chain-0', 2049))
        self.assertFalse(store.chain_reaches(b'chain-1', 0))
        again = self.reopen(store)
        self.assertEqual(again.lookup(CKPT, key_of('same')).extra, 7)
        self.assertTrue(again.chain_reaches(b'chain-0', 2048))

    def test_the_digest_is_computed_and_kept(self):
        store = self.open()
        self.put(store, 'a')
        self.assertEqual(store.lookup(KV, key_of('a')).digest, hashlib.sha256(payload_of('a')).digest())
        again = self.reopen(store)
        self.assertEqual(again.lookup(KV, key_of('a')).digest, hashlib.sha256(payload_of('a')).digest())

    def test_read_many_runs_are_merged_and_a_verify_failure_is_reported_per_entry(self):
        store = self.open()
        self.put(store, *['b%d' % n for n in range(6)])
        pairs = [(KV, key_of('b%d' % n)) for n in range(6)]
        entries = store.pin(pairs)
        calls = []
        original = store.bf.read_run

        def counting(offset, lengths, views):
            calls.append(len(lengths))
            return original(offset, lengths, views)

        store.bf.read_run = counting
        outs = [bytearray(SMALL) for _ in entries]

        def verify(entry, view):
            return entry.key != key_of('b3')

        results = store.read_many(entries, [memoryview(out) for out in outs], verify=verify)
        store.unpin(entries)
        self.assertEqual(results, [None, None, None, 'digest', None, None])
        self.assertEqual(calls, [6], 'six adjacent blocks are one device call')
        self.assertEqual(bytes(outs[0]), payload_of('b0'))

    def test_a_device_read_error_is_an_io_result_not_an_exception(self):
        store = self.open()
        self.put(store, 'a')
        entries = store.pin([(KV, key_of('a'))])
        store.bf.fail = lambda what, offset, length: (_ for _ in ()).throw(IOError('dead')) if what == 'read' else None
        out = bytearray(SMALL)
        self.assertEqual(store.read_many(entries, [memoryview(out)]), ['io'])
        store.unpin(entries)


class ExpiryTests(StoreCase):
    def test_ttl_counts_from_the_last_use_and_a_touch_restarts_it(self):
        store = self.open(ttl_s=1000)
        self.put(store, 'a', 'b')
        self.clock.now += 600
        store.touch(KV, key_of('b'))
        self.clock.now += 600
        self.assertIsNone(store.lookup(KV, key_of('a')), 'a: 1200 s since written')
        self.assertIsNotNone(store.lookup(KV, key_of('b')), 'b: 600 s since used')
        self.assertEqual(store.sweep(), 1)
        self.assertEqual(store.stats['expired_ttl'], 1)
        self.assertEqual(sorted(store.entries), [(KV, key_of('b'))])
        used = store.logical_bytes
        self.assertEqual(used, SMALL)

    def test_an_expired_entry_stays_gone_after_a_restart_and_its_space_is_free(self):
        store = self.open(ttl_s=1000)
        self.put(store, 'a', 'b')
        free_before = store.allocator.free
        self.clock.now += 2000
        store.sweep()
        self.assertEqual(store.allocator.free, free_before + 2 * SMALL)
        again = self.reopen(store, ttl_s=1000)
        self.assertEqual(again.snapshot_stats()['entries'], 0)
        self.assertEqual(again.allocator.free, free_before + 2 * SMALL)

    def test_the_age_cap_counts_from_the_write_whatever_the_use(self):
        store = self.open(ttl_s=10 ** 9, max_age_s=1000)
        self.put(store, 'a')
        self.clock.now += 800
        store.touch(KV, key_of('a'))
        self.clock.now += 400
        self.assertIsNone(store.lookup(KV, key_of('a')))
        self.assertEqual(store.sweep(), 1)
        self.assertEqual(store.stats['expired_age'], 1)

    def test_an_expired_key_can_be_stored_again(self):
        store = self.open(ttl_s=1000)
        self.put(store, 'a')
        self.clock.now += 2000
        self.assertEqual(store.put_batch([item('a')]), [True])
        self.assertIsNotNone(store.lookup(KV, key_of('a')))
        self.assertEqual(store.snapshot_stats()['entries'], 1)

    def test_a_pinned_expired_entry_is_not_swept_from_under_its_reader(self):
        store = self.open(ttl_s=1000)
        self.put(store, 'a')
        entries = store.pin([(KV, key_of('a'))])
        self.clock.now += 2000
        self.assertEqual(store.sweep(), 0)
        store.unpin(entries)
        self.assertEqual(store.sweep(), 1)

    def test_touches_are_journaled_so_a_restart_keeps_a_session_alive(self):
        store = self.open(ttl_s=1000)
        self.put(store, 'a')
        self.clock.now += 900
        store.touch(KV, key_of('a'))
        self.clock.now += 100
        store.sweep()                                               # journals the touch (over TOUCH_JOURNAL_S)
        again = self.reopen(store, ttl_s=1000)
        self.clock.now += 500
        self.assertIsNotNone(again.lookup(KV, key_of('a')), 'last used 600 s ago, not 1500')


class CapacityTests(StoreCase):
    def test_the_least_recently_used_is_evicted_to_make_room(self):
        store = self.open(size=1 << 20, snap_bytes=8 * ALIGN, journal_bytes=8 * ALIGN)
        capacity = store.capacity_bytes() // SMALL
        names = ['e%d' % n for n in range(capacity + 3)]
        for name in names:
            self.put(store, name)
            self.clock.now += 1
        self.assertIsNone(store.lookup(KV, key_of(names[0])))
        self.assertIsNone(store.lookup(KV, key_of(names[2])))
        self.assertIsNotNone(store.lookup(KV, key_of(names[-1])))
        self.assertEqual(store.stats['evicted_lru'], 3)
        self.assertLessEqual(store.logical_bytes, store.capacity_bytes())

    def test_a_touch_protects_an_entry_from_eviction(self):
        store = self.open(size=1 << 20, snap_bytes=8 * ALIGN, journal_bytes=8 * ALIGN)
        capacity = store.capacity_bytes() // SMALL
        for n in range(capacity):
            self.put(store, 'e%d' % n)
        store.touch(KV, key_of('e0'))
        self.put(store, 'new')
        self.assertIsNotNone(store.lookup(KV, key_of('e0')))
        self.assertIsNone(store.lookup(KV, key_of('e1')))

    def test_max_bytes_and_the_free_floor_bound_the_logical_size(self):
        store = self.open(max_bytes=10 * SMALL)
        for n in range(15):
            self.put(store, 'm%d' % n)
        self.assertEqual(store.logical_bytes, 10 * SMALL)
        floor = store.geometry.data_bytes - 6 * SMALL
        store.close()
        os.remove(self.path)
        store = self.open(free_floor_bytes=floor)
        for n in range(15):
            self.put(store, 'f%d' % n)
        self.assertEqual(store.logical_bytes, 6 * SMALL)
        self.assertGreaterEqual(store.allocator.free, floor)

    def test_an_object_larger_than_the_tier_is_refused_and_nothing_else_is_disturbed(self):
        store = self.open(max_bytes=4 * SMALL)
        self.put(store, 'a')
        result = store.put_batch([item('big', size=8 * SMALL)])
        self.assertEqual(result, [False])
        self.assertEqual(store.stats['put_refused'], 1)
        self.assertIsNotNone(store.lookup(KV, key_of('a')))

    def test_the_index_capacity_bounds_the_entry_count(self):
        store = self.open(size=2 << 20, snap_bytes=2 * ALIGN, journal_bytes=16 * ALIGN)
        limit = store.max_entries()
        self.assertEqual(limit, 20)
        for n in range(limit + 5):
            self.put(store, 'x%d' % n, size=ALIGN)
        self.assertLessEqual(store.snapshot_stats()['entries'], limit)
        self.assertGreaterEqual(store.stats['evicted_lru'], 5)
        again = self.reopen(store, snap_bytes=2 * ALIGN, journal_bytes=16 * ALIGN)           # and a snapshot of a full index fits its slot
        self.assertLessEqual(again.snapshot_stats()['entries'], limit)

    def test_one_batch_that_does_not_fit_in_a_run_is_placed_piecemeal(self):
        store = self.open(size=1 << 20, snap_bytes=8 * ALIGN, journal_bytes=8 * ALIGN)
        capacity = store.capacity_bytes() // SMALL
        for n in range(capacity):
            self.put(store, 'h%d' % n)
        # free every other block: the free space is fragmented into one-block holes
        store.delete([(KV, key_of('h%d' % n)) for n in range(0, capacity, 2)])
        holes = store.allocator.fragments()
        self.assertGreater(holes, 2)
        want = ['p%d' % n for n in range(min(holes, 4))]
        self.assertTrue(all(store.put_batch([item(name) for name in want])))
        for name in want:
            self.assertEqual(self.read(store, name), payload_of(name))
        again = self.reopen(store, snap_bytes=8 * ALIGN, journal_bytes=8 * ALIGN)
        for name in want:
            self.assertEqual(self.read(again, name), payload_of(name))


class DeletionTests(StoreCase):
    def test_a_tenant_is_deleted_with_one_record_and_stays_deleted(self):
        store = self.open()
        store.put_batch([item('a%d' % n, tenant='alpha') for n in range(5)] + [item('b%d' % n, tenant='bravo') for n in range(3)])
        records = store.stats['journal_records']
        self.assertEqual(store.delete_tenant('alpha'), 5)
        self.assertEqual(store.stats['journal_records'] - records, 1)
        self.assertEqual(store.tenant_entries('alpha'), 0)
        self.assertEqual(store.tenant_entries('bravo'), 3)
        again = self.reopen(store)
        self.assertEqual(again.tenant_entries('alpha'), 0)
        self.assertEqual(again.tenant_entries('bravo'), 3)
        self.assertEqual(self.read(again, 'b0'), payload_of('b0'))
        self.assertIsNone(again.lookup(KV, key_of('a0')))

    def test_a_put_after_a_tenant_delete_is_kept_by_the_replay(self):
        store = self.open()
        self.put(store, 'a0')
        store.delete_tenant('alpha')
        self.put(store, 'a1')
        again = self.reopen(store)
        self.assertIsNone(again.lookup(KV, key_of('a0')))
        self.assertIsNotNone(again.lookup(KV, key_of('a1')))

    def test_the_tenant_tag_never_reaches_the_device(self):
        store = self.open()
        self.put(store, 'a', tenant='tenant-tag-secret-123')
        store.close()
        self.assertNotIn(b'tenant-tag-secret-123', Path(self.path).read_bytes())

    def test_wipe_removes_everything_durably_and_frees_the_space(self):
        store = self.open()
        self.put(store, *['w%d' % n for n in range(6)])
        free = store.allocator.free
        self.assertEqual(store.wipe(), 6)
        self.assertEqual(store.allocator.free, free + 6 * SMALL)
        again = self.reopen(store)
        self.assertEqual(again.snapshot_stats()['entries'], 0)
        self.put(again, 'after')
        self.assertEqual(self.read(again, 'after'), payload_of('after'))

    def test_drop_where_removes_the_entries_of_another_layout(self):
        store = self.open()
        store.put_batch([item('old1', fp=b'oldlayou'), item('old2', fp=b'oldlayou'), item('new1')])
        self.assertEqual(store.drop_where(lambda entry: entry.fp != FP), 2)
        self.assertEqual(store.stats['fp_drops'], 2)
        again = self.reopen(store)
        self.assertEqual(sorted(entry.key for entry in again.entries.values()), [key_of('new1')])

    def test_discard_is_asked_for_freed_extents_when_enabled(self):
        calls = []

        class Discarding(BlockFile):
            def discard(self, offset, length):
                calls.append((offset, length))
                return True

        store = NvmeStore.open(self.path, size=4 << 20, snap_bytes=16 * ALIGN, journal_bytes=16 * ALIGN, file_factory=Discarding, clock=self.clock, discard=True)
        self.opened.append(store)
        self.put(store, 'a')
        store.delete([(KV, key_of('a'))])
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][1], SMALL)
        self.assertEqual(store.stats['discarded_bytes'], SMALL)


class PinTests(StoreCase):
    def test_a_pinned_extent_is_not_reused_until_the_reader_lets_go(self):
        store = self.open(size=1 << 20, snap_bytes=8 * ALIGN, journal_bytes=8 * ALIGN)
        capacity = store.capacity_bytes() // SMALL
        for n in range(capacity):
            self.put(store, 'q%d' % n)
        entries = store.pin([(KV, key_of('q0'))])
        store.delete([(KV, key_of('q0'))])
        self.assertIsNone(store.lookup(KV, key_of('q0')), 'deleted for every new reader')
        self.assertEqual(store.allocator.free, store.geometry.data_bytes - capacity * SMALL, 'the extent is still held')
        self.put(store, 'filler')                                    # evicts others, not the pinned extent
        out = bytearray(SMALL)
        self.assertEqual(store.read_many(entries, [memoryview(out)]), [None])
        self.assertEqual(bytes(out), payload_of('q0'), 'the bytes the reader pinned are still there')
        free_before = store.allocator.free
        store.unpin(entries)
        self.assertEqual(store.allocator.free, free_before + SMALL)

    def test_eviction_skips_pinned_entries(self):
        store = self.open(size=1 << 20, snap_bytes=8 * ALIGN, journal_bytes=8 * ALIGN)
        capacity = store.capacity_bytes() // SMALL
        for n in range(capacity):
            self.put(store, 'r%d' % n)
        pinned = store.pin([(KV, key_of('r0'))])
        self.put(store, 'new')
        self.assertIsNotNone(store.lookup(KV, key_of('r0')))
        self.assertIsNone(store.lookup(KV, key_of('r1')))
        store.unpin(pinned)

    def test_when_everything_is_pinned_a_put_is_refused_not_forced(self):
        store = self.open(size=1 << 20, snap_bytes=8 * ALIGN, journal_bytes=8 * ALIGN)
        capacity = store.capacity_bytes() // SMALL
        for n in range(capacity):
            self.put(store, 's%d' % n)
        pinned = store.pin([(KV, key_of('s%d' % n)) for n in range(capacity)])
        self.assertEqual(store.put_batch([item('late')]), [False])
        store.unpin(pinned)
        self.assertEqual(store.put_batch([item('late')]), [True])


class JournalTests(StoreCase):
    def test_a_full_journal_snapshots_and_the_state_survives_many_cycles(self):
        store = self.open(size=4 << 20, snap_bytes=32 * ALIGN, journal_bytes=8 * ALIGN)
        rng = random.Random(7)
        model = {}
        for step in range(120):
            name = 'j%d' % rng.randrange(40)
            if rng.random() < 0.35 and name in model:
                store.delete([(KV, key_of(name))])
                del model[name]
            else:
                if name not in model:
                    self.assertTrue(all(store.put_batch([item(name, size=rng.choice([ALIGN, 2 * ALIGN, ODD]))])))
                    model[name] = True
            if step % 40 == 39:
                store = self.reopen(store, size=4 << 20, snap_bytes=32 * ALIGN, journal_bytes=8 * ALIGN)
        self.assertGreater(store.stats['snapshots'] + store.epoch, 3)
        self.assertEqual(sorted(entry.key for entry in store.entries.values()), sorted(key_of(name) for name in model))
        for name in model:
            self.assertEqual(hashlib.sha256(self.read(store, name)).digest(), store.lookup(KV, key_of(name)).digest)

    def test_stale_records_of_an_older_epoch_are_never_replayed(self):
        store = self.open(size=4 << 20, snap_bytes=32 * ALIGN, journal_bytes=8 * ALIGN)
        self.put(store, 'a', 'b', 'c')
        store.snapshot()                                             # the old records stay in the journal, under the old epoch
        store.delete([(KV, key_of('a'))])
        again = self.reopen(store, size=4 << 20, snap_bytes=32 * ALIGN, journal_bytes=8 * ALIGN)
        self.assertEqual(sorted(entry.key for entry in again.entries.values()), sorted([key_of('b'), key_of('c')]))


class CrashTests(StoreCase):
    """The device loses power at every device call of an operation; a random part of the unsynced writes reaches the media. The reopened store must be sound."""

    GEOMETRY = dict(size=2 << 20, snap_bytes=16 * ALIGN, journal_bytes=16 * ALIGN)

    def make(self, crash_at=None):
        store = NvmeStore.open(self.path, file_factory=SimDisk, clock=self.clock, threads=1, **self.GEOMETRY)
        self.opened.append(store)
        store.bf.crash_at = crash_at
        store.bf.calls = 0
        return store

    def sound(self, store):
        """The invariants of any reopened store; -> the set of keys."""
        entries = sorted(store.entries.values(), key=lambda entry: entry.offset)
        end = store.geometry.data_off
        for entry in entries:
            self.assertGreaterEqual(entry.offset, end, 'extents overlap')
            end = entry.offset + entry.alloc
            self.assertLessEqual(end, store.geometry.data_off + store.geometry.data_bytes)
            out = bytearray(entry.nbytes)
            store.bf.read_run(entry.offset, [entry.nbytes], [memoryview(out)])
            self.assertEqual(hashlib.sha256(bytes(out)).digest(), entry.digest, 'an indexed entry whose bytes are not the bytes it was stored with')
        free = store.allocator.free
        self.assertEqual(free + sum(entry.alloc for entry in entries), store.geometry.data_bytes)
        return set(entry.key for entry in entries)

    def run_crashes(self, setup, operation, allowed, seeds=4, label=''):
        """Run setup on a fresh device, then operation with a crash at call 1, 2, ... until the operation completes; after each crash reopen and check `allowed`
        (a function of the reopened store's key set and the setup's key set)."""
        # first, a dry run to count the operation's device calls
        os.path.exists(self.path) and os.remove(self.path)
        store = self.make()
        before = setup(store)
        base_calls = store.bf.calls
        operation(store)
        total = store.bf.calls - base_calls
        self.assertGreater(total, 0)
        store.close()
        checked = 0
        for crash_call in range(1, total + 1):
            for seed in range(seeds):
                os.path.exists(self.path) and os.remove(self.path)
                store = self.make()
                before = setup(store)
                store.bf.pending = []                                   # setup ended synced
                store.bf.crash_at = store.bf.calls + crash_call
                with self.assertRaises(SimDisk.Crash):
                    operation(store)
                store.bf.crash(random.Random(1000 * crash_call + seed))
                os.close(store.bf.fd)
                store._closed = True
                reopened = NvmeStore.open(self.path, clock=self.clock, threads=1, **self.GEOMETRY)
                self.opened.append(reopened)
                keys = self.sound(reopened)
                allowed(keys, before, '%s crash at call %d of %d, seed %d' % (label, crash_call, total, seed))
                checked += 1
        self.assertGreaterEqual(checked, total)

    def test_a_crash_in_the_middle_of_a_put_leaves_the_old_state_or_the_new(self):
        new = ['n%d' % n for n in range(6)]

        def setup(store):
            store.put_batch([item('old%d' % n) for n in range(3)])
            return set(key_of('old%d' % n) for n in range(3))

        def operation(store):
            store.put_batch([item(name) for name in new])

        def allowed(keys, before, why):
            added = keys - before
            self.assertTrue(before <= keys, why + ': an old entry vanished')
            self.assertIn(added, (set(), set(key_of(name) for name in new)), why + ': a half-stored batch (one journal record carries it)')

        self.run_crashes(setup, operation, allowed, label='put')

    def test_a_crash_in_a_multi_record_put_leaves_a_prefix_of_the_batch(self):
        names = ['m%03d' % n for n in range(nvme.PUT_RECORD_ENTRIES * 2 + 5)]

        def setup(store):
            return set()

        def operation(store):
            store.put_batch([item(name, size=ALIGN, index=n) for n, name in enumerate(names)])

        def allowed(keys, before, why):
            present = [key_of(name) in keys for name in names]
            count = sum(present)
            self.assertEqual(present, [True] * count + [False] * (len(names) - count), why + ': not a prefix of the batch')
            self.assertIn(count, (0, nvme.PUT_RECORD_ENTRIES, 2 * nvme.PUT_RECORD_ENTRIES, len(names)), why + ': a record is whole or absent')

        self.run_crashes(setup, operation, allowed, seeds=2, label='multi')

    def test_a_crash_in_a_delete_never_leaves_an_entry_with_overwritten_bytes(self):
        def setup(store):
            store.put_batch([item('d%d' % n) for n in range(4)])
            return set(key_of('d%d' % n) for n in range(4))

        def operation(store):
            store.delete([(KV, key_of('d1')), (KV, key_of('d2'))])
            store.put_batch([item('reuse%d' % n) for n in range(2)])          # the freed extents are reused at once

        def allowed(keys, before, why):
            self.assertTrue({key_of('d0'), key_of('d3')} <= keys, why)
            deleted = {key_of('d1'), key_of('d2')} - keys
            self.assertIn(deleted, (set(), {key_of('d1'), key_of('d2')}), why + ': a half-done delete')
            if keys & {key_of('reuse0'), key_of('reuse1')}:
                self.assertEqual(deleted, {key_of('d1'), key_of('d2')}, why + ': a new entry in space whose old owner was not durably deleted')

        self.run_crashes(setup, operation, allowed, label='delete')

    def test_a_crash_in_a_tenant_delete_or_a_wipe_is_all_or_nothing(self):
        def setup(store):
            store.put_batch([item('t%d' % n, tenant='alpha') for n in range(3)] + [item('u%d' % n, tenant='bravo') for n in range(3)])
            return set(key_of('t%d' % n) for n in range(3)) | set(key_of('u%d' % n) for n in range(3))

        def operation(store):
            store.delete_tenant('alpha')

        def allowed(keys, before, why):
            alpha = set(key_of('t%d' % n) for n in range(3))
            self.assertIn(keys & alpha, (set(), alpha), why)
            self.assertTrue(set(key_of('u%d' % n) for n in range(3)) <= keys, why)

        self.run_crashes(setup, operation, allowed, label='tenant')

        def wipe(store):
            store.wipe()

        def allowed_wipe(keys, before, why):
            self.assertIn(keys, (set(), before), why)

        self.run_crashes(setup, wipe, allowed_wipe, label='wipe')

    def test_a_crash_in_a_snapshot_keeps_the_old_pair_or_takes_the_new(self):
        def setup(store):
            store.put_batch([item('s%d' % n) for n in range(4)])
            return set(key_of('s%d' % n) for n in range(4))

        def operation(store):
            store.snapshot()

        def allowed(keys, before, why):
            self.assertEqual(keys, before, why)

        self.run_crashes(setup, operation, allowed, label='snapshot')

    def test_a_crash_while_the_journal_wraps_into_a_snapshot_loses_nothing_that_was_synced(self):
        names = ['w%03d' % n for n in range(40)]

        def setup(store):
            return set()

        def operation(store):
            for name in names:
                store.put_batch([item(name, size=ALIGN)])

        def allowed(keys, before, why):
            present = [key_of(name) in keys for name in names]
            count = sum(present)
            self.assertEqual(present, [True] * count + [False] * (len(names) - count), why + ': a later put survived an earlier one that did not')

        self.run_crashes(setup, operation, allowed, seeds=2, label='wrap')


class ConcurrencyTests(StoreCase):
    def test_lookups_and_pins_run_while_a_writer_stores_and_evicts(self):
        store = self.open(size=2 << 20, snap_bytes=16 * ALIGN, journal_bytes=16 * ALIGN, threads=2)
        self.put(store, 'steady')
        stop = threading.Event()
        errors = []

        def reader():
            try:
                while not stop.is_set():
                    entry = store.lookup(KV, key_of('steady'))
                    if entry is not None:
                        out = bytearray(entry.nbytes)
                        pinned = store.pin([(KV, key_of('steady'))])[0]
                        if pinned is not None:
                            try:
                                got = store.read_many([pinned], [memoryview(out)])
                            finally:
                                store.unpin([pinned])
                            if got == [None] and bytes(out) != payload_of('steady'):
                                errors.append('wrong bytes')
                    store.touch(KV, key_of('steady'))
                    store.chain_reaches(b'chain-0', 1)
            except Exception as error:  # noqa: BLE001
                errors.append(repr(error))

        thread = threading.Thread(target=reader)
        thread.start()
        try:
            for n in range(80):
                store.put_batch([item('c%d' % n), item('d%d' % n)])
        finally:
            stop.set()
            thread.join()
        self.assertEqual(errors, [])
        self.assertLessEqual(store.logical_bytes, store.capacity_bytes())


class BenchTests(StoreCase):
    def test_the_bench_writes_reads_and_reports_on_a_file(self):
        import nvme_store_bench

        lines = []
        result = nvme_store_bench.run(self.path, gib=0.004, block_bytes=8 * ALIGN, batch_blocks=16, threads=2, verify='all', out=lines.append)
        self.assertGreater(result['blocks'], 100)
        self.assertEqual(result['read_failures'], 0)
        self.assertGreater(result['write_gbps'], 0)
        self.assertGreater(result['read_sequential_gbps'], 0)
        self.assertEqual(len(lines), 3)
        self.assertEqual(result['puts'], result['blocks'])

    def test_a_block_device_is_refused_without_both_flags(self):
        import nvme_store_bench
        from unittest import mock

        with open(self.path, 'wb') as handle:
            handle.truncate(1 << 20)
        with mock.patch('stat.S_ISBLK', return_value=True):
            with self.assertRaises(SystemExit):
                nvme_store_bench.main([self.path])
            with self.assertRaises(SystemExit):
                nvme_store_bench.main([self.path, '--format-force'])


if __name__ == '__main__':
    unittest.main()
