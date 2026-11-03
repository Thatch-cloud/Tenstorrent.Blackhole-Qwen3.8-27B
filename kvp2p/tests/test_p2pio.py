"""P2PIO over fake cards (``fake_card``) and a real file as the drive (O_DIRECT where the file system allows it): both directions byte for byte, the iovec
order, short-I/O loops, restore priority, re-aim only when idle, the aim epochs, the hard bounds and the mutations of them, the DMA canary, and shutdown.
No vLLM, no numpy, no device: stdlib only."""

import ast
import errno
import inspect
import logging
import os
import random
import subprocess
import sys
import threading
import time
import types
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
PACKAGE_ROOT = os.environ.get('TTKVP2P_PACKAGE_ROOT') or os.path.dirname(HERE)
for entry in (HERE, PACKAGE_ROOT):
    if entry not in sys.path:
        sys.path.insert(0, entry)

import fake_card as fc  # noqa: E402
from ttkvp2p import cardlink as cl  # noqa: E402
from ttkvp2p import p2pio as pio  # noqa: E402

logging.getLogger('ttkvp2p').addHandler(logging.NullHandler())      # (the layer logs its errors; the tests that care use assertLogs)

PAGE = 4096
SHARD = fc.SMALL_SHARD
SLOT = fc.SMALL_SLOT


class IOCase(unittest.TestCase):
    def env(self, **kwargs):
        env = fc.Env(**kwargs)
        self.addCleanup(env.close)
        return env

    def settle(self, seconds=0.08):
        time.sleep(seconds)


def prime_drive(env, slots, tag_of):
    """Write ``tag_of(slot)``'s object into each given drive slot."""
    for slot in slots:
        env.drive_write(env.drive_off(slot), env.drive_object(tag_of(slot)))


class RoundTripTests(IOCase):
    def test_a_spill_puts_each_cards_shard_in_order_on_the_drive(self):
        e = self.env()
        self.assertEqual(e.direct, pio.fd_is_direct(e.fd), 'the fixture reports honestly whether O_DIRECT is in use (the file system may refuse it)')
        n = 3
        for j in range(n):
            for d in range(4):
                e.put(d, 0, 1 + j, e.card_object(d, ('o', j)))
        result = e.io.write_subrun(e.drive_off(7), 0, n, first_obj=1).result(10)
        self.assertEqual(result, n * e.object_bytes)
        for j in range(n):
            self.assertEqual(e.drive_read(e.drive_off(7) + j * e.object_bytes, e.object_bytes), e.drive_object(('o', j)), 'object %d' % j)
        self.assertEqual(e.drive_read(e.drive_off(7) + n * e.object_bytes, e.object_bytes), bytes(e.object_bytes), 'nothing beyond the run was written')
        call = e.ops.calls[0]
        self.assertEqual((call['kind'], call['offset'], call['lengths']), ('write', e.drive_off(7), [SHARD] * (4 * n)), 'one pwritev, shard 0..3 of object 0, then object 1, ...')

    def test_a_restore_puts_each_shard_on_its_card_and_touches_nothing_else(self):
        e = self.env()
        prime_drive(e, range(3), lambda slot: ('o', slot))
        result = e.io.read_piece(e.drive_off(0), 3, 3, first_obj=2).result(10)
        self.assertEqual(result, 3 * e.object_bytes)
        for d in range(4):
            for j in range(3):
                self.assertEqual(e.get(d, 3, 2 + j), e.card_object(d, ('o', j)), 'card %d object %d' % (d, j))
            for untouched in (0, 1, 5):
                self.assertEqual(e.get(d, 3, untouched), bytes(SHARD), 'card %d object %d must be untouched' % (d, untouched))
        call = e.ops.calls[0]
        self.assertEqual((call['kind'], call['offset'], call['lengths']), ('read', e.drive_off(0), [SHARD] * 12))

    def test_spill_then_restore_into_another_bank_is_byte_exact(self):
        e = self.env()
        for j in range(SLOT):
            for d in range(4):
                e.put(d, 1, j, e.card_object(d, ('x', j)))
        e.io.write_subrun(e.drive_off(11), 1, SLOT - 2).result(10)
        e.io.write_subrun(e.drive_off(11 + SLOT - 2), 1, 2, first_obj=SLOT - 2).result(10)
        e.io.read_piece(e.drive_off(11), 6, SLOT).result(10)
        for d in range(4):
            for j in range(SLOT):
                self.assertEqual(e.get(d, 6, j), e.card_object(d, ('x', j)))

    def test_with_the_real_gemma_geometry(self):
        e = self.env(rig_kwargs=dict(shard_object_bytes=fc.GEMMA_SHARD, slot_objects=3), data_slots=8)
        self.assertEqual(e.io.object_bytes, 2785280)
        self.assertEqual(e.io.shard_object_bytes, 696320)
        for j in range(2):
            for d in range(4):
                e.put(d, 0, j, e.card_object(d, ('g', j)))
        self.assertEqual(e.io.write_subrun(e.drive_off(2), 0, 2).result(30), 2 * 2785280)
        self.assertEqual(e.drive_read(e.drive_off(2), 2 * e.object_bytes), e.drive_object(('g', 0)) + e.drive_object(('g', 1)))
        e.io.read_piece(e.drive_off(2), 4, 2, first_obj=1).result(30)
        for d in range(4):
            self.assertEqual(e.get(d, 4, 1), e.card_object(d, ('g', 0)))
            self.assertEqual(e.get(d, 4, 2), e.card_object(d, ('g', 1)))
        self.assertEqual(e.ops.calls[0]['lengths'], [696320] * 8)

    def test_banks_with_different_addresses(self):
        e = self.env(rig_kwargs=dict(per_bank_offset=0x10000))
        prime_drive(e, [0], lambda slot: ('b', slot))
        e.io.read_piece(e.drive_off(0), 5, 1, first_obj=3).result(10)
        for d in range(4):
            self.assertEqual(e.get(d, 5, 3), e.card_object(d, ('b', 0)))


class VectorIoTests(unittest.TestCase):
    """The short-I/O loop on its own, with arbitrary counts (the drive may return any multiple of its block size, a signal may stop a call anywhere)."""

    @staticmethod
    def reader(source, counts, calls):
        """A preadv over ``source`` returning at most the next count each time."""
        queue = list(counts)

        def function(fd, buffers, offset):
            calls.append((offset, [memoryview(b).nbytes for b in buffers]))
            room = queue.pop(0) if queue else 1 << 40
            done = 0
            position = offset
            for buffer in buffers:
                view = memoryview(buffer)
                take = min(view.nbytes, room - done, len(source) - position)
                if take <= 0:
                    break
                view[:take] = source[position:position + take]
                position += take
                done += take
                if take < view.nbytes:
                    break
            return done
        return function

    def test_random_short_counts_move_every_byte_exactly_once(self):
        rng = random.Random(11)
        for round_ in range(200):
            sizes = [rng.choice((1, 7, 4096, 5000)) for _ in range(rng.randint(1, 9))]
            total = sum(sizes)
            source = rng.randbytes(total + 100)
            offset = rng.randint(0, 50)
            buffers = [bytearray(size) for size in sizes]
            views = [memoryview(b) for b in buffers]
            counts = [rng.randint(1, 6000) for _ in range(rng.randint(0, 12))]
            calls = []
            moved = pio.vector_io(self.reader(source, counts, calls), 0, views, offset, total)
            self.assertEqual(moved, total)
            self.assertEqual(b''.join(buffers), source[offset:offset + total], 'round %d counts %s' % (round_, counts))
            for view in views:
                view.release()           # fails if a slice made for a partly consumed iovec was not released
            position = offset
            for (call_offset, lengths), count in zip(calls, counts):
                self.assertEqual(call_offset, position)
                position += count if count < sum(lengths) else sum(lengths)

    def test_a_partly_consumed_iovec_is_continued_not_repeated(self):
        source = bytes(range(256)) * 16
        buffers = [bytearray(1000), bytearray(1000)]
        calls = []
        pio.vector_io(self.reader(source, [300, 1200], calls), 0, [memoryview(b) for b in buffers], 0, 2000)
        self.assertEqual(calls[0], (0, [1000, 1000]))
        self.assertEqual(calls[1], (300, [700, 1000]))
        self.assertEqual(calls[2], (1500, [500]))
        self.assertEqual(b''.join(buffers), source[:2000])

    def test_a_zero_count_is_end_of_file_and_not_a_media_error(self):
        calls = []
        with self.assertRaises(pio.P2PShortIO) as caught:
            pio.vector_io(self.reader(b'x' * 10, [], calls), 0, [memoryview(bytearray(100))], 0, 100)
        self.assertIsNone(caught.exception.errno)
        self.assertIsInstance(caught.exception, OSError)

    def test_a_count_beyond_the_call_is_a_bug_below_us(self):
        with self.assertRaises(pio.P2PIOError):
            pio.vector_io(lambda fd, buffers, offset: 101, 0, [memoryview(bytearray(100))], 0, 100)

    def test_an_oserror_keeps_its_errno(self):
        def function(fd, buffers, offset):
            raise OSError(errno.EIO, 'media')
        with self.assertRaises(OSError) as caught:
            pio.vector_io(function, 0, [memoryview(bytearray(100))], 0, 100)
        self.assertEqual(caught.exception.errno, errno.EIO)


class ShortIoTests(IOCase):
    def test_short_counts_across_iovec_boundaries_still_land_every_byte(self):
        e = self.env()
        prime_drive(e, range(3), lambda slot: ('s', slot))
        e.ops.shorts = [PAGE, 5 * PAGE, 4 * PAGE]
        e.io.read_piece(e.drive_off(0), 2, 3).result(10)
        calls = e.ops.calls
        self.assertEqual([c['offset'] - e.drive_off(0) for c in calls[:3]], [0, PAGE, 6 * PAGE], 'each call continues where the last one stopped')
        self.assertEqual(sum(c['returned'] for c in calls), 3 * e.object_bytes)
        for d in range(4):
            for j in range(3):
                self.assertEqual(e.get(d, 2, j), e.card_object(d, ('s', j)))

    def test_short_writes_are_continued_too(self):
        e = self.env()
        for j in range(2):
            for d in range(4):
                e.put(d, 1, j, e.card_object(d, ('w', j)))
        e.ops.shorts = [3 * PAGE, PAGE, 7 * PAGE]
        self.assertEqual(e.io.write_subrun(e.drive_off(4), 1, 2).result(10), 2 * e.object_bytes)
        self.assertEqual(e.drive_read(e.drive_off(4), 2 * e.object_bytes), e.drive_object(('w', 0)) + e.drive_object(('w', 1)))
        self.assertGreaterEqual(len(e.ops.calls), 4)

    def test_end_of_file_fails_the_call_without_errno_and_the_layer_goes_on(self):
        e = self.env()
        prime_drive(e, [0], lambda slot: ('e', slot))
        e.ops.zero_next = 1
        future = e.io.read_piece(e.drive_off(0), 2, 1)
        error = future.exception(10)
        self.assertIsInstance(error, pio.P2PShortIO)
        self.assertIsNone(error.errno)
        self.assertEqual(e.io.stats()['short_ios'], 1)
        self.assertIsNone(e.io.dead)
        self.assertEqual(e.io.read_piece(e.drive_off(0), 2, 1).result(10), e.object_bytes)
        self.assertEqual(e.runtime.latched, [], 'an I/O error is the worker\'s strike to count, not a latch')

    def test_a_media_error_keeps_its_errno_and_the_layer_goes_on(self):
        e = self.env()
        e.ops.errors = [OSError(errno.EIO, 'media error'), OSError(errno.ENODEV, 'gone')]
        first = e.io.read_piece(e.drive_off(0), 2, 1).exception(10)
        self.assertEqual(first.errno, errno.EIO)
        second = e.io.read_piece(e.drive_off(0), 2, 1).exception(10)
        self.assertEqual(second.errno, errno.ENODEV, 'the worker decides what ENODEV means')
        self.assertEqual(e.io.stats()['io_errors'], 2)
        self.assertIsNone(e.io.dead)
        prime_drive(e, [0], lambda slot: ('m', slot))
        self.assertEqual(e.io.read_piece(e.drive_off(0), 2, 1).result(10), e.object_bytes)


class SplitTests(IOCase):
    def test_a_long_restore_can_be_cut_into_calls_and_is_one_future(self):
        e = self.env(rig_kwargs=dict(slot_objects=12), io_kwargs=dict(restore_run_objects=4))
        prime_drive(e, range(10), lambda slot: ('l', slot))
        result = e.io.read_piece(e.drive_off(0), 2, 10, first_obj=1).result(10)
        self.assertEqual(result, 10 * e.object_bytes)
        self.assertEqual([c['total'] // e.object_bytes for c in e.ops.calls], [4, 4, 2])
        self.assertEqual([c['offset'] for c in e.ops.calls], [e.drive_off(0), e.drive_off(4), e.drive_off(8)])
        for d in range(4):
            for j in range(10):
                self.assertEqual(e.get(d, 2, 1 + j), e.card_object(d, ('l', j)))

    def test_a_failure_in_one_call_stops_the_rest_of_the_request(self):
        e = self.env(rig_kwargs=dict(slot_objects=12), io_kwargs=dict(restore_run_objects=4))
        prime_drive(e, range(10), lambda slot: ('l', slot))
        e.ops.fail_at = {1: OSError(errno.EIO, 'media error')}
        error = e.io.read_piece(e.drive_off(0), 2, 10).exception(10)
        self.assertEqual(error.errno, errno.EIO)
        self.assertEqual(len(e.ops.calls), 2, 'the third call was never issued')
        self.assertEqual(e.io.wait_idle(5), True)
        self.assertEqual(e.io.stats()['inflight_calls'], 0)

    def test_a_spill_sub_run_is_one_call(self):
        e = self.env(config_kwargs=dict(spill_subrun_objects=6))
        e.io.write_subrun(e.drive_off(0), 0, 6).result(10)
        self.assertEqual(len(e.ops.calls), 1)

    def test_the_default_is_one_call_per_run_as_the_design_says(self):
        e = self.env()
        prime_drive(e, range(SLOT), lambda slot: ('d', slot))
        e.io.read_piece(e.drive_off(0), 2, SLOT).result(10)
        self.assertEqual(len(e.ops.calls), 1)
        self.assertEqual(e.ops.calls[0]['lengths'], [SHARD] * (4 * SLOT))


class SchedulingTests(IOCase):
    def test_a_restore_goes_before_queued_spills_and_waits_only_for_the_one_in_flight(self):
        e = self.env()
        (release, entered), = e.ops.hold_next(1)
        s1 = e.io.write_subrun(e.drive_off(0), 0, 2)
        self.assertTrue(entered.wait(5))
        s2 = e.io.write_subrun(e.drive_off(2), 1, 2)
        s3 = e.io.write_subrun(e.drive_off(4), 0, 2)
        r1 = e.io.read_piece(e.drive_off(10), 3, 2)
        self.settle()
        self.assertEqual(len(e.ops.calls), 1, 'nothing starts while the spill in flight drains and a restore is queued')
        self.assertFalse(s2.done() or s3.done() or r1.done())
        release.set()
        for future in (s1, s2, s3, r1):
            future.result(10)
        order = [(c['kind'], c['offset']) for c in e.ops.calls]
        self.assertEqual(order, [('write', e.drive_off(0)), ('read', e.drive_off(10)), ('write', e.drive_off(2)), ('write', e.drive_off(4))])
        self.assertEqual([c['aimed'] for c in e.ops.calls], [[0] * 4, [3] * 4, [1] * 4, [0] * 4], 'every call ran with all four windows on its own bank')
        self.assertEqual(e.rig.reaim_while_busy, [])
        stats = e.io.stats()
        self.assertEqual(stats['aims'], 4)
        self.assertEqual(stats['restore_behind_spill'], 1)
        self.assertEqual(stats['spill_held_for_restore'], 2)
        self.assertGreaterEqual(stats['restore_behind_spill_wait_s_max'], 0.05, 'the drain time W1 bounds is measured')

    def test_calls_for_the_banks_of_the_current_epoch_share_it_without_a_re_aim(self):
        e = self.env()
        (release, entered), = e.ops.hold_next(1)
        r1 = e.io.read_piece(e.drive_off(0), 2, 1)
        self.assertTrue(entered.wait(5))
        r2 = e.io.read_piece(e.drive_off(1), 2, 1, first_obj=1)
        r2.result(10)                                  # joined the epoch while r1 was still in flight
        self.assertFalse(r1.done())
        release.set()
        r1.result(10)
        self.assertEqual(e.io.stats()['aims'], 1)

    def test_a_call_for_another_bank_waits_for_everything_in_flight_and_only_then_moves_the_windows(self):
        e = self.env()
        (release, entered), = e.ops.hold_next(1)
        r1 = e.io.read_piece(e.drive_off(0), 2, 1)
        self.assertTrue(entered.wait(5))
        before = [len(board.configures) for board in e.rig.boards]
        r2 = e.io.read_piece(e.drive_off(1), 3, 1)
        self.settle()
        self.assertEqual([len(board.configures) for board in e.rig.boards], before, 'no CONFIGURE_TLB while a bio is in flight')
        self.assertFalse(r2.done())
        release.set()
        r1.result(10)
        r2.result(10)
        self.assertEqual(e.rig.reaim_while_busy, [])
        self.assertEqual([c['aimed'] for c in e.ops.calls], [[2] * 4, [3] * 4])

    def test_the_queue_depth_counts_objects_in_flight(self):
        for depth, second_runs_at_once in ((4, False), (8, True)):
            with self.subTest(depth=depth):
                e = self.env(config_kwargs=dict(io_queue_depth_objects=depth))
                (release1, entered1), (release2, entered2) = e.ops.hold_next(2)
                r1 = e.io.read_piece(e.drive_off(0), 2, 4)
                self.assertTrue(entered1.wait(5))
                r2 = e.io.read_piece(e.drive_off(4), 2, 4)
                self.assertEqual(entered2.wait(0.2), second_runs_at_once)
                release1.set()
                release2.set()
                r1.result(10)
                r2.result(10)

    def test_a_call_larger_than_the_queue_depth_still_runs_alone(self):
        e = self.env(config_kwargs=dict(io_queue_depth_objects=2))
        self.assertEqual(e.io.read_piece(e.drive_off(0), 2, 5).result(10), 5 * e.object_bytes)

    def test_the_thread_count_limits_calls_in_flight(self):
        e = self.env(config_kwargs=dict(io_threads=1))
        (release1, entered1), (release2, entered2) = e.ops.hold_next(2)
        r1 = e.io.read_piece(e.drive_off(0), 2, 1)
        self.assertTrue(entered1.wait(5))
        r2 = e.io.read_piece(e.drive_off(1), 2, 1, first_obj=1)
        self.assertFalse(entered2.wait(0.2))
        release1.set()
        release2.set()
        r1.result(10)
        r2.result(10)
        self.assertEqual(e.io.stats()['inflight_calls_max'], 1)

    def test_a_storm_of_mixed_calls_from_three_threads(self):
        e = self.env(rig_kwargs=dict(slot_objects=6), data_slots=64)
        restores = [(i, 2 + i % 6, i // 6) for i in range(24)]            # (drive slot, bank, object): every destination distinct
        spills = [(30 + i, i % 2, i // 2, i) for i in range(12)]          # (drive slot, bank, object, tag)
        prime_drive(e, [r[0] for r in restores], lambda slot: ('r', slot))
        for _, bank, obj, tag in spills:
            for d in range(4):
                e.put(d, bank, obj, e.card_object(d, ('p', tag)))
        bank_of_offset = {e.drive_off(slot): bank for slot, bank, _ in restores}
        bank_of_offset.update({e.drive_off(slot): bank for slot, bank, _, _ in spills})
        jobs = [('r', r) for r in restores] + [('s', s) for s in spills]
        random.Random(5).shuffle(jobs)
        futures = []
        lock = threading.Lock()

        def run(chunk):
            for kind, job in chunk:
                if kind == 'r':
                    future = e.io.read_piece(e.drive_off(job[0]), job[1], 1, first_obj=job[2])
                else:
                    future = e.io.write_subrun(e.drive_off(job[0]), job[1], 1, first_obj=job[2])
                with lock:
                    futures.append(future)
                time.sleep(0.001)

        threads = [threading.Thread(target=run, args=(jobs[k::3],)) for k in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        for future in futures:
            self.assertEqual(future.result(30), e.object_bytes)
        self.assertEqual(e.rig.reaim_while_busy, [], 'a window was moved with a bio in flight')
        for call in e.ops.calls:
            self.assertEqual(call['aimed'], [bank_of_offset[call['offset']]] * 4, 'a bio ran with a window on the wrong bank: %s' % call)
        for slot, bank, obj in restores:
            for d in range(4):
                self.assertEqual(e.get(d, bank, obj), e.card_object(d, ('r', slot)), 'restore of slot %d' % slot)
        for slot, bank, obj, tag in spills:
            self.assertEqual(e.drive_read(e.drive_off(slot), e.object_bytes), e.drive_object(('p', tag)), 'spill of slot %d' % slot)
        stats = e.io.stats()
        self.assertEqual(stats['calls_read'], 24)
        self.assertEqual(stats['calls_write'], 12)

    def test_the_scheduler_thread_failing_stops_the_layer_and_records_why(self):
        e = self.env()

        def broken():
            raise RuntimeError('boom')
        e.io._decide = broken
        future = e.io.read_piece(e.drive_off(0), 2, 1)
        error = future.exception(10)
        self.assertIsInstance(error, pio.P2PDead)
        self.assertIn('scheduler thread failed', str(error))
        self.assertIn('scheduler thread failed', e.io.dead)
        self.assertEqual(e.runtime.latched, [], 'the scheduler thread never calls into the runtime: the worker latches from the engine thread when it sees io.dead')
        self.assertEqual(e.ops.calls, [])


class BoundsTests(IOCase):
    """Every request is checked against the slot, the banks of its direction and the data region before anything is queued. A violation raises, latches
    the tier, stops the layer and issues no syscall."""

    def cases(self, e):
        n = 3
        last = SLOT - n
        off = e.drive_off(0)
        hi = e.data_hi
        ob = e.object_bytes
        return [
            ('a read into a spill bank', lambda io: io.read_piece(off, 0, n)),
            ('a write from a restore bank', lambda io: io.write_subrun(off, 2, n)),
            ('a read into a bank that is not staging', lambda io: io.read_piece(off, 9, n)),
            ('a negative bank', lambda io: io.read_piece(off, -1, n)),
            ('the object range one past the slot (read)', lambda io: io.read_piece(off, 2, n, first_obj=last + 1)),
            ('the object range one past the slot (write)', lambda io: io.write_subrun(off, 0, n, first_obj=last + 1)),
            ('a negative first object', lambda io: io.read_piece(off, 2, n, first_obj=-1)),
            ('no objects', lambda io: io.read_piece(off, 2, 0)),
            ('a negative count', lambda io: io.write_subrun(off, 0, -1)),
            ('more objects than the slot', lambda io: io.read_piece(off, 2, SLOT + 1)),
            ('a spill sub-run longer than spill_subrun_objects', lambda io: io.write_subrun(off, 0, 5)),
            ('a float offset', lambda io: io.read_piece(float(off), 2, n)),
            ('a boolean bank', lambda io: io.read_piece(off, True, n)),
            ('a string count', lambda io: io.read_piece(off, 2, '3')),
            ('the superblock', lambda io: io.read_piece(0, 2, n)),
            ('the DMA canary block', lambda io: io.write_subrun(2 * PAGE, 0, n)),
            ('one object below the data region', lambda io: io.read_piece(e.DATA_OFF - ob, 2, 1)),
            ('one page below the data region', lambda io: io.write_subrun(e.DATA_OFF - PAGE, 0, 1)),
            ('a run that ends one page past the data region', lambda io: io.read_piece(hi - n * ob + PAGE, 2, n)),
            ('a run that starts past the end', lambda io: io.read_piece(hi, 2, 1)),
            ('an offset that is not 4 KiB aligned', lambda io: io.read_piece(off + 1, 2, n)),
            ('an offset 512 bytes in', lambda io: io.write_subrun(off + 512, 0, n)),
        ]

    def test_each_violation_raises_latches_stops_and_issues_nothing(self):
        names = [name for name, _ in self.cases(fc.Env(start=False))]
        for index, name in enumerate(names):
            with self.subTest(violation=name):
                e = self.env(config_kwargs=dict(spill_subrun_objects=4))
                case = self.cases(e)[index][1]
                with self.assertRaises(cl.BoundsViolation):
                    case(e.io)
                self.assertEqual(e.ops.calls, [], 'no syscall')
                self.assertTrue(e.runtime.latched, 'the tier is latched')
                self.assertIsNotNone(e.io.dead)
                self.assertEqual(e.io.stats()['bounds_violations'], 1)
                again = e.io.read_piece(e.drive_off(0), 2, 1)
                self.assertIsInstance(again.exception(5), pio.P2PDead, 'a layer that saw a violation issues nothing more')
                self.assertEqual(e.ops.calls, [])

    def test_integers_that_are_not_ints_are_accepted_but_floats_bools_and_strings_are_not(self):
        class Index(object):             # (what a numpy integer is: an object with __index__)
            def __init__(self, value):
                self.value = value

            def __index__(self):
                return self.value

        e = self.env()
        prime_drive(e, [0], lambda slot: ('n', slot))
        self.assertEqual(e.io.read_piece(Index(e.drive_off(0)), Index(2), Index(1), first_obj=Index(1)).result(10), e.object_bytes)
        for d in range(4):
            self.assertEqual(e.get(d, 2, 1), e.card_object(d, ('n', 0)))
        link = e.links[0]
        self.assertEqual(link.expected_slice_offset(Index(2), Index(1)), e.rig.data_offset(2, 1))
        self.assertEqual(len(link.slot_view(2, Index(0), Index(2))), 2)

    def test_the_edges_themselves_are_fine(self):
        e = self.env(config_kwargs=dict(spill_subrun_objects=4))
        ob = e.object_bytes
        self.assertEqual(e.io.read_piece(e.DATA_OFF, 2, SLOT).result(10), SLOT * ob, 'the whole slot at the start of the data region')
        self.assertEqual(e.io.read_piece(e.data_hi - ob, 2, 1, first_obj=SLOT - 1).result(10), ob, 'the last object of the slot from the last object of the region')
        self.assertEqual(e.io.write_subrun(e.DATA_OFF, 0, 4).result(10), 4 * ob)
        self.assertEqual(e.io.write_subrun(e.DATA_OFF + PAGE * 0 + ob * 10, 1, 2, first_obj=SLOT - 2).result(10), 2 * ob)
        self.assertEqual(e.runtime.latched, [])
        self.assertIsNone(e.io.dead)

    def test_a_bad_request_while_others_are_queued_fails_them_rather_than_run_them(self):
        e = self.env()
        (release, entered), = e.ops.hold_next(1)
        r1 = e.io.read_piece(e.drive_off(0), 2, 1)
        self.assertTrue(entered.wait(5))
        queued = e.io.read_piece(e.drive_off(1), 3, 1)
        with self.assertRaises(cl.BoundsViolation):
            e.io.read_piece(e.drive_off(2), 0, 1)
        self.assertIsInstance(queued.exception(5), pio.P2PDead)
        release.set()
        self.assertEqual(r1.result(10), e.object_bytes, 'the call already in flight completes: DMA cannot be taken back')
        self.assertEqual(len(e.ops.calls), 1)


class IovecMutationTests(IOCase):
    """The check right before each syscall, fed the planted mistakes of the design: off by one, the wrong bank, the wrong card, a stale epoch."""

    def plan(self, e, bank=2, first=1, n=2):
        for link in e.links:
            link.aim(bank)
        req = pio._Request('read', bank, 0.0)
        call = pio._Call(req, first, n, e.drive_off(0))
        call.tokens = tuple(link.aim_epoch for link in e.links)
        iovs = e.io._build_iovecs(call)
        return call, iovs

    def refuses(self, e, call, iovs, text):
        with self.assertRaises(cl.BoundsViolation) as caught:
            e.io._check_iovecs(call, iovs)
        self.assertIn(text, str(caught.exception))
        self.assertTrue(e.runtime.latched, 'latched')
        self.assertIsNotNone(e.io.dead)
        self.assertEqual(e.ops.calls, [])

    def test_the_plan_as_built_passes(self):
        e = self.env()
        call, iovs = self.plan(e)
        e.io._check_iovecs(call, iovs)
        self.assertEqual(len(iovs), 8)
        self.assertEqual([e.links[d].locate(iovs[j * 4 + d]) for j in range(2) for d in range(4)],
                         [e.rig.data_offset(2, 1 + j) for j in range(2) for d in range(4)], 'object-major, shard-minor')
        self.assertEqual(e.runtime.latched, [])

    def test_off_by_one_object(self):
        for delta in (-1, 1):
            with self.subTest(delta=delta):
                e = self.env()
                call, iovs = self.plan(e, first=2)
                iovs[1] = e.links[1].slot_view(2, 2 + delta, 1)[0]
                self.refuses(e, call, iovs, 'iovec 1')

    def test_a_view_one_page_off(self):
        e = self.env()
        call, iovs = self.plan(e)
        iovs[0] = e.links[0]._view[e.rig.canary_offset(2):e.rig.canary_offset(2) + SHARD]          # starts at the canary page
        self.refuses(e, call, iovs, 'iovec 0')

    def test_a_view_inside_the_object_but_not_at_its_start(self):
        e = self.env()
        call, iovs = self.plan(e)
        base = e.rig.data_offset(2, 1)
        iovs[2] = e.links[2]._view[base + 1:base + 1 + SHARD]
        self.refuses(e, call, iovs, 'iovec 2')

    def test_the_wrong_card(self):
        e = self.env()
        call, iovs = self.plan(e)
        iovs[0], iovs[1] = iovs[1], iovs[0]
        self.refuses(e, call, iovs, 'iovec 0')

    def test_the_wrong_bank_when_banks_have_different_addresses(self):
        e = self.env(rig_kwargs=dict(per_bank_offset=0x10000))
        call, iovs = self.plan(e)
        iovs[3] = e.links[3].slot_view(3, 1, 1)[0]
        self.refuses(e, call, iovs, 'iovec 3')

    def test_the_wrong_bank_when_the_windows_are_aimed_elsewhere(self):
        e = self.env()
        call, iovs = self.plan(e)
        e.links[1].aim(5)
        self.refuses(e, call, iovs, 'aimed at bank 5')

    def test_a_stale_epoch(self):
        e = self.env()
        call, iovs = self.plan(e)
        e.links[2].aim(2)                       # same bank, but aimed again since the plan was made
        self.refuses(e, call, iovs, 'epoch')

    def test_too_few_or_too_many_iovecs(self):
        e = self.env()
        call, iovs = self.plan(e)
        iovs.pop()
        self.refuses(e, call, iovs, '7 iovecs')
        e2 = self.env()
        call, iovs = self.plan(e2)
        iovs.append(iovs[0])
        self.refuses(e2, call, iovs, '9 iovecs')

    def test_an_iovec_of_the_wrong_length(self):
        for change in (lambda v: v[:-PAGE], lambda v: v[:-1]):
            e = self.env()
            call, iovs = self.plan(e)
            iovs[3] = change(iovs[3])
            self.refuses(e, call, iovs, 'iovec 3')

    def test_an_iovec_that_is_host_memory(self):
        e = self.env()
        call, iovs = self.plan(e)
        iovs[2] = memoryview(bytearray(SHARD))
        self.refuses(e, call, iovs, 'iovec 2')

    def test_a_link_in_the_wrong_slot_of_the_shard_map(self):
        e = self.env()
        call, iovs = self.plan(e)
        e.links[0].shard = 1
        self.refuses(e, call, iovs, 'shard')

    def test_a_drive_range_outside_the_data_region_at_the_syscall(self):
        for off in (0, 2 * PAGE):
            e = self.env()
            call, iovs = self.plan(e)
            call.drive_off = off
            self.refuses(e, call, iovs, 'outside the data region')

    def test_a_planted_mistake_is_caught_even_when_the_call_is_issued_the_normal_way(self):
        """The same stale-epoch mistake through the scheduler: another thread re-aims a card behind the layer's back."""
        e = self.env()
        e.io.read_piece(e.drive_off(0), 2, 1).result(10)
        e.links[3].aim(2)                      # (the layer thinks the epoch is still the old one)
        future = e.io.read_piece(e.drive_off(1), 2, 1, first_obj=1)
        error = future.exception(10)
        self.assertIsInstance(error, cl.StaleAim)
        self.assertEqual(len(e.ops.calls), 1, 'the stale call never reached the drive')
        self.assertEqual(e.runtime.latched, [], 'found on the scheduler thread: recorded, not latched from there')
        self.assertTrue(e.io.dead.startswith(pio.FATAL_PREFIX), e.io.dead)
        self.assertEqual(e.links[0].inflight, 0, 'the cards that had counted the call in flight were released again')


class CanaryTests(IOCase):
    def test_a_canary_mismatch_on_a_re_aim_stops_everything_before_a_single_bio(self):
        e = self.env()
        e.rig.boards[1].ignore_configure = True
        future = e.io.read_piece(e.drive_off(0), 4, 1)
        error = future.exception(10)
        self.assertIsInstance(error, pio.P2PDead)
        self.assertEqual(e.ops.calls, [])
        self.assertIn('re-aim to bank 4', e.io.dead)
        self.assertTrue(e.io.dead.startswith(pio.FATAL_PREFIX), e.io.dead)
        self.assertEqual(e.runtime.latched, [], 'found on the scheduler thread: recorded, not latched from there')
        self.assertEqual(e.io.stats()['canary_mismatches'], 1)
        self.assertIsInstance(e.io.read_piece(e.drive_off(0), 4, 1).exception(5), pio.P2PDead)
        self.assertIsInstance(e.io.write_subrun(e.drive_off(0), 0, 1).exception(5), pio.P2PDead)

    def test_a_mismatch_fails_everything_that_was_queued_behind_it(self):
        e = self.env()
        e.rig.boards[2].alias = {5: 3}
        futures = [e.io.read_piece(e.drive_off(i), 5, 1, first_obj=i) for i in range(3)]
        for future in futures:
            self.assertIsInstance(future.exception(10), pio.P2PDead)
        self.assertEqual(e.ops.calls, [])

    def test_the_dma_canary_round_trips_a_page_per_card_both_ways(self):
        e = self.env()
        e.io.dma_canary()
        calls = e.ops.calls
        self.assertEqual([(c['kind'], c['offset'], c['total']) for c in calls], [('write', 2 * PAGE, PAGE), ('read', 2 * PAGE, PAGE)] * 4)
        self.assertEqual(e.drive_read(2 * PAGE, PAGE), e.io._canary_pattern(3), 'the last card\'s page is on the drive')
        self.assertEqual(e.runtime.latched, [])
        self.assertIsNone(e.io.epoch_bank, 'the windows are somewhere the epoch does not know')
        prime_drive(e, [0], lambda slot: ('c', slot))
        e.io.read_piece(e.drive_off(0), 2, 1).result(10)           # the first data call re-aims all four
        for d in range(4):
            self.assertEqual(e.get(d, 2, 0), e.card_object(d, ('c', 0)))
        for link in e.links:
            self.assertEqual(link.inflight, 0)
        # the canary pages themselves were never overwritten
        for d, board in enumerate(e.rig.boards):
            for bank in range(8):
                page = board.content(bank)[e.rig.canary_offset(bank):e.rig.canary_offset(bank) + PAGE]
                self.assertEqual(page, cl.canary_page(e.links[d]._nonce, board.bdf, bank))

    def test_a_drive_that_does_not_write_the_card_fails_the_dma_canary(self):
        e = self.env()
        e.ops.on_call = lambda kind, record: setattr(e.ops, 'swallow_next', 1) if kind == 'read' else None
        with self.assertRaises(pio.DmaCanaryError):
            e.io.dma_canary()
        self.assertTrue(e.runtime.latched)
        self.assertIsNotNone(e.io.dead)

    def test_a_drive_that_does_not_take_the_page_fails_the_dma_canary(self):
        e = self.env()
        e.ops.on_call = lambda kind, record: setattr(e.ops, 'swallow_next', 1) if kind == 'write' else None
        with self.assertRaises(pio.DmaCanaryError):
            e.io.dma_canary()

    def test_the_dma_canary_needs_an_idle_layer(self):
        e = self.env()
        (release, entered), = e.ops.hold_next(1)
        future = e.io.read_piece(e.drive_off(0), 2, 1)
        self.assertTrue(entered.wait(5))
        with self.assertRaises(pio.P2PIOError):
            e.io.dma_canary()
        release.set()
        future.result(10)

    def test_the_dma_canary_only_writes_inside_its_block(self):
        e = self.env(io_kwargs=dict(canary_offset=4 * PAGE))
        with self.assertRaises(cl.BoundsViolation):
            e.io.dma_canary()
        self.assertEqual(e.ops.calls, [])


class ShutdownTests(IOCase):
    def test_close_fails_the_queue_lets_the_call_in_flight_finish_and_returns_true(self):
        e = self.env()
        (release, entered), = e.ops.hold_next(1)
        r1 = e.io.read_piece(e.drive_off(0), 2, 1)
        self.assertTrue(entered.wait(5))
        r2 = e.io.read_piece(e.drive_off(1), 3, 1)
        s1 = e.io.write_subrun(e.drive_off(2), 0, 1)
        outcome = []
        closer = threading.Thread(target=lambda: outcome.append(e.io.close(timeout=10)))
        closer.start()
        self.assertIsInstance(r2.exception(5), pio.P2PClosed)
        self.assertIsInstance(s1.exception(5), pio.P2PClosed)
        self.assertFalse(r1.done(), 'the call in flight is not abandoned')
        release.set()
        closer.join(10)
        self.assertEqual(outcome, [True])
        self.assertEqual(r1.result(5), e.object_bytes)
        self.assertEqual(len(e.ops.calls), 1)
        self.assertIsInstance(e.io.read_piece(e.drive_off(0), 2, 1).exception(5), pio.P2PClosed, 'after close a request is refused, not queued')
        self.assertTrue(e.io.close(), 'closing twice is fine')

    def test_a_drive_that_will_not_answer_leaves_the_links_mapped(self):
        e = self.env()
        (release, entered), = e.ops.hold_next(1)
        e.io.read_piece(e.drive_off(0), 2, 1)
        self.assertTrue(entered.wait(5))
        with self.assertLogs('ttkvp2p', level='ERROR'):
            self.assertFalse(e.io.close(timeout=0.2))
        for link in e.links:
            with self.assertLogs('ttkvp2p', level='WARNING'):
                link.close()
            self.assertTrue(link.closed)
        self.assertEqual(e.rig.kmd.open_fds, 4 * 3, 'chardev, p2pmem file and pin of every card are still open: a DMA may be running into them')
        release.set()

    def test_the_drive_descriptor_is_the_runtimes_and_stays_open_after_close(self):
        e = self.env()
        e.io.close()
        os.fstat(e.fd)
        e2 = self.env(start=False)
        e2.links = e2.rig.open_links()
        e2.runtime = e2.wp3a_runtime()
        e2.build_io(drive_fd=None, drive_range=None)
        self.assertEqual(e2.io.read_piece(e2.drive_off(0), 2, 1).result(10), e2.object_bytes)
        e2.io.close()
        os.fstat(e2.fd)

    def test_threads_are_named_for_the_per_thread_cpu_gate_and_stop_at_close(self):
        e = self.env()
        e.io.read_piece(e.drive_off(0), 2, 1).result(10)
        names = {t.name for t in threading.enumerate()}
        self.assertIn('kvp2p-sched', names)
        self.assertTrue(any(name.startswith('kvp2p-io') for name in names), names)
        self.assertTrue(all(c['thread'].startswith('kvp2p-io') for c in e.ops.calls))
        self.assertGreater(e.io.stats()['io_thread_cpu_s'], 0.0, 'the per-thread CPU time of the I/O threads is measured')
        e.io.close()
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline and 'kvp2p-sched' in {t.name for t in threading.enumerate()}:
            time.sleep(0.01)
        self.assertNotIn('kvp2p-sched', {t.name for t in threading.enumerate()})


class DriveSourceTests(IOCase):
    """Where the descriptor, the data region and the canary offset come from: the runtime, with the names of WP3a's ``KVP2PRuntime`` (``drive.fd``, ``store.data_off`` /
    ``num_slots`` / ``object_bytes``, ``canary_offset``). Nothing is guessed, and the device is never opened here."""

    def built(self, **runtime_overrides):
        e = self.env(start=False)
        e.links = e.rig.open_links()
        e.runtime = e.wp3a_runtime(**runtime_overrides)
        return e

    def refused(self, e, text, **overrides):
        with self.assertRaises(pio.P2PIOError) as caught:
            e.build_io(drive_fd=None, drive_range=None, **overrides)
        self.assertIn(text, str(caught.exception))
        self.assertNotIn('kvp2p-sched', {t.name for t in threading.enumerate()}, 'a layer that failed to start leaves no thread')
        self.assertFalse(hasattr(e.ops, 'opened'), 'the drive operations have no open: this layer never opens the device')

    def test_the_wp3a_runtime_gives_the_descriptor_the_data_region_and_the_canary_offset(self):
        e = self.built(num_slots=5)
        e.build_io(drive_fd=None, drive_range=None)
        self.assertEqual(e.io._drive_range, (e.DATA_OFF, e.DATA_OFF + 5 * e.object_bytes), 'the fence is exactly the region the runtime states')
        self.assertEqual(e.io._canary_offset, 2 * PAGE)
        prime_drive(e, [4], lambda slot: ('r', slot))
        self.assertEqual(e.io.read_piece(e.drive_off(4), 2, 1).result(10), e.object_bytes, 'the last slot of the region')
        for off in (e.drive_off(5), e.drive_off(6) + PAGE, e.drive_off(4) + PAGE):
            with self.assertRaises(cl.BoundsViolation, msg='offset %d is past num_slots' % off):
                e.io.read_piece(off, 2, 1)
        self.assertEqual(len(e.ops.calls), 1)

    def test_the_wp3a_runtime_makes_the_whole_start_up_work_without_any_keyword(self):
        e = self.built()
        e.build_io(drive_fd=None, drive_range=None)
        e.io.dma_canary()
        self.assertEqual([(c['kind'], c['offset']) for c in e.ops.calls[:2]], [('write', 2 * PAGE), ('read', 2 * PAGE)])
        self.assertEqual(e.runtime.latched, [])

    def test_the_keywords_win_over_the_runtime(self):
        e = self.built(num_slots=2)
        e.build_io(drive_fd=e.fd, drive_range=(e.DATA_OFF, e.data_hi))
        self.assertEqual(e.io._drive_range, (e.DATA_OFF, e.data_hi))

    def test_no_drive_at_all_is_a_clean_error_and_the_configured_path_is_not_opened(self):
        e = self.env(start=False)
        e.links = e.rig.open_links()
        e.config.nvme_device = e.path
        self.refused(e, 'no drive')
        e.runtime = types.SimpleNamespace(latch=lambda reason: None, drive=types.SimpleNamespace(), store=types.SimpleNamespace(data_off=e.DATA_OFF, num_slots=3, object_bytes=e.object_bytes))
        self.refused(e, 'no drive')

    def test_a_runtime_that_states_no_data_region_is_refused_instead_of_a_guessed_fence(self):
        for label, store in (('no store', None), ('no data_off', dict(num_slots=3)), ('no num_slots', dict(data_off=1 << 20)), ('zero slots', dict(data_off=1 << 20, num_slots=0)),
                             ('a string', dict(data_off='1048576', num_slots=3))):
            with self.subTest(label):
                e = self.built()
                if store is None:
                    del e.runtime.store
                else:
                    e.runtime.store = types.SimpleNamespace(**store)
                self.refused(e, 'no data region')

    def test_the_runtimes_object_size_must_be_the_cards(self):
        e = self.built(object_bytes=4 * fc.SMALL_SHARD + PAGE)
        self.refused(e, 'objects are')

    def test_the_region_must_still_clear_the_superblock_and_the_canary_block(self):
        e = self.built(data_off=2 * PAGE)
        self.refused(e, 'superblock')

    def test_the_runtimes_canary_offset_must_be_inside_the_canary_block(self):
        for offset in (0, PAGE, 4 * PAGE, 2 * PAGE + 1, 'x'):
            with self.subTest(offset=offset):
                e = self.built(canary_offset=offset)
                self.refused(e, 'canary')
        e = self.built(canary_offset=3 * PAGE)            # the second page of the block is fine, and is the one used
        e.build_io(drive_fd=None, drive_range=None)
        e.io.dma_canary()
        self.assertEqual({c['offset'] for c in e.ops.calls}, {3 * PAGE})
        e = self.built(canary_offset=None)                # a runtime that says nothing: the frozen block start
        e.build_io(drive_fd=None, drive_range=None)
        self.assertEqual(e.io._canary_offset, 2 * PAGE)

    def test_a_descriptor_that_is_not_an_integer_is_refused(self):
        for bad in ('3', 3.0, True, object()):
            with self.subTest(bad=bad):
                e = self.built()
                e.runtime.drive.fd = bad
                self.refused(e, 'no drive')


class DirectIoTests(IOCase):
    """Buffered I/O on the drive would make the CPU copy the cards' BAR pages: a descriptor that is not O_DIRECT is refused outside a test."""

    def setUp(self):
        if not hasattr(os, 'O_DIRECT'):
            self.skipTest('this platform has no O_DIRECT')

    def test_a_buffered_descriptor_is_refused(self):
        e = self.env(start=False)
        e.links = e.rig.open_links()
        fd = os.open(e.path, os.O_RDWR)
        self.addCleanup(os.close, fd)
        self.assertFalse(pio.fd_is_direct(fd))
        with self.assertRaises(pio.P2PIOError) as caught:
            e.build_io(drive_fd=fd, allow_buffered=False)
        self.assertIn('O_DIRECT', str(caught.exception))
        self.assertNotIn('kvp2p-sched', {t.name for t in threading.enumerate()})

    def test_a_buffered_descriptor_is_taken_only_where_a_test_says_so(self):
        e = self.env(start=False)
        e.links = e.rig.open_links()
        fd = os.open(e.path, os.O_RDWR)
        self.addCleanup(os.close, fd)
        e.build_io(drive_fd=fd, allow_buffered=True)
        self.assertEqual(e.io.read_piece(e.drive_off(0), 2, 1).result(10), e.object_bytes)

    def test_a_direct_descriptor_passes_the_default_and_a_bad_one_does_not(self):
        e = self.env(start=False)
        e.links = e.rig.open_links()
        if not e.direct:
            self.skipTest('this file system does not take O_DIRECT')
        self.assertTrue(pio.fd_is_direct(e.fd))
        e.build_io(allow_buffered=False)
        self.assertFalse(pio.fd_is_direct(-1))
        self.assertFalse(pio.fd_is_direct(10 ** 6))


class ConstructionTests(IOCase):
    def refused(self, e, text, **overrides):
        with self.assertRaises(pio.P2PIOError) as caught:
            e.build_io(**overrides)
        self.assertIn(text, str(caught.exception))
        self.assertNotIn('kvp2p-sched', {t.name for t in threading.enumerate()}, 'a layer that failed to start leaves no thread')

    def test_links_that_have_not_proved_their_canaries_are_refused(self):
        e = self.env(start=False)
        e.links = [e.rig.make_link(d) for d in range(4)]
        self.refused(e, 'has not proved')

    def test_the_shards_must_be_exactly_the_mesh_positions(self):
        e = self.env()
        e.io.close()
        e.links[1].shard = 0
        self.refused(e, 'shards')

    def test_links_may_come_in_any_order(self):
        e = self.env()
        e.io.close()
        e.links = list(reversed(e.links))
        e.build_io()
        prime_drive(e, [0], lambda slot: ('o', slot))
        e.io.read_piece(e.drive_off(0), 2, 1).result(10)
        for d in range(4):
            self.assertEqual(e.get(d, 2, 0), e.card_object(d, ('o', 0)), 'shard d goes to card d whatever the list order')

    def test_the_shard_map_must_agree_with_the_staging_record(self):
        e = self.env(start=False)
        e.links = e.rig.open_links()
        info = e.rig.staging_info()
        info['devices'][0]['pci_bus_id'], info['devices'][1]['pci_bus_id'] = info['devices'][1]['pci_bus_id'], info['devices'][0]['pci_bus_id']
        with self.assertRaises(pio.P2PIOError) as caught:
            pio.P2PIO(e.links, e.runtime, e.config, info, drive_fd=e.fd, drive_ops=e.ops, drive_range=(e.DATA_OFF, e.data_hi))
        self.assertIn('staging record', str(caught.exception))

    def test_bank_sets(self):
        e = self.env()
        e.io.close()
        e.config.restore_banks = (1, 2, 3)
        self.refused(e, 'disjoint')
        e.config.restore_banks = (2, 3, 9)
        self.refused(e, 'no bank')
        e.config.restore_banks = ()
        self.refused(e, 'non-empty')

    def test_sizes(self):
        e = self.env()
        e.io.close()
        e.config.spill_subrun_objects = 300
        self.refused(e, 'spill_subrun_objects')
        e.config.spill_subrun_objects = 0
        self.refused(e, 'spill_subrun_objects')
        e.config.spill_subrun_objects = 8
        e.config.slot_objects = 0
        self.refused(e, 'holds no object')

    def test_the_data_range(self):
        e = self.env()
        e.io.close()
        self.refused(e, 'superblock', drive_range=(2 * PAGE, e.data_hi))
        self.refused(e, 'aligned', drive_range=(e.DATA_OFF + 1, e.data_hi))
        self.refused(e, 'non-empty', drive_range=(e.DATA_OFF, e.DATA_OFF))

    def test_the_slot_is_the_smaller_of_the_config_and_the_links(self):
        e = self.env(config_kwargs=dict(slot_objects=4))
        self.assertEqual(e.io.slot_objects, 4)
        with self.assertRaises(cl.BoundsViolation):
            e.io.read_piece(e.drive_off(0), 2, 1, first_obj=4)


class FutureResolutionTests(IOCase):
    """A future resolves only when no call of its request is still in flight. The worker releases the bank and the slots' in-flight counts when it sees the future done,
    so a future that resolved early would let a bank be reused, or a slot be trimmed, under a DMA that is still running. This arises when a request is cut into several calls
    (``restore_run_objects`` shorter than the run), a knob the W1 sweeps may turn."""

    def split(self, **kwargs):
        io_kwargs = dict(restore_run_objects=2)
        io_kwargs.update(kwargs.pop('io_kwargs', {}))
        e = self.env(rig_kwargs=dict(slot_objects=12), io_kwargs=io_kwargs, **kwargs)
        prime_drive(e, range(8), lambda slot: ('f', slot))
        return e

    def wait_for(self, condition, what):
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            if condition():
                return
            time.sleep(0.005)
        self.fail('timed out waiting for %s' % what)

    def inflight_when_resolved(self, e, future):
        seen = []
        future.add_done_callback(lambda done: seen.append(e.io.stats()['inflight_calls']))
        return seen

    def test_an_error_in_one_call_does_not_resolve_the_request_while_a_sibling_is_in_flight(self):
        e = self.split()
        (release, entered), = e.ops.hold_next(1)
        e.ops.fail_at = {1: OSError(errno.EIO, 'media error')}               # the second call fails at once, the first is held
        future = e.io.read_piece(e.drive_off(0), 2, 4)                       # two calls of two objects, both in flight together
        seen = self.inflight_when_resolved(e, future)
        self.assertTrue(entered.wait(5))
        self.wait_for(lambda: e.io.stats()['io_errors'] == 1, 'the failing call to return')
        self.settle()
        self.assertEqual(e.io.stats()['inflight_calls'], 1)
        self.assertFalse(future.done(), 'resolved on the first error while the sibling call was still in flight')
        release.set()
        self.assertEqual(future.exception(5).errno, errno.EIO)
        self.assertEqual(seen, [0], 'the future resolved with a call still in flight')

    def test_an_errored_request_with_calls_left_to_issue_waits_for_the_call_in_flight(self):
        e = self.split(config_kwargs=dict(io_threads=2))
        (release, entered), = e.ops.hold_next(1)
        e.ops.fail_at = {1: OSError(errno.EIO, 'media error')}
        future = e.io.read_piece(e.drive_off(0), 2, 6)                       # three calls: the first held, the second fails, the third is never issued
        seen = self.inflight_when_resolved(e, future)
        self.assertTrue(entered.wait(5))
        self.wait_for(lambda: e.io.stats()['io_errors'] == 1, 'the failing call to return')
        self.settle()
        self.assertFalse(future.done(), 'the scheduler resolved an errored request that still had a call in flight')
        self.assertEqual(len(e.ops.calls), 2, 'the third call was never issued')
        release.set()
        self.assertEqual(future.exception(5).errno, errno.EIO)
        self.assertEqual(seen, [0])
        self.assertEqual(e.io.stats()['queued_restore'], 0)

    def test_a_partly_issued_request_is_not_resolved_by_the_death_of_the_layer_while_a_call_is_in_flight(self):
        e = self.split(config_kwargs=dict(io_threads=1))
        (release, entered), = e.ops.hold_next(1)
        future = e.io.read_piece(e.drive_off(0), 2, 6)                       # three calls, one thread: the first is held, two are still to issue
        seen = self.inflight_when_resolved(e, future)
        self.assertTrue(entered.wait(5))
        with self.assertRaises(cl.BoundsViolation):
            e.io.read_piece(e.drive_off(0), 0, 1)                            # a bad request stops the layer and fails what is queued
        self.settle()
        self.assertFalse(future.done(), 'resolved by the death of the layer with its first call still in flight')
        release.set()
        self.assertIsInstance(future.exception(5), pio.P2PDead)
        self.assertEqual(seen, [0])
        self.assertEqual(len(e.ops.calls), 1, 'the calls that were never issued stay unissued')

    def test_a_partly_issued_request_is_not_resolved_by_close_while_a_call_is_in_flight(self):
        e = self.split(config_kwargs=dict(io_threads=1))
        (release, entered), = e.ops.hold_next(1)
        future = e.io.read_piece(e.drive_off(0), 2, 6)
        seen = self.inflight_when_resolved(e, future)
        self.assertTrue(entered.wait(5))
        outcome = []
        closer = threading.Thread(target=lambda: outcome.append(e.io.close(timeout=10)))
        closer.start()
        self.settle()
        self.assertFalse(future.done(), 'resolved by close() with its first call still in flight')
        release.set()
        closer.join(10)
        self.assertEqual(outcome, [True])
        self.assertIsInstance(future.exception(5), pio.P2PClosed)
        self.assertEqual(seen, [0])

    def test_a_request_that_succeeds_resolves_once_with_all_its_bytes(self):
        e = self.split()
        future = e.io.read_piece(e.drive_off(0), 2, 6)
        seen = self.inflight_when_resolved(e, future)
        self.assertEqual(future.result(10), 6 * e.object_bytes)
        self.assertEqual(seen, [0])
        self.assertEqual(len(e.ops.calls), 3)


class ThreadContractTests(IOCase):
    """The layer's own threads never call into the runtime (I7: the runtime belongs to the engine thread). A fault found there stops the layer and is recorded in
    ``dead``, which the worker reads and latches from the engine thread. A fault found on a caller's thread is latched on that thread."""

    @staticmethod
    def filed_as(reason):
        """The part of the runtime's latch classifier (WP3a ``health.latch_code``) that decides between these reasons, in its order."""
        text = (reason or '').lower()
        if 'control' in text and 'off' in text:
            return 'control_off'
        if 'consecutive job failure' in text or 'strike' in text:
            return 'strikes'
        if 'no completion' in text or 'stall' in text:
            return 'stall'
        if 'aer' in text:
            return 'aer_uncorrectable'
        if 'device lost' in text or 'enodev' in text or 'enxio' in text or 'not live' in text or 'reads 0 bytes' in text:
            return 'device_lost'
        if 'bound' in text or 'canary' in text:
            return 'bounds_violation'
        return 'op_throw'

    def test_a_violation_found_on_a_pool_thread_is_recorded_and_the_runtime_is_not_called(self):
        e = self.env()
        e.io.read_piece(e.drive_off(0), 2, 1).result(10)
        e.links[1].locate = lambda view: 12345                               # the check right before the syscall (an I/O thread) now finds a buffer in the wrong place
        error = e.io.read_piece(e.drive_off(1), 2, 1, first_obj=1).exception(10)
        self.assertIsInstance(error, cl.BoundsViolation)
        self.assertTrue(e.io.dead.startswith(pio.FATAL_PREFIX), e.io.dead)
        self.assertEqual(self.filed_as(e.io.dead), 'bounds_violation')
        self.assertEqual((e.runtime.latched, e.runtime.latch_threads), ([], []))
        self.assertEqual(len(e.ops.calls), 1, 'the second call never reached the drive')

    def test_a_card_that_moved_found_on_the_scheduler_thread_is_recorded_and_the_runtime_is_not_called(self):
        e = self.env()
        e.io.read_piece(e.drive_off(0), 2, 1).result(10)
        e.links[3].aim(2)                                                    # behind the layer's back
        self.assertIsInstance(e.io.read_piece(e.drive_off(1), 2, 1, first_obj=1).exception(10), cl.StaleAim)
        self.assertEqual(self.filed_as(e.io.dead), 'bounds_violation', 'a stale aim is a window fault, not an internal error: %s' % e.io.dead)
        self.assertEqual(e.runtime.latched, [])

    def test_a_canary_mismatch_on_a_re_aim_is_filed_as_a_bounds_violation(self):
        e = self.env()
        e.rig.boards[1].ignore_configure = True
        e.io.read_piece(e.drive_off(0), 4, 1).exception(10)
        self.assertEqual(self.filed_as(e.io.dead), 'bounds_violation', e.io.dead)
        self.assertEqual(e.runtime.latched, [])

    def test_a_failing_re_aim_that_is_not_a_canary_mismatch_is_filed_the_same_way(self):
        e = self.env()
        e.rig.kmd.fail['configure'] = errno.EIO
        e.io.read_piece(e.drive_off(0), 4, 1).exception(10)
        self.assertIn('re-aim to bank 4 failed', e.io.dead)
        self.assertEqual(self.filed_as(e.io.dead), 'bounds_violation', e.io.dead)
        self.assertEqual(e.runtime.latched, [])

    def test_a_bad_request_is_latched_on_the_callers_thread(self):
        e = self.env()
        with self.assertRaises(cl.BoundsViolation):
            e.io.read_piece(e.drive_off(0), 0, 1)
        self.assertEqual(len(e.runtime.latched), 1)
        self.assertEqual(e.runtime.latch_threads, [threading.current_thread().name])
        self.assertEqual(self.filed_as(e.runtime.latched[0]), 'bounds_violation')
        self.assertEqual(e.runtime.latched[0], e.io.dead)

    def test_the_start_up_dma_canary_is_latched_on_the_callers_thread_and_filed_as_a_bounds_violation(self):
        e = self.env()
        e.ops.on_call = lambda kind, record: setattr(e.ops, 'swallow_next', 1) if kind == 'read' else None
        with self.assertRaises(pio.DmaCanaryError):
            e.io.dma_canary()
        self.assertEqual(e.runtime.latch_threads, [threading.current_thread().name])
        self.assertEqual(self.filed_as(e.runtime.latched[0]), 'bounds_violation', e.runtime.latched[0])

    def test_the_first_reason_is_kept_and_the_runtime_is_latched_once(self):
        e = self.env()
        with self.assertRaises(cl.BoundsViolation):
            e.io.read_piece(e.drive_off(0), 0, 1)
        first = e.io.dead
        with self.assertRaises(cl.BoundsViolation):
            e.io.read_piece(e.drive_off(0), 9, 1)
        self.assertEqual(e.io.dead, first)
        self.assertEqual(len(e.runtime.latched), 1)

    def test_a_scheduler_crash_is_an_internal_error_not_a_bounds_violation(self):
        e = self.env()
        e.io._decide = lambda: 1 / 0
        e.io.read_piece(e.drive_off(0), 2, 1).exception(10)
        self.assertTrue(e.io.dead.startswith('P2PIO internal error'), e.io.dead)
        self.assertEqual(self.filed_as(e.io.dead), 'op_throw')
        self.assertEqual(e.runtime.latched, [])


class DeadLayerTests(IOCase):
    """A call decided before the layer died is not issued afterwards (the ``P2PDead`` docstring: nothing more is issued)."""

    def test_a_layer_that_dies_while_the_cards_count_a_call_issues_no_syscall(self):
        e = self.env()
        link = e.links[0]
        original = link.begin_io

        def begin_io(bank, epoch):
            e.io._die('%s: another thread found a violation' % pio.FATAL_PREFIX)
            link.begin_io = original
            return original(bank, epoch)
        link.begin_io = begin_io
        future = e.io.read_piece(e.drive_off(0), 2, 1)
        self.assertIsInstance(future.exception(10), pio.P2PDead)
        self.assertEqual(e.ops.calls, [], 'a syscall was issued after the layer was dead')
        self.assertTrue(e.io.wait_idle(5))
        self.assertEqual([link.inflight for link in e.links], [0] * 4)
        self.assertEqual(e.rig.inflight, 0)

    def test_a_layer_that_dies_while_the_iovecs_are_built_issues_no_syscall(self):
        e = self.env()
        link = e.links[2]
        original = link.slot_view

        def slot_view(bank, first_obj, n_obj):
            e.io._die('%s: another thread found a violation' % pio.FATAL_PREFIX)
            return original(bank, first_obj, n_obj)
        link.slot_view = slot_view
        future = e.io.read_piece(e.drive_off(0), 2, 1)
        self.assertIsInstance(future.exception(10), pio.P2PDead)
        self.assertEqual(e.ops.calls, [])
        self.assertTrue(e.io.wait_idle(5))
        self.assertEqual([link.inflight for link in e.links], [0] * 4)

    def test_a_call_decided_just_before_the_layer_died_does_not_even_count_in_flight_on_the_cards(self):
        e = self.env()
        counted = []
        for link in e.links:
            original = link.begin_io
            link.begin_io = lambda bank, epoch, original=original: counted.append(bank) or original(bank, epoch)
        issue = e.io._do_issue

        def late(req, call):
            e.io._die('%s: another thread found a violation' % pio.FATAL_PREFIX)
            return issue(req, call)
        e.io._do_issue = late
        future = e.io.read_piece(e.drive_off(0), 2, 1)
        self.assertIsInstance(future.exception(10), pio.P2PDead)
        self.assertEqual((counted, e.ops.calls), ([], []))
        self.assertTrue(e.io.wait_idle(5))
        self.assertEqual(e.io.stats()['inflight_calls'], 0)


class WallClockDeadlineTests(IOCase):
    """The deadlines of close() and wait_idle() are wall time: an injected clock (for the statistics) that stands still must not make them wait forever."""

    def test_close_with_a_clock_that_stands_still_and_a_stuck_call_returns(self):
        e = self.env(io_kwargs=dict(clock=lambda: 0.0))
        (release, entered), = e.ops.hold_next(1)
        e.io.read_piece(e.drive_off(0), 2, 1)
        self.assertTrue(entered.wait(5))
        outcome = []
        closer = threading.Thread(target=lambda: outcome.append(e.io.close(timeout=0.2)), daemon=True)
        closer.start()
        closer.join(3.0)
        alive = closer.is_alive()
        release.set()
        closer.join(5)
        self.assertFalse(alive, 'close(timeout=0.2) was still waiting after 3 s')
        self.assertEqual(outcome, [False])

    def test_wait_idle_with_a_clock_that_stands_still_returns(self):
        e = self.env(io_kwargs=dict(clock=lambda: 0.0))
        (release, entered), = e.ops.hold_next(1)
        e.io.read_piece(e.drive_off(0), 2, 1)
        self.assertTrue(entered.wait(5))
        started = time.monotonic()
        self.assertFalse(e.io.wait_idle(0.2))
        self.assertLess(time.monotonic() - started, 3.0)
        release.set()
        self.assertTrue(e.io.wait_idle(5))


class InterfaceTests(unittest.TestCase):
    def test_signatures_are_the_frozen_ones_plus_one_optional_keyword(self):
        sig = inspect.signature
        read, write = sig(pio.P2PIO.read_piece), sig(pio.P2PIO.write_subrun)
        self.assertEqual(list(read.parameters), ['self', 'drive_off', 'bank', 'n_obj', 'first_obj'])
        self.assertEqual(list(write.parameters), ['self', 'drive_off', 'bank', 'n_obj', 'first_obj'])
        self.assertEqual(read.parameters['first_obj'].default, 0)
        self.assertEqual(write.parameters['first_obj'].default, 0)
        init = list(sig(pio.P2PIO.__init__).parameters)
        self.assertEqual(init[:5], ['self', 'links', 'runtime', 'config', 'staging_info'], 'the order the WP4 factory calls it in')
        for name, parameter in list(sig(pio.P2PIO.__init__).parameters.items())[5:]:
            self.assertEqual(parameter.kind, inspect.Parameter.KEYWORD_ONLY, name)
        self.assertEqual(sig(pio.P2PIO.__init__).parameters['allow_buffered'].default, False, 'production never passes it: the default refuses a buffered descriptor')
        for method in ('dma_canary', 'close', 'stats', 'wait_idle'):
            self.assertTrue(callable(getattr(pio.P2PIO, method)))

    def test_the_futures_are_concurrent_futures(self):
        import concurrent.futures
        e = fc.Env()
        try:
            self.assertIsInstance(e.io.read_piece(e.drive_off(0), 2, 1), concurrent.futures.Future)
            self.assertIsInstance(e.io.write_subrun(e.drive_off(0), 0, 1), concurrent.futures.Future)
            self.assertTrue(e.io.wait_idle(5))
        finally:
            e.close()

    def test_a_future_cannot_be_cancelled_the_dma_of_a_started_call_cannot_be_taken_back(self):
        e = fc.Env()
        try:
            (release, entered), = e.ops.hold_next(1)
            future = e.io.read_piece(e.drive_off(0), 2, 1)
            self.assertTrue(entered.wait(5))
            self.assertFalse(future.cancel())
            release.set()
            future.result(10)
        finally:
            e.close()


class ImportTests(unittest.TestCase):
    def test_importing_starts_nothing_opens_nothing_and_pulls_in_no_device_stack(self):
        code = (
            "import sys, threading, os\n"
            "sys.path.insert(0, %r)\n"
            "before = len(os.listdir('/proc/self/fd'))\n"
            "import ttkvp2p.cardlink, ttkvp2p.p2pio\n"
            "assert threading.active_count() == 1, threading.enumerate()\n"
            "assert len(os.listdir('/proc/self/fd')) == before\n"
            "for name in ('numpy', 'vllm', 'ttnn', 'torch', 'qwen_kv_read'):\n"
            "    assert name not in sys.modules, name\n"
            "print('clean')\n" % PACKAGE_ROOT)
        out = subprocess.run([sys.executable, '-I', '-c', code], capture_output=True, text=True, timeout=60)
        self.assertEqual((out.returncode, out.stdout.strip()), (0, 'clean'), out.stderr)

    def test_no_syntax_newer_than_python_3_10(self):
        for name in ('ttkvp2p/cardlink.py', 'ttkvp2p/p2pio.py', 'tests/fake_card.py', 'tests/test_cardlink.py', 'tests/test_p2pio.py', 'tests/test_p2pio_mutants.py',
                     'tests/test_wp4_wiring.py'):
            path = os.path.join(PACKAGE_ROOT, name)
            self.assertTrue(os.path.exists(path), '%s is a file of this work package and must exist' % name)
            with open(path) as handle:
                ast.parse(handle.read(), filename=name, feature_version=(3, 10))


if __name__ == '__main__':
    unittest.main()
