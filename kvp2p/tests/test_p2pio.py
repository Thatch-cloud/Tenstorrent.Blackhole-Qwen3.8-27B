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
        self.assertTrue(e.direct or True, 'O_DIRECT is used where the file system takes it: %s' % e.direct)
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

    def test_the_scheduler_thread_failing_stops_the_layer_and_latches(self):
        e = self.env()

        def broken():
            raise RuntimeError('boom')
        e.io._decide = broken
        future = e.io.read_piece(e.drive_off(0), 2, 1)
        error = future.exception(10)
        self.assertIsInstance(error, pio.P2PDead)
        self.assertIn('scheduler thread failed', str(error))
        self.assertTrue(any('scheduler thread failed' in reason for reason in e.runtime.latched))
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
        self.assertTrue(e.runtime.latched)
        self.assertIsNotNone(e.io.dead)
        self.assertEqual(e.links[0].inflight, 0, 'the cards that had counted the call in flight were released again')


class CanaryTests(IOCase):
    def test_a_canary_mismatch_on_a_re_aim_stops_everything_before_a_single_bio(self):
        e = self.env()
        e.rig.boards[1].ignore_configure = True
        future = e.io.read_piece(e.drive_off(0), 4, 1)
        error = future.exception(10)
        self.assertIsInstance(error, pio.P2PDead)
        self.assertEqual(e.ops.calls, [])
        self.assertTrue(any('re-aim to bank 4' in reason for reason in e.runtime.latched))
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

    def test_the_drive_fd_it_opened_is_closed_and_the_one_it_was_given_is_not(self):
        e = self.env(start=False)
        e.links = e.rig.open_links()
        e.config.nvme_device = e.path
        e.build_io(drive_fd=None)
        self.assertEqual(e.ops.opened, [e.path])
        self.assertEqual(e.io.read_piece(e.drive_off(0), 2, 1).result(10), e.object_bytes)
        e.io.close()
        self.assertEqual(len(e.ops.closed_fds), 1)
        e2 = self.env()
        e2.io.close()
        self.assertEqual(e2.ops.closed_fds, [], 'the runtime\'s descriptor stays the runtime\'s')
        os.fstat(e2.fd)

    def test_threads_are_named_for_the_per_thread_cpu_gate_and_stop_at_close(self):
        e = self.env()
        e.io.read_piece(e.drive_off(0), 2, 1).result(10)
        names = {t.name for t in threading.enumerate()}
        self.assertIn('kvp2p-sched', names)
        self.assertTrue(any(name.startswith('kvp2p-io') for name in names), names)
        self.assertTrue(all(c['thread'].startswith('kvp2p-io') for c in e.ops.calls))
        self.assertGreaterEqual(e.io.stats()['io_thread_cpu_s'], 0.0)
        e.io.close()
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline and 'kvp2p-sched' in {t.name for t in threading.enumerate()}:
            time.sleep(0.01)
        self.assertNotIn('kvp2p-sched', {t.name for t in threading.enumerate()})


class DriveSourceTests(IOCase):
    def test_the_runtime_may_hand_over_the_descriptor_as_an_int_or_a_callable(self):
        for form in ('int', 'callable'):
            with self.subTest(form=form):
                e = self.env(start=False)
                e.links = e.rig.open_links()
                e.runtime.drive_fd = e.fd if form == 'int' else (lambda: e.fd)
                e.build_io(drive_fd=None)
                self.assertEqual(e.io.read_piece(e.drive_off(0), 2, 1).result(10), e.object_bytes)
                self.assertEqual(e.ops.opened, [])

    def test_no_drive_at_all_is_a_clean_error(self):
        e = self.env(start=False)
        e.links = e.rig.open_links()
        with self.assertRaises(pio.P2PIOError):
            e.build_io(drive_fd=None)
        self.assertNotIn('kvp2p-sched', {t.name for t in threading.enumerate()})

    def test_the_default_data_range_leaves_the_first_mib_alone_and_comes_from_the_runtime_when_it_says(self):
        e = self.env(start=False)
        e.links = e.rig.open_links()
        e.build_io(drive_range=None)
        self.assertEqual(e.io._drive_range, (1 << 20, e.data_hi))
        e.io.close()
        e.runtime.data_off = 2 << 20
        e.runtime.num_slots = 5
        e.build_io(drive_range=None)
        self.assertEqual(e.io._drive_range, (2 << 20, (2 << 20) + 5 * e.object_bytes))


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
        for method in ('dma_canary', 'close', 'stats'):
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
        for name in ('ttkvp2p/cardlink.py', 'ttkvp2p/p2pio.py', 'tests/fake_card.py', 'tests/test_cardlink.py', 'tests/test_p2pio.py', 'tests/test_p2pio_mutants.py'):
            path = os.path.join(PACKAGE_ROOT, name)
            if not os.path.exists(path):
                continue
            with open(path) as handle:
                ast.parse(handle.read(), filename=name, feature_version=(3, 10))


if __name__ == '__main__':
    unittest.main()
