"""The TTP2PWorker job state machine, one invariant at a time (design 2.4: I1-I5, I8, I11, I13), driven by hand against the numpy
card and the file drive: no scheduler, so each test controls exactly when a bio returns, when an op completes and what the clock says."""

import errno
import threading
import time
import unittest

import numpy as np

from fakes import LEAD, OBJ, TENSORS
from harness import RigCase, WorkerKit

NULL_FREE = list(range(1, 9))


class Lifecycle(RigCase):
    def kit(self, **kwargs):
        rig = self.rig(with_scheduler=False, **kwargs)
        return rig, WorkerKit(rig)

    # ------------------------------------------------------------------------------------------------ the data path
    def test_a_store_then_a_load_through_the_worker_is_byte_exact(self):
        rig, kit = self.kit()
        source, target = [1, 2, 3, 4, 5, 6], [20, 21, 22, 23, 24, 25]
        kit.fill_blocks(source, 'data')
        kit.store(1, source)
        kit.pump_until_reported(1)
        self.assertTrue(kit.results[1].success)
        self.assertEqual(kit.results[1].transfer_size, 6 * OBJ)
        for i, slot in enumerate(kit.written[1][0]):                    # the drive holds the objects the cards held
            self.assertTrue(np.array_equal(rig.drive.slot_object(slot), kit.object_of(source[i])))
        kit.load(2, 1, target)
        kit.pump_until_reported(2)
        self.assertTrue(kit.results[2].success)
        for src, dst in zip(source, target):
            self.assertTrue(np.array_equal(kit.object_of(src), kit.object_of(dst)))
        self.assertEqual(sum(rig.runtime.allocator.inflight), 0)
        self.assertEqual(rig.worker.free_banks, {'spill': [0, 1], 'restore': [2, 3, 4, 5, 6, 7]})

    def test_a_job_larger_than_a_slot_runs_as_several_pieces_in_the_ring(self):
        rig, kit = self.kit(slot_objects=4)
        source = list(range(1, 22))                                     # 21 objects = 6 pieces over 2 spill banks
        kit.fill_blocks(source, 'big')
        kit.store(1, source)
        kit.pump_until_reported(1)
        self.assertTrue(kit.results[1].success)
        target = list(range(100, 121))
        kit.load(2, 1, target)
        kit.pump_until_reported(2)
        self.assertTrue(kit.results[2].success)
        for src, dst in zip(source, target):
            self.assertTrue(np.array_equal(kit.object_of(src), kit.object_of(dst)))
        reads = [c for c in rig.io.calls if c[0] == 'read']
        self.assertEqual({c[2] for c in reads} <= {2, 3, 4, 5, 6, 7}, True, 'loads only ever use restore banks')
        writes = [c for c in rig.io.calls if c[0] == 'write']
        self.assertEqual({c[2] for c in writes} <= {0, 1}, True, 'stores only ever use spill banks')

    def test_no_op_touches_the_canary_page_the_null_block_or_leaves_the_slot(self):
        rig, kit = self.kit(slot_objects=4)
        source = list(range(1, 14))
        kit.fill_blocks(source, 'bounds')
        kit.store(1, source)
        kit.pump_until_reported(1)
        kit.load(2, 1, list(range(50, 63)))
        kit.pump_until_reported(2)
        gathers = rig.card.names('gather') + rig.card.names('scatter')
        self.assertTrue(gathers)                                        # FakeOps asserts every offset and block; here, one more look
        for _seq, _name, info in gathers:
            self.assertGreaterEqual(info['offset'], LEAD + 4096)
            self.assertNotIn(0, info['blocks'])

    # ------------------------------------------------------------------------------------------------ I1
    def test_every_job_is_reported_exactly_once_and_then_tombstoned(self):
        rig, kit = self.kit()
        kit.fill_blocks([1, 2, 3], 'once')
        slots, gens = kit.store(1, [1, 2, 3])
        kit.pump_until_reported(1)
        for _ in range(40):
            kit.pump_once()
        self.assertEqual(kit.reports, {1: 1})
        self.assertIn(1, rig.worker.tombstones)
        # a late wait, a second submit of the same id and a retry: nothing is reported again
        rig.worker.wait({1})
        from vllm.v1.kv_offload.base import GPULoadStoreSpec
        from ttkvp2p.spec import NvmeLoadStoreSpec
        self.assertTrue(rig.worker.submit_store(1, GPULoadStoreSpec([1, 2, 3], [3], [0]), NvmeLoadStoreSpec(slots, gens)))
        for _ in range(40):
            kit.pump_once()
        self.assertEqual(kit.reports, {1: 1})
        self.assertEqual(rig.worker.active_job_ids, [])

    def test_a_failed_job_is_reported_once_even_when_its_bios_return_late_and_fail_again(self):
        rig, kit = self.kit()
        kit.use_fake_clock()
        kit.fill_blocks([1, 2, 3, 4], 'late')
        kit.store(1, [1, 2, 3, 4])
        kit.pump_until_reported(1)
        hold, = rig.io.hold_next()
        kit.load(2, 1, [30, 31, 32, 33])
        kit.pump_once()
        kit.advance(5.0)                                               # past the deadline (2 s + bytes / 1 GB/s)
        kit.pump_once()
        self.assertEqual((kit.results[2].success, kit.results[2].reason), (False, 'deadline'))
        rig.io.read_errors.append(OSError(errno.EIO, 'late media error'))
        hold.set()                                                      # the stuck bio now returns, and fails
        for _ in range(60):
            kit.pump_once()
            time.sleep(0.001)
        self.assertEqual(kit.reports[2], 1)
        self.assertEqual(sum(rig.runtime.allocator.inflight), 0)
        self.assertEqual(rig.worker.quarantined_banks, [])
        self.assertEqual(rig.runtime.bad, set(), 'a late error of a job that already failed on its deadline proves nothing')

    def test_the_advisory_flush_read_never_decides_anything(self):
        rig, kit = self.kit()
        kit.fill_blocks([1, 2], 'adv')
        kit.store(1, [1, 2])
        kit.pump_until_reported(1)

        def broken_flush():
            raise OSError(errno.EIO, 'the uncached window read failed')

        rig.worker.be.flush = broken_flush
        kit.load(2, 1, [30, 31])
        kit.pump_until_reported(2)
        self.assertTrue(kit.results[2].success, 'the digest decides, not the advisory read')
        self.assertTrue(np.array_equal(kit.object_of(1), kit.object_of(30)))

    # ------------------------------------------------------------------------------------------------ I8
    def test_wait_accepts_unknown_terminal_and_load_ids(self):
        rig, kit = self.kit()
        kit.fill_blocks([1, 2], 'w')
        kit.store(1, [1, 2])
        kit.pump_until_reported(1)
        rig.worker.wait({999, 1, 5, 7})                                 # unknown and terminal: no error, no effect
        hold, = rig.io.hold_next()
        kit.load(3, 1, [10, 11])
        kit.pump_once()
        rig.worker.wait({3})                                            # a load id: fail it, quarantine its slot
        self.assertEqual(rig.worker.quarantined_banks, [2])
        kit.pump_once()
        self.assertEqual((kit.results[3].success, kit.results[3].reason), (False, 'reset'))
        self.assertEqual(rig.worker.strikes, 0, 'a reset is not a strike')
        hold.set()
        kit.pump(lambda: not rig.worker.quarantined_banks)
        self.assertEqual(rig.worker.free_banks['restore'], [2, 3, 4, 5, 6, 7])

    def test_a_flushed_store_is_gathered_before_the_reallocating_forward(self):
        rig, kit = self.kit()
        original = [1, 2, 3]
        kit.fill_blocks(original, 'original')
        before = [kit.object_of(b).copy() for b in original]
        kit.store(1, original)                                          # submitted, not yet serviced: a budget-delayed gather
        self.assertEqual(rig.card.names('gather'), [])
        rig.worker.wait({1})                                            # jobs_to_flush: the blocks are about to be reallocated
        self.assertEqual(len(rig.card.names('gather')), TENSORS)        # one single-tensor gather per tensor, enqueued now
        self.assertEqual(len(rig.card.names('digest')), 1)
        rig.card.mark('forward-that-reallocates')
        kit.fill_blocks(original, 'someone else\'s KV')                 # the next owner's forward overwrites the blocks
        kit.pump_until_reported(1)
        self.assertTrue(kit.results[1].success)
        for i, slot in enumerate(kit.written[1][0]):
            self.assertTrue(np.array_equal(rig.drive.slot_object(slot), before[i]), 'the drive holds the OLD owner\'s KV')

    def test_without_the_flush_the_same_store_would_read_the_new_owners_kv(self):
        """The control for the test above: it is the flush, not luck, that keeps the stored bytes right."""
        rig, kit = self.kit()
        kit.fill_blocks([1, 2, 3], 'original')
        before = [kit.object_of(b).copy() for b in (1, 2, 3)]
        kit.store(1, [1, 2, 3])
        kit.fill_blocks([1, 2, 3], 'someone else\'s KV')                # the forward ran first; the gather had not
        kit.pump_until_reported(1)
        slot = kit.written[1][0][0]
        self.assertFalse(np.array_equal(rig.drive.slot_object(slot), before[0]))

    def test_a_flushed_store_with_no_free_spill_slot_fails_without_reading_its_source(self):
        rig, kit = self.kit(extra={'spill_banks': [0]})
        kit.fill_blocks([1, 2, 3, 4], 'x')
        rig.io.gate.clear()                                             # the drive is slow: the only spill bank stays busy
        kit.store(1, [1, 2])
        kit.pump(lambda: rig.worker.free_banks['spill'] == [], what='the spill bank taken')
        gathers_before = len(rig.card.names('gather'))
        kit.store(2, [3, 4])
        rig.worker.wait({2})
        kit.pump_once()
        self.assertEqual((kit.results[2].success, kit.results[2].reason), (False, 'no_slot'))
        self.assertEqual(len(rig.card.names('gather')), gathers_before, 'the source of the failed job was never read')
        self.assertEqual(rig.worker.strikes, 0)
        self.assertEqual(rig.runtime.state, 'AVAILABLE')
        rig.io.gate.set()
        kit.pump_until_reported(1)
        self.assertTrue(kit.results[1].success)

    # ------------------------------------------------------------------------------------------------ budgets
    def test_the_step_budget_limits_device_objects_and_idle_steps_are_unlimited(self):
        rig, kit = self.kit(slot_objects=8, extra={'device_budget_objects_per_step': 3})
        kit.fill_blocks(range(1, 9), 'b')
        kit.store(1, list(range(1, 9)))
        seen = []
        for _ in range(3):
            before = rig.worker.stats['objects_gathered']
            kit.pump_once()
            seen.append(rig.worker.stats['objects_gathered'] - before)
        self.assertEqual(seen, [3, 3, 2])
        kit.pump_until_reported(1)
        kit.fill_blocks(range(11, 19), 'c')
        kit.store(2, list(range(11, 19)))
        before = rig.worker.stats['objects_gathered']
        kit.pump_once(decode_active=False)                              # no decode batch: the budget does not apply
        self.assertEqual(rig.worker.stats['objects_gathered'] - before, 8)

    def test_a_budgeted_load_verifies_every_chunk(self):
        rig, kit = self.kit(slot_objects=8, extra={'device_budget_objects_per_step': 3})
        source = list(range(1, 9))
        kit.fill_blocks(source, 'chunks')
        kit.store(1, source)
        kit.pump_until_reported(1)
        kit.load(2, 1, list(range(40, 48)))
        scatters = []
        while 2 not in kit.results:
            before = rig.worker.stats['objects_scattered']
            kit.pump_once()
            scatters.append(rig.worker.stats['objects_scattered'] - before)
            time.sleep(0.001)
        self.assertTrue(kit.results[2].success)
        self.assertTrue(all(n <= 3 for n in scatters))
        self.assertEqual(sum(scatters), 8)
        for src, dst in zip(source, range(40, 48)):
            self.assertTrue(np.array_equal(kit.object_of(src), kit.object_of(dst)))

    # ------------------------------------------------------------------------------------------------ I13 deadlines
    def test_the_deadline_scales_with_the_bytes_and_is_decided_at_a_service_point(self):
        rig, kit = self.kit(extra={'load_deadline_base_s': 1.0, 'load_deadline_gbps': 0.001})
        kit.use_fake_clock()
        kit.fill_blocks(range(1, 9), 'd')
        kit.store(1, list(range(1, 9)))
        kit.pump_until_reported(1)
        rig.io.gate.clear()
        kit.load(2, 1, list(range(30, 38)))
        kit.pump_once()
        # 8 objects x 5,120 B at 0.001 GB/s = 0.04096 s on top of the 1 s base
        kit.advance(1.040)
        kit.pump_once()
        self.assertNotIn(2, kit.results)
        kit.advance(0.002)                                              # past the deadline, but nothing has serviced yet
        self.assertEqual(rig.worker.active_job_ids, [2])
        kit.pump_once()
        self.assertEqual((kit.results[2].success, kit.results[2].reason), (False, 'deadline'))
        self.assertEqual(rig.runtime.bad, set(), 'a deadline never marks the keys BAD')
        rig.io.gate.set()

    # ------------------------------------------------------------------------------------------------ I2
    def test_a_scatter_is_refused_after_the_job_failed(self):
        rig, kit = self.kit(slot_objects=4, workers=2)
        source = list(range(1, 9))                                      # two pieces of four
        kit.fill_blocks(source, 'refuse')
        kit.store(1, source)
        kit.pump_until_reported(1)
        slots = kit.written[1][0]
        rig.drive.corrupt(slots[1])                                     # piece 0 cannot verify, however often it is re-read
        gate0, gate1 = rig.io.hold_next(2)                              # piece 0's read passes (hold 0 set), piece 1's is held
        gate0.set()
        target = list(range(40, 48))
        kit.load(2, 1, target)
        kit.pump(lambda: 2 in kit.results, what='the load failed')
        self.assertEqual((kit.results[2].success, kit.results[2].reason), (False, 'digest'))
        piece1_blocks = target[4:]
        scatters_for_piece1 = [e for e in rig.card.names('scatter') if set(e[2]['blocks']) & set(piece1_blocks)]
        self.assertEqual(scatters_for_piece1, [], 'no scatter for the second piece was enqueued after (or before) the failure')
        count = len(rig.card.names('scatter'))
        gate1.set()                                                     # its bio lands now, into a quarantined slot
        kit.pump(lambda: not rig.worker.quarantined_banks)
        self.assertEqual(len(rig.card.names('scatter')), count, 'a landed piece of a failed job is never scattered')
        self.assertEqual(rig.runtime.bad, {(slots[1], kit.written[1][1][1])}, 'only the object that still disagrees after one re-read goes BAD')
        self.assertEqual(kit.reports[2], 1)

    # ------------------------------------------------------------------------------------------------ I3
    def test_a_quarantined_slot_is_not_reused_until_its_last_bio_returns(self):
        rig, kit = self.kit(extra={'restore_banks': [2]})
        kit.use_fake_clock()
        kit.fill_blocks([1, 2, 3, 4], 'q')
        kit.store(1, [1, 2])
        kit.store(5, [3, 4])
        kit.pump_until_reported(1, 5)
        snapshot = [t.copy() for t in rig.card.kv]
        hold, = rig.io.hold_next()
        kit.load(2, 1, [30, 31])
        kit.pump_once()
        kit.advance(5.0)
        kit.pump_once()
        self.assertEqual(kit.results[2].reason, 'deadline')
        self.assertEqual(rig.worker.quarantined_banks, [2])
        self.assertEqual(rig.worker.free_banks['restore'], [])
        # a new load needs a restore slot and the only one is quarantined: it waits, it does not borrow
        kit.load(3, 5, [34, 35])
        for _ in range(10):
            kit.pump_once()
            time.sleep(0.001)
        self.assertNotIn(3, kit.results)
        self.assertEqual(len([c for c in rig.io.calls if c[0] == 'read']), 1)
        hold.set()                                                      # the late DMA lands in the quarantined slot only
        kit.pump_until_reported(3)
        self.assertTrue(kit.results[3].success)
        for before, after in zip(snapshot, rig.card.kv):
            changed = np.nonzero((before != after).any(axis=(0, 2)))[0]
            self.assertEqual(set(changed.tolist()), {34, 35}, 'KV changed only where the second load wrote')

    def test_a_slot_with_a_device_sequence_still_queued_is_held_back_when_its_job_fails(self):
        rig, kit = self.kit()
        kit.fill_blocks([1, 2, 3], 'q2')
        rig.card.hold = True                                            # the gather and its digest are enqueued but not done
        kit.store(1, [1, 2, 3])
        kit.pump_once()
        self.assertEqual(len(rig.card.names('gather')), TENSORS)
        rig.runtime.latch('control off')                                # the job is failed, no bio was ever issued
        kit.pump_once()
        self.assertEqual(kit.results[1].reason, 'latched')
        self.assertEqual(rig.worker.quarantined_banks, [0], 'the queue may still read the staging slot')
        self.assertEqual(rig.io.calls, [])
        rig.card.release()
        kit.pump_once()
        self.assertEqual(rig.worker.quarantined_banks, [])

    def test_a_device_op_throw_quarantines_the_slot_behind_a_trailing_event_and_latches(self):
        rig, kit = self.kit()
        kit.fill_blocks([1, 2, 3], 't')
        rig.card.fail_plan['gather'] = 5                               # the fifth single-tensor gather throws: a half-enqueued sequence
        rig.card.hold = True                                           # and nothing completes until we say so
        kit.store(1, [1, 2, 3])
        kit.pump_once()
        self.assertEqual((kit.results[1].success, kit.results[1].reason), (False, 'throw'))
        self.assertEqual(rig.runtime.state, 'LATCHED')
        self.assertEqual(rig.worker.quarantined_banks, [0], 'the half-enqueued slot is out of circulation')
        ops_after = len(rig.card.log)
        events = rig.card.names('event')
        self.assertEqual(len(events), 1, 'exactly one trailing record_event was enqueued after the throw')
        kit.store(2, [1, 2])                                           # after a latch: failed at once, no device op
        kit.pump_once()
        self.assertEqual((kit.results[2].success, kit.results[2].reason), (False, 'latched'))
        self.assertEqual(len(rig.card.log), ops_after)
        rig.card.release()                                              # the trailing event completes
        kit.pump_once()
        self.assertEqual(rig.worker.quarantined_banks, [])
        self.assertIn(0, rig.worker.free_banks['spill'])

    def test_a_trailing_event_that_cannot_be_enqueued_loses_the_slot_for_good(self):
        rig, kit = self.kit()
        kit.fill_blocks([1, 2], 't')
        rig.card.fail_plan['gather'] = 2
        rig.card.fail_plan['event'] = 1
        kit.store(1, [1, 2])
        kit.pump_once()
        self.assertEqual(kit.results[1].reason, 'throw')
        self.assertEqual(rig.worker.quarantined_banks, [0])
        for _ in range(5):
            kit.pump_once()
        self.assertEqual(rig.worker.quarantined_banks, [0], 'never released: nothing proves the queue is clear')

    # ------------------------------------------------------------------------------------------------ I5
    def test_a_slot_that_moved_on_is_not_read(self):
        rig, kit = self.kit()
        kit.fill_blocks([1, 2], 's')
        kit.store(1, [1, 2])
        kit.pump_until_reported(1)
        slots, gens, _ = kit.written[1]
        reads_before = len(rig.io.calls)
        kit.load(2, 1, [30, 31])
        rig.runtime.allocator.gen[slots[1]] += 1                        # reset_cache, TRIM and a new owner before the bio started
        kit.pump_until_reported(2)
        self.assertEqual((kit.results[2].success, kit.results[2].reason), (False, 'stale'))
        self.assertEqual(len(rig.io.calls), reads_before, 'no read of a reallocated slot')
        self.assertEqual(rig.runtime.bad, set())
        self.assertEqual(rig.worker.strikes, 0)
        self.assertEqual(sum(rig.runtime.allocator.inflight), 0)

    def test_inflight_counts_cover_every_bio_and_return_to_zero(self):
        rig, kit = self.kit(slot_objects=4)
        kit.fill_blocks(range(1, 10), 'f')
        kit.store(1, list(range(1, 10)))
        peak = 0
        while 1 not in kit.results:
            kit.pump_once()
            peak = max(peak, sum(rig.runtime.allocator.inflight))
            time.sleep(0.0005)
        self.assertEqual(sum(rig.runtime.allocator.inflight), 0)
        self.assertEqual(rig.runtime.allocator.begin_refused, 0)

    # ------------------------------------------------------------------------------------------------ I13 proof
    def test_a_digest_mismatch_is_re_read_once_and_a_transient_one_does_not_mark_anything_bad(self):
        rig, kit = self.kit()
        kit.fill_blocks([1, 2, 3], 'r')
        kit.store(1, [1, 2, 3])
        kit.pump_until_reported(1)
        rig.runtime.faults['digest'] += 1                               # the first verification sees a wrong digest, the re-read is right
        kit.load(2, 1, [30, 31, 32])
        kit.pump_until_reported(2)
        self.assertTrue(kit.results[2].success)
        self.assertEqual(rig.worker.stats['reread'], 1)
        self.assertEqual(rig.runtime.bad, set())
        self.assertEqual(len([c for c in rig.io.calls if c[0] == 'read']), 2, 'the re-read went back to the drive')

    def test_a_torn_landing_is_cured_by_the_re_read_because_it_fetches_the_bytes_again(self):
        rig, kit = self.kit()
        kit.fill_blocks([1, 2, 3], 'torn')
        kit.store(1, [1, 2, 3])
        kit.pump_until_reported(1)
        rig.io.tear_next_reads = 1                                      # the first landing is torn; the drive itself is fine
        kit.load(2, 1, [30, 31, 32])
        kit.pump_until_reported(2)
        self.assertTrue(kit.results[2].success)
        self.assertEqual(rig.worker.stats['digest_mismatch'], 1)
        self.assertEqual(rig.runtime.bad, set(), 'one mismatch is not proof')
        for src, dst in zip([1, 2, 3], [30, 31, 32]):
            self.assertTrue(np.array_equal(kit.object_of(src), kit.object_of(dst)))
        rig.io.tear_next_reads = 2                                      # torn every time: the second mismatch is proof
        kit.load(3, 1, [33, 34, 35])
        kit.pump_until_reported(3)
        self.assertEqual(kit.results[3].reason, 'digest')
        self.assertEqual(len(rig.runtime.bad), 1)

    def test_a_media_error_marks_the_run_bad_and_other_errors_do_not(self):
        rig, kit = self.kit()
        kit.fill_blocks([1, 2, 3], 'm')
        slots, gens = kit.store(1, [1, 2, 3])
        kit.pump_until_reported(1)
        rig.io.read_errors.append(OSError(errno.EIO, 'media'))
        kit.load(2, 1, [30, 31, 32])
        kit.pump_until_reported(2)
        self.assertEqual(kit.results[2].reason, 'eio')
        self.assertEqual(rig.runtime.bad, set(zip(slots, gens)))
        rig.runtime.bad.clear()
        rig.io.read_errors.append(OSError(errno.EAGAIN, 'busy'))
        kit.load(3, 1, [33, 34, 35])
        kit.pump_until_reported(3)
        self.assertEqual(kit.results[3].reason, 'io')
        self.assertEqual(rig.runtime.bad, set(), 'only EIO and EILSEQ are proof of a bad medium')

    def test_a_short_transfer_fails_the_job(self):
        rig, kit = self.kit()
        kit.fill_blocks([1, 2], 'sh')
        kit.store(1, [1, 2])
        kit.pump_until_reported(1)
        rig.io.short_next = 1
        kit.load(2, 1, [30, 31])
        kit.pump_until_reported(2)
        self.assertEqual(kit.results[2].reason, 'short')

    # ------------------------------------------------------------------------------------------------ latch
    def test_three_consecutive_strikes_latch_and_a_success_in_between_does_not(self):
        rig, kit = self.kit()
        kit.fill_blocks([1, 2], 'l')
        kit.store(1, [1, 2])
        kit.pump_until_reported(1)

        def failing_load(job_id):
            rig.io.read_errors.append(OSError(errno.EIO, 'media'))
            kit.load(job_id, 1, [30 + job_id, 40 + job_id])
            kit.pump_until_reported(job_id)
            self.assertEqual(kit.results[job_id].reason, 'eio')

        failing_load(10)
        failing_load(11)
        self.assertEqual(rig.worker.strikes, 2)
        kit.load(12, 1, [50, 51])                                       # a success resets the count
        kit.pump_until_reported(12)
        self.assertTrue(kit.results[12].success)
        self.assertEqual(rig.worker.strikes, 0)
        failing_load(13)
        failing_load(14)
        self.assertEqual(rig.runtime.state, 'AVAILABLE')
        rig.worker.wait({999})
        failing_load(15)                                                # three in a row
        self.assertEqual(rig.runtime.state, 'LATCHED')
        self.assertIn('3 consecutive', rig.runtime.latch_reason)

    def test_neutral_failures_neither_count_nor_reset(self):
        rig, kit = self.kit()
        kit.fill_blocks([1, 2], 'n')
        kit.store(1, [1, 2])
        kit.pump_until_reported(1)
        for job_id in (10, 11):
            rig.io.read_errors.append(OSError(errno.EIO, 'media'))
            kit.load(job_id, 1, [30 + job_id, 40 + job_id])
            kit.pump_until_reported(job_id)
        hold, = rig.io.hold_next()
        kit.load(12, 1, [60, 61])
        kit.pump_once()
        rig.worker.wait({12})                                           # a reset-flush: neutral
        kit.pump_once()
        self.assertEqual(kit.results[12].reason, 'reset')
        self.assertEqual(rig.worker.strikes, 2)
        hold.set()
        self.assertEqual(rig.runtime.state, 'AVAILABLE')

    def test_a_device_lost_errno_latches_at_once(self):
        rig, kit = self.kit()
        kit.fill_blocks([1, 2], 'd')
        kit.store(1, [1, 2])
        kit.pump_until_reported(1)
        rig.io.read_errors.append(OSError(errno.ENODEV, 'gone'))
        kit.load(2, 1, [30, 31])
        kit.pump_until_reported(2)
        self.assertEqual(rig.runtime.state, 'LATCHED')

    def test_no_completion_for_the_stall_window_latches_and_fails_what_is_in_flight(self):
        rig, kit = self.kit()
        kit.use_fake_clock()
        kit.fill_blocks([1, 2, 3, 4], 'st')
        kit.store(1, [1, 2])
        kit.pump_until_reported(1)
        rig.io.gate.clear()                                             # the drive stops answering
        kit.store(2, [3, 4])
        kit.pump(lambda: rig.worker.free_banks['spill'] != [0, 1], what='the write issued')
        kit.advance(19.0)
        kit.pump_once()
        self.assertEqual(rig.runtime.state, 'AVAILABLE')
        kit.advance(2.0)
        kit.pump_once()
        self.assertEqual(rig.runtime.state, 'LATCHED')
        self.assertIn('no completion', rig.runtime.latch_reason)
        self.assertEqual(kit.results[2].reason, 'latched')
        self.assertEqual(len(rig.worker.quarantined_banks), 1, 'the bank with a write in flight is held back')
        rig.io.gate.set()
        kit.pump(lambda: not rig.worker.quarantined_banks)

    def test_a_latched_worker_fails_new_jobs_at_once_with_no_device_op(self):
        rig, kit = self.kit()
        kit.fill_blocks([1, 2], 'x')
        rig.runtime.latch('control off')
        log = len(rig.card.log)
        kit.store(1, [1, 2])
        kit.pump_once()
        self.assertEqual((kit.results[1].success, kit.results[1].reason), (False, 'latched'))
        self.assertEqual(len(rig.card.log), log)
        self.assertEqual(rig.io.calls, [])

    def test_in_flight_loads_are_failed_at_the_next_service_point_after_a_latch(self):
        rig, kit = self.kit()
        kit.fill_blocks([1, 2], 'x')
        kit.store(1, [1, 2])
        kit.pump_until_reported(1)
        hold, = rig.io.hold_next()
        kit.load(2, 1, [30, 31])
        kit.pump_once()
        rig.runtime.latch('control off')                                # the kill switch
        kit.pump_once()
        self.assertEqual((kit.results[2].success, kit.results[2].reason), (False, 'latched'))
        hold.set()
        kit.pump(lambda: not rig.worker.quarantined_banks)

    # ------------------------------------------------------------------------------------------------ bad specs
    def test_malformed_jobs_fail_at_submit_and_are_still_reported_once(self):
        rig, kit = self.kit()
        from vllm.v1.kv_offload.base import GPULoadStoreSpec
        from ttkvp2p.spec import NvmeLoadStoreSpec
        worker = rig.worker
        self.assertTrue(worker.submit_store(1, GPULoadStoreSpec([1, 2], [2], [0]), NvmeLoadStoreSpec([5], [1])))               # 2 blocks, 1 slot
        self.assertTrue(worker.submit_store(2, GPULoadStoreSpec([0], [1], [0]), NvmeLoadStoreSpec([5], [1])))                  # the null block
        self.assertTrue(worker.submit_store(3, GPULoadStoreSpec([10 ** 6], [1], [0]), NvmeLoadStoreSpec([5], [1])))            # past the pool
        self.assertTrue(worker.submit_load(4, NvmeLoadStoreSpec([5], [1], None), GPULoadStoreSpec([1], [1], [0])))             # no expected digests
        self.assertTrue(worker.submit_load(5, NvmeLoadStoreSpec([5], [1], np.zeros((2, 4), np.uint64)), GPULoadStoreSpec([1], [1], [0])))
        poisoned = NvmeLoadStoreSpec([6], [1])
        poisoned.poison = 'I6: test'
        self.assertTrue(worker.submit_store(6, GPULoadStoreSpec([1], [1], [0]), poisoned))
        kit.pump(lambda: len(kit.results) == 6)
        self.assertTrue(all(r.reason == 'bad_spec' for r in kit.results.values()), kit.results)
        self.assertEqual(rig.card.log, [])
        self.assertEqual(rig.worker.strikes, 0)

    # ------------------------------------------------------------------------------------------------ test-only fault hooks
    def test_the_runtime_fault_hooks_drive_the_same_paths(self):
        rig, kit = self.kit(extra={'load_deadline_base_s': 0.05})
        kit.fill_blocks([1, 2, 3], 'h')
        slots, gens = kit.store(1, [1, 2, 3])
        kit.pump_until_reported(1)
        rig.runtime.faults['eio'] += 1
        kit.load(2, 1, [30, 31, 32])
        kit.pump_until_reported(2)
        self.assertEqual(kit.results[2].reason, 'eio')
        rig.runtime.bad.clear()
        rig.runtime.faults['stall'] += 1
        kit.load(3, 1, [30, 31, 32])
        kit.pump_until_reported(3)
        self.assertEqual(kit.results[3].reason, 'deadline')
        self.assertEqual(rig.worker.quarantined_banks, [], 'a stalled piece issued no bio, so nothing is outstanding')
        rig.runtime.faults['throw'] += 1
        kit.load(4, 1, [30, 31, 32])
        kit.pump_until_reported(4)
        self.assertEqual(kit.results[4].reason, 'throw')
        self.assertEqual(rig.runtime.state, 'LATCHED')

    # ------------------------------------------------------------------------------------------------ shutdown (2.9)
    def test_shutdown_fails_jobs_waits_for_bios_then_closes_and_releases_the_runtime(self):
        rig, kit = self.kit(extra={'shutdown_drain_s': 5.0})
        kit.fill_blocks([1, 2], 'z')
        kit.store(1, [1, 2])
        kit.pump_until_reported(1)
        hold, = rig.io.hold_next()
        kit.load(2, 1, [30, 31])
        kit.pump_once()
        timer = threading.Timer(0.2, hold.set)
        timer.start()
        self.addCleanup(timer.cancel)
        started = time.monotonic()
        rig.worker.shutdown()
        self.assertGreaterEqual(time.monotonic() - started, 0.15, 'it waited for the bio')
        self.assertEqual(rig.factory.closed, 1)
        self.assertEqual(rig.runtime.released, ['worker'])
        self.assertEqual(sum(rig.runtime.allocator.inflight), 0)

    def test_shutdown_does_not_unmap_while_a_bio_is_still_out(self):
        rig, kit = self.kit(extra={'shutdown_drain_s': 0.1})
        kit.fill_blocks([1, 2], 'z')
        kit.store(1, [1, 2])
        kit.pump_until_reported(1)
        hold, = rig.io.hold_next()
        kit.load(2, 1, [30, 31])
        kit.pump_once()
        rig.worker.shutdown()
        self.assertEqual(rig.factory.closed, 0, 'no exit with DMA in flight on purpose: the mappings stay')
        hold.set()


if __name__ == '__main__':
    unittest.main()


class IdleWait(RigCase):
    def test_a_zero_token_step_waits_for_a_completion_instead_of_spinning(self):
        rig = self.rig(with_scheduler=False, extra={'idle_wait_ms': 3000.0})
        kit = WorkerKit(rig)
        kit.fill_blocks([1, 2], 'idle')
        kit.store(1, [1, 2])
        kit.pump_until_reported(1)
        hold, = rig.io.hold_next()
        kit.load(2, 1, [30, 31])
        kit.pump_once()                                                 # the read is out
        started = time.monotonic()
        kit.pump_once()                                                 # a normal step never waits (the wait would be 3 s)
        self.assertLess(time.monotonic() - started, 1.0)
        timer = threading.Timer(0.05, hold.set)
        timer.start()
        self.addCleanup(timer.cancel)
        started = time.monotonic()
        kit.pump_once(idle=True)                                        # a zero-token step: blocks until the bio returns (about 50 ms), not 3 s
        elapsed = time.monotonic() - started
        self.assertGreater(elapsed, 0.02)
        self.assertLess(elapsed, 2.0)
        kit.pump_until_reported(2)
        started = time.monotonic()
        kit.pump_once(idle=True)                                        # nothing in flight: no wait at all
        self.assertLess(time.monotonic() - started, 1.0)


class SingleTerminalReport(RigCase):
    def test_a_second_terminal_decision_for_a_job_reports_nothing(self):
        """I1 at the API of the state machine: whatever path decides a job's fate a second time, only the first report exists."""
        rig = self.rig(with_scheduler=False)
        kit = WorkerKit(rig)
        worker = rig.worker
        kit.fill_blocks([1, 2, 3, 4], 'twice')
        kit.store(1, [1, 2])
        kit.pump_until_reported(1)
        done_job = worker._jobs.get(1)
        self.assertIsNone(done_job, 'a reported job is forgotten')
        hold, = rig.io.hold_next()
        kit.store(2, [3, 4])
        kit.pump(lambda: sum(rig.runtime.allocator.inflight) > 0, what='the write is out')
        job = worker._jobs[2]
        worker._fail(job, 'eio')
        worker._fail(job, 'deadline')                       # a second failure for the same job
        worker._finish_ok(job)                              # a late "success"
        worker._queue_result(job)                           # and a direct second report
        hold.set()
        kit.pump_until_reported(2)
        for _ in range(20):
            kit.pump_once()
        self.assertEqual(kit.reports, {1: 1, 2: 1})
        self.assertEqual(kit.results[2].reason, 'eio', 'the first decision stands')

    def test_the_connector_worker_drops_a_duplicate_report_before_the_scheduler_could_trip_on_it(self):
        from ttkvp2p.worker import TTTransferResult

        rig = self.rig(with_scheduler=False)
        conn = rig.worker_conn.connector_worker

        class Twice:
            calls = 0

            def get_finished(self):
                self.calls += 1
                if self.calls == 1:
                    return [TTTransferResult(5, True), TTTransferResult(5, True)]
                return [TTTransferResult(5, False, reason='late')]

            def shutdown(self):
                pass

        real, conn.worker = conn.worker, Twice()
        try:
            conn._load_jobs[5] = 'req-5'
            finished = conn.get_finished(set())
            self.assertEqual(finished, (set(), {'req-5'}))
            first = conn.build_connector_worker_meta()                  # step n's output
            self.assertEqual(first.completed_jobs, {5: 1})
            self.assertEqual(conn.get_finished(set()), (set(), set()))  # step n + 1: the worker says it again, now as a failure
            self.assertIsNone(conn.build_connector_worker_meta(), 'a second completion of the id would be a KeyError in the scheduler')
            self.assertEqual(conn.take_failed_recving(), set())
        finally:
            conn.worker = real


class InternalErrors(RigCase):
    """A bug in the state machine is latched and survived: the engine never sees the exception, every job is still reported once."""

    def test_an_unexpected_exception_in_a_service_pass_latches_the_tier_and_fails_the_jobs(self):
        rig = self.rig(with_scheduler=False)
        rig.expected_internal_errors = 1
        kit = WorkerKit(rig)
        kit.fill_blocks([1, 2], 'bug')
        kit.store(1, [1, 2])
        kit.store(2, [1, 2])

        def boom():
            raise ZeroDivisionError('a bug')

        rig.worker._advance = boom
        kit.pump_once()                                                 # must not raise
        self.assertEqual(rig.runtime.state, 'LATCHED')
        self.assertIn('internal error', rig.runtime.latch_reason)
        kit.pump_until_reported(1, 2)
        self.assertEqual({kit.results[1].reason, kit.results[2].reason}, {'latched'})
        self.assertEqual(rig.worker.stats['internal_error'], 1)
        for _ in range(5):
            kit.pump_once()
        self.assertEqual(kit.reports, {1: 1, 2: 1})

    def test_an_unexpected_exception_in_submit_or_wait_never_reaches_the_caller(self):
        rig = self.rig(with_scheduler=False)
        rig.expected_internal_errors = None
        kit = WorkerKit(rig)
        kit.fill_blocks([1, 2], 'bug2')

        def boom(*args, **kwargs):
            raise KeyError('a bug')

        rig.worker._make_job = boom
        self.assertTrue(rig.worker.submit_store(1, None, None))         # always True, even when it could not even read the spec
        self.assertEqual(rig.runtime.state, 'LATCHED')
        rig.worker.wait({1, 2, 3})                                      # and wait() accepts anything
        rig.worker._reap_runs = boom
        rig.worker.wait({1})                                            # a second failure inside wait
        kit.pump_once()
        self.assertGreaterEqual(rig.worker.stats['internal_error'], 3)
