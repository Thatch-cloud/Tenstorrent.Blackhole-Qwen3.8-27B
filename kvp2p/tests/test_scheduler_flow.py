"""The worker and connector inside vLLM's real scheduler: failures become recomputes (never a hang, never a defer loop), a reset
during a load cannot deliver stale KV, a mis-ordered manager is caught, and the fault matrix (EIO, digest, stall, throw) ends in a
finished request with exactly the right KV and a tier in the state the design says."""

import errno
import unittest
from types import SimpleNamespace


from harness import BLOCK, RigCase, tokens

PROMPT = 2300
FULL_BLOCKS = PROMPT // BLOCK
OBJECTS = FULL_BLOCKS + 5 * 8


class Flow(RigCase):
    def stored_rig(self, name='p', **kwargs):
        rig = self.rig(**kwargs)
        prompt = tokens(PROMPT, name)
        rig.add('seed', prompt)
        rig.drain()
        self.assertEqual(rig.manager.written(), OBJECTS)
        rig.reset_device_cache()                        # the device prefix cache is empty; the drive still has the session
        return rig, prompt

    # ------------------------------------------------------------------------------------------------ failure, then recompute
    def test_an_eio_fails_the_load_the_request_recomputes_and_nothing_defers_forever(self):
        rig, prompt = self.stored_rig()
        rig.io.read_errors.append(OSError(errno.EIO, 'media error'))
        rig.add('r2', prompt)
        rig.finish('r2', limit=400)                     # a defer loop would exhaust the limit
        self.assertEqual(rig.worker.stats['fail_eio'], 1)
        self.assertEqual(rig.worker.stats['done_load'], 0)
        self.assertEqual(len(rig.loads), 1, 'the recompute did not look up and load again')
        self.assertTrue(rig.runtime.bad, 'the keys behind the media error are BAD')
        # the BAD entries are gone from the index; the others are untouched
        self.assertLess(rig.manager.written(), OBJECTS)
        self.assertEqual(rig.assert_drive_exact(), rig.manager.written())
        rig.drain()
        rig.reset_device_cache()
        rig.add('r3', prompt)
        rig.finish('r3', limit=400)
        self.assertEqual(len(rig.loads), 1, 'a BAD prefix key misses, so there is nothing left to load')
        rig.drain()
        self.assertEqual(sum(rig.runtime.allocator.inflight), 0)

    def test_a_deadline_keeps_the_keys_valid_and_the_requests_recompute_does_not_look_up_again(self):
        rig, prompt = self.stored_rig(extra={'load_deadline_base_s': 0.05})
        hold, = rig.io.hold_next()
        rig.add('r2', prompt)
        rig.finish('r2', limit=2000)
        self.assertEqual(rig.worker.stats['fail_deadline'], 1)
        self.assertGreater(rig.manager.lookups_refused_no_retry, 0, 'the no-retry flag turned the recompute\'s lookups into misses')
        self.assertEqual(len(rig.loads), 1)
        self.assertEqual(rig.runtime.bad, set())
        self.assertEqual(rig.manager.written(), OBJECTS, 'a deadline never invalidates the keys')
        hold.set()                                      # the stuck bio returns into a quarantined slot
        rig.drain()
        rig.run_until(lambda: not rig.worker.quarantined_banks, what='the quarantine released')
        rig.reset_device_cache()
        rig.add('r3', prompt)                           # another request, no flag: the same keys load fine
        rig.finish('r3')
        self.assertEqual(rig.worker.stats['done_load'], 1)
        self.assertEqual(len(rig.loads), 2)

    def test_a_transient_digest_mismatch_is_re_read_and_the_load_succeeds(self):
        rig, prompt = self.stored_rig()
        rig.runtime.faults['digest'] += 1
        rig.add('r2', prompt)
        rig.finish('r2')
        self.assertEqual(rig.worker.stats['reread'], 1)
        self.assertEqual(rig.worker.stats['done_load'], 1)
        self.assertEqual(rig.runtime.bad, set())

    def test_a_persistent_digest_mismatch_marks_exactly_that_object_bad_and_recomputes(self):
        rig, prompt = self.stored_rig()
        victim_key = next(k for k in rig.manager.index if int.from_bytes(k[-4:], 'big') == 0)
        entry = rig.manager.index[victim_key]
        rig.drive.corrupt(entry.slot)
        rig.add('r2', prompt)
        rig.finish('r2', limit=400)
        self.assertEqual(rig.worker.stats['fail_digest'], 1)
        self.assertEqual(rig.runtime.bad, {(entry.slot, entry.gen)})
        self.assertNotIn(victim_key, rig.manager.index, 'complete_load removed the BAD key')
        self.assertEqual(rig.manager.written(), OBJECTS - 1)

    # ------------------------------------------------------------------------------------------------ the fault matrix
    def test_a_device_op_throw_during_a_restore_latches_the_tier_and_the_engine_carries_on(self):
        rig, prompt = self.stored_rig()
        rig.card.fail_plan['scatter'] = 3
        rig.add('r2', prompt)
        rig.finish('r2', limit=400)
        self.assertEqual(rig.runtime.state, 'LATCHED')
        self.assertEqual(rig.worker.stats['fail_throw'], 1)
        stores_before = rig.worker.stats['submit_store']
        loads_before = len(rig.loads)
        rig.add('r3', tokens(PROMPT, 'fresh'))          # new work: no hits, no stores, the engine just serves it
        rig.add('r4', prompt)
        rig.finish('r3', 'r4', limit=400)
        self.assertEqual(rig.worker.stats['submit_store'], stores_before)
        self.assertEqual(len(rig.loads), loads_before)
        # the half-enqueued sequence of the throw is quarantined behind a trailing event, and released once that event completes
        rig.run_until(lambda: not rig.worker.quarantined_banks, what='the quarantine released')
        self.assertEqual(rig.worker.free_banks, {'spill': [0, 1], 'restore': [2, 3, 4, 5, 6, 7]}, 'every staging bank is back in circulation')

    def test_the_kill_switch_latches_a_load_in_flight_and_the_request_recomputes(self):
        rig, prompt = self.stored_rig()
        hold, = rig.io.hold_next()
        rig.add('r2', prompt)
        rig.run_until(lambda: any(c[0] == 'read' for c in rig.io.calls), what='the read issued')
        rig.runtime.latch('control off')
        rig.finish('r2', limit=400)
        self.assertEqual(rig.worker.stats['fail_latched'], 1)
        hold.set()
        rig.drain()
        rig.run_until(lambda: not rig.worker.quarantined_banks, what='the quarantine released')
        self.assertEqual(rig.manager.lookup(next(iter(rig.manager.index)), SimpleNamespace(get_state=lambda cls: None)).name, 'MISS')

    # ------------------------------------------------------------------------------------------------ reset_cache during a load
    def test_a_reset_during_a_load_cannot_deliver_stale_kv_and_the_slots_are_not_reused_under_a_bio(self):
        rig, prompt = self.stored_rig(workers=2)
        hold, = rig.io.hold_next()
        rig.add('r2', prompt)
        rig.run_until(lambda: any(c[0] == 'read' for c in rig.io.calls), what='the read issued')
        allocator = rig.runtime.allocator
        under_bio = [slot for slot, count in enumerate(allocator.inflight) if count]
        self.assertTrue(under_bio)
        self.assertTrue(rig.scheduler.reset_connector_cache())          # POST /reset_prefix_cache?reset_external=true
        self.assertEqual(len(rig.manager.index), 0, 'the reset emptied the index')
        for slot in under_bio:
            self.assertIn(slot, allocator.pending_free)
            self.assertNotIn(slot, allocator.free, 'a slot with a bio in flight is not allocatable (I5)')
        rig.add('r3', tokens(PROMPT, 'after the reset'))               # a new session stores while the old bio is still out
        rig.finish('r2', 'r3', limit=800)
        self.assertEqual(allocator.handed_out_in_flight, [])
        self.assertEqual(rig.worker.stats['fail_reset'], 1)
        hold.set()                                                      # the old bio lands into a quarantined slot
        rig.drain()
        rig.run_until(lambda: not rig.worker.quarantined_banks and sum(allocator.inflight) == 0, what='everything returned')
        self.assertEqual(rig.assert_drive_exact(), rig.manager.written())
        self.assertGreater(rig.manager.written(), 0)
        for slot in under_bio:
            self.assertTrue(slot in allocator.free or slot in {e.slot for e in rig.manager.index.values()})

    # ------------------------------------------------------------------------------------------------ an abort during a load
    def test_aborting_a_request_while_its_load_is_in_flight_frees_everything_once_the_load_lands(self):
        from vllm.v1.request import RequestStatus

        rig, prompt = self.stored_rig()
        hold, = rig.io.hold_next()
        rig.add('r2', prompt)
        rig.run_until(lambda: any(c[0] == 'read' for c in rig.io.calls), what='the read issued')
        rig.scheduler.finish_requests('r2', RequestStatus.FINISHED_ABORTED)
        for _ in range(5):
            rig.step()
        hold.set()                                                      # the load lands for a request that is gone
        rig.drain()
        rig.run_until(lambda: sum(rig.runtime.allocator.inflight) == 0 and not rig.worker.quarantined_banks, what='all returned')
        pool = rig.scheduler.kv_cache_manager.block_pool
        self.assertEqual(pool.get_num_free_blocks(), rig.num_blocks - 1, 'every block of the aborted request is free again')
        self.assertEqual(rig.manager.written(), OBJECTS, 'a finished load leaves the keys valid')
        rig.add('r3', prompt)                                           # and the session still restores
        rig.finish('r3')
        self.assertEqual(rig.worker.stats['done_load'], 2)

    # ------------------------------------------------------------------------------------------------ I6
    def test_a_manager_that_returns_the_keys_in_another_order_is_caught_and_stores_nothing_wrong(self):
        rig = self.rig()
        rig.manager.shuffle_keys = True
        rig.add('r1', tokens(PROMPT, 'shuffled'))
        rig.drain()
        self.assertGreater(rig.connector_scheduler.misalignments, 0)
        self.assertEqual(rig.worker.stats['done_store'], 0)
        self.assertGreater(rig.worker.stats['fail_bad_spec'], 0)
        self.assertEqual(rig.manager.written(), 0, 'no key points at bytes that are not its own')
        self.assertEqual(rig.runtime.state, 'AVAILABLE', 'a refused misaligned job is not a strike')
        rig.manager.shuffle_keys = False                                # the same engine, a manager that keeps the order
        rig.add('r2', tokens(PROMPT, 'ordered'))
        rig.drain()
        self.assertEqual(rig.manager.written(), OBJECTS)
        self.assertEqual(rig.assert_drive_exact(), OBJECTS)

    def test_the_block_hash_check_catches_a_job_that_reads_another_keys_block(self):
        from ttkvp2p.connector import TTP2PConnectorScheduler
        keys = [b'a' * 32 + (0).to_bytes(4, 'big'), b'b' * 32 + (0).to_bytes(4, 'big')]
        pool = SimpleNamespace(blocks={1: SimpleNamespace(block_hash=keys[1]), 2: SimpleNamespace(block_hash=None)})
        check = SimpleNamespace(_req_status={}, _block_pool=pool)
        job = SimpleNamespace(req_id='nobody', src_spec=SimpleNamespace(block_ids=[1, 2]), dst_spec=SimpleNamespace(slot_ids=[3, 4]))
        problem = TTP2PConnectorScheduler._misalignment(check, job, keys)
        self.assertIn('hashed for another key', problem)
        pool.blocks[1] = SimpleNamespace(block_hash=keys[0])
        self.assertIsNone(TTP2PConnectorScheduler._misalignment(check, job, keys))

    # ------------------------------------------------------------------------------------------------ I9
    def test_stores_are_submitted_through_get_transfer_results(self):
        from vllm.distributed.kv_transfer.kv_connector.v1.offloading.common import OffloadingConnectorMetadata, TransferJob
        from vllm.v1.kv_offload.base import GPULoadStoreSpec
        from ttkvp2p.spec import NvmeLoadStoreSpec
        from harness import WorkerKit

        rig = self.rig(with_scheduler=False)
        kit = WorkerKit(rig)
        kit.fill_blocks([1, 2, 3], 'i9')
        connector = rig.worker_conn
        slots, gens = kit.alloc(3)
        job = TransferJob('req', GPULoadStoreSpec([1, 2, 3], [3], [0]), NvmeLoadStoreSpec(slots, gens))
        connector.bind_connector_metadata(OffloadingConnectorMetadata(load_jobs={}, store_jobs={7: job}))
        # the worker's own get_finished knows nothing of the metadata: stores are deferred by prepare_store_kv, which only the
        # connector's get_finished runs, and the connector's get_transfer_results is the only caller of that
        rig.worker_conn.connector_worker.worker.get_finished()
        self.assertEqual(connector.connector_worker._unsubmitted_store_jobs, [])
        results = connector.get_transfer_results(set())
        self.assertEqual([j for j, _, _ in connector.connector_worker._unsubmitted_store_jobs], [7])
        self.assertEqual((results.finished_sending, results.finished_recving, results.failed_recving), (set(), set(), set()))
        connector.clear_connector_metadata()
        # the next step starts the store, and it completes through the same path
        connector.bind_connector_metadata(OffloadingConnectorMetadata(load_jobs={}, store_jobs={}))
        connector.start_load_kv(None)
        self.assertEqual(rig.worker.stats['submit_store'], 1)
        done = False
        for _ in range(500):
            connector.get_transfer_results(set())
            meta = connector.build_connector_worker_meta()
            if meta is not None and 7 in meta.completed_jobs:
                done = True
                break
            import time
            time.sleep(0.001)
        self.assertTrue(done)
        self.assertTrue(rig.runtime.store_results[7].success)

    def test_a_failed_receive_is_in_both_sets_and_the_assertion_holds(self):
        from vllm.distributed.kv_transfer.kv_connector.v1.offloading.common import OffloadingConnectorMetadata, TransferJob
        from vllm.v1.kv_offload.base import GPULoadStoreSpec
        from ttkvp2p.spec import NvmeLoadStoreSpec

        rig = self.rig(with_scheduler=False)
        connector = rig.worker_conn
        # a load whose spec is malformed is failed at submit and still reported, once, as finished and failed
        bad = TransferJob('req-9', NvmeLoadStoreSpec([1], [1], None), GPULoadStoreSpec([1], [1], [0]))
        connector.bind_connector_metadata(OffloadingConnectorMetadata(load_jobs={9: bad}, store_jobs={}))
        connector.start_load_kv(None)
        results = connector.get_transfer_results(set())
        self.assertEqual(results.finished_recving, {'req-9'})
        self.assertEqual(results.failed_recving, {'req-9'})
        meta = connector.build_connector_worker_meta()
        self.assertEqual(set(meta.completed_jobs), {9})
        self.assertEqual(meta.failed_loads, {9: 'bad_spec'})
        again = connector.get_transfer_results(set())
        self.assertEqual((again.finished_recving, again.failed_recving), (set(), set()))
        self.assertIsNone(connector.build_connector_worker_meta())


class FlushThroughTheRealScheduler(RigCase):
    """C15: the scheduler's fence protects an in-flight forward's writes; a store's gather is protected by the flush, which our wait() honours."""

    def reallocation_rig(self, **extra):
        # A pool this small cannot hold a second session beside the cached blocks of the first, so the second session's
        # allocation reuses the blocks the first one's still-queued store job reads.
        return self.rig(num_blocks=150, **extra)

    def test_a_finished_sessions_blocks_reallocated_before_its_store_ran_are_gathered_first_and_the_drive_is_exact(self):
        rig = self.reallocation_rig(slot_objects=32)                     # two spill slots hold the whole 57-object job
        rig.add('first', tokens(PROMPT, 'first'))
        rig.finish('first')
        rig.add('second', tokens(PROMPT, 'second'))                    # arrives while the first's store is still queued
        rig.drain()
        self.assertTrue(any(rig.flushes), 'the scenario happened: the scheduler put the first session\'s store job in jobs_to_flush')
        if not rig.async_mode:
            # with a step of slack (async scheduling) the gather has already run by the time the blocks are reallocated
            self.assertGreater(rig.worker.stats['flush_gathered'], 0, 'a store that had not run yet was gathered by the flush')
        self.assertEqual(rig.worker.stats['fail_no_slot'], 0)
        self.assertEqual(rig.worker.stats['done_store'], 2, 'both sessions\' stores reached the drive')
        # whatever reached the drive is exactly its key's object: a gather that ran after the second forward would have stored
        # the second session's KV under the first session's hashes
        self.assertGreater(rig.assert_drive_exact(), 0)
        self.assertEqual(rig.runtime.allocator.handed_out_in_flight, [])

    def test_with_no_free_spill_slot_the_flushed_store_is_dropped_and_nothing_wrong_is_stored(self):
        rig = self.reallocation_rig(extra={'spill_banks': [0]})                # one 8-object slot cannot hold a 57-object job
        rig.io.gate.clear()                                              # the only spill bank stays busy behind a slow drive
        rig.add('warm', tokens(400, 'warm'))                             # one small store occupies it
        rig.run_until(lambda: rig.worker.free_banks['spill'] == [], what='the spill bank taken')
        rig.add('first', tokens(PROMPT, 'first'))
        rig.finish('first')
        rig.add('second', tokens(PROMPT, 'second'))
        rig.finish('second', limit=1000)
        self.assertGreater(rig.worker.stats['fail_no_slot'], 0)
        rig.io.gate.set()
        rig.drain()
        self.assertEqual(rig.worker.strikes, 0)
        self.assertEqual(rig.assert_drive_exact(), rig.manager.written())
        self.assertEqual(rig.runtime.state, 'AVAILABLE')


class AsyncSchedulingFlow(Flow):
    def rig(self, **kwargs):
        kwargs.setdefault('async_scheduling', True)
        return super().rig(**kwargs)


class AsyncSchedulingFlush(FlushThroughTheRealScheduler):
    def rig(self, **kwargs):
        kwargs.setdefault('async_scheduling', True)
        return super().rig(**kwargs)


if __name__ == '__main__':
    unittest.main()
