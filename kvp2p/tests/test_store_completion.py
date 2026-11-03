"""How a store job ends in the manager (design 2.5 steps 5 and 6; the seam decision of review B2).

vLLM's own ``update_connector_output`` calls ``manager.complete_store(keys, ctx)`` for a completed store job: the job's key SET, no job
id, no digests, and no way to say the store failed. The scheduler role of the connector therefore settles every store job itself: it
takes the job's ``StoreResult`` from the runtime and hands the manager ``complete_store(keys in the manager's order, ctx, success=...,
digests=...)``, suppressing vLLM's own call. These tests drive vLLM's real scheduler and check what the (dict-backed) manager is told,
and that nothing is left ALLOCATED, which would make every lookup of those keys defer.
"""

import errno
import unittest

import numpy as np

from fakes import SHARD_OBJ, digest_bytes
from harness import RigCase, tokens

PROMPT = 2300
OBJECTS = PROMPT // 128 + 5 * 8


class Lossy(dict):
    """A store_results that forgets what the worker leaves in it: the pruned-before-consumed case."""

    def __setitem__(self, key, value):
        pass


class FailedWithDigests(dict):
    """A store_results whose entries say the job failed but still carry digests: the verdict, not the presence of digests, decides."""

    def __setitem__(self, key, value):
        value.success = False
        super().__setitem__(key, value)


class WrongSlots(dict):
    """A store_results whose entries name other drive slots than the job's store spec did."""

    def __setitem__(self, key, value):
        value.slot_ids = tuple(slot + 1 for slot in value.slot_ids)
        super().__setitem__(key, value)


class StoreCompletion(RigCase):
    def assert_nothing_left_allocated(self, rig):
        self.assertEqual([k for k, e in rig.manager.index.items() if e.state == 'ALLOCATED'], [],
                         'an ALLOCATED entry answers HIT_PENDING for ever: every lookup of its key would defer')

    # ------------------------------------------------------------------------------------------------ success
    def test_a_successful_store_hands_the_manager_the_keys_in_its_order_with_each_keys_own_digest(self):
        rig = self.rig()
        rig.add('r1', tokens(PROMPT, 'digests'))
        rig.drain()
        self.assertEqual(rig.manager.written(), OBJECTS)
        # one call per store job, from the connector (vLLM's own call is suppressed), every one a success with one digest row per key
        calls = rig.manager.complete_store_calls
        self.assertEqual(len(calls), rig.worker.stats['done_store'])
        self.assertTrue(all(success for _, success, _ in calls), calls)
        self.assertEqual(sum(n for n, _, _ in calls), OBJECTS)
        self.assertTrue(all(shape == (n, 4) for n, _, shape in calls), calls)
        # and each entry holds the digest of ITS OWN object: job.keys is a set, so a pairing by position would scramble them
        for key, entry in rig.manager.index.items():
            obj = rig.expected_object(key)
            want = [digest_bytes(obj[d * SHARD_OBJ:(d + 1) * SHARD_OBJ]) for d in range(4)]
            self.assertEqual([int(v) for v in entry.digest], want, 'the digest of another key\'s bytes is stored under this key')
        self.assertEqual(rig.runtime.store_results, {}, 'the scheduler role consumed every result')
        self.assertEqual(rig.connector_scheduler._store_jobs, {})

    def test_the_restore_verifies_against_those_digests(self):
        rig = self.rig()
        prompt = tokens(PROMPT, 'verified')
        rig.add('r1', prompt)
        rig.drain()
        rig.reset_device_cache()
        rig.add('r2', prompt)
        rig.finish('r2')
        self.assertEqual(rig.worker.stats['done_load'], 1)
        self.assertEqual(rig.worker.stats['digest_mismatch'], 0, 'the digests the manager kept are the digests of what the card holds')

    # ------------------------------------------------------------------------------------------------ B2: failure
    def test_a_failed_store_reaches_complete_store_as_a_failure_and_nothing_is_left_allocated(self):
        rig = self.rig()
        prompt = tokens(PROMPT, 'failing')
        rig.io.write_errors.append(OSError(errno.EAGAIN, 'a transient write error'))
        rig.add('r1', prompt)
        rig.drain()
        self.assertEqual(rig.worker.stats['fail_io'], 1)
        self.assertEqual(rig.worker.stats['done_store'], 0)
        calls = rig.manager.complete_store_calls
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][1:], (False, None), 'the verdict is the job\'s: a failed store is complete_store(success=False)')
        self.assertEqual(rig.manager.written(), 0)
        self.assertEqual(len(rig.manager.index), 0)
        allocator = rig.runtime.allocator
        self.assertEqual((len(allocator.free), allocator.pending_free, sum(allocator.inflight)), (rig.runtime.num_slots, set(), 0),
                         'every drive slot of the failed store is allocatable again')
        # the session can be stored again, and a lookup of its keys is a plain miss, not a deferral
        rig.reset_device_cache()
        rig.add('r2', prompt)
        rig.finish('r2', limit=400)
        rig.drain()
        self.assertEqual(rig.manager.written(), OBJECTS)
        self.assertEqual(rig.assert_drive_exact(), OBJECTS)

    def test_a_latched_tier_fails_the_stores_in_flight_and_they_reach_complete_store_as_failures(self):
        rig = self.rig()
        hold, = rig.io.hold_next()
        rig.add('r1', tokens(PROMPT, 'latching'))
        rig.run_until(lambda: any(call[0] == 'write' for call in rig.io.calls), what='the first write issued')
        rig.runtime.latch('control off')                                # the kill switch, with a store job in flight
        rig.finish('r1', limit=400)
        hold.set()
        rig.drain()
        self.assertEqual(rig.manager.written(), 0)
        self.assertTrue(rig.manager.complete_store_calls)
        self.assertTrue(all(success is False for _, success, _ in rig.manager.complete_store_calls), rig.manager.complete_store_calls)
        self.assert_nothing_left_allocated(rig)

    def test_a_store_the_worker_left_no_result_for_is_a_failed_store(self):
        """A pruned result (more than 1,024 reported before it was consumed) must read as a failure, never as a success."""
        rig = self.rig()
        rig.runtime.store_results = Lossy()
        rig.add('r1', tokens(PROMPT, 'pruned'))
        rig.drain()
        self.assertEqual(rig.worker.stats['done_store'], 1, 'the worker did store it')
        self.assertEqual(rig.manager.complete_store_calls[0][1:], (False, None))
        self.assertEqual(len(rig.manager.index), 0)

    def test_the_workers_verdict_decides_not_the_presence_of_digests(self):
        rig = self.rig()
        rig.runtime.store_results = FailedWithDigests()
        rig.add('r1', tokens(PROMPT, 'verdict'))
        rig.drain()
        self.assertEqual(rig.manager.written(), 0)
        self.assertTrue(all(success is False for _, success, _ in rig.manager.complete_store_calls))
        self.assertEqual(len(rig.manager.index), 0)

    def test_a_result_that_names_other_drive_slots_than_the_job_is_not_trusted(self):
        rig = self.rig()
        rig.runtime.store_results = WrongSlots()
        rig.add('r1', tokens(PROMPT, 'wrong-slots'))
        rig.drain()
        self.assertEqual(rig.manager.written(), 0, 'digests that may belong to other slots are never attached to these keys')
        self.assertTrue(all(success is False for _, success, _ in rig.manager.complete_store_calls))
        self.assertEqual(len(rig.manager.index), 0)

    # ------------------------------------------------------------------------------------------------ the stock call is suppressed
    def test_vllms_own_call_for_a_job_the_connector_did_not_settle_reaches_the_manager_as_a_failure(self):
        rig = self.rig()
        rig.add('r1', tokens(300, 'x'))
        status = next(iter(rig.connector_scheduler._req_status.values()))
        keys = {b'k' * 32 + (0).to_bytes(4, 'big')}
        rig.connector_scheduler.manager.complete_store(keys, status.req_context)        # what the stock loop calls: a bare set, success=True
        self.assertEqual(rig.manager.complete_store_calls, [(1, False, None)])

    # ------------------------------------------------------------------------------------------------ reset
    def test_a_store_job_completing_after_a_reset_settles_nothing_and_leaves_nothing_behind(self):
        rig = self.rig(workers=2)
        hold, = rig.io.hold_next()
        rig.add('r1', tokens(PROMPT, 'reset-store'))
        rig.run_until(lambda: any(call[0] == 'write' for call in rig.io.calls), what='the first write issued')
        self.assertTrue(rig.connector_scheduler._store_jobs)
        self.assertTrue(rig.scheduler.reset_connector_cache())
        self.assertEqual(rig.connector_scheduler._store_jobs, {}, 'jobs from before the reset are forgotten')
        hold.set()
        rig.drain()
        rig.run_until(lambda: not rig.worker.quarantined_banks and sum(rig.runtime.allocator.inflight) == 0, what='everything returned')
        self.assertEqual(rig.runtime.store_results, {}, 'the result of the pre-reset job was taken and dropped')
        self.assertEqual(rig.connector_scheduler._store_jobs, {})
        self.assert_nothing_left_allocated(rig)
        self.assertEqual(rig.assert_drive_exact(), rig.manager.written())

    # ------------------------------------------------------------------------------------------------ a broken manager
    def test_a_manager_that_raises_in_complete_store_latches_the_tier_and_the_keys_are_given_up(self):
        rig = self.rig()
        real = rig.manager.complete_store

        def flaky(keys, req_context, success=True, digests=None):
            if success:
                raise RuntimeError('a bug in the manager')
            return real(keys, req_context, success=success, digests=digests)

        rig.manager.complete_store = flaky
        rig.add('r1', tokens(PROMPT, 'flaky'))
        rig.drain()                                                     # the engine never sees the exception
        self.assertEqual(rig.runtime.state, 'LATCHED')
        self.assertIn('complete_store', rig.runtime.latch_reason)
        self.assertEqual(len(rig.manager.index), 0, 'the fallback call gave the keys up, so no lookup defers on them')
        rig.add('r2', tokens(PROMPT, 'after'))
        rig.finish('r2', limit=400)


class AsyncStoreCompletion(StoreCompletion):
    """The same with async scheduling: two batches in flight, so every completion reaches update_connector_output a step late."""

    def rig(self, **kwargs):
        kwargs.setdefault('async_scheduling', True)
        return super().rig(**kwargs)


if __name__ == '__main__':
    unittest.main()
