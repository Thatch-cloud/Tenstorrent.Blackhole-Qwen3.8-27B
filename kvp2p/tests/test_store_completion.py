"""How a store job ends in the manager (design 2.5 steps 5 and 6; the seam decision of review B2; AMENDMENT A1 of the interface).

vLLM's own ``update_connector_output`` calls ``manager.complete_store(keys, ctx)`` for a completed store job: the job's key SET, no job
id, no digests, and no way to say the store failed. The scheduler role of the connector therefore settles every store job itself: it
takes the job's ``StoreResult`` from the runtime and hands the manager ``complete_store(keys in the manager's order, ctx, success=...,
digests=...)``, suppressing vLLM's own call. These tests drive vLLM's real scheduler and check what the (dict-backed) manager is told,
and that nothing is left ALLOCATED, which would make every lookup of those keys defer.

The managers here have EXACTLY the signature of amendment A1 (``A1Manager``: it raises on any argument the amendment does not allow, and
a raise latches the tier), a slot-permuting allocator (``PermutingKVManager``: a digest paired through the slot's rank, not the key's
position, lands on another key's object), and a stream of digests that break the contract in every way a result can.
"""

import errno
import inspect
import unittest

import numpy as np

from fakes import SHARD_OBJ, A1Manager, FakeNvmeKVManager, PermutingKVManager, digest_bytes
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


def tampered(change):
    """A store_results whose successful entries carry ``change(digests)`` instead of what the worker computed."""

    class Tampered(dict):
        def __setitem__(self, key, value):
            if value.digests is not None:
                value.digests = change(value.digests)
            super().__setitem__(key, value)

    return Tampered()


# every way a result can break the contract of the digests: [n keys, chips] uint64
BROKEN_DIGESTS = {
    'one row short': lambda d: d[:-1],
    'one row too many': lambda d: np.vstack([d, d[:1]]),
    'three columns for four chips': lambda d: d[:, :3].copy(),
    'five columns for four chips': lambda d: np.hstack([d, d[:, :1]]),
    'one-dimensional': lambda d: d.reshape(-1),
    'three-dimensional': lambda d: d[:, :, None],
    'signed 64 bit': lambda d: d.astype(np.int64),
    'float 64': lambda d: d.astype(np.float64),
    'unsigned 32 bit': lambda d: d.astype(np.uint32),
    'a list of lists': lambda d: d.tolist(),
    'none at all': lambda d: None,
}


class ResultProblemTests(unittest.TestCase):
    """``TTP2PConnectorScheduler._result_problem`` on its own: what a result must be to be handed over as a success."""

    KEYS = [b'a' * 32 + (0).to_bytes(4, 'big'), b'b' * 32 + (0).to_bytes(4, 'big')]

    def problem(self, digests, *, chips=4, slots=(5, 6), result_slots=(5, 6), keys=None, geometry=True):
        from types import SimpleNamespace

        from ttkvp2p.connector import TTP2PConnectorScheduler
        from ttkvp2p.worker import StoreResult

        keys = self.KEYS if keys is None else keys
        spec = SimpleNamespace(layout_geometry=SimpleNamespace(chips=chips) if geometry else None)
        result = StoreResult(job_id=1, success=True, slot_ids=tuple(result_slots), generations=(1,) * len(slots), digests=digests)
        store_spec = SimpleNamespace(slot_ids=list(slots), generations=[1] * len(slots))
        job_status = SimpleNamespace(keys=set(self.KEYS))
        return TTP2PConnectorScheduler._result_problem(SimpleNamespace(_spec=spec), result, store_spec, keys, job_status)

    def test_exactly_what_the_contract_promises_is_accepted(self):
        self.assertIsNone(self.problem(np.zeros((2, 4), dtype=np.uint64)))
        self.assertIsNone(self.problem(np.zeros((2, 3), dtype=np.uint64), chips=3), 'the chip count is the layout\'s, not a constant 4')

    def test_each_way_to_break_it_is_named(self):
        good = np.zeros((2, 4), dtype=np.uint64)
        self.assertIn('not the keys of the job', self.problem(good, keys=self.KEYS[:1] + [b'c' * 36]))
        self.assertIn('other drive slots', self.problem(good, result_slots=(5, 7)))
        self.assertIn('3 drive slots for 2 keys', self.problem(good, slots=(5, 6, 7), result_slots=(5, 6, 7)))
        self.assertIn('not an ndarray', self.problem(good.tolist()))
        self.assertIn('not uint64', self.problem(good.astype(np.int64)))
        self.assertIn('for 2 keys of a 4-chip layout', self.problem(good[:, :3]))
        self.assertIn('for 2 keys of a 4-chip layout', self.problem(good[:1]))
        self.assertIn('chip count of the layout is unknown', self.problem(good, geometry=False))


class PermutingA1Manager(PermutingKVManager, A1Manager):
    """Both: the A1 signature with its strict checks, and slots that come out of order."""


class RequiredDigestsManager(A1Manager):
    """The guard accepts a ``digests`` without a default, so the connector must pass it on EVERY call, failures included."""

    def complete_store(self, keys, req_context, success, digests):
        super().complete_store(keys, req_context, success=success, digests=digests)


class StoreCompletion(RigCase):
    def assert_each_key_holds_its_own_digest(self, rig):
        for key in rig.manager.index:
            obj = rig.expected_object(key)
            want = [digest_bytes(obj[d * SHARD_OBJ:(d + 1) * SHARD_OBJ]) for d in range(4)]
            self.assertEqual(rig.entry_digest(key), want, 'the digest of another key\'s bytes is stored under this key')

    def assert_nothing_left_allocated(self, rig):
        self.assertEqual(rig.allocated_keys(), [],
                         'an ALLOCATED entry answers HIT_PENDING for ever: every lookup of its key would defer')

    # ------------------------------------------------------------------------------------------------ success
    def test_a_successful_store_hands_the_manager_the_keys_in_its_order_with_each_keys_own_digest(self):
        rig = self.rig()
        rig.add('r1', tokens(PROMPT, 'digests'))
        rig.drain()
        self.assertEqual(rig.manager.written(), OBJECTS)
        # one call per store job, from the connector (vLLM's own call is suppressed), every one a success with one digest row per key
        calls = rig.complete_store_calls
        self.assertEqual(len(calls), rig.worker.stats['done_store'])
        self.assertTrue(all(success for _, success, _ in calls), calls)
        self.assertEqual(sum(n for n, _, _ in calls), OBJECTS)
        self.assertTrue(all(shape == (n, 4) for n, _, shape in calls), calls)
        # and each entry holds the digest of ITS OWN object: job.keys is a set, so a pairing by position would scramble them
        self.assert_each_key_holds_its_own_digest(rig)
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
        calls = rig.complete_store_calls
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][1:], (False, None), 'the verdict is the job\'s: a failed store is complete_store(success=False)')
        self.assertEqual(rig.manager.written(), 0)
        self.assertEqual(len(rig.manager.index), 0)
        self.assertTrue(rig.slots_all_returned(), 'every drive slot of the failed store is allocatable again')
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
        self.assertTrue(rig.complete_store_calls)
        self.assertTrue(all(success is False for _, success, _ in rig.complete_store_calls), rig.complete_store_calls)
        self.assert_nothing_left_allocated(rig)

    def test_a_store_the_worker_left_no_result_for_is_a_failed_store(self):
        """A pruned result (more than 1,024 reported before it was consumed) must read as a failure, never as a success."""
        rig = self.rig()
        rig.runtime.store_results = Lossy()
        rig.add('r1', tokens(PROMPT, 'pruned'))
        rig.drain()
        self.assertEqual(rig.worker.stats['done_store'], 1, 'the worker did store it')
        self.assertEqual(rig.complete_store_calls[0][1:], (False, None))
        self.assertEqual(len(rig.manager.index), 0)

    def test_the_workers_verdict_decides_not_the_presence_of_digests(self):
        rig = self.rig()
        rig.runtime.store_results = FailedWithDigests()
        rig.add('r1', tokens(PROMPT, 'verdict'))
        rig.drain()
        self.assertEqual(rig.manager.written(), 0)
        self.assertTrue(all(success is False for _, success, _ in rig.complete_store_calls))
        self.assertEqual(len(rig.manager.index), 0)

    def test_a_result_that_names_other_drive_slots_than_the_job_is_not_trusted(self):
        rig = self.rig()
        rig.runtime.store_results = WrongSlots()
        rig.add('r1', tokens(PROMPT, 'wrong-slots'))
        rig.drain()
        self.assertEqual(rig.manager.written(), 0, 'digests that may belong to other slots are never attached to these keys')
        self.assertTrue(all(success is False for _, success, _ in rig.complete_store_calls))
        self.assertEqual(len(rig.manager.index), 0)

    def test_a_result_that_breaks_the_digest_contract_is_a_failed_store(self):
        """[keys, chips] uint64, exactly: a wrong shape, dtype or type is never handed over as a success. Digests of the wrong shape would
        make every later restore of those keys a bad_spec (no strike, no BAD), a miss that nothing ever explains."""
        for name, change in BROKEN_DIGESTS.items():
            with self.subTest(name):
                rig = self.rig()
                rig.runtime.store_results = tampered(change)
                with self.assertLogs('ttkvp2p.connector', level='ERROR') as logged:
                    rig.add('r1', tokens(PROMPT, 'broken ' + name))
                    rig.drain()
                self.assertIn('cannot be trusted', ' '.join(logged.output))
                calls = rig.complete_store_calls
                self.assertTrue(calls)
                self.assertTrue(all(call[1:] == (False, None) for call in calls), calls)
                self.assertEqual(len(rig.manager.index), 0, 'the keys are dropped, none is kept with digests nobody can vouch for')
                self.assertEqual(rig.runtime.state, 'AVAILABLE', 'a distrusted result is a miss, not a strike')
                self.assert_nothing_left_allocated(rig)

    # ------------------------------------------------------------------------------------------------ AMENDMENT A1
    def test_the_fake_managers_have_the_signature_of_amendment_a1_word_for_word(self):
        """complete_store(self, keys, req_context, success=True, digests=None): the fakes must not drift to whatever the connector calls."""
        from ttkvp2p.spec import RefusedManager

        empty = inspect.Parameter.empty
        want = [('self', inspect.Parameter.POSITIONAL_OR_KEYWORD, empty), ('keys', inspect.Parameter.POSITIONAL_OR_KEYWORD, empty),
                ('req_context', inspect.Parameter.POSITIONAL_OR_KEYWORD, empty),
                ('success', inspect.Parameter.POSITIONAL_OR_KEYWORD, True), ('digests', inspect.Parameter.POSITIONAL_OR_KEYWORD, None)]
        for cls in (FakeNvmeKVManager, A1Manager, PermutingKVManager, PermutingA1Manager, RefusedManager):
            with self.subTest(cls.__name__):
                got = [(p.name, p.kind, p.default) for p in inspect.signature(cls.complete_store).parameters.values()]
                self.assertEqual(got, want)

    def test_a_manager_with_exactly_the_a1_signature_is_told_everything_it_needs_and_nothing_it_does_not_allow(self):
        rig = self.rig(manager_class=A1Manager)
        rig.add('r1', tokens(PROMPT, 'a1'))
        rig.drain()
        self.assertEqual(rig.runtime.state, 'AVAILABLE', 'A1Manager raises on any call the amendment does not allow, and a raise latches: %s' % rig.runtime.latch_reason)
        self.assertEqual(rig.manager.written(), OBJECTS)
        calls = rig.manager.a1_calls
        self.assertEqual(len(calls), rig.worker.stats['done_store'])
        self.assertTrue(all(success and digests.shape == (len(keys), 4) for keys, success, digests in calls))
        self.assertEqual(sum(len(keys) for keys, _, _ in calls), OBJECTS)
        self.assert_each_key_holds_its_own_digest(rig)
        self.assertEqual(rig.assert_drive_exact(), OBJECTS)

    def test_a_manager_with_the_a1_signature_gets_digests_none_on_a_failed_store(self):
        rig = self.rig(manager_class=A1Manager)
        rig.io.write_errors.append(OSError(errno.EAGAIN, 'a transient write error'))
        rig.add('r1', tokens(PROMPT, 'a1-failing'))
        rig.drain()
        self.assertEqual(rig.runtime.state, 'AVAILABLE', rig.runtime.latch_reason)
        self.assertEqual(len(rig.manager.a1_calls), 1)
        keys, success, digests = rig.manager.a1_calls[0]
        self.assertEqual((success, digests), (False, None))
        self.assertEqual(len(rig.manager.index), 0)

    def test_a_manager_whose_digests_has_no_default_is_passed_it_on_every_call(self):
        rig = self.rig(manager_class=RequiredDigestsManager)
        prompt = tokens(PROMPT, 'required')
        rig.io.write_errors.append(OSError(errno.EAGAIN, 'a transient write error'))
        rig.add('r1', prompt)
        rig.drain()
        self.assertEqual([call[1:] for call in rig.manager.a1_calls], [(False, None)])
        rig.reset_device_cache()
        rig.add('r2', prompt)
        rig.drain()
        self.assertEqual(rig.runtime.state, 'AVAILABLE', rig.runtime.latch_reason)
        self.assertEqual(rig.manager.written(), OBJECTS)

    def test_a_slot_permuting_manager_still_gets_each_keys_own_digest_and_restores_exactly(self):
        """The slot of key i is not the i-th allocated and not monotone in i. Digest i still belongs to key i (the worker's digest row i is
        the digest of source block i, which went to slot_ids[i]); a connector that paired digests with keys through the slots' rank would
        attach another key's digest, and the restore of that key would then verify against the wrong object (or fail as a mismatch)."""
        rig = self.rig(manager_class=PermutingA1Manager)
        prompt = tokens(PROMPT, 'permuted')
        rig.add('r1', prompt)
        rig.drain()
        self.assertEqual(rig.runtime.state, 'AVAILABLE', rig.runtime.latch_reason)
        self.assertEqual(rig.manager.written(), OBJECTS)
        self.assertTrue(any(len(slots) > 1 and slots != sorted(slots) for _, slots in rig.manager.prepared),
                        'the allocator did permute: %s' % [slots for _, slots in rig.manager.prepared][:3])
        for keys, slots in rig.manager.prepared:                    # the manager's own key-to-slot map is the store spec's
            self.assertEqual([rig.manager.index[key].slot for key in keys], slots)
        self.assert_each_key_holds_its_own_digest(rig)
        self.assertEqual(rig.assert_drive_exact(), OBJECTS, 'every slot holds the object of the key whose entry names it')
        rig.reset_device_cache()                                    # and the session restores from those slots, digest-verified
        rig.add('r2', prompt)
        rig.finish('r2')
        self.assertEqual(rig.worker.stats['done_load'], 1)
        self.assertEqual(rig.worker.stats['digest_mismatch'], 0)
        self.assertEqual(rig.runtime.state, 'AVAILABLE')

    # ------------------------------------------------------------------------------------------------ the stock call is suppressed
    def test_vllms_own_call_for_a_job_the_connector_did_not_settle_reaches_the_manager_as_a_failure(self):
        rig = self.rig()
        rig.add('r1', tokens(300, 'x'))
        status = next(iter(rig.connector_scheduler._req_status.values()))
        keys = {b'k' * 32 + (0).to_bytes(4, 'big')}
        rig.connector_scheduler.manager.complete_store(keys, status.req_context)        # what the stock loop calls: a bare set, success=True
        self.assertEqual(rig.complete_store_calls, [(1, False, None)])

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


# The same tests again on the stand-in runtime and manager (tests/fakes.py): the connector and the worker are held to the seam without the WP3a modules
# under them, so a change in either side shows up on its own.


class StandInStoreCompletion(StoreCompletion):
    FAKE_TIER = True


class StandInAsyncStoreCompletion(AsyncStoreCompletion):
    FAKE_TIER = True


if __name__ == '__main__':
    unittest.main()
