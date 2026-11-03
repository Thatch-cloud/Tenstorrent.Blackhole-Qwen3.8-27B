"""KVP2PRuntime and SlotAllocator: the I5 slot rules, the shared process-wide runtime (I10), the host checks of design 2.9 (including the
zero-retention and retention_ack refusals), the control directory, the state file, the open-time discard.
"""

import json
import os
import random
import threading
import unittest
from types import SimpleNamespace

import numpy as np

import tier_kit
from tier_kit import OBJ, TierCase

from ttkvp2p import health, runtime as runtime_module, store
from ttkvp2p.runtime import (
    S_DRAINING, S_QUARANTINED, S_TRIM_WAIT, S_TRIMMING, S_USED, STATE_AVAILABLE, STATE_CLOSING, STATE_LATCHED,
    STATE_UNAVAILABLE, KVP2PRuntime, SlotAllocator, StoreResultsMap, _Runs,
)


def free_allocator(n):
    allocator = SlotAllocator(n)
    for first, count in allocator.take_trim_runs(10 ** 6, 10 ** 6):
        allocator.trim_done(first, count, True)
    return allocator


class Runs(unittest.TestCase):
    def test_runs_coalesce_and_split(self):
        runs = _Runs()
        runs.add(10, 5)
        runs.add(20, 5)
        runs.add(15, 5)
        self.assertEqual((runs.starts, runs.lens, runs.total), ([10], [15], 15))
        self.assertEqual(runs.take(4), [10, 11, 12, 13])
        self.assertEqual((runs.starts, runs.lens), ([14], [11]))

    def test_take_prefers_one_contiguous_run(self):
        runs = _Runs()
        runs.add(0, 3)
        runs.add(10, 10)
        self.assertEqual(runs.take(5), [10, 11, 12, 13, 14], 'a run of 5 exists: no split across runs')
        self.assertEqual(runs.take(4), [15, 16, 17, 18])

    def test_take_spans_runs_when_none_is_long_enough(self):
        runs = _Runs()
        runs.add(0, 2)
        runs.add(10, 2)
        self.assertEqual(runs.take(3), [0, 1, 10])
        self.assertIsNone(runs.take(5))

    def test_add_slots_builds_runs_from_any_order(self):
        runs = _Runs()
        runs.add_slots([5, 3, 4, 9, 10, 1])
        self.assertEqual((runs.starts, runs.lens), ([1, 3, 9], [1, 3, 2]))


class AllocatorRules(unittest.TestCase):
    """Invariant I5: a slot is allocatable only when no entry names it, no bio is in flight on it, and its TRIM completed."""

    def test_every_slot_starts_waiting_for_its_trim(self):
        allocator = SlotAllocator(100)
        self.assertEqual(allocator.n_free, 0)
        self.assertIsNone(allocator.alloc(1), 'nothing is allocatable before the open-time discard completes')
        runs = allocator.take_trim_runs(1, 40)
        self.assertEqual(runs, [(0, 40)])
        self.assertIsNone(allocator.alloc(1), 'a slot being TRIMmed is not allocatable')
        allocator.trim_done(0, 40, True)
        self.assertEqual(allocator.alloc(40), list(range(40)), 'allocatable as its run completes')
        self.assertIsNone(allocator.alloc(1), 'the rest is still waiting')

    def test_each_reallocation_bumps_the_generation(self):
        allocator = free_allocator(8)
        first = allocator.alloc(2)
        self.assertEqual([allocator.generation(s) for s in first], [1, 1])
        allocator.release_many(first)
        for run in allocator.take_trim_runs(5, 100):
            allocator.trim_done(*run, True)
        second = allocator.alloc(8)
        self.assertEqual(sorted(second), list(range(8)))
        self.assertEqual([allocator.generation(s) for s in first], [2, 2])

    def test_a_released_slot_is_not_allocatable_until_its_trim_completes(self):
        allocator = free_allocator(4)
        slots = allocator.alloc(4)
        allocator.release(slots[0])
        self.assertEqual(allocator.state_name(slots[0]), 'trim_wait')
        self.assertIsNone(allocator.alloc(1))
        (run,) = allocator.take_trim_runs(1, 10)
        self.assertEqual(run, (slots[0], 1))
        self.assertIsNone(allocator.alloc(1), 'issued is not done')
        allocator.trim_done(*run, True)
        self.assertEqual(allocator.alloc(1), [slots[0]])

    def test_a_slot_with_a_bio_in_flight_drains_before_it_is_trimmed(self):
        allocator = free_allocator(4)
        slots = allocator.alloc(2)
        gens = [allocator.generation(s) for s in slots]
        self.assertTrue(allocator.io_begin(slots, gens))
        allocator.release_many(slots)
        self.assertEqual([allocator.state_name(s) for s in slots], ['draining', 'draining'])
        self.assertEqual(allocator.take_trim_runs(5, 10), [], 'no TRIM is issued under a bio')
        allocator.io_end([slots[0]])
        self.assertEqual(allocator.take_trim_runs(5, 10), [(slots[0], 1)])
        self.assertEqual(allocator.state_name(slots[1]), 'draining')
        allocator.io_end([slots[1]])
        self.assertEqual(allocator.state_name(slots[1]), 'trim_wait')

    def test_io_begin_refuses_a_stale_generation_a_free_slot_and_counts_nothing(self):
        allocator = free_allocator(6)
        slots = allocator.alloc(3)
        gens = [allocator.generation(s) for s in slots]
        self.assertFalse(allocator.io_begin(slots, [gens[0], gens[1] + 1, gens[2]]))
        self.assertEqual(allocator.total_inflight(), 0, 'a refused begin counts nothing')
        allocator.release(slots[2])
        self.assertFalse(allocator.io_begin(slots, gens), 'a released slot takes no new bio')
        self.assertFalse(allocator.io_begin([5], [0]), 'a FREE slot takes no bio')
        self.assertFalse(allocator.io_begin([99], [1]), 'a slot that does not exist')
        self.assertTrue(allocator.io_begin(slots[:2], gens[:2]))
        self.assertEqual(allocator.total_inflight(), 2)

    def test_a_late_bio_of_a_stale_job_cannot_start_on_a_reallocated_slot(self):
        allocator = free_allocator(2)
        (slot,) = allocator.alloc(1)
        old_generation = allocator.generation(slot)
        allocator.release(slot)
        for run in allocator.take_trim_runs(5, 10):
            allocator.trim_done(*run, True)
        self.assertEqual(allocator.alloc(2), [slot, 1 - slot] if slot == 0 else [slot, 1 - slot])
        self.assertFalse(allocator.io_begin([slot], [old_generation]))
        self.assertTrue(allocator.io_begin([slot], [allocator.generation(slot)]))

    def test_a_failed_trim_quarantines_the_slots_for_good(self):
        allocator = free_allocator(4)
        slots = allocator.alloc(4)
        allocator.release_many(slots[:2])
        (run,) = allocator.take_trim_runs(5, 10)
        allocator.trim_done(*run, False)
        self.assertEqual(allocator.count(S_QUARANTINED), 2)
        self.assertEqual(allocator.trim_failed, 2)
        self.assertIsNone(allocator.alloc(1))
        self.assertEqual(allocator.take_trim_runs(5, 10), [])

    def test_invalidate_all_sends_every_slot_back_through_trim_and_respects_bios(self):
        allocator = free_allocator(10)
        slots = allocator.alloc(6)
        self.assertTrue(allocator.io_begin([slots[0]], [allocator.generation(slots[0])]))
        allocator.invalidate_all()
        self.assertEqual(allocator.n_free, 0)
        self.assertEqual(allocator.state_name(slots[0]), 'draining')
        self.assertEqual(allocator.count(S_TRIM_WAIT), 9)
        total = 0
        for run in allocator.take_trim_runs(100, 100):
            total += run[1]
            allocator.trim_done(*run, True)
        self.assertEqual(total, 9)
        allocator.io_end([slots[0]])
        for run in allocator.take_trim_runs(100, 100):
            allocator.trim_done(*run, True)
        self.assertEqual(allocator.n_free, 10)

    def test_a_second_release_of_the_same_slot_is_harmless(self):
        allocator = free_allocator(3)
        (slot,) = allocator.alloc(1)
        allocator.release(slot)
        allocator.release(slot)
        self.assertEqual(allocator.count(S_TRIM_WAIT), 1)

    def test_quarantine_takes_a_used_slot_out_of_circulation(self):
        allocator = free_allocator(3)
        (slot,) = allocator.alloc(1)
        allocator.quarantine(slot)
        self.assertEqual(allocator.state_name(slot), 'quarantined')
        allocator.release(slot)
        self.assertEqual(allocator.state_name(slot), 'quarantined')

    def test_a_random_workload_never_hands_out_a_slot_that_is_not_free_and_never_loses_one(self):
        """A model check: random alloc / release / io / trim steps against a plain-Python model of the five states."""
        rng = random.Random(1234)
        n = 200
        allocator = free_allocator(n)
        model_used, model_bio = set(), collections_counter()
        issued = []
        for step in range(6000):
            action = rng.random()
            if action < 0.30:
                want = rng.randint(1, 12)
                free_before = allocator.n_free
                got = allocator.alloc(want)
                if got is None:
                    self.assertLess(free_before, want)
                else:
                    self.assertEqual(len(got), want)
                    self.assertTrue(model_used.isdisjoint(got), 'alloc handed out a USED slot')
                    for slot in got:
                        self.assertEqual(model_bio[slot], 0, 'alloc handed out a slot with a bio in flight (I5)')
                    model_used.update(got)
            elif action < 0.50 and model_used:
                victims = rng.sample(sorted(model_used), min(len(model_used), rng.randint(1, 6)))
                allocator.release_many(victims)
                model_used.difference_update(victims)
            elif action < 0.65 and model_used:
                slot = rng.choice(sorted(model_used))
                self.assertTrue(allocator.io_begin([slot], [allocator.generation(slot)]))
                model_bio[slot] += 1
            elif action < 0.80:
                busy = [s for s, c in model_bio.items() if c > 0]
                if busy:
                    slot = rng.choice(busy)
                    allocator.io_end([slot])
                    model_bio[slot] -= 1
            elif action < 0.92:
                issued.extend(allocator.take_trim_runs(rng.randint(1, 3), rng.randint(1, 20)))
            elif issued:
                run = issued.pop(rng.randrange(len(issued)))
                allocator.trim_done(*run, True)
            counts = [allocator.count(i) for i in range(6)]
            self.assertEqual(sum(counts), n, 'a slot was lost or duplicated')
            self.assertEqual(counts[S_USED], len(model_used))
            for slot in range(n):
                if model_bio[slot] > 0:
                    self.assertIn(allocator.state_name(slot), ('used', 'draining'), 'a slot under a bio went to TRIM or FREE (I5)')
        draining = sum(1 for s in range(n) if allocator.state_name(s) == 'draining')
        self.assertEqual(draining, allocator.count(S_DRAINING))


def collections_counter():
    import collections
    return collections.defaultdict(int)


class StoreResultsIndex(unittest.TestCase):
    def result(self, job, slots, gens, ok=True):
        return SimpleNamespace(job_id=job, success=ok, slot_ids=tuple(slots), generations=tuple(gens),
                               digests=np.arange(len(slots) * 4, dtype=np.uint64).reshape(-1, 4) + job * 1000 if ok else None)

    def test_pairing_is_an_index_lookup_by_slot_and_generation(self):
        results = StoreResultsMap()
        results[7] = self.result(7, [10, 11, 12], [1, 1, 3])
        found, position = results.lookup(11, 1)
        self.assertEqual((found.job_id, position), (7, 1))
        self.assertIsNone(results.lookup(11, 2), 'another generation of the slot is another store')
        self.assertIsNone(results.lookup(99, 1))

    def test_pop_clear_and_overwrite_keep_the_index_exact(self):
        results = StoreResultsMap()
        results[1] = self.result(1, [0, 1], [1, 1])
        results[2] = self.result(2, [2], [1])
        results.pop(1)
        self.assertIsNone(results.lookup(0, 1))
        self.assertIsNotNone(results.lookup(2, 1))
        results[2] = self.result(2, [5], [4])
        self.assertIsNone(results.lookup(2, 1))
        self.assertIsNotNone(results.lookup(5, 4))
        results.clear()
        self.assertIsNone(results.lookup(5, 4))
        self.assertEqual(len(results), 0)

    def test_a_pruned_result_is_gone_from_the_index_too(self):
        results = StoreResultsMap()
        results[1] = self.result(1, [3], [1])
        results.pop(1, None)
        results.pop(1, None)
        self.assertIsNone(results.lookup(3, 1))

    def test_a_newer_store_of_the_same_slot_generation_wins_and_the_old_unindexing_does_not_remove_it(self):
        results = StoreResultsMap()
        results[1] = self.result(1, [3], [1])
        results[2] = self.result(2, [3], [1])
        results.pop(1)
        found, _ = results.lookup(3, 1)
        self.assertEqual(found.job_id, 2)


class SharedRuntime(TierCase):
    """Invariant I10: one runtime per drive path; the worker opens, the scheduler attaches; release is reference counted."""

    def test_both_roles_get_one_runtime_and_only_the_worker_opens_it(self):
        self.tier()
        scheduler_first = KVP2PRuntime.acquire(self.cfg, 'scheduler')
        self.assertFalse(scheduler_first.host_ok, 'attaching does not open the drive')
        self.assertIsNone(scheduler_first.store)
        self.assertEqual(scheduler_first.state, STATE_UNAVAILABLE)
        worker = KVP2PRuntime.acquire(self.cfg, 'worker')
        self.runtimes.append(worker)
        self.assertIs(worker, scheduler_first)
        self.assertTrue(worker.host_ok, worker.host_reason)
        self.assertEqual(sorted(worker.roles), ['scheduler', 'worker'])

    def test_the_worker_first_then_the_scheduler_is_the_same_instance(self):
        self.tier()
        worker = self.open_runtime()
        scheduler = KVP2PRuntime.acquire(self.cfg, 'scheduler')
        self.assertIs(worker, scheduler)

    def test_a_different_drive_path_is_a_different_runtime(self):
        self.tier()
        a = KVP2PRuntime.acquire(self.cfg, 'scheduler')
        other = type(self.cfg)(**{**self.cfg.__dict__, 'nvme_device': self.cfg.nvme_device + '-2'})
        b = KVP2PRuntime.acquire(other, 'scheduler')
        self.assertIsNot(a, b)

    def test_shutdown_is_reference_counted_the_last_release_closes(self):
        self.tier()
        runtime = self.open_runtime()
        KVP2PRuntime.acquire(self.cfg, 'scheduler')
        runtime.release('worker')                 # the executor shuts down before the scheduler
        self.assertEqual(runtime.state, STATE_AVAILABLE)
        self.assertIsNotNone(runtime.store, 'the scheduler still holds the runtime')
        runtime.release('scheduler')
        self.assertEqual(runtime.state, STATE_CLOSING)
        self.assertIsNone(runtime.store)
        self.assertNotIn(self.cfg.nvme_device, KVP2PRuntime._instances)
        self.assertEqual(self.tree.read_state()['state'], 'CLOSING', 'a final state file says CLOSING')

    def test_the_drive_is_released_so_a_new_engine_can_claim_it(self):
        self.tier()
        runtime = self.open_runtime()
        runtime.release('worker')
        again = store.open_store(self.tree.drive, layout_id=self.tree.layout_id, object_bytes=OBJ, allow_regular=True)
        again.close()

    def test_a_drive_with_a_bio_in_flight_is_left_open(self):
        self.tier()
        runtime = self.open_runtime()
        (slot,) = runtime.allocator.alloc(1)
        runtime.allocator.io_begin([slot], [runtime.allocator.generation(slot)])
        runtime.release('worker')
        self.assertEqual(runtime.state, STATE_CLOSING)
        self.assertIsNone(runtime.store, 'dropped from the runtime ...')
        # ... and the descriptor was NOT closed: the old holder still has the claim (no exit with a DMA in flight)
        with self.assertRaises(store.DeviceBusy):
            store.open_store(self.tree.drive, layout_id=self.tree.layout_id, object_bytes=OBJ, allow_regular=True)

    def test_an_unknown_role_is_refused_and_an_extra_release_is_tolerated(self):
        self.tier()
        with self.assertRaises(ValueError):
            KVP2PRuntime.acquire(self.cfg, 'driver')
        runtime = self.open_runtime()
        runtime.release('worker')
        runtime.release('worker')

    def test_the_interface_members_exist(self):
        self.tier()
        runtime = self.open_runtime()
        for name in ('state', 'allocator', 'store_results', 'latch', 'poll_control', 'write_state', 'release', 'set_available', 'mark_bad',
                     'slot_offset', 'host_ok', 'count', 'observe_restore', 'take_fault'):
            self.assertTrue(hasattr(runtime, name), name)
        self.assertIsInstance(runtime.allocator, SlotAllocator)
        self.assertIsInstance(runtime.store_results, dict)


class OpenAndDiscard(TierCase):
    def test_the_open_discards_the_whole_data_region_and_slots_come_free_as_runs_complete(self):
        self.tier(num_slots=40, sync_io=False)
        with open(self.tree.drive, 'r+b') as handle:                        # stale data from a previous process
            handle.seek(self.tree.data_off)
            handle.write(b'\xa5' * (40 * OBJ))
        gate = threading.Event()
        calls = []

        original = store.DriveFile.discard

        def gated(drive, offset, length):
            calls.append((offset, length))
            gate.wait(5)
            return original(drive, offset, length)

        store.DriveFile.discard = gated
        self.addCleanup(setattr, store.DriveFile, 'discard', original)
        self.addCleanup(gate.set)
        runtime = self.open_runtime(settle=False)
        self.assertTrue(runtime.host_ok, runtime.host_reason)
        runtime.service()
        self.assertEqual(runtime.allocator.n_free, 0, 'nothing is allocatable before the first discard completes')
        self.assertEqual(runtime.allocator.count(S_TRIM_WAIT) + runtime.allocator.count(S_TRIMMING), 40)
        gate.set()
        self.assertTrue(runtime.settle(5.0))
        self.assertEqual(runtime.allocator.n_free, 40)
        with open(self.tree.drive, 'rb') as handle:
            handle.seek(self.tree.data_off)
            self.assertEqual(handle.read(40 * OBJ), bytes(40 * OBJ), 'every slot was discarded')
        self.assertEqual(sum(length for _, length in calls), 40 * OBJ)

    def test_the_discard_goes_in_runs_of_at_most_1_gib(self):
        self.assertEqual(store.DISCARD_RUN_BYTES, 1 << 30)
        allocator = SlotAllocator(1000)
        per_run = (1 << 30) // 2785280
        runs = allocator.take_trim_runs(100, per_run)
        self.assertTrue(all(count <= 385 for _, count in runs))
        self.assertEqual(per_run, 385)

    def test_a_failed_discard_run_quarantines_its_slots_and_the_rest_still_serve(self):
        self.tier(num_slots=64)
        drive_discard = store.DriveFile.discard
        calls = {'n': 0}

        def flaky(drive, offset, length):
            calls['n'] += 1
            if calls['n'] == 1:
                raise OSError(5, 'media error')
            return drive_discard(drive, offset, length)

        store.DriveFile.discard = flaky
        self.addCleanup(setattr, store.DriveFile, 'discard', drive_discard)
        # one huge run would take everything; shrink the run size so that only part fails
        saved = store.DISCARD_RUN_BYTES
        store.DISCARD_RUN_BYTES = 16 * OBJ
        self.addCleanup(setattr, store, 'DISCARD_RUN_BYTES', saved)
        runtime = self.open_runtime()
        self.assertEqual(runtime.allocator.count(S_QUARANTINED), 16)
        self.assertEqual(runtime.allocator.n_free, 48)
        self.assertEqual(runtime.counters['trim_failed'], 16)

    def test_a_device_gone_trim_error_latches(self):
        self.tier(num_slots=8)
        original = store.DriveFile.discard

        def gone(drive, offset, length):
            raise OSError(19, 'No such device')

        store.DriveFile.discard = gone
        self.addCleanup(setattr, store.DriveFile, 'discard', original)
        runtime = self.open_runtime(settle=False)
        runtime.settle(2.0)
        self.assertEqual(runtime.state, STATE_LATCHED)
        self.assertEqual(runtime.latch_code, 'device_lost')

    def test_the_slot_offset_comes_from_the_superblock(self):
        self.tier(num_slots=10)
        runtime = self.open_runtime()
        self.assertEqual(runtime.slot_offset(3), self.tree.data_off + 3 * OBJ)
        self.assertEqual(runtime.canary_offset, 8192)
        self.assertLess(runtime.canary_offset + runtime.canary_bytes, self.tree.data_off + 1)


class HostChecks(TierCase):
    """Design 2.9 step 1: every host check runs before any device work; a failure leaves the tier UNAVAILABLE with no device footprint."""

    def refused(self, code, **kw):
        runtime = self.open_runtime()
        self.assertFalse(runtime.host_ok, 'the host checks passed but %r was expected' % code)
        self.assertEqual(runtime.host_code, code, runtime.host_reason)
        self.assertIsNone(runtime.store, 'a refused tier holds no drive')
        self.assertEqual(runtime.allocator.num_slots, 0, 'and no slots')
        self.assertNotEqual(runtime.state, STATE_AVAILABLE)
        runtime.set_available()
        self.assertNotEqual(runtime.state, STATE_AVAILABLE, 'set_available cannot revive a refused runtime')
        return runtime

    def test_all_good_passes(self):
        self.tier()
        runtime = self.open_runtime()
        self.assertTrue(runtime.host_ok, runtime.host_reason)
        self.assertEqual(runtime.state, STATE_AVAILABLE)
        self.assertEqual(runtime.allocator.num_slots, 64)

    # --- retention consent and zero retention
    def test_retention_ack_is_required(self):
        self.tier(retention_ack=False)
        self.refused('retention_ack_missing')

    def test_zero_retention_refuses_in_each_of_its_forms(self):
        for name, value in (('THATCH_ZERO_RETENTION', '1'), ('THATCH_ZERO_RETENTION', 'true'), ('THATCH_NO_LOG', '1'),
                            ('THATCH_ATTESTATION_PRIVATE_KEY', 'some-key-material')):
            self.setUp()
            self.tier(environ={name: value})
            runtime = self.refused('zero_retention')
            self.assertTrue(runtime.host_reason.startswith('zero_retention'))

    def test_a_zero_retention_marker_file_refuses(self):
        self.tier()
        marker = os.path.join(os.path.dirname(self.cfg.control_dir.rstrip('/')), health.ZERO_RETENTION_MARKER_NAME)
        open(marker, 'w').close()
        self.refused('zero_retention')

    def test_zero_retention_off_values_do_not_refuse(self):
        self.tier(environ={'THATCH_ZERO_RETENTION': '0', 'THATCH_NO_LOG': '0'})
        self.assertTrue(self.open_runtime().host_ok)

    def test_zero_retention_wins_over_a_missing_ack(self):
        self.tier(retention_ack=False, environ={'THATCH_ZERO_RETENTION': '1'})
        self.refused('zero_retention')

    def test_ttl_over_the_floor_is_refused(self):
        self.tier(ttl_s=86401)
        self.refused('ttl_over_floor')

    def test_require_sed_false_is_gate_only(self):
        self.tier(require_sed=False)
        self.assertTrue(self.open_runtime().host_ok, 'with THATCH_KVP2P_TEST=1 it is allowed')
        self.runtimes.pop().release('worker')
        KVP2PRuntime.reset_registry()
        self.setUp()
        self.tier(require_sed=False)
        runtime_module.set_host_env(tier_kit.make_env(self.tree, self.clock, environ={'THATCH_KVP2P_TEST': '0'}))
        self.refused('sed_not_required')

    def test_durable_persistence_is_v1b(self):
        self.tier(persistence='durable')
        self.refused('schema_invalid')

    def test_a_regular_file_as_the_drive_needs_the_test_flag(self):
        self.tier()
        runtime_module.set_host_env(tier_kit.make_env(self.tree, self.clock, environ={'THATCH_KVP2P_TEST': '0'}))
        self.refused('device_absent')

    # --- the cache-drive status file
    def test_a_missing_status_file_means_at_rest_unverified(self):
        self.tier()
        os.unlink(os.path.join(self.tree.status_dir, 'status.json'))
        self.refused('at_rest_unverified')

    def test_a_stale_status_file_is_refused(self):
        self.tier()
        self.clock.advance(121)
        self.refused('status_stale')

    def test_a_status_file_from_the_future_is_refused(self):
        self.tier()
        self.tree.write_status(self.clock, timestamp=health.iso_utc(self.clock.now + 600))
        self.refused('status_stale')

    def test_the_status_file_must_be_fresh_but_may_be_nearly_two_minutes_old(self):
        self.tier()
        self.clock.advance(119)
        self.assertTrue(self.open_runtime().host_ok)

    def test_the_status_must_name_the_drive_and_say_ok_unlocked_opal_and_gen4(self):
        cases = [
            ({'by_id': ''}, 'at_rest_unverified'), ({'state': 'not_live'}, 'device_dead'), ({'state': 'locked'}, 'at_rest_unverified'),
            ({'sed': 'none'}, 'at_rest_unverified'), ({'locking_enabled': False}, 'at_rest_unverified'), ({'locked': True}, 'at_rest_unverified'),
            ({'link_speed': '8.0 GT/s PCIe'}, 'link_below_floor'), ({'link_speed': '2.5 GT/s PCIe'}, 'link_below_floor'),
            ({'link_speed': 'unknown'}, 'link_below_floor'), ({'timestamp': 'garbage'}, 'status_stale'),
        ]
        for change, code in cases:
            self.setUp()
            self.tier()
            self.tree.write_status(self.clock, **change)
            runtime = self.refused(code)
            self.assertIn(code, runtime.host_reason)

    def test_a_32_gt_link_passes(self):
        self.tier()
        self.tree.write_status(self.clock, link_speed='32.0 GT/s PCIe')
        self.assertTrue(self.open_runtime().host_ok)

    def test_a_drive_claimed_under_another_by_id_name_fails_the_claim(self):
        self.tier()
        self.tree.write_status(self.clock, by_id='the-cloned-drive')
        self.refused('superblock_unknown')

    # --- the drive
    def test_a_blank_drive_is_superblock_unknown(self):
        self.tier(claimed=False)
        self.refused('superblock_unknown')

    def test_a_layout_mismatch(self):
        self.tier(layout='gemma-chipmajor-v1')
        runtime = self.open_runtime()
        self.assertFalse(runtime.host_ok)
        self.assertIn(runtime.host_code, ('layout_mismatch', 'superblock_unknown'))

    def test_a_drive_held_by_another_process_is_claim_busy(self):
        self.tier()
        holder = store.DriveFile(self.tree.drive, allow_regular=True)
        self.addCleanup(holder.close)
        self.refused('claim_busy')

    def test_a_drive_too_small_for_one_maximum_session_is_capacity_short(self):
        self.tier(num_slots=64)
        store.MIN_SESSION_OBJECTS[self.tree.layout_id] = 100
        self.addCleanup(store.MIN_SESSION_OBJECTS.pop, self.tree.layout_id, None)
        self.refused('capacity_short')

    def test_the_floor_and_the_reserve_count_against_the_session_room(self):
        self.tier(num_slots=64, free_floor_pct=50)
        store.MIN_SESSION_OBJECTS[self.tree.layout_id] = 40
        self.addCleanup(store.MIN_SESSION_OBJECTS.pop, self.tree.layout_id, None)
        self.refused('capacity_short')

    def test_the_gemma_session_needs_2088_objects(self):
        self.assertEqual(store.min_session_objects('gemma-chipmajor-v1'), 2088)
        self.assertEqual(store.min_session_objects('anything-else'), 1)

    # --- p2pmem
    def test_p2pmem_needs_one_allocate_file_per_card(self):
        self.tier()
        for count in (0, 3, 5):
            self.setUp()
            self.tier()
            import shutil
            shutil.rmtree(os.path.join(self.root, 'p2pmem'))
            self.tree.write_p2pmem(cards=count)
            self.refused('p2pmem_unavailable')

    def test_a_slice_that_is_already_allocated_is_refused(self):
        self.tier()
        self.tree.write_p2pmem(available=4 << 29)
        runtime = self.refused('p2pmem_unavailable')
        self.assertIn('already allocated', runtime.host_reason)

    def test_a_slice_the_kernel_does_not_know_is_refused(self):
        self.tier()
        os.unlink(os.path.join(self.tree.sysfs_pci, 'card-2', 'p2pmem', 'size'))
        self.refused('p2pmem_unavailable')

    def test_the_mmap_of_every_slice_must_succeed(self):
        mapped = []

        def probe(path, size):
            mapped.append(os.path.basename(os.path.dirname(path)))
            if len(mapped) == 3:
                raise OSError(13, 'Permission denied')          # an AppArmor profile without the allocate rule

        self.tier(probe=probe)
        runtime = self.refused('p2pmem_unavailable')
        self.assertEqual(mapped, ['card-0', 'card-1', 'card-2'])
        self.assertIn('card-2', runtime.host_reason)

    def test_a_good_node_maps_all_four_slices(self):
        mapped = []
        self.tier(probe=lambda path, size: mapped.append((os.path.basename(os.path.dirname(path)), size)))
        self.assertTrue(self.open_runtime().host_ok)
        self.assertEqual([name for name, _ in mapped], ['card-0', 'card-1', 'card-2', 'card-3'])
        self.assertTrue(all(size == 4 << 30 for _, size in mapped))

    # --- no device footprint
    def test_a_refusal_never_touches_the_drive(self):
        self.tier(retention_ack=False)
        before = open(self.tree.drive, 'rb').read()
        runtime = self.open_runtime()
        self.assertEqual(runtime.host_code, 'retention_ack_missing')
        self.assertEqual(open(self.tree.drive, 'rb').read(), before)
        holder = store.DriveFile(self.tree.drive, allow_regular=True)             # and it never even opened it: we can take the claim
        holder.close()

    def test_the_refusal_is_written_to_the_state_file(self):
        self.tier(retention_ack=False)
        self.open_runtime()
        state = self.tree.read_state()
        self.assertFalse(state['host_ok'])
        self.assertTrue(state['host_reason'].startswith('retention_ack_missing'))
        self.assertEqual(state['state'], 'UNAVAILABLE')


class Latch(TierCase):
    def test_a_latch_turns_the_tier_off_and_keeps_the_first_reason(self):
        self.tier()
        runtime = self.open_runtime()
        runtime.latch('3 consecutive job failures (last: eio)')
        runtime.latch('a device op threw')
        self.assertEqual(runtime.state, STATE_LATCHED)
        self.assertEqual(runtime.latch_code, 'strikes')
        self.assertIn('consecutive', runtime.latch_reason)
        self.assertEqual(dict(runtime.latched_total), {'strikes': 1})
        runtime.set_available()
        self.assertEqual(runtime.state, STATE_LATCHED, 'nothing but a control on revives a latch')

    def test_latch_reasons_map_into_the_closed_vocabulary(self):
        for text, code in (('3 consecutive job failures (last: digest)', 'strikes'), ('no completion for 30.0 s with work in flight', 'stall'),
                           ('a device op threw', 'op_throw'), ('internal error in submit_store: KeyError', 'op_throw'),
                           ('control off', 'control_off'), ('uncorrectable AER on the drive path', 'aer_uncorrectable'),
                           ('device lost: ENODEV', 'device_lost'), ('canary mismatch on bank 3', 'bounds_violation'),
                           ('bounds violation', 'bounds_violation'), ('the daily write cap', 'write_cap')):
            self.assertEqual(health.latch_code(text), code, text)
            self.assertIn(code, health.LATCH_CODES)

    def test_a_latch_on_a_closing_runtime_is_ignored(self):
        self.tier()
        runtime = self.open_runtime()
        runtime.release('worker')
        runtime.latch('too late')
        self.assertEqual(runtime.state, STATE_CLOSING)


class ControlDirectory(TierCase):
    """3.6: commands are files NNNN-<verb> polled once a second; off/on are levels (the highest number wins), the rest apply once."""

    def poll(self, runtime, seconds=2.0):
        self.clock.advance(seconds)
        runtime.service()

    def test_off_latches_now_and_on_clears_it(self):
        self.tier()
        runtime = self.open_runtime()
        self.tree.command(1, 'off')
        self.poll(runtime)
        self.assertEqual(runtime.state, STATE_LATCHED)
        self.assertEqual(runtime.latch_code, 'control_off')
        self.tree.command(2, 'on')
        self.poll(runtime)
        self.assertEqual(runtime.state, STATE_AVAILABLE)
        self.assertIsNone(runtime.latch_code)

    def test_the_highest_numbered_level_file_is_the_switch(self):
        self.tier()
        runtime = self.open_runtime()
        self.tree.command(5, 'off')
        self.tree.command(3, 'on')
        self.poll(runtime)
        self.assertEqual(runtime.state, STATE_LATCHED, 'off at 5 outranks on at 3')
        self.tree.command(9, 'on')
        self.poll(runtime)
        self.assertEqual(runtime.state, STATE_AVAILABLE)

    def test_on_clears_only_an_off_not_another_latch(self):
        self.tier()
        runtime = self.open_runtime()
        runtime.latch('a device op threw')
        self.tree.command(1, 'on')
        self.poll(runtime)
        self.assertEqual(runtime.state, STATE_LATCHED)
        self.assertEqual(runtime.latch_code, 'op_throw')

    def test_an_off_from_before_the_start_survives_a_restart(self):
        self.tier()
        self.tree.command(4, 'off')
        runtime = self.open_runtime()
        self.assertFalse(runtime.host_ok, 'the kill switch is re-read at every open: no worker, no device footprint')
        self.assertEqual(runtime.state, STATE_LATCHED)
        self.assertEqual(runtime.latch_code, 'control_off')
        self.assertIsNone(runtime.store)

    def test_on_after_a_start_that_was_refused_reports_that_a_restart_is_needed(self):
        self.tier()
        path = self.tree.command(4, 'off')
        runtime = self.open_runtime()
        os.unlink(path)
        self.tree.command(5, 'on')
        self.poll(runtime)
        self.assertNotEqual(runtime.state, STATE_AVAILABLE, 'the tier never started; only a restart starts it')
        self.assertIn('restart', json.dumps(runtime.snapshot(self.clock.now)['control']['acks']))

    def test_the_directory_is_polled_once_a_second_not_every_pass(self):
        self.tier()
        runtime = self.open_runtime()
        self.clock.advance(5)
        runtime.service()
        self.tree.command(1, 'off')
        runtime.service()
        self.assertEqual(runtime.state, STATE_AVAILABLE, 'same second: not polled yet')
        self.clock.advance(1.1)
        runtime.service()
        self.assertEqual(runtime.state, STATE_LATCHED)

    def test_names_that_do_not_match_are_ignored(self):
        self.tier()
        runtime = self.open_runtime()
        for name in ('off', 'x-off', '0001-offf', '0001_off', '1-reset', 'README', '0002-OFF'):
            open(os.path.join(self.tree.ctl_dir, name), 'w').close()
        self.poll(runtime)
        self.assertEqual(runtime.state, STATE_AVAILABLE)

    def test_wipe_invalidates_and_sends_every_slot_back_through_trim_once(self):
        self.tier()
        runtime = self.open_runtime()
        slots = runtime.allocator.alloc(10)
        self.assertEqual(runtime.allocator.n_used, 10)
        self.tree.command(1, 'wipe')
        self.poll(runtime)
        self.assertTrue(runtime.settle(5.0))
        self.assertEqual(runtime.allocator.n_free, 64)
        self.assertEqual(runtime.counters['wipes'], 1)
        runtime.allocator.alloc(3)
        self.poll(runtime)
        self.assertEqual(runtime.counters['wipes'], 1, 'an edge verb applies once')
        self.assertEqual(runtime.allocator.n_used, 3)

    def test_test_only_verbs_are_refused_without_the_test_flag(self):
        self.tier()
        runtime_module.set_host_env(tier_kit.make_env(self.tree, self.clock, environ={'THATCH_KVP2P_TEST': '0'}))
        self.tree.cfg_probe = None
        runtime = KVP2PRuntime.acquire(self.cfg, 'scheduler')            # attach only: the point is the verb handling, not the drive
        self.runtimes.append(runtime)
        self.tree.command(1, 'fault', 'eio 2')
        self.tree.command(2, 'selftest')
        self.poll(runtime)
        results = [ack['result'] for ack in runtime.snapshot(self.clock.now)['control']['acks']]
        self.assertTrue(all(r.startswith('refused: test-only') for r in results), results)
        self.assertFalse(runtime.take_fault('eio'))
        self.assertFalse(runtime.pop_selftest())

    def test_fault_and_selftest_work_under_the_test_flag(self):
        self.tier()
        runtime = self.open_runtime()
        self.tree.command(1, 'fault', 'eio 2')
        self.tree.command(2, 'fault', 'throw')
        self.tree.command(3, 'fault', 'bogus 1')
        self.tree.command(4, 'selftest')
        self.poll(runtime)
        self.assertTrue(runtime.take_fault('eio'))
        self.assertTrue(runtime.take_fault('eio'))
        self.assertFalse(runtime.take_fault('eio'), 'two faults were armed')
        self.assertTrue(runtime.take_fault('throw'))
        self.assertTrue(runtime.pop_selftest())
        self.assertFalse(runtime.pop_selftest())
        acks = {ack['seq']: ack['result'] for ack in runtime.snapshot(self.clock.now)['control']['acks']}
        self.assertEqual(acks[1], 'ok')
        self.assertTrue(acks[3].startswith('refused'))

    def test_delete_tenant_is_acknowledged_as_unsupported_in_v1a(self):
        self.tier()
        runtime = self.open_runtime()
        self.tree.command(1, 'delete_tenant', '0123456789abcdef')
        self.poll(runtime)
        acks = runtime.snapshot(self.clock.now)['control']['acks']
        self.assertTrue(acks[-1]['result'].startswith('unsupported'))

    def test_edge_verbs_apply_in_sequence_order_once_each(self):
        self.tier()
        runtime = self.open_runtime()
        self.tree.command(2, 'fault', 'digest')
        self.tree.command(1, 'fault', 'eio')
        self.poll(runtime)
        acks = [(a['seq'], a['verb']) for a in runtime.snapshot(self.clock.now)['control']['acks']]
        self.assertEqual(acks, [(1, 'fault'), (2, 'fault')])
        self.poll(runtime)
        self.assertEqual(len(runtime.snapshot(self.clock.now)['control']['acks']), 2)


class StatusPolling(TierCase):
    def test_an_uncorrectable_aer_increment_latches(self):
        self.tier()
        self.tree.write_status(self.clock, aer={'drive': {'correctable': 4, 'uncorrectable': 0}, 'switch_port': {'correctable': 9, 'uncorrectable': 0}})
        runtime = self.open_runtime()
        self.clock.advance(6)
        self.tree.write_status(self.clock, aer={'drive': {'correctable': 40, 'uncorrectable': 0}, 'switch_port': {'correctable': 90, 'uncorrectable': 0}})
        runtime.service()
        self.assertEqual(runtime.state, STATE_AVAILABLE, 'correctable errors are recovered by the link layer: an alert, never a latch')
        self.clock.advance(6)
        self.tree.write_status(self.clock, aer={'drive': {'correctable': 40, 'uncorrectable': 1}, 'switch_port': {'correctable': 90, 'uncorrectable': 0}})
        runtime.service()
        self.assertEqual(runtime.state, STATE_LATCHED)
        self.assertEqual(runtime.latch_code, 'aer_uncorrectable')

    def test_the_drive_dying_latches_device_lost(self):
        self.tier()
        runtime = self.open_runtime()
        self.clock.advance(6)
        self.tree.write_status(self.clock, state='not_live')
        runtime.service()
        self.assertEqual(runtime.state, STATE_LATCHED)
        self.assertEqual(runtime.latch_code, 'device_lost')

    def test_a_missing_status_file_while_running_is_not_a_latch(self):
        self.tier()
        runtime = self.open_runtime()
        os.unlink(os.path.join(self.tree.status_dir, 'status.json'))
        self.clock.advance(6)
        runtime.service()
        self.assertEqual(runtime.state, STATE_AVAILABLE)

    def test_the_aer_counters_are_read_from_any_shape(self):
        facts = health.StatusFacts
        data = {'aer': [{'dev': 'a', 'aer_dev_correctable': {'total': 3}, 'aer_dev_nonfatal': {'uncorrectable': 2}},
                        {'ce_count': 1, 'uncorrectable_total': 5}]}
        self.assertEqual(health._sum_aer(data['aer'], health._is_uncorrectable), 7)
        self.assertEqual(health._sum_aer(data['aer'], health._is_correctable), 0)
        self.assertEqual(health._sum_aer({'x': {'correctable': 3, 'uncorrectable': 1}}, health._is_correctable), 3)


class StateFile(TierCase):
    def test_the_state_file_carries_the_standard_fields(self):
        self.tier(environ={'THATCH_KVTIER_SPAWN_ID': 'spawn-1234'})
        runtime = self.open_runtime()
        runtime.write_state(self.clock.now, force=True)
        state = self.tree.read_state()
        self.assertEqual(state['schema'], 1)
        self.assertEqual(state['spawn_id'], 'spawn-1234')
        self.assertEqual(state['written_at'], health.iso_utc(self.clock.now))
        self.assertEqual(state['state'], 'AVAILABLE')
        self.assertEqual(state['persistence'], 'process')
        for key in ('latch_reason', 'counters', 'bytes_written_today', 'write_cap_bytes', 'capacity', 'restore', 'drive', 'engine_heartbeat', 'control'):
            self.assertIn(key, state)
        self.assertEqual(state['write_cap_bytes'], self.cfg.daily_write_cap_bytes)
        self.assertEqual(state['drive']['link_gts'], 16.0)

    def test_the_spawn_id_is_null_when_the_env_does_not_carry_one(self):
        self.tier()
        runtime = self.open_runtime()
        runtime.write_state(self.clock.now, force=True)
        self.assertIsNone(self.tree.read_state()['spawn_id'])

    def test_it_is_rewritten_every_ten_seconds_and_not_more_often(self):
        self.tier()
        runtime = self.open_runtime()
        runtime.write_state(self.clock.now, force=True)
        first = self.tree.read_state()['written_at']
        self.clock.advance(4)
        runtime.service()
        self.assertEqual(self.tree.read_state()['written_at'], first)
        self.clock.advance(7)
        runtime.service()
        self.assertNotEqual(self.tree.read_state()['written_at'], first)

    def test_it_is_written_atomically_and_world_readable(self):
        self.tier()
        runtime = self.open_runtime()
        runtime.write_state(self.clock.now, force=True)
        mode = os.stat(self.tree.state_file).st_mode & 0o777
        self.assertEqual(mode, 0o644)
        leftovers = [n for n in os.listdir(os.path.dirname(self.tree.state_file)) if '.tmp.' in n]
        self.assertEqual(leftovers, [])

    def test_a_state_file_that_cannot_be_written_never_breaks_the_tier(self):
        self.tier(state_file='/proc/self/no-such-dir/state.json')
        runtime = self.open_runtime()
        runtime.service()
        self.assertEqual(runtime.state, STATE_AVAILABLE)

    def test_the_heartbeat_is_the_time_of_the_last_schedule(self):
        self.tier()
        runtime = self.open_runtime()
        runtime.heartbeat(1234.5)
        runtime.write_state(self.clock.now, force=True)
        self.assertEqual(self.tree.read_state()['engine_heartbeat'], 1234.5)

    def test_restore_percentiles_come_from_the_workers_observations(self):
        self.tier()
        runtime = self.open_runtime()
        for seconds in range(1, 101):
            runtime.observe_restore(seconds / 100.0)
        runtime.write_state(self.clock.now, force=True)
        restore = self.tree.read_state()['restore']
        self.assertEqual(restore['count'], 100)
        self.assertAlmostEqual(restore['p50_s'], 0.5, delta=0.02)
        self.assertAlmostEqual(restore['p99_s'], 0.99, delta=0.02)

    def test_failures_by_cause_come_from_the_workers_counters(self):
        self.tier()
        runtime = self.open_runtime()
        runtime.count('fail_eio', 2)
        runtime.count('fail_deadline')
        runtime.count('done_store', 5)
        runtime.write_state(self.clock.now, force=True)
        counters = self.tree.read_state()['counters']
        self.assertEqual(counters['failures'], {'eio': 2, 'deadline': 1})
        self.assertEqual(counters['stores'], 5)

    def test_a_latched_state_file_names_the_reason_and_code(self):
        self.tier()
        runtime = self.open_runtime()
        runtime.latch('3 consecutive job failures (last: eio)')
        runtime.write_state(self.clock.now, force=True)
        state = self.tree.read_state()
        self.assertEqual((state['state'], state['latch_code']), ('LATCHED', 'strikes'))
        self.assertIn('consecutive', state['latch_reason'])
        self.assertEqual(state['counters']['latched'], {'strikes': 1})


if __name__ == '__main__':
    unittest.main()
