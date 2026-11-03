"""The REAL runtime and the REAL NvmeKVManager inside vLLM 0.30.0's real Scheduler, KV cache manager and OffloadingConnectorScheduler (through the
ttkvp2p subclasses), against the numpy card and the file drive: the contract test of the manager against the real ``OffloadingManager`` usage.

``harness.Rig(real_tier=...)`` swaps the fakes for the runtime and manager under test. The drive file is claimed by the real claim tool and
opened by the real host checks; the rig's FakeIO moves the bytes, the runtime owns the slots. Every step the rig asserts that every block of
every running request holds exactly the KV its hash names, so wrong KV is never accepted whatever the tier does.
"""

import errno
import unittest

import numpy as np

import tier_kit
from harness import BLOCK, Rig, tokens
from tier_kit import OBJ, RealTier

from ttkvp2p.runtime import S_USED, STATE_AVAILABLE, STATE_LATCHED

PROMPT = 2300                                # 17 full blocks + 5 sliding groups x 8 = 57 objects
OBJECTS = PROMPT // BLOCK + 5 * 8


class RealTierCase(unittest.TestCase):
    def rig(self, tier_extra=None, **kwargs) -> Rig:
        tier = RealTier(**(tier_extra or {}))
        rig = Rig(real_tier=tier, **kwargs)
        rig.tier = tier
        self.addCleanup(rig.close)
        return rig

    # --- accessors for the real manager
    def written_keys(self, rig):
        mgr = rig.manager
        return [k for k in mgr.index if mgr.entry_state(k) == 'written']

    def assert_drive_exact(self, rig) -> int:
        """Every WRITTEN entry's slot on the drive holds exactly the object its key names."""
        checked = 0
        for key in self.written_keys(rig):
            slot = rig.manager.index[key]
            self.assertTrue(np.array_equal(rig.drive.slot_object(slot), rig.expected_object(key)), 'slot %d does not hold its key\'s object' % slot)
            checked += 1
        return checked

    def stored_rig(self, name='p', **kwargs):
        rig = self.rig(**kwargs)
        prompt = tokens(PROMPT, name)
        rig.add('seed', prompt)
        rig.drain()
        self.assertEqual(rig.manager.written(), OBJECTS)
        rig.reset_device_cache()                     # the device prefix cache is empty; the drive still has the session
        return rig, prompt


class RoundTrip(RealTierCase):
    def test_a_session_is_stored_through_the_real_tier_and_restored_byte_exact(self):
        rig, prompt = self.stored_rig()
        self.assertEqual(rig.runtime.state, STATE_AVAILABLE)
        self.assertEqual(self.assert_drive_exact(rig), OBJECTS)
        rig.add('r2', prompt)
        rig.finish('r2')
        self.assertEqual(len(rig.loads), 1)
        self.assertEqual(rig.worker.stats['done_load'], 1)
        self.assertEqual(rig.manager.hits >= OBJECTS // 6, True)
        rig.drain()
        self.assertEqual(rig.runtime.allocator.total_inflight(), 0)

    def test_the_digests_the_manager_hands_the_worker_are_the_stores_source_digests(self):
        rig, prompt = self.stored_rig()
        for key in self.written_keys(rig):
            slot = rig.manager.index[key]
            found = rig.runtime.store_results.lookup(slot, rig.runtime.allocator.generation(slot))
            self.assertIsNotNone(found, 'the worker left a result for every stored slot')
            result, position = found
            self.assertTrue(np.array_equal(rig.manager._digest[slot], np.asarray(result.digests[position], dtype=np.uint64)))
            break

    def test_two_sessions_share_nothing_and_both_restore(self):
        rig = self.rig()
        a, b = tokens(PROMPT, 'a'), tokens(PROMPT, 'b')
        rig.add('a', a)
        rig.add('b', b)
        rig.drain()
        self.assertEqual(rig.manager.written(), 2 * OBJECTS)
        rig.reset_device_cache()
        rig.add('a2', a)
        rig.add('b2', b)
        rig.finish('a2', 'b2')
        self.assertEqual(rig.worker.stats['done_load'], 2)
        self.assertEqual(self.assert_drive_exact(rig), 2 * OBJECTS)

    def test_the_state_file_reports_what_happened(self):
        rig, prompt = self.stored_rig()
        rig.add('r2', prompt)
        rig.finish('r2')
        rig.runtime.write_state(rig.tier.clock.now, force=True)
        state = rig.tier.tree.read_state()
        self.assertEqual(state['state'], 'AVAILABLE')
        self.assertEqual(state['counters']['stores'], rig.worker.stats['done_store'])
        self.assertEqual(state['counters']['loads'], 1)
        self.assertEqual(state['capacity']['entries'], OBJECTS)
        self.assertEqual(state['bytes_written_today'], OBJECTS * OBJ)
        self.assertGreaterEqual(state['restore']['count'], 1)


class FailedStores(RealTierCase):
    """The seam of the failed store: vLLM's stock update_connector_output calls complete_store(keys, ctx) with no success flag and no job id.
    The manager finds the job's own result by (slot, generation) and drops the keys of a failed job."""

    def test_a_failed_store_job_leaves_no_entry_that_a_lookup_can_reach(self):
        rig = self.rig()
        rig.io.write_errors.append(OSError(errno.EIO, 'media error'))
        rig.add('r1', tokens(PROMPT, 'x'))
        rig.drain()
        self.assertEqual(rig.worker.stats['fail_io'], 1)
        mgr = rig.manager
        self.assertGreater(mgr.evictions['store_failed'], 0, 'the failed job\'s entries were dropped')
        states = {mgr.entry_state(k) for k in mgr.index}
        self.assertEqual(states, {'written'} if mgr.index else set(), 'no entry is left half-written or in flight')
        failed = [r for r in rig.runtime.store_results.values() if not r.success]
        self.assertEqual(len(failed), 1)
        for slot, generation in zip(failed[0].slot_ids, failed[0].generations):
            self.assertNotEqual(rig.runtime.allocator.state_name(slot), 'used' if slot not in mgr.index.values() else 'x')
            self.assertNotIn(slot, mgr.index.values(), 'the failed job\'s slot is not named by the index')
        self.assertEqual(self.assert_drive_exact(rig), mgr.written())

    def test_after_a_failed_store_the_same_prompt_is_computed_not_restored_from_garbage(self):
        rig = self.rig()
        for _ in range(3):
            rig.io.write_errors.append(OSError(errno.EIO, 'media error'))
        rig.add('r1', tokens(PROMPT, 'x'))
        rig.drain()
        rig.reset_device_cache()
        rig.add('r2', tokens(PROMPT, 'x'))
        rig.finish('r2')                                     # the rig asserts exact KV at every step
        self.assertEqual(self.assert_drive_exact(rig), rig.manager.written())

    def test_the_gather_flush_failure_without_a_spill_slot_is_a_dropped_store(self):
        rig = self.rig()
        rig.add('r1', tokens(PROMPT, 'x'))
        rig.drain()
        self.assertEqual(rig.manager.written(), OBJECTS)
        self.assertEqual(rig.runtime.allocator.count(S_USED), OBJECTS)


class FailedLoads(RealTierCase):
    def test_an_eio_marks_the_keys_bad_and_the_manager_removes_them_when_the_load_completes(self):
        rig, prompt = self.stored_rig()
        rig.io.read_errors.append(OSError(errno.EIO, 'media error'))
        rig.add('r2', prompt)
        rig.finish('r2', limit=600)
        self.assertEqual(rig.worker.stats['fail_eio'], 1)
        self.assertEqual(len(rig.loads), 1, 'the recompute did not look up and defer again')
        self.assertLess(rig.manager.written(), OBJECTS)
        self.assertGreater(rig.manager.evictions['bad'], 0)
        self.assertEqual(rig.runtime.bad, set(), 'the marks were forgotten with the removed entries')
        self.assertEqual(self.assert_drive_exact(rig), rig.manager.written())
        rig.drain()
        self.assertEqual(rig.runtime.allocator.total_inflight(), 0)

    def test_a_persistent_digest_mismatch_marks_exactly_that_object_bad(self):
        rig, prompt = self.stored_rig()
        victim = next(k for k in self.written_keys(rig) if int.from_bytes(k[-4:], 'big') == 0)
        slot = rig.manager.index[victim]
        rig.drive.corrupt(slot)
        rig.add('r2', prompt)
        rig.finish('r2', limit=600)
        self.assertEqual(rig.worker.stats['fail_digest'], 1)
        self.assertNotIn(victim, rig.manager.index)
        self.assertEqual(rig.manager.written(), OBJECTS - 1)

    def test_a_deadline_keeps_the_keys_valid_for_other_requests(self):
        rig, prompt = self.stored_rig(extra={'load_deadline_base_s': 0.05})
        hold, = rig.io.hold_next()
        rig.add('r2', prompt)
        rig.finish('r2', limit=2000)
        self.assertEqual(rig.worker.stats['fail_deadline'], 1)
        self.assertGreater(rig.manager.miss_reasons['no_retry'], 0, 'the flag turned the recompute\'s lookups into misses')
        self.assertEqual(rig.manager.written(), OBJECTS, 'a deadline never invalidates the keys (I13)')
        hold.set()
        rig.drain()
        rig.run_until(lambda: not rig.worker.quarantined_banks, what='the quarantine released')
        rig.reset_device_cache()
        rig.add('r3', prompt)
        rig.finish('r3')
        self.assertEqual(rig.worker.stats['done_load'], 1)


class ResetWithSlotReuse(RealTierCase):
    def test_a_reset_during_a_load_never_reuses_a_slot_under_a_bio(self):
        rig, prompt = self.stored_rig(workers=2, num_slots=120)
        hold, = rig.io.hold_next()
        rig.add('r2', prompt)
        rig.run_until(lambda: any(c[0] == 'read' for c in rig.io.calls), what='the read issued')
        allocator = rig.runtime.allocator
        under_bio = [int(s) for s in np.nonzero(allocator.inflight)[0]]
        self.assertTrue(under_bio)
        self.assertTrue(rig.scheduler.reset_connector_cache())                  # POST /reset_prefix_cache?reset_external=true
        self.assertEqual(rig.manager.entries, 0)
        self.assertEqual(rig.manager.epoch, 1)
        for slot in under_bio:
            self.assertEqual(allocator.state_name(slot), 'draining', 'a slot with a bio in flight is not allocatable (I5)')
        rig.add('r3', tokens(PROMPT, 'after the reset'))                      # a new session stores while the old bio is still out
        rig.finish('r2', 'r3', limit=1500)
        for slot in under_bio:
            self.assertNotIn(slot, set(rig.manager.index.values()), 'a slot with a bio in flight was handed to a new entry')
        self.assertEqual(rig.worker.stats['fail_reset'], 1)
        hold.set()
        rig.drain()
        rig.run_until(lambda: not rig.worker.quarantined_banks and allocator.total_inflight() == 0, what='everything returned')
        self.assertEqual(self.assert_drive_exact(rig), rig.manager.written())
        self.assertGreater(rig.manager.written(), 0)

    def test_after_a_reset_the_old_session_is_gone_and_the_next_one_stores_normally(self):
        rig, prompt = self.stored_rig()
        self.assertTrue(rig.scheduler.reset_connector_cache())
        rig.runtime.settle()
        rig.add('r2', prompt)
        rig.finish('r2')
        self.assertEqual(len(rig.loads), 0, 'nothing to restore: the index was invalidated')
        rig.drain()
        self.assertEqual(rig.manager.written(), OBJECTS)


class TtlAndControl(RealTierCase):
    def test_a_session_older_than_the_ttl_is_not_restored(self):
        rig, prompt = self.stored_rig(tier_extra={'ttl_s': 3600})
        clock = rig.tier.clock
        rig.step()
        clock.advance(3000)
        rig.step()
        rig.add('young', prompt)
        rig.finish('young')
        self.assertEqual(len(rig.loads), 1, 'inside the TTL it restores')
        rig.drain()
        rig.reset_device_cache()
        rig.step()
        clock.advance(3601)
        rig.step()
        rig.step()
        self.assertEqual(rig.manager.entries, 0, 'expired at the end of a pass')
        rig.add('old', prompt)
        rig.finish('old')
        self.assertEqual(len(rig.loads), 1, 'no second load: it was computed')

    def test_the_kill_switch_stops_hits_and_stores_and_the_engine_carries_on(self):
        rig, prompt = self.stored_rig()
        rig.tier.tree.command(1, 'off')
        rig.tier.clock.advance(2)
        rig.step()
        rig.step()
        self.assertEqual(rig.runtime.state, STATE_LATCHED)
        self.assertEqual(rig.runtime.latch_code, 'control_off')
        rig.add('r2', prompt)
        rig.add('r3', tokens(PROMPT, 'fresh'))
        rig.finish('r2', 'r3')
        self.assertEqual(len(rig.loads), 0)
        self.assertEqual(rig.manager.written(), OBJECTS, 'nothing new was stored')

    def test_off_then_on_brings_the_tier_back(self):
        rig, prompt = self.stored_rig()
        rig.tier.tree.command(1, 'off')
        rig.tier.clock.advance(2)
        rig.step()
        rig.step()
        self.assertEqual(rig.runtime.state, STATE_LATCHED)
        rig.tier.tree.command(2, 'on')
        rig.tier.clock.advance(2)
        rig.step()
        rig.step()
        self.assertEqual(rig.runtime.state, STATE_AVAILABLE)
        rig.add('r2', prompt)
        rig.finish('r2')
        self.assertEqual(len(rig.loads), 1)

    def test_a_latch_by_the_worker_turns_lookups_into_misses(self):
        rig, prompt = self.stored_rig()
        rig.runtime.latch('3 consecutive job failures (last: eio)')
        rig.add('r2', prompt)
        rig.finish('r2')
        self.assertEqual(len(rig.loads), 0)


class AllOrNothingInTheRealScheduler(RealTierCase):
    """I12 with the real scheduler: the connector scheduler's tag carries the prompt size (the WP4 change this report asks for) and the
    manager knows the KV groups; a request whose whole boundary does not fit stores NOTHING."""

    def rig_with_prompt_sizes(self, **kwargs):
        rig = self.rig(**kwargs)
        scheduler = rig.connector_scheduler
        original = scheduler.on_new_request

        def tagged(request):
            original(request)
            tier_kit.ReqTag
            scheduler._req_status[request.request_id].req_context.get_state(tier_kit.ReqTag).num_prompt_tokens = request.num_prompt_tokens

        scheduler.on_new_request = tagged
        rig.manager.bind_groups((None, 8, 8, 8, 8, 8))
        return rig

    def test_a_boundary_that_does_not_fit_stores_not_even_its_first_chunk(self):
        rig = self.rig_with_prompt_sizes(num_slots=100, max_batched=1024)       # 2300 tokens prefill in 3 chunks
        rig.add('a', tokens(PROMPT, 'a'))
        rig.add('b', tokens(PROMPT, 'b'))
        rig.drain()
        mgr = rig.manager
        self.assertEqual(mgr.admitted_objects, OBJECTS, 'one whole boundary admitted')
        self.assertEqual(mgr.written(), OBJECTS, 'and exactly one: the other request wrote nothing, not even a prefix of its prompt')
        self.assertEqual(mgr.dropped['no_slot'] + mgr.dropped['capacity'], 1)
        self.assertGreaterEqual(mgr.dropped['denied'], 1, 'its later chunks were dropped on the cached verdict')
        self.assertEqual(self.assert_drive_exact(rig), OBJECTS)
        self.assertEqual(rig.runtime.allocator.count(S_USED), OBJECTS)

    def test_both_fit_when_the_drive_has_room(self):
        rig = self.rig_with_prompt_sizes(num_slots=200, max_batched=1024)
        rig.add('a', tokens(PROMPT, 'a'))
        rig.add('b', tokens(PROMPT, 'b'))
        rig.drain()
        self.assertEqual(rig.manager.written(), 2 * OBJECTS)
        self.assertEqual(rig.manager.snapshot()['reserved_slots'], 0)

    def test_the_daily_cap_denies_a_whole_boundary(self):
        rig = self.rig_with_prompt_sizes(num_slots=200, max_batched=1024, tier_extra={'daily_write_cap_bytes': 100 * OBJ})
        rig.add('a', tokens(PROMPT, 'a'))
        rig.add('b', tokens(PROMPT, 'b'))
        rig.drain()
        self.assertEqual(rig.manager.written(), OBJECTS)
        self.assertEqual(rig.manager.dropped['cap'], 1)
        self.assertEqual(rig.manager.snapshot()['bytes_written_today'], OBJECTS * OBJ)


class Shutdown(RealTierCase):
    def test_shutting_down_the_engine_releases_the_runtime_in_the_right_order(self):
        rig = self.rig()
        runtime = rig.runtime
        rig.add('a', tokens(PROMPT, 'a'))
        rig.drain()
        self.assertEqual(sorted(runtime.roles), ['scheduler', 'worker'])
        rig.close()
        self.assertEqual(runtime.roles, [])
        self.assertEqual(runtime.state, 'CLOSING')
        self.assertIsNone(runtime.store)


if __name__ == '__main__':
    unittest.main()
