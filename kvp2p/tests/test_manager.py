"""NvmeKVManager against vLLM 0.30.0's OffloadingManager contract: lookup / prepare / complete, the I5, I6, I7, I12 and I13 rules, TTL with a
frozen clock, LRU and the capacity floor, the daily write cap, all-or-nothing admission, reset_cache, BAD keys.

These tests call the manager the way the real OffloadingConnectorScheduler does (keys as lists, job keys as sets, a ``ReqContext`` carrying
the connector's ``ReqTag``); the scheduler-driven tests are in test_tier_rig.py.
"""

import unittest

import numpy as np

import tier_kit
from tier_kit import OBJ, TierCase, key_for, req

from vllm.v1.kv_offload.base import (
    LookupResult, Medium, OffloadingManager, PrepareStoreOutput, RequestOffloadingContext, ScheduleEndContext,
)
from vllm.v1.kv_offload.cpu.manager import CPUOffloadingManager

from ttkvp2p.manager import ALLOCATED_REAP_S, NvmeKVManager, PairingError, estimate_objects
from ttkvp2p.runtime import S_TRIM_WAIT, S_USED, STATE_LATCHED
from ttkvp2p.spec import NvmeLoadStoreSpec
from ttkvp2p.worker import StoreResult

GEMMA_GROUPS = (None, 8, 8, 8, 8, 8)


def digest_rows(slots):
    """A deterministic digest per object: [n, 4] uint64 derived from the slot."""
    return np.array([[s * 7 + 1, s * 7 + 2, s * 7 + 3, s * 7 + 4] for s in slots], dtype=np.uint64)


class ManagerCase(TierCase):
    def build(self, num_slots=64, **cfg):
        self.tier(num_slots=num_slots, **cfg)
        self.runtime = self.open_runtime()
        self.mgr = self.manager(self.runtime)
        self.job = 0
        return self.mgr

    def end_pass(self):
        self.mgr.on_schedule_end(ScheduleEndContext(new_req_ids=(), preempted_req_ids=()))

    def settle(self):
        self.assertTrue(self.runtime.settle(5.0))

    def store(self, keys, ctx):
        return self.mgr.prepare_store(keys, ctx)

    def finish_store(self, out, ctx, ok=True, via='runtime'):
        """The worker reports the job: a StoreResult in the runtime, then the scheduler's complete_store(job.keys as a SET, ctx)."""
        self.job += 1
        spec = out.store_spec
        result = StoreResult(self.job, ok, tuple(spec.slot_ids), tuple(spec.generations), digest_rows(spec.slot_ids) if ok else None,
                             '' if ok else 'eio')
        self.runtime.store_results[self.job] = result
        if via == 'result':
            self.mgr.complete_store(set(out.keys_to_store), ctx, success=ok, result=result)
        else:
            self.mgr.complete_store(set(out.keys_to_store), ctx)           # exactly what vLLM's update_connector_output calls
        return result

    def put(self, count, ctx=None, first=0, group=0):
        """Store ``count`` fresh keys and complete the job."""
        ctx = ctx or req('seed')
        keys = [key_for(first + i, group) for i in range(count)]
        out = self.store(keys, ctx)
        self.assertEqual(out.keys_to_store, keys)
        self.finish_store(out, ctx)
        return keys


class Contract(ManagerCase):
    """The shape vLLM 0.30.0's OffloadingConnectorScheduler relies on."""

    def test_it_is_a_cpu_offloading_manager_for_the_storage_medium(self):
        mgr = self.build()
        self.assertIsInstance(mgr, CPUOffloadingManager)
        self.assertIsInstance(mgr, OffloadingManager)
        self.assertIs(mgr.medium, Medium.STORAGE)

    def test_the_abstract_methods_are_all_implemented(self):
        self.build()
        self.assertEqual(NvmeKVManager.__abstractmethods__, frozenset())

    def test_on_new_request_returns_a_request_context(self):
        mgr = self.build()
        self.assertIsInstance(mgr.on_new_request(req('a')), RequestOffloadingContext)

    def test_lookup_returns_a_lookup_result_for_every_state(self):
        mgr = self.build()
        ctx = req('a')
        self.assertIs(mgr.lookup(key_for(1), ctx), LookupResult.MISS)
        out = self.store([key_for(1)], ctx)
        self.assertIs(mgr.lookup(key_for(1), ctx), LookupResult.HIT_PENDING)
        self.finish_store(out, ctx)
        self.assertIs(mgr.lookup(key_for(1), ctx), LookupResult.HIT)

    def test_prepare_store_returns_a_store_output_never_none_and_a_refusal_is_an_empty_key_list(self):
        mgr = self.build(num_slots=4)
        ctx = req('a')
        out = self.store([key_for(i) for i in range(4)], ctx)
        self.assertIsInstance(out, PrepareStoreOutput)
        self.assertEqual(len(out.keys_to_store), 4)
        self.assertIsInstance(out.store_spec, NvmeLoadStoreSpec)
        self.assertIs(out.store_spec.medium, Medium.STORAGE)
        self.assertEqual(out.evicted_keys, [])
        full = self.store([key_for(100 + i) for i in range(3)], req('b'))
        self.assertIsInstance(full, PrepareStoreOutput, 'None would be a warning and a retry on every step')
        self.assertEqual(full.keys_to_store, [])
        self.assertEqual(full.store_spec.slot_ids, [])

    def test_a_cached_verdict_with_no_slot_left_is_still_an_empty_drop_never_none(self):
        mgr = self.build(num_slots=6)
        a = req('a')
        self.assertEqual(len(self.store([key_for(1)], a).keys_to_store), 1)          # admitted; the verdict is cached in a's tag
        self.assertEqual(len(self.store([key_for(100 + i) for i in range(5)], req('b')).keys_to_store), 5)     # the rest of the drive
        out = self.store([key_for(2), key_for(3)], a)
        self.assertIsNotNone(out)
        self.assertEqual(out.keys_to_store, [])

    def test_the_state_names_are_the_ones_the_worker_compares_against(self):
        from ttkvp2p import runtime as runtime_module, spec
        for name in ('STATE_UNAVAILABLE', 'STATE_AVAILABLE', 'STATE_LATCHED', 'STATE_CLOSING'):
            self.assertEqual(getattr(runtime_module, name), getattr(spec, name))

    def test_a_manager_built_before_the_drive_is_open_picks_the_slot_table_up_when_it_opens(self):
        self.tier(num_slots=16)
        attached = KVP2PRuntimeScheduler(self.cfg)
        mgr = self.mgr = self.manager(attached)
        self.assertIs(mgr.lookup(key_for(1), req('a')), LookupResult.MISS)
        self.assertEqual(self.store([key_for(1)], req('a')).keys_to_store, [], 'the tier is not up: stores drop')
        self.runtime = self.open_runtime()
        self.job = 0
        self.end_pass()
        keys = self.put(3)
        self.assertEqual(mgr.written(), 3)
        self.assertIs(mgr.lookup(keys[0], req('b')), LookupResult.HIT)

    def test_keys_already_held_are_not_stored_again_and_an_all_held_call_is_a_drop(self):
        mgr = self.build()
        ctx = req('a')
        keys = self.put(5)
        out = self.store(keys + [key_for(50)], ctx)
        self.assertEqual(out.keys_to_store, [key_for(50)])
        again = self.store(keys, ctx)
        self.assertEqual(again.keys_to_store, [])

    def test_the_scheduler_calls_work_with_collections_of_any_kind(self):
        mgr = self.build()
        keys = self.put(4)
        ctx = req('a')
        mgr.touch(keys, ctx)
        mgr.touch(tuple(keys), ctx)
        spec = mgr.prepare_load(tuple(keys[:2]), ctx)
        mgr.complete_load(set(keys[:2]), ctx)
        self.assertEqual(len(spec.slot_ids), 2)

    def test_the_manager_has_no_pending_work_and_no_stats_events_of_its_own(self):
        mgr = self.build()
        self.assertFalse(mgr.has_pending_work())
        self.assertEqual(list(mgr.take_events()), [])
        self.assertIsNone(mgr.get_stats())

    def test_shutdown_releases_the_scheduler_role_once(self):
        mgr = self.build()
        before = list(self.runtime.roles)
        self.assertIn('scheduler', before) if 'scheduler' in before else None
        self.runtime.roles.append('scheduler')
        mgr.shutdown()
        mgr.shutdown()
        self.assertEqual(self.runtime.roles.count('scheduler'), 0)
        self.assertIsNone(self.runtime._manager)


class LookupRules(ManagerCase):
    def test_miss_when_the_tier_is_not_available(self):
        mgr = self.build()
        keys = self.put(1)
        ctx = req('a')
        self.assertIs(mgr.lookup(keys[0], ctx), LookupResult.HIT)
        self.runtime.latch('control off')
        self.assertIs(mgr.lookup(keys[0], ctx), LookupResult.MISS)
        self.assertEqual(mgr.miss_reasons['unavailable'], 1)
        self.assertEqual(self.store([key_for(77)], ctx).keys_to_store, [], 'a latched tier drops stores too')
        self.assertEqual(mgr.dropped['unavailable'], 1)

    def test_a_lookup_before_the_runtime_is_open_misses(self):
        self.tier()
        runtime = KVP2PRuntimeScheduler(self.cfg)
        mgr = self.manager(runtime)
        self.assertIs(mgr.lookup(key_for(1), req('a')), LookupResult.MISS)

    def test_miss_for_a_request_whose_load_failed_without_proof_the_no_retry_flag(self):
        mgr = self.build()
        keys = self.put(2)
        ctx = req('a', no_retry=True)
        self.assertIs(mgr.lookup(keys[0], ctx), LookupResult.MISS)
        self.assertIs(mgr.lookup(keys[0], req('b')), LookupResult.HIT, 'the keys stay valid for other requests (I13)')
        self.assertEqual(mgr.miss_reasons['no_retry'], 1)

    def test_miss_for_a_bad_entry_even_before_its_loads_complete(self):
        mgr = self.build()
        keys = self.put(2)
        ctx = req('a')
        spec = mgr.prepare_load(keys, ctx)
        self.runtime.mark_bad(spec.slot_ids[0], spec.generations[0])
        self.assertIs(mgr.lookup(keys[0], req('b')), LookupResult.MISS)
        self.assertIs(mgr.lookup(keys[1], req('b')), LookupResult.HIT)

    def test_expired_entries_miss_against_the_frozen_now(self):
        mgr = self.build(ttl_s=1000)
        keys = self.put(1)
        self.end_pass()
        self.clock.advance(999)
        self.assertIs(mgr.lookup(keys[0], req('a')), LookupResult.HIT)
        self.end_pass()
        self.clock.advance(2)
        self.assertIs(mgr.lookup(keys[0], req('a')), LookupResult.MISS)
        self.assertEqual(mgr.miss_reasons['expired'], 1)

    def test_a_foreign_tenant_misses_only_when_a_salt_is_required(self):
        mgr = self.build(require_salt=True)
        owner = req('a', tenant=b'\x01' * 8)
        out = self.store([key_for(1)], owner)
        self.finish_store(out, owner)
        self.assertIs(mgr.lookup(key_for(1), req('b', tenant=b'\x01' * 8)), LookupResult.HIT)
        self.assertIs(mgr.lookup(key_for(1), req('c', tenant=b'\x02' * 8)), LookupResult.MISS)
        self.assertIs(mgr.lookup(key_for(1), req('d')), LookupResult.MISS, 'no salt, no sharing')
        self.assertIs(mgr.lookup(key_for(1), req('e', tag=False)), LookupResult.MISS)
        self.assertEqual(mgr.miss_reasons['foreign_tenant'], 1)
        self.assertEqual(mgr.miss_reasons['no_salt'], 2)

    def test_tenants_are_ignored_when_no_salt_is_required(self):
        mgr = self.build(require_salt=False)
        owner = req('a', tenant=b'\x01' * 8)
        out = self.store([key_for(1)], owner)
        self.finish_store(out, owner)
        self.assertIs(mgr.lookup(key_for(1), req('c', tenant=b'\x02' * 8)), LookupResult.HIT)
        self.assertIs(mgr.lookup(key_for(1), req('d')), LookupResult.HIT)

    def test_a_store_that_never_completes_stops_deferring_lookups(self):
        mgr = self.build()
        ctx = req('a')
        self.store([key_for(1)], ctx)
        self.assertIs(mgr.lookup(key_for(1), req('b')), LookupResult.HIT_PENDING)
        self.end_pass()
        self.clock.advance(ALLOCATED_REAP_S + 1)
        self.assertIs(mgr.lookup(key_for(1), req('b')), LookupResult.MISS)
        self.end_pass()
        self.assertEqual(mgr.entries, 0, 'the stuck entry is given up at the end of the pass')
        self.assertEqual(self.runtime.allocator.count(S_USED), 0)

    def test_multimodal_requests_may_hit_but_never_store(self):
        mgr = self.build()
        keys = self.put(2)
        mm = req('mm', has_mm=True)
        self.assertIs(mgr.lookup(keys[0], mm), LookupResult.HIT)
        out = self.store([key_for(90)], mm)
        self.assertEqual(out.keys_to_store, [])
        self.assertEqual(mgr.dropped['multimodal'], 1)


def KVP2PRuntimeScheduler(cfg):
    from ttkvp2p.runtime import KVP2PRuntime
    return KVP2PRuntime.acquire(cfg, 'scheduler')


class StoreOrder(ManagerCase):
    """I6: slot ids come back 1:1 with keys_to_store, in the order asked, and the manager asserts it."""

    def test_slots_pair_one_to_one_with_the_keys_in_the_order_asked(self):
        mgr = self.build()
        keys = [key_for(i, group=i % 6) for i in (9, 3, 7, 1, 8, 2)]
        out = self.store(keys, req('a'))
        self.assertEqual(out.keys_to_store, keys)
        index = mgr.index
        self.assertEqual([index[k] for k in out.keys_to_store], out.store_spec.slot_ids)
        self.assertEqual(len(set(out.store_spec.slot_ids)), len(keys))
        self.assertEqual(out.store_spec.generations, [self.runtime.allocator.generation(s) for s in out.store_spec.slot_ids])

    def test_a_shuffling_allocator_cannot_break_the_pairing(self):
        mgr = self.build()
        real_alloc = self.runtime.allocator.alloc

        def shuffled(count):
            slots = real_alloc(count)
            return None if slots is None else slots[::2] + slots[1::2][::-1]

        self.runtime.allocator.alloc = shuffled
        keys = [key_for(i) for i in range(9)]
        out = self.store(keys, req('a'))
        self.assertEqual(out.keys_to_store, keys)
        index = mgr.index
        for key, slot, generation in zip(out.keys_to_store, out.store_spec.slot_ids, out.store_spec.generations):
            self.assertEqual(index[key], slot, 'slot_ids[i] is the slot of keys_to_store[i], whatever order the allocator used')
            self.assertEqual(self.runtime.allocator.generation(slot), generation)

    def test_a_pairing_that_does_not_hold_is_caught_rolled_back_and_latches(self):
        mgr = self.build()
        original = mgr._check_pairing

        def corrupt(output):
            output.store_spec.slot_ids[0], output.store_spec.slot_ids[1] = output.store_spec.slot_ids[1], output.store_spec.slot_ids[0]
            return original(output)

        mgr._check_pairing = corrupt
        out = self.store([key_for(1), key_for(2), key_for(3)], req('a'))
        self.assertEqual(out.keys_to_store, [], 'nothing is stored under a pairing nobody can verify')
        self.assertEqual(mgr.entries, 0)
        self.assertEqual(self.runtime.allocator.count(S_USED), 0, 'the slots went back')
        self.assertEqual(self.runtime.state, STATE_LATCHED)
        self.assertEqual(self.runtime.latch_code, 'op_throw')
        self.assertEqual(mgr.dropped['pairing'], 1)

    def test_check_pairing_rejects_each_kind_of_swap(self):
        mgr = self.build()
        out = self.store([key_for(1), key_for(2)], req('a'))
        mgr._check_pairing(out)
        spec = out.store_spec
        for broken in (NvmeLoadStoreSpec(spec.slot_ids[::-1], spec.generations), NvmeLoadStoreSpec(spec.slot_ids, spec.generations[::-1][:1] + [99]),
                       NvmeLoadStoreSpec(spec.slot_ids[:1], spec.generations[:1]), NvmeLoadStoreSpec([50, 51], spec.generations)):
            with self.assertRaises(PairingError):
                mgr._check_pairing(PrepareStoreOutput(out.keys_to_store, broken, []))

    def test_duplicate_keys_in_one_call_are_stored_once(self):
        mgr = self.build()
        out = self.store([key_for(1), key_for(2), key_for(1)], req('a'))
        self.assertEqual(out.keys_to_store, [key_for(1), key_for(2)])

    def test_store_threshold_two_stores_a_key_from_its_second_offer(self):
        mgr = self.build(store_threshold=2)
        first = self.store([key_for(1), key_for(2)], req('a'))
        self.assertEqual(first.keys_to_store, [])
        second = self.store([key_for(1)], req('b'))
        self.assertEqual(second.keys_to_store, [key_for(1)])


class CompleteStore(ManagerCase):
    """The failed-store seam: complete_store(keys, ctx, success=...) with the digests of the job's own result, found by (slot, generation)."""

    def test_a_successful_store_makes_the_keys_loadable_with_the_workers_digests(self):
        mgr = self.build()
        keys = [key_for(i) for i in range(5)]
        out = self.store(keys, req('a'))
        result = self.finish_store(out, req('a'))
        spec = mgr.prepare_load(keys, req('b'))
        self.assertTrue(np.array_equal(spec.expected, result.digests), 'the expected digests are the source digests of the store (I4)')
        self.assertEqual(spec.slot_ids, out.store_spec.slot_ids)

    def test_the_stock_call_with_a_set_of_keys_in_any_order_pairs_by_slot_not_by_position(self):
        mgr = self.build()
        keys = [key_for(i) for i in range(30)]
        out = self.store(keys, req('a'))
        self.finish_store(out, req('a'))                    # complete_store(set(keys), ctx): a set has no order
        spec = mgr.prepare_load(keys, req('b'))
        for key, slot, row in zip(keys, spec.slot_ids, spec.expected):
            self.assertEqual(int(row[0]), slot * 7 + 1, 'key %d got another object\'s digest' % int.from_bytes(key[:32], 'big'))

    def test_success_false_drops_the_entries_and_the_slots_go_through_trim(self):
        mgr = self.build()
        keys = [key_for(i) for i in range(4)]
        out = self.store(keys, req('a'))
        self.finish_store(out, req('a'), ok=False)
        self.assertEqual(mgr.entries, 0)
        for key in keys:
            self.assertIs(mgr.lookup(key, req('b')), LookupResult.MISS)
        self.assertEqual(self.runtime.allocator.count(S_TRIM_WAIT), 4, 'released under I5: not allocatable before the TRIM')
        self.assertEqual(mgr.evictions['store_failed'], 4)

    def test_an_explicit_failure_flag_beats_a_result_that_says_success(self):
        mgr = self.build()
        keys = [key_for(i) for i in range(3)]
        out = self.store(keys, req('a'))
        self.job += 1
        result = StoreResult(self.job, True, tuple(out.store_spec.slot_ids), tuple(out.store_spec.generations), digest_rows(out.store_spec.slot_ids))
        self.runtime.store_results[self.job] = result
        mgr.complete_store(set(keys), req('a'), success=False)
        self.assertEqual(mgr.entries, 0)

    def test_a_missing_or_pruned_result_means_the_store_failed(self):
        mgr = self.build()
        keys = [key_for(i) for i in range(3)]
        out = self.store(keys, req('a'))
        mgr.complete_store(set(keys), req('a'))             # the worker never left a result (or pruned it)
        self.assertEqual(mgr.entries, 0)

    def test_a_result_for_another_generation_of_the_slot_does_not_count(self):
        mgr = self.build()
        keys = [key_for(1)]
        out = self.store(keys, req('a'))
        self.job += 1
        stale = StoreResult(self.job, True, tuple(out.store_spec.slot_ids), (out.store_spec.generations[0] + 1,), digest_rows(out.store_spec.slot_ids))
        self.runtime.store_results[self.job] = stale
        mgr.complete_store(set(keys), req('a'))
        self.assertEqual(mgr.entries, 0, 'never publish bytes that a different store wrote')

    def test_a_failed_job_result_fails_every_key_of_the_job(self):
        mgr = self.build()
        keys = [key_for(i) for i in range(3)]
        out = self.store(keys, req('a'))
        self.finish_store(out, req('a'), ok=False)
        self.assertEqual(mgr.entries, 0)

    def test_the_result_can_be_handed_over_directly(self):
        mgr = self.build()
        keys = [key_for(i) for i in range(6)]
        out = self.store(keys, req('a'))
        self.finish_store(out, req('a'), via='result')
        self.assertEqual(mgr.written(), 6)
        spec = mgr.prepare_load(keys, req('b'))
        self.assertEqual(int(spec.expected[2][0]), out.store_spec.slot_ids[2] * 7 + 1)

    def test_a_handed_over_failure_drops_the_keys(self):
        mgr = self.build()
        keys = [key_for(i) for i in range(6)]
        out = self.store(keys, req('a'))
        self.finish_store(out, req('a'), ok=False, via='result')
        self.assertEqual(mgr.entries, 0)

    def test_pairing_is_an_index_lookup_not_a_scan_over_the_results(self):
        mgr = self.build(num_slots=3000)
        for job in range(1, 1025):                          # the worker keeps up to 1,024 reported results
            slots = (job % 2900,)
            self.runtime.store_results[job] = StoreResult(job, True, slots, (1,), digest_rows(slots))
        ctx = req('a')
        keys = [key_for(i) for i in range(2088)]            # a 262k session
        out = self.store(keys, ctx)
        self.job = 5000
        result = StoreResult(self.job, True, tuple(out.store_spec.slot_ids), tuple(out.store_spec.generations), digest_rows(out.store_spec.slot_ids))
        self.runtime.store_results[self.job] = result
        calls = []
        real_lookup = self.runtime.store_results.lookup
        self.runtime.store_results.lookup = lambda s, g: calls.append((s, g)) or real_lookup(s, g)
        import time
        started = time.perf_counter()
        mgr.complete_store(set(keys), ctx)
        elapsed = time.perf_counter() - started
        self.assertEqual(len(calls), 2088, 'one index lookup per key')
        self.assertEqual(mgr.written(), 2088)
        self.assertLess(elapsed, 1.0)

    def test_a_repeated_completion_and_a_completion_after_a_reset_change_nothing(self):
        mgr = self.build()
        keys = [key_for(i) for i in range(3)]
        out = self.store(keys, req('a'))
        result = self.finish_store(out, req('a'))
        mgr.complete_store(set(keys), req('a'))
        self.assertEqual(mgr.written(), 3)
        mgr.reset_cache()
        mgr.complete_store(set(keys), req('a'))
        mgr.complete_store(set(keys), req('a'), success=False)
        self.assertEqual(mgr.entries, 0)


class LoadsAndPins(ManagerCase):
    def test_prepare_load_pins_and_copies_the_expected_digests(self):
        mgr = self.build()
        keys = self.put(6)
        ctx = req('a')
        spec = mgr.prepare_load(keys[:3], ctx)
        self.assertEqual(spec.slot_ids, [mgr.index[k] for k in keys[:3]])
        self.assertEqual(spec.generations, [self.runtime.allocator.generation(s) for s in spec.slot_ids])
        self.assertEqual(spec.expected.shape, (3, 4))
        self.assertEqual(spec.expected.dtype, np.uint64)
        spec.expected[0, 0] = 12345                       # a copy: changing the job's digests does not touch the index
        again = mgr.prepare_load(keys[:1], ctx)
        self.assertNotEqual(int(again.expected[0, 0]), 12345)
        empty = mgr.prepare_load([], ctx)
        self.assertEqual((empty.slot_ids, empty.generations, empty.expected.shape), ([], [], (0, 4)))

    def test_a_pinned_entry_survives_pressure_and_expiry_until_its_load_completes(self):
        mgr = self.build(ttl_s=100, num_slots=20)
        keys = self.put(5)
        ctx = req('a')
        mgr.prepare_load(keys[:2], ctx)
        self.end_pass()
        self.clock.advance(10_000)
        self.end_pass()
        self.assertEqual(mgr.entries, 2, 'only the pinned ones are left: they are being read')
        mgr.complete_load(set(keys[:2]), ctx)
        self.end_pass()
        self.assertEqual(mgr.entries, 2, 'complete_load stamped a use, so they live another ttl')
        self.clock.advance(101)
        self.end_pass()
        self.assertEqual(mgr.entries, 0)

    def test_two_loads_pin_twice_and_both_must_complete(self):
        mgr = self.build(ttl_s=100)
        keys = self.put(2)
        mgr.prepare_load(keys, req('a'))
        mgr.prepare_load(keys, req('b'))
        mgr.complete_load(set(keys), req('a'))
        self.clock.advance(1000)
        self.end_pass()
        self.assertEqual(mgr.entries, 2, 'one load is still reading them')
        mgr.complete_load(set(keys), req('b'))
        self.clock.advance(1000)
        self.end_pass()
        self.assertEqual(mgr.entries, 0)

    def test_a_key_that_vanished_before_prepare_load_poisons_the_spec_instead_of_crashing_the_engine(self):
        mgr = self.build()
        keys = self.put(3)
        mgr.reset_cache()
        spec = mgr.prepare_load(keys, req('a'))
        self.assertIsNotNone(spec.poison)
        self.assertEqual(len(spec.slot_ids), 3)
        self.assertEqual(len(spec.generations), 3)
        self.assertEqual(self.runtime.counters['mgr_prepare_load_missing'], 1)

    def test_complete_load_of_a_reset_entry_is_ignored(self):
        mgr = self.build()
        keys = self.put(2)
        mgr.prepare_load(keys, req('a'))
        mgr.reset_cache()
        mgr.complete_load(set(keys), req('a'))
        self.assertEqual(mgr.entries, 0)


class BadKeys(ManagerCase):
    """I13: BAD only on the worker's proof; a BAD entry is removed when its last load completes."""

    def test_a_bad_entry_is_removed_at_ref_zero_and_its_slot_goes_through_trim(self):
        mgr = self.build()
        keys = self.put(3)
        spec = mgr.prepare_load(keys, req('a'))
        self.runtime.mark_bad(spec.slot_ids[1], spec.generations[1])
        self.assertEqual(mgr.entries, 3, 'still pinned: kept until the load completes')
        mgr.complete_load(set(keys), req('a'))
        self.assertEqual(mgr.entries, 2)
        self.assertNotIn(keys[1], mgr.index)
        self.assertEqual(self.runtime.allocator.state_name(spec.slot_ids[1]), 'trim_wait')
        self.assertEqual(mgr.evictions['bad'], 1)
        self.assertEqual(self.runtime.bad, set(), 'the mark is forgotten with the entry')

    def test_a_bad_mark_on_an_unpinned_entry_removes_it_at_once(self):
        mgr = self.build()
        keys = self.put(2)
        slot = mgr.index[keys[0]]
        self.runtime.mark_bad(slot, self.runtime.allocator.generation(slot))
        self.assertEqual(mgr.entries, 1)

    def test_a_stale_bad_mark_for_another_generation_changes_nothing(self):
        mgr = self.build()
        keys = self.put(1)
        slot = mgr.index[keys[0]]
        self.runtime.mark_bad(slot, self.runtime.allocator.generation(slot) + 5)
        self.assertEqual(mgr.entries, 1)
        self.assertIs(mgr.lookup(keys[0], req('a')), LookupResult.HIT)


class FrozenNowAndRaces(ManagerCase):
    """I7: only the engine thread mutates; 'now' is frozen per pass; eviction happens at the end of a pass."""

    def test_now_is_frozen_between_the_first_call_of_a_pass_and_the_end_of_it(self):
        mgr = self.build(ttl_s=100)
        keys = self.put(1)
        self.end_pass()
        self.clock.advance(99.5)
        self.assertIs(mgr.lookup(keys[0], req('a')), LookupResult.HIT)
        self.clock.advance(5)                              # the clock runs past the TTL in the middle of the pass ...
        self.assertIs(mgr.lookup(keys[0], req('b')), LookupResult.HIT, '... and a second lookup of the same pass agrees with the first')
        mgr.prepare_load(keys, req('a'))                   # the pass goes on to prepare the load it promised
        self.end_pass()
        self.assertEqual(mgr.entries, 1, 'it is pinned: the expiry at the end of the pass cannot take it')

    def test_an_entry_that_answered_hit_is_still_there_for_prepare_load_after_the_ttl_ran_out(self):
        mgr = self.build(ttl_s=100)
        keys = self.put(2)
        self.end_pass()
        self.clock.advance(99.9)
        self.assertIs(mgr.lookup(keys[0], req('a')), LookupResult.HIT)
        self.clock.advance(60)
        spec = mgr.prepare_load([keys[0]], req('a'))
        self.assertIsNone(spec.poison)
        self.end_pass()
        self.assertEqual(mgr.entries, 2, 'the pass saw time as of its first lookup: nothing was due, so nothing expired')
        self.clock.advance(1)
        self.end_pass()
        self.assertEqual(mgr.entries, 1, 'the next pass sees the new time: the unpinned one expired, the pinned one did not')

    def test_expiry_and_lru_eviction_happen_only_in_on_schedule_end(self):
        mgr = self.build(ttl_s=10, num_slots=10, free_floor_pct=50)
        keys = self.put(5)
        self.end_pass()
        self.clock.advance(1000)
        mgr.lookup(keys[0], req('a'))
        mgr.touch([keys[1]], req('a'))
        self.store([key_for(99)], req('c'))
        self.assertEqual(mgr.entries, 5, 'no lookup, touch or prepare evicts anything')
        self.end_pass()
        self.assertEqual(mgr.entries, 1, 'the one touched keeps its stamp; the others expired')
        self.assertIn(keys[1], mgr.index)

    def test_drive_completions_arrive_on_another_thread_but_are_applied_only_by_the_engine_thread(self):
        self.tier(num_slots=32, sync_io=False)
        runtime = self.open_runtime()
        mgr = self.manager(runtime)
        keys = []
        for i in range(8):
            keys.append(key_for(i))
        ctx = req('a')
        out = mgr.prepare_store(keys, ctx)
        self.job = 1
        runtime.store_results[1] = StoreResult(1, True, tuple(out.store_spec.slot_ids), tuple(out.store_spec.generations), digest_rows(out.store_spec.slot_ids))
        mgr.complete_store(set(keys), ctx)
        self.assertIs(mgr.lookup(keys[0], ctx), LookupResult.HIT)
        mgr.reset_cache()                                   # releases 8 slots: their TRIMs run on the I/O thread
        import time
        time.sleep(0.1)                                     # ... and finish there ...
        before = runtime.allocator.n_free
        self.assertEqual(runtime.allocator.count(S_TRIM_WAIT), 8, 'nothing is submitted until the engine thread runs the pump')
        runtime.service()
        deadline = time.time() + 5
        while runtime.allocator.n_free == before and time.time() < deadline:
            time.sleep(0.01)
            runtime.service()
        self.assertEqual(runtime.allocator.n_free, 32, '... and the engine thread applied the completion')

    def test_the_end_of_a_pass_pumps_the_runtime(self):
        mgr = self.build(num_slots=16)
        keys = self.put(4)
        mgr.reset_cache()
        self.assertEqual(self.runtime.allocator.count(S_TRIM_WAIT), 4)
        self.end_pass()
        self.assertEqual(self.runtime.allocator.count(S_TRIM_WAIT), 0)
        self.end_pass()
        self.assertEqual(self.runtime.allocator.n_free, 16)
        self.assertIsNotNone(self.runtime.last_schedule_wall, 'the heartbeat')


class TtlAndLru(ManagerCase):
    def test_ttl_counts_from_the_last_use_not_from_the_store(self):
        mgr = self.build(ttl_s=1000)
        keys = self.put(2)
        self.end_pass()
        self.clock.advance(900)
        mgr.touch([keys[0]], req('a'))
        self.end_pass()
        self.clock.advance(900)
        self.end_pass()
        self.assertEqual(list(mgr.index), [keys[0]], 'the untouched one expired at 1800 s, the touched one lives until 1900')
        self.clock.advance(101)
        self.end_pass()
        self.assertEqual(mgr.entries, 0)

    def test_an_expired_entrys_slot_is_trimmed_before_it_can_be_reused(self):
        mgr = self.build(ttl_s=100, num_slots=4)
        self.put(4)
        self.assertEqual(self.store([key_for(99)], req('x')).keys_to_store, [], 'full')
        self.end_pass()
        self.clock.advance(101)
        self.end_pass()                                        # evicts and issues the TRIMs; their completions are applied by the NEXT pass
        self.assertEqual(mgr.entries, 0)
        self.assertEqual(self.store([key_for(99)], req('x')).keys_to_store, [], 'evicted but not yet trimmed: still not allocatable (I5)')
        self.settle()
        self.assertEqual(self.runtime.allocator.n_free, 4)
        self.assertEqual(len(self.store([key_for(99)], req('y')).keys_to_store), 1)

    def test_load_completion_counts_as_a_use(self):
        mgr = self.build(ttl_s=100)
        keys = self.put(1)
        self.end_pass()
        self.clock.advance(60)
        mgr.prepare_load(keys, req('a'))
        self.clock.advance(30)
        mgr.complete_load(set(keys), req('a'))
        self.clock.advance(60)
        self.end_pass()
        self.assertEqual(mgr.entries, 1)

    def test_lru_evicts_the_coldest_first(self):
        mgr = self.build(num_slots=10, free_floor_pct=30)       # capacity 7
        keys = self.put(7)
        self.end_pass()
        mgr.touch(keys[:3], req('a'))                          # the first three are the hot ones
        out = self.store([key_for(100)], req('b'))
        self.assertEqual(out.keys_to_store, [], 'a full tier drops a store (the free floor is kept)')
        self.assertEqual(self.runtime.allocator.n_free, 3, 'the floor: 3 of 10 stay free')

    def test_the_standing_reserve_below_the_floor_is_kept_by_lru_eviction(self):
        reserve = 3 * OBJ
        mgr = self.build(num_slots=20, free_floor_pct=10, reserve_bytes=reserve)
        self.assertEqual(mgr.capacity_slots, 18)
        self.assertEqual(mgr.reserve_slots, 3)
        keys = self.put(18)
        self.end_pass()
        self.assertEqual(mgr.entries, 15, '18 - 3: the oldest three were evicted to keep room for a new session')
        self.assertNotIn(keys[0], mgr.index)
        self.assertNotIn(keys[2], mgr.index)
        self.assertIn(keys[3], mgr.index)
        self.assertEqual(mgr.evictions['lru'], 3)

    def test_lru_order_follows_touch(self):
        mgr = self.build(num_slots=10, free_floor_pct=0, reserve_bytes=3 * OBJ)
        keys = self.put(10)
        mgr.touch(keys[:2], req('a'))
        self.end_pass()
        self.assertEqual(mgr.entries, 7)
        self.assertIn(keys[0], mgr.index)
        self.assertIn(keys[1], mgr.index)
        self.assertNotIn(keys[2], mgr.index)
        self.assertNotIn(keys[4], mgr.index)

    def test_capacity_is_all_slots_minus_the_free_floor(self):
        mgr = self.build(num_slots=100, free_floor_pct=5)
        self.assertEqual(mgr.capacity_slots, 95)
        self.assertEqual(mgr.floor_slots, 5)

    def test_the_free_floor_is_rounded_up(self):
        mgr = self.build(num_slots=64, free_floor_pct=5)
        self.assertEqual(mgr.floor_slots, 4)             # 3.2 -> 4
        self.assertEqual(mgr.capacity_slots, 60)

    def test_a_drive_filled_to_capacity_still_serves_hits_and_drops_stores(self):
        mgr = self.build(num_slots=20, free_floor_pct=25)
        keys = self.put(15)
        self.assertEqual(self.store([key_for(500)], req('a')).keys_to_store, [])
        self.assertIs(mgr.lookup(keys[0], req('a')), LookupResult.HIT)

    def test_eviction_per_pass_is_bounded(self):
        from ttkvp2p import manager as manager_module
        saved = manager_module.EVICT_PER_PASS
        manager_module.EVICT_PER_PASS = 5
        self.addCleanup(setattr, manager_module, 'EVICT_PER_PASS', saved)
        mgr = self.build(ttl_s=100, num_slots=40)
        self.put(20)
        self.end_pass()
        self.clock.advance(1000)
        self.end_pass()
        self.assertEqual(mgr.entries, 15)
        self.end_pass()
        self.assertEqual(mgr.entries, 10)


class Admission(ManagerCase):
    """I12: all or nothing per request boundary, decided once, cached in the ReqTag, slots and bytes reserved for the request."""

    def build_gemma_like(self, num_slots=200, **cfg):
        self.tier(num_slots=num_slots, **cfg)
        self.runtime = self.open_runtime()
        self.mgr = self.manager(self.runtime, groups=GEMMA_GROUPS)
        self.job = 0
        return self.mgr

    def keys_for(self, tokens, base=0):
        """The keys a request of ``tokens`` tokens stores at its boundary: every full block of group 0, the last 8 of groups 1-5."""
        full = tokens // 128
        keys = [key_for(base * 100000 + i, 0) for i in range(full)]
        for group in range(1, 6):
            keys += [key_for(base * 100000 + i, group) for i in range(max(0, full - 8), full)]
        return keys

    def test_the_whole_boundary_of_a_262k_request_is_2088_objects(self):
        self.assertEqual(estimate_objects(262144, GEMMA_GROUPS), 2088)
        self.assertEqual(estimate_objects(262144, GEMMA_GROUPS), 2048 + 5 * 8)
        self.assertEqual(estimate_objects(0, GEMMA_GROUPS), 0)
        self.assertEqual(estimate_objects(5 * 128, GEMMA_GROUPS), 5 + 5 * 5, 'a short prompt has fewer than 8 blocks per sliding group')
        self.assertEqual(estimate_objects(262144, (None, 8, 8, 8, 8, 8), 128), 2088)
        self.assertEqual(estimate_objects(262144, (None,) + (None,) * 5), 6 * 2048, 'the dense fallback (retention interval None) would store 12,288')

    def test_a_request_that_fits_is_admitted_and_its_slots_are_reserved_up_front(self):
        mgr = self.build_gemma_like(num_slots=200)
        ctx = req('a', prompt_tokens=6000)                 # 46 full blocks: 46 + 5 * 8 = 86 objects
        first = self.keys_for(6000)
        chunk = first[:32]                                  # the first prefill chunk offers only some of them
        out = self.store(chunk, ctx)
        self.assertEqual(len(out.keys_to_store), 32)
        self.assertTrue(ctx.get_state(tier_kit.ReqTag).admitted)
        self.assertEqual(mgr.snapshot()['reserved_slots'], 86 - 32)
        self.assertEqual(mgr.admitted_objects, 86)

    def test_all_or_nothing_a_request_that_does_not_fit_stores_nothing_at_all(self):
        mgr = self.build_gemma_like(num_slots=100)
        a = req('a', prompt_tokens=6000)
        out = self.store(self.keys_for(6000, 1)[:32], a)
        self.assertEqual(len(out.keys_to_store), 32)
        b = req('b', prompt_tokens=6000)
        denied = self.store(self.keys_for(6000, 2)[:32], b)
        self.assertEqual(denied.keys_to_store, [], '14 slots are free but request b needs 86: not even its first chunk is written')
        self.assertFalse(b.get_state(tier_kit.ReqTag).admitted)
        self.assertEqual(mgr.dropped['no_slot'], 1)
        again = self.store(self.keys_for(6000, 2)[32:46], b)
        self.assertEqual(again.keys_to_store, [], 'the verdict is cached: later chunks of a denied request are dropped too')
        self.assertEqual(mgr.dropped['denied'], 1)
        self.assertEqual(mgr.entries, 32)

    def test_the_reservation_protects_a_request_from_later_arrivals(self):
        mgr = self.build_gemma_like(num_slots=100)
        a = req('a', prompt_tokens=6000)                    # reserves 86
        keys_a = self.keys_for(6000, 1)
        self.store(keys_a[:32], a)
        small = req('s', prompt_tokens=4096)                # 32 + 40 = 72 objects: does not fit in the 14 that are left unreserved
        self.assertEqual(self.store(self.keys_for(4096, 3)[:32], small).keys_to_store, [])
        rest = self.store(keys_a[32:], a)                   # request a still gets the rest of what it was promised
        self.assertEqual(len(rest.keys_to_store), 86 - 32)
        self.assertEqual(mgr.snapshot()['reserved_slots'], 0)

    def test_the_unused_part_of_a_reservation_is_released_when_the_request_finishes(self):
        mgr = self.build_gemma_like(num_slots=100)
        a = req('a', prompt_tokens=6000)
        self.store(self.keys_for(6000, 1)[:32], a)
        self.assertEqual(mgr.snapshot()['reserved_slots'], 54)
        mgr.on_request_finished(a)                          # aborted early
        self.assertEqual(mgr.snapshot()['reserved_slots'], 0)
        b = req('b', prompt_tokens=2048)                    # 16 + 40 = 56 objects
        self.assertEqual(len(self.store(self.keys_for(2048, 4)[:16], b).keys_to_store), 16)

    def test_min_store_tokens_is_judged_on_the_prompt(self):
        mgr = self.build_gemma_like(num_slots=200, min_store_tokens=4096)
        short = req('s', prompt_tokens=4000)
        self.assertEqual(self.store(self.keys_for(4000)[:20], short).keys_to_store, [])
        self.assertEqual(mgr.dropped['short_prompt'], 1)
        enough = req('e', prompt_tokens=4096)
        self.assertEqual(len(self.store(self.keys_for(4096, 2)[:32], enough).keys_to_store), 32)

    def test_without_the_prompt_size_the_call_itself_is_judged(self):
        mgr = self.build(num_slots=200, min_store_tokens=4096)
        short = req('s')
        self.assertEqual(self.store([key_for(i) for i in range(20)], short).keys_to_store, [])
        self.assertEqual(mgr.dropped['short_prompt'], 1)
        long = req('l')
        self.assertEqual(len(self.store([key_for(100 + i) for i in range(32)], long).keys_to_store), 32)

    def test_the_verdict_is_cached_in_the_req_tag_and_a_tagless_request_is_cached_by_id(self):
        mgr = self.build_gemma_like(num_slots=200)
        tagged = req('a', prompt_tokens=6000)
        self.store(self.keys_for(6000)[:10], tagged)
        self.assertIs(tagged.get_state(tier_kit.ReqTag).admitted, True)
        tagless = req('t', tag=False)
        self.store([key_for(70000 + i) for i in range(5)], tagless)
        self.assertEqual(mgr._verdicts['t'], True)
        mgr.on_request_finished(tagless)
        self.assertNotIn('t', mgr._verdicts)

    def test_the_capacity_floor_counts_against_a_whole_boundary(self):
        mgr = self.build_gemma_like(num_slots=100, free_floor_pct=20)      # capacity 80
        a = req('a', prompt_tokens=6000)                    # 86 > 80
        self.assertEqual(self.store(self.keys_for(6000)[:32], a).keys_to_store, [])
        self.assertEqual(mgr.dropped['no_slot'] + mgr.dropped['capacity'], 1)

    def test_a_denied_boundary_charges_nothing(self):
        mgr = self.build_gemma_like(num_slots=50)
        a = req('a', prompt_tokens=6000)
        self.store(self.keys_for(6000)[:32], a)
        self.assertEqual(mgr.snapshot()['bytes_written_today'], 0)
        self.assertEqual(mgr.admitted_objects, 0)


class DailyCap(ManagerCase):
    def test_the_cap_is_charged_at_admission_for_the_whole_boundary(self):
        mgr = self.build(num_slots=300, daily_write_cap_bytes=100 * OBJ)
        mgr.groups = GEMMA_GROUPS
        a = req('a', prompt_tokens=6000)                    # 86 objects
        keys = AdmissionKeys.keys_for(6000, 1)
        self.assertEqual(len(self.store(keys[:32], a).keys_to_store), 32)
        self.assertEqual(mgr.snapshot()['bytes_written_today'], 86 * OBJ)
        b = req('b', prompt_tokens=6000)
        self.assertEqual(self.store(AdmissionKeys.keys_for(6000, 2)[:32], b).keys_to_store, [], '86 + 86 > 100')
        self.assertEqual(mgr.dropped['cap'], 1)

    def test_the_window_is_a_rolling_day_not_a_calendar_day(self):
        mgr = self.build(num_slots=300, daily_write_cap_bytes=100 * OBJ)
        mgr.groups = GEMMA_GROUPS
        self.store(AdmissionKeys.keys_for(6000, 1)[:32], req('a', prompt_tokens=6000))
        self.clock.advance(23 * 3600)
        self.assertEqual(self.store(AdmissionKeys.keys_for(6000, 2)[:32], req('b', prompt_tokens=6000)).keys_to_store, [], 'still inside the day')
        self.clock.advance(2 * 3600 + 10)
        out = self.store(AdmissionKeys.keys_for(6000, 3)[:32], req('c', prompt_tokens=6000))
        self.assertEqual(len(out.keys_to_store), 32, 'the first day aged out of the window')
        self.assertEqual(mgr.snapshot()['bytes_written_today'], 86 * OBJ)

    def test_a_requests_unused_reservation_is_refunded_to_the_cap(self):
        mgr = self.build(num_slots=300, daily_write_cap_bytes=100 * OBJ)
        mgr.groups = GEMMA_GROUPS
        a = req('a', prompt_tokens=6000)
        self.store(AdmissionKeys.keys_for(6000, 1)[:32], a)
        mgr.on_request_finished(a)
        self.assertEqual(mgr.snapshot()['bytes_written_today'], 32 * OBJ, 'only what was admitted counts')
        b = req('b', prompt_tokens=2048)
        self.assertEqual(len(self.store(AdmissionKeys.keys_for(2048, 2)[:16], b).keys_to_store), 16)

    def test_a_zero_cap_stores_nothing_and_a_huge_cap_never_bites(self):
        mgr = self.build(num_slots=60, daily_write_cap_bytes=0)
        self.assertEqual(self.store([key_for(i) for i in range(5)], req('a')).keys_to_store, [])
        self.assertEqual(mgr.dropped['cap'], 1)

    def test_without_a_prompt_size_each_call_is_charged_as_it_is_admitted(self):
        mgr = self.build(num_slots=300, daily_write_cap_bytes=40 * OBJ)
        self.assertEqual(len(self.store([key_for(i) for i in range(30)], req('a')).keys_to_store), 30)
        self.assertEqual(self.store([key_for(100 + i) for i in range(30)], req('b')).keys_to_store, [])

    def test_the_cap_does_not_stop_hits(self):
        mgr = self.build(num_slots=300, daily_write_cap_bytes=40 * OBJ)
        keys = self.put(30)
        self.assertIs(mgr.lookup(keys[0], req('x')), LookupResult.HIT)
        spec = mgr.prepare_load(keys[:5], req('x'))
        self.assertEqual(len(spec.slot_ids), 5)


class AdmissionKeys:
    @staticmethod
    def keys_for(tokens, base=0):
        full = tokens // 128
        keys = [key_for(base * 100000 + i, 0) for i in range(full)]
        for group in range(1, 6):
            keys += [key_for(base * 100000 + i, group) for i in range(max(0, full - 8), full)]
        return keys


class ResetCache(ManagerCase):
    """reset_cache: an in-RAM invalidate plus an epoch bump; the slots go back through TRIM under I5; never a wipe."""

    def test_reset_invalidates_bumps_the_epoch_and_sends_the_slots_through_trim(self):
        mgr = self.build(num_slots=32)
        keys = self.put(10)
        self.assertEqual(mgr.epoch, 0)
        mgr.reset_cache()
        self.assertEqual(mgr.epoch, 1)
        self.assertEqual(mgr.entries, 0)
        for key in keys:
            self.assertIs(mgr.lookup(key, req('a')), LookupResult.MISS)
        self.assertEqual(self.runtime.allocator.count(S_TRIM_WAIT), 10)
        self.assertEqual(self.runtime.allocator.n_free, 22)
        self.settle()
        self.assertEqual(self.runtime.allocator.n_free, 32)
        self.assertEqual(mgr.evictions['reset'], 10)

    def test_reset_is_not_a_wipe_it_does_not_discard_free_or_untouched_state(self):
        mgr = self.build(num_slots=16)
        self.put(4)
        runs_before = self.runtime.counters['trim_slots']
        mgr.reset_cache()
        self.settle()
        self.assertEqual(self.runtime.counters['trim_slots'] - runs_before, 4, 'only the released slots were trimmed, not the whole region')

    def test_a_slot_under_a_bio_is_not_reused_after_a_reset_until_the_bio_returns(self):
        mgr = self.build(num_slots=8)
        keys = self.put(8)
        spec = mgr.prepare_load(keys[:2], req('a'))
        self.assertTrue(self.runtime.allocator.io_begin(spec.slot_ids, spec.generations))     # the worker started the read
        mgr.reset_cache()
        self.settle()
        self.assertEqual(self.runtime.allocator.state_name(spec.slot_ids[0]), 'draining')
        out = self.store([key_for(100 + i) for i in range(8)], req('b'))
        self.assertEqual(out.keys_to_store, [], 'only 6 slots are allocatable: all-or-nothing drops 8')
        out = self.store([key_for(100 + i) for i in range(6)], req('b'))
        self.assertEqual(len(out.keys_to_store), 6)
        for slot in out.store_spec.slot_ids:
            self.assertNotIn(slot, spec.slot_ids, 'I5: a slot with a bio in flight was handed out')
        self.runtime.allocator.io_end(spec.slot_ids)
        self.settle()
        self.assertEqual(self.runtime.allocator.n_free, 2)

    def test_the_old_jobs_late_io_begin_is_refused_after_the_reset(self):
        mgr = self.build(num_slots=8)
        keys = self.put(4)
        spec = mgr.prepare_load(keys, req('a'))
        mgr.reset_cache()
        self.assertFalse(self.runtime.allocator.io_begin(spec.slot_ids, spec.generations), 'a read of a released slot never starts')
        self.settle()
        self.store([key_for(100 + i) for i in range(4)], req('b'))
        self.assertFalse(self.runtime.allocator.io_begin(spec.slot_ids, spec.generations), 'nor after the slots were reallocated')

    def test_reset_drops_reservations_and_verdicts_and_lets_a_new_request_in(self):
        mgr = self.build(num_slots=100)
        mgr.groups = GEMMA_GROUPS
        a = req('a', prompt_tokens=6000)
        self.store(AdmissionKeys.keys_for(6000, 1)[:32], a)
        mgr.reset_cache()
        self.assertEqual(mgr.snapshot()['reserved_slots'], 0)
        self.settle()
        b = req('b', prompt_tokens=6000)
        self.assertEqual(len(self.store(AdmissionKeys.keys_for(6000, 2)[:32], b).keys_to_store), 32)


class WipeVerb(ManagerCase):
    def test_wipe_through_the_control_directory_invalidates_the_index(self):
        mgr = self.build(num_slots=32)
        self.put(6)
        self.tree.command(1, 'wipe')
        self.clock.advance(2)
        self.end_pass()
        self.assertEqual(mgr.entries, 0)
        self.assertEqual(mgr.evictions['wipe'], 6)
        self.settle()
        self.assertEqual(self.runtime.allocator.n_free, 32)


class EnginePassFuzz(ManagerCase):
    """A randomized engine: passes of lookups, loads, stores, completions, time jumps and resets against a small drive whose TRIMs complete on
    another thread. The invariants of I5 and I7 must hold at every step: a HIT is always loadable by prepare_load in its own pass, no slot with a
    bio in flight is ever handed out, no slot is lost, and the tables agree with the allocator."""

    def run_engine(self, seed, passes, *, async_io):
        import random
        import time
        rng = random.Random(seed)
        self.tier(num_slots=48, sync_io=not async_io, ttl_s=500, free_floor_pct=10, reserve_bytes=4 * OBJ)
        runtime = self.open_runtime()
        mgr = self.mgr = self.manager(runtime)
        self.runtime = runtime
        self.job = 0
        allocator = runtime.allocator
        universe = [key_for(i, group=i % 2) for i in range(80)]
        inflight_stores, loads = [], []           # (out, ctx) of stores not yet completed; (keys, ctx, slots, gens) of loads not yet completed
        bios = []                                 # (slots, gens) the "worker" started
        poisoned = 0
        for step in range(passes):
            ctx = req('r%d' % step)
            # lookups of the pass, then the loads the scheduler promised for the HITs
            hits = [k for k in rng.sample(universe, 12) if mgr.lookup(k, ctx) is LookupResult.HIT]
            if hits:
                spec = mgr.prepare_load(hits, ctx)
                if spec.poison:
                    poisoned += 1
                else:
                    loads.append((hits, ctx, spec))
                    if rng.random() < 0.7 and allocator.io_begin(spec.slot_ids, spec.generations):
                        bios.append((spec.slot_ids, spec.generations))
            # the end of the pass: eviction, TRIM pump
            if rng.random() < 0.15:
                self.clock.advance(rng.choice([1, 50, 300, 700]))
            self.end_pass()
            # stores of the pass
            wanted = rng.sample(universe, rng.randint(1, 9))
            out = mgr.prepare_store(wanted, ctx)
            if out.keys_to_store:
                for slot in out.store_spec.slot_ids:
                    self.assertEqual(int(allocator.inflight[slot]), 0, 'a slot with a bio in flight was handed out (I5)')
                inflight_stores.append((out, ctx))
            # completions that arrive later
            if inflight_stores and rng.random() < 0.6:
                out, octx = inflight_stores.pop(rng.randrange(len(inflight_stores)))
                self.finish_store(out, octx, ok=rng.random() < 0.9)
            if loads and rng.random() < 0.5:
                keys, lctx, spec = loads.pop(rng.randrange(len(loads)))
                mgr.complete_load(set(keys), lctx)
            if bios and rng.random() < 0.5:
                slots, gens = bios.pop(rng.randrange(len(bios)))
                allocator.io_end(slots)
            if rng.random() < 0.03:
                mgr.reset_cache()
                loads.clear()
                inflight_stores.clear()
            if async_io:
                time.sleep(0.0002)
            # invariants
            counts = [allocator.count(i) for i in range(6)]
            self.assertEqual(sum(counts), allocator.num_slots, 'a slot was lost')
            self.assertEqual(counts[S_USED], mgr.entries, 'the index and the allocator disagree about which slots are in use')
            for key, slot in mgr.index.items():
                self.assertEqual(mgr._slot_key[slot], key)
                self.assertEqual(allocator.state_name(slot), 'used')
        self.assertEqual(poisoned, 0, 'a HIT entry vanished before its prepare_load (I7)')
        for slots, gens in bios:
            allocator.io_end(slots)
        return mgr

    def test_the_invariants_hold_with_trims_completing_synchronously(self):
        for seed in (1, 2, 3):
            self.setUp()
            self.run_engine(seed, 400, async_io=False)

    def test_the_invariants_hold_with_trims_completing_on_another_thread(self):
        for seed in (11, 12):
            self.setUp()
            self.run_engine(seed, 300, async_io=True)


class Accounting(ManagerCase):
    def test_the_snapshot_feeds_the_state_file(self):
        mgr = self.build(num_slots=32)
        keys = self.put(5)
        mgr.lookup(keys[0], req('a'))
        mgr.lookup(key_for(999), req('a'))
        self.runtime.write_state(self.clock.now, force=True)
        state = self.tree.read_state()
        counters = state['counters']
        self.assertEqual((counters['hits'], counters['misses'], counters['lookups']), (1, 1, 2))
        self.assertEqual(state['capacity']['entries'], 5)
        self.assertEqual(state['capacity']['capacity_slots'], 32)
        self.assertEqual(state['bytes_written_today'], 5 * OBJ)

    def test_the_slot_table_is_modest(self):
        mgr = self.build(num_slots=64)
        per_slot = (mgr._state.itemsize + mgr._tenant.itemsize + mgr._created.itemsize + mgr._last_used.itemsize + mgr._digest.itemsize * 4
                    + self.runtime.allocator.gen.itemsize + self.runtime.allocator.inflight.itemsize + self.runtime.allocator.state.itemsize)
        self.assertLess(per_slot, 100, 'bytes per slot of the structure-of-arrays table (the dict and the LRU add about 300 B per entry)')


if __name__ == '__main__':
    unittest.main()
