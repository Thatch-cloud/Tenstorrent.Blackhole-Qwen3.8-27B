"""The manager and the allocator at the slot count of a 1 TB drive (B3 of the WP3a review).

A 1 TB drive holds 359,030 Gemma objects; with the 5% free floor the index can hold 340,886. The CPU tests of the earlier rounds ran at 64 to
400 slots and could not see the engine thread stall: every ``release_many`` recounted ALL slots (0.8 ms), the manager released one slot per
call, so one 262k session's 2,088 releases took 1.7 s and one ``on_schedule_end`` of an eviction burst (4,096 victims) about 3.4 s.

These tests build a 340,732-object index on a 359,030-slot drive through the real runtime (a sparse file stands in for the namespace; the
kit's tiny layout keeps it at 1.8 GB of address space and nothing is written to it) and bound the latency of every manager call against it. The
bounds are about 20 to 100 times the measured numbers, so a loaded machine does not fail them, and about 10 times below what the pre-fix code
needed, so a per-call scan over the slots or the index does.

``bench_scale.py`` of the WP3a work (private) prints p50, p99 and max per call for the same workload.
"""

import gc
import os
import time
import unittest

import numpy as np

from tier_kit import OBJ, TierCase, key_for, req

from vllm.v1.kv_offload.base import ScheduleEndContext

SLOTS = 359_030                       # 1 TB of Gemma objects (2,785,280 B each)
ENTRIES = 340_732                     # an index nearly at the capacity of that drive
GEMMA = (None, 8, 8, 8, 8, 8)
PASS = ScheduleEndContext(new_req_ids=(), preempted_req_ids=())


def show(what, seconds):
    """TTKVP2P_SHOW_TIMES=1 prints what the bounds below were measured against."""
    if os.environ.get('TTKVP2P_SHOW_TIMES'):
        print('  [%s] %.2f ms' % (what, seconds * 1e3))


class ProductionScale(TierCase):
    def build(self, entries):
        self.tier(num_slots=SLOTS, ttl_s=86400, free_floor_pct=5, reserve_bytes=3084 * OBJ, min_store_tokens=4096, daily_write_cap_bytes=10 ** 15)
        self.runtime = self.open_runtime()
        self.mgr = self.manager(self.runtime, groups=GEMMA)
        self.chunks = []
        filled, base = 0, 0
        while filled < entries:
            n = min(10_000, entries - filled)
            ctx = req('fill%d' % base, prompt_tokens=(n + 40) * 128)
            keys = [key_for(base * 100_000 + i) for i in range(n)]
            out = self.mgr.prepare_store(keys, ctx)
            self.assertEqual(len(out.keys_to_store), n)
            self.mgr.complete_store(out.keys_to_store, ctx, success=True, digests=np.zeros((n, 4), dtype=np.uint64))
            self.mgr.on_request_finished(ctx)
            self.chunks.append(keys)
            filled += n
            base += 1
        self.assertEqual((self.mgr.entries, self.mgr.written()), (entries, entries))
        return self.mgr

    def timed(self, fn, *args):
        started = time.perf_counter()
        result = fn(*args)
        return time.perf_counter() - started, result

    def end_pass(self):
        return self.timed(self.mgr.on_schedule_end, PASS)[0]

    def test_an_idle_night_of_ttl_expiry_and_an_lru_burst_never_stall_the_engine_thread(self):
        mgr = self.build(ENTRIES)
        allocator = self.runtime.allocator
        slowest = self.end_pass()                                        # 340,732 > the 337,802 target: the first pass evicts 2,930 by LRU
        self.assertEqual(mgr.evictions['lru'], ENTRIES - (mgr.capacity_slots - mgr.reserve_slots))
        self.clock.advance(86400 + 10)                                   # the engine idled for a day: every entry is past its TTL
        passes = 0
        while mgr.entries and passes < 200:
            slowest = max(slowest, self.end_pass())
            self.runtime.settle(30)
            passes += 1
        self.assertEqual(mgr.entries, 0)
        self.assertLessEqual(passes, 84, 'at most 4,096 evictions per pass')
        self.assertLess(slowest, 0.4, 'the slowest on_schedule_end took %.3f s (3.4 s before the fix)' % slowest)
        self.assertEqual(allocator.recount()[:6], [allocator.count(i) for i in range(6)])
        self.assertEqual(allocator.n_free, allocator.num_slots, 'every slot went through TRIM and came back')
        mgr.check_invariants()

    def test_session_sized_calls_cost_milliseconds_not_the_size_of_the_drive(self):
        mgr = self.build(60_000)                                         # (a fuller index changes nothing below; this keeps the test short)
        # a 262k session fails its store: 2,088 entries leave in one batched release
        ctx = req('big', prompt_tokens=262144)
        keys = [key_for(9_000_000 + i, 0) for i in range(2048)] + [key_for(9_000_000 + i, g) for g in range(1, 6) for i in range(2040, 2048)]
        took, out = self.timed(mgr.prepare_store, keys, ctx)
        self.assertEqual(len(out.keys_to_store), 2088)
        self.assertLess(took, 0.25, 'prepare_store of 2,088 keys took %.3f s' % took)
        took, _ = self.timed(mgr.complete_store, out.keys_to_store, ctx, False)
        self.assertLess(took, 0.25, 'complete_store(failure) of 2,088 keys took %.3f s (1.7 s of single releases before the fix)' % took)
        self.assertEqual(mgr.entries, 60_000)
        # a session of 2,088 stored objects is looked up, touched, loaded and released again
        hot = self.chunks[0][:2088]
        took, _ = self.timed(lambda: [mgr.lookup(key, ctx) for key in hot])
        self.assertLess(took, 0.25, '2,088 lookups took %.3f s' % took)
        took, _ = self.timed(mgr.touch, hot, ctx)
        self.assertLess(took, 0.25, 'touch of 2,088 keys took %.3f s' % took)
        took, spec = self.timed(mgr.prepare_load, hot, ctx)
        self.assertLess(took, 0.25, 'prepare_load of 2,088 keys took %.3f s' % took)
        self.assertIsNone(spec.poison)
        took, _ = self.timed(mgr.complete_load, hot, ctx)
        self.assertLess(took, 0.25, 'complete_load of 2,088 keys took %.3f s' % took)
        # one BAD entry among them
        spec = mgr.prepare_load(hot[:100], ctx)
        self.runtime.mark_bad(spec.slot_ids[7], spec.generations[7])
        took, _ = self.timed(mgr.complete_load, hot[:100], ctx)
        self.assertLess(took, 0.1)
        self.assertEqual(mgr.entry_state(hot[7]), 'none')
        mgr.check_invariants()

    def test_the_p99_latency_of_every_call_stays_far_below_a_decode_step(self):
        mgr = self.build(200_000)
        samples = {}

        def timed(name, fn, *args):
            started = time.perf_counter()
            result = fn(*args)
            samples.setdefault(name, []).append(time.perf_counter() - started)
            return result

        for session in range(6):                                          # chunked prefill of six 262k sessions, 64 keys per step
            ctx = req('new%d' % session, prompt_tokens=262144)
            full = [key_for(9_000_000 + session * 10_000 + i, 0) for i in range(2048)]
            tails = [key_for(9_000_000 + session * 10_000 + i, g) for g in range(1, 6) for i in range(2040, 2048)]
            for chunk in [full[i:i + 64] for i in range(0, 2048, 64)] + [tails]:
                out = timed('prepare_store', mgr.prepare_store, chunk, ctx)
                self.assertEqual(len(out.keys_to_store), len(chunk))
                timed('complete_store', mgr.complete_store, out.keys_to_store, ctx, True, np.zeros((len(chunk), 4), dtype=np.uint64))
                timed('on_schedule_end', mgr.on_schedule_end, PASS)
            mgr.on_request_finished(ctx)
        probe = req('serve')
        for _ in range(3):
            for keys in self.chunks[:8]:
                for key in keys[:256]:
                    timed('lookup', mgr.lookup, key, probe)
        p99 = {name: float(np.percentile(values, 99)) for name, values in samples.items()}
        report = ', '.join('%s %.2f ms' % (name, value * 1e3) for name, value in sorted(p99.items()))
        self.assertLess(p99['lookup'], 0.005, report)
        self.assertLess(p99['prepare_store'], 0.05, report)
        self.assertLess(p99['complete_store'], 0.05, report)
        self.assertLess(p99['on_schedule_end'], 0.1, report)
        mgr.check_invariants()


class FragmentedFreeList(TierCase):
    """The free list is a sorted list of runs. Many small stores and scattered releases fragment it; the calls must stay cheap."""

    def test_scattered_releases_over_a_fragmented_drive_stay_cheap(self):
        import random
        rng = random.Random(5)
        self.tier(num_slots=SLOTS, ttl_s=86400, free_floor_pct=0, reserve_bytes=0, min_store_tokens=0, daily_write_cap_bytes=10 ** 15)
        runtime = self.open_runtime()
        mgr = self.manager(runtime)                                       # no group map: every call is judged alone, so small stores are admitted
        allocator = runtime.allocator
        ctx = req('churn')
        sizes = (1, 1, 2, 3, 5, 8, 13, 64, 96)
        held, next_key, slowest = [], 0, 0.0
        while allocator.n_free > SLOTS // 3:                              # fill two thirds of the drive with small sessions
            n = rng.choice(sizes)
            keys = [key_for(next_key + i) for i in range(n)]
            next_key += n
            out = mgr.prepare_store(keys, ctx)
            mgr.complete_store(out.keys_to_store, ctx, True, np.zeros((len(out.keys_to_store), 4), dtype=np.uint64))
            held.append(keys)
        rng.shuffle(held)
        for victims in range(0, len(held) // 2):                          # release half of them in random order, in bursts of 200 sessions
            if victims % 200:
                continue
            batch = [key for keys in held[victims:victims + 200] for key in keys]
            started = time.perf_counter()
            mgr._remove_many(batch, 'lru')
            slowest = max(slowest, time.perf_counter() - started)
        runs = len(allocator._trim_wait.starts)
        self.assertGreater(runs, 1000, 'the TRIM queue really is fragmented (%d runs)' % runs)
        show('burst of 200 scattered sessions released, %d runs waiting' % runs, slowest)
        self.assertLess(slowest, 0.1, 'the slowest burst of 200 scattered sessions (%d runs waiting) took %.3f s' % (runs, slowest))
        runtime.settle(60)
        started = time.perf_counter()
        out = mgr.prepare_store([key_for(next_key + i) for i in range(2088)], ctx)
        took = time.perf_counter() - started
        self.assertEqual(len(out.keys_to_store), 2088)
        show('2,088-slot allocation from %d free runs' % len(allocator._free.starts), took)
        self.assertLess(took, 0.05, 'a 2,088-slot allocation from %d free runs took %.3f s' % (len(allocator._free.starts), took))
        mgr.check_invariants()
        self.assertEqual(allocator.recount()[:6], [allocator.count(i) for i in range(6)])

    def test_the_trim_completions_of_a_fragmented_drive_are_applied_without_stalling_a_pass(self):
        """Review M-frag: about 164,000 scattered one-slot runs wait for their TRIM and the pump queues 4,096 at a time. Each pass applies up to 4,096
        completions and takes the next 4,096 runs off the queue; the free list takes them in ONE merge and the queue gives them up in ONE slice (it was
        a list insertion and a list deletion per run: 116 ms for the worst pass of the reviewer's bench)."""
        import random
        rng = random.Random(9)
        self.tier(num_slots=SLOTS, ttl_s=86400, free_floor_pct=0, reserve_bytes=0, min_store_tokens=0, daily_write_cap_bytes=10 ** 15)
        runtime = self.open_runtime()
        mgr = self.manager(runtime)
        allocator = runtime.allocator
        ctx = req('fill')
        keys, filled = [], 0
        while filled < SLOTS - 5000:                                      # a drive nearly full of 1-key entries
            n = min(10_000, SLOTS - 5000 - filled)
            batch = [key_for(filled + i) for i in range(n)]
            out = mgr.prepare_store(batch, ctx)
            mgr.complete_store(out.keys_to_store, ctx, True, np.zeros((len(out.keys_to_store), 4), dtype=np.uint64))
            keys.extend(batch)
            filled += n
        scattered = keys[::2]
        rng.shuffle(scattered)
        for i in range(0, len(scattered), 4096):
            mgr._remove_many(scattered[i:i + 4096], 'lru')
        runtime.store.drive.discard = lambda offset, length: None        # the engine thread's own cost: the kit's sync I/O would add 4,096 real discards a pass
        waiting = len(allocator._trim_wait.starts)
        self.assertGreater(waiting, 100_000, 'the TRIM queue is as fragmented as a 1 TB drive after a day of churn (%d runs)' % waiting)
        slowest, passes = 0.0, 0
        gc.collect()
        gc.disable()                                  # a full collection over the vLLM and numpy objects of this process is 80 ms, and is not what is measured
        try:
            while allocator.trim_backlog() and passes < 400:
                started = time.perf_counter()
                runtime.service()
                slowest = max(slowest, time.perf_counter() - started)
                passes += 1
        finally:
            gc.enable()
        self.assertEqual(allocator.trim_backlog(), 0)
        show('worst service pass over %d scattered TRIM runs (%d passes)' % (waiting, passes), slowest)
        self.assertLess(slowest, 0.07, 'the slowest service pass took %.3f s (about 170 ms before the fix, on a quiet machine)' % slowest)
        self.assertGreater(len(allocator._free.starts), 100_000)
        self.assertEqual(allocator.recount()[:6], [allocator.count(i) for i in range(6)])
        mgr.check_invariants()


if __name__ == '__main__':
    unittest.main()
