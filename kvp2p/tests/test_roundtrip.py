"""A KV session leaves the card and comes back, byte for byte, through vLLM's real scheduler (hybrid: one full-attention group
and five sliding-window groups), the connector, the worker, a numpy card and a file drive."""

import unittest

from harness import BLOCK, GROUPS, RigCase, tokens

PROMPT = 2300           # 17 full blocks; the sliding groups store their last 8 blocks before the prompt boundary
FULL_BLOCKS = PROMPT // BLOCK


class RoundTripTests(RigCase):
    def test_a_session_stores_then_restores_byte_exact_across_all_groups(self):
        rig = self.rig()
        rig.add('r1', tokens(PROMPT, 'session'))
        rig.drain()
        # 17 full-attention objects and 8 per sliding group (2,088 = 2,048 + 40 at 262k: full blocks plus 5 x 8)
        by_group = {}
        for key, entry in rig.manager.index.items():
            assert entry.state == 'WRITTEN'
            by_group[int.from_bytes(key[-4:], 'big')] = by_group.get(int.from_bytes(key[-4:], 'big'), 0) + 1
        self.assertEqual(by_group, {0: FULL_BLOCKS, **{g: 8 for g in range(1, GROUPS)}})
        self.assertEqual(rig.assert_drive_exact(), FULL_BLOCKS + 5 * 8)
        self.assertEqual(rig.worker.stats['done_store'], 1)

        rig.reset_device_cache()                        # POST /reset_prefix_cache?reset_external=false
        rig.add('r2', tokens(PROMPT, 'session'))
        rig.finish('r2')
        self.assertEqual(rig.loads, [('r2', (FULL_BLOCKS, 8, 8, 8, 8, 8))])      # the hit reaches the last full block
        self.assertEqual(rig.worker.stats['done_load'], 1)
        self.assertEqual(rig.worker.stats['objects_scattered'], FULL_BLOCKS + 5 * 8)
        # every step of the run checked that each non-null block below the computed tokens holds exactly its hash's content
        self.assertEqual(rig.runtime.allocator.handed_out_in_flight, [])
        self.assertEqual(sum(rig.runtime.allocator.inflight), 0)

    def test_a_returning_session_with_new_text_restores_the_prefix_and_prefills_the_rest(self):
        rig = self.rig()
        prompt = tokens(PROMPT, 'turn1')
        rig.add('t1', prompt)
        rig.drain()
        rig.reset_device_cache()
        follow_up = prompt + tokens(700, 'turn2')
        rig.add('t2', follow_up)
        rig.finish('t2')
        self.assertEqual(rig.loads, [('t2', (FULL_BLOCKS, 8, 8, 8, 8, 8))])
        self.assertEqual(rig.worker.stats['objects_scattered'], FULL_BLOCKS + 5 * 8)

    def test_a_cold_prompt_misses_and_a_device_hit_loads_nothing(self):
        rig = self.rig()
        rig.add('a', tokens(PROMPT, 'x'))
        rig.drain()
        before = dict(rig.worker.stats)
        rig.add('b', tokens(PROMPT, 'y'))                # another prompt: no hit
        rig.drain()
        self.assertEqual(rig.worker.stats['submit_load'], 0)
        self.assertGreater(rig.worker.stats['done_store'], before['done_store'])
        rig.add('c', tokens(PROMPT, 'x'))                # the device prefix cache still has it: the local hit wins
        rig.drain()
        self.assertEqual(rig.worker.stats['submit_load'], 0)

    def test_fragmented_slots_use_object_offsets_inside_the_slot_and_subruns(self):
        rig = self.rig(num_slots=240, extra={'spill_subrun_objects': 4})
        allocator = rig.runtime.allocator
        everything = allocator.alloc(240)
        for slot in everything[::3]:                     # every third slot is free again: no two allocations are adjacent
            allocator.release(slot)
        rig.add('r1', tokens(4000, 'long'))              # 31 full blocks + 40 sliding objects
        rig.drain()
        writes = [c for c in rig.io.calls if c[0] == 'write']
        self.assertTrue(writes)
        self.assertTrue(all(call[3] <= 4 for call in writes), 'sub-runs are at most spill_subrun_objects')
        self.assertTrue(any(call[4] > 0 for call in writes), 'scattered slots land at object offsets inside the staging slot')
        self.assertEqual(rig.assert_drive_exact(), 31 + 40)
        rig.reset_device_cache()
        rig.add('r2', tokens(4000, 'long'))
        rig.finish('r2')
        reads = [c for c in rig.io.calls if c[0] == 'read']
        self.assertTrue(any(call[4] > 0 for call in reads))
        self.assertEqual(rig.worker.stats['objects_scattered'], 31 + 40)
        self.assertEqual(sum(rig.runtime.allocator.inflight), 0)

    def test_two_sessions_back_to_back_share_the_banks(self):
        rig = self.rig()
        for name in ('s1', 's2', 's3'):
            rig.add(name, tokens(PROMPT, name))
        rig.drain()
        self.assertEqual(rig.assert_drive_exact(), 3 * (FULL_BLOCKS + 40))
        rig.reset_device_cache()
        for name in ('s1', 's2', 's3'):
            rig.add('again-' + name, tokens(PROMPT, name))
        rig.drain()
        self.assertEqual(rig.worker.stats['done_load'], 3)


class AsyncSchedulingRoundTripTests(RoundTripTests):
    """The same tests with async scheduling: two batches in flight, so every completion reaches the scheduler one step late."""

    def rig(self, **kwargs):
        kwargs.setdefault('async_scheduling', True)
        return super().rig(**kwargs)


if __name__ == '__main__':
    unittest.main()
