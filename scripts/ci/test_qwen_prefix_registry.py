"""qwen_prefix_registry: the G1 checkpoint LRU, the grant ledger and the kill switch.

Pure python, no vLLM: runs in the 3.11 CPU suite. The P0a probe's check 15 (LRU by bytes never
evicts a pinned checkpoint) and the polling half of check 16 (the kill switch) live here; the
scheduler-driven checks are test_qwen_prefix_scheduler_patch (fakes) and
test_qwen_prefix_scheduler_vllm (real vLLM objects).
"""

import importlib.util
import os
import sys
import unittest
from array import array
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import qwen_prefix_registry as registry_module  # noqa: E402
from qwen_prefix_registry import (  # noqa: E402
    BLOCK, CHUNK, CHECKPOINT_NBYTES, Grant, KillSwitch, PrefixRegistry, REGISTRY_KEY, current_registry,
    floor_chunk, shared_registry, store_budget_bytes)

IDS = list(range(CHUNK))


def request(tokens):
    return SimpleNamespace(all_token_ids=list(tokens))


def grant(req_id, q=0, h=0, key=None, checkpoint=None, plan=(), tokens=None, orphans=0):
    return Grant(req_id, q, h, key, checkpoint, list(plan), request(tokens or range(3 * CHUNK)), orphans)


class Silence(object):
    """Swallow the [PINDIAG] lines a test provokes on purpose."""

    def __enter__(self):
        self.saved = sys.stderr
        sys.stderr = open(os.devnull, 'w')

    def __exit__(self, *exc):
        sys.stderr.close()
        sys.stderr = self.saved


class ConstantsTests(unittest.TestCase):
    def test_the_geometry_is_the_model_s(self):
        self.assertEqual((CHUNK, BLOCK), (2048, 64))
        self.assertEqual(CHUNK % BLOCK, 0)
        # 48 layers x (fp32 [1,24,128,128] + bf16 [1,3,5120]) x 2 chips, design section 2.0.3.
        self.assertEqual(CHECKPOINT_NBYTES, 153944064)
        self.assertEqual([floor_chunk(n) for n in (0, 2047, 2048, 2049, 6143)], [0, 0, 2048, 2048, 4096])

    def test_budget_from_the_environment(self):
        self.assertEqual(store_budget_bytes({}), 8 << 30)
        self.assertEqual(store_budget_bytes({'QWEN_PREFIX_STORE_GIB': '0.5'}), 1 << 29)
        self.assertEqual(store_budget_bytes({'QWEN_PREFIX_STORE_GIB': ''}), 8 << 30)
        for bad in ('eight', '-1', 'nan'):
            with self.assertRaises(ValueError):
                store_budget_bytes({'QWEN_PREFIX_STORE_GIB': bad})
        self.assertEqual(PrefixRegistry(environ={'QWEN_PREFIX_STORE_GIB': '2'}).budget_bytes, 2 << 30)

    def test_reuse_is_off_unless_exactly_one(self):
        self.assertTrue(registry_module.reuse_enabled({'QWEN_PREFIX_REUSE': '1'}))
        for value in (None, '', '0', 'true', 'yes', ' 1'):
            environ = {} if value is None else {'QWEN_PREFIX_REUSE': value}
            self.assertFalse(registry_module.reuse_enabled(environ))


class CheckpointTests(unittest.TestCase):
    def test_lru_by_bytes_never_evicts_a_pinned_checkpoint(self):
        """P0a check 15."""
        registry = PrefixRegistry(budget_bytes=300)
        for index in range(3):
            registry.put(b'k%d' % index, CHUNK, IDS, nbytes=100)
        registry.entries[b'k0'].pins = 1
        registry.put(b'k3', CHUNK, IDS, nbytes=100)
        self.assertEqual(list(registry.entries), [b'k0', b'k2', b'k3'])
        self.assertEqual(registry.bytes, 300)
        self.assertEqual(registry.stats['evicted_lru'], 1)
        self.assertIsNone(registry.put(b'k4', CHUNK, IDS, nbytes=301))
        self.assertEqual(registry.stats['capture_skipped_budget'], 1)

    def test_only_whole_chunks_with_their_token_ids(self):
        registry = PrefixRegistry(budget_bytes=1 << 20)
        for pos, size in ((CHUNK + 64, CHUNK + 64), (CHUNK, CHUNK - 1), (0, 0)):
            with self.assertRaises(ValueError):
                registry.put(b'bad', pos, list(range(size)), nbytes=1)
        self.assertFalse(registry.entries)

    def test_the_token_check_is_exact(self):
        registry = PrefixRegistry(budget_bytes=1 << 20)
        entry = registry.put(b'k', CHUNK, IDS, nbytes=1)
        self.assertIsInstance(entry.token_ids, array)
        self.assertTrue(entry.matches(IDS))
        self.assertTrue(entry.matches(tuple(IDS)))
        self.assertFalse(entry.matches(IDS[:-1]))
        self.assertFalse(entry.matches(IDS[:-1] + [IDS[-1] + 1]))

    def test_a_recapture_replaces_unless_pinned(self):
        registry = PrefixRegistry(budget_bytes=1 << 20)
        first = registry.put(b'k', CHUNK, IDS, rec='a', nbytes=10)
        first.pins = 1
        self.assertIs(registry.put(b'k', CHUNK, IDS, rec='b', nbytes=10), first)
        self.assertEqual(registry.stats['capture_kept_pinned'], 1)
        first.pins = 0
        second = registry.put(b'k', CHUNK, IDS, rec='c', nbytes=20)
        self.assertIsNot(second, first)
        self.assertEqual((registry.bytes, registry.stats['capture_replaced']), (20, 1))

    def test_drop_counts_by_reason(self):
        registry = PrefixRegistry(budget_bytes=1 << 20)
        registry.put(b'a', CHUNK, IDS, nbytes=5)
        registry.put(b'b', CHUNK, IDS, nbytes=5)
        registry.drop(b'a', 'coupled')
        registry.drop(b'b', 'no-such-reason')
        self.assertIsNone(registry.drop(b'missing', 'coupled'))
        self.assertEqual((registry.stats['evicted_coupled'], registry.stats['dropped'], registry.bytes), (1, 1, 0))


class GrantTests(unittest.TestCase):
    def setUp(self):
        self.registry = PrefixRegistry(budget_bytes=1 << 20)
        self.entry = self.registry.put(b'k2048', CHUNK, IDS, nbytes=10)

    def test_commit_only_what_the_output_admits_at_q(self):
        registry = self.registry
        registry.begin_step()
        registry.stage(grant('a', q=CHUNK, h=2 * CHUNK + 64, key=b'k2048', checkpoint=self.entry,
                             plan=[(2 * CHUNK, b'k4096')]))
        registry.stage(grant('b', q=CHUNK, h=CHUNK, key=b'k2048', checkpoint=self.entry))
        registry.stage(grant('c', q=0, h=64, plan=[(CHUNK, b'x')]))
        registry.stage(grant('d', q=CHUNK, h=CHUNK, key=b'k2048', checkpoint=self.entry))
        with Silence():
            done = registry.commit({'a': CHUNK, 'c': 0, 'd': 0})
        self.assertEqual(sorted(g.req_id for g in done), ['a', 'c'])
        self.assertIsNone(registry.grant_for('b'))
        self.assertIsNone(registry.grant_for('d'))
        self.assertEqual(self.entry.pins, 1)
        stats = registry.stats
        self.assertEqual((stats['dropped_attempts'], stats['commit_mismatch'], stats['grants'],
                          stats['admissions'], stats['grant_tokens']), (1, 1, 1, 2, CHUNK))
        self.assertEqual(stats['trim_loss_tokens'], (2 * CHUNK + 64 - CHUNK) + 64)
        self.assertEqual(stats['kv_hit_without_checkpoint'], 1)
        self.assertEqual(len(registry.grant_for('a').tokens), 2 * CHUNK)
        self.assertIsNone(registry.grant_for('a').request)
        self.assertFalse(registry.staged)

    def test_the_pin_lasts_one_step(self):
        registry = self.registry
        registry.begin_step()
        registry.stage(grant('a', q=CHUNK, h=CHUNK, key=b'k2048', checkpoint=self.entry))
        registry.commit({'a': CHUNK})
        self.assertEqual(registry.pins(), 1)
        registry.begin_step()
        self.assertEqual(registry.pins(), 0)
        self.assertIsNone(registry.grant_for('a'))

    def test_staging_is_idempotent_across_attempts(self):
        registry = self.registry
        registry.begin_step()
        for _ in range(3):
            registry.stage(grant('a', q=CHUNK, h=CHUNK, key=b'k2048', checkpoint=self.entry))
        self.assertEqual(len(registry.staged), 1)
        self.assertEqual(self.entry.pins, 0)
        registry.commit({})
        self.assertEqual(registry.stats['dropped_attempts'], 1)

    def test_forget_releases_the_pin(self):
        registry = self.registry
        registry.begin_step()
        registry.stage(grant('a', q=CHUNK, h=CHUNK, key=b'k2048', checkpoint=self.entry))
        registry.commit({'a': CHUNK})
        registry.stage(grant('w', q=CHUNK, h=CHUNK, key=b'k2048', checkpoint=self.entry))
        self.assertTrue(registry.forget_request('a'))
        self.assertTrue(registry.forget_request('w'))
        self.assertFalse(registry.forget_request('never'))
        self.assertEqual((registry.pins(), registry.stats['freed_requests']), (0, 2))

    def test_capture_stores_only_planned_boundaries_and_never_raises(self):
        registry = self.registry
        registry.begin_step()
        tokens = list(range(1000, 1000 + 3 * CHUNK))
        registry.stage(Grant('a', 0, 64, None, None, [(CHUNK, b'p2048'), (2 * CHUNK, b'p4096')],
                             request(tokens)))
        registry.commit({'a': 0})
        with Silence():
            stored = registry.capture('a', 2 * CHUNK, rec='r', carry='c', nbytes=7, ms=3.5)
            self.assertIsNone(registry.capture('a', 3 * CHUNK))
            self.assertIsNone(registry.capture('nobody', CHUNK))
        self.assertEqual((stored.pos, stored.rec, stored.carry, stored.nbytes), (2 * CHUNK, 'r', 'c', 7))
        self.assertTrue(stored.matches(tokens[0:2 * CHUNK]))
        self.assertEqual(registry.stats['capture_failures'], 2)
        self.assertEqual(registry.stats['capture_ms'], 3.5)
        registry.budget_bytes = 1 << 30
        self.assertEqual(registry.capture('a', CHUNK).nbytes, CHECKPOINT_NBYTES)
        registry.note_restore(12.0)
        self.assertEqual((registry.stats['restores'], registry.stats['restore_ms']), (1, 12.0))

    def test_clear_keeps_admitted_grants_restorable(self):
        registry = self.registry
        registry.begin_step()
        registry.stage(grant('a', q=CHUNK, h=CHUNK, key=b'k2048', checkpoint=self.entry))
        registry.commit({'a': CHUNK})
        registry.clear()
        self.assertFalse(registry.entries)
        self.assertIs(registry.grant_for('a').checkpoint, self.entry)
        self.assertEqual(registry.bytes, 0)

    def test_disable_latches_and_refuses_captures(self):
        registry = self.registry
        registry.begin_step()
        registry.stage(Grant('a', 0, 0, None, None, [(CHUNK, b'p')], request(range(CHUNK + 5))))
        registry.commit({'a': 0})
        registry.disable('kill switch')
        registry.disable('again')
        self.assertEqual(registry.disabled, 'kill switch')
        self.assertIsNone(registry.capture('a', CHUNK))
        self.assertIsNone(registry.put(b'z', CHUNK, IDS, nbytes=1))
        self.assertEqual(registry.stats['capture_disabled'], 2)
        self.assertFalse(registry.entries)
        self.assertEqual(registry.snapshot()['disabled'], 'kill switch')

    def test_snapshot(self):
        snap = self.registry.snapshot()
        self.assertEqual((snap['entries'], snap['bytes'], snap['pins'], snap['budget_bytes']), (1, 10, 0, 1 << 20))
        self.assertTrue(set(registry_module.STAT_NAMES) <= set(snap))


class OwnerTests(unittest.TestCase):
    def test_one_live_scheduler_per_registry(self):
        class Owner(object):
            pass

        registry = PrefixRegistry(budget_bytes=1 << 20)
        first, second = Owner(), Owner()
        registry.bind(first)
        registry.bind(first)
        with self.assertRaises(RuntimeError):
            registry.bind(second)
        registry.put(b'k', CHUNK, IDS, nbytes=1)
        del first
        import gc
        gc.collect()
        registry.bind(second)
        self.assertFalse(registry.entries, 'a rebuilt engine starts from an empty registry')

    def test_a_dead_owner_held_only_by_a_cycle_is_replaced(self):
        """The graft and its scheduler reference each other, so a dead engine lingers until gc."""
        class Owner(object):
            pass

        import gc
        registry = PrefixRegistry(budget_bytes=1 << 20)
        first = Owner()
        first.cycle = first
        registry.bind(first)
        del first
        gc.disable()
        try:
            registry.bind(Owner())
        finally:
            gc.enable()


class KillSwitchTests(unittest.TestCase):
    def test_polls_at_most_once_per_interval_and_latches(self):
        """P0a check 16, the polling half (the scheduler half is in the graft tests)."""
        now = [0.0]
        present = [False]
        calls = []

        def exists(path):
            calls.append(path)
            return present[0]

        switch = KillSwitch('/flag', 1.0, clock=lambda: now[0], exists=exists)
        self.assertFalse(switch.poll())
        present[0] = True
        now[0] = 0.5
        self.assertFalse(switch.poll(), 'within the interval the flag is not read')
        self.assertEqual(len(calls), 1)
        now[0] = 1.0
        self.assertTrue(switch.poll())
        self.assertTrue(switch.engaged)
        present[0] = False
        now[0] = 10.0
        self.assertFalse(switch.poll(), 'engaged is reported once')
        self.assertTrue(switch.engaged, 'and latches: removing the flag does not re-enable')
        self.assertEqual(len(calls), 2)

    def test_no_path_never_engages_and_errors_read_as_absent(self):
        self.assertFalse(KillSwitch(None).poll())

        def broken(path):
            raise OSError('stale mount')

        switch = KillSwitch('/flag', 0.0, exists=broken)
        self.assertFalse(switch.poll())
        self.assertFalse(switch.engaged)

    def test_the_default_flag_is_on_the_persistent_mount(self):
        self.assertEqual(registry_module.KILL_SWITCH_PATH, '/models/.qwen-c2/prefix-reuse.off')
        self.assertEqual(registry_module.KILL_SWITCH_POLL_S, 1.0)


class SharedRegistryTests(unittest.TestCase):
    def setUp(self):
        self.saved = sys.modules.pop(REGISTRY_KEY, None)

    def tearDown(self):
        sys.modules.pop(REGISTRY_KEY, None)
        if self.saved is not None:
            sys.modules[REGISTRY_KEY] = self.saved

    def test_two_copies_of_the_module_share_one_registry(self):
        """The plugin package and the model tree may load this file under two names."""
        self.assertIsNone(current_registry())
        spec = importlib.util.spec_from_file_location('qwen_prefix_registry_copy', registry_module.__file__)
        copy = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(copy)
        first = shared_registry(environ={})
        self.assertIs(copy.shared_registry(), first)
        self.assertIs(copy.current_registry(), first)
        self.assertIs(sys.modules[REGISTRY_KEY].registry, first)

    def test_the_request_ids_kwarg_matches_the_runner_patch(self):
        import qwen_prefix_runner_patch

        self.assertEqual(registry_module.REQUEST_IDS_KWARG, qwen_prefix_runner_patch.REQUEST_IDS_KWARG)


if __name__ == '__main__':
    unittest.main()
