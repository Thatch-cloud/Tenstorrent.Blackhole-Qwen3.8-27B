"""Per-tenant quotas in the prefix store (tp4/prefix-tiers; docs/prefix-store-hygiene.md "Tenant quotas"), and the cold-tier client interface.

* the strict parser (qwen_prefix_registry.quota_config);
* the host tier's per-tenant bytes (TierStore: the cap evicts the tenant's OWN oldest records first, the floor spares a tenant's last bytes from everyone else, both
  off by default and then byte-identical to the store that was);
* the device pool's per-tenant tokens (qwen_prefix_scheduler_patch.DeviceQuota) on a faithful copy of vLLM's block pool and free queue: the eviction order the
  module states (own excess first, then vLLM's order skipping protected blocks, an override only when nothing else will do), the counters (aggregates, no tenant in
  any key), and the wiring through the real install();
* the cold-tier client's fake (cold_tier_client).
"""

import hashlib
import json
import os
import random
import sys
import types
import unittest
from collections import OrderedDict
from types import SimpleNamespace
from unittest import mock

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import cold_tier_client as cold  # noqa: E402
import qwen_prefix_registry as registry_module  # noqa: E402
import qwen_prefix_scheduler_patch as patch_module  # noqa: E402
from qwen_prefix_registry import PrefixRegistry, TierStore, quota_config  # noqa: E402
from test_qwen_prefix_scheduler_patch import (  # noqa: E402
    Blocks, Coordinator, FakeScheduler, HashMap, Quiet, Request, fake_vllm_modules, tokens)

BLOCK = registry_module.BLOCK
CHUNK = registry_module.CHUNK
GIB = 1 << 30
DEVICE_CAP = registry_module.ENV_QUOTA_DEVICE_CAP
DEVICE_FLOOR = registry_module.ENV_QUOTA_DEVICE_FLOOR
HOST_CAP = registry_module.ENV_QUOTA_HOST_CAP
HOST_FLOOR = registry_module.ENV_QUOTA_HOST_FLOOR


# -- (1) the parser -------------------------------------------------------------------------------------------

class ParserTests(unittest.TestCase):
    def test_nothing_named_is_no_quota_and_zero_is_nothing(self):
        self.assertIsNone(quota_config({}))
        self.assertIsNone(quota_config({DEVICE_CAP: '', DEVICE_FLOOR: ''}))
        self.assertIsNone(quota_config({DEVICE_CAP: '0', DEVICE_FLOOR: '0'}))

    def test_the_device_numbers_are_tokens_in_whole_blocks(self):
        config = quota_config({DEVICE_CAP: '262144', DEVICE_FLOOR: '65536'})
        self.assertEqual((config.device_cap_blocks, config.device_floor_blocks), (4096, 1024))
        self.assertTrue(config.device_on and not config.host_on)
        self.assertEqual(config.describe(), dict(device_cap_tokens=262144, device_floor_tokens=65536, host_cap_bytes=0, host_floor_bytes=0))
        only_floor = quota_config({DEVICE_FLOOR: '4096'})
        self.assertEqual((only_floor.device_cap_blocks, only_floor.device_floor_blocks), (0, 64), 'a floor alone is a quota: no cap')

    def test_every_malformed_value_refuses_to_start(self):
        bad = [({DEVICE_CAP: '1e3'}, 'digits only'), ({DEVICE_CAP: '-64'}, 'digits only'), ({DEVICE_CAP: ' 4096'}, 'digits only'), ({DEVICE_CAP: '4096.0'}, 'digits only'),
               ({DEVICE_CAP: '٤٠٩٦'}, 'digits only'), ({DEVICE_CAP: '100'}, 'multiple of the 64-token block'),
               ({DEVICE_CAP: '1024'}, 'below one 2048-token chunk'), ({DEVICE_FLOOR: 'abc'}, 'digits only'),
               ({DEVICE_CAP: '4096', DEVICE_FLOOR: '8192'}, 'exceeds'), ({DEVICE_FLOOR: '63'}, 'multiple')]
        for environ, why in bad:
            with self.subTest(environ=environ):
                with self.assertRaises(ValueError) as caught:
                    quota_config(environ)
                self.assertIn(why, str(caught.exception))

    def test_a_host_quota_needs_the_tier_and_sane_numbers(self):
        with self.assertRaises(ValueError) as caught:
            quota_config({HOST_CAP: '1'})
        self.assertIn('need the host KV tier', str(caught.exception))
        config = quota_config({HOST_CAP: '12', HOST_FLOOR: '4'}, tier_kv_bytes=24 * GIB)
        self.assertEqual((config.host_cap_bytes, config.host_floor_bytes), (12 * GIB, 4 * GIB))
        self.assertTrue(config.host_on and not config.device_on)
        for environ in ({HOST_CAP: 'inf'}, {HOST_CAP: 'nan'}, {HOST_CAP: '-1'}, {HOST_CAP: 'x'}, {HOST_CAP: '2', HOST_FLOOR: '3'}, {HOST_FLOOR: '13'}):
            with self.subTest(environ=environ):
                with self.assertRaises(ValueError):
                    quota_config(environ, tier_kv_bytes=24 * GIB)

    def test_the_registry_reads_it_and_adds_its_counters_only_when_a_quota_is_on(self):
        plain = PrefixRegistry(budget_bytes=100, environ={})
        self.assertIsNone(plain.quota)
        self.assertFalse(any(name.startswith('quota_') for name in plain.stats))
        self.assertFalse(any(name.startswith('quota_') for name in plain.snapshot()), 'the stats of an engine without a quota are what they were')
        on = PrefixRegistry(budget_bytes=100, environ={DEVICE_CAP: '8192'})
        self.assertEqual(on.quota.device_cap_blocks, 128)
        for name in registry_module.QUOTA_STAT_NAMES:
            self.assertEqual(on.stats[name], 0, name)
        snapshot = on.snapshot()
        self.assertEqual((snapshot['quota_on'], snapshot['quota_dev_tenants'], snapshot['quota_dev_at_cap']), (1, 0, 0))
        with self.assertRaises(ValueError):
            PrefixRegistry(budget_bytes=100, environ={DEVICE_CAP: '1000'})


# -- (2) the host tier's bytes --------------------------------------------------------------------------------------

def payload(size=100, fill=1):
    return bytes([fill]) * size


class HostQuotaTests(unittest.TestCase):
    def store(self, cap=1000, cap_per=0, floor=0, policy='lru', slab=0):
        counters = dict.fromkeys(registry_module.QUOTA_STAT_NAMES, 0)
        store = TierStore(cap, policy, digest_inline=True, tenant_cap=cap_per, tenant_floor=floor, counters=counters)
        if slab:
            self.assertTrue(store.configure(slab))
        return store, counters

    def fill(self, store, tenant, count, start=0, size=100):
        for index in range(count):
            self.assertTrue(store.put(('%s-%d' % (tenant, start + index)).encode(), payload(size, start + index + 1), tenant))

    def test_a_tenant_over_its_cap_evicts_its_own_oldest_records_first(self):
        store, counters = self.store(cap=1000, cap_per=400)
        self.fill(store, 'b', 2)
        self.fill(store, 'a', 6)
        self.assertEqual(store.tenant_bytes['a'], 400, 'a is held at its cap')
        self.assertEqual(store.tenant_bytes['b'], 200, 'b was never touched')
        self.assertEqual(sorted(k for k in store.records if k.startswith(b'a-')), [b'a-2', b'a-3', b'a-4', b'a-5'], 'a lost its oldest')
        self.assertEqual((counters['quota_host_evictions_own'], counters['quota_host_evictions_global']), (2, 0))

    def test_without_the_quota_the_same_flood_evicts_the_other_tenant(self):
        store, counters = self.store(cap=600)
        self.fill(store, 'b', 2)
        self.fill(store, 'a', 6)
        self.assertEqual(store.tenant_bytes.get('b', 0), 0, 'plain LRU: a flood pushes b out')
        self.assertEqual(counters['quota_host_evictions_own'], 0)

    def test_the_floor_spares_a_tenants_last_bytes_from_everyone_else(self):
        store, counters = self.store(cap=600, floor=200)
        self.fill(store, 'b', 2)
        self.fill(store, 'a', 8)
        self.assertEqual(store.tenant_bytes['b'], 200, 'b keeps its floor through the flood')
        self.assertEqual(store.tenant_bytes['a'], 400)
        self.assertGreater(counters['quota_host_floor_protected'], 0)
        self.assertGreater(counters['quota_host_evictions_global'], 0)

    def test_above_the_floor_a_tenants_oldest_records_are_fair_game(self):
        store, _ = self.store(cap=600, floor=200)
        self.fill(store, 'b', 4)
        self.fill(store, 'a', 4)
        self.assertEqual(store.tenant_bytes['b'], 200, 'b was brought down to its floor, not below')
        self.assertEqual(store.tenant_bytes['a'], 400)
        self.assertEqual(sorted(k for k in store.records if k.startswith(b'b-')), [b'b-2', b'b-3'], 'b lost its oldest above the floor')

    def test_when_only_protected_records_remain_the_spill_is_refused_not_forced(self):
        store, counters = self.store(cap=300, floor=300)
        self.fill(store, 'b', 3)
        self.assertFalse(store.put(b'a-0', payload(100, 9), 'a'), 'every record is another tenant\'s floor')
        self.assertEqual(store.tenant_bytes, {'b': 300})
        self.assertEqual(store.stats['refused_full'], 1)

    def test_the_fair_policy_skips_protected_records_too(self):
        store, _ = self.store(cap=600, floor=200, policy='fair')
        self.fill(store, 'b', 2)
        self.fill(store, 'a', 8)
        self.assertEqual(store.tenant_bytes['b'], 200)

    def test_a_record_larger_than_the_cap_is_refused_and_pinned_ones_stay(self):
        store, counters = self.store(cap=1000, cap_per=250)
        self.assertFalse(store.put(b'big', payload(300), 'a'))
        self.assertEqual(counters['quota_host_refused'], 1)
        self.fill(store, 'a', 2)
        store.pin([b'a-0', b'a-1'])
        self.assertFalse(store.put(b'a-9', payload(100, 3), 'a'), 'the tenant\'s own records are pinned: no room under its cap')
        store.unpin([b'a-0', b'a-1'])
        self.assertTrue(store.put(b'a-9', payload(100, 3), 'a'))

    def test_slab_mode_reserves_under_the_cap_and_commits(self):
        store, counters = self.store(cap=1000, cap_per=300, slab=100)
        number = 0
        for tenant, count in (('b', 2), ('a', 3), ('a', 2)):
            slots = store.reserve(count, [tenant] * count)
            self.assertEqual(len(slots), count)
            for slot in slots:
                slot.view[:] = payload(100, number % 250 + 1)
                store.commit(slot, ('%s-%d' % (tenant, number)).encode(), tenant)
                number += 1
        self.assertEqual(store.tenant_bytes['a'], 300, 'a is held at its cap')
        self.assertEqual(store.tenant_bytes['b'], 200)
        self.assertEqual(store.reserved, {}, 'every reservation was committed')
        self.assertEqual(counters['quota_host_evictions_own'], 2, 'the second batch of a evicted a\'s two oldest records')
        self.assertEqual(sorted(k for k in store.records if k.startswith(b'a-')), [b'a-4', b'a-5', b'a-6'])
        slots = store.reserve(2, ['c', 'c'])
        self.assertEqual(store.reserved, {'c': 200})
        store.abort(slots)
        self.assertEqual(store.reserved, {})

    def test_a_batch_of_reservations_counts_what_it_already_holds(self):
        store, counters = self.store(cap=1000, cap_per=300, slab=100)
        slots = store.reserve(6, ['a'] * 6)
        self.assertEqual(len(slots), 3, 'the cap bounds the reservations of one batch: three of a\'s six fit')
        self.assertEqual(counters['quota_host_refused'], 1)
        store.abort(slots)

    def test_off_it_is_the_store_that_was(self):
        for seed in range(6):
            rng = random.Random(seed)
            plain = TierStore(1000, 'lru', digest_inline=True)
            quiet = TierStore(1000, 'lru', digest_inline=True, tenant_cap=0, tenant_floor=0, counters={})
            for step in range(300):
                key = b'k%d' % rng.randrange(30)
                tenant = rng.choice(['a', 'b', 'c', None])
                size = rng.choice([100, 100, 150, 400])
                if rng.random() < 0.8:
                    results = [plain.put(key, payload(size, step % 250), tenant), quiet.put(key, payload(size, step % 250), tenant)]
                    self.assertEqual(results[0], results[1])
                else:
                    plain.remove(key, evicted=rng.random() < 0.5)
                    quiet.remove(key, evicted=False)
                    if key in plain.records != key in quiet.records:
                        quiet.remove(key)
                self.assertEqual(list(plain.records), list(quiet.records), 'seed %d step %d' % (seed, step))
            self.assertEqual(plain.tenant_bytes, quiet.tenant_bytes)

    def test_the_aggregates_name_no_tenant(self):
        store, counters = self.store(cap=1000, cap_per=300)
        self.fill(store, 'tenant-secret-a', 5)
        self.fill(store, 'tenant-secret-b', 5)
        self.assertEqual(store.tenants_at_cap(), 2)
        text = json.dumps(dict(counters, **store.snapshot()))
        self.assertNotIn('secret', text)


# -- (3) the device pool: a faithful copy of vLLM's block pool and free queue ----------------------------------------------

class ShimBlock(object):
    """vLLM's KVCacheBlock, the fields the graft and the quota read."""

    def __init__(self, block_id):
        self.block_id = block_id
        self.ref_cnt = 0
        self.block_hash = None
        self.prev_free_block = None
        self.next_free_block = None
        self.is_null = False


class ShimQueue(object):
    """vLLM's FreeKVCacheBlockQueue (v1/core/kv_cache_utils.py): a doubly linked list with fake head and tail, popleft_n, remove, append, prepend_n."""

    def __init__(self, blocks):
        self.num_free_blocks = len(blocks)
        for index in range(len(blocks)):
            if index > 0:
                blocks[index].prev_free_block = blocks[index - 1]
            if index < len(blocks) - 1:
                blocks[index].next_free_block = blocks[index + 1]
        self.fake_free_list_head = ShimBlock(-1)
        self.fake_free_list_tail = ShimBlock(-1)
        if blocks:
            self.fake_free_list_head.next_free_block = blocks[0]
            blocks[0].prev_free_block = self.fake_free_list_head
            self.fake_free_list_tail.prev_free_block = blocks[-1]
            blocks[-1].next_free_block = self.fake_free_list_tail
        else:
            self.fake_free_list_head.next_free_block = self.fake_free_list_tail
            self.fake_free_list_tail.prev_free_block = self.fake_free_list_head

    def popleft_n(self, count):
        if count == 0:
            return []
        assert self.num_free_blocks >= count
        self.num_free_blocks -= count
        current = self.fake_free_list_head.next_free_block
        out = []
        for _ in range(count):
            out.append(current)
            last = current
            current = current.next_free_block
            last.prev_free_block = None
            last.next_free_block = None
        if current is not None:
            self.fake_free_list_head.next_free_block = current
            current.prev_free_block = self.fake_free_list_head
        return out

    def remove(self, block):
        if block.prev_free_block is None or block.next_free_block is None:
            raise RuntimeError('remove() called on an invalid block: %s' % block.block_id)
        block.prev_free_block.next_free_block = block.next_free_block
        block.next_free_block.prev_free_block = block.prev_free_block
        block.prev_free_block = block.next_free_block = None
        self.num_free_blocks -= 1

    def append(self, block):
        last = self.fake_free_list_tail.prev_free_block
        last.next_free_block = block
        block.prev_free_block = last
        block.next_free_block = self.fake_free_list_tail
        self.fake_free_list_tail.prev_free_block = block
        self.num_free_blocks += 1

    def append_n(self, blocks):
        for block in blocks:
            self.append(block)

    def prepend_n(self, blocks):
        if not blocks:
            return
        first = self.fake_free_list_head.next_free_block
        previous = self.fake_free_list_head
        for block in blocks:
            block.prev_free_block = previous
            previous.next_free_block = block
            previous = block
        previous.next_free_block = first
        first.prev_free_block = previous
        self.num_free_blocks += len(blocks)

    def order(self):
        out, block = [], self.fake_free_list_head.next_free_block
        while block is not self.fake_free_list_tail:
            out.append(block.block_id)
            block = block.next_free_block
        return out


class QueuePool(object):
    """vLLM's BlockPool (v1/core/block_pool.py) reduced to what the graft and the quota call: the free queue, ref counts, the hash map, eviction on reuse, free_blocks
    (cached blocks to the tail, uncached to the front), touch, cache_full_blocks."""

    def __init__(self, count):
        self.blocks = [ShimBlock(index) for index in range(count)]
        self.free_block_queue = ShimQueue(list(self.blocks))
        self.cached_block_hash_to_block = HashMap()
        self.cached_block_hashes_by_block = {}
        self.hash_block_size = BLOCK
        self.evicted = []

    def get_num_free_blocks(self):
        return self.free_block_queue.num_free_blocks

    def get_new_blocks(self, count):
        if count > self.get_num_free_blocks():
            raise ValueError('Cannot get %d free blocks from the pool' % count)
        taken = self.free_block_queue.popleft_n(count)
        for block in taken:
            self._maybe_evict_cached_block(block)
            assert block.ref_cnt == 0
            block.ref_cnt += 1
        return taken

    def _maybe_evict_cached_block(self, block):
        if block.block_hash is None:
            return False
        self.cached_block_hash_to_block.pop(block.block_hash, block)
        block.block_hash = None
        self.evicted.append(block.block_id)
        return True

    def touch(self, blocks):
        for block in blocks:
            if block.ref_cnt == 0 and not block.is_null:
                self.free_block_queue.remove(block)
            block.ref_cnt += 1

    def free_blocks(self, ordered_blocks):
        hashed, plain = [], []
        for block in ordered_blocks:
            block.ref_cnt -= 1
            if block.ref_cnt == 0 and not block.is_null:
                (plain if block.block_hash is None else hashed).append(block)
        self.free_block_queue.prepend_n(plain)
        self.free_block_queue.append_n(hashed)

    def cache_full_blocks(self, request, blocks, num_cached_blocks, num_full_blocks, block_size, kv_cache_group_id, block_mask=None):
        for index in range(num_cached_blocks, num_full_blocks):
            key = (request.block_hashes[index], 0)
            blocks[index].block_hash = key
            self.cached_block_hash_to_block.insert(key, blocks[index])


class QueueManager(object):
    """The KV cache manager over a QueuePool: get_computed_blocks (the leading cached blocks), allocate_slots (touch the hits, allocate the rest)."""

    def __init__(self, num_blocks):
        self.block_pool = QueuePool(num_blocks)
        self.coordinator = Coordinator(self.block_pool)
        self.enable_caching = True
        self.empty_kv_cache_blocks = Blocks(([],))

    def create_kv_cache_blocks(self, blocks):
        return Blocks(blocks)

    def get_computed_blocks(self, request):
        hits = []
        limit = (request.num_tokens - 1) // BLOCK
        for index in range(limit):
            block = self.block_pool.cached_block_hash_to_block.get_one_block((request.block_hashes[index], 0))
            if block is None:
                break
            hits.append(block)
        return Blocks((hits,)), len(hits) * BLOCK

    def allocate_slots(self, request, num_new_tokens, new_computed_blocks=None):
        hit = list(new_computed_blocks) if new_computed_blocks else []
        needed = -(-request.num_tokens // BLOCK) - len(hit)
        pool = self.block_pool
        evictable = sum(1 for block in hit if block.ref_cnt == 0)
        if needed > pool.get_num_free_blocks() - evictable:
            return None
        pool.touch(hit)
        return hit + pool.get_new_blocks(needed)


class QueueScheduler(FakeScheduler):
    """FakeScheduler over the vLLM-shaped pool: admission goes through the manager's allocate_slots, a finishing request frees through the pool's free_blocks (reversed:
    the last block is evicted first), as vLLM's single-type manager does."""

    def __init__(self, num_blocks=400, max_model_len=65536):
        super().__init__(num_blocks=num_blocks, max_model_len=max_model_len)
        self.kv_cache_manager = QueueManager(num_blocks)

    def schedule(self):
        admitted = []
        manager = self.kv_cache_manager
        while self.waiting:
            request = self.waiting[0]
            blocks, start = manager.get_computed_blocks(request)
            hit = list(blocks.blocks[0]) if start else []
            owned = manager.allocate_slots(request, request.num_tokens - start, new_computed_blocks=hit)
            if owned is None:
                break
            self.waiting.pop(0)
            single = manager.coordinator.single_type_managers[0]
            single.req_to_blocks[request.request_id] = owned
            single.num_cached_block[request.request_id] = len(hit)
            manager.coordinator.cache_blocks(request, request.num_tokens)
            admitted.append((request.request_id, start))
        from test_qwen_prefix_scheduler_patch import output
        return output(admitted)

    def finish(self, request):
        single = self.kv_cache_manager.coordinator.single_type_managers[0]
        owned = single.req_to_blocks.pop(request.request_id)
        self.kv_cache_manager.block_pool.free_blocks(reversed(owned))
        single.num_cached_block.pop(request.request_id, None)
        self._free_request(request)


def salt_of(tenant):
    return 'salt-%s' % tenant


def session(name, tenant, blocks, seed=0):
    """A request of `blocks` blocks (a multiple of the 2048-token chunk, so the cap publishes every block) of its own tokens under the tenant's salt."""
    assert blocks % (CHUNK // BLOCK) == 0
    return Request('%s' % name, tokens(blocks * BLOCK, seed or (sum(map(ord, name)) + 1)), salt=salt_of(tenant))


class Rig(object):
    """The real install() over QueueScheduler and a registry with the quota environment."""

    def __init__(self, num_blocks=400, **environ):
        self.patches = [mock.patch.dict(sys.modules, fake_vllm_modules()), mock.patch.dict(os.environ, {'QWEN_PREFIX_STATS_PATH': ''})]
        for patcher in self.patches:
            patcher.start()
        self.lines = []
        self.scheduler = QueueScheduler(num_blocks=num_blocks)
        self.registry = PrefixRegistry(budget_bytes=1 << 40, environ=dict(environ))
        with Quiet():
            self.registry.enable_mid_loop_capture()
        self.graft = patch_module.install(self.scheduler, registry=self.registry, kill_switch_path=None, logger=self.logger,
                                          stats=registry_module.StatsExport(path='', interval_s=0.0, logger=self.logger))
        self.pool = self.scheduler.kv_cache_manager.block_pool

    def close(self):
        for patcher in reversed(self.patches):
            patcher.stop()

    def logger(self, message, *values):
        self.lines.append(message % values if values else message)

    def run(self, request, finish=True):
        self.scheduler.add(request)
        out = self.scheduler.schedule()
        assert [data.req_id for data in out.scheduled_new_reqs] == [request.request_id], 'the request was not admitted'
        if finish:
            self.scheduler.finish(request)
        return out

    def finish(self, request):
        self.scheduler.finish(request)

    def cached(self, request):
        """How many of the request's blocks the pool still serves as cached."""
        count = 0
        for index in range(len(request.block_hashes)):
            if self.pool.cached_block_hash_to_block.get_one_block((request.block_hashes[index], 0)) is None:
                break
            count += 1
        return count

    def cached_total(self, request):
        return sum(1 for index in range(len(request.block_hashes)) if self.pool.cached_block_hash_to_block.get_one_block((request.block_hashes[index], 0)) is not None)


class DeviceQuotaTests(unittest.TestCase):
    def rig(self, num_blocks=400, **environ):
        made = Rig(num_blocks=num_blocks, **environ)
        self.addCleanup(made.close)
        return made

    def test_off_nothing_is_wrapped_and_the_stats_are_what_they_were(self):
        made = self.rig()
        self.assertIsNone(made.graft.quota)
        self.assertNotIn('free_blocks', made.pool.__dict__)
        self.assertNotIn('get_new_blocks', made.pool.__dict__)
        self.assertFalse(any('tenant quota' in line for line in made.lines))
        self.assertFalse(any(name.startswith('quota_') for name in made.registry.snapshot()))

    def test_on_it_installs_and_says_so_with_numbers_only(self):
        made = self.rig(**{DEVICE_CAP: str(64 * BLOCK), DEVICE_FLOOR: str(32 * BLOCK)})
        self.assertIsNotNone(made.graft.quota)
        lines = [line for line in made.lines if line.startswith('tenant quota on')]
        self.assertEqual(lines, ['tenant quota on device_cap_tokens=4096 device_floor_tokens=2048 host_cap_bytes=0 host_floor_bytes=0 device=1 host=0'])
        for name in ('free_blocks', 'get_new_blocks', 'allocate_slots'):
            self.assertTrue(name in made.pool.__dict__ or name in made.scheduler.kv_cache_manager.__dict__, name)

    def test_a_flood_over_its_cap_evicts_its_own_blocks_and_not_the_other_tenants(self):
        made = self.rig(num_blocks=200, **{DEVICE_CAP: str(64 * BLOCK)})
        b = session('b-session', 'b', 32)
        made.run(b)
        floods = [session('a-flood-%d' % index, 'a', 64) for index in range(6)]
        for flood in floods:
            made.run(flood)
        self.assertEqual(made.cached_total(b), 32, 'b\'s whole session is still cached after a flood of 384 blocks through a 200-block pool')
        stats = made.registry.stats
        self.assertGreater(stats['quota_dev_evictions_own'], 0)
        self.assertEqual(stats['quota_dev_evictions_global'], 0, 'nothing was evicted by vLLM\'s own order: a\'s excess was always first')
        self.assertGreater(stats['quota_dev_demoted_blocks'], 0)
        resident = made.graft.quota.resident
        self.assertLessEqual(resident['salt-a'], 64 + 64, 'a never holds more than its cap beyond the request in service')

    def test_the_same_flood_without_the_quota_evicts_the_other_tenant(self):
        made = self.rig(num_blocks=200)
        b = session('b-session', 'b', 32)
        made.run(b)
        for index in range(6):
            made.run(session('a-flood-%d' % index, 'a', 64))
        self.assertLess(made.cached_total(b), 32, 'negative control: plain LRU lets the flood reach b\'s blocks')

    def test_the_floor_protects_a_tenant_from_a_flood_even_with_no_cap(self):
        made = self.rig(num_blocks=200, **{DEVICE_FLOOR: str(32 * BLOCK)})
        b = session('b-session', 'b', 32)
        made.run(b)
        for index in range(6):
            made.run(session('a-flood-%d' % index, 'a', 64))
        self.assertEqual(made.cached_total(b), 32, 'b sat at its floor: the flood evicted a\'s own idle blocks and the never-cached ones, not b\'s')
        stats = made.registry.stats
        self.assertGreater(stats['quota_dev_floor_protected'], 0)
        self.assertEqual(stats['quota_dev_floor_overrides'], 0)
        self.assertGreater(stats['quota_dev_evictions_global'], 0, 'with no cap a\'s blocks leave by vLLM\'s order, among the unprotected')

    def test_above_the_floor_a_tenants_blocks_are_fair_game_down_to_the_floor(self):
        made = self.rig(num_blocks=200, **{DEVICE_FLOOR: str(32 * BLOCK)})
        b = session('b-session', 'b', 64)
        made.run(b)
        for index in range(6):
            made.run(session('a-flood-%d' % index, 'a', 64))
        self.assertEqual(made.cached_total(b), 32, 'b held 64 blocks with a floor of 32: the flood took exactly the 32 above it')
        self.assertEqual(made.graft.quota.resident['salt-b'], 32)

    def test_when_the_floors_leave_nothing_the_allocation_still_happens_as_an_override(self):
        made = self.rig(num_blocks=96, **{DEVICE_FLOOR: str(32 * BLOCK)})
        made.run(session('b1', 'b', 32))
        made.run(session('c1', 'c', 32))
        # 32 free blocks are left, and 64 idle protected ones: a 64-block request must take 32 protected blocks
        made.run(session('a1', 'a', 64))
        stats = made.registry.stats
        self.assertGreater(stats['quota_dev_floor_overrides'], 0)
        self.assertEqual(made.pool.get_num_free_blocks(), 96 - 0, 'the request finished and freed everything')

    def test_the_allocating_tenant_may_take_its_own_blocks_below_its_floor(self):
        made = self.rig(num_blocks=96, **{DEVICE_FLOOR: str(32 * BLOCK)})
        a1 = session('a1', 'a', 32)
        made.run(a1)
        made.run(session('a2', 'a', 32))
        made.run(session('a3', 'a', 32))
        made.run(session('a4', 'a', 64))
        self.assertEqual(made.registry.stats['quota_dev_floor_overrides'], 0, 'a is the only tenant: nothing to protect it from')

    def test_a_block_touched_back_into_use_is_not_demoted_and_comes_back_when_freed(self):
        made = self.rig(num_blocks=300, **{DEVICE_CAP: str(64 * BLOCK)})
        a1 = session('a1', 'a', 32)
        made.run(a1)
        made.run(session('a2', 'a', 32))
        again = Request('a1-again', list(a1.all_token_ids) + tokens(2048, 99), salt=salt_of('a'))
        made.run(again, finish=False)
        made.run(session('a3', 'a', 64))
        made.finish(again)
        quota = made.graft.quota
        self.assertEqual(quota.resident['salt-a'], sum(1 for owner in quota.owner.values() if owner == 'salt-a'))
        self.assertTrue(all(block.ref_cnt >= 0 for block in made.pool.blocks))
        queue = made.pool.free_block_queue
        self.assertEqual(len(queue.order()), queue.num_free_blocks, 'the free queue stayed consistent')

    def test_reset_prefix_cache_forgets_the_owners(self):
        made = self.rig(num_blocks=200, **{DEVICE_CAP: str(64 * BLOCK)})
        made.run(session('a1', 'a', 32))
        self.assertTrue(made.graft.quota.owner)
        self.assertTrue(made.scheduler.reset_prefix_cache())
        self.assertEqual((made.graft.quota.owner, made.graft.quota.resident), ({}, {}))

    def test_the_counters_are_aggregates_and_name_no_tenant(self):
        made = self.rig(num_blocks=200, **{DEVICE_CAP: str(64 * BLOCK), DEVICE_FLOOR: str(32 * BLOCK)})
        made.run(session('b1', 'b', 32))
        for index in range(4):
            made.run(session('a-flood-%d' % index, 'a', 64))
        snapshot = made.registry.snapshot()
        self.assertEqual(snapshot['quota_dev_tenants'], 2)
        self.assertEqual(snapshot['quota_dev_at_cap'], 1)
        text = json.dumps(snapshot, sort_keys=True) + '\n'.join(made.lines)
        self.assertNotIn('salt-a', text)
        self.assertNotIn('salt-b', text)
        keys = [name for name in snapshot if name.startswith('quota_')]
        self.assertFalse([name for name in keys if 'salt' in name or 'tenant' in name and name not in ('quota_dev_tenants', 'quota_host_tenants')])

    def test_numbers_the_pool_cannot_honour_are_refused_at_install(self):
        for environ, needle in (({DEVICE_FLOOR: str(96 * BLOCK)}, 'more than half'), ({DEVICE_CAP: str(300 * BLOCK)}, 'more than the whole')):
            with self.subTest(environ=environ):
                with self.assertRaises(patch_module.PrefixInstallError) as caught:
                    self.rig(num_blocks=100, **environ)
                self.assertIn(needle, str(caught.exception))

    def test_an_install_without_the_free_queue_is_refused(self):
        patches = [mock.patch.dict(sys.modules, fake_vllm_modules()), mock.patch.dict(os.environ, {'QWEN_PREFIX_STATS_PATH': ''})]
        for patcher in patches:
            patcher.start()
            self.addCleanup(patcher.stop)
        scheduler = FakeScheduler(num_blocks=100)           # the reduced pool: no free queue, no prepend_n
        registry = PrefixRegistry(budget_bytes=1 << 40, environ={DEVICE_CAP: str(64 * BLOCK)})
        with self.assertRaises(patch_module.PrefixInstallError) as caught:
            patch_module.install(scheduler, registry=registry, kill_switch_path=None, logger=lambda *a: None, stats=None)
        self.assertIn('block_pool.free_block_queue', str(caught.exception))
        self.assertIn('allocate_slots', str(caught.exception))

    def test_a_random_workload_keeps_the_invariants_and_the_floor(self):
        for seed in range(5):
            rng = random.Random(seed)
            made = self.rig(num_blocks=160, **{DEVICE_CAP: str(64 * BLOCK), DEVICE_FLOOR: str(32 * BLOCK)})
            self.addCleanup(lambda: None)
            running = []
            tenants = ['a', 'b', 'c']
            for step in range(120):
                if running and rng.random() < 0.5:
                    made.finish(running.pop(rng.randrange(len(running))))
                    continue
                request = session('r%d-%d' % (seed, step), rng.choice(tenants), rng.choice([32, 32, 64]), seed=seed * 1000 + step + 7)
                made.scheduler.add(request)
                out = made.scheduler.schedule()
                if out.scheduled_new_reqs:
                    running.append(request)
                else:
                    made.scheduler.waiting.remove(request)
                quota = made.graft.quota
                queue = made.pool.free_block_queue
                self.assertEqual(len(queue.order()), queue.num_free_blocks)
                self.assertEqual(sum(quota.resident.values()), len(quota.owner))
                cached = {block.block_id for block in made.pool.blocks if block.block_hash is not None}
                self.assertEqual(cached, set(quota.owner), 'the owner map is exactly the cached blocks (seed %d step %d)' % (seed, step))
            for request in running:
                made.finish(request)
            made.close()
            made.patches = []


# -- (3b) the same quota on vLLM's own BlockPool and FreeKVCacheBlockQueue (skipped where vLLM is not importable) -----------------------------------

try:
    from vllm.v1.core.block_pool import BlockPool as RealBlockPool
    from vllm.v1.core.kv_cache_utils import make_block_hash_with_group_id as real_hash
except Exception:        # noqa: BLE001 - any import trouble means "no vLLM here"
    RealBlockPool = None


@unittest.skipIf(RealBlockPool is None, 'vLLM is not importable here (the 3.11 CPU suite); the vLLM workflow and the serving image run this class')
class RealPoolTests(unittest.TestCase):
    """DeviceQuota's calls on the real classes: the free queue's prepend_n / remove / append and the pool's free_blocks / get_new_blocks, as the graft wires them."""

    def setUp(self):
        self.pool = RealBlockPool(64, True, BLOCK)
        self.stats = dict.fromkeys(registry_module.QUOTA_STAT_NAMES, 0)
        graft = SimpleNamespace(registry=SimpleNamespace(stats=self.stats), block_pool=self.pool)
        config = SimpleNamespace(device_cap_blocks=16, device_floor_blocks=8)       # (below the parser's 2048-token minimum: the pool here is 64 blocks)
        self.quota = patch_module.DeviceQuota(graft, config)
        original = self.pool._maybe_evict_cached_block

        def evict(block):
            evicted = original(block)
            if evicted:
                self.quota.on_evict(block)
            return evicted

        self.pool._maybe_evict_cached_block = evict
        self.serial = 0

    def publish(self, tenant, count):
        blocks = self.pool.get_new_blocks(count)
        for block in blocks:
            self.serial += 1
            key = real_hash(hashlib.sha256(b'%d' % self.serial).digest(), 0)
            block.block_hash = key
            self.pool.cached_block_hash_to_block.insert(key, block)
        self.quota.note_published(blocks, tenant)
        return blocks

    def release(self, blocks):
        ordered = list(reversed(blocks))        # vLLM frees a request's blocks tail first: the last block is the first to leave
        self.pool.free_blocks(ordered)
        self.quota.on_free(ordered)

    def test_the_quota_calls_exist_on_the_real_classes(self):
        for name in patch_module.QUOTA_POOL_INTERNALS:
            self.assertTrue(hasattr(self.pool, name), name)
        for name in patch_module.QUOTA_QUEUE_INTERNALS:
            self.assertTrue(hasattr(self.pool.free_block_queue, name), name)

    def test_the_excess_is_evicted_first_oldest_first_and_the_queue_stays_whole(self):
        b = self.publish('b', 8)
        self.release(b)
        a = self.publish('a', 12)
        self.release(a)
        a2 = self.publish('a', 12)        # a now holds 24 > its cap of 16 (12 idle + 12 in use)
        self.assertEqual(self.stats['quota_dev_demoted_blocks'], 8, 'the 8 oldest of a\'s idle blocks were put at the front')
        front = self.pool.free_block_queue.fake_free_list_head.next_free_block
        self.assertEqual(front.block_id, a[11].block_id, 'oldest first: the block freed first is the one at the very front')
        taken = self.pool.get_new_blocks(8)        # the allocation that follows takes them, not b's
        self.assertEqual(sorted(block.block_id for block in taken), sorted(block.block_id for block in a[4:]))
        self.assertEqual(self.stats['quota_dev_evictions_own'], 8)
        self.assertEqual(self.stats['quota_dev_evictions_global'], 0)
        self.assertEqual(self.quota.resident['b'], 8)
        queue = self.pool.free_block_queue
        count, block = 0, queue.fake_free_list_head.next_free_block
        while block is not queue.fake_free_list_tail:
            count += 1
            block = block.next_free_block
        self.assertEqual(count, queue.num_free_blocks)
        self.assertEqual(a2[0].ref_cnt, 1)

    def test_the_floor_sets_protected_blocks_aside_until_nothing_else_is_left(self):
        self.quota.cap = 0                   # the floor alone (the cap would put c's excess at the front, ahead of b)
        b = self.publish('b', 8)             # b sits exactly at its floor
        c = self.publish('c', 55)            # ... and c takes the rest of the pool (63 usable blocks: vLLM keeps the null block)
        self.release(b)
        self.release(c)                      # the queue: b's 8 (oldest), then c's 55
        self.quota.allocating = 'a'
        self.quota.protect(30)
        self.assertEqual(self.stats['quota_dev_floor_protected'], 8, 'b\'s 8 blocks were set aside at the tail')
        taken = self.pool.get_new_blocks(30)
        self.assertFalse({block.block_id for block in taken} & {block.block_id for block in b}, 'b\'s blocks were not among the 30')
        self.assertTrue({block.block_id for block in taken} <= {block.block_id for block in c}, 'all 30 were c\'s, oldest first')
        self.assertEqual(self.stats['quota_dev_floor_overrides'], 0)
        self.release(taken)
        self.quota.protect(63)               # the 30 uncached, 17 of c's 25 (c is held to its floor of 8 too), and nothing else: 16 short
        self.assertEqual(self.stats['quota_dev_floor_overrides'], 16)
        self.assertEqual(self.stats['quota_dev_floor_protected'], 8 + 16)


# -- (4) the cold-tier client interface ----------------------------------------------------------------------------------

class ColdTierTests(unittest.TestCase):
    LAYOUT = 'layout-1'

    def key(self, tenant='t1', block=b'h1', layout=None):
        return cold.ColdKey(tenant, layout or self.LAYOUT, block)

    def test_a_store_then_a_fetch_writes_into_the_callers_buffer(self):
        tier = cold.InMemoryColdTier(self.LAYOUT)
        data = bytes(range(64))
        self.assertEqual(tier.store(self.key(), data).result(timeout=5), len(data))
        buffer = bytearray(64)
        self.assertEqual(tier.fetch(self.key(), memoryview(buffer)).result(timeout=5), 64)
        self.assertEqual(bytes(buffer), data)

    def test_the_key_carries_the_tenant_and_the_layout(self):
        tier = cold.InMemoryColdTier(self.LAYOUT)
        tier.store(self.key('t1'), b'x' * 8).result(timeout=5)
        self.assertEqual(tier.fetch(self.key('t2'), memoryview(bytearray(8))).result(timeout=5), 0, 'another tenant cannot name the block')
        with self.assertRaises(cold.ColdLayoutError):
            tier.fetch(self.key(layout='layout-2'), memoryview(bytearray(8))).result(timeout=5)
        with self.assertRaises(cold.ColdLayoutError):
            tier.store(self.key(layout='layout-2'), b'x').result(timeout=5)

    def test_a_missing_block_is_zero_bytes_and_a_small_buffer_is_an_error(self):
        tier = cold.InMemoryColdTier(self.LAYOUT)
        self.assertEqual(tier.fetch(self.key(), memoryview(bytearray(8))).result(timeout=5), 0)
        tier.store(self.key(), b'y' * 16).result(timeout=5)
        with self.assertRaises(cold.ColdBufferError):
            tier.fetch(self.key(), memoryview(bytearray(8))).result(timeout=5)

    def test_it_is_asynchronous_and_a_failure_is_the_futures(self):
        gate_open = []
        tier = cold.InMemoryColdTier(self.LAYOUT, delay=lambda: gate_open.append(1))
        future = tier.store(self.key(), b'z' * 4)
        self.assertEqual(future.result(timeout=5), 4)
        self.assertEqual(gate_open, [1])
        failing = cold.InMemoryColdTier(self.LAYOUT, fail=lambda operation, key: operation == 'fetch')
        failing.store(self.key(), b'q' * 4).result(timeout=5)
        with self.assertRaises(cold.ColdTierError):
            failing.fetch(self.key(), memoryview(bytearray(4))).result(timeout=5)

    def test_the_interface_is_abstract_and_the_fake_fills_it(self):
        with self.assertRaises(TypeError):
            cold.ColdTierClient()
        self.assertTrue(issubclass(cold.InMemoryColdTier, cold.ColdTierClient))
        tier = cold.InMemoryColdTier(self.LAYOUT)
        for name in ('fetch', 'store', 'contains', 'close'):
            self.assertTrue(callable(getattr(tier, name)), name)

    def test_the_key_text_names_no_salt_and_nothing_platform_specific(self):
        text = repr(self.key('tenant-tag'))
        self.assertIn('tenant-tag', text)       # the tenant id is the caller's opaque tag: it is the key, and the key is not a log line
        with open(os.path.join(HERE, 'cold_tier_client.py'), encoding='utf-8') as handle:
            source = handle.read().lower()
        for word in ('kubernetes', 'k8s', 'nvme', 'ghcr', 'docker', ' pod'):
            self.assertNotIn(word, source.replace('ndarray', ''), word)


if __name__ == '__main__':
    unittest.main()
