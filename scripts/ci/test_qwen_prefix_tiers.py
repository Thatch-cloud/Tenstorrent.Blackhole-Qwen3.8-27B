"""The checkpoint registry's store hygiene (WP1) and telemetry (WP0): sizes, the fair policy, supersession, the admission
classes, the reuse distance, the periodic lines.

Pure python, no vLLM: runs in the 3.11 CPU suite. The registry is driven directly (grants staged and committed by hand) and
through the scheduler graft on the fakes of test_qwen_prefix_scheduler_patch, whose FakeGdnModel restores every checkpoint
against the cold chain of the row's own tokens - so every scenario below also proves that what a store policy leaves in the
registry still serves byte-identical state.

Three properties are held throughout:
  * flag-off identity - with QWEN_PREFIX_EVICT unset (lru), QWEN_PREFIX_SUPERSEDE unset (0) and the telemetry on or off, the
    registry makes the same grants, trims, captures and evictions as the plain LRU it was (a reference model below, a random
    sequence of operations, and the graft driven twice);
  * exact or absent - a retired or evicted checkpoint is a miss, never a different state;
  * the telemetry only reads - it changes no counter the registry had before it.
"""

import hashlib
import os
import random
import re
import shutil
import sys
import tempfile
import unittest
from array import array
from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import levern_policy  # noqa: E402
import prefix_judge  # noqa: E402
import qwen_prefix_metrics as metrics  # noqa: E402
import qwen_prefix_registry as registry_module  # noqa: E402
import qwen_prefix_scheduler_patch as patch_module  # noqa: E402
import serving_c2_contract as contract  # noqa: E402
from qwen_prefix_registry import (  # noqa: E402
    BLOCK, CHUNK, CHECKPOINT_NBYTES, CHECKPOINT_NBYTES_BF16, CHECKPOINT_NBYTES_FP32, CLASSES, Grant, PrefixRegistry,
    StatsExport, checkpoint_nbytes, host_gauges, tenant_of_salt)
from test_qwen_prefix_scheduler_patch import (  # noqa: E402
    FakeGdnModel, FakeScheduler, Quiet, Request, fake_vllm_modules, tokens as make_tokens)

BF16 = {'QWEN35_GDN_STATE_BF16': '1'}
LEGACY_STATS = list(registry_module.LEGACY_STAT_NAMES)


def salt_of(tag):
    return 'qps1.%s.%s' % (tag, 'f' * 64)


def span(count, seed):
    return make_tokens(count, seed)


def keys_of(request):
    return request.block_hashes


class Clock(object):
    def __init__(self, now=1000.0):
        self.now = now

    def __call__(self):
        return self.now


# ------------------------------------------------------------------------------------------------
# WP1: the sizes
# ------------------------------------------------------------------------------------------------
class SizeTests(unittest.TestCase):
    def test_the_two_dtypes_by_independent_arithmetic(self):
        """48 GDN layers x (value-head state [48, 128, 128] + conv carry [3, 10240] in bf16) over the whole mesh."""
        fp32 = 48 * (48 * 128 * 128 * 4 + 3 * 10240 * 2)
        bf16 = 48 * (48 * 128 * 128 * 2 + 3 * 10240 * 2)
        self.assertEqual((fp32, bf16), (153944064, 78446592))
        self.assertEqual((CHECKPOINT_NBYTES_FP32, CHECKPOINT_NBYTES_BF16), (fp32, bf16))
        self.assertEqual(checkpoint_nbytes({}), fp32)
        self.assertEqual(checkpoint_nbytes(BF16), bf16)
        self.assertEqual(checkpoint_nbytes(state_bf16=True), bf16)
        self.assertEqual(checkpoint_nbytes(state_bf16=False), fp32)

    def test_only_exactly_one_selects_bf16_as_the_gdn_layer_reads_it(self):
        for value in (None, '', '0', 'true', 'yes', ' 1', '10'):
            environ = {} if value is None else {'QWEN35_GDN_STATE_BF16': value}
            self.assertEqual(checkpoint_nbytes(environ), CHECKPOINT_NBYTES_FP32, repr(value))
        self.assertTrue(registry_module.gdn_state_bf16(BF16))

    def test_it_does_not_depend_on_how_many_chips_share_the_state(self):
        """What the model reads (rec_state [chips, heads per chip, 128, 128] and conv_carry [chips, 3, channels per chip], summed
        over layers) is the same bytes at two chips of 24 heads and 5120 channels and at four of 12 and 2560."""
        for chips, heads, channels in ((2, 24, 5120), (4, 12, 2560)):
            for itemsize, expected in ((4, CHECKPOINT_NBYTES_FP32), (2, CHECKPOINT_NBYTES_BF16)):
                read = 48 * (chips * heads * 128 * 128 * itemsize + chips * 3 * channels * 2)
                self.assertEqual(read, expected, (chips, heads, itemsize))

    def test_a_capture_without_a_size_is_charged_the_dtype_s(self):
        for environ, expected in (({}, CHECKPOINT_NBYTES_FP32), (BF16, CHECKPOINT_NBYTES_BF16)):
            registry = PrefixRegistry(budget_bytes=1 << 40, environ=dict(environ))
            self.assertEqual(registry.checkpoint_nbytes, expected)
            stored = capture_once(registry, nbytes=None)
            self.assertEqual((stored.nbytes, registry.bytes), (expected, expected))

    def test_the_process_constant_is_its_environment_s(self):
        self.assertEqual(CHECKPOINT_NBYTES, checkpoint_nbytes(os.environ))
        self.assertEqual(levern_policy.PARK_NBYTES, CHECKPOINT_NBYTES)

    def test_a_default_store_holds_twice_the_checkpoints_with_the_bf16_state(self):
        self.assertEqual(registry_module.DEFAULT_STORE_GIB, 8.0)
        self.assertEqual(prefix_judge.store_entries(8.0), 55)
        self.assertEqual(prefix_judge.store_entries(8.0, state_bf16=True), 109)
        self.assertEqual(prefix_judge.store_entries(0.5), 3)
        self.assertEqual(prefix_judge.store_entries(0.5, state_bf16=True), 6)
        self.assertEqual(prefix_judge.Oracle().capacity, 55)
        self.assertEqual(prefix_judge.Oracle(state_bf16=True).capacity, 109)
        self.assertEqual(prefix_judge.Oracle(capacity=7, state_bf16=True).capacity, 7)

    def test_the_standalone_mirrors_agree_with_the_registry(self):
        for bf16 in (False, True):
            want = checkpoint_nbytes(state_bf16=bf16)
            self.assertEqual(levern_policy.park_nbytes(state_bf16=bf16), want)
            self.assertEqual(prefix_judge.checkpoint_nbytes(bf16), want)
        self.assertEqual(levern_policy.park_nbytes({}), CHECKPOINT_NBYTES_FP32)
        self.assertEqual(levern_policy.park_nbytes(BF16), CHECKPOINT_NBYTES_BF16)
        self.assertEqual(prefix_judge.CHECKPOINT_NBYTES, CHECKPOINT_NBYTES_FP32)

    def test_lever_n_guards_the_host_with_the_size_it_really_parks(self):
        levern_policy.reset_state()
        self.addCleanup(levern_policy.reset_state)
        self.assertEqual(levern_policy.observed_park_nbytes({}), CHECKPOINT_NBYTES_FP32)
        self.assertEqual(levern_policy.observed_park_nbytes(BF16), CHECKPOINT_NBYTES_BF16)
        for bad in (None, 0, -5, 'x'):
            levern_policy.note_park_nbytes(bad)
        self.assertEqual(levern_policy.observed_park_nbytes(BF16), CHECKPOINT_NBYTES_BF16)
        levern_policy.note_park_nbytes(12345)
        self.assertEqual(levern_policy.observed_park_nbytes({}), 12345, 'the last scratch the route really read')
        levern_policy.reset_state()
        self.assertEqual(levern_policy.observed_park_nbytes({}), CHECKPOINT_NBYTES_FP32)

    def test_may_preempt_asks_for_the_observed_size(self):
        import levern_route
        import levern_scheduler

        levern_policy.reset_state()
        self.addCleanup(levern_policy.reset_state)
        env = {'QWEN_FAST_LEVER_N': '1', 'QWEN_PREFIX_REUSE': '1', 'QWEN_FAST_STICKY_SESSIONS': '1', 'QWEN_FAST_LEVERN_ROUNDS': '1',
               'QWEN_FAST_LEVERN_TTFT_TARGET_S': '0', 'QWEN_FAST_LEVERN_PARK': 'host', 'QWEN_FAST_LEVERN_SHORT_TOKENS': '16384'}
        runtime = levern_scheduler.LevernRuntime(levern_policy.config(env), log=mock.Mock(), merged=levern_policy.merged_config(env))
        scheduler = SimpleNamespace(max_num_running_reqs=8)
        active = SimpleNamespace(request_id='long')
        asked = []
        with mock.patch.object(levern_route, 'host_memory_ok', side_effect=lambda nbytes, environ=None: asked.append(nbytes) or True):
            self.assertTrue(runtime.may_preempt(scheduler, [], active, 'short', 0))
            levern_policy.note_park_nbytes(CHECKPOINT_NBYTES_BF16)
            self.assertTrue(runtime.may_preempt(scheduler, [], active, 'short', 0))
        self.assertEqual(asked, [levern_policy.park_nbytes(), CHECKPOINT_NBYTES_BF16])

    def test_the_park_route_reports_what_it_read(self):
        import levern_route

        source = Path(levern_route.__file__).read_text(encoding='utf-8')
        self.assertIn('levern_policy.note_park_nbytes(nbytes)', source)

    def test_no_document_still_says_154_mb_for_the_production_checkpoint(self):
        for name in ('qwen_prefix_registry.py', 'levern_policy.py', 'levern_route.py', 'prefix_judge.py'):
            self.assertNotIn('154 MB', (HERE / name).read_text(encoding='utf-8'), name)


def capture_once(registry, nbytes=10, size=CHUNK, tenant=None, chain=None, gap=None, name='a', salt='s'):
    request = Request(name, span(size + 100, size + len(name)), salt=salt)
    key = request.block_hashes[size // BLOCK - 1]
    registry.begin_step()
    registry.stage(Grant(name, 0, 0, None, None, [(size, key)], request, chain=chain, tenant=tenant, gap=gap))
    registry.commit({name: 0})
    return registry.capture(name, size, rec='r', carry='c', nbytes=nbytes, loop_pos=size)


# ------------------------------------------------------------------------------------------------
# WP1: the flags
# ------------------------------------------------------------------------------------------------
class FlagTests(unittest.TestCase):
    def test_defaults(self):
        registry = PrefixRegistry(budget_bytes=1, environ={})
        self.assertEqual((registry.evict_policy, registry.supersede, registry.telemetry), ('lru', False, True))
        snap = registry.snapshot()
        self.assertEqual((snap['evict_fair'], snap['supersede'], snap['telemetry']), (0, 0, 1))

    def test_each_flag_is_strict(self):
        for name, bad in (('QWEN_PREFIX_EVICT', 'Fair'), ('QWEN_PREFIX_EVICT', 'lfu'), ('QWEN_PREFIX_EVICT', '1'),
                          ('QWEN_PREFIX_SUPERSEDE', 'true'), ('QWEN_PREFIX_SUPERSEDE', '2'), ('QWEN_PREFIX_TELEMETRY', 'yes'),
                          ('QWEN_PREFIX_GHOST_ENTRIES', '-1'), ('QWEN_PREFIX_GHOST_ENTRIES', 'many')):
            with self.assertRaises(ValueError, msg='%s=%s' % (name, bad)):
                PrefixRegistry(budget_bytes=1, environ={name: bad})
        registry = PrefixRegistry(budget_bytes=1, environ={'QWEN_PREFIX_EVICT': 'fair', 'QWEN_PREFIX_SUPERSEDE': '1',
                                                           'QWEN_PREFIX_TELEMETRY': '0', 'QWEN_PREFIX_GHOST_ENTRIES': '7'})
        self.assertEqual((registry.evict_policy, registry.supersede, registry.telemetry, registry.ghost_limit), ('fair', True, False, 0))
        self.assertEqual(PrefixRegistry(budget_bytes=1, environ={'QWEN_PREFIX_GHOST_ENTRIES': '7'}).ghost_limit, 7)
        self.assertEqual(PrefixRegistry(budget_bytes=1, environ={'QWEN_PREFIX_EVICT': '', 'QWEN_PREFIX_SUPERSEDE': ''}).evict_policy, 'lru')

    def test_the_contract_polices_the_profile(self):
        def problems(**env):
            return contract.prefix_policy_problems({'env': dict({'QWEN_PREFIX_REUSE': '1'}, **env)})

        self.assertEqual(problems(), [])
        self.assertEqual(problems(QWEN_PREFIX_EVICT='fair', QWEN_PREFIX_SUPERSEDE='1', QWEN_PREFIX_TELEMETRY='0',
                                  QWEN_PREFIX_GHOST_ENTRIES='1024'), [])
        self.assertTrue(any('QWEN_PREFIX_EVICT' in p for p in problems(QWEN_PREFIX_EVICT='mru')))
        self.assertTrue(any('QWEN_PREFIX_SUPERSEDE' in p for p in problems(QWEN_PREFIX_SUPERSEDE='yes')))
        self.assertTrue(any('QWEN_PREFIX_GHOST_ENTRIES' in p for p in problems(QWEN_PREFIX_GHOST_ENTRIES='-3')))
        inert = contract.prefix_policy_problems({'env': {'QWEN_PREFIX_EVICT': 'fair'}})
        self.assertEqual(len(inert), 1)
        self.assertIn('inert', inert[0])
        self.assertEqual(contract.prefix_policy_problems({'env': {}}), [])
        self.assertEqual(contract.prefix_policy_problems({}), [])
        # prefix_reuse_problems reports them beside the engine-flag problems, with or without the switch
        self.assertTrue(any('QWEN_PREFIX_EVICT' in p for p in contract.prefix_reuse_problems(
            {'env': {'QWEN_PREFIX_EVICT': 'fair'}})))

    def test_the_policy_names_are_the_registry_s(self):
        self.assertEqual(sorted(contract.PREFIX_POLICY_FLAGS),
                         sorted([registry_module.ENV_EVICT, registry_module.ENV_SUPERSEDE, registry_module.ENV_TELEMETRY]))
        self.assertEqual(contract.PREFIX_GHOST_FLAG, registry_module.ENV_GHOST)
        self.assertEqual(contract.PREFIX_POLICY_FLAGS[registry_module.ENV_EVICT], registry_module.EVICT_POLICIES)

    def test_the_tenant_is_the_salt_s(self):
        self.assertEqual(tenant_of_salt('qps1.tenantTAG1.' + 'a' * 64), 'tenantTAG1')
        self.assertEqual(tenant_of_salt('plain-salt'), 'plain-salt')
        self.assertEqual(tenant_of_salt('qps1..mac'), 'qps1..mac')
        self.assertEqual(tenant_of_salt(None), '')
        self.assertEqual(tenant_of_salt(''), '')


# ------------------------------------------------------------------------------------------------
# WP1: flag-off identity - the plain LRU
# ------------------------------------------------------------------------------------------------
class ReferenceLru(object):
    """The registry's eviction as it was before the policies: keys in LRU order, bytes, pinned entries skipped."""

    def __init__(self, budget):
        self.budget = budget
        self.order = OrderedDict()
        self.evicted = 0
        self.skipped = 0

    def put(self, key, nbytes, pinned_keys):
        if nbytes > self.budget:
            self.skipped += 1
            return
        if key in self.order:
            if key in pinned_keys:
                self.order.move_to_end(key)
                return
            del self.order[key]
        self.order[key] = nbytes
        total = sum(self.order.values())
        for victim in list(self.order):
            if total <= self.budget:
                break
            if victim in pinned_keys:
                continue
            total -= self.order.pop(victim)
            self.evicted += 1

    def touch(self, key):
        if key in self.order:
            self.order.move_to_end(key)

    def drop(self, key):
        self.order.pop(key, None)


class FlagOffIdentityTests(unittest.TestCase):
    def run_ops(self, seed, environ, steps=400):
        rng = random.Random(seed)
        registry = PrefixRegistry(budget_bytes=60, environ=environ)
        model = ReferenceLru(60)
        ids = list(range(CHUNK))
        for step in range(steps):
            op = rng.random()
            key = b'k%d' % rng.randrange(14)
            pinned = {k for k, e in registry.entries.items() if e.pins}
            if op < 0.55:
                size = rng.choice([5, 10, 10, 20, 40, 61])
                tenant = rng.choice(['a', 'b', None])
                registry.put(key, CHUNK, ids, nbytes=size, chain=b'c%d' % rng.randrange(3), tenant=tenant)
                model.put(key, size, pinned)
            elif op < 0.7:
                registry.touch(key)
                model.touch(key)
            elif op < 0.8:
                registry.drop(key, 'coupled')
                model.drop(key)
            elif op < 0.9:
                if key in registry.entries:
                    registry.entries[key].pins = rng.choice([0, 1])
            self.assertEqual(list(registry.entries), list(model.order), 'seed %d step %d' % (seed, step))
            self.assertEqual(registry.bytes, sum(model.order.values()))
        self.assertEqual(registry.stats['evicted_lru'], model.evicted)
        self.assertEqual(registry.stats['capture_skipped_budget'], model.skipped)
        self.assertEqual((registry.stats['evicted_fair'], registry.stats['evicted_superseded']), (0, 0))
        return registry

    def test_unset_is_the_plain_lru_on_a_random_sequence(self):
        for seed in range(8):
            self.run_ops(seed, {})

    def test_the_explicit_defaults_are_the_same_lru(self):
        for seed in range(4):
            self.run_ops(seed, {'QWEN_PREFIX_EVICT': 'lru', 'QWEN_PREFIX_SUPERSEDE': '0', 'QWEN_PREFIX_TELEMETRY': '0'})

    def test_the_indexes_follow_the_entries(self):
        registry = self.run_ops(3, {})
        by_chain = {}
        held = {}
        for key, entry in registry.entries.items():
            by_chain.setdefault(entry.chain, set()).add(key)
            held[entry.tenant] = held.get(entry.tenant, 0) + entry.nbytes
        self.assertEqual(registry.by_chain, by_chain)
        self.assertEqual(registry.tenant_bytes, held)


# ------------------------------------------------------------------------------------------------
# WP1: the fair policy
# ------------------------------------------------------------------------------------------------
class FairPolicyTests(unittest.TestCase):
    def registry(self, budget, **extra):
        return PrefixRegistry(budget_bytes=budget, environ=dict({'QWEN_PREFIX_EVICT': 'fair'}, **extra))

    def put(self, registry, key, tenant, nbytes=10):
        return registry.put(key, CHUNK, list(range(CHUNK)), nbytes=nbytes, tenant=tenant)

    def test_the_biggest_holder_pays_not_the_global_oldest(self):
        registry = self.registry(50)
        self.put(registry, b'b1', 'quiet')
        for index in range(4):
            self.put(registry, b'a%d' % index, 'loud')
        self.assertEqual(list(registry.entries), [b'b1', b'a0', b'a1', b'a2', b'a3'])
        self.put(registry, b'a4', 'loud')
        self.assertIn(b'b1', registry.entries, 'the quiet tenant\'s oldest entry stays although it is the global oldest')
        self.assertNotIn(b'a0', registry.entries, 'the loud tenant\'s oldest goes')
        self.assertEqual((registry.stats['evicted_fair'], registry.stats['evicted_lru']), (1, 0))

    def test_plain_lru_would_have_evicted_the_quiet_tenant(self):
        registry = PrefixRegistry(budget_bytes=50, environ={})
        self.put(registry, b'b1', 'quiet')
        for index in range(5):
            self.put(registry, b'a%d' % index, 'loud')
        self.assertNotIn(b'b1', registry.entries)

    def test_a_flood_does_not_push_an_idle_tenant_out(self):
        """The plan's simulation at small scale: tenant B holds two checkpoints and goes idle; tenant A turns over many."""
        for policy, survivors in (('lru', 0), ('fair', 2)):
            registry = PrefixRegistry(budget_bytes=60, environ={'QWEN_PREFIX_EVICT': policy})
            self.put(registry, b'b1', 'B')
            self.put(registry, b'b2', 'B')
            for index in range(20):
                self.put(registry, b'a%d' % index, 'A')
            self.assertEqual(len([k for k in registry.entries if k.startswith(b'b')]), survivors, policy)
            self.assertLessEqual(registry.bytes, 60)

    def test_by_bytes_not_by_count(self):
        registry = self.registry(100)
        self.put(registry, b'big', 'x', nbytes=60)
        self.put(registry, b's1', 'y', nbytes=10)
        self.put(registry, b's2', 'y', nbytes=10)
        self.put(registry, b's3', 'y', nbytes=10)
        self.put(registry, b's4', 'y', nbytes=10)
        self.put(registry, b's5', 'y', nbytes=10)
        self.assertNotIn(b'big', registry.entries, 'x holds 60 bytes, y 50: x pays although y holds more entries')
        self.assertEqual(sorted(registry.entries), [b's1', b's2', b's3', b's4', b's5'])

    def test_a_tie_goes_to_the_tenant_with_the_older_candidate(self):
        registry = self.registry(40)
        self.put(registry, b'x1', 'x')
        self.put(registry, b'y1', 'y')
        self.put(registry, b'x2', 'x')
        self.put(registry, b'y2', 'y')
        self.put(registry, b'z', 'z', nbytes=1)
        self.assertNotIn(b'x1', registry.entries)
        self.assertEqual(list(registry.entries), [b'y1', b'x2', b'y2', b'z'])

    def test_pinned_checkpoints_never_go_and_the_new_one_is_spared(self):
        registry = self.registry(30)
        pinned = self.put(registry, b'p', 'loud')
        pinned.pins = 1
        self.put(registry, b'q', 'loud')
        self.put(registry, b'r', 'quiet')
        self.put(registry, b's', 'loud')
        self.assertIn(b'p', registry.entries)
        self.assertNotIn(b'q', registry.entries, 'the loud tenant\'s oldest unpinned entry goes')
        self.assertIn(b's', registry.entries, 'just stored: spared while another tenant\'s older entry could go')
        self.assertLessEqual(registry.bytes, 30)

    def test_everything_pinned_leaves_the_store_over_budget_as_the_lru_does(self):
        for policy in ('lru', 'fair'):
            registry = PrefixRegistry(budget_bytes=1 << 20, environ={'QWEN_PREFIX_EVICT': policy})
            for key in (b'a', b'b'):
                self.put(registry, key, 't').pins = 1
            registry.budget_bytes = 15
            registry.begin_step()
            self.assertEqual(sorted(registry.entries), [b'a', b'b'], policy)
            self.assertEqual(registry.bytes, 20)
            registry.entries[b'a'].pins = 0
            registry.begin_step()
            self.assertEqual(sorted(registry.entries), [b'b'], policy)

    def test_the_new_checkpoint_goes_last_when_nothing_else_can(self):
        registry = self.registry(15)
        self.put(registry, b'p', 't').pins = 1
        self.assertIsNotNone(self.put(registry, b'n', 't'))
        self.assertEqual(list(registry.entries), [b'p'], 'only pinned data is safe: the newcomer is the victim, as under the lru')

    def test_fair_eviction_is_counted_apart_and_remembered(self):
        registry = self.registry(20)
        self.put(registry, b'a', 'x')
        self.put(registry, b'b', 'x')
        self.put(registry, b'c', 'x')
        self.assertEqual((registry.stats['evicted_fair'], registry.stats['evicted_lru']), (1, 0))

    def test_a_budget_change_by_begin_step_is_enforced_fairly_too(self):
        registry = self.registry(1 << 20)
        for key, tenant in ((b'a1', 'a'), (b'b1', 'b'), (b'a2', 'a')):
            self.put(registry, key, tenant)
        registry.budget_bytes = 20
        registry.begin_step()
        self.assertEqual(sorted(registry.entries), [b'a2', b'b1'])


# ------------------------------------------------------------------------------------------------
# WP1: supersession
# ------------------------------------------------------------------------------------------------
class Conversation(object):
    """One session's tokens and the requests it sends: a shared system block, then its own text."""

    def __init__(self, tag, seed, system, salt=None):
        self.salt = salt or salt_of(tag)
        self.seed = seed
        self.history = list(system)
        self.sent = []

    def prompt(self, extra, label):
        return self.history + span(extra, self.seed * 1000 + len(self.sent) * 17 + len(label))

    def turn(self, extra, label=None, base=None):
        label = label or 't%d' % len(self.sent)
        tokens = (base if base is not None else self.history) + span(extra, self.seed * 1000 + len(self.sent) * 17 + len(label))
        request = Request('%s-%s' % (self.salt[5:13], label), tokens, salt=self.salt)
        self.sent.append(request)
        return request


class SupersessionHarness(object):
    """Drive a registry the way the scheduler and the model do, per request: stage and commit the grant, take the captures."""

    def __init__(self, supersede=True, **extra):
        environ = dict({'QWEN_PREFIX_SUPERSEDE': '1' if supersede else '0'}, **extra)
        self.registry = PrefixRegistry(budget_bytes=1 << 40, environ=environ)

    def serve(self, request, boundaries, gap=None, q=None):
        """Admit `request` with a grant at q (the best resident ancestor when None) and capture at each boundary."""
        registry = self.registry
        hashes = request.block_hashes
        entry = None
        if q is None:
            for pos in range((len(request.all_token_ids) // CHUNK) * CHUNK, 0, -CHUNK):
                found = registry.get(hashes[pos // BLOCK - 1])
                if found is not None and found.matches(request.all_token_ids[0:pos]):
                    entry, q = found, pos
                    break
            q = q or 0
        elif q:
            entry = registry.get(hashes[q // BLOCK - 1])
        plan = [(pos, hashes[pos // BLOCK - 1]) for pos in boundaries if pos > q]
        registry.begin_step()
        registry.stage(Grant(request.request_id, q, q, hashes[q // BLOCK - 1] if q else None, entry, plan, request,
                             chain=hashes[0], tenant=tenant_of_salt(request.cache_salt), gap=gap))
        registry.commit({request.request_id: q})
        for pos, _ in plan:
            registry.capture(request.request_id, pos, rec=('state', pos), carry='c', nbytes=10, loop_pos=pos)
        registry.begin_step()
        registry.forget_request(request.request_id)
        return q

    def longest(self, request):
        """The deepest resident checkpoint that is the start of request's tokens (what an exact trim would pick)."""
        best = 0
        for entry in self.registry.entries.values():
            if entry.pos > best and entry.matches(request.all_token_ids[0:entry.pos]):
                best = entry.pos
        return best


class SupersessionTests(unittest.TestCase):
    system = span(2200, 99)

    def conversation(self, tag='alpha', seed=1):
        return Conversation(tag, seed, self.system)

    def three_turns(self, supersede=True, **extra):
        harness = SupersessionHarness(supersede, **extra)
        conv = self.conversation()
        boundaries = []
        for turn, extra_tokens in enumerate((3000, 3000, 3000)):
            request = conv.turn(extra_tokens)
            top = (len(request.all_token_ids) // CHUNK) * CHUNK
            harness.serve(request, [top])
            boundaries.append(top)
            conv.history = request.all_token_ids + span(500, 7000 + turn)
        return harness, conv, boundaries

    def test_the_turn_before_the_previous_one_is_retired(self):
        harness, conv, boundaries = self.three_turns()
        registry = harness.registry
        self.assertEqual(len(set(boundaries)), 3)
        resident = sorted(entry.pos for entry in registry.entries.values())
        self.assertEqual(resident, boundaries[1:], 'the previous turn and the newest stay; the oldest is dead weight')
        self.assertEqual((registry.stats['evicted_superseded'], registry.stats['evicted_lru']), (1, 0))
        self.assertEqual(registry.bytes, 20)
        self.assertEqual(registry.snapshot()['supersede'], 1)

    def test_off_by_default_nothing_is_retired(self):
        harness, conv, boundaries = self.three_turns(supersede=False)
        self.assertEqual(sorted(entry.pos for entry in harness.registry.entries.values()), boundaries)
        self.assertEqual(harness.registry.stats['supersede_checks'], 0)
        unset = SupersessionHarness.__new__(SupersessionHarness)
        unset.registry = PrefixRegistry(budget_bytes=1 << 40, environ={})
        self.assertFalse(unset.registry.supersede)

    def test_another_conversation_that_shares_only_the_first_block_is_never_touched(self):
        harness = SupersessionHarness()
        registry = harness.registry
        mine, other = self.conversation('alpha', 1), self.conversation('alpha', 2)
        other.salt = mine.salt
        first = other.turn(3500)
        harness.serve(first, [(len(first.all_token_ids) // CHUNK) * CHUNK])
        other_keys = set(registry.entries)
        self.assertEqual(first.block_hashes[0], mine.turn(100).block_hashes[0], 'the same first block: one chain')
        for turn in range(4):
            request = mine.turn(3000)
            harness.serve(request, [(len(request.all_token_ids) // CHUNK) * CHUNK])
            mine.history = request.all_token_ids + span(300, 5000 + turn)
        self.assertTrue(other_keys <= set(registry.entries), 'the other conversation\'s checkpoint is still resident')
        self.assertGreaterEqual(registry.stats['evicted_superseded'], 1)
        self.assertEqual(sum(1 for e in registry.entries.values() if e.chain == first.block_hashes[0]), 3,
                         'the other one plus my previous and newest turns')

    def test_a_pinned_older_checkpoint_stays(self):
        harness = SupersessionHarness()
        registry = harness.registry
        conv = self.conversation()
        requests = []
        for turn in range(2):
            request = conv.turn(3000)
            requests.append(request)
            harness.serve(request, [(len(request.all_token_ids) // CHUNK) * CHUNK])
            conv.history = request.all_token_ids + span(300, 4000 + turn)
        oldest = min(registry.entries.values(), key=lambda entry: entry.pos)
        oldest.pins = 1
        request = conv.turn(3000)
        harness.serve(request, [(len(request.all_token_ids) // CHUNK) * CHUNK], q=0)
        self.assertIn(oldest.key, registry.entries)
        self.assertEqual(registry.stats['supersede_kept_pinned'], 1)

    def test_a_shared_gap_boundary_stays(self):
        harness = SupersessionHarness()
        registry = harness.registry
        conv = self.conversation()
        first = conv.turn(6000)
        top = (len(first.all_token_ids) // CHUNK) * CHUNK
        gap = 2 * CHUNK
        harness.serve(first, [gap, top], gap=gap, q=0)
        shared = registry.get(first.block_hashes[gap // BLOCK - 1])
        self.assertTrue(shared.shared)
        self.assertFalse(registry.get(first.block_hashes[top // BLOCK - 1]).shared)
        conv.history = first.all_token_ids + span(300, 4100)
        for turn in range(3):
            request = conv.turn(3000)
            harness.serve(request, [(len(request.all_token_ids) // CHUNK) * CHUNK])
            conv.history = request.all_token_ids + span(300, 4200 + turn)
        self.assertIn(shared.key, registry.entries, 'other sessions\' prefixes reach it: only the byte budget may take it')
        self.assertGreaterEqual(registry.stats['supersede_kept_shared'], 1)

    def test_a_branch_point_stays(self):
        """Two admissions resumed from the same checkpoint: something other than the next turn depends on it."""
        harness = SupersessionHarness()
        registry = harness.registry
        conv = self.conversation()
        first = conv.turn(3000)
        harness.serve(first, [(len(first.all_token_ids) // CHUNK) * CHUNK])
        fork_point = registry.get(first.block_hashes[(len(first.all_token_ids) // CHUNK) * CHUNK // BLOCK - 1])
        history = first.all_token_ids + span(300, 4300)
        for branch in range(2):
            request = Request('branch-%d' % branch, history + span(3000, 4400 + branch), salt=conv.salt)
            harness.serve(request, [(len(request.all_token_ids) // CHUNK) * CHUNK])
        self.assertEqual(fork_point.grants, 2)
        deeper = Request('deep-0', history + span(3000, 4400) + span(300, 4500) + span(3000, 4600), salt=conv.salt)
        harness.serve(deeper, [(len(deeper.all_token_ids) // CHUNK) * CHUNK])
        deepest = Request('deep-1', deeper.all_token_ids + span(300, 4700) + span(3000, 4800), salt=conv.salt)
        harness.serve(deepest, [(len(deepest.all_token_ids) // CHUNK) * CHUNK])
        self.assertIn(fork_point.key, registry.entries)
        self.assertGreaterEqual(registry.stats['supersede_kept_branch'], 1)

    def test_a_retry_or_edit_of_the_last_message_still_finds_the_previous_turn(self):
        for supersede in (False, True):
            harness, conv, boundaries = self.three_turns(supersede)
            # the history of the last turn's prompt, then another last message: the previous turn's boundary is what it needs
            retry = Request('retry', conv.history + span(3000, 31337), salt=conv.salt)
            self.assertEqual(harness.longest(retry), boundaries[2], 'the newest turn is a prefix of the history: %s' % supersede)
            edit_before_newest = Request('edit', conv.sent[2].all_token_ids[0:boundaries[1] + 700] + span(3000, 31338), salt=conv.salt)
            self.assertEqual(harness.longest(edit_before_newest), boundaries[1], supersede)

    def test_rewriting_history_below_the_previous_turn_loses_the_old_checkpoint_and_nothing_else(self):
        off, conv_off, bounds = self.three_turns(False)
        on, conv_on, _ = self.three_turns(True)
        rewrite = Request('rewrite', conv_off.sent[2].all_token_ids[0:bounds[0] + 700] + span(3000, 31339), salt=conv_off.salt)
        self.assertEqual(off.longest(rewrite), bounds[0])
        self.assertEqual(on.longest(rewrite), 0, 'the documented limit: history rewritten below the previous turn')

    def test_what_is_served_is_still_exactly_the_cold_state_of_the_request(self):
        harness, conv, boundaries = self.three_turns()
        for entry in harness.registry.entries.values():
            request = conv.sent[boundaries.index(entry.pos)]
            self.assertTrue(entry.matches(request.all_token_ids[0:entry.pos]))
            self.assertEqual(entry.rec, ('state', entry.pos))

    def test_a_capture_that_kept_a_pinned_one_does_not_retire(self):
        harness = SupersessionHarness()
        registry = harness.registry
        conv = self.conversation()
        for turn in range(2):
            request = conv.turn(3000)
            harness.serve(request, [(len(request.all_token_ids) // CHUNK) * CHUNK])
            conv.history = request.all_token_ids + span(300, 4000 + turn)
        checks = registry.stats['supersede_checks']
        newest = max(registry.entries.values(), key=lambda entry: entry.pos)
        newest.pins = 1
        again = Request('again', newest.token_ids.tolist() + span(100, 9), salt=conv.salt)
        registry.begin_step()
        registry.stage(Grant('again', 0, 0, None, None, [(newest.pos, newest.key)], again, chain=newest.chain))
        registry.commit({'again': 0})
        self.assertIs(registry.capture('again', newest.pos, nbytes=10, loop_pos=newest.pos), newest)
        self.assertEqual(registry.stats['supersede_checks'], checks)

    def test_in_flight_captures_carry_the_same_identity(self):
        harness = SupersessionHarness()
        registry = harness.registry
        conv = self.conversation()
        request = conv.turn(9000)
        hashes = request.block_hashes
        plan = [(2 * CHUNK, hashes[2 * CHUNK // BLOCK - 1]), (4 * CHUNK, hashes[4 * CHUNK // BLOCK - 1])]
        registry.begin_step()
        registry.stage(Grant('split', 0, 0, None, None, plan, request, chain=hashes[0], tenant='alpha', gap=2 * CHUNK))
        registry.commit({'split': 0}, scheduled={'split': CHUNK})
        self.assertIn('split', registry.inflight)
        registry.begin_step()
        registry.capture('split', 2 * CHUNK, nbytes=10, loop_pos=2 * CHUNK)
        registry.capture('split', 4 * CHUNK, nbytes=10, loop_pos=4 * CHUNK)
        low, high = registry.get(plan[0][1]), registry.get(plan[1][1])
        self.assertEqual((low.shared, high.shared, low.tenant, high.chain), (True, False, 'alpha', hashes[0]))


# ------------------------------------------------------------------------------------------------
# The graft: whole conversations through the scheduler's trim, the cap, the commit and the model
# ------------------------------------------------------------------------------------------------
class Rig(object):
    """A FakeScheduler under the graft, its registry and the cold-chain checking model."""

    def __init__(self, environ=None, num_blocks=3000, budget=1 << 40, clock=None, sticky=False, mid_loop=True):
        self.patches = [mock.patch.dict(sys.modules, fake_vllm_modules()),
                        mock.patch.dict(os.environ, {'QWEN_PREFIX_STATS_PATH': ''})]
        for patcher in self.patches:
            patcher.start()
        self.lines = []
        self.scheduler = FakeScheduler(num_blocks=num_blocks)
        kwargs = {} if clock is None else {'clock': clock}
        self.registry = PrefixRegistry(budget_bytes=budget, environ=dict(environ or {}), **kwargs)
        if mid_loop:
            with Quiet():
                self.registry.enable_mid_loop_capture()
        self.export = StatsExport(path='', interval_s=0.0, logger=self.logger,
                                  **({} if clock is None else {'clock': clock}))
        self.graft = patch_module.install(self.scheduler, registry=self.registry, kill_switch_path=None, logger=self.logger,
                                          stats=self.export)
        self.model = FakeGdnModel(self.registry)
        self.rows = []

    def close(self):
        for patcher in reversed(self.patches):
            patcher.stop()

    def logger(self, message, *values):
        self.lines.append(message % values if values else message)

    def serve(self, request, model=True):
        """Admit, run the model's side, finish. -> (start_pos, grant)"""
        self.scheduler.add(request)
        out = self.scheduler.schedule()
        row = None
        for data in out.scheduled_new_reqs:
            grant = self.registry.grant_for(data.req_id)
            if data.req_id == request.request_id:
                row = (data.num_computed_tokens, grant)
            if model:
                with Quiet():
                    self.model.prefill(data.req_id, self.scheduler.requests[data.req_id].all_token_ids, data.num_computed_tokens)
        self.scheduler.finish(request)
        self.rows.append(row)
        return row

    def state(self):
        """Everything observable of the registry the policies must not move when off."""
        registry = self.registry
        return dict(entries=[(key, entry.pos, entry.nbytes) for key, entry in registry.entries.items()],
                    bytes=registry.bytes, stats={name: registry.stats[name] for name in LEGACY_STATS},
                    rows=[None if row is None else (row[0], row[1].q if row[1] else 0, row[1].h if row[1] else 0,
                                                    row[1].capture_positions() if row[1] else []) for row in self.rows])


def rig(testcase, **kwargs):
    made = Rig(**kwargs)
    testcase.addCleanup(made.close)
    return made


def scenario(rig_, seed, conversations=4, turns=7):
    """Interleaved conversations sharing one system block per tenant: new, continued, retried, edited deep, short. -> labels"""
    rng = random.Random(seed)
    system = {tenant: span(2300, 500 + index) for index, tenant in enumerate(('alpha', 'bravo'))}
    convs = [Conversation(('alpha', 'bravo')[index % 2] + ('X' if index >= 2 else ''), 10 + index, system[('alpha', 'bravo')[index % 2]],
                          salt=salt_of(('alpha', 'bravo')[index % 2])) for index in range(conversations)]
    labels = []
    counts = [0] * conversations
    for _ in range(conversations * turns):
        index = rng.randrange(conversations)
        conv = convs[index]
        kind = 'new' if counts[index] == 0 else rng.choice(['continue', 'continue', 'continue', 'retry', 'deep'])
        counts[index] += 1
        last = conv.sent[-1] if conv.sent else None
        label = 'c%d-%d' % (index, counts[index])
        if kind == 'new':
            request = conv.turn(rng.randrange(2500, 5000), label)
        elif kind in ('continue', 'retry'):
            request = conv.turn(rng.randrange(1500, 3500), label)
        else:
            cut = rng.randrange(CHUNK, max(CHUNK + 1, len(last.all_token_ids) - 3000))
            request = conv.turn(rng.randrange(1500, 3500), label, base=last.all_token_ids[0:cut])
        rig_.serve(request)
        labels.append((kind, request))
        if kind in ('new', 'continue'):
            conv.history = request.all_token_ids + span(rng.randrange(100, 700), seed + len(labels))
        elif kind == 'retry':
            pass    # the next turn resends the same history with another last message
    return labels


# The registry and the scheduler graft as they were before the store policies and the telemetry existed (tp4/next-2): this scenario
# set was run against that code, and the sha256 of everything observable (each admission's start, Q, h and capture plan; the
# checkpoints held, in LRU order, with their bytes; every counter the registry then had; the restores) is pinned. It must not move
# while the new flags are off.
GOLDEN_SHA256 = 'f98554562b3315f708778b7cadfa2e65a9a5779a61b44074814903efc537d37e'
GOLDEN_CONFIGS = ((700, 6), (3000, 1 << 40), (500, 3))


def golden_run(seed, num_blocks, budget, environ):
    rng = random.Random(seed)
    with mock.patch.dict(sys.modules, fake_vllm_modules()), mock.patch.dict(os.environ, {'QWEN_PREFIX_STATS_PATH': ''}):
        scheduler = FakeScheduler(num_blocks=num_blocks)
        registry = PrefixRegistry(budget_bytes=budget, environ=dict(environ))
        with Quiet():
            registry.enable_mid_loop_capture()
        patch_module.install(scheduler, registry=registry, kill_switch_path=None, logger=lambda message, *values: None,
                             stats=patch_module.StatsExport(path='', logger=lambda message, *values: None))
        model = FakeGdnModel(registry)
        systems = {tenant: make_tokens(2300, 500 + index) for index, tenant in enumerate(('alpha', 'bravo'))}
        history, sent, rows, counts = {}, {}, [], {}
        for step in range(40):
            index = rng.randrange(4)
            tenant = ('alpha', 'bravo')[index % 2]
            counts[index] = counts.get(index, 0) + 1
            kind = 'new' if counts[index] == 1 else rng.choice(['continue', 'continue', 'retry', 'deep'])
            last = sent.get(index)
            base = history.get(index, systems[tenant])
            if kind == 'deep':
                base = last[0:rng.randrange(CHUNK, max(CHUNK + 1, len(last) - 3000))]
            extra = rng.randrange(2500, 5000) if kind == 'new' else rng.randrange(1500, 3500)
            text = list(base) + make_tokens(extra, 1000 * index + step)
            request = Request('r%d-%d' % (index, step), text, salt=salt_of(tenant))
            scheduler.add(request)
            for data in scheduler.schedule().scheduled_new_reqs:
                grant = registry.grant_for(data.req_id)
                if data.req_id == request.request_id:
                    rows.append([data.num_computed_tokens, grant.q if grant else 0, grant.h if grant else 0,
                                 grant.capture_positions() if grant else []])
                with Quiet():
                    model.prefill(data.req_id, scheduler.requests[data.req_id].all_token_ids, data.num_computed_tokens)
            scheduler.finish(request)
            sent[index] = text
            if kind in ('new', 'continue'):
                history[index] = text + make_tokens(rng.randrange(100, 700), seed + step)
        return dict(rows=rows, entries=[[key.hex(), entry.pos, entry.nbytes] for key, entry in registry.entries.items()],
                    bytes=registry.bytes, stats={name: registry.stats[name] for name in LEGACY_STATS},
                    restored=sorted(model.restored.items()), chunks=sorted(model.chunks.items()))


def golden_digest(environ):
    import json

    runs = {'%d-%d-%d' % (seed, blocks, budget): golden_run(seed, blocks, budget, environ)
            for seed in range(3) for blocks, budget in GOLDEN_CONFIGS}
    return hashlib.sha256(json.dumps(runs, sort_keys=True, separators=(',', ':')).encode()).hexdigest(), runs


class FlagOffGoldenTests(unittest.TestCase):
    def test_the_registry_with_the_flags_off_is_the_registry_that_was(self):
        for environ in ({}, {'QWEN_PREFIX_TELEMETRY': '0'}, {'QWEN_PREFIX_EVICT': 'lru', 'QWEN_PREFIX_SUPERSEDE': '0'},
                        {'QWEN_PREFIX_GHOST_ENTRIES': '0'}):
            digest, runs = golden_digest(environ)
            self.assertEqual(digest, GOLDEN_SHA256, environ)
        totals = {name: sum(run['stats'][name] for run in runs.values()) for name in ('grants', 'evicted_lru', 'evicted_coupled')}
        self.assertTrue(all(totals.values()), 'the scenarios grant, evict by budget and by the pool: %s' % totals)
        self.assertEqual(sum(len(run['rows']) for run in runs.values()), 9 * 40)

    def test_the_policies_on_move_it(self):
        """The pin is not blind to the feature: with supersession or the fair policy on, the same scenarios differ."""
        for environ in ({'QWEN_PREFIX_SUPERSEDE': '1'}, {'QWEN_PREFIX_EVICT': 'fair'}):
            self.assertNotEqual(golden_digest(environ)[0], GOLDEN_SHA256, environ)


class TelemetryIsOnlyAReadTests(unittest.TestCase):
    def test_telemetry_on_or_off_the_graft_does_exactly_the_same(self):
        for seed in range(3):
            states = []
            for environ in ({}, {'QWEN_PREFIX_TELEMETRY': '0'}, {'QWEN_PREFIX_TELEMETRY': '1', 'QWEN_PREFIX_GHOST_ENTRIES': '5'}):
                made = Rig(environ=environ, num_blocks=700, budget=6)
                try:
                    scenario(made, seed)
                    states.append(made.state())
                finally:
                    made.close()
            self.assertEqual(states[0], states[1], 'seed %d: telemetry off' % seed)
            self.assertEqual(states[0], states[2], 'seed %d: a tiny ghost memory' % seed)
            self.assertGreater(states[0]['stats']['grants'], 0)
            self.assertGreater(states[0]['stats']['evicted_lru'], 0, 'the scenario evicts')

    def test_explicit_off_flags_are_the_unset_defaults(self):
        base = Rig(environ={}, num_blocks=700, budget=6)
        explicit = Rig(environ={'QWEN_PREFIX_EVICT': 'lru', 'QWEN_PREFIX_SUPERSEDE': '0'}, num_blocks=700, budget=6)
        try:
            scenario(base, 4)
            scenario(explicit, 4)
            self.assertEqual(base.state(), explicit.state())
        finally:
            base.close()
            explicit.close()

    def test_every_granted_state_is_the_cold_state_whatever_the_policies(self):
        """FakeGdnModel raises when a restore differs from the cold chain of its row, so reaching the end proves it."""
        for seed in range(3):
            for environ in ({'QWEN_PREFIX_SUPERSEDE': '1'}, {'QWEN_PREFIX_EVICT': 'fair'},
                            {'QWEN_PREFIX_EVICT': 'fair', 'QWEN_PREFIX_SUPERSEDE': '1'}):
                made = Rig(environ=environ, num_blocks=700, budget=7)
                try:
                    scenario(made, seed)
                    self.assertGreater(sum(made.model.restored.values()), 0)
                finally:
                    made.close()

    def test_the_grant_carries_identity_without_changing_what_it_decides(self):
        made = rig(self)
        request = Request('a', span(5000, 3), salt=salt_of('tenantAAAA'))
        start, grant = made.serve(request)
        self.assertEqual((start, grant.q, grant.capture_positions()), (0, 0, [4096]))
        self.assertEqual((grant.chain, grant.tenant, grant.gap), (request.block_hashes[0], 'tenantAAAA', None))
        follow = Request('b', request.all_token_ids + span(3000, 4), salt=salt_of('tenantAAAA'))
        start, grant = made.serve(follow)
        self.assertEqual((start, grant.q, grant.h, grant.gap), (4096, 4096, 4096, None))
        # a sibling that shares 3000 tokens: vLLM hits 2944, the trim has no checkpoint at 2048, so the gap boundary is captured
        sibling = Request('c', request.all_token_ids[0:3000] + span(3000, 5), salt=salt_of('tenantAAAA'))
        start, grant = made.serve(sibling)
        self.assertEqual((start, grant.h, grant.q, grant.gap), (0, 2944, 0, 2048))
        self.assertEqual(grant.capture_positions(), [2048, 4096])
        shared = made.registry.get(request.block_hashes[2048 // BLOCK - 1])
        own = made.registry.get(sibling.block_hashes[4096 // BLOCK - 1])
        self.assertTrue(shared.shared)
        self.assertFalse(own.shared)
        self.assertEqual((shared.tenant, own.tenant, shared.chain, own.chain),
                         ('tenantAAAA', 'tenantAAAA', request.block_hashes[0], request.block_hashes[0]))
        # no gap when the trim found a checkpoint at the last chunk the hit reaches
        wide = Request('d', request.all_token_ids[0:4500] + span(300, 6), salt=salt_of('tenantAAAA'))
        start, grant = made.serve(wide)
        self.assertEqual((start, grant.q, grant.gap), (4096, 4096, None))

    def test_the_end_boundary_is_the_plan_s_prompt_candidate(self):
        made = rig(self)
        graft = made.graft
        for length in (100, 2047, 2048, 5000, 8192):
            request = Request('x', span(length, 1))
            plan, _, _ = graft.plan(request, 0, 0)
            planned = [pos for pos, _ in plan]
            boundary = graft.end_boundary(request)
            self.assertEqual(planned, [boundary] if boundary > 0 else [], length)
        graft.sticky = graft.drop_last = True
        request = Request('x', span(8192, 1))
        plan, _, _ = graft.plan(request, 0, 0)
        self.assertEqual([pos for pos, _ in plan], [graft.end_boundary(request)])
        self.assertEqual(graft.end_boundary(request), 8192 - CHUNK)


class SupersessionOnTheGraftTests(unittest.TestCase):
    def test_continuations_and_retries_are_served_alike_with_and_without_it(self):
        """Never drops what could still serve: every request that extends its own previous or newest turn is granted the same Q."""
        for seed in range(4):
            off, on = Rig(environ={}), Rig(environ={'QWEN_PREFIX_SUPERSEDE': '1'})
            try:
                labels_off = scenario(off, seed, conversations=3, turns=7)
                labels_on = scenario(on, seed, conversations=3, turns=7)
                self.assertEqual([(kind, r.request_id) for kind, r in labels_off], [(kind, r.request_id) for kind, r in labels_on])
                differing = []
                for index, ((kind, request), a, b) in enumerate(zip(labels_off, off.rows, on.rows)):
                    q_off, q_on = a[1].q if a[1] else 0, b[1].q if b[1] else 0
                    self.assertLessEqual(q_on, q_off, 'seed %d request %d: it never serves more than the full store' % (seed, index))
                    if kind in ('new', 'continue', 'retry'):
                        self.assertEqual(q_on, q_off, 'seed %d %s %s' % (seed, kind, request.request_id))
                    elif q_on != q_off:
                        differing.append(request.request_id)
                self.assertLessEqual(on.registry.bytes, off.registry.bytes)
                self.assertTrue(set(on.registry.entries) <= set(off.registry.entries), 'only fewer checkpoints, never others')
                self.assertGreater(on.registry.stats['evicted_superseded'], 0, 'seed %d: the scenario retires something' % seed)
            finally:
                off.close()
                on.close()

    def test_a_long_conversation_holds_two_checkpoints_not_one_per_turn(self):
        made = rig(self, environ={'QWEN_PREFIX_SUPERSEDE': '1'})
        conv = Conversation('alpha', 1, span(2300, 500))
        for turn in range(8):
            request = conv.turn(2600, 't%d' % turn)
            made.serve(request)
            conv.history = request.all_token_ids + span(200, 900 + turn)
        mine = [e for e in made.registry.entries.values() if e.chain == conv.sent[0].block_hashes[0]]
        self.assertEqual(len(mine), 2)
        self.assertEqual(made.registry.stats['evicted_superseded'], 6)
        plain = rig(self)
        conv2 = Conversation('alpha', 1, span(2300, 500))
        for turn in range(8):
            request = conv2.turn(2600, 't%d' % turn)
            plain.serve(request)
            conv2.history = request.all_token_ids + span(200, 900 + turn)
        self.assertEqual(len(plain.registry.entries), 8)


class SupersessionNamesItsOwnCostTests(unittest.TestCase):
    def test_an_unpinned_previous_turn_is_kept_and_only_the_older_ones_go(self):
        """Pins protect the checkpoint a request restored; the rule itself keeps the previous turn without them."""
        registry = PrefixRegistry(budget_bytes=1 << 30, environ={'QWEN_PREFIX_SUPERSEDE': '1'})
        conv = Conversation('alpha', 1, span(2300, 500))
        text = conv.turn(14000).all_token_ids
        hashes = Request('probe', text, salt=conv.salt).block_hashes
        for pos in (2 * CHUNK, 3 * CHUNK, 4 * CHUNK):
            registry.put(hashes[pos // BLOCK - 1], pos, text[0:pos], nbytes=10, chain=hashes[0], tenant='alpha')
        request = Request('new', text, salt=conv.salt)
        registry.begin_step()
        registry.stage(Grant('new', 0, 0, None, None, [(5 * CHUNK, hashes[5 * CHUNK // BLOCK - 1])], request, chain=hashes[0]))
        registry.commit({'new': 0})
        registry.capture('new', 5 * CHUNK, nbytes=10, loop_pos=5 * CHUNK)
        self.assertEqual(sorted(entry.pos for entry in registry.entries.values()), [4 * CHUNK, 5 * CHUNK])
        self.assertEqual(registry.stats['evicted_superseded'], 2)

    def test_a_request_that_needed_a_retired_checkpoint_is_counted_as_the_cost(self):
        made = rig(self, environ={'QWEN_PREFIX_SUPERSEDE': '1'})
        conv = Conversation('alpha', 1, span(2300, 500))
        requests = []
        for turn in range(3):
            request = conv.turn(2600, 't%d' % turn)
            requests.append(request)
            made.serve(request)
            conv.history = request.all_token_ids + span(200, 900 + turn)
        stats = made.registry.stats
        self.assertEqual(stats['evicted_superseded'], 1)
        first_boundary = (len(requests[0].all_token_ids) // CHUNK) * CHUNK
        # history rewritten after the first turn's boundary but before the second's: only the retired checkpoint would have served it
        rewrite = Request('rewrite', requests[2].all_token_ids[0:first_boundary + 300] + span(3000, 55), salt=conv.salt)
        start, grant = made.serve(rewrite)
        self.assertEqual(start, 0)
        self.assertEqual((stats['class_ckpt_evicted'], stats['class_ckpt_evicted_superseded']), (1, 1))
        self.assertEqual(stats['lost_tokens_ckpt_evicted'], first_boundary)
        # while a retry of the last message still hits
        retry = Request('retry', requests[2].all_token_ids + span(300, 56) + span(3000, 57), salt=conv.salt)
        start, grant = made.serve(retry)
        self.assertGreater(start, 0)
        self.assertEqual(stats['class_returning_served'], 3, 'the two continuations and the retry')


class FairPolicyOnTheGraftTests(unittest.TestCase):
    def flood(self, policy):
        made = Rig(environ={'QWEN_PREFIX_EVICT': policy}, budget=3)
        try:
            quiet = Conversation('quietTENANT', 1, span(2300, 11), salt=salt_of('quietTENANT'))
            first = quiet.turn(3000, 'q1')
            made.serve(first)
            quiet.history = first.all_token_ids + span(200, 12)
            for index in range(6):
                loud = Conversation('loudTENANT%d' % index, 20 + index, span(2300, 40 + index), salt=salt_of('loudTENANT'))
                made.serve(loud.turn(3000, 'l%d' % index))
            back = quiet.turn(1500, 'q2')
            start, grant = made.serve(back)
            return start, made.registry
        finally:
            made.close()

    def test_the_quiet_tenant_returns_to_a_hit_under_fair_and_a_miss_under_lru(self):
        start_lru, registry_lru = self.flood('lru')
        start_fair, registry_fair = self.flood('fair')
        self.assertEqual(start_lru, 0)
        self.assertGreater(start_fair, 0)
        self.assertGreater(registry_fair.stats['evicted_fair'], 0)
        self.assertEqual(registry_fair.stats['evicted_lru'], 0)
        self.assertEqual(registry_lru.stats['evicted_fair'], 0)


# ------------------------------------------------------------------------------------------------
# WP0: the admission classes
# ------------------------------------------------------------------------------------------------
class ClassifyTests(unittest.TestCase):
    """The registry's side, driven by hand: note_attempt for the trim's answer, commit for the step's admission."""

    def setUp(self):
        self.clock = Clock()
        self.registry = PrefixRegistry(budget_bytes=1 << 40, environ={}, clock=self.clock)

    def admit(self, request, h=0, q=0, kind=None, start=None, end=None, captured=True, plan=()):
        registry = self.registry
        hashes = request.block_hashes
        prompt = len(request.all_token_ids)
        end = (prompt // CHUNK) * CHUNK if end is None else end
        registry.note_attempt(request.request_id, hashes, prompt, end, h, q, kind)
        registry.begin_step()
        entry = registry.get(hashes[q // BLOCK - 1]) if q else None
        registry.stage(Grant(request.request_id, q, h, hashes[q // BLOCK - 1] if q else None, entry,
                             [(pos, hashes[pos // BLOCK - 1]) for pos in plan], request, chain=hashes[0]))
        registry.commit({request.request_id: q if start is None else start})
        if captured:
            for pos in plan:
                registry.capture(request.request_id, pos, nbytes=10, loop_pos=pos)

    def classes(self):
        return {name: self.registry.stats['class_' + name] for name in CLASSES if self.registry.stats['class_' + name]}

    def first(self, tokens=4300, seed=1, salt=None, name='a'):
        request = Request(name, span(tokens, seed), salt=salt or salt_of('alpha'))
        self.admit(request, plan=[4096])
        return request

    def test_first_then_served(self):
        a = self.first()
        self.assertEqual(self.classes(), {'first_turn': 1})
        b = Request('b', a.all_token_ids + span(3000, 2), salt=a.cache_salt)
        self.clock.now += 40
        self.admit(b, h=4096, q=4096, plan=[6144])
        self.assertEqual(self.classes(), {'first_turn': 1, 'returning_served': 1})
        self.assertEqual(self.registry.stats['returning_sessions'], 1)
        histograms = self.registry.histograms()
        served = histograms['reuse_seconds']['served']
        self.assertEqual(sum(served['counts']), 1)
        self.assertEqual(served['counts'][2], 1, '40 s falls in the le=60 bucket')
        self.assertEqual(served['sum'], 40.0)
        self.assertEqual(sum(histograms['reuse_seconds']['missed']['counts']), 0)
        self.assertEqual(histograms['reuse_tokens']['served']['sum'], 0, 'no other request was admitted in between')

    def test_reuse_distance_in_tokens_counts_the_others_only(self):
        a = self.first()
        for index in range(3):
            self.admit(Request('o%d' % index, span(5000, 50 + index), salt=salt_of('other')), plan=[4096])
        b = Request('b', a.all_token_ids + span(3000, 2), salt=a.cache_salt)
        self.admit(b, h=4096, q=4096, plan=[6144])
        cell = self.registry.histograms()['reuse_tokens']['served']
        self.assertEqual(cell['sum'], 15000)
        bounds = self.registry.histograms()['reuse_tokens']['bounds']
        self.assertEqual(cell['counts'][bounds.index(16384)], 1, '15000 <= 16384')
        self.assertEqual(self.registry.stats['admitted_prompt_tokens'], 4300 + 15000 + len(b.all_token_ids))

    def test_the_buckets_are_le_and_the_top_one_is_open(self):
        registry = self.registry
        seconds = registry_module.REUSE_SECONDS_BOUNDS
        for value in (0, 10, 10.0001, 86400, 86401, 10 ** 9):
            registry._observe('seconds', 'missed', value)
        counts = registry.histograms()['reuse_seconds']['missed']['counts']
        self.assertEqual(len(counts), len(seconds) + 1)
        self.assertEqual((counts[0], counts[1], counts[len(seconds) - 1], counts[len(seconds)]), (2, 1, 1, 2))

    def test_kv_evicted(self):
        a = self.first()
        b = Request('b', a.all_token_ids + span(3000, 2), salt=a.cache_salt)
        self.admit(b, h=2880, q=0, plan=[6144])
        self.assertEqual(self.classes(), {'first_turn': 1, 'kv_evicted': 1})
        self.assertEqual(self.registry.stats['lost_tokens_kv_evicted'], 4096)
        self.assertEqual(sum(self.registry.histograms()['reuse_seconds']['missed']['counts']), 1)

    def test_kv_evicted_with_a_partial_hit_at_an_older_boundary_counts_the_hit(self):
        a = self.first()
        c = Request('c', a.all_token_ids[0:2100] + span(100, 8), salt=a.cache_salt)
        b = Request('b', a.all_token_ids + span(3000, 2), salt=a.cache_salt)
        self.admit(c, plan=[2048])
        self.admit(b, h=3000, q=2048, plan=[6144])
        self.assertEqual(self.registry.stats['class_kv_evicted'], 1)
        self.assertEqual(self.registry.stats['class_kv_evicted_hit'], 1)
        self.assertEqual(self.registry.stats['lost_tokens_kv_evicted'], 4096 - 2048)

    def test_ckpt_evicted_by_every_reason(self):
        def returning(reason):
            registry = PrefixRegistry(budget_bytes=1 << 40, environ={}, clock=self.clock)
            self.registry = registry
            a = self.first()
            key = a.block_hashes[4096 // BLOCK - 1]
            if reason == 'budget':
                registry.budget_bytes = 5
                registry.put(b'other', CHUNK, list(range(CHUNK)), nbytes=1)
            elif reason == 'coupled':
                registry.drop(key, 'coupled')
            elif reason == 'cleared':
                registry.clear()
            elif reason == 'other':
                registry.drop(key, 'dropped')
            elif reason == 'superseded':
                registry._remove(key, 'superseded')
            self.assertNotIn(key, registry.entries)
            b = Request('b', a.all_token_ids + span(3000, 2), salt=a.cache_salt)
            self.admit(b, h=4096, q=0, plan=[6144])
            return registry.stats

        for reason in registry_module.GONE_REASONS:
            stats = returning(reason)
            self.assertEqual(stats['class_ckpt_evicted'], 1, reason)
            self.assertEqual(stats['class_ckpt_evicted_' + reason], 1, reason)
            self.assertEqual(stats['lost_tokens_ckpt_evicted'], 4096)

    def test_ckpt_missing_when_it_was_never_stored(self):
        request = Request('a', span(4300, 1), salt=salt_of('alpha'))
        self.admit(request, plan=[4096], captured=False)
        b = Request('b', request.all_token_ids + span(3000, 2), salt=request.cache_salt)
        self.admit(b, h=4096, q=0, plan=[6144])
        self.assertEqual(self.classes(), {'first_turn': 1, 'ckpt_missing': 1})
        self.assertEqual(self.registry.stats['lost_tokens_ckpt_missing'], 4096)

    def test_refused_names_a_token_mismatch(self):
        a = self.first()
        entry = self.registry.get(a.block_hashes[4096 // BLOCK - 1])
        entry.token_ids[100] ^= 1
        b = Request('b', a.all_token_ids + span(3000, 2), salt=a.cache_salt)
        self.assertFalse(self.registry.tokens_match('b', 4096, entry, lambda: b.all_token_ids))
        self.admit(b, h=4096, q=0, plan=[6144])
        self.assertEqual((self.registry.stats['class_refused'], self.registry.stats['class_refused_mismatch']), (1, 1))
        self.assertEqual(self.registry.stats['lost_tokens_refused'], 4096)
        again = Request('c', a.all_token_ids + span(3000, 3), salt=a.cache_salt)
        self.admit(again, h=4096, q=0, plan=[6144])
        self.assertEqual((self.registry.stats['class_refused'], self.registry.stats['class_refused_mismatch']), (2, 1),
                         'a refusal for another reason (same-step, ceiling) is not a mismatch')

    def test_rewritten_is_a_known_first_block_with_no_earlier_boundary(self):
        a = self.first()
        sibling = Request('s', a.all_token_ids[0:300] + span(4000, 9), salt=a.cache_salt)
        self.admit(sibling, h=256, q=0, plan=[4096])
        self.assertEqual(self.classes(), {'first_turn': 1, 'rewritten': 1})
        elsewhere = Request('e', span(4300, 77), salt=a.cache_salt)
        self.admit(elsewhere, plan=[4096])
        self.assertEqual(self.registry.stats['class_first_turn'], 2, 'another first block: a cold first turn')
        other_tenant = Request('t', a.all_token_ids, salt=salt_of('bravo'))
        self.admit(other_tenant, plan=[4096])
        self.assertEqual(self.registry.stats['class_first_turn'], 3, 'the salt is in the first block: another tenant never matches')

    def test_a_sibling_hit_through_a_shared_prefix_is_a_first_turn_that_counts_its_hit(self):
        a = self.first()
        sibling = Request('s', a.all_token_ids[0:2100] + span(4200, 9), salt=a.cache_salt)
        self.admit(sibling, h=2048, q=2048, plan=[6144])
        self.assertEqual(self.registry.stats['class_rewritten'], 1)
        self.assertEqual(self.registry.stats['class_rewritten_hit'], 1)

    def test_unsalted_denied_and_short(self):
        unsalted = Request('u', span(5000, 1), salt=None)
        self.admit(unsalted, kind='unsalted')
        denied = Request('d', span(5000, 2), salt=salt_of('alpha'))
        self.admit(denied, kind='denied')
        short = Request('x', span(1500, 3), salt=salt_of('alpha'))
        self.admit(short, end=0)
        self.assertEqual(self.classes(), {'unsalted': 1, 'denied': 1, 'short': 1})
        self.assertEqual(self.registry.ghost.keys() & set(unsalted.block_hashes + denied.block_hashes), set(),
                         'a refused request leaves no session record')
        self.assertEqual(self.registry.stats['admitted_prompt_tokens'], 5000 + 5000 + 1500)

    def test_a_resent_prompt_of_a_whole_chunk_is_judged_by_the_boundary_it_can_hit(self):
        a = Request('a', span(4096, 1), salt=salt_of('alpha'))
        self.admit(a, plan=[4096])
        again = Request('again', list(a.all_token_ids), salt=a.cache_salt)
        # vLLM cannot hit the last token: h <= 4032, and the best boundary is 2048
        self.admit(again, h=4032, q=0, plan=[])
        self.assertEqual(self.registry.stats['class_ckpt_missing'] + self.registry.stats['class_kv_evicted']
                         + self.registry.stats['class_ckpt_evicted'] + self.registry.stats['class_refused'], 1)
        self.assertEqual(self.registry.stats['class_kv_evicted'], 0, 'the KV reached what a resent prompt can reach')

    def test_each_admission_is_classified_once_even_when_readmitted(self):
        a = self.first()
        total = sum(self.registry.stats['class_' + name] for name in CLASSES)
        self.registry.note_attempt('a', a.block_hashes, 4300, 4096, 4096, 4096)
        self.registry.begin_step()
        self.registry.commit({'a': 4096})
        self.assertEqual(sum(self.registry.stats['class_' + name] for name in CLASSES), total, 'a preempted request, admitted again')
        self.registry.forget_request('a')
        self.assertNotIn('a', self.registry.classified)
        self.assertEqual(self.registry.stats['admitted_prompt_tokens'], 4300)

    def test_a_request_not_admitted_is_not_counted_and_the_last_attempt_wins(self):
        a = self.first()
        b = Request('b', a.all_token_ids + span(3000, 2), salt=a.cache_salt)
        for h, q in ((0, 0), (64, 0), (4096, 4096)):
            self.registry.note_attempt('b', b.block_hashes, len(b.all_token_ids), 6144, h, q)
        self.assertEqual(sum(self.registry.stats['class_' + name] for name in CLASSES), 1)
        self.registry.begin_step()
        self.registry.commit({})
        self.assertIn('b', self.registry.attempts)
        self.registry.forget_request('b')
        self.assertNotIn('b', self.registry.attempts, 'an aborted waiting request leaves nothing behind')
        self.registry.note_attempt('b', b.block_hashes, len(b.all_token_ids), 6144, 4096, 4096)
        self.registry.begin_step()
        self.registry.commit({'b': 4096})
        self.assertEqual(self.registry.stats['class_returning_served'], 1)

    def test_the_admitted_start_is_what_counts(self):
        a = self.first()
        b = Request('b', a.all_token_ids + span(3000, 2), salt=a.cache_salt)
        self.admit(b, h=4096, q=4096, start=0, plan=[])
        self.assertEqual(self.registry.stats['class_returning_served'], 0)
        self.assertEqual(self.registry.stats['class_ckpt_evicted'] + self.registry.stats['class_ckpt_missing']
                         + self.registry.stats['class_refused'], 1)

    def test_the_memory_is_bounded(self):
        registry = PrefixRegistry(budget_bytes=1 << 40, environ={'QWEN_PREFIX_GHOST_ENTRIES': '3'}, clock=self.clock)
        self.registry = registry
        for index in range(8):
            self.admit(Request('r%d' % index, span(4300, 100 + index), salt=salt_of('alpha%d' % index)), plan=[4096])
        self.assertEqual(len(registry.ghost), 3)
        self.assertLessEqual(len(registry.chains), 3)
        self.assertEqual(registry.snapshot()['ghost_now'], 3)
        for index in range(registry_module.ATTEMPTS_LIMIT + 5):
            registry.note_attempt('w%d' % index, [b'h'], 100, 0, 0, 0)
        self.assertEqual(len(registry.attempts), registry_module.ATTEMPTS_LIMIT)

    def test_ghost_memory_zero_still_classifies_without_remembering(self):
        registry = PrefixRegistry(budget_bytes=1 << 40, environ={'QWEN_PREFIX_GHOST_ENTRIES': '0'}, clock=self.clock)
        self.registry = registry
        a = self.first()
        b = Request('b', a.all_token_ids + span(3000, 2), salt=a.cache_salt)
        self.admit(b, h=4096, q=4096, plan=[6144])
        self.assertEqual(len(registry.ghost), 0)
        self.assertEqual(self.classes(), {'first_turn': 1, 'rewritten': 1})

    def test_a_rebuilt_engine_forgets_the_sessions(self):
        class Owner(object):
            pass

        first = Owner()
        self.registry.bind(first)
        a = self.first()
        self.assertTrue(self.registry.ghost)
        del first
        import gc
        gc.collect()
        self.registry.bind(Owner())
        self.assertFalse(self.registry.ghost or self.registry.chains or self.registry.attempts or self.registry.classified)

    def test_disable_and_clear(self):
        a = self.first()
        self.registry.note_attempt('z', a.block_hashes, 4300, 4096, 0, 0)
        self.registry.disable('kill switch')
        self.assertFalse(self.registry.attempts)
        self.assertTrue(self.registry.ghost, 'the memory survives a latch: the sessions still existed')

    def test_a_failure_is_counted_and_never_raised(self):
        a = self.first()
        self.registry.attempts['boom'] = object()
        with Quiet():
            self.registry.begin_step()
            self.registry.commit({'boom': 0})
        self.assertEqual(self.registry.stats['class_failures'], 1)
        self.registry.note_attempt('x', None, 100, 0, 0, 0)
        self.assertEqual(self.registry.stats['class_failures'], 2)

    def test_telemetry_off_records_nothing(self):
        registry = PrefixRegistry(budget_bytes=1 << 40, environ={'QWEN_PREFIX_TELEMETRY': '0'}, clock=self.clock)
        self.registry = registry
        a = self.first()
        b = Request('b', a.all_token_ids + span(3000, 2), salt=a.cache_salt)
        self.admit(b, h=4096, q=4096, plan=[6144])
        self.assertEqual(sum(registry.stats['class_' + name] for name in CLASSES), 0)
        self.assertEqual((registry.ghost, registry.attempts, registry.histograms()), (OrderedDict(), OrderedDict(), {}))
        self.assertEqual(registry.snapshot()['telemetry'], 0)

    def test_every_admission_lands_in_exactly_one_class(self):
        for seed in range(3):
            made = Rig(environ={}, num_blocks=700, budget=6)
            try:
                scenario(made, seed)
                stats = made.registry.stats
                self.assertEqual(sum(stats['class_' + name] for name in CLASSES), len(made.rows))
                self.assertEqual(stats['admitted_prompt_tokens'],
                                 sum(len(made.scheduler.requests[req].all_token_ids) for req in made.scheduler.requests))
                self.assertEqual(stats['class_failures'], 0)
            finally:
                made.close()


class ClassifyOnTheGraftTests(unittest.TestCase):
    def test_the_whole_taxonomy_through_the_scheduler(self):
        clock = Clock(5000.0)
        made = rig(self, environ={}, num_blocks=150, clock=clock)
        stats = made.registry.stats
        a = Request('a1', span(4200, 1), salt=salt_of('alpha'))
        made.serve(a)
        self.assertEqual(stats['class_first_turn'], 1)
        clock.now += 30
        a2 = Request('a2', a.all_token_ids + span(600, 2), salt=a.cache_salt)
        start, grant = made.serve(a2)
        self.assertEqual((start, stats['class_returning_served']), (4096, 1))
        # another tenant takes the pool: a's KV tail is evicted, and its checkpoint goes with the boundary block
        filler = Request('f', span(6000, 3), salt=salt_of('bravo'))
        made.serve(filler)
        self.assertGreater(stats['evicted_coupled'], 0)
        clock.now += 600
        a3 = Request('a3', a2.all_token_ids + span(600, 4), salt=a.cache_salt)
        start, grant = made.serve(a3)
        self.assertEqual(start, 0)
        self.assertEqual(stats['class_kv_evicted'], 1)
        self.assertGreater(stats['lost_tokens_kv_evicted'], 0)
        seconds = made.registry.histograms()['reuse_seconds']
        self.assertEqual((sum(seconds['served']['counts']), sum(seconds['missed']['counts'])), (1, 1))
        self.assertEqual(seconds['missed']['sum'], 600.0, 'seconds since a2, its previous turn')
        unsalted = Request('u', span(3000, 5), salt=None)
        made.serve(unsalted)
        denied = Request('s', span(3000, 6), salt=salt_of('alpha'), resumable=True)
        made.serve(denied)
        short = Request('t', span(900, 7), salt=salt_of('alpha'))
        made.serve(short)
        self.assertEqual((stats['class_unsalted'], stats['class_denied'], stats['class_short']), (1, 1, 1))
        self.assertEqual(stats['class_failures'], 0)

    def test_a_checkpoint_the_budget_evicted_with_the_kv_still_resident(self):
        made = rig(self, environ={}, budget=1)
        stats = made.registry.stats
        a = Request('a', span(4200, 1), salt=salt_of('alpha'))
        made.serve(a)
        made.serve(Request('b', span(4200, 2), salt=salt_of('bravo')))
        self.assertEqual(stats['evicted_lru'], 1)
        made.serve(Request('a2', a.all_token_ids + span(600, 3), salt=a.cache_salt))
        self.assertEqual((stats['class_ckpt_evicted'], stats['class_ckpt_evicted_budget']), (1, 1))
        self.assertEqual(stats['lost_tokens_ckpt_evicted'], 4096)

    def test_a_checkpoint_that_was_never_taken(self):
        made = rig(self, environ={})
        stats = made.registry.stats
        a = Request('a', span(4200, 1), salt=salt_of('alpha'))
        made.serve(a, model=False)
        made.serve(Request('a2', a.all_token_ids + span(600, 3), salt=a.cache_salt))
        self.assertEqual(stats['class_ckpt_missing'], 1)

    def test_a_corrupted_checkpoint_is_refused_and_named(self):
        made = rig(self, environ={})
        stats = made.registry.stats
        text = span(4200, 5)
        made.serve(Request('x1', text[0:2100], salt=salt_of('alpha')))
        x2 = Request('x2', text, salt=salt_of('alpha'))
        made.serve(x2)
        made.registry.get(x2.block_hashes[63]).token_ids[100] ^= 1
        made.serve(Request('b', text[0:4096] + span(300, 6), salt=salt_of('alpha')))
        self.assertEqual((stats['class_refused'], stats['class_refused_mismatch'], stats['token_mismatches']), (1, 1, 1))

    def test_sticky_sessions_judge_the_boundary_the_drop_leaves(self):
        made = rig(self, environ={})
        made.graft.sticky = made.graft.drop_last = True
        a = Request('a', span(8300, 1), salt=salt_of('alpha'))
        self.assertEqual(made.graft.end_boundary(a), 8192 - CHUNK)
        made.registry.note_attempt('a', a.block_hashes, 8300, made.graft.end_boundary(a), 0, 0)
        self.assertEqual(made.registry.attempts['a'].end, 6144)


# ------------------------------------------------------------------------------------------------
# WP0: the periodic lines and the host gauges
# ------------------------------------------------------------------------------------------------
PAIR = re.compile(r'(\w+)=(\S+)')


def parse(line):
    return dict(PAIR.findall(line))


class TierLineTests(unittest.TestCase):
    def registry_with_returns(self):
        clock = Clock()
        registry = PrefixRegistry(budget_bytes=1 << 30, environ={'QWEN_PREFIX_SUPERSEDE': '1'}, clock=clock)
        harness = ClassifyTests('test_first_then_served')
        harness.registry, harness.clock = registry, clock
        a = harness.first()
        clock.now += 90
        b = Request('b', a.all_token_ids + span(3000, 2), salt=a.cache_salt)
        harness.admit(b, h=4096, q=4096, plan=[6144])
        return registry

    def test_the_tier_line_is_key_value_and_complete(self):
        registry = self.registry_with_returns()
        lines = registry.telemetry_lines(host={'host_rss_bytes': 123456, 'host_mem_available_bytes': 7890000})
        self.assertEqual(len(lines), 2)
        fields = parse(lines[0])
        self.assertTrue(lines[0].startswith('tier '))
        self.assertEqual((fields['rss'], fields['avail']), ('123456', '7890000'))
        self.assertEqual((fields['reg_entries'], fields['reg_bytes'], fields['adm'], fields['returning']), ('2', '20', '2', '1'))
        self.assertEqual((fields['first_turn'], fields['returning_served'], fields['evict'], fields['supersede']), ('1', '1', 'lru', '1'))
        for name in CLASSES:
            self.assertIn(name, fields)
        for name in ('lost_kv', 'lost_ckpt', 'lost_missing', 'lost_refused', 'superseded', 'evicted_lru', 'evicted_fair', 'evicted_coupled'):
            self.assertIn(name, fields)

    def test_the_reuse_line_carries_both_histograms(self):
        registry = self.registry_with_returns()
        fields = parse(registry.telemetry_lines(host={})[1])
        bounds_s = [int(x) for x in fields['bounds_s'].split(',')]
        counts, total = fields['served_s'].split(':')
        counts = [int(x) for x in counts.split(',')]
        self.assertEqual(len(counts), len(bounds_s) + 1)
        self.assertEqual((sum(counts), int(total), counts[bounds_s.index(120)]), (1, 90, 1))
        self.assertEqual(sum(int(x) for x in fields['missed_s'].split(':')[0].split(',')), 0)
        self.assertEqual(len(fields['bounds_tok'].split(',')) + 1, len(fields['served_tok'].split(':')[0].split(',')))

    def test_a_host_that_cannot_be_read_leaves_the_keys_out(self):
        registry = PrefixRegistry(budget_bytes=1, environ={})
        fields = parse(registry.telemetry_lines(host={})[0])
        self.assertNotIn('rss', fields)
        self.assertNotIn('avail', fields)

    def test_the_cadence_and_the_force(self):
        now = [100.0]
        lines = []
        registry = PrefixRegistry(budget_bytes=1, environ={})
        export = StatsExport(path='', interval_s=30.0, clock=lambda: now[0], logger=lambda message, *v: lines.append(message % v if v else message))
        self.assertTrue(export.maybe_tier(registry))
        self.assertFalse(export.maybe_tier(registry), 'within the interval')
        now[0] += 29
        self.assertFalse(export.maybe_tier(registry))
        now[0] += 2
        self.assertTrue(export.maybe_tier(registry), 'whether or not a counter moved')
        self.assertTrue(export.maybe_tier(registry, force=True))
        self.assertEqual(len([line for line in lines if line.startswith('tier ')]), 3)
        self.assertEqual(len([line for line in lines if line.startswith('reuse ')]), 3)
        self.assertEqual(export.tiers, 3)

    def test_nothing_without_telemetry_and_never_a_raise(self):
        lines = []
        registry = PrefixRegistry(budget_bytes=1, environ={'QWEN_PREFIX_TELEMETRY': '0'})
        export = StatsExport(path='', interval_s=0.0, logger=lambda message, *v: lines.append(message % v if v else message))
        self.assertFalse(export.maybe_tier(registry))
        self.assertEqual(lines, [])
        self.assertFalse(export.maybe_tier(object()), 'a stand-in registry')

        class Broken(object):
            telemetry = True

            def telemetry_lines(self):
                raise RuntimeError('boom')

        self.assertFalse(export.maybe_tier(Broken()))
        self.assertTrue(lines[-1].startswith('telemetry lines failed: RuntimeError'))

    def test_the_stats_line_and_file_are_what_they_were(self):
        """maybe_export is untouched: the same json line on change, the same file; the tier lines are a separate call."""
        directory = tempfile.mkdtemp(prefix='qwen-prefix-tier-')
        self.addCleanup(shutil.rmtree, directory, True)
        path = os.path.join(directory, 'stats.json')
        lines = []
        registry = PrefixRegistry(budget_bytes=1 << 20, environ={})
        export = StatsExport(path=path, interval_s=0.0, logger=lambda message, *v: lines.append(message % v if v else message))
        self.assertTrue(export.maybe_export(registry))
        self.assertTrue(all(line.startswith('stats {') for line in lines))
        self.assertEqual(sorted(os.listdir(directory)), ['stats.json'])

    def test_the_graft_logs_the_lines_with_its_stats_and_its_policy(self):
        made = rig(self, environ={'QWEN_PREFIX_EVICT': 'fair'})
        made.serve(Request('a', span(4200, 1), salt=salt_of('alpha')))
        self.assertTrue(any(line.startswith('tier ') for line in made.lines))
        policy = [line for line in made.lines if line.startswith('install policy ')]
        self.assertEqual(len(policy), 1)
        fields = parse(policy[0])
        self.assertEqual((fields['evict'], fields['supersede'], fields['telemetry']), ('fair', '0', '1'))
        self.assertEqual(int(fields['checkpoint_bytes']), CHECKPOINT_NBYTES)
        self.assertEqual(int(fields['ghost_entries']), 65536)

    def test_host_gauges_from_proc(self):
        proc = tempfile.mkdtemp(prefix='qwen-prefix-proc-')
        self.addCleanup(shutil.rmtree, proc, True)
        os.makedirs(os.path.join(proc, 'self'))
        with open(os.path.join(proc, 'self', 'status'), 'w') as handle:
            handle.write('Name:\tpython\nVmPeak:\t  999 kB\nVmRSS:\t  2048 kB\nThreads:\t4\n')
        with open(os.path.join(proc, 'meminfo'), 'w') as handle:
            handle.write('MemTotal:       99999 kB\nMemFree: 10 kB\nMemAvailable:   4096 kB\n')
        self.assertEqual(host_gauges(proc), {'host_rss_bytes': 2048 * 1024, 'host_mem_available_bytes': 4096 * 1024})
        self.assertEqual(host_gauges(os.path.join(proc, 'absent')), {})
        with open(os.path.join(proc, 'meminfo'), 'w') as handle:
            handle.write('MemAvailable: lots kB\n')
        self.assertEqual(host_gauges(proc), {'host_rss_bytes': 2048 * 1024})
        live = host_gauges()
        if sys.platform.startswith('linux'):
            self.assertGreater(live['host_rss_bytes'], 0)


if __name__ == '__main__':
    unittest.main()
