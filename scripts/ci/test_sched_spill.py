"""Complete spills of the host KV tier (QWEN_PREFIX_HOST_TIER_SPILL_MODE=complete; docs/tp4-sched-1.md section 3).

The per-flush cap (QWEN_PREFIX_HOST_TIER_SPILL_MAX_BLOCKS, 512) drops what it cuts for good: the blocks vLLM evicted are reused by the very allocation that evicted them, so
their bytes cannot wait for a later step. The flush that exceeds it is a restore (TierHooks.load allocates the whole session at once and evicts as many cached blocks as it
restores: 3,393 of the 3,905 blocks a 254k session displaced in the TC1 card run were dropped, and the sessions they belonged to came back cold). In complete mode the flush
spills everything up to a hard maximum, in batches. These tests drive the scheduler graft over a fake vLLM pool (test_qwen_prefix_tiers.Rig) with the cap scaled down to 16
blocks, so the same shapes run in milliseconds:

  * the default (cap) is byte for byte what it was: no new stats key, no new line, the same drops;
  * the same traffic, cap against complete: the session that was displaced by a restore comes back from the tier instead of cold, and every block the cap would have dropped is
    counted (tier_spill_cap_lifted);
  * batches, a failing read in the middle of a flush, a tier too small, the hard maximum, the slow-flush line;
  * the configuration, the markers and the profile contract.

    nice python3 -B -m unittest test_sched_spill      (from scripts/ci)
"""

import json
from pathlib import Path
import sys
import unittest
from unittest import mock

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import prefix_markers  # noqa: E402
import qwen_prefix_registry as registry_module  # noqa: E402
import qwen_prefix_scheduler_patch as patch_module  # noqa: E402
import serving_c2_contract as contract  # noqa: E402
import test_qwen_prefix_tiers as tiers  # noqa: E402  (module import, so that its TestCases are not collected here again)
from test_qwen_prefix_scheduler_patch import Request  # noqa: E402

MODE = 'QWEN_PREFIX_HOST_TIER_SPILL_MODE'
CAP = 'QWEN_PREFIX_HOST_TIER_SPILL_MAX_BLOCKS'
HARD = 'QWEN_PREFIX_HOST_TIER_SPILL_HARD_MAX_BLOCKS'
BATCH = 'QWEN_PREFIX_HOST_TIER_SPILL_BATCH_BLOCKS'
WARN = 'QWEN_PREFIX_HOST_TIER_SPILL_WARN_MS'
COMPLETE = {MODE: 'complete'}
SPILL_LINE = r'^host tier spill blocks=\d+ bytes=\d+ ms=[0-9.]+ held=\d+$'


def displaced_by_a_restore(env, tier=400):
    """The TC1 shape at 1/32 scale. A's first turn (129 blocks), then another tenant's 219-block session on a 300-block pool, which evicts a third of A; A returns (the restore
    allocates what A lost, and every block it takes from the filler's cached session is a block to spill); then the filler returns. -> (rig, A's start, the filler's start)"""
    made = tiers.Rig(tier=tier, num_blocks=300, budget=1 << 20, tier_env=env)
    a = Request('a1', tiers.span(8200, 1), salt=tiers.salt_of('alpha'))
    made.serve(a)
    made.serve(Request('f1', tiers.span(14000, 2), salt=tiers.salt_of('bravo')))
    start_a, _ = made.serve(Request('a2', a.all_token_ids + tiers.span(700, 3), salt=a.cache_salt))
    start_f, _ = made.serve(Request('f2', tiers.span(14000, 2) + tiers.span(300, 9), salt=tiers.salt_of('bravo')))
    return made, start_a, start_f


class Rigs(unittest.TestCase):
    def displaced(self, env, **kwargs):
        made, start_a, start_f = displaced_by_a_restore(env, **kwargs)
        self.addCleanup(made.close)
        return made, start_a, start_f


class ConfigTests(unittest.TestCase):
    def plan(self, tier=True, **env):
        config = registry_module.tier_config({registry_module.ENV_TIER_GIB: '32'}, 8 << 30) if tier else None
        return registry_module.spill_plan(env, config)

    def test_the_default_and_the_cap_mode_have_no_plan(self):
        for env in ({}, {MODE: ''}, {MODE: 'cap'}):
            self.assertIsNone(self.plan(**env))
            self.assertIsNone(self.plan(tier=False, **env), 'the cap mode needs nothing')

    def test_complete_has_its_documented_defaults(self):
        plan = self.plan(**COMPLETE)
        self.assertEqual((plan.hard_max, plan.batch, plan.warn_ms), (4096, 512, 1000))
        self.assertEqual(registry_module.DEFAULT_TIER_SPILL_HARD_MAX_BLOCKS * registry_module.BLOCK, 262144, 'a 262,144-token session is the largest restore')
        plan = self.plan(**{MODE: 'complete', HARD: '8192', BATCH: '256', WARN: '250'})
        self.assertEqual((plan.hard_max, plan.batch, plan.warn_ms), (8192, 256, 250))

    def test_every_malformed_setting_refuses_to_start(self):
        cases = [dict(**{MODE: 'all'}), dict(**{MODE: 'Complete'}),
                 {HARD: '8192'}, {BATCH: '256'}, {WARN: '10'}, dict(**{MODE: 'cap', BATCH: '256'}),
                 {MODE: 'complete', HARD: '0'}, {MODE: 'complete', HARD: '4096.0'}, {MODE: 'complete', HARD: '-1'}, {MODE: 'complete', HARD: ' 1'},
                 {MODE: 'complete', BATCH: '0'}, {MODE: 'complete', BATCH: 'x'}, {MODE: 'complete', WARN: '0'}, {MODE: 'complete', WARN: '600001'},
                 {MODE: 'complete', HARD: '511'},                                    # below the cap (512): the mode would spill less
                 {MODE: 'complete', HARD: '1024', BATCH: '2048'},                    # a batch above the hard maximum
                 {MODE: 'complete', HARD: '65537'}]
        for env in cases:
            with self.subTest(env=env), self.assertRaises(ValueError):
                self.plan(**env)

    def test_complete_needs_the_tier(self):
        with self.assertRaisesRegex(ValueError, 'needs the host KV tier'):
            self.plan(tier=False, **COMPLETE)

    def test_the_registry_reads_it_and_adds_its_counters_only_then(self):
        off = registry_module.PrefixRegistry(budget_bytes=1 << 30, environ={registry_module.ENV_TIER_GIB: '32'})
        on = registry_module.PrefixRegistry(budget_bytes=1 << 30, environ={registry_module.ENV_TIER_GIB: '32', MODE: 'complete'})
        self.assertIsNone(off.spill_plan)
        self.assertIsNotNone(on.spill_plan)
        # (the exact-resume turn telemetry, on by default and counting only, adds its own turn_* counters to every registry: tp4/next-1)
        self.assertEqual(set(off.stats) - set(registry_module.TURN_STAT_NAMES), set(registry_module.STAT_NAMES))
        self.assertEqual(set(on.stats) - set(off.stats), set(registry_module.SPILL_STAT_NAMES))
        json.dumps(on.stats)
        with self.assertRaises(ValueError):
            registry_module.PrefixRegistry(budget_bytes=1 << 30, environ={BATCH: '256'})


class FlagOffTests(Rigs):
    def test_the_default_run_keeps_its_stats_its_drops_and_its_lines(self):
        made, start_a, start_f = self.displaced({CAP: '16'})
        stats = made.registry.stats
        self.assertEqual(set(stats) - set(registry_module.TURN_STAT_NAMES), set(registry_module.STAT_NAMES), 'no counter of the complete mode')
        self.assertIsNone(made.registry.spill_plan)
        self.assertGreater(stats['tier_spill_dropped_cap'], 0)
        self.assertEqual((start_a, start_f), (0, 0), 'the capped session did not come back, and neither did the one its restore displaced')
        spill = [line for line in made.lines if line.startswith('host tier spill')]
        self.assertTrue(spill)
        for line in spill:
            self.assertRegex(line, SPILL_LINE)
        self.assertFalse([line for line in made.lines if 'spill complete' in line])
        self.assertTrue(made.check_kv())

    def test_a_complete_flush_that_the_cap_never_cut_is_the_default_flush(self):
        results = []
        for env in ({}, COMPLETE):
            made = tiers.Rig(tier=64, num_blocks=tiers.SMALL_POOL, budget=1 << 20, tier_env=env)
            self.addCleanup(made.close)
            a, (start, grant) = tiers.two_turn_scenario(made)
            stats = made.registry.stats
            results.append((start, grant.q, stats['tier_spill_blocks'], stats['tier_spill_bytes'], stats['tier_restore_blocks'], stats['tier_spill_dropped_cap'],
                            sorted(made.registry.tier.records), made.check_kv()))
        self.assertEqual(results[0], results[1])
        self.assertEqual(results[0][:2], (4096, 4096))

    def test_the_default_still_keeps_the_lowest_blocks_of_a_chain_when_the_cap_bites(self):
        made = tiers.Rig(tier=64, num_blocks=tiers.SMALL_POOL, budget=1 << 20, tier_env={CAP: '3'})
        self.addCleanup(made.close)
        tiers.two_turn_scenario(made)
        stats = made.registry.stats
        self.assertEqual((stats['tier_spill_blocks'], stats['tier_spill_dropped_cap']), (3, 5))


class CompleteTests(Rigs):
    def test_the_session_a_restore_displaced_comes_back_from_the_tier_instead_of_cold(self):
        capped, capped_a, capped_f = self.displaced({CAP: '16'})
        complete, complete_a, complete_f = self.displaced(dict({CAP: '16'}, **COMPLETE))
        self.assertEqual((capped_a, capped_f), (0, 0))
        self.assertEqual((complete_a, complete_f), (8192, 12288), 'both sessions are hits again')
        stats = complete.registry.stats
        self.assertEqual(stats['tier_spill_dropped_cap'], 0, 'nothing is cut')
        self.assertGreater(stats['tier_spill_cap_lifted'], 0)
        self.assertEqual(stats['tier_restore_requests'], 2)
        self.assertEqual(stats['tier_digest_failures'], 0)
        self.assertGreater(stats['tier_spill_blocks'], capped.registry.stats['tier_spill_blocks'])
        self.assertTrue(complete.check_kv(), 'every cached block, restored ones included, holds its hash\'s bytes')
        # what the cap would have dropped is exactly what the complete mode spilled beyond it, flush by flush
        self.assertEqual(stats['tier_spill_cap_lifted'], sum(max(0, int(line.split('lifted=')[1])) for line in complete.lines if line.startswith('host tier spill blocks=')))

    def test_the_counters_and_the_lines_say_what_a_flush_did(self):
        made, _a, _f = self.displaced(dict({CAP: '16'}, **COMPLETE))
        stats = made.registry.stats
        self.assertEqual(stats['tier_spill_batches'], stats['tier_spill_flushes'] - 0, 'one batch a flush at the default batch size')
        self.assertEqual(stats['tier_spill_max_blocks'], max(int(line.split('blocks=')[1].split()[0]) for line in made.lines if line.startswith('host tier spill blocks=')))
        self.assertGreater(stats['tier_spill_max_blocks'], 16, 'a flush beyond the cap')
        self.assertEqual(stats['tier_spill_slow'], 0)
        install = [line for line in made.lines if line.startswith('host tier spill complete on')]
        self.assertEqual(install, ['host tier spill complete on cap_blocks=16 hard_max_blocks=4096 batch_blocks=512 warn_ms=1000'])
        for line in made.lines:
            if line.startswith('host tier spill blocks='):
                self.assertRegex(line, r'^host tier spill blocks=\d+ bytes=\d+ ms=[0-9.]+ held=\d+ batches=\d+ lifted=\d+$')
        json.dumps(stats)

    def test_batches_split_the_reads_and_change_nothing_else(self):
        whole, a_whole, f_whole = self.displaced(dict({CAP: '16'}, **COMPLETE))
        split, a_split, f_split = self.displaced(dict({CAP: '16', BATCH: '8'}, **COMPLETE))
        self.assertEqual((a_split, f_split), (a_whole, f_whole))
        self.assertEqual(sorted(split.registry.tier.records), sorted(whole.registry.tier.records), 'the same blocks in the tier')
        self.assertEqual(split.registry.stats['tier_spill_blocks'], whole.registry.stats['tier_spill_blocks'])
        self.assertGreater(split.registry.stats['tier_spill_batches'], whole.registry.stats['tier_spill_batches'])
        sizes = [len(ids) for ids in split.kv.reads]
        self.assertTrue(sizes and max(sizes) <= 8, 'no device read is larger than a batch: %r' % sizes)
        self.assertTrue(split.check_kv())

    def test_a_read_that_fails_in_the_middle_loses_its_batch_and_the_rest_not_what_was_stored(self):
        made = tiers.Rig(tier=400, num_blocks=300, budget=1 << 20, tier_env=dict({CAP: '16', BATCH: '8'}, **COMPLETE))
        self.addCleanup(made.close)
        real, calls = made.kv.read_blocks, []

        def flaky(ids, into=None):
            calls.append(len(ids))
            if len(calls) == 2:
                raise RuntimeError('device read refused')
            return real(ids, into=into)
        made.kv.read_blocks = flaky
        a = Request('a1', tiers.span(8200, 1), salt=tiers.salt_of('alpha'))
        made.serve(a)
        made.serve(Request('f1', tiers.span(14000, 2), salt=tiers.salt_of('bravo')))
        stats = made.registry.stats
        self.assertEqual(stats['tier_spill_failures'], 1)
        self.assertEqual(stats['tier_spill_blocks'], 8, 'the first batch was stored before the second failed')
        self.assertEqual(len(made.registry.tier.records), 8)
        self.assertEqual(made.registry.tier.snapshot()['free_slots'] + 8, made.registry.tier.snapshot()['slots'], 'the failed batch\'s slots went back')
        self.assertTrue(made.registry.tier_on, 'one failure does not latch the tier')
        # ...and the next flush succeeds, which resets the count
        made.serve(Request('g1', tiers.span(9000, 5), salt=tiers.salt_of('carol')))
        self.assertEqual(made.registry.stats['tier_spill_failures'], 1)
        self.assertEqual(made.registry.tier_on, True)

    def test_three_failing_flushes_in_a_row_latch_the_tier_as_before(self):
        made = tiers.Rig(tier=64, num_blocks=100, budget=1 << 20, tier_env=COMPLETE)
        self.addCleanup(made.close)
        made.kv.fail_read = True
        for index in range(4):
            made.serve(Request('x%d' % index, tiers.span(4200, 10 + index), salt=tiers.salt_of('t%d' % index)))
        self.assertEqual(made.registry.stats['tier_latched'], 1)
        self.assertFalse(made.registry.tier_on)

    def test_a_tier_too_small_drops_what_does_not_fit_and_counts_it(self):
        made, _a, _f = self.displaced(dict({CAP: '16'}, **COMPLETE), tier=10)
        stats = made.registry.stats
        self.assertLessEqual(len(made.registry.tier.records), 10)
        self.assertGreater(stats['tier_spill_dropped_full'] + stats['tier_evicted'], 0)
        self.assertTrue(made.check_kv(), 'whatever was dropped, every cached block holds its hash\'s bytes')
        self.assertEqual(stats['tier_digest_failures'], 0)

    def test_the_hard_maximum_still_cuts_and_counts_it_as_the_cap_did(self):
        made, _a, _f = self.displaced(dict({CAP: '16', HARD: '20'}, **COMPLETE))
        stats = made.registry.stats
        self.assertGreater(stats['tier_spill_dropped_cap'], 0)
        self.assertLessEqual(stats['tier_spill_max_blocks'], 20)
        self.assertTrue(made.check_kv())

    def test_a_flush_past_the_warning_bound_is_counted_and_named(self):
        made = tiers.Rig(tier=400, num_blocks=300, budget=1 << 20, tier_env=dict({CAP: '16', WARN: '5'}, **COMPLETE))
        self.addCleanup(made.close)
        clock = iter(i * 0.010 for i in range(100000))                  # every perf_counter call is 10 ms later: each flush reads as slow
        with mock.patch.object(patch_module.time, 'perf_counter', side_effect=lambda: next(clock)):
            made.serve(Request('a1', tiers.span(8200, 1), salt=tiers.salt_of('alpha')))
            made.serve(Request('f1', tiers.span(14000, 2), salt=tiers.salt_of('bravo')))
        stats = made.registry.stats
        self.assertGreater(stats['tier_spill_slow'], 0)
        slow = [line for line in made.lines if line.startswith('host tier spill slow')]
        self.assertEqual(len(slow), stats['tier_spill_slow'])
        self.assertRegex(slow[0], r'^host tier spill slow blocks=\d+ ms=[0-9.]+ warn_ms=5 batches=\d+$')

    def test_a_host_short_of_memory_still_spills_nothing(self):
        made = tiers.Rig(tier=64, num_blocks=tiers.SMALL_POOL, budget=1 << 20,
                         tier_env=dict(COMPLETE, QWEN_PREFIX_HOST_TIER_MIN_AVAILABLE_GIB='1'))
        self.addCleanup(made.close)
        with mock.patch.object(registry_module, 'host_gauges', return_value={'host_mem_available_bytes': 1 << 20}):
            tiers.two_turn_scenario(made)
        stats = made.registry.stats
        self.assertEqual(stats['tier_spill_blocks'], 0)
        self.assertGreater(stats['tier_spill_dropped_governor'], 0)


class MarkerTests(unittest.TestCase):
    def test_the_gate_reads_the_new_lines_and_the_old_one_with_its_new_fields(self):
        lines = ['2026-10-19T10:00:00.000000000Z [PINDIAG] prefix: host tier spill complete on cap_blocks=512 hard_max_blocks=4096 batch_blocks=512 warn_ms=1000',
                 '2026-10-19T10:00:01.000000000Z [PINDIAG] prefix: host tier spill blocks=3905 bytes=8701214720 ms=6320.1 held=25769410560 batches=8 lifted=3393',
                 '2026-10-19T10:00:02.000000000Z [PINDIAG] prefix: host tier spill slow blocks=3905 ms=6320.1 warn_ms=1000 batches=8',
                 '2026-10-19T10:00:03.000000000Z [PINDIAG] prefix: host tier spill blocks=1 bytes=2228224 ms=2.4 held=2228224']
        scanned = prefix_markers.scan(lines)
        self.assertEqual([(entry['hard_max_blocks'], entry['batch_blocks']) for entry in scanned['tier_spill_complete']], [(4096, 512)])
        self.assertEqual([(entry['blocks'], entry.get('batches'), entry.get('lifted')) for entry in scanned['tier_spills']], [(3905, 8, 3393), (1, None, None)])
        self.assertEqual([(entry['blocks'], entry['warn_ms']) for entry in scanned['tier_spill_slow']], [(3905, 1000)])

    def test_a_scan_of_an_engine_without_the_mode_has_empty_lists(self):
        scanned = prefix_markers.scan(['[PINDIAG] prefix: host tier spill blocks=1 bytes=2228224 ms=2.4 held=2228224'])
        self.assertEqual((scanned['tier_spill_complete'], scanned['tier_spill_slow'], len(scanned['tier_spills'])), ([], [], 1))


class ContractTests(unittest.TestCase):
    def problems(self, gate=False, **env):
        base = {'QWEN_PREFIX_REUSE': '1', 'QWEN_PREFIX_STORE_GIB': '8', 'QWEN_PREFIX_HOST_TIER_GIB': '32'}
        base.update(env)
        profile = {'name': 'x-gate' if gate else 'x', 'env': base}
        if gate:
            profile['gate_only'] = True
        return contract.prefix_policy_problems(profile)

    def test_the_names_agree_with_the_registry(self):
        self.assertEqual(contract.PREFIX_SPILL_MODE, registry_module.ENV_TIER_SPILL_MODE)
        self.assertEqual(set(contract.PREFIX_SPILL_KNOBS), {registry_module.ENV_TIER_SPILL_HARD_MAX, registry_module.ENV_TIER_SPILL_BATCH, registry_module.ENV_TIER_SPILL_WARN_MS})
        self.assertEqual(contract.PREFIX_POLICY_FLAGS[MODE], registry_module.TIER_SPILL_MODES)
        self.assertIn((MODE, 'complete'), contract.PREFIX_GATE_ONLY_VALUES)
        for name in (MODE,) + contract.PREFIX_SPILL_KNOBS:
            self.assertIn(name, contract.PREFIX_TIER_ENV_NAMES, 'an inherited value is dropped under a profile that does not name it')

    def test_a_gate_only_profile_may_name_the_mode_and_its_knobs(self):
        self.assertEqual(self.problems(gate=True, **COMPLETE), [])
        self.assertEqual(self.problems(gate=True, **{MODE: 'complete', HARD: '4096', BATCH: '512', WARN: '1000'}), [])
        self.assertEqual(self.problems(gate=True, **{MODE: 'cap'}), [])

    def test_a_traffic_profile_may_not_name_it_active(self):
        problems = self.problems(**COMPLETE)
        self.assertTrue([problem for problem in problems if MODE in problem], problems)
        self.assertEqual(self.problems(**{MODE: 'cap'}), [problem for problem in self.problems() ])

    def test_malformed_settings_are_named(self):
        self.assertTrue([p for p in self.problems(gate=True, **{MODE: 'all'}) if MODE in p])
        self.assertTrue([p for p in self.problems(gate=True, **{BATCH: '256'}) if 'set without' in p])
        self.assertTrue([p for p in self.problems(gate=True, **{MODE: 'complete', BATCH: '2.5'}) if 'plain digits' in p])
        no_tier = {'QWEN_PREFIX_REUSE': '1', MODE: 'complete'}
        problems = contract.prefix_policy_problems({'name': 'x-gate', 'gate_only': True, 'env': no_tier})
        self.assertTrue([p for p in problems if 'needs the host KV tier' in p], problems)


if __name__ == '__main__':
    unittest.main()
