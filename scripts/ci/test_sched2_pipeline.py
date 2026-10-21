"""tp4/sched-2, NP3 (docs/tp4-sched2.md): admission pipelining and the memory-ledger trim, on the CPU.

Three switches, each default off, and the claim of each is the same: the decisions and the device work are the ones of the eager order and only WHEN something happens moves.
    QWEN_FAST_ADMISSION_HOLD_REUSE   the DRAM hold reuses a recent allocator reading: the verdicts equal the real reading's wherever the allocator has not moved, and a reading
                                     that is old, over-used, tight or taken under another parked-set signature is read again
    QWEN_FAST_ADMISSION_LEDGER_TRIM  the memory ledger ends after the attach: no bracket, no admission point
    QWEN_FAST_ADMISSION_SEED_FIRST   the sampler returns the seed before the bind; the schedule() wrapper binds first; the result, the bridge call, the hook and the final state are the eager ones

    nice python3 -B -m unittest test_sched2_pipeline      (from scripts/ci)
"""

import json
import os
from contextlib import nullcontext
from pathlib import Path
import random
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent / 'speculative-decoding/harness'))

import admission_pipeline as pipeline  # noqa: E402
import profile_twins  # noqa: E402
import memory_ledger  # noqa: E402
import serving_c2_contract as contract  # noqa: E402
import serving_prefill_admission as admission  # noqa: E402
import serving_request_factory  # noqa: E402
import serving_runtime  # noqa: E402
import test_serving_extent_memory  # noqa: E402,F401  (module import, so that its TestCases are not collected here again)
import test_serving_lifecycle  # noqa: E402,F401
from serving_lifecycle import FastServingLifecycle  # noqa: E402
import test_serving_fast_policy  # noqa: E402  (module imports throughout: a TestCase class imported by name would be collected here again)
import test_serving_request_factory  # noqa: E402
import test_serving_worker_hook  # noqa: E402
from test_serving_extent_memory import MB  # noqa: E402


RESERVE = 256 * 2 ** 20
PREFIX = 'c2-packed-tp4-8x262k-ship-prefix-levern-w2-er-fx-traffic-s5'


def environ(**extra):
    base = {name: value for name, value in os.environ.items() if name not in pipeline.NAMES + (admission.LEAN_FLAG, 'QWEN_FAST_EXTENT_REPLAY')}
    base.update(extra)
    return patch.dict(os.environ, base, clear=True)


class Clock(object):
    def __init__(self, now=1000.0):
        self.now = now

    def __call__(self):
        return self.now


class Fresh(unittest.TestCase):
    def setUp(self):
        pipeline.reset()
        admission.forget_readings()
        self.addCleanup(pipeline.reset)
        self.addCleanup(admission.forget_readings)


class CountingPool(object):
    """A pool whose DRAM and trace statistics count their calls: the allocator reads the lever removes."""

    def __init__(self, free=4_800 * MB, largest=4_700 * MB, trace=148 * MB, chips=4):
        self.free, self.largest, self.trace, self.chips = free, largest, trace, chips
        self.dram_calls = self.trace_calls = 0

    def dram_statistics(self):
        self.dram_calls += 1
        return [dict(chip=index, free=self.free, largest_free=self.largest, allocated=0, total=self.free) for index in range(self.chips)]

    def trace_statistics(self):
        self.trace_calls += 1
        return [dict(chip=index, largest_free=self.trace, allocated=0) for index in range(self.chips)]


class FlagTests(Fresh):
    def test_the_switches_are_off_by_default_and_read_as_one_only(self):
        for check in (pipeline.seed_first_enabled, pipeline.hold_reuse_enabled, pipeline.ledger_trim_enabled):
            self.assertIs(check({}), False)
        for name, check in ((pipeline.SEED_FLAG, pipeline.seed_first_enabled), (pipeline.HOLD_FLAG, pipeline.hold_reuse_enabled), (pipeline.LEDGER_FLAG, pipeline.ledger_trim_enabled)):
            for value in ('0', '', 'true', '2', ' 1'):
                self.assertIs(check({name: value}), False, (name, value))
            self.assertIs(check({name: '1'}), True)
        self.assertIs(pipeline.any_enabled({}), False)
        self.assertIs(pipeline.any_enabled({pipeline.LEDGER_FLAG: '1'}), True)

    def test_every_module_that_names_a_switch_names_the_same_one(self):
        self.assertEqual(admission.HOLD_REUSE_FLAG, pipeline.HOLD_FLAG)
        self.assertEqual(serving_request_factory.LEDGER_TRIM_FLAG, pipeline.LEDGER_FLAG)
        self.assertEqual(tuple(serving_runtime.ADMISSION_PIPELINE_FLAGS), pipeline.SWITCHES)
        self.assertEqual(tuple(contract.SCHED2_SWITCHES), pipeline.SWITCHES)
        self.assertEqual(set(contract.SCHED2_HOLD_KNOBS), set(pipeline.HOLD_KNOBS))
        for name, (low, high) in contract.SCHED2_HOLD_KNOBS.items():
            self.assertEqual((low, high), pipeline.HOLD_DEFAULTS[name][1:], name)

    def test_problems_names_every_malformed_setting_and_every_missing_dependency(self):
        ok = {pipeline.HOLD_FLAG: '1', 'QWEN_FAST_EXTENT_REPLAY': '1'}
        self.assertEqual(pipeline.problems({}), [])
        self.assertEqual(pipeline.problems(ok), [])
        self.assertEqual(pipeline.problems({pipeline.LEDGER_FLAG: '1'}), [])
        for name in pipeline.SWITCHES:
            self.assertIn('must be 0 or 1', ' '.join(pipeline.problems({name: 'yes'})))
        for name, bad in ((pipeline.HOLD_AGE_FLAG, '50'), (pipeline.HOLD_AGE_FLAG, '60001'), (pipeline.HOLD_REUSES_FLAG, '0'), (pipeline.HOLD_REUSES_FLAG, '17'),
                          (pipeline.HOLD_DRIFT_FLAG, '15'), (pipeline.HOLD_DRIFT_FLAG, '1025'), (pipeline.HOLD_DRIFT_FLAG, '1.5'), (pipeline.HOLD_DRIFT_FLAG, '-3')):
            self.assertIn('whole number', ' '.join(pipeline.problems(dict(ok, **{name: bad}))), (name, bad))
        for name in pipeline.HOLD_KNOBS:
            self.assertIn('would be ignored', ' '.join(pipeline.problems({name: '200'})), name)
        self.assertIn('QWEN_FAST_EXTENT_REPLAY', ' '.join(pipeline.problems({pipeline.HOLD_FLAG: '1'})))
        seed = {pipeline.SEED_FLAG: '1'}
        self.assertEqual(len(pipeline.problems(seed)), 3)
        self.assertEqual(pipeline.problems(dict(seed, QWEN_FAST_PARKED_ENGINES='1', QWEN_FAST_ADMISSION_LEAN='1', QWEN_FAST_LEVER_N='1')), [])

    def test_the_profile_contract_reads_the_same_verdicts(self):
        cases = [{}, {pipeline.LEDGER_FLAG: '1'}, {pipeline.HOLD_FLAG: '1'}, {pipeline.HOLD_FLAG: '1', 'QWEN_FAST_EXTENT_REPLAY': '1'},
                 {pipeline.HOLD_FLAG: '1', 'QWEN_FAST_EXTENT_REPLAY': '1', pipeline.HOLD_AGE_FLAG: '500', pipeline.HOLD_DRIFT_FLAG: '128'},
                 {pipeline.HOLD_AGE_FLAG: '500'}, {pipeline.HOLD_FLAG: '1', 'QWEN_FAST_EXTENT_REPLAY': '1', pipeline.HOLD_REUSES_FLAG: '0'},
                 {pipeline.SEED_FLAG: '1'}, {pipeline.SEED_FLAG: '1', 'QWEN_FAST_PARKED_ENGINES': '1', 'QWEN_FAST_ADMISSION_LEAN': '1', 'QWEN_FAST_LEVER_N': '1'},
                 {pipeline.SEED_FLAG: 'maybe'}, {pipeline.LEDGER_FLAG: '2'}]
        for env in cases:
            process = bool(pipeline.problems(env))
            profile = bool(contract.sched2_problems({'name': 'x-gate', 'gate_only': True, 'env': dict(env)}))
            self.assertEqual(process, profile, env)

    def test_a_traffic_profile_carries_none_of_the_switches_whatever_its_waiver_names(self):
        base = {'name': 'traffic', 'env': {'QWEN_FAST_EXTENT_REPLAY': '1', 'QWEN_FAST_PARKED_ENGINES': '1', 'QWEN_FAST_ADMISSION_LEAN': '1', 'QWEN_FAST_LEVER_N': '1'}}
        self.assertEqual(contract.sched2_problems(base), [])
        for name in pipeline.SWITCHES:
            profile = json.loads(json.dumps(base))
            profile['env'][name] = '1'
            self.assertTrue(any('gate arm' in problem for problem in contract.sched2_problems(profile)), name)
            gate = dict(profile, gate_only=True)
            self.assertEqual(contract.sched2_problems(gate), [], name)
            profile['env'][name] = '0'
            self.assertEqual(contract.sched2_problems(profile), [], 'an explicit 0 is a traffic profile\'s too')

    def test_the_ship_5_candidate_and_every_profile_that_exists_name_none_of_them(self):
        import make_s7_profiles

        data = json.loads((HERE / 'qwen_c2_profiles.json').read_text(encoding='utf-8'))
        profiles = data['profiles']
        self.assertIn(PREFIX, profiles)
        for name, profile in profiles.items():
            if name in profile_twins.twin_names() and name.startswith('c2-packed-tp4-8x262k-ship-prefix-levern-w2-er-sch-s5-'):
                continue
            if name in make_s7_profiles.owned_names():
                continue        # (tp4/s7: the ship-7 candidate carries seed first and the ledger trim under its waiver, its audited twins seed first: test_s7_profiles)
            env = profile.get('env') or {}
            self.assertEqual([flag for flag in pipeline.NAMES if flag in env], [], name)


class HoldReadingTests(Fresh):
    DRIFT = 256

    def state(self, **kw):
        self.clock = Clock()
        return pipeline.HoldReadings(clock=self.clock, **kw)

    def reading(self, free=4_800, largest=4_700, trace=148):
        return dict(free=free * MB, largest_free=largest * MB, trace_largest_free=trace * MB)

    def clears(self, free_need=1_000, contiguous=700):
        def short_of(r):
            return admission.split_short(r['free'], r['largest_free'], free_need * MB, RESERVE, r['trace_largest_free'], contiguous=contiguous * MB, trace_need=0)
        return short_of

    def test_nothing_kept_nothing_reused(self):
        state = self.state()
        self.assertIsNone(state.lookup((), self.clears()))

    def test_a_young_wide_reading_is_reused_and_counts_each_use(self):
        state = self.state()
        state.store(self.reading(), ('a',))
        self.clock.now += 2.0
        reading, meta = state.lookup(('a',), self.clears())
        self.assertEqual((reading['free'], reading['largest_free']), (4_800 * MB, 4_700 * MB), 'the reading as taken, not the discounted one')
        self.assertEqual(meta['reuses'], 1)
        self.assertAlmostEqual(meta['age_ms'], 2000.0)
        self.assertEqual(meta['read_at'], 1000.0)
        self.assertEqual(state.lookup(('a',), self.clears())[1]['reuses'], 2)
        self.assertEqual(state.lookup(('a',), self.clears())[1]['reuses'], 3)
        self.assertIsNone(state.lookup(('a',), self.clears()), 'the fourth hold on one reading reads (the default limit is three)')
        self.assertEqual((state.reads, state.reused), (1, 3))

    def test_a_new_reading_starts_the_count_again(self):
        state = self.state(max_reuses=1)
        state.store(self.reading(), ())
        self.assertIsNotNone(state.lookup((), self.clears()))
        self.assertIsNone(state.lookup((), self.clears()))
        state.store(self.reading(), ())
        self.assertIsNotNone(state.lookup((), self.clears()))

    def test_an_old_reading_is_read_again(self):
        state = self.state(max_age_ms=5000)
        state.store(self.reading(), ())
        self.clock.now += 5.0
        self.assertIsNotNone(state.lookup((), self.clears()))
        self.clock.now += 0.001
        self.assertIsNone(state.lookup((), self.clears()))

    def test_a_clock_that_ran_backwards_is_not_trusted(self):
        state = self.state()
        state.store(self.reading(), ())
        self.clock.now -= 1.0
        self.assertIsNone(state.lookup((), self.clears()))

    def test_another_signature_is_another_allocator(self):
        state = self.state()
        state.store(self.reading(), ('parked', 8))
        self.assertIsNone(state.lookup(('parked', 7), self.clears()))
        self.assertIsNotNone(state.lookup(('parked', 8), self.clears()))

    def test_the_discount_is_one_drift_more_per_use_and_the_largest_block_is_halved(self):
        # The contiguous term is 700 MB: the largest block, 4,700 MB, halves to 2,350 MB, less 256 MB for each use so far including this one. Use k needs 2,350 - 256 k >= 700:
        # k = 6 has 814 MB, k = 7 would have 558 MB.
        state = self.state(max_reuses=16)
        state.store(self.reading(largest=4_700), ())
        uses = 0
        while state.lookup((), self.clears(contiguous=700)) is not None:
            uses += 1
        self.assertEqual(uses, 6)

    def test_a_tight_reading_is_never_reused(self):
        state = self.state()
        state.store(self.reading(free=1_200, largest=1_100), ())
        self.assertIsNone(state.lookup((), self.clears(free_need=1_000, contiguous=700)))

    def test_a_trace_term_that_applies_needs_a_trace_reading(self):
        state = self.state()
        state.store(dict(free=4_800 * MB, largest_free=4_700 * MB, trace_largest_free=10 * MB), ())

        def short_of(r):
            return admission.split_short(r['free'], r['largest_free'], 1_000 * MB, RESERVE, r['trace_largest_free'], contiguous=700 * MB)

        self.assertIsNone(state.lookup((), short_of), 'a 10 MB trace block is short of the 44 MB an engine\'s traces need')

    def test_clear_forgets(self):
        state = self.state()
        state.store(self.reading(), ())
        state.clear()
        self.assertIsNone(state.lookup((), self.clears()))

    def test_the_discount_is_never_more_optimistic_than_an_allocator_that_takes_the_drift_from_both(self):
        """Randomised: an allocator model that takes at most DRIFT bytes per admission from the free total and from the largest block (and splits the block in two once, the worst a
        low-address-first allocator can do to it). Whenever a reuse says fit, the true state at that hold fits the same terms."""
        generator = random.Random(20261021)
        misses = 0
        for _case in range(400):
            free = generator.randrange(1_500, 6_000) * MB
            largest = generator.randrange(1_200, int(free // MB) + 1) * MB
            need = generator.randrange(500, 1_800) * MB
            contiguous = generator.randrange(400, 1_300) * MB
            state = self.state(max_reuses=8, drift_mb=self.DRIFT)
            state.store(dict(free=free, largest_free=largest, trace_largest_free=148 * MB), ())
            true_free, true_largest = free, largest
            for admission_index in range(10):
                taken = generator.randrange(0, self.DRIFT + 1) * MB
                true_free -= taken
                true_largest = max(true_largest // 2, true_largest - taken) if admission_index == 0 and generator.random() < 0.5 else true_largest - taken
                short_of = self.clears(free_need=need // MB, contiguous=contiguous // MB)
                hit = state.lookup((), short_of)
                if hit is None:
                    continue
                truth = admission.split_short(true_free, true_largest, need, RESERVE, 148 * MB, contiguous=contiguous, trace_need=0)
                if truth:
                    misses += 1
        self.assertEqual(misses, 0)


class HoldPredicateTests(Fresh):
    def predicate(self, pool, clock=None, parked=None, signature=None):
        return admission.dram_predicate(pool, RESERVE, parked=parked, signature=signature)

    def test_without_the_switch_every_hold_reads_and_the_detail_is_what_it_was(self):
        with environ(QWEN_FAST_EXTENT_REPLAY='1'):
            pool = CountingPool()
            admits = self.predicate(pool)
            first = admits(3_600)
            second = admits(3_600)
        self.assertEqual((pool.dram_calls, pool.trace_calls), (2, 2))
        self.assertEqual(first, second)
        self.assertEqual(sorted(first[1]), ['free', 'largest_free', 'need', 'short', 'trace_largest_free'])

    def test_with_the_switch_the_first_hold_reads_and_the_next_ones_reuse_until_the_limit(self):
        with environ(QWEN_FAST_EXTENT_REPLAY='1', **{pipeline.HOLD_FLAG: '1'}):
            pool = CountingPool()
            admits = self.predicate(pool)
            answers = [admits(3_600) for _ in range(5)]
        self.assertEqual((pool.dram_calls, pool.trace_calls), (2, 2), 'a read, three reuses, a read')
        self.assertTrue(all(ok for ok, _detail in answers))
        self.assertNotIn('read_at', answers[0][1])
        self.assertEqual([detail.get('reuses') for _ok, detail in answers], [None, 1, 2, 3, None])
        self.assertEqual(answers[1][1]['extra_allowance'], 1 * 256 * MB)
        self.assertEqual(answers[3][1]['extra_allowance'], 3 * 256 * MB)

    def test_verdicts_equal_the_real_readings_wherever_the_allocator_stands_still(self):
        """A grid of pool sizes and prompt lengths around the terms (the boundary the ballast ladder of the engine-reuse gate walks): the switch changes no verdict."""
        sizes = [(free, largest) for free in (900, 1_200, 1_500, 2_000, 3_000, 4_800) for largest in (500, 800, 1_100, 1_500, 2_500, 4_700) if largest <= free]
        for free, largest in sizes:
            for prompt in (54, 2_047, 2_048, 3_600, 65_000, 205_000):
                with environ(QWEN_FAST_EXTENT_REPLAY='1'):
                    real = self.predicate(CountingPool(free * MB, largest * MB))
                    want = [real(prompt) for _ in range(6)]
                with environ(QWEN_FAST_EXTENT_REPLAY='1', **{pipeline.HOLD_FLAG: '1'}):
                    pipeline.reset()
                    got_pool = CountingPool(free * MB, largest * MB)
                    reuse = self.predicate(got_pool)
                    got = [reuse(prompt) for _ in range(6)]
                self.assertEqual([ok for ok, _ in got], [ok for ok, _ in want], (free, largest, prompt))
                for (_ok, left), (_ok2, right) in zip(got, want):
                    self.assertEqual((left['free'], left['largest_free'], left['need'], left['short']), (right['free'], right['largest_free'], right['need'], right['short']))

    def test_a_pool_that_shrinks_below_the_terms_is_noticed_at_the_next_read(self):
        with environ(QWEN_FAST_EXTENT_REPLAY='1', **{pipeline.HOLD_FLAG: '1'}):
            pool = CountingPool(4_800 * MB, 4_700 * MB)
            admits = self.predicate(pool)
            self.assertTrue(admits(3_600)[0])
            for _ in range(3):
                self.assertTrue(admits(3_600)[0])
            pool.free, pool.largest = 600 * MB, 500 * MB
            ok, detail = admits(3_600)
        self.assertFalse(ok, 'the fifth hold reads (the limit), and sees the shrunken pool')
        self.assertIn('free', detail['short'])

    def test_a_held_prompt_reads_again_every_time(self):
        with environ(QWEN_FAST_EXTENT_REPLAY='1', **{pipeline.HOLD_FLAG: '1'}):
            pool = CountingPool(900 * MB, 800 * MB)
            admits = self.predicate(pool)
            for _ in range(4):
                self.assertFalse(admits(3_600)[0])
        self.assertEqual(pool.dram_calls, 4, 'a short reading is never kept as a clearance')

    def test_a_signature_change_forces_a_read(self):
        signature = {'value': ('a',)}
        with environ(QWEN_FAST_EXTENT_REPLAY='1', **{pipeline.HOLD_FLAG: '1'}):
            pool = CountingPool()
            admits = self.predicate(pool, signature=lambda: signature['value'])
            admits(3_600)
            admits(3_600)
            signature['value'] = ('b',)
            admits(3_600)
        self.assertEqual(pool.dram_calls, 2)

    def test_the_knobs_of_the_process_are_the_ones_the_state_is_built_from(self):
        with environ(QWEN_FAST_EXTENT_REPLAY='1', **{pipeline.HOLD_FLAG: '1', pipeline.HOLD_AGE_FLAG: '2500', pipeline.HOLD_REUSES_FLAG: '5', pipeline.HOLD_DRIFT_FLAG: '64'}):
            pipeline.reset()
            state = pipeline.hold_state()
        self.assertEqual((state.max_age_s, state.max_reuses, state.drift), (2.5, 5, 64 * MB))

    def test_a_reused_reading_carries_its_age_and_its_extra_allowance_to_the_backstop(self):
        """The hold used an older reading: the backstop's reuse counts the age from the allocator read and takes the admissions since from its margin too."""
        clock = Clock()
        detail = dict(free=4_800 * MB, largest_free=4_700 * MB, trace_largest_free=148 * MB, need=1_000 * MB, short=(), read_at=clock.now - 9.0, reuses=3, extra_allowance=3 * 256 * MB)
        with environ(**{admission.LEAN_FLAG: '1'}):
            admission.note_reading('r', detail, clock=clock)
            need = admission.parked_backstop_need(RESERVE, rebind=admission.PARKED_REBIND_BYTES, single=0)
            self.assertTrue(admission.reading_clears('r', need, RESERVE, trace_need=0, clock=clock), 'does not use it up')
            kept = admission.reusable_reading('r', need, RESERVE, trace_need=0, clock=clock)
            self.assertIsNotNone(kept)
            self.assertAlmostEqual(kept['age_ms'], 9000.0)
            narrow = dict(detail, largest_free=2_000 * MB)
            admission.note_reading('s', narrow, clock=clock)
            self.assertIsNone(admission.reusable_reading('s', need, RESERVE, trace_need=0, clock=clock), '2,000 MB halves to 1,000, less 512 + 768 is under zero')

    def test_a_reading_without_a_reuse_is_kept_exactly_as_before(self):
        clock = Clock()
        with environ(**{admission.LEAN_FLAG: '1'}):
            admission.note_reading('r', dict(free=4_800 * MB, largest_free=4_700 * MB, trace_largest_free=148 * MB), clock=clock)
            kept = admission._LEAN_READINGS['r']
        self.assertEqual(sorted(kept), ['at', 'free', 'largest_free', 'trace_largest_free'])
        self.assertEqual(kept['at'], clock.now)

    def test_reading_clears_is_false_without_the_lean_flag_or_a_reading_and_never_uses_one_up(self):
        clock = Clock()
        need = admission.backstop_need(RESERVE)
        with environ():
            self.assertFalse(admission.reading_clears('r', need, RESERVE, clock=clock))
        with environ(**{admission.LEAN_FLAG: '1'}):
            self.assertFalse(admission.reading_clears('absent', need, RESERVE, clock=clock))
            admission.note_reading('r', dict(free=4_800 * MB, largest_free=4_700 * MB, trace_largest_free=148 * MB), clock=clock)
            self.assertTrue(admission.reading_clears('r', need, RESERVE, clock=clock))
            self.assertTrue(admission.reading_clears('r', need, RESERVE, clock=clock))
            clock.now += admission.LEAN_READING_MAX_AGE_S + 1
            self.assertFalse(admission.reading_clears('r', need, RESERVE, clock=clock))

    def test_parked_signature_names_the_states_the_reading_did_not_see(self):
        class Entry(object):
            def __init__(self, state):
                self.state = state

        parked = SimpleNamespace(closed=False, off=False, fallbacks=0, unparks=0, reparks=0, slots=[Entry('parked'), Entry('serving')], single_released=lambda entry: False)
        first = pipeline.parked_signature(parked)
        self.assertEqual(first, pipeline.parked_signature(parked))
        parked.slots[1].state = 'parked'
        self.assertNotEqual(first, pipeline.parked_signature(parked))
        parked.slots[1].state = 'serving'
        parked.fallbacks = 1
        self.assertNotEqual(first, pipeline.parked_signature(parked))
        parked.fallbacks = 0
        parked.reparks = 1
        self.assertNotEqual(first, pipeline.parked_signature(parked))
        self.assertEqual(pipeline.parked_signature(None), ())

    def test_an_unreadable_parked_set_is_never_the_same_twice(self):
        broken = SimpleNamespace(closed=False, off=False, slots=None)
        self.assertNotEqual(pipeline.parked_signature(broken), pipeline.parked_signature(broken))


class RegisterTests(Fresh):
    def test_the_attach_builds_the_predicate_with_a_signature_only_under_the_switch(self):
        calls = []

        def fake_predicate(pool, reserve, **kw):
            calls.append(kw)
            return lambda prompt: (True, {})

        parked = SimpleNamespace(arrival_terms=lambda: None, arrival_rebind_bytes=lambda: 100 * MB, closed=False, off=False, slots=[], fallbacks=0, unparks=0, reparks=0, single_released=lambda entry: False)
        with patch.object(admission, 'dram_predicate', fake_predicate), patch.object(admission, 'dram_reading', lambda pool: (None, 'x')):
            with environ():
                serving_request_factory.register_dram_admission(CountingPool(), log=lambda *a: None)()
                serving_request_factory.register_dram_admission(CountingPool(), log=lambda *a: None, parked=parked)()
            with environ(**{pipeline.HOLD_FLAG: '1'}):
                serving_request_factory.register_dram_admission(CountingPool(), log=lambda *a: None, parked=parked)()
                serving_request_factory.register_dram_admission(CountingPool(), log=lambda *a: None)()
        self.assertEqual([sorted(kw) for kw in calls], [[], ['parked'], ['parked', 'signature'], ['signature']])
        self.assertEqual(calls[2]['signature'](), pipeline.parked_signature(parked))


class LedgerTrimTests(Fresh):
    def setUp(self):
        super().setUp()
        serving_request_factory._REBIND_BRACKETED.clear()
        self.addCleanup(serving_request_factory._REBIND_BRACKETED.clear)
        self.addCleanup(memory_ledger.end)

    def test_the_bracket_is_never_taken_under_the_trim_and_is_a_slots_first_under_lean(self):
        with environ():
            self.assertFalse(serving_request_factory.lean_bracket_skipped(0), 'neither flag: every rebind is bracketed')
        with environ(**{admission.LEAN_FLAG: '1'}):
            self.assertFalse(serving_request_factory.lean_bracket_skipped(3))
            self.assertTrue(serving_request_factory.lean_bracket_skipped(3))
        serving_request_factory._REBIND_BRACKETED.clear()
        with environ(**{admission.LEAN_FLAG: '1', pipeline.LEDGER_FLAG: '1'}):
            for slot in (0, 3, 3, 7):
                self.assertTrue(serving_request_factory.lean_bracket_skipped(slot), slot)
            self.assertEqual(serving_request_factory._REBIND_BRACKETED, set(), 'nothing is recorded')
        with environ(**{pipeline.LEDGER_FLAG: '1'}):
            self.assertTrue(serving_request_factory.lean_bracket_skipped(1), 'the trim alone is enough')

    def test_trim_ledger_ends_an_active_ledger_once_and_every_reading_after_it_is_a_no_op(self):
        ledger = memory_ledger.begin(SimpleNamespace(), None, log=lambda *_: None, emit=lambda *_: None) if False else None
        fake = SimpleNamespace(phase=Mock(return_value='read'), before=Mock(return_value='token'), after=Mock())
        memory_ledger._active = fake
        lines = []
        self.assertEqual(memory_ledger.record('P7', point='after_attach'), 'read')
        self.assertTrue(pipeline.trim_ledger(memory_ledger, log=lambda *a: lines.append(a)))
        self.assertIsNone(memory_ledger.active())
        self.assertIsNone(memory_ledger.record('prefill', point='before'))
        self.assertIsNone(memory_ledger.before('rebind', estimate=1))
        self.assertIsNone(memory_ledger.after('token'))
        self.assertIsNone(memory_ledger.engine_admitted('r'))
        self.assertIsNone(memory_ledger.first_packed_round())
        self.assertIsNone(memory_ledger.record('P13', point='before_shutdown'))
        fake.phase.assert_called_once()
        self.assertEqual(lines, [(pipeline.LEDGER_TRIM_LINE,)])
        self.assertFalse(pipeline.trim_ledger(memory_ledger, log=lambda *a: lines.append(a)), 'nothing left to end')
        self.assertEqual(len(lines), 1)


def lifecycle_world(seed_first=None, scheduler=None, bridge_ms=0.0, clock=None):
    """test_serving_lifecycle's fixture, with a seed-first policy and a scheduler class the wrapper can install on."""
    worker, bridge, _, decode = test_serving_worker_hook.WorkerHookTests().fixture()
    worker.model_runner.execute_model.return_value = None
    worker.model_runner.sample_tokens.return_value = SimpleNamespace(req_ids=['request'], sampled_token_ids=[[10]])
    _, _, _, arguments = test_serving_request_factory.RequestFactoryTests().fixture()
    new = SimpleNamespace(req_id='request', prompt_token_ids=[1] * 4096, num_computed_tokens=0, mm_features=[], prompt_embeds=None, lora_request=None,
                          sampling_params=arguments['state'].sampling_params)
    scheduled = SimpleNamespace(finished_req_ids=set(), scheduled_new_reqs=[new], scheduled_cached_reqs=SimpleNamespace(req_ids=[]), scheduled_spec_decode_tokens={},
                                num_scheduled_tokens={'request': 4096}, total_num_scheduled_tokens=4096)
    from contextlib import nullcontext

    capture = SimpleNamespace(capture=Mock(side_effect=lambda: nullcontext()), close=Mock())
    events = []

    def factory(state, features):
        events.append(('bridge', clock.now if clock is not None else None))
        if clock is not None:
            clock.now += bridge_ms / 1000.0
        features.close()
        return bridge

    build = Mock(side_effect=factory)
    scheduler = scheduler or type('Scheduler', (), {'schedule': lambda self: events.append(('schedule', clock.now if clock is not None else None)) or 'scheduled'})
    with patch.object(FastServingLifecycle, '_install_admission', staticmethod(lambda config: 'tests.Scheduler')), \
            patch.object(FastServingLifecycle, '_install_quarantine', staticmethod(lambda config: None)), \
            patch.dict(os.environ, {'QWEN_FAST_ANY_REQUEST': '1'}), \
            patch('serving_request_quarantine.scheduler_class', lambda config, importer=None: scheduler):
        lifecycle = FastServingLifecycle(worker, config=test_serving_fast_policy.FastPolicyTests().fixture(), capture_factory=Mock(return_value=capture), bridge_factory=build, eos_ids=(99,),
                                         cancelled=lambda: False, **({} if seed_first is None else dict(seed_first=seed_first)))
    return lifecycle, worker, bridge, capture, build, scheduled, decode, scheduler, events


def policy(parked_entry=SimpleNamespace(index=2), clears=True, clock=None, log=None, **parked_extra):
    parked = SimpleNamespace(closed=False, off=False, fault=None, peek=lambda: parked_entry, **parked_extra)
    seed = pipeline.SeedFirst(parked, lambda request_id, need: clears, lambda: 1_000 * MB, log=log or (lambda *a: None), clock=clock or Clock())
    return seed, parked


class SeedFirstPolicyTests(Fresh):
    def test_it_defers_only_a_rebind_whose_cold_fallback_clears(self):
        state = SimpleNamespace(req_id='r1')
        seed, parked = policy()
        self.assertIsNone(seed.refusal(state))
        self.assertTrue(seed.defer(state.req_id, state))
        self.assertEqual(seed.deferred, 1)

    def test_every_reason_to_stay_in_the_sampler_is_named_and_counted(self):
        state = SimpleNamespace(req_id='r1')
        lines = []
        cases = [(dict(parked_entry=None), 'cold build'), (dict(clears=False), 'does not clear'), (dict(), None)]
        seed, parked = policy(parked_entry=None, log=lambda *a: lines.append(a))
        self.assertFalse(seed.defer(state.req_id, state))
        parked.peek = lambda: SimpleNamespace(index=1)
        parked.off = True
        self.assertFalse(seed.defer(state.req_id, state))
        parked.off = False
        parked.closed = True
        self.assertFalse(seed.defer(state.req_id, state))
        parked.closed = False
        parked.fault = 'rebind'
        self.assertFalse(seed.defer(state.req_id, state))
        parked.fault = None
        seed.clears = lambda request_id, need: False
        self.assertFalse(seed.defer(state.req_id, state))
        seed.clears = Mock(side_effect=RuntimeError('x'))
        self.assertFalse(seed.defer(state.req_id, state))
        parked.peek = Mock(side_effect=RuntimeError('y'))
        seed.clears = lambda request_id, need: True
        self.assertFalse(seed.defer(state.req_id, state))
        self.assertEqual(seed.deferred, 0)
        self.assertEqual(sum(seed.eager.values()), 7)
        self.assertEqual(len(seed.eager), 6, 'a closed set and an off one are the same reason')
        self.assertTrue(all(line[0] == pipeline.SEED_EAGER_LINE for line in lines))

    def test_the_clearance_is_asked_with_the_request_and_the_cold_fallbacks_need(self):
        seen = []
        seed, _ = policy()
        seed.clears = lambda request_id, need: seen.append((request_id, need)) or True
        seed.defer('abc', SimpleNamespace())
        self.assertEqual(seen, [('abc', 1_000 * MB)])

    def test_install_registers_the_completion_and_wraps_schedule_once(self):
        modules = {}
        done = []
        scheduler = type('S', (), {'schedule': lambda self, *a, **k: done.append(('schedule', a, k)) or 'out'})
        seed, _ = policy()
        lifecycle = SimpleNamespace(complete_deferred=lambda: done.append('complete'))
        seed.install(lifecycle, 'config', modules=modules, resolve=lambda config: scheduler)
        first = scheduler.schedule
        seed.install(lifecycle, 'config', modules=modules, resolve=lambda config: scheduler)
        self.assertIs(scheduler.schedule, first, 'wrapped once')
        self.assertEqual(scheduler().schedule(1, flag=2), 'out')
        self.assertEqual(done, ['complete', ('schedule', (1,), {'flag': 2})], 'the completion runs first')
        seed.uninstall()
        self.assertNotIn(pipeline.HOLDER_KEY, modules)
        done.clear()
        scheduler().schedule()
        self.assertEqual(done, [('schedule', (), {})], 'a wrapper left on the class calls nothing once the lifecycle has gone')

    def test_a_scheduler_without_schedule_is_refused_by_name(self):
        seed, _ = policy()
        with self.assertRaisesRegex(ValueError, 'schedule'):
            seed.install(SimpleNamespace(complete_deferred=lambda: None), 'config', modules={}, resolve=lambda config: object)

    def test_complete_pending_with_nothing_registered_does_nothing(self):
        pipeline.complete_pending({})
        pipeline.complete_pending({pipeline.HOLDER_KEY: SimpleNamespace()})


class SeedFirstLifecycleTests(Fresh):
    def drive(self, seed_first, **kw):
        world = lifecycle_world(seed_first, **kw)
        lifecycle, worker, bridge, capture, build, scheduled, decode, scheduler, events = world
        self.addCleanup(lambda: lifecycle.closed or lifecycle.close())
        self.assertIsNone(worker.execute_model(scheduled))
        result = worker.sample_tokens(None)
        return world + (result,)

    def script(self, seed_first):
        """The engine's whole story of one request, step by step: prefill, seed, the next scheduling call, the drafts, one decode round, the finish, the close. Everything observable."""
        from types import ModuleType

        lifecycle, worker, bridge, capture, build, scheduled, decode, scheduler, events, seed = self.drive(seed_first)
        scheduler().schedule()
        outputs = ModuleType('vllm.v1.outputs')
        outputs.ModelRunnerOutput = outputs.DraftTokenIds = SimpleNamespace
        with patch.dict('sys.modules', {'vllm.v1.outputs': outputs}):
            drafts = worker.take_draft_token_ids().draft_token_ids
            result = worker.execute_model(decode)
        finished = SimpleNamespace(finished_req_ids={'request'}, total_num_scheduled_tokens=0)
        worker.execute_model(finished)
        after_finish = dict(closed=bridge.request.closed, hook=lifecycle.hook, request_id=lifecycle.request_id, decoding=list(lifecycle.decoding_ids), draft_none=worker.take_draft_token_ids())
        lifecycle.close()
        return dict(seed=seed.sampled_token_ids, drafts=drafts, decode=result.sampled_token_ids, after_finish=after_finish, capture_close=capture.close.call_count,
                    builds=build.call_count, restored=not hasattr(worker, '_qwen_fast_lifecycle'), failed=lifecycle.failed)

    def test_the_whole_story_of_a_request_is_the_same_with_the_seed_first_as_without(self):
        eager = self.script(None)
        seed, _ = policy()
        lazy = self.script(seed)
        self.assertEqual(seed.deferred, 1)
        self.assertEqual(eager, lazy)
        self.assertEqual(eager['builds'], 1)
        self.assertEqual(eager['seed'], [[10]])

    def test_the_seed_goes_out_before_the_bind_and_the_wrapper_binds_before_the_next_schedule(self):
        seed, _ = policy()
        lifecycle, worker, bridge, capture, build, scheduled, decode, scheduler, events, result = self.drive(seed)
        self.assertEqual(result.sampled_token_ids, [[10]], 'the sampler\'s output, unchanged')
        build.assert_not_called()
        self.assertIsNotNone(lifecycle.deferred)
        self.assertEqual(lifecycle.request_id, 'request', 'still in its prefill phase: the gate stays held until the bind')
        self.assertIsNone(lifecycle.hook)
        self.assertEqual(lifecycle.decoding_ids, [])
        self.assertEqual(scheduler().schedule(), 'scheduled')
        build.assert_called_once()
        self.assertEqual([name for name, _at in events], ['bridge', 'schedule'], 'the bind, then the scheduler\'s own call')
        self.assertIsNone(lifecycle.deferred)
        self.assertIsNone(lifecycle.request_id)
        self.assertEqual(lifecycle.decoding_ids, ['request'])

    def test_the_bridge_factory_is_called_once_with_the_state_and_the_capture_either_way(self):
        for seed_first in (None, policy()[0]):
            lifecycle, worker, bridge, capture, build, scheduled, decode, scheduler, events, result = self.drive(seed_first)
            scheduler().schedule()
            build.assert_called_once()
            state, features = build.call_args[0]
            self.assertIs(state, bridge.state)
            self.assertIs(features, capture)

    def test_the_next_execute_binds_first_where_no_wrapper_ran(self):
        seed, _ = policy()
        lifecycle, worker, bridge, capture, build, scheduled, decode, scheduler, events, result = self.drive(seed)
        build.assert_not_called()
        from types import ModuleType

        outputs = ModuleType('vllm.v1.outputs')
        outputs.ModelRunnerOutput = outputs.DraftTokenIds = SimpleNamespace
        with patch.dict('sys.modules', {'vllm.v1.outputs': outputs}):
            result = worker.execute_model(decode)
        build.assert_called_once()
        self.assertEqual(result.sampled_token_ids, [list(range(11, 27))], 'the decode ran on the bound request')
        self.assertIsNone(lifecycle.deferred)

    def test_a_refused_defer_binds_in_the_sampler_as_before(self):
        seed, _ = policy(clears=False)
        lifecycle, worker, bridge, capture, build, scheduled, decode, scheduler, events, result = self.drive(seed)
        build.assert_called_once()
        self.assertIsNone(lifecycle.deferred)
        self.assertEqual(lifecycle.decoding_ids, ['request'])
        self.assertEqual(sum(seed.eager.values()), 1)

    def test_a_bind_that_fails_after_the_seed_stops_the_engine_and_names_the_request(self):
        from serving_request_factory import RequestRefused

        seed, _ = policy()
        lifecycle, worker, bridge, capture, build, scheduled, decode, scheduler, events, result = self.drive(seed)
        build.side_effect = RequestRefused('DRAM backstop: short')
        with self.assertRaises(RequestRefused):
            scheduler().schedule()
        self.assertTrue(lifecycle.failed)
        with self.assertRaisesRegex(ValueError, 'failed or closed'):
            worker.execute_model(decode)

    def test_closing_with_a_bind_pending_releases_the_capture_and_unregisters(self):
        seed, _ = policy()
        lifecycle, worker, bridge, capture, build, scheduled, decode, scheduler, events, result = self.drive(seed)
        self.assertIsNotNone(sys.modules.get(pipeline.HOLDER_KEY))
        lifecycle.close()
        capture.close.assert_called()
        self.assertIsNone(lifecycle.deferred)
        self.assertIsNone(sys.modules.get(pipeline.HOLDER_KEY))
        build.assert_not_called()

    def test_no_seed_first_policy_means_no_wrapper_and_the_eager_bind(self):
        lifecycle, worker, bridge, capture, build, scheduled, decode, scheduler, events = lifecycle_world(None)
        self.addCleanup(lifecycle.close)
        self.assertFalse(getattr(scheduler.schedule, pipeline.SCHEDULE_WRAPPED, False))
        worker.execute_model(scheduled)
        worker.sample_tokens(None)
        build.assert_called_once()
        self.assertIsNone(lifecycle.deferred)
        self.assertIsNone(lifecycle.seed_first)

    def test_a_lifecycle_without_the_admission_cap_refuses_the_policy_by_name(self):
        seed, _ = policy()
        worker, *_ = test_serving_worker_hook.WorkerHookTests().fixture()
        with patch.object(FastServingLifecycle, '_install_admission', staticmethod(lambda config: None)), patch.object(FastServingLifecycle, '_install_quarantine', staticmethod(lambda config: None)), \
                patch.dict(os.environ, {'QWEN_FAST_ANY_REQUEST': '1'}):
            with self.assertRaisesRegex(ValueError, 'one-fresh-prefill cap'):
                FastServingLifecycle(worker, config=test_serving_fast_policy.FastPolicyTests().fixture(), capture_factory=Mock(), bridge_factory=Mock(), eos_ids=(99,), cancelled=lambda: False, seed_first=seed)

    def test_the_first_token_leaves_earlier_by_the_bind_and_the_next_schedule_starts_when_it_did(self):
        """The timing the lever exists for, on a fake clock: the bind takes 275 ms. Eager, the sampler returns after it. Seed first, the sampler returns at once and the bind
        runs inside the next schedule() wrapper, so that schedule() begins (and the alternator's clock reads it) at the very moment the eager order's would."""
        def run(seed_first, clock):
            lifecycle, worker, bridge, capture, build, scheduled, decode, scheduler, events, result = self.drive(seed_first, clock=clock, bridge_ms=275.0)
            seed_at = clock.now
            scheduler().schedule()
            return seed_at, dict((name, at) for name, at in events), lifecycle

        eager_seed, eager_events, _ = run(None, Clock(10.0))
        lazy_clock = Clock(10.0)
        seed, _ = policy(clock=lazy_clock)
        lazy_seed, lazy_events, lifecycle = run(seed, lazy_clock)
        self.assertAlmostEqual(eager_seed - lazy_seed, 0.275, places=6, msg='the seed leaves earlier by the whole bind')
        self.assertAlmostEqual(eager_events['schedule'], lazy_events['bridge'] + 0.275, places=6)
        self.assertAlmostEqual(eager_events['schedule'], lazy_events['schedule'], places=6, msg='the scheduler\'s clock reads the same instant')
        self.assertEqual(lifecycle.decoding_ids, ['request'])

    def test_a_second_arrival_is_held_until_its_bind_and_binds_after_the_first(self):
        seed, _ = policy()
        lifecycle, worker, bridge, capture, build, scheduled, decode, scheduler, events, result = self.drive(seed)
        sampler = worker.model_runner.sample_tokens          # the hook, built at the bind, replaces the runner's
        lifecycle.packed_step = lambda entries, *, cancelled: []
        scheduler().schedule()
        second = SimpleNamespace(req_id='second', prompt_token_ids=[1] * 4096, num_computed_tokens=0, mm_features=[], prompt_embeds=None, lora_request=None,
                                 sampling_params=scheduled.scheduled_new_reqs[0].sampling_params)
        step = SimpleNamespace(finished_req_ids=set(), scheduled_new_reqs=[second], scheduled_cached_reqs=SimpleNamespace(req_ids=[]), scheduled_spec_decode_tokens={},
                               num_scheduled_tokens={'second': 4096}, total_num_scheduled_tokens=4096)
        sampler.return_value = SimpleNamespace(req_ids=['second'], sampled_token_ids=[[11]])
        worker.model_runner.requests['second'] = SimpleNamespace(output_token_ids=[11], req_id='second')
        other = SimpleNamespace(**dict(vars(bridge), request=SimpleNamespace(session=SimpleNamespace(request_id='second'), close=Mock()), close=Mock()))
        build.side_effect = lambda state, features: (features.close(), other)[1]
        self.assertIsNone(worker.execute_model(step))
        worker.sample_tokens(None)
        self.assertEqual(lifecycle.request_id, 'second', 'held until its bind')
        self.assertEqual(lifecycle.decoding_ids, ['request'])
        scheduler().schedule()
        self.assertEqual(lifecycle.decoding_ids, ['request', 'second'])
        self.assertIsNone(lifecycle.request_id)


class PacingIdentityTests(Fresh):
    """The real alternator on the plugin's scheduler over a fake clock: the step time it measures for a prefill step, the debt it owes the decoders and every decision it takes are the
    same whether the bind ran in the sampler (eager) or at the start of the next schedule() call (seed first), because the completion runs BEFORE the alternator's begin()."""

    def story(self, seed_first):
        import test_levern_scheduler as base

        clock = base.Clock()
        cls, runtime, log = base.install_levern({'QWEN_FAST_LEVER_N': '1', 'QWEN_FAST_LEVERN_PREFILL_SHARE': '0.5', 'QWEN_FAST_LEVERN_MAX_ROUNDS': '64'}, clock=clock)
        scheduler = base.new_scheduler(cls)
        for index in range(3):
            scheduler.add_decoder('d%d' % index)
        scheduler.add_request(base.Request('cold-a', 3000))
        scheduler.add_request(base.Request('cold-b', 3000))
        pending = []

        class Lifecycle(object):
            def complete_deferred(self):
                if pending:
                    pending.pop()
                    clock.advance_ms(BIND_MS)

        seed = pipeline.SeedFirst(SimpleNamespace(), lambda request_id, need: True, lambda: 1, log=lambda *a: None, clock=clock)
        if seed_first:
            seed.install(Lifecycle(), 'config', modules={}, resolve=lambda config: cls)
        seeds, decisions = [], []
        for _ in range(60):
            before = {request.request_id: request.num_computed_tokens for request in scheduler.running}
            result = scheduler.schedule()
            decisions.append((runtime.alternator.last_kind, round(runtime.alternator.owed_ms, 3), runtime.alternator.owed_rounds, round(clock.now, 6)))
            new = [request.req_id for request in result.scheduled_new_reqs]
            if new:
                clock.advance_ms(PREFILL_MS)                    # the prefill step: the seed is sampled
                if seed_first:
                    seeds.append(clock.now)                     # the first token leaves now
                    pending.append(True)                        # the bind runs inside the next schedule() wrapper
                else:
                    clock.advance_ms(BIND_MS)                   # the bind inside the sampler
                    seeds.append(clock.now)                     # the first token leaves after it
            else:
                clock.advance_ms(ROUND_MS)                      # a decode round
        return decisions, seeds, clock.now

    def test_the_alternator_sees_the_same_steps_the_same_debt_and_the_same_clock(self):
        eager = self.story(False)
        lazy = self.story(True)
        self.assertEqual(eager[0], lazy[0], 'every decision, every debt, the clock at every scheduling call')
        self.assertEqual(eager[2], lazy[2], 'the run ends at the same instant')
        self.assertEqual(len(eager[1]), 2, 'both cold prompts were prefilled')
        for early, late in zip(lazy[1], eager[1]):
            self.assertAlmostEqual(late - early, BIND_MS / 1000.0, places=6, msg='the first token leaves earlier by the bind in the seed-first order')

    def test_the_debt_is_charged_for_the_bind_either_way(self):
        """dt of the prefill step includes the bind in both orders, so the decoders are owed the same time."""
        eager, lazy = self.story(False)[0], self.story(True)[0]
        owed = [entry[1] for entry in eager]
        self.assertTrue(any(value > 0 for value in owed))
        self.assertEqual(owed, [entry[1] for entry in lazy])


PREFILL_MS, BIND_MS, ROUND_MS = 700.0, 275.0, 80.0


class AttachWiringTests(Fresh):
    """serving_runtime's attach: with none of the switches it builds the lifecycle exactly as before; with them, the policy object, the strict check and the ledger's end are wired in."""

    def attach(self, extra_env=None, parked=None, strict=True):
        import test_serving_runtime

        test = test_serving_runtime.RuntimeAttachmentTests()
        lines = []
        # (The harness's fake worker has no prefill_forward, so it cannot take Lever N: a test that needs QWEN_FAST_LEVER_N=1 for the policy's own check leaves the check out here, and
        # FlagTests holds the check itself.)
        checked = nullcontext() if strict else patch.object(pipeline, 'problems', lambda environ=None: [])
        with patch.object(pipeline, 'SeedFirst') as seed, patch.object(pipeline, 'trim_ledger') as trim, patch.object(pipeline, '_log', lambda *a: lines.append(a)), checked:
            test.exercise(extra_env=extra_env, parked=parked)
        test.installed = [line for line in lines if line[0] == pipeline.INSTALLED_LINE]
        return test, seed, trim

    def test_without_a_switch_nothing_is_built_and_nothing_trimmed(self):
        test, seed, trim = self.attach()
        seed.assert_not_called()
        trim.assert_not_called()
        self.assertEqual(test.installed, [], 'no line either')

    def test_the_ledger_trim_ends_the_ledger_once_after_the_lifecycle_is_built(self):
        test, seed, trim = self.attach(extra_env={pipeline.LEDGER_FLAG: '1'})
        trim.assert_called_once()
        self.assertIs(trim.call_args.args[0], memory_ledger)
        seed.assert_not_called()
        self.assertEqual(test.installed, [(pipeline.INSTALLED_LINE, 0, 0, 1, '-', '-', '-')])

    def test_seed_first_builds_the_policy_over_the_parked_set(self):
        env = {pipeline.SEED_FLAG: '1', 'QWEN_FAST_ADMISSION_LEAN': '1'}
        test, seed, trim = self.attach(extra_env=env, parked={}, strict=False)
        seed.assert_called_once()
        self.assertIs(seed.call_args.args[0], test.parked_sets[0])
        self.assertTrue(callable(seed.call_args.args[1]) and callable(seed.call_args.args[2]))
        trim.assert_not_called()
        self.assertEqual(test.installed, [(pipeline.INSTALLED_LINE, 1, 0, 0, '-', '-', '-')])


if __name__ == '__main__':
    unittest.main()
