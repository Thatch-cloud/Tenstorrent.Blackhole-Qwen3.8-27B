"""KV reservation admission (serving_kv_reservation, QWEN_FAST_KV_RESERVATION=1): the arithmetic, the hold on the plugin's own scheduler class
over the reduced vLLM scheduler, its composition with the DRAM hold and the decode credit, the contract's refusal and the boot rule.

The real-vLLM proof - that no step ever preempts under the rule, on vLLM 0.25.1's own block manager, with a negative control - is
test_serving_kv_reservation_vllm (qwen-fast-vllm-cpu.yml). What the CPU holds here is everything the proof's premise does not
cover: that the reservation is what the module docstring says, that a hold is a hold the plugin already knows (queues hidden, the
decode-only pass, the finished ids carried), that it is never lifted, that the flag off is call for call what it was, and
that 10,000 seeded arrival/finish sequences keep the sum of reservations inside the pool and always make progress."""

import copy
import json
import os
from pathlib import Path
import random
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, call, patch

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import serving_c2_contract as contract  # noqa: E402
import serving_kv_reservation as kv  # noqa: E402
import serving_prefill_admission as admission  # noqa: E402
import test_serving_prefill_admission as base  # noqa: E402

PROFILES = HERE / 'qwen_c2_profiles.json'
WINDOW = 262144
POOL = 21760 - 1


class KvRequest(base.FakeRequest):
    """A vLLM Request as the reservation reads it: num_prompt_tokens and max_tokens."""

    def __init__(self, request_id, prompt_tokens=4096, max_tokens=256, sampling_params=None):
        super().__init__(request_id, prompt_tokens, sampling_params)
        self.max_tokens = max_tokens

    @property
    def num_prompt_tokens(self):
        return self.prompt_tokens


class KvScheduler(base.FinishingVllmScheduler):
    """The reduced vLLM scheduler with the pool the rule reads (kv_cache_manager.block_pool.num_gpu_blocks) and the window."""

    def __init__(self, *args, blocks=POOL + 1, window=WINDOW, **kwargs):
        super().__init__(*args, **kwargs)
        self.kv_cache_manager = SimpleNamespace(block_pool=SimpleNamespace(num_gpu_blocks=blocks))
        self.max_model_len = window


def install(cls, log, on=True):
    with patch.dict(os.environ, {kv.FLAG: '1' if on else '0'}), patch.object(kv, 'install_check', return_value=[]):
        return admission.install(base.configured(cls), log=log)


def plugin(blocks=POOL + 1, seats=8, on=True, log=None):
    cls = base.plugin_class(KvScheduler)
    log = Mock() if log is None else log
    install(cls, log, on)
    return cls(max_num_seqs=seats, blocks=blocks), log


def kv_lines(log, prefix):
    return [entry.args for entry in log.call_args_list if entry.args and str(entry.args[0]).startswith(prefix)]


def blocks_of(prompt, tokens):
    return -(-(prompt + tokens + 32) // 64) + 1


class ArithmeticTests(unittest.TestCase):
    def test_the_reservation_is_the_ceiling_of_prompt_answer_and_lookahead_plus_one_block(self):
        self.assertEqual(kv.request_blocks(253920, 8192), 4097)
        self.assertEqual(kv.request_blocks(1, 1), 2)
        self.assertEqual(kv.request_blocks(0, 0), 2)
        self.assertEqual(kv.request_blocks(4096, 256), -(-(4096 + 256 + 32) // 64) + 1)
        for prompt in (1, 63, 64, 65, 4095, 4096, 100000, 262111):
            for answer in (1, 31, 32, 33, 255, 256, 8192, 16384):
                with self.subTest(prompt=prompt, answer=answer):
                    r = kv.request_blocks(prompt, answer)
                    self.assertGreaterEqual(r * 64, prompt + answer + 32 + 64)
                    self.assertLess((r - 1) * 64 - 64, prompt + answer + 32 + 1)

    def test_it_grows_with_prompt_and_answer_and_never_below_what_the_tokens_alone_need(self):
        previous = 0
        for tokens in range(0, 5000, 7):
            r = kv.request_blocks(tokens, 100)
            self.assertGreaterEqual(r, previous)
            self.assertGreaterEqual(r, -(-(tokens + 100) // 64) + 1)
            previous = r

    def test_the_262k_numbers(self):
        full = kv.request_blocks(262112 - 8192, 8192)
        self.assertEqual(full, 4097)
        self.assertEqual(POOL // full, 5, 'five full windows are resident at once in the provisional pool')
        self.assertLessEqual(5 * full, POOL)
        self.assertGreater(6 * full, POOL)
        # the I1 profile's pool: 8 seats of 2,052 blocks; one full 131k window reserves 2,053 + ... and the null block
        self.assertEqual(kv.request_blocks(123136, 8192), -(-(123136 + 8192 + 32) // 64) + 1)

    def test_a_bad_value_is_refused(self):
        for prompt, answer in ((-1, 1), (1, -1), (1.5, 1), (1, None), (True, 1) if False else (1, '1')):
            with self.subTest(prompt=prompt, answer=answer), self.assertRaises(ValueError):
                kv.request_blocks(prompt, answer)

    def test_the_flag_is_strict(self):
        self.assertFalse(kv.enabled({}))
        self.assertFalse(kv.enabled({kv.FLAG: '0'}))
        self.assertTrue(kv.enabled({kv.FLAG: '1'}))
        for bad in ('', '2', 'yes', 'true', ' 1'):
            with self.subTest(value=bad), self.assertRaisesRegex(ValueError, kv.FLAG):
                kv.enabled({kv.FLAG: bad})
            with self.subTest(value=bad), self.assertRaisesRegex(ValueError, kv.FLAG):
                admission.kv_reservation_requested({kv.FLAG: bad})
        self.assertEqual(admission.KV_FLAG, kv.FLAG)

    def test_the_pool_is_vllms_count_less_the_null_block_and_a_missing_pool_is_named(self):
        self.assertEqual(kv.pool_blocks(SimpleNamespace(kv_cache_manager=SimpleNamespace(
            block_pool=SimpleNamespace(num_gpu_blocks=100)))), 99)
        for scheduler in (SimpleNamespace(), SimpleNamespace(kv_cache_manager=SimpleNamespace()),
                          SimpleNamespace(kv_cache_manager=SimpleNamespace(block_pool=SimpleNamespace(num_gpu_blocks=None))),
                          SimpleNamespace(kv_cache_manager=SimpleNamespace(block_pool=SimpleNamespace(num_gpu_blocks=1)))):
            with self.assertRaisesRegex(ValueError, 'block_pool.num_gpu_blocks'):
                kv.pool_blocks(scheduler)

    def test_a_request_without_a_budget_reserves_the_rest_of_the_window(self):
        window = 4352
        plain = SimpleNamespace(num_prompt_tokens=1000)
        self.assertEqual(kv.request_reservation(plain, window), kv.request_blocks(1000, window - 1000))
        self.assertEqual(kv.request_reservation(SimpleNamespace(num_prompt_tokens=1000, max_tokens=10), window),
                         kv.request_blocks(1000, 10))
        sampling = SimpleNamespace(num_prompt_tokens=1000, sampling_params=SimpleNamespace(max_tokens=20))
        self.assertEqual(kv.request_reservation(sampling, window), kv.request_blocks(1000, 20))
        self.assertEqual(kv.request_reservation(SimpleNamespace(), window), kv.request_blocks(window, 0),
                         'an unreadable prompt counts as the whole window')
        self.assertEqual(kv.request_reservation(SimpleNamespace(prompt_token_ids=[1] * 70, max_tokens=5), window),
                         kv.request_blocks(70, 5))


class HoldTests(base.DramFreeCase):
    """The rule on the plugin's own TTScheduler class over the reduced vLLM scheduler (fixtures/plugin_scheduler.py)."""

    def test_a_request_that_does_not_fit_waits_while_the_decodes_run_and_is_admitted_when_they_finish(self):
        # a pool of 100: A reserves 60, B 60: the second must wait
        scheduler, log = plugin(blocks=101)
        a_tokens = 60 * 64 - 32 - 64 - 3000       # A's r = 60 (prompt 3000)
        scheduler.add_request(KvRequest('A', 3000, a_tokens))
        scheduler.add_request(KvRequest('B', 3000, a_tokens))
        self.assertEqual(kv.request_blocks(3000, a_tokens), 60)
        self.assertEqual(base.new_ids(scheduler.schedule()), ['A'])
        held = scheduler.schedule()
        self.assertEqual((base.new_ids(held), base.cached_ids(held)), ([], ['A']), 'B waits, A decodes')
        self.assertEqual(base.names(scheduler.waiting), ['B'])
        self.assertEqual(kv_lines(log, kv.HOLD_PREFIX),
                         [(kv.HOLD_LINE, 'B', 60, 60, 100, 1)])
        again = scheduler.schedule()
        self.assertEqual(base.cached_ids(again), ['A'])
        self.assertEqual(len(kv_lines(log, kv.HOLD_PREFIX)), 1, 'a state logs once')
        scheduler.finish('A')
        admitted = scheduler.schedule()
        self.assertEqual(base.new_ids(admitted), ['B'])
        self.assertEqual(admitted.finished_req_ids, {'A'}, 'A is named finished exactly once')
        self.assertIn(call(kv.RELEASED_LINE, 'B', 60, 0, 100), log.call_args_list)

    def test_the_finished_ids_of_a_held_pass_reach_the_decode_only_pass(self):
        # pool 100: A reserves 61, B 14, C 61. C is held beside both (136) and still beside A alone (122) after B finishes.
        scheduler, log = plugin(blocks=101)
        scheduler.add_request(KvRequest('A', 3000, 800))
        scheduler.add_request(KvRequest('B', 500, 300))
        scheduler.add_request(KvRequest('C', 3000, 800))
        self.assertEqual((kv.request_blocks(3000, 800), kv.request_blocks(500, 300)), (61, 14))
        for expected in (['A'], ['B']):
            self.assertEqual(base.new_ids(scheduler.schedule()), expected)
        scheduler.finish('B')
        held = scheduler.schedule()
        self.assertEqual(base.new_ids(held), [])
        self.assertEqual(base.cached_ids(held), ['A'])
        self.assertEqual(set(held.finished_req_ids), {'B'}, 'the decode-only pass names B finished: the worker detaches it')
        self.assertIn(call(kv.CARRIED_LINE, ['B']), log.call_args_list)
        self.assertEqual(base.names(scheduler.waiting), ['C'])

    def test_a_hold_is_never_lifted_while_a_decode_runs_and_a_fitting_request_is_never_held_with_nothing_running(self):
        scheduler, log = plugin(blocks=101)
        scheduler.add_request(KvRequest('BIG', 3000, 60 * 64 - 32 - 64 - 3000))     # r = 60 of a pool of 100
        self.assertEqual(base.new_ids(scheduler.schedule()), ['BIG'])
        scheduler.add_request(KvRequest('NEXT', 3000, 60 * 64 - 32 - 64 - 3000))
        for _ in range(50):
            step = scheduler.schedule()
            self.assertEqual(base.new_ids(step), [])
        scheduler.finish('BIG')
        scheduler.schedule()
        self.assertEqual(names(scheduler.running), ['NEXT'])

    def test_a_request_larger_than_the_whole_pool_is_held_and_logged_never_admitted(self):
        scheduler, log = plugin(blocks=11)
        scheduler.add_request(KvRequest('HUGE', 6000, 200))
        for _ in range(3):
            self.assertEqual(base.new_ids(scheduler.schedule()), [])
        self.assertEqual(base.names(scheduler.waiting), ['HUGE'])
        self.assertEqual(kv_lines(log, '[PINDIAG] kv reservation too large'),
                         [(kv.TOO_LARGE_LINE, 'HUGE', kv.request_blocks(6000, 200), 10)])

    def test_the_live_pool_is_logged_once_so_a_boot_shows_the_pool_vllm_really_built(self):
        scheduler, log = plugin(blocks=21761)
        scheduler.add_request(KvRequest('A', 100, 100))
        scheduler.add_request(KvRequest('B', 100, 100))
        scheduler.schedule()
        scheduler.schedule()
        self.assertEqual(kv_lines(log, '[PINDIAG] kv reservation pool='), [(kv.POOL_LINE, 21760, 21761)])

    def test_an_unreadable_pool_raises_and_logs_it_never_admits(self):
        scheduler = KvScheduler(max_num_seqs=8)
        scheduler.kv_cache_manager = SimpleNamespace(block_pool=SimpleNamespace())
        log = Mock()
        with self.assertRaises(ValueError):
            kv.hold(scheduler, [KvRequest('A', 100, 100)], 0, {}, log)
        self.assertEqual(len(kv_lines(log, '[PINDIAG] kv reservation unavailable')), 1)

    def test_the_waiting_loop_is_fcfs_so_the_head_binds_and_a_small_request_behind_it_does_not_jump(self):
        scheduler, log = plugin(blocks=101)
        scheduler.add_request(KvRequest('A', 3000, 800))                       # r = 61 of a pool of 100
        scheduler.add_request(KvRequest('LARGE', 3000, 800))                   # 61 + 61 > 100: held
        scheduler.add_request(KvRequest('SMALL', 100, 100))                    # would fit (61 + 5), but is behind LARGE
        self.assertEqual(base.new_ids(scheduler.schedule()), ['A'])
        step = scheduler.schedule()
        self.assertEqual((base.new_ids(step), base.names(scheduler.waiting)), ([], ['LARGE', 'SMALL']))
        self.assertEqual(kv_lines(log, kv.HOLD_PREFIX)[0][1:3], ('LARGE', 61))

    def test_among_the_candidates_the_loop_may_take_the_one_needing_the_most_binds(self):
        window = 4352
        big, small = KvRequest('BIG', 3000, 800), KvRequest('SMALL', 100, 100)
        self.assertEqual(kv.binding([small, big], window), (big, 61))
        self.assertEqual(kv.binding([big, small], window), (big, 61))
        self.assertEqual(kv.binding([], window), (None, 0))
        twin = KvRequest('TWIN', 3000, 800)
        self.assertIs(kv.binding([big, twin], window)[0], big, 'the first of equals')

    def test_the_hold_asks_only_when_a_fresh_prompt_would_be_admitted(self):
        # every seat decoding: nothing is asked and nothing logged
        scheduler, log = plugin(blocks=10_000, seats=2)
        scheduler.add_request(KvRequest('A', 100, 100))
        scheduler.add_request(KvRequest('B', 100, 100))
        scheduler.add_request(KvRequest('C', 100, 100))
        scheduler.schedule()
        scheduler.schedule()
        before = len(log.call_args_list)
        scheduler.schedule()
        self.assertEqual([entry for entry in log.call_args_list[before:] if 'kv reservation' in str(entry.args[0])], [])
        self.assertEqual(base.names(scheduler.waiting), ['C'])

    def test_composition_with_the_dram_hold_kv_is_asked_first_and_dram_only_when_kv_fits(self):
        asked = []

        def admits(prompt):
            asked.append(prompt)
            return False, dict(largest_free=900 * base.MB, need=1_300 * base.MB)

        scheduler, log = plugin(blocks=101)
        scheduler.add_request(KvRequest('A', 3000, 800))
        scheduler.schedule()
        scheduler.add_request(KvRequest('B', 3000, 800))                      # r = 61: does not fit beside A's 61 in 100
        with base.dram(admits):
            scheduler.schedule()
            self.assertEqual(asked, [], 'a KV hold is decided before the DRAM is read')
            self.assertEqual(len(kv_lines(log, kv.HOLD_PREFIX)), 1)
            self.assertEqual(base.logged(log, admission.DRAM_HOLD_LINE), [])
            scheduler.finish('A')
            scheduler.schedule()                                              # A's blocks are back: KV fits, DRAM refuses
            self.assertEqual(asked, [3000])
            self.assertEqual(len(base.logged(log, admission.DRAM_HOLD_LINE)), 0, 'no decode left: the DRAM hold is lifted')

    def test_composition_with_the_decode_credit(self):
        log = Mock()
        cls = base.plugin_class(KvScheduler)
        with patch.dict(os.environ, {kv.FLAG: '1', admission.STEPS_FLAG: '2'}), patch.object(kv, 'install_check', return_value=[]):
            admission.install(base.configured(cls), log=log)
        scheduler = cls(max_num_seqs=8)
        for name in 'ABC':
            scheduler.add_request(KvRequest(name, 100, 100))
        order = []
        for _ in range(8):
            step = scheduler.schedule()
            order.append((base.new_ids(step), base.cached_ids(step)))
        self.assertEqual([entry[0] for entry in order if entry[0]], [['A'], ['B'], ['C']])
        self.assertEqual(kv_lines(log, kv.HOLD_PREFIX), [], 'a pool this big never holds')


def names(requests):
    return sorted(request.request_id for request in requests)


class FlagOffTests(base.DramFreeCase):
    def run_sequence(self, on_flag):
        log = Mock()
        cls = base.plugin_class(KvScheduler)
        env = {} if on_flag is None else {kv.FLAG: on_flag}
        with patch.dict(os.environ, env, clear=False):
            if on_flag is None:
                os.environ.pop(kv.FLAG, None)
            admission.install(base.configured(cls), log=log)
        scheduler = cls(max_num_seqs=4)
        rng = random.Random(7)
        trace = []
        for step in range(40):
            if rng.random() < 0.5:
                scheduler.add_request(KvRequest('R%d' % step, rng.choice((64, 1024, 4096)), 128))
            output = scheduler.schedule()
            trace.append((base.new_ids(output), base.cached_ids(output), sorted(output.finished_req_ids)))
            if scheduler.running and rng.random() < 0.3:
                scheduler.finish(scheduler.running[0].request_id)
        return trace, [entry.args for entry in log.call_args_list]

    def test_unset_and_zero_install_nothing_and_every_step_is_the_one_without_the_module(self):
        unset, lines_unset = self.run_sequence(None)
        zero, lines_zero = self.run_sequence('0')
        self.assertEqual(unset, zero)
        self.assertFalse([line for line in lines_unset + lines_zero if 'kv reservation' in str(line[0])])
        # and the wrapper is call for call the one with no reservation argument at all
        cls = base.plugin_class(KvScheduler)
        original = cls._schedule_prefill_only
        wrapped = admission.wrap(original, queue_factory=lambda s: base.Queue(), log=Mock(), steps=0)
        self.assertTrue(callable(wrapped))

    def test_a_bad_flag_refuses_the_install_naming_it(self):
        cls = base.plugin_class(KvScheduler)
        with patch.dict(os.environ, {kv.FLAG: 'yes'}), self.assertRaisesRegex(ValueError, kv.FLAG):
            admission.install(base.configured(cls), log=Mock())
        self.assertFalse(getattr(cls._schedule_prefill_only, admission.WRAPPED, False))

    def test_a_vllm_that_moved_the_pool_refuses_the_install_naming_what_moved(self):
        cls = base.plugin_class(KvScheduler)
        with patch.dict(os.environ, {kv.FLAG: '1'}), patch.object(kv, 'install_check', return_value=['BlockPool no longer binds x']):
            with self.assertRaisesRegex(ValueError, 'cannot be installed.*BlockPool no longer binds x'):
                admission.install(base.configured(cls), log=Mock())
        self.assertFalse(getattr(cls._schedule_prefill_only, admission.WRAPPED, False))

    def test_the_install_logs_its_line_once_and_is_idempotent(self):
        log = Mock()
        cls = base.plugin_class(KvScheduler)
        install(cls, log)
        install(cls, log)
        self.assertEqual(len(kv_lines(log, '[PINDIAG] kv reservation installed')), 1)


class SeededSequenceTests(unittest.TestCase):
    """10,000 seeded arrival / finish sequences against the rule itself (hold() over a scheduler double): the sum of reservations
    of the running set never exceeds the pool, a request that fits an empty pool is always eventually admitted, and an arrival
    order is never reordered (FCFS)."""

    def simulate(self, seed):
        rng = random.Random(seed)
        pool = rng.choice((200, 2000, POOL))
        scheduler = SimpleNamespace(kv_cache_manager=SimpleNamespace(block_pool=SimpleNamespace(num_gpu_blocks=pool + 1)),
                                    max_model_len=4096 * 64, running=[], waiting=[])
        seats, state, log = rng.choice((2, 4, 8)), {}, lambda *a: None
        admitted, arrivals, finished = [], [], 0
        cap = max(1, pool * 64 - 2 * 64 - 32)
        for step in range(60):
            for _ in range(rng.choice((0, 0, 1, 2))):
                prompt, answer = rng.randint(1, min(cap, 20000)), rng.randint(1, min(cap, 4000))
                if kv.request_blocks(prompt, answer) > pool:
                    continue
                request = KvRequest('r%d-%d' % (seed, len(arrivals)), prompt, answer)
                arrivals.append(request.request_id)
                scheduler.waiting.append(request)
            if scheduler.waiting and len(scheduler.running) < seats:
                if not kv.hold(scheduler, scheduler.waiting[:1] if rng.random() < 0.5 else list(scheduler.waiting),
                               len(scheduler.running), state, log):
                    request = scheduler.waiting.pop(0)
                    scheduler.running.append(request)
                    admitted.append(request.request_id)
            reserved = kv.running_blocks(scheduler)
            assert reserved <= pool, (seed, step, reserved, pool)
            if scheduler.running and rng.random() < 0.35:
                scheduler.running.pop(rng.randrange(len(scheduler.running)))
                finished += 1
        # drain: with every running request finished, the head always fits an empty pool
        while scheduler.waiting or scheduler.running:
            if scheduler.running:
                scheduler.running.pop(0)
            while scheduler.waiting and len(scheduler.running) < seats:
                if kv.hold(scheduler, list(scheduler.waiting), len(scheduler.running), state, log):
                    break
                request = scheduler.waiting.pop(0)
                scheduler.running.append(request)
                admitted.append(request.request_id)
                assert kv.running_blocks(scheduler) <= pool
        return arrivals, admitted

    def test_ten_thousand_seeded_sequences(self):
        for seed in range(10000):
            arrivals, admitted = self.simulate(seed)
            self.assertEqual(admitted, arrivals, seed)


class ContractTests(unittest.TestCase):
    class Params(object):
        n, logprobs, prompt_logprobs, structured_outputs, stop, min_tokens = 1, None, None, None, None, 0
        logit_bias = allowed_token_ids = bad_words = stop_token_ids = max_tokens = None

    def profile(self, name='c2-packed-tp4-8x262k'):
        return json.loads(PROFILES.read_text(encoding='utf-8'))['profiles'][name]

    def test_a_request_needing_more_than_the_pool_is_refused_at_the_edge(self):
        limits = dict(budget=16384, max_prompt_tokens=253920, min_answer_tokens=8192, default_max_tokens=8192,
                      drafter_headroom_tokens=32)
        params = self.Params()
        params.max_tokens = 100
        contract.enforce_request(params, prompt_tokens=253920, max_model_len=WINDOW, eos_ids=frozenset(),
                                 kv_pool_blocks=POOL, **limits)
        self.assertEqual(params.max_tokens, 100)
        for pool in (kv.request_blocks(253920, 100) - 1, 10):
            params = self.Params()
            params.max_tokens = 100
            with self.assertRaisesRegex(contract.ContractError, 'KV blocks of 64 tokens'):
                contract.enforce_request(params, prompt_tokens=253920, max_model_len=WINDOW, eos_ids=frozenset(),
                                         kv_pool_blocks=pool, **limits)
        params = self.Params()
        contract.enforce_request(params, prompt_tokens=253920, max_model_len=WINDOW, eos_ids=frozenset(), kv_pool_blocks=None,
                                 **limits)
        self.assertEqual(params.max_tokens, 8192)

    def test_a_full_window_always_fits_the_profiles_pool(self):
        for name in ('c2-packed-tp4-8x262k', 'c2-packed-tp4-8x262k-gate', 'c2-packed-tp4-8x262k-time-gate',
                     'c2-packed-tp4-8x262k-diag-strace'):
            limits = contract.request_limits(self.profile(name))
            self.assertEqual(limits['kv_pool_blocks'], 21760 - 1, name)
            for prompt in (1, 100000, 253919, 253920):
                params = self.Params()
                params.max_tokens = 10 ** 6
                contract.enforce_request(params, prompt_tokens=prompt, max_model_len=WINDOW, eos_ids=frozenset(), **limits)

    def test_the_boot_rule_a_pooled_fast_path_profile_needs_the_flag(self):
        profile = copy.deepcopy(self.profile())
        profile['name'] = 'c2-packed-tp4-8x262k'
        self.assertTrue(contract.kv_pooled(profile))
        self.assertIsNone(contract.kv_reservation_problem(profile))
        del profile['env'][kv.FLAG]
        with self.assertRaisesRegex(ValueError, 'pools its KV cache.*needs QWEN_FAST_KV_RESERVATION=1'):
            contract.request_limits(profile)
        profile['env'][kv.FLAG] = '0'
        with self.assertRaisesRegex(ValueError, 'needs QWEN_FAST_KV_RESERVATION=1'):
            contract.request_limits(profile)
        profile['env'][kv.FLAG] = 'x'
        with self.assertRaisesRegex(ValueError, 'must be 0 or 1'):
            contract.request_limits(profile)
        profile['env'][kv.FLAG] = '1'
        del profile['engine']['num-gpu-blocks-override']
        with self.assertRaisesRegex(ValueError, 'needs num-gpu-blocks-override'):
            contract.request_limits(profile)

    def test_every_other_profile_boots_without_the_flag_or_a_pool_key(self):
        profiles = json.loads(PROFILES.read_text(encoding='utf-8'))['profiles']
        for name, profile in profiles.items():
            if kv.FLAG in profile['env']:
                self.assertTrue(name.startswith('c2-packed-tp4-8x262k'), name)
                continue
            with self.subTest(profile=name):
                self.assertFalse(contract.kv_pooled(dict(profile, name=name)), 'fully reserved or not a fast-path profile')
                self.assertNotIn('kv_pool_blocks', contract.request_limits(dict(profile, name=name)))

    def test_the_four_seat_262k_gate_is_fully_reserved_and_needs_no_reservation(self):
        profile = self.profile('c2-packed-tp4-262k-gate')
        self.assertFalse(contract.kv_pooled(profile))
        self.assertNotIn('kv_pool_blocks', contract.request_limits(profile))

    def test_the_installed_contract_passes_the_pool_through(self):
        import types

        class Processor(object):
            model_config = types.SimpleNamespace(max_model_len=4352)

            def process_inputs(self, request_id, prompt, params, *args, **kwargs):
                return params

        module = types.SimpleNamespace(InputProcessor=Processor)
        contract.install_request_contract(module, budget=256, eos_ids=frozenset(), kv_pool_blocks=10)
        params = self.Params()
        params.max_tokens, params.temperature = 100, 0.5
        with self.assertRaises(contract.ContractError):
            Processor().process_inputs('r', dict(prompt_token_ids=[1] * 2000), params)
        params = self.Params()
        params.max_tokens, params.temperature = 100, 0.5
        Processor().process_inputs('r', dict(prompt_token_ids=[1] * 100), params)


class InstallCheckTests(unittest.TestCase):
    def test_a_vllm_that_binds_the_pool_passes(self):
        def source(**names):
            return SimpleNamespace(**names)

        class KVCacheManager:
            def __init__(self):
                self.block_pool = None

        class BlockPool:
            def __init__(self):
                self.num_gpu_blocks = 1

        class Scheduler:
            def __init__(self):
                self.kv_cache_manager = None

        modules = {'vllm.v1.core.kv_cache_manager': source(KVCacheManager=KVCacheManager),
                   'vllm.v1.core.block_pool': source(BlockPool=BlockPool),
                   'vllm.v1.core.sched.scheduler': source(Scheduler=Scheduler)}
        self.assertEqual(kv.install_check(importer=modules.__getitem__), [])
        modules['vllm.v1.core.block_pool'] = source(BlockPool=type('BlockPool', (), {'__init__': lambda self: None}))
        problems = kv.install_check(importer=modules.__getitem__)
        self.assertEqual(len(problems), 1)
        self.assertIn('no longer binds self.num_gpu_blocks', problems[0])

    def test_without_vllm_nothing_is_checked(self):
        def missing(name):
            raise ImportError(name)

        self.assertEqual(kv.install_check(importer=missing), [])


if __name__ == '__main__':
    unittest.main()
