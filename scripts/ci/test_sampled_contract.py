"""The sampled-verify request contract (QWEN_FAST_SAMPLED, docs/tp4-sampled-verify.md), pure CPU.

What is held:
  - flag OFF (unset or 0, every shipped profile) is byte-identical to the greedy contract: a differential fuzz against a verbatim copy of the old
    enforce_request (the same refusals with the same messages, the same mutations of the params, nothing added), and a golden table;
  - flag ON honours temperature, top_p and top_k (the model's defaults for a field left None; temperature 0 stays greedy; top_k 0 or above K_MAX clamps to
    K_MAX), carries the request seed (the client's, else random) in extra_args and clears params.seed, and still refuses everything the fast path cannot do;
    penalties and min_p are NOT refused (clients send them today and must keep working): they are forced off as the greedy contract forces them, greedy or not,
    and a request that sent one gets one log line;
  - the switch is the profile's alone: no shipped profile names it (only the two gate-only sampled profiles of make_sampled_profiles do), an inherited value is
    dropped, a profile that turns it on needs the four-card mesh, C2-any, the request contract and a gate-only profile, and a launch that moves the model's
    generation-config defaults is refused;
  - it fails closed: the engine-side gate (serving_fast_policy.validate_request_sampling) refuses a non-greedy request until the attach proved every verifier can
    draw (test_sampled_verify, test_sampled_packed_path), so no sampled request reaches a verifier that would commit argmax for it;
  - the contract's constants are the sampler module's.
"""
import contextlib
import io
import json
import os
from pathlib import Path
import random
import sys
import tempfile
import types
import unittest
from unittest import mock

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(HERE))

import sampled_accept as sa  # noqa: E402
import serving_c2_contract as contract  # noqa: E402

PROFILES = HERE / 'qwen_c2_profiles.json'
ContractError = contract.ContractError
prompt_room = contract.prompt_room
omitted_max_tokens = contract.omitted_max_tokens


def legacy_enforce_request(params, *, prompt_tokens, max_model_len, budget, eos_ids, max_prompt_tokens=None,
                    min_answer_tokens=None, default_max_tokens=None, drafter_headroom_tokens=0, kv_pool_blocks=None):
    # (the docstring is dropped; everything below is the body as it stood at the base of the sampled-verify branch)
    if getattr(params, 'n', 1) != 1:
        raise ContractError('n must be 1 on this model')
    if getattr(params, 'logprobs', None) is not None or getattr(params, 'prompt_logprobs', None) is not None:
        raise ContractError('logprobs are not supported on this model')
    if getattr(params, 'structured_outputs', None) is not None:
        raise ContractError('structured output (response_format, or tool_choice "required" or a named tool) '
                            'is not supported on this model; use tool_choice "auto"')
    if (getattr(params, 'logit_bias', None) or getattr(params, 'allowed_token_ids', None)
            or getattr(params, 'bad_words', None)):
        raise ContractError('logit_bias, allowed_token_ids and bad_words are not supported on this model')
    if getattr(params, 'stop', None):
        raise ContractError('stop strings are not supported on this model')
    if getattr(params, 'min_tokens', 0):
        raise ContractError('min_tokens is not supported on this model')
    if any(token not in eos_ids for token in (getattr(params, 'stop_token_ids', None) or ())):
        raise ContractError('stop_token_ids other than the model end-of-sequence tokens are not supported')
    # The profile may cap prompts below context less budget: the KV cache is sized for
    # max-num-seqs x (max_prompt_tokens + budget), not x max_model_len, and the fast path
    # has no preemption to fall back on when a request outgrows it (prompt_room).
    room = prompt_room(max_model_len, budget, max_prompt_tokens, min_answer_tokens, drafter_headroom_tokens)
    if prompt_tokens is not None and prompt_tokens > room:
        if min_answer_tokens is None:
            raise ContractError('prompt of %d tokens exceeds the %d-token prompt limit of this model (%d-token '
                                'output budget)' % (prompt_tokens, room, budget))
        raise ContractError('prompt of %d tokens exceeds the %d-token prompt limit of this model (at least %d '
                            'tokens of answer room)' % (prompt_tokens, room, min_answer_tokens))
    # Greedy: the fast path's verifier commits argmax tokens whatever these say, and the
    # first token is sampled by vLLM from them, so they must agree with it.
    params.temperature = 0.0
    params.top_p = 1.0
    params.top_k = 0
    params.min_p = 0.0
    params.presence_penalty = 0.0
    params.frequency_penalty = 0.0
    params.repetition_penalty = 1.0
    params.seed = None
    limit = budget if prompt_tokens is None else min(budget, max_model_len - drafter_headroom_tokens - prompt_tokens)
    if default_max_tokens is not None and omitted_max_tokens(params.max_tokens, prompt_tokens=prompt_tokens,
                                                             max_model_len=max_model_len):
        params.max_tokens = min(default_max_tokens, limit)
    elif params.max_tokens is None or params.max_tokens > limit:
        params.max_tokens = limit
    if kv_pool_blocks is not None and prompt_tokens is not None:
        from serving_kv_reservation import request_blocks

        needed = request_blocks(prompt_tokens, params.max_tokens)
        if needed > kv_pool_blocks:
            raise ContractError('a request of %d prompt tokens and up to %d answer tokens needs %d KV blocks of 64 tokens, '
                                'and the server holds %d in all' % (prompt_tokens, params.max_tokens, needed, kv_pool_blocks))
    return params


class Params(object):
    """A vLLM SamplingParams as the contract sees it (the fixture of test_serving_c2_contract, with neutral penalties and no seed)."""

    def __init__(self, **values):
        self.n, self.logprobs, self.prompt_logprobs = 1, None, None
        self.structured_outputs, self.logit_bias, self.allowed_token_ids, self.bad_words = None, None, None, None
        self.stop, self.stop_token_ids, self.min_tokens = None, None, 0
        self.temperature, self.top_p, self.top_k, self.min_p = 1.0, 0.95, 20, 0.0
        self.presence_penalty, self.frequency_penalty, self.repetition_penalty = 0.0, 0.0, 1.0
        self.seed, self.max_tokens = None, None
        for key, value in values.items():
            setattr(self, key, value)


EOS = frozenset((248046, 248044))
COMMON = dict(prompt_tokens=1000, max_model_len=131328, budget=4096, eos_ids=EOS, max_prompt_tokens=61440)
LEGACY_CHANGED = {'temperature', 'top_p', 'top_k', 'min_p', 'presence_penalty', 'frequency_penalty', 'repetition_penalty', 'seed', 'max_tokens'}


def enforce(params, sampled=None, **overrides):
    arguments = dict(COMMON, **overrides)
    if sampled is None:
        return contract.enforce_request(params, **arguments)
    return contract.enforce_request(params, sampled=sampled, **arguments)


def legacy(params, **overrides):
    return legacy_enforce_request(params, **dict(COMMON, **overrides))


def outcome(function, params, **overrides):
    """('ok', the params' attributes afterwards) or (the exception class name, its message)."""
    try:
        function(params, **overrides)
    except ValueError as error:
        return type(error).__name__, str(error)
    return 'ok', dict(vars(params))


def random_values(rng):
    """Mostly requests the contract serves (so the served paths are fuzzed as hard as the refusals), some that it refuses."""
    if rng.random() < 0.6:
        return dict(temperature=rng.choice((0.0, 0.3, 0.7, 1.0, 1.5, None)), top_p=rng.choice((1.0, 0.95, 0.5, None)),
                    top_k=rng.choice((-1, 0, 1, 20, 64, 500, None)), min_p=rng.choice((0.0, 0.1)), presence_penalty=rng.choice((0.0, 1.5)),
                    frequency_penalty=rng.choice((0.0, -0.5)), repetition_penalty=rng.choice((1.0, 1.05)), seed=rng.choice((None, 0, 7, -3, 2 ** 70)),
                    max_tokens=rng.choice((None, 1, 100, 4096, 100000)), stop_token_ids=rng.choice((None, [248046])))
    return dict(
        n=rng.choice((1, 1, 1, 2)), logprobs=rng.choice((None, None, None, 1)), prompt_logprobs=rng.choice((None, None, None, 0)),
        structured_outputs=rng.choice((None, None, None, 'json')), logit_bias=rng.choice((None, None, None, {1: 1.0})),
        allowed_token_ids=rng.choice((None, None, None, [1])), bad_words=rng.choice((None, None, None, ['x'])),
        stop=rng.choice((None, None, None, ['a'])), min_tokens=rng.choice((0, 0, 0, 3)),
        stop_token_ids=rng.choice((None, None, [248046], [5])),
        temperature=rng.choice((0.0, 0.3, 0.7, 1.0, 1.5, None)), top_p=rng.choice((1.0, 0.95, 0.5, None)), top_k=rng.choice((-1, 0, 1, 20, 64, 500, None)),
        min_p=rng.choice((0.0, 0.0, 0.1)), presence_penalty=rng.choice((0.0, 0.0, 1.5)), frequency_penalty=rng.choice((0.0, 0.0, -0.5)),
        repetition_penalty=rng.choice((1.0, 1.0, 1.05)), seed=rng.choice((None, None, 0, 7, -3, 2 ** 70)),
        max_tokens=rng.choice((None, 1, 100, 4096, 100000)))


def random_configuration(rng):
    return dict(prompt_tokens=rng.choice((None, 1, 1000, 61440, 61441, 130000)), budget=rng.choice((256, 4096, 16384)),
                max_prompt_tokens=rng.choice((None, 61440)), min_answer_tokens=rng.choice((None, 128, 8192)),
                default_max_tokens=rng.choice((None, 8192)), drafter_headroom_tokens=rng.choice((0, 0, 32)))


class FlagOffIsTheGreedyContractTest(unittest.TestCase):
    def test_a_differential_fuzz_against_the_old_function(self):
        rng = random.Random(20261022)
        counts = dict(ok=0, ContractError=0, ValueError=0)
        for _ in range(4000):
            values, configuration = random_values(rng), random_configuration(rng)
            wanted = outcome(legacy, Params(**values), **configuration)
            counts[wanted[0]] += 1
            with mock.patch.dict(os.environ):
                os.environ.pop(contract.SAMPLED_SWITCH, None)
                unset = outcome(enforce, Params(**values), **configuration)
                os.environ[contract.SAMPLED_SWITCH] = '0'
                zero = outcome(enforce, Params(**values), **configuration)
            explicit = outcome(enforce, Params(**values), sampled=False, **configuration)
            for label, got in (('unset', unset), ('0', zero), ('sampled=False', explicit)):
                self.assertEqual(got, wanted, (label, values, configuration))
        self.assertGreater(counts['ok'], 300)
        self.assertGreater(counts['ContractError'], 300)

    def test_a_golden_table(self):
        """Hand-written expectations: what a client sends, what the engine gets."""
        cases = (
            (dict(temperature=0.7, top_p=0.8, top_k=50, seed=1234, repetition_penalty=1.05, max_tokens=None),
             dict(temperature=0.0, top_p=1.0, top_k=0, min_p=0.0, presence_penalty=0.0, frequency_penalty=0.0, repetition_penalty=1.0, seed=None, max_tokens=4096)),
            (dict(temperature=None, top_p=None, top_k=None, seed=None, max_tokens=100),
             dict(temperature=0.0, top_p=1.0, top_k=0, min_p=0.0, presence_penalty=0.0, frequency_penalty=0.0, repetition_penalty=1.0, seed=None, max_tokens=100)),
            (dict(temperature=1.0, presence_penalty=1.5, min_p=0.2, seed=-1, max_tokens=10 ** 6),
             dict(temperature=0.0, top_p=1.0, top_k=0, min_p=0.0, presence_penalty=0.0, frequency_penalty=0.0, repetition_penalty=1.0, seed=None, max_tokens=4096)),
        )
        for values, expected in cases:
            for sampled in (None, False):
                params = enforce(Params(**values), sampled=sampled)
                self.assertEqual({name: getattr(params, name) for name in expected}, expected)
                self.assertFalse(hasattr(params, 'extra_args'))
                self.assertEqual(set(vars(params)) - set(vars(Params())), set())

    def test_only_the_legacy_attributes_are_touched(self):
        before = Params(temperature=0.5, seed=3)
        reference = dict(vars(before))
        enforce(before)
        self.assertEqual({name for name in vars(before) if vars(before)[name] != reference[name]} - LEGACY_CHANGED, set())
        self.assertEqual(set(vars(before)), set(reference))

    def test_a_malformed_switch_is_an_error_not_a_silent_off(self):
        for value in ('yes', 'true', '2', ''):
            with mock.patch.dict(os.environ, {contract.SAMPLED_SWITCH: value}), self.assertRaises(ValueError):
                enforce(Params())
        self.assertFalse(contract.sampled_enabled({}))
        self.assertFalse(contract.sampled_enabled({contract.SAMPLED_SWITCH: '0'}))
        self.assertTrue(contract.sampled_enabled({contract.SAMPLED_SWITCH: '1'}))


class FlagOnTest(unittest.TestCase):
    def sampled(self, **values):
        return enforce(Params(**values), sampled=True)

    def test_the_requests_terms_are_honoured(self):
        params = self.sampled(temperature=0.7, top_p=0.8, top_k=50, seed=1234, max_tokens=100)
        self.assertEqual((params.temperature, params.top_p, params.top_k, params.min_p), (0.7, 0.8, 50, 0.0))
        self.assertEqual((params.presence_penalty, params.frequency_penalty, params.repetition_penalty), (0.0, 0.0, 1.0))
        self.assertIsNone(params.seed)
        self.assertEqual(params.extra_args, {contract.SAMPLED_SEED_KEY: 1234})
        self.assertEqual(params.max_tokens, 100)

    def test_a_field_left_none_takes_the_models_default(self):
        params = self.sampled(temperature=None, top_p=None, top_k=None)
        self.assertEqual((params.temperature, params.top_p, params.top_k), (1.0, 0.95, 20))
        params = self.sampled(temperature=0.5, top_p=None, top_k=None)
        self.assertEqual((params.temperature, params.top_p, params.top_k), (0.5, 0.95, 20))
        params = self.sampled(top_p=0.3, top_k=None)
        self.assertEqual((params.temperature, params.top_p, params.top_k), (1.0, 0.3, 20))

    def test_the_models_defaults_are_the_samplers(self):
        self.assertEqual(contract.SAMPLED_MODEL_DEFAULTS, sa.MODEL_DEFAULTS)
        self.assertEqual(contract.SAMPLED_K_MAX, sa.K_MAX)
        params = self.sampled(temperature=None, top_p=None, top_k=None)
        self.assertEqual(sa.SamplingSpec(params.temperature, params.top_p, params.top_k), sa.SamplingSpec())

    def test_top_k_zero_negative_or_above_the_maximum_clamps_to_it(self):
        for sent, kept in ((0, 64), (-1, 64), (1, 1), (20, 20), (63, 63), (64, 64), (65, 64), (1000, 64), (10 ** 9, 64)):
            params = self.sampled(top_k=sent)
            self.assertEqual(params.top_k, kept, sent)
            self.assertEqual(params.top_k, sa.SamplingSpec(1.0, 0.95, sent).effective_top_k, sent)

    def test_temperature_zero_stays_greedy_exactly_as_the_greedy_contract(self):
        for values in (dict(temperature=0, top_p=0.3, top_k=5, seed=9), dict(temperature=0.0), dict(temperature=0, seed=None, top_k=None, top_p=None)):
            params = self.sampled(**values)
            reference = Params(**values)
            legacy(reference)
            self.assertEqual(vars(params), vars(reference))
            self.assertFalse(hasattr(params, 'extra_args'))

    def test_the_seed_is_the_clients_or_a_random_64_bit_one(self):
        self.assertEqual(self.sampled(seed=0).extra_args[contract.SAMPLED_SEED_KEY], 0)
        self.assertEqual(self.sampled(seed=-1).extra_args[contract.SAMPLED_SEED_KEY], (1 << 64) - 1)
        self.assertEqual(self.sampled(seed=2 ** 64 + 5).extra_args[contract.SAMPLED_SEED_KEY], 5)
        seeds = [self.sampled(seed=None).extra_args[contract.SAMPLED_SEED_KEY] for _ in range(300)]
        self.assertEqual(len(set(seeds)), 300)
        self.assertTrue(all(0 <= seed < 1 << 64 for seed in seeds))
        self.assertGreater(max(seed.bit_length() for seed in seeds), 60)
        for seed in seeds[:5]:
            self.assertEqual(sa.request_seed(seed), seed)         # the key the sampler module reads back
        params = self.sampled(seed=5)
        self.assertIsNone(params.seed)                            # the pinned samplers refuse a block while a request seed is active

    def test_extra_args_keep_the_others_and_the_seed_field_wins_over_the_key(self):
        params = self.sampled(seed=7, extra_args={'qwen_lane': 'fast', contract.SAMPLED_SEED_KEY: 99})
        self.assertEqual(params.extra_args, {'qwen_lane': 'fast', contract.SAMPLED_SEED_KEY: 7})
        original = {'qwen_lane': 'fast'}
        params = self.sampled(seed=7, extra_args=original)
        self.assertEqual(original, {'qwen_lane': 'fast'})         # not mutated in place

    def test_the_hook_is_idempotent_and_a_client_sent_key_is_a_seed(self):
        """Run twice, a request keeps its first seed (params.seed was cleared the first time); the key sent as vllm_xargs is a seed like `seed`."""
        once = self.sampled(seed=None)
        twice = enforce(once, sampled=True)
        self.assertEqual(twice.extra_args, once.extra_args)
        snapshot = dict(vars(twice))
        self.assertEqual(vars(enforce(twice, sampled=True)), snapshot)
        self.assertEqual(self.sampled(seed=None, extra_args={contract.SAMPLED_SEED_KEY: 99}).extra_args[contract.SAMPLED_SEED_KEY], 99)
        for junk in ('x', 1.5, None, True):
            seed = self.sampled(seed=None, extra_args={contract.SAMPLED_SEED_KEY: junk}).extra_args[contract.SAMPLED_SEED_KEY]
            self.assertIsInstance(seed, int)
            self.assertNotEqual(seed, junk)

    def test_what_the_fast_path_cannot_serve_is_still_refused(self):
        for values in (dict(n=2), dict(logprobs=1), dict(prompt_logprobs=0), dict(structured_outputs=object()), dict(logit_bias={1: 1.0}),
                       dict(allowed_token_ids=[1]), dict(bad_words=['x']), dict(stop=['\n\n']), dict(min_tokens=5), dict(stop_token_ids=[5])):
            with self.assertRaises(ContractError, msg=repr(values)):
                self.sampled(**values)
        self.sampled(stop_token_ids=[248046])
        with self.assertRaises(ContractError):
            enforce(Params(), sampled=True, prompt_tokens=61441)               # the prompt cap is the greedy contract's

    def sampled_logged(self, **values):
        with contextlib.redirect_stderr(io.StringIO()) as log:
            params = self.sampled(**values)
        return params, [line for line in log.getvalue().splitlines() if 'ignored' in line]

    def test_penalties_and_min_p_are_forced_off_not_refused_greedy_or_not(self):
        for values in (dict(presence_penalty=0.5), dict(frequency_penalty=-0.5), dict(repetition_penalty=1.05), dict(min_p=0.05)):
            for temperature in (0.0, 0.7):
                with self.subTest(values=values, temperature=temperature):
                    params, lines = self.sampled_logged(temperature=temperature, **values)
                    self.assertEqual((params.min_p, params.presence_penalty, params.frequency_penalty, params.repetition_penalty), (0.0, 0.0, 0.0, 1.0))
                    self.assertEqual(params.temperature, 0.0 if temperature == 0.0 else 0.7)
                    self.assertEqual(len(lines), 1, lines)
                    self.assertIn(next(iter(values)), lines[0])
                    legacy_params = Params(temperature=temperature, **values)
                    legacy(legacy_params)
                    self.assertEqual((legacy_params.min_p, legacy_params.presence_penalty, legacy_params.frequency_penalty, legacy_params.repetition_penalty),
                                     (params.min_p, params.presence_penalty, params.frequency_penalty, params.repetition_penalty), 'the same neutral values the greedy contract writes')

    def test_the_log_names_every_dropped_term_once_per_request(self):
        params, lines = self.sampled_logged(presence_penalty=0.5, frequency_penalty=0.1, repetition_penalty=1.2, min_p=0.1)
        self.assertEqual(len(lines), 1)
        for name in ('presence_penalty', 'frequency_penalty', 'repetition_penalty', 'min_p'):
            self.assertIn(name, lines[0])
        with contextlib.redirect_stderr(io.StringIO()) as again:
            enforce(params, sampled=True)                                   # a second pass through the hook: the terms are neutral now
        self.assertEqual(again.getvalue(), '')

    def test_neutral_terms_log_nothing(self):
        for values in (dict(), dict(presence_penalty=0.0, frequency_penalty=0.0, repetition_penalty=1.0, min_p=0.0), dict(presence_penalty=None, repetition_penalty=None, min_p=None)):
            self.assertEqual(self.sampled_logged(**values)[1], [], values)

    def test_a_request_refused_for_another_reason_logs_nothing_about_its_penalties(self):
        with contextlib.redirect_stderr(io.StringIO()) as log, self.assertRaises(ContractError):
            self.sampled(presence_penalty=0.5, temperature=-1.0)
        self.assertNotIn('ignored', log.getvalue())

    def test_the_flag_off_contract_drops_them_silently_as_it_always_did(self):
        params = Params(presence_penalty=0.5, min_p=0.1)
        with contextlib.redirect_stderr(io.StringIO()) as log:
            enforce(params, sampled=False)
        self.assertEqual(log.getvalue(), '')
        self.assertEqual((params.presence_penalty, params.min_p), (0.0, 0.0))

    def test_malformed_terms_are_refused(self):
        for values in (dict(temperature=-0.1), dict(temperature=float('nan')), dict(temperature=float('inf')), dict(temperature='1'), dict(temperature=True),
                       dict(top_p=0.0), dict(top_p=1.5), dict(top_p=float('nan')), dict(top_p='0.9'), dict(top_k=2.5), dict(top_k='3'), dict(top_k=True),
                       dict(seed=1.5), dict(seed='7'), dict(seed=True)):
            with self.subTest(values=values), self.assertRaises(ContractError):
                self.sampled(**values)

    def test_the_budget_clamps_are_the_greedy_contracts(self):
        for values in (dict(max_tokens=None), dict(max_tokens=100), dict(max_tokens=10 ** 6)):
            for prompt_tokens in (None, 1000, 61440):
                got = enforce(Params(**values), sampled=True, prompt_tokens=prompt_tokens).max_tokens
                self.assertEqual(got, legacy(Params(**values), prompt_tokens=prompt_tokens).max_tokens, (values, prompt_tokens))

    def test_the_installed_hook_reads_the_switch_per_request(self):
        seen = []

        class InputProcessor(object):
            model_config = types.SimpleNamespace(max_model_len=131328)

            def process_inputs(self, request_id, prompt, params, *args, **kwargs):
                seen.append((params.temperature, params.top_p, params.top_k, params.seed, getattr(params, 'extra_args', None)))
                return 'request'

        module = types.ModuleType('fake_input_processor_sampled')
        module.InputProcessor = InputProcessor
        with mock.patch.dict(os.environ, {contract.SAMPLED_SWITCH: '1'}), contextlib.redirect_stderr(io.StringIO()) as log:
            contract.install_request_contract(module, budget=4096, eos_ids=EOS, max_prompt_tokens=61440)
            InputProcessor().process_inputs('r1', {'prompt_token_ids': [1] * 10}, Params(temperature=0.6, top_k=0, seed=11))
            os.environ[contract.SAMPLED_SWITCH] = '0'
            InputProcessor().process_inputs('r2', {'prompt_token_ids': [1] * 10}, Params(temperature=0.6, top_k=0, seed=11))
        self.assertEqual(seen, [(0.6, 0.95, 64, None, {contract.SAMPLED_SEED_KEY: 11}), (0.0, 1.0, 0, None, None)])
        self.assertIn('sampling honoured', log.getvalue())


class ProfileAndBootTest(unittest.TestCase):
    GATE = 'c2-packed-tp4-8x262k-gate'

    def document(self):
        with open(PROFILES, encoding='utf-8') as handle:
            return json.load(handle)

    def test_no_shipped_profile_or_image_file_names_the_switch(self):
        import make_sampled_profiles

        document = self.document()
        owned = set(make_sampled_profiles.twin_names())
        for name, profile in document['profiles'].items():
            if name in owned:
                continue
            self.assertNotIn(contract.SAMPLED_SWITCH, profile.get('env', {}), name)
            self.assertNotIn(contract.SAMPLED_AUDIT_SWITCH, profile.get('env', {}), name)
            self.assertEqual(contract.sampled_problems(dict(profile, name=name)), [], name)
            self.assertFalse(contract.sampled_on(profile), name)
        for name in owned:
            self.assertTrue(contract.sampled_on(document['profiles'][name]), name)
            self.assertIs(document['profiles'][name]['gate_only'], True, name)
            self.assertEqual(contract.sampled_problems(dict(document['profiles'][name], name=name)), [], name)
        self.assertEqual(sum(profile.get('env', {}).get(contract.SAMPLED_SWITCH) == '1' for profile in document['profiles'].values()), len(owned))
        named = []
        for pattern in ('docker/*', '.github/workflows/*.yml'):
            for path in sorted(ROOT.glob(pattern)):
                if path.is_file() and contract.SAMPLED_SWITCH in path.read_text(encoding='utf-8', errors='replace'):
                    named.append(str(path.relative_to(ROOT)))
        self.assertEqual(named, [])

    def test_the_switch_is_validated_on_the_profile(self):
        good = {'QWEN_FAST_ANY_REQUEST': '1', 'QWEN_FAST_TP': '4', contract.SAMPLED_SWITCH: '1'}
        base = dict(name='p', gate_only=True, mesh_device='P150x4', env=good)
        self.assertEqual(contract.sampled_problems(base), [])
        self.assertTrue(contract.sampled_on(base))
        self.assertEqual(contract.sampled_problems(dict(base, env=dict(good, **{contract.SAMPLED_AUDIT_SWITCH: '1'}))), [])
        self.assertEqual(contract.sampled_problems(dict(base, env=dict(good, **{contract.SAMPLED_SWITCH: '0'}))), [])
        self.assertEqual(contract.sampled_problems(dict(base, env={})), [])
        self.assertEqual(contract.sampled_problems(dict(base, env=dict(good, **{contract.SAMPLED_SWITCH: 1}))), [])   # (a JSON 1 is str()ed like every flag value)
        for bad in ('yes', 'true', '', 'True', 2):
            problems = contract.sampled_problems(dict(base, env=dict(good, **{contract.SAMPLED_SWITCH: bad})))
            self.assertTrue(any('must be 0 or 1' in p for p in problems), bad)
        for broken, needle in ((dict(base, gate_only=False), 'gate-only'), (dict(base, gate_only=None), 'gate-only'),
                               (dict(base, env={'QWEN_FAST_TP': '4', contract.SAMPLED_SWITCH: '1'}), 'QWEN_FAST_ANY_REQUEST'),
                               (dict(base, env=dict(good, QWEN_FAST_ANY_REQUEST='0')), 'QWEN_FAST_ANY_REQUEST'),
                               (dict(base, request_contract=False), 'request contract'),
                               (dict(base, mesh_device=None), 'four-card'), (dict(base, env=dict(good, QWEN_FAST_TP='2')), 'four-card'),
                               (dict(base, env={'QWEN_FAST_ANY_REQUEST': '1', 'QWEN_FAST_TP': '4', contract.SAMPLED_AUDIT_SWITCH: '1'}), 'needs QWEN_FAST_SAMPLED=1'),
                               (dict(base, env=dict(good, **{contract.SAMPLED_SWITCH: '0', contract.SAMPLED_AUDIT_SWITCH: '1'})), 'needs QWEN_FAST_SAMPLED=1'),
                               (dict(base, env=dict(good, **{contract.SAMPLED_AUDIT_SWITCH: 'yes'})), 'must be 0 or 1')):
            problems = contract.sampled_problems(broken)
            self.assertTrue(any(needle in p for p in problems), (needle, problems))

    def test_an_inherited_switch_is_dropped_unless_the_profile_names_it(self):
        profile = self.document()['profiles'][self.GATE]
        environ = {contract.SAMPLED_SWITCH: '1', contract.SAMPLED_AUDIT_SWITCH: '1', 'OTHER': 'x'}
        contract.apply_environment(profile, environ)
        self.assertNotIn(contract.SAMPLED_SWITCH, environ)
        self.assertNotIn(contract.SAMPLED_AUDIT_SWITCH, environ)
        self.assertEqual(environ['OTHER'], 'x')
        named = dict(profile, env=dict(profile['env'], **{contract.SAMPLED_SWITCH: '1'}))
        environ = {}
        contract.apply_environment(named, environ)
        self.assertEqual(environ[contract.SAMPLED_SWITCH], '1')

    def test_a_launch_that_moves_the_generation_config_is_refused(self):
        for argv in ([], ['--model', 'm'], ['--generation-config', 'auto'], ['--generation-config=auto'], ['--generation_config', 'auto']):
            self.assertEqual(contract.sampled_launch_problems(argv), [], argv)
        for argv in (['--generation-config', 'vllm'], ['--generation-config=/some/dir'], ['--generation-config'], ['--override-generation-config', '{"top_k": 5}'],
                     ['--override-generation-config={"temperature": 0.2}']):
            self.assertTrue(contract.sampled_launch_problems(argv), argv)

    def boot(self, flag, argv_extra=(), api=True):
        document = self.document()
        profile = document['profiles'][self.GATE]
        if flag is not None:
            profile['env'][contract.SAMPLED_SWITCH] = flag
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'profiles.json'
            path.write_text(json.dumps(document), encoding='utf-8')
            environ = {'QWEN_C2_SERVING': '1', 'QWEN_C2_PROFILES': str(path), 'QWEN_C2_GATE': '1'}
            saved = (list(sys.meta_path), list(sys.argv), contract.resolve_snapshot)
            contract.resolve_snapshot = lambda profile: '/snapshot'
            sys.argv[:] = ['python3'] + list(argv_extra)
            try:
                with mock.patch.dict(os.environ, {'QWEN_C2_PROFILE': self.GATE}), contextlib.redirect_stderr(io.StringIO()):
                    result = contract.boot(environ=environ, orig_argv=['python3', '-m', contract.API_SERVER] if api else ['python3', '-c', 'pass'])
            finally:
                sys.meta_path[:] = saved[0]
                sys.argv[:] = saved[1]
                contract.resolve_snapshot = saved[2]
        return result, environ

    def test_boot_is_unchanged_without_the_switch_and_checks_it_with(self):
        self.assertIn(self.GATE, self.document()['profiles'])
        for api in (True, False):
            result, environ = self.boot(None, api=api)
            self.assertEqual(result['name'], self.GATE)
            self.assertNotIn(contract.SAMPLED_SWITCH, environ)
        # the launched generation config is only checked for a profile that turns the switch on
        self.boot(None, ['--generation-config', 'vllm'])
        result, environ = self.boot('1', ['--generation-config=auto'])
        self.assertEqual(environ[contract.SAMPLED_SWITCH], '1')
        self.boot('1')
        with self.assertRaisesRegex(ValueError, 'cannot serve sampled verification with this launch'):
            self.boot('1', ['--generation-config', 'vllm'])
        with self.assertRaisesRegex(ValueError, 'cannot serve sampled verification with this launch'):
            self.boot('1', ['--override-generation-config', '{"top_k": 1}'])
        with self.assertRaisesRegex(ValueError, 'misuses the sampled-verify switch.*must be 0 or 1'):
            self.boot('maybe', api=False)


class FailsClosedTest(unittest.TestCase):
    def test_the_engine_side_gate_refuses_a_sampled_request_until_the_attach_proved_the_verifiers_can_draw(self):
        """A request the contract lets through with temperature > 0 is refused at the engine (ending only that request under C2-any) until sampled_verify.attach has
        proved every block draws (test_sampled_verify holds the open gate), never committed as argmax."""
        import serving_fast_policy as policy

        params = types.SimpleNamespace(temperature=0, n=1, max_tokens=256, min_tokens=0, ignore_eos=False, logprobs=None, prompt_logprobs=None,
                                       presence_penalty=0, frequency_penalty=0, repetition_penalty=1, stop=None, stop_token_ids=None,
                                       structured_outputs=None, logit_bias=None, allowed_token_ids=None, bad_words=None)
        with mock.patch.dict(os.environ, {contract.SAMPLED_SWITCH: '1'}):
            policy.validate_request_sampling(params, prompt_tokens=100)
            sampled = enforce(Params(temperature=0.7, seed=3), sampled=True)
            params.temperature = sampled.temperature
            with self.assertRaises(ValueError):
                policy.validate_request_sampling(params, prompt_tokens=100)
            for field, value in (('top_p', 0.9), ('top_k', 5)):
                greedy = types.SimpleNamespace(**dict(vars(params), temperature=0))
                setattr(greedy, field, value)
                policy.validate_request_sampling(greedy, prompt_tokens=100)      # (temperature 0 reads no other term: unchanged)


if __name__ == '__main__':
    unittest.main()
