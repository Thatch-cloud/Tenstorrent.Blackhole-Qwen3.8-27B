from types import SimpleNamespace
import unittest

from serving_fast_policy import internal_batch_capacity, validate_fast_config, validate_request_sampling


class FastPolicyTests(unittest.TestCase):
    def fixture(self):
        return SimpleNamespace(additional_config={'qwen_fast_t16': True},
            scheduler_config=SimpleNamespace(max_num_seqs=1, async_scheduling=False),
            parallel_config=SimpleNamespace(tensor_parallel_size=1, pipeline_parallel_size=1, data_parallel_size=1),
            cache_config=SimpleNamespace(block_size=64, enable_prefix_caching=False),
            lora_config=None, model_config=SimpleNamespace(max_model_len=4352),
            speculative_config=SimpleNamespace(method='dflash', num_speculative_tokens=15,
                draft_sample_method='greedy', rejection_sample_method='standard'))

    def test_preserves_internal_eight_slots_with_one_scheduler_request(self):
        config = self.fixture()
        self.assertEqual(internal_batch_capacity(config), 8)
        self.assertEqual(config.scheduler_config.max_num_seqs, 1)
        self.assertFalse(validate_fast_config(config)['serving_qualified'])

    def test_default_capacity_unchanged(self):
        config = self.fixture()
        config.additional_config = {}
        for capacity in (1, 2, 8):
            config.scheduler_config.max_num_seqs = capacity
            self.assertEqual(internal_batch_capacity(config), capacity)

    def test_long_contexts_admitted_in_whole_pages(self):
        """The 4352 pin is now a floor: capture planning is position-parameterised."""
        for model_len in (4352, 8192, 65536, 163840):
            config = self.fixture()
            config.model_config.max_model_len = model_len
            profile = validate_fast_config(config)
            self.assertEqual(profile['max_model_len'], model_len)
            self.assertEqual(profile['context_tokens'], model_len - 256)
            self.assertEqual(profile['output_budget'], 256)

    def test_short_or_partial_page_lengths_still_rejected(self):
        for model_len in (4096, 4351, 4353, 8100):
            config = self.fixture()
            config.model_config.max_model_len = model_len
            with self.assertRaises(ValueError):
                validate_fast_config(config)

    def test_scheduler_request_pin_is_unchanged(self):
        """Concurrency stays pinned: session state in the fast path is singular."""
        for capacity in (2, 4, 8):
            config = self.fixture()
            config.scheduler_config.max_num_seqs = capacity
            with self.assertRaises(ValueError):
                validate_fast_config(config)

    def sampling(self, **overrides):
        base = dict(temperature=0, n=1, max_tokens=256, min_tokens=0, ignore_eos=False,
                    logprobs=None, prompt_logprobs=None, presence_penalty=0,
                    frequency_penalty=0, repetition_penalty=1, stop=None,
                    stop_token_ids=None, structured_outputs=None, logit_bias=None,
                    allowed_token_ids=None, bad_words=None)
        base.update(overrides)
        return SimpleNamespace(**base)

    def test_prompt_length_is_a_bound_not_an_equality(self):
        """The 4096 pin was the initial coding profile, not a correctness requirement."""
        for prompt_tokens in (1, 4096, 30000, 65536, 163840):
            validate_request_sampling(self.sampling(), prompt_tokens=prompt_tokens)

    def test_output_tokens_bounded_by_the_budget(self):
        for max_tokens in (1, 64, 256):
            validate_request_sampling(self.sampling(max_tokens=max_tokens), prompt_tokens=4096)
        for max_tokens in (0, 257, 1024):
            with self.assertRaises(ValueError):
                validate_request_sampling(self.sampling(max_tokens=max_tokens), prompt_tokens=4096)

    def test_determinism_constraints_stay_exact(self):
        """These are what make output bit-comparable; loosening them invalidates every
        correctness claim downstream."""
        for field, value in (('temperature', 0.7), ('n', 2), ('min_tokens', 1),
                             ('ignore_eos', True), ('presence_penalty', 0.1),
                             ('frequency_penalty', 0.1), ('repetition_penalty', 1.1),
                             ('logprobs', 5), ('prompt_logprobs', 5), ('stop', ['x']),
                             ('structured_outputs', object()), ('logit_bias', {1: 1.0}),
                             ('allowed_token_ids', [1]), ('bad_words', ['x'])):
            with self.assertRaises(ValueError):
                validate_request_sampling(self.sampling(**{field: value}), prompt_tokens=4096)

    def test_invalid_prompt_lengths_rejected(self):
        for prompt_tokens in (0, -1, 'x', None):
            with self.assertRaises(ValueError):
                validate_request_sampling(self.sampling(), prompt_tokens=prompt_tokens)

    def test_unsupported_configs_rejected(self):
        for group, field, value in (('scheduler_config', 'max_num_seqs', 2),
                ('scheduler_config', 'async_scheduling', True), ('parallel_config', 'tensor_parallel_size', 2),
                ('cache_config', 'enable_prefix_caching', True), ('cache_config', 'block_size', 32),
                ('model_config', 'max_model_len', 4351), ('model_config', 'max_model_len', 8100),
                ('speculative_config', 'num_speculative_tokens', 31),
                ('speculative_config', 'method', 'ngram'), ('speculative_config', 'draft_sample_method', 'probabilistic')):
            config = self.fixture()
            setattr(getattr(config, group), field, value)
            with self.assertRaises(ValueError):
                validate_fast_config(config)

    def test_string_opt_in_is_rejected(self):
        config = self.fixture()
        config.additional_config['qwen_fast_t16'] = 'false'
        with self.assertRaises(ValueError):
            internal_batch_capacity(config)

    def test_unimplemented_sampling_features_fail_explicitly(self):
        def parameters():
            return SimpleNamespace(temperature=0, n=1, max_tokens=256, min_tokens=0, ignore_eos=False,
                logprobs=None, prompt_logprobs=None, presence_penalty=0, frequency_penalty=0,
                repetition_penalty=1, stop=[], stop_token_ids=[])

        validate_request_sampling(parameters(), prompt_tokens=4096)
        for field, value in (('temperature', .7), ('n', 2), ('max_tokens', 512), ('min_tokens', 1),
                ('ignore_eos', True), ('logprobs', 1), ('repetition_penalty', 1.1),
                ('stop', ['END']), ('stop_token_ids', [13]), ('structured_outputs', object()),
                ('logit_bias', {13: 1}), ('allowed_token_ids', [13]), ('bad_words', ['bad'])):
            sample = parameters()
            setattr(sample, field, value)
            with self.assertRaises(ValueError):
                validate_request_sampling(sample, prompt_tokens=4096)
