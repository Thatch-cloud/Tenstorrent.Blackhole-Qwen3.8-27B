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

    def test_unsupported_configs_rejected(self):
        for group, field, value in (('scheduler_config', 'max_num_seqs', 2),
                ('scheduler_config', 'async_scheduling', True), ('parallel_config', 'tensor_parallel_size', 2),
                ('cache_config', 'enable_prefix_caching', True), ('cache_config', 'block_size', 32),
                ('model_config', 'max_model_len', 65536), ('speculative_config', 'num_speculative_tokens', 31),
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
