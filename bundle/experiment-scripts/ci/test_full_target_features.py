from types import SimpleNamespace
import unittest

import torch

from full_target_features import feature_prompts, verify_features


class TargetFeatureValidationTests(unittest.TestCase):
    def test_template_budget_is_an_upper_bound_not_an_exact_length(self):
        calls = []

        def make_prompt(tokenizer, budget, variant):
            calls.append(budget)
            return list(range(budget // 32 * 32))

        prompts = feature_prompts(None, make_prompt, (63, 64, 65, 127, 128, 129))
        self.assertEqual(calls, [129, 258])
        self.assertEqual([len(prompt) for prompt in prompts], [63, 64, 65, 127, 128, 129])
        self.assertTrue(all(prompt == prompts[-1][:len(prompt)] for prompt in prompts))

    def test_non_growing_template_is_bounded(self):
        calls = []

        def make_prompt(tokenizer, budget, variant):
            calls.append(budget)
            return [1]

        with self.assertRaisesRegex(ValueError, 'Insufficient feature prompt tokens'):
            feature_prompts(None, make_prompt, (129,))
        self.assertEqual(calls, [129, 258, 516, 1032])

    def test_short_contexts_slice_a_template_sized_base(self):
        calls = []

        def make_prompt(tokenizer, length, seed):
            calls.append((tokenizer, length, seed))
            if length < 100:
                raise ValueError('Template exceeds prompt budget')
            return list(range(length))

        prompts = feature_prompts('tokenizer', make_prompt, (63, 64, 65))
        self.assertEqual(calls, [('tokenizer', 128, 0)])
        self.assertEqual([len(prompt) for prompt in prompts], [63, 64, 65])
        self.assertEqual(prompts[0], prompts[2][:63])
        with self.assertRaises(ValueError):
            feature_prompts(None, lambda *args: [1], (63,))

    def fixture(self, *, between_layers=False, corrupt_state=False, missing_shard=False):
        model = SimpleNamespace(layers=[SimpleNamespace(forward=lambda hidden: hidden + 1) for index in range(4)])
        state = dict(value=0, calls=0)
        owned = {}

        def snapshot(value):
            result = value.clone()
            owned[id(result)] = result
            return result

        def release(value):
            del owned[id(value)]

        def prefill(prompt):
            state['value'] = 0
            return 2

        def decode(token, position, trace):
            self.assertFalse(trace)
            self.assertEqual((token, position), (2, 63))
            state['calls'] += 1
            hidden = torch.arange(64).reshape(1, 1, 8, 8).bfloat16()
            buffers = []
            for index, layer in enumerate(model.layers):
                hidden = layer.forward(hidden)
                buffers.append(hidden)
                if between_layers and index == 1:
                    hidden = hidden + 1
            logits = hidden.clone()
            for buffer in buffers:
                buffer.zero_()
            state['value'] = 2 if corrupt_state and state['calls'] == 3 else 1
            return logits

        callbacks = dict(prefill=prefill, decode=decode, live_digest=lambda: state['value'],
            kv_digest=lambda count: count, inactive_digest=lambda: 'inactive', snapshot=snapshot,
            release=release, storage_ids=lambda value: (value.data_ptr(),),
            local_host=lambda value: [part.clone() for part in value.chunk(1 if missing_shard else 2, dim=-1)])
        return model, callbacks, owned

    def test_owned_features_match_independent_next_layer_inputs(self):
        model, callbacks, owned = self.fixture()
        forwards = [layer.forward for layer in model.layers]
        report = verify_features(model, [1] * 63, [0, 2], **callbacks)
        self.assertEqual(len(report['checks']), 4)
        self.assertTrue(all(check['exact'] for check in report['checks']))
        self.assertEqual([layer.forward for layer in model.layers], forwards)
        self.assertFalse(hasattr(model, '_qwen_target_feature_capture'))
        self.assertEqual(owned, {})

    def test_changed_boundary_state_or_missing_chip_cannot_pass(self):
        for options in (dict(between_layers=True), dict(corrupt_state=True), dict(missing_shard=True)):
            model, callbacks, owned = self.fixture(**options)
            forwards = [layer.forward for layer in model.layers]
            with self.subTest(options=options), self.assertRaises(AssertionError):
                verify_features(model, [1] * 63, [1, 2], **callbacks)
            self.assertEqual(owned, {})
            self.assertEqual([layer.forward for layer in model.layers], forwards)
            self.assertFalse(hasattr(model, '_qwen_target_feature_capture'))

    def test_final_layer_cannot_be_used_as_next_input_oracle(self):
        model, callbacks, owned = self.fixture()
        with self.assertRaises(ValueError):
            verify_features(model, [1] * 63, [3], **callbacks)
        self.assertEqual(owned, {})
