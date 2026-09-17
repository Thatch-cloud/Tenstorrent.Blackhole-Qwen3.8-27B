from types import SimpleNamespace
import unittest

import torch

from full_batched_features import verify_batched_features


class BatchedFeatureTests(unittest.TestCase):
    def fixture(self, *, reverse_rows=False, wrong_state=False):
        model = SimpleNamespace(args=SimpleNamespace(vocab_size=7),
            layers=[SimpleNamespace(forward=lambda hidden: hidden + 1) for index in range(3)])
        owned, state = {}, dict(rows=0)

        def snapshot(value):
            result = value.clone()
            owned[id(result)] = result
            return result

        def prefill(prompt):
            state['rows'] = 0
            return 2

        def forward(tokens, position, batched):
            values = torch.tensor(tokens).reshape(1, 1, len(tokens), 1).bfloat16()
            values = values + torch.arange(position, position + len(tokens)).reshape(1, 1, -1, 1)
            hidden = values.expand(1, 1, len(tokens), 5120).clone().bfloat16()
            if batched and reverse_rows:
                hidden = hidden.flip(2)
            buffers = []
            for layer in model.layers:
                hidden = layer.forward(hidden)
                buffers.append(hidden)
            logits = torch.zeros(len(tokens), 7)
            for index, token in enumerate(tokens):
                logits[index, (token + 1) % 7] = 1
            for buffer in buffers:
                buffer.zero_()
            state['rows'] += len(tokens) + int(batched and wrong_state)
            return [logits.clone(), logits.clone()] if batched else logits

        def decode(token, position, trace):
            self.assertFalse(trace)
            return forward([token], position, False)

        callbacks = dict(prefill=prefill, decode=decode, batch_decode=lambda tokens, position: forward(tokens, position, True),
            live_digest=lambda: state['rows'], kv_digest=lambda count: count, inactive_digest=lambda: 'inactive',
            snapshot=snapshot, release=lambda value: owned.pop(id(value)), storage_ids=lambda value: (value.data_ptr(),),
            local_host=lambda value: [part.clone() for part in value.chunk(2, dim=-1)])
        return model, callbacks, owned

    def test_each_batched_row_matches_its_serial_position(self):
        for rows in (8, 16, 32):
            model, callbacks, owned = self.fixture()
            result = verify_batched_features(model, [1] * 63, (0, 2), rows, **callbacks)
            self.assertEqual(len(result['checks']), 4)
            self.assertEqual(result['input_tokens'], [(index + 2) % 7 for index in range(rows)])
            self.assertEqual(owned, {})
            self.assertFalse(hasattr(model, '_qwen_target_feature_capture'))

    def test_reordered_features_or_wrong_state_fail_and_release(self):
        for options in (dict(reverse_rows=True), dict(wrong_state=True)):
            model, callbacks, owned = self.fixture(**options)
            original = [layer.forward for layer in model.layers]
            with self.subTest(options=options), self.assertRaises(AssertionError):
                verify_batched_features(model, [1] * 63, (0, 2), 8, **callbacks)
            self.assertEqual(owned, {})
            self.assertEqual([layer.forward for layer in model.layers], original)

    def test_invalid_geometry_fails_before_prefill(self):
        model, callbacks, owned = self.fixture()
        for rows in (True, 4, 64):
            with self.assertRaises(ValueError):
                verify_batched_features(model, [1] * 63, (0, 2), rows, **callbacks)
        self.assertEqual(owned, {})
