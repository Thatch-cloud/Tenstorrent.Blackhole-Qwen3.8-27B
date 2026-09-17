from types import SimpleNamespace
import unittest

import torch

from full_prefill_features import verify_prefill_features


class PrefillFeatureTests(unittest.TestCase):
    def fixture(self, *, shifted=False, corrupt_state=False, missing_shard=False, repeat=False):
        model = SimpleNamespace(args=SimpleNamespace(vocab_size=5120),
            layers=[SimpleNamespace(forward=lambda hidden: hidden + 1) for index in range(4)])
        state = dict(position=0)
        owned = {}

        def snapshot(value):
            result = value.clone()
            if shifted and hasattr(model, '_qwen_target_feature_capture'):
                result = result.roll(1, dims=2)
            owned[id(result)] = result
            return result

        def release(value):
            del owned[id(value)]

        def prefill(prompt):
            length = len(prompt)
            bucket = ((length + 127) // 128) * 128
            hidden = torch.arange(bucket).reshape(1, 1, bucket, 1).expand(1, 1, bucket, 5120).clone().bfloat16()
            buffers = []
            for layer in model.layers:
                hidden = layer.forward(hidden)
                buffers.append(hidden)
                if repeat:
                    hidden = layer.forward(hidden)
                    buffers.append(hidden)
            result = hidden[:, :, length - 1].clone()
            for value in buffers:
                value.zero_()
            state['position'] = length + int(corrupt_state and hasattr(model, '_qwen_target_feature_capture'))
            return result

        def decode(token, position, trace):
            self.assertFalse(trace)
            state['position'] = position + 1
            return torch.full((1, 5120), float(position))

        callbacks = dict(prefill_logits=prefill, decode=decode, live_digest=lambda: state['position'],
            kv_digest=lambda length: (length, state['position']), inactive_digest=lambda: 'inactive',
            snapshot=snapshot, release=release, storage_ids=lambda value: (value.data_ptr(),),
            local_host=lambda value: [part.clone() for part in value.chunk(1 if missing_shard else 2, dim=-1)])
        return model, callbacks, owned

    def test_valid_rows_exclude_bucket_padding_and_preserve_continuation(self):
        for length in (63, 64, 65, 127, 128, 129):
            model, callbacks, owned = self.fixture()
            report = verify_prefill_features(model, [1] * length, (0, 2), **callbacks)
            self.assertEqual(len(report['checks']), 4)
            self.assertTrue(all(check['valid_rows'] == length for check in report['checks']))
            self.assertTrue(all(check['padding_rows'] == ((length + 127) // 128) * 128 - length
                                for check in report['checks']))
            self.assertEqual(report['correction_steps'], 1)
            self.assertFalse(owned)
            self.assertFalse(hasattr(model, '_qwen_target_feature_capture'))

    def test_shifted_rows_state_drift_missing_shard_and_multiple_chunks_fail(self):
        for options in (dict(shifted=True), dict(corrupt_state=True), dict(missing_shard=True), dict(repeat=True)):
            model, callbacks, owned = self.fixture(**options)
            forwards = [layer.forward for layer in model.layers]
            with self.subTest(options=options), self.assertRaises((AssertionError, RuntimeError)):
                verify_prefill_features(model, [1] * 63, (0, 2), **callbacks)
            self.assertFalse(owned)
            self.assertEqual([layer.forward for layer in model.layers], forwards)
            self.assertFalse(hasattr(model, '_qwen_target_feature_capture'))

    def test_unbounded_prompt_and_final_layer_are_rejected(self):
        model, callbacks, owned = self.fixture()
        for length, taps in ((0, (0,)), (257, (0,)), (63, (3,)), (63, (True,))):
            with self.assertRaises(ValueError):
                verify_prefill_features(model, [1] * length, taps, **callbacks)
        self.assertFalse(owned)
