from types import SimpleNamespace
import unittest

from mtp_hidden_capture import MTPHiddenCapture


class MTPHiddenCaptureTests(unittest.TestCase):
    def fixture(self):
        def tensor(storage, value):
            return SimpleNamespace(shape=(1, 1, 8, 5120), dtype='bf16', layout='tile',
                                   storage=storage, value=value)
        normalized, destination = tensor(1, 7), tensor(2, 0)
        model = SimpleNamespace(_final_norm_decode=lambda hidden: normalized)
        def copy(source, target):
            target.value = source.value
        capture = MTPHiddenCapture(model, destination, copy=copy, storage_ids=lambda value: (value.storage,))
        return model, normalized, destination, capture

    def test_sharded_head_can_bypass_lm_head_without_losing_normalized_hidden(self):
        model, normalized, destination, capture = self.fixture()
        original = model._final_norm_decode
        for value in (7, 19, 7):
            normalized.value = value
            with capture.capture():
                self.assertIs(model._final_norm_decode(object()), normalized)
            self.assertIs(capture.output(), destination)
            self.assertEqual(destination.value, value)
            self.assertIs(model._final_norm_decode, original)
            self.assertFalse(hasattr(model, '_qwen_mtp_hidden_capture'))

    def test_missing_repeated_nested_and_failed_forward_invalidate_capture(self):
        for failure in ('missing', 'repeated', 'nested', 'forward'):
            model, _, _, capture = self.fixture()
            original = model._final_norm_decode
            with self.assertRaises(RuntimeError):
                with capture.capture():
                    if failure != 'missing':
                        model._final_norm_decode(None)
                    if failure == 'repeated':
                        model._final_norm_decode(None)
                    elif failure == 'nested':
                        with capture.capture():
                            pass
                    elif failure == 'forward':
                        raise RuntimeError('target failed after normalization')
            with self.assertRaises(RuntimeError):
                capture.output()
            self.assertIs(model._final_norm_decode, original)

    def test_alias_moved_and_wrong_shape_fail_before_copy(self):
        for fault in ('alias', 'moved', 'shape'):
            model, normalized, destination, capture = self.fixture()
            if fault == 'alias':
                normalized.storage = destination.storage
            elif fault == 'moved':
                destination.storage = 3
            else:
                normalized.shape = (1, 1, 1, 5120)
            with self.assertRaises(ValueError):
                with capture.capture():
                    model._final_norm_decode(None)
            self.assertEqual(destination.value, 0)

    def test_same_numeric_addresses_on_different_chips_are_not_aliases(self):
        model, normalized, destination, _ = self.fixture()
        capture = MTPHiddenCapture(model, destination,
            copy=lambda source, target: setattr(target, 'value', source.value),
            storage_ids=lambda value: (value.storage, value.storage))
        with capture.capture():
            model._final_norm_decode(None)
        self.assertEqual(capture.output().value, normalized.value)


if __name__ == '__main__':
    unittest.main()
