from types import SimpleNamespace
import unittest
from unittest.mock import Mock

from dspark_prefill import FeatureChunk, FullHistoryCapture
from prefill_prefix_features import prefix_features


def chunk(start, rows):
    return FeatureChunk(start, rows,
        tuple(SimpleNamespace(shape=(1, 1, rows, 2560)) for unused in range(5)))


class PrefixFeaturesTests(unittest.TestCase):
    def fixture(self):
        operations = SimpleNamespace()
        model = SimpleNamespace(_forward_prefill_chunk_masked_tp=Mock())
        owner = FullHistoryCapture(operations, model, 4096)
        owner.chunks = [chunk(0, 2048), chunk(2048, 2048)]
        owner.children = [SimpleNamespace(close=Mock()), SimpleNamespace(close=Mock())]
        owner.started = owner.complete = True
        owner.cursor = 4096
        return operations, model, owner

    def test_complete_history_reuses_prefix_without_taking_ownership(self):
        operations, model, owner = self.fixture()
        original = owner.outputs()
        suffix_child = SimpleNamespace(close=Mock())
        with prefix_features(operations, model, 6144, owner, 4096) as capture:
            self.assertEqual(capture.cursor, 4096)
            self.assertIs(capture.chunks[0], original[0])
            with capture.capture():
                capture.chunks.append(chunk(4096, 2048))
                capture.children.append(suffix_child)
                capture.cursor = 6144
            self.assertEqual([value.start for value in capture.outputs()], [0, 2048, 4096])
            with self.assertRaisesRegex(ValueError, 'borrows'):
                owner.close()
        suffix_child.close.assert_called_once()
        self.assertEqual(owner.outputs(), original)
        for child in owner.children:
            child.close.assert_not_called()
        owner.close()
        self.assertTrue(owner.closed)

    def test_failed_suffix_capture_releases_only_suffix(self):
        operations, model, owner = self.fixture()
        child = SimpleNamespace(close=Mock())
        with self.assertRaisesRegex(RuntimeError, 'suffix'):
            with prefix_features(operations, model, 6144, owner, 4096) as capture:
                with capture.capture():
                    capture.children.append(child)
                    raise RuntimeError('suffix')
        child.close.assert_called_once()
        self.assertEqual(len(owner.outputs()), 2)
        self.assertNotIn('close', owner.__dict__)
        self.assertFalse(hasattr(owner, '_qwen_prefix_feature_borrower'))

    def test_missing_prefix_and_nested_borrow_rejected(self):
        operations, model, owner = self.fixture()
        for boundary in (128, 6144):
            with self.assertRaises(ValueError):
                with prefix_features(operations, model, 8192, owner, boundary):
                    self.fail('Invalid prefix admitted')
        with prefix_features(operations, model, 6144, owner, 4096):
            with self.assertRaises(ValueError):
                with prefix_features(operations, model, 6144, owner, 4096):
                    self.fail('Nested borrower admitted')

    def test_closed_or_different_model_owner_rejected(self):
        operations, model, owner = self.fixture()
        with self.assertRaises(ValueError):
            with prefix_features(operations, object(), 6144, owner, 4096):
                self.fail('Other model admitted')
        owner.close()
        with self.assertRaises(ValueError):
            with prefix_features(operations, model, 6144, owner, 4096):
                self.fail('Closed storage admitted')


if __name__ == '__main__':
    unittest.main()
