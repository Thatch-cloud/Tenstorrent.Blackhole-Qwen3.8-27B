from types import SimpleNamespace
import unittest
from unittest.mock import Mock

from prefill_prefix_boundary import checkpoint_boundary


class PrefixBoundaryTests(unittest.TestCase):
    def fixture(self):
        order = []
        model = SimpleNamespace(_forward_prefill_chunk_masked_tp=Mock(
            side_effect=lambda tokens, rows, start, pages, bucket: order.append(('chunk', start)) or 'hidden'))
        checkpoint = SimpleNamespace(capture=Mock(side_effect=lambda position: order.append(('save', position))),
            operations=SimpleNamespace(deallocate=Mock()))
        return model, checkpoint, order

    def test_capture_occurs_before_suffix_mutation(self):
        model, checkpoint, order = self.fixture()
        original = model._forward_prefill_chunk_masked_tp
        with checkpoint_boundary(model, checkpoint, 4096) as evidence:
            for start in (0, 2048, 4096):
                model._forward_prefill_chunk_masked_tp(None, 2048, start, None, 2048)
        self.assertEqual(order, [('chunk', 0), ('chunk', 2048), ('save', 4096), ('chunk', 4096)])
        self.assertTrue(evidence['complete'] and evidence['restored'])
        self.assertIs(model._forward_prefill_chunk_masked_tp, original)

    def test_checkpoint_failure_releases_hidden_and_restores_wrapper(self):
        model, checkpoint, unused = self.fixture()
        checkpoint.capture.side_effect = RuntimeError('copy')
        original = model._forward_prefill_chunk_masked_tp
        with self.assertRaisesRegex(RuntimeError, 'copy'):
            with checkpoint_boundary(model, checkpoint, 2048) as evidence:
                model._forward_prefill_chunk_masked_tp(None, 2048, 0, None, 2048)
        checkpoint.operations.deallocate.assert_called_once_with('hidden')
        self.assertFalse(evidence['complete'])
        self.assertTrue(evidence['restored'])
        self.assertIs(model._forward_prefill_chunk_masked_tp, original)

    def test_missing_boundary_is_not_publishable(self):
        model, checkpoint, unused = self.fixture()
        with self.assertRaisesRegex(ValueError, 'not reached'):
            with checkpoint_boundary(model, checkpoint, 4096) as evidence:
                model._forward_prefill_chunk_masked_tp(None, 2048, 0, None, 2048)
        self.assertFalse(evidence['complete'])
        checkpoint.capture.assert_not_called()

    def test_crossing_skipping_and_nested_boundaries_rejected(self):
        for rows, start in ((4096, 0), (2048, 2048)):
            model, checkpoint, unused = self.fixture()
            with self.assertRaises(ValueError):
                with checkpoint_boundary(model, checkpoint, 2048):
                    model._forward_prefill_chunk_masked_tp(None, rows, start, None, rows)
            checkpoint.capture.assert_not_called()
        model, checkpoint, unused = self.fixture()
        with self.assertRaises(ValueError):
            with checkpoint_boundary(model, checkpoint, 2048):
                with checkpoint_boundary(model, checkpoint, 2048):
                    self.fail('Nested boundary admitted')


if __name__ == '__main__':
    unittest.main()
