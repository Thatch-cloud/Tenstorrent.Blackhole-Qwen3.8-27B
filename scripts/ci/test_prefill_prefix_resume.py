from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import torch

from prefill_prefix_resume import resume_eager_prefill


class PrefixResumeTests(unittest.TestCase):
    def fixture(self):
        operations = SimpleNamespace(synchronize_device=Mock(), deallocate=Mock())
        model = SimpleNamespace(device='mesh', _set_vision_merge=Mock(),
            _forward_prefill_chunk_masked_tp=Mock(side_effect=lambda *args, **kwargs: object()),
            prefill_masked_bucket=Mock(return_value='tail-logits'),
            _masked_bucket_logits_tp=Mock(return_value='full-logits'),
            _reset_gdn_state_for_new_sequence=Mock())
        return operations, model

    def test_changed_suffix_runs_only_absolute_suffix_chunks(self):
        operations, model = self.fixture()
        tokens = torch.arange(8193).reshape(1, -1)
        restore = Mock(side_effect=lambda position:
            self.assertEqual(model._forward_prefill_chunk_masked_tp.call_count, 0))
        result = resume_eager_prefill(operations, model, tokens, 'pages',
            actual_len=8193, prefix_position=4096, restore=restore)
        self.assertEqual(result, 'tail-logits')
        restore.assert_called_once_with(4096)
        calls = model._forward_prefill_chunk_masked_tp.call_args_list
        self.assertEqual([call.args[2] for call in calls], [4096, 6144])
        self.assertTrue(torch.equal(calls[0].args[0], tokens[:, 4096:6144]))
        tail = model.prefill_masked_bucket.call_args
        self.assertEqual(tail.kwargs['chunk_start'], 8192)
        self.assertEqual(tail.kwargs['actual_len'], 1)
        self.assertEqual(operations.deallocate.call_count, 2)
        model._reset_gdn_state_for_new_sequence.assert_not_called()

    def test_tail_only_needs_no_cached_hidden_tensor(self):
        operations, model = self.fixture()
        resume_eager_prefill(operations, model, torch.zeros(1, 4097), 'pages',
            actual_len=4097, prefix_position=4096, restore=Mock())
        model._forward_prefill_chunk_masked_tp.assert_not_called()
        operations.deallocate.assert_not_called()
        self.assertEqual(model.prefill_masked_bucket.call_args.kwargs['chunk_start'], 4096)

    def test_exact_chunk_suffix_uses_last_hidden_then_releases(self):
        operations, model = self.fixture()
        result = resume_eager_prefill(operations, model, torch.zeros(1, 6144), 'pages',
            actual_len=6144, prefix_position=4096, restore=Mock())
        self.assertEqual(result, 'full-logits')
        operations.deallocate.assert_called_once_with(model._masked_bucket_logits_tp.call_args.args[0])
        model.prefill_masked_bucket.assert_not_called()

    def test_failed_restore_never_executes_suffix(self):
        operations, model = self.fixture()
        with self.assertRaisesRegex(RuntimeError, 'restore'):
            resume_eager_prefill(operations, model, torch.zeros(1, 4097), 'pages',
                actual_len=4097, prefix_position=4096, restore=Mock(side_effect=RuntimeError('restore')))
        model._forward_prefill_chunk_masked_tp.assert_not_called()
        model.prefill_masked_bucket.assert_not_called()

    def test_failed_logits_releases_hidden(self):
        operations, model = self.fixture()
        model._masked_bucket_logits_tp.side_effect = RuntimeError('logits')
        with self.assertRaisesRegex(RuntimeError, 'logits'):
            resume_eager_prefill(operations, model, torch.zeros(1, 6144), 'pages',
                actual_len=6144, prefix_position=4096, restore=Mock())
        operations.deallocate.assert_called_once()

    def test_invalid_boundary_rejected_before_restore(self):
        operations, model = self.fixture()
        restore = Mock()
        for position in (0, 128, 4096, True):
            with self.assertRaises(ValueError):
                resume_eager_prefill(operations, model, torch.zeros(1, 4096), 'pages',
                    actual_len=4096, prefix_position=position, restore=restore)
        restore.assert_not_called()


if __name__ == '__main__':
    unittest.main()
