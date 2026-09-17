from types import SimpleNamespace
import unittest
from unittest.mock import Mock

from draft_operation_audit import audit_operations


class DraftOperationAuditTests(unittest.TestCase):
    def test_preserves_arguments_outputs_and_restores_bindings_after_success(self):
        result = object()
        matmul, gather, synchronize, progress = Mock(return_value=result), Mock(), Mock(), Mock()
        operations = SimpleNamespace(matmul=matmul, synchronize_device=synchronize,
            experimental=SimpleNamespace(all_gather_async=gather))
        mesh = object()
        with audit_operations(operations, mesh, progress):
            self.assertIs(operations.matmul('left', 'right', dtype='fp32'), result)
            operations.experimental.all_gather_async('input', num_links=4)
        matmul.assert_called_once_with('left', 'right', dtype='fp32')
        gather.assert_called_once_with('input', num_links=4)
        self.assertIs(operations.matmul, matmul)
        self.assertIs(operations.experimental.all_gather_async, gather)
        self.assertEqual(synchronize.call_count, 2)
        self.assertEqual([call.kwargs['phase'] for call in progress.call_args_list],
            ['enqueue', 'synchronize', 'complete'] * 2)

    def test_failure_retains_last_incomplete_operation_and_restores_bindings(self):
        original, progress = Mock(), Mock()
        operations = SimpleNamespace(generic_op=original, experimental=SimpleNamespace(),
            synchronize_device=Mock(side_effect=RuntimeError('device stalled')))
        with self.assertRaisesRegex(RuntimeError, 'device stalled'):
            with audit_operations(operations, object(), progress):
                operations.generic_op('descriptor')
        self.assertIs(operations.generic_op, original)
        self.assertEqual([call.kwargs['phase'] for call in progress.call_args_list], ['enqueue', 'synchronize'])
