from contextlib import redirect_stdout
import io
import os
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from dspark_draft_profile import DraftProfile
from request_verifier_profile import PROFILER_FLAGS


class DraftProfileTests(unittest.TestCase):
    def fixture(self):
        operations = SimpleNamespace(execute_trace=Mock(return_value='trace_result'),
            synchronize_device=Mock(), ReadDeviceProfiler=Mock())
        mesh = object()
        class Prepared:
            def __init__(self):
                self.trace, self.mesh, self.operations, self.closed = 7, mesh, operations, False
                self.device = SimpleNamespace(max_drafts=15, position=4096)
                self.operations.execute_trace(mesh, self.trace, cq_id=0, blocking=True)

            def propose(self, token, count):
                self.device.position += 3
                self.operations.execute_trace(mesh, self.trace, cq_id=0, blocking=True)
                return (token,) * count
        return operations, mesh, Prepared

    def test_records_real_warmup_and_proposals_without_changing_results(self):
        operations, mesh, prepared_type = self.fixture()
        original = (operations.execute_trace, prepared_type.__init__, prepared_type.propose)
        with patch.dict(os.environ, {name: '1' for name in PROFILER_FLAGS}), redirect_stdout(io.StringIO()):
            observer = DraftProfile(operations, mesh, signpost=Mock())
            with observer.install(prepared_type):
                operations.execute_trace(mesh, 99, blocking=False)
                prepared = prepared_type()
                for token in range(3):
                    self.assertEqual(prepared.propose(token, 15), (token,) * 15)
            result = observer.summary()
        self.assertEqual(result['trace_counts'], {7: 4})
        self.assertEqual([record['position'] for record in result['records']], [4096, 4099, 4102, 4105])
        self.assertEqual(original[0].call_count, 5)
        self.assertEqual(operations.ReadDeviceProfiler.call_count, 8)
        self.assertIs(operations.execute_trace, original[0])
        self.assertIs(prepared_type.__init__, original[1])
        self.assertIs(prepared_type.propose, original[2])

    def test_failure_restores_hooks_and_cannot_qualify(self):
        operations, mesh, prepared_type = self.fixture()
        original = operations.execute_trace
        original.side_effect = RuntimeError('device failed')
        with patch.dict(os.environ, {name: '1' for name in PROFILER_FLAGS}), redirect_stdout(io.StringIO()):
            observer = DraftProfile(operations, mesh, signpost=Mock())
            with self.assertRaisesRegex(RuntimeError, 'device failed'):
                with observer.install(prepared_type):
                    prepared_type()
            self.assertIs(operations.execute_trace, original)
            self.assertTrue(observer.restored)
            self.assertEqual(observer.records, [])
            with self.assertRaises(ValueError):
                observer.summary()

    def test_missing_profiler_configuration_rejected(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(ValueError):
                DraftProfile(None, None, signpost=Mock())


if __name__ == '__main__':
    unittest.main()
