from contextlib import contextmanager
import os
import inspect
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from dspark_64k_score_scope import score_scope


class ScoreScopeTests(unittest.TestCase):
    def test_loaded_audit_precedes_capture_and_restores(self):
        order, records = [], []

        class Device:
            def prepare_trace(self, anchor, *, audit=False):
                order.append('capture')
                self.audit = audit

        class Arm:
            def __init__(self, device, *, hardware_audit):
                self.restored = False
                self.evidence = hardware_audit

            @contextmanager
            def install(self):
                order.append('install')
                try:
                    yield
                finally:
                    self.restored = True
                    order.append('restore')

            def summary(self):
                self_test.assertTrue(self.restored)
                return dict(restored=True, calls=2)

        self_test = self

        def audit(*args):
            order.append('learned_audit')
            return {'passed': True}

        def measure(operations, model, prompt, predecessor, successor, *, audit_features, proposal_trace):
            Device().prepare_trace(1, audit=audit_features)
            return {'exact': True}

        module = SimpleNamespace(measure_dspark_request=measure)
        original = Device.prepare_trace
        with patch.dict(os.environ, QWEN_64K_SCORE_AUDIT='1', QWEN_64K_SHARED_QK_AUDIT='1',
                QWEN_DSPARK_SFPU_REQUEST_SCREEN='1', QWEN_DSPARK_SFPU_TIMED='0'):
            with score_scope(module, Device, Arm, audit, records):
                signature = inspect.signature(module.measure_dspark_request)
                self.assertEqual(signature, inspect.signature(measure))
                bound = signature.bind(None, SimpleNamespace(mesh_device=None), [0] * 65536,
                    None, None, audit_features=True, proposal_trace=True)
                self.assertEqual(len(bound.arguments['prompt']), 65536)
                result = module.measure_dspark_request(None, SimpleNamespace(mesh_device=None),
                    [0] * 65536, None, None, audit_features=True, proposal_trace=True)
        self.assertEqual(order, ['learned_audit', 'install', 'capture', 'restore'])
        self.assertIs(Device.prepare_trace, original)
        self.assertIs(module.measure_dspark_request, measure)
        self.assertEqual(len(records), 1)
        self.assertTrue(result['exact'])
