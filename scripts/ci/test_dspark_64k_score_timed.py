from contextlib import contextmanager
import inspect
import os
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from dspark_64k_score_timed import measurement_scope


class ScoreTimingTests(unittest.TestCase):
    def test_clean_capture_uses_candidate_until_request_finishes(self):
        events, records = [], []
        class Device:
            def prepare_trace(self, anchor, *, audit=False):
                self_test.assertFalse(audit)
                events.append('capture')
        class Arm:
            def __init__(self, device, *, hardware_audit):
                self_test.assertTrue(hardware_audit['passed'])
            @contextmanager
            def install(self):
                events.append('install')
                try:
                    yield
                finally:
                    events.append('restore')
            def summary(self):
                self_test.assertEqual(events[-1], 'restore')
                return dict(calls=2, restored=True)
        self_test = self
        def hardware(*args):
            events.append('loaded_weight_audit')
            return dict(passed=True)
        def measure(prompt, operations, model, predecessor, successor, *, audit_features,
                max_new_tokens, proposal_trace):
            Device().prepare_trace(1, audit=audit_features)
            events.append('complete_request')
            return dict(exact=True)
        module = SimpleNamespace(measure_dspark_request=measure)
        original_prepare = Device.prepare_trace
        with patch.dict(os.environ, QWEN_64K_SCORE_TIMED='1', QWEN_64K_SHARED_QK_TIMED='1',
                QWEN_DSPARK_SFPU_TIMED='1', QWEN_DSPARK_SFPU_REQUEST_SCREEN='0'):
            with measurement_scope(module, Device, Arm, hardware, records):
                self.assertEqual(inspect.signature(module.measure_dspark_request), inspect.signature(measure))
                result = module.measure_dspark_request([1] * 65536, None, SimpleNamespace(mesh_device=None),
                    None, None, audit_features=False, max_new_tokens=256, proposal_trace=True)
        self.assertEqual(events, ['loaded_weight_audit', 'install', 'capture', 'complete_request', 'restore'])
        self.assertTrue(result['exact'])
        self.assertEqual(len(records), 1)
        self.assertIs(module.measure_dspark_request, measure)
        self.assertIs(Device.prepare_trace, original_prepare)

    def test_rejects_unselected_timing(self):
        with patch.dict(os.environ, QWEN_64K_SCORE_TIMED='0'):
            with self.assertRaises(ValueError):
                with measurement_scope(None, None, None, None, []):
                    self.fail('Unselected timing entered')
