from contextlib import contextmanager
from types import SimpleNamespace
import os
import unittest
from unittest.mock import patch

from dspark_64k_mlp_scope import audit_scope, validate_result


class ReintegrationTests(unittest.TestCase):
    def test_all_layers_and_restoration_required(self):
        valid = dict(rows=16, layers=64, hits=[1] * 64, restored=True,
            native_bindings_unchanged=True, extra_weight_allocations=0)
        validate_result(valid)
        for field in valid:
            invalid = dict(valid)
            invalid.pop(field)
            with self.assertRaises(ValueError):
                validate_result(invalid)

    def test_original_request_called_inside_scope_and_restored(self):
        active = []

        def measure(operations, model, prompt, *, audit_features=False, captured_publication=False,
                target_attention_t16=False, commit_only_gdn=False, proposal_trace=False, native_attention=False):
            self.assertTrue(active)
            return {'exact': True}

        module = SimpleNamespace(measure_dspark_request=measure)

        class Arm:
            audit = dict(rows=16, layers=64, hits=[1] * 64, restored=True,
                native_bindings_unchanged=True, extra_weight_allocations=0)

            @contextmanager
            def install(self):
                active.append(True)
                try:
                    yield
                finally:
                    active.clear()

        with patch.dict(os.environ, QWEN_SPLITK_COMBINED='1', QWEN_64K_MLP_AUDIT='1',
                QWEN_DSPARK_SFPU_REQUEST_SCREEN='1', QWEN_DSPARK_SFPU_TIMED='0'):
            with audit_scope(module, lambda *args: Arm(), None):
                with self.assertRaisesRegex(ValueError, 'Isolated'):
                    module.measure_dspark_request(None, None, [0] * 65536)
                result = module.measure_dspark_request(None, None, [0] * 65536,
                    audit_features=True, captured_publication=True, target_attention_t16=True,
                    commit_only_gdn=True, proposal_trace=True, native_attention=True)
                self.assertTrue(result['exact'])
                self.assertFalse(result['mlp_64k_reintegration']['performance_qualified'])
                self.assertFalse(active)
        self.assertIs(module.measure_dspark_request, measure)
