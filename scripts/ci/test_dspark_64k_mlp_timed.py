from contextlib import contextmanager
from types import SimpleNamespace
import os
import unittest
from unittest.mock import patch

import dspark_splitk_timed as splitk
from dspark_64k_mlp_timed import identity_scope, measurement_scope, SCREEN_RUN


class MlpTimingTests(unittest.TestCase):
    def test_identity_restores(self):
        original = splitk.SCREEN_RUN
        with identity_scope():
            self.assertEqual(splitk.SCREEN_RUN, SCREEN_RUN)
        self.assertEqual(splitk.SCREEN_RUN, original)

    def test_clean_measurement_only(self):
        active = []

        def measure(operations, model, prompt, *, audit_features, max_new_tokens,
                captured_publication, target_attention_t16, commit_only_gdn, proposal_trace, native_attention):
            self.assertTrue(active)
            return dict(exact=True)

        class Arm:
            audit = dict(rows=16, layers=64, hits=[2] * 64, restored=True,
                native_bindings_unchanged=True, extra_weight_allocations=0)

            @contextmanager
            def install(self):
                active.append(True)
                try:
                    yield
                finally:
                    active.clear()

        module = SimpleNamespace(measure_dspark_request=measure)
        options = dict(audit_features=False, max_new_tokens=256, captured_publication=True,
            target_attention_t16=True, commit_only_gdn=True, proposal_trace=True, native_attention=True)
        with patch.object(splitk, 'require_timed'), patch.dict(os.environ,
                QWEN_64K_MLP_TIMED='1', QWEN_64K_MLP_AUDIT='0'):
            with measurement_scope(module, lambda *args: Arm(), None):
                result = module.measure_dspark_request(None, None, [0] * 65536, **options)
                self.assertIn('mlp_64k_reintegration', result)
                with self.assertRaises(ValueError):
                    module.measure_dspark_request(None, None, [0] * 65536,
                        **dict(options, audit_features=True))
        self.assertIs(module.measure_dspark_request, measure)
        self.assertFalse(active)
