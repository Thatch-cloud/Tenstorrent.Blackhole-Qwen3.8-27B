from contextlib import contextmanager
import os
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from frozen_gdn_norm_scope import runtime_scope
from frozen_gdn_norm_gate import REPORT_SHA256


class NormScopeTests(unittest.TestCase):
    def test_composed_arms_and_restoration(self):
        active = []
        observed = []
        route = Mock()
        variants = SimpleNamespace(validate_route=route, REPORT_SHA256='original')
        native_build = Mock(return_value=(1, 2, 3))
        scope = SimpleNamespace(build=native_build)
        native_gate = Mock(return_value={'report_sha256': 'original'})
        gate = SimpleNamespace(qualify=native_gate)
        history = SimpleNamespace(StableHistoryKV=object())
        publication = SimpleNamespace()

        @contextmanager
        def writer(history_class, publication_module, records):
            active.append(True)
            try:
                yield
            finally:
                active.pop()
                records.append(dict(initial_position=32768, capacity=33024, failed=False,
                    restored=True, prepared=10, committed=10, discarded=0, max_touched_rows=64))

        def measure(**kwargs):
            observed.append((kwargs['gdn_shared_qk'], bool(active), gate.qualify()))
            for layer in range(48):
                scope.build(None, None, [], root='native')
            return {}

        full = SimpleNamespace(measure_dspark_request=measure)
        modules = dict(full_dspark_request=full, gdn_shared_qk_scope=scope,
            gdn_shared_qk_gate=gate, gdn_shared_qk_variants=variants,
            dspark_stable_history=history, dspark_publication_scope=publication)
        environment = dict(QWEN_FROZEN_COMBINED_RUNTIME='1', QWEN_CARDS_ALLOCATED='1',
            QWEN_DSPARK_REQUEST_CONTEXT='32768', TT_METAL_HOME='native')
        options = dict(captured_publication=True, fused_t16_mlp=True,
            target_attention_t16=True, score_layout=True)
        with patch.dict(os.environ, environment, clear=True), patch.dict('sys.modules', modules), \
                patch('frozen_incremental_scope.qualify'), \
                patch('frozen_incremental_scope.incremental_history', writer), \
                patch('frozen_gdn_norm_scope.qualify', return_value={'report_sha256': REPORT_SHA256}), \
                patch('frozen_gdn_norm_scope.build', return_value=(1, 2, 3)) as candidate_build:
            with runtime_scope('.'):
                control = full.measure_dspark_request(**options)
                candidate = full.measure_dspark_request(**options, gdn_shared_qk=True)
                self.assertTrue(control['incremental_history']['enabled'])
                self.assertTrue(candidate['incremental_history']['enabled'])
                variants.validate_route(control, 'control')
                variants.validate_route(candidate, 'publication')
                with self.assertRaises(ValueError):
                    variants.validate_route(control, 'publication')
            self.assertEqual(candidate_build.call_count, 48)
            with self.assertRaisesRegex(RuntimeError, 'injected'):
                with runtime_scope('.'):
                    raise RuntimeError('injected')
        self.assertEqual(native_build.call_count, 48)
        self.assertEqual([(shared, incremental) for shared, incremental, evidence in observed],
            [(True, True), (True, True)])
        self.assertEqual(observed[1][2]['report_sha256'], REPORT_SHA256)
        self.assertIs(full.measure_dspark_request, measure)
        self.assertIs(variants.validate_route, route)
        self.assertIs(scope.build, native_build)
        self.assertIs(gate.qualify, native_gate)
        self.assertEqual(variants.REPORT_SHA256, 'original')
        self.assertFalse(active)


if __name__ == '__main__':
    unittest.main()
