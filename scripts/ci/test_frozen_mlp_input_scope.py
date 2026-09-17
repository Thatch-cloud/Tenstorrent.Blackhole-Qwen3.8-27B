from contextlib import contextmanager
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from frozen_mlp_input_scope import runtime_scope
from frozen_mlp_input_gate import REPORT_SHA256


class InputScopeTests(unittest.TestCase):
    def test_both_arms_keep_base_and_only_candidate_changes_mlp(self):
        original_projection, candidate_projection = object(), object()
        original_admission = Mock(return_value='baseline')
        fusion = SimpleNamespace(FusedProjection=original_projection, qualify_simulator=original_admission)
        variants = SimpleNamespace(REPORT_SHA256='baseline')
        observations = []

        def measure(**kwargs):
            observations.append((kwargs['gdn_shared_qk'], fusion.FusedProjection, fusion.qualify_simulator()))
            return dict(base_enabled=kwargs['gdn_shared_qk'])

        def route(value, arm):
            self.assertEqual(arm, 'publication')
            self.assertTrue(value['base_enabled'])
            self.assertEqual(variants.REPORT_SHA256,
                REPORT_SHA256 if value['mlp_input_prefetch']['enabled'] else 'baseline')

        full = SimpleNamespace(measure_dspark_request=measure)
        shared = SimpleNamespace(validate_route=route)
        entered = []

        @contextmanager
        def base(directory):
            entered.append(True)
            try:
                yield
            finally:
                entered.pop()

        modules = dict(full_dspark_request=full, fused_t16_scope=fusion,
            dspark_fusion_variants=variants, gdn_shared_qk_variants=shared)
        candidate = SimpleNamespace(FusedProjection=candidate_projection)
        spec = SimpleNamespace(loader=SimpleNamespace(exec_module=lambda module: None))
        evidence = dict(report_sha256=REPORT_SHA256)
        with patch.dict('sys.modules', modules), patch('frozen_mlp_input_scope.norm_scope', base), \
                patch('frozen_mlp_input_scope.qualify', return_value=evidence), \
                patch('frozen_mlp_input_scope.importlib.util.spec_from_file_location', return_value=spec), \
                patch('frozen_mlp_input_scope.importlib.util.module_from_spec', return_value=candidate):
            with runtime_scope('.'):
                self.assertTrue(entered)
                control = full.measure_dspark_request()
                candidate_result = full.measure_dspark_request(gdn_shared_qk=True)
                shared.validate_route(control, 'control')
                shared.validate_route(candidate_result, 'publication')
                with self.assertRaises(ValueError):
                    shared.validate_route(control, 'publication')
            with self.assertRaisesRegex(RuntimeError, 'injected'):
                with runtime_scope('.'):
                    raise RuntimeError('injected')
        self.assertEqual(observations, [(True, original_projection, 'baseline'),
            (True, candidate_projection, evidence)])
        self.assertIs(full.measure_dspark_request, measure)
        self.assertIs(shared.validate_route, route)
        self.assertIs(fusion.FusedProjection, original_projection)
        self.assertIs(fusion.qualify_simulator, original_admission)
        self.assertEqual(variants.REPORT_SHA256, 'baseline')
        self.assertFalse(entered)


if __name__ == '__main__':
    unittest.main()
