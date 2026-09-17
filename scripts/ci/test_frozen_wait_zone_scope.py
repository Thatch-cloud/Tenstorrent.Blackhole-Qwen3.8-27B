from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from frozen_wait_zone_scope import runtime_scope, validate_marker_audit
from frozen_wait_zone_gate import SIMULATOR_SHA256, HARDWARE_SHA256


class MarkerScopeTests(unittest.TestCase):
    def test_diagnostic_scope_restores_and_rejects_timing_arm(self):
        original_projection, candidate_projection = object(), object()
        admission = Mock(return_value='baseline')
        fusion = SimpleNamespace(FusedProjection=original_projection, qualify_simulator=admission)
        variants = SimpleNamespace(REPORT_SHA256='baseline')
        evidence = dict(report_sha256=SIMULATOR_SHA256)

        def measure(**kwargs):
            self.assertIs(fusion.FusedProjection, candidate_projection)
            self.assertEqual(fusion.qualify_simulator(), evidence)
            return dict(instrumented_timing=True, committed_tokens_per_second=None)

        def route(value, arm):
            self.assertEqual(variants.REPORT_SHA256, SIMULATOR_SHA256)
            self.assertEqual(arm, 'publication')

        full = SimpleNamespace(measure_dspark_request=measure)
        shared = SimpleNamespace(validate_route=route)
        modules = dict(full_dspark_request=full, fused_t16_scope=fusion,
            dspark_fusion_variants=variants, gdn_shared_qk_variants=shared)
        environment = dict(QWEN_COMBINED_TRACE_PROFILE='1', QWEN_FROZEN_COMBINED_RUNTIME='1',
            QWEN_CARDS_ALLOCATED='1', QWEN_DSPARK_REQUEST_CONTEXT='32768')
        spec = SimpleNamespace(loader=SimpleNamespace(exec_module=lambda module: None))
        candidate = SimpleNamespace(FusedProjection=candidate_projection)
        with patch.dict('os.environ', environment, clear=True), patch.dict('sys.modules', modules), \
                patch('frozen_wait_zone_scope.qualify', return_value=evidence), \
                patch('frozen_wait_zone_scope.importlib.util.spec_from_file_location', return_value=spec), \
                patch('frozen_wait_zone_scope.importlib.util.module_from_spec', return_value=candidate):
            with runtime_scope('.'):
                with self.assertRaises(ValueError):
                    full.measure_dspark_request(combined_profile=False)
                value = full.measure_dspark_request(combined_profile=True, gdn_shared_qk=True,
                    fused_t16_mlp=True, audit_features=True)
                shared.validate_route(value, 'publication')
                with self.assertRaises(ValueError):
                    shared.validate_route(value, 'control')
            with self.assertRaisesRegex(RuntimeError, 'injected'):
                with runtime_scope('.'):
                    raise RuntimeError('injected')
        self.assertIs(full.measure_dspark_request, measure)
        self.assertIs(shared.validate_route, route)
        self.assertIs(fusion.FusedProjection, original_projection)
        self.assertIs(fusion.qualify_simulator, admission)
        self.assertEqual(variants.REPORT_SHA256, 'baseline')

    def test_throughput_claim_rejected(self):
        with self.assertRaises(ValueError):
            validate_marker_audit(dict(instrumented_timing=True, committed_tokens_per_second=200,
                mlp_wait_zones=dict(enabled=True, simulator_report_sha256=SIMULATOR_SHA256,
                    hardware_marker_sha256=HARDWARE_SHA256)))

    def test_unallocated_environment_rejected(self):
        with patch.dict('os.environ', {}, clear=True), self.assertRaises(ValueError):
            with runtime_scope('.'):
                pass


if __name__ == '__main__':
    unittest.main()
