import os
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from frozen_mlp_buffer_scope import runtime_scope
from frozen_mlp_buffer_gate import REPORT_SHA256


class BufferScopeTests(unittest.TestCase):
    def test_requires_allocated_hardware(self):
        with patch.dict(os.environ, {}, clear=True), self.assertRaises(ValueError):
            with runtime_scope('.'):
                pass

    def test_both_arms_use_shared_qk_only_candidate_changes_projection(self):
        native = object()
        candidate = object()
        qualification = lambda: {'report_sha256': 'control'}
        fused = SimpleNamespace(FusedProjection=native, qualify_simulator=qualification)
        fusion = SimpleNamespace(REPORT_SHA256='control')
        calls = []

        def measure(**kwargs):
            calls.append((kwargs['gdn_shared_qk'], fused.FusedProjection, fused.qualify_simulator()))
            return {}

        original_route = Mock()
        shared = SimpleNamespace(validate_route=original_route)
        full = SimpleNamespace(measure_dspark_request=measure)
        modules = dict(full_dspark_request=full, fused_t16_scope=fused,
            dspark_fusion_variants=fusion, gdn_shared_qk_variants=shared)
        environment = dict(QWEN_FROZEN_COMBINED_RUNTIME='1', QWEN_CARDS_ALLOCATED='1',
            QWEN_DSPARK_REQUEST_CONTEXT='32768')
        spec = SimpleNamespace(loader=SimpleNamespace(exec_module=lambda module: None))
        options = dict(captured_publication=True, fused_t16_mlp=True,
            target_attention_t16=True, score_layout=True)
        with patch.dict(os.environ, environment, clear=True), patch.dict('sys.modules', modules), \
                patch('frozen_mlp_buffer_scope.qualify', return_value={'report_sha256': REPORT_SHA256}), \
                patch('frozen_mlp_buffer_scope.importlib.util.spec_from_file_location', return_value=spec), \
                patch('frozen_mlp_buffer_scope.importlib.util.module_from_spec',
                    return_value=SimpleNamespace(FusedProjection=candidate)):
            with runtime_scope('.'):
                control = full.measure_dspark_request(**options)
                changed = full.measure_dspark_request(**options, gdn_shared_qk=True)
                shared.validate_route(control, 'control')
                shared.validate_route(changed, 'publication')
                with self.assertRaises(ValueError):
                    shared.validate_route(control, 'publication')
        self.assertEqual([entry[0] for entry in calls], [True, True])
        self.assertIs(calls[0][1], native)
        self.assertIs(calls[1][1], candidate)
        self.assertEqual(calls[1][2]['report_sha256'], REPORT_SHA256)
        self.assertIs(fused.FusedProjection, native)
        self.assertIs(fused.qualify_simulator, qualification)
        self.assertIs(full.measure_dspark_request, measure)
        self.assertIs(shared.validate_route, original_route)
        self.assertEqual(fusion.REPORT_SHA256, 'control')
        self.assertEqual([call.args[1] for call in original_route.call_args_list], ['publication', 'publication'])


if __name__ == '__main__':
    unittest.main()
