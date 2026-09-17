import os
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from frozen_gdn_cache_scope import runtime_scope
from frozen_gdn_cache_gate import REPORT_SHA256


class GdnCacheScopeTests(unittest.TestCase):
    def test_unallocated_hardware_rejected(self):
        with patch.dict(os.environ, {}, clear=True), self.assertRaises(ValueError):
            with runtime_scope('.'):
                pass

    def test_only_candidate_builds_change_and_scopes_restore(self):
        original_build = Mock(return_value=(1, 2, 3))
        scope = SimpleNamespace(build=original_build)
        gate = SimpleNamespace(qualify=lambda *args: {'report_sha256': 'control'})
        original_qualify = gate.qualify
        route = Mock()
        variants = SimpleNamespace(validate_route=route, REPORT_SHA256='control')
        shared_flags, admissions = [], []

        def measure(**kwargs):
            shared_flags.append(kwargs['gdn_shared_qk'])
            admissions.append(gate.qualify())
            for layer in range(48):
                scope.build(None, None, [], root='pinned')
            return {}

        full = SimpleNamespace(measure_dspark_request=measure)
        modules = dict(full_dspark_request=full, gdn_shared_qk_scope=scope,
            gdn_shared_qk_gate=gate, gdn_shared_qk_variants=variants)
        environment = dict(QWEN_FROZEN_COMBINED_RUNTIME='1', QWEN_CARDS_ALLOCATED='1',
            QWEN_DSPARK_REQUEST_CONTEXT='32768', TT_METAL_HOME='pinned')
        options = dict(captured_publication=True, fused_t16_mlp=True, target_attention_t16=True, score_layout=True)
        with patch.dict(os.environ, environment, clear=True), patch.dict('sys.modules', modules), \
                patch('frozen_gdn_cache_scope.qualify', return_value={'report_sha256': REPORT_SHA256}), \
                patch('frozen_gdn_cache_scope.build', return_value=(1, 2, 3)) as cached:
            with runtime_scope('.'):
                control = full.measure_dspark_request(**options)
                candidate = full.measure_dspark_request(**options, gdn_shared_qk=True)
                variants.validate_route(control, 'control')
                variants.validate_route(candidate, 'publication')
                with self.assertRaises(ValueError):
                    variants.validate_route(control, 'publication')
        self.assertEqual(shared_flags, [True, True])
        self.assertEqual(original_build.call_count, 48)
        self.assertEqual(cached.call_count, 48)
        self.assertEqual(admissions[1]['report_sha256'], REPORT_SHA256)
        self.assertEqual(candidate['gdn_input_cache']['builds'], 48)
        self.assertIs(scope.build, original_build)
        self.assertIs(gate.qualify, original_qualify)
        self.assertIs(full.measure_dspark_request, measure)
        self.assertIs(variants.validate_route, route)
        self.assertEqual(variants.REPORT_SHA256, 'control')


if __name__ == '__main__':
    unittest.main()
