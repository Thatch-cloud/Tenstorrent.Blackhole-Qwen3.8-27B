from contextlib import contextmanager
import os
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from frozen_draft_tail_scope import runtime_scope
from frozen_draft_tail_gate import HARDWARE_SHA256


class DraftTailScopeTests(unittest.TestCase):
    def test_both_arms_retain_norm_scope_and_restore_assembly(self):
        native = Mock(return_value='original')
        layer = SimpleNamespace(append_queries=native)
        route = Mock()
        variants = SimpleNamespace(validate_route=route)
        observed, active = [], []

        @contextmanager
        def norm(directory):
            active.append(True)
            try:
                yield
            finally:
                active.pop()

        def measure(**kwargs):
            self.assertTrue(active)
            self.assertIs(kwargs['gdn_shared_qk'], True)
            observed.append([layer.append_queries(position=33024, proposals=15) for _ in range(10)])
            return {}

        full = SimpleNamespace(measure_dspark_request=measure)
        modules = dict(full_dspark_request=full, dspark_native_cached_layer=layer,
            gdn_shared_qk_variants=variants)
        with patch.dict('sys.modules', modules), patch.dict(os.environ, {}, clear=True), \
                patch('frozen_draft_tail_scope.norm_scope', norm), \
                patch('frozen_draft_tail_scope.qualify_hardware', return_value={'passed': True}), \
                patch('frozen_draft_tail_scope.append_queries', return_value='candidate') as candidate:
            with runtime_scope('.'):
                control = full.measure_dspark_request()
                selected = full.measure_dspark_request(gdn_shared_qk=True)
                self.assertFalse(control['draft_tail']['enabled'])
                self.assertEqual(selected['draft_tail'], dict(enabled=True, calls=10, report_sha256=HARDWARE_SHA256))
                variants.validate_route(control, 'control')
                variants.validate_route(selected, 'publication')
                with self.assertRaises(ValueError):
                    variants.validate_route(control, 'publication')
                with self.assertRaises(ValueError):
                    full.measure_dspark_request(gdn_shared_qk=1)
            self.assertEqual(candidate.call_count, 10)
            self.assertIs(layer.append_queries, native)
            self.assertIs(full.measure_dspark_request, measure)
            self.assertIs(variants.validate_route, route)
            self.assertEqual(observed, [['original'] * 10, ['candidate'] * 10])
            self.assertEqual([call.args[1] for call in route.call_args_list], ['publication', 'publication'])
            with self.assertRaisesRegex(RuntimeError, 'injected'):
                with runtime_scope('.'):
                    raise RuntimeError('injected')
            self.assertIs(layer.append_queries, native)

    def test_profiler_cannot_contaminate_comparison(self):
        from frozen_draft_tail_scope import comparison_scope
        with patch.dict(os.environ, {'QWEN_COMBINED_TRACE_PROFILE': '1'}):
            with self.assertRaisesRegex(ValueError, 'Unprofiled'):
                with comparison_scope('.'):
                    self.fail('Profiled comparison was admitted')
