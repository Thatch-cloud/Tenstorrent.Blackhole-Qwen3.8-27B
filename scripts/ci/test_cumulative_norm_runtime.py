import copy
from contextlib import contextmanager
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import cumulative_norm_runtime as runtime


class CumulativeNormTests(unittest.TestCase):
    def test_staged_selector_preserves_default_and_rejects_unknown_flag(self):
        calls = []

        @contextmanager
        def original(directory):
            calls.append(directory)
            yield 'original'

        module = SimpleNamespace(runtime_scope=original)
        with patch.dict('sys.modules', {'frozen_gdn_norm_scope': module}), \
                patch.dict('os.environ', {'QWEN_CUMULATIVE_NORM': '0'}):
            with runtime.selected_runtime_scope('sources') as result:
                self.assertEqual(result, 'original')
            self.assertEqual(calls, ['sources'])
        with patch.dict('os.environ', {'QWEN_CUMULATIVE_NORM': 'invalid'}):
            with self.assertRaises(ValueError):
                with runtime.selected_runtime_scope('sources'):
                    self.fail('Unknown normalization flag accepted')

    def test_selection_controls_builder_admission_and_reporting(self):
        prefetch = Mock(return_value=(1, 2, 3))
        scatter = Mock(return_value=(1, 2, 3))
        original_build, original_gate = Mock(), Mock()
        scope = SimpleNamespace(build=original_build)
        gate = SimpleNamespace(qualify=original_gate)
        routes = []
        variants = SimpleNamespace(REPORT_SHA256='native')
        variants.validate_route = lambda request, arm: routes.append(variants.REPORT_SHA256)
        original_route = variants.validate_route

        def measure(**options):
            for layer in range(48):
                scope.build(None, None, [])
            return dict(gdn_shared_qk=dict(admission=gate.qualify()))

        full = SimpleNamespace(measure_dspark_request=measure)
        modules = dict(full_dspark_request=full, gdn_shared_qk_scope=scope,
                       gdn_shared_qk_gate=gate, gdn_shared_qk_variants=variants)
        with patch.dict('sys.modules', modules), \
                patch.dict('os.environ', {'QWEN_CUMULATIVE_NORM': '1', 'TT_METAL_HOME': 'native'}), \
                patch.object(runtime, 'qualify_prefetch', return_value=dict(report_sha256=runtime.PREFETCH_SHA256)), \
                patch.object(runtime, 'qualify_scatter', return_value=dict(report_sha256=runtime.SCATTER_SHA256)), \
                patch.object(runtime, 'prefetch_build', prefetch), patch.object(runtime, 'scatter_build', scatter):
            with runtime.normalization_scope('.'):
                runtime.require_active()
                control = full.measure_dspark_request(gdn_shared_qk=True)
                with runtime.select_norm('scatter'):
                    candidate = full.measure_dspark_request(gdn_shared_qk=True)
                after = full.measure_dspark_request(gdn_shared_qk=True)
                for request in (control, candidate, after):
                    variants.validate_route(request, 'publication')
                changed = copy.deepcopy(candidate)
                changed['gdn_norm_prefetch'] = control['gdn_norm_prefetch']
                with self.assertRaisesRegex(ValueError, 'must not be reported'):
                    runtime.validate_identity(changed)
                changed = copy.deepcopy(candidate)
                changed['gdn_shared_qk']['admission']['report_sha256'] = runtime.PREFETCH_SHA256
                with self.assertRaisesRegex(ValueError, 'must agree'):
                    runtime.validate_identity(changed)
                with runtime.select_norm('scatter'), patch.object(runtime, 'scatter_build', side_effect=RuntimeError('kernel')):
                    with self.assertRaisesRegex(RuntimeError, 'kernel'):
                        full.measure_dspark_request(gdn_shared_qk=True)
                self.assertIs(scope.build, original_build)
                self.assertIs(gate.qualify, original_gate)
        self.assertEqual(prefetch.call_count, 96)
        self.assertEqual(scatter.call_count, 48)
        self.assertEqual(routes, [runtime.PREFETCH_SHA256, runtime.SCATTER_SHA256, runtime.PREFETCH_SHA256])
        self.assertIs(full.measure_dspark_request, measure)
        self.assertIs(variants.validate_route, original_route)
        self.assertEqual(variants.REPORT_SHA256, 'native')
        with self.assertRaisesRegex(ValueError, 'must be installed'):
            runtime.require_active()

    def test_unknown_nested_and_failed_selection_restore_context(self):
        with self.assertRaises(ValueError):
            with runtime.select_norm('unknown'):
                self.fail('Invalid policy entered')
        with self.assertRaisesRegex(RuntimeError, 'request'):
            with runtime.select_norm('scatter'):
                with self.assertRaises(ValueError):
                    with runtime.select_norm('prefetch'):
                        self.fail('Nested policy entered')
                raise RuntimeError('request')
        self.assertIsNone(runtime._selected.get())
