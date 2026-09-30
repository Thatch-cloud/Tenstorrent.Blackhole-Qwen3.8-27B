from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import cumulative_register_scope as scope


class RegisterScopeTests(unittest.TestCase):
    def exercise(self, *, failed_call=False):
        original_projection, original_qualification, down_forward = object(), object(), object()
        fused = SimpleNamespace(FusedProjection=original_projection, qualify_simulator=original_qualification,
            FusedT16Arm=SimpleNamespace(forward=down_forward))
        implementation = Mock(return_value='output')
        implementation.manifest = {'kernel': 'qualified'}
        if failed_call:
            implementation.side_effect = RuntimeError('kernel failed')
        candidate = SimpleNamespace(FusedProjection=Mock(return_value=implementation))
        spec = SimpleNamespace(loader=Mock())
        evidence = {'report_sha256': scope.REPORT_SHA256}
        with patch.dict('sys.modules', {'fused_t16_scope': fused}), \
                patch.object(scope, 'qualify', return_value=evidence), \
                patch.object(scope.importlib.util, 'spec_from_file_location', return_value=spec), \
                patch.object(scope.importlib.util, 'module_from_spec', return_value=candidate):
            try:
                with scope.scoped_register_epilogue('.', runtime_root='native') as audit:
                    self.assertIs(fused.FusedT16Arm.forward, down_forward)
                    self.assertIs(fused.qualify_simulator(), evidence)
                    with self.assertRaisesRegex(ValueError, 'Nested'):
                        with scope.scoped_register_epilogue('.', runtime_root='native'):
                            self.fail('Nested scope accepted')
                    with self.assertRaisesRegex(ValueError, 'only for T16'):
                        fused.FusedProjection(token_rows=32)
                    for layer in range(64):
                        projection = fused.FusedProjection(token_rows=16)
                        self.assertEqual(projection.manifest, {'kernel': 'qualified'})
                        self.assertEqual(projection('input'), 'output')
            finally:
                self.assertIs(fused.FusedProjection, original_projection)
                self.assertIs(fused.qualify_simulator, original_qualification)
                self.assertIs(fused.FusedT16Arm.forward, down_forward)
        return audit

    def test_keeps_down_forward_and_accounts_for_all_layers(self):
        audit = self.exercise()
        request = dict(fused_t16_mlp=dict(hits=[1] * 64, passed_simulator=scope.REPORT_SHA256))
        scope.validate_request(request, audit)
        for field, value in (('calls', 63), ('constructions', 0), ('restored', False),
                             ('report_sha256', 'wrong')):
            with self.assertRaises(ValueError):
                scope.validate_request(request, dict(audit, **{field: value}))
        request['fused_t16_mlp']['hits'][17] = 0
        with self.assertRaises(ValueError):
            scope.validate_request(request, audit)

    def test_failed_kernel_restores_bindings(self):
        with self.assertRaisesRegex(RuntimeError, 'kernel failed'):
            self.exercise(failed_call=True)
