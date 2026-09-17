from contextlib import contextmanager
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import cumulative_register_runtime as runtime
from test_cumulative_fusion_validation import fixture, BASELINE_SHA256, REGISTER_SHA256


class RegisterRuntimeTests(unittest.TestCase):
    def exercise(self, *, fail=False, missing=False, double=False):
        active = []

        @contextmanager
        def register(*args, **kwargs):
            audit = dict(report_sha256=REGISTER_SHA256, constructions=64, calls=128, restored=False)
            active.append(True)
            try:
                yield audit
            finally:
                active.pop()
                audit['restored'] = True

        def measure():
            if fail and active:
                raise RuntimeError('request failed')
            result = fixture(bool(active))
            del result['register_epilogue']
            return result

        full = SimpleNamespace(measure_dspark_request=measure)
        fusion = SimpleNamespace(REPORT_SHA256=BASELINE_SHA256)
        observed = []
        route = lambda result, arm: observed.append(fusion.REPORT_SHA256)
        variants = SimpleNamespace(validate_route=route)
        modules = dict(full_dspark_request=full, gdn_shared_qk_variants=variants, dspark_fusion_variants=fusion)
        with patch.dict('sys.modules', modules), patch.object(runtime, 'scoped_register_epilogue', register):
            try:
                with runtime.runtime_scope('.', runtime_root='native') as candidate:
                    control = full.measure_dspark_request()
                    with candidate():
                        with self.assertRaisesRegex(ValueError, 'Nested'):
                            with candidate():
                                self.fail('Nested candidate accepted')
                        if not missing:
                            result = full.measure_dspark_request()
                            self.assertNotIn('register_epilogue', result)
                        if double:
                            full.measure_dspark_request()
                    self.assertTrue(result['register_epilogue']['restored'])
                    after = full.measure_dspark_request()
                    for request in (control, result, after):
                        variants.validate_route(request, 'publication')
            finally:
                self.assertFalse(active)
                self.assertIs(full.measure_dspark_request, measure)
                self.assertIs(variants.validate_route, route)
                self.assertEqual(fusion.REPORT_SHA256, BASELINE_SHA256)
        self.assertEqual(observed, [BASELINE_SHA256, REGISTER_SHA256, BASELINE_SHA256])

    def test_control_candidate_control_identity_and_restoration(self):
        self.exercise()

    def test_failure_restores_every_binding(self):
        with self.assertRaisesRegex(RuntimeError, 'request failed'):
            self.exercise(fail=True)

    def test_missing_or_multiple_measurements_rejected(self):
        for options in (dict(missing=True), dict(double=True)):
            with self.assertRaisesRegex(ValueError, 'Exactly one'):
                self.exercise(**options)
