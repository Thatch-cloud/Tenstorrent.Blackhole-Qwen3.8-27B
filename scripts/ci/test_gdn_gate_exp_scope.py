import hashlib
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import gdn_gate_exp_scope as scope
from gdn_gate_exp_fusion import transform
from test_gdn_gate_exp_fusion import fixture


class GateExpScopeTests(unittest.TestCase):
    def test_qualified_compute_only_scope_restores_after_success_or_failure(self):
        kernel = dict(scope.KERNEL, control_sha256=hashlib.sha256(fixture().encode()).hexdigest(),
                      candidate_sha256=hashlib.sha256(transform(fixture()).encode()).hexdigest())
        for failed in (False, True):
            native = {'recurrence': dict(compute=fixture(), reader='reader', writer='writer'),
                      'norm_gate': dict(compute='norm')}
            original = lambda root: native
            pipeline = SimpleNamespace(load_kernels=original)
            admission = dict(report_sha256=scope.REPORT_SHA256, kernel=kernel)
            with patch.dict('sys.modules', {'gdn_shared_qk_pipeline': pipeline}), \
                    patch.object(scope, 'KERNEL', kernel):
                def construct():
                    with scope.scoped_gate_exp(admission) as audit:
                        with self.assertRaises(ValueError):
                            with scope.scoped_gate_exp(admission):
                                pass
                        changed = pipeline.load_kernels('root')
                        self.assertEqual(changed['norm_gate'], native['norm_gate'])
                        self.assertEqual(changed['recurrence']['reader'], 'reader')
                        self.assertEqual(changed['recurrence']['writer'], 'writer')
                        self.assertEqual(audit['kernels'], [kernel])
                        if failed:
                            raise RuntimeError('construction failed')
                    self.assertTrue(audit['restored'])
                if failed:
                    with self.assertRaises(RuntimeError):
                        construct()
                else:
                    construct()
                self.assertIs(pipeline.load_kernels, original)
                self.assertEqual(native['recurrence']['compute'], fixture())

    def test_unqualified_admission_and_unreviewed_compute_rejected(self):
        original = lambda root: {'recurrence': dict(compute=fixture())}
        pipeline = SimpleNamespace(load_kernels=original)
        with patch.dict('sys.modules', {'gdn_shared_qk_pipeline': pipeline}):
            with self.assertRaises(ValueError):
                with scope.scoped_gate_exp({}):
                    pass
            with self.assertRaises(ValueError):
                with scope.scoped_gate_exp(dict(report_sha256=scope.REPORT_SHA256,
                                                kernel=scope.KERNEL)):
                    pipeline.load_kernels('root')
            self.assertIs(pipeline.load_kernels, original)
