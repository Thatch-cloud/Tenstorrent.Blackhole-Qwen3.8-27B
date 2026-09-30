import hashlib
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import gdn_dual_state_scope as scope
from gdn_dual_state_copy import transform
from test_gdn_dual_state_copy import fixture


class DualStateScopeTests(unittest.TestCase):
    def test_composes_with_norm_prefetch_and_restores_on_failure(self):
        before = fixture()
        kernel = dict(scope.KERNEL, control_sha256=hashlib.sha256(before.encode()).hexdigest(),
            candidate_sha256=hashlib.sha256(transform(before).encode()).hexdigest())
        for fail in (False, True):
            original = lambda root: {'recurrence': {'compute': before, 'reader': 'reader', 'writer': 'writer'},
                'norm_gate': {'reader': 'serial'}}
            pipeline = SimpleNamespace(load_kernels=original)
            with patch.dict('sys.modules', {'gdn_shared_qk_pipeline': pipeline}), patch.object(scope, 'KERNEL', kernel):
                try:
                    with scope.scoped_copy(dict(report_sha256=scope.REPORT_SHA256, kernel=kernel)) as audit:
                        selected = pipeline.load_kernels
                        def prefetch(root):
                            result = selected(root)
                            result['norm_gate']['reader'] = 'prefetched'
                            return result
                        with patch.object(pipeline, 'load_kernels', prefetch):
                            result = pipeline.load_kernels('runtime')
                            self.assertEqual(result['norm_gate']['reader'], 'prefetched')
                            self.assertEqual(result['recurrence']['compute'], transform(before))
                            self.assertEqual(result['recurrence']['reader'], 'reader')
                            self.assertEqual(result['recurrence']['writer'], 'writer')
                            if fail:
                                raise RuntimeError('construction failed')
                except RuntimeError:
                    self.assertTrue(fail)
            self.assertIs(pipeline.load_kernels, original)
            self.assertTrue(audit['restored'])
            self.assertEqual(audit['kernels'], [kernel])

    def test_unadmitted_scope_rejected_without_replacing_loader(self):
        original = lambda root: {}
        pipeline = SimpleNamespace(load_kernels=original)
        with patch.dict('sys.modules', {'gdn_shared_qk_pipeline': pipeline}):
            with self.assertRaises(ValueError):
                with scope.scoped_copy({}):
                    self.fail('Unadmitted scope entered')
        self.assertIs(pipeline.load_kernels, original)
