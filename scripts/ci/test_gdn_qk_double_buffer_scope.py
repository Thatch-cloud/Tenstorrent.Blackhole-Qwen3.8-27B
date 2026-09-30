import hashlib
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import gdn_qk_double_buffer_scope as scope
from gdn_qk_double_buffer import reader
from gdn_qk_double_buffer import ORIGINAL_ZERO
from gdn_vsplit import cb_plan


class DoubleBufferScopeTests(unittest.TestCase):
    def test_reader_only_composition_with_norm_prefetch_and_failure_restoration(self):
        before = 'before\n' + ORIGINAL_ZERO + 'after\n'
        kernel = dict(scope.KERNEL, control_sha256=hashlib.sha256(before.encode()).hexdigest(),
            candidate_sha256=hashlib.sha256(reader(before).encode()).hexdigest())
        for failed in (False, True):
            original = lambda root: dict(recurrence=dict(reader=before, compute='compute', writer='writer'),
                norm_gate=dict(reader='serial'))
            pipeline = SimpleNamespace(load_kernels=original, cb_plan=cb_plan)
            with patch.dict('sys.modules', {'gdn_shared_qk_pipeline': pipeline}), patch.object(scope, 'KERNEL', kernel):
                try:
                    with scope.scoped_double_buffer(dict(report_sha256=scope.REPORT_SHA256, kernel=kernel)) as audit:
                        selected = pipeline.load_kernels
                        def prefetch(root):
                            result = selected(root)
                            result['norm_gate']['reader'] = 'prefetched'
                            return result
                        with patch.object(pipeline, 'load_kernels', prefetch):
                            result = pipeline.load_kernels('runtime')
                            self.assertEqual(pipeline.cb_plan('recurrence')[1][10], 8)
                            self.assertEqual(result['norm_gate']['reader'], 'prefetched')
                            self.assertEqual(result['recurrence'], dict(reader=reader(before), compute='compute', writer='writer'))
                            if failed:
                                raise RuntimeError('construction failed')
                except RuntimeError:
                    self.assertTrue(failed)
            self.assertIs(pipeline.load_kernels, original)
            self.assertIs(pipeline.cb_plan, cb_plan)
            self.assertTrue(audit['restored'])
            self.assertEqual(audit['kernels'], [kernel])

    def test_unadmitted_scope_never_installs(self):
        original = lambda root: {}
        pipeline = SimpleNamespace(load_kernels=original, cb_plan=cb_plan)
        with patch.dict('sys.modules', {'gdn_shared_qk_pipeline': pipeline}):
            with self.assertRaises(ValueError):
                with scope.scoped_double_buffer({}):
                    self.fail('Unadmitted scope entered')
        self.assertIs(pipeline.load_kernels, original)

    def test_reader_without_buffer_construction_is_rejected_and_restored(self):
        before = ORIGINAL_ZERO
        kernel = dict(scope.KERNEL, control_sha256=hashlib.sha256(before.encode()).hexdigest(),
            candidate_sha256=hashlib.sha256(reader(before).encode()).hexdigest())
        original = lambda root: dict(recurrence=dict(reader=before))
        pipeline = SimpleNamespace(load_kernels=original, cb_plan=cb_plan)
        with patch.dict('sys.modules', {'gdn_shared_qk_pipeline': pipeline}), patch.object(scope, 'KERNEL', kernel):
            with self.assertRaisesRegex(ValueError, 'two-slot rings'):
                with scope.scoped_double_buffer(dict(report_sha256=scope.REPORT_SHA256, kernel=kernel)):
                    pipeline.load_kernels('runtime')
        self.assertIs(pipeline.load_kernels, original)
        self.assertIs(pipeline.cb_plan, cb_plan)
