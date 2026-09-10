"""Scoped candidate selection must never leak into following controls."""

from pathlib import Path
import unittest

import gdn_norm_scatter_scope as scope


class ScopeTests(unittest.TestCase):
    def test_restore_on_success_and_exception(self):
        original = scope.scatter.batch.load_kernels
        for fail in (False, True):
            try:
                with scope.scoped_reader() as audit:
                    self.assertIsNot(scope.scatter.batch.load_kernels, original)
                    if fail:
                        raise RuntimeError('injected request failure')
            except RuntimeError:
                self.assertTrue(fail)
            self.assertIs(scope.scatter.batch.load_kernels, original)
            self.assertTrue(audit['restored'])

    def test_nested_override_rejected(self):
        with scope.scoped_reader():
            with self.assertRaises(ValueError):
                with scope.scoped_reader():
                    self.fail('Nested override accepted')

    def test_candidate_differs_only_in_reader(self):
        root = Path(__file__).resolve().parents[2] / 'hardware-evidence.local/34009341359/qwen-hardware-inventory-34009341359/gdn-source'
        if not root.exists():
            self.skipTest('Pinned native source unavailable')
        control = scope.scatter.batch.load_kernels(root)
        with scope.scoped_reader() as audit:
            candidate = scope.scatter.batch.load_kernels(root)
        self.assertEqual(control['recurrence'], candidate['recurrence'])
        for role in ('compute', 'writer'):
            self.assertEqual(control['norm_gate'][role], candidate['norm_gate'][role])
        self.assertNotEqual(control['norm_gate']['reader'], candidate['norm_gate']['reader'])
        self.assertEqual(len(audit['loads']), 1)
        self.assertTrue(audit['restored'])


if __name__ == '__main__':
    unittest.main()
