from contextlib import contextmanager
import unittest
from unittest.mock import patch

import cumulative_t16_scope as scope


class CumulativeScopeTests(unittest.TestCase):
    def exercise(self, fail_entry=False, fail_body=False):
        active, audits = [], {}

        @contextmanager
        def direct(admission, directory):
            self.assertEqual(admission, 'direct-admission')
            audit = dict(hits=96, restored=False)
            audits['direct'] = audit
            active.append('direct')
            try:
                yield audit
            finally:
                self.assertEqual(active.pop(), 'direct')
                audit['restored'] = True

        @contextmanager
        def compact(admission, directory):
            self.assertEqual(admission, 'compact-admission')
            self.assertEqual(active, ['direct'])
            if fail_entry:
                raise RuntimeError('compact entry failed')
            audit = dict(calls=2, steps=30, restored=False)
            audits['compact'] = audit
            active.append('compact')
            try:
                yield audit
            finally:
                self.assertEqual(active.pop(), 'compact')
                audit['restored'] = True

        with patch.object(scope, 'scoped_direct_windows', direct), \
                patch.object(scope, 'scoped_compact_scores', compact):
            try:
                with scope.scoped_cumulative_t16('direct-admission', 'compact-admission', '.') as audit:
                    self.assertEqual(active, ['direct', 'compact'])
                    if fail_body:
                        raise RuntimeError('request failed')
            finally:
                self.assertEqual(active, [])
                self.assertTrue(all(value['restored'] for value in audits.values()))
        return audit

    def test_both_components_active_and_restored(self):
        audit = self.exercise()
        request = dict(gdn_shared_qk=dict(loads=[{}] * 96), score_layout=dict(calls=2))
        scope.validate_request(request, audit)
        for component, field in (('direct', 'hits'), ('compact', 'calls'), ('compact', 'steps')):
            changed = {name: dict(value) for name, value in audit.items()}
            changed[component][field] = 0
            with self.assertRaises(ValueError):
                scope.validate_request(request, changed)

    def test_entry_failure_unwinds_target_route(self):
        with self.assertRaisesRegex(RuntimeError, 'compact entry failed'):
            self.exercise(fail_entry=True)

    def test_request_failure_unwinds_both_routes(self):
        with self.assertRaisesRegex(RuntimeError, 'request failed'):
            self.exercise(fail_body=True)
