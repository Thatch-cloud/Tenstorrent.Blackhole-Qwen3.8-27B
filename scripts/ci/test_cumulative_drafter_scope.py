from contextlib import contextmanager
import unittest
from unittest.mock import patch

import cumulative_t16_scope as scope


class DrafterScopeTests(unittest.TestCase):
    def exercise(self, fail=False):
        active = []

        @contextmanager
        def target(*arguments):
            audit = dict(hits=48, restored=False)
            active.append('target')
            try:
                yield audit
            finally:
                active.pop()
                audit['restored'] = True

        @contextmanager
        def down(*arguments):
            audit = dict(hits=[1] * 64, restored=False)
            active.append('down')
            try:
                yield audit
            finally:
                active.pop()
                audit['restored'] = True

        with patch.object(scope, 'scoped_direct_windows', target), \
                patch.object(scope, 'scoped_down_grid', down), \
                patch.object(scope, 'scoped_compact_scores') as compact:
            try:
                with scope.scoped_cumulative_t16({}, None, '.', down_admission={}, drafter='dflash2') as audit:
                    self.assertEqual(active, ['target', 'down'])
                    if fail:
                        raise RuntimeError('request failed')
            finally:
                self.assertEqual(active, [])
                compact.assert_not_called()
                self.assertTrue(all(component['restored'] for component in audit.values()))
        return audit

    def test_dflash_preserves_shared_target_routes_without_markov(self):
        audit = self.exercise()
        request = dict(selected_drafter='dflash2', gdn_shared_qk=dict(loads=[{}] * 48),
                       fused_t16_mlp=dict(hits=[1] * 64))
        scope.validate_request(request, audit, drafter='dflash2')
        audit['down']['hits'][0] = 0
        with self.assertRaises(ValueError):
            scope.validate_request(request, audit, drafter='dflash2')

    def test_failed_request_restores_target_routes(self):
        with self.assertRaisesRegex(RuntimeError, 'request failed'):
            self.exercise(fail=True)

    def test_invalid_drafter_admission_fails_before_patching(self):
        for drafter, compact in (('unknown', None), ('dspark', None), ('dflash2', {})):
            with self.subTest(drafter=drafter), patch.object(scope, 'scoped_direct_windows') as target:
                with self.assertRaises(ValueError):
                    with scope.scoped_cumulative_t16({}, compact, '.', drafter=drafter):
                        self.fail('Invalid admission entered')
                target.assert_not_called()

    def test_drafter_identity_and_selector_contamination_rejected(self):
        for selected, extra in (('dspark', {}), ('dflash2', {'compact': {}})):
            with self.assertRaises(ValueError):
                scope.validate_request(dict(selected_drafter=selected), extra, drafter='dflash2')
