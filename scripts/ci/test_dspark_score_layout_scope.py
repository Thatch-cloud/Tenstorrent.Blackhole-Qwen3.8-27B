from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import dspark_score_layout_scope as scope


class ScoreScopeTests(unittest.TestCase):
    def setUp(self):
        self.device = SimpleNamespace(closed=False, max_drafts=15, mesh=SimpleNamespace(shape=[1, 2]),
            operations=object(), predecessor=object(), successor=object())
        self.module = SimpleNamespace(markov=scope.native)
        self.arguments = (self.device.operations, object(), SimpleNamespace(shape=(1, 1, 15, 248320)),
            self.device.predecessor, self.device.successor, [])
        for name, value in (('validate_hardware', 'test-digest'), ('addresses', [1, 2])):
            patcher = patch.object(scope, name, return_value=value)
            patcher.start()
            self.addCleanup(patcher.stop)

    def test_missing_gate_never_installs(self):
        arm = scope.ScoreLayoutArm(self.device, hardware_audit={'weight_bindings': [[1, 2], [1, 2]]})
        with patch.object(scope, 'qualify', side_effect=ValueError('missing')), self.assertRaises(ValueError):
            with arm.install(self.module):
                self.fail('Gate bypassed')
        self.assertIs(self.module.markov, scope.native)

    def test_routes_only_owned_request_and_restores(self):
        arm = scope.ScoreLayoutArm(self.device, hardware_audit={'weight_bindings': [[1, 2], [1, 2]]})
        result = object()
        with patch.object(scope, 'qualify', return_value={'passed': True}), patch.object(
                scope, 'candidate', return_value=result) as candidate:
            with arm.install(self.module):
                self.assertIs(self.module.markov(*self.arguments), result)
                self.assertEqual(candidate.call_count, 1)
                self.assertIs(candidate.call_args.args[1], self.device.mesh)
                with self.assertRaises(ValueError):
                    with scope.ScoreLayoutArm(self.device, hardware_audit={'weight_bindings': [[1, 2], [1, 2]]}).install(self.module):
                        self.fail('Nested hook accepted')
        self.assertIs(self.module.markov, scope.native)
        self.assertEqual(arm.summary()['calls'], 1)

    def test_exception_restores_without_qualifying(self):
        arm = scope.ScoreLayoutArm(self.device, hardware_audit={'weight_bindings': [[1, 2], [1, 2]]})
        with patch.object(scope, 'qualify', return_value={}), patch.object(scope, 'candidate', side_effect=RuntimeError):
            with self.assertRaises(RuntimeError):
                with arm.install(self.module):
                    self.module.markov(*self.arguments)
        self.assertIs(self.module.markov, scope.native)
        with self.assertRaises(ValueError):
            arm.summary()

    def test_foreign_owner_rejected_before_candidate(self):
        arm = scope.ScoreLayoutArm(self.device, hardware_audit={'weight_bindings': [[1, 2], [1, 2]]})
        with patch.object(scope, 'qualify', return_value={}), patch.object(scope, 'candidate') as candidate:
            with self.assertRaises(ValueError):
                with arm.install(self.module):
                    self.module.markov(object(), *self.arguments[1:])
            candidate.assert_not_called()
        self.assertIs(self.module.markov, scope.native)

    def test_external_hook_not_overwritten(self):
        arm = scope.ScoreLayoutArm(self.device, hardware_audit={'weight_bindings': [[1, 2], [1, 2]]})
        foreign = Mock()
        with patch.object(scope, 'qualify', return_value={}), self.assertRaises(RuntimeError):
            with arm.install(self.module):
                self.module.markov = foreign
        self.assertIs(self.module.markov, foreign)
        self.assertTrue(arm.failed)
        self.assertFalse(arm.restored)


if __name__ == '__main__':
    unittest.main()
