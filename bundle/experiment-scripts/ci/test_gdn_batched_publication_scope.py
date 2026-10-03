from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import gdn_batched_publication_scope as scope
from test_gdn_commit_dma import CommitDmaTests


class PublicationScopeTests(unittest.TestCase):
    def fixture(self, rows=16):
        mesh = SimpleNamespace(shape=(1, 2))
        layers = [[SimpleNamespace(shape=shape) for shape in CommitDmaTests().fixture(rows)] for unused in range(48)]
        module = SimpleNamespace(prepare=scope.native)
        return mesh, layers, module, scope.PublicationArm(mesh, [layer[10:15] for layer in layers])

    def test_qualified_scope_prepares_all_prefixes_and_restores(self):
        mesh, layers, module, arm = self.fixture()
        operation = Mock()
        with patch.object(scope, 'qualify', return_value={'simulator': 'fixture'}), patch.object(scope, 'candidate', return_value=operation) as candidate:
            with arm.install(module):
                for prefix in range(17):
                    self.assertIs(module.prepare(mesh, layers, prefix), operation)
            self.assertEqual(candidate.call_count, 17)
        self.assertIs(module.prepare, scope.native)
        self.assertFalse(arm.summary()['hardware_qualified'])

    def test_missing_simulator_proof_never_installs(self):
        mesh, layers, module, arm = self.fixture()
        with patch.object(scope, 'qualify', side_effect=ValueError('missing simulator proof')):
            with self.assertRaises(ValueError):
                with arm.install(module):
                    self.fail('Unqualified scope installed')
        self.assertIs(module.prepare, scope.native)

    def test_foreign_state_is_rejected_and_binding_restored(self):
        mesh, layers, module, arm = self.fixture()
        layers[0][10] = SimpleNamespace(shape=layers[0][10].shape)
        with patch.object(scope, 'qualify', return_value={}):
            with self.assertRaises(ValueError):
                with arm.install(module):
                    module.prepare(mesh, layers, 0)
        self.assertIs(module.prepare, scope.native)
        with self.assertRaises(ValueError):
            arm.summary()

    def test_external_hook_is_not_overwritten(self):
        mesh, layers, module, arm = self.fixture()
        foreign = Mock()
        with patch.object(scope, 'qualify', return_value={}):
            with self.assertRaises(RuntimeError):
                with arm.install(module):
                    module.prepare = foreign
        self.assertIs(module.prepare, foreign)

    def test_smaller_buckets_keep_native_preparation(self):
        mesh, layers, module, arm = self.fixture(8)
        replacement = Mock()
        module.prepare = replacement
        with patch.object(scope, 'native', replacement), patch.object(scope, 'qualify', return_value={}):
            with arm.install(module):
                module.prepare(mesh, layers, 4)
            replacement.assert_called_once_with(mesh, layers, 4)
        self.assertEqual(arm.native_fallbacks, 1)
        with self.assertRaises(ValueError):
            arm.summary()
