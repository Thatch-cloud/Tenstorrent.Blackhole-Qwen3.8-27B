"""Check ownership and restoration without opening a device."""

from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import gdn_shared_qk_scope as scope


class ScopeTests(unittest.TestCase):
    def operations(self):
        return SimpleNamespace(empty=Mock(side_effect=lambda *args, **kwargs: object()),
            generic_op=Mock(), synchronize_device=Mock(), deallocate=Mock(),
            float32='fp32', bfloat16='bf16', ROW_MAJOR_LAYOUT='row', TILE_LAYOUT='tile',
            DRAM_MEMORY_CONFIG='dram', L1_MEMORY_CONFIG='l1')

    def invoke(self):
        return scope.split.execute(object(), SimpleNamespace(shape=(1, 16, 5120)),
            object(), object(), object(), z=object(), norm_w=object(), experimental=True,
            batch_norm=True, synchronize=False, output_memory='l1')

    def test_retains_only_preparation_until_scope_exit(self):
        operations = self.operations()
        original = scope.split.execute
        with patch.object(scope, 'build', return_value=[([], object()) for unused in range(3)]):
            with scope.scoped_shared_qk(operations, {}) as audit:
                output = self.invoke()
                self.assertEqual(len(output), 3)
                operations.deallocate.assert_not_called()
                operations.synchronize_device.assert_not_called()
                self.assertEqual(operations.generic_op.call_count, 3)
            self.assertEqual(operations.deallocate.call_count, 2)
            self.assertEqual(operations.synchronize_device.call_count, 1)
            self.assertTrue(audit['restored'] and audit['released'])
            self.assertIs(scope.split.execute, original)

    def test_build_failure_releases_all_allocations(self):
        operations = self.operations()
        original = scope.split.execute
        with patch.object(scope, 'build', side_effect=ValueError('build failed')):
            with self.assertRaisesRegex(ValueError, 'build failed'):
                with scope.scoped_shared_qk(operations, {}):
                    self.invoke()
        self.assertEqual(operations.deallocate.call_count, 5)
        self.assertIs(scope.split.execute, original)

    def test_other_width_uses_original(self):
        operations = self.operations()
        with patch.object(scope.split, 'execute', return_value='native') as original:
            original._shared_qk_override = False
            with scope.scoped_shared_qk(operations, {}):
                result = scope.split.execute(None, SimpleNamespace(shape=(1, 1, 5120)),
                    None, None, None, z=None, norm_w=None)
                self.assertEqual(result, 'native')
            original.assert_called_once()
        operations.empty.assert_not_called()

    def test_nested_scope_rejected(self):
        operations = self.operations()
        with scope.scoped_shared_qk(operations, {}):
            with self.assertRaisesRegex(ValueError, 'Nested'):
                with scope.scoped_shared_qk(operations, {}):
                    pass
