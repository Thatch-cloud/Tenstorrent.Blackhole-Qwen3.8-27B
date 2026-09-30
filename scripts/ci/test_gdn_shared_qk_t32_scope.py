from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import gdn_shared_qk_t32_scope as scope


class T32ScopeTests(unittest.TestCase):
    def setUp(self):
        self.operations = SimpleNamespace(empty=Mock(side_effect=lambda *args, **kwargs: object()),
            generic_op=Mock(), synchronize_device=Mock(), deallocate=Mock(),
            float32='fp32', bfloat16='bf16', ROW_MAJOR_LAYOUT='row', TILE_LAYOUT='tile',
            DRAM_MEMORY_CONFIG='dram', L1_MEMORY_CONFIG='l1')
        self.admission = dict(report_sha256=scope.REPORT_SHA256, rows=32, simulator_qualified=True)
        self.builder = Mock(return_value=[([], object()) for unused in range(3)])

    def invoke(self, rows=32):
        return scope.split.execute(object(), SimpleNamespace(shape=(1, rows, 5120)),
            object(), object(), object(), z=object(), norm_w=object(), experimental=True,
            batch_norm=True, synchronize=False, output_memory='l1')

    def test_t32_buffers_retained_and_native_binding_restored(self):
        original = scope.split.execute
        with scope.scoped_shared_qk_t32(self.operations, self.admission, builder=self.builder) as audit:
            self.assertEqual(len(self.invoke()), 3)
            self.operations.deallocate.assert_not_called()
            self.operations.synchronize_device.assert_not_called()
            shapes = [call.args[0] for call in self.operations.empty.call_args_list]
            self.assertEqual(shapes, [(32, 1, 96, 32), (32, 24, 128, 128),
                (1, 32, 3072), (1, 32, 1024), (1, 32, 1024)])
        self.assertEqual(self.operations.generic_op.call_count, 3)
        self.assertEqual(self.operations.deallocate.call_count, 2)
        self.assertTrue(audit['restored'] and audit['released'])
        self.assertIs(scope.split.execute, original)

    def test_short_tails_use_original_and_are_counted(self):
        with patch.object(scope.split, 'execute', return_value='native') as native:
            native._shared_qk_override = False
            with scope.scoped_shared_qk_t32(self.operations, self.admission, builder=self.builder) as audit:
                for rows in (1, 8, 16):
                    self.assertEqual(self.invoke(rows), 'native')
            self.assertEqual(audit['fallbacks'], 3)
            self.builder.assert_not_called()

    def test_failed_build_releases_all_buffers(self):
        original = scope.split.execute
        self.builder.side_effect = ValueError('build failed')
        with self.assertRaisesRegex(ValueError, 'build failed'):
            with scope.scoped_shared_qk_t32(self.operations, self.admission, builder=self.builder):
                self.invoke()
        self.assertEqual(self.operations.deallocate.call_count, 5)
        self.assertIs(scope.split.execute, original)

    def test_t16_admission_rejected(self):
        with self.assertRaises(ValueError):
            with scope.scoped_shared_qk_t32(self.operations, {**self.admission, 'rows': 16}, builder=self.builder):
                self.fail('T16 evidence admitted T32')


if __name__ == '__main__':
    unittest.main()
