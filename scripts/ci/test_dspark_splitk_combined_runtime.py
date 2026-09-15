from contextlib import contextmanager
import os
import unittest
from unittest.mock import patch

import dspark_splitk_combined_runtime as candidate


class CombinedRuntimeTests(unittest.TestCase):
    def test_binding_executes_qualified_geometry_and_restores(self):
        original = candidate.dspark_native_cached_layer.attend
        kernel = dict(source_before='original', source_active='qualified')
        admission = dict(component=dict(kernel=kernel), build=dict(binaries={'library': 'digest'}))

        @contextmanager
        def kernel_scope(root):
            yield kernel

        def adapter(context):
            self.assertEqual(context, 65536)
            return lambda: candidate.dspark_splitk_attention.execute_folded('device')

        def draft_execute(*args, **kwargs):
            self.assertEqual(os.environ.get('QWEN_SPLITK_FP32_INTERMEDIATES'), '1')
            return 'result'

        with patch.object(candidate, 'require_screen'), \
                patch.object(candidate, 'validate_combined', return_value=admission), \
                patch.object(candidate, 'kernel_scope', kernel_scope), \
                patch.object(candidate, 'digest', return_value='original'), \
                patch.object(candidate.dspark_splitk_attention, 'adapter', side_effect=adapter), \
                patch.dict(os.environ, QWEN_SPLITK_FP32_INTERMEDIATES='0'), \
                patch.object(candidate.dspark_splitk_attention, 'execute_folded', side_effect=draft_execute) as execute:
            with candidate.runtime_scope('.', '.', 'build') as record:
                self.assertEqual(os.environ.get('QWEN_SPLITK_FP32_INTERMEDIATES'), '0')
                self.assertEqual(candidate.dspark_native_cached_layer.attend(), 'result')
                self.assertEqual(os.environ.get('QWEN_SPLITK_FP32_INTERMEDIATES'), '0')
            execute.assert_called_once_with('device', key_chunk_size=256, max_cores_per_head=8,
                stripe_keys=False, fp32_dest_acc=True)
            self.assertEqual(record['attention_calls'], 1)
            self.assertTrue(record['kernel_restored'])
            self.assertIs(candidate.dspark_native_cached_layer.attend, original)
            with self.assertRaisesRegex(ValueError, 'did not execute'):
                with candidate.runtime_scope('.', '.', 'build'):
                    pass
            with self.assertRaisesRegex(RuntimeError, 'request failed'):
                with candidate.runtime_scope('.', '.', 'build'):
                    raise RuntimeError('request failed')
            self.assertIs(candidate.dspark_native_cached_layer.attend, original)

    def test_override_installed_after_legacy_scope(self):
        order = []

        @contextmanager
        def legacy(directory, report_path, *, factory_root, build_path):
            order.append('legacy-enter')
            yield 'admission'
            order.append('legacy-exit')

        @contextmanager
        def runtime(directory, root, build_path):
            order.append('splitk-enter')
            yield {'active': True}
            order.append('splitk-exit')

        records = []
        with patch.object(candidate, 'require_screen'), \
                patch.object(candidate.dspark_64k_entry, 'runtime_scope', legacy), \
                patch.object(candidate, 'runtime_scope', runtime):
            with candidate.entry_scope(records):
                with candidate.dspark_64k_entry.runtime_scope('.', 'report', factory_root='root', build_path='build') as value:
                    self.assertEqual(value, 'admission')
                    order.append('request')
            self.assertIs(candidate.dspark_64k_entry.runtime_scope, legacy)
        self.assertEqual(order, ['legacy-enter', 'splitk-enter', 'request', 'splitk-exit', 'legacy-exit'])
        self.assertEqual(records, [{'active': True}])


if __name__ == '__main__':
    unittest.main()
