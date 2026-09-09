from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock

from dspark_history_projection import geometry, project
from test_dspark_projection import operations, tensor


class HistoryProjectionTests(unittest.TestCase):
    def test_full_4k_rows_are_partitioned_across_eight_worker_rows(self):
        configuration = geometry(4096)
        self.assertEqual(configuration['compute_with_storage_grid_size'],(8,8))
        self.assertEqual(configuration['per_core_M'],16)
        self.assertEqual(configuration['per_core_N'],2)
        self.assertEqual(configuration['in0_block_w'],4)
        self.assertEqual(configuration['out_subblock_h']*configuration['out_subblock_w'],2)

    def test_supported_row_counts_cover_the_entire_projection(self):
        for rows in range(32,4097,32):
            configuration = geometry(rows)
            self.assertGreaterEqual(configuration['per_core_M']*8*32,rows)
            self.assertLess(configuration['per_core_M']*8*32-rows,256)
        for rows in (True,0,31,33,4097,8192):
            with self.assertRaises(ValueError):
                geometry(rows)

    def test_full_learned_kv_shape_precision_and_ownership(self):
        runtime = operations()
        runtime.MatmulMultiCoreReuseMultiCastProgramConfig = MagicMock()
        mesh = SimpleNamespace(compute_with_storage_grid_size=lambda:SimpleNamespace(x=11,y=10))
        context,weight = tensor((1,1,4096,5120)),tensor((1,1,5120,512))
        partial,projected = tensor((1,1,4096,512),'fp32'),tensor((1,1,4096,512))
        runtime.matmul.return_value = partial
        runtime.typecast.return_value = projected
        owned = []
        result = project(runtime,mesh,context,weight,lambda value:owned.append(value) or value)
        self.assertEqual(result,dict(partial=partial,projected=projected))
        self.assertEqual(owned,[partial,projected])
        runtime.MatmulMultiCoreReuseMultiCastProgramConfig.assert_called_once_with(**geometry(4096))
        self.assertEqual(runtime.matmul.call_args.args,(context,weight))
        self.assertEqual(runtime.matmul.call_args.kwargs['dtype'],'fp32')
        runtime.typecast.assert_called_once_with(partial,'bf16')

    def test_wrong_shapes_dtypes_and_worker_capacity_are_rejected(self):
        runtime = operations()
        mesh = SimpleNamespace(compute_with_storage_grid_size=lambda:SimpleNamespace(x=8,y=8))
        for context,weight in ((tensor((1,1,4096,2560)),tensor((1,1,5120,512))),
                (tensor((1,1,4096,5120)),tensor((1,1,2560,512))),
                (tensor((1,1,4096,5120)),tensor((1,1,5120,1024))),
                (tensor((1,1,4096,5120),'fp32'),tensor((1,1,5120,512)))):
            with self.assertRaises(ValueError):
                project(runtime,mesh,context,weight,lambda value:value)
        small = SimpleNamespace(compute_with_storage_grid_size=lambda:SimpleNamespace(x=8,y=7))
        with self.assertRaises(ValueError):
            project(runtime,small,tensor((1,1,4096,5120)),tensor((1,1,5120,512)),lambda value:value)
        runtime.matmul.assert_not_called()


if __name__ == '__main__':
    unittest.main()
