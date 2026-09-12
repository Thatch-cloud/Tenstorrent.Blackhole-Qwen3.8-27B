from types import SimpleNamespace
import unittest

from output_projection_grid import widen


class OutputGridTests(unittest.TestCase):
    def configuration(self):
        return SimpleNamespace(compute_with_storage_grid_size=(11, 3), in0_block_w=8,
            out_subblock_h=1, out_subblock_w=1, per_core_M=1, per_core_N=5,
            fuse_batch=True, fused_activation=None, mcast_in0=True)

    def test_only_grid_and_output_partition_change(self):
        operations = SimpleNamespace(MatmulMultiCoreReuseMultiCast1DProgramConfig=SimpleNamespace)
        original = self.configuration()
        candidate = widen(operations, original)
        self.assertEqual(candidate.compute_with_storage_grid_size, (11, 6))
        self.assertEqual(candidate.per_core_N, 3)
        for name in vars(original).keys() - {'compute_with_storage_grid_size', 'per_core_N'}:
            self.assertEqual(getattr(candidate, name), getattr(original, name))
        self.assertEqual(original.compute_with_storage_grid_size, (11, 3))

    def test_unqualified_geometry_rejected(self):
        for name, value in (('per_core_M', 2), ('per_core_N', 4), ('out_subblock_w', 2),
                ('mcast_in0', False), ('fused_activation', 'silu')):
            original = self.configuration()
            setattr(original, name, value)
            with self.subTest(name=name), self.assertRaises(ValueError):
                widen(None, original)
        with self.assertRaises(ValueError):
            widen(None, self.configuration(), grid=(11, 10))
