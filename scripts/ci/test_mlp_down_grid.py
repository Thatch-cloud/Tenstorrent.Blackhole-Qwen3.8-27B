from types import SimpleNamespace
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from mlp_down_grid import widen


class DownGridTests(unittest.TestCase):
    def original(self):
        return SimpleNamespace(compute_with_storage_grid_size=SimpleNamespace(x=11, y=3),
            in0_block_w=8, out_subblock_h=1, out_subblock_w=1, per_core_M=1, per_core_N=5,
            fuse_batch=True, fused_activation=None, mcast_in0=True)

    def test_only_output_distribution_changes(self):
        original = self.original()
        candidate = widen(SimpleNamespace(MatmulMultiCoreReuseMultiCast1DProgramConfig=SimpleNamespace), original)
        self.assertEqual(candidate.compute_with_storage_grid_size, (11, 8))
        self.assertEqual(candidate.per_core_N, 2)
        for name in vars(original).keys() - {'compute_with_storage_grid_size', 'per_core_N'}:
            self.assertEqual(getattr(candidate, name), getattr(original, name))
        self.assertEqual(original.per_core_N, 5)
        self.assertEqual(original.compute_with_storage_grid_size.y, 3)

    def test_changed_native_policy_fails_closed(self):
        for name, value in (('in0_block_w', 4), ('per_core_M', 2), ('out_subblock_w', 2),
                ('per_core_N', 4), ('mcast_in0', False), ('fused_activation', 'silu'),
                ('compute_with_storage_grid_size', SimpleNamespace(x=8, y=4))):
            original = self.original()
            setattr(original, name, value)
            with self.subTest(name=name), self.assertRaises(ValueError):
                widen(None, original)

    def test_staging_is_explicit_and_cannot_reapply(self):
        from mlp_down_grid_stage import stage

        with TemporaryDirectory() as temporary:
            scripts = Path(temporary) / 'scripts/ci'
            scripts.mkdir(parents=True)
            (scripts / 'gdn-output-grid-probe.py').write_text('original')
            result = stage(temporary)
            self.assertEqual(result['projection'], 'mlp_down')
            self.assertFalse(result['simulator_qualified'])
            source = (scripts / 'gdn-output-grid-probe.py').read_text()
            for required in ("projection='mlp_down'", 'range(3)', 'for index in (1, 2, 0)',
                    "report['negative_controls'].append", "operand='weight'", "operand='activation'",
                    "packer_l1_acc=True", "len(report['checks']) != 12"):
                self.assertIn(required, source)
            with self.assertRaises(ValueError):
                stage(temporary)


if __name__ == '__main__':
    unittest.main()
