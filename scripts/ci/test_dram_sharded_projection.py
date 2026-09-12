import unittest

from dram_sharded_projection import geometry


class DramShardedProjectionTests(unittest.TestCase):
    def test_real_geometry_preserves_precision_and_covers_banks(self):
        for banks in (7, 8):
            for name in ('gate', 'up', 'down'):
                plan = geometry(name, banks)
                self.assertEqual(plan['dtype'], 'bfloat8_b' if name == 'down' else 'bfloat4_b')
                self.assertEqual(plan['in0_block_w'], 8)
                self.assertEqual(plan['input_shard'][1] * plan['workers'], plan['inner'])
                self.assertEqual(plan['output_shard'][1] * plan['workers'], plan['width'])
                self.assertEqual(plan['input_shard'][1] % (plan['in0_block_w'] * 32), 0)
                self.assertGreaterEqual(plan['weight_shard'][1] * banks, plan['width'])
                self.assertLess(plan['weight_shard'][1] * banks - plan['width'], banks * 32)

    def test_unqualified_geometry_rejected(self):
        for banks in (True, 0, 6, 9, 12):
            with self.assertRaises(ValueError):
                geometry('gate', banks)
        with self.assertRaises(ValueError):
            geometry('unknown', 8)
