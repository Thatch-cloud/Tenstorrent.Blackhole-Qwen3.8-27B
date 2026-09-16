import subprocess
import unittest

from frozen_recipe_context import REVISION
from frozen_target_replay import adapt_target_probe


class TargetReplayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original = subprocess.check_output(['git', 'show',
            f'{REVISION}:scripts/ci/target-t16-attention-8k-probe.py'], text=True)

    def test_selected_geometry_and_runtime_kv_preserve_replay_checks(self):
        changed = adapt_target_probe(self.original)
        self.assertIn("for capacity in (selected_geometry()['capacity'],):", changed)
        self.assertIn("kv_dtype='bfloat16'", changed)
        self.assertNotIn('ttnn.bfloat8_b', changed)
        self.assertIn("'frozen_context_geometry.py'", changed)
        start = '                original_cache = '
        self.assertEqual(self.original.split(start, 1)[1], changed.split(start, 1)[1])
        self.assertIn('starts = [first, first + 17, capacity - rows, first]', changed)

    def test_source_drift_and_reapplication_rejected(self):
        with self.assertRaises(ValueError):
            adapt_target_probe(self.original.replace('for capacity in (8448,):', 'for capacity in (4352,):'))
        with self.assertRaises(ValueError):
            adapt_target_probe(adapt_target_probe(self.original))


if __name__ == '__main__':
    unittest.main()
