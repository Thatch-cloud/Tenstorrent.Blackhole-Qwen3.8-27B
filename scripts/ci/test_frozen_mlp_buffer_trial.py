import subprocess
import unittest

from frozen_mlp_buffer_trial import manifest, transform
from frozen_recipe_context import REVISION


class MlpBufferTrialTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = subprocess.check_output(['git', 'show',
            f'{REVISION}:scripts/ci/fused_1d.py'], text=True)

    def test_only_fifo_sizes_and_manifest_change(self):
        candidate = transform(self.source)
        self.assertEqual(candidate.split('def fused_compute', 1)[1].split('class FusedProjection', 1)[0],
            self.source.split('def fused_compute', 1)[1].split('class FusedProjection', 1)[0])
        self.assertEqual(candidate.split('        mesh_program =', 1)[1],
            self.source.split('        mesh_program =', 1)[1])
        evidence = manifest(self.source)
        self.assertEqual(evidence['extra_input_bytes_per_core'], 32768)
        self.assertEqual(evidence['extra_weight_bytes_per_worker'], 55296)
        self.assertFalse(evidence['simulator_qualified'])
        self.assertFalse(evidence['performance_qualified'])

    def test_unknown_or_already_changed_source_rejected(self):
        for source in ('', transform(self.source)):
            with self.assertRaises(ValueError):
                transform(source)


if __name__ == '__main__':
    unittest.main()
