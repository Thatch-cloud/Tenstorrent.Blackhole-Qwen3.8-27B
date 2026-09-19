from pathlib import Path
import tempfile
import unittest

from mlp_down_grid_t32_stage import adapt_probe, stage


class T32DownGridStageTests(unittest.TestCase):
    def test_only_width_changes_in_probe(self):
        original = Path(__file__).with_name('mlp-down-grid-probe.py').read_text()
        candidate = adapt_probe(original)
        restored = candidate.replace('T32 MLP-down', 'T16 MLP-down').replace('rows=32,', 'rows=16,')
        restored = restored.replace('(2, 1, 32,', '(2, 1, 16,')
        restored = restored.replace('create_matmul_1d_decode_progcfg(32,', 'create_matmul_1d_decode_progcfg(16,')
        self.assertEqual(restored, original)
        with self.assertRaises(ValueError):
            adapt_probe(candidate)

    def test_staging_preserves_native_grid_helper(self):
        with tempfile.TemporaryDirectory() as temporary:
            scripts = Path(temporary) / 'scripts/ci'
            scripts.mkdir(parents=True)
            (scripts / 'gdn-output-grid-probe.py').write_text('original probe')
            (scripts / 'simulator-suite.sh').write_text('frozen_sim_phase.py --phase probe --seconds 510 --output report')
            report = stage(temporary)
            self.assertEqual(report['rows'], 32)
            self.assertFalse(report['hardware_qualified'])
            self.assertIn('rows=32,', (scripts / 'gdn-output-grid-probe.py').read_text())
            self.assertEqual((scripts / 'mlp_down_grid.py').read_bytes(),
                Path(__file__).with_name('mlp_down_grid.py').read_bytes())
            with self.assertRaises(ValueError):
                stage(temporary)


if __name__ == '__main__':
    unittest.main()
