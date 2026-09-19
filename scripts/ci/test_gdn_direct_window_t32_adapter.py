from pathlib import Path
import tempfile
import unittest

from gdn_direct_window_t32_adapter import payloads
from gdn_direct_window_t32_stage import stage


class T32DirectWindowAdapterTests(unittest.TestCase):
    def test_staged_probe_retains_native_control_and_complete_matrix(self):
        directory = Path(__file__).parent
        with tempfile.TemporaryDirectory() as temporary:
            scripts = Path(temporary) / 'scripts/ci'
            scripts.mkdir(parents=True)
            for name in ('gdn_conv_windows.py', 'gdn_conv_windows.cpp', 'attention_batch.py',
                    'gdn_multitoken_conv.py'):
                (scripts / name).write_bytes((directory / name).read_bytes())
            (scripts / 'simulator-suite.sh').write_text('frozen_sim_phase.py --phase probe --seconds 510 --output report')
            report = stage(temporary)
            self.assertEqual(report['rows'], 32)
            self.assertFalse(report['hardware_qualified'])
            probe = (scripts / 'gdn-output-grid-probe.py').read_text()
            self.assertIn('batch=32,', probe)
            self.assertIn('rows=32, hardware_qualified=False', probe)
            self.assertIn("len(report['checks']) != 56", probe)
            self.assertIn("len(report['immutable_checks']) != 88", probe)
            self.assertIn("float('nan')", probe)
            self.assertIn('from gdn_direct_window_t32_device import', probe)
            for name in self.originals:
                self.assertEqual((scripts / name).read_text(), self.originals[name])
            with self.assertRaises(ValueError):
                stage(temporary)

    def setUp(self):
        directory = Path(__file__).parent
        self.originals = {name: (directory / name).read_text() for name in
            ('gdn_direct_window.py', 'gdn_direct_window_device.py')}
        self.sources = payloads(self.originals)

    def test_all_causal_positions_and_full_tile_faces(self):
        namespace = {}
        exec(self.sources['gdn_direct_window_t32.py'], namespace)
        for token in range(32):
            for slot in range(4):
                position = token + slot - 3
                expected = ('history', position + 4) if position < 0 else ('projected', position)
                self.assertEqual(namespace['causal_source'](token, slot), expected)
        for token in (-1, 32):
            with self.assertRaises(ValueError):
                namespace['causal_source'](token, 0)
        original = {}
        exec(self.originals['gdn_direct_window.py'], original)
        native = 'before\n' + original['START'] + 'old window\n' + original['END'] + 'after\n'
        generated = namespace['reader'](native)
        self.assertIn('B == 32', generated)
        self.assertIn('token < 32;', generated)
        for row in ('source_row', 'token'):
            self.assertIn(f'({row} / 16) * 1024 + ({row} % 16) * 32 + face * 512', generated)
        offsets = [1024 * (row // 16) + 32 * (row % 16) + 512 * face + word * 4
            for row in range(32) for face in range(2) for word in range(8)]
        self.assertEqual(sorted(offsets), list(range(0, 2048, 4)))
        self.assertTrue(generated.startswith('before\n'))
        self.assertTrue(generated.endswith(original['END'] + 'after\n'))

    def test_descriptor_widens_rows_without_changing_math(self):
        device = self.sources['gdn_direct_window_t32_device.py']
        self.assertIn('reader=[4, 160, 1, 32, 1, 258', device)
        self.assertNotIn('(1, 16,', device)
        for line in self.originals['gdn_direct_window_device.py'].splitlines():
            if ('compute=' in line or 'fp32_dest_acc_en=' in line
                    or line.startswith(('IO_PAGES =', 'FP32_PAGES ='))):
                self.assertIn(line, device)
        with self.assertRaises(ValueError):
            payloads({**self.originals, 'gdn_direct_window.py': self.sources['gdn_direct_window_t32.py']})


if __name__ == '__main__':
    unittest.main()
