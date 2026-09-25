from pathlib import Path
import unittest
from unittest.mock import patch

from mlp_block_stream_projection import adapt_projection
from mlp_progressive_input import projection
from mlp_progressive_input_gate import hardware_reader, hardware_source, qualify, READER_SHA256
from mlp_progressive_input_report import digest
from mlp_register_epilogue import adapt_projection as register_projection


class ProgressiveGateTests(unittest.TestCase):
    def test_physical_reader_is_exact_simulator_source(self):
        source = Path(__file__).with_name('fused_1d_input.cpp').read_text()
        self.assertEqual(digest(hardware_reader(source).encode()), READER_SHA256)
        with self.assertRaises(ValueError):
            hardware_reader(source + '\n')

    def test_only_explicit_admission_and_source_delivery_change(self):
        source = register_projection(Path(__file__).with_name('fused_1d.py').read_text(), nearest_away=True)
        candidate = hardware_source(source)
        restored = candidate.replace('        require_hardware(os.environ)\n',
            '        from mlp_progressive_input import require_simulator\n        require_simulator()\n').replace(
            'kernel_source=progressive_reader(Path(__file__).with_name("fused_1d_input.cpp").read_text()),\n'
            '                source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE, core_ranges=all_cores,',
            'kernel_source=str(Path(__file__).with_name("fused_1d_input.cpp")), core_ranges=all_cores,')
        self.assertEqual(restored, projection(adapt_projection(source)))

    def test_serial_qualification_cannot_be_bypassed(self):
        with patch('mlp_progressive_input_gate.qualify_serial', side_effect=ValueError('serial rejected')):
            with self.assertRaisesRegex(ValueError, 'serial rejected'):
                qualify('.', '.', '.', {})


if __name__ == '__main__':
    unittest.main()
