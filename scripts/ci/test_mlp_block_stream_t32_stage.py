import os
from pathlib import Path
import unittest
from unittest.mock import patch

from mlp_block_stream_probe import adapt_probe
from mlp_block_stream_projection import adapt_projection
from mlp_block_stream_t32_stage import payloads, require_simulator
from mlp_register_epilogue import adapt_projection as register_projection


class T32BlockStreamStageTests(unittest.TestCase):
    def test_only_width_and_simulator_routing_change(self):
        directory = Path(__file__).parent
        source = (directory / 'fused_1d.py').read_text()
        originals = {'fused_1d.py': adapt_projection(register_projection(source, nearest_away=True)),
            'fused-batch-probe.py': adapt_probe((directory / 'fused-batch-probe.py').read_text()),
            'mlp_block_stream_projection.py': (directory / 'mlp_block_stream_projection.py').read_text()}
        generated = payloads(originals)
        projection = generated['fused_1d.py']
        restored = projection.replace('token_rows != 32', 'token_rows != 16').replace(
            'T32 target-math simulator candidate', 'T16 target-math simulator candidate').replace(
            'from mlp_block_stream_t32_projection import', 'from mlp_block_stream_projection import').replace(
            '        from mlp_block_stream_t32_stage import require_simulator\n        require_simulator()\n', '')
        self.assertEqual(restored, originals['fused_1d.py'])
        helper = generated['mlp_block_stream_t32_projection.py']
        self.assertEqual(helper.replace('projection.token_rows != 32', 'projection.token_rows != 16').replace(
            'Exact T32', 'Exact T16'), originals['mlp_block_stream_projection.py'])
        self.assertIn('trace_rows = (32,)', generated['fused-batch-probe.py'])
        self.assertIn('for rows in (32,):', generated['fused-batch-probe.py'])
        self.assertIn('stream_bytes_after', generated['fused-batch-probe.py'])
        with self.assertRaises(ValueError):
            payloads({**originals, 'fused_1d.py': projection})

    def test_hardware_remains_rejected(self):
        with patch.dict(os.environ, {}, clear=True), self.assertRaises(ValueError):
            require_simulator()
        environment = dict(QWEN_SIM_ONLY='1', TT_METAL_SIMULATOR='fixture')
        with patch.dict(os.environ, environment, clear=True):
            require_simulator()
        for flag in ('QWEN_HARDWARE_TESTS', 'QWEN_CARDS_ALLOCATED'):
            with patch.dict(os.environ, {**environment, flag: '1'}, clear=True), self.assertRaises(ValueError):
                require_simulator()


if __name__ == '__main__':
    unittest.main()
