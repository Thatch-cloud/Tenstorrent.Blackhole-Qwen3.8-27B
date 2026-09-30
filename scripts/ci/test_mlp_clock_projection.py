from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from mlp_clock_projection import instrument_projection, remove_projection, validate_buffers


class ClockProjectionTests(unittest.TestCase):
    def test_original_projection_round_trip_and_persistent_bindings(self):
        original = Path(__file__).with_name('fused_1d.py').read_text()
        candidate = instrument_projection(original)
        self.assertEqual(remove_projection(candidate), original)
        self.assertEqual(candidate.count('ttnn.empty('), original.count('ttnn.empty('))
        self.assertIn('11 * self.rows, input_samples.buffer_address()]', candidate)
        self.assertIn('output_tiles, weight_samples.buffer_address()]', candidate)
        self.assertIn('[value, self.weights, output, *self.sample_buffers]', candidate)
        with self.assertRaises(ValueError):
            instrument_projection(candidate)
        with self.assertRaises(ValueError):
            instrument_projection(original.replace('11 * self.rows]', '99]'))

    def test_buffer_geometry_ownership_and_alias_validation(self):
        mesh = object()
        operations = SimpleNamespace(uint32='uint32', ROW_MAJOR_LAYOUT='row', DRAM_MEMORY_CONFIG='dram',
            get_device_tensors=lambda tensor: tensor.shards)

        def tensor(rows, address):
            return SimpleNamespace(shape=(rows, 32), dtype='uint32', layout='row',
                device=lambda: mesh, memory_config=lambda: 'dram',
                shards=[SimpleNamespace(device=lambda: SimpleNamespace(id=lambda: 0),
                    buffer_address=lambda: address) for chip in (0, 2)])

        with patch.dict('sys.modules', ttnn=operations):
            inputs, weights = tensor(2, 128), tensor(1, 512)
            validate_buffers(mesh, (inputs, weights))
            for invalid in (None, [inputs, weights], (inputs,), (inputs, tensor(1, 128)),
                            (tensor(1, 128), weights)):
                with self.assertRaises(ValueError):
                    validate_buffers(mesh, invalid)
            with self.assertRaises(ValueError):
                validate_buffers(object(), (inputs, weights))
            weights.shards.pop()
            with self.assertRaises(ValueError):
                validate_buffers(mesh, (inputs, weights))
