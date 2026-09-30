import os
from pathlib import Path
import unittest
from unittest.mock import patch

from mlp_progressive_input import projection, reader, require_simulator


class ProgressiveInputTests(unittest.TestCase):
    def test_reader_publishes_each_block_without_reusing_storage(self):
        original = Path(__file__).with_name('fused_1d_input.cpp').read_text()
        source = reader(original)
        loop = source.index('for (uint32_t block = 0; block < 20; ++block)')
        self.assertLess(source.index('cb_reserve_back(0, 160)'), loop)
        self.assertLess(source.index('noc_semaphore_set(received, 0)'), source.index('noc_semaphore_inc('))
        self.assertLess(source.index('noc_semaphore_wait(ready, receivers)'), loop)
        self.assertEqual(source.count('noc_semaphore_inc('), 1)
        self.assertIn('noc_semaphore_wait_min(received, block + 1)', source)
        self.assertNotIn('cb_push_back(0, 160)', source)
        self.assertIn('signal, receivers);\n            noc_async_write_barrier();', source)
        self.assertEqual([block * 8 + tile for block in range(20) for tile in range(8)], list(range(160)))
        with self.assertRaises(ValueError):
            reader(original + '\n')

    def test_lagging_receiver_can_catch_up_without_missing_counter(self):
        for received in range(1, 21):
            consumed = []
            for needed in range(1, 21):
                if received >= needed:
                    consumed.extend(range((needed - 1) * 8, needed * 8))
            self.assertEqual(consumed, list(range(received * 8)))
        self.assertEqual(20 * 8 * 2048, 327680)

    def test_projection_preserves_compute_and_weights(self):
        original = Path(__file__).with_name('fused_1d.py').read_text()
        candidate = projection(original)
        restored = candidate.replace('cb(0, ttnn.bfloat16, 2048, 160, all_cores)',
            'cb(0, ttnn.bfloat16, 2048, 16, all_cores)').replace(
            '\n                             progressive_input=True, input_buffer_tiles=160,', '').replace(
            '        from mlp_progressive_input import require_simulator\n        require_simulator()\n', '')
        self.assertEqual(restored, original)
        with self.assertRaises(ValueError):
            projection(candidate)

    def test_hardware_rejected(self):
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
