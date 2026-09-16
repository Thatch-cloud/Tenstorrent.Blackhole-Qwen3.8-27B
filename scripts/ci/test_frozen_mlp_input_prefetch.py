import subprocess
import unittest

from frozen_recipe_context import REVISION
from frozen_mlp_input_prefetch import reader, projection, memory_budget


class InputPrefetchTests(unittest.TestCase):
    def source(self, name):
        return subprocess.check_output(['git', 'show', f'{REVISION}:scripts/ci/{name}'], text=True)

    def test_input_pages_and_multicast_chunks_cover_identical_payload(self):
        old_pages = [block * 8 + tile for block in range(20) for tile in range(8)]
        self.assertEqual(old_pages, list(range(160)))
        chunks = [(block * 8 * 2048, (block + 1) * 8 * 2048) for block in range(20)]
        self.assertEqual(chunks[0][0], 0)
        self.assertEqual(chunks[-1][1], 160 * 2048)
        self.assertTrue(all(end == following for (_, end), (following, _) in zip(chunks, chunks[1:])))
        self.assertTrue(all(end - start == 16384 for start, end in chunks))

    def test_reader_preserves_multicast_order_and_packet_size(self):
        candidate = reader(self.source('fused_1d_input.cpp'))
        self.assertNotIn('block * 8 + tile', candidate)
        self.assertIn('tile < 160', candidate)
        for operation in ('cb_reserve_back', 'cb_push_back', 'cb_wait_front', 'cb_pop_front'):
            self.assertIn(f'{operation}(0, 160);', candidate)
        self.assertIn('noc_async_write_multicast(chunk, target, 8 * 2048, receivers)', candidate)
        self.assertLess(candidate.index('noc_async_write_barrier();'), candidate.index('noc_semaphore_set(received, 1);'))
        with self.assertRaises(ValueError):
            reader(candidate)

    def test_projection_changes_only_input_capacity_and_metadata(self):
        source = self.source('fused_1d.py')
        candidate = projection(source)
        restored = candidate.replace('cb(0, ttnn.bfloat16, 2048, 160, all_cores)',
            'cb(0, ttnn.bfloat16, 2048, 16, all_cores)').replace(
            '\n                             full_k_input_prefetch=True, input_buffer_tiles=160,', '')
        self.assertEqual(restored, source)
        self.assertEqual(memory_budget()['worker_cb_bytes'], 425984)
        self.assertEqual(memory_budget()['extra_input_bytes_per_core'], 294912)
        with self.assertRaises(ValueError):
            projection(candidate)


if __name__ == '__main__':
    unittest.main()
