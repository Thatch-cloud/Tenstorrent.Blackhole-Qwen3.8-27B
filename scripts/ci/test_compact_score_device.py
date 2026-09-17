from pathlib import Path
import unittest
from unittest.mock import patch

from compact_score_device import execute_local_winners, geometry


class CompactDeviceTests(unittest.TestCase):
    def test_hardware_execution_rejected_before_allocation(self):
        with patch.dict('os.environ', {}, clear=True):
            with self.assertRaisesRegex(ValueError, 'simulator-only'):
                execute_local_winners(None, None, None, None, 0, None)

    def test_full_vocabulary_tile_partitions_cover_once(self):
        for width in (64, 248320):
            for limit in (1, 64, 110):
                workers, tiles = geometry((1, 1, 15, width), (1, 1, 1, width), 14, limit)
                tasks = [task for worker in range(workers)
                         for task in range(worker * tiles // workers, (worker + 1) * tiles // workers)]
                self.assertEqual(tasks, list(range(tiles)))
                self.assertTrue(all((worker + 1) * tiles // workers > worker * tiles // workers
                                    for worker in range(workers)))

    def test_sfpu_arithmetic_unchanged_and_matching_partition_loops(self):
        directory = Path(__file__).parent
        original = (directory / 'dspark_score_layout_compute.cpp').read_text()
        compute = (directory / 'compact_score_compute.cpp').read_text()
        candidate_loop = 'for (uint32_t task = worker * tiles / workers; task < (worker + 1) * tiles / workers; ++task)'
        original_loop = 'for (uint32_t task = worker; task < tiles; task += workers)'
        self.assertEqual(compute.replace(candidate_loop, original_loop), original)
        reader = (directory / 'compact_score_io.cpp').read_text()
        self.assertIn(candidate_loop, reader)
        self.assertIn('scores[(lane / 16) * 256 + lane % 16]', reader)
        self.assertIn('record[2] = nonfinite;', reader)
        self.assertIn('noc_async_write(get_write_ptr(0), output.get_noc_addr(worker), 32);', reader)
        self.assertEqual(reader.count('noc_async_write('), 1)
        self.assertGreater(reader.index('cb_reserve_back(0, 1);', reader.index('cb_pop_front(16, 1);')),
                           reader.index('cb_pop_front(16, 1);'))
