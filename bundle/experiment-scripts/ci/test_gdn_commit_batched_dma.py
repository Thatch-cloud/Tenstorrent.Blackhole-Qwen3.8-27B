import unittest
from unittest.mock import Mock, patch

from gdn_commit_batched_dma import publish, validate_shapes
from test_gdn_commit_dma import CommitDmaTests


class BatchedCommitDmaTests(unittest.TestCase):
    def test_same_publication_geometry_and_prefix_contract(self):
        for rows in (2, 4, 8, 16, 32):
            for prefix in range(rows + 1):
                for count in (1, 2, 48):
                    layers = [CommitDmaTests().fixture(rows)] * count
                    self.assertEqual(validate_shapes(layers, prefix), rows)
        for prefix in (-1, 17, True):
            with self.assertRaises(ValueError):
                validate_shapes([CommitDmaTests().fixture(16)], prefix)

    def test_candidate_publication_executes_only_candidate_preparation(self):
        operation = Mock()
        with patch('gdn_commit_batched_dma.prepare', return_value=operation) as prepare:
            publish('mesh', 'layers', 8)
        prepare.assert_called_once_with('mesh', 'layers', 8)
        operation.assert_called_once_with()

    def test_two_worker_eight_page_batches_partition_all_tasks(self):
        for count in (384, 640):
            workers = [[batch + lane * 2 + worker
                for batch in range(0, count, 16) for lane in range(8)] for worker in range(2)]
            self.assertFalse(set(workers[0]) & set(workers[1]))
            self.assertEqual(sorted(workers[0] + workers[1]), list(range(count)))
        for prefix in range(33):
            token = max(0, prefix - 1)
            offset = ((token // 16) * 512 + token % 16 * 16) * 2
            self.assertLessEqual(offset + 512 + 32, 2048)
        self.assertEqual(7 * 4096 + 2048 + 2048, 32768)
