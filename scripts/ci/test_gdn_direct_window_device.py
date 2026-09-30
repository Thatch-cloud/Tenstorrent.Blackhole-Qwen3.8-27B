import unittest
from unittest.mock import patch

from gdn_direct_window_device import execute, work, IO_PAGES, FP32_PAGES


class DirectDescriptorTests(unittest.TestCase):
    def test_hardware_rejected_before_allocation(self):
        with patch.dict('os.environ', {}, clear=True), self.assertRaisesRegex(ValueError, 'simulator-only'):
            execute(None, None, None, None, None, None, None, None)

    def test_native_work_partition_and_private_scratch(self):
        workers = work(11, 10)
        self.assertEqual(len(workers), 81)
        self.assertEqual(sum(worker[4] for worker in workers), 1)
        self.assertEqual(workers[-1][3:], (0, 1))
        self.assertEqual([page for _, _, start, count, _ in workers for page in range(start, start + count)], list(range(160)))
        self.assertEqual(IO_PAGES[14] * 2048, 8192)
        self.assertFalse(set(IO_PAGES) & set(FP32_PAGES))
        self.assertEqual(IO_PAGES[0], 8)
        self.assertEqual(IO_PAGES[5], 8)
