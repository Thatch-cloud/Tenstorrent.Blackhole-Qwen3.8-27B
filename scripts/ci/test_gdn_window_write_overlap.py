from pathlib import Path
import unittest

from gdn_window_write_overlap import schedule, transform_builder, transform_reader


class WindowWriteOverlapTests(unittest.TestCase):
    def test_each_window_has_disjoint_storage_until_page_barrier(self):
        outstanding = set()
        for page, operation, slot in schedule(4):
            if operation == 'issue':
                address = (5 + slot) * 2048
                self.assertNotIn(address, outstanding)
                self.assertGreaterEqual(address, 5 * 2048)
                self.assertLessEqual(address + 2048, 9 * 2048)
                outstanding.add(address)
            else:
                self.assertEqual(len(outstanding), 4)
                outstanding.clear()
        self.assertFalse(outstanding)

    def test_only_scratch_ownership_and_write_barrier_change(self):
        directory = Path(__file__).parent
        original = (directory / 'gdn_conv_windows.cpp').read_text()
        candidate = transform_reader(original)
        self.assertEqual(candidate.count('noc_async_write_barrier();'), 1)
        restored = candidate.replace('source_tiles + (5 + slot) * 2048', 'source_tiles + 5 * 2048')
        restored = restored.replace('\n        noc_async_write_barrier();', '')
        restored = restored.replace('noc_async_write_tile(page, destination, scratch);',
            'noc_async_write_tile(page, destination, scratch);\n        noc_async_write_barrier();')
        self.assertEqual(restored, original)
        with self.assertRaises(ValueError):
            transform_reader(candidate)
        builder = (directory / 'gdn_conv_windows.py').read_text()
        changed = transform_builder(builder)
        self.assertEqual(changed.replace('total_size=9 * 2048,', 'total_size=6 * 2048,'), builder)
        compile(changed, 'gdn_conv_windows.py', 'exec')
