import hashlib
from pathlib import Path
import tempfile
import unittest

from qwen_cache_read_probe import read_prefix


class CacheReadTests(unittest.TestCase):
    def test_bounded_prefix_and_repeat_preserve_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'cache.bin'
            payload = b'0123456789' * 1000
            path.write_bytes(payload)
            first = read_prefix(path, 100)
            self.assertEqual(first['bytes_read'], 100)
            self.assertTrue(first['prefix_only'])
            self.assertEqual(first['sha256'], hashlib.sha256(payload[:100]).hexdigest())
            self.assertEqual(first['sha256'], read_prefix(path, 100)['sha256'])
            self.assertEqual(path.read_bytes(), payload)

    def test_small_file_reads_to_eof(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'cache.bin'
            path.write_bytes(b'abc')
            result = read_prefix(path, 100)
            self.assertEqual(result['bytes_read'], 3)
            self.assertFalse(result['prefix_only'])
