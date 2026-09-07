import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from fixture_integrity_probe import inspect_file


class FixtureIntegrityTests(unittest.TestCase):
    def test_stable_and_changed_reads(self):
        data = b'0123456789'
        expected = hashlib.sha256(data).hexdigest()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'fixture'
            path.write_bytes(data)
            self.assertTrue(inspect_file(path, expected)['passed'])
            with patch.object(Path, 'read_bytes', return_value=b'012X456789'):
                result = inspect_file(path, expected)
            self.assertFalse(result['passed'])
            self.assertEqual(result['streamed'], expected)
            self.assertEqual(result['whole'], result['rehashed'])
            self.assertEqual(result['differences'], [dict(offset=3, whole_byte=88, stream_byte=51)])
            with patch.object(Path, 'read_bytes', side_effect=[data, b'012X456789']):
                result = inspect_file(path, expected)
            self.assertFalse(result['passed'])
            self.assertEqual(result['whole'], expected)
            self.assertEqual(result['differences'], [])
            self.assertEqual(result['second_differences'], [dict(offset=3, whole_byte=88, stream_byte=51)])


if __name__ == '__main__':
    unittest.main()
