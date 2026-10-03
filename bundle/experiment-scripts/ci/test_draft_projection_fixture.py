from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from draft_projection_fixture import read_range


class DraftProjectionFixtureTests(unittest.TestCase):
    def test_only_exact_partial_responses_are_accepted(self):
        response = SimpleNamespace(status=206, headers={'Content-Range': 'bytes 8-11/99'}, read=Mock(return_value=b'abcd'))
        manager = Mock(__enter__=Mock(return_value=response), __exit__=Mock(return_value=False))
        with patch('urllib.request.urlopen', return_value=manager):
            self.assertEqual(read_range('https://example.test/weights', 8, 4), (b'abcd', 99))
        response.read.assert_called_once_with(5)

    def test_full_wrong_or_oversized_responses_fail(self):
        for status, content_range, data in ((200, 'bytes 8-11/99', b'abcd'),
                (206, 'bytes 0-3/99', b'abcd'), (206, 'bytes 8-11/11', b'abcd'),
                (206, 'bytes 8-11/99', b'abcde'), (206, '', b'abcd')):
            response = SimpleNamespace(status=status, headers={'Content-Range': content_range}, read=Mock(return_value=data))
            manager = Mock(__enter__=Mock(return_value=response), __exit__=Mock(return_value=False))
            with patch('urllib.request.urlopen', return_value=manager), self.assertRaises(ValueError):
                read_range('https://example.test/weights', 8, 4)
        for start, length in ((-1, 8), (0, 0), (0, 2097153), (True, 8)):
            with self.assertRaises(ValueError):
                read_range('https://example.test/weights', start, length)
