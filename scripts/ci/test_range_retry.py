import http.client
import unittest
from unittest.mock import patch

from draft_projection_fixture import read_range


class RangeRetryTests(unittest.TestCase):
    def test_transient_read_retries_identical_range(self):
        with patch('draft_projection_fixture._read_range_once', side_effect=[TimeoutError(), (b'ab', 10)]) as request, \
                patch('draft_projection_fixture.time.sleep') as sleep:
            self.assertEqual(read_range('url', 3, 2), (b'ab', 10))
            self.assertEqual(request.call_count, 2)
            self.assertTrue(all(call.args == ('url', 3, 2) for call in request.call_args_list))
            sleep.assert_called_once_with(1)

    def test_retry_budget_is_bounded(self):
        with patch('draft_projection_fixture._read_range_once', side_effect=http.client.IncompleteRead(b'a', 2)) as request, \
                patch('draft_projection_fixture.time.sleep') as sleep:
            with self.assertRaises(http.client.IncompleteRead):
                read_range('url', 3, 2)
            self.assertEqual(request.call_count, 3)
            self.assertEqual(sleep.call_count, 2)

    def test_invalid_range_response_is_not_retried(self):
        with patch('draft_projection_fixture._read_range_once', side_effect=ValueError('wrong range')) as request, \
                patch('draft_projection_fixture.time.sleep') as sleep:
            with self.assertRaises(ValueError):
                read_range('url', 3, 2)
            request.assert_called_once()
            sleep.assert_not_called()
