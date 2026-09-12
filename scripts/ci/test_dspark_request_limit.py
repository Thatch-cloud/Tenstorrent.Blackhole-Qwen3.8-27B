import unittest

from dspark_request_limit import request_limit


class RequestLimitTests(unittest.TestCase):
    def test_defaults(self):
        self.assertEqual(request_limit(), 257)
        self.assertEqual(request_limit(short_default=True), 256)

    def test_matched_limit(self):
        self.assertEqual(request_limit(65, short_default=True), 65)

    def test_reject_invalid(self):
        for value in (True, False, 0, 1, 258, 65.0, '65'):
            with self.subTest(value=value), self.assertRaises(ValueError):
                request_limit(value)
