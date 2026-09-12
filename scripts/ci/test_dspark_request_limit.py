import unittest

from dspark_request_limit import request_limit


class RequestLimitTests(unittest.TestCase):
    def test_existing_budgets_are_preserved(self):
        self.assertEqual(request_limit(), 257)
        self.assertEqual(request_limit(short_default=True), 256)
        self.assertEqual(request_limit(t32=True), 65)

    def test_matched_budget(self):
        self.assertEqual(request_limit(65, short_default=True), 65)
        self.assertEqual(request_limit(65, t32=True), 65)

    def test_invalid_budgets(self):
        for value in (True, False, 0, 1, 258, 65.0, '65'):
            with self.subTest(value=value), self.assertRaises(ValueError):
                request_limit(value)
        with self.assertRaises(ValueError):
            request_limit(257, t32=True)
