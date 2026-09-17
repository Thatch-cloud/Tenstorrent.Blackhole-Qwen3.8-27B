import unittest
from unittest.mock import patch

from dspark_confidence_fixture import TENSORS, decode, fetch
from dspark_intake import CHECKPOINT_BYTES


class ConfidenceFixtureTests(unittest.TestCase):
    def test_corrupt_tensor_bytes_rejected(self):
        for name in TENSORS:
            with self.subTest(name=name), self.assertRaises(ValueError):
                decode(name, b'corrupt')

    def test_unknown_tensor_rejected(self):
        with self.assertRaises(KeyError):
            decode('untrusted', b'')

    def test_invalid_header_prevents_payload_requests(self):
        with patch('dspark_confidence_fixture.read_range', return_value=(b'bad', CHECKPOINT_BYTES)) as read:
            with self.assertRaises(ValueError):
                fetch()
        self.assertEqual(read.call_count, 1)
