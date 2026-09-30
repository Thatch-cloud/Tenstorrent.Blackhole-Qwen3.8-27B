import unittest
from unittest.mock import patch

from dspark_confidence_parity import upstream_class


class UpstreamSourceTests(unittest.TestCase):
    def test_unpinned_code_rejected_before_parsing(self):
        with patch('dspark_confidence_parity.ast.parse') as parse:
            with self.assertRaises(ValueError):
                upstream_class(b'raise RuntimeError("must not execute")')
        parse.assert_not_called()
