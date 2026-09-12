import os
import unittest
from unittest.mock import patch

from dspark_context_selection import request_context
from target_t16_attention_gate import validate_request_option


class ContextTests(unittest.TestCase):
    def test_default(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(request_context(), 4096)

    def test_explicit_eight_k(self):
        with patch.dict(os.environ, {'QWEN_DSPARK_REQUEST_CONTEXT': '8192'}):
            self.assertEqual(request_context(), 8192)
            options = dict(rows=16, position=8192, remaining=256, replay=True,
                norm_batch=True, native_sampling=True, group_rows=4, short_context=False)
            validate_request_option(True, **options)
            with self.assertRaises(ValueError):
                validate_request_option(True, **dict(options, position=4096))

    def test_reject_unqualified_context(self):
        for value in ('', '8193', '16384', '262144', '8k'):
            with self.subTest(value=value), patch.dict(os.environ, {'QWEN_DSPARK_REQUEST_CONTEXT': value}):
                with self.assertRaises(ValueError):
                    request_context()
