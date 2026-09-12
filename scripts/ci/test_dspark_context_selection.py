import os
import unittest
from unittest.mock import patch

from dspark_context_selection import request_context, validate_history_capacity
from target_t16_attention_gate import validate_request_option


class ContextTests(unittest.TestCase):
    def test_output_headroom_is_part_of_capacity(self):
        validate_history_capacity(4096, 256)
        validate_history_capacity(7936, 256)
        for context, output in ((8192, 256), (7937, 256), (8192, 2), (4096, True)):
            with self.subTest(context=context, output=output), self.assertRaises(ValueError):
                validate_history_capacity(context, output)

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
