from copy import deepcopy
import os
import unittest
from unittest.mock import patch

from dspark_splitk_timed import require_timed, validate_execution


class SplitKTimingTests(unittest.TestCase):
    def test_audit_selection_and_global_precision_rejected(self):
        flags = dict(QWEN_DSPARK_SFPU_TIMED='1', QWEN_DSPARK_SFPU_REQUEST_SCREEN='0',
            QWEN_TARGET_T16_64K_REQUEST='1', QWEN_DSPARK_CENTER_TILE_FILL='1')
        with patch('dspark_splitk_timed.require_selected'), patch.dict(os.environ, flags, clear=True):
            require_timed()
            for name, value in (('QWEN_DSPARK_SFPU_REQUEST_SCREEN', '1'),
                    ('QWEN_SPLITK_FP32_INTERMEDIATES', '1')):
                with patch.dict(os.environ, {name: value}), self.assertRaises(ValueError):
                    require_timed()

    def test_runtime_identity_restoration_and_execution_required(self):
        expected = dict(component={'qualified': True}, kernel={'source_active': 'hash'},
            combined_build_binaries={'native': 'hash'})
        record = dict(expected, attention_calls=50, kernel_restored=True)
        validate_execution([record], expected)
        for name in (*expected, 'attention_calls', 'kernel_restored'):
            changed = deepcopy(record)
            changed.pop(name)
            with self.assertRaises(ValueError):
                validate_execution([changed], expected)
        with self.assertRaises(ValueError):
            validate_execution([record, record], expected)
