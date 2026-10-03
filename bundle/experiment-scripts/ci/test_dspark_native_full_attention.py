from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from dspark_native_full_attention import execute


class NativeFullAttentionTests(unittest.TestCase):
    def test_passes_original_operands_to_native_without_external_conversions(self):
        operations, mesh = SimpleNamespace(), object()
        operands = [object() for index in range(4)]
        owned, output = [], object()
        with patch('dspark_native_full_attention.validate_inputs') as validate, patch(
                'dspark_native_full_attention.draft_sdpa', return_value=output) as native:
            self.assertIs(execute(operations, mesh, *operands, owned,
                context_rows=4384, proposals=15, mask_validated=True), output)
        validate.assert_called_once_with(operations, *operands, 4384, 15, True)
        native.assert_called_once_with(operations, *operands, key_chunk_size=64)
        self.assertEqual(owned, [output])

    def test_invalid_layout_cannot_reach_native_dispatch(self):
        with patch('dspark_native_full_attention.validate_inputs', side_effect=ValueError('layout')):
            with patch('dspark_native_full_attention.draft_sdpa') as native:
                with self.assertRaisesRegex(ValueError, 'layout'):
                    execute(Mock(), object(), *[object() for index in range(4)], [],
                        context_rows=4384, proposals=15)
                native.assert_not_called()
