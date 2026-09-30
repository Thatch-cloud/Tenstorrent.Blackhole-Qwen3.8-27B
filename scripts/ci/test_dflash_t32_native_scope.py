import os
import unittest
from unittest.mock import Mock, patch

import dflash_t32_cached_adapter
import dflash_t32_native_attention
import dflash_t32_native_scope as scope


class T32NativeScopeTests(unittest.TestCase):
    environment = dict(QWEN_T32_COMBINED_EXPERIMENT='1', QWEN_CARDS_ALLOCATED='1',
        QWEN_HARDWARE_TESTS='1', QWEN_FROZEN_COMBINED_RUNTIME='1')

    def test_no_bare_flag_admits_native_t32(self):
        with patch.dict(os.environ, self.environment, clear=True), self.assertRaises(ValueError):
            scope.require_active()

    def test_request_guard_and_binding_restoration(self):
        original_guard = dflash_t32_cached_adapter.require_simulator
        original_attention = dflash_t32_native_attention.attention
        attention = Mock(return_value='output')
        admission = dict(block_rows=32, approximate_proposals=True)
        with patch.dict(os.environ, self.environment, clear=True), \
                patch.object(scope, 'admit', return_value=admission), \
                patch.object(dflash_t32_native_attention, 'native_attention', attention):
            with self.assertRaisesRegex(RuntimeError, 'request failed'):
                with scope.scoped_native_t32('.', '.', '.', '.'):
                    self.assertEqual(dflash_t32_cached_adapter.require_simulator(), admission)
                    self.assertEqual(dflash_t32_native_attention.attention(None, None, None, None, None), 'output')
                    with self.assertRaises(ValueError):
                        with scope.scoped_native_t32('.', '.', '.', '.'):
                            self.fail('Nested admission')
                    with patch.dict(os.environ, {'QWEN_CARDS_ALLOCATED': '0'}), self.assertRaises(ValueError):
                        dflash_t32_native_attention.attention(None, None, None, None, None)
                    raise RuntimeError('request failed')
            self.assertIs(dflash_t32_cached_adapter.require_simulator, original_guard)
            self.assertIs(dflash_t32_native_attention.attention, original_attention)
            with self.assertRaises(ValueError):
                scope.require_active()


if __name__ == '__main__':
    unittest.main()
