import os
from pathlib import Path
import unittest
from unittest.mock import patch

from dflash_t32_cached_adapter import adapt_sources, require_simulator


class CachedT32AdapterTests(unittest.TestCase):
    def test_dedicated_t32_guards_preserve_t16_routes(self):
        directory = Path(__file__).parent
        original = {name: (directory / name).read_text() for name in
            ('dflash_device.py', 'draft_attention_branch.py', 'dflash_proposal_trace.py')}
        result = adapt_sources(original)
        self.assertIn('block_rows not in (8, 16)', original['dflash_device.py'])
        self.assertIn('block_rows not in (8, 16, 32)', result['dflash_device.py'])
        self.assertLess(result['dflash_device.py'].index('require_simulator()'),
            result['dflash_device.py'].index('self.operations, self.model'))
        for name in ('draft_attention_branch.py', 'dflash_proposal_trace.py'):
            self.assertIn('from dflash_t16_native_scope import require_active', result[name])
            self.assertIn('from dflash_t32_cached_adapter import require_simulator', result[name])
            self.assertIn('from dflash_t32_native_attention import', result[name])
        with self.assertRaises(ValueError):
            adapt_sources(result)

    def test_no_hardware_admission(self):
        with patch.dict(os.environ, {}, clear=True), self.assertRaises(ValueError):
            require_simulator()
        environment = dict(QWEN_SIM_ONLY='1', TT_METAL_SIMULATOR='sim')
        with patch.dict(os.environ, environment, clear=True):
            require_simulator()
        for name in ('QWEN_HARDWARE_TESTS', 'QWEN_CARDS_ALLOCATED'):
            with patch.dict(os.environ, {**environment, name: '1'}, clear=True), self.assertRaises(ValueError):
                require_simulator()


if __name__ == '__main__':
    unittest.main()
