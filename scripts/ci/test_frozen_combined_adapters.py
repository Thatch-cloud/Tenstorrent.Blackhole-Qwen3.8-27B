import os
import subprocess
import unittest
from unittest.mock import patch

from frozen_combined_adapters import adapt_admission
from frozen_combined_runtime import qualify
from frozen_recipe_context import REVISION


class CombinedAdmissionTests(unittest.TestCase):
    def test_selected_admission_preserves_binary_and_factory_checks(self):
        source = subprocess.check_output(['git', 'show',
            f'{REVISION}:scripts/ci/dspark_8k_admission.py'], text=True)
        changed = adapt_admission(source)
        namespace = {}
        exec(changed, namespace)
        namespace['validate_request'](32768, 256)
        for context in (8192, 16384, 65536, 131072, 262144):
            with self.assertRaises(ValueError):
                namespace['validate_request'](context, 256)
        self.assertEqual(namespace['history_limit'](), 8192)
        token = namespace['_ADMISSION'].set({'capacity': 33024})
        try:
            self.assertEqual(namespace['history_limit'](), 33024)
        finally:
            namespace['_ADMISSION'].reset(token)
        start, end = '    factory_sha256 = verify_factory', '    admission = dict('
        self.assertEqual(source.split(start)[1].split(end)[0], changed.split(start)[1].split(end)[0])

    def test_default_or_simulator_runtime_cannot_enter_hardware_admission(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(ValueError, 'Explicit allocated offline'):
                qualify('.')
        with patch.dict(os.environ, {'QWEN_FROZEN_COMBINED_RUNTIME': '1',
                'QWEN_DSPARK_REQUEST_CONTEXT': '32768', 'QWEN_HARDWARE_TESTS': '1',
                'QWEN_CARDS_ALLOCATED': '1', 'TT_METAL_SIMULATOR': '1'}, clear=True):
            with self.assertRaisesRegex(ValueError, 'Explicit allocated offline'):
                qualify('.')


if __name__ == '__main__':
    unittest.main()
