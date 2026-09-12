from contextlib import ExitStack
import hashlib
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import t32_hardware_kernel as kernel


class HardwareKernelTests(unittest.TestCase):
    def test_simulator_policy_rejected_before_evidence_or_runtime_access(self):
        with patch.dict(os.environ, {'QWEN_SIM_ONLY': '1'}, clear=True), patch.object(kernel, 'qualify') as qualify:
            with self.assertRaises(ValueError), kernel.installed('/missing', '/missing', '/missing'):
                self.fail('Entered hardware scope')
            qualify.assert_not_called()

    def test_original_sources_restored_after_request_failure(self):
        with tempfile.TemporaryDirectory() as temporary, ExitStack() as stack:
            root = Path(temporary)
            directory = root / kernel.KERNEL_DIRECTORY
            directory.mkdir(parents=True)
            original = {name: ('original-' + name).encode() for name in kernel.SOURCE_HASHES}
            for name, payload in original.items():
                (directory / name).write_bytes(payload)
            binary = root / 'binary'
            binary.write_bytes(b'runtime')
            stack.enter_context(patch.dict(os.environ, {'QWEN_PROJECTION_LINKS': '4'}, clear=True))
            stack.enter_context(patch.object(kernel, 'validate', return_value={'backend': 'hardware'}))
            stack.enter_context(patch.object(kernel, 'qualify', return_value={'full_request_qualified': False}))
            stack.enter_context(patch.object(kernel, 'RUNTIME', {'binary': hashlib.sha256(b'runtime').hexdigest()}))
            stack.enter_context(patch.object(kernel, 'build_sources', return_value={name: b'candidate' for name in original}))
            with self.assertRaisesRegex(RuntimeError, 'request failed'):
                with kernel.installed(root, 'report', 'sources'):
                    self.assertTrue(all((directory / name).read_bytes() == b'candidate' for name in original))
                    raise RuntimeError('request failed')
            self.assertTrue(all((directory / name).read_bytes() == payload for name, payload in original.items()))
            self.assertFalse((directory / '.qwen-precise-draft.lock').exists())


if __name__ == '__main__':
    unittest.main()
