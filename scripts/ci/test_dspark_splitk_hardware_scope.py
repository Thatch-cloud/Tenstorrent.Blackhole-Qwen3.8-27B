import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import dspark_splitk_hardware_scope as scope


class HardwareScopeTests(unittest.TestCase):
    def test_restores_source_after_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / scope.HEADER
            source.parent.mkdir(parents=True)
            original = b'native kernel'
            source.write_bytes(original)
            with patch.object(scope, 'SOURCE_SHA256', hashlib.sha256(original).hexdigest()), \
                    patch.object(scope, 'transform', return_value='candidate kernel'), \
                    patch.object(scope, 'recurrence_transform', side_effect=lambda source: source), \
                    patch.object(scope, 'correction_rounding', side_effect=lambda source: source):
                with self.assertRaisesRegex(RuntimeError, 'probe failed'):
                    with scope.kernel_scope(directory) as evidence:
                        self.assertIn(b'QWEN_SPLITK_NATIVE_EXPERIMENT', source.read_bytes())
                        self.assertEqual(evidence['source_active'], hashlib.sha256(source.read_bytes()).hexdigest())
                        raise RuntimeError('probe failed')
                self.assertEqual(source.read_bytes(), original)
                self.assertFalse(source.with_suffix('.splitk-hardware.lock').exists())
                source.with_suffix('.splitk-hardware.lock').touch()
                with self.assertRaises(FileExistsError):
                    with scope.kernel_scope(directory):
                        self.fail('Existing owner must prevent mutation')
                self.assertEqual(source.read_bytes(), original)

    def test_rejects_unknown_source(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / scope.HEADER
            source.parent.mkdir(parents=True)
            source.write_bytes(b'unknown')
            with self.assertRaisesRegex(ValueError, 'pinned decode'):
                with scope.kernel_scope(directory):
                    self.fail('Unknown source must not run')


if __name__ == '__main__':
    unittest.main()
