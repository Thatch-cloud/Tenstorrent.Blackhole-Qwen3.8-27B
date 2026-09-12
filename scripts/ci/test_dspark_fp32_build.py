from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from dspark_fp32_build import restore_registrations
from sdpa_graft_build import implementation_sources


class RegistrationTests(unittest.TestCase):
    def test_restores_all_existing_implementations(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            target = root / 'ttnn/cpp/ttnn/operations/transformer/sources.cmake'
            target.parent.mkdir(parents=True)
            target.write_bytes(b'set(TTNN_OP_TRANSFORMER_SRCS\n    existing.cpp\n)\n')
            with patch('dspark_fp32_build.audit_registrations', return_value={}):
                evidence = restore_registrations(root)
            result = target.read_text()
            self.assertIn('    existing.cpp\n', result)
            for name in implementation_sources():
                self.assertEqual(result.count('    ' + name + '\n'), 1)
            self.assertEqual(len(evidence['source_after']), 64)

    def test_unknown_registration_fails_without_edit(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            target = root / 'ttnn/cpp/ttnn/operations/transformer/sources.cmake'
            target.parent.mkdir(parents=True)
            target.write_bytes(b'unknown')
            with self.assertRaises(ValueError):
                restore_registrations(root)
            self.assertEqual(target.read_bytes(), b'unknown')
