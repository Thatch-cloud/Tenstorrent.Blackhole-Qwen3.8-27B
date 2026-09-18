import hashlib
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import serving_image_preflight as preflight


class ImagePreflightTests(unittest.TestCase):
    def check(self, root, *, imported=None):
        with patch.object(preflight, 'SOURCES', {'source.py': hashlib.sha256(b'pinned').hexdigest()}), \
                patch.object(preflight, 'DEPENDENCIES', {'test_dependency': ('entry',)}), \
                patch.object(preflight.importlib, 'import_module', return_value=imported or SimpleNamespace(entry=lambda: None)), \
                patch('serving_request_factory.device_components', return_value=SimpleNamespace(engine=lambda: None)):
            return preflight.audit(root)

    def test_all_checks_pass_without_claiming_serving_acceptance(self):
        with TemporaryDirectory() as directory:
            (Path(directory) / 'source.py').write_bytes(b'pinned')
            report = self.check(directory)
            self.assertTrue(report['passed'])
            self.assertEqual(report['failures'], {})
            for name in ('devices_accessed', 'weights_loaded', 'serving_qualified', 'performance_qualified'):
                self.assertFalse(report[name])

    def test_missing_and_changed_sources_fail(self):
        with TemporaryDirectory() as directory:
            self.assertIn('source.py', self.check(directory)['failures'])
            (Path(directory) / 'source.py').write_bytes(b'changed')
            report = self.check(directory)
            self.assertFalse(report['passed'])
            self.assertIn('fingerprint mismatch', report['failures']['source.py'])

    def test_missing_dependency_is_reported_alongside_source_failure(self):
        with TemporaryDirectory() as directory:
            report = self.check(directory, imported=SimpleNamespace())
            self.assertFalse(report['passed'])
            self.assertEqual(set(report['failures']), {'source.py', 'test_dependency'})
