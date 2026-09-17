from pathlib import Path
import tempfile
import unittest

from qwen_load_diagnostic import inspect_tree, filesystem


class LoadDiagnosticTests(unittest.TestCase):
    def test_source_snapshot_is_bounded_and_read_only(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'one.py').write_text('value = 1\n')
            (root / 'two.py').write_text('value = 2\n')
            report = inspect_tree(root, limit=1)
            self.assertTrue(report['truncated'])
            self.assertEqual(len(report['files']), 1)
            self.assertEqual(report['files'][0]['source'], 'value = 1\n')
            self.assertEqual((root / 'two.py').read_text(), 'value = 2\n')
            self.assertEqual(inspect_tree(root, byte_limit=1)['files'], [])

    def test_missing_filesystem_is_explicit(self):
        with tempfile.TemporaryDirectory() as directory:
            self.assertEqual(filesystem(Path(directory) / 'missing')['unavailable'], 'FileNotFoundError')
