import importlib.util
from pathlib import Path
import tempfile
import unittest


spec = importlib.util.spec_from_file_location('gdn_source_audit', Path(__file__).with_name('gdn-source-audit.py'))
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


class OwnershipExportTests(unittest.TestCase):
    def test_requires_slice_implementation(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(RuntimeError):
                audit.ownership_sources(Path(directory))

    def test_only_bounded_source_roots_and_extensions(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'ttnn/cpp/ttnn/operations/data_movement/slice'
            source.mkdir(parents=True)
            expected = source / 'slice.cpp'
            expected.write_text('slice source')
            (source / 'not-source.bin').write_bytes(b'ignored')
            (root / 'unrelated.cpp').write_text('not exported')
            paths, missing = audit.ownership_sources(root)
            self.assertEqual(paths, [expected])
            self.assertIn('ttnn/cpp/ttnn/operations/data_movement/clone', missing)

    def test_oversized_export_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'ttnn/cpp/ttnn/operations/data_movement/slice'
            source.mkdir(parents=True)
            (source / 'slice.cpp').write_bytes(b' ' * (4 * 1024 * 1024 + 1))
            with self.assertRaises(RuntimeError):
                audit.ownership_sources(root)


if __name__ == '__main__':
    unittest.main()
