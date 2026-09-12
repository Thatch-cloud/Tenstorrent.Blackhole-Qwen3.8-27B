import importlib.util
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest


spec = importlib.util.spec_from_file_location('sim_native_owner', Path(__file__).parents[2] / 'optimisation/sim/run-native-fixed-attention.py')
owner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(owner)


class NativeOwnerTests(unittest.TestCase):
    def test_timeout_leftovers_are_restored(self):
        with TemporaryDirectory() as folder:
            directory = Path(folder)
            source, lock = directory / 'kernel', directory / '.lock'
            source.write_bytes(b'patched')
            lock.touch()
            owner.restore_owned_sources(directory, {'kernel': b'original'}, {'kernel': b'patched'}, lock)
            self.assertEqual(source.read_bytes(), b'original')
            self.assertFalse(lock.exists())

    def test_unexpected_external_change_is_not_overwritten(self):
        with TemporaryDirectory() as folder:
            directory = Path(folder)
            source, lock = directory / 'kernel', directory / '.lock'
            source.write_bytes(b'external')
            lock.touch()
            with self.assertRaises(ValueError):
                owner.restore_owned_sources(directory, {'kernel': b'original'}, {'kernel': b'patched'}, lock)
            self.assertEqual(source.read_bytes(), b'external')
            self.assertTrue(lock.exists())

    def test_normal_child_cleanup_needs_no_rewrite(self):
        with TemporaryDirectory() as folder:
            directory = Path(folder)
            (directory / 'kernel').write_bytes(b'original')
            owner.restore_owned_sources(directory, {'kernel': b'original'}, {'kernel': b'patched'}, directory / '.lock')
