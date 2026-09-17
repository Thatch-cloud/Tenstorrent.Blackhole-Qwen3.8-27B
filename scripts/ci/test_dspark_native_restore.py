import hashlib
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import dspark_native_restore as native


class DSparkNativeRestoreTests(unittest.TestCase):
    def test_only_exact_pinned_image_source_is_restored_to_verified_git_bytes(self):
        original,patched = b'original',b'patched'
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root/native.SOURCE
            path.parent.mkdir(parents=True)
            path.write_bytes(patched)
            with patch.object(native,'IMAGE_SHA256',hashlib.sha256(patched).hexdigest()), \
                    patch.object(native,'SIMULATED_SHA256',hashlib.sha256(original).hexdigest()), \
                    patch.object(native.subprocess,'run',return_value=SimpleNamespace(stdout=original)) as command:
                result = native.restore(root)
                self.assertEqual(path.read_bytes(),original)
                self.assertTrue(result['binary_rebuild_or_verified_restore_required'])
                command.assert_called_once()
                self.assertIn(native.REVISION+':'+native.SOURCE,command.call_args.args[0])
                native.restore(root)
                command.assert_called_once()

    def test_unknown_input_or_wrong_git_source_never_changes_the_file(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root/native.SOURCE
            path.parent.mkdir(parents=True)
            path.write_bytes(b'input')
            with patch.object(native.subprocess,'run',return_value=SimpleNamespace(stdout=b'wrong')) as command:
                with self.assertRaises(ValueError):
                    native.restore(root)
                command.assert_not_called()
                with patch.object(native,'IMAGE_SHA256',hashlib.sha256(b'input').hexdigest()):
                    with self.assertRaises(ValueError):
                        native.restore(root)
                self.assertEqual(path.read_bytes(),b'input')


if __name__=='__main__':
    unittest.main()
