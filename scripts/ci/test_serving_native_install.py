import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import serving_native_install as install


class NativeInstallTests(unittest.TestCase):
    def test_bundled_patch_is_hash_checked_before_git_application(self):
        with TemporaryDirectory() as directory:
            patch_file = Path(directory) / 'sdpa-graft-registration.patch'
            patch_file.write_bytes(b'patch')
            with patch.object(install.subprocess, 'run') as run:
                install.apply_patch_file(Path(directory), patch_file, install.digest(patch_file))
                self.assertEqual(run.call_args_list[0].args[0],
                    ['git', '-C', directory, 'apply', '--check', str(patch_file)])
                self.assertEqual(run.call_count, 2)
                with self.assertRaises(ValueError):
                    install.apply_patch_file(Path(directory), patch_file, '0' * 64)
                self.assertEqual(run.call_count, 2)

    def test_binary_and_builder_bytes_are_checked_before_native_edits(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            binary = root / 'binary.so'
            binary.write_bytes(b'binary')
            builder = root / 'builder.py'
            builder.write_text('source')
            inputs = {'builders': {'builder.py': install.digest(builder)}}
            key = hashlib.sha256(json.dumps(inputs, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
            manifest = {'inputs': inputs, 'binary_sha256': install.digest(binary)}
            with patch.object(install, 'CACHE_KEY', key), patch.object(install, 'BINARY_SHA256', install.digest(binary)):
                self.assertEqual(install.validate_cache(manifest, root, binary), inputs)
                builder.write_text('changed')
                with self.assertRaisesRegex(ValueError, 'builder changed'):
                    install.validate_cache(manifest, root, binary)
                binary.write_bytes(b'changed')
                with self.assertRaisesRegex(ValueError, 'cache provenance'):
                    install.validate_cache(manifest, root, binary)
