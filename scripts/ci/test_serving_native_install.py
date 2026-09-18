import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import serving_native_install as install


class NativeInstallTests(unittest.TestCase):
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
