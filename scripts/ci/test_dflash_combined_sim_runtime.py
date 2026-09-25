import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import dflash_combined_sim_runtime as runtime
from dspark_fp32_intermediates import REPLACEMENT
from frozen_context_geometry import factory_selector


class CombinedSimulatorRuntimeTests(unittest.TestCase):
    def test_factory_transform_matches_frozen_recipe(self):
        historical_selector = factory_selector().replace('Skt == 8200 || ', '')
        self.assertEqual(runtime.REPLACEMENT, REPLACEMENT.replace('Skt == 272', historical_selector))
        with self.assertRaisesRegex(ValueError, 'original SDPA factory'):
            runtime.factory_bytes(b'unreviewed factory')

    def test_cache_identity_and_binary_are_checked_before_mutation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            inputs = dict(recipe='fixture')
            binary = b'fixture binary'
            checksum = hashlib.sha256(binary).hexdigest()
            key = hashlib.sha256(json.dumps(inputs, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
            (root / '_ttnncpp.so').write_bytes(binary)
            (root / 'manifest.json').write_text(json.dumps(dict(inputs=inputs, binary_sha256=checksum)))
            with patch.object(runtime, 'CACHE_KEY', key), patch.object(runtime, 'BINARY_SHA256', checksum), \
                    patch.object(runtime, 'factory_bytes', side_effect=RuntimeError('validated cache')):
                for name in runtime.BINARIES:
                    path = root / name
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(binary)
                self.assertEqual(runtime.binary_hashes(root), dict.fromkeys(runtime.BINARIES, checksum))
                factory = root / runtime.FACTORY
                factory.parent.mkdir(parents=True, exist_ok=True)
                factory.write_bytes(b'original')
                with self.assertRaisesRegex(RuntimeError, 'validated cache'):
                    runtime.install(root, root)
                (root / '_ttnncpp.so').write_bytes(b'corrupt')
                with self.assertRaisesRegex(ValueError, 'cache entry'):
                    runtime.install(root, root)
                self.assertEqual(factory.read_bytes(), b'original')
                (root / runtime.BINARIES[0]).write_bytes(b'changed')
                with self.assertRaisesRegex(ValueError, 'runtime binary'):
                    runtime.binary_hashes(root)


if __name__ == '__main__':
    unittest.main()
