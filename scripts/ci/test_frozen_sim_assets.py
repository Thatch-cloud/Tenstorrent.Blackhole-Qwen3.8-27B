import hashlib
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import frozen_sim_assets as assets


class AssetTests(unittest.TestCase):
    def test_cold_warm_and_corrupt_cache(self):
        payload = b'pinned simulator fixture'
        checksum = hashlib.sha256(payload).hexdigest()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cache = root / 'cache'

            def download(command, **options):
                self.assertEqual(options, dict(check=True, timeout=25))
                self.assertIn('--max-time', command)
                Path(command[-1]).write_bytes(payload)

            with patch.object(assets, 'ASSETS', (('sim.so', 'https://example.invalid/sim', checksum),)), \
                    patch.object(assets.subprocess, 'run', side_effect=download) as fetch:
                self.assertFalse(assets.prepare(cache, root / 'cold')[0]['cache_hit'])
                fetch.assert_called_once()
                fetch.reset_mock()
                self.assertTrue(assets.prepare(cache, root / 'warm')[0]['cache_hit'])
                fetch.assert_not_called()
                (cache / checksum).write_bytes(b'corrupt')
                self.assertFalse(assets.prepare(cache, root / 'repaired')[0]['cache_hit'])
                self.assertEqual((root / 'repaired/sim.so').read_bytes(), payload)
                with self.assertRaisesRegex(ValueError, 'Fresh asset destination'):
                    assets.prepare(cache, root / 'warm')

    def test_download_failure_never_publishes_cache(self):
        checksum = hashlib.sha256(b'expected').hexdigest()
        for failure in ('hash', 'timeout'):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)

                def download(command, **options):
                    if failure == 'timeout':
                        raise subprocess.TimeoutExpired(command, 25)
                    Path(command[-1]).write_bytes(b'wrong')

                with patch.object(assets, 'ASSETS', (('sim.so', 'https://example.invalid/sim', checksum),)), \
                        patch.object(assets.subprocess, 'run', side_effect=download):
                    with self.assertRaises((ValueError, subprocess.TimeoutExpired)):
                        assets.prepare(root / 'cache', root / 'destination')
                self.assertFalse((root / 'cache' / checksum).exists())
                self.assertFalse((root / 'destination/sim.so').exists())


if __name__ == '__main__':
    unittest.main()
