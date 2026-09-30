import hashlib
from pathlib import Path
import tempfile
import unittest

from prefill_cache_inventory import GENERATOR, MODEL, inventory


class PrefillCacheInventoryTests(unittest.TestCase):
    def fixture(self, root):
        for name in (MODEL + '/model.py', GENERATOR):
            source = root / name
            source.parent.mkdir(parents=True, exist_ok=True)
            source.write_bytes(b'raise RuntimeError("must not import runtime")\n')

    def test_retains_sources_without_execution_and_requires_fresh_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.fixture(root)
            output = root / 'result'
            report = inventory(root, output)
            self.assertEqual(len(report['files']), 2)
            self.assertFalse(report['devices_opened'])
            self.assertFalse(report['prefix_cache_enabled'])
            for name, record in report['files'].items():
                self.assertEqual((output / 'sources' / name).read_bytes(), (root / name).read_bytes())
                self.assertEqual(record['sha256'], hashlib.sha256((root / name).read_bytes()).hexdigest())
            with self.assertRaises(ValueError):
                inventory(root, output)

    def test_missing_and_oversized_sources_do_not_publish_inventory(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.fixture(root)
            (root / GENERATOR).write_bytes(b'x' * (512 * 1024 + 1))
            with self.assertRaises(ValueError):
                inventory(root, root / 'result')
            self.assertFalse((root / 'result').exists())
            (root / GENERATOR).unlink()
            with self.assertRaises(ValueError):
                inventory(root, root / 'result')


if __name__ == '__main__':
    unittest.main()
