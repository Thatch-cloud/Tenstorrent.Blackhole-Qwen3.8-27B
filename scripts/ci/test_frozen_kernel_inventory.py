import hashlib
from pathlib import Path
import tempfile
import unittest

from frozen_kernel_inventory import inventory, KERNELS


class InventoryTests(unittest.TestCase):
    def test_exporter_retained_without_kernel_marker(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / 'runtime'
            exporter = root / 'tt_metal/tools/tracy/export.py'
            exporter.parent.mkdir(parents=True)
            exporter.write_text('def export_zones(): pass\n')
            output = Path(temporary) / 'output'
            result = inventory(root, output)
            self.assertEqual(len(result['files']), 1)
            self.assertEqual((output / 'sources' / exporter.relative_to(root)).read_bytes(),
                exporter.read_bytes())

    def test_bounded_read_only_inventory(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / 'runtime'
            for name in KERNELS:
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b'kernel')
            profiler = root / 'tt_metal/hw/inc/kernel_profiler.hpp'
            profiler.parent.mkdir(parents=True, exist_ok=True)
            profiler.write_text('DeviceZoneScopedN\ncb_wait\n')
            output = Path(temporary) / 'output'
            result = inventory(root, output)
            self.assertFalse(result['devices_opened'])
            self.assertEqual(len(result['files']), 5)
            self.assertEqual(result['files'][0]['sha256'], hashlib.sha256(b'kernel').hexdigest())
            self.assertEqual(result['profiler_candidates'][0]['total_matches'], 2)
            self.assertEqual((output / 'sources' / KERNELS[0]).read_bytes(), b'kernel')
            with self.assertRaises(ValueError):
                inventory(root, output)


if __name__ == '__main__':
    unittest.main()
