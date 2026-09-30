import hashlib
from pathlib import Path
import tempfile
import unittest

from gdn_conv_inventory import COMPONENT, inventory


class ConvInventoryTests(unittest.TestCase):
    def test_bounded_export_and_fresh_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / 'runtime'
            kernel = root / COMPONENT / 'device/kernels/reader.cpp'
            kernel.parent.mkdir(parents=True)
            kernel.write_bytes(b'native source')
            output = Path(temporary) / 'output'
            report = inventory(root, output)
            self.assertFalse(report['devices_opened'])
            self.assertFalse(report['model_weights_loaded'])
            self.assertEqual(report['files'][0]['sha256'], hashlib.sha256(b'native source').hexdigest())
            self.assertEqual((output / 'sources' / kernel.relative_to(root)).read_bytes(), b'native source')
            self.assertEqual(kernel.read_bytes(), b'native source')
            with self.assertRaises(ValueError):
                inventory(root, output)

    def test_missing_and_oversized_graft_fail_before_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / 'output'
            with self.assertRaises(ValueError):
                inventory(root, output)
            kernel = root / COMPONENT / 'device/kernels/reader.cpp'
            kernel.parent.mkdir(parents=True)
            kernel.write_bytes(b'x' * (512 * 1024 + 1))
            with self.assertRaises(ValueError):
                inventory(root, output)
            self.assertFalse(output.exists())
