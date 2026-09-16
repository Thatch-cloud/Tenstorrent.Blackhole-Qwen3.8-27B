import hashlib
from pathlib import Path
import tempfile
import unittest

from frozen_wait_raw_export import export


class RawExportTests(unittest.TestCase):
    def test_preserves_all_marker_rows_including_duplicate_and_incomplete(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            header = b'ARCH: blackhole, CHIP_FREQ[MHz]: 1350\nzone name,type\n'
            marker = b'QWEN_MLP_INPUT_FREE,ZONE_START\n'
            raw = header + marker + b'OTHER,ZONE_START\n' + marker
            source, destination = root / 'raw.csv', root / 'selected.csv'
            source.write_bytes(raw)
            result = export(source, destination)
            self.assertEqual(source.read_bytes(), raw)
            self.assertEqual(destination.read_bytes(), header + marker * 2)
            self.assertEqual(result['source_sha256'], hashlib.sha256(raw).hexdigest())
            self.assertEqual(result['selected_sha256'], hashlib.sha256(destination.read_bytes()).hexdigest())
            self.assertEqual(result['selected_rows'], 2)
            self.assertIsNone(result['committed_tg'])
            with self.assertRaises(ValueError):
                export(source, source)

    def test_empty_or_wrong_architecture_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for raw in (b'', b'ARCH: other,\nzone name\nQWEN_MLP_X\n',
                    b'ARCH: blackhole,\nzone name\nOTHER\n'):
                (root / 'raw.csv').write_bytes(raw)
                with self.assertRaises(ValueError):
                    export(root / 'raw.csv', root / 'selected.csv')
