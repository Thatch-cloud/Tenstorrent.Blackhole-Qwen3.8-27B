import hashlib
from io import BytesIO
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import cumulative_native_sources as sources


class NativeSourcesTests(unittest.TestCase):
    def exercise(self, *, corrupt_download=False, corrupt_existing=False, invalid_report=False, register=False):
        payload = b'pinned native source'
        expected = hashlib.sha256(payload).hexdigest()
        raw = json.dumps(dict(sources={'/opt/tt-metal/' + name: expected for name in sources.PATHS})).encode()
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            report = root / 'report.json'
            report.write_bytes(raw)
            destination = root / 'native'
            if corrupt_existing:
                target = destination / sources.PATHS[0]
                target.parent.mkdir(parents=True)
                target.write_bytes(b'wrong')
            with patch.object(sources, 'REPORT_SHA256', 'invalid' if invalid_report else hashlib.sha256(raw).hexdigest()), \
                    patch.object(sources, 'TYPECAST', expected), patch.object(sources, 'HARDWARE_PACKER', expected), \
                    patch.object(sources, 'validate_report'), \
                    patch.object(sources, 'urlopen', side_effect=lambda *args, **kwargs:
                        BytesIO(b'wrong' if corrupt_download else payload)) as download:
                if corrupt_download or corrupt_existing or invalid_report:
                    with self.assertRaises(ValueError):
                        sources.restore(report, destination, register_epilogue=register)
                    self.assertFalse((destination / sources.PATHS[-1]).exists())
                    if corrupt_existing or invalid_report:
                        download.assert_not_called()
                    return
                result = sources.restore(report, destination, register_epilogue=register)
                self.assertEqual(len(result['sources']), 9 if register else 7)
                self.assertIs(result['register_epilogue'], register)
                self.assertFalse(result['hardware_qualified'])
                for call in download.call_args_list:
                    self.assertIn('/' + sources.REVISION + '/', call.args[0])
                download.reset_mock()
                self.assertEqual(sources.restore(report, destination, register_epilogue=register), result)
                download.assert_not_called()

    def test_pinned_download_and_verified_reuse(self):
        self.exercise()

    def test_bad_download_writes_no_partial_inventory(self):
        self.exercise(corrupt_download=True)

    def test_existing_corruption_is_not_overwritten(self):
        self.exercise(corrupt_existing=True)

    def test_unqualified_report_cannot_request_downloads(self):
        self.exercise(invalid_report=True)

    def test_register_adds_pinned_cast_and_physical_packer(self):
        self.exercise(register=True)
