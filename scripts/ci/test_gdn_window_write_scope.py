import hashlib
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from gdn_window_write_gate import REPORT_SHA256
from gdn_window_write_scope import scoped_window_writes


class WindowScopeTests(unittest.TestCase):
    def test_t16_only_restore_and_failure(self):
        for failed in (False, True):
            with tempfile.TemporaryDirectory() as temporary:
                directory = Path(temporary)
                candidate = directory / 'window-write-candidate'
                candidate.mkdir()
                source = ('def build_windows(*args):\n    raise RuntimeError("device failure")\n' if failed
                          else 'def build_windows(*args):\n    return "candidate"\n')
                hashes = {}
                for name in ('gdn_conv_windows.py', 'gdn_conv_windows.cpp',
                        'window-write-candidate/gdn_conv_windows.py', 'window-write-candidate/gdn_conv_windows.cpp'):
                    payload = source.encode() if name.endswith('.py') else b'kernel'
                    (directory / name).write_bytes(payload)
                    hashes[name] = hashlib.sha256(payload).hexdigest()
                native = lambda *args: 'native'
                module = SimpleNamespace(build_windows=native)
                admission = dict(report_sha256=REPORT_SHA256, passed=True, source_hashes=hashes, logical_width=8240)
                with patch.dict('sys.modules', {'gdn_conv_windows': module}):
                    try:
                        with scoped_window_writes(admission, directory) as audit:
                            self.assertEqual(module.build_windows(None, SimpleNamespace(shape=(1, 8, 8256)), []), 'native')
                            with self.assertRaisesRegex(ValueError, 'Unqualified T16 window geometry'):
                                module.build_windows(None, SimpleNamespace(shape=(1, 16, 8256)), [])
                            with self.assertRaises(ValueError):
                                with scoped_window_writes(admission, directory):
                                    self.fail('Nested override entered')
                            self.assertEqual(module.build_windows(None, SimpleNamespace(shape=(1, 16, 8240)),
                                [SimpleNamespace(shape=(1, 1, 5120))] * 4), 'candidate')
                    except RuntimeError:
                        self.assertTrue(failed)
                    self.assertIs(module.build_windows, native)
                    self.assertTrue(audit['restored'])
                    self.assertEqual(audit['hits'], 0 if failed else 1)
                    self.assertEqual(audit['fallbacks'], 1)
                    self.assertEqual(audit['shapes']['(1, 16, 8256)'], 1)
                    (candidate / 'gdn_conv_windows.cpp').write_bytes(b'changed')
                    with self.assertRaises(ValueError):
                        with scoped_window_writes(admission, directory):
                            self.fail('Changed kernel admitted')

    def test_missing_admission_rejected(self):
        with patch.dict('sys.modules', {'gdn_conv_windows': SimpleNamespace()}):
            with self.assertRaises(ValueError):
                with scoped_window_writes({}, '.'):
                    self.fail('Missing admission accepted')
