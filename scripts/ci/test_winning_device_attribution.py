from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import winning_device_attribution as attribution


class WinningDeviceAttributionTests(unittest.TestCase):
    def test_unpinned_request_rejected_before_parsing(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / 'request.json').write_bytes(b'{}')
            with self.assertRaisesRegex(ValueError, 'retained completed experiment'):
                attribution.analyze(root)

    def test_complete_device_only_report_is_not_throughput(self):
        self.check_report()

    def test_incomplete_or_changed_report_rejected(self):
        for changes in ({'passed': False}, {'closed_cleanly': False},
                {'committed_tg': 200}, {'sources_after': {'changed': 'hash'}},
                {'request_output_limit': 256}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.check_report(changes)

    def check_report(self, changes=None):
        request = dict(length=4096, arm='publication', exact=True, state_exact=True, inactive_exact=True)
        report = dict(passed=True, closed_cleanly=True, committed_tg=None, request_output_limit=64,
            request_checks=[request], sources={'script': 'hash'}, sources_after={'script': 'hash'},
            native_sources={'native': 'hash'}, native_sources_after={'native': 'hash'})
        report.update(changes or {})
        payload = json.dumps(report).encode()
        device = b'header\n'
        with tempfile.TemporaryDirectory() as temporary, ExitStack() as stack:
            root = Path(temporary)
            (root / 'metadata').mkdir()
            (root / 'request.json').write_bytes(payload)
            (root / 'metadata/cpp_device_perf_report.csv').write_bytes(device)
            (root / 'console.log').write_text('markers')
            stack.enter_context(patch.object(attribution, 'REQUEST_SHA256', hashlib.sha256(payload).hexdigest()))
            stack.enter_context(patch.object(attribution, 'DEVICE_SHA256', hashlib.sha256(device).hexdigest()))
            records = stack.enter_context(patch.object(attribution, 'validate_records', return_value=['records']))
            traces = stack.enter_context(patch.object(attribution, 'analyze_traces', return_value=['devices']))
            result = attribution.analyze(root)
            records.assert_called_once_with(request, 'markers')
            self.assertEqual(traces.call_args.args[0], ['records'])
            self.assertEqual(traces.call_args.kwargs, {'full_rows': 16})
            self.assertTrue(result['passed'])
            self.assertFalse(result['performance_qualified'])
            self.assertFalse(result['host_tracy_available'])
            self.assertIsNone(result['committed_tg'])


if __name__ == '__main__':
    unittest.main()
