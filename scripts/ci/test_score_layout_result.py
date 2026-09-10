import hashlib
import io
import tarfile
import unittest

from score_layout_result import EXTRA_SOURCES, request_diagnostics, validate_sources


class ScoreLayoutSourceTests(unittest.TestCase):
    def test_host_diagnostics_recomputed_and_absence_preserved(self):
        from request_host_health import summarize
        request = dict(arm='control', instrumented_timing=False, committed_decode_tokens=52,
            decode_ms=500, prefill_ms=1200)
        self.assertIsNone(request_diagnostics([request])[0]['host_health'])
        request['host_health'] = summarize(dict(monotonic_ns=1, cgroup={'cpu.stat': 'nr_throttled 1'}),
            dict(monotonic_ns=2, cgroup={'cpu.stat': 'nr_throttled 3'}))
        self.assertEqual(request_diagnostics([request])[0]['host_health']['counter_deltas']['cpu.stat']['nr_throttled'], 2)
        request['host_health']['counter_deltas']['cpu.stat']['nr_throttled'] = 0
        with self.assertRaises(ValueError):
            request_diagnostics([request])

    def fixture(self):
        files = {'scripts/ci/probe.py': b'probe', 'scripts/ci/kernel.cpp': b'kernel',
            'speculative-decoding/harness/session.py': b'session', 'scripts/ci/result.json': b'{}'}
        files.update({'scripts/ci/' + name: name.encode() for name in EXTRA_SOURCES})
        output = io.BytesIO()
        with tarfile.open(fileobj=output, mode='w') as archive:
            for name, data in files.items():
                member = tarfile.TarInfo(name)
                member.size = len(data)
                archive.addfile(member, io.BytesIO(data))
        sources = {'probe.py': hashlib.sha256(b'probe').hexdigest(),
            'kernel.cpp': hashlib.sha256(b'kernel').hexdigest(),
            '../../speculative-decoding/harness/session.py': hashlib.sha256(b'session').hexdigest()}
        sources.update({name: hashlib.sha256(name.encode()).hexdigest() for name in EXTRA_SOURCES})
        return dict(sources=sources, sources_after=dict(sources), native_sources={'runtime': 'hash'},
            native_sources_after={'runtime': 'hash'}), output.getvalue()

    def test_complete_sources_include_harness_and_exclude_report_data(self):
        report, snapshot = self.fixture()
        self.assertEqual(validate_sources(report, snapshot), 3 + len(EXTRA_SOURCES))

    def test_missing_extra_changed_or_native_drift_rejected(self):
        for mutation in ('missing', 'extra', 'changed', 'native'):
            report, snapshot = self.fixture()
            if mutation == 'missing':
                report['sources'].pop('probe.py')
            elif mutation == 'extra':
                report['sources']['extra.py'] = 'hash'
            elif mutation == 'changed':
                report['sources_after']['probe.py'] = 'wrong'
            else:
                report['native_sources_after'] = {}
            with self.assertRaises(ValueError):
                validate_sources(report, snapshot)
