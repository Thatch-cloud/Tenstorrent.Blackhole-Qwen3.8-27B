import hashlib
import io
import tarfile
import unittest

from score_layout_result import EXTRA_SOURCES, validate_sources


class ScoreLayoutSourceTests(unittest.TestCase):
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
