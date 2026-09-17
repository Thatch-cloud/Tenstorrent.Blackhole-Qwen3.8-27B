import hashlib
import io
import tarfile
import unittest

from score_layout_result import EXTRA_SOURCES, publication_attribution, request_diagnostics, validate_sources


class ScoreLayoutSourceTests(unittest.TestCase):
    def test_complete_publication_attribution_and_rejection_controls(self):
        import copy
        stages = ('features', 'prepare_history', 'publish_target', 'commit_history')
        request = dict(blocks=[dict(position=4096, committed=11, select_commit_ms=8)],
            publication_diagnostics=dict(records=[dict(position=4096, prefix=11, stage=stage,
                passed=True, host_ms=1, process_cpu_ms=2, gc_pauses=[]) for stage in stages]))
        self.assertEqual(publication_attribution(request), request['publication_diagnostics'])
        self.assertIsNone(publication_attribution({}))
        for mutation in ('missing', 'prefix', 'nan', 'gc', 'outside'):
            changed = copy.deepcopy(request)
            records = changed['publication_diagnostics']['records']
            if mutation == 'missing':
                records.pop()
            elif mutation == 'prefix':
                records[0]['prefix'] = 12
            elif mutation == 'nan':
                records[0]['host_ms'] = float('nan')
            elif mutation == 'gc':
                records[0]['gc_pauses'] = [dict(generation=2, duration_ms=10)]
            else:
                changed['blocks'][0]['select_commit_ms'] = 3
            with self.assertRaises(ValueError):
                publication_attribution(changed)

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
