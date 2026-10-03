"""record_packed_any_evidence_tp4 --capacity 262144: the recorder for the 262,144-token window's own record.

Three passing 262k reports become a record packed_any_admission qualifies at capacity 262,144 and ONLY the 262k file and its pin are
written: the 131,328 record and EVIDENCE_TP4_SHA256 stay byte for byte. A 131k report is refused at 262,144 and a 262k report is
refused without --capacity (no silent cross-over in either direction)."""

import copy
import hashlib
import json
from pathlib import Path
import re
import shutil
import sys
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import packed_any_admission as admission  # noqa: E402
import record_packed_any_evidence_tp4 as recorder  # noqa: E402
import test_record_packed_any_evidence_tp4 as base  # noqa: E402

CAPACITY = 262144
COMMIT = 'b' * 40


def cb1_262k():
    report = base.cb1_report()
    report.update(capacity=CAPACITY, extents=list(admission.K1_EXTENTS_262K))
    return report


def cb2a_262k():
    rows = base.comparisons(dict(k2_native_vs_extent=2030, x7_narrow_vs_wide=300, x7_extent_vs_wide=300, z_trace_vs_eager=450,
                                 z_trace_vs_reference=450))
    report = base.cb2a_report()
    report.update(capacity=CAPACITY, comparisons=rows, tally=base.tally_of(rows),
                  cb2a=dict(cb2_extents=list(admission.CB2_EXTENTS_262K)))
    report['decision']['k2_coverage'] = dict(full=True, covered=2030, design=2030, short=[])
    report['verdict_line'] = ('K64J_CARD verdict=PASS capacity=262144 extent=none k2=2030/2030 k2_rows=29730/29730 '
                              'k2_floor_differing=0/3280 k2_verdict=PASS k2_coverage=2030/2030 x7=600/600 z=900/900 z_families=15 '
                              'kv_heads=1')
    return report


def cb2b_262k():
    report = base.cb2b_report()
    report['capacity'] = CAPACITY
    report['r2_families_replayed'] = sorted(set(report['r2_families_replayed']) | {262144})
    report['verdict_line'] = report['verdict_line'].replace('families=', 'families=', 1).replace(
        ' chips=1of4', ' capacity=262144 chips=1of4')
    return report


class Workspace262:
    def __init__(self, test):
        self.dir = Path(tempfile.mkdtemp(prefix='recorder262-test-'))
        test.addCleanup(shutil.rmtree, str(self.dir), True)
        self.evidence = self.dir / 'packed_any_evidence_tp4_262144.json'
        self.evidence131 = self.dir / 'packed_any_evidence_tp4.json'
        self.admission = self.dir / 'packed_any_admission.py'
        shutil.copyfile(admission.EVIDENCE_TP4_262K, self.evidence)
        shutil.copyfile(admission.EVIDENCE_TP4, self.evidence131)
        source = (HERE / 'packed_any_admission.py').read_text(encoding='utf-8')
        pin = hashlib.sha256(self.evidence.read_bytes()).hexdigest()
        assert source.count(admission.EVIDENCE_TP4_262K_SHA256) == 1
        self.admission.write_text(source.replace(admission.EVIDENCE_TP4_262K_SHA256, pin), encoding='utf-8', newline=chr(10))
        self.before131 = (self.evidence131.read_bytes(), self.pin131())

    def report(self, name, report):
        path = self.dir / name
        path.write_text(json.dumps(report, indent=2), encoding='utf-8')
        return str(path)

    def argv(self, cb1=None, cb2a=None, cb2b=None, capacity=CAPACITY, extra=()):
        args = ['--evidence', str(self.evidence), '--admission', str(self.admission), '--sources-root', str(HERE)]
        if capacity is not None:
            args += ['--capacity', str(capacity)]
        if cb1 is not None:
            args += ['--cb1', self.report('card-e2a.json', cb1), '--cb1-run', '2001', '--cb1-tag', 'experiment/c2-serving-v951']
        if cb2a is not None:
            args += ['--cb2a', self.report('card-e2b.json', cb2a), '--cb2a-run', '2002', '--cb2a-tag', 'experiment/c2-serving-v952']
        if cb2b is not None:
            args += ['--cb2b', self.report('reader-e2c.json', cb2b), '--cb2b-run', '2003', '--cb2b-tag',
                     'experiment/c2-serving-v953', '--cb2b-commit', COMMIT, '--cb2b-image', 'tp4-262k8-1']
        return args + list(extra)

    def run(self, **kwargs):
        lines = []
        code = recorder.main(self.argv(**kwargs), out=lines.append)
        return code, lines

    def pin(self, name='EVIDENCE_TP4_262K_SHA256'):
        return re.search(r"^%s = '([0-9a-f]{64})'" % name, self.admission.read_text(encoding='utf-8'), re.M).group(1)

    def pin131(self):
        return self.pin('EVIDENCE_TP4_SHA256')


class Record262Tests(unittest.TestCase):
    def test_three_262k_reports_become_a_record_the_admission_qualifies_at_262k_and_only_its_pin_follows(self):
        work = Workspace262(self)
        code, lines = work.run(cb1=cb1_262k(), cb2a=cb2a_262k(), cb2b=cb2b_262k())
        self.assertEqual(code, 0, lines)
        payload = work.evidence.read_bytes()
        self.assertNotIn(b'\r', payload)
        self.assertEqual(work.pin(), hashlib.sha256(payload).hexdigest())
        record = json.loads(payload.decode('utf-8'))
        self.assertEqual(record['capacity'], CAPACITY)
        self.assertEqual(admission.evidence_problems(record, str(HERE), tp=4, capacity=CAPACITY), [])
        admission.check_evidence(str(work.evidence), expected_sha256=work.pin(), sources_root=str(HERE), tp=4, capacity=CAPACITY)
        self.assertEqual([record['sections'][name]['status'] for name in admission.SECTIONS], ['PASS'] * 3)
        self.assertEqual([record['sections'][name]['capacity'] for name in admission.SECTIONS], [CAPACITY] * 3)
        self.assertIn('2,030 tickets', record['what'])
        self.assertNotIn('SKELETON', record['what'])
        self.assertEqual(record['sections']['CB2a']['k2']['tickets'], [2030, 2030])
        self.assertEqual(record['sections']['CB2a']['x7'], [600, 600])
        self.assertIn('--capacity 262144', record['sections']['CB1']['harness'])
        # the 131,328 record and its pin are never touched
        self.assertEqual((work.evidence131.read_bytes(), work.pin131()), work.before131)
        self.assertEqual(work.pin131(), admission.EVIDENCE_TP4_SHA256)

    def test_the_repos_own_files_are_not_touched_by_a_test_run(self):
        before = [path.read_bytes() for path in (admission.EVIDENCE_TP4, admission.EVIDENCE_TP4_262K, HERE / 'packed_any_admission.py')]
        Workspace262(self).run(cb1=cb1_262k())
        self.assertEqual(before, [path.read_bytes() for path in (admission.EVIDENCE_TP4, admission.EVIDENCE_TP4_262K,
                                                                    HERE / 'packed_any_admission.py')])

    def test_the_default_evidence_path_is_the_capacitys_own(self):
        self.assertEqual(admission.evidence_path(4, CAPACITY).name, 'packed_any_evidence_tp4_262144.json')
        self.assertEqual(recorder.parse_args(['--capacity', '262144']).evidence, None)
        self.assertEqual(recorder.parse_args([]).capacity, 131328)
        with self.assertRaises(SystemExit):
            import contextlib
            import io
            with contextlib.redirect_stderr(io.StringIO()):
                recorder.parse_args(['--capacity', '200000'])

    def test_a_131k_report_is_not_262k_evidence(self):
        for label, kwargs in (('CB1', dict(cb1=base.cb1_report())), ('CB2a', dict(cb2a=base.cb2a_report())),
                              ('CB2b', dict(cb2b=base.cb2b_report()))):
            work = Workspace262(self)
            code, lines = work.run(**kwargs)
            self.assertEqual(code, 1, (label, lines))
            self.assertTrue(any('capacity' in line for line in lines), (label, lines))
            self.assertEqual(json.loads(work.evidence.read_text())['sections'][label]['status'], 'PENDING')

    def test_a_262k_report_is_not_131k_evidence(self):
        # without --capacity the recorder keeps the 131,328 rules: the wide reports' extents/capacity are not its design set only
        # where they differ (the capacity), and the 131k file is the one it would write
        work = Workspace262(self)
        code, lines = work.run(cb2b=cb2b_262k(), capacity=None, extra=['--evidence', str(work.evidence131)])
        self.assertEqual(code, 1, lines)
        self.assertTrue(any('capacity 262144, not 131328' in line for line in lines), lines)
        self.assertEqual((work.evidence131.read_bytes(), work.pin131()), work.before131)

    def test_each_262k_shortfall_is_refused_alone(self):
        cases = {
            'CB1 without the full window': ('cb1', lambda r: r.update(extents=list(admission.K1_EXTENTS_262K[:-1]))),
            'CB1 without 196864': ('cb1', lambda r: r.update(extents=[e for e in admission.K1_EXTENTS_262K if e != 196864])),
            'CB2a without the window family': ('cb2a', lambda r: r['cb2a'].update(cb2_extents=list(admission.CB2_EXTENTS))),
            'CB2a K2 at 1980': ('cb2a', lambda r: r['decision'].update(k2_coverage=dict(full=True, covered=1980, design=1980, short=[]))),
            'CB2a X7 below 600': ('cb2a', lambda r: r.update(comparisons=[c for c in r['comparisons']
                                                                           if c['label'] != 'x7_narrow_vs_wide/0'])),
            'CB2b without the named window': ('cb2b', lambda r: r.update(r2_families_replayed=[
                f for f in r['r2_families_replayed'] if f != 262144])),
            'CB2b capacity word missing': ('cb2b', lambda r: r.update(verdict_line=r['verdict_line'].replace(' capacity=262144', ''))),
        }
        reports = dict(cb1=cb1_262k, cb2a=cb2a_262k, cb2b=cb2b_262k)
        for label, (key, mutate) in cases.items():
            work = Workspace262(self)
            report = reports[key]()
            mutate(report)
            if label == 'CB2a X7 below 600':
                report['tally'] = base.tally_of(report['comparisons'])
            code, lines = work.run(**{key: report})
            self.assertEqual(code, 1, (label, lines))
            self.assertEqual(json.loads(work.evidence.read_text())['sections'][{'cb1': 'CB1', 'cb2a': 'CB2a', 'cb2b': 'CB2b'}[key]]
                             ['status'], 'PENDING', label)

    def test_the_131k_recorder_path_is_what_it_was(self):
        work = base.Workspace(self)
        code, lines = work.run(cb1=base.cb1_report(), cb2a=base.cb2a_report(), cb2b=base.cb2b_report())
        self.assertEqual(code, 0, lines)
        record = json.loads(work.evidence.read_text())
        self.assertNotIn('capacity', record)
        self.assertEqual(admission.evidence_problems(record, str(HERE), tp=4), [])
        self.assertTrue(all('capacity' not in record['sections'][name] or name == 'CB2b' for name in ('CB1', 'CB2a')))


if __name__ == '__main__':
    unittest.main()
