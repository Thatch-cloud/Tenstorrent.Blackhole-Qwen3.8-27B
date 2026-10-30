"""pf3_report: the traced-tail pair reader and the trace-region headroom reader, on synthetic logs in the formats the lever and the memory ledger print.

The headroom figures in HEADROOM_LOG are the aggregates of the ship-5 window (2026-10-19): a 640 MiB trace region, 69.9 MB used after the attach, 410.1 MB after the first packed round (the eight
parked engines), 419.2 MB at most over the run, i.e. a least free of 251.9 MB; the audited twin held 205.6 / 545.8 / 556.3 MB, i.e. 114.8 MB free."""

import contextlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import pf3_report as report  # noqa: E402

LED = '2026-10-19 12:50:54.650 | INFO     | memory_ledger:log_line:176 - [MEMLEDGER] '


def ledger(used_after_attach, used_first_round, used_max, total=671.1):
    lines = []
    for chip in range(4):
        lines.append(LED + 'phase=P7 point=after_attach chip%d trace_total=%.1fMB trace_used=%.1fMB trace_free=%.1fMB trace_largest_free=%.1fMB'
                     % (chip, total, used_after_attach, total - used_after_attach, total - used_after_attach))
    for chip in range(4):
        lines.append(LED + 'phase=P12 point=first_packed_round chip%d trace_total=%.1fMB trace_used=%.1fMB trace_free=%.1fMB trace_largest_free=%.1fMB'
                     % (chip, total, used_first_round, total - used_first_round, total - used_first_round))
    lines.append(LED + 'after op=pair point=slots=6,7 chip0 cost=108.2MB largest_free=3715.7MB trace_used=%.1fMB trace_largest_free=%.1fMB' % (used_max, total - used_max))
    lines.append(LED + 'phase=P7 point=after_attach chip0 allocated=25.255GB free=8.253GB largest_free=8044.5MB total=33.508GB known=25.2GB residual=0.0GB')
    return lines


HEADROOM_LOG = ledger(69.9, 410.1, 419.2)
AUDITED_LOG = ledger(205.6, 545.8, 556.3)


def tails(mode, values, start=2048):
    lines = []
    for bucket, ms in values:
        extra = ' stage_ms=3.1 replay_ms=%.1f' % (ms - 3.1) if mode == 'traced' else ''
        lines.append('[PF3] tail mode=%s bucket=%d valid=%d start=%d ms=%.1f%s' % (mode, bucket, bucket - 5, start, ms, extra))
    return lines


def route(start, ms):
    return '[PINDIAG] lever N route req=r start=%d end=%d prompt=%d final=1 wrote_slot=1 ms=%.1f programs=1->1 window=0 source=COLD captured=-' % (start, start + 3000, start + 3000, ms)


class Files(object):
    def __init__(self, case):
        self.case, self.paths = case, []

    def write(self, lines):
        handle = tempfile.NamedTemporaryFile('w', suffix='.log', delete=False, encoding='utf-8')
        handle.write('\n'.join(lines) + '\n')
        handle.close()
        self.case.addCleanup(os.unlink, handle.name)
        return handle.name


class PairTests(unittest.TestCase):
    def setUp(self):
        self.files = Files(self)
        self.control = self.files.write(tails('eager', [(128, 330.0), (128, 334.0), (512, 345.0), (1024, 395.0)]) + [route(100000, 1200.0)] + ledger(69.9, 410.1, 419.2))
        self.arm = self.files.write(['[PINDIAG] prefill tail trace engaged buckets=128,256,512,1024 traces=4 sink=16500 taps=5,19,33,47,61 arena_mb=320 arena_address=1,2 capture_ms=9000']
                                    + tails('traced', [(128, 120.0), (128, 124.0), (512, 130.0), (1024, 180.0)]) + [route(100000, 990.0)] + ledger(150.0, 490.0, 499.0))

    def test_a_pair_reads_the_saving_per_bucket_and_the_trace_region_taken(self):
        summary = report.summarise(report.read([self.control]), report.read([self.arm]))
        self.assertEqual(summary['problems'], [])
        self.assertEqual(summary['control'][128]['mean'], 332.0)
        self.assertEqual(summary['arm'][128]['mean'], 122.0)
        self.assertEqual(summary['savings'][128]['saving_ms'], 210.0)
        self.assertEqual(summary['savings'][1024]['saving_ms'], 215.0)
        self.assertEqual(summary['arena_mb'], 320)
        self.assertEqual(summary['trace_region_taken_mb'], 80.1)
        self.assertEqual(summary['routes']['control']['96-160k']['mean'], 1200.0)
        self.assertEqual(summary['routes']['arm']['96-160k']['mean'], 990.0)

    def test_the_command_line_prints_the_verdict_and_exits_zero_for_a_pair(self):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = report.main(['--control', self.control, '--arm', self.arm])
        self.assertEqual(code, 0)
        self.assertIn('VERDICT: a pair', out.getvalue())
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            self.assertEqual(report.main(['--control', self.control, '--arm', self.arm, '--json']), 0)
        self.assertEqual(json.loads(out.getvalue())['problems'], [])

    def test_a_pair_that_is_not_a_pair_is_named(self):
        cases = {
            'no engaged': (self.control, self.files.write(tails('traced', [(128, 120.0)]))),
            'fell back': (self.control, self.files.write(['[PINDIAG] prefill tail trace engaged arena_mb=320', '[PINDIAG] prefill tail trace fell back reason=staging'] + tails('traced', [(128, 1.0)]))),
            'mismatch': (self.control, self.files.write(['[PINDIAG] prefill tail trace engaged arena_mb=320', '[PINDIAG] prefill tail trace audit mismatch n=1'] + tails('traced', [(128, 1.0)]))),
            'inexact audit': (self.control, self.files.write(['[PINDIAG] prefill tail trace engaged arena_mb=320', '[PINDIAG] prefill tail trace audit n=1 exact=False bucket=128'] + tails('traced', [(128, 1.0)]))),
            'control ran the lever': (self.files.write(['[PINDIAG] prefill tail trace engaged arena_mb=320'] + tails('traced', [(128, 1.0)])), self.arm),
            'control timed nothing': (self.files.write(['nothing']), self.arm),
            'arm timed no traced tail': (self.control, self.files.write(['[PINDIAG] prefill tail trace engaged arena_mb=320'] + tails('eager', [(128, 330.0)]))),
        }
        for name, (control, arm) in cases.items():
            with self.subTest(name):
                self.assertTrue(report.summarise(report.read([control]), report.read([arm]))['problems'], name)
                with contextlib.redirect_stdout(io.StringIO()):
                    self.assertEqual(report.main(['--control', control, '--arm', arm]), 1)

    def test_an_unreadable_log_exits_two(self):
        with contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(report.main(['--control', '/nonexistent/x.log', '--arm', self.arm]), 2)

    def test_two_logs_of_one_arm_are_one_population(self):
        second = self.files.write(tails('traced', [(128, 126.0)]))
        arm = report.read([self.arm, second])
        self.assertEqual(sum(1 for entry in arm['tails'] if entry[1] == 128), 3)


class HeadroomTests(unittest.TestCase):
    def setUp(self):
        self.files = Files(self)

    def test_the_ship5_figures_are_read_back(self):
        chips = report.headroom([self.files.write(HEADROOM_LOG)])
        self.assertEqual(sorted(chips), [0, 1, 2, 3])
        self.assertEqual(chips[0]['total'], 671.1)
        self.assertEqual((chips[0]['used_min'], chips[0]['used_max']), (69.9, 419.2))
        self.assertEqual(round(chips[0]['free_min'], 1), 251.9)

    def test_four_traces_fit_the_traffic_candidate_and_not_the_audited_twin_at_the_high_end(self):
        traffic = report.headroom([self.files.write(HEADROOM_LOG)])
        audited = report.headroom([self.files.write(AUDITED_LOG)])
        low, high, free, verdict = report.fit(traffic, 4, 3400, 6000, 16000)
        self.assertEqual((round(low, 1), round(high, 1), round(free, 1)), (81.6, 217.6, 251.9))
        self.assertEqual(verdict, 'FITS at both ends')
        low, high, free, verdict = report.fit(audited, 4, 3400, 6000, 16000)
        self.assertEqual(round(free, 1), 114.8)
        self.assertEqual(verdict, 'FITS at the low estimate only')
        low, high, free, verdict = report.fit(audited, 2, 3400, 6000, 16000)
        self.assertEqual(verdict, 'FITS at both ends')
        self.assertEqual(report.fit(traffic, 9, 3400, 16000, 16000)[3], 'DOES NOT FIT')

    def test_a_log_without_trace_lines_is_unknown_not_fitting(self):
        chips = report.headroom([self.files.write(['nothing here'])])
        self.assertEqual(chips, {})
        self.assertTrue(report.fit(chips, 4, 3400, 6000, 16000)[3].startswith('UNKNOWN'))

    def test_the_command_line(self):
        path = self.files.write(HEADROOM_LOG)
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            self.assertEqual(report.main(['--headroom', path]), 0)
        self.assertIn('FITS at both ends', out.getvalue())
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(report.main(['--headroom', self.files.write(AUDITED_LOG)]), 1)
        with self.assertRaises(SystemExit):
            report.main([])
        with self.assertRaises(SystemExit):
            report.main(['--headroom', path, '--control', path])


if __name__ == '__main__':
    unittest.main()
