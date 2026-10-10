"""CPU tests of the trace_ms ABAB reader: the parse of [PACKED-PHASE] lines, the per-third comparison, the host-skew flag and the verdict rules.

Run: python -B -m unittest discover -s optimisation/ttnn-op/shard_argmax -p 'test_*.py'
"""

import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import trace_ms_abab as abab  # noqa: E402

LINE = ('(EngineCore pid=66) 2026-10-03 23:56:23.097 | INFO     | packed_verifier:diagnostic:215 - [PACKED-PHASE] round=%d users=4 bind_ms=%.2f '
        'input_ms=%.2f trace_ms=%.2f sync_ms=0.11 readback_ms=%.2f live=%d idle=-')


def log(directory, name, per_third, live=4, counts=(40, 40, 40), inputs=(6.5, 6.5, 6.5), readback=0.9):
    """A log with `counts` live-N lines per third, trace_ms per_third[i] (with a little spread), one live=2 line between each, and noise lines."""
    path = Path(directory) / name
    lines = ['ordinary engine log line']
    number = 0
    for third, count in enumerate(counts):
        for index in range(count):
            number += 1
            spread = ((index * 7) % 5 - 2) * 0.01
            lines.append(LINE % (number, 3.3, inputs[third], per_third[third] + spread, readback, live))
            lines.append(LINE % (number, 3.3, inputs[third], 1.0, readback, 2))
    lines.append('[PACKED-PHASE] malformed')
    path.write_text('\n'.join(lines) + '\n')
    return str(path)


class ParseTests(unittest.TestCase):
    def test_only_the_live_lines_asked_for_are_read(self):
        with tempfile.TemporaryDirectory() as directory:
            path = log(directory, 'a.log', (55.0, 62.0, 83.0))
            four = abab.read_rounds(path, 4)
            two = abab.read_rounds(path, 2)
        self.assertEqual((len(four), len(two)), (120, 120))
        self.assertAlmostEqual(abab.median(four[:40], 2), 55.0, places=1)
        self.assertTrue(all(item[2] == 1.0 for item in two))

    def test_thirds_partition_the_rounds_in_order(self):
        rounds = list(range(10))
        parts = abab.thirds(rounds, 3)
        self.assertEqual([len(part) for part in parts], [3, 3, 4])
        self.assertEqual([item for part in parts for item in part], rounds)


class VerdictTests(unittest.TestCase):
    def pairs(self, directory, control, lever, **keywords):
        a1 = log(directory, 'a1.log', control)
        b1 = log(directory, 'b1.log', lever)
        a2 = log(directory, 'a2.log', control)
        b2 = log(directory, 'b2.log', lever, **keywords)
        return a1, b1, a2, b2

    def run_cli(self, a1, b1, a2, b2, *extra):
        lines = []
        status = abab.main(['--control', a1, a2, '--lever', b1, b2, *extra], out=lines.append)
        return status, lines

    def test_a_lever_that_saves_three_quarters_of_a_millisecond_in_every_third_of_both_pairs_passes(self):
        with tempfile.TemporaryDirectory() as directory:
            status, lines = self.run_cli(*self.pairs(directory, (55.0, 62.0, 83.0), (54.25, 61.25, 82.25)))
        self.assertEqual(status, 0)
        self.assertTrue(lines[-1].startswith('TRACE_ABAB verdict=PASS'), lines[-1])
        self.assertTrue(any('pair=1 live=4' in line and 'delta_ms=-0.750' in line for line in lines))

    def test_a_lever_that_saves_nothing_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            status, lines = self.run_cli(*self.pairs(directory, (55.0, 62.0, 83.0), (55.0, 62.0, 83.0)))
        self.assertEqual(status, 1)
        self.assertIn('TRACE_ABAB verdict=FAIL', lines[-1])

    def test_a_saving_between_half_and_all_of_the_requirement_is_inconclusive(self):
        with tempfile.TemporaryDirectory() as directory:
            status, lines = self.run_cli(*self.pairs(directory, (55.0, 62.0, 83.0), (54.6, 61.6, 82.6)))
        self.assertEqual(status, 2)
        self.assertIn('TRACE_ABAB verdict=INCONCLUSIVE', lines[-1])

    def test_pairs_that_disagree_are_inconclusive(self):
        with tempfile.TemporaryDirectory() as directory:
            a1 = log(directory, 'a1.log', (55.0, 62.0, 83.0))
            b1 = log(directory, 'b1.log', (54.0, 61.0, 82.0))
            a2 = log(directory, 'a2.log', (55.0, 62.0, 83.0))
            b2 = log(directory, 'b2.log', (54.5, 61.5, 82.5))
            status, lines = self.run_cli(a1, b1, a2, b2)
        self.assertEqual(status, 2)
        self.assertIn('disagree', lines[-1])

    def test_too_few_rounds_are_inconclusive(self):
        with tempfile.TemporaryDirectory() as directory:
            a1 = log(directory, 'a1.log', (55.0, 62.0, 83.0), counts=(5, 5, 5))
            b1 = log(directory, 'b1.log', (54.0, 61.0, 82.0), counts=(5, 5, 5))
            lines = []
            status = abab.main(['--control', a1, '--lever', b1], out=lines.append)
        self.assertEqual(status, 2)
        self.assertIn('INCONCLUSIVE', lines[-1])
        self.assertIn('fewer than', lines[-1])

    def test_a_loaded_host_is_named_but_does_not_hide_the_device_number(self):
        with tempfile.TemporaryDirectory() as directory:
            a1 = log(directory, 'a1.log', (55.0, 62.0, 83.0))
            b1 = log(directory, 'b1.log', (54.25, 61.25, 82.25), inputs=(10.8, 10.8, 6.5), readback=1.8)
            a2 = log(directory, 'a2.log', (55.0, 62.0, 83.0))
            b2 = log(directory, 'b2.log', (54.25, 61.25, 82.25))
            status, lines = self.run_cli(a1, b1, a2, b2)
        self.assertEqual(status, 0)
        skewed = [line for line in lines if 'pair=1 split=' in line and 'host_skew=True' in line]
        self.assertEqual(len(skewed), 2)
        self.assertTrue(any('pair=1 split=3' in line and 'host_skew=False' in line for line in lines))

    def test_the_ts_pairs_numbers_read_as_they_were_measured(self):
        # TS1/TS2 and TS3/TS4: trace_ms 62.64 -> 61.94 and 62.63 -> 61.87 at live=4
        with tempfile.TemporaryDirectory() as directory:
            status, lines = self.run_cli(*self.pairs(directory, (55.97, 62.66, 83.33), (55.25, 61.96, 82.56)))
        self.assertEqual(status, 0)

    def test_the_json_report_carries_the_pairs_and_the_verdict(self):
        with tempfile.TemporaryDirectory() as directory:
            a1, b1, a2, b2 = self.pairs(directory, (55.0, 62.0, 83.0), (54.25, 61.25, 82.25))
            target = str(Path(directory) / 'out.json')
            status, _ = self.run_cli(a1, b1, a2, b2, '--json', target)
            report = json.loads(Path(target).read_text())
        self.assertEqual((status, report['verdict'], len(report['pairs']), len(report['pairs'][0]['splits'])), (0, 'PASS', 2, 3))

    def test_bad_input_is_refused(self):
        with tempfile.TemporaryDirectory() as directory, patch('sys.stderr'):
            a1 = log(directory, 'a1.log', (55.0, 62.0, 83.0))
            self.assertEqual(abab.main(['--control', a1, '--lever', a1, a1], out=lambda line: None), 2)           # unequal counts
            self.assertEqual(abab.main(['--control', a1, '--lever', str(Path(directory) / 'missing.log')], out=lambda line: None), 2)
            empty = Path(directory) / 'empty.log'
            empty.write_text('nothing\n')
            self.assertEqual(abab.main(['--control', a1, '--lever', str(empty)], out=lambda line: None), 2)


if __name__ == '__main__':
    unittest.main()
