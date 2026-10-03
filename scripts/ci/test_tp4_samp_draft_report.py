"""tp4_samp_draft_report: the tau rule of the tp4/samp-draft window, on synthetic container logs and on stats."""

import io
import json
import os
import tempfile
import unittest
from contextlib import redirect_stdout

import tp4_samp_draft_report as report


def log_of(committed_per_block, requests=1):
    """A container log with `requests` close records, each holding one full-draft block (rows 16) per entry of `committed_per_block`
    plus a terminal block (excluded from the statistics)."""
    lines = []
    for _ in range(requests):
        blocks = [dict(position=100 + 16 * index, rows=16, committed=committed, verifier=dict(packed=True, users=4))
                  for index, committed in enumerate(list(committed_per_block) + [3])]
        lines.append('(EngineCore pid=57) ' + json.dumps(dict(stage='fast_serving_phases', blocks=blocks, cancelled=False,
                                                              finished=True)))
    return '\n'.join(lines) + '\n'


def stats(rounds, mean, positions=None):
    return dict(rounds=rounds, mean=mean, positions=positions if positions is not None else [0.9, 0.8, 0.7])


class ArmStatsTests(unittest.TestCase):
    def test_the_full_draft_rounds_and_mean_come_from_the_phase_records(self):
        found = report.arm_stats(log_of([4, 6, 8, 10], requests=3))
        self.assertEqual(found['rounds'], 12)
        self.assertAlmostEqual(found['mean'], 7.0)
        self.assertEqual(len(found['positions']), 15)

    def test_a_log_with_no_full_draft_round_has_no_stats(self):
        self.assertIsNone(report.arm_stats('nothing here\n'))


class PoolTests(unittest.TestCase):
    def test_the_pool_weights_each_log_by_its_rounds(self):
        pooled = report.pool([stats(1000, 4.0, [1.0]), stats(3000, 8.0, [0.5]), None])
        self.assertEqual(pooled['rounds'], 4000)
        self.assertAlmostEqual(pooled['mean'], 7.0)
        self.assertAlmostEqual(pooled['positions'][0], 0.625)

    def test_nothing_to_pool_is_none(self):
        self.assertIsNone(report.pool([None, None]))


class JudgeTests(unittest.TestCase):
    def test_an_identical_lever_passes(self):
        found = report.judge([stats(2500, 7.0), stats(2500, 7.0)], [stats(2500, 7.0)])
        self.assertEqual(found['verdict'], 'PASS')
        self.assertEqual(found['reasons'], [])
        self.assertAlmostEqual(found['tau_ratio'], 1.0)

    def test_a_lever_within_one_percent_passes_and_below_it_fails(self):
        self.assertEqual(report.judge([stats(3000, 7.0)], [stats(3000, 6.94)])['verdict'], 'PASS')
        failed = report.judge([stats(3000, 7.0)], [stats(3000, 6.9)])
        self.assertEqual(failed['verdict'], 'FAIL')
        self.assertIn('below', failed['reasons'][0])

    def test_a_per_position_drop_of_more_than_two_points_fails_even_at_equal_tau(self):
        found = report.judge([stats(3000, 7.0, [0.9, 0.8])], [stats(3000, 7.0, [0.9, 0.77])])
        self.assertEqual(found['verdict'], 'FAIL')
        self.assertIn('position(s) 2', found['reasons'][0])

    def test_too_few_rounds_on_either_side_is_insufficient_not_a_pass(self):
        self.assertEqual(report.judge([stats(1000, 7.0)], [stats(5000, 7.0)])['verdict'], 'INSUFFICIENT')
        self.assertEqual(report.judge([stats(5000, 7.0)], [stats(1999, 7.0)])['verdict'], 'INSUFFICIENT')
        self.assertEqual(report.judge([stats(5000, 7.0)], [stats(2000, 7.0)])['verdict'], 'PASS')

    def test_controls_that_do_not_reproduce_each_other_are_insufficient(self):
        found = report.judge([stats(3000, 7.0), stats(3000, 7.2)], [stats(3000, 7.1)])
        self.assertEqual(found['verdict'], 'INSUFFICIENT')
        self.assertGreater(found['spread'], report.CONTROL_TOLERANCE)

    def test_a_failure_is_not_softened_by_noisy_controls(self):
        found = report.judge([stats(3000, 7.0), stats(3000, 7.2)], [stats(3000, 6.0)])
        self.assertEqual(found['verdict'], 'FAIL')

    def test_a_missing_side_is_insufficient(self):
        self.assertEqual(report.judge([None], [stats(3000, 7.0)])['verdict'], 'INSUFFICIENT')
        self.assertEqual(report.judge([stats(3000, 7.0)], [None])['verdict'], 'INSUFFICIENT')

    def test_the_summary_line_names_the_verdict_and_the_numbers(self):
        line = report.summary_line(report.judge([stats(3000, 7.0)], [stats(3000, 7.0)]))
        self.assertTrue(line.startswith('[SAMP-DRAFT-TAU] verdict=PASS control_rounds=3000 control_tau=7.0000'))
        self.assertIn('ratio=1.0000', line)


class CommandTests(unittest.TestCase):
    def run_main(self, controls, levers, *extra):
        with tempfile.TemporaryDirectory() as folder:
            paths = []
            for name, text in (*[('a%d.log' % index, text) for index, text in enumerate(controls)],
                               *[('b%d.log' % index, text) for index, text in enumerate(levers)]):
                path = os.path.join(folder, name)
                with open(path, 'w', encoding='utf-8') as handle:
                    handle.write(text)
                paths.append(path)
            out = os.path.join(folder, 'out.json')
            buffer = io.StringIO()
            with redirect_stdout(buffer):
                code = report.main(['--control', *paths[:len(controls)], '--lever', *paths[len(controls):], '--json', out, *extra])
            with open(out, encoding='utf-8') as handle:
                return code, buffer.getvalue(), json.load(handle)

    def test_exit_codes_follow_the_verdict(self):
        same = log_of([8] * 700, requests=4)
        code, output, payload = self.run_main([same, same], [same])
        self.assertEqual((code, payload['verdict']), (0, 'PASS'))
        self.assertIn('[SAMP-DRAFT-TAU] verdict=PASS', output)
        worse = log_of([6] * 700, requests=4)
        self.assertEqual(self.run_main([same, same], [worse])[0], 1)
        self.assertEqual(self.run_main([log_of([8] * 10), log_of([8] * 10)], [log_of([8] * 10)])[0], 2)


if __name__ == '__main__':
    unittest.main()
