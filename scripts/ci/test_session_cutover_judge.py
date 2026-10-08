"""session_cutover_judge on synthetic timed logs (the timed block T0s TA1 TL1 TA2 TL2 TA3 of the chained session).

Run at py 3.11: `py -3.11 -B -m unittest test_session_cutover_judge` from scripts/ci."""

import json
from pathlib import Path
import sys
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import session_cutover_judge as judge  # noqa: E402
import test_w2ln_timing_compare as fakes  # noqa: E402


def load_log(value, samples=6):
    return '\n'.join('[LOAD] %d %.2f' % (1000 + 60 * index, value) for index in range(samples))


def make_dir(root, name, seconds, load=1.0, with_load=True, log_name='container.log', count=150):
    directory = Path(root) / name / 'c2-serving-1'
    directory.mkdir(parents=True)
    (directory / log_name).write_text(fakes.shaped(fakes.uniform(seconds), count=count), encoding='utf-8')
    if with_load:
        (directory / 'load.log').write_text(load_log(load), encoding='utf-8')
    return str(Path(root) / name)


def run(root, t0s=0.200, a=(0.200, 0.2005, 0.1995), l=(0.2005, 0.2005), baseline_load=1.0, extra=(), count=150, l_load=1.0):
    args = ['--t0s', make_dir(root, 'T0s', t0s, count=count), '--a'] + [make_dir(root, 'TA%d' % (i + 1), v, count=count) for i, v in enumerate(a)]
    args += ['--l'] + [make_dir(root, 'TL%d' % (i + 1), v, load=l_load, count=count) for i, v in enumerate(l)]
    if baseline_load is not None:
        args += ['--baseline-load', make_dir(root, 'S0', 0.2, load=baseline_load)]
    lines = []
    status = judge.main(args + list(extra), out=lines.append)
    return status, json.loads(lines[0])


class JudgeTests(unittest.TestCase):
    def test_a_lever_n_cost_inside_the_floor_and_production_bytes_inside_it_hold(self):
        with tempfile.TemporaryDirectory() as tmp:
            status, report = run(tmp)
        self.assertEqual(status, 0, report)
        self.assertTrue(report['holds'])
        self.assertEqual(sorted(report['windows']), ['32k', 'steady'])
        self.assertTrue(all(entry['holds'] for entry in report['windows'].values()))
        self.assertEqual(len(report['windows']['steady']['pairs']), 3)

    def test_a_lever_n_cost_past_the_floor_fails_naming_the_pair(self):
        with tempfile.TemporaryDirectory() as tmp:
            status, report = run(tmp, l=(0.2005, 0.214))
        self.assertEqual(status, 1)
        reasons = ' '.join(report['windows']['steady']['reasons'])
        self.assertIn('TA2-TL2', reasons)
        self.assertNotIn('TA1-TL1', reasons)

    def test_a_faster_lever_n_is_no_cost(self):
        with tempfile.TemporaryDirectory() as tmp:
            status, report = run(tmp, l=(0.190, 0.190))
        self.assertEqual(status, 0, report)

    def test_production_bytes_off_by_more_than_the_floor_in_either_direction_fail(self):
        for value in (0.214, 0.186):
            with self.subTest(t0s=value), tempfile.TemporaryDirectory() as tmp:
                status, report = run(tmp, t0s=value)
                self.assertEqual(status, 1)
                self.assertIn('TA1-T0s', ' '.join(report['windows']['steady']['reasons']))

    def test_a_void_pair_is_never_inside(self):
        with tempfile.TemporaryDirectory() as tmp:
            status, report = run(tmp, count=20)
        self.assertEqual(status, 1)
        self.assertIn('never read as inside', ' '.join(report['windows']['steady']['reasons']))

    def test_a_loaded_lever_n_job_voids_its_pair(self):
        with tempfile.TemporaryDirectory() as tmp:
            status, report = run(tmp, l_load=9.0)
        self.assertEqual(status, 1)
        self.assertIn('VOID', ' '.join(report['windows']['steady']['reasons']))

    def test_no_baseline_refuses_unless_the_load_rule_is_waived(self):
        with tempfile.TemporaryDirectory() as tmp:
            status, report = run(tmp, baseline_load=None)
        self.assertEqual(status, 1)
        self.assertFalse(report['holds'])
        self.assertIn('--baseline-load', report['error'])
        with tempfile.TemporaryDirectory() as tmp:
            status, report = run(tmp, baseline_load=None, extra=['--no-load-check'])
        self.assertEqual(status, 0, report)

    def test_a_missing_log_fails_closed_with_an_error_not_a_pass(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = ['--t0s', make_dir(tmp, 'T0s', 0.2, log_name='other.txt'), '--a'] + [make_dir(tmp, 'TA%d' % i, 0.2) for i in (1, 2, 3)]
            args += ['--l', make_dir(tmp, 'TL1', 0.2), make_dir(tmp, 'TL2', 0.2), '--no-load-check']
            lines = []
            status = judge.main(args, out=lines.append)
        self.assertEqual(status, 1)
        self.assertIn('T0s', json.loads(lines[0])['error'])

    def test_server_log_is_read_when_there_is_no_container_log(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = ['--t0s', make_dir(tmp, 'T0s', 0.2, log_name='server.log'), '--a'] + [make_dir(tmp, 'TA%d' % i, 0.2, log_name='server.log') for i in (1, 2, 3)]
            args += ['--l', make_dir(tmp, 'TL1', 0.2, log_name='server.log'), make_dir(tmp, 'TL2', 0.2, log_name='server.log'), '--no-load-check']
            status = judge.main(args, out=lambda text: None)
        self.assertEqual(status, 0)


if __name__ == '__main__':
    unittest.main()
