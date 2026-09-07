import importlib.util
from pathlib import Path
import unittest


spec = importlib.util.spec_from_file_location('vsplit_timing_gate', Path(__file__).with_name('gdn-vsplit-timing.py'))
gate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gate)


class TimingMatrixTests(unittest.TestCase):
    def fixture(self):
        return [dict(seed=seed, rows=rows, timed_replays=120, exact=True, refreshed_checks=2)
                for seed in (0, 1, 2) for rows in (1, 2, 4, 8, 16, 32)]

    def test_complete_matrix_has_2160_replays(self):
        fixtures = self.fixture()
        gate.validate_matrix(fixtures)
        self.assertEqual(sum(record['timed_replays'] for record in fixtures), 2160)

    def test_stage_timing_requires_complete_finite_exact_matrix(self):
        fixtures = self.fixture()
        with self.assertRaises(AssertionError):
            gate.validate_matrix(fixtures, stage_timing=True)
        for record in fixtures:
            record['stage_attribution'] = dict(exact=True, timed_replays=120, recurrence_ms=.4, norm_gate_ms=.6)
        gate.validate_matrix(fixtures, stage_timing=True)
        fixtures[0]['stage_attribution']['norm_gate_ms'] = float('nan')
        with self.assertRaises(AssertionError):
            gate.validate_matrix(fixtures, stage_timing=True)

    def test_missing_or_duplicate_width_rejected(self):
        for duplicate in (False, True):
            fixtures = self.fixture()
            fixtures.pop()
            if duplicate:
                fixtures.append(fixtures[0])
            with self.assertRaises(AssertionError):
                gate.validate_matrix(fixtures)

    def test_inexact_or_short_replay_count_rejected(self):
        for key, value in (('exact', False), ('timed_replays', 119), ('refreshed_checks', 0)):
            fixtures = self.fixture()
            fixtures[0][key] = value
            with self.assertRaises(AssertionError):
                gate.validate_matrix(fixtures)
