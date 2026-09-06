import importlib.util
from pathlib import Path
import unittest


spec = importlib.util.spec_from_file_location('attention_group_gate', Path(__file__).with_name('attention-group-timing.py'))
gate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gate)


class AttentionTimingTests(unittest.TestCase):
    def test_parallel_matrix_requires_complete_bundles(self):
        fixtures = self.fixtures()
        for fixture in fixtures:
            fixture['parallel_plan'] = gate.parallel_groups(fixture['start'], fixture['rows'])
        gate.validate_matrix(fixtures, parallel=True)
        fixtures[-1]['parallel_plan'] = []
        with self.assertRaises(AssertionError):
            gate.validate_matrix(fixtures, parallel=True)

    def test_tree_matrix_requires_distinct_bounded_group_plans(self):
        fixtures = self.fixtures()
        for fixture in fixtures:
            fixture['groups'] = gate.chunk_groups(fixture['start'], fixture['rows'])
            fixture['candidate_groups'] = gate.chunk_groups(fixture['start'], fixture['rows'], max_group_rows=8)
        gate.validate_matrix(fixtures, tree_scratch=True)
        fixtures[-1]['candidate_groups'] = fixtures[-1]['groups']
        with self.assertRaises(AssertionError):
            gate.validate_matrix(fixtures, tree_scratch=True)

    def fixtures(self):
        return [dict(seed=seed, rows=rows, start=start, exact=True, timed_replays=120, refreshed_checks=2,
                     samples=[dict(arm=arm, replay_ms=[0.1] * 10)
                              for arm in ['control', 'candidate', 'candidate', 'control'] * 3])
                for seed in (0, 1, 2) for rows in (1, 2, 4, 8, 16, 32) for start in (4095, 16383)]

    def test_complete_matrix(self):
        fixtures = self.fixtures()
        gate.validate_matrix(fixtures)
        self.assertEqual(sum(value['timed_replays'] for value in fixtures), 4320)

    def test_missing_or_duplicate_geometry_rejected(self):
        fixtures = self.fixtures()
        with self.assertRaises(AssertionError):
            gate.validate_matrix(fixtures[:-1])
        fixtures[-1] = fixtures[0]
        with self.assertRaises(AssertionError):
            gate.validate_matrix(fixtures)

    def test_invalid_evidence_rejected(self):
        for key, value in (('exact', False), ('timed_replays', 119), ('refreshed_checks', 0)):
            fixtures = self.fixtures()
            fixtures[0][key] = value
            with self.assertRaises(AssertionError):
                gate.validate_matrix(fixtures)
        for value in (0, float('nan'), float('inf'), True):
            fixtures = self.fixtures()
            fixtures[0]['samples'][0]['replay_ms'][0] = value
            with self.assertRaises(AssertionError):
                gate.validate_matrix(fixtures)
