from copy import deepcopy
import json
import unittest

from tensix_stream_gate import NATIVE_SOURCES, SOURCES, qualify
from tensix_weight_stream import stream_geometry


class TensixStreamGateTests(unittest.TestCase):
    def fixture(self):
        return dict(passed=True, closed_cleanly=True, backend='simulator',
            sources={name: 'a' * 64 for name in SOURCES}, native_sources={name: 'b' * 64 for name in NATIVE_SOURCES},
            geometry=json.loads(json.dumps(stream_geometry('gate', 5))),
            eager_checks=[dict(pattern=pattern, arm=arm, chip=chip, exact_packed_words=True, inputs_unchanged=True)
                for pattern in range(2) for arm in range(2) for chip in range(2)],
            replay_checks=[dict(repetition=repetition, pattern=pattern, arm=arm, chip=chip,
                exact_packed_words=True, inputs_unchanged=True, bindings_stable=True)
                for repetition, pattern in enumerate((0, 1, 0)) for arm in range(2) for chip in range(2)],
            negative_controls=[dict(arm=arm, chip=chip, stale_detected=True) for arm in range(2) for chip in range(2)])

    def test_complete_short_gate_does_not_claim_full_weights(self):
        report = self.fixture()
        result = qualify(report, report['sources'], report['native_sources'])
        self.assertTrue(result['passed'])
        self.assertFalse(result['full_projection'])

    def test_full_gate_and_down_require_their_actual_block_counts(self):
        for projection, blocks in (('gate', 20), ('down', 34)):
            report = self.fixture()
            report['geometry'] = json.loads(json.dumps(stream_geometry(projection, blocks)))
            self.assertTrue(qualify(report, report['sources'], report['native_sources'])['full_projection'])

    def test_missing_duplicate_or_falsified_checks_fail_closed(self):
        original = self.fixture()
        for field in ('eager_checks', 'replay_checks', 'negative_controls'):
            for mutation in ('missing', 'duplicate', 'flag', 'bool_index'):
                report = deepcopy(original)
                rows = report[field]
                if mutation == 'missing':
                    rows.pop()
                elif mutation == 'duplicate':
                    rows[1] = rows[0]
                elif mutation == 'flag':
                    flag = 'stale_detected' if field == 'negative_controls' else 'exact_packed_words'
                    rows[0][flag] = 1
                else:
                    rows[0]['chip'] = False
                with self.subTest(field=field, mutation=mutation), self.assertRaises(ValueError):
                    qualify(report, original['sources'], original['native_sources'])

    def test_scope_errors_drift_and_incomplete_cleanup_are_not_passes(self):
        original = self.fixture()
        for mutation in ('backend', 'passed', 'closed_cleanly', 'error', 'sources', 'native_sources', 'geometry'):
            report = deepcopy(original)
            if mutation in ('sources', 'native_sources'):
                report[mutation].pop(next(iter(report[mutation])))
            elif mutation == 'geometry':
                report['geometry']['blocks'] = 20
            else:
                report[mutation] = 'hardware' if mutation == 'backend' else 'invalid'
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                qualify(report, original['sources'], original['native_sources'])


if __name__ == '__main__':
    unittest.main()
