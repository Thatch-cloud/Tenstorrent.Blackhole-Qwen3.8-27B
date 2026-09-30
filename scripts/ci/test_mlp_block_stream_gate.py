import copy
import unittest

from mlp_block_stream_gate import validate_report


class BlockStreamGateTests(unittest.TestCase):
    def fixture(self):
        return dict(passed=True, backend='simulator', timings=[],
            checks=[dict(rows=16, chip=chip, exact=True) for chip in range(2)],
            stream_bytes_before=['a' * 64, 'b' * 64], stream_bytes_after=['a' * 64, 'b' * 64],
            trace_replays=[dict(rows=16, passed=True, timings=[],
                checks=[dict(arm=arm, repetition=repetition, pattern=pattern, chip=chip, exact=True)
                    for repetition, pattern in enumerate((0, 1, 0))
                    for arm in ('control', 'fused') for chip in range(2)],
                negative_controls=[dict(arm=arm, chip=chip, stale_input_detected=True)
                    for arm in ('control', 'fused') for chip in range(2)])])

    def test_complete_matrix(self):
        validate_report(self.fixture())

    def test_missing_or_duplicate_replay_cannot_pass(self):
        for field in ('checks', 'negative_controls'):
            report = self.fixture()
            report['trace_replays'][0][field].pop()
            with self.assertRaises(ValueError):
                validate_report(report)
            report = self.fixture()
            rows = report['trace_replays'][0][field]
            rows[1] = copy.deepcopy(rows[0])
            with self.assertRaises(ValueError):
                validate_report(report)

    def test_changed_stream_or_failed_execution_cannot_pass(self):
        for field, invalid in (('stream_bytes_after', ['c' * 64, 'b' * 64]),
                ('passed', False), ('backend', 'hardware'), ('timings', [1]), ('error', 'failure')):
            report = self.fixture()
            report[field] = invalid
            with self.assertRaises(ValueError):
                validate_report(report)


if __name__ == '__main__':
    unittest.main()
