import copy
import unittest

from frozen_wait_combined_report import expected_executions


class CombinedWaitReportTests(unittest.TestCase):
    def fixture(self):
        attribution = dict(passed=True, context=32768, streams=1, devices=[dict(
            trace_id=7, device=str(chip), steady_replays=2,
            replays=[dict(rows=16, replay_session=replay) for replay in (1, 2, 3)]) for chip in (0, 1)])
        events = [dict(chip=chip, host_id=1000 + layer * 2 + chip, trace_id=7, replay=replay)
            for chip in (0, 1) for layer in range(64) for replay in (1, 2, 3)]
        return events, attribution

    def test_complete_actual_replays(self):
        events, attribution = self.fixture()
        self.assertEqual(len(expected_executions(events, attribution)), 384)

    def test_missing_layer_or_session_rejected(self):
        events, attribution = self.fixture()
        for candidate in (events[1:], [event for event in events if event['replay'] != 3]):
            with self.assertRaises(ValueError):
                expected_executions(candidate, attribution)

    def test_program_identity_must_remain_stable(self):
        events, attribution = self.fixture()
        events[0]['host_id'] = 999
        with self.assertRaises(ValueError):
            expected_executions(events, attribution)

    def test_failed_or_single_chip_attribution_rejected(self):
        events, attribution = self.fixture()
        for mutation in (lambda value: value.update(passed=False), lambda value: value['devices'].pop()):
            candidate = copy.deepcopy(attribution)
            mutation(candidate)
            with self.assertRaises(ValueError):
                expected_executions(events, candidate)


if __name__ == '__main__':
    unittest.main()
