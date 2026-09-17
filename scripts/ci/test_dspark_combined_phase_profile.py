import unittest
from unittest.mock import patch

from dspark_combined_phase_profile import attach_phases, profile_request_scope
from dspark_prepared_proposal import TracedDSparkDevice
from test_dspark_proposal_phase_profile import PhaseProfileTests


class CombinedPhaseTests(unittest.TestCase):
    def test_bounded_samples_preserve_proposals_and_restore_hooks(self):
        records = []
        first, first_clock = PhaseProfileTests().fixture()
        second, second_clock = PhaseProfileTests().fixture()

        def original(device, anchor, count):
            return device.prepared.propose(anchor, count)

        with patch.object(TracedDSparkDevice, 'propose', original):
            with profile_request_scope(records, clock=lambda: first_clock() + second_clock()):
                for device in (first, second):
                    for ordinal in range(5):
                        self.assertEqual(TracedDSparkDevice.propose(device, 7, 15), [7] * 15)
            self.assertIs(TracedDSparkDevice.propose, original)
        self.assertEqual(len(records), 6)
        result = attach_phases({'arms': {'scatter': {'pp': 2000, 'committed_tg': 14}}}, records)
        self.assertIsNone(result['arms']['scatter']['committed_tg'])
        self.assertIsNone(result['arms']['scatter']['pp'])
        self.assertEqual(result['arms']['scatter']['instrumented_committed_tg'], 14)
        self.assertFalse(result['performance_qualified'])

    def test_incomplete_profile_cannot_pass(self):
        with self.assertRaises(ValueError):
            attach_phases({'arms': {}}, [])


if __name__ == '__main__':
    unittest.main()
