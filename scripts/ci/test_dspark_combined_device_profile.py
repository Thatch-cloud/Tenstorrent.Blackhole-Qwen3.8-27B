from contextlib import redirect_stdout
import io
import os
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from dspark_combined_device_profile import attach_device_profile, profile_device_scope
from dspark_draft_profile import DraftProfile
from dspark_prepared_proposal import TracedDSparkDevice
from request_verifier_profile import PROFILER_FLAGS


class CombinedDeviceProfileTests(unittest.TestCase):
    def fixture(self, trace):
        operations = SimpleNamespace(execute_trace=Mock(), synchronize_device=Mock(), ReadDeviceProfiler=Mock())
        device = SimpleNamespace(operations=operations, mesh=object(), position=65536, max_drafts=15)
        device.prepared = SimpleNamespace(audit=False, closed=False, trace=trace, device=device,
            operations=operations, mesh=device.mesh)
        return device

    def test_profiles_only_first_real_replay_and_preserves_outputs(self):
        devices = [self.fixture(7), self.fixture(8)]
        records = []

        def propose(device, anchor, count):
            device.operations.execute_trace(device.mesh, device.prepared.trace, cq_id=0, blocking=True)
            return [anchor] * count

        factory = lambda operations, mesh: DraftProfile(operations, mesh, signpost=Mock())
        with patch.dict(os.environ, {name: '1' for name in PROFILER_FLAGS}), redirect_stdout(io.StringIO()), \
                patch.object(TracedDSparkDevice, 'propose', propose):
            with profile_device_scope(records, observer_type=factory):
                for device in devices:
                    for ordinal in range(4):
                        self.assertEqual(TracedDSparkDevice.propose(device, 3, 15), [3] * 15)
            self.assertIs(TracedDSparkDevice.propose, propose)
        self.assertEqual(len(records), 2)
        for device in devices:
            self.assertEqual(device.operations.execute_trace.call_count, 4)
            self.assertEqual(device.operations.ReadDeviceProfiler.call_count, 2)
        result = attach_device_profile({'arms': {'scatter': {'pp': 2000, 'committed_tg': 14}}}, records)
        self.assertIsNone(result['arms']['scatter']['committed_tg'])
        self.assertFalse(result['performance_qualified'])

    def test_missing_profiles_rejected(self):
        with self.assertRaises(ValueError):
            attach_device_profile({'arms': {}}, [])


if __name__ == '__main__':
    unittest.main()
