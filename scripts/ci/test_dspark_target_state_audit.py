from types import SimpleNamespace
import unittest
from unittest.mock import Mock

from dspark_target_state_audit import TargetStateAuditedDrafter, TargetStateChanged


class TargetStateAuditTests(unittest.TestCase):
    def test_unchanged_state_preserves_result(self):
        draft = SimpleNamespace(propose=Mock(return_value=(1, 2)))
        snapshot = Mock(side_effect=[['same'], ['same']])
        self.assertEqual(TargetStateAuditedDrafter(draft, snapshot).propose(3, 2), (1, 2))
        draft.propose.assert_called_once_with(3, 2)

    def test_changed_state_identifies_capture_and_replay(self):
        for phase in ('proposal_capture', 'proposal_replay'):
            draft = SimpleNamespace(propose=Mock(), prepare_trace=Mock())
            wrapper = TargetStateAuditedDrafter(draft, Mock(side_effect=[['before'], ['after']]))
            with self.assertRaises(TargetStateChanged) as failure:
                if phase == 'proposal_capture':
                    wrapper.prepare_trace(3, audit=True)
                else:
                    wrapper.propose(3, 2)
            self.assertEqual(failure.exception.evidence['phase'], phase)
