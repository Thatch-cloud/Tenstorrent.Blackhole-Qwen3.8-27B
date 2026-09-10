from types import SimpleNamespace
import unittest
from unittest.mock import Mock

from dspark_target_state_audit import TargetStateAuditedDrafter, TargetStateChanged


class TargetStateAuditTests(unittest.TestCase):
    def test_saved_verifier_corruption_detected_even_when_live_state_is_unchanged(self):
        draft = SimpleNamespace(propose=Mock(return_value=(1, 2)))
        wrapper = TargetStateAuditedDrafter(draft, Mock(return_value=['same']),
            protected_snapshot=Mock(side_effect=[['saved'], ['overwritten']]))
        with self.assertRaises(TargetStateChanged) as failure:
            wrapper.propose(3, 2)
        self.assertEqual(failure.exception.evidence['phase'], 'proposal_replay_protected_verifier_storage')
        self.assertEqual(failure.exception.evidence['before'], ['saved'])
        self.assertEqual(failure.exception.evidence['after'], ['overwritten'])

    def test_unbound_verifier_has_no_protected_storage(self):
        draft = SimpleNamespace(propose=Mock(return_value=(1, 2)))
        wrapper = TargetStateAuditedDrafter(draft, Mock(return_value=['same']),
            protected_snapshot=Mock(return_value=None))
        self.assertEqual(wrapper.propose(3, 2), (1, 2))

    def test_verifier_setup_corruption_is_detected_before_replay(self):
        draft = SimpleNamespace(position=4096, prepare_trace=Mock(), propose=Mock())
        wrapper = TargetStateAuditedDrafter(draft,
            Mock(side_effect=[['initial'], ['initial'], ['initial'], ['changed']]))
        wrapper.prepare_trace(3, audit=True)
        with self.assertRaises(TargetStateChanged) as failure:
            wrapper.propose(3, 15)
        self.assertEqual(failure.exception.evidence['phase'], 'before_proposal_at_initial_frontier')
        draft.propose.assert_not_called()

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
