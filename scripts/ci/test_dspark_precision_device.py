from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from dspark_precision_device import device_type


class PrecisionDeviceTests(unittest.TestCase):
    def fixture(self, **overrides):
        backend = Mock()
        events = []

        class Base:
            def __init__(self, **options):
                self.closed, self.prepared = False, None
                self.proposal_layer, self.max_drafts = backend, 15
                self.__dict__.update(overrides)
                events.append('history-built-before-policy')

            def propose(self, anchor, count):
                return self.proposal_layer('operations', anchor, count)

            def close(self):
                if self.prepared is not None:
                    self.prepared.close()
                self.closed = True
                events.append('closed')

        return Base, backend, events

    def test_only_candidate_instance_changes_and_trace_owns_teardown(self):
        base, backend, events = self.fixture()
        candidate = device_type(base, backend)(native_attention=True)
        control = base(native_attention=True)
        self.assertIs(control.proposal_layer, backend)
        self.assertEqual(candidate.precision_layer_calls, 0)
        with self.assertRaises(ValueError):
            candidate.propose(10, 15)
        candidate.prepared = SimpleNamespace(close=Mock())
        with patch('dspark_precision_device.execute', return_value='result') as selected:
            self.assertEqual(candidate.propose(10, 15), 'result')
            selected.assert_called_once_with(backend, 'operations', 10, 15)
        self.assertEqual(candidate.precision_layer_calls, 1)
        candidate.close()
        candidate.prepared.close.assert_called_once()
        with self.assertRaises(ValueError):
            candidate.proposal_layer('operations')
        self.assertFalse(control.closed)
        self.assertEqual(events[-1], 'closed')

    def test_rejects_wrong_policy_before_allocation(self):
        base, backend, events = self.fixture()
        with self.assertRaises(ValueError):
            device_type(base, backend)(native_attention=False)
        self.assertEqual(events, [])

    def test_invalid_initialized_device_is_closed(self):
        for overrides in (dict(max_drafts=31), dict(proposal_layer=Mock()),
                dict(prepared=SimpleNamespace(close=Mock()))):
            base, backend, events = self.fixture(**overrides)
            with self.assertRaises(ValueError):
                device_type(base, backend)(native_attention=True)
            self.assertEqual(events[-1], 'closed')

    def test_failed_layer_does_not_count_as_success(self):
        base, backend, events = self.fixture()
        candidate = device_type(base, backend)(native_attention=True)
        with patch('dspark_precision_device.execute', side_effect=RuntimeError('device failure')):
            with self.assertRaises(RuntimeError):
                candidate.proposal_layer('operations')
        self.assertEqual(candidate.precision_layer_calls, 0)
        candidate.close()
