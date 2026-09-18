from contextlib import contextmanager
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, Mock, patch

from serving_gather_experiment import ARMS, GatherExperiment, from_environment


class GatherExperimentTests(unittest.TestCase):
    def test_default_is_disabled(self):
        with patch.dict('os.environ', {}, clear=True):
            self.assertIsNone(from_environment('.', '.'))

    def test_warmups_then_abba_retains_scope_until_request_close(self):
        events = []

        @contextmanager
        def scope(admission):
            audit = dict(restored=False, kernels=[admission])
            events.append('enter')
            try:
                yield audit
            finally:
                events.append('restore')
                audit['restored'] = True

        experiment = GatherExperiment('admitted', scope)
        with patch('builtins.print') as output:
            for ordinal, arm in enumerate(ARMS):
                events.clear()
                request = experiment.create(lambda: SimpleNamespace(close=Mock(side_effect=lambda owner: events.append('close'))))
                self.assertEqual(experiment.ordinal, ordinal)
                with self.assertRaises(ValueError):
                    experiment.create(Mock())
                request.close('owner')
                self.assertEqual(events, ['enter', 'close', 'restore'] if arm == 'grouped' else ['close'])
                self.assertFalse(experiment.active)
                request.close('owner')
            self.assertEqual(output.call_count, 6)
        with self.assertRaises(ValueError):
            experiment.create(Mock())

    def test_failed_close_does_not_restore_live_trace_scope(self):
        scope = MagicMock()
        experiment = GatherExperiment({}, scope)
        experiment.ordinal = 1
        context = scope.return_value
        request = experiment.create(lambda: SimpleNamespace(close=Mock(side_effect=RuntimeError('live trace'))))
        with self.assertRaises(RuntimeError):
            request.close('owner')
        context.__exit__.assert_not_called()
        self.assertTrue(experiment.active)
