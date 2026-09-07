import unittest
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import Mock

from full_replay import validate_fixture, warm_feature_fixture


class FullReplayTests(unittest.TestCase):
    def test_feature_copy_warmup_runs_outside_trace_and_releases(self):
        events = []

        @contextmanager
        def capture():
            events.append('hook-enter')
            yield
            events.append('hook-exit')

        output = object()
        fixture = SimpleNamespace(run=lambda: events.append('forward') or output,
                                  close=lambda: events.append('fixture-close'))
        features = SimpleNamespace(capture=capture, close=lambda: events.append('features-close'))
        operations = SimpleNamespace(synchronize_device=lambda mesh: events.append('synchronize'),
            deallocate=lambda value: events.append('output-free') if value is output else self.fail('Wrong output'))
        warm_feature_fixture(fixture, features, operations, object())
        self.assertEqual(events, ['hook-enter', 'forward', 'hook-exit', 'synchronize',
                                  'features-close', 'output-free', 'fixture-close'])

    def test_failed_warmup_closes_features_and_fixture(self):
        @contextmanager
        def capture():
            yield

        fixture = SimpleNamespace(run=Mock(side_effect=RuntimeError('warmup failed')), close=Mock())
        features = SimpleNamespace(capture=capture, close=Mock())
        operations = SimpleNamespace(synchronize_device=Mock(), deallocate=Mock())
        with self.assertRaisesRegex(RuntimeError, 'warmup failed'):
            warm_feature_fixture(fixture, features, operations, object())
        fixture.close.assert_called_once()
        features.close.assert_called_once()
        operations.deallocate.assert_not_called()

    def test_all_replay_shapes_include_both_correction_tokens(self):
        for rows in (2, 16, 32):
            for first in (0, 1, rows):
                for second in (1, rows):
                    validate_fixture(rows, first, second, first + rows + 3)
                    with self.assertRaises(ValueError):
                        validate_fixture(rows, first, second, first + rows + 2)

    def test_rejects_unsupported_or_ambiguous_replay_geometry(self):
        for values in ((64, 1, 1, 100), (2, -1, 1, 100), (2, 1, 0, 100),
                       (16, 8, 1, 100), (16, 1, True, 100), (2.0, 1, 1, 100)):
            with self.assertRaises(ValueError):
                validate_fixture(*values)
