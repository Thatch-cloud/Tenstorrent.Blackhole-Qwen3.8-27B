import unittest
from copy import deepcopy
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import patch

import matched_combined_request as candidate
from matched_combined_request import validate_result


class MatchedCombinedRequestTests(unittest.TestCase):
    def fixture(self):
        return (dict(passed=True, correctness_screen_passed=True, closed_cleanly=True,
            request_checks=[dict(instrumented_timing=True, blocks=[{}, {}])]),
            [dict(prepared=3, committed=2, discarded=1, max_touched_rows=64, failed=False, restored=True)],
            [dict(committed=False)])

    def test_complete_audit(self):
        validate_result(*self.fixture())

    def test_missing_commit_or_warmup_rejected(self):
        for field, value in (('committed', 1), ('prepared', 2), ('discarded', 0),
                ('max_touched_rows', 65536), ('failed', True), ('restored', False)):
            with self.subTest(field=field):
                report, updates, warmups = self.fixture()
                updates[0][field] = value
                with self.assertRaises(ValueError):
                    validate_result(report, updates, warmups)

    def test_incomplete_or_clean_timing_is_not_an_audit(self):
        original, updates, warmups = self.fixture()
        for field in ('passed', 'correctness_screen_passed', 'closed_cleanly'):
            report = deepcopy(original)
            report[field] = False
            with self.assertRaises(ValueError):
                validate_result(report, updates, warmups)
        original['request_checks'][0]['instrumented_timing'] = False
        with self.assertRaises(ValueError):
            validate_result(original, updates, warmups)

    def test_writer_warmup_discards_and_restores_constructor(self):
        events = []

        class Arm:
            def __init__(self, history, *, audit=False):
                events.append(('initialize', audit))
                self.projection = SimpleNamespace(outputs='outputs', close=lambda: events.append('close'))

        constructor = Arm.__init__
        history = SimpleNamespace(position=65536,
            prepare_projected=lambda outputs, rows, *, position: events.append((outputs, rows, position)) or 'pending',
            discard_publication=lambda publication: events.append(('discard', publication)))
        module = SimpleNamespace(CapturedPublicationArm=Arm)
        warmups = []
        with patch.object(candidate, 'incremental_history', return_value=nullcontext()):
            with candidate.publication_scope(object, module, [], warmups):
                Arm(history, audit=True)
                with self.assertRaises(ValueError):
                    Arm(history, audit=False)
        self.assertIs(Arm.__init__, constructor)
        self.assertEqual(events, [('initialize', True), ('outputs', 32, 65536), ('discard', 'pending')])
        self.assertEqual(len(warmups), 1)
        self.assertFalse(warmups[0]['committed'])


if __name__ == '__main__':
    unittest.main()
