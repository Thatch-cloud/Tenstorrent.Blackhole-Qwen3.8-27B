import unittest
from copy import deepcopy
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import patch

import matched_combined_request as candidate
from matched_cached_build import cached_only
from matched_combined_request import validate_result


class MatchedCombinedRequestTests(unittest.TestCase):
    def test_original_full_kv_callback_is_not_replaced(self):
        callback = lambda valid: valid
        arguments = dict(audit_features=True, kv_digest=callback)
        records = []
        candidate.retain_legacy_kv(arguments, records, lambda record: None)
        self.assertIs(arguments['kv_digest'], callback)
        self.assertTrue(records[0]['full_prefix_checks'])
        self.assertFalse(records[0]['reader_comparison'])

    def test_missing_cache_fails_without_compilation(self):
        original = lambda cache, inputs: None
        module = SimpleNamespace(inspect_entry=original)
        with cached_only(module):
            with self.assertRaisesRegex(ValueError, 'prepare the build separately'):
                module.inspect_entry('cache', {})
        self.assertIs(module.inspect_entry, original)

    def test_cache_identity_validation_is_not_bypassed(self):
        def invalid(cache, inputs):
            raise ValueError('binary hash changed')

        module = SimpleNamespace(inspect_entry=invalid)
        with cached_only(module):
            with self.assertRaisesRegex(ValueError, 'binary hash changed'):
                module.inspect_entry('cache', {})
        manifest = dict(inputs={'source': 'exact'}, binary_sha256='binary')
        module.inspect_entry = lambda cache, inputs: manifest
        with cached_only(module):
            self.assertIs(module.inspect_entry('cache', manifest['inputs']), manifest)

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
