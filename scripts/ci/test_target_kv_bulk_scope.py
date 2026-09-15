import unittest

from target_kv_bulk_scope import qualified_callback


class BulkScopeTests(unittest.TestCase):
    def test_compares_full_and_partial_then_selects_faster(self):
        calls, records, emitted = [], [], []
        ticks = iter((0, 4, 4, 5, 5, 9, 9, 10, 10, 11))
        def original(valid):
            calls.append(('old', valid))
            return [valid]
        def candidate(valid):
            calls.append(('new', valid))
            return [valid]
        callback = qualified_callback(original, candidate, records, emitted.append, clock=lambda: next(ticks))
        self.assertEqual(callback(65536), [65536])
        self.assertEqual(callback(65547), [65547])
        self.assertEqual(calls, [('old', 65536), ('new', 65536),
            ('old', 65535), ('new', 65535), ('new', 65547)])
        self.assertTrue(records[-1]['candidate'])

    def test_mismatch_fails_closed(self):
        callback = qualified_callback(lambda valid: [valid], lambda valid: [0], [], lambda record: None)
        with self.assertRaises(AssertionError):
            callback(65536)

    def test_first_check_after_gold_decode_retains_requested_frontier(self):
        calls = []
        def operation(valid):
            calls.append(valid)
            return ['digest', valid]
        callback = qualified_callback(operation, operation, [], lambda record: None)
        self.assertEqual(callback(65552), ['digest', 65552])
        self.assertEqual(calls, [65536, 65536, 65535, 65535, 65552, 65552])
        self.assertEqual(callback(65536), ['digest', 65536])

    def test_slower_candidate_keeps_reference(self):
        records = []
        ticks = iter((0, 1, 1, 5, 5, 6, 6, 10, 10, 11))
        callback = qualified_callback(lambda valid: [valid], lambda valid: [valid],
            records, lambda record: None, clock=lambda: next(ticks))
        callback(65536)
        callback(65547)
        self.assertFalse(records[-1]['candidate'])

    def test_rejects_wrong_initial_frontier(self):
        callback = qualified_callback(None, None, [], lambda record: None)
        with self.assertRaises(ValueError):
            callback(4096)
