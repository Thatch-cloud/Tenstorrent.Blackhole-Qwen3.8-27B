import unittest

from stage_profile import StageProfile, direct_calls
from full_batch_attribution import aggregate


class StageProfileTests(unittest.TestCase):
    def test_aggregation_reconciles_exclusive_categories(self):
        records = [dict(category="model.block", layer=None, calls=1, inclusive_ms=10, exclusive_ms=1),
                   dict(category="row", layer=0, calls=2, inclusive_ms=4, exclusive_ms=4),
                   dict(category="row", layer=1, calls=2, inclusive_ms=5, exclusive_ms=5)]
        totals = aggregate(records)
        self.assertEqual(totals[0]["category"], "row")
        self.assertEqual(totals[0]["calls"], 4)
        records[0]["exclusive_ms"] = 2
        with self.assertRaises(ValueError):
            aggregate(records)

    def test_nested_exclusive_time_does_not_double_count(self):
        times = iter((0.0, 1.0, 3.0, 5.0))
        fences = []
        profiler = StageProfile(lambda: fences.append(True), lambda: next(times))
        profiler.begin()
        with profiler.scope("outer", 7):
            with profiler.scope("inner"):
                pass
        records = {record["category"]: record for record in profiler.finish()}
        self.assertEqual(records["outer"]["inclusive_ms"], 5000)
        self.assertEqual(records["outer"]["exclusive_ms"], 3000)
        self.assertEqual(records["inner"]["exclusive_ms"], 2000)
        self.assertEqual(records["inner"]["layer"], 7)
        self.assertEqual(len(fences), 4)

    def test_disabled_wrapper_adds_no_fences_or_records(self):
        profiler = StageProfile(lambda: self.fail("Unexpected fence"))
        self.assertEqual(profiler.wrap("disabled", lambda value: value + 1)(2), 3)
        self.assertEqual(profiler.records, {})

    def test_exception_unwinds_scope(self):
        times = iter((0.0, 1.0))
        profiler = StageProfile(lambda: None, lambda: next(times))
        profiler.begin()
        with self.assertRaises(ValueError):
            with profiler.scope("failure"):
                raise ValueError("device")
        self.assertEqual(profiler.stack, [])
        self.assertEqual(profiler.finish()[0]["calls"], 1)

    def test_direct_decoder_calls_exclude_attention_methods(self):
        source = '''def forward(self, value):
    hidden = self.attention_norm(value)
    hidden = self.attention.forward_decode(hidden)
    return self.feed_forward(self.ffn_norm(hidden))
'''
        self.assertEqual(direct_calls(source), ["attention_norm", "feed_forward", "ffn_norm"])


if __name__ == "__main__":
    unittest.main()
