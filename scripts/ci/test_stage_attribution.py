import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from stage_profile import StageProfile, direct_calls, gdn_namespace
from full_batch_attribution import aggregate


class StageAttributionTests(unittest.TestCase):
    def test_gdn_namespace_preserves_globals_arguments_and_returns(self):
        conv = Mock(return_value=("conv", "beta", "gate"))
        recurrence = Mock(return_value=("output", "state"))
        transformer = SimpleNamespace(gdn_decode_conv_gates=conv, sentinel=object())
        operations = SimpleNamespace(transformer=transformer, sentinel=object())
        namespace = dict(ttnn=operations, recurrent_gated_delta_rule_decode_packed_ttnn=recurrence)
        profiler = StageProfile(lambda: None)
        local = gdn_namespace(namespace, profiler)
        profiler.begin()
        self.assertEqual(local["ttnn"].transformer.gdn_decode_conv_gates("input", batch=1), ("conv", "beta", "gate"))
        self.assertEqual(local["recurrent_gated_delta_rule_decode_packed_ttnn"]("input", inplace_state=False), ("output", "state"))
        self.assertEqual({record["category"] for record in profiler.finish()}, {"gdn.conv_gates", "gdn.recurrence_norm_gate"})
        conv.assert_called_once_with("input", batch=1)
        recurrence.assert_called_once_with("input", inplace_state=False)
        self.assertIs(namespace["ttnn"], operations)
        self.assertIs(operations.transformer, transformer)
        self.assertIs(transformer.gdn_decode_conv_gates, conv)
        self.assertIs(namespace["recurrent_gated_delta_rule_decode_packed_ttnn"], recurrence)
        self.assertIs(local["ttnn"].sentinel, operations.sentinel)
        self.assertIs(local["ttnn"].transformer.sentinel, transformer.sentinel)

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
