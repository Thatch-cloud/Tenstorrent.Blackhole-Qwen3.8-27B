import copy
import unittest

from full_model_fusion import paired_block, supported_decode


class FullModelFusionTests(unittest.TestCase):
    def test_only_explicit_b1_decode_is_selected(self):
        self.assertTrue(supported_decode([1, 1, 1, 5120], True))
        self.assertFalse(supported_decode([1, 1, 1, 5120], False))
        for shape in ([1, 1, 2, 5120], [1, 1, 32, 5120], [1, 1, 2048, 2560], [1, 5120]):
            self.assertFalse(supported_decode(shape, True))

    def records(self):
        return [dict(arm=arm, token_ids=[1, 2, 3], decode_seconds=elapsed)
                for arm, elapsed in zip(("control", "fused", "fused", "control"), (1, .9, .9, 1))]

    def test_paired_block_uses_decode_steps_not_prefill_token(self):
        result = paired_block(self.records())
        self.assertAlmostEqual(result["latency_change"], -.1)
        self.assertEqual(result["control_host_tok_s"], 2)

    def test_inexact_tokens_fail_before_timing_summary(self):
        for index in range(4):
            records = self.records()
            records[index]["token_ids"] = [1, 2, 4]
            with self.assertRaises(ValueError):
                paired_block(records)

    def test_invalid_timing_and_arm_order_fail(self):
        for elapsed in (0, -1, float("inf"), float("nan")):
            records = self.records()
            records[1]["decode_seconds"] = elapsed
            with self.assertRaises(ValueError):
                paired_block(records)
        records = copy.deepcopy(self.records())
        records[1]["arm"] = "control"
        with self.assertRaises(ValueError):
            paired_block(records)
