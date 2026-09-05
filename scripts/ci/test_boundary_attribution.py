"""Diagnostic comparisons distinguish state changes from token changes."""

import copy
import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location("attribution", Path(__file__).with_name("compare-boundary-diagnostics.py"))
attribution = importlib.util.module_from_spec(spec)
spec.loader.exec_module(attribution)


class AttributionTests(unittest.TestCase):
    def records(self):
        return [[dict(kind="input", phase="prefill", tokens={"sha256": "prompt"}, prompt_lens=[2049]),
                 dict(kind="slot", recurrent=["same", "changed"], convolution=[["same"]]),
                 dict(kind="output", phase="prefill", decode_step=0, logits="same", top_ids=[7], top_values=[1.])]]

    def test_identical_records(self):
        results = attribution.compare(self.records(), self.records())
        self.assertEqual(results[0]["changed_state_layers"], dict(recurrent=[], convolution=[]))
        self.assertTrue(results[0]["outputs"][0]["equal"])

    def test_layer_difference_localized(self):
        control = self.records()
        arm = copy.deepcopy(control)
        arm[0][1]["recurrent"][1] = "different"
        result = attribution.compare(control, arm)[0]
        self.assertEqual(result["changed_state_layers"]["recurrent"], [1])

    def test_unpaired_or_different_prompts_rejected(self):
        with self.assertRaises(ValueError):
            attribution.compare(self.records(), [])
        arm = self.records()
        arm[0][0]["tokens"] = {"sha256": "other"}
        with self.assertRaises(ValueError):
            attribution.compare(self.records(), arm)


if __name__ == "__main__":
    unittest.main()
