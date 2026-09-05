"""Measured arms must actually engage their claimed sampler."""

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location("sampling_client", Path(__file__).with_name("sampling-client.py"))
client = importlib.util.module_from_spec(spec)
spec.loader.exec_module(client)


class EngagementTests(unittest.TestCase):
    def check(self, selected=True, force=True, trace="all", expected=True):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            entry = dict(request_id="cmpl-test-label-uuid", is_decode=True, selected=selected,
                         force_argmax=force, trace_mode=trace)
            (root / "sampling-engagement.jsonl").write_text(json.dumps(entry) + "\n")
            return client.engagement(root, "test-label", expected)

    def test_device_requires_force_argmax_and_trace(self):
        self.assertTrue(self.check()["selected"])
        for options in (dict(selected=False), dict(force=False), dict(trace="none")):
            with self.assertRaises(AssertionError):
                self.check(**options)

    def test_host_fallback_is_not_a_device_measurement(self):
        self.assertFalse(self.check(selected=False, expected=False)["selected"])


if __name__ == "__main__":
    unittest.main()
