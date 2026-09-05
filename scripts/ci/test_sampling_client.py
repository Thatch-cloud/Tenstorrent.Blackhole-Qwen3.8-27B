"""Measured arms must actually engage their claimed sampler."""

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

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

    def test_fallback_matrix_checks_paths_and_exact_output(self):
        with mock.patch.object(client, "measure", return_value=dict(token_ids_sha256="ids", output_sha256="text")) as measure:
            checks = client.fallback_checks([1], Path("unused"))
            self.assertEqual(len(checks), 7)
            self.assertEqual(measure.call_count, 14)
            self.assertEqual([call.args[4] for call in measure.call_args_list],
                             [False, True, False, True, False, False, False, False, False, False, False, False, False, True])
            self.assertEqual(measure.call_args_list[0].kwargs["seed"], 123)
            self.assertEqual(measure.call_args_list[2].kwargs["seed"], 456)
        with mock.patch.object(client, "measure", side_effect=[
                dict(token_ids_sha256="one", output_sha256="text"),
                dict(token_ids_sha256="two", output_sha256="text")]):
            with self.assertRaises(AssertionError):
                client.fallback_checks([1], Path("unused"))

    def test_measure_preserves_explicit_seed(self):
        with mock.patch.object(client.baseline, "request_stream", return_value={}) as request, \
                mock.patch.object(client, "engagement"):
            client.measure([1], 32, "label", Path("unused"), True, seed=456)
            self.assertEqual(request.call_args.kwargs["seed"], 456)


if __name__ == "__main__":
    unittest.main()
