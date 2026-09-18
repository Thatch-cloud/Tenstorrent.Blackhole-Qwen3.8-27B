"""Host-only tests for the Phase-0 identity capture script."""

import json
import os
import tempfile
import unittest
from pathlib import Path

from fabric_relay_phase0_identity import capture, main, resolve_by_id


class ResolveTests(unittest.TestCase):
    def test_maps_blackhole_serials_only(self):
        import tempfile as tf
        with tf.TemporaryDirectory() as root:
            for name in ("blackhole-K123", "blackhole-K456", "unrelated"):
                open(os.path.join(root, name), "w").close()
            mapping, present = resolve_by_id(root)
        self.assertTrue(present)
        self.assertEqual(sorted(mapping), ["blackhole-K123", "blackhole-K456"])

    def test_missing_dir_reports_absent(self):
        mapping, present = resolve_by_id("/nonexistent/path/xyz")
        self.assertFalse(present)
        self.assertEqual(mapping, {})


class CaptureTests(unittest.TestCase):
    def test_capture_shapes_snapshot_without_devices(self):
        proc = {
            "/proc/meminfo": "MemTotal  100 GB\n",
            "/proc/pressure/cpu": "some avg10=1.0",
            "/proc/pressure/memory": "full avg10=0.0",
            "/proc/pressure/io": "some avg10=2.0",
        }

        def read_text(path):
            return proc[path]

        def run_command(cmd, timeout=0):
            class Result:
                returncode = 0
                stdout = "board state ok"
            return Result()

        snapshot = capture(read_text=read_text, run_command=run_command)
        self.assertEqual(snapshot["schema"], "fabric-relay-phase0-identity-v1")
        self.assertIn("MemTotal", snapshot["meminfo_total_line"])
        self.assertEqual(snapshot["host_pressure"]["cpu"], "some avg10=1.0")
        self.assertEqual(snapshot["tt_smi_state"]["returncode"], 0)

    def test_missing_smi_is_captured_not_fatal(self):
        snapshot = capture(read_text=lambda p: "", run_command=self._raise_oserror)
        self.assertIn("error", snapshot["tt_smi_state"])

    @staticmethod
    def _raise_oserror(cmd, timeout=0):
        raise OSError("no such file")


class CliTests(unittest.TestCase):
    def test_main_writes_artifact_with_sha256(self):
        with tempfile.TemporaryDirectory() as root:
            out = Path(root) / "identity.json"
            main(["--out", str(out)])
            artifact = json.loads(out.read_text(encoding="utf-8"))
        self.assertEqual(artifact["artifact_path"], str(out))
        self.assertRegex(artifact["artifact_sha256"], r"^[0-9a-f]{64}$")


if __name__ == "__main__":
    unittest.main()
