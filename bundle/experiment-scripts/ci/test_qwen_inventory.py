"""Test inventory orchestration with a fake Docker CLI, never a daemon."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

SCRIPT = Path(__file__).with_name("qwen-inventory.sh")
FIXTURE = Path(__file__).with_name("fixtures") / "docker"


class InventoryTests(unittest.TestCase):
    def run_inventory(self, **overrides):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            binaries = root / "bin"
            binaries.mkdir()
            for name in ("docker", "git"):
                binary = binaries / name
                shutil.copyfile(FIXTURE, binary)
                binary.chmod(0o755)
            log = root / "calls.jsonl"
            environment = dict(os.environ, PATH=str(binaries) + os.pathsep + os.environ["PATH"],
                               MOCK_DOCKER_LOG=str(log), **overrides)
            result = subprocess.run(["bash", str(SCRIPT)], cwd=root, env=environment,
                                    capture_output=True, text=True, timeout=15)
            calls = [json.loads(line) for line in log.read_text().splitlines()]
            return result, calls

    def test_daemon_probe_works_without_runner_device_tree(self):
        result, calls = self.run_inventory()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        creation = next(call["args"] for call in calls if call["args"][0] == "create")
        for required in ("--read-only", "--cap-drop", "--network", "--user", "--entrypoint"):
            self.assertIn(required, creation)
        self.assertNotIn("--privileged", creation)
        self.assertNotIn("--device", creation)
        mounts = [creation[index + 1] for index, value in enumerate(creation) if value == "--mount"]
        self.assertTrue(all(value.endswith(",readonly") for value in mounts))
        self.assertEqual([call["args"] for call in calls if call["args"][0] == "rm"], [["rm", "-f", "probe-owned-id"]])
        start = next(call for call in calls if call["args"][0] == "start")
        self.assertIn("/host-sys/bus/pci/devices", start["stdin"])
        self.assertIn("idleness is UNVERIFIED", start["stdin"])

    def test_failure_cleans_only_owned_probe(self):
        result, calls = self.run_inventory(MOCK_PROBE_FAIL="1")
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(calls[-1]["args"], ["rm", "-f", "probe-owned-id"])

    def test_missing_image_does_not_pull_or_create(self):
        result, calls = self.run_inventory(MOCK_NO_IMAGE="1")
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse(any(call["args"][0] in ("pull", "create", "start", "rm") for call in calls))

    def test_create_failure_does_not_remove_another_container(self):
        result, calls = self.run_inventory(MOCK_CREATE_FAIL="1")
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse(any(call["args"][0] in ("start", "rm") for call in calls))


if __name__ == "__main__":
    unittest.main(verbosity=2)
