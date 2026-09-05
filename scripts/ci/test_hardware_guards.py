"""Hardware entry points must reject implicit execution, even without TTNN installed."""

import os
import importlib.util
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock
import stat
from types import SimpleNamespace


class HardwareGuardTests(unittest.TestCase):
    def run_guard(self, script, arguments, **environment):
        with tempfile.TemporaryDirectory() as directory:
            env = dict(os.environ, TT_METAL_HOME=directory, **environment)
            env.pop("QWEN_HARDWARE_TESTS", None)
            result = subprocess.run([sys.executable, "-B", str(script), *arguments], env=env,
                                    text=True, capture_output=True, timeout=15)
            self.assertNotEqual(result.returncode, 0)
            self.assertNotIn("No module named 'ttnn'", result.stderr)
            return result.stderr

    def test_hardware_gate_requires_explicit_authorization(self):
        error = self.run_guard(Path(__file__).with_name("hardware-correctness.py"),
                               ["--suite", "audit", "--output", "/tmp/unused-qwen-guard.json"])
        self.assertIn("Explicit hardware authorization", error)

    def test_gdn_prefix_gate_requires_explicit_authorization(self):
        error = self.run_guard(Path(__file__).with_name("gdn-prefix.py"), [])
        self.assertIn("Explicit hardware allocation", error)

    def test_prefill_hardware_gate_requires_explicit_authorization(self):
        script = Path(__file__).resolve().parents[2] / "optimisation/sim/prefill-state.py"
        self.assertIn("Hardware requires explicit authorization", self.run_guard(script, ["--hardware"]))

    def test_prefill_default_cannot_fall_back_to_hardware(self):
        script = Path(__file__).resolve().parents[2] / "optimisation/sim/prefill-state.py"
        self.assertIn("Simulator required", self.run_guard(script, [], TT_METAL_SIMULATOR=""))

    def allocation_probe(self, device_count=2):
        spec = importlib.util.spec_from_file_location("device_owners", Path(__file__).with_name("device-owners.py"))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        nodes = []
        for index, name in enumerate(("0", "2")[:device_count]):
            node = mock.Mock()
            node.name = name
            node.stat.return_value = SimpleNamespace(st_mode=stat.S_IFCHR, st_rdev=index)
            nodes.append(node)
        def path_factory(value):
            path = mock.MagicMock()
            if value == "/host-dev/tenstorrent":
                path.iterdir.return_value = nodes
            elif value == "/host-dev/tenstorrent/by-id":
                path.__truediv__.side_effect = lambda board: SimpleNamespace(
                    resolve=lambda: SimpleNamespace(name={"blackhole-3707293C249A5E67": "0",
                                                          "blackhole-CEF5729692C19E6D": "2"}[board]))
            else:
                raise AssertionError(f"Unexpected host access: {value}")
            return path
        with mock.patch.object(module, "Path", side_effect=path_factory), \
                mock.patch.dict(os.environ, QWEN_CARDS_ALLOCATED="1"), mock.patch("builtins.print"):
            module.main()

    def test_operator_allocation_does_not_scan_host_processes(self):
        self.allocation_probe()

    def test_operator_allocation_still_requires_both_boards(self):
        with self.assertRaisesRegex(RuntimeError, "Expected two card device nodes"):
            self.allocation_probe(device_count=1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
