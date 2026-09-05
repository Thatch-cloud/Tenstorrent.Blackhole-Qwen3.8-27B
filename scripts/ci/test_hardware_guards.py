"""Hardware entry points must reject implicit execution, even without TTNN installed."""

import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


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

    def test_prefill_hardware_gate_requires_explicit_authorization(self):
        script = Path(__file__).resolve().parents[2] / "optimisation/sim/prefill-state.py"
        self.assertIn("Hardware requires explicit authorization", self.run_guard(script, ["--hardware"]))

    def test_prefill_default_cannot_fall_back_to_hardware(self):
        script = Path(__file__).resolve().parents[2] / "optimisation/sim/prefill-state.py"
        self.assertIn("Simulator required", self.run_guard(script, [], TT_METAL_SIMULATOR=""))


if __name__ == "__main__":
    unittest.main(verbosity=2)
