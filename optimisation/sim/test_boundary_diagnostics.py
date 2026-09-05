"""Diagnostic hooks must not touch device tensors or alter host state."""

import os
import ast
import importlib.util
from pathlib import Path
import subprocess
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

import boundary_diagnostics as diagnostics


class DiagnosticsTests(unittest.TestCase):
    def test_staging_transforms_pinned_originals_without_editing_checkout(self):
        root = Path(os.environ.get("SIM_ROOT", "/opt/ttsim"))
        for script, checkout, prefix, files in (
            ("stage-continuation.py", "tt-metal", "models/demos/blackhole/qwen36/tt/", ["model.py", "qwen36_vllm.py"]),
            ("stage-plugin.py", "vllm-tt-plugin", "src/vllm_tt_plugin/", ["model_runner.py", "model_input.py", "scheduler.py", "platform.py"]),
        ):
            spec = importlib.util.spec_from_file_location("stager", Path(__file__).with_name(script))
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            for name in files:
                original = subprocess.check_output(["git", "-C", str(root / checkout), "show", f"HEAD:{prefix}{name}"], text=True)
                updated = module.transform(name, original)
                ast.parse(updated)
                self.assertNotEqual(updated, original)

    def test_fingerprint_preserves_bfloat16_and_shape(self):
        tensor = torch.arange(8).to(torch.bfloat16).reshape(2, 4)
        previous = tensor.clone()
        report = diagnostics.fingerprint(tensor)
        self.assertEqual(report["dtype"], "torch.bfloat16")
        self.assertEqual(report["shape"], [2, 4])
        self.assertTrue(torch.equal(tensor, previous))
        self.assertNotEqual(report["sha256"], diagnostics.fingerprint(tensor + 1)["sha256"])

    def test_disabled_input_never_inspects_model(self):
        with patch.dict(os.environ, QWEN_BOUNDARY_DIAGNOSTICS="0"):
            diagnostics.record_input(None, None)
            diagnostics.record_slot(0, None, None)
            diagnostics.record_output(None)
        self.assertIsNone(diagnostics.CONTEXT)

    def test_only_first_three_decode_steps_recorded(self):
        inputs = SimpleNamespace(prompt_lens=None, input_positions=torch.tensor([2049]),
                                 input_tokens=torch.tensor([[74830]]))
        with patch.dict(os.environ, QWEN_BOUNDARY_DIAGNOSTICS="1"), \
                patch.object(diagnostics, "DECODE_COUNTS", {}), patch.object(diagnostics, "emit") as emit:
            for step in range(5):
                diagnostics.record_input(inputs, ["test"])
                diagnostics.record_output(torch.tensor([[1., 2., 3.]]))
            self.assertEqual(emit.call_count, 6)


if __name__ == "__main__":
    unittest.main()
