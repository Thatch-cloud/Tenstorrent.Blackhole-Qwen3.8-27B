"""TP2 greedy-only admission cannot bypass original sampling constraints."""

import ast
import importlib.util
import os
from pathlib import Path
import subprocess
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

import sampling_experiment as experiment


class SamplingExperimentTests(unittest.TestCase):
    def setUp(self):
        environment = patch.dict(os.environ, QWEN_TP2_SAMPLING_EXPERIMENT="1")
        environment.start()
        self.addCleanup(environment.stop)

    def runner(self, arm="device", temperature=0., penalties=False):
        return SimpleNamespace(input_batch=SimpleNamespace(req_ids=[f"cmpl-qwen-sampling-{arm}-test", None],
            sampling=SimpleNamespace(temperature=torch.tensor([temperature])), no_penalties=not penalties))

    def test_only_reviewed_topology(self):
        self.assertTrue(experiment.enable_tp2((1, 2), 248320))
        for shape, vocab in (((1, 4), 248320), ((1, 2), 100)):
            with self.assertRaises(ValueError):
                experiment.enable_tp2(shape, vocab)

    def test_flag_off_preserves_original(self):
        with patch.dict(os.environ, QWEN_TP2_SAMPLING_EXPERIMENT="0"):
            self.assertTrue(experiment.select_sampling(None, True, True))
            self.assertFalse(experiment.select_sampling(None, False, True))
            self.assertFalse(experiment.enable_tp2((1, 4), 100))
            experiment.require_greedy(False)

    def test_original_constraints_cannot_be_overridden(self):
        self.assertFalse(experiment.select_sampling(self.runner(), False, True))
        self.assertTrue(experiment.select_sampling(self.runner(), True, True))
        self.assertFalse(experiment.select_sampling(self.runner(), True, False))

    def test_non_greedy_penalties_and_host_labels_stay_host(self):
        for runner in (self.runner(temperature=.8), self.runner(penalties=True), self.runner(arm="host"), self.runner(arm="unknown")):
            self.assertFalse(experiment.select_sampling(runner, True, True))

    def test_generic_topk_cannot_execute(self):
        with self.assertRaises(RuntimeError):
            experiment.require_greedy(False)
        experiment.require_greedy(True)

    def test_multiple_requests_fall_back(self):
        runner = self.runner()
        runner.input_batch.req_ids = ["qwen-sampling-device-one", "qwen-sampling-device-two"]
        self.assertFalse(experiment.select_sampling(runner, True, True))

    def test_transforms_pinned_sources_and_preserves_original_gate(self):
        spec = importlib.util.spec_from_file_location("stage_sampling", Path(__file__).with_name("stage-sampling.py"))
        stager = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(stager)
        root = Path(os.environ.get("SIM_ROOT", "/opt/ttsim"))
        for checkout, path in (("tt-metal", "models/demos/blackhole/qwen36/tt/model.py"),
                               ("tt-metal", "models/common/sampling/tt_sampling.py"),
                               ("vllm-tt-plugin", "src/vllm_tt_plugin/model_runner.py")):
            source = subprocess.check_output(["git", "-C", str(root / checkout), "show", f"HEAD:{path}"], text=True)
            transformed = stager.transform(Path(path).name, source)
            tree = ast.parse(transformed)
            if path.endswith("model_runner.py"):
                original = next(node for node in ast.walk(ast.parse(source))
                                if isinstance(node, ast.FunctionDef) and node.name == "check_perform_device_sampling")
                retained = next(node for node in ast.walk(tree)
                                if isinstance(node, ast.FunctionDef) and node.name == "_qwen_original_sampling_check")
                retained.name = original.name
                self.assertEqual(ast.dump(original), ast.dump(retained))


if __name__ == "__main__":
    unittest.main()
