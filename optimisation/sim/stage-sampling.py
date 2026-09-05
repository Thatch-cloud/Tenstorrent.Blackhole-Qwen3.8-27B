"""Stage only a pinned TP2 greedy experiment in a disposable container."""

import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess

spec = importlib.util.spec_from_file_location("stage", Path(__file__).with_name("stage-continuation.py"))
stage = importlib.util.module_from_spec(spec)
spec.loader.exec_module(stage)


def transform(name, source):
    if name == "model.py":
        return stage.patch_method(source, "__init__", ((
            "        self._supports_on_device_sampling = (\n",
            "        from qwen_sampling_experiment import enable_tp2\n"
            "        self._qwen_tp2_greedy_only = enable_tp2(mesh_shape, args.vocab_size)\n"
            "        self._supports_on_device_sampling = self._qwen_tp2_greedy_only or (\n",
        ), (
            "            args.pad_logits_to_power_of_2 = True\n",
            "            args.pad_logits_to_power_of_2 = not self._qwen_tp2_greedy_only\n",
        )))
    if name == "tt_sampling.py":
        return stage.patch_method(source, "forward", ((
            "        if self._force_argmax_sampling:\n",
            "        from qwen_sampling_experiment import require_greedy\n"
            "        require_greedy(self._force_argmax_sampling)\n"
            "        if self._force_argmax_sampling:\n",
        ),))
    if name == "model_runner.py":
        source = stage.patch_method(source, "check_perform_device_sampling", ((
            "    def check_perform_device_sampling(\n", "    def _qwen_original_sampling_check(\n",
        ), (
            "        return True\n",
            "        return True\n\n"
            "    def check_perform_device_sampling(self, is_decode, has_structured_outputs):\n"
            "        from qwen_sampling_experiment import select_sampling\n"
            "        original = self._qwen_original_sampling_check(is_decode, has_structured_outputs)\n"
            "        return select_sampling(self, original, is_decode)\n",
        )))
        return stage.patch_method(source, "_forward_with_model_input", ((
            "        return _SyncForward(\n",
            "        from qwen_sampling_experiment import record_execution\n"
            "        record_execution(self, model_input)\n"
            "        return _SyncForward(\n",
        ),))
    raise ValueError(name)


def main():
    pending = {}
    revisions = {}
    for root, revision, paths in (
        (Path("/opt/tt-metal"), stage.REVISION, ("models/demos/blackhole/qwen36/tt/model.py", "models/common/sampling/tt_sampling.py")),
        (Path("/opt/vllm-tt-plugin"), "bf77cd63756fc891b8fb7f7cb3f5c1420f0e044c", ("src/vllm_tt_plugin/model_runner.py",)),
    ):
        actual = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
        if actual != revision:
            raise SystemExit(f"Unreviewed source revision: {actual}")
        revisions[str(root)] = actual
        for relative in paths:
            original = subprocess.check_output(["git", "-C", str(root), "show", f"HEAD:{relative}"], text=True)
            path = root / relative
            updated = transform(path.name, original)
            ast.parse(updated)
            if path.read_text() not in (original, updated):
                raise SystemExit(f"Refusing unrelated changes: {path}")
            pending[path] = updated
    helper = Path("/opt/tt-metal/qwen_sampling_experiment.py")
    contents = Path(__file__).with_name("sampling_experiment.py").read_text()
    if helper.exists() and helper.read_text() != contents:
        raise SystemExit("Existing sampling helper differs")
    pending[helper] = contents
    for path, contents in pending.items():
        path.write_text(contents)
    print(json.dumps(dict(revisions=revisions, sha256={str(path): hashlib.sha256(contents.encode()).hexdigest()
                                                     for path, contents in pending.items()}), indent=2))


if __name__ == "__main__":
    main()
