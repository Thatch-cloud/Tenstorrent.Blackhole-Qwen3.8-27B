"""Stage the single-lane interleaving experiment into a pinned plugin checkout."""

import argparse
import ast
import hashlib
import importlib.util
import json
import subprocess
from pathlib import Path

REVISION = "bf77cd63756fc891b8fb7f7cb3f5c1420f0e044c"
BASE = "src/vllm_tt_plugin/"
spec = importlib.util.spec_from_file_location("stage_continuation", Path(__file__).with_name("stage-continuation.py"))
stage = importlib.util.module_from_spec(spec)
spec.loader.exec_module(stage)


def transform(name, source):
    if name == "scheduler.py":
        source = stage.patch_method(source, "schedule", ((
            "        mode = self._forced_mode\n",
            "        from vllm_tt_plugin.qwen_interleave import enabled, choose_prefill\n"
            "        mode = self._forced_mode\n"
            "        if enabled() and mode == TTSchedulingMode.DEFAULT:\n"
            "            mode = TTSchedulingMode.from_prefill_intent(int(choose_prefill(\n"
            "                self, has_pending_prefill, has_running_decode)))\n",
        ), (
            "            result = self._schedule_prefill_only()\n",
            "            result = self._schedule_prefill_only()\n"
            "            if (enabled() and self._forced_mode == TTSchedulingMode.DEFAULT\n"
            "                    and result.total_num_scheduled_tokens == 0 and has_running_decode):\n"
            "                result = self._schedule_decode_only()\n",
        )))
        return stage.patch_method(source, "_schedule_prefill_only", ((
            "        pure_decodes = ",
            "        from vllm_tt_plugin.qwen_interleave import single_prefill\n"
            "        with single_prefill(self) as active:\n"
            "            if active:\n"
            "                return super().schedule()\n"
            "        pure_decodes = ",
        ),))
    if name == "platform.py":
        return stage.patch_method(source, "_apply_chunked_prefill_policy", ((
            "    scheduler_config = vllm_config.scheduler_config\n",
            "    from vllm_tt_plugin.qwen_interleave import validate_config\n"
            "    if validate_config(vllm_config):\n"
            "        return\n"
            "    scheduler_config = vllm_config.scheduler_config\n",
        ),))
    if name == "model_input.py":
        anchor = "    intermediate_prefill_mask: torch.Tensor | None = None\n"
        if source.count(anchor) != 1:
            raise ValueError("Ambiguous TTModelInput anchor")
        return source.replace(anchor, anchor + "    prefill_request_identity: list[tuple[str, int]] | None = None\n")
    if name == "model_runner.py":
        source = stage.patch_method(source, "_forward_with_model_input", ((
            "        sampling_params = model_input.tt_sampling_params\n",
            "        from qwen_boundary_diagnostics import record_input, record_output\n"
            "        record_input(model_input, self.input_batch.req_ids)\n"
            "        sampling_params = model_input.tt_sampling_params\n",
        ), (
            "        return _SyncForward(\n",
            "        record_output(tt_out)\n"
            "        return _SyncForward(\n",
        )))
        source = stage.patch_method(source, "_release_dead_state_slots", ((
            "        for req_id in scheduler_output.finished_req_ids:\n",
            "        from vllm_tt_plugin.qwen_interleave import release_requests\n"
            "        release_requests(self, scheduler_output)\n"
            "        for req_id in scheduler_output.finished_req_ids:\n",
        ),))
        source = stage.patch_method(source, "_prepare_model_inputs", ((
            "        return TTModelInput(\n",
            "        from vllm_tt_plugin.qwen_interleave import identities\n"
            "        prefill_identity = identities(self, row_req_ids) if is_prompt else None\n"
            "        return TTModelInput(\n"
            "            prefill_request_identity=prefill_identity,\n",
        ),))
        return stage.patch_method(source, "submit_prefill", ((
            "        if self.request_specific_rope:\n",
            "        from vllm_tt_plugin.qwen_interleave import enabled, submit\n"
            "        if enabled():\n"
            "            return submit(self, model_input, kwargs)\n"
            "        if self.request_specific_rope:\n",
        ),))
    raise ValueError(name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    root = args.source.resolve()
    revision = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    if revision != REVISION:
        raise SystemExit(f"Unreviewed plugin revision: {revision}")
    pending = {}
    for name in ("scheduler.py", "platform.py", "model_input.py", "model_runner.py"):
        relative = BASE + name
        original = subprocess.check_output(["git", "-C", str(root), "show", f"HEAD:{relative}"], text=True)
        updated = transform(name, original)
        ast.parse(updated)
        path = root / relative
        if path.read_text() not in (original, updated):
            raise SystemExit(f"Refusing unrelated changes: {path}")
        pending[path] = updated
    helper = root / BASE / "qwen_interleave.py"
    contents = Path(__file__).with_name("interleave.py").read_text()
    if helper.exists() and helper.read_text() != contents:
        raise SystemExit(f"Existing helper differs; review before restaging: {helper}")
    pending[helper] = contents
    if args.apply:
        for path, contents in pending.items():
            if not path.exists() or path.read_text() != contents:
                path.write_text(contents)
    print(json.dumps({"revision": revision, "applied": args.apply,
                      "sha256": {str(path.relative_to(root)): hashlib.sha256(contents.encode()).hexdigest()
                                 for path, contents in pending.items()}}, indent=2))


if __name__ == "__main__":
    main()
