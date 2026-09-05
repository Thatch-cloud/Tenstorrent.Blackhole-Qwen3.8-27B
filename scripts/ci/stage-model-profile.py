"""Scope the pinned profiler report to the explicitly measured decode trace."""

import ast
import importlib.util
import json
import os
from pathlib import Path
import subprocess


def transform(source):
    staging = Path(os.environ.get("QWEN_PROFILE_STAGE_HELPER", "/experiment-optimisation/sim/stage-continuation.py"))
    spec = importlib.util.spec_from_file_location("stage", staging)
    stage = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(stage)
    return stage.patch_method(source, "process_ops", ((
        "    ops, signposts, traceReplays = import_tracy_op_logs(logFolder)\n",
        "    ops, signposts, traceReplays = import_tracy_op_logs(logFolder)\n"
        "    generation = json.loads((Path(output_folder) / 'generation.json').read_text())\n"
        "    if not generation['passed']:\n"
        "        raise AssertionError('Generation oracle failed')\n"
        "    ops = {key: op for key, op in ops.items()\n"
        "           if str(op.get('metal_trace_id')) == str(generation['decode_trace_id'])}\n"
        "    if {str(op.get('device_id')) for op in ops.values()} != {'0', '1'}:\n"
        "        raise AssertionError('Both-chip decode metadata required')\n",
    ),))


def main():
    root = Path("/opt/tt-metal")
    revision = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    if revision != "9f9cd4fd590f4b606bd0981a4fe0b6403eb38ec9":
        raise RuntimeError("Unreviewed profiler revision")
    relative = "tools/tracy/process_ops_logs.py"
    source = subprocess.check_output(["git", "-C", str(root), "show", f"HEAD:{relative}"], text=True)
    updated = transform(source)
    ast.parse(updated)
    path = root / relative
    if path.read_text() not in (source, updated):
        raise RuntimeError("Refusing unrelated profiler changes")
    path.write_text(updated)
    print(json.dumps(dict(revision=revision, path=relative, scope="Report selection only; no device code changes")))


if __name__ == "__main__":
    main()
