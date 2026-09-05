"""Stage opt-in continuation into a reviewed disposable TT-Metal checkout."""

import argparse
import ast
import hashlib
import json
import subprocess
from pathlib import Path

REVISION = "9f9cd4fd590f4b606bd0981a4fe0b6403eb38ec9"
BASE = "models/demos/blackhole/qwen36/tt/"


def patch_method(source, name, changes):
    methods = [node for node in ast.walk(ast.parse(source)) if isinstance(node, ast.FunctionDef) and node.name == name]
    if len(methods) != 1:
        raise ValueError(f"Expected one {name} method")
    method = methods[0]
    lines = source.splitlines(keepends=True)
    body = "".join(lines[method.lineno - 1:method.end_lineno])
    for original, replacement in changes:
        if body.count(original) != 1:
            raise ValueError(f"Missing or ambiguous {name} anchor: {original!r}")
        body = body.replace(original, replacement, 1)
    return "".join(lines[:method.lineno - 1]) + body + "".join(lines[method.end_lineno:])


def transform(name, source):
    if name == "model.py":
        updated = patch_method(source, "_prefill_traced_chunked_tp", (
            ("tail_real, vision_tokens=None\n", "tail_real, vision_tokens=None, *, start_pos=0, is_last=True\n"),
            ("        self._reset_gdn_state_for_new_sequence()\n",
             "        if start_pos == 0:\n            self._reset_gdn_state_for_new_sequence()\n"),
            ("        for c in range(num_full):\n", "        for c in range(start_pos // chunk_size, num_full):\n"),
            ("        if tail_real > 0:\n", "        if not is_last:\n            return None\n\n        if tail_real > 0:\n"),
        ))
        method = next(node for node in ast.walk(ast.parse(updated))
                      if isinstance(node, ast.FunctionDef) and node.name == "_prefill_traced_chunked_tp")
        lines = updated.splitlines(keepends=True)
        first_dma = next(index for index in range(method.lineno, method.end_lineno)
                         if lines[index].startswith("        pt_host = "))
        guarded = "        try:\n" + "".join("    " + line if line.strip() else line
                                                 for line in lines[first_dma:method.end_lineno])
        guarded += ("        finally:\n"
                    "            try:\n"
                    "                ttnn.synchronize_device(self.device)\n"
                    "            except BaseException:\n"
                    "                self._prefill_failed_dma_refs = dict(locals())\n"
                    "                raise\n")
        return "".join(lines[:first_dma]) + guarded + "".join(lines[method.end_lineno:])
    if name == "qwen36_vllm.py":
        return patch_method(source, "prefill_forward", ((
            "        model = self.model[0]\n",
            '        if os.environ.get("QWEN_PREFILL_CONTINUATION", "0") == "1":\n'
            "            from qwen_prefill_continuation import dispatch_prefill\n"
            "\n"
            "            return dispatch_prefill(self, tokens, page_table, prompt_lens, kwargs)\n"
            "        model = self.model[0]\n",
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
        raise SystemExit(f"Unreviewed source revision: {revision}")
    pending = {}
    for name in ("model.py", "qwen36_vllm.py"):
        relative = BASE + name
        original = subprocess.check_output(["git", "-C", str(root), "show", f"HEAD:{relative}"], text=True)
        updated = transform(name, original)
        ast.parse(updated)
        path = root / relative
        if path.read_text() not in (original, updated):
            raise SystemExit(f"Refusing to overwrite unrelated changes: {path}")
        pending[path] = updated
    helper = root / "qwen_prefill_continuation.py"
    contents = Path(__file__).with_name("continuation.py").read_text()
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
