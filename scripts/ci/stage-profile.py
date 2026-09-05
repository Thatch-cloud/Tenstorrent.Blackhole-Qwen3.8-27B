"""Match the pinned attention fixture cache dtype to the shipped model flag."""

import ast
import hashlib
import json
from pathlib import Path
import subprocess
import sys

REVISION = "9f9cd4fd590f4b606bd0981a4fe0b6403eb38ec9"
RELATIVE = "models/demos/blackhole/qwen36/tests/test_attention_tp.py"


def transform(source):
    functions = [node for node in ast.walk(ast.parse(source))
                 if isinstance(node, ast.FunctionDef) and node.name == "test_attention_tp_paged"]
    if len(functions) != 1:
        raise ValueError("Expected one paged attention test")
    caches = [node for node in ast.walk(functions[0])
              if isinstance(node, ast.FunctionDef) and node.name == "mk_cache"]
    if len(caches) != 1:
        raise ValueError("Expected one scoped cache fixture")
    cache = caches[0]
    lines = source.splitlines(keepends=True)
    body = "".join(lines[cache.lineno - 1:cache.end_lineno])
    original = "dtype=ttnn.bfloat16,"
    if body.count(original) != 1:
        raise ValueError("Unexpected cache dtype anchor")
    body = body.replace(original,
                        'dtype=ttnn.bfloat8_b if os.environ.get("QWEN_SDPA_BF8") == "1" else ttnn.bfloat16,')
    updated = "".join(lines[:cache.lineno - 1]) + body + "".join(lines[cache.end_lineno:])
    ast.parse(updated)
    return updated


def main():
    root = Path(sys.argv[1]).resolve()
    revision = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    if revision != REVISION:
        raise SystemExit(f"Unreviewed source revision: {revision}")
    original = subprocess.check_output(["git", "-C", str(root), "show", f"HEAD:{RELATIVE}"], text=True)
    updated = transform(original)
    path = root / RELATIVE
    if path.read_text() not in (original, updated):
        raise SystemExit("Refusing to overwrite unrelated fixture changes")
    path.write_text(updated)
    print(json.dumps(dict(revision=revision, path=RELATIVE,
                          original_sha256=hashlib.sha256(original.encode()).hexdigest(),
                          staged_sha256=hashlib.sha256(updated.encode()).hexdigest()), indent=2))


if __name__ == "__main__":
    main()
