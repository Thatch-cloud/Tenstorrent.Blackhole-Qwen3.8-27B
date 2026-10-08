"""Stage the production model.py with the narrowed (region read) prefix audit: <out>/prod-audit/model.py.

    python3 stage_prod_audit.py --image qwen-fast-serving:ci-be9e184e672756c8eee040f03e48dbff58e68fda --out ~/opgraft-KVR
    python3 stage_prod_audit.py --original model.orig.py --out ./graft          # no docker: the P8 original from a file

Why: the P1-CTL re-run (W-2's precondition) must run PRODUCTION bytes, and tp4-serve-10's audit reads the whole KV pool per request
(~8.4 minutes at 8 x 262k). The image's model.py is the P8 original run through this checkout's prefix stage, which changed only the
audit (docs/prefix-audit-cost.md); mounting this file over it (c2_prefix_gate --kvread-mount, job key C2_PREFIX_KVREAD_MOUNT=1), with
the qwen_kv_read extension, gives production's engine with the cheap audit. The file is the stage's own output: the original must be
the pinned one (qwen_prefix_model_patch.SOURCE_SHA256) and the output the pinned graft (PATCHED_SHA256), or this refuses. The gate's
anchor probe holds the served model.py to the same pin, so a stale file is refused before any container boots.

--verify-against FILE (the model.py the production image serves) prints the methods that differ from it: for tp4-serve-10 they are the
audit's methods and the prefill loop's collect-then-audit-once-per-step change, all inert without QWEN_PREFIX_AUDIT=1.
"""

import argparse
import ast
import hashlib
import os
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
CI = HERE.parent.parent.parent / 'scripts' / 'ci'
ORIGINAL_PATH = '/opt/tt-metal/models/demos/blackhole/qwen36/tt/model.py'


def sha(data):
    return hashlib.sha256(data).hexdigest()


def methods(source):
    tree = ast.parse(source)
    out = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    out[node.name + '.' + item.name] = ast.get_source_segment(source, item)
    return out


def method_diff(served, staged):
    a, b = methods(served), methods(staged)
    return dict(changed=sorted(n for n in a if n in b and a[n] != b[n]), added=sorted(set(b) - set(a)), removed=sorted(set(a) - set(b)))


def original_from_image(image):
    result = subprocess.run(['docker', 'run', '--rm', '--network', 'none', '--entrypoint', 'cat', image, ORIGINAL_PATH],
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=300)
    if result.returncode != 0:
        raise SystemExit('cannot read %s from %s: %s' % (ORIGINAL_PATH, image, result.stderr.decode('utf-8', 'replace')[-300:]))
    return result.stdout


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--image', help='the P8 base image holding the pinned original model.py')
    parser.add_argument('--original', help='the pinned original model.py as a file (instead of --image)')
    parser.add_argument('--out', required=True, help='the graft directory (the rig\'s ~/opgraft-KVR): writes <out>/prod-audit/model.py')
    parser.add_argument('--verify-against', help='the model.py the production image serves: print the methods that differ')
    options = parser.parse_args(argv)
    if bool(options.image) == bool(options.original):
        parser.error('exactly one of --image and --original')
    sys.path.insert(0, str(CI))
    sys.dont_write_bytecode = True
    import qwen_prefix_model_patch as patcher
    data = original_from_image(options.image) if options.image else Path(options.original).read_bytes()
    if sha(data) != patcher.SOURCE_SHA256[patcher.MODEL_FILE]:
        raise SystemExit('the original model.py is %s, not the pinned %s' % (sha(data), patcher.SOURCE_SHA256[patcher.MODEL_FILE]))
    staged = patcher.patch_model(data.decode('utf-8')).encode('utf-8')   # pinned both ways
    out = Path(options.out) / 'prod-audit'
    out.mkdir(parents=True, exist_ok=True)
    (out / 'model.py').write_bytes(staged)
    (out / 'model.py.sha256').write_text('%s  model.py\n' % sha(staged), encoding='utf-8', newline='\n')
    print('prod-audit/model.py %s (pinned %s) from the original %s' % (sha(staged)[:12], patcher.PATCHED_SHA256[patcher.MODEL_FILE][:12], sha(data)[:12]))
    if options.verify_against:
        served = Path(options.verify_against).read_bytes()
        diff = method_diff(served.decode('utf-8'), staged.decode('utf-8'))
        print('against the served %s: %s' % (sha(served)[:12], diff))
    manifest = Path(options.out) / 'MANIFEST.sha256'
    if manifest.is_file():
        lines = sorted('%s  ./%s' % (sha(p.read_bytes()), p.relative_to(options.out).as_posix())
                       for p in Path(options.out).rglob('*') if p.is_file() and p.name not in ('MANIFEST.sha256', 'build.log'))
        manifest.write_text('\n'.join(lines) + '\n', encoding='utf-8', newline='\n')
        print('MANIFEST.sha256 rewritten: %d files' % len(lines))
    return 0


if __name__ == '__main__':
    sys.exit(main())
