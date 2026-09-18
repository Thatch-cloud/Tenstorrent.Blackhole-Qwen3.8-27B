"""Read-only packaging inventory; never initializes TT devices or loads weights."""

import argparse
import hashlib
from importlib import metadata
import json
from pathlib import Path
import subprocess


CACHE_KEY = 'b501e1084c8fdb949b203b8ede33bf0049f45686a80944ca8ceab532f3a02c90'
BINARY_SHA256 = '4b7299c1c9233b25aad310bc9a9d751a0631c6af0602cb1a151f934b4bfa07ea'


def fingerprint(path):
    path = Path(path)
    if not path.is_file():
        return dict(present=False)
    digest = hashlib.sha256()
    with path.open('rb') as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(chunk)
    return dict(present=True, bytes=path.stat().st_size, sha256=digest.hexdigest())


def inventory(cache, staged, runtime):
    entry = Path(cache) / 'dspark-native-v1' / CACHE_KEY
    binary = fingerprint(entry / '_ttnncpp.so')
    manifest = entry / 'manifest.json'
    saved = json.loads(manifest.read_text()) if manifest.is_file() else None
    packages = {}
    for name in ('torch', 'vllm', 'vllm-tt-plugin', 'ttnn'):
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            packages[name] = None
    files = ('model_batch.py', 'verifier_engine.py', 'dflash_combined_request.py',
        'draft_kv_slide.cpp', 'draft_kv_slide_gate.py', 'frozen_combined_runtime.py',
        'target_t16_attention_gate.py', 'dflash_t16_native_scope.py')
    return dict(device_access=False, weights_loaded=False, serving_qualified=False,
        cache_key=CACHE_KEY, cached_binary=binary, cache_manifest=saved,
        expected_binary_sha256=BINARY_SHA256,
        cached_binary_matches=binary.get('sha256') == BINARY_SHA256,
        staged_files={name: fingerprint(Path(staged) / 'scripts/ci' / name) for name in files},
        runtime_revision=subprocess.check_output(['git', '-C', str(runtime), 'rev-parse', 'HEAD'], text=True).strip(),
        installed_packages=packages)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    arguments = parser.parse_args()
    report = inventory('/inventory-cache', '/inventory-staged', '/opt/tt-metal')
    arguments.output.write_text(json.dumps(report, indent=2) + '\n')
