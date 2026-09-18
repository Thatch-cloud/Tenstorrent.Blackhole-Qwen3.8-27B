"""Recreate cached native source inputs without compiling or accessing devices."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

from serving_image_inventory import BINARY_SHA256, CACHE_KEY


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_cache(manifest, scripts, binary):
    inputs = manifest.get('inputs', {})
    key = hashlib.sha256(json.dumps(inputs, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    if key != CACHE_KEY or manifest.get('binary_sha256') != BINARY_SHA256 or digest(binary) != BINARY_SHA256:
        raise ValueError('Exact combined runtime cache provenance and binary required')
    for name, checksum in inputs['builders'].items():
        if Path(name).name != name or digest(Path(scripts) / name) != checksum:
            raise ValueError('Cached runtime builder changed: ' + name)
    return inputs


def apply_patch_file(root, patch_file, expected):
    if digest(patch_file) != expected:
        raise ValueError('Native patch differs from cached build inputs')
    for options in (['--check'], []):
        subprocess.run(['git', '-C', str(root), 'apply', *options, str(patch_file)], check=True, timeout=20)


def install(root, scripts, optimisation, binary, manifest):
    root, scripts, optimisation, binary = map(Path, (root, scripts, optimisation, binary))
    inputs = validate_cache(manifest, scripts, binary)
    revision = subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip()
    if revision != '9f9cd4fd590f4b606bd0981a4fe0b6403eb38ec9':
        raise ValueError('Pinned native runtime checkout required')
    from dspark_native_restore import restore
    from dspark_fp32_intermediates import SOURCE, transform
    from sdpa_tree_scratch import audit as scratch_audit
    from sdpa_graft_build import audit as graft_audit
    from lazy_ccl_links import apply as apply_links

    factory = inputs['draft_8k_factory']
    if factory['source'] != SOURCE:
        raise ValueError('Unexpected cached SDPA factory path')
    source = root / SOURCE
    if digest(source) != factory['source_before']:
        raise ValueError('SDPA source does not match pre-transform cache provenance')
    transformed = transform(source.read_bytes())
    if hashlib.sha256(transformed).hexdigest() != factory['factory_sha256']:
        raise ValueError('Staged geometry does not recreate cached factory bytes')
    scratch = factory['target_tree_scratch']
    if digest(scripts / 'frozen_combined_runtime.py') != scratch['adapter_sha256']:
        raise ValueError('Frozen scratch adapter differs from cache provenance')
    scratch_audit(root)
    graft = graft_audit(root)
    restored = restore(root)
    source.write_bytes(transformed)
    apply_patch_file(root, optimisation / 'sim/sdpa-tree-scratch.patch', scratch['patch_sha256'])
    if scratch_audit(root, patched=True) != scratch['sources']:
        raise ValueError('Restored scratch sources differ from cached runtime')
    apply_patch_file(root, Path('/tmp/ccl-graft-registration.patch'), inputs['registration_patch'])
    links = apply_links(root)
    for name in ('build_Release/lib/_ttnncpp.so', 'build_Release/ttnn/_ttnncpp.so'):
        destination = root / name
        if not destination.is_file():
            raise ValueError('Existing runtime binary location required: ' + name)
        shutil.copyfile(binary, destination)
        if digest(destination) != BINARY_SHA256:
            raise ValueError('Installed runtime binary mismatch')
    return dict(cache_key=CACHE_KEY, binary_sha256=BINARY_SHA256, factory_sha256=digest(source),
        slice=restored, graft=graft, links=links, scratch=scratch,
        compiled=False, devices_accessed=False, serving_qualified=False, performance_qualified=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for name in ('root', 'scripts', 'optimisation', 'binary', 'manifest', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    arguments = parser.parse_args()
    report = install(arguments.root, arguments.scripts, arguments.optimisation, arguments.binary,
        json.loads(arguments.manifest.read_text()))
    arguments.output.write_text(json.dumps(report, indent=2) + '\n')
