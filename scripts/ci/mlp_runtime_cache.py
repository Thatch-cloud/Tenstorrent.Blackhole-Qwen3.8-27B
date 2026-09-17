"""Isolated native build reuse for the opt-in DRAM MLP hardware suite."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from dspark_hardware_gate import digest
from dspark_native_restore import SOURCE, IMAGE_SHA256
from dspark_runtime_cache import IMAGE, cache_key, inspect_entry, store_entry


BUILDERS = ('ccl-links-build.sh', 'sdpa_graft_build.py', 'lazy_ccl_links.py',
    'mlp_runtime_cache.py', 'dspark_runtime_cache.py', 'dspark_hardware_gate.py', 'dspark_native_restore.py')


def inputs_for(root, scripts, patch):
    if digest(root / SOURCE) != IMAGE_SHA256:
        raise ValueError('Unmodified pinned image slice required; DSpark-restored builds cannot be reused')
    return dict(image=IMAGE, builders={name: digest(scripts / name) for name in BUILDERS},
        registration_patch=digest(patch), slice_sha256=IMAGE_SHA256,
        base_binary_sha256=digest(root / 'build_Release/lib/_ttnncpp.so'))


def main():
    if (os.environ.get('QWEN_HARDWARE_TESTS') != '1' or os.environ.get('QWEN_CARDS_ALLOCATED') != '1'
            or os.environ.get('QWEN_DRAM_MLP') != '1' or os.environ.get('QWEN_CCL_LAZY_BUILD') != '1'
            or os.environ.get('TT_METAL_SIMULATOR') or os.environ.get('TT_METAL_HOME') != '/opt/tt-metal'):
        raise ValueError('Allocated opt-in DRAM MLP hardware container required')
    root, scripts = Path('/opt/tt-metal'), Path(__file__).parent
    patch = Path('/tmp/ccl-graft-registration.patch')
    cache, output = Path('/experiment-cache/dram-mlp-native-v1'), Path('/experiment/results')
    started = time.monotonic()
    inputs = inputs_for(root, scripts, patch)
    manifest = inspect_entry(cache, inputs)
    hit = manifest is not None
    if hit:
        subprocess.run([sys.executable, str(scripts / 'sdpa_graft_build.py')], check=True)
        subprocess.run(['git', '-C', str(root), 'apply', '--check', str(patch)], check=True)
        subprocess.run(['git', '-C', str(root), 'apply', str(patch)], check=True)
        subprocess.run([sys.executable, str(scripts / 'lazy_ccl_links.py'), '--root', str(root),
            '--output', str(output / 'ccl-links-source.json')], check=True)
        binary = cache / cache_key(inputs) / '_ttnncpp.so'
        for name in ('build_Release/lib/_ttnncpp.so', 'build_Release/ttnn/_ttnncpp.so'):
            shutil.copy2(binary, root / name)
    else:
        subprocess.run(['bash', str(scripts / 'ccl-links-build.sh')], check=True)
        manifest = store_entry(cache, inputs, root / 'build_Release/ttnn/_ttnncpp.so')
    for name in ('build_Release/lib/_ttnncpp.so', 'build_Release/ttnn/_ttnncpp.so'):
        if digest(root / name) != manifest['binary_sha256']:
            raise ValueError('Both runtime libraries must match the verified cached binary')
    (output / 'ccl-links-build.sha256').write_text(
        manifest['binary_sha256'] + '  /opt/tt-metal/build_Release/lib/_ttnncpp.so\n')
    subprocess.run([sys.executable, str(scripts / 'dram-projection-binding-check.py')], check=True)
    (output / 'dram-mlp-runtime-cache.json').write_text(json.dumps(dict(cache_hit=hit,
        cache_key=cache_key(inputs), build_seconds=time.monotonic() - started, **manifest), indent=2) + '\n')


if __name__ == '__main__':
    main()
