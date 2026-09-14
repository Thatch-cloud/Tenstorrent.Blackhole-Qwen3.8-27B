"""Content-addressed isolated CCL build cache for the pinned DSpark hardware container."""

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

from dspark_hardware_gate import digest


IMAGE = 'sha256:f1e9b1a64b4f7aa04cd3d3b36fefed4d47320bfdd0f4d108d2ca85a932cf9465'
BUILDERS = ('ccl-links-build.sh','sdpa_graft_build.py','lazy_ccl_links.py','dspark_runtime_cache.py',
    'dspark_hardware_gate.py','dspark_native_restore.py','dspark_8k_build.py',
    'dspark_fp32_intermediates.py','dspark_attention_8k_gate.py')


def cache_key(inputs):
    return hashlib.sha256(json.dumps(inputs,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def inspect_entry(cache, inputs):
    directory = Path(cache)/cache_key(inputs)
    if not directory.exists():
        return None
    manifest = json.loads((directory/'manifest.json').read_text())
    if manifest.get('inputs')!=inputs or digest(directory/'_ttnncpp.so')!=manifest.get('binary_sha256'):
        raise ValueError('Cached runtime provenance or binary hash changed')
    return manifest


def store_entry(cache, inputs, binary):
    cache = Path(cache)
    cache.mkdir(parents=True,exist_ok=True)
    existing = inspect_entry(cache,inputs)
    if existing is not None:
        if existing['binary_sha256']!=digest(binary):
            raise ValueError('Refusing to replace a different binary under an existing cache key')
        return existing
    directory = Path(tempfile.mkdtemp(prefix=cache_key(inputs)+'.partial-',dir=cache))
    shutil.copy2(binary,directory/'_ttnncpp.so')
    manifest = dict(inputs=inputs,binary_sha256=digest(directory/'_ttnncpp.so'))
    (directory/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    directory.rename(cache/cache_key(inputs))
    return manifest


def main():
    if (os.environ.get('QWEN_HARDWARE_TESTS')!='1' or os.environ.get('QWEN_CARDS_ALLOCATED')!='1'
            or os.environ.get('QWEN_CCL_LAZY_BUILD')!='1' or os.environ.get('TT_METAL_SIMULATOR')
            or os.environ.get('TT_METAL_HOME')!='/opt/tt-metal'):
        raise ValueError('Allocated pinned hardware container required for native runtime cache')
    root,scripts = Path('/opt/tt-metal'),Path(__file__).parent
    from dspark_native_restore import SOURCE, SIMULATED_SHA256

    if digest(root/SOURCE)!=SIMULATED_SHA256:
        raise ValueError('Exact simulated slice source required before build/cache restore')
    patch = Path('/tmp/ccl-graft-registration.patch')
    cache,output = Path('/experiment-cache/dspark-native-v1'),Path('/experiment/results')
    inputs = dict(image=IMAGE,builders={name:digest(scripts/name) for name in BUILDERS},registration_patch=digest(patch))
    from dspark_context_selection import request_context
    from dspark_8k_build import prepare, completed
    factory = prepare(root, scripts, enabled=request_context() == 8192)
    if factory is not None:
        inputs['draft_8k_factory'] = factory
    factory_64k = None
    if request_context() == 65536:
        from dspark_64k_build import prepare as prepare_64k
        factory_64k = prepare_64k(root, scripts, scripts / 'dspark-ladder-hardware-65536.json')
        inputs['draft_64k_factory'] = factory_64k
    cache_selection = os.environ.get('QWEN_DSPARK_BIAS_CACHE', '0')
    if cache_selection not in ('0', '1'):
        raise ValueError('Explicit zero or one bias cache selection required')
    cache_factory = None
    if cache_selection == '1':
        from dspark_cached_markov_build import prepare as prepare_cache
        cache_factory = prepare_cache(root, scripts, enabled=True)
        inputs['markov_cache_factory'] = cache_factory
        inputs['markov_cache_builders'] = {name: digest(scripts / name) for name in (
            'dspark_cached_markov_build.py', 'dspark_cached_markov_gate.py', 'markov_sparse_fp32.py')}
    manifest = inspect_entry(cache,inputs)
    hit = manifest is not None
    if hit:
        subprocess.run([sys.executable,str(scripts/'sdpa_graft_build.py')],check=True)
        subprocess.run(['git','-C',str(root),'apply','--check',str(patch)],check=True)
        subprocess.run(['git','-C',str(root),'apply',str(patch)],check=True)
        subprocess.run([sys.executable,str(scripts/'lazy_ccl_links.py'),'--root',str(root),
            '--output',str(output/'ccl-links-source.json')],check=True)
        binary = cache/cache_key(inputs)/'_ttnncpp.so'
        for name in ('build_Release/lib/_ttnncpp.so','build_Release/ttnn/_ttnncpp.so'):
            shutil.copy2(binary,root/name)
    else:
        subprocess.run(['bash',str(scripts/'ccl-links-build.sh')],check=True)
        manifest = store_entry(cache,inputs,root/'build_Release/ttnn/_ttnncpp.so')
    for name in ('build_Release/lib/_ttnncpp.so','build_Release/ttnn/_ttnncpp.so'):
        if digest(root/name)!=manifest['binary_sha256']:
            raise ValueError('Both runtime library paths must match the content-addressed build')
    (output/'ccl-links-build.sha256').write_text(manifest['binary_sha256']+'  /opt/tt-metal/build_Release/lib/_ttnncpp.so\n')
    subprocess.run([sys.executable,'-c',
        'import ttnn; names=("attn_decode_prep","gdn_decode_norm_gate","gdn_decode_conv_gates","decode_gated_delta_rule_packed"); '
        'assert all(callable(getattr(ttnn.transformer,name)) for name in names)'],check=True)
    if factory is not None:
        evidence = completed(root, factory, manifest['binary_sha256'], import_passed=True)
        evidence['cache_key'] = cache_key(inputs)
        (output/'dspark-8k-hardware-build.json').write_text(json.dumps(evidence,indent=2)+'\n')
    if factory_64k is not None:
        from dspark_64k_build import completed as completed_64k
        evidence = completed_64k(root, factory_64k, manifest['binary_sha256'], import_passed=True)
        evidence['cache_key'] = cache_key(inputs)
        (output/'dspark-64k-hardware-build.json').write_text(json.dumps(evidence,indent=2)+'\n')
    if cache_factory is not None:
        from dspark_cached_markov_build import completed as completed_cache
        subprocess.run([sys.executable, '-c', 'import ttnn; assert callable(ttnn.sparse_matmul)'], check=True)
        evidence = completed_cache(root, cache_factory, manifest['binary_sha256'], import_passed=True)
        evidence['cache_key'] = cache_key(inputs)
        (output/'dspark-cached-markov-hardware-build.json').write_text(json.dumps(evidence,indent=2)+'\n')
    (output/'dspark-runtime-cache.json').write_text(json.dumps(dict(cache_hit=hit,cache_key=cache_key(inputs),**manifest),indent=2)+'\n')


if __name__=='__main__':
    main()
