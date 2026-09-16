"""Reuse only identical historical simulator builds; never substitute a newer factory."""

import json
import os
from pathlib import Path
import shutil
import subprocess
from unittest.mock import patch

import dspark_fp32_build as baseline
from dspark_hardware_gate import digest
from frozen_binary_cache import IMAGE, cache_key, inspect_entry, store_entry


def main():
    root, scripts = Path('/opt/tt-metal'), Path(__file__).parent
    cache = Path('/frozen-simulator-cache/native-stats-v1')
    original = subprocess.run
    state = {}
    scratch = None
    selection = os.environ.get('QWEN_FROZEN_TARGET_SCRATCH', '0')
    if selection not in ('0', '1'):
        raise ValueError('Explicit target scratch build selection required')
    if selection == '1':
        from sdpa_tree_scratch import audit
        audit(root)
        source_patch = Path('/simulator-support/sdpa-tree-scratch.patch')
        original(['git', '-C', str(root), 'apply', '--check', str(source_patch)], check=True, timeout=10)
        original(['git', '-C', str(root), 'apply', str(source_patch)], check=True, timeout=10)
        scratch = dict(sources=audit(root, patched=True), patch_sha256=digest(source_patch),
            audit_sha256=digest(scripts / 'sdpa_tree_scratch.py'))

    def run(command, *arguments, **keywords):
        expected = ['ninja', '-C', str(root / 'build_Release'), '-j', '8', 'ttnncpp']
        if command != expected:
            return original(command, *arguments, **keywords)
        if state:
            raise ValueError('Exactly one historical factory build required')
        transformer = root / 'ttnn/cpp/ttnn/operations/transformer'
        inputs = dict(image=IMAGE, backend='simulator',
            factory=digest(root / baseline.SOURCE),
            registration=digest(transformer / 'sources.cmake'),
            implementations={name: digest(transformer / name) for name in baseline.implementation_sources()},
            packer=digest(root / 'tt_metal/tt-llk/tt_llk_blackhole/common/inc/cpack_common.h'),
            builders={name: digest(scripts / name) for name in
                ('dspark_fp32_build.py', 'dspark_fp32_intermediates.py',
                 'frozen_sim_build_cache.py', 'frozen_binary_cache.py', 'dspark_hardware_gate.py')},
            base_binaries={name: digest(root / name) for name in
                ('build_Release/lib/_ttnncpp.so', 'build_Release/ttnn/_ttnncpp.so')})
        if scratch is not None:
            inputs['target_tree_scratch'] = scratch
        manifest = inspect_entry(cache, inputs)
        state.update(inputs=inputs, cache_hit=manifest is not None)
        print(json.dumps(dict(stage='frozen_factory_cache', hit=manifest is not None)), flush=True)
        if manifest is None:
            if os.environ.get('QWEN_FROZEN_BUILD_ONLY') != '1':
                raise ValueError('Build cache missing; prepare it separately before numerical testing')
            return original(command, *arguments, **keywords)
        for name in ('build_Release/lib/_ttnncpp.so', 'build_Release/ttnn/_ttnncpp.so'):
            shutil.copy2(cache / cache_key(inputs) / '_ttnncpp.so', root / name)
        return subprocess.CompletedProcess(command, 0)

    with patch.object(baseline.subprocess, 'run', run):
        baseline.main()
    if not state:
        raise ValueError('Historical build was not intercepted')
    baseline.validate_manifest(root, '/experiment/results/dspark-fp32-build.json')
    if scratch is not None:
        from sdpa_tree_scratch import audit
        if audit(root, patched=True) != scratch['sources']:
            raise ValueError('Target scratch sources changed during build')
    manifest = store_entry(cache, state['inputs'], root / 'build_Release/ttnn/_ttnncpp.so')
    Path('/experiment/results/frozen-build-cache.json').write_text(json.dumps(dict(
        cache_hit=state['cache_hit'], cache_key=cache_key(state['inputs']),
        binary_sha256=manifest['binary_sha256'], original_build_audit_passed=True,
        target_tree_scratch=scratch), indent=2) + '\n')


if __name__ == '__main__':
    main()
