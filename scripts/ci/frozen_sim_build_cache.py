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
    manifest = store_entry(cache, state['inputs'], root / 'build_Release/ttnn/_ttnncpp.so')
    Path('/experiment/results/frozen-build-cache.json').write_text(json.dumps(dict(
        cache_hit=state['cache_hit'], cache_key=cache_key(state['inputs']),
        binary_sha256=manifest['binary_sha256'], original_build_audit_passed=True), indent=2) + '\n')


if __name__ == '__main__':
    main()
