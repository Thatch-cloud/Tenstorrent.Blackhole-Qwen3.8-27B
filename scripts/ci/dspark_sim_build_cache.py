"""Cache the simulator factory library while retaining the original build audits."""

import json
import os
from pathlib import Path
import shutil
import subprocess
from unittest.mock import patch

import dspark_fp32_build as baseline
import dspark_ladder_build as ladder
from dspark_hardware_gate import digest
from dspark_runtime_cache import IMAGE, cache_key, inspect_entry, store_entry


def build_inputs(root, scripts):
    transformer = root / 'ttnn/cpp/ttnn/operations/transformer'
    names = set(ladder.BUILDERS) | {'dspark_fp32_build.py', 'dspark_fp32_intermediates.py',
        'dspark_sim_build_cache.py', 'dspark_runtime_cache.py'}
    return dict(image=IMAGE, backend='simulator', builders={name: digest(scripts / name) for name in sorted(names)},
        factory=digest(root / baseline.SOURCE), registration=digest(transformer / 'sources.cmake'),
        implementations={name: digest(transformer / name) for name in baseline.implementation_sources()},
        packer=digest(root / 'tt_metal/tt-llk/tt_llk_blackhole/common/inc/cpack_common.h'))


def main():
    if (os.environ.get('QWEN_SIM_ONLY') != '1' or not os.environ.get('TT_METAL_SIMULATOR')
            or Path('/dev/tenstorrent').exists() or os.environ.get('QWEN_CARDS_ALLOCATED') == '1'
            or os.environ.get('QWEN_SCORE_BITWISE') != '1'):
        raise ValueError('Dedicated CPU-only score smoke required')
    root, scripts = Path('/opt/tt-metal'), Path(__file__).parent
    cache = Path('/simulator-build-cache/ladder-v1')
    original = subprocess.run
    state = {}

    def run(command, *arguments, **keywords):
        expected = ['ninja', '-C', str(root / 'build_Release'), '-j', '8', 'ttnncpp']
        if command != expected:
            return original(command, *arguments, **keywords)
        if state:
            raise ValueError('Exactly one factory build interception required')
        inputs = build_inputs(root, scripts)
        manifest = inspect_entry(cache, inputs)
        state.update(inputs=inputs, cache_hit=manifest is not None)
        if manifest is None:
            return original(command, *arguments, **keywords)
        binary = cache / cache_key(inputs) / '_ttnncpp.so'
        for name in ('build_Release/lib/_ttnncpp.so', 'build_Release/ttnn/_ttnncpp.so'):
            shutil.copy2(binary, root / name)
        return subprocess.CompletedProcess(command, 0)

    with patch.object(baseline.subprocess, 'run', run):
        ladder.main()
    if not state:
        raise ValueError('Factory build was not intercepted')
    manifest = store_entry(cache, state['inputs'], root / 'build_Release/ttnn/_ttnncpp.so')
    output = Path('/experiment/results/dspark-simulator-build-cache.json')
    output.write_text(json.dumps(dict(cache_hit=state['cache_hit'],
        cache_key=cache_key(state['inputs']), binary_sha256=manifest['binary_sha256']), indent=2) + '\n')


if __name__ == '__main__':
    main()
