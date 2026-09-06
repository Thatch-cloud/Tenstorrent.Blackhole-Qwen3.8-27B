"""Read-only source and incremental-build availability audit; never opens cards."""

import hashlib
import json
from pathlib import Path
import shutil
import subprocess

from sdpa_tree_scratch import HASHES, ROOT


def main():
    root = Path('/opt/tt-metal')
    sources = {name: hashlib.sha256((root / ROOT / name).read_bytes()).hexdigest() for name in HASHES}
    config = root / 'ttnn/cpp/ttnn/operations/transformer/sdpa_config.hpp'
    report = dict(scope='Read-only native SDPA source and build availability; no device operations or build',
        sources=sources, expected_sources=HASHES, source_match=sources == HASHES,
        config_sha256=hashlib.sha256(config.read_bytes()).hexdigest(),
        worker_cap=[line.strip() for line in config.read_text().splitlines() if 'max_cores_per_head_batch' in line],
        tools={name: shutil.which(name) for name in ('ninja', 'cmake', 'clang++', 'g++')}, builds={})
    for directory in ('build', 'build_Release'):
        cache = root / directory / 'CMakeCache.txt'
        report['builds'][directory] = dict(ninja=(root / directory / 'build.ninja').is_file(),
            compiler=[line for line in cache.read_text().splitlines() if line.startswith((
                'CMAKE_CXX_COMPILER:', 'CMAKE_BUILD_TYPE:', 'CMAKE_HOME_DIRECTORY:'))] if cache.is_file() else [])
    transformer = root / 'ttnn/cpp/ttnn/operations/transformer'
    report['transformer_registration'] = {name: (transformer / name).read_text()
        for name in ('CMakeLists.txt', 'sources.cmake') if (transformer / name).is_file()}
    report['grafted_sources'] = {}
    for directory in transformer.iterdir():
        if directory.is_dir() and directory.name.startswith(('attn_', 'gdn_')):
            report['grafted_sources'][directory.name] = [str(path.relative_to(transformer))
                for path in directory.rglob('*') if path.is_file() and path.suffix in ('.cpp', '.hpp', '.cmake')]
    report['graft_symbols'] = {}
    for name in ('build_Release/lib/_ttnncpp.so', 'ttnn/ttnn/_ttnn.so'):
        library = root / name
        if library.is_file() and shutil.which('nm'):
            result = subprocess.run(['nm', '-D', '-C', str(library)], capture_output=True, text=True, timeout=30, check=True)
            report['graft_symbols'][name] = [line for line in result.stdout.splitlines()
                if any(symbol in line for symbol in ('attn_decode_prep(', 'gdn_decode_norm_gate(', 'gdn_decode_conv_gates('))]
    Path('/experiment/results/sdpa-tree-audit.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
