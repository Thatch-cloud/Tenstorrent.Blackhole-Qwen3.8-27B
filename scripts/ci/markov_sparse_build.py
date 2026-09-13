"""Bounded disposable simulator build for native sparse FP32 reload correction."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from dspark_hardware_gate import digest
from dspark_fp32_build import restore_registrations
from markov_sparse_fp32 import SOURCE, SOURCE_SHA256, ANCHOR, INSERT, CONFIG, CONFIG_REPLACEMENT, transform


BINARY_PATHS = ('build_Release/lib/_ttnncpp.so', 'build_Release/ttnn/_ttnncpp.so')
BUILDERS = ('markov_sparse_build.py', 'markov_sparse_fp32.py', 'dspark_fp32_build.py', 'sdpa_graft_build.py')


def validate(root, manifest):
    root = Path(root)
    report = json.loads(Path(manifest).read_text())
    if report.get('passed') is not True or report.get('source_before') != SOURCE_SHA256:
        raise ValueError('Completed pinned sparse build required')
    source = (root / SOURCE).read_bytes()
    if source.count(INSERT.encode()) != 1 or source.count(CONFIG_REPLACEMENT.encode()) != 1:
        raise ValueError('Exact sparse reload transformation required')
    original = source.replace(INSERT.encode(), b'').replace(CONFIG_REPLACEMENT.encode(), CONFIG.encode())
    if transform(original) != source or digest(root / SOURCE) != report.get('source_after'):
        raise ValueError('Sparse factory changed')
    binaries = report.get('binaries_after', {})
    if set(binaries) != set(BINARY_PATHS) or len(set(binaries.values())) != 1:
        raise ValueError('Both rebuilt library paths required')
    if any(digest(root / name) != expected for name, expected in binaries.items()):
        raise ValueError('Rebuilt libraries changed')
    if report.get('import_passed') is not True:
        raise ValueError('TTNN import smoke required')
    for name in BUILDERS:
        if report.get('builders', {}).get(name) != digest(Path(__file__).with_name(name)):
            raise ValueError('Builder changed')
    return report


def main():
    if (os.environ.get('QWEN_SIM_ONLY') != '1' or not os.environ.get('TT_METAL_SIMULATOR')
            or os.environ.get('QWEN_SIM_CASE') != 'markov-sparse-dot'
            or Path('/dev/tenstorrent').exists() or os.environ.get('QWEN_CARDS_ALLOCATED') == '1'
            or os.environ.get('TT_METAL_HOME') != '/opt/tt-metal'):
        raise ValueError('Dedicated device-free simulator container required')
    root = Path('/opt/tt-metal')
    output = Path('/experiment/results/markov-sparse-build.json')
    if output.exists():
        raise ValueError('Fresh build report required')
    factory = root / SOURCE
    candidate = transform(factory.read_bytes())
    report = dict(passed=False, source_before=digest(factory),
        binaries_before={name: digest(root / name) for name in BINARY_PATHS},
        builders={name: digest(Path(__file__).with_name(name)) for name in BUILDERS},
        jobs=8, timeout_seconds=1800, device_access=False)
    try:
        report['registration'] = restore_registrations(root)
        factory.write_bytes(candidate)
        report['source_after'] = digest(factory)
        subprocess.run(['ninja', '-C', str(root / 'build_Release'), '-j', '8', 'ttnncpp'], check=True, timeout=1800)
        built, loaded = [root / name for name in reversed(BINARY_PATHS)]
        if built.resolve() != loaded.resolve():
            shutil.copy2(built, loaded)
        report['binaries_after'] = {name: digest(root / name) for name in BINARY_PATHS}
        if factory.read_bytes() != candidate:
            raise ValueError('Factory changed during build')
        subprocess.run([sys.executable, '-c', 'import ttnn; assert callable(ttnn.sparse_matmul); '
            'assert callable(ttnn.transformer.attn_decode_prep)'], check=True, timeout=60)
        report['import_passed'] = True
        report['passed'] = True
    finally:
        output.write_text(json.dumps(report, indent=2) + '\n')
    validate(root, output)


if __name__ == '__main__':
    main()
