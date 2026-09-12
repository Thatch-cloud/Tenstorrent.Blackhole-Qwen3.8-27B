"""Bounded factory rebuild inside the disposable CPU-only simulator container."""

import json
import os
from pathlib import Path
import shutil
import subprocess

from dspark_hardware_gate import digest
from dspark_fp32_intermediates import SOURCE, SOURCE_SHA256, ANCHOR, REPLACEMENT, transform


def validate_manifest(root, output):
    report = json.loads(Path(output).read_text())
    root = Path(root)
    if report.get('passed') is not True or report.get('source_before') != SOURCE_SHA256:
        raise ValueError('Completed pinned factory rebuild required')
    source = (root / SOURCE).read_bytes()
    if source.count(REPLACEMENT.encode()) != 1:
        raise ValueError('Unique rebuilt factory variant required')
    original = source.replace(REPLACEMENT.encode(), ANCHOR.encode())
    if transform(original) != source or digest(root / SOURCE) != report.get('source_after'):
        raise ValueError('Rebuilt factory differs from exact transformation')
    expected = {'build_Release/lib/_ttnncpp.so', 'build_Release/ttnn/_ttnncpp.so'}
    binaries = report.get('binaries_after', {})
    if set(binaries) != expected or len(set(binaries.values())) != 1:
        raise ValueError('Both binary paths must contain the rebuilt library')
    if any(digest(root / name) != value for name, value in binaries.items()):
        raise ValueError('Rebuilt binary changed')
    for name in ('dspark_fp32_build.py', 'dspark_fp32_intermediates.py'):
        if report.get('builders', {}).get(name) != digest(Path(__file__).with_name(name)):
            raise ValueError('Factory builder changed')
    return report


def main():
    if (os.environ.get('QWEN_SIM_ONLY') != '1' or not os.environ.get('TT_METAL_SIMULATOR')
            or os.environ.get('QWEN_SIM_CASE') != 'dspark-native-8k-attention'
            or Path('/dev/tenstorrent').exists() or os.environ.get('QWEN_CARDS_ALLOCATED') == '1'
            or os.environ.get('TT_METAL_HOME') != '/opt/tt-metal'):
        raise ValueError('Dedicated device-free simulator container required')
    root = Path('/opt/tt-metal')
    output = Path('/experiment/results/dspark-fp32-build.json')
    if output.exists():
        raise ValueError('Fresh build manifest required')
    factory = root / SOURCE
    before = factory.read_bytes()
    candidate = transform(before)
    binaries = ('build_Release/lib/_ttnncpp.so', 'build_Release/ttnn/_ttnncpp.so')
    report = dict(passed=False, source_before=digest(factory),
        binaries_before={name: digest(root / name) for name in binaries},
        builders={name: digest(Path(__file__).with_name(name)) for name in
            ('dspark_fp32_build.py', 'dspark_fp32_intermediates.py')},
        jobs=8, timeout_seconds=1800, device_access=False)
    try:
        factory.write_bytes(candidate)
        report['source_after'] = digest(factory)
        subprocess.run(['ninja', '-C', str(root / 'build_Release'), '-j', '8', 'ttnncpp'],
            check=True, timeout=1800)
        built, loaded = [root / name for name in reversed(binaries)]
        if built.resolve() != loaded.resolve():
            shutil.copy2(built, loaded)
        report['binaries_after'] = {name: digest(root / name) for name in binaries}
        if factory.read_bytes() != candidate:
            raise ValueError('Factory source changed during build')
        report['passed'] = True
    finally:
        output.write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
