"""Bounded factory rebuild inside the disposable CPU-only simulator container."""

import json
import os
from pathlib import Path
import shutil
import subprocess

from dspark_hardware_gate import digest
from dspark_fp32_intermediates import SOURCE, transform


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
