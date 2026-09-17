"""Bounded factory rebuild inside the disposable CPU-only simulator container."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from dspark_hardware_gate import digest
from dspark_fp32_intermediates import SOURCE, SOURCE_SHA256, ANCHOR, REPLACEMENT, transform
from sdpa_graft_build import audit as audit_registrations, implementation_sources


def restore_registrations(root):
    evidence = audit_registrations(root)
    directory = root / 'ttnn/cpp/ttnn/operations/transformer'
    registration = directory / 'sources.cmake'
    source = registration.read_bytes()
    anchor = b'set(TTNN_OP_TRANSFORMER_SRCS\n'
    if source.count(anchor) != 1:
        raise ValueError('Unique transformer registration anchor required')
    additions = ''.join('    ' + name + '\n' for name in implementation_sources()).encode()
    registration.write_bytes(source.replace(anchor, anchor + additions))
    evidence['source_after'] = digest(registration)
    return evidence


def restore_factory_source(source, replacement):
    return source.replace(replacement.encode(), ANCHOR.encode())


def validate_manifest(root, output):
    report = json.loads(Path(output).read_text())
    root = Path(root)
    if report.get('passed') is not True or report.get('source_before') != SOURCE_SHA256:
        raise ValueError('Completed pinned factory rebuild required')
    source = (root / SOURCE).read_bytes()
    enabled = report.get('factory_enabled')
    if type(enabled) is not bool:
        raise ValueError('Explicit factory variant evidence required')
    replacement = REPLACEMENT if enabled else REPLACEMENT.replace(
        'qwen_draft_fp32_intermediates =\n', 'qwen_draft_fp32_intermediates = false &&\n')
    if source.count(replacement.encode()) != 1:
        raise ValueError('Unique rebuilt factory variant required')
    original = restore_factory_source(source, replacement)
    if transform(original, enabled=enabled) != source or digest(root / SOURCE) != report.get('source_after'):
        raise ValueError('Rebuilt factory differs from exact transformation')
    expected = {'build_Release/lib/_ttnncpp.so', 'build_Release/ttnn/_ttnncpp.so'}
    binaries = report.get('binaries_after', {})
    if set(binaries) != expected or len(set(binaries.values())) != 1:
        raise ValueError('Both binary paths must contain the rebuilt library')
    if any(digest(root / name) != value for name, value in binaries.items()):
        raise ValueError('Rebuilt binary changed')
    registration = report.get('registration', {})
    directory = root / 'ttnn/cpp/ttnn/operations/transformer'
    if digest(directory / 'sources.cmake') != registration.get('source_after'):
        raise ValueError('Rebuilt operation registration changed')
    if set(registration.get('implementation_sources', {})) != set(implementation_sources()):
        raise ValueError('Complete transformer implementations required')
    for name, value in registration['implementation_sources'].items():
        if digest(directory / name) != value:
            raise ValueError('Registered operation implementation changed')
    if report.get('import_passed') is not True:
        raise ValueError('Rebuilt TTNN import check required')
    for name in ('dspark_fp32_build.py', 'dspark_fp32_intermediates.py'):
        if report.get('builders', {}).get(name) != digest(Path(__file__).with_name(name)):
            raise ValueError('Factory builder changed')
    return report


def main(*, hardware=False):
    if type(hardware) is not bool:
        raise ValueError('Explicit build backend required')
    if hardware:
        from dspark_ladder_backend import require_backend
        require_backend(os.environ, hardware=True, device_present=Path('/dev/tenstorrent').exists())
        if os.environ.get('TT_METAL_HOME') != '/opt/tt-metal':
            raise ValueError('Disposable pinned runtime required')
    elif (os.environ.get('QWEN_SIM_ONLY') != '1' or not os.environ.get('TT_METAL_SIMULATOR')
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
    control = os.environ.get('QWEN_DRAFT_FP32_CONTROL', '0')
    if control not in ('0', '1'):
        raise ValueError('Explicit rebuilt control flag required')
    enabled = control == '0'
    candidate = transform(before, enabled=enabled)
    binaries = ('build_Release/lib/_ttnncpp.so', 'build_Release/ttnn/_ttnncpp.so')
    report = dict(passed=False, precision_variant='stats-only', factory_enabled=enabled, source_before=digest(factory),
        binaries_before={name: digest(root / name) for name in binaries},
        builders={name: digest(Path(__file__).with_name(name)) for name in
            ('dspark_fp32_build.py', 'dspark_fp32_intermediates.py')},
        jobs=8, timeout_seconds=1800, device_access=False)
    try:
        report['registration'] = restore_registrations(root)
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
        subprocess.run([sys.executable, '-c',
            'import ttnn; assert callable(ttnn.transformer.attn_decode_prep); '
            'assert callable(ttnn.transformer.scaled_dot_product_attention)'], check=True, timeout=60)
        report['import_passed'] = True
        report['passed'] = True
    finally:
        output.write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
