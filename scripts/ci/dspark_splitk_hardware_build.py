"""Bounded source-keyed hardware build for the simulator-qualified split-K kernel."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from dspark_fp32_build import restore_registrations
from dspark_hardware_gate import digest
from dspark_ladder_backend import require_backend
from dspark_runtime_cache import IMAGE, cache_key, store_entry
from dspark_splitk_fp32_factory import SOURCE, transform
from dspark_splitk_sim_gate import qualify
from sdpa_graft_build import implementation_sources
from dspark_splitk_compile_cache import find_entry


BUILDERS = ('dspark_splitk_hardware_build.py', 'dspark_splitk_fp32_factory.py',
    'dspark_splitk_sim_gate.py', 'dspark_runtime_cache.py', 'dspark_fp32_build.py',
    'sdpa_graft_build.py', 'dspark_ladder_backend.py', 'feature_projection.py',
    'dspark_hardware_gate.py', 'dspark_attention_8k_gate.py', 'dspark_fp32_intermediates.py',
    'dspark_splitk_compile_cache.py')
BINARIES = ('build_Release/lib/_ttnncpp.so', 'build_Release/ttnn/_ttnncpp.so')
REPORT = '/experiment/results/dspark-splitk-hardware-build.json'


def fingerprints(directory):
    return {name: digest(Path(directory) / name) for name in BUILDERS}


def validate_build(root, directory, report_path, simulator_report):
    admission = qualify(directory, simulator_report)
    report = json.loads(Path(report_path).read_text())
    if (report.get('passed') is not True or report.get('import_passed') is not True
            or report.get('backend') != 'hardware' or report.get('image') != IMAGE
            or report.get('builders') != fingerprints(directory)
            or report.get('simulator_report_sha256') != admission['report_sha256']
            or report.get('source_after') != admission['factory_source_after']
            or digest(Path(root) / SOURCE) != admission['factory_source_after']):
        raise ValueError('Exact simulator-qualified hardware factory build required')
    binaries = report.get('binaries', {})
    if set(binaries) != set(BINARIES) or len(set(binaries.values())) != 1:
        raise ValueError('Two identical rebuilt hardware libraries required')
    for name, checksum in binaries.items():
        if digest(Path(root) / name) != checksum:
            raise ValueError('Loaded split-K hardware binary changed')
    registration = report.get('registration', {})
    transformer = Path(root) / 'ttnn/cpp/ttnn/operations/transformer'
    if digest(transformer / 'sources.cmake') != registration.get('source_after'):
        raise ValueError('Hardware operation registration changed')
    sources = registration.get('implementation_sources', {})
    if set(sources) != set(implementation_sources()):
        raise ValueError('Complete hardware operation implementations required')
    for name, checksum in sources.items():
        if digest(transformer / name) != checksum:
            raise ValueError('Hardware operation implementation changed')
    return report


def main():
    require_backend(os.environ, hardware=True, device_present=Path('/dev/tenstorrent').exists())
    if os.environ.get('QWEN_SPLITK_ATTENTION') != '1' or os.environ.get('TT_METAL_HOME') != '/opt/tt-metal':
        raise ValueError('Explicit disposable split-K hardware build required')
    root, directory = Path('/opt/tt-metal'), Path(__file__).parent
    simulator_report = directory / 'dspark-splitk-simulator.json'
    admission = qualify(directory, simulator_report)
    output = Path(REPORT)
    if output.exists():
        raise ValueError('Fresh hardware build report required')
    factory = root / SOURCE
    if digest(factory) != admission['factory_source_before']:
        raise ValueError('Exact pre-transform decode factory required')
    report = dict(passed=False, import_passed=False, backend='hardware', image=IMAGE,
        builders=fingerprints(directory), simulator_report_sha256=admission['report_sha256'],
        source_before=digest(factory), opens_devices=False, jobs=8, build_timeout_seconds=330)
    try:
        factory.write_text(transform(factory.read_text()))
        report['source_after'] = digest(factory)
        if report['source_after'] != admission['factory_source_after']:
            raise ValueError('Hardware factory differs from qualified simulator factory')
        report['registration'] = restore_registrations(root)
        inputs = dict(image=IMAGE, backend='hardware', builders=report['builders'],
            simulator_report=admission['report_sha256'], factory=report['source_after'],
            registration=report['registration'])
        cache = Path('/experiment-cache/splitk-hardware-v1')
        inputs, manifest = find_entry(cache, inputs)
        report['compile_inputs'] = inputs
        report.update(cache_hit=manifest is not None, cache_key=cache_key(inputs))
        if manifest is None:
            subprocess.run(['ninja', '-C', str(root / 'build_Release'), '-j', '8', 'ttnncpp'],
                check=True, timeout=330)
            built = root / BINARIES[1]
        else:
            built = cache / cache_key(inputs) / '_ttnncpp.so'
        for name in BINARIES:
            destination = root / name
            if built.resolve() != destination.resolve():
                shutil.copy2(built, destination)
        subprocess.run([sys.executable, '-c',
            'import ttnn; assert callable(ttnn.transformer.scaled_dot_product_attention_decode)'],
            check=True, timeout=20)
        report['import_passed'] = True
        if digest(factory) != admission['factory_source_after']:
            raise ValueError('Hardware factory changed during build')
        store_entry(cache, inputs, root / BINARIES[1])
        report['binaries'] = {name: digest(root / name) for name in BINARIES}
        report['passed'] = True
    finally:
        output.write_text(json.dumps(report, indent=2) + '\n')
    validate_build(root, directory, output, simulator_report)


if __name__ == '__main__':
    main()
