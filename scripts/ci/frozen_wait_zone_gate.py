"""Admission for diagnostic MLP scopes, not serving or throughput promotion."""

import hashlib
import json
from pathlib import Path


SIMULATOR_SHA256 = 'd246e6e3b23fb6a5615cafe261c90dc5e15db195cb0e78f4f6a47641584088fd'
HARDWARE_SHA256 = '08b18726cb651df3536c9143767d52432dca7fe585bfd23cb069d97f7677920e'
PROJECTION_SHA256 = '4c215d945e723ef3d8c2be1757e7c170bf21eebb01e204359243d766c6c7cf93'


def retained(path, expected):
    raw = Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError('Exact retained diagnostic report required: ' + str(path))
    return json.loads(raw)


def qualify(directory):
    from fused_t16_admission import qualify_simulator

    directory = Path(directory)
    baseline = qualify_simulator(directory)
    simulator = retained(directory / 'frozen-wait-zones-simulator.json', SIMULATOR_SHA256)
    hardware = retained(directory / 'frozen-wait-zones-hardware.json', HARDWARE_SHA256)
    if (simulator.get('passed') is not True or hardware.get('passed') is not True
            or hardware.get('numerics_passed') is not True or hardware.get('diagnostic_only') is not True
            or hardware.get('committed_tg') is not None
            or hardware.get('simulator_report_sha256') != SIMULATOR_SHA256
            or len(hardware.get('samples', [])) != 80):
        raise ValueError('Passed numerical and complete paired-marker evidence required')
    expected = dict(next(kernel for kernel in baseline['kernels'] if kernel['token_rows'] == 16))
    expected['reader_sha256'] = dict(simulator['kernels'][0]['reader_sha256'])
    if simulator.get('kernels') != [expected]:
        raise ValueError('Only reader diagnostic scopes may differ from the admitted projection')
    candidate = directory / 'frozen-wait-zone-candidate'
    hashes = dict(expected['reader_sha256'], **{'fused_1d.py': PROJECTION_SHA256})
    for name, checksum in hashes.items():
        if hashlib.sha256((candidate / name).read_bytes()).hexdigest() != checksum:
            raise ValueError('Diagnostic projection source changed: ' + name)
    return dict(report_sha256=SIMULATOR_SHA256, hardware_marker_sha256=HARDWARE_SHA256,
        kernels=[expected], passed=True, diagnostic_only=True, performance_qualified=False)
