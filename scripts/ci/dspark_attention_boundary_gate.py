"""Exact mixed draft/target simulator admission for the guarded attention header."""

import hashlib
import json
from pathlib import Path


RUN_ID = 34830639528
REPORT_SHA256 = '5e469f628570cc02e4fb256a9bc475719f8e7ce9cb90c15ec267bc1d526fb18d'


def qualify(directory, report_path):
    payload = Path(report_path).read_bytes()
    if hashlib.sha256(payload).hexdigest() != REPORT_SHA256:
        raise ValueError('Exact completed mixed-attention simulation required')
    report = json.loads(payload)
    expected = dict(passed=True, closed_cleanly=True, backend='simulator', context=128,
        capacity=384, stage='complete', candidate='draft-target-header-boundary',
        numerical_tolerances=dict(rtol=.01, atol=.01), performance_qualified=False)
    if any(report.get(name) != value for name, value in expected.items()):
        raise ValueError('Complete unchanged-tolerance draft simulation required')
    if (report['sources'] != report['sources_after'] or report['native_sources'] != report['native_sources_after']
            or report['shared_active_header_sha256'] != report['kernel_audit']['patched']['compute_common.hpp']):
        raise ValueError('Both kernels must use the same unchanged guarded header')
    target = report['target_attention']
    if (target.get('passed') is not True or target.get('closed') is not True
            or target.get('backend') != 'simulator' or target.get('sources') != target.get('sources_after')
            or target.get('stale_controls') != 2 or target.get('mask_poison_controls') != 8):
        raise ValueError('Complete native-versus-folded target controls required')
    for name, count in (('checks', 8), ('mask_checks', 16), ('source_checks', 4), ('unpoisoned_replay', 2)):
        values = target.get(name, [])
        if len(values) != count or any(value.get('exact') is not True for value in values):
            raise ValueError('Exact target controls required: ' + name)
    for dependencies in (report['sources'], report['candidate_sources'], target['sources'],
            report['factory_build']['builders'], report['factory_build']['ladder_builders']):
        for name, checksum in dependencies.items():
            if hashlib.sha256((Path(directory) / name).read_bytes()).hexdigest() != checksum:
                raise ValueError('Mixed-attention simulator dependency changed: ' + name)
    return dict(run_id=RUN_ID, report_sha256=REPORT_SHA256, simulator_qualified=True,
        hardware_qualified=False, performance_qualified=False)
