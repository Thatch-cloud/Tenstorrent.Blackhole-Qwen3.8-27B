"""Direct FP32 staging admission from mixed-kernel simulation and unchanged draft outputs."""

import hashlib
import json
from pathlib import Path


RUN_ID = 34840912016
REPORT_SHA256 = 'a83bb4f482be40242a7dedf9ebed68310e4e2fd2028c9cb6689921b0d5df33e2'


def qualify(directory, report_path):
    payload = Path(report_path).read_bytes()
    if hashlib.sha256(payload).hexdigest() != REPORT_SHA256:
        raise ValueError('Exact completed mixed-attention simulation required')
    report = json.loads(payload)
    expected = dict(passed=True, closed_cleanly=True, backend='simulator', context=128,
        capacity=384, stage='complete', candidate='center-tile-fill',
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
    reference_outputs = {
        (0, 0): '398d4f644070f9dfa386d8b373fe6451cc973d82aec14a88e2f345cd962b0e78',
        (0, 1): 'b3fff750f9da675680a4b42bd17cda5dfbc4ebd61fdfab8dbd9a59050f11c57c',
        (1, 0): '9b283f9abbe870ef90ad588364b5a730cbbf3c07a1c016ee7411fda572bca95d',
        (1, 1): '64af03be8391e10273ab55dd6c4a9b3522fa3a6d702126759aa875eab3d84b82'}
    for field in ('eager_checks', 'replay_checks'):
        values = report[field]
        if (len(values) != 4 or any(value.get('passed') is not True for value in values)
                or {(value['case'], value['chip']): value['sha256'] for value in values} != reference_outputs):
            raise ValueError('Draft outputs must retain baseline simulator 34830639528 hashes')
    for dependencies in (report['sources'], report['candidate_sources'], target['sources'],
            report['factory_build']['builders'], report['factory_build']['ladder_builders']):
        for name, checksum in dependencies.items():
            if hashlib.sha256((Path(directory) / name).read_bytes()).hexdigest() != checksum:
                raise ValueError('Mixed-attention simulator dependency changed: ' + name)
    return dict(run_id=RUN_ID, report_sha256=REPORT_SHA256, simulator_qualified=True,
        hardware_qualified=False, performance_qualified=False)
