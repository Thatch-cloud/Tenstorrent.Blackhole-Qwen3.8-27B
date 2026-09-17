"""Pinned sixteen-worker simulator evidence; no model or speed qualification."""

import json
from pathlib import Path

from dspark_attention_8k_gate import require_matrix
from dspark_hardware_gate import digest
from dspark_splitk_maxima_gate import FACTORY_SHA256, qualify as qualify_control
from dspark_splitk_sim_gate import verify_sources


REPORT_SHA256 = '5b4f00edc06a1c449574c1a81368dcece1376cd077ed4924fb81f927a62e3448'


def qualify(directory, report_path, baseline_path=None):
    baseline_path = Path(directory) / 'dspark-splitk-simulator.json' if baseline_path is None else Path(baseline_path)
    qualify_control(directory, baseline_path)
    control = json.loads(baseline_path.read_bytes())
    if digest(report_path) != REPORT_SHA256:
        raise ValueError('Exact passing sixteen-worker simulator report required')
    report = json.loads(Path(report_path).read_bytes())
    if (report.get('passed') is not True or report.get('closed_cleanly') is not True
            or report.get('backend') != 'simulator' or report.get('capacity') != 384
            or report.get('positions') != [128, 369] or report.get('proposal_rows') != 15
            or report.get('candidate') != 'splitk-maxima-worker-limit16'
            or report.get('numerical_tolerances') != dict(rtol=.01, atol=.01)
            or report.get('splitk_execution_calls') != 3):
        raise ValueError('Complete unchanged simulator fixture required')
    expected = dict(control['diagnostic_override'], max_cores_per_head=16, local_maximum_storage='float32')
    call = dict(key_chunk_size=256, requested_worker_limit=8, selected_worker_limit=16,
        stripe_keys=False, fp32_dest_acc=True)
    if report.get('diagnostic_override') != expected or report.get('worker_calls') != [call] * 3:
        raise ValueError('Only the worker limit may change')
    for field, count, flag in (('input_checks', 48, 'exact'), ('layout_checks', 16, 'passed'),
            ('fixture_controls', 8, 'detected'), ('stale_controls', 2, 'detected')):
        checks = report.get(field, [])
        if len(checks) != count or any(check.get(flag) is not True for check in checks):
            raise ValueError('Complete input, layout and negative controls required')
    for mode, cases in (('eager', (0, 1)), ('replay', (1, 0))):
        records = report[mode + '_checks']
        require_matrix(records, ('ordinal', 'case', 'chip'),
            {(ordinal, case, chip) for ordinal, case in enumerate(cases) for chip in range(2)}, 'passed')
        if any(record.get('failed_elements') != 0 or record.get('numerical_close') is not True
                or mode == 'replay' and record.get('replay_exact') is not True for record in records):
            raise ValueError('Numerical and changing-input exact replay checks required')
    target = report['target_attention']
    factory = report['splitk_factory']
    if (target.get('passed') is not True or target.get('closed') is not True
            or factory.get('passed') is not True or factory.get('device_access') is not False
            or factory.get('source_after') != FACTORY_SHA256
            or factory.get('builders') != control['splitk_factory']['builders']):
        raise ValueError('Same admitted factory and independent native target gate required')
    for field in ('sources', 'native_sources'):
        if report.get(field) != report.get(field + '_after'):
            raise ValueError('Simulator sources changed')
    if target['sources'] != target['sources_after']:
        raise ValueError('Native target sources changed')
    for sources in (report['sources'], report['candidate_sources'], factory['builders'], target['sources']):
        verify_sources(directory, sources)
    return dict(report_sha256=REPORT_SHA256, worker_limit=16, key_chunk_size=256,
        factory_sha256=FACTORY_SHA256, simulator_qualified=True, hardware_qualified=False,
        full_request_qualified=False, performance_qualified=False, serving_qualified=False)
