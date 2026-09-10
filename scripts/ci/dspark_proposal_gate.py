"""Fail-closed fixed-storage proposal prerequisite; report pin is set only after independent simulation audit."""

import json
from pathlib import Path

from dspark_hardware_gate import digest


REPORT = 'dspark-fixed-attention-simulator.json'
SHA256 = None
COUNTS = dict(eager_checks=4, replay_checks=4, input_checks=48, layout_checks=16, fixture_controls=8, stale_controls=2)


def qualify(directory):
    directory = Path(directory)
    if SHA256 is None:
        raise ValueError('Fixed-capacity proposal simulation has not yet been independently qualified')
    path = directory / REPORT
    if digest(path) != SHA256 or path.with_suffix('.exit-status').read_text().strip() != '0':
        raise ValueError('Pinned clean fixed-capacity attention report required')
    report = json.loads(path.read_text())
    if (report.get('passed') is not True or report.get('closed_cleanly') is not True
            or report.get('backend') != 'simulator' or report.get('stage') != 'complete'
            or report.get('sources') != report.get('sources_after')
            or report.get('native_sources') != report.get('native_sources_after')
            or report.get('positions') != [4096, 4109] or report.get('capacity') != 4384
            or report.get('proposal_rows') != 15 or report.get('chunks') != [[0, 2048], [2048, 4096], [4096, 4416]]
            or report.get('numerical_tolerances') != dict(rtol=.01, atol=.01)
            or {name: len(report.get(name, [])) for name in COUNTS} != COUNTS):
        raise ValueError('Complete unchanged fixed-history numerical and changing-input replay matrix required')
    for name in COUNTS:
        flag = 'exact' if name == 'input_checks' else 'detected' if name in ('fixture_controls', 'stale_controls') else 'passed'
        if any(value.get(flag) is not True for value in report[name]):
            raise ValueError('Every fixed-capacity comparison must pass')
    for name, checksum in report['sources'].items():
        if name != '../../optimisation/sim/run-dispatch-probe.sh' and digest(directory / name) != checksum:
            raise ValueError('Changed simulated fixed-capacity source: ' + name)
    return {REPORT: SHA256}
