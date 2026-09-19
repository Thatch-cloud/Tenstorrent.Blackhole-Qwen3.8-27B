"""Validate complete explicit-slot DMA evidence; does not admit hardware or model use."""

import hashlib
import json
from pathlib import Path


SOURCES = ('gdn-slot-copy-probe.py', 'gdn_slot_copy.py', 'gdn_slot_copy.cpp',
           'gdn_state_copy.py', 'attention_batch.py')


def validate(report, source_root, exit_status):
    if (str(exit_status).strip() != '0' or report.get('passed') is not True
            or report.get('closed_cleanly') is not True or report.get('backend') != 'simulator'
            or report.get('performance_qualified') is not False
            or report.get('model_integrated') is not False):
        raise ValueError('Clean successful simulator-only state-copy evidence required')
    root = Path(source_root)
    hashes = {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in SOURCES}
    if report.get('sources') != hashes or report.get('sources_after') != hashes:
        raise ValueError('Exact unchanged candidate sources required')
    expected = {(slot, direction, phase, role, tensor, chip)
                for slot in range(8) for direction in ('load', 'store')
                for phase in ('eager', 'replay', 'changed-replay')
                for role in ('source', 'destination') for tensor in range(5) for chip in range(2)}
    observed = set()
    checks = report.get('checks')
    if not isinstance(checks, list) or len(checks) != len(expected):
        raise ValueError('Complete eight-slot two-direction two-chip matrix required')
    for check in checks:
        if (not isinstance(check, dict) or check.get('exact') is not True
                or any(type(check.get(field)) is not int for field in ('slot', 'tensor', 'chip'))
                or any(type(check.get(field)) is not str for field in ('direction', 'phase', 'role'))):
            raise ValueError('Exact typed state-copy check required')
        key = tuple(check[field] for field in ('slot', 'direction', 'phase', 'role', 'tensor', 'chip'))
        if key not in expected or key in observed:
            raise ValueError('Unexpected or duplicate state-copy check')
        observed.add(key)
    return dict(checks=len(observed), slots=8, chips=2, sources=hashes,
        simulator_component_qualified=True, hardware_qualified=False,
        model_integrated=False, performance_qualified=False)


def qualify(report_path, source_root):
    report_path = Path(report_path)
    raw = report_path.read_bytes()
    result = validate(json.loads(raw), source_root, report_path.with_suffix('.exit-status').read_text())
    result['report_sha256'] = hashlib.sha256(raw).hexdigest()
    return result
