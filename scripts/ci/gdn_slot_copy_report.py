"""Require the complete two-chip slot transport matrix and exact executed sources."""

import hashlib
import json
from pathlib import Path


def validate(report, directory):
    expected = {(slot, direction, ordinal, operand, name, chip)
        for slot in (0, 1, 7) for direction in ('save', 'restore')
        for ordinal in (-1, 0, 1, 2) for operand in range(5)
        for name in ('source_unchanged', 'destination_all_slots') for chip in (0, 1)}
    if (report.get('passed') is not True or report.get('closed_cleanly') is not True
            or report.get('backend') != 'simulator' or report.get('model_integrated') is not False
            or report.get('performance_qualified') is not False):
        raise ValueError('Clean simulator-only evidence required')
    observed = []
    for check in report.get('checks', []):
        if check.get('exact') is not True:
            raise ValueError('Every source and complete destination must be exact')
        observed.append(tuple(check.get(name) for name in ('slot', 'direction', 'ordinal', 'operand', 'name', 'chip')))
    if len(observed) != len(expected) or set(observed) != expected:
        raise ValueError('Complete unique slot save/restore replay matrix required')
    directory = Path(directory)
    sources = {name: hashlib.sha256((directory / name).read_bytes()).hexdigest()
        for name in ('history-append-probe.py', 'gdn_slot_copy.py', 'gdn_slot_copy.cpp', 'gdn_state_copy.py')}
    if (report.get('sources') != sources
            or (directory / 'history-append-probe.py').read_bytes() != (directory / 'gdn-slot-copy-probe.py').read_bytes()):
        raise ValueError('Executed slot-copy sources differ from reviewed sources')
    return dict(passed=True, checks=len(expected), model_integrated=False, performance_qualified=False)


if __name__ == '__main__':
    import sys

    print(json.dumps(validate(json.loads(Path(sys.argv[1]).read_text()), sys.argv[2])))
