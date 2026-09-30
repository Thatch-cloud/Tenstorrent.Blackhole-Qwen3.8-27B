"""Validate the complete sliding-transport matrix, not model or performance acceptance."""

import hashlib
import json
from pathlib import Path


def validate(report, directory, *, staged_probe=True):
    expected = {(history, prefix, ordinal, name, chip)
        for history, prefix in ((31, 2), (2047, 2), (2048, 1), (2048, 16), (2048, 32))
        for ordinal in (-1, 0, 1, 2)
        for name in ('active_unchanged', 'delta_unchanged', 'sliding_output')
        for chip in (0, 1)}
    if (report.get('passed') is not True or report.get('closed_cleanly') is not True
            or report.get('backend') != 'simulator' or report.get('performance_qualified') is not False
            or report.get('model_integrated') is not False):
        raise ValueError('Clean simulator-only evidence required')
    observed = []
    for check in report.get('checks', []):
        if check.get('exact') is not True:
            raise ValueError('Every output and immutable-input comparison must be exact')
        observed.append(tuple(check.get(name) for name in ('history', 'prefix', 'ordinal', 'name', 'chip')))
    if len(observed) != len(expected) or set(observed) != expected:
        raise ValueError('Complete unique eager and replay matrix required')
    directory = Path(directory)
    sources = {name: hashlib.sha256((directory / ('draft-kv-slide-probe.py'
        if name == 'history-append-probe.py' and not staged_probe else name)).read_bytes()).hexdigest()
        for name in ('history-append-probe.py', 'draft_kv_slide.py', 'draft_kv_slide.cpp')}
    if report.get('sources') != sources:
        raise ValueError('Executed sources differ from candidate')
    if staged_probe and (directory / 'history-append-probe.py').read_bytes() != (directory / 'draft-kv-slide-probe.py').read_bytes():
        raise ValueError('Staged probe differs from reviewed probe')
    return dict(passed=True, checks=len(expected), performance_qualified=False, model_integrated=False)


if __name__ == '__main__':
    import sys

    print(json.dumps(validate(json.loads(Path(sys.argv[1]).read_text()), sys.argv[2])))
