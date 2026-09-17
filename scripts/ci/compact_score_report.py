"""Validate the complete compact-selection simulator evidence matrix."""

import argparse
import hashlib
import json
from pathlib import Path


def validate(report, source_directory):
    if (report.get('passed') is not True or report.get('closed_cleanly') is not True
            or report.get('backend') != 'simulator' or report.get('vocabulary') != 248320
            or report.get('stage') != 'complete' or report.get('performance_qualified') is not False
            or report.get('error')):
        raise ValueError('Complete successful full-vocabulary simulator evidence required')
    expected = {(pattern, step, mode, chip)
                for mode, patterns in (('eager', (0,)), ('replay', (1, 2, 3, 0)))
                for pattern in patterns for step in (0, 14) for chip in (0, 1)}
    for name, wanted, fields in (
            ('checks', expected, ('pattern', 'step', 'mode', 'chip')),
            ('immutable_checks', {key + (operand,) for key in expected for operand in (0, 1)},
             ('pattern', 'step', 'mode', 'chip', 'operand'))):
        records = report.get(name, [])
        if (len(records) != len(wanted) or any(record.get('exact') is not True for record in records)
                or {tuple(record.get(field) for field in fields) for record in records} != wanted):
            raise ValueError('Missing, duplicate or failed evidence: ' + name)
    names = ('gdn-output-grid-probe.py', 'compact_score_device.py', 'compact_score_io.cpp',
             'compact_score_compute.cpp', 'compact_score_reduce.cpp', 'dspark_score_layout.py',
             'dspark_score_layout_io.cpp', 'dspark_score_layout_compute.cpp', 'attention_batch.py',
             'gdn_multitoken_conv.py')
    sources = report.get('sources', {})
    if set(sources) != set(names) or sources != report.get('sources_after'):
        raise ValueError('Complete unchanged source closure required')
    for name in names:
        local = 'compact-score-probe.py' if name == 'gdn-output-grid-probe.py' else name
        digest = hashlib.sha256((Path(source_directory) / local).read_bytes()).hexdigest()
        if sources[name] != digest:
            raise ValueError('Simulator source differs from candidate: ' + name)
    return dict(simulator_qualified=True, hardware_qualified=False, checks=len(expected),
                immutable_checks=len(expected) * 2, sources=sources)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('report', type=Path)
    options = parser.parse_args()
    print(json.dumps(validate(json.loads(options.report.read_text()), Path(__file__).parent), indent=2))
