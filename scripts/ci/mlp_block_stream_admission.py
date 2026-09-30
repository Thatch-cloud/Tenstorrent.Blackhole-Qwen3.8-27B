"""Validate transport evidence before attempting numerical MLP qualification."""

import hashlib
import argparse
import json
from pathlib import Path

from mlp_block_stream import geometry


SOURCES = ('mlp_block_stream.py', 'mlp_block_stream.cpp', 'mlp-block-stream-probe.py')


def admit_transport(report, source_root):
    if (report.get('passed') is not True or report.get('closed_cleanly') is not True
            or report.get('backend') != 'simulator' or 'error' in report):
        raise ValueError('Successful clean simulator transport execution required')
    if any(report.get(field) is not False for field in
            ('mlp_qualified', 'performance_qualified', 'hardware_qualified')):
        raise ValueError('Transport evidence cannot qualify arithmetic or performance')
    sources = {name: hashlib.sha256((Path(source_root) / name).read_bytes()).hexdigest()
        for name in SOURCES}
    if report.get('sources') != sources or report.get('sources_after') != sources:
        raise ValueError('Transport source identity changed or differs from candidate')
    expected = {(pairs, blocks, pattern, chip): geometry(pairs, blocks)['stream_pages']
        for pairs, blocks in ((8, 2), (272, 1)) for pattern in (0, 1) for chip in (0, 1)}
    checks = report.get('checks')
    if not isinstance(checks, list) or len(checks) != len(expected):
        raise ValueError('Complete transport matrix required')
    seen = set()
    for check in checks:
        if not isinstance(check, dict):
            raise ValueError('Structured transport check required')
        key = tuple(check.get(field) for field in ('pairs', 'blocks', 'pattern', 'chip'))
        if (any(type(value) is not int for value in key) or key not in expected or key in seen
                or type(check.get('pages')) is not int or check['pages'] != expected[key]
                or check.get('source_unchanged') is not True or check.get('exact_all_words') is not True):
            raise ValueError('Exact unique transport geometry, pattern and chip checks required')
        seen.add(key)
    return dict(transport_qualified=True, mlp_qualified=False, hardware_qualified=False,
        performance_qualified=False, sources=sources, checks=len(seen))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, required=True)
    options = parser.parse_args()
    result = admit_transport(json.loads(options.report.read_text()), Path(__file__).parent)
    print(json.dumps(result, sort_keys=True))


if __name__ == '__main__':
    main()
