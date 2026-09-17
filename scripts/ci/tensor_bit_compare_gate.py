"""Independent gate for exact comparator controls; no publication qualification."""

import hashlib
import json
from pathlib import Path


SOURCES = ('tensor-bit-compare-probe.py', 'tensor_bit_compare.py', 'tensor_bit_compare.cpp',
    'attention_batch.py', 'feature_projection.py', 'gdn_multitoken_conv.py')


def validate(report, directory, exit_status):
    if (str(exit_status).strip() != '0' or report.get('passed') is not True
            or report.get('closed_cleanly') is not True or report.get('stage') != 'complete'
            or report.get('backend') != 'simulator' or 'error' in report):
        raise ValueError('Complete closed simulator comparator execution required')
    sources = {name: hashlib.sha256((Path(directory) / name).read_bytes()).hexdigest() for name in SOURCES}
    if report.get('sources') != sources or report.get('sources_after') != sources:
        raise ValueError('Exact tested comparator and control source fingerprints required')
    expected = []
    for mode in ('eager', 'replay'):
        for width in (32, 2048):
            cases = [(None, case) for case in range(3)] if mode == 'eager' else list(enumerate((0, 1, 2, 0)))
            for repetition, case in cases:
                for chip in range(2):
                    expected.append(dict(width=width, case=case, mode=mode, repetition=repetition,
                        chip=chip, exact=True, inputs_unchanged=True, poisoned_counters_replaced=True))
    if report.get('checks') != expected:
        raise ValueError('Every known mismatch, padding, worker and replay control must pass')
    return dict(checks=28, scope='Exact BF16 physical comparator controls only', publication_qualified=False)


def qualify(directory):
    directory = Path(directory)
    path = directory / 'tensor-bit-compare-simulator.json'
    result = validate(json.loads(path.read_text()), directory, path.with_suffix('.exit-status').read_text())
    return dict(result, report_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
