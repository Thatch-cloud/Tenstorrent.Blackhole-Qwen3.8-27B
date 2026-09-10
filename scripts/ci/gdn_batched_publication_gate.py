"""Independent complete-matrix gate for batched publication simulator evidence."""

import hashlib
import json
from pathlib import Path


SOURCES = ('gdn-batched-publication-probe.py', 'gdn_commit_dma.py', 'gdn_commit_dma.cpp',
    'gdn_commit_batched_dma.py', 'gdn_commit_batched_dma.cpp', 'attention_batch.py',
    'feature_projection.py', 'gdn_multitoken_conv.py')
PADDED = (1, 2, 3, 4, 6, 7, 8, 9, 11, 12, 13, 14, 16, 17, 18, 19)


def validate(report, directory, layers, exit_status):
    if type(layers) is not int or layers not in (1, 48):
        raise ValueError('Explicit one-layer or complete 48-layer simulator geometry required')
    if (str(exit_status).strip() != '0' or report.get('passed') is not True
            or report.get('closed_cleanly') is not True or report.get('padding_audited') is not True
            or report.get('backend') != 'simulator' or report.get('stage') != 'complete'
            or report.get('rows') != 16 or report.get('layers') != layers or 'error' in report):
        raise ValueError('Complete clean simulator execution and physical padding audit required')
    current = {name: hashlib.sha256((Path(directory) / name).read_bytes()).hexdigest() for name in SOURCES}
    if report.get('sources') != current or report.get('sources_after') != current:
        raise ValueError('All tested kernel, adapter and probe sources must remain unchanged')
    prefixes = tuple(range(17)) if layers == 1 else (0, 1, 8, 16)
    schedule = [(arm, pattern, prefix, None) for prefix in prefixes for pattern in range(2)
        for arm in ('native', 'candidate')]
    schedule += [('replay', pattern, prefix, repetition) for prefix in prefixes
        for repetition, pattern in enumerate((0, 1, 0))]
    checks, padding, poison = [], [], []
    for arm, pattern, prefix, repetition in schedule:
        for layer in range(layers):
            for chip in range(2):
                entry = dict(arm=arm, pattern=pattern, prefix=prefix, repetition=repetition,
                    layer=layer, chip=chip, exact=True)
                checks.append(entry)
                padding.extend(dict(entry, operand=operand) for operand in PADDED)
                poison.append(dict(pattern=pattern, layer=layer, chip=chip, exact=True))
    if report.get('checks') != checks or report.get('padding_checks') != padding or report.get('poison_checks') != poison:
        raise ValueError('Complete ordered native/candidate/replay, padding and poison matrices required')
    return dict(layers=layers, rows=16, checks=len(checks), padding_checks=len(padding),
        poison_checks=len(poison), hardware_qualified=False, serving_qualified=False)


def qualify(directory):
    directory = Path(directory)
    results = []
    for layers in (1, 48):
        path = directory / f'gdn-batched-publication-{layers}-simulator.json'
        report = json.loads(path.read_text())
        result = validate(report, directory, layers, path.with_suffix('.exit-status').read_text())
        results.append(dict(result, report_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    return dict(scope='Synthetic publication only; learned-state continuation and full requests remain required',
        simulator_results=results, hardware_qualified=False, serving_qualified=False)
