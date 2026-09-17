"""Require complete, source-bound wide-page simulator evidence before hardware use."""

import hashlib
import json
from pathlib import Path

from ordered_cache import HASHES


SOURCES = ('ladder-cache-probe.py', 'frozen_ladder_ordered_cache.py', 'frozen_context_geometry.py',
    'ordered_cache.py', 'attention_batch.py')
RUNTIME = '9f9cd4fd590f4b606bd0981a4fe0b6403eb38ec9'


def validate(report, context):
    if (type(context) is not int or context not in (65536, 131072, 261888)
            or type(report.get('context')) is not int or report['context'] != context
            or any(report.get(key) is not True for key in ('passed', 'closed_cleanly'))
            or report.get('backend') != 'simulator' or report.get('stage') != 'complete'
            or any(report.get(key) is not False for key in ('performance_qualified', 'model_integrated'))
            or report.get('error') or report.get('native_hashes') != HASHES
            or set(report.get('sources', {})) != set(SOURCES)
            or report.get('sources') != report.get('sources_after')
            or set(report.get('generated_hashes', {})) != set(HASHES)):
        raise ValueError('Complete unchanged-source simulator cache qualification required')
    for checksum in (*report['sources'].values(), *report['generated_hashes'].values()):
        if (type(checksum) is not str or len(checksum) != 64
                or any(character not in '0123456789abcdef' for character in checksum)):
            raise ValueError('Explicit source fingerprints required')
    expected = {(seed, chip, name) for seed in (0, 1) for chip in (0, 1)
        for name in ('replay', 'input_unchanged')}
    expected.update((0, chip, 'eager') for chip in (0, 1))
    seen = set()
    for check in report.get('checks', []):
        if (check.get('exact') is not True or check.get('context') != context
                or any(type(check.get(key)) is not int for key in ('context', 'seed', 'chip'))):
            raise ValueError('Explicit exact check identities required')
        identity = (check.get('seed'), check.get('chip'), check.get('name'))
        if identity not in expected or identity in seen:
            raise ValueError('Unexpected or duplicate cache check')
        seen.add(identity)
    if seen != expected:
        raise ValueError('All eager, changed-input replay and unchanged-input checks required')
    return report


def qualify(directory, evidence, expected_sha256, context):
    directory, evidence = Path(directory), Path(evidence)
    raw = (evidence / 'ladder-cache.json').read_bytes()
    if (hashlib.sha256(raw).hexdigest() != expected_sha256
            or (evidence / 'ladder-cache.exit-status').read_text().strip() != '0'
            or (evidence / 'simulator-runtime.txt').read_text().strip() != RUNTIME):
        raise ValueError('Pinned clean simulator cache report and runtime required')
    report = validate(json.loads(raw), context)
    for name, checksum in report['sources'].items():
        if hashlib.sha256((directory / name).read_bytes()).hexdigest() != checksum:
            raise ValueError('Simulator-qualified cache source changed: ' + name)
    return report
