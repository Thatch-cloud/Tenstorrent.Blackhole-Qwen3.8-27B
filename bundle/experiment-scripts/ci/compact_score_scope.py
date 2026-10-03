"""Single-request compact feedback override; no global serving configuration."""

from contextlib import contextmanager
import hashlib
import importlib
from pathlib import Path
from unittest.mock import patch

from compact_score_hardware_sources import payloads
from compact_score_gate import REPORT_SHA256


@contextmanager
def scoped_compact_scores(admission, directory):
    import dspark_score_layout_scope
    from compact_score_report import validate

    directory = Path(directory)
    report = admission.get('report')
    if (admission.get('report_sha256') != REPORT_SHA256 or not isinstance(report, dict)
            or validate(report, directory).get('simulator_qualified') is not True):
        raise ValueError('Source-qualified compact feedback simulator report required')
    generated = payloads(directory)
    for name, source in generated.items():
        if (directory / name).read_bytes() != source.encode():
            raise ValueError('Hardware adapter differs from exact qualified-source transformation: ' + name)
    device = importlib.import_module('compact_score_hardware_device')
    candidate = importlib.import_module('compact_score_hardware_markov')
    for module, name in ((device, 'compact_score_hardware_device.py'), (candidate, 'compact_score_hardware_markov.py')):
        if Path(module.__file__).resolve() != (directory / name).resolve():
            raise ValueError('Hardware adapter imported from another checkout')
    original = dspark_score_layout_scope.candidate
    if getattr(original, '_compact_score_override', False):
        raise ValueError('Nested compact feedback override forbidden')
    audit = dict(calls=0, steps=0, restored=False, hardware_sources={
        name: hashlib.sha256(source.encode()).hexdigest() for name, source in generated.items()})

    def execute(operations, mesh, anchor, logits, predecessor, successor, owned, **options):
        if tuple(logits.shape) != (1, 1, 15, 248320) or list(mesh.shape) != [1, 2]:
            raise ValueError('Complete fifteen-step full-vocabulary two-chip feedback required')
        records = candidate.execute(operations, mesh, anchor, logits, predecessor, successor, owned, **options)
        if len(records) != 15 or any('diagnostic' not in record for record in records):
            raise ValueError('Every compact feedback validity record must be retained')
        audit['calls'] += 1
        audit['steps'] += len(records)
        return records

    execute._compact_score_override = True
    try:
        with patch.object(dspark_score_layout_scope, 'candidate', execute):
            try:
                yield audit
            finally:
                if dspark_score_layout_scope.candidate is not execute:
                    raise ValueError('Compact feedback override changed outside request owner')
    finally:
        audit['restored'] = dspark_score_layout_scope.candidate is original
        validate(report, directory)
        for name, source in generated.items():
            if (directory / name).read_bytes() != source.encode():
                raise ValueError('Hardware adapter changed during request')
