"""Retained captured-publication simulator admission, not full-request qualification."""

import hashlib
import json
from pathlib import Path


REPORT_SHA256 = '4bd749d6381cb7e1f5be69276d5a5011c9e6cfa7c30182b44dd09f3d1b115914'
SOURCES = ('dspark_publication_trace.py', 'dspark_captured_publication.py', 'dspark_history.py',
    'dspark_stable_history.py', 'dspark_projection.py', 'dspark_mesh.py', 'dspark_layer.py',
    'dspark_rotary_device.py', 'dspark_rope_tables.py', 'feature_projection.py',
    'attention_batch.py', 'gdn_multitoken_conv.py')


def qualify(path, directory):
    payload = Path(path).read_bytes()
    if hashlib.sha256(payload).hexdigest() != REPORT_SHA256:
        raise ValueError('Exact retained publication simulator report required')
    report = json.loads(payload)
    expected = [dict(position=4096, prefix=15, committed=True, exact=True),
        dict(position=4111, prefix=1, committed=False, exact=True),
        dict(position=4111, prefix=32, committed=True, exact=True)]
    if (report.get('passed') is not True or report.get('closed_cleanly') is not True
            or report.get('publication_checks') != expected
            or report.get('replay_checks') != [dict(tensors=20, exact=True)] * 6
            or report.get('attention') != report.get('attention_after')
            or report.get('sources') != report.get('sources_after')):
        raise ValueError('Complete stable captured projection and transaction evidence required')
    for name in SOURCES:
        if hashlib.sha256((Path(directory) / name).read_bytes()).hexdigest() != report['sources'][name]:
            raise ValueError('Simulator-qualified publication source changed: ' + name)
    return dict(report_sha256=REPORT_SHA256, checked_sources=list(SOURCES),
        full_request_qualified=False, scope='Retained simulator projection and transaction checks only')
