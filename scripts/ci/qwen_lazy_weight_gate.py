"""Immutable combined lazy-loader correctness admission, not speed acceptance."""

import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import dspark_64k_score_gate as score

RUN = 34936162975
SHA256 = '739f8e203fcfdeeec552a7d947b5b540f45f99eb2c7490f824a71b27dad4130e'


def qualify(directory, report_path):
    with patch.object(score, 'SCREEN_RUN', RUN), patch.object(score, 'SCREEN_SHA256', SHA256):
        admission = score.qualify(directory, report_path)
    report = json.loads(Path(report_path).read_bytes())
    loader = report['lazy_weight_load']
    expected = ('dspark-64k-lazy-load-request.py', 'qwen_lazy_weight_load.py')
    if loader.get('failure') is not None or set(loader['sources']) != set(expected):
        raise ValueError('Successful source-pinned lazy loader required')
    for name in expected:
        if hashlib.sha256((Path(directory) / name).read_bytes()).hexdigest() != loader['sources'][name]:
            raise ValueError('Audited lazy-loader source changed: ' + name)
    records = loader['records']
    if (len(records) != 1 or records[0].get('restored') is not True
            or records[0].get('calls') != 320 or records[0].get('packed_calls') != 64
            or records[0].get('materializations') != 0 or records[0].get('packed_materializations') != 0):
        raise ValueError('All cached shard and packed MLP loads must have been exercised and restored')
    return dict(admission, lazy_loader=loader, lazy_loader_audit_run=RUN)
