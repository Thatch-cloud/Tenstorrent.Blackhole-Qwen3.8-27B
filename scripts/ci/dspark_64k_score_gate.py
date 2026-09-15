"""Pin the complete combined score-layout correctness audit before timing."""

import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import dspark_64k_shared_gate as shared
from dspark_score_layout_hardware_gate import validate_hardware


SCREEN_RUN = 34929473486
SCREEN_SHA256 = 'c027988a232f47f7a27bc7382aa09c02f6b9117af9e3c754e782f0d75ee23288'


def qualify(directory, report_path):
    with patch.object(shared, 'SCREEN_RUN', SCREEN_RUN), patch.object(shared, 'SCREEN_SHA256', SCREEN_SHA256):
        admission = shared.qualify(directory, report_path)
    score = json.loads(Path(report_path).read_bytes())['score_layout_64k']
    expected = ('dspark-64k-score-request.py', 'dspark_64k_score_scope.py', 'dspark_score_layout_scope.py',
        'dspark_score_layout_hardware_audit.py', 'dspark_score_layout_hardware_gate.py',
        'target_kv_bulk_audit.py', 'target_kv_bulk_scope.py')
    if set(score['sources']) != set(expected) or len(score['records']) != 1:
        raise ValueError('One complete source-pinned score-layout audit required')
    for name in expected:
        if hashlib.sha256((Path(directory) / name).read_bytes()).hexdigest() != score['sources'][name]:
            raise ValueError('Audited score-layout source changed: ' + name)
    record = score['records'][0]
    if record.get('restored') is not True or type(record.get('calls')) is not int or record['calls'] < 1:
        raise ValueError('Used and restored score-layout scope required')
    if validate_hardware(record['hardware_audit'], directory) != record['hardware_audit_sha256']:
        raise ValueError('Exact learned full-vocabulary score audit required')
    return dict(admission, score_audit=record, score_sources=score['sources'])
