"""Completed full-history combined MLP audit; not a performance result."""

import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import dspark_splitk_request_gate as splitk
from dspark_64k_mlp_scope import validate_result


SCREEN_RUN = 34924260072
SCREEN_SHA256 = 'ad3481c13bfba26c7b95ec787a4afc81d5a24947dfe83e01d22cbd0b3791072e'


def qualify(directory, report_path):
    with patch.object(splitk, 'SCREEN_RUN', SCREEN_RUN), patch.object(splitk, 'SCREEN_SHA256', SCREEN_SHA256):
        admission = splitk.qualify(directory, report_path)
    report = json.loads(Path(report_path).read_bytes())
    sources = report['mlp_reintegration_sources']
    expected = ('dspark-64k-mlp-request.py', 'dspark_64k_mlp_scope.py', 'fused_t16_scope.py',
        'fused_t16_admission.py', 'fused_1d.py')
    if set(sources) != set(expected):
        raise ValueError('Complete audited MLP source manifest required')
    for name in expected:
        if hashlib.sha256((Path(directory) / name).read_bytes()).hexdigest() != sources[name]:
            raise ValueError('Audited MLP source changed: ' + name)
    audit = admission['request']['mlp_64k_reintegration']['audit']
    validate_result(audit)
    weights = audit['weight_audit']
    if (weights.get('passed') is not True or len(weights.get('checks', [])) != 256
            or any(check.get('exact') is not True for check in weights['checks'])):
        raise ValueError('Exact packed weights for both projections and chips required')
    return dict(admission, mlp_audit=audit, mlp_sources=sources)
