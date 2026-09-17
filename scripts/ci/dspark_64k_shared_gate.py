"""Retain the completed combined shared-Q/K audit before clean timing."""

import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import dspark_64k_mlp_gate as mlp
from dspark_64k_shared_qk import validate_shared


SCREEN_RUN = 34925963042
SCREEN_SHA256 = 'f1c53b0d1a8dfeb1ece5cd3f13e771480f7e35a365e460365c4ebb89aca380e8'


def qualify(directory, report_path):
    with patch.object(mlp, 'SCREEN_RUN', SCREEN_RUN), patch.object(mlp, 'SCREEN_SHA256', SCREEN_SHA256):
        evidence = mlp.qualify(directory, report_path)
    shared = json.loads(Path(report_path).read_bytes())['shared_qk_64k']
    if len(shared['records']) != 1:
        raise ValueError('One complete shared-Q/K audit required')
    validate_shared(shared['records'][0])
    expected = ('dspark-64k-shared-qk-request.py', 'dspark_64k_shared_qk.py',
        'gdn_shared_qk_scope.py', 'gdn_shared_qk_gate.py')
    if set(shared['sources']) != set(expected):
        raise ValueError('Complete shared-Q/K source manifest required')
    for name in expected:
        if hashlib.sha256((Path(directory) / name).read_bytes()).hexdigest() != shared['sources'][name]:
            raise ValueError('Audited shared-Q/K source changed: ' + name)
    return dict(evidence, shared_audit=shared['records'][0], shared_sources=shared['sources'])
