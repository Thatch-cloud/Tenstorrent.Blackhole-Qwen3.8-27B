"""Exact fresh maxima and incremental-history correctness admission."""

import json
from pathlib import Path
from unittest.mock import patch

import qwen_lazy_weight_gate as loader
from dspark_splitk_combined_build import digest
from matched_combined_build import admission
from matched_combined_request import validate_result


SCREEN_RUN = 35038298030
SCREEN_SHA256 = 'e9129dcb65a87d7bc25463d38eca83c4a83a276e1a6521fab0dec3b47cb4f329'


def qualify(directory, report_path):
    with patch.object(loader, 'RUN', SCREEN_RUN), patch.object(loader, 'SHA256', SCREEN_SHA256):
        evidence = loader.qualify(directory, report_path)
    report = json.loads(Path(report_path).read_bytes())
    matched = report['matched_combined']
    validate_result(report, matched['updates'], matched['warmups'])
    if (matched.get('failure') is not None or report.get('checkpoint_closed') is not True
            or matched['sources'] != matched['sources_after']
            or matched['components'] != admission(directory)
            or evidence['combined_scope']['component'] != matched['components']):
        raise ValueError('Same complete source-stable maxima/incremental runtime required')
    for name, expected in matched['sources'].items():
        if Path(name).name != name or digest(Path(directory) / name) != expected:
            raise ValueError('Audited combined runtime changed: ' + name)
    parameters = report.get('device_parameter_checks', [])
    if len(parameters) != 120 or any(check.get('exact') is not True for check in parameters):
        raise ValueError('All learned parameter checks must remain exact')
    return dict(evidence, matched_combined=matched)
