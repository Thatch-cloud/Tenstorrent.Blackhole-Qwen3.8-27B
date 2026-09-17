"""Pinned FP32-maxima simulator admission; no hardware or throughput acceptance."""

import json
from pathlib import Path
from unittest.mock import patch

import dspark_splitk_sim_gate as baseline


REPORT_SHA256 = '24a5659adba79424b6dee5106257b44469c22a8b8fa7341e2d788b1359b54d2e'
FACTORY_SHA256 = '9406dd82227d067da33670b6d68d4f48d7a78a24b93c66c0a458c7b6c2f7443e'


def qualify(directory, report_path):
    with patch.object(baseline, 'REPORT_SHA256', REPORT_SHA256):
        result = baseline.qualify(directory, report_path)
    report = json.loads(Path(report_path).read_text())
    if (report.get('candidate') != 'splitk-local-maxima-fp32-ablation'
            or result['factory_source_after'] != FACTORY_SHA256):
        raise ValueError('Exact qualified maxima candidate required')
    if len(report['input_checks']) != 48 or any(record.get('exact') is not True for record in report['input_checks']):
        raise ValueError('All unchanged-input checks required')
    if len(report['layout_checks']) != 16 or any(record.get('passed') is not True for record in report['layout_checks']):
        raise ValueError('All layout checks required')
    return result
