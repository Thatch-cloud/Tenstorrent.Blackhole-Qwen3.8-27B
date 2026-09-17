"""Exact native-comparison evidence for folded-verifier ladder components."""

import json
from pathlib import Path

from dspark_hardware_gate import digest
from dspark_splitk_sim_gate import verify_sources
from matched_target_geometry import geometry


REPORTS = {
    4096: 'a71348c2d1d3991752d39cdc1256a27c2b6cbe77ae128eb04380345cb6f54b2c',
    8192: 'ec07916a78069cddd047c50b629d6b0d5e76bf04dddca7df0d9ffda428352221',
    16384: '1e75fc5aeecc34187a519855b983d5d99b4202d9f53bd2f0536374da40bc7690',
    32768: '94257b7a9d66adf93cd9f5c6d2a805fb821dcb4dc1520a8d3086e04a38cbeb8f',
    65536: '097fdfc263dd6a325dec428d05f44f0231113a1faa00b12b314856e9667ce315',
}


def qualify(directory, report_path, context):
    if type(context) is not int or context not in REPORTS or digest(report_path) != REPORTS[context]:
        raise ValueError('Exact passing verifier context report required')
    report = json.loads(Path(report_path).read_text())
    expected = geometry(context)
    if (report.get('passed') is not True or report.get('closed') is not True
            or report.get('backend') != 'hardware'
            or report.get('geometry') != dict(expected, starts=list(expected['starts']))
            or report.get('stale_controls') != 2 or report.get('mask_poison_controls') != 8):
        raise ValueError('Complete verifier scope and negative controls required')
    for field, count in (('checks', 8), ('mask_checks', 16), ('source_checks', 4), ('unpoisoned_replay', 2)):
        records = report.get(field, [])
        if len(records) != count or any(record.get('exact') is not True for record in records):
            raise ValueError('All native comparisons and replay checks must be exact')
    for field in ('sources', 'context_sources'):
        if report.get(field) != report.get(field + '_after'):
            raise ValueError('Source-stable verifier experiment required')
        verify_sources(directory, report[field])
    verify_sources(directory, report['build']['builders'])
    return dict(context=context, capacity=expected['capacity'], report_sha256=REPORTS[context],
        component_qualified=True, full_request_qualified=False,
        performance_qualified=False, serving_qualified=False)
