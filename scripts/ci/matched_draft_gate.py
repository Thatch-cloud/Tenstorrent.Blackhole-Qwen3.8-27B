"""Exact source-pinned component admission for FP32-maxima draft contexts."""

import json
from pathlib import Path

from dspark_hardware_gate import digest
from dspark_splitk_maxima_gate import FACTORY_SHA256
from dspark_splitk_sim_gate import verify_sources
from matched_context_geometry import geometry


REPORTS = {
    4096: '513f8760e9635e8056a43ad1f480f6ae7b5e0ac978e53ff17203d7289d726391',
    8192: '9e9b418bcbc07f50971486f140b69bbceaeee32619ba06e793278491c7ef40a9',
    16384: '4860f3cf31ce951a3ea3004af7251e095b886f7173f24e06ba2a94da27bfbf8b',
    32768: 'f9d1bc607a699490c4a4d87a3b0311bd24e7c2d9d54f4730c8cd624e900e3670',
    65536: 'b9fe17a85c014ab51a500af09e3551bdeb12b11d8e0051b4c4c1013bdc948623',
}


def qualify(directory, report_path, context):
    if type(context) is not int or context not in REPORTS or digest(report_path) != REPORTS[context]:
        raise ValueError('Exact passing hardware report for this context required')
    report = json.loads(Path(report_path).read_text())
    if (report.get('passed') is not True or report.get('closed_cleanly') is not True
            or report.get('backend') != 'hardware' or report.get('geometry') != geometry(context)
            or report.get('numerical_tolerances') != dict(rtol=.01, atol=.01)
            or report.get('sources') != report.get('sources_after')):
        raise ValueError('Complete source-stable context correctness gate required')
    if len(report['checks']) != 72 or any(record.get('passed') is not True for record in report['checks']):
        raise ValueError('Every numerical, replay, input and layout check must pass')
    if len(report['fixture_controls']) != 8 or any(record.get('detected') is not True for record in report['fixture_controls']):
        raise ValueError('Every poison and frontier control must pass')
    build = report['build']
    if (build.get('passed') is not True or build.get('import_passed') is not True
            or build.get('source_after') != FACTORY_SHA256):
        raise ValueError('Qualified FP32-maxima hardware factory required')
    verify_sources(directory, report['sources'])
    verify_sources(directory, build['builders'])
    return dict(context=context, report_sha256=REPORTS[context], kernel=report['kernel'],
        factory_sha256=FACTORY_SHA256, component_qualified=True,
        full_request_qualified=False, performance_qualified=False, serving_qualified=False)
