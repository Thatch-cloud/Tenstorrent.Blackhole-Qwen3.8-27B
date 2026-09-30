"""Independently recompute combined GDN input-order correctness and full-cycle TG."""

import argparse
import hashlib
import json
from pathlib import Path

from gdn_input_overlap_comparison import summarize
from dspark_request_experiment import summarize as summarize_requests
from mlp_weight_pipeline_report import validate_audit, validate_fusion, NORM_SHA256, HISTORY_SHA256


def validate_route(request, arm):
    from gdn_input_overlap_gate import REPORT_SHA256
    identity = request.get('gdn_input_overlap', {})
    enabled, builds = identity.get('overlapped_inputs'), identity.get('builds')
    norm, history = request.get('gdn_norm_prefetch', {}), request.get('incremental_history', {})
    if (arm != 'publication' or type(enabled) is not bool or type(builds) is not int
            or identity.get('restored') is not True
            or identity.get('report_sha256') != (REPORT_SHA256 if enabled else None)
            or (enabled and (builds <= 0 or builds % 48 or builds != norm.get('builds')))
            or (not enabled and builds != 0)):
        raise ValueError('All norm-prefetched recurrence layers must use the admitted reader policy')
    validate_fusion(request)
    if (norm.get('enabled') is not True or norm.get('report_sha256') != NORM_SHA256
            or type(norm.get('builds')) is not int or norm['builds'] < 48 or norm['builds'] % 48
            or history.get('enabled') is not True or history.get('report_sha256') != HISTORY_SHA256):
        raise ValueError('Unchanged winning target norm and incremental history required')


def validate(report):
    if (report.get('passed') is not True or report.get('closed_cleanly') is not True
            or report.get('stage') != 'complete' or report.get('streams') != 1
            or report.get('ctx_tokens') != 4096 or report.get('sampler_links') != 4
            or report.get('fresh_context_audit') is not True):
        raise ValueError('Clean complete four-link single-stream 4K comparison required')
    for before, after in (('sources', 'sources_after'), ('native_sources', 'native_sources_after'),
            ('input_overlap_sources', 'input_overlap_sources_after')):
        if not report.get(before) or report.get(before) != report.get(after):
            raise ValueError('Unchanged native, script and reader sources required')
    comparison = summarize(report.get('request_checks', []), summarize_requests, validate_audit, validate_route)
    if comparison != report.get('input_overlap_comparison'):
        raise ValueError('Saved comparison differs from independently recomputed complete-cycle evidence')
    return comparison


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('report', type=Path)
    options = parser.parse_args()
    raw = options.report.read_bytes()
    print(json.dumps(dict(validate(json.loads(raw)), report_sha256=hashlib.sha256(raw).hexdigest()), indent=2))
