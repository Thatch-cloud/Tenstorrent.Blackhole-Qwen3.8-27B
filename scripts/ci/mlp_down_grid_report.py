"""Independently recompute combined MLP-down correctness and full-cycle TG."""

import argparse
import hashlib
import json
from pathlib import Path

from mlp_down_grid_comparison import summarize
from dspark_request_experiment import summarize as summarize_requests
from mlp_weight_pipeline_report import validate_audit, validate_fusion, NORM_SHA256, HISTORY_SHA256
from cumulative_norm_validation import validate_norm_history


def validate_route(request, arm, *, norm_policy='prefetch'):
    from mlp_down_grid_gate import REPORT_SHA256
    identity = request.get('mlp_down_grid', {})
    enabled, hits = identity.get('wider_down'), identity.get('hits')
    if (arm != 'publication' or type(enabled) is not bool or not isinstance(hits, list) or len(hits) != 64
            or identity.get('restored') is not True
            or identity.get('report_sha256') != (REPORT_SHA256 if enabled else None)
            or (enabled and (any(type(count) is not int or count <= 0 for count in hits)
                or hits != request.get('fused_t16_mlp', {}).get('hits')))
            or (not enabled and hits != [0] * 64)):
        raise ValueError('All fused MLP calls must match the admitted down-grid route')
    validate_fusion(request)
    validate_norm_history(request, norm_policy)


def validate(report):
    if (report.get('passed') is not True or report.get('closed_cleanly') is not True
            or report.get('stage') != 'complete' or report.get('streams') != 1
            or report.get('ctx_tokens') != 4096 or report.get('sampler_links') != 4
            or report.get('fresh_context_audit') is not True):
        raise ValueError('Clean complete four-link single-stream 4K comparison required')
    for before, after in (('sources', 'sources_after'), ('native_sources', 'native_sources_after'),
            ('mlp_down_grid_sources', 'mlp_down_grid_sources_after')):
        if not report.get(before) or report.get(before) != report.get(after):
            raise ValueError('Unchanged native, script and projection sources required')
    comparison = summarize(report.get('request_checks', []), summarize_requests, validate_audit, validate_route)
    if comparison != report.get('mlp_down_grid_comparison'):
        raise ValueError('Saved comparison differs from independently recomputed complete-cycle evidence')
    return comparison


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('report', type=Path)
    options = parser.parse_args()
    raw = options.report.read_bytes()
    print(json.dumps(dict(validate(json.loads(raw)), report_sha256=hashlib.sha256(raw).hexdigest()), indent=2))
