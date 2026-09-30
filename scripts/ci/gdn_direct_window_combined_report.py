"""Independently recompute combined direct-window correctness and full-cycle TG."""

import argparse
import hashlib
import json
from pathlib import Path

from gdn_direct_window_comparison import summarize, repeatability
from dspark_request_experiment import summarize as summarize_requests
from mlp_weight_pipeline_report import validate_audit, validate_fusion, NORM_SHA256, HISTORY_SHA256
from cumulative_norm_validation import validate_norm_history
from cumulative_fusion_validation import validate_fusion_policy


def validate_route(request, arm, *, norm_policy='prefetch', fusion_policy='baseline'):
    from gdn_direct_window_gate import REPORT_SHA256
    identity = request.get('gdn_direct_window', {})
    enabled, hits = identity.get('direct'), identity.get('hits')
    if (arm != 'publication' or type(enabled) is not bool or type(hits) is not int
            or identity.get('restored') is not True
            or identity.get('report_sha256') != (REPORT_SHA256 if enabled else None)
            or (enabled and (hits < 48 or hits % 48
                or hits != len(request.get('gdn_shared_qk', {}).get('loads', []))))
            or (not enabled and hits != 0)):
        raise ValueError('All T16 GDN window calls must match the admitted write route')
    validate_fusion_policy(request, fusion_policy)
    validate_norm_history(request, norm_policy)


def validate(report, *, norm_scatter=False, register_epilogue=False):
    if type(norm_scatter) is not bool or type(register_epilogue) is not bool:
        raise ValueError('Explicit normalization composition policy required')
    if (report.get('passed') is not True or report.get('closed_cleanly') is not True
            or report.get('stage') != 'complete' or report.get('streams') != 1
            or report.get('ctx_tokens') != 4096 or report.get('sampler_links') != 4
            or report.get('fresh_context_audit') is not True):
        raise ValueError('Clean complete four-link single-stream 4K comparison required')
    for before, after in (('sources', 'sources_after'), ('native_sources', 'native_sources_after'),
            ('gdn_direct_window_sources', 'gdn_direct_window_sources_after')):
        if not report.get(before) or report.get(before) != report.get(after):
            raise ValueError('Unchanged native, script and projection sources required')
    def route(request, arm):
        policy = 'scatter' if norm_scatter and request.get('gdn_direct_window', {}).get('direct') is True else 'prefetch'
        fusion = 'register' if register_epilogue and request.get('gdn_direct_window', {}).get('direct') is True else 'baseline'
        validate_route(request, arm, norm_policy=policy, fusion_policy=fusion)

    comparison = summarize(report.get('request_checks', []), summarize_requests, validate_audit, route)
    if comparison != report.get('gdn_direct_window_comparison'):
        raise ValueError('Saved comparison differs from independently recomputed complete-cycle evidence')
    return dict(comparison, measurement_quality=repeatability(comparison))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('report', type=Path)
    options = parser.parse_args()
    raw = options.report.read_bytes()
    print(json.dumps(dict(validate(json.loads(raw)), report_sha256=hashlib.sha256(raw).hexdigest()), indent=2))
