"""Independently recompute complete-request read-order comparison evidence."""

import argparse
import hashlib
import json
from pathlib import Path

from dspark_intake import TAPS
from dspark_request_experiment import summarize as summarize_requests
from fused_t16_admission import REPORT_SHA256 as BASELINE_SHA256
from frozen_gdn_norm_gate import REPORT_SHA256 as NORM_SHA256
from history_append_hardware_gate import REPORT_SHA256 as HISTORY_SHA256
from mlp_read_order_comparison import summarize
from mlp_read_order_gate import REPORT_SHA256


def validate_route(request, arm):
    identity = request.get('weight_read_order', {})
    enabled = identity.get('staggered')
    if (type(enabled) is not bool or arm != 'publication'
            or identity.get('simulator_report_sha256') != (REPORT_SHA256 if enabled else None)):
        raise ValueError('Source-bound reader identity required')
    validate_fusion(request, REPORT_SHA256 if enabled else BASELINE_SHA256)
    norm, history = request.get('gdn_norm_prefetch', {}), request.get('incremental_history', {})
    if (norm.get('enabled') is not True or norm.get('report_sha256') != NORM_SHA256
            or type(norm.get('builds')) is not int or norm['builds'] < 48 or norm['builds'] % 48
            or history.get('enabled') is not True or history.get('report_sha256') != HISTORY_SHA256):
        raise ValueError('Winning norm prefetch and incremental publication required in both arms')


def validate_fusion(request, expected_sha256=BASELINE_SHA256):
    fusion = request.get('fused_t16_mlp', {})
    if (fusion.get('passed_simulator') != expected_sha256
            or fusion.get('rows') != 16 or fusion.get('layers') != 64
            or fusion.get('restored') is not True or fusion.get('native_bindings_unchanged') is not True
            or fusion.get('extra_weight_allocations') != 0 or len(fusion.get('hits', [])) != 64
            or any(type(count) is not int or count < 1 for count in fusion['hits'])
            or len(set(fusion['hits'])) != 1):
        raise ValueError('Restored all-layer T16 fusion and retained weights required')
    weights = fusion.get('weight_audit', {})
    checks = weights.get('checks', [])
    expected = {(layer, offset, chip) for layer in range(64) for offset in (0, 1) for chip in (0, 1)}
    if (weights.get('passed') is not True or len(checks) != 256
            or {(check.get('layer'), check.get('offset'), check.get('chip')) for check in checks} != expected
            or any(check.get('exact') is not True or check.get('pages') != 43520
                or check.get('mismatched_words') != 0 for check in checks)):
        raise ValueError('Complete actual target packed-weight audit required')


def validate_audit(request):
    blocks = [block for block in request['blocks'] if block['rows'] > 1]
    checks = request.get('dspark', {}).get('proposal_checks', [])
    if (not blocks or [check.get('position') for check in checks] != [4096, *(block['position'] for block in blocks)]
            or any(check.get('tensors') != 6 or check.get('exact') is not True for check in checks)):
        raise ValueError('Complete changed-input proposal audit required')
    if request.get('gdn_verify_checks') != [dict(position=block['position'], rows=block['rows'], unchanged=True) for block in blocks]:
        raise ValueError('Complete deferred-state publication audit required')
    expected = {(block['position'], block['committed'], tap, chip) for block in blocks for tap in TAPS for chip in range(2)}
    features = request.get('dspark', {}).get('feature_checks', [])
    if (len(features) != len(expected) or any(check.get('exact') is not True for check in features)
            or {(check.get('position'), check.get('rows'), check.get('tap'), check.get('chip')) for check in features} != expected):
        raise ValueError('Both chips and all five target taps must match after every committed block')


def validate(report):
    if (report.get('passed') is not True or report.get('closed_cleanly') is not True
            or report.get('stage') != 'complete' or report.get('streams') != 1 or report.get('ctx_tokens') != 4096
            or report.get('sampler_links') != 4 or report.get('fresh_context_audit') is not True):
        raise ValueError('Clean complete four-link single-stream 4K run required')
    for before, after in (('sources', 'sources_after'), ('native_sources', 'native_sources_after'),
            ('read_order_sources', 'read_order_sources_after')):
        if not report.get(before) or report.get(before) != report.get(after):
            raise ValueError('Unchanged script, native and candidate sources required')
    result = summarize(report.get('request_checks', []), summarize_requests, validate_audit, validate_route)
    if result != report.get('read_order_comparison'):
        raise ValueError('Recomputed complete-cycle comparison differs from saved summary')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('report', type=Path)
    options = parser.parse_args()
    raw = options.report.read_bytes()
    print(json.dumps(dict(validate(json.loads(raw)), report_sha256=hashlib.sha256(raw).hexdigest()), indent=2))
