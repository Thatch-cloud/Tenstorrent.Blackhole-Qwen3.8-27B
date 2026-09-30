"""Independently recompute combined drafter-precision correctness and full-cycle TG."""

import argparse
import hashlib
import json
from pathlib import Path

from dspark_precision_comparison import summarize
from dspark_request_experiment import summarize as summarize_requests
from mlp_weight_pipeline_report import validate_audit, validate_fusion, NORM_SHA256, HISTORY_SHA256


def validate_route(request, arm):
    precision = request.get('drafter_precision')
    identity = request.get('drafter_precision_evidence', {})
    reviewed = json.loads(Path(__file__).with_name('dspark-precision-reviewed.json').read_bytes())
    calls = identity.get('layer_calls')
    if (arm != 'publication' or precision not in ('hifi4', 'hifi2')
            or identity.get('closed') is not True or type(calls) is not int
            or (precision == 'hifi2' and (calls <= 0 or calls % 5))
            or (precision == 'hifi4' and calls != 0)
            or identity.get('report_sha256') != (reviewed if precision == 'hifi2' else None)):
        raise ValueError('Closed source-admitted precision device and complete layer execution required')
    validate_fusion(request)
    norm, history = request.get('gdn_norm_prefetch', {}), request.get('incremental_history', {})
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
            ('drafter_precision_sources', 'drafter_precision_sources_after')):
        if not report.get(before) or report.get(before) != report.get(after):
            raise ValueError('Unchanged native, script and precision sources required')
    comparison = summarize(report.get('request_checks', []), summarize_requests, validate_audit, validate_route)
    if comparison != report.get('drafter_precision_comparison'):
        raise ValueError('Saved comparison differs from independently recomputed complete-cycle evidence')
    return comparison


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('report', type=Path)
    options = parser.parse_args()
    raw = options.report.read_bytes()
    print(json.dumps(dict(validate(json.loads(raw)), report_sha256=hashlib.sha256(raw).hexdigest()), indent=2))
