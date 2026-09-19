"""Independently validate and recompute matched complete norm-reader requests."""

import argparse
import hashlib
import json
from pathlib import Path

from dspark_request_experiment import summarize as summarize_requests
from frozen_draft_tail_gate import HARDWARE_SHA256 as TAIL_SHA256
from history_append_hardware_gate import REPORT_SHA256 as HISTORY_SHA256
from mlp_read_order_report import validate_audit, validate_fusion
from shared_qk_norm_comparison import summarize
from shared_qk_norm_comparison_scope import validate_identity


def validate_route(request, arm):
    expected = validate_identity(request)
    if arm != 'publication':
        raise ValueError('Winning publication policy required')
    validate_fusion(request)
    gdn = request.get('gdn_shared_qk', {})
    loads = gdn.get('loads', [])
    if (gdn.get('restored') is not True or gdn.get('released') is not True
            or gdn.get('admission', {}).get('report_sha256') != expected
            or len(loads) != request['norm_reader']['builds']
            or any(value != dict(rows=16, programs=3, retained_preparation_buffers=2) for value in loads)):
        raise ValueError('Qualified restored complete shared-Q/K execution required')
    tail = request.get('draft_tail', {})
    if (tail.get('enabled') is not True or tail.get('report_sha256') != TAIL_SHA256
            or type(tail.get('calls')) is not int or tail['calls'] <= 0 or tail['calls'] % 10):
        raise ValueError('Winning draft-tail assembly required')
    history = request.get('incremental_history', {})
    records = history.get('records', [])
    if (history.get('enabled') is not True or history.get('report_sha256') != HISTORY_SHA256
            or len(records) != 1):
        raise ValueError('One qualified incremental history session required')
    record = records[0]
    if (record.get('initial_position') != 4096 or record.get('capacity') != 4352
            or record.get('failed') is not False or record.get('restored') is not True
            or not 0 < record.get('committed', 0) <= record.get('prepared', 0)
            or record['prepared'] != record['committed'] + record.get('discarded', -1)
            or not 32 <= record.get('max_touched_rows', 0) <= 96):
        raise ValueError('Complete bounded incremental publication required')


def validate(report):
    if (report.get('passed') is not True or report.get('closed_cleanly') is not True
            or report.get('stage') != 'complete' or report.get('streams') != 1
            or report.get('ctx_tokens') != 4096 or report.get('sampler_links') != 4
            or report.get('fresh_context_audit') is not True):
        raise ValueError('Clean complete four-link single-stream 4K run required')
    for before, after in (('sources', 'sources_after'), ('native_sources', 'native_sources_after'),
            ('norm_reader_sources', 'norm_reader_sources_after')):
        if not report.get(before) or report.get(before) != report.get(after):
            raise ValueError('Unchanged script, native and comparison sources required')
    result = summarize(report.get('request_checks', []), summarize_requests, validate_audit, validate_route)
    if result != report.get('norm_reader_comparison'):
        raise ValueError('Recomputed whole-cycle comparison differs from saved report')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('report', type=Path)
    options = parser.parse_args()
    raw = options.report.read_bytes()
    print(json.dumps(dict(validate(json.loads(raw)), report_sha256=hashlib.sha256(raw).hexdigest()), indent=2))
