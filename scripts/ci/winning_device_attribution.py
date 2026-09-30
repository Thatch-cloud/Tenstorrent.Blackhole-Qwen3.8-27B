"""Recover device-only attribution from the completed, hash-pinned 35185624322 request."""

import csv
import hashlib
import json
from pathlib import Path
import sys

from request_verifier_profile_report import analyze_traces, validate_records


REQUEST_SHA256 = 'da769ef226bdeb3c2f17183da870aea14a7fdfe9a569f4f8091067c0877d5e76'
DEVICE_SHA256 = '4c579676a287ac57eccee7762f7ad8a4b8f0845fa0f5c49c08da71fb634d848a'


def analyze(root):
    root = Path(root)
    request_path = root / 'request.json'
    device_path = root / 'metadata/cpp_device_perf_report.csv'
    for path, expected in ((request_path, REQUEST_SHA256), (device_path, DEVICE_SHA256)):
        with path.open('rb') as stream:
            if hashlib.file_digest(stream, 'sha256').hexdigest() != expected:
                raise ValueError('Only the retained completed experiment is admitted')
    report = json.loads(request_path.read_text())
    if (report.get('passed') is not True or report.get('closed_cleanly') is not True
            or report.get('committed_tg') is not None or report.get('request_output_limit') != 64
            or len(report.get('request_checks', [])) != 1
            or report.get('sources') != report.get('sources_after')
            or report.get('native_sources') != report.get('native_sources_after')):
        raise ValueError('Complete unchanged audited request required')
    request = report['request_checks'][0]
    if (request.get('length') != 4096 or request.get('arm') != 'publication'
            or any(request.get(key) is not True for key in ('exact', 'state_exact', 'inactive_exact'))):
        raise ValueError('Exact winning 4K request required')
    records = validate_records(request, (root / 'console.log').read_text(errors='replace'))
    with device_path.open(newline='') as stream:
        devices = analyze_traces(records, csv.DictReader(stream), full_rows=16)
    return dict(passed=True, run=35185624322, context=4096, streams=1, output_limit=64,
        request_sha256=REQUEST_SHA256, device_report_sha256=DEVICE_SHA256,
        devices=devices, committed_tg=None, performance_qualified=False,
        host_tracy_available=False, scope=__doc__, caveats=[
            'Workflow failed after the request; host Tracy export is missing.',
            'Device replay counts and console markers validated independently.',
            'Per-chip operation sums may overlap; neither a critical-path proof nor throughput.',
            'Generic kernel identities inferred from source geometry require separate confirmation.'])


if __name__ == '__main__':
    print(json.dumps(analyze(sys.argv[1]), indent=2))
