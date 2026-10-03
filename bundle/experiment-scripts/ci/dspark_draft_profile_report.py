"""Validate actual fifteen-query drafter device attribution, never throughput."""

from collections import defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys

from dspark_intake import TAPS
from dspark_target_attention_variants import validate_route
from request_verifier_profile_report import analyze_traces
from tensix_stream_gate import matrix


def validate(report, console):
    requests = report.get('request_checks')
    if (any(report.get(name) is not True for name in ('passed', 'closed_cleanly', 'instrumented_timing', 'correctness_only'))
            or report.get('stage') != 'complete' or report.get('error')
            or report.get('profile_family') != 'dspark-draft' or report.get('committed_tg') is not None
            or report.get('pp') is not None or report.get('eligible_for_serving') is not False
            or not report.get('sources') or report.get('sources') != report.get('sources_after')
            or not report.get('native_sources') or report.get('native_sources') != report.get('native_sources_after')
            or not isinstance(requests, list) or len(requests) != 1):
        raise ValueError('One clean unchanged attribution-only drafter request required')
    request = requests[0]
    draft = request.get('dspark', {})
    if (request.get('length') != 4096 or request.get('lookup_max_rows') != 16
            or any(request.get(name) is not True for name in ('exact', 'state_exact', 'inactive_exact',
                'instrumented_timing', 'commit_only_gdn', 'norm_batch', 'native_sampling_rows'))
            or any(draft.get(name) is not True for name in ('native_attention', 'proposal_trace', 'audit_features'))
            or request.get('committed_tokens_per_second') is not None):
        raise ValueError('Audited unchanged precise-native DSpark configuration required')
    validate_route(request, 'parallel')
    blocks = request.get('blocks', [])
    if len(blocks) < 3:
        raise ValueError('Complete multi-block request required')
    matrix(draft.get('feature_checks'), ('position', 'tap', 'chip'),
        {(block['position'], tap, chip) for block in blocks for tap in TAPS for chip in range(2)}, ('exact',))
    positions = [request['length'], *(block['position'] for block in blocks if block['rows'] > 1)]
    checks = draft.get('proposal_checks', [])
    if ([value.get('position') for value in checks] != positions
            or any(value.get('exact') is not True or value.get('tensors') != 6 for value in checks)
            or request.get('gdn_verify_checks') != [dict(position=block['position'], rows=block['rows'], unchanged=True)
                for block in blocks if block['rows'] > 1]):
        raise ValueError('Complete eager/replay proposal and unpublished GDN state audits required')
    profile = request.get('draft_profile', {})
    records = profile.get('records', [])
    positions = [request['length'], *positions]
    if profile.get('restored') is not True or len(records) != len(positions):
        raise ValueError('Restored observer with capture warmup and every proposal replay required')
    traces = {record.get('trace_id') for record in records}
    if len(traces) != 1 or any(type(trace) is not int or trace < 0 for trace in traces):
        raise ValueError('One actual proposal trace required')
    trace = next(iter(traces))
    if {int(key): value for key, value in profile.get('trace_counts', {}).items()} != {trace: len(records)}:
        raise ValueError('Exact proposal trace counts required')
    previous = -1
    for index, (record, position) in enumerate(zip(records, positions, strict=True)):
        role = 'capture_warmup' if index == 0 else 'proposal'
        duration = record.get('instrumented_host_ms')
        if (any(type(record.get(name)) is not int or record[name] != value for name, value in
                    (('block', index), ('position', position), ('rows', 15), ('trace_ordinal', index)))
                or record.get('role') != role or record.get('first_replay') is not (index == 0)
                or type(duration) not in (float, int) or not math.isfinite(duration) or duration <= 0):
            raise ValueError('Ordered real proposal metadata required')
        label = f'qwen_draft_{index}_pos{position}_trace{trace}_{role}'
        begin, end = 'QWEN_DRAFT_PROFILE_BEGIN ' + label, 'QWEN_DRAFT_PROFILE_END ' + label
        if (record.get('label') != label or console.count(begin) != 1 or console.count(end) != 1
                or not previous < console.index(begin) < console.index(end)
                or 'markers were dropped' in console.split(begin, 1)[1].split(end, 1)[0]):
            raise ValueError('Complete ordered markers without dropped events required')
        previous = console.index(end)
    return request, records


def attribute(records, rows):
    devices = analyze_traces(records, rows, full_rows=15)
    for device in devices:
        sessions = {replay['replay_session'] for replay in device['replays'] if not replay['first_replay']}
        calls = defaultdict(list)
        for row in rows:
            if (row.get('METAL TRACE ID') == str(device['trace_id']) and row.get('DEVICE ID') == device['device']
                    and row.get('METAL TRACE REPLAY SESSION ID')
                    and int(row['METAL TRACE REPLAY SESSION ID']) in sessions):
                calls[int(row['GLOBAL CALL COUNT'])].append(row)
        device['captured_operations'] = []
        for call, entries in sorted(calls.items()):
            if (len(entries) != len(sessions)
                    or len({entry['METAL TRACE REPLAY SESSION ID'] for entry in entries}) != len(sessions)
                    or len({(entry['OP NAME'], entry['CORE COUNT']) for entry in entries}) != 1):
                raise ValueError('Stable complete captured-operation identity across every replay required')
            device['captured_operations'].append(dict(global_call_count=call, op=entries[0]['OP NAME'],
                cores=int(entries[0]['CORE COUNT']), samples=len(entries),
                median_kernel_ns=statistics.median(float(entry['DEVICE KERNEL DURATION [ns]']) for entry in entries),
                operand_metadata={key: value for key, value in entries[0].items()
                    if key.startswith(('INPUT_0_', 'INPUT_1_', 'OUTPUT_0_'))}))
    return devices


def main(root):
    root = Path(root)
    result = dict(passed=False, scope=__doc__)
    try:
        path = root / 'request.json'
        report = json.loads(path.read_text())
        request, records = validate(report, (root / 'console.log').read_text(errors='replace'))
        for name, expected in report['sources'].items():
            if hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest() != expected:
                raise ValueError('Request source fingerprint changed: ' + name)
        device_path = root / 'metadata/cpp_device_perf_report.csv'
        with device_path.open(newline='') as stream:
            devices = attribute(records, list(csv.DictReader(stream)))
        result.update(passed=True, context=request['length'], devices=devices,
            request_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            device_report_sha256=hashlib.sha256(device_path.read_bytes()).hexdigest(),
            caveats=['Instrumented trace replay excludes proposal staging, audit and token-readback costs.',
                'Per-chip kernel intervals and overlapping operation sums are not a critical path or TG.'])
    except BaseException as error:
        result['error'] = f'{type(error).__name__}: {error}'
        raise
    finally:
        (root / 'attribution.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(dict(passed=True, context=request['length'], chips=len(devices), scope=__doc__)), flush=True)


if __name__ == '__main__':
    main(sys.argv[1])
