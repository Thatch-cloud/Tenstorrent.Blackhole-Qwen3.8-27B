"""Attribute complete request verifier replays per chip, without reporting instrumented TG."""

from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys

from full_dflash_request import summarize_dflash_requests


def validate_request(report, console, *, family='dflash2'):
    if report.get('passed') is not True or report.get('instrumented_timing') is not True or report.get('correctness_only') is not True:
        raise ValueError('Passed attribution-only complete request required')
    requests = report.get('request_checks', [])
    if family == 'dflash2':
        summary = summarize_dflash_requests(requests, audit_only=True)
        request = requests[0]
        if (summary['context'] != 4096 or any(request.get(key) is not True for key in ('commit_only_gdn', 'fused_convolution', 'cache_history'))
                or request.get('cache_projection_capture') is not False or request['dflash'].get('block_rows') != 8):
            raise ValueError('Profile the qualified cached 4K T8 eager-update configuration')
    elif family == 'dspark':
        if (len(requests) != 1 or report.get('profile_family') != 'dspark' or report.get('closed_cleanly') is not True
                or report.get('committed_tg') is not None or report.get('sources') != report.get('sources_after')
                or not report.get('sources') or report.get('native_sources') != report.get('native_sources_after')
                or not report.get('native_sources')):
            raise ValueError('One complete unchanged DSpark attribution request required')
        request = requests[0]
        draft = request.get('dspark', {})
        if (request.get('length') != 4096 or request.get('lookup_max_rows') != 16
                or any(request.get(key) is not True for key in ('exact', 'state_exact', 'inactive_exact',
                    'instrumented_timing', 'commit_only_gdn', 'norm_batch', 'native_sampling_rows'))
                or any(draft.get(key) is not True for key in ('native_attention', 'proposal_trace', 'audit_features'))
                or request.get('committed_tokens_per_second') is not None
                or request.get('verifier_profile', {}).get('full_rows') != 16):
            raise ValueError('Profile the audited native-attention T16 DSpark configuration')
        features = draft.get('feature_checks', [])
        if len(features) != 10 * len(request['blocks']) or any(value.get('exact') is not True for value in features):
            raise ValueError('Complete exact target-feature publication checks required')
    else:
        raise ValueError('Explicit supported verifier profile family required')
    profile = request.get('verifier_profile', {})
    records = profile.get('records', [])
    if len(records) != len(request['blocks']) or len(records) < 3 or not profile.get('host_calls'):
        raise ValueError('Every actual verification and bounded host attribution required')
    counts = Counter()
    for index, (record, block) in enumerate(zip(records, request['blocks'], strict=True)):
        trace = record.get('trace_id')
        if (type(trace) is not int or trace < 0 or record.get('block') != index
                or record.get('position') != block['position'] or record.get('rows') != block['rows']
                or record.get('trace_ordinal') != counts[trace] or record.get('first_replay') is not (counts[trace] == 0)
                or not math.isfinite(record.get('instrumented_host_ms', float('nan'))) or record['instrumented_host_ms'] <= 0):
            raise ValueError('Record must identify its actual request block and trace ordinal')
        label = f'qwen_request_verify_{index}_pos{block["position"]}_t{block["rows"]}_trace{trace}'
        begin, end = 'QWEN_REQUEST_VERIFY_BEGIN ' + label, 'QWEN_REQUEST_VERIFY_END ' + label
        if (record.get('label') != label or console.count(begin) != 1 or console.count(end) != 1
                or console.index(begin) >= console.index(end)):
            raise ValueError('Unambiguous verification markers required')
        if 'markers were dropped' in console.split(begin, 1)[1].split(end, 1)[0]:
            raise ValueError('Dropped profiler markers inside actual verification')
        counts[trace] += 1
    if {int(key): value for key, value in profile.get('trace_counts', {}).items()} != dict(counts):
        raise ValueError('Complete trace invocation counts required')
    return request, records


def intervals(rows):
    spans, scales = [], []
    for row in rows:
        start, end = int(row['DEVICE KERNEL START CYCLE']), int(row['DEVICE KERNEL END CYCLE'])
        duration = float(row['DEVICE KERNEL DURATION [ns]'])
        if start < 0 or end <= start or not math.isfinite(duration) or duration <= 0:
            raise ValueError('Positive kernel durations and ordered device timestamps required')
        spans.append((start, end))
        scales.append(duration / (end - start))
    scale = statistics.median(scales)
    if any(abs(float(row['DEVICE KERNEL DURATION [ns]']) - (end - start) * scale)
            > max(2.0, float(row['DEVICE KERNEL DURATION [ns]']) * 0.002)
            for row, (start, end) in zip(rows, spans, strict=True)):
        raise ValueError('Device timestamps do not establish a consistent clock conversion')
    spans.sort()
    first, last = spans[0]
    covered = 0
    for start, end in spans[1:]:
        if start > last:
            covered += last - first
            first, last = start, end
        else:
            last = max(last, end)
    covered += last - first
    envelope = max(end for start, end in spans) - spans[0][0]
    return dict(clock_ns_per_cycle_estimate=scale, kernel_envelope_cycles=envelope,
        kernel_envelope_ns_estimate=envelope * scale, kernel_union_ns_estimate=covered * scale,
        uncovered_interval_ns_estimate=(envelope - covered) * scale)


def analyze_traces(records, rows, *, full_rows=8):
    if type(full_rows) is not int or full_rows not in (8, 16):
        raise ValueError('Explicit supported verifier attribution width required')
    by_trace = defaultdict(list)
    for record in records:
        by_trace[record['trace_id']].append(record)
    sessions = defaultdict(list)
    for row in rows:
        trace = row.get('METAL TRACE ID', '')
        replay = row.get('METAL TRACE REPLAY SESSION ID', '')
        if not trace or not replay or int(trace) not in by_trace:
            continue
        sessions[(int(trace), row['DEVICE ID'], int(replay))].append(row)
    output = []
    for trace, trace_records in by_trace.items():
        devices = {device for identifier, device, replay in sessions if identifier == trace}
        if devices != {'0', '1'}:
            raise ValueError('Both physical chips required for every verifier trace')
        reference_sessions = None
        for device in sorted(devices):
            replays = sorted(replay for identifier, chip, replay in sessions if identifier == trace and chip == device)
            if len(replays) != len(trace_records) or (reference_sessions is not None and replays != reference_sessions):
                raise ValueError('Runtime replay count must exactly match the observed request calls on both chips')
            reference_sessions = replays
            records_out, group_totals = [], defaultdict(list)
            coverage = None
            for replay, record in zip(replays, trace_records, strict=True):
                operations = sessions[(trace, device, replay)]
                groups = Counter((row['OP NAME'], row['CORE COUNT']) for row in operations)
                if coverage is not None and groups != coverage:
                    raise ValueError('Operation/core coverage changes across the same captured trace')
                coverage = groups
                measured = dict(block=record['block'], position=record['position'], rows=record['rows'], replay_session=replay,
                    first_replay=record['first_replay'], operations=len(operations), **intervals(operations))
                records_out.append(measured)
                if not record['first_replay']:
                    totals = defaultdict(float)
                    for row in operations:
                        totals[(row['OP NAME'], row['CORE COUNT'])] += float(row['DEVICE KERNEL DURATION [ns]'])
                    for key, total in totals.items():
                        group_totals[key].append(total)
            steady = [entry for entry in records_out if not entry['first_replay']]
            output.append(dict(trace_id=trace, device=device, replays=records_out, steady_replays=len(steady),
                steady_medians={key: statistics.median(entry[key] for entry in steady) for key in (
                    'kernel_envelope_ns_estimate', 'kernel_union_ns_estimate', 'uncovered_interval_ns_estimate')} if steady else {},
                operation_core_groups=sorted([dict(op=key[0], cores=key[1], operations_per_replay=coverage[key],
                    median_summed_kernel_ns=statistics.median(values)) for key, values in group_totals.items()],
                    key=lambda entry: -entry['median_summed_kernel_ns'])))
    if not any(entry['steady_replays'] >= 2 and all(replay['rows'] == full_rows for replay in entry['replays']) for entry in output):
        raise ValueError(f'Multiple steady actual T{full_rows} replays required')
    return output


def main(root, family='dflash2'):
    root = Path(root)
    result = dict(passed=False, scope='Instrumented current-request verifier attribution, not throughput or held-out coding quality')
    try:
        request_path = root / 'request.json'
        request, records = validate_request(json.loads(request_path.read_text()), (root / 'console.log').read_text(errors='replace'), family=family)
        paths = [root / location / 'cpp_device_perf_report.csv' for location in ('metadata', '.logs')]
        profile_path = next((path for path in paths if path.is_file()), None)
        if profile_path is None:
            raise ValueError('Runtime C++ device operation report required')
        with profile_path.open(newline='') as stream:
            devices = analyze_traces(records, csv.DictReader(stream), full_rows=16 if family == 'dspark' else 8)
        result.update(passed=True, context=4096, streams=1, devices=devices, host_calls=request['verifier_profile']['host_calls'],
            request_sha256=hashlib.sha256(request_path.read_bytes()).hexdigest(),
            device_report_sha256=hashlib.sha256(profile_path.read_bytes()).hexdigest(),
            caveats=['Kernel envelopes are per-chip observed intervals, not a dependency critical-path proof.',
                'Clock conversion is estimated from the same runtime timestamp/duration pairs.',
                'Operation sums may overlap and must not be added across chips or converted to TG.',
                'Uncovered intervals do not by themselves identify host or dispatch stalls.',
                'Host call durations include native extension waits, not CPU utilization.'])
    except BaseException as error:
        result['error'] = f'{type(error).__name__}: {error}'
        raise
    finally:
        (root / 'attribution.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(dict(passed=True, context=result['context'], traces=len(result['devices']), scope=result['scope'])), flush=True)


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else 'dflash2')
