"""Validate and attribute all native/streamed MLP replays per chip, without converting them to TG."""

from collections import defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys

from request_verifier_profile_report import analyze_traces
from tensix_mlp_hardware_gate import validate_sources
from tensix_mlp_weight_views import qualify_views
from tensix_stream_gate import matrix


def validate_profile(report, console):
    if (not isinstance(report, dict) or report.get('stage') != 'complete' or report.get('backend') != 'hardware'
            or report.get('error') or any(report.get(field) is not True for field in
                ('passed', 'closed_cleanly', 'instrumented_timing', 'correctness_only', 'dram_boundary', 'native_collective'))
            or report.get('eligible_for_full_model_gate') is not False or report.get('all_samples_retained') is not False
            or report.get('blocks') != [] or report.get('timed_checks') != []
            or any(name in report for name in ('control_ms', 'candidate_ms', 'tg_tokens_per_second'))
            or any(type(report.get(name)) is not int or report[name] != expected for name, expected in
                (('rows', 8), ('streams', 1), ('layer', 0), ('pool_buffers', 2), ('repeats_per_sample', 1), ('collective_links', 4)))
            or report.get('seeds') != [1659, 2670, 3781]
            or report.get('matched_requested_links') != dict(default=4, axis0=4, axis1=4)):
        raise ValueError('Complete attribution-only real-weight T8 MLP result required; no promotion or TG')
    qualify_views(report.get('weight_views'))
    if (report.get('native_weights') != {name: check['native'] for name, check in report['weight_views'].items()}
            or report.get('native_weights_after') != report['native_weights']):
        raise ValueError('Native control weights must remain unchanged')
    matrix(report.get('eager_checks'), ('pattern', 'chip'),
        {(pattern, chip) for pattern in range(3) for chip in range(2)}, ('exact',))
    matrix(report.get('trace_checks'), ('pattern', 'arm', 'chip'),
        {(pattern, arm, chip) for pattern in range(3) for arm in range(2) for chip in range(2)}, ('exact',))
    matrix(report.get('negative_controls'), ('arm', 'chip'),
        {(arm, chip) for arm in range(2) for chip in range(2)}, ('stale_detected',))
    matrix(report.get('input_checks'), ('pattern', 'tensor', 'chip'),
        {(pattern, tensor, chip) for pattern in range(3) for tensor in range(4) for chip in range(2)},
        ('packed_words_unchanged', 'bindings_stable', 'pool_reused'))
    matrix(report.get('profile_checks'), ('pattern', 'sample', 'chip'),
        {(pattern, sample, chip) for pattern in range(3) for sample in range(4) for chip in range(2)}, ('exact',))
    profile = report.get('profile', {})
    trace_ids, records = profile.get('trace_ids'), profile.get('records')
    if (not isinstance(trace_ids, list) or len(trace_ids) != 2
            or any(type(trace) is not int or trace < 0 for trace in trace_ids) or len(set(trace_ids)) != 2
            or profile.get('trace_counts') != [10, 10] or not isinstance(records, list) or len(records) != 20):
        raise ValueError('Two distinct traces and every actual replay required')
    expected = []
    for pattern in range(3):
        for arm in range(2):
            expected.append((arm, pattern, 'audit', -1))
            if pattern == 0:
                expected.append((arm, pattern, 'stale', -1))
        expected.extend((arm, pattern, 'measurement', sample) for sample, arm in enumerate((0, 1, 1, 0)))
    counts = [0, 0]
    for index, (record, (arm, pattern, role, sample)) in enumerate(zip(records, expected, strict=True)):
        integers = dict(block=index, arm=arm, pattern=pattern, sample=sample, rows=8,
            trace_id=trace_ids[arm], trace_ordinal=counts[arm])
        if (any(type(record.get(name)) is not int or record[name] != value for name, value in integers.items())
                or record.get('role') != role or record.get('first_replay') is not (counts[arm] == 0)
                or type(record.get('instrumented_host_ms')) not in (int, float)
                or not math.isfinite(record['instrumented_host_ms']) or record['instrumented_host_ms'] <= 0):
            raise ValueError('Exact ordered replay roles, trace ordinals and instrumented durations required')
        label = f'qwen_mlp_arm{arm}_pattern{pattern}_{role}_{sample}_ordinal{counts[arm]}'
        begin, end = 'QWEN_MLP_PROFILE_BEGIN ' + label, 'QWEN_MLP_PROFILE_END ' + label
        if (record.get('label') != label or console.count(begin) != 1 or console.count(end) != 1
                or console.index(begin) >= console.index(end)
                or 'markers were dropped' in console.split(begin, 1)[1].split(end, 1)[0]):
            raise ValueError('One complete marker pair without dropped device events required per replay')
        counts[arm] += 1
    return records


def attribute(records, rows):
    devices = analyze_traces([{**record, 'position': None} for record in records], rows)
    sessions = defaultdict(list)
    for row in rows:
        if row.get('METAL TRACE ID') and row.get('METAL TRACE REPLAY SESSION ID'):
            sessions[(int(row['METAL TRACE ID']), row['DEVICE ID'], int(row['METAL TRACE REPLAY SESSION ID']))].append(row)
    for device in devices:
        for replay in device['replays']:
            record = records[replay['block']]
            replay.pop('position')
            replay.update(arm=record['arm'], pattern=record['pattern'], role=record['role'], sample=record['sample'])
        measured = [replay for replay in device['replays'] if replay['role'] == 'measurement']
        if len(measured) != 6 or len({replay['arm'] for replay in measured}) != 1:
            raise ValueError('All six ABBA measurements required on each chip and arm')
        device['arm'] = 'native' if measured[0]['arm'] == 0 else 'streamed'
        device['measurement_medians'] = {name: statistics.median(replay[name] for replay in measured) for name in
            ('kernel_envelope_ns_estimate', 'kernel_union_ns_estimate', 'uncovered_interval_ns_estimate')}
        calls, reference = defaultdict(list), None
        for replay in measured:
            operations = sessions[(device['trace_id'], device['device'], replay['replay_session'])]
            identifiers = [int(row['GLOBAL CALL COUNT']) for row in operations]
            if len(identifiers) != len(set(identifiers)) or (reference is not None and set(identifiers) != reference):
                raise ValueError('Every captured operation must be present exactly once in every measured replay')
            reference = set(identifiers)
            for row in operations:
                calls[int(row['GLOBAL CALL COUNT'])].append(row)
        device['measured_operations'] = []
        for ordinal, (call, operations) in enumerate(sorted(calls.items())):
            if len({(row['OP NAME'], row['CORE COUNT']) for row in operations}) != 1:
                raise ValueError('A captured operation cannot change identity between replays')
            entry = dict(ordinal=ordinal, global_call_count=call, op=operations[0]['OP NAME'],
                cores=int(operations[0]['CORE COUNT']), samples=len(operations),
                median_kernel_ns=statistics.median(float(row['DEVICE KERNEL DURATION [ns]']) for row in operations),
                risc_medians_ns={})
            for processor in ('BRISC', 'NCRISC', 'TRISC0', 'TRISC1', 'TRISC2', 'ERISC'):
                values = [row.get(f'DEVICE {processor} KERNEL DURATION [ns]', '') for row in operations]
                if not any(value not in ('', None) for value in values):
                    continue
                if any(value in ('', None) or not math.isfinite(float(value)) or float(value) < 0 for value in values):
                    raise ValueError('Consistent finite per-RISC durations required when present')
                entry['risc_medians_ns'][processor] = statistics.median(float(value) for value in values)
            device['measured_operations'].append(entry)
    return devices


def main(root):
    root = Path(root)
    result = dict(passed=False, scope='Instrumented native/streamed MLP device attribution; not TG or promotion')
    try:
        report_path, profile_path = root / 'mlp.json', root / 'metadata/cpp_device_perf_report.csv'
        report = json.loads(report_path.read_text())
        validate_sources(report, Path(__file__).parent)
        records = validate_profile(report, (root / 'console.log').read_text(errors='replace'))
        with profile_path.open(newline='') as stream:
            devices = attribute(records, list(csv.DictReader(stream)))
        result.update(passed=True, devices=devices, mlp_sha256=hashlib.sha256(report_path.read_bytes()).hexdigest(),
            device_report_sha256=hashlib.sha256(profile_path.read_bytes()).hexdigest(),
            caveats=['Operation order and durations locate the regression, not its causal mechanism.',
                'RISC durations include waits and may span different cores; they are not pure compute or DRAM service times.',
                'Per-chip envelopes and overlapping operation/RISC sums are not an end-to-end critical path or TG.'])
    except BaseException as error:
        result['error'] = f'{type(error).__name__}: {error}'
        raise
    finally:
        (root / 'attribution.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(dict(passed=True, chips_and_arms=len(devices), scope=result['scope'])), flush=True)


if __name__ == '__main__':
    main(sys.argv[1])
