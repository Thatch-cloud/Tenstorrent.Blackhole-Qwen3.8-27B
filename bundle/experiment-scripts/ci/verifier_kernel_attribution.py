"""Selected-arm kernel diagnostics; not a complete profiling gate or critical-path measurement."""

import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import statistics


def selected_metadata(stream, wanted):
    cache, selected = {}, {}
    for row in csv.DictReader(stream, delimiter=';', quotechar='`'):
        message = row['MessageName']
        if not ('TT_DNN' in message or 'TT_METAL' in message) or 'OP' not in message:
            continue
        if ' ->\n' in message:
            try:
                data = json.loads(message.split(' ->\n', 1)[1])
            except json.JSONDecodeError:
                continue
            if 'op_hash' not in data:
                continue
            device, op_hash, call = (int(data[key]) for key in ('device_id', 'op_hash', 'global_call_count'))
            info = data.get('kernel_info', {})
            sources = tuple(sorted({kernel['source'] for kind in ('compute_kernels', 'datamovement_kernels')
                                    for kernel in info.get(kind, [])}))
            shapes = tuple(tuple(tensor.get('shape', {}).get(axis, '') for axis in ('W', 'Z', 'Y', 'X'))
                           for tensor in data.get('input_tensors', [])[:2])
            metadata = (data['op_code'], sources, shapes)
            cache[device, op_hash] = metadata
        else:
            fields = message.split(':', 1)[-1].split(',')
            if len(fields) < 5:
                continue
            op_hash, device, call = int(fields[1]), int(fields[2]), int(fields[4])
            metadata = cache.get((device, op_hash))
        if (device, call) in wanted:
            if metadata is None:
                raise ValueError('Selected operation has no cached metadata')
            if (device, call) in selected and selected[device, call] != metadata:
                raise ValueError('Conflicting selected operation metadata')
            selected[device, call] = metadata
    if set(selected) != wanted:
        raise ValueError(f'Missing metadata for {len(wanted - set(selected))} selected operations')
    return selected


def analyze(root, arm):
    generation = json.loads((root / 'full-gdn-device-loop.json').read_text())
    if generation.get('passed') is not True:
        raise ValueError('Passed numerical verifier report required')
    selected = {}
    if sorted(timing['length'] for timing in generation.get('timings', [])) != [4095, 16383]:
        raise ValueError('Both matched coding contexts required')
    console = (root / 'verifier-profile/console.log').read_text(encoding='utf-8', errors='replace')
    for timing in generation['timings']:
        if timing.get('rows') != 8 or timing.get('exact') is not True:
            raise ValueError('Exact T8 numerical checks required')
        records = [record for record in timing['device_profile']['records'] if record['arm'] == arm]
        if len(records) != 3 or {record['repeat'] for record in records} != {0, 1, 2}:
            raise ValueError('Three exact labeled replays required')
        if len({record['trace_id'] for record in records}) != 1:
            raise ValueError('One stable trace per context and arm required')
        for record in records:
            if not record['exact']:
                raise ValueError('Inexact replay')
            begin = 'QWEN_VERIFIER_PROFILE_BEGIN ' + record['label']
            end = 'QWEN_VERIFIER_PROFILE_END ' + record['label']
            if console.count(begin) != 1 or console.count(end) != 1 or console.index(end) < console.index(begin):
                raise ValueError('Missing measured boundaries')
            if 'markers were dropped' in console.split(begin, 1)[1].split(end, 1)[0]:
                raise ValueError('Selected arm lost measured markers')
            selected[str(record['trace_id'])] = timing['length']
    folder = root / 'verifier-profile/metadata'
    with (folder / 'cpp_device_perf_report.csv').open(newline='') as stream:
        rows = [row for row in csv.DictReader(stream) if row['METAL TRACE ID'] in selected
                and row['METAL TRACE REPLAY SESSION ID']]
    wanted = {(int(row['DEVICE ID']), int(row['GLOBAL CALL COUNT'])) for row in rows}
    with (folder / 'tracy_ops_data.csv').open(encoding='utf-8', newline='') as stream:
        metadata = selected_metadata(stream, wanted)
    groups = defaultdict(lambda: defaultdict(list))
    sessions = defaultdict(set)
    for row in rows:
        device, call = int(row['DEVICE ID']), int(row['GLOBAL CALL COUNT'])
        trace, replay = row['METAL TRACE ID'], int(row['METAL TRACE REPLAY SESSION ID'])
        signature = metadata[device, call]
        duration = float(row['DEVICE KERNEL DURATION [ns]'])
        if not 0 < duration < float('inf'):
            raise ValueError('Invalid kernel duration')
        groups[trace, device, signature][replay].append(duration)
        sessions[trace, device].add(replay)
    if set(sessions) != {(trace, device) for trace in selected for device in (0, 1)}:
        raise ValueError('Both chips required for every selected trace')
    results = []
    for (trace, device), available in sorted(sessions.items()):
        if len(available) != 3:
            raise ValueError('Exactly three replay sessions required')
        operations = []
        for (group_trace, chip, signature), replays in groups.items():
            if (group_trace, chip) != (trace, device):
                continue
            if set(replays) != available or len({len(values) for values in replays.values()}) != 1:
                raise ValueError('Per-kernel coverage changed across replays')
            operations.append(dict(operation=signature[0], kernels=signature[1], inputs=signature[2],
                calls_per_replay=len(next(iter(replays.values()))),
                median_summed_kernel_ms=statistics.median(sum(values) for values in replays.values()) / 1e6))
        results.append(dict(context=selected[trace], chip=device, arm=arm, trace_id=trace,
                            groups=sorted(operations, key=lambda item: -item['median_summed_kernel_ms'])))
    return dict(scope=__doc__, results=results)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--arm', choices=('batch', 'control', 'serial'), default='batch')
    parser.add_argument('--output', type=Path, required=True)
    options = parser.parse_args()
    result = analyze(options.root, options.arm)
    options.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(dict(scope=result['scope'], summaries=[dict(context=item['context'], chip=item['chip'],
        groups=item['groups'][:5]) for item in result['results']]), indent=2))
