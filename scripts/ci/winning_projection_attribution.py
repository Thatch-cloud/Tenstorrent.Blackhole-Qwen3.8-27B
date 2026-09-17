"""Separate source-consistent projection sequences in the retained combined trace."""

from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
import statistics
import sys

from winning_device_attribution import analyze


PREDECESSORS = {
    ('GenericOpDeviceOperation', '99'): ('fused_mlp_down', 64),
    ('GenericOpDeviceOperation', '24'): ('gdn_output', 48),
    ('SliceDeviceOperation', '96'): ('attention_output', 16),
}


def separate(rows):
    ordered = sorted(rows, key=lambda row: int(row['DEVICE KERNEL START CYCLE']))
    groups = defaultdict(list)
    for index, row in enumerate(ordered):
        if (row['OP NAME'], row['CORE COUNT']) != ('MatmulDeviceOperation', '32'):
            continue
        if index == 0 or index + 1 == len(ordered):
            raise ValueError('Complete projection neighborhood required')
        previous, following = ordered[index - 1], ordered[index + 1]
        identity = previous['OP NAME'], previous['CORE COUNT']
        if identity not in PREDECESSORS or (following['OP NAME'], following['CORE COUNT']) != (
                'ReduceScatterMinimalAsyncDeviceOperation', '10'):
            raise ValueError('Unrecognized projection sequence; do not guess source identity')
        if not (int(previous['DEVICE KERNEL END CYCLE']) <= int(row['DEVICE KERNEL START CYCLE'])
                <= int(row['DEVICE KERNEL END CYCLE']) <= int(following['DEVICE KERNEL START CYCLE'])):
            raise ValueError('Ordered nonoverlapping neighbors required for sequence inference')
        duration = float(row['DEVICE KERNEL DURATION [ns]'])
        if not 0 < duration < 1e9:
            raise ValueError('Finite positive bounded projection duration required')
        groups[PREDECESSORS[identity][0]].append(duration / 1e6)
    if {name: len(values) for name, values in groups.items()} != dict(PREDECESSORS.values()):
        raise ValueError('All 64 MLP, 48 GDN and 16 attention output projections required')
    return {name: dict(calls=len(values), summed_ms=sum(values), median_call_ms=statistics.median(values))
        for name, values in groups.items()}


def main(root, scope_source):
    root = Path(root)
    evidence = analyze(root)
    report = json.loads((root / 'request.json').read_bytes())
    source = Path(scope_source)
    matching = [digest for name, digest in report['sources'].items() if Path(name).name == 'fused_t16_scope.py']
    if matching != [hashlib.sha256(source.read_bytes()).hexdigest()]:
        raise ValueError('Retained fused-MLP source must match the reviewed local forward sequence')
    expected = {(device['trace_id'], device['device'], replay['replay_session']): replay['operations']
        for device in evidence['devices'] for replay in device['replays']
        if replay['rows'] == 16 and not replay['first_replay']}
    sessions = defaultdict(list)
    with (root / 'metadata/cpp_device_perf_report.csv').open(newline='') as stream:
        for row in csv.DictReader(stream):
            if not row['METAL TRACE ID'] or not row['METAL TRACE REPLAY SESSION ID']:
                continue
            key = int(row['METAL TRACE ID']), row['DEVICE ID'], int(row['METAL TRACE REPLAY SESSION ID'])
            if key in expected:
                sessions[key].append(row)
    if not expected or {key: len(rows) for key, rows in sessions.items()} != expected:
        raise ValueError('Every admitted steady replay must be present')
    groups = defaultdict(list)
    for (trace, chip, replay), rows in sessions.items():
        for name, result in separate(rows).items():
            groups[chip, name].append(result)
    return dict(run=evidence['run'], request_sha256=evidence['request_sha256'],
        device_report_sha256=evidence['device_report_sha256'], scope_source_sha256=matching[0],
        source_sequence_verified=True,
        source_identity_is_inferred=True, committed_tg=None, performance_qualified=False,
        scope='Source-consistent ordered sequences, not profiler source labels or active compute',
        groups=[dict(chip=chip, projection=name, replays=len(values), calls=values[0]['calls'],
            median_summed_ms=statistics.median(value['summed_ms'] for value in values),
            median_call_ms=statistics.median(value['median_call_ms'] for value in values))
            for (chip, name), values in sorted(groups.items())])


if __name__ == '__main__':
    print(json.dumps(main(sys.argv[1], sys.argv[2]), indent=2))
