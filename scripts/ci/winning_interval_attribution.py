"""Observed exclusive kernel intervals, not a dependency critical path or speedup prediction."""

from collections import Counter, defaultdict
import csv
import json
from pathlib import Path
import statistics
import sys

from request_verifier_profile_report import intervals
from winning_device_attribution import analyze


def coverage(rows):
    timing = intervals(rows)
    events = defaultdict(Counter)
    for row in rows:
        group = (row['OP NAME'], row['CORE COUNT'])
        events[int(row['DEVICE KERNEL START CYCLE'])][group] += 1
        events[int(row['DEVICE KERNEL END CYCLE'])][group] -= 1
    active, inclusive, exclusive = Counter(), Counter(), Counter()
    overlap = 0
    previous = min(events)
    for position, changes in sorted(events.items()):
        elapsed = position - previous
        groups = [group for group, count in active.items() if count > 0]
        for group in groups:
            inclusive[group] += elapsed
        if len(groups) == 1:
            exclusive[groups[0]] += elapsed
        elif len(groups) > 1:
            overlap += elapsed
        active.update(changes)
        if any(count < 0 for count in active.values()):
            raise ValueError('Invalid kernel interval balance')
        previous = position
    if any(active.values()):
        raise ValueError('Unclosed kernel intervals')
    scale = timing['clock_ns_per_cycle_estimate'] / 1e6
    return dict(groups={group: dict(inclusive_ms=duration * scale, exclusive_ms=exclusive[group] * scale)
        for group, duration in inclusive.items()}, overlapping_groups_ms=overlap * scale)


def main(root):
    root = Path(root)
    evidence = analyze(root)
    expected = {(device['trace_id'], device['device'], replay['replay_session']): replay['operations']
        for device in evidence['devices'] for replay in device['replays']
        if replay['rows'] == 16 and not replay['first_replay']}
    sessions = defaultdict(list)
    with (root / 'metadata/cpp_device_perf_report.csv').open(newline='') as stream:
        for row in csv.DictReader(stream):
            if not row.get('METAL TRACE ID') or not row.get('METAL TRACE REPLAY SESSION ID'):
                continue
            key = (int(row['METAL TRACE ID']), row['DEVICE ID'], int(row['METAL TRACE REPLAY SESSION ID']))
            if key in expected:
                sessions[key].append(row)
    if not expected or {key: len(rows) for key, rows in sessions.items()} != expected:
        raise ValueError('Every operation in each admitted steady replay required')
    groups = defaultdict(list)
    overlaps = defaultdict(list)
    for (trace, chip, replay), rows in sessions.items():
        result = coverage(rows)
        overlaps[chip].append(result['overlapping_groups_ms'])
        for (operation, cores), values in result['groups'].items():
            groups[(chip, operation, cores)].append(values)
    output = []
    for (chip, operation, cores), repeats in groups.items():
        if len(repeats) != len(overlaps[chip]):
            raise ValueError('Group coverage changed between admitted replays')
        output.append(dict(chip=chip, operation=operation, cores=cores, replays=len(repeats),
            **{name: statistics.median(value[name] for value in repeats) for name in ('inclusive_ms', 'exclusive_ms')}))
    return dict(run=evidence['run'], request_sha256=evidence['request_sha256'],
        device_report_sha256=evidence['device_report_sha256'],
        groups=sorted(output, key=lambda value: (value['chip'], -value['exclusive_ms'])),
        overlapping_groups_ms={chip: statistics.median(values) for chip, values in overlaps.items()},
        scope=__doc__, committed_tg=None, performance_qualified=False)


if __name__ == '__main__':
    print(json.dumps(main(sys.argv[1]), indent=2))
