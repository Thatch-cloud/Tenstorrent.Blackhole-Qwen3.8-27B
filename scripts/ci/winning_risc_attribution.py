"""Per-RISC duration envelopes from validated winning T16 device replays, not stall attribution."""

from collections import defaultdict
import csv
import json
import math
from pathlib import Path
import statistics
import sys

from winning_device_attribution import analyze


RISCS = ('BRISC', 'NCRISC', 'TRISC0', 'TRISC1', 'TRISC2')


def summarize(devices, rows):
    expected = {(device['trace_id'], device['device'], replay['replay_session']): replay['operations']
        for device in devices for replay in device['replays'] if replay['rows'] == 16 and not replay['first_replay']}
    counts = defaultdict(int)
    sessions = defaultdict(lambda: defaultdict(list))
    for row in rows:
        if not row.get('METAL TRACE ID') or not row.get('METAL TRACE REPLAY SESSION ID'):
            continue
        key = (int(row['METAL TRACE ID']), row['DEVICE ID'], int(row['METAL TRACE REPLAY SESSION ID']))
        if key not in expected:
            continue
        counts[key] += 1
        group = (*key, row['OP NAME'], row['CORE COUNT'])
        values = sessions[group]
        for risc in RISCS:
            value = row.get(f'DEVICE {risc} KERNEL DURATION [ns]', '')
            if value in ('', None):
                continue
            duration = float(value)
            if not math.isfinite(duration) or duration < 0:
                raise ValueError('Finite nonnegative RISC duration or explicit missing value required')
            values[risc].append(duration)
    if not expected or dict(counts) != expected:
        raise ValueError('Every operation from each validated steady T16 replay required')
    groups = defaultdict(list)
    for (trace, chip, replay, op, cores), values in sessions.items():
        groups[(trace, chip, op, cores)].append({risc: dict(calls=len(values[risc]), ns=sum(values[risc])) for risc in RISCS})
    output = []
    for (trace, chip, op, cores), repeats in sorted(groups.items()):
        if any(len({entry[risc]['calls'] for entry in repeats}) != 1 for risc in RISCS):
            raise ValueError('RISC measurement coverage changed between steady replays')
        output.append(dict(trace_id=trace, chip=chip, op=op, cores=cores, replays=len(repeats),
            riscs={risc: dict(calls_per_replay=repeats[0][risc]['calls'],
                median_summed_ms=statistics.median(entry[risc]['ns'] for entry in repeats) / 1e6
                    if repeats[0][risc]['calls'] else None) for risc in RISCS}))
    return output


def main(root):
    root = Path(root)
    evidence = analyze(root)
    with (root / 'metadata/cpp_device_perf_report.csv').open(newline='') as stream:
        groups = summarize(evidence['devices'], csv.DictReader(stream))
    return dict(run=evidence['run'], request_sha256=evidence['request_sha256'],
        device_report_sha256=evidence['device_report_sha256'], groups=groups,
        scope=__doc__, committed_tg=None, performance_qualified=False,
        caveat='RISC envelopes include waits and can overlap; these are not active compute or memory-stall counters.')


if __name__ == '__main__':
    print(json.dumps(main(sys.argv[1]), indent=2))
