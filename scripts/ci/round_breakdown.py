"""Decompose the packed decode round by phase, grouped by how many users were live.

Reads a server.log with [PHASE] lines. A round starts at an `execute total=N ... spec=K`
line with K > 0 (a speculative decode step for K users) and ends at the next such line.
Within a round, every `<label> ... end X ms` duration is summed by label.

Usage: round_breakdown.py <server.log>
"""
import re
import statistics
import sys
from collections import defaultdict

TS = re.compile(r'(\d\d):(\d\d):(\d\d)\.(\d{3}) \|')
EXEC = re.compile(r'\[PHASE\] execute total=(\d+) new=(\d+) cached=(\d+) spec=(\d+)')
END = re.compile(r'\[PHASE\] ([a-z_]+) .*? end ([0-9.]+) ms')


def seconds(line):
    m = TS.search(line)
    if not m:
        return None
    h, mi, s, ms = (int(g) for g in m.groups())
    return h * 3600 + mi * 60 + s + ms / 1000.0


def main(path):
    rounds = []          # (users, wall_ms, {label: ms})
    current = None
    for line in open(path, encoding='utf-8', errors='replace'):
        e = EXEC.search(line)
        if e:
            t = seconds(line)
            spec = int(e.group(4))
            if current is not None and t is not None and current['t'] is not None:
                current['wall'] = (t - current['t']) * 1000.0
                rounds.append(current)
            current = None
            if spec > 0 and t is not None:
                current = {'t': t, 'users': spec, 'phases': defaultdict(float)}
            continue
        if current is not None:
            m = END.search(line)
            if m:
                current['phases'][m.group(1)] += float(m.group(2))

    by_users = defaultdict(list)
    for r in rounds:
        if 'wall' in r:
            by_users[r['users']].append(r)

    for users in sorted(by_users, reverse=True):
        rs = by_users[users]
        walls = sorted(r['wall'] for r in rs)
        # Drop the slowest decile: prefill steps interleaved between decode steps
        # land inside a "round" window and are not decode cost.
        keep = rs if len(rs) < 10 else [r for r in rs if r['wall'] <= walls[int(len(walls) * 0.9) - 1]]
        med = statistics.median(r['wall'] for r in keep)
        print('users=%d  rounds=%d  median round %.1f ms' % (users, len(rs), med))
        labels = defaultdict(list)
        for r in keep:
            for k, v in r['phases'].items():
                labels[k].append(v)
        accounted = 0.0
        for k in sorted(labels, key=lambda k: -statistics.median(labels[k])):
            v = statistics.median(labels[k])
            accounted += v
            print('    %-20s %7.1f ms  (%d rounds)' % (k, v, len(labels[k])))
        print('    %-20s %7.1f ms  (round minus the sum of named phases)' % ('UNACCOUNTED', med - accounted))
        print()


if __name__ == '__main__':
    main(sys.argv[1])
