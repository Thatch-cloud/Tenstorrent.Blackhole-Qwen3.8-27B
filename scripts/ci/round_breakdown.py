"""Decompose the packed decode round by phase, grouped by how many users were live.

Reads a server.log with [PHASE] lines. A round starts at an `execute total=N ... spec=K`
line with K > 0 (a speculative decode step for K users) and ends at the next such line,
so each window is one full steady-state cycle: this round's verify and commits plus the
next round's proposals.

Phases NEST. `propose_pair` runs inside `prepare_proposals`, so summing every `end X ms`
line counts the pair passes twice. This tracks begin/end depth and adds only TOP-LEVEL
phases to the round total; nested phases are listed separately, indented, and are
already inside their parent's figure. An earlier version summed everything, and a
verifier caught it overstating the named total by the pair time (~6 ms at four users).

Usage: round_breakdown.py <server.log>
"""
import re
import statistics
import sys
from collections import defaultdict

TS = re.compile(r'(\d\d):(\d\d):(\d\d)\.(\d{3}) \|')
EXEC = re.compile(r'\[PHASE\] execute total=(\d+) new=(\d+) cached=(\d+) spec=(\d+)')
BEGIN = re.compile(r'\[PHASE\] ([a-z_]+) .*?\bbegin\b')
END = re.compile(r'\[PHASE\] ([a-z_]+) .*? end ([0-9.]+) ms')


def seconds(line):
    m = TS.search(line)
    if not m:
        return None
    h, mi, s, ms = (int(g) for g in m.groups())
    return h * 3600 + mi * 60 + s + ms / 1000.0


def main(path):
    rounds = []
    current = None
    stack = []
    for line in open(path, encoding='utf-8', errors='replace'):
        e = EXEC.search(line)
        if e:
            t = seconds(line)
            if current is not None and t is not None:
                current['wall'] = (t - current['t']) * 1000.0
                rounds.append(current)
            current = None
            spec = int(e.group(4))
            if spec > 0 and t is not None:
                current = {'t': t, 'users': spec,
                           'top': defaultdict(float), 'nested': defaultdict(float)}
            continue
        m = END.search(line)
        if m:
            name, ms = m.group(1), float(m.group(2))
            if name in stack:
                # Remove the most recent open instance of this phase.
                del stack[len(stack) - 1 - stack[::-1].index(name)]
            nested = bool(stack)
            if current is not None:
                (current['nested'] if nested else current['top'])[name] += ms
            continue
        b = BEGIN.search(line)
        if b:
            stack.append(b.group(1))

    by_users = defaultdict(list)
    for r in rounds:
        by_users[r['users']].append(r)

    for users in sorted(by_users, reverse=True):
        rs = by_users[users]
        walls = sorted(r['wall'] for r in rs)
        # Drop the slowest decile: prefill steps interleaved between decode steps land
        # inside a round window and are not decode cost.
        keep = rs if len(rs) < 10 else [r for r in rs if r['wall'] <= walls[int(len(walls) * 0.9) - 1]]
        med = statistics.median(r['wall'] for r in keep)
        print('users=%d  rounds=%d  median round %.1f ms' % (users, len(rs), med))
        top = defaultdict(list)
        nested = defaultdict(list)
        for r in keep:
            for k, v in r['top'].items():
                top[k].append(v)
            for k, v in r['nested'].items():
                nested[k].append(v)
        accounted = 0.0
        for k in sorted(top, key=lambda k: -statistics.median(top[k])):
            v = statistics.median(top[k])
            accounted += v
            print('    %-20s %7.1f ms  (%d rounds)' % (k, v, len(top[k])))
        for k in sorted(nested, key=lambda k: -statistics.median(nested[k])):
            print('      (nested) %-11s %7.1f ms  - already inside its parent' % (k, statistics.median(nested[k])))
        print('    %-20s %7.1f ms  (round minus the sum of top-level phases)' % ('UNACCOUNTED', med - accounted))
        print()


if __name__ == '__main__':
    main(sys.argv[1])
