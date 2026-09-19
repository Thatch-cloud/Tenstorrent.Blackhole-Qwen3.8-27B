"""Attribute a tt-metal device profile by operation. Observed intervals, not a critical path.

Written after doing this by hand four times. The rules that matter, learned the hard way:

  * Discard the first replay of each trace - it carries compilation.
  * Identify an operation by its position in the execution sequence, not by core count
    alone. A 99-core GenericOp looked like unattributed overhead until the surrounding
    ops showed it between a LayerNorm and the down projection, i.e. MLP gate/up.
  * Call counts are the cleanest identity hint: an op running once per layer tracks the
    layer count, so 48 and 64 distinguish the two layer families in a hybrid model.
  * Durations include waits and can overlap, so sums are not a dependency-graph critical
    path and must not be converted into a throughput claim.
"""

import argparse
import collections
import csv
import io
import statistics as st
from pathlib import Path

NS = 1e6


def load(path, device):
    per = collections.defaultdict(lambda: collections.defaultdict(lambda: collections.defaultdict(float)))
    calls = collections.defaultdict(collections.Counter)
    order = collections.defaultdict(list)
    with io.open(path, encoding='utf-8', errors='replace', newline='') as handle:
        for row in csv.DictReader(handle):
            if row.get('DEVICE ID') != device:
                continue
            try:
                ns = float(row.get('DEVICE KERNEL DURATION [ns]') or 0)
                cores = int(float(row.get('CORE COUNT') or 0))
                gc = int(float(row.get('GLOBAL CALL COUNT') or 0))
            except (TypeError, ValueError):
                continue
            trace = row.get('METAL TRACE ID') or 'untraced'
            session = row.get('METAL TRACE REPLAY SESSION ID', '')
            op = row.get('OP NAME', '')
            per[trace][session][(op, cores)] += ns
            calls[trace][(op, cores)] += 1
            order[trace].append((gc, op, cores, ns))
    return per, calls, order


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--csv', type=Path, required=True)
    parser.add_argument('--device', default='0')
    parser.add_argument('--trace', help='restrict to one METAL TRACE ID')
    parser.add_argument('--sequence', type=int, default=0,
                        help='print the first N ops in execution order, which is how an '
                             'ambiguous op gets identified')
    parser.add_argument('--layers', default='48,64',
                        help='layer counts to flag call-count multiples of')
    options = parser.parse_args()

    per, calls, order = load(options.csv, options.device)
    layers = [int(v) for v in options.layers.split(',')]
    traces = [options.trace] if options.trace else sorted(
        per, key=lambda t: -sum(st.median([sum(per[t][s].values()) for s in per[t]]) for _ in (0,)))

    for trace in traces:
        if trace not in per:
            print('no such trace: %s' % trace)
            continue
        sessions = sorted(per[trace])
        steady = sessions[1:] if len(sessions) > 1 else sessions
        keys = set()
        for s in steady:
            keys |= set(per[trace][s])
        rows = sorted(((st.median([per[trace][s].get(k, 0.0) for s in steady]) / NS, k)
                       for k in keys), reverse=True)
        total = sum(ms for ms, _ in rows)
        if total <= 0:
            continue
        print('=== trace %s | %d replays (first discarded) | median %.2f ms ==='
              % (trace, len(sessions), total))
        print('%-42s %7s %8s %7s %6s  %s' % ('op', 'cores', 'ms', 'calls', '%', 'per layer?'))
        for ms, (op, cores) in rows[:18]:
            n = calls[trace][(op, cores)] // max(len(sessions), 1)
            tag = ','.join('x%d' % L for L in layers if n and n % L == 0)
            print('%-42s %7d %8.3f %7d %5.1f%%  %s'
                  % (op.replace('DeviceOperation', '')[:42], cores, ms, n, 100 * ms / total, tag))
        print()

    if options.sequence and traces:
        trace = traces[0]
        seq = sorted(order[trace])[:options.sequence]
        print('--- first %d ops of trace %s in execution order ---' % (len(seq), trace))
        for gc, op, cores, ns in seq:
            print('  %-42s cores=%3d %9.1f us' % (op.replace('DeviceOperation', '')[:42], cores, ns / 1e3))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
