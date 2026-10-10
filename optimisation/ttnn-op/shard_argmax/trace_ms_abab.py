"""ABAB read of a verify-trace lever on trace_ms, from the engine logs of four timed arms (A control, B lever, A control, B lever).

    python3 trace_ms_abab.py --control A1/container.log A2/container.log --lever B1/container.log B2/container.log

WHY THIS AND NOT tok/s. The aggregate tok/s of a smoke test carries the host: staging (input_ms), binding, readback and the scheduler all move with
the load on the machine that runs the cards. S1's only timed ABAB so far (TS1-TS4) shows it: TS2's code_32k aggregate was 135.27 against controls of
156.22 and 157.86, yet its [PACKED-PHASE] trace_ms was 0.70 ms LOWER than the control's at every quantile, while its input_ms was 2.3 ms higher and its
readback 0.9 ms higher in the same tests (host contention); the repeat TS4 sat at parity on the aggregate and 0.76 ms lower on trace_ms. A device-side
lever is read on the device-side number: the median trace_ms of the 64-row block verifies (live=4), per arm and per third of the run (the tests of a
smoke run differ in context length, which moves trace_ms by tens of ms, so a pair is compared third against third), with the host phases beside it so a
loaded arm is named.

Per pair (control i, lever i), per third of the live-N lines and overall: the control and lever medians of trace_ms, their delta, and the ratio of the
medians of input_ms (HOST-SKEW when it leaves 0.85 .. 1.15: the arms did not see the same host). Verdict:

  PASS          every pair's overall delta is at most -min_gain_ms and no third of any pair has a delta above -min_gain_ms / 2;
  FAIL          a pair's overall delta is above -min_gain_ms / 2 (the lever does not save what it should);
  INCONCLUSIVE  fewer than MIN_ROUNDS live-N lines in any arm, or the pairs disagree by more than DISAGREE_MS, or the deltas sit between the two lines.

Exit: 0 PASS; 1 FAIL; 2 INCONCLUSIVE or bad input. Stdlib only.
"""

import argparse
import json
import re
import statistics
import sys

LINE = re.compile(r'\[PACKED-PHASE\] round=(\d+) users=(\d+) bind_ms=([\d.]+) input_ms=([\d.]+) trace_ms=([\d.]+) sync_ms=([\d.]+) '
                  r'readback_ms=([\d.]+) live=(\d+)')
MIN_ROUNDS = 100
DISAGREE_MS = 0.3
SKEW = (0.85, 1.15)


def read_rounds(path, live):
    """[(bind, input, trace, sync, readback)] of every [PACKED-PHASE] line of the log with live == `live`, in log order."""
    rounds = []
    with open(path, errors='replace') as handle:
        for line in handle:
            match = LINE.search(line)
            if match and int(match.group(8)) == live:
                rounds.append(tuple(float(match.group(index)) for index in (3, 4, 5, 6, 7)))
    return rounds


def median(rounds, index):
    return statistics.median(item[index] for item in rounds)


def thirds(rounds, splits):
    size = len(rounds) // splits
    return [rounds[index * size:(index + 1) * size if index < splits - 1 else len(rounds)] for index in range(splits)]


def compare(control, lever, splits):
    """One pair's numbers: the overall and per-split medians of trace_ms and input_ms and the deltas."""
    def block(a, b):
        ratio = median(b, 1) / median(a, 1) if median(a, 1) else float('inf')
        return dict(control_trace_ms=round(median(a, 2), 3), lever_trace_ms=round(median(b, 2), 3), delta_ms=round(median(b, 2) - median(a, 2), 3),
                    control_input_ms=round(median(a, 1), 3), lever_input_ms=round(median(b, 1), 3), input_ratio=round(ratio, 3),
                    host_skew=not SKEW[0] <= ratio <= SKEW[1],
                    control_readback_ms=round(median(a, 4), 3), lever_readback_ms=round(median(b, 4), 3), n=(len(a), len(b)))

    found = block(control, lever)
    found['splits'] = [block(a, b) for a, b in zip(thirds(control, splits), thirds(lever, splits)) if a and b]
    return found


def verdict(pairs, min_gain_ms, minimum=MIN_ROUNDS):
    """('PASS' | 'FAIL' | 'INCONCLUSIVE', [reason]) from the pairs' comparisons."""
    reasons = []
    for index, pair in enumerate(pairs, 1):
        if min(pair['n']) < minimum:
            reasons.append('pair %d has %d/%d live rounds, fewer than %d' % (index, pair['n'][0], pair['n'][1], minimum))
    if reasons:
        return 'INCONCLUSIVE', reasons
    deltas = [pair['delta_ms'] for pair in pairs]
    for index, pair in enumerate(pairs, 1):
        if pair['delta_ms'] > -min_gain_ms / 2:
            reasons.append('pair %d saves %.3f ms of trace_ms, less than half of the required %.3f' % (index, -pair['delta_ms'], min_gain_ms))
    if reasons:
        return 'FAIL', reasons
    if len(deltas) > 1 and max(deltas) - min(deltas) > DISAGREE_MS:
        return 'INCONCLUSIVE', ['the pairs disagree: deltas %s differ by more than %.2f ms' % (deltas, DISAGREE_MS)]
    for index, pair in enumerate(pairs, 1):
        if pair['delta_ms'] > -min_gain_ms:
            reasons.append('pair %d saves %.3f ms, short of the required %.3f' % (index, -pair['delta_ms'], min_gain_ms))
        for number, split in enumerate(pair['splits'], 1):
            if split['delta_ms'] > -min_gain_ms / 2:
                reasons.append('pair %d split %d saves only %.3f ms' % (index, number, -split['delta_ms']))
    return ('INCONCLUSIVE', reasons) if reasons else ('PASS', [])


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--control', nargs='+', required=True, help='engine logs of the control arms (A)')
    parser.add_argument('--lever', nargs='+', required=True, help='engine logs of the lever arms (B), in the same order as the controls')
    parser.add_argument('--live', type=int, default=4, help='live users of the lines compared (4 = the 64-row block verify)')
    parser.add_argument('--splits', type=int, default=3)
    parser.add_argument('--min-gain-ms', type=float, default=0.5)
    parser.add_argument('--json', default=None)
    return parser


def main(argv=None, out=print):
    options = build_parser().parse_args(argv)
    if len(options.control) != len(options.lever) or options.splits < 1 or options.min_gain_ms <= 0:
        sys.stderr.write('refusing: one lever log per control log, splits >= 1, min gain > 0\n')
        return 2
    pairs = []
    try:
        for control_path, lever_path in zip(options.control, options.lever):
            control, lever = read_rounds(control_path, options.live), read_rounds(lever_path, options.live)
            if not control or not lever:
                sys.stderr.write('no [PACKED-PHASE] lines with live=%d in %s or %s\n' % (options.live, control_path, lever_path))
                return 2
            pairs.append(compare(control, lever, options.splits))
    except OSError as error:
        sys.stderr.write('%s\n' % error)
        return 2
    for index, pair in enumerate(pairs, 1):
        out('TRACE_ABAB pair=%d live=%d control_trace_ms=%.3f lever_trace_ms=%.3f delta_ms=%.3f input_ratio=%.3f host_skew=%s n=%d/%d' % (
            index, options.live, pair['control_trace_ms'], pair['lever_trace_ms'], pair['delta_ms'], pair['input_ratio'], pair['host_skew'], *pair['n']))
        for number, split in enumerate(pair['splits'], 1):
            out('TRACE_ABAB pair=%d split=%d control_trace_ms=%.3f lever_trace_ms=%.3f delta_ms=%.3f input_ratio=%.3f host_skew=%s' % (
                index, number, split['control_trace_ms'], split['lever_trace_ms'], split['delta_ms'], split['input_ratio'], split['host_skew']))
    text, reasons = verdict(pairs, options.min_gain_ms)
    out('TRACE_ABAB verdict=%s min_gain_ms=%.3f%s' % (text, options.min_gain_ms, ''.join(' | ' + reason for reason in reasons)))
    if options.json:
        with open(options.json, 'w') as handle:
            json.dump(dict(verdict=text, reasons=reasons, pairs=pairs, live=options.live, min_gain_ms=options.min_gain_ms), handle, indent=2, sort_keys=True)
    return {'PASS': 0, 'FAIL': 1}.get(text, 2)


if __name__ == '__main__':
    sys.exit(main())
