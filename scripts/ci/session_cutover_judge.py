"""The cutover's timing rule, read from the session's timed block (references/tp4-session-0808-jobs/ORDER.txt, CUTOVER-JUDGE).

The rule (W-2's cutover rules, owner-approved 2026-10-07): Lever N's steady cost on the window image sits INSIDE the A-to-A drift floor at the steady AND the 32k
shape, and the production bytes (T0s, tp4-serve-10) sit inside the floor of the control (TA1). The timed jobs are T0s, TA1, TL1, TA2, TL2, TA3 (the pairs
(TA1, TL1) and (TA2, TL2); the floor is the spread of the three A runs' median round times). The reading is w2ln_timing_compare's: eight-live fully packed rounds,
paired by mean-context bucket inside one mean-position window, never pooled across lengths; a pair with fewer matched rounds than the pre-registered minimum is
VOID, and a VOID pair is never 'inside'.

    python3 -B scripts/ci/session_cutover_judge.py --t0s DIR --a DIR DIR DIR --l DIR DIR --baseline-load DIR

Each DIR is a job's artifact directory (the driver's {DIR:<job>}); the container log is `container.log` (else `server.log`) and the host load log `load.log`,
found at any depth. `--baseline-load` names the serving baseline whose load.log calibrates the load limit (S0-CTL: the engine's own threads raise the load average, so
the idle host is no baseline); without it, or with a baseline that has no samples, the judge refuses (exit 1, an `error` in the JSON) unless `--no-load-check` says the load is not to be read.

'Inside' for Lever N is one-sided: the pair's delta (TL minus TA, milliseconds a round) must not exceed the floor; a Lever N arm that is FASTER is no cost. For the
production bytes it is two-sided (|T0s - TA1| <= floor): the window image must run at production's speed with Lever N off.

Exit 0 only when every window holds; 1 when a window does not, or when a log is missing or the load cannot be calibrated (the JSON says which pair and why, or carries an
`error`: a judge that cannot read its input fails closed, and the driver's dry run, which runs it on fixtures, tells a crash (exit 2: bad arguments, a traceback) from a 'no' by the exit status).
"""

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import w2ln_timing_compare as timing  # noqa: E402

WINDOWS = ('steady', '32k')
LOG_NAMES = ('container.log', 'server.log')
LOAD_NAME = 'load.log'


class Missing(Exception):
    pass


def find(directory, names):
    """The first file of `names` (in order) found at any depth under `directory`, or None."""
    for name in names:
        for root, _dirs, files in sorted(os.walk(directory)):
            if name in files:
                return os.path.join(root, name)
    return None


def read(path):
    with open(path, encoding='utf-8', errors='replace') as handle:
        return handle.read()


def job_texts(directory, label):
    log = find(directory, LOG_NAMES)
    if log is None:
        raise Missing('%s: no %s under %s' % (label, ' or '.join(LOG_NAMES), directory))
    load = find(directory, (LOAD_NAME,))
    return dict(log=read(log), load=read(load) if load else '')


def pair(a, b, window, max_load, max_gap, min_matched):
    result = timing.compare(a['log'], b['log'], timing.DEFAULT_LIVE, timing.DEFAULT_BUCKET, min_matched, window)
    if max_load is not None:
        timing.apply_load(result, a['log'] + '\n' + a['load'], b['log'] + '\n' + b['load'], max_load, max_gap)
    return result


def judge(t0s, a_runs, l_runs, max_load=None, max_gap=timing.DEFAULT_LOAD_GAP, min_matched=timing.DEFAULT_MIN_MATCHED, windows=WINDOWS):
    """-> (holds, report). `t0s` and each run are dict(log=, load=)."""
    report, holds = {}, True
    for window in windows:
        floor = timing.drift_floor_ms([run['log'] for run in a_runs], timing.DEFAULT_LIVE, window)
        entry = dict(floor_ms=floor, pairs=[], reasons=[])
        if floor is None:
            entry['reasons'].append('no A-to-A floor: fewer than two control runs read in this window')
        for index, (a, l) in enumerate(zip(a_runs, l_runs), start=1):
            found = pair(a, l, window, max_load, max_gap, min_matched)
            row = dict(pair='TA%d-TL%d' % (index, index), verdict=found['verdict'], delta_ms=found.get('delta_ms'), matched=found.get('matched'),
                       reason=found.get('reason'))
            if found['verdict'] != 'MEASURED':
                entry['reasons'].append('%s is %s (%s): never read as inside' % (row['pair'], found['verdict'], found.get('reason')))
            elif floor is not None and found['delta_ms'] > floor:
                entry['reasons'].append('%s: Lever N costs %.2f ms a round, past the %.2f ms floor' % (row['pair'], found['delta_ms'], floor))
            entry['pairs'].append(row)
        found = pair(a_runs[0], t0s, window, max_load, max_gap, min_matched)
        row = dict(pair='TA1-T0s', verdict=found['verdict'], delta_ms=found.get('delta_ms'), matched=found.get('matched'), reason=found.get('reason'))
        if found['verdict'] != 'MEASURED':
            entry['reasons'].append('TA1-T0s is %s (%s): never read as inside' % (found['verdict'], found.get('reason')))
        elif floor is not None and abs(found['delta_ms']) > floor:
            entry['reasons'].append('TA1-T0s: the production bytes differ by %.2f ms a round, past the %.2f ms floor' % (found['delta_ms'], floor))
        entry['pairs'].append(row)
        entry['holds'] = not entry['reasons']
        holds = holds and entry['holds']
        report[window] = entry
    return holds, report


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--t0s', required=True, help='the artifact directory of T0s (production bytes)')
    parser.add_argument('--a', nargs=3, required=True, metavar='DIR', help='TA1 TA2 TA3')
    parser.add_argument('--l', nargs=2, required=True, metavar='DIR', help='TL1 TL2 (paired with TA1 and TA2)')
    parser.add_argument('--baseline-load', help="a serving baseline's artifact directory (S0-CTL): its load.log calibrates the load limit")
    parser.add_argument('--no-load-check', action='store_true', help='read the pairs without the load rule (a stated exception)')
    parser.add_argument('--load-margin', type=float, default=timing.DEFAULT_LOAD_MARGIN)
    parser.add_argument('--load-gap', type=float, default=timing.DEFAULT_LOAD_GAP)
    parser.add_argument('--min-matched', type=int, default=timing.DEFAULT_MIN_MATCHED)
    return parser


def main(argv=None, out=print):
    options = build_parser().parse_args(argv)
    try:
        t0s = job_texts(options.t0s, 'T0s')
        a_runs = [job_texts(path, 'TA%d' % (index + 1)) for index, path in enumerate(options.a)]
        l_runs = [job_texts(path, 'TL%d' % (index + 1)) for index, path in enumerate(options.l)]
        max_load = None
        if not options.no_load_check:
            if not options.baseline_load:
                raise Missing('no --baseline-load: the load limit cannot be calibrated (--no-load-check reads without it)')
            baseline = find(options.baseline_load, (LOAD_NAME,))
            max_load = timing.load_limit(read(baseline), options.load_margin) if baseline else None
            if max_load is None:
                raise Missing('the serving baseline %s has no [LOAD] samples: the load limit cannot be calibrated' % options.baseline_load)
    except Missing as error:
        out(json.dumps(dict(holds=False, error=str(error))))
        return 1
    holds, report = judge(t0s, a_runs, l_runs, max_load, options.load_gap, options.min_matched)
    out(json.dumps(dict(holds=holds, max_load=max_load, windows=report), sort_keys=True))
    return 0 if holds else 1


if __name__ == '__main__':
    sys.exit(main())
