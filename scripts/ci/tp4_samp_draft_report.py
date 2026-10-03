"""The tau report of the tp4/samp-draft window: control arms against a lever arm, from the timed arms' container logs.

A drafter change never changes the served text (the target verifies every draft); what it can change is the proposals, hence the
acceptance length tau. D2a (conv I/O) and D2c (head copies) hand the compute kernels byte-identical tiles, so their tau must equal the
control's; the report states the rule it applies instead of assuming that:

  - the controls (the two A arms of the ABAB, run on the same image) must reproduce each other: full-draft mean tau within
    CONTROL_TOLERANCE (greedy decoding with identical proposals is deterministic, so a spread here is host noise in WHICH rounds ran, not
    in tau; a wide spread means the pool is too thin to judge a lever);
  - the lever's pooled full-draft mean tau must be at least the controls' pooled mean less TOLERANCE (1%), over at least MIN_ROUNDS
    full-draft rounds on each side;
  - no per-position acceptance rate of the lever may fall more than POSITION_POINTS (2 points) below the controls'.

Exit 0: the rule holds. 1: it does not. 2: too few full-draft rounds to judge (the report says which side). Reads each log with
acceptance_report.report (the same four-source reader the gates use); the log paths are arguments, nothing is fetched.

    python scripts/ci/tp4_samp_draft_report.py --control A1.log A2.log --lever B.log [--lever B2.log] [--json out.json]

Stdlib plus acceptance_report; py 3.7 importable.
"""

import argparse
import json
import sys
from pathlib import Path

import acceptance_report

TOLERANCE = 0.01
CONTROL_TOLERANCE = 0.01
POSITION_POINTS = 0.02
MIN_ROUNDS = 2000


def arm_stats(log_text):
    """{'rounds', 'mean', 'positions'} of one container log's full-draft acceptance, or None when it has no full-draft round."""
    result = acceptance_report.report(log_text)
    overall = result.get('overall') or {}
    full = overall.get('full_draft')
    if not full or not full.get('rounds'):
        return None
    return dict(rounds=full['rounds'], mean=full['mean_emitted'], positions=list(overall.get('per_position_full_draft') or []))


def pool(stats):
    """Rounds-weighted pooled mean and per-position rates of several logs' stats (None entries are skipped), or None."""
    stats = [entry for entry in stats if entry]
    rounds = sum(entry['rounds'] for entry in stats)
    if not rounds:
        return None
    mean = sum(entry['rounds'] * entry['mean'] for entry in stats) / rounds
    width = max((len(entry['positions']) for entry in stats), default=0)
    positions = []
    for index in range(width):
        pairs = [(entry['rounds'], entry['positions'][index]) for entry in stats
                 if index < len(entry['positions']) and entry['positions'][index] is not None]
        weight = sum(count for count, _ in pairs)
        positions.append(sum(count * rate for count, rate in pairs) / weight if weight else None)
    return dict(rounds=rounds, mean=mean, positions=positions)


def judge(controls, levers, *, tolerance=TOLERANCE, control_tolerance=CONTROL_TOLERANCE, position_points=POSITION_POINTS,
          min_rounds=MIN_ROUNDS):
    """The verdict dict for lists of per-log stats (see arm_stats): {'verdict': 'PASS'|'FAIL'|'INSUFFICIENT', 'reasons': [...], ...}."""
    control, lever = pool(controls), pool(levers)
    reasons, verdict = [], 'PASS'
    if control is None or lever is None:
        return dict(verdict='INSUFFICIENT', reasons=['no full-draft round in the %s logs' % (
            'control' if control is None else 'lever')], control=control, lever=lever, spread=None)
    for name, arm in (('control', control), ('lever', lever)):
        if arm['rounds'] < min_rounds:
            verdict = 'INSUFFICIENT'
            reasons.append('%s pools %d full-draft rounds, fewer than the %d the rule needs' % (name, arm['rounds'], min_rounds))
    spread = None
    live = [entry for entry in controls if entry]
    if len(live) >= 2:
        means = [entry['mean'] for entry in live]
        spread = (max(means) - min(means)) / control['mean']
        if spread > control_tolerance:
            reasons.append('the controls differ by %.2f%% of their mean (more than %.2f%%): the pool is too noisy to judge a lever'
                           % (100 * spread, 100 * control_tolerance))
            if verdict == 'PASS':
                verdict = 'INSUFFICIENT'
    floor = control['mean'] * (1 - tolerance)
    if lever['mean'] < floor:
        verdict = 'FAIL'
        reasons.append('lever tau %.4f is below the controls\' %.4f less %.1f%% (%.4f)' % (
            lever['mean'], control['mean'], 100 * tolerance, floor))
    drops = []
    for index, (mine, theirs) in enumerate(zip(lever['positions'], control['positions']), 1):
        if mine is not None and theirs is not None and theirs - mine > position_points:
            drops.append((index, theirs, mine))
    if drops:
        verdict = 'FAIL'
        reasons.append('per-position acceptance fell more than %.0f points at position(s) %s' % (
            100 * position_points, ', '.join('%d (%.3f -> %.3f)' % drop for drop in drops)))
    return dict(verdict=verdict, reasons=reasons, control=control, lever=lever, spread=spread,
                tau_ratio=lever['mean'] / control['mean'])


def summary_line(found):
    control, lever = found.get('control'), found.get('lever')
    if control is None or lever is None:
        return '[SAMP-DRAFT-TAU] verdict=%s %s' % (found['verdict'], '; '.join(found['reasons']))
    return ('[SAMP-DRAFT-TAU] verdict=%s control_rounds=%d control_tau=%.4f lever_rounds=%d lever_tau=%.4f ratio=%.4f spread=%s'
            % (found['verdict'], control['rounds'], control['mean'], lever['rounds'], lever['mean'], found['tau_ratio'],
               '-' if found['spread'] is None else '%.4f' % found['spread']))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--control', type=Path, nargs='+', required=True, help='the control arms\' container logs (A1 A2 ...)')
    parser.add_argument('--lever', type=Path, nargs='+', required=True, help='the lever arms\' container logs')
    parser.add_argument('--tolerance', type=float, default=TOLERANCE)
    parser.add_argument('--min-rounds', type=int, default=MIN_ROUNDS)
    parser.add_argument('--json', type=Path, help='also write the verdict here')
    options = parser.parse_args(argv)
    read = lambda path: arm_stats(path.read_text(encoding='utf-8', errors='replace'))  # noqa: E731
    found = judge([read(path) for path in options.control], [read(path) for path in options.lever],
                  tolerance=options.tolerance, min_rounds=options.min_rounds)
    print(summary_line(found))
    for reason in found['reasons']:
        print('[SAMP-DRAFT-TAU] ' + reason)
    if options.json:
        options.json.write_text(json.dumps(found, indent=2), encoding='utf-8')
    return {'PASS': 0, 'FAIL': 1, 'INSUFFICIENT': 2}[found['verdict']]


if __name__ == '__main__':
    sys.exit(main())
