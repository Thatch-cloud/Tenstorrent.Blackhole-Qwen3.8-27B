"""The tp4/octo-3 timed read (stdlib plus c2_smoke_check's log reader, Python 3.7 syntax): octo-everywhere against today's m3 rounds, ABAB over four boots.

    python3 scripts/ci/octo3_report.py --control-a C1.smoke --arm-a L1.smoke --control-b C2.smoke --arm-b L2.smoke [--arm-container L1.container] [--profile P] [--gain 0.05]

The octo-3 profiles run QWEN_FAST_OCTO=live: every eligible 8-live round is an octo round and none is an m3 one, so the control is not a second shape of the same boot (tp4/octo-2's
alternation, read by octo2_report.py) but the fused stack's own profile on another boot of the same image (references/fusion-jobs/O3). The rounds of the two boots cannot be paired: an
octo round commits at most 8 tokens a seat and an m3 round 16, so the accepted prefixes differ round by round and only the TEXTS agree. What is compared is what the smoke measures
for the whole all-live window of each test (c2_serving_smoke.live_report: the window every one of the eight users is live in): the aggregate decode rate and the slowest user's.

Per test (concurrent8_steady, and the deep ones concurrent8_code_32k_long and concurrent8_code_128k_long, whose budgets outlast the prefill queue so that eight seats decode together at depth - the
800-token deep tests never have six seats live at once and so never run an octo round):

  ratio A = arm A / control A, ratio B = arm B / control B on `live4_agg_tok_s` (the aggregate tokens per second of the all-live window)
  floors  = control B / control A and arm B / arm A (the boot-to-boot drift of the same profile)
  verdict = GAIN when both ratios exceed 1 + max(gain, both floors' distance from 1); LOSS when either falls below 1 - max(floors); else PARITY; UNREAD when a window is missing.

SHIP-GO needs no LOSS in any test and a GAIN in concurrent8_steady; with --arm-container the octo judge must be clean on the arm's container log (the shape executed). The text read is
octo_compare.py (every answer equal, user by user), run on the same four logs. This module prints OCTO3_REPORT lines and exits 0 (SHIP-GO), 1 (NO-GO) or 2 (unread).
"""
import argparse
import json
import sys
from pathlib import Path

TESTS = ('concurrent8_steady', 'concurrent8_code_32k_long', 'concurrent8_code_128k_long')
DEEP = TESTS[1:]
METRIC = 'live4_agg_tok_s'
MIN_WINDOW_S = 10.0           # an all-live window shorter than this is a tail, not a measurement
GAIN = 0.05


def window(results, test):
    """(aggregate rate, slowest user's rate, window seconds) of a test's all-live window, or None when the log has no such window."""
    entry = (results or {}).get(test)
    if not isinstance(entry, dict) or 'error' in entry:
        return None
    aggregate = entry.get('aggregate')
    if not isinstance(aggregate, dict):
        return None
    rate, slowest, seconds = aggregate.get(METRIC), aggregate.get('live4_min_tok_s'), aggregate.get('live4_window_s')
    if not isinstance(rate, (int, float)) or not isinstance(seconds, (int, float)) or seconds < MIN_WINDOW_S or rate <= 0:
        return None
    return float(rate), (float(slowest) if isinstance(slowest, (int, float)) else None), float(seconds)


def verdict(ratio_a, ratio_b, floor_control, floor_arm, gain=GAIN):
    """GAIN, PARITY or LOSS from the two pair ratios and the two floors (each a ratio of two boots of one profile)."""
    noise = max(abs(floor_control - 1.0), abs(floor_arm - 1.0))
    margin = max(gain, noise)
    if ratio_a > 1.0 + margin and ratio_b > 1.0 + margin:
        return 'GAIN'
    if ratio_a < 1.0 - noise and ratio_b < 1.0 - noise:
        return 'LOSS'
    if min(ratio_a, ratio_b) < 1.0 - max(noise, 0.02):
        return 'LOSS'
    return 'PARITY'


def read(control_a, arm_a, control_b, arm_b, gain=GAIN):
    """(rows, ship_go, notes): rows are dicts per test (ratios, floors, verdict); ship_go is True, False or None (unread)."""
    rows, notes = [], []
    for test in TESTS:
        windows = [window(log, test) for log in (control_a, arm_a, control_b, arm_b)]
        if any(item is None for item in windows):
            rows.append(dict(test=test, verdict='UNREAD', note='a boot has no all-live window of %d s or more in this test' % MIN_WINDOW_S))
            continue
        ca, aa, cb, ab = (item[0] for item in windows)
        ratio_a, ratio_b, floor_control, floor_arm = aa / ca, ab / cb, cb / ca, ab / aa
        rows.append(dict(test=test, control_a=ca, arm_a=aa, control_b=cb, arm_b=ab, ratio_a=round(ratio_a, 4), ratio_b=round(ratio_b, 4),
                         floor_control=round(floor_control, 4), floor_arm=round(floor_arm, 4), verdict=verdict(ratio_a, ratio_b, floor_control, floor_arm, gain),
                         slowest=[item[1] for item in windows], window_s=[round(item[2], 1) for item in windows]))
    verdicts = {row['test']: row['verdict'] for row in rows}
    if verdicts.get(TESTS[0]) == 'UNREAD':
        return rows, None, ['concurrent8_steady is unread: nothing to ship on']
    losses = [test for test, state in verdicts.items() if state == 'LOSS']
    unread = [test for test, state in verdicts.items() if state == 'UNREAD']
    if unread:
        notes.append('unread: %s (a deep test whose budget did not outlast the prefill queue, or a boot that failed)' % ', '.join(unread))
    go = not losses and verdicts.get(TESTS[0]) == 'GAIN'
    if losses:
        notes.append('LOSS in %s' % ', '.join(losses))
    return rows, go, notes


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    for name in ('control-a', 'arm-a', 'control-b', 'arm-b'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--arm-container', type=Path, action='append', default=[], help='an arm boot\'s container log (octo_judge must be clean on it); repeatable')
    parser.add_argument('--profile', help='the arm profile (its env drives octo_judge)')
    parser.add_argument('--profiles', type=Path)
    parser.add_argument('--gain', type=float, default=GAIN)
    options = parser.parse_args(argv)
    import c2_smoke_check as check
    try:
        logs = [check.smoke_results(path.read_text(encoding='utf-8', errors='replace')) for path in (options.control_a, options.arm_a, options.control_b, options.arm_b)]
        containers = [path.read_text(encoding='utf-8', errors='replace') for path in options.arm_container]
        env = check.profile_env(options.profile, options.profiles or check.DEFAULT_PROFILES) if options.profile else None
    except (OSError, ValueError) as error:
        print('OCTO3_REPORT unreadable: %s' % error, file=sys.stderr)
        return 2
    rows, go, notes = read(*logs, gain=options.gain)
    for row in rows:
        print('OCTO3_REPORT %s' % json.dumps(row, sort_keys=True))
    problems = []
    if containers and env is not None:
        import octo_judge

        for index, text in enumerate(containers):
            found, facts = octo_judge.judge(env, text)
            print('OCTO3_REPORT judge arm=%d %s' % (index + 1, json.dumps(facts, sort_keys=True)))
            problems += ['arm container %d: %s' % (index + 1, problem) for problem in found]
    for note in notes + problems:
        print('OCTO3_REPORT note: %s' % note)
    if go is None:
        print('OCTO3_REPORT SHIP-UNREAD')
        return 2
    go = go and not problems
    print('OCTO3_REPORT %s' % ('SHIP-GO' if go else 'SHIP-NO-GO'))
    return 0 if go else 1


if __name__ == '__main__':
    sys.exit(main())
