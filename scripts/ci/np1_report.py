"""The tp4/np1-fastlane timed read (stdlib plus octo3_report and c2_smoke_check, Python 3.7 syntax): the lone-user padded round (QWEN_FAST_SOLO_PACKED=1) against the approved ship-4 profile, ABAB over four boots.

    python3 scripts/ci/np1_report.py --control-a C1.smoke --arm-a L1.smoke --control-b C2.smoke --arm-b L2.smoke [--arm-container L1.container] [--profile P] [--gain 0.10]

The control is the approved ship-4 profile itself (a lone user decodes on the 4-row per-request engine, two or three on the padded 64-row block, four on the full block, six and more on the octo
block); the arm is its gate-only twin with QWEN_FAST_SOLO_PACKED=1 (references/fusion-jobs/NP1), which differs for the lone user alone (and for a lone user of the other block). The tests are the shapes
c2_serving_smoke adds for it, one boot each: np1_live1, np1_live2 and np1_live4 (one, two and four users started together on about 3,500-token real-code prompts, every answer to exactly 2,000
tokens) and concurrent8_steady (eight users: the octo block, which the flag never touches, so a parity row that shows the boot-to-boot floor and catches a regression the flag could cause there).

The rounds of two boots cannot be paired, so what is compared is what the smoke measures for the whole all-live window of each test (c2_serving_smoke.live_report): the aggregate decode rate
(`live4_agg_tok_s`, whatever the user count), as octo3_report reads it:

  ratio A = arm A / control A, ratio B = arm B / control B; floors = control B / control A and arm B / arm A (the boot-to-boot drift of one profile)
  verdict = GAIN when both ratios exceed 1 + max(gain, both floors' distance from 1); LOSS when they fall below 1 - the floors; else PARITY; UNREAD when a window is missing or shorter than 10 s.

GO needs: a GAIN at np1_live1 (the lone user, where the flag acts), no LOSS in any other test, and - with --arm-container - octo_judge clean on each arm boot's container log under --profile (a padded
round of a lone live user RAN, and no eligible lone-user round went to the per-request engines). The text read is octo_compare.py (every answer equal, user by user), run on the same four logs.
This module prints NP1_REPORT lines and exits 0 (GO), 1 (NO-GO) or 2 (unread).
"""
import argparse
import json
import sys
from pathlib import Path

import octo3_report

TESTS = ('np1_live1', 'np1_live2', 'np1_live4', 'concurrent8_steady')
ACTS_ON = TESTS[0]
GAIN = 0.10


def read(control_a, arm_a, control_b, arm_b, gain=GAIN):
    """(rows, go, notes): rows are dicts per test (ratios, floors, verdict); go is True, False or None (unread)."""
    rows, notes = [], []
    for test in TESTS:
        windows = [octo3_report.window(log, test) for log in (control_a, arm_a, control_b, arm_b)]
        if any(item is None for item in windows):
            rows.append(dict(test=test, verdict='UNREAD', note='a boot has no all-live window of %d s or more in this test' % octo3_report.MIN_WINDOW_S))
            continue
        ca, aa, cb, ab = (item[0] for item in windows)
        ratio_a, ratio_b, floor_control, floor_arm = aa / ca, ab / cb, cb / ca, ab / aa
        rows.append(dict(test=test, control_a=ca, arm_a=aa, control_b=cb, arm_b=ab, ratio_a=round(ratio_a, 4), ratio_b=round(ratio_b, 4),
                         floor_control=round(floor_control, 4), floor_arm=round(floor_arm, 4),
                         verdict=octo3_report.verdict(ratio_a, ratio_b, floor_control, floor_arm, gain), slowest=[item[1] for item in windows],
                         window_s=[round(item[2], 1) for item in windows]))
    verdicts = {row['test']: row['verdict'] for row in rows}
    if verdicts.get(ACTS_ON) == 'UNREAD':
        return rows, None, ['%s is unread: the lone user\'s window is the whole claim' % ACTS_ON]
    losses = [test for test, state in verdicts.items() if state == 'LOSS']
    unread = [test for test, state in verdicts.items() if state == 'UNREAD']
    if unread:
        notes.append('unread: %s' % ', '.join(unread))
    if losses:
        notes.append('LOSS in %s' % ', '.join(losses))
    return rows, (not losses and verdicts.get(ACTS_ON) == 'GAIN'), notes


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
        print('NP1_REPORT unreadable: %s' % error, file=sys.stderr)
        return 2
    rows, go, notes = read(*logs, gain=options.gain)
    for row in rows:
        print('NP1_REPORT %s' % json.dumps(row, sort_keys=True))
    problems = []
    if containers and env is not None:
        import octo_judge

        for index, text in enumerate(containers):
            found, facts = octo_judge.judge(env, text)
            print('NP1_REPORT judge arm=%d %s' % (index + 1, json.dumps(facts, sort_keys=True)))
            problems += ['arm container %d: %s' % (index + 1, problem) for problem in found]
    for note in notes + problems:
        print('NP1_REPORT note: %s' % note)
    if go is None:
        print('NP1_REPORT UNREAD')
        return 2
    go = go and not problems
    print('NP1_REPORT %s' % ('GO' if go else 'NO-GO'))
    return 0 if go else 1


if __name__ == '__main__':
    sys.exit(main())
