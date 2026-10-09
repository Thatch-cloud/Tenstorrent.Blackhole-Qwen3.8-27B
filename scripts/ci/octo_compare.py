"""octo_compare: an octo-T8 arm against its flag-off control, answer for answer (stdlib only, Python 3.7 syntax).

The octo shape's claim is exactness: a round of eight seats x eight rows commits the target's own greedy tokens, only fewer or more of them per round, so every
text equals the text of the same user on the control (the arm's parent profile, the same image). Two arms run the same smoke tests and this tool reads the two smoke
logs (SMOKE_JSON) and, optionally, the two container logs, and requires:

  - every test both arms ran: every stream's content hash, reasoning hash, completion token count and finish reason equal, user by user, and every exact-length row
    equal (levern_compare's own comparison, the same one the engine-reuse gate uses);
  - with the arm's container log: octo_judge.judge clean (the shape executed, alternated, compiled nothing on a switch, no lone-user round left to the engines);
  - and that it compared SOMETHING: no common test is a failure (exit 2), not a pass.

It also prints, never gates, the per-user decode rate of every test both arms ran (the median of the users' decode_tok_s, the arm over the control) - the aggregate
read across two boots, beside the paired read of one alternate boot (octo_judge --verdict).

  python octo_compare.py --control smoke-control.log --octo smoke-octo.log [--octo-container o.log] [--profile P]
Exit 0 identical; 1 a mismatch; 2 nothing to compare or an unreadable log."""

import argparse
import json
import sys
from pathlib import Path

import c2_smoke_check as check
import levern_compare
import octo_judge


def median(values):
    ordered = sorted(values)
    if not ordered:
        return None
    middle = len(ordered) // 2
    return ordered[middle] if len(ordered) % 2 else (ordered[middle - 1] + ordered[middle]) / 2.0


def rates(control_smoke, octo_smoke):
    """[(test, control median decode tok/s, arm median, ratio)] for every streamed test both arms ran with readable rates. Informational."""
    control, arm = check.smoke_results(control_smoke) or {}, check.smoke_results(octo_smoke) or {}
    found = []
    for name in sorted(set(control) & set(arm)):
        left, right = control[name], arm[name]
        if not isinstance(left, dict) or not isinstance(right, dict) or 'error' in left or 'error' in right:
            continue
        first = [user['decode_tok_s'] for user in left.get('users') or [] if isinstance(user, dict) and isinstance(user.get('decode_tok_s'), (int, float))]
        second = [user['decode_tok_s'] for user in right.get('users') or [] if isinstance(user, dict) and isinstance(user.get('decode_tok_s'), (int, float))]
        if first and second and median(first):
            found.append((name, round(median(first), 2), round(median(second), 2), round(median(second) / median(first), 4)))
    return found


def compare(control_smoke, octo_smoke, octo_container=None, env=None):
    """(problems, compared, report lines) of two arms' smoke logs and, optionally, the arm's container log."""
    mismatches, compared, report = levern_compare.compare(control_smoke, octo_smoke, None, None)
    problems, lines = list(mismatches), list(report)
    if octo_container is not None:
        found, facts = octo_judge.judge(env or dict(QWEN_FAST_OCTO='alternate'), octo_container)
        problems += ['octo arm: ' + text for text in found]
        lines.append('OCTO_COMPARE judge %s' % json.dumps(facts, sort_keys=True))
    for name, control_rate, arm_rate, ratio in rates(control_smoke, octo_smoke):
        lines.append('OCTO_COMPARE rate %s control=%s octo=%s ratio=%s' % (name, control_rate, arm_rate, ratio))
    return problems, compared, lines


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--control', type=Path, required=True, help='the control arm\'s smoke log')
    parser.add_argument('--octo', type=Path, required=True, help='the octo arm\'s smoke log')
    parser.add_argument('--octo-container', type=Path)
    parser.add_argument('--profile', help='the octo arm\'s profile name (its env drives the judge)')
    parser.add_argument('--profiles', type=Path, default=check.DEFAULT_PROFILES)
    options = parser.parse_args(argv)
    try:
        read = lambda path: path.read_text(encoding='utf-8', errors='replace') if path else None  # noqa: E731
        env = check.profile_env(options.profile, options.profiles) if options.profile else None
        problems, compared, lines = compare(read(options.control), read(options.octo), read(options.octo_container), env)
    except (OSError, ValueError) as error:
        print('OCTO_COMPARE unreadable: %s' % error, file=sys.stderr)
        return 2
    for line in lines:
        print(line)
    for problem in problems:
        print('OCTO_COMPARE MISMATCH: %s' % problem)
    print('OCTO_COMPARE %s' % json.dumps(dict(ok=not problems and compared > 0, compared=compared, problems=len(problems))))
    if problems:
        return 1
    if not compared:
        print('OCTO_COMPARE nothing was compared: no test in common', file=sys.stderr)
        return 2
    return 0


if __name__ == '__main__':
    sys.exit(main())
