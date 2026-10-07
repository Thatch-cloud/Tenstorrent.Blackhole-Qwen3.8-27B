"""parked_compare: a parked-engines arm against its flag-off control, answer for answer (engine reuse at TP4, stdlib only).

Engine reuse's claim is exactness: an engine rebound to a request produces the same tokens as a cold build. Two arms run the same smoke tests on the same
image - the control (the profile's parent, or its -audit-r2 twin for the audited pair) and the parked arm - and this tool reads the two smoke logs
(SMOKE_JSON) and, optionally, the two container logs, and requires (mode exact, the default):

  - every test both arms ran: every stream's content hash, reasoning hash, completion token count and finish reason equal, user by user, and every exact-length
    or budget row equal in content hash, token count, finish reason and prompt token count (levern_compare's own comparison);
  - with the container logs: the Lever N digest lines (GDN slot, logits, KV pages of a finished prefill) equal per prompt, the parked arm's log clean under
    parked_judge.judge (its widths, programs, fallbacks, digests), and the drafter-equivalence judge (the solo requests' accepted-prefix sequences identical, the overlapping requests' mean acceptance within tolerance);
  - and that it compared SOMETHING: no common test and no common digest is a failure (exit 2), not a pass.

The other modes judge a negative-control arm against the control (parked_judge.negative_verdict): carry and pages must CHANGE tokens, drafter must keep
every token and fail the equivalence judge, widths must show a rebind whose widths are not a cold engine's. `--turns` prints the ER5 pairing of the
parked_turns test (per turn wall time and longest gap, paired by index; never a verdict).

  python parked_compare.py --control smoke-control.log --parked smoke-parked.log [--control-container c.log --parked-container p.log] [--mode exact]
Exit 0 identical (or the control did what it exists to do); 1 a mismatch; 2 nothing to compare or an unreadable log."""

import argparse
import json
import sys
from pathlib import Path

import c2_smoke_check as check
import levern_compare
import parked_judge

MODES = ('exact', 'carry', 'pages', 'drafter', 'widths')


def compare(control_smoke, parked_smoke, control_container=None, parked_container=None, mode='exact', env=None):
    """(problems, compared, report lines) of two arms' smoke logs and, optionally, container logs."""
    mismatches, compared, report = levern_compare.compare(control_smoke, parked_smoke, control_container, parked_container)
    problems, lines = [], list(report)
    if mode == 'exact':
        problems += mismatches
        if parked_container is not None:
            found, facts = parked_judge.judge(env or dict(QWEN_FAST_PARKED_ENGINES='1'), parked_container)
            problems += ['parked arm: ' + text for text in found]
            # R2 (the replay ledger): the parked arm's verifier raises on a replay between a verify and its publication, so its log carries none; the
            # flag-off control logs one instead of raising, so today's ordering is a number in the report and cannot stop the window by itself.
            control_r2, parked_r2 = parked_judge.r2_violations(control_container), parked_judge.r2_violations(parked_container)
            lines.append('PARKED_COMPARE R2 violations control=%d parked=%d' % (control_r2, parked_r2))
            if parked_r2:
                problems.append('parked arm: %d R2 violation line(s): a trace replayed between a verify and its publication' % parked_r2)
            lines.append('PARKED_COMPARE judge %s' % json.dumps(facts))
            # The exact judge over the requests that ran alone; the requests that overlapped (levern_equal_busy, churn) by acceptance, because their
            # draft paths follow admission timing.
            solo, shortfalls, detail = parked_judge.equivalence(control_container, parked_container, only='solo')
            problems += ['drafter equivalence: ' + text for text in solo]
            lines += ['PARKED_COMPARE drafter equivalence shortfall: %s' % text for text in shortfalls]
            lines.append('PARKED_COMPARE drafter equivalence %s' % json.dumps(detail))
            if parked_judge.accepted_sequences(control_container, 'concurrent') and parked_judge.accepted_sequences(parked_container, 'concurrent'):
                busy, busy_shortfalls, busy_detail = parked_judge.equivalence(control_container, parked_container, solo=False, only='concurrent')
                problems += ['concurrent drafting: ' + text for text in busy]
                lines += ['PARKED_COMPARE concurrent drafting shortfall: %s' % text for text in busy_shortfalls]
                lines.append('PARKED_COMPARE concurrent drafting %s' % json.dumps(busy_detail))
    else:
        found, facts = parked_judge.negative_verdict(mode, mismatches, control_container or '', parked_container or '')
        problems += ['negative control %s: %s' % (mode, text) for text in found]
        lines.append('PARKED_COMPARE negative %s' % json.dumps(facts))
    control_results, parked_results = check.smoke_results(control_smoke), check.smoke_results(parked_smoke)
    if control_results and parked_results and 'parked_turns' in control_results and 'parked_turns' in parked_results:
        rows, summary = parked_judge.paired_turns((control_results['parked_turns'] or {}).get('users') or [],
                                                  (parked_results['parked_turns'] or {}).get('users') or [])
        lines.append('PARKED_COMPARE turns %s' % json.dumps(summary))
        for row in rows:
            lines.append('PARKED_COMPARE turn %s' % json.dumps(row))
        for name, results in (('control', control_results), ('parked', parked_results)):
            entry = results['parked_turns'] or {}
            lines.append('PARKED_COMPARE turns %s makespan_s=%s tokens_per_s=%s completion_tokens=%s' % (
                name, entry.get('makespan_s'), entry.get('tokens_per_s'), entry.get('completion_tokens')))
    return problems, compared, lines


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--control', type=Path, required=True, help='the control arm\'s smoke log')
    parser.add_argument('--parked', type=Path, required=True, help='the parked arm\'s smoke log')
    parser.add_argument('--control-container', type=Path)
    parser.add_argument('--parked-container', type=Path)
    parser.add_argument('--mode', choices=MODES, default='exact')
    parser.add_argument('--profile', help='the parked arm\'s profile name (its env drives the judge)')
    parser.add_argument('--profiles', type=Path, default=check.DEFAULT_PROFILES)
    options = parser.parse_args(argv)
    if bool(options.control_container) != bool(options.parked_container):
        print('PARKED_COMPARE unreadable: the two container logs go together', file=sys.stderr)
        return 2
    try:
        read = lambda path: path.read_text(encoding='utf-8', errors='replace') if path else None  # noqa: E731
        env = check.profile_env(options.profile, options.profiles) if options.profile else None
        problems, compared, lines = compare(read(options.control), read(options.parked), read(options.control_container), read(options.parked_container),
                                            options.mode, env)
    except (OSError, ValueError) as error:
        print('PARKED_COMPARE unreadable: %s' % error, file=sys.stderr)
        return 2
    for line in lines:
        print(line)
    for problem in problems:
        print('PARKED_COMPARE MISMATCH: %s' % problem)
    print('PARKED_COMPARE %s' % json.dumps(dict(mode=options.mode, ok=not problems and (compared > 0 or options.mode != 'exact'), compared=compared,
                                                 problems=len(problems))))
    if problems:
        return 1
    if not compared and options.mode == 'exact':
        print('PARKED_COMPARE nothing was compared: no test in common and no digest in common', file=sys.stderr)
        return 2
    return 0


if __name__ == '__main__':
    sys.exit(main())
