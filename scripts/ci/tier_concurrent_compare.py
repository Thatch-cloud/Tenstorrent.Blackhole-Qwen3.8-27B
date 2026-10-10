"""Two tier-concurrent arms side by side (c2_prefix_gate --plan tier-concurrent / tier-concurrent-timed; prefix_concurrent): the control profile (no host tier) and the tier
profile, each run as a job of its own (the ABAB pairs), on the same seats and the same returning session.

    python3 scripts/ci/tier_concurrent_compare.py <A results dir or records.jsonl> <B ...> [<C> <D>] [--label-a control --label-b tier ...]

A is the reference (the control profile), B the arm under test. Giving C and D (the repeat pair of an ABAB) also compares D with C and prints A against C, the FLOOR: how much two
runs of the very same control profile differ in stall and round time (and, for a text, in nothing: a control that differs from its own repeat makes the stack, not the tier, the
finding). What is read, per pair:

  TEXT      every seat's output token ids over the tokens both arms streamed (a seat is aborted when the scenario ends, at different lengths in different arms, so the common
            prefix is what is comparable): IDENTICAL, DIVERGED at the first token that differs, or SHORT (fewer than MIN_COMMON_TOKENS in common: nothing is shown). The returning
            session's answer, the long prompt's and the cold twin's are compared whole: the restored answer of B must equal A's cold re-prefill and its own cold twin.
  STALL     each window's round time p50/p99 and longest wait of any seat, B against A (percent), each arm against its own control window, and the stall budget (the Lever N decode
            gap floor, 8 s) for both arms: a longest wait over it is a problem.

With --quota (two paths: the quota-control arm and the quota-isolated arm of the tier-quota plan, each a results directory) it compares what that plan sends instead: tenant B's
and tenant A's returning answers (cases quota-b-return, quota-a-return) over the whole output, which must be IDENTICAL across the two arms - the same prompts, greedy, so a quota
that changes an answer is a defect - and reads no seats and no stall.

Exit 0 only when every compared text is IDENTICAL, both arms hold the same seats, and no stall is over the budget; 1 otherwise; 2 for a missing input.
Stdlib only, Python 3.7 syntax: it runs on the rig host.
"""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import prefix_concurrent as concurrent  # noqa: E402

WHOLE_CASES = (concurrent.RETURN_CASE, concurrent.SPILL_CASE, concurrent.COLD_CASE)
QUOTA_CASES = ('quota-b-return', 'quota-a-return')


def whole_rows(records_a, records_b, cases=WHOLE_CASES):
    """[row] for the returning turn, the long prompt and the cold twin (or the cases given): their token ids (and finish reasons) equal whole."""
    rows = []
    for case in cases:
        a, b = concurrent.first_record(records_a, case), concurrent.first_record(records_b, case)
        if a is None and b is None:
            continue
        if a is None or b is None:
            rows.append(dict(case=case, verdict='MISSING', detail='only %s sent it' % ('B' if a is None else 'A')))
            continue
        ids_a, ids_b = a.get('token_ids'), b.get('token_ids')
        if ids_a is None or ids_b is None:
            rows.append(dict(case=case, verdict='ERROR', detail='no output token ids recorded in %s' % ('A' if ids_a is None else 'B')))
            continue
        common = min(len(ids_a), len(ids_b))
        at = next((index for index in range(common) if ids_a[index] != ids_b[index]), None)
        if at is None and len(ids_a) != len(ids_b):
            at = common
        if at is None and a.get('finish') == b.get('finish'):
            rows.append(dict(case=case, verdict='IDENTICAL', detail='%d tokens' % len(ids_a)))
        else:
            rows.append(dict(case=case, verdict='DIVERGED', detail='first differing output token %s (A %d tokens %s, B %d tokens %s)' % (
                at, len(ids_a), a.get('finish'), len(ids_b), b.get('finish'))))
    return rows


def stall_lines(stats_a, stats_b, label_a, label_b):
    """One line per window both arms measured: round p50/p99 and the longest wait, B against A (percent)."""
    lines = []
    for name in concurrent.WINDOWS:
        a, b = stats_a.get(name), stats_b.get(name)
        if a is None or b is None:
            continue
        change = None
        if a.get('p99_s') and b.get('p99_s') is not None:
            change = round((b['p99_s'] / a['p99_s'] - 1.0) * 100.0, 1)
        lines.append('window %s: round p50/p99 %s/%s s (%s) vs %s/%s s (%s); longest wait %s s vs %s s; seats %s vs %s; round p99 %s %s' % (
            name, a['p50_s'], a['p99_s'], label_a, b['p50_s'], b['p99_s'], label_b, a['max_gap_s'], b['max_gap_s'], a['seats'], b['seats'], label_b,
            'n/a' if change is None else '%+.1f%% against %s' % (change, label_a)))
    return lines


def compare_quota(records_a, records_b, label_a='control', label_b='isolated'):
    """-> dict(verdict, problems, lines, whole). The two arms of the tier-quota plan: tenant B's and tenant A's returning answers, whole."""
    problems, lines = [], []
    whole = whole_rows(records_a, records_b, QUOTA_CASES)
    if not whole:
        problems.append('neither arm sent a quota-b-return or a quota-a-return request')
    for row in whole:
        lines.append('%s %s vs %s: %s (%s)' % (row['case'], label_a, label_b, row['verdict'], row['detail']))
        if row['verdict'] != 'IDENTICAL':
            problems.append('%s %s: %s' % (row['case'], row['verdict'], row['detail']))
    missing = [case for case in QUOTA_CASES if case not in [row['case'] for row in whole]]
    if whole and missing:
        problems.append('no %s in either arm' % ', '.join(missing))
    return dict(verdict='FAIL' if problems else 'PASS', problems=problems, lines=lines, whole=whole)


def compare_pair(records_a, records_b, label_a='A', label_b='B', minimum=concurrent.MIN_COMMON_TOKENS, budget=concurrent.DEFAULT_GAP_BUDGET_S, control_s=None):
    """-> dict(verdict, problems, lines, seats, whole, stats). Two arms' text and stall, as the module docstring says."""
    problems, lines = [], []
    seats = concurrent.compare_seats(records_a, records_b, minimum)
    whole = whole_rows(records_a, records_b)
    texts_a, texts_b = concurrent.seat_texts(records_a), concurrent.seat_texts(records_b)
    for label, mine, other, other_label in ((label_a, texts_a, texts_b, label_b), (label_b, texts_b, texts_a, label_a)):
        lone = sorted(set(mine) - set(other))
        if lone:
            problems.append('seats only %s ran: %s (%s lacks them)' % (label, ', '.join(lone), other_label))
    if not seats:
        problems.append('no seat is present in both arms')
    counts = dict((verdict, sum(1 for row in seats if row['verdict'] == verdict)) for verdict in ('IDENTICAL', 'DIVERGED', 'SHORT'))
    lines.append('seat text %s vs %s: %d seats compared, %d identical, %d diverged, %d short (identical over %s tokens in common at the fewest)' % (
        label_a, label_b, len(seats), counts['IDENTICAL'], counts['DIVERGED'], counts['SHORT'], min([row['tokens'] for row in seats] or [0])))
    for row in seats:
        if row['verdict'] == 'DIVERGED':
            problems.append('%s diverged at output token %d of %d in common (%s %d tokens, %s %d tokens)' % (
                row['conv'], row['at'], row['tokens'], label_a, row['lengths'][0], label_b, row['lengths'][1]))
        elif row['verdict'] == 'SHORT':
            problems.append('%s: only %d tokens in common (under %d): its text proves nothing' % (row['conv'], row['tokens'], minimum))
    for row in whole:
        lines.append('%s %s vs %s: %s (%s)' % (row['case'], label_a, label_b, row['verdict'], row['detail']))
        if row['verdict'] != 'IDENTICAL':
            problems.append('%s %s: %s' % (row['case'], row['verdict'], row['detail']))
    stats_a, stats_b = concurrent.stats_of(records_a, control_s), concurrent.stats_of(records_b, control_s)
    lines += concurrent.render(label_a, stats_a)
    lines += concurrent.render(label_b, stats_b)
    lines += stall_lines(stats_a, stats_b, label_a, label_b)
    for label, stats in ((label_a, stats_a), (label_b, stats_b)):
        if not stats:
            problems.append('%s: no window could be read (no returning turn in its records)' % label)
        problems += ['%s %s' % (label, text) for text in concurrent.budget_problems(stats, budget)]
    return dict(verdict='FAIL' if problems else 'PASS', problems=problems, lines=lines, seats=seats, whole=whole, stats=(stats_a, stats_b))


def main(argv=None, out=print):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('paths', nargs='+', help='A B [C D]: results directories or records.jsonl files')
    parser.add_argument('--labels', default='control,tier,control-repeat,tier-repeat', help='comma-separated names for A B C D')
    parser.add_argument('--budget', type=float, default=concurrent.DEFAULT_GAP_BUDGET_S, help='the stall budget in seconds (the Lever N decode gap floor)')
    parser.add_argument('--min-tokens', type=int, default=concurrent.MIN_COMMON_TOKENS)
    parser.add_argument('--quota', action='store_true', help='compare the two arms of the tier-quota plan (A control, B isolated): the returning answers of tenants B and A')
    options = parser.parse_args(argv)
    if options.quota:
        if len(options.paths) != 2:
            out('refused: --quota takes exactly two paths (the quota-control and the quota-isolated results)')
            return 2
        labels = ['control', 'isolated']
        try:
            loaded = [concurrent.load_records(path) for path in options.paths]
        except (IOError, OSError, ValueError) as error:
            out('refused: %s' % error)
            return 2
        result = compare_quota(loaded[0], loaded[1], labels[0], labels[1])
        for line in result['lines']:
            out(line)
        for problem in result['problems']:
            out('problem: %s' % problem)
        out('tier quota %s vs %s: %s' % (labels[0], labels[1], result['verdict']))
        return 0 if result['verdict'] == 'PASS' else 1
    if len(options.paths) not in (2, 4):
        out('refused: give A B (one pair) or A B C D (the ABAB: two pairs and the control-to-control floor)')
        return 2
    labels = (options.labels.split(',') + ['A', 'B', 'C', 'D'])[:len(options.paths)]
    try:
        loaded = [concurrent.load_records(path) for path in options.paths]
    except (IOError, OSError, ValueError) as error:
        out('refused: %s' % error)
        return 2
    verdicts = []
    pairs = [(0, 1)] + ([(2, 3)] if len(loaded) == 4 else [])
    for first, second in pairs:
        result = compare_pair(loaded[first], loaded[second], labels[first], labels[second], options.min_tokens, options.budget)
        for line in result['lines']:
            out(line)
        for problem in result['problems']:
            out('problem: %s' % problem)
        out('tier concurrent %s vs %s: %s' % (labels[first], labels[second], result['verdict']))
        verdicts.append(result['verdict'])
    if len(loaded) == 4:
        floor = compare_pair(loaded[0], loaded[2], labels[0], labels[2], options.min_tokens, options.budget)
        out('FLOOR (the same control profile twice, %s vs %s): the differences above that are not the tier\'s are of this size' % (labels[0], labels[2]))
        for line in floor['lines']:
            out(line)
        for problem in floor['problems']:
            out('floor problem: %s' % problem)
        out('tier concurrent floor %s vs %s: %s' % (labels[0], labels[2], floor['verdict']))
        verdicts.append(floor['verdict'])
    return 0 if all(verdict == 'PASS' for verdict in verdicts) else 1


if __name__ == '__main__':
    sys.exit(main())
