"""The exact-resume ABAB's reading (docs/tp4-exact-resume.md 5.2): two runs of the prefix gate's exact-resume-timed plan (one arm per
job), a control (the combined arm without the switches) and the lever (the same plus QWEN_PREFIX_STICKY_UNDROP and
QWEN_PREFIX_WINDOW_CARRY), paired turn by turn.

    python3 scripts/ci/exact_resume_compare.py <control results dir or records.jsonl> <lever ...> [--label-a A --label-b B]

For every salted continuation (role hit) of both runs, keyed by (conversation, turn): the prompt P, each arm's reused Q (the marker
row's, else the oracle's) and tail P - Q, each arm's TTFT and the difference; and the hit's output token ids compared across the arms.
Both arms are byte-exact against their cold twins (E0), so the same messages give the same answers: an answer that differs is a FAIL,
reported once per conversation (every later prompt of it differs). Exit 0 when every paired hit is identical and neither run lacks a
turn the other ran, 1 otherwise, 2 for a missing input. Stdlib only, Python 3.7 syntax: it runs on the rig host.
"""
import argparse
import json
import os
import sys


def load(path):
    if os.path.isdir(path):
        path = os.path.join(path, 'records.jsonl')
    if not os.path.isfile(path):
        return None
    with open(path, encoding='utf-8') as handle:
        return [json.loads(line) for line in handle if line.strip()]


def q_of(record):
    markers = record.get('markers') or {}
    if isinstance(markers.get('q'), int):
        return markers['q']
    return (record.get('expected') or {}).get('q') or 0


def hits(records):
    out = {}
    for record in records or ():
        if record.get('role') == 'hit' and record.get('continuation') and record.get('ok'):
            out[(record.get('conv'), record.get('turn'))] = record
    return out


def compare(records_a, records_b, label_a='A', label_b='B'):
    a, b = hits(records_a), hits(records_b)
    rows, problems, diverged = [], [], set()
    for key in sorted(set(a) | set(b), key=lambda item: (str(item[0]), item[1] if isinstance(item[1], int) else -1)):
        left, right = a.get(key), b.get(key)
        if left is None or right is None:
            problems.append('%s turn %s ran only in %s' % (key[0], key[1], label_a if right is None else label_b))
            continue
        if key[0] in diverged:
            continue
        same = left.get('token_ids') == right.get('token_ids')
        if not same:
            diverged.add(key[0])
            problems.append('%s turn %s: the answers differ between %s and %s (both are exact against their cold twins)' % (
                key[0], key[1], label_a, label_b))
        prompt = left.get('prompt_tokens') or 0
        qa, qb = q_of(left), q_of(right)
        ta, tb = left.get('ttft_s'), right.get('ttft_s')
        rows.append(dict(conv=key[0], turn=key[1], prompt=prompt, q_a=qa, q_b=qb, tail_a=prompt - qa, tail_b=prompt - qb,
                         ttft_a=ta, ttft_b=tb, delta=(tb - ta) if isinstance(ta, (int, float)) and isinstance(tb, (int, float)) else None,
                         identical=same))
    return rows, problems


def render(rows, problems, label_a, label_b, out=print):
    out('conversation turn P | Q %s Q %s | tail %s tail %s | TTFT %s TTFT %s delta | answer' % (label_a, label_b, label_a, label_b,
                                                                                              label_a, label_b))
    for row in rows:
        out('%s %s %d | %d %d | %d %d | %s %s %s | %s' % (
            row['conv'], row['turn'], row['prompt'], row['q_a'], row['q_b'], row['tail_a'], row['tail_b'],
            '-' if row['ttft_a'] is None else '%.2f' % row['ttft_a'], '-' if row['ttft_b'] is None else '%.2f' % row['ttft_b'],
            '-' if row['delta'] is None else '%+.2f' % row['delta'], 'IDENTICAL' if row['identical'] else 'DIFFERS'))
    by_conv = {}
    for row in rows:
        if row['delta'] is not None:
            by_conv.setdefault(row['conv'], []).append(row)
    for conv, items in sorted(by_conv.items()):
        out('%s: %d turns, mean TTFT %s %.2f s %s %.2f s (%+.2f s), mean tail %s %d %s %d' % (
            conv, len(items), label_a, sum(r['ttft_a'] for r in items) / len(items), label_b, sum(r['ttft_b'] for r in items) / len(items),
            sum(r['delta'] for r in items) / len(items), label_a, sum(r['tail_a'] for r in items) // len(items),
            label_b, sum(r['tail_b'] for r in items) // len(items)))
    for problem in problems:
        out('problem: %s' % problem)
    out('EXACT_RESUME_COMPARE verdict=%s turns=%d problems=%d' % ('PASS' if not problems and rows else 'FAIL', len(rows), len(problems)))


def main(argv=None, out=print):
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('a')
    parser.add_argument('b')
    parser.add_argument('--label-a', default='control')
    parser.add_argument('--label-b', default='lever')
    options = parser.parse_args(argv)
    records_a, records_b = load(options.a), load(options.b)
    if records_a is None or records_b is None:
        out('missing input: %s' % (options.a if records_a is None else options.b))
        return 2
    rows, problems = compare(records_a, records_b, options.label_a, options.label_b)
    render(rows, problems, options.label_a, options.label_b, out)
    return 0 if rows and not problems else 1


if __name__ == '__main__':
    sys.exit(main())
