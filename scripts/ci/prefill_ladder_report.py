"""Render the prefill ladder results of a smoke run as markdown (stdlib only).

    python3 prefill_ladder_report.py <results> [--second <results>]

<results> is the smoke's SMOKE_JSON object, a file holding it, or a smoke log that carries a "SMOKE_JSON {...}" line. The tests it reads (c2_serving_smoke.py):

  prefill_ladder_solo    four prompts (about 4k, 32k, 128k and 254k tokens), one request at a time with nothing else running: TTFT and prefill tok/s
  prefill_ladder_busy    the same four sizes, each arriving beside seven decoders (today's policy): TTFT, prefill tok/s, busy over solo, the decoders' worst gap
  prefill_few_decoders   the Lever N governor's shape: one decoder and a 120k-token arrival, two decoders and a 254k-token arrival

With --second the first file is arm A (the production profile) and the second is arm B (the adaptive twin): a "B over A" table gives the TTFT ratio per rung and shape,
and the decoders' worst gap of both arms beside it. A ratio below 1 is B faster to first token. Nothing here gates: the numbers are recorded, and the ratio of one pair is
one pair (the ABAB order and the noise floor are the pack's ORDER.txt rules).

Exit 0 once something was rendered; 2 on an unreadable file or one with none of the three tests.
"""

import argparse
import json
import sys

SOLO, BUSY, FEW = 'prefill_ladder_solo', 'prefill_ladder_busy', 'prefill_few_decoders'
TESTS = (SOLO, BUSY, FEW)


def load(path):
    """The smoke results dict of `path`: the whole file as JSON, else its last SMOKE_JSON line."""
    with open(path, encoding='utf-8') as handle:
        text = handle.read()
    marker = 'SMOKE_JSON '
    lines = [line for line in text.splitlines() if line.startswith(marker)]
    if lines:
        return json.loads(lines[-1][len(marker):])
    value = json.loads(text)
    if isinstance(value, dict) and isinstance(value.get('smoke'), dict):
        value = value['smoke']
    if not isinstance(value, dict):
        raise ValueError('%s holds no smoke results object' % path)
    return value


def number(value, digits=1):
    return '-' if not isinstance(value, (int, float)) or isinstance(value, bool) else ('%.*f' % (digits, value))


def ratio(top, bottom):
    if isinstance(top, (int, float)) and isinstance(bottom, (int, float)) and bottom > 0:
        return round(top / bottom, 3)
    return None


def rungs_of(results, name):
    """{rung tokens: row} of a ladder test, ascending; {} when the test did not run or failed whole."""
    entry = results.get(name)
    if not isinstance(entry, dict) or 'error' in entry or not isinstance(entry.get('rungs'), dict):
        return {}
    found = {}
    for key, row in entry['rungs'].items():
        if str(key).isdigit() and isinstance(row, dict):
            found[int(key)] = row
    return dict(sorted(found.items()))


def shapes_of(results):
    entry = results.get(FEW)
    if not isinstance(entry, dict) or 'error' in entry:
        return []
    return [row for row in entry.get('shapes') or [] if isinstance(row, dict)]


def rows(results):
    """[(key, label, row)] in report order: the solo rungs, the busy rungs, the few-decoder shapes. key is stable across arms, label is for the table."""
    found = []
    for tokens, row in rungs_of(results, SOLO).items():
        found.append((('solo', tokens), 'solo %s' % format(tokens, ','), row))
    for tokens, row in rungs_of(results, BUSY).items():
        found.append((('busy', tokens), 'busy (7 decoders) %s' % format(tokens, ','), row))
    for row in shapes_of(results):
        decoders, target = row.get('shape_decoders'), row.get('target_tokens')
        found.append((('few', decoders, target), '%s decoder(s), %s' % (decoders, format(target, ',') if isinstance(target, int) else '?'), row))
    return found


def worst_gap(row):
    return row.get('longest_gap_s') if isinstance(row, dict) else None


def render_one(results, title):
    lines = ['### %s' % title, '']
    solo, busy = rungs_of(results, SOLO), rungs_of(results, BUSY)
    if solo or busy:
        lines += ['| rung (tokens) | prompt tokens | solo TTFT s | solo prefill tok/s | busy TTFT s | busy prefill tok/s | busy / solo | decoders live at first token | worst decoder gap s |',
                  '|---|---|---|---|---|---|---|---|---|']
        for tokens in sorted(set(solo) | set(busy)):
            one, two = solo.get(tokens, {}), busy.get(tokens, {})
            asked = two.get('busy_over_solo')
            if asked is None:
                asked = ratio(two.get('ttft_s'), one.get('ttft_s'))
            problem = one.get('error') or two.get('error')
            lines.append('| %s | %s | %s | %s | %s | %s | %s | %s | %s |%s' % (
                format(tokens, ','), number(one.get('prompt_tokens', two.get('prompt_tokens')), 0), number(one.get('ttft_s'), 2), number(one.get('prefill_tok_s')),
                number(two.get('ttft_s'), 2), number(two.get('prefill_tok_s')), number(asked, 3), number(two.get('decoders_live_at_first_token'), 0),
                number(worst_gap(two), 2), ' ERROR: %s' % problem if problem else ''))
        lines.append('')
    shapes = shapes_of(results)
    if shapes:
        lines += ['| shape | prompt tokens | TTFT s | prefill tok/s | decoders live at first token | worst decoder gap s | decoder tokens in the prefill window (est. tok/s) |',
                  '|---|---|---|---|---|---|---|']
        for row in shapes:
            decoders, target = row.get('shape_decoders'), row.get('target_tokens')
            window = row.get('window') or {}
            lines.append('| %s decoder(s), arrival of about %s | %s | %s | %s | %s | %s | %s |%s' % (
                decoders, format(target, ',') if isinstance(target, int) else '?', number(row.get('prompt_tokens'), 0), number(row.get('ttft_s'), 2),
                number(row.get('prefill_tok_s')), number(row.get('decoders_live_at_first_token'), 0), number(worst_gap(row), 2),
                number(window.get('total_est_tok_s')), ' ERROR: %s' % row['error'] if row.get('error') else ''))
        lines.append('')
    if len(lines) == 2:
        lines += ['No prefill ladder test is in these results.', '']
    return lines


def same_text(one, two):
    """'same' when both arms answered the case and the answers hash equal, 'DIFFERS' when they hash differently (greedy text must be byte-identical across the arms),
    '-' when either side has no answer hash (a failed case, or a smoke that predates the hash)."""
    left, right = one.get('content_sha256'), two.get('content_sha256')
    if not left or not right:
        return '-'
    return 'same' if left == right else 'DIFFERS'


def render_pair(first, second):
    """The B-over-A table: per rung and shape, A's TTFT, B's TTFT and B / A, both arms' worst decoder gap and whether the arrival's answer text is byte-identical."""
    a_rows, b_rows = dict((key, row) for key, _label, row in rows(first)), dict((key, row) for key, _label, row in rows(second))
    lines = ['### B over A (A = first file, B = second file; a ratio below 1 is B faster to first token)', '',
             '| case | A TTFT s | B TTFT s | B / A | A worst decoder gap s | B worst decoder gap s | arrival text |', '|---|---|---|---|---|---|---|']
    seen = set()
    for key, label, _row in rows(first) + rows(second):
        if key in seen:
            continue
        seen.add(key)
        one, two = a_rows.get(key, {}), b_rows.get(key, {})
        lines.append('| %s | %s | %s | %s | %s | %s | %s |' % (label, number(one.get('ttft_s'), 2), number(two.get('ttft_s'), 2),
                                                             number(ratio(two.get('ttft_s'), one.get('ttft_s')), 3), number(worst_gap(one), 2), number(worst_gap(two), 2),
                                                             same_text(one, two)))
    lines.append('')
    return lines


def render(first, second=None):
    lines = ['## Prefill ladder', ''] + render_one(first, 'A' if second is not None else 'Results')
    if second is not None:
        lines += render_one(second, 'B')
        lines += render_pair(first, second)
    return '\n'.join(lines)


def has_tests(results):
    return any(name in results for name in TESTS)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('results')
    parser.add_argument('--second', default=None, help='the other arm\'s results: adds the B-over-A table')
    options = parser.parse_args(argv)
    try:
        first = load(options.results)
        second = load(options.second) if options.second else None
    except (OSError, ValueError) as error:
        sys.stderr.write('prefill_ladder_report: %s\n' % error)
        return 2
    if not has_tests(first) or (second is not None and not has_tests(second)):
        sys.stderr.write('prefill_ladder_report: no prefill ladder test (%s) in the results\n' % ', '.join(TESTS))
        return 2
    print(render(first, second))
    return 0


if __name__ == '__main__':
    sys.exit(main())
