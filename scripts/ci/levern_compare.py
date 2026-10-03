"""levern_compare: the interleaved arm against the non-interleaved control, answer for answer and digest for digest (Lever N at TP4, stdlib only).

Lever N's claim is exactness: a prefill that arrives in steps with decode rounds between them produces the same KV pages, the same GDN state, the same
logits and therefore the same tokens as the whole prompt did (docs/lever-n-tp4-design-2026-10-04.md section 3.4). Two arms run the same smoke tests on the
same image - the control (chunked prefill off, the whole-prompt path: c2-packed-tp4-8x262k-best-time-gate, or best-levern-control-audit for the digests) and
the interleaved one (c2-packed-tp4-8x262k-best-levern-time-gate, -r1-time-gate or -audit) - and this tool reads the two smoke logs (SMOKE_JSON) and, for the
audited pair, the two container logs, and requires:

  - every test both arms ran: every stream's content hash, reasoning hash, completion token count and finish reason equal, user by user (the seven decoders and
    the arrival of the stall tests included), and every exact-length row (levern_equal, levern_equal_long, levern_equal_busy) equal in content hash, token count,
    finish reason and the prompt token count the server reported;
  - the digest lines ("[PINDIAG] lever N digest": the GDN slot, the last-position logits and the KV pages of a finished prefill), per prompt length in log order,
    equal between the two container logs;
  - and that it compared SOMETHING: a comparison with no common test and no common digest is a failure (exit 2), not a pass.

It also prints, never gates, the stall tests' numbers side by side: the arrival's time to first token, the worst decoder gap, and the decoders' progress inside the
arrival's prefill window (zero on the control arm, which freezes them).

  python levern_compare.py --control smoke-control.log --levern smoke-levern.log [--control-container c.log --levern-container l.log]
Exit 0 identical; 1 a mismatch; 2 nothing to compare or an unreadable log."""

import argparse
import json
import sys
from pathlib import Path

import c2_smoke_check as check

USER_FIELDS = (('content_sha256', 'content hash'), ('reasoning_sha256', 'reasoning hash'), ('tokens', 'completion tokens'), ('finish', 'finish reason'))
ROW_FIELDS = (('content_sha256', 'content hash'), ('tokens', 'completion tokens'), ('finish', 'finish reason'), ('prompt_tokens', 'prompt tokens'))
STALL_TESTS = check.STALL_TESTS


def compare_users(name, a, b):
    """(mismatches, compared) of two arms' `users` lists for one test."""
    mismatches, compared = [], 0
    if len(a) != len(b):
        return ['%s: %d users in the control, %d in the interleaved arm' % (name, len(a), len(b))], 0
    for index, (left, right) in enumerate(zip(a, b)):
        label = '%s user %d' % (name, index)
        if not isinstance(left, dict) or not isinstance(right, dict) or 'error' in left or 'error' in right:
            mismatches.append('%s: an arm has no answer to compare (control %s, interleaved %s)' % (
                label, (left or {}).get('error', 'ok') if isinstance(left, dict) else left,
                (right or {}).get('error', 'ok') if isinstance(right, dict) else right))
            continue
        for key, what in USER_FIELDS:
            if left.get(key) != right.get(key):
                mismatches.append('%s: %s differs (control %r, interleaved %r)' % (label, what, left.get(key), right.get(key)))
        compared += 1
    return mismatches, compared


def compare_rows(name, a, b):
    """(mismatches, compared) of two arms' exact-length `prompts` rows for one test."""
    mismatches, compared = [], 0
    if sorted(a) != sorted(b):
        return ['%s: the arms ran different prompt lengths (control %s, interleaved %s)' % (name, sorted(a, key=int), sorted(b, key=int))], 0
    for length in sorted(a, key=int):
        left, right, label = a[length], b[length], '%s prompt %s' % (name, length)
        if 'error' in left or 'error' in right:
            mismatches.append('%s: an arm has no answer to compare (control %s, interleaved %s)' % (label, left.get('error', 'ok'), right.get('error', 'ok')))
            continue
        for key, what in ROW_FIELDS:
            if left.get(key) != right.get(key):
                mismatches.append('%s: %s differs (control %r, interleaved %r)' % (label, what, left.get(key), right.get(key)))
        compared += 1
    return mismatches, compared


def compare_digests(a_text, b_text):
    """(mismatches, compared): the digest lines of the two container logs, per prompt length in log order."""
    def grouped(text):
        found = {}
        for row in check.levern_facts(text)['digests']:
            found.setdefault(row['prompt'], []).append((row['slot'], row['logits'], row['kv']))
        return found

    left, right = grouped(a_text), grouped(b_text)
    mismatches, compared = [], 0
    for prompt in sorted(set(left) | set(right)):
        one, two = left.get(prompt, []), right.get(prompt, [])
        if len(one) != len(two):
            mismatches.append('digests of the %d-token prompt: %d in the control, %d in the interleaved arm' % (prompt, len(one), len(two)))
            continue
        for index, (x, y) in enumerate(zip(one, two)):
            for part, what in zip(range(3), ('GDN slot', 'logits', 'KV pages')):
                if x[part] != y[part]:
                    mismatches.append('digest of the %d-token prompt (run %d): the %s differs (control %s, interleaved %s)'
                                      % (prompt, index + 1, what, x[part], y[part]))
            compared += 1
    return mismatches, compared


def stall_report(name, a, b):
    """Side-by-side numbers of one stall test, as text lines (never a verdict)."""
    lines = []
    for label, entry in (('control', a), ('interleaved', b)):
        window = entry.get('window') or {}
        lines.append('LEVERN_COMPARE %s %s: arrival_ttft_s=%s longest_gap_s=%s seats_progressing=%s/%s min_chunk_rate=%s total_est_tok_s=%s' % (
            name, label, entry.get('arrival_ttft_s'), entry.get('longest_gap_s'), window.get('seats_progressing'), window.get('seats'),
            window.get('min_chunk_rate'), window.get('total_est_tok_s')))
    ttft = [entry.get('arrival_ttft_s') for entry in (a, b)]
    if all(isinstance(value, (int, float)) and value for value in ttft):
        lines.append('LEVERN_COMPARE %s: arrival TTFT interleaved / control = %.2f' % (name, ttft[1] / ttft[0]))
    return lines


def compare(control_smoke, levern_smoke, control_container=None, levern_container=None):
    """(mismatches, compared, report lines) of two arms' smoke logs and, optionally, container logs."""
    left, right = check.smoke_results(control_smoke), check.smoke_results(levern_smoke)
    if left is None or right is None:
        return ['no SMOKE_JSON line in the %s smoke log' % ('control' if left is None else 'interleaved')], 0, []
    mismatches, compared, report = [], 0, []
    for name in sorted(set(left) & set(right)):
        one, two = left[name], right[name]
        if not isinstance(one, dict) or not isinstance(two, dict):
            continue
        if 'error' in one or 'error' in two:
            mismatches.append('%s: an arm errored (control %s, interleaved %s)' % (name, one.get('error', 'ok'), two.get('error', 'ok')))
            continue
        if isinstance(one.get('prompts'), dict) and isinstance(two.get('prompts'), dict):
            found, count = compare_rows(name, one['prompts'], two['prompts'])
            mismatches += found
            compared += count
        if isinstance(one.get('users'), list) and isinstance(two.get('users'), list):
            found, count = compare_users(name, one['users'], two['users'])
            mismatches += found
            compared += count
        if name in STALL_TESTS:
            report += stall_report(name, one, two)
    if control_container is not None and levern_container is not None:
        found, count = compare_digests(control_container, levern_container)
        mismatches += found
        compared += count
        report.append('LEVERN_COMPARE digests compared: %d' % count)
    return mismatches, compared, report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--control', type=Path, required=True, help='the control arm\'s smoke log')
    parser.add_argument('--levern', type=Path, required=True, help='the interleaved arm\'s smoke log')
    parser.add_argument('--control-container', type=Path)
    parser.add_argument('--levern-container', type=Path)
    options = parser.parse_args(argv)
    if bool(options.control_container) != bool(options.levern_container):
        print('LEVERN_COMPARE unreadable: the two container logs go together', file=sys.stderr)
        return 2
    try:
        read = lambda path: path.read_text(encoding='utf-8', errors='replace') if path else None  # noqa: E731
        mismatches, compared, report = compare(read(options.control), read(options.levern), read(options.control_container),
                                               read(options.levern_container))
    except OSError as error:
        print('LEVERN_COMPARE unreadable: %s' % error, file=sys.stderr)
        return 2
    for line in report:
        print(line)
    for mismatch in mismatches:
        print('LEVERN_COMPARE MISMATCH: %s' % mismatch)
    print('LEVERN_COMPARE %s' % json.dumps(dict(identical=not mismatches and compared > 0, compared=compared, mismatches=len(mismatches))))
    if mismatches:
        return 1
    if not compared:
        print('LEVERN_COMPARE nothing was compared: no test in common and no digest in common', file=sys.stderr)
        return 2
    return 0


if __name__ == '__main__':
    sys.exit(main())
