"""speed_window_compare: two matrix-gate result trees of the four-card speed window, compared offline, stdlib only.

The speed window's exactness jobs (E1 slide on, E2 slide off, E4 the timed profile) each write, under their results tree,
<arm>/m3native-gate.json for the arms matrix-solo and matrix-concurrent. This compares two such trees, user by user.

Why the texts alone are not enough for the drafter's K/V slide: the drafter's K/V only changes what is PROPOSED and
accepted. The greedy verifier emits the target's own argmax whatever the draft holds, so a slide that wrote wrong bytes
inside the banks would still give the same text and show up only as a different (lower) acceptance. So the comparison is
two-fold:
  texts     every user's text, solo and concurrent, must be equal (the tokens are the target's own);
  prefixes  the per-user accepted-prefix sequence of the SOLO arm (the [SEQ-PUBLISH] prefix= of each step, carried in the
            report as s2.paths) must be identical: one user decodes deterministically, so the same drafts give the same
            accepted lengths, and a different sequence is a different draft, that is, a wrong slide. The concurrent
            arm's sequence is reported, and judged only with --strict-concurrent-prefixes (its round mix can differ
            between runs).

Exit 0 identical, 1 a difference, 2 an unreadable or incomparable tree (a missing arm, or no path records: use
--texts-only to accept that, knowing the slide is then unchecked).

  python speed_window_compare.py FIRST_RESULTS SECOND_RESULTS [--texts-only] [--strict-concurrent-prefixes] [--json OUT]
"""

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from real_text_compare import decode_paths, first_divergence  # noqa: E402

ARMS = ('matrix-solo', 'matrix-concurrent')
REPORT = 'm3native-gate.json'


def find_report(root, arm):
    """The arm's m3native-gate.json under root (root/<arm>/ or any depth below it), or None."""
    root = Path(root)
    direct = root / arm / REPORT
    if direct.is_file():
        return direct
    found = sorted(root.rglob('%s/%s' % (arm, REPORT)))
    return found[0] if found else None


def load(root, arm):
    path = find_report(root, arm)
    if path is None:
        raise ValueError('%s: no %s/%s' % (root, arm, REPORT))
    return json.loads(path.read_text(encoding='utf-8'))


def user_texts(report):
    return [(stream or {}).get('text') for stream in report.get('streams') or []]


def user_prefixes(report, user):
    """(path, prefix) per round for one user from the report's path records, or None when it carries none."""
    entry = ((((report or {}).get('s2') or {}).get('paths') or {}).get('users') or {}).get(str(user))
    if not entry or entry.get('encoded') is None:
        return None
    return [(record['path'], record['prefix']) for record in decode_paths(entry['encoded'])]


def first_difference(first, second):
    for index, (a, b) in enumerate(zip(first, second)):
        if a != b:
            return index
    return None if len(first) == len(second) else min(len(first), len(second))


def compare_arm(first, second, texts_only=False):
    """One arm's per-user rows: text equality with the first divergence, and the prefix sequences' equality."""
    a_texts, b_texts = user_texts(first), user_texts(second)
    rows = []
    for user in range(max(len(a_texts), len(b_texts))):
        a = a_texts[user] if user < len(a_texts) else None
        b = b_texts[user] if user < len(b_texts) else None
        row = dict(user=user, text_equal=a is not None and a == b, text_first_divergence=None,
                   prefix_equal=None, prefix_first_divergence=None, rounds=None)
        if a is not None and b is not None and a != b:
            row['text_first_divergence'] = first_divergence(a, b)
        if not texts_only:
            pa, pb = user_prefixes(first, user), user_prefixes(second, user)
            if pa is not None and pb is not None:
                row['rounds'] = (len(pa), len(pb))
                row['prefix_equal'] = pa == pb
                row['prefix_first_divergence'] = None if pa == pb else first_difference(pa, pb)
        rows.append(row)
    return rows


def compare_trees(first_root, second_root, texts_only=False, strict_concurrent=False):
    """(result, code): result {arms: {arm: rows}, problems: [...]}; code 0, 1 or 2 (module docstring)."""
    arms, problems, incomparable = {}, [], []
    for arm in ARMS:
        try:
            first, second = load(first_root, arm), load(second_root, arm)
        except (OSError, ValueError) as error:
            incomparable.append('%s: %s' % (arm, error))
            continue
        rows = compare_arm(first, second, texts_only)
        arms[arm] = rows
        if not rows:
            incomparable.append('%s: no streams' % arm)
        for row in rows:
            if not row['text_equal']:
                problems.append('%s user %d: text differs (first at character %s)'
                                % (arm, row['user'], row['text_first_divergence']))
            if not texts_only and row['prefix_equal'] is None:
                incomparable.append('%s user %d: no accepted-prefix records in one or both trees' % (arm, row['user']))
            elif row['prefix_equal'] is False and (arm == 'matrix-solo' or strict_concurrent):
                problems.append('%s user %d: accepted-prefix sequence differs (first at round %s): a different draft'
                                % (arm, row['user'], row['prefix_first_divergence']))
    result = dict(first=str(first_root), second=str(second_root), arms=arms, problems=problems,
                  incomparable=incomparable, texts_only=texts_only)
    code = 1 if problems else (2 if incomparable else 0)
    return result, code


def render(result, code):
    lines = ['speed_window_compare: %s  vs  %s' % (result['first'], result['second'])]
    for arm, rows in result['arms'].items():
        for row in rows:
            if row['prefix_equal'] is None:
                prefix = 'n/a'
            elif row['prefix_equal']:
                prefix = 'IDENTICAL'
            else:
                prefix = 'DIFFERS at round %s' % row['prefix_first_divergence']
            lines.append('  %-17s user %d  text %-9s  prefixes %s' % (
                arm, row['user'], 'IDENTICAL' if row['text_equal'] else 'DIFFERS', prefix))
    lines += ['  PROBLEM: ' + text for text in result['problems']]
    lines += ['  INCOMPARABLE: ' + text for text in result['incomparable']]
    lines.append('verdict: %s' % {0: 'IDENTICAL', 1: 'DIFFERENT', 2: 'INCOMPARABLE'}[code])
    return '\n'.join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('first', type=Path)
    parser.add_argument('second', type=Path)
    parser.add_argument('--texts-only', action='store_true', help='skip the accepted-prefix comparison')
    parser.add_argument('--strict-concurrent-prefixes', action='store_true',
                        help='also fail on a differing concurrent-arm prefix sequence (only where the round mix matches)')
    parser.add_argument('--json', type=Path)
    options = parser.parse_args(argv)
    result, code = compare_trees(options.first, options.second, options.texts_only, options.strict_concurrent_prefixes)
    print(render(result, code))
    if options.json:
        options.json.write_text(json.dumps(result, indent=2), encoding='utf-8')
    return code


if __name__ == '__main__':
    sys.exit(main())
