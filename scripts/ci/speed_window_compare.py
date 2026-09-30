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

The same argument judges a BATCHED draft (a packed pair, the four-user quad): a user drafted inside a batch must be proposed
exactly what its own single-user capture proposes, and only the accepted prefixes can show it (the text is the target's).
  --position-keyed         the concurrent arm's prefix sequences are compared by the POSITION each round drafted at, not by
                           round index, over the positions both trees drafted at (a scheduling difference - an extra
                           sequential step, another round boundary - moves round indexes and never moves a position's
                           draft); a user needs --min-coverage of the shorter arm's positions in common to count
  --batched-vs-singles     within each tree, every user's concurrent-arm rounds (drafted in batches where the users allow)
                           against its own solo-arm rounds (one user never batches), by position: the same text and the same
                           position must give the same accepted prefix. One tree is enough (SECOND may be left out); given
                           two, each tree is judged and the trees are then compared as above.
Only full-width rounds count (a tail round of fewer drafts has another accepted prefix by construction).
  --users 2,3              judge only these users (0-based stream indexes): two trees served the SAME prompt for a user only
                           where the real-text corpus fixes it, by (user index, prompt length)

Exit 0 identical, 1 a difference, 2 an unreadable or incomparable tree (a missing arm, or no path records: use
--texts-only to accept that, knowing the slide is then unchecked).

  python speed_window_compare.py FIRST_RESULTS [SECOND_RESULTS] [--texts-only] [--strict-concurrent-prefixes]
                                 [--position-keyed] [--batched-vs-singles] [--min-coverage 0.5] [--users 2,3] [--json OUT]
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
FULL_ROWS = 16            # one anchor and fifteen drafts: the block width of a T16 round
MIN_COVERAGE = 0.5


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


def user_records(report, user):
    """One user's decoded path records (real_text_compare.decode_paths), or None when the report carries none."""
    entry = ((((report or {}).get('s2') or {}).get('paths') or {}).get('users') or {}).get(str(user))
    if not entry or entry.get('encoded') is None:
        return None
    return decode_paths(entry['encoded'])


def user_prefixes(report, user):
    """(path, prefix) per round from the report's path records, or None when it carries none."""
    records = user_records(report, user)
    return None if records is None else [(record['path'], record['prefix']) for record in records]


def first_difference(first, second):
    for index, (a, b) in enumerate(zip(first, second)):
        if a != b:
            return index
    return None if len(first) == len(second) else min(len(first), len(second))


def keyed_by_position(records):
    """{position: [(prefix, emitted)]} of the full-width rounds of a user's records, in round order. A packed round names no
    row count (it is always the block width); a sequential round names its own, and only 16 is a full block."""
    found = {}
    for record in records:
        if record.get('position') is None or record.get('prefix') is None:
            continue
        if record.get('rows') not in (None, FULL_ROWS):
            continue
        found.setdefault(record['position'], []).append((record['prefix'], record.get('emitted')))
    return found


def agree(left, right):
    """Two rounds' (prefix, emitted) lists at one position: the prefixes always, the emitted counts where both carry one."""
    if len(left) != len(right):
        return False
    return all(a[0] == b[0] and (a[1] is None or b[1] is None or a[1] == b[1]) for a, b in zip(left, right))


def compare_positions(first, second, min_coverage=MIN_COVERAGE):
    """(entry, identical) of two users' records lists compared by position: `common` positions both drafted at,
    `differing` among them, `coverage` = common / the shorter arm's positions, and the first difference. Identical needs a
    common position, no difference and the coverage."""
    left, right = keyed_by_position(first), keyed_by_position(second)
    common = sorted(set(left) & set(right))
    differing = [position for position in common if not agree(left[position], right[position])]
    shorter = min(len(left), len(right))
    coverage = len(common) / shorter if shorter else 0.0
    entry = dict(positions=(len(left), len(right)), common=len(common), only_first=len(left) - len(common),
                 only_second=len(right) - len(common), differing=len(differing), coverage=round(coverage, 3),
                 first_difference_position=differing[0] if differing else None)
    return entry, bool(common) and not differing and coverage >= min_coverage


def compare_arm(first, second, texts_only=False, position_keyed=False, min_coverage=MIN_COVERAGE, users=None):
    """One arm's per-user rows: text equality with the first divergence, and the prefix sequences' equality (by round index,
    or by position under position_keyed)."""
    a_texts, b_texts = user_texts(first), user_texts(second)
    rows = []
    for user in range(max(len(a_texts), len(b_texts))):
        if users is not None and user not in users:
            continue
        a = a_texts[user] if user < len(a_texts) else None
        b = b_texts[user] if user < len(b_texts) else None
        row = dict(user=user, text_equal=a is not None and a == b, text_first_divergence=None,
                   prefix_equal=None, prefix_first_divergence=None, rounds=None)
        if a is not None and b is not None and a != b:
            row['text_first_divergence'] = first_divergence(a, b)
        if not texts_only:
            ra, rb = user_records(first, user), user_records(second, user)
            if ra is not None and rb is not None:
                if position_keyed:
                    entry, identical = compare_positions(ra, rb, min_coverage)
                    row.update(rounds=entry['positions'], prefix_equal=identical, positions=entry,
                               prefix_first_divergence=entry['first_difference_position'])
                else:
                    pa = [(record['path'], record['prefix']) for record in ra]
                    pb = [(record['path'], record['prefix']) for record in rb]
                    row['rounds'] = (len(pa), len(pb))
                    row['prefix_equal'] = pa == pb
                    row['prefix_first_divergence'] = None if pa == pb else first_difference(pa, pb)
        rows.append(row)
    return rows


def compare_within(root, min_coverage=MIN_COVERAGE, users=None):
    """One tree: every user's concurrent-arm rounds (drafted in batches where the users allow) against its own solo-arm rounds
    (one user never batches), by position. -> (rows, problems, incomparable)."""
    try:
        concurrent, solo = load(root, 'matrix-concurrent'), load(root, 'matrix-solo')
    except (OSError, ValueError) as error:
        return [], [], ['%s' % error]
    a_texts, b_texts = user_texts(concurrent), user_texts(solo)
    rows, problems, incomparable = [], [], []
    for user in range(max(len(a_texts), len(b_texts))):
        if users is not None and user not in users:
            continue
        a = a_texts[user] if user < len(a_texts) else None
        b = b_texts[user] if user < len(b_texts) else None
        row = dict(user=user, text_equal=a is not None and a == b, prefix_equal=None, positions=None)
        if not row['text_equal']:
            problems.append('%s user %d: the concurrent text differs from the solo text (first at character %s)'
                            % (root, user, first_divergence(a or '', b or '')))
        ra, rb = user_records(concurrent, user), user_records(solo, user)
        if ra is None or rb is None:
            incomparable.append('%s user %d: no accepted-prefix records in the concurrent or the solo arm' % (root, user))
        else:
            entry, identical = compare_positions(ra, rb, min_coverage)
            row.update(prefix_equal=identical, positions=entry)
            if entry['differing']:
                problems.append('%s user %d: the concurrent accepted prefix differs from the solo one at %d of %d shared '
                                'positions (first at position %s): the batched draft is not the single-user draft'
                                % (root, user, entry['differing'], entry['common'], entry['first_difference_position']))
            elif not identical:
                incomparable.append('%s user %d: only %d of the shorter arm\'s positions are shared (coverage %s, needed %s)'
                                    % (root, user, entry['common'], entry['coverage'], min_coverage))
        rows.append(row)
    if not rows:
        incomparable.append('%s: no streams' % root)
    return rows, problems, incomparable


def compare_trees(first_root, second_root, texts_only=False, strict_concurrent=False, position_keyed=False,
                  min_coverage=MIN_COVERAGE, users=None):
    """(result, code): result {arms: {arm: rows}, problems: [...]}; code 0, 1 or 2 (module docstring)."""
    arms, problems, incomparable = {}, [], []
    for arm in ARMS:
        try:
            first, second = load(first_root, arm), load(second_root, arm)
        except (OSError, ValueError) as error:
            incomparable.append('%s: %s' % (arm, error))
            continue
        keyed = position_keyed and arm == 'matrix-concurrent'
        rows = compare_arm(first, second, texts_only, keyed, min_coverage, users)
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
                if keyed:
                    detail = row['positions']
                    if detail['differing']:
                        problems.append('%s user %d: accepted prefix differs at %d of %d shared positions (first at position '
                                        '%s): a different draft' % (arm, row['user'], detail['differing'], detail['common'],
                                                                    detail['first_difference_position']))
                    else:
                        incomparable.append('%s user %d: only %d positions shared (coverage %s, needed %s)'
                                            % (arm, row['user'], detail['common'], detail['coverage'], min_coverage))
                else:
                    problems.append('%s user %d: accepted-prefix sequence differs (first at round %s): a different draft'
                                    % (arm, row['user'], row['prefix_first_divergence']))
    result = dict(first=str(first_root), second=str(second_root), arms=arms, problems=problems,
                  incomparable=incomparable, texts_only=texts_only, position_keyed=position_keyed)
    code = 1 if problems else (2 if incomparable else 0)
    return result, code


def compare_batched_vs_singles(roots, min_coverage=MIN_COVERAGE, users=None):
    """(result, code) of the within-tree comparison of every root."""
    trees, problems, incomparable = {}, [], []
    for root in roots:
        rows, found, missing = compare_within(root, min_coverage, users)
        trees[str(root)] = rows
        problems += found
        incomparable += missing
    result = dict(within=trees, problems=problems, incomparable=incomparable, min_coverage=min_coverage)
    return result, 1 if problems else (2 if incomparable else 0)


def render(result, code):
    lines = ['speed_window_compare: %s  vs  %s' % (result['first'], result['second'])]
    for arm, rows in result['arms'].items():
        for row in rows:
            if row['prefix_equal'] is None:
                prefix = 'n/a'
            elif row['prefix_equal']:
                prefix = 'IDENTICAL'
            else:
                prefix = 'DIFFERS at %s %s' % ('position' if 'positions' in row else 'round', row['prefix_first_divergence'])
            lines.append('  %-17s user %d  text %-9s  prefixes %s' % (
                arm, row['user'], 'IDENTICAL' if row['text_equal'] else 'DIFFERS', prefix))
    lines += ['  PROBLEM: ' + text for text in result['problems']]
    lines += ['  INCOMPARABLE: ' + text for text in result['incomparable']]
    lines.append('verdict: %s' % {0: 'IDENTICAL', 1: 'DIFFERENT', 2: 'INCOMPARABLE'}[code])
    return '\n'.join(lines)


def render_within(result, code):
    lines = ['speed_window_compare: concurrent (batched drafts) against solo (singles), by position']
    for root, rows in result['within'].items():
        lines.append('  %s' % root)
        for row in rows:
            detail = row['positions']
            if detail is None:
                lines.append('    user %d  text %-9s  prefixes n/a' % (row['user'], 'IDENTICAL' if row['text_equal'] else 'DIFFERS'))
                continue
            lines.append('    user %d  text %-9s  prefixes %s  (%d shared positions of %s, coverage %s)' % (
                row['user'], 'IDENTICAL' if row['text_equal'] else 'DIFFERS',
                'IDENTICAL' if row['prefix_equal'] else 'DIFFERS at position %s' % detail['first_difference_position']
                if detail['differing'] else 'TOO FEW SHARED', detail['common'], detail['positions'], detail['coverage']))
    lines += ['  PROBLEM: ' + text for text in result['problems']]
    lines += ['  INCOMPARABLE: ' + text for text in result['incomparable']]
    lines.append('verdict: %s' % {0: 'IDENTICAL', 1: 'DIFFERENT', 2: 'INCOMPARABLE'}[code])
    return '\n'.join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('first', type=Path)
    parser.add_argument('second', type=Path, nargs='?')
    parser.add_argument('--texts-only', action='store_true', help='skip the accepted-prefix comparison')
    parser.add_argument('--strict-concurrent-prefixes', action='store_true',
                        help='also fail on a differing concurrent-arm prefix sequence (only where the round mix matches, '
                             'or with --position-keyed)')
    parser.add_argument('--position-keyed', action='store_true',
                        help="compare the concurrent arm's prefixes by the position each round drafted at")
    parser.add_argument('--batched-vs-singles', action='store_true',
                        help="within each tree, the concurrent arm's prefixes against the solo arm's, by position")
    parser.add_argument('--min-coverage', type=float, default=MIN_COVERAGE,
                        help='the share of the shorter arm\'s positions two arms must have in common (default 0.5)')
    parser.add_argument('--users', help='judge only these users, comma-separated 0-based stream indexes (default: every user)')
    parser.add_argument('--json', type=Path)
    options = parser.parse_args(argv)
    users = None
    if options.users is not None:
        try:
            users = {int(part) for part in options.users.split(',') if part.strip() != ''}
        except ValueError:
            users = None
        if not users or min(users) < 0:
            parser.error('--users is a comma-separated list of stream indexes, 0 or more')
    if options.second is None and not options.batched_vs_singles:
        parser.error('SECOND_RESULTS is needed unless --batched-vs-singles judges one tree')
    if not 0 < options.min_coverage <= 1:
        parser.error('--min-coverage is a share in (0, 1]')
    code, text, document = 0, [], {}
    if options.batched_vs_singles:
        roots = [options.first] + ([options.second] if options.second is not None else [])
        within, within_code = compare_batched_vs_singles(roots, options.min_coverage, users)
        text.append(render_within(within, within_code))
        document['within'] = within
        code = within_code
    if options.second is not None:
        result, pair_code = compare_trees(options.first, options.second, options.texts_only,
                                          options.strict_concurrent_prefixes, options.position_keyed, options.min_coverage,
                                          users)
        text.append(render(result, pair_code))
        document['trees'] = result
        code = 1 if 1 in (code, pair_code) else (2 if 2 in (code, pair_code) else 0)
    print('\n'.join(text))
    if options.json:
        options.json.write_text(json.dumps(document if options.batched_vs_singles else document['trees'], indent=2),
                                encoding='utf-8')
    return code


if __name__ == '__main__':
    sys.exit(main())
