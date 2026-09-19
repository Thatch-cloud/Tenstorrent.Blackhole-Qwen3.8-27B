"""Decide the GDN conv dispatch gate, and carry the verdict into the exit code.

Three things must hold, and a speedup alone is none of them:

  1. the lever moved     the patched arm's log must show full=True and the
                         native conv1d path taken; an arm that silently kept
                         the FIR would report a speedup of zero and read as a
                         physical result
  2. tokens identical    greedy generation from the same prompt, so any
                         numerical difference in the conv path shows up as
                         diverging ids rather than a tolerance argument
  3. both arms complete  a crashed arm is not a pass

The reporter's own failure mode is worth naming: an earlier gate in this project
printed FAIL and exited 0, so CI went green on a failed run. main() returns the
verdict and sys.exit carries it.
"""

import argparse
import io
import json
import sys


def load(path):
    return json.load(io.open(path, encoding='utf-8'))


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--baseline', required=True)
    parser.add_argument('--fixed', required=True)
    parser.add_argument('--fixed-log', required=True)
    parser.add_argument('--json')
    options = parser.parse_args()

    base = load(options.baseline)
    fixed = load(options.fixed)
    log = io.open(options.fixed_log, encoding='utf-8', errors='replace').read()

    report = {'baseline_arm': base.get('arm'), 'fixed_arm': fixed.get('arm')}

    # 1. the lever moved
    full_true = log.count('full=True')
    full_false = log.count('full=False')
    report['marker_full_true'] = full_true
    report['marker_full_false'] = full_false
    lever_moved = full_true > 0

    # 2. tokens identical, per length
    checked = []
    bykey = dict((r.get('target_tokens'), r) for r in fixed.get('results', []))
    for row in base.get('results', []):
        key = row.get('target_tokens')
        peer = bykey.get(key, {})
        same = (row.get('token_ids') is not None
                and row.get('token_ids') == peer.get('token_ids'))
        checked.append({
            'target_tokens': key,
            'identical': bool(same),
            'baseline_ids': row.get('token_ids'),
            'fixed_ids': peer.get('token_ids'),
            'baseline_tok_s': row.get('tokens_per_s'),
            'fixed_tok_s': peer.get('tokens_per_s'),
            'speedup': (round(peer['tokens_per_s'] / row['tokens_per_s'], 3)
                        if row.get('tokens_per_s') and peer.get('tokens_per_s') else None),
        })
    report['checked'] = checked

    complete = bool(base.get('arms_complete')) and bool(fixed.get('arms_complete'))
    identical = bool(checked) and all(c['identical'] for c in checked)

    report['controls'] = {'lever_moved': lever_moved,
                          'tokens_identical': identical,
                          'both_arms_complete': complete}
    report['gate_passed'] = lever_moved and identical and complete

    speeds = [c['speedup'] for c in checked if c['speedup']]
    if speeds:
        report['mean_speedup'] = round(sum(speeds) / len(speeds), 3)

    print(json.dumps(report, indent=2))
    print()
    for c in checked:
        print('  %7s tokens  identical=%-5s  %s -> %s tok/s  speedup %s'
              % (c['target_tokens'], c['identical'], c['baseline_tok_s'],
                 c['fixed_tok_s'], c['speedup']))
    print()
    for name, value in report['controls'].items():
        print('  control %-20s %s' % (name, value))
    print()
    print('GATE %s' % ('PASSED' if report['gate_passed'] else 'FAILED'))

    if options.json:
        io.open(options.json, 'w', encoding='utf-8', newline='\n').write(
            json.dumps(report, indent=2) + '\n')
    return 0 if report['gate_passed'] else 1


if __name__ == '__main__':
    sys.exit(main())
