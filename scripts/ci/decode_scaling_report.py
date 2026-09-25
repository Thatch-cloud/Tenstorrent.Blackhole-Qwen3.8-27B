"""Tabulate the decode-scaling ladder: does per-sequence device work amortise?

Cells on one iso-product line carry the same total KV (users x context), so if the
stack is purely KV-bandwidth-bound they measure the same inter-token latency. If
per-sequence work scales with users, the higher-user cell on a line is slower, and the
slope across a line is the number that decides whether four or eight users is reachable.

Reads the per-cell cycle.json artifacts already downloaded to a directory.
"""

import argparse
import json
import collections
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, required=True)
    options = parser.parse_args()

    cells = []
    for path in sorted(options.directory.glob('**/cycle.json')):
        try:
            d = json.loads(path.read_text())
        except ValueError:
            continue
        streams = [s for s in (d.get('streams') or []) if s]
        prompt = next((s.get('prompt_tokens') for s in streams if s.get('prompt_tokens')), None)
        cells.append(dict(users=d.get('users'), context=d.get('context'),
                          itl=d.get('itl_ms_median'), rate=d.get('tokens_per_user_per_s'),
                          prompt_tokens=prompt, ready=d.get('ready'),
                          ttft=next((s.get('ttft_s') for s in streams if s.get('ttft_s')), None),
                          error=d.get('error') or next((s.get('error') for s in streams if s.get('error')), None)))
    if not cells:
        print('no cells found under %s' % options.directory)
        return 1

    print('%-7s %9s %12s %10s %10s %10s' % ('users', 'context', 'prompt_tok', 'ITL ms', 'tok/s', 'ttft s'))
    for c in sorted(cells, key=lambda c: (c['users'] or 0, c['context'] or 0)):
        if c['error']:
            print('%-7s %9s   ERROR %s' % (c['users'], c['context'], str(c['error'])[:60]))
            continue
        print('%-7s %9s %12s %10s %10s %10s'
              % (c['users'], c['context'], c['prompt_tokens'], c['itl'], c['rate'],
                 round(c['ttft'], 1) if c['ttft'] else None))

    good = [c for c in cells if c['itl'] and c['prompt_tokens']]
    lines = collections.defaultdict(list)
    for c in good:
        lines[c['users'] * c['context']].append(c)
    print()
    print('iso-product lines (same total KV, so same ITL if per-sequence work amortises)')
    for product in sorted(lines):
        row = sorted(lines[product], key=lambda c: c['users'])
        if len(row) < 2:
            continue
        base = row[0]
        parts = ['%du=%.1fms' % (c['users'], c['itl']) for c in row]
        slope = row[-1]['itl'] / base['itl']
        per_user_factor = row[-1]['users'] / base['users']
        print('  product %8d  %s' % (product, '  '.join(parts)))
        print('      %dx the users -> %.2fx the ITL  (%s)'
              % (per_user_factor, slope,
                 'amortises' if slope < 1.25 else
                 'partially amortises' if slope < per_user_factor * 0.7 else 'scales with users'))

    # PP curve, free from ttft
    pp = [(c['prompt_tokens'], c['ttft'], c['users']) for c in cells
          if c['ttft'] and c['prompt_tokens']]
    if pp:
        print()
        print('prefill (ttft), which is a PP curve for free')
        for tok, ttft, users in sorted(pp):
            print('  %7d prompt tokens x %d users -> %6.1f s -> %7.0f tok/s'
                  % (tok, users, ttft, tok * users / ttft))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
