"""Report the M1 equality gate, stating only what the run observed."""

import argparse
import json
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--log', type=Path)
    options = parser.parse_args()
    if not options.report.is_file() or not options.report.read_text().strip():
        print('No structured report; tail of stdout:')
        if options.log and options.log.is_file():
            print('\n'.join(options.log.read_text(errors='replace').splitlines()[-30:]))
        return 1
    report = json.loads(options.report.read_text())
    print('context=%s chunk_size=%s targets=%s'
          % (report.get('context'), report.get('chunk_size'), report.get('targets')))
    for arm in ('baseline', 'resumable'):
        data = report.get(arm) or {}
        print('%-10s ready=%s completions=%d %s'
              % (arm, data.get('ready'), len(data.get('completions') or []),
                 ('ERROR ' + data['error'][:200]) if data.get('error') else ''))
    if report.get('fatal'):
        print('FATAL %s' % report['fatal'][:300])
    for entry in report.get('comparisons') or []:
        if not entry.get('both_present'):
            print('  %-12s missing from one arm' % entry['name'])
            continue
        mark = 'IDENTICAL' if entry['identical'] else 'DIVERGED'
        print('  %-12s prompt_tokens=%-7s %s  baseline %.2fs  resumable %.2fs'
              % (entry['name'], entry.get('prompt_tokens'), mark,
                 entry.get('baseline_seconds') or 0.0, entry.get('resumable_seconds') or 0.0))
        if not entry['identical']:
            print('        first divergence at char %s' % entry.get('first_divergence'))
            print('        baseline  %r' % (entry.get('baseline_at') or '')[:60])
            print('        resumable %r' % (entry.get('resumable_at') or '')[:60])
    controls = report.get('controls')
    if controls:
        for name in sorted(controls):
            print('control  %-28s %s' % (name, 'ok' if controls[name] else 'FAILED'))
    passed = bool(report.get('gate_passed'))
    print('\nM1 GATE %s  (%s lengths checked)'
          % ('PASSED' if passed else 'NOT PASSED', report.get('lengths_checked')))
    return 0 if passed else 1


if __name__ == '__main__':
    sys.exit(main())
