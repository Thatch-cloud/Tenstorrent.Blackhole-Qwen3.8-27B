"""Report the M2 alternation gate, stating only what the run observed."""

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
    print('seqs=%s chunk_size not shown (see gate scope) decode_tokens=%s prefill_target_tokens=%s'
          % (report.get('seqs'), report.get('decode_tokens'), report.get('prefill_target_tokens')))
    for arm in ('m2', 'baseline'):
        data = report.get(arm) or {}
        print('%-10s ready=%s decode_tokens=%s steady_gap_median=%ss overlap_gap_max=%ss %s'
              % (arm, data.get('ready'), data.get('decode_tokens_received'),
                 data.get('steady_gap_median'), data.get('overlap_gap_max'),
                 ('ERROR ' + data['error'][:200]) if data.get('error') else ''))
        if data.get('decode_errors'):
            print('           decode_errors=%s' % data['decode_errors'])
    if report.get('fatal'):
        print('FATAL %s' % report['fatal'][:300])
    print('m2_bounded=%s (budget %ss)  baseline_bounded=%s'
          % (report.get('m2_bounded'), report.get('gap_budget_seconds'),
             report.get('baseline_bounded')))
    controls = report.get('controls')
    if controls:
        for name in sorted(controls):
            print('control  %-28s %s' % (name, 'ok' if controls[name] else 'FAILED'))
    passed = bool(report.get('gate_passed'))
    print('\nM2 GATE %s' % ('PASSED' if passed else 'NOT PASSED'))
    return 0 if passed else 1


if __name__ == '__main__':
    sys.exit(main())
