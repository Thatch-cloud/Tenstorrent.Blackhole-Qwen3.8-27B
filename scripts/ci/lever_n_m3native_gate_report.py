"""Report the M3native gate, stating only what the run observed."""

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
    print('users=%s context=%s prompt_tokens=%s prompt_base=%s offset=%s'
          % (report.get('users'), report.get('context'), report.get('prompt_tokens'),
             report.get('prompt_base'), report.get('prompt_user_offset')))
    print('ready=%s references_loaded=%s users_checked=%s/%s'
          % (report.get('ready'), report.get('references_loaded'), report.get('users_checked'),
             report.get('users')))
    if report.get('fatal'):
        print('FATAL %s' % report['fatal'][:300])
    for entry in report.get('comparisons') or []:
        if not entry.get('reference_present'):
            print('  user %-2s prompt_base=%-6s NO REFERENCE'
                 % (entry.get('user'), entry.get('prompt_base')))
            continue
        mark = 'IDENTICAL PREFIX' if entry.get('identical_prefix') else 'DIVERGED'
        print('  user %-2s prompt_base=%-6s %-17s reference_len=%-6s actual_len=%-6s %s'
              % (entry.get('user'), entry.get('prompt_base'), mark, entry.get('reference_len'),
                 entry.get('actual_len'), ('ERROR ' + entry['error'][:120]) if entry.get('error') else ''))
    phase = report.get('packed_phase')
    if phase:
        print('[PACKED-PHASE] trace_ms rounds=%s min=%s mean=%s max=%s'
              % (phase.get('rounds'), phase.get('trace_ms_min'), phase.get('trace_ms_mean'),
                 phase.get('trace_ms_max')))
    else:
        print('[PACKED-PHASE] no rounds observed')
    print('native_m3 marker present: %s' % report.get('native_m3_marker_present'))
    print('retired-binder rounds observed: %s' % report.get('retired_binder_rounds_observed'))
    if report.get('retired_binder_calls_nonzero'):
        print('retired-binder calls LEAKED: %s' % report['retired_binder_calls_nonzero'])
    passed = bool(report.get('gate_passed'))
    print('\nM3NATIVE GATE %s' % ('PASSED' if passed else 'NOT PASSED'))
    return 0 if passed else 1


if __name__ == '__main__':
    sys.exit(main())
