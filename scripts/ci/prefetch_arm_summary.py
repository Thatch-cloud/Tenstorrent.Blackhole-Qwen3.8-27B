"""Summarise a checkpoint 4 arm report, stating only what the run observed."""

import argparse
import json
from pathlib import Path

STAGES = ('weight_built', 'gcb_built', 'program_config_built', 'matmul_ran', 'pcc_passed')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--log', type=Path)
    options = parser.parse_args()
    if not options.report.is_file() or not options.report.read_text().strip():
        print('No structured report; tail of raw log:')
        if options.log and options.log.is_file():
            print('\n'.join(options.log.read_text(errors='replace').splitlines()[-30:]))
        return
    report = json.loads(options.report.read_text())
    print('supported:', report.get('supported'))
    if 'fatal' in report:
        print('FATAL:', report['fatal'][-900:])
    for arm in report.get('arms', []):
        geom = arm.get('geometry') or {}
        print('\n--- %s / %s (native %s) ---' % (arm['projection'], arm['dtype'], arm['native_dtype']))
        print('    receivers=%s grid=%s banks=%s per_core_N=%s in0_block_w=%s block_count=%s' % (
            geom.get('receivers'), geom.get('grid'), geom.get('banks'),
            geom.get('per_core_N'), arm.get('in0_block_w'), arm.get('block_count')))
        reached = [s for s in STAGES if arm.get(s)]
        print('    reached: %s' % (', '.join(reached) or 'nothing'))
        if arm.get('pcc_message'):
            print('    pcc: %s' % arm['pcc_message'])
        if arm.get('error'):
            print('    error tail: %s' % arm['error'][-700:])


if __name__ == '__main__':
    main()
