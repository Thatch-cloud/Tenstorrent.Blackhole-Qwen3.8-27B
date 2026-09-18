"""Summarise a prefetcher run report for the CI log, claiming nothing it did not observe."""

import argparse
import json
from pathlib import Path

FIELDS = ('design_doc_found', 'mesh_open', 'supported', 'gcb_created',
          'receiver_cores', 'grid', 'prefetcher_started', 'prefetcher_stopped',
          'mesh_closed', 'stopped_at')


def summarize(report):
    lines = []
    for field in FIELDS:
        if field in report:
            lines.append('  %-20s %s' % (field, report[field]))
    kernel_ran = bool(report.get('prefetcher_started'))
    lines.append('  %-20s %s' % ('kernel_ran', kernel_ran))
    lines.append('  %-20s %s' % ('override_set', report.get('override_set')))
    for key in ('error', 'stop_error', 'close_error', 'fatal'):
        if report.get(key):
            lines.append('%s tail: %s' % (key, str(report[key])[-800:]))
    return '\n'.join(lines), kernel_ran


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--log', type=Path)
    options = parser.parse_args()
    if not options.report.is_file() or not options.report.read_text().strip():
        print('No structured report produced; tail of the raw log:')
        if options.log and options.log.is_file():
            print('\n'.join(options.log.read_text(errors='replace').splitlines()[-25:]))
        return
    report = json.loads(options.report.read_text())
    text, kernel_ran = summarize(report)
    print(text)
    print('\nCHECKPOINT 2 %s' % ('SATISFIED' if kernel_ran else 'NOT YET SATISFIED'))


if __name__ == '__main__':
    main()
