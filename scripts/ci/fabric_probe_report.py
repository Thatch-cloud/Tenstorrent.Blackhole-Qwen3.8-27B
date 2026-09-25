"""Summarise a fabric init probe and optionally publish the verdict as a step output."""

import argparse
import json
from pathlib import Path

INTERESTING = ('Fabric Router Sync', 'handshake', 'TT_THROW', 'Timeout')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--log', type=Path)
    parser.add_argument('--github-output', type=Path)
    parser.add_argument('--key', default='fabric_up')
    options = parser.parse_args()

    verdict = 'unknown'
    if options.report.is_file() and options.report.read_text().strip():
        report = json.loads(options.report.read_text())
        verdict = 'yes' if report.get('fabric_up') else 'no'
        print('fabric_up: %s' % report.get('fabric_up'))
        print('configs available: %s' % report.get('fabric_config_values'))
        for attempt in report.get('attempts', []):
            print('  %-14s opened=%s %s' % (attempt.get('config'),
                                            attempt.get('opened'),
                                            (attempt.get('error') or '')[:220]))
        if report.get('fatal'):
            print('fatal: %s' % report['fatal'][-400:])
    else:
        print('no structured report')

    if options.log and options.log.is_file():
        text = options.log.read_text(errors='replace')
        hits = [line.strip() for line in text.splitlines()
                if any(needle in line for needle in INTERESTING)]
        for line in hits[:6]:
            print('  | %s' % line[:230])

    if options.github_output:
        with options.github_output.open('a') as handle:
            handle.write('%s=%s\n' % (options.key, verdict))
    print('VERDICT %s=%s' % (options.key, verdict))


if __name__ == '__main__':
    main()
