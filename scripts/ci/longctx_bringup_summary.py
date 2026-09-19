"""Summarise the long-context bring-up, stating only what the run observed."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--log', type=Path)
    options = parser.parse_args()
    if not options.report.is_file() or not options.report.read_text().strip():
        print('No structured report; tail of raw stdout:')
        if options.log and options.log.is_file():
            print('\n'.join(options.log.read_text(errors='replace').splitlines()[-30:]))
        return
    report = json.loads(options.report.read_text())
    plan = report['plan']
    print('PLAN   %d users x %d ctx -> %d blocks, %.2f GiB KV'
          % (plan['users'], plan['context'], plan['blocks'], plan['kv_gib']))
    print('SPEC   %s' % report.get('speculative'))
    print('READY  %s   stage=%s   startup_s=%s'
          % (report.get('ready'), report.get('stage'), report.get('startup_seconds')))
    if report.get('error'):
        print('ERROR  %s' % report['error'][:400])
    if report.get('smoke_completion') is not None:
        print('SMOKE  %r' % report['smoke_completion'][:160])
    if report.get('smoke_error'):
        print('SMOKE ERROR %s' % report['smoke_error'][:300])
    log = report.get('log') or {}
    for key in ('kv_cache_lines', 'gpu_blocks_lines', 'oom_lines'):
        for line in log.get(key, []):
            print('  [%-3s] %s' % (key[:3], line[:200]))
    print('CHECKPOINT A %s' % ('SATISFIED' if report.get('ready') else 'NOT SATISFIED'))


if __name__ == '__main__':
    main()
