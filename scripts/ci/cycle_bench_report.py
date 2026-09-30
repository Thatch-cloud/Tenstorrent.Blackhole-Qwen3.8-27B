"""Summarise the concurrent cycle bench against the 200 tok/s per user target."""

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
    print('users=%s context=%s blocks=%s' % (report.get('users'), report.get('context'),
                                             report.get('blocks')))
    print('ready=%s startup_s=%s' % (report.get('ready'), report.get('startup_seconds')))
    if report.get('error'):
        print('ERROR %s' % report['error'][:400])
    for index, stream in enumerate(report.get('streams') or []):
        if not stream:
            continue
        if stream.get('error'):
            print('  stream %d ERROR %s' % (index, stream['error'][:200]))
        else:
            print('  stream %d tokens=%s ttft=%.2fs wall=%.2fs'
                  % (index, stream.get('tokens'), stream.get('ttft_s') or 0.0,
                     stream.get('wall_s') or 0.0))
    if report.get('itl_ms_median') is not None:
        target = report['target_itl_ms']
        print('')
        print('ITL median %.3f ms   mean %.3f ms   p90 %.3f ms'
              % (report['itl_ms_median'], report['itl_ms_mean'], report['itl_ms_p90']))
        print('per user   %.1f tok/s     aggregate %.1f tok/s'
              % (report['tokens_per_user_per_s'], report['aggregate_tokens_per_s']))
        print('target     %.1f tok/s per user (%.1f ms ITL)'
              % (report['target_tokens_per_user'], target))
        print('FRACTION OF TARGET %.2fx  (need %.2fx further speedup)'
              % (report['fraction_of_target'],
                 1.0 / report['fraction_of_target'] if report['fraction_of_target'] else 0))
    else:
        print('no ITL samples collected')


if __name__ == '__main__':
    main()
