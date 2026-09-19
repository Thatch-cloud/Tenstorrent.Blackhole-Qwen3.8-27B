"""Decide a decode A/B, with the regression floor that a prefill gate learned to have.

Token equality is not checked here and does not need to be: the CCL link change
was already gated for output equality on the prefill path (run 35430783967,
tokens byte-identical). This run answers the only question left, which is whether
it is faster in decode, where issue 55125 recorded +2.4% when the override was
first measured.

The controls are the ones that survived today:

  lever moved       the marker must appear in the fixed arm's log, and the
                    marker carries its value so it cannot match a patch whose
                    request was ignored
  no regression     run 35431417507 passed every correctness control on a change
                    that made serving 14x slower, because nothing asked whether
                    it got slower. A floor is a control.
  both arms real    a missing or errored arm is a failure, not a partial pass

A decode improvement worth shipping has to clear the floor AND beat it by more
than the run-to-run spread, so the reporter prints the margin rather than
declaring victory on a fraction of a percent.
"""

import argparse
import io
import json
import sys


def load(path):
    return json.load(io.open(path, encoding='utf-8'))


def rate(report):
    for key in ('tokens_per_user_per_s', 'aggregate_tokens_per_s'):
        value = report.get(key)
        if value:
            return float(value), key
    return None, None


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--baseline', required=True)
    parser.add_argument('--fixed', required=True)
    parser.add_argument('--fixed-log', required=True)
    parser.add_argument('--marker', required=True)
    parser.add_argument('--min-speedup', type=float, default=0.95)
    parser.add_argument('--worth-shipping', type=float, default=1.02,
                        help='below this the change is not distinguishable from noise')
    parser.add_argument('--json')
    options = parser.parse_args()

    base = load(options.baseline)
    fixed = load(options.fixed)
    log = io.open(options.fixed_log, encoding='utf-8', errors='replace').read()

    base_rate, base_key = rate(base)
    fixed_rate, _ = rate(fixed)
    hits = log.count(options.marker)

    report = {'marker': options.marker, 'marker_hits': hits, 'metric': base_key,
              'baseline': base_rate, 'fixed': fixed_rate,
              'baseline_error': base.get('error'), 'fixed_error': fixed.get('error')}

    both_real = bool(base_rate and fixed_rate
                     and not base.get('error') and not fixed.get('error'))
    speedup = round(fixed_rate / base_rate, 4) if both_real else None
    report['speedup'] = speedup

    lever_moved = hits > 0
    no_regression = bool(speedup) and speedup >= options.min_speedup
    report['controls'] = {'lever_moved': lever_moved,
                          'no_regression': no_regression,
                          'both_arms_real': both_real}
    report['gate_passed'] = lever_moved and no_regression and both_real
    if speedup:
        report['worth_shipping'] = speedup >= options.worth_shipping
        report['margin_pct'] = round(100 * (speedup - 1.0), 2)

    print(json.dumps(report, indent=2))
    print()
    print('  %s: %s -> %s  speedup %s' % (base_key, base_rate, fixed_rate, speedup))
    for name, value in report['controls'].items():
        print('  control %-18s %s' % (name, value))
    print()
    if report['gate_passed'] and not report.get('worth_shipping'):
        print('GATE PASSED but the margin is %.2f%%, under the %.0f%% worth-shipping bar.'
              % (report.get('margin_pct', 0.0), 100 * (options.worth_shipping - 1)))
        print('A serving default should not change for a number this close to noise.')
    print('GATE %s' % ('PASSED' if report['gate_passed'] else 'FAILED'))

    if options.json:
        io.open(options.json, 'w', encoding='utf-8', newline='\n').write(
            json.dumps(report, indent=2) + '\n')
    return 0 if report['gate_passed'] else 1


if __name__ == '__main__':
    sys.exit(main())
