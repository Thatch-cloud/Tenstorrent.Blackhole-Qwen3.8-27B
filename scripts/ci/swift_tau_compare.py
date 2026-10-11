"""Compare the drafter's acceptance on two checkpoints from two tau-lab public summaries (stdlib only, Python 3.7 syntax).

    python3 scripts/ci/swift_tau_compare.py --base <tau-lab-summary.json of the base run> --other <tau-lab-summary.json of the other checkpoint's run> [--json-out compare.json]

Each C2_ACTIONS=reset taulab job (references/fusion-jobs/SW, T1 on the base and T2 on the other checkpoint, the same image and profile and drafter) uploads taulab/tau-lab-summary.json
in its artifact: aggregates only (tau = tokens committed per speculative round, per arm, per context bucket, per draft position). The prompts are the same; the answers are not (the
checkpoints generate different text), so this is an UNPAIRED read: the pooled tau with its bootstrap interval, each arm's tau and its per-turn p10 / p50, each context bucket's tau
and the acceptance at each draft position, base against other.

The verdict is on the pooled tau of the gate-G arms and their 95% intervals: LOWER when the other's interval lies wholly under the base's, HIGHER when wholly over it, else
INDISTINGUISHABLE. A speculative decode round costs the same whatever it commits, so the decode rate moves with tau: a ratio of 0.93 is about 7% fewer tokens per round
(the decode ABAB of the same window, jobs C1 L1 C2 L2, measures the rate itself). Exit 0 when both runs are complete, 2 when a run is missing or incomplete (the numbers are then
printed as partial).
"""
import argparse
import json
import sys


def load(path):
    with open(path, encoding='utf-8') as handle:
        return json.load(handle)


def ratio(a, b):
    return None if a in (None, 0) or b is None else b / float(a)


def fmt(value, digits=3):
    return 'n/a' if value is None else '%.*f' % (digits, value)


def interval(summary):
    ci = (summary.get('k_inputs') or {}).get('ci95') or {}
    return ci.get('low'), ci.get('high')


def verdict(base, other):
    (base_low, base_high), (other_low, other_high) = interval(base), interval(other)
    if None in (base_low, base_high, other_low, other_high):
        return 'NOT_ESTABLISHED'
    if other_high < base_low:
        return 'LOWER'
    if other_low > base_high:
        return 'HIGHER'
    return 'INDISTINGUISHABLE'


def compare(base, other):
    rows = []
    b_k, o_k = base.get('k_inputs') or {}, other.get('k_inputs') or {}
    rows.append(('pooled', b_k.get('pooled_tau'), o_k.get('pooled_tau')))
    for label in ('p10', 'p50', 'mean'):
        rows.append(('per_turn_' + label, (b_k.get('per_turn') or {}).get(label), (o_k.get('per_turn') or {}).get(label)))
    for arm in sorted(set(base.get('arms') or {}) & set(other.get('arms') or {})):
        a, b = base['arms'][arm], other['arms'][arm]
        rows.append(('arm_%s_tau' % arm, a.get('tau'), b.get('tau')))
        for label in ('p10', 'p50'):
            rows.append(('arm_%s_%s' % (arm, label), (a.get('tau_per_turn') or {}).get(label), (b.get('tau_per_turn') or {}).get(label)))
    for name in sorted(set(b_k.get('buckets') or {}) & set(o_k.get('buckets') or {})):
        rows.append(('bucket_%s' % name, b_k['buckets'][name].get('tau'), o_k['buckets'][name].get('tau')))
    positions = zip((base.get('positions') or {}).get('lab') or [], (other.get('positions') or {}).get('lab') or [])
    for index, (a, b) in enumerate(positions, 1):
        rows.append(('position_%d' % index, a, b))
    return [dict(name=name, base=a, other=b, ratio=ratio(a, b)) for name, a, b in rows]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--base', required=True)
    parser.add_argument('--other', required=True)
    parser.add_argument('--json-out')
    options = parser.parse_args(argv)
    try:
        base, other = load(options.base), load(options.other)
    except (OSError, ValueError) as error:
        print('SWIFT_TAU unreadable: %s' % error, file=sys.stderr)
        return 2
    complete = bool(base.get('complete')) and bool(other.get('complete'))
    rows = compare(base, other)
    answer = verdict(base, other)
    print('SWIFT_TAU complete=%s (base %s, other %s) image base=%s other=%s' % (complete, base.get('complete'), other.get('complete'), base.get('image_tag'), other.get('image_tag')))
    if base.get('image_tag') != other.get('image_tag'):
        print('SWIFT_TAU WARNING: the two runs are on different images; the pair is not a clean checkpoint comparison')
    for row in rows:
        print('SWIFT_TAU %-22s base %s other %s other/base %s' % (row['name'], fmt(row['base']), fmt(row['other']), fmt(row['ratio'])))
    low_b, high_b = interval(base)
    low_o, high_o = interval(other)
    print('SWIFT_TAU verdict=%s pooled tau interval base %s..%s other %s..%s' % (answer, fmt(low_b), fmt(high_b), fmt(low_o), fmt(high_o)))
    if options.json_out:
        with open(options.json_out, 'w', encoding='utf-8') as handle:
            json.dump(dict(complete=complete, verdict=answer, rows=rows), handle, indent=1)
    return 0 if complete else 2


if __name__ == '__main__':
    sys.exit(main())
