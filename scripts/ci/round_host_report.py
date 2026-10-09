"""round_host_report: where the host time of a verified round goes, from the per-round ledger (round_host.py, `[PACKED-ROUND-HOST]`), and the
paired reading of two arms phase by phase (docs/tp4-round-host.md).

    python3 scripts/ci/round_host_report.py summary <server log> [--live 8] [--window 32k]
    python3 scripts/ci/round_host_report.py pair <A server log> <B server log> [--live 8] [--window 32k] [--bucket 1024] [--min-matched 100]

WHAT IT READS. Every `[PACKED-ROUND-HOST]` line of a log (the control arm writes them under QWEN_FAST_TP4_ROUND_HOST_LOG alone, the lever arm under the
same flag and LEAN), keeping the steps with `live` live users. A step's fields are milliseconds of HOST time:
    gap     the end of the last step's drafts to this step's entry (vLLM's scheduler and the worker hook between two execute_model calls)
    entry   step entry to the packed step (admission, the storage check, the runner's state update, the page bindings)
    v0/v1   each block's verify phase (v*_in the staged inputs, v*_tr the blocking trace, v*_sy the replay's checks, v*_rb the prediction read back)
    c0/c1   each block's commits (the publication enqueue and the bookkeeping)
    ed      the early draft; ed_pre its host work before the first quad launches, launch the quads' host prep, window the pre-stage window (host work
            hidden under the quads), fence the wait for the quads, collect the draft read back, flush the GDN commit enqueue, select the FP64 selection,
            tail the rest of the draft (tickets and the cache of the drafts)
    step    the packed step as a whole (v0 + c0 + v1 + c1 and the glue between)
A field the step did not have is `-` and is left out of that field's statistics.

HOW IT PAIRS. By mean-position bucket (the ledger's `pos`, the mean frontier of the live users), never by round index: the two arms admit their seats at
different offsets. Per bucket the medians of the two arms are compared; a field's delta is the mean of the per-bucket deltas weighted by min(nA, nB). A
field with fewer than `min_matched` matched steps is VOID. The inputs of every lever are on the CRITICAL path or hidden under a device trace, and
this report cannot tell which: it says how much HOST time each phase took, and the round time (w2ln_timing_compare.py pair) says how much of it was on the
path. Read both: a phase that fell and a round that did not means the phase was hidden.

Stdlib only.
"""

import argparse
import json
import re
import statistics
import sys

import w2ln_timing_compare

LEDGER = re.compile(r'\[PACKED-ROUND-HOST\] round=(\d+) live=(\S+) pos=(\S+) ((?:\w+=\S+ )+)sel=(\d+) read=(\d+) keyed=(\d+) guard=(\d+)')
DEFAULT_BUCKET = 1024
DEFAULT_MIN_MATCHED = 100
FIELD_ORDER = ('gap', 'entry', 'v0', 'v0_in', 'v0_tr', 'v0_sy', 'v0_rb', 'c0', 'v1', 'v1_in', 'v1_tr', 'v1_sy', 'v1_rb', 'c1', 'ed', 'ed_pre', 'launch',
               'window', 'fence', 'collect', 'flush', 'select', 'tail', 'step')


def steps_of(log_text, live=8, window=None):
    """[{position, fields, sel, read, keyed, guard}] of a log's ledger lines with `live` live users (and a mean position inside `window`, a
    w2ln_timing_compare window name or a (low, high) pair)."""
    bounds = w2ln_timing_compare.resolve_window(window)
    found = []
    for _round, users, position, fields, sel, read, keyed, guard in LEDGER.findall(log_text):
        if users != str(live) or position == '-':
            continue
        position = int(position)
        if bounds is not None and not bounds[0] <= position < bounds[1]:
            continue
        values = {}
        for item in fields.split():
            name, _, value = item.partition('=')
            if value != '-':
                values[name] = float(value)
        found.append(dict(position=position, fields=values, sel=int(sel), read=int(read), keyed=int(keyed), guard=int(guard)))
    return found


def field_names(steps):
    seen = {name for step in steps for name in step['fields']}
    return [name for name in FIELD_ORDER if name in seen]


def summary(log_text, live=8, window=None):
    """{field: dict(n, mean, p50, p90)} and the paths taken (the share of steps with each counter above 0)."""
    steps = steps_of(log_text, live, window)
    out = dict(live=live, window=window, steps=len(steps), fields={})
    for name in field_names(steps):
        values = sorted(step['fields'][name] for step in steps if name in step['fields'])
        out['fields'][name] = dict(n=len(values), mean=round(sum(values) / len(values), 3), p50=round(values[len(values) // 2], 3),
                                   p90=round(values[min(len(values) - 1, len(values) * 9 // 10)], 3))
    if steps:
        out['paths'] = {key: round(sum(1 for step in steps if step[key] > 0) / len(steps), 3) for key in ('sel', 'read', 'keyed', 'guard')}
    return out


def pair(log_a, log_b, live=8, window=None, bucket=DEFAULT_BUCKET, min_matched=DEFAULT_MIN_MATCHED):
    """Each field of arm B against arm A, paired by mean-position bucket: {field: dict(verdict, matched, a_median, b_median, delta)}; the delta is
    B minus A in ms (negative is less host time)."""
    steps_a, steps_b = steps_of(log_a, live, window), steps_of(log_b, live, window)
    result = dict(live=live, window=window, steps_a=len(steps_a), steps_b=len(steps_b), fields={})
    for name in [item for item in FIELD_ORDER if item in set(field_names(steps_a)) & set(field_names(steps_b))]:
        a, b = {}, {}
        for steps, table in ((steps_a, a), (steps_b, b)):
            for step in steps:
                if name in step['fields']:
                    table.setdefault(step['position'] // bucket, []).append(step['fields'][name])
        weight, total, rows = 0, 0.0, []
        for key in sorted(set(a) & set(b)):
            matched = min(len(a[key]), len(b[key]))
            delta = statistics.median(b[key]) - statistics.median(a[key])
            rows.append((key * bucket, matched, statistics.median(a[key]), statistics.median(b[key]), delta))
            weight += matched
            total += delta * matched
        entry = dict(matched=weight)
        if weight < min_matched:
            entry.update(verdict='VOID', reason='%d matched steps, under %d' % (weight, min_matched))
        else:
            keep_a = [value for key in set(a) & set(b) for value in a[key]]
            keep_b = [value for key in set(a) & set(b) for value in b[key]]
            entry.update(verdict='MEASURED', delta_ms=round(total / weight, 3), a_median=round(statistics.median(keep_a), 3),
                         b_median=round(statistics.median(keep_b), 3), a_mean=round(sum(keep_a) / len(keep_a), 3),
                         b_mean=round(sum(keep_b) / len(keep_b), 3))
        result['fields'][name] = entry
    return result


def render_summary(found):
    lines = ['live=%s window=%s steps=%d' % (found['live'], found['window'] or 'all', found['steps'])]
    for name, item in found['fields'].items():
        lines.append('%-8s n=%-5d mean=%8.3f p50=%8.3f p90=%8.3f' % (name, item['n'], item['mean'], item['p50'], item['p90']))
    if 'paths' in found:
        lines.append('paths taken (share of steps): ' + ' '.join('%s=%.2f' % item for item in found['paths'].items()))
    return '\n'.join(lines)


def render_pair(found):
    lines = ['live=%s window=%s steps A=%d B=%d (B minus A, ms; negative is less host time)' % (
        found['live'], found['window'] or 'all', found['steps_a'], found['steps_b'])]
    for name, item in found['fields'].items():
        if item['verdict'] == 'VOID':
            lines.append('%-8s VOID (%s)' % (name, item['reason']))
        else:
            lines.append('%-8s matched=%-5d A p50=%8.3f B p50=%8.3f  delta=%+8.3f   (means %8.3f -> %8.3f)' % (
                name, item['matched'], item['a_median'], item['b_median'], item['delta_ms'], item['a_mean'], item['b_mean']))
    return '\n'.join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)
    one = sub.add_parser('summary', help='the per-field statistics of one log')
    one.add_argument('log')
    two = sub.add_parser('pair', help='arm B against arm A, field by field')
    two.add_argument('a_log')
    two.add_argument('b_log')
    two.add_argument('--bucket', type=int, default=DEFAULT_BUCKET)
    two.add_argument('--min-matched', type=int, default=DEFAULT_MIN_MATCHED)
    for each in (one, two):
        each.add_argument('--live', type=int, default=8)
        each.add_argument('--window', choices=sorted(w2ln_timing_compare.WINDOWS))
        each.add_argument('--json', action='store_true')
    options = parser.parse_args(argv)
    def read(path):
        with open(path, encoding='utf-8', errors='replace') as handle:
            return handle.read()

    if options.command == 'summary':
        found = summary(read(options.log), options.live, options.window)
        print(json.dumps(found) if options.json else render_summary(found))
        return 0 if found['steps'] else 1
    found = pair(read(options.a_log), read(options.b_log), options.live, options.window, options.bucket, options.min_matched)
    print(json.dumps(found) if options.json else render_pair(found))
    return 0 if found['fields'] else 1


if __name__ == '__main__':
    sys.exit(main())
