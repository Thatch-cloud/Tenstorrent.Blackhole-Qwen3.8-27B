"""Read the host-gap instrument's lines (hostgap_instr, QWEN_FAST_TP4_HOSTGAP_LOG=1 and _PROBE=1) out of a serving container log.

    python3 scripts/ci/hostgap_read.py <container log> [--slow-factor 1.5] [--min-spans 20] [--json out.json]

What it prints, with PROBE ROUNDS DROPPED from every table (a probe round runs its window after the quads, not under them, so its times are not round times):

  ROUNDS   the per-step lines: wall, thread CPU, involuntary context switches, collections;
  SPANS    per span name (prestage per block, window, fence, launch, phase:propose_quad, collect, verify_stage, readback, ...): n and the median, p90 and mean of wall and CPU,
           the involuntary switches and collections per span, and the longest single copy;
  SLOW     the spans whose wall time is more than --slow-factor times their own median, each CLASSIFIED by what the host's own account says it spent the excess on:
             gc               collections took at least half the excess;
             blocked-copy     one host call (a copy, a trace enqueue, a fence, a read) took at least half the excess while the thread's CPU time stayed below half of wall;
             descheduled      involuntary context switches and CPU time under 70 percent of wall: the scheduler took the CPU away;
             cpu-bound        CPU time at least 80 percent of wall and no collection: Python or native work, not waiting;
             wait             anything else (CPU well under wall, no switch, no long call: blocked outside the timed calls);
  PROBE    the probe lines: the timed synchronize after the two quad launches, which is 2Q directly (twoq_ms from the first non-blocking trace enqueue to the end of the wait).
  WINDOW   the existing [PACKED-HOSTGAP-WINDOW] lines: the share of rounds whose fence wait after the window was under 1 ms (the window is the critical path) and the window's own time.

Stdlib only. The same parsing is what test_hostgap_instr pins."""

import argparse
import json
import re
import statistics
import sys

KEYVALUE = re.compile(r'(?<![A-Za-z0-9_])([A-Za-z_][A-Za-z0-9_]*)=(-?[0-9][0-9.]*|[^\s]+)')
NUMBER = re.compile(r'-?[0-9]+(\.[0-9]+)?')
KINDS = ('copy', 'up', 'tnb', 'tb', 'fence', 'read', 'd2h')
SLOW_FACTOR = 1.5
MIN_SPANS = 20
CAUSES = ('gc', 'blocked-copy', 'descheduled', 'cpu-bound', 'wait')


def fields(line):
    out = {}
    for key, value in KEYVALUE.findall(line):
        out[key] = float(value) if NUMBER.fullmatch(value) else value
    return out


def parse(text):
    """{rounds: [dict], spans: [dict], probes: [dict], windows: [dict]} from a log's text, in order."""
    found = dict(rounds=[], spans=[], probes=[], windows=[])
    for line in text.splitlines():
        for marker, target in (('[PACKED-HOSTGAP-ROUND]', 'rounds'), ('[PACKED-HOSTGAP-SPAN]', 'spans'), ('[PACKED-HOSTGAP-PROBE]', 'probes'),
                               ('[PACKED-HOSTGAP-WINDOW]', 'windows')):
            at = line.find(marker)
            if at >= 0:
                found[target].append(fields(line[at + len(marker):]))
                break
    return found


def percentile(values, fraction):
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(fraction * len(ordered)))]


def stats(values):
    if not values:
        return dict(n=0, p50=0.0, p90=0.0, mean=0.0)
    return dict(n=len(values), p50=statistics.median(values), p90=percentile(values, 0.9), mean=statistics.fmean(values))


def longest_call(span):
    """(kind, ms) of the longest single host call the span made."""
    best = ('-', 0.0)
    for kind in KINDS:
        value = span.get(kind + '_max_ms')
        if isinstance(value, float) and value > best[1]:
            best = (kind, value)
    return best


def classify(span, median):
    """The cause of one slow span (CAUSES): see the module docstring."""
    wall, cpu = span.get('wall_ms', 0.0), span.get('cpu_ms', 0.0)
    excess = max(wall - median, 1e-9)
    if span.get('gc_ms', 0.0) >= 0.5 * excess:
        return 'gc'
    kind, longest = longest_call(span)
    if longest >= 0.5 * excess and cpu < 0.5 * wall:
        return 'blocked-copy'
    if span.get('nivcsw', 0) > 0 and cpu < 0.7 * wall:
        return 'descheduled'
    if cpu >= 0.8 * wall:
        return 'cpu-bound'
    return 'wait'


def analyse(parsed, slow_factor=SLOW_FACTOR, min_spans=MIN_SPANS):
    probe_steps = {int(item['step']) for item in parsed['probes'] if 'step' in item}
    probe_steps |= {int(item['step']) for item in parsed['rounds'] if item.get('probe') == 1.0 and 'step' in item}
    rounds = [item for item in parsed['rounds'] if int(item.get('step', -1)) not in probe_steps]
    spans = [item for item in parsed['spans'] if int(item.get('step', -1)) not in probe_steps]
    report = dict(probe_steps=sorted(probe_steps), rounds=dict(
        wall_ms=stats([item['wall_ms'] for item in rounds if 'wall_ms' in item]), cpu_ms=stats([item['cpu_ms'] for item in rounds if 'cpu_ms' in item]),
        nivcsw=sum(item.get('nivcsw', 0) for item in rounds), gc_n=sum(item.get('gc_n', 0) for item in rounds),
        gc_ms=sum(item.get('gc_ms', 0.0) for item in rounds)), spans={}, slow={})
    names = {}
    for item in spans:
        key = str(item.get('name', '?')) + ('' if 'block' not in item else '/' + str(item['block']))
        names.setdefault(key, []).append(item)
    slow_steps = set()
    for key, items in sorted(names.items()):
        walls = [item['wall_ms'] for item in items if 'wall_ms' in item]
        entry = dict(wall_ms=stats(walls), cpu_ms=stats([item.get('cpu_ms', 0.0) for item in items]),
                     nivcsw_mean=statistics.fmean([item.get('nivcsw', 0) for item in items]), gc_n=sum(item.get('gc_n', 0) for item in items),
                     longest_call=max((longest_call(item) for item in items), key=lambda pair: pair[1]))
        report['spans'][key] = entry
        if len(items) >= min_spans:
            median = entry['wall_ms']['p50']
            slow = [item for item in items if item['wall_ms'] > slow_factor * median]
            causes = {cause: 0 for cause in CAUSES}
            for item in slow:
                causes[classify(item, median)] += 1
                slow_steps.add(int(item.get('step', -1)))
            report['slow'][key] = dict(n=len(items), slow=len(slow), share=len(slow) / float(len(items)), median_ms=median, causes=causes)
    steps = {int(item.get('step', -1)) for item in spans}
    report['slow_step_share'] = len(slow_steps) / float(len(steps)) if steps else 0.0
    probes = [item for item in parsed['probes'] if isinstance(item.get('twoq_ms'), float)]
    report['probe'] = dict(n=len(parsed['probes']), twoq_ms=stats([item['twoq_ms'] for item in probes]),
                           sync_ms=stats([item['sync_ms'] for item in parsed['probes'] if 'sync_ms' in item]),
                           launch_ms=stats([item['launch_ms'] for item in parsed['probes'] if 'launch_ms' in item]))
    windows = [item for item in parsed['windows'] if 'fence_wait_ms' in item]
    report['window'] = dict(n=len(windows), window_ms=stats([item['window_ms'] for item in windows if 'window_ms' in item]),
                            fence_wait_ms=stats([item['fence_wait_ms'] for item in windows]),
                            critical_share=(sum(1 for item in windows if item['fence_wait_ms'] < 1.0) / float(len(windows))) if windows else 0.0)
    return report


def render(report):
    def row(name, entry):
        return '%-34s n=%-5d p50=%8.2f p90=%8.2f mean=%8.2f' % (name, entry['n'], entry['p50'], entry['p90'], entry['mean'])

    out = ['HOSTGAP READ (probe steps dropped: %s)' % (report['probe_steps'] or 'none'), '', 'ROUNDS']
    out.append(row('wall_ms', report['rounds']['wall_ms']))
    out.append(row('cpu_ms', report['rounds']['cpu_ms']))
    out.append('involuntary switches %d, collections %d (%.2f ms)' % (report['rounds']['nivcsw'], report['rounds']['gc_n'], report['rounds']['gc_ms']))
    out += ['', 'SPANS (wall ms; cpu p50; involuntary switches a span; longest single host call)']
    for key, entry in report['spans'].items():
        out.append('%s cpu_p50=%.2f nivcsw=%.2f longest=%s %.2f' % (row(key, entry['wall_ms']), entry['cpu_ms']['p50'], entry['nivcsw_mean'], entry['longest_call'][0],
                                                                  entry['longest_call'][1]))
    out += ['', 'SLOW SPANS (more than the factor times their own median), by cause']
    for key, entry in report['slow'].items():
        out.append('%-34s %d of %d (%.1f%%) median %.2f ms: %s' % (key, entry['slow'], entry['n'], 100 * entry['share'], entry['median_ms'],
                                                                 ' '.join('%s=%d' % (cause, entry['causes'][cause]) for cause in CAUSES)))
    out.append('steps with at least one slow span: %.1f%%' % (100 * report['slow_step_share']))
    out += ['', 'PROBE (the timed synchronize after the two quad launches)']
    out.append('%d probes; ' % report['probe']['n'] + '; '.join('%s p50 %.2f p90 %.2f' % (name, report['probe'][name]['p50'], report['probe'][name]['p90'])
                                                                   for name in ('twoq_ms', 'sync_ms', 'launch_ms')))
    out += ['', 'WINDOW (stage 0 lines)']
    out.append('%d windows; window_ms p50 %.2f; fence_wait_ms p50 %.2f; fence wait under 1 ms (the window is the critical path) in %.1f%% of rounds' % (
        report['window']['n'], report['window']['window_ms']['p50'], report['window']['fence_wait_ms']['p50'], 100 * report['window']['critical_share']))
    return '\n'.join(out)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('log')
    parser.add_argument('--slow-factor', type=float, default=SLOW_FACTOR)
    parser.add_argument('--min-spans', type=int, default=MIN_SPANS)
    parser.add_argument('--json')
    options = parser.parse_args(argv)
    with open(options.log, encoding='utf-8', errors='replace') as handle:
        text = handle.read()
    report = analyse(parse(text), options.slow_factor, options.min_spans)
    print(render(report))
    if options.json:
        with open(options.json, 'w', encoding='utf-8') as handle:
            json.dump(report, handle, indent=2, sort_keys=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
