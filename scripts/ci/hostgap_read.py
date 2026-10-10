"""Read the host-gap instrument's lines (hostgap_instr, QWEN_FAST_TP4_HOSTGAP_LOG=1 and _PROBE=1) out of a serving container log.

    python3 scripts/ci/hostgap_read.py <container log> [--slow-factor 1.5] [--min-spans 20] [--json out.json]

What it prints, with PROBE ROUNDS DROPPED from every table (a probe round runs its window after the quads, not under them, so its times are not round times):

  ROUNDS   the per-step lines: wall, thread CPU, involuntary context switches, collections;
  SPANS    per span name (prestage per block, window, fence, launch, phase:propose_quad, collect, verify_stage, readback, ...): n and the median, p90 and mean of wall and CPU,
           the involuntary switches and collections per span, and the longest single copy;
  SLOW     the spans whose wall time is more than --slow-factor times their own median, each CLASSIFIED by what the host's own account says it spent the excess on:
             gc               collections took at least half the excess;
             blocked-copy     one host call that hands work to the device (a copy, an upload, a trace enqueue) took at least half the excess while the CPU time stayed below half of wall;
             device-wait      the same for a call that waits for the device by design (a fence, a blocking trace, a blocking read);
             descheduled      involuntary context switches and CPU time under 70 percent of wall: the scheduler took the CPU away;
             cpu-bound        CPU time at least 80 percent of wall and no collection: Python or native work, not waiting;
             wait             anything else (CPU well under wall, no switch, no long call: blocked outside the timed calls);
  PROBE    the probe lines: the timed synchronize after the two quad launches, which is 2Q directly (twoq_ms from the first non-blocking trace enqueue to the end of the wait).
  EIGHT-LIVE   the steps in which both blocks were pre-staged, apart from the rest (a median over every step mixes in the one-block rounds, whose window is shorter than one quad), and
           the EXPOSED line: launch + window + fence of the eight-live steps against the probe's first enqueue + 2Q, the host time that is on the critical path;
  CPU BANDWIDTH / THREADS   the cgroup's throttle counters over the rounds and the thread census (hostgap_instr): quota or contention, and which threads are busy;
  FENCE    what the fences waited behind: the copies and trace enqueues made since the previous synchronization point.
  A span whose longest call is a wait the device owes (a fence, a blocking trace, a blocking read) is `device-wait`; `blocked-copy` is for copies, uploads and trace enqueues only.
  WINDOW   the existing [PACKED-HOSTGAP-WINDOW] lines: the share of rounds whose fence wait after the window was under 1 ms (the window is the critical path) and the window's own time.

Stdlib only. The same parsing is what test_hostgap_instr pins."""

import argparse
import collections
import json
import re
import statistics
import sys

KEYVALUE = re.compile(r'(?<![A-Za-z0-9_])([A-Za-z_][A-Za-z0-9_]*)=(-?[0-9][0-9.]*|[^\s]+)')
NUMBER = re.compile(r'-?[0-9]+(\.[0-9]+)?')
KINDS = ('copy', 'up', 'tnb', 'tb', 'fence', 'read', 'd2h')
ENQUEUE_KINDS = ('copy', 'up', 'tnb', 'd2h')       # calls that hand work to the device and return: a long one means the queue or a lock was full
WAITING_KINDS = ('tb', 'fence', 'read')            # calls that wait for the device by design: a long one is the device's time, not a blocked host
SLOW_FACTOR = 1.5
MIN_SPANS = 20
CAUSES = ('gc', 'blocked-copy', 'device-wait', 'descheduled', 'cpu-bound', 'wait')


def fields(line):
    out = {}
    for key, value in KEYVALUE.findall(line):
        out[key] = float(value) if NUMBER.fullmatch(value) else value
    return out


def parse(text):
    """{rounds, spans, probes, windows, threads, blocked: [dict]} from a log's text, in order."""
    found = dict(rounds=[], spans=[], probes=[], windows=[], threads=[], blocked=[])
    for line in text.splitlines():
        for marker, target in (('[PACKED-HOSTGAP-ROUND]', 'rounds'), ('[PACKED-HOSTGAP-SPAN]', 'spans'), ('[PACKED-HOSTGAP-PROBE]', 'probes'),
                               ('[PACKED-HOSTGAP-WINDOW]', 'windows'), ('[PACKED-HOSTGAP-THREADS]', 'threads'), ('[PACKED-HOSTGAP-BLOCKED]', 'blocked')):
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


def longest_call(span, kinds=KINDS):
    """(kind, ms) of the longest single host call of the given kinds that the span made."""
    best = ('-', 0.0)
    for kind in kinds:
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
    if longest_call(span, ENQUEUE_KINDS)[1] >= 0.5 * excess and cpu < 0.5 * wall:
        return 'blocked-copy'
    if longest_call(span, WAITING_KINDS)[1] >= 0.5 * excess and cpu < 0.5 * wall:
        return 'device-wait'
    if span.get('nivcsw', 0) > 0 and cpu < 0.7 * wall:
        return 'descheduled'
    if cpu >= 0.8 * wall:
        return 'cpu-bound'
    return 'wait'


def span_key(item):
    return str(item.get('name', '?')) + ('' if 'block' not in item else '/' + str(item['block']))


def table(items, key):
    values = [item[key] for item in items if key in item and isinstance(item[key], float)]
    return stats(values)


def thread_summary(censuses):
    """The census lines folded: the thread count, the limits, and the threads that were busy, by name: [(comm, mean percent of a CPU over the censuses it showed in, the most it
    showed, involuntary switches summed, its distinct tids)]."""
    by = {}
    for item in censuses:
        for entry in str(item.get('busy', '-')).split(','):
            parts = entry.split(':')
            if len(parts) != 5:
                continue
            comm, tid, percent, inv, cpu = parts
            try:
                share, switches = float(percent.rstrip('%')), int(inv.replace('inv', ''))
            except ValueError:
                continue
            row = by.setdefault(comm, dict(shares=[], max=0.0, inv=0, tids=set()))
            row['shares'].append(share)
            row['max'] = max(row['max'], share)
            row['inv'] += switches
            row['tids'].add(tid)
    rows = [(comm, statistics.fmean(row['shares']), row['max'], row['inv'], len(row['tids'])) for comm, row in by.items()]
    rows.sort(key=lambda row: -row[1])
    return dict(n=len(censuses), threads=stats([item['threads'] for item in censuses if 'threads' in item]),
                quota=sorted({str(item.get('quota', '-')) for item in censuses}), cpuset=sorted({str(item.get('cpuset', '-')) for item in censuses}),
                allowed=sorted({str(item.get('allowed', '-')) for item in censuses}), busy=rows[:10])


def analyse(parsed, slow_factor=SLOW_FACTOR, min_spans=MIN_SPANS):
    probe_steps = {int(item['step']) for item in parsed['probes'] if 'step' in item}
    probe_steps |= {int(item['step']) for item in parsed['rounds'] if item.get('probe') == 1.0 and 'step' in item}
    census_steps = {int(item['step']) for item in parsed['rounds'] if item.get('census') == 1.0 and 'step' in item}
    census_steps |= {int(item['step']) for item in parsed['threads'] if 'step' in item}            # the census runs inside the interval it names (the one the entry opens)
    dropped = probe_steps | census_steps
    rounds = [item for item in parsed['rounds'] if int(item.get('step', -1)) not in dropped]
    spans = [item for item in parsed['spans'] if int(item.get('step', -1)) not in dropped]
    report = dict(probe_steps=sorted(probe_steps), census_steps=sorted(census_steps), rounds=dict(
        wall_ms=stats([item['wall_ms'] for item in rounds if 'wall_ms' in item]), cpu_ms=stats([item['cpu_ms'] for item in rounds if 'cpu_ms' in item]),
        nivcsw=sum(item.get('nivcsw', 0) for item in rounds), gc_n=sum(item.get('gc_n', 0) for item in rounds),
        gc_ms=sum(item.get('gc_ms', 0.0) for item in rounds)), spans={}, slow={})
    names = {}
    for item in spans:
        names.setdefault(span_key(item), []).append(item)
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
    # EIGHT-LIVE steps (a span of block B: both blocks pre-staged) against the rest: a median over every step mixes the one-block rounds, whose window is shorter than one quad.
    eight_steps = {int(item['step']) for item in spans if item.get('block') == 'B' and 'step' in item}
    by_step = {}
    for item in spans:
        by_step.setdefault(int(item.get('step', -1)), []).append(item)
    report['split'] = {}
    for label, wanted in (('eight_live', True), ('other', False)):
        selected = [item for step, items in by_step.items() if (step in eight_steps) == wanted for item in items]
        report['split'][label] = dict(steps=len({int(item.get('step', -1)) for item in selected}), spans={
            key: dict(wall_ms=table(items, 'wall_ms'), cpu_ms=table(items, 'cpu_ms'), nivcsw_mean=statistics.fmean([item.get('nivcsw', 0) for item in items]))
            for key, items in sorted({span_key(item): [x for x in selected if span_key(x) == span_key(item)] for item in selected}.items())})
    probes = [item for item in parsed['probes'] if isinstance(item.get('twoq_ms'), float)]
    report['probe'] = dict(n=len(parsed['probes']), twoq_ms=stats([item['twoq_ms'] for item in probes]),
                           sync_ms=stats([item['sync_ms'] for item in parsed['probes'] if 'sync_ms' in item]),
                           launch_ms=stats([item['launch_ms'] for item in parsed['probes'] if 'launch_ms' in item]),
                           first_enqueue_ms=stats([item['first_enqueue_ms'] for item in probes if isinstance(item.get('first_enqueue_ms'), float)]))
    eight = report['split']['eight_live']['spans']
    if probes and 'launch' in eight and 'window' in eight:
        host = eight['launch']['wall_ms']['p50'] + eight['window']['wall_ms']['p50'] + (eight['fence']['wall_ms']['p50'] if 'fence' in eight else 0.0)
        device = report['probe']['first_enqueue_ms']['p50'] + report['probe']['twoq_ms']['p50']
        report['exposed'] = dict(host_ms=host, device_ms=device, exposed_ms=host - device)
    windows = [item for item in parsed['windows'] if 'fence_wait_ms' in item]
    report['window'] = dict(n=len(windows), window_ms=stats([item['window_ms'] for item in windows if 'window_ms' in item]),
                            fence_wait_ms=stats([item['fence_wait_ms'] for item in windows]),
                            critical_share=(sum(1 for item in windows if item['fence_wait_ms'] < 1.0) / float(len(windows))) if windows else 0.0)
    report['window']['by_blocks'] = {}
    for count in sorted({int(item['blocks']) for item in windows if 'blocks' in item}):
        chosen = [item for item in windows if int(item.get('blocks', 0)) == count]
        report['window']['by_blocks'][count] = dict(n=len(chosen), window_ms=stats([item['window_ms'] for item in chosen if 'window_ms' in item]),
                                                    fence_wait_ms=stats([item['fence_wait_ms'] for item in chosen]),
                                                    critical_share=sum(1 for item in chosen if item['fence_wait_ms'] < 1.0) / float(len(chosen)))
    withs = [item for item in rounds if 'thr_n' in item]
    report['throttle'] = dict(rounds=len(withs), periods=sum(item.get('periods', 0) for item in withs), throttled_periods=sum(item['thr_n'] for item in withs),
                              throttled_ms=sum(item.get('thr_ms', 0.0) for item in withs),
                              rounds_throttled=sum(1 for item in withs if item['thr_n'] > 0))
    queued = [item for item in spans if 'fence_q_copy' in item]
    report['fence_queue'] = dict(n=len(queued), copies=stats([item['fence_q_copy'] for item in queued]), traces=stats([item['fence_q_tnb'] for item in queued]))
    report['threads'] = thread_summary(parsed['threads'])
    report['blocked'] = blocked_summary([item for item in parsed['blocked'] if int(item.get('step', -1)) not in dropped])
    return report


def blocked_summary(items):
    """The single calls of 1 ms or more that handed work to the device: how many, how long against the CPU they used, where they were, how long after the last trace enqueue and
    the last synchronization point, and how much was queued behind that point."""
    bucket = lambda value: '-' if not isinstance(value, float) else ('<1ms' if value < 1 else '1-5ms' if value < 5 else '5-20ms' if value < 20 else '>=20ms')
    return dict(n=len(items), ms=stats([item['ms'] for item in items if 'ms' in item]), cpu_ms=stats([item['cpu_ms'] for item in items if 'cpu_ms' in item]),
                asleep=sum(1 for item in items if item.get('cpu_ms', 0.0) < 0.25 * item.get('ms', 0.0)),
                by_kind=collections.Counter(str(item.get('kind')) for item in items), by_span=collections.Counter(str(item.get('in')) for item in items),
                since_tnb=collections.Counter(bucket(item.get('since_tnb_ms')) for item in items),
                since_sync=collections.Counter(bucket(item.get('since_sync_ms')) for item in items),
                queued_tnb=collections.Counter(int(item.get('queued_tnb', 0)) for item in items),
                queued_copy=stats([item['queued_copy'] for item in items if 'queued_copy' in item]))


def render(report):
    def row(name, entry):
        return '%-34s n=%-5d p50=%8.2f p90=%8.2f mean=%8.2f' % (name, entry['n'], entry['p50'], entry['p90'], entry['mean'])

    out = ['HOSTGAP READ (probe steps dropped: %s; census steps dropped: %d)' % (report['probe_steps'] or 'none', len(report['census_steps'])), '', 'ROUNDS']
    out.append(row('wall_ms', report['rounds']['wall_ms']))
    out.append(row('cpu_ms', report['rounds']['cpu_ms']))
    out.append('involuntary switches %d, collections %d (%.2f ms)' % (report['rounds']['nivcsw'], report['rounds']['gc_n'], report['rounds']['gc_ms']))
    out += ['', 'SPANS (wall ms; cpu p50; involuntary switches a span; longest single host call)']
    for key, entry in report['spans'].items():
        out.append('%s cpu_p50=%.2f nivcsw=%.2f longest=%s %.2f' % (row(key, entry['wall_ms']), entry['cpu_ms']['p50'], entry['nivcsw_mean'], entry['longest_call'][0],
                                                                  entry['longest_call'][1]))
    for label, title in (('eight_live', 'EIGHT-LIVE STEPS (both blocks pre-staged)'), ('other', 'ALL OTHER STEPS (one block, ramp, drain)')):
        part = report['split'][label]
        out += ['', '%s: %d steps' % (title, part['steps'])]
        for key in ('window', 'launch', 'fence', 'collect', 'phase:early_draft', 'prestage/A', 'prestage/B', 'verify_stage/A', 'readback/A'):
            if key in part['spans']:
                entry = part['spans'][key]
                out.append('%s cpu_p50=%.2f nivcsw=%.2f' % (row(key, entry['wall_ms']), entry['cpu_ms']['p50'], entry['nivcsw_mean']))
    if 'exposed' in report:
        exposed = report['exposed']
        out.append('EXPOSED HOST TIME (eight-live): launch + window + fence p50 = %.2f ms against the probe\'s first enqueue + 2Q = %.2f ms: %.2f ms of host work is on the critical path' % (
            exposed['host_ms'], exposed['device_ms'], exposed['exposed_ms']))
    out += ['', 'SLOW SPANS (more than the factor times their own median), by cause']
    for key, entry in report['slow'].items():
        out.append('%-34s %d of %d (%.1f%%) median %.2f ms: %s' % (key, entry['slow'], entry['n'], 100 * entry['share'], entry['median_ms'],
                                                                 ' '.join('%s=%d' % (cause, entry['causes'][cause]) for cause in CAUSES)))
    out.append('steps with at least one slow span: %.1f%%' % (100 * report['slow_step_share']))
    out += ['', 'PROBE (the timed synchronize after the two quad launches)']
    out.append('%d probes; ' % report['probe']['n'] + '; '.join('%s p50 %.2f p90 %.2f' % (name, report['probe'][name]['p50'], report['probe'][name]['p90'])
                                                                   for name in ('twoq_ms', 'sync_ms', 'launch_ms', 'first_enqueue_ms')))
    out += ['', 'WINDOW (stage 0 lines)']
    out.append('%d windows; window_ms p50 %.2f; fence_wait_ms p50 %.2f; fence wait under 1 ms (the window is the critical path) in %.1f%% of rounds' % (
        report['window']['n'], report['window']['window_ms']['p50'], report['window']['fence_wait_ms']['p50'], 100 * report['window']['critical_share']))
    for count, entry in report['window']['by_blocks'].items():
        out.append('  blocks=%d: %d windows, window_ms p50 %.2f, fence_wait_ms p50 %.2f, critical in %.1f%%' % (count, entry['n'], entry['window_ms']['p50'],
                                                                                                          entry['fence_wait_ms']['p50'], 100 * entry['critical_share']))
    queue = report['fence_queue']
    if queue['n']:
        out += ['', 'WHAT THE FENCE WAITED BEHIND (copies and trace enqueues since the previous synchronization point): %d fences; copies p50 %.0f p90 %.0f; trace enqueues p50 %.0f p90 %.0f' % (
            queue['n'], queue['copies']['p50'], queue['copies']['p90'], queue['traces']['p50'], queue['traces']['p90'])]
    throttle = report['throttle']
    out += ['', 'CPU BANDWIDTH (the container cgroup): %d rounds read, %d of %d periods throttled (%.1f ms), %d rounds with at least one throttle' % (
        throttle['rounds'], throttle['throttled_periods'], throttle['periods'], throttle['throttled_ms'], throttle['rounds_throttled'])]
    blocked = report['blocked']
    if blocked['n']:
        out += ['', 'BLOCKED CALLS (a copy, upload or trace enqueue of 1 ms or more): %d; ms p50 %.2f p90 %.2f, CPU used p50 %.2f ms; %d of them asleep (CPU under a quarter of the time)' % (
            blocked['n'], blocked['ms']['p50'], blocked['ms']['p90'], blocked['cpu_ms']['p50'], blocked['asleep']),
            '  by kind %s; in span %s' % (dict(blocked['by_kind']), dict(blocked['by_span'].most_common(6))),
            '  started after the last trace enqueue: %s; after the last synchronization point: %s; trace enqueues already queued: %s; copies already queued p50 %.0f p90 %.0f' % (
                dict(blocked['since_tnb']), dict(blocked['since_sync']), dict(sorted(blocked['queued_tnb'].items())), blocked['queued_copy']['p50'], blocked['queued_copy']['p90'])]
    threads = report['threads']
    if threads['n']:
        out += ['', 'THREADS (%d censuses): quota %s cpuset %s main thread allowed %s; threads p50 %.0f' % (
            threads['n'], ','.join(threads['quota']), ','.join(threads['cpuset']), ','.join(threads['allowed']), threads['threads']['p50'])]
        for comm, mean, top, inv, tids in threads['busy']:
            out.append('  %-24s mean %5.1f%% of a CPU, most %5.1f%%, %d involuntary switches, %d thread(s)' % (comm, mean, top, inv, tids))
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
