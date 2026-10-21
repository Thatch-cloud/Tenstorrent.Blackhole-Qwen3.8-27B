"""Read the traced-tail pair of the tp4/prefill-3 window from its container logs, and the trace-region headroom of any attach (stdlib only).

    python3 -B scripts/ci/pf3_report.py --control <log> [--control <log> ...] --arm <log> [--arm <log> ...] [--json]
    python3 -B scripts/ci/pf3_report.py --headroom <log> [--headroom <log> ...] [--traces N] [--ops N] [--bytes-per-op LOW,HIGH] [--json]

PAIR. A log is a container log of one serving job. The CONTROL is the job of the -pf3-s5-ctl profile (QWEN_FAST_PREFILL_TAIL_TRACE=0: the eager tail, timed) and the ARM the job of -pf3-s5 (=1).
Two logs of one arm (the A and B of an ABAB) are read as one population. What it reads, per arm:

  '[PF3] tail mode=<eager|traced> bucket=.. valid=.. start=.. ms=.. [stage_ms=.. replay_ms=..]'   the quantity the lever changes, once per tail pass: the whole masked forward as the
                                                                                                  caller sees it, to the device sync that follows it
  '[PINDIAG] lever N route req=.. start=.. end=.. prompt=.. final=1 ... ms=..'                    the final step's own time, so the saving is read against the step it is part of
  '[PINDIAG] prefill tail trace engaged ... arena_mb=.. arena_address=..'                          once, after the warm
  '[MEMLEDGER] phase=P7 ... trace_used=..'                                                         the trace region after the attach (the lever's own traces are in it)

and prints the tail milliseconds (count, mean, median, p90) per bucket for each arm, the difference of the means, the final-step route milliseconds by context band, the trace region the
traces took (arm minus control, after the attach), and the verdict lines. EXIT 1 when the pair is not a pair: the arm shows no 'engaged' line, any 'fell back' or 'audit mismatch' line, no
traced tail; the control shows an 'engaged' line, a traced tail, or no eager timing line. Exit 2 on an unreadable log. Nothing here gates the saving itself: one pair is one pair, and the
control-to-control floor is the pack's rule.

HEADROOM. The '[MEMLEDGER]' trace lines of a log (the phases P7, P12 and P13 and every 'after op=' point), per chip: the region's size, the most it held, the least free it ever had; and, for
`--traces N` traces of `--ops` ops each at `--bytes-per-op LOW,HIGH` bytes an op (the ESTIMATE of docs/tp4-prefill3.md section 3, to be replaced by the arm-minus-control figure of a
PAIR), whether they fit the least free the run showed. Exit 1 when the high estimate does not fit.
"""

import argparse
import json
import re
import sys

TAIL = re.compile(r'\[PF3\] tail mode=(eager|traced|audited) bucket=(\d+) valid=(\d+) start=(\d+) ms=([\d.]+)(?: stage_ms=([\d.]+) replay_ms=([\d.]+))?')
ROUTE = re.compile(r'lever N route req=\S+ start=(\d+) end=(\d+) prompt=(\d+) final=(\d) wrote_slot=(\d) ms=([\d.]+)')
ENGAGED = '[PINDIAG] prefill tail trace engaged'
FELL_BACK = '[PINDIAG] prefill tail trace fell back'
MISMATCH = '[PINDIAG] prefill tail trace audit mismatch'
AUDIT = re.compile(r'\[PINDIAG\] prefill tail trace audit n=(\d+) exact=(True|False)')
ARENA = re.compile(r'arena_mb=(\d+)')
LEDGER_TRACE = re.compile(r'\[MEMLEDGER\] (?:phase=(P\d+)|(?:after|before) op=(\S+))[^\n]*? chip(\d+)[^\n]*?trace_used=([\d.]+)MB(?: trace_free=([\d.]+)MB)? ?(?:trace_largest_free=([\d.]+)MB)?')
LEDGER_TOTAL = re.compile(r'\[MEMLEDGER\] phase=P\d+[^\n]*? chip(\d+) trace_total=([\d.]+)MB trace_used=([\d.]+)MB trace_free=([\d.]+)MB')
LEDGER_DRAM = re.compile(r'\[MEMLEDGER\] phase=P7 point=after_attach chip(\d+) allocated=([\d.]+)GB free=([\d.]+)GB')
BANDS = ((0, 32768, '0-32k'), (32768, 98304, '32-96k'), (98304, 163840, '96-160k'), (163840, 1 << 30, '160k+'))


def read(paths):
    """One arm's population from its logs."""
    arm = dict(tails=[], routes=[], engaged=0, fell_back=0, mismatch=0, audits=[], modes=set(), arena=[], trace_after_attach=[], dram_free_after_attach=[])
    for path in paths:
        with open(path, encoding='utf-8', errors='replace') as handle:
            for line in handle:
                found = TAIL.search(line)
                if found:
                    mode, bucket, valid, start, ms = found.group(1), int(found.group(2)), int(found.group(3)), int(found.group(4)), float(found.group(5))
                    arm['tails'].append((mode, bucket, valid, start, ms))
                    arm['modes'].add(mode)
                    continue
                route = ROUTE.search(line)
                if route and route.group(4) == '1':
                    arm['routes'].append((int(route.group(1)), float(route.group(6))))
                    continue
                if ENGAGED in line:
                    arm['engaged'] += 1
                    arena = ARENA.search(line)
                    if arena:
                        arm['arena'].append(int(arena.group(1)))
                elif FELL_BACK in line:
                    arm['fell_back'] += 1
                elif MISMATCH in line:
                    arm['mismatch'] += 1
                else:
                    audit = AUDIT.search(line)
                    if audit:
                        arm['audits'].append(audit.group(2) == 'True')
                        continue
                    ledger = LEDGER_TRACE.search(line)
                    if ledger and ledger.group(1) == 'P7' and ledger.group(3) == '0':
                        arm['trace_after_attach'].append(float(ledger.group(4)))
                    dram = LEDGER_DRAM.search(line)
                    if dram and dram.group(1) == '0':
                        arm['dram_free_after_attach'].append(float(dram.group(3)))
    return arm


def quantile(values, fraction):
    ordered = sorted(values)
    if not ordered:
        return None
    return ordered[min(len(ordered) - 1, int(round(fraction * (len(ordered) - 1))))]


def stats(values):
    if not values:
        return dict(n=0, mean=None, median=None, p90=None)
    return dict(n=len(values), mean=round(sum(values) / len(values), 1), median=round(quantile(values, 0.5), 1), p90=round(quantile(values, 0.9), 1))


def by_bucket(tails, modes):
    out = {}
    for mode, bucket, _valid, _start, ms in tails:
        if mode in modes:
            out.setdefault(bucket, []).append(ms)
    return dict((bucket, stats(values)) for bucket, values in sorted(out.items()))


def band_means(routes):
    out = {}
    for low, high, name in BANDS:
        values = [ms for start, ms in routes if low <= start < high]
        out[name] = dict(n=len(values), mean=round(sum(values) / len(values), 1) if values else None)
    return out


def summarise(control, arm):
    """The numbers and the problems of a pair."""
    problems = []
    if arm['engaged'] < 1:
        problems.append('the arm logged no "%s" line: the lever was mounted and never executed' % ENGAGED)
    if arm['fell_back']:
        problems.append('the arm logged %d "fell back" line(s)' % arm['fell_back'])
    if arm['mismatch'] or any(not exact for exact in arm['audits']):
        problems.append('the arm logged an audit mismatch')
    if 'traced' not in arm['modes']:
        problems.append('the arm timed no traced tail')
    if control['engaged']:
        problems.append('the control logged an "engaged" line: it ran the lever')
    if 'eager' not in control['modes'] or 'traced' in control['modes']:
        problems.append('the control must time the eager tail alone (mode=eager lines, none of mode=traced)')
    if not arm['tails'] or not control['tails']:
        problems.append('an arm has no tail pass (no "[PF3] tail" line)')
    eager = by_bucket(control['tails'], ('eager',))
    traced = by_bucket(arm['tails'], ('traced',))
    savings = {}
    for bucket in sorted(set(eager) & set(traced)):
        before, after = eager[bucket]['mean'], traced[bucket]['mean']
        savings[bucket] = dict(saving_ms=round(before - after, 1), fraction=round((before - after) / before, 3) if before else None)
    attach = None
    if control['trace_after_attach'] and arm['trace_after_attach']:
        attach = round(arm['trace_after_attach'][-1] - control['trace_after_attach'][-1], 1)
    return dict(control=eager, arm=traced, savings=savings, routes=dict(control=band_means(control['routes']), arm=band_means(arm['routes'])),
                audits=dict(n=len(arm['audits']), exact=sum(1 for exact in arm['audits'] if exact)), arena_mb=arm['arena'][-1] if arm['arena'] else None,
                trace_region_taken_mb=attach,
                dram_free_after_attach_mb_delta=(round((arm['dram_free_after_attach'][-1] - control['dram_free_after_attach'][-1]) * 1000.0, 0)
                                                 if control['dram_free_after_attach'] and arm['dram_free_after_attach'] else None),
                problems=problems)


def render(summary):
    def cell(value):
        return '-' if value is None else '%.1f' % value

    lines = ['tail milliseconds per bucket: control (eager) | arm (traced)']
    for bucket in sorted(set(summary['control']) | set(summary['arm'])):
        control, arm = summary['control'].get(bucket, dict(n=0, mean=None, median=None, p90=None)), summary['arm'].get(bucket, dict(n=0, mean=None, median=None, p90=None))
        saving = summary['savings'].get(bucket)
        lines.append('  %5d  n=%3d mean %8s median %8s p90 %8s | n=%3d mean %8s median %8s p90 %8s%s' % (
            bucket, control['n'], cell(control['mean']), cell(control['median']), cell(control['p90']), arm['n'], cell(arm['mean']), cell(arm['median']), cell(arm['p90']),
            '' if saving is None else ' | saves %s ms (%s)' % (cell(saving['saving_ms']), '-' if saving['fraction'] is None else '%.0f%%' % (100 * saving['fraction']))))
    lines.append('route milliseconds of the final steps by context band (control | arm):')
    for _low, _high, name in BANDS:
        control, arm = summary['routes']['control'][name], summary['routes']['arm'][name]
        lines.append('  %-8s n=%3d mean %8s | n=%3d mean %8s' % (name, control['n'], cell(control['mean']), arm['n'], cell(arm['mean'])))
    lines.append('trace region the lever took after the attach (arm minus control, chip 0): %s MB; arena %s MB; free DRAM after the attach, arm minus control: %s MB' % (
        cell(summary['trace_region_taken_mb']), '-' if summary['arena_mb'] is None else summary['arena_mb'], cell(summary['dram_free_after_attach_mb_delta'])))
    lines.append('audit lines: %d, exact %d' % (summary['audits']['n'], summary['audits']['exact']))
    lines += ['PROBLEM: ' + problem for problem in summary['problems']]
    lines.append('VERDICT: %s' % ('NOT A PAIR' if summary['problems'] else 'a pair (read the saving against the control-to-control floor)'))
    return '\n'.join(lines)


def headroom(paths):
    """Per chip: dict(total, used_max, used_min, free_min) in MB, from every trace line of the logs."""
    chips = {}
    for path in paths:
        with open(path, encoding='utf-8', errors='replace') as handle:
            for line in handle:
                total = LEDGER_TOTAL.search(line)
                if total:
                    chip = chips.setdefault(int(total.group(1)), dict(total=None, used=[], free=[]))
                    chip['total'] = float(total.group(2))
                    chip['used'].append(float(total.group(3)))
                    chip['free'].append(float(total.group(4)))
                    continue
                found = LEDGER_TRACE.search(line)
                if found:
                    chip = chips.setdefault(int(found.group(3)), dict(total=None, used=[], free=[]))
                    chip['used'].append(float(found.group(4)))
                    if found.group(6):
                        chip['free'].append(float(found.group(6)))
    return dict((index, dict(total=chip['total'], used_max=max(chip['used']) if chip['used'] else None, used_min=min(chip['used']) if chip['used'] else None,
                             free_min=min(chip['free']) if chip['free'] else None)) for index, chip in sorted(chips.items()))


def fit(chips, traces, ops, low, high):
    """(low need MB, high need MB, least free MB over the chips, verdict) for `traces` traces of `ops` ops at low..high bytes an op."""
    need_low, need_high = traces * ops * low / 1e6, traces * ops * high / 1e6
    free = min((chip['free_min'] for chip in chips.values() if chip['free_min'] is not None), default=None)
    if free is None:
        return need_low, need_high, None, 'UNKNOWN (no trace lines in the logs: was the memory ledger on?)'
    if need_high <= free:
        return need_low, need_high, free, 'FITS at both ends'
    if need_low <= free:
        return need_low, need_high, free, 'FITS at the low estimate only'
    return need_low, need_high, free, 'DOES NOT FIT'


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--control', action='append', default=[])
    parser.add_argument('--arm', action='append', default=[])
    parser.add_argument('--headroom', action='append', default=[])
    parser.add_argument('--traces', type=int, default=4)
    parser.add_argument('--ops', type=int, default=3400)
    parser.add_argument('--bytes-per-op', default='6000,16000')
    parser.add_argument('--json', action='store_true')
    arguments = parser.parse_args(argv)
    if bool(arguments.headroom) == bool(arguments.control or arguments.arm):
        parser.error('give --headroom logs, or --control and --arm logs')
    try:
        if arguments.headroom:
            low, high = (int(value) for value in arguments.bytes_per_op.split(','))
            chips = headroom(arguments.headroom)
            need_low, need_high, free, verdict = fit(chips, arguments.traces, arguments.ops, low, high)
            result = dict(chips=chips, traces=arguments.traces, ops=arguments.ops, need_mb=[round(need_low, 1), round(need_high, 1)], least_free_mb=free, verdict=verdict)
            if arguments.json:
                print(json.dumps(result, indent=1, sort_keys=True))
            else:
                for index, chip in chips.items():
                    print('chip %d: region %s MB, used %s..%s MB, least free %s MB' % (index, chip['total'], chip['used_min'], chip['used_max'], chip['free_min']))
                print('%d traces of %d ops at %d..%d bytes an op: %.1f..%.1f MB against %s MB free: %s' % (arguments.traces, arguments.ops, low, high, need_low, need_high, free, verdict))
            return 0 if verdict.startswith('FITS at both') else 1
        if not arguments.control or not arguments.arm:
            parser.error('--control and --arm both')
        control, arm = read(arguments.control), read(arguments.arm)
    except OSError as error:
        sys.stderr.write('unreadable log: %s\n' % error)
        return 2
    summary = summarise(control, arm)
    print(json.dumps(summary, indent=1, sort_keys=True) if arguments.json else render(summary))
    return 1 if summary['problems'] else 0


if __name__ == '__main__':
    sys.exit(main())
