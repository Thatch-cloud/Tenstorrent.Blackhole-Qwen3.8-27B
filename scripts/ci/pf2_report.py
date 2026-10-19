"""Read the device-side hand-off pair of the tp4/prefill-2 window from its container logs (stdlib only).

    python3 -B scripts/ci/pf2_report.py --control <log> [--control <log> ...] --arm <log> [--arm <log> ...] [--json]

A log is a container log of one serving job (a smoke step or a prefix step). The CONTROL is the job of the -pf2-s4-ctl profile (QWEN_FAST_PREFILL_HANDOFF=0: the host path, timed) and the ARM the job of
-pf2-s4 (=1). Two logs of one arm (the A and B of an ABAB) are read as one population. What it reads, per arm:

  '[PF2] handoff mode=<host|device> ms=.. read_ms=.. [clone_ms=..] write_ms=.. layers=..'     the quantity the lever changes, once per final step: the host snapshot of the GDN
                                                                                             scratch plus the slot write (host), the clones plus the slot write (device)
  '[PINDIAG] lever N route req=.. start=.. end=.. prompt=.. final=1 ... ms=..'               the step's own time, so the saving is read against the step it is part of

and prints the hand-off milliseconds (count, mean, median, p90) of each arm, the difference of the means (the saving per final step), the route milliseconds of the final steps by context band
for each arm, and the verdict lines. EXIT 1 when the pair is not a pair: the arm shows no 'engaged' line or any 'fell back' / 'audit mismatch' line, the control shows an 'engaged' line or no host
timing line, or an arm has no final step. Exit 2 on an unreadable log. Nothing here gates the saving itself: one pair is one pair, and the control-to-control floor is the pack's ORDER.txt rule.
"""

import argparse
import json
import re
import sys

HANDOFF = re.compile(r'\[PF2\] handoff mode=(host|device) ms=([\d.]+) read_ms=([\d.]+)(?: clone_ms=([\d.]+))? write_ms=([\d.]+) layers=(\d+)')
ROUTE = re.compile(r'lever N route req=\S+ start=(\d+) end=(\d+) prompt=(\d+) final=(\d) wrote_slot=(\d) ms=([\d.]+)')
ENGAGED = '[PINDIAG] prefill handoff engaged'
FELL_BACK = '[PINDIAG] prefill handoff fell back'
MISMATCH = '[PINDIAG] prefill handoff audit mismatch'
AUDIT = re.compile(r'\[PINDIAG\] prefill handoff audit n=(\d+) exact=(True|False)')
BANDS = ((0, 32768, '0-32k'), (32768, 98304, '32-96k'), (98304, 163840, '96-160k'), (163840, 1 << 30, '160k+'))


def read(paths):
    """One arm's population from its logs: dict(handoff=[(mode, ms, read, clone, write)], routes=[(start, ms)], engaged, fell_back, mismatch, audits=[bool])."""
    arm = dict(handoff=[], routes=[], engaged=0, fell_back=0, mismatch=0, audits=[], modes=set())
    for path in paths:
        with open(path, encoding='utf-8', errors='replace') as handle:
            for line in handle:
                found = HANDOFF.search(line)
                if found:
                    mode, ms, read_ms, clone_ms, write_ms = found.group(1), float(found.group(2)), float(found.group(3)), found.group(4), float(found.group(5))
                    arm['handoff'].append((mode, ms, read_ms, float(clone_ms) if clone_ms is not None else 0.0, write_ms))
                    arm['modes'].add(mode)
                    continue
                route = ROUTE.search(line)
                if route and route.group(4) == '1':
                    arm['routes'].append((int(route.group(1)), float(route.group(6))))
                    continue
                if ENGAGED in line:
                    arm['engaged'] += 1
                elif FELL_BACK in line:
                    arm['fell_back'] += 1
                elif MISMATCH in line:
                    arm['mismatch'] += 1
                else:
                    audit = AUDIT.search(line)
                    if audit:
                        arm['audits'].append(audit.group(2) == 'True')
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
    if not any(mode == 'device' for mode, *_ in arm['handoff']):
        problems.append('the arm timed no device hand-off')
    if control['engaged']:
        problems.append('the control logged an "engaged" line: it ran the lever')
    if 'host' not in control['modes'] or 'device' in control['modes']:
        problems.append('the control must time the host path alone (mode=host lines, none of mode=device)')
    if not arm['handoff'] or not control['handoff']:
        problems.append('an arm has no final step (no "[PF2] handoff" line)')
    host = stats([ms for _mode, ms, *_ in control['handoff']])
    device = stats([ms for _mode, ms, *_ in arm['handoff']])
    saving = round(host['mean'] - device['mean'], 1) if host['mean'] is not None and device['mean'] is not None else None
    parts = dict(read=stats([read for _m, _ms, read, _c, _w in control['handoff']]), write_host=stats([write for _m, _ms, _r, _c, write in control['handoff']]),
                 clone=stats([clone for _m, _ms, _r, clone, _w in arm['handoff']]), write_device=stats([write for _m, _ms, _r, _c, write in arm['handoff']]))
    return dict(control=host, arm=device, saving_ms_per_final_step=saving, saving_fraction=round(saving / host['mean'], 3) if saving is not None and host['mean'] else None,
                parts=parts, routes=dict(control=band_means(control['routes']), arm=band_means(arm['routes'])),
                audits=dict(n=len(arm['audits']), exact=sum(1 for exact in arm['audits'] if exact)), problems=problems)


def render(summary):
    def cell(value):
        return '-' if value is None else '%.1f' % value

    lines = ['hand-off milliseconds per final step: control (host path) n=%d mean %s median %s p90 %s; arm (device) n=%d mean %s median %s p90 %s'
             % (summary['control']['n'], cell(summary['control']['mean']), cell(summary['control']['median']), cell(summary['control']['p90']),
                summary['arm']['n'], cell(summary['arm']['mean']), cell(summary['arm']['median']), cell(summary['arm']['p90'])),
             'saving per final step: %s ms (%s of the control\'s hand-off)' % (cell(summary['saving_ms_per_final_step']),
                                                                             '-' if summary['saving_fraction'] is None else '%.0f%%' % (100 * summary['saving_fraction'])),
             'parts: control read %s + write %s ms; arm clone %s + write %s ms' % (cell(summary['parts']['read']['mean']), cell(summary['parts']['write_host']['mean']),
                                                                              cell(summary['parts']['clone']['mean']), cell(summary['parts']['write_device']['mean'])),
             'route milliseconds of the final steps by context band (control | arm):']
    for _low, _high, name in BANDS:
        control, arm = summary['routes']['control'][name], summary['routes']['arm'][name]
        lines.append('  %-8s n=%3d mean %8s | n=%3d mean %8s' % (name, control['n'], cell(control['mean']), arm['n'], cell(arm['mean'])))
    lines.append('audit lines: %d, exact %d' % (summary['audits']['n'], summary['audits']['exact']))
    lines += ['PROBLEM: ' + problem for problem in summary['problems']]
    lines.append('VERDICT: %s' % ('NOT A PAIR' if summary['problems'] else 'a pair (read the saving against the control-to-control floor)'))
    return '\n'.join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--control', action='append', default=[], required=True)
    parser.add_argument('--arm', action='append', default=[], required=True)
    parser.add_argument('--json', action='store_true')
    arguments = parser.parse_args(argv)
    try:
        control, arm = read(arguments.control), read(arguments.arm)
    except OSError as error:
        sys.stderr.write('unreadable log: %s\n' % error)
        return 2
    summary = summarise(control, arm)
    print(json.dumps(summary, indent=1, sort_keys=True) if arguments.json else render(summary))
    return 1 if summary['problems'] else 0


if __name__ == '__main__':
    sys.exit(main())
