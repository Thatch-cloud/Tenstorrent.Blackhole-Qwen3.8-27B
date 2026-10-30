"""Where the admission time goes, read from a serving container log (tp4/sched-2, docs/tp4-sched2.md). Stdlib only.

    python3 -B scripts/ci/sched2_report.py <container log>                 the per-segment table, returning turns and fresh prompts apart
    python3 -B scripts/ci/sched2_report.py <container log> --second <log>  the same for a second log, side by side (the ABAB reader: control, then the arm)
    python3 -B scripts/ci/sched2_report.py <container log> --burst         the cycle of the first burst of fresh prompts, one line an arrival

The log is the container's own ([PINDIAG] lines, loguru's timestamp). One ADMISSION is the lines of one request id between its KV reservation fit and the engine it is bound to:

    kv reservation fit / dram admission fit      the scheduler's hold (the DRAM allocator reading is taken between them)
    lever N step ... kind=prefill                the prefill step is scheduled
    lever N route ... final=1 ... ms=            the prefill's own clock (a synchronous device call: the host waits in it)
    admission lean backstop reused               the first line after the prefill (the drain: what the sampler waited for after the route's own clock)
    admission lean adopt / rebind phases / split  A the slot adoption and the host checks, B the placement and the window join, C the rebind, with the phases of C
    sticky engine built                          the bind (A + B + C)
    admission seed first deferred / built        tp4/sched-2 (SEED_FIRST): the bind followed the seed; `ms` is the bind that the first token did not wait for (the gain), `after_seed_ms` how long the token had been out when the bind began
    admission hold reading reused                tp4/sched-2 (HOLD_REUSE): a hold that took no allocator reading

THE SERIAL CHAIN between one arrival's prefill and the next arrival's prefill is: hold (the next arrival's) + drain + bind. `bracket` is the part of C that is not the rebind's own
phases (the memory ledger's before and after readings, about 170 to 310 ms and only at a slot's first rebind), `readback` the part of A that is the conv-slice readback (only at a
native slot's first adoption). Every figure prints `-` when no line is behind it.
"""

import argparse
import re
import statistics
import sys
from datetime import datetime

TS = re.compile(r'(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d\.\d{3}) \|')
KV = re.compile(r'kv reservation (?:fit|hold)\b.*?request=(\S+)')
DRAM = re.compile(r'dram admission fit .*?request=(\S+)')
STEP = re.compile(r'lever N step n=\d+ kind=prefill seats=(\d+) req=(\S+) start=(\d+) tokens=(\d+) end=(\d+) prompt=(\d+) final=(\d)')
ROUTE = re.compile(r'lever N route req=(\S+) start=(\d+) end=(\d+) prompt=(\d+) final=(\d) .*?\bms=([\d.]+)')
REUSED = re.compile(r'admission lean backstop reused the admission reading request=(\S+) age_ms=')
ADOPT = re.compile(r'admission lean adopt request=(\S+) slot=(\d+) readback=(\d) ms=([\d.]+)')
PHASES = re.compile(r'admission lean rebind phases request=(\S+) slot=(\d+) fence_ms=([\d.]+) rezero_ms=([\d.]+) project_ms=([\d.]+) copy_ms=([\d.]+) history_ms=([\d.]+) engine_ms=([\d.]+)')
SPLIT = re.compile(r'admission lean split request=(\S+) a_ms=([\d.]+) b_ms=([\d.]+) c_ms=([\d.]+) total_ms=([\d.]+)')
BUILT = re.compile(r'sticky engine built req=(\S+) ms=([\d.]+) frontier=(\d+) prompt=(\d+)(?: kind=(\w+))?')
SEED_DEFERRED = re.compile(r'admission seed first deferred request=(\S+)')
SEED_EAGER = re.compile(r'admission seed first eager request=(\S+) reason=(.*)$')
SEED_BUILT = re.compile(r'admission seed first built request=(\S+) ms=([\d.]+) after_seed_ms=([\d.]+)')
HOLD_REUSED = re.compile(r'admission hold reading reused prompt=(\S+) age_ms=(\d+) reuses=(\d+)')
HOLD_LINE = re.compile(r'dram hold prompt=')
LEDGER_TRIM = re.compile(r'admission ledger trim')
INSTALLED = re.compile(r'admission pipeline installed: seed_first=(\d) hold_reuse=(\d) ledger_trim=(\d)')


def stamp(line):
    match = TS.search(line)
    return None if match is None else datetime.strptime(match.group(1), '%Y-%m-%d %H:%M:%S.%f').timestamp() * 1000.0


def parse(lines):
    """({request id: its admission}, the run's facts). An admission is complete when it has a KV fit, a step, a route, a split and a bind."""
    admissions, facts = {}, dict(installed=None, hold_reused=0, hold_holds=0, ledger_trimmed=False, seed_eager={}, hold_reuse_ages=[])

    def get(request):
        return admissions.setdefault(request, {})

    for line in lines:
        at = stamp(line)
        if at is None:
            if line.startswith('{"stage": "memory_ledger_before"') and '"op": "rebind"' in line:
                facts['bracket_lines'] = facts.get('bracket_lines', 0) + 1
            continue
        match = INSTALLED.search(line)
        if match:
            facts['installed'] = tuple(int(value) for value in match.groups())
            continue
        if LEDGER_TRIM.search(line):
            facts['ledger_trimmed'] = True
            continue
        if HOLD_LINE.search(line):
            facts['hold_holds'] += 1
            continue
        match = HOLD_REUSED.search(line)
        if match:
            facts['hold_reused'] += 1
            facts['hold_reuse_ages'].append(float(match.group(2)))
            continue
        match = KV.search(line)
        if match:
            get(match.group(1))['kv'] = at
            continue
        match = DRAM.search(line)
        if match:
            get(match.group(1))['dram'] = at
            continue
        match = STEP.search(line)
        if match:
            row = get(match.group(2))
            row.update(step=at, seats=int(match.group(1)), start=int(match.group(3)), tokens=int(match.group(4)), prompt=int(match.group(6)))
            continue
        match = ROUTE.search(line)
        if match and match.group(5) == '1':
            row = get(match.group(1))
            row.update(route=at, route_ms=float(match.group(6)))
            continue
        match = REUSED.search(line)
        if match:
            get(match.group(1))['reused'] = at
            continue
        match = ADOPT.search(line)
        if match:
            row = get(match.group(1))
            row.update(adopt_ms=float(match.group(4)), readback=int(match.group(3)), native_slot=int(match.group(2)))
            continue
        match = PHASES.search(line)
        if match:
            get(match.group(1))['phases'] = [float(value) for value in match.groups()[2:]]
            continue
        match = SPLIT.search(line)
        if match:
            row = get(match.group(1))
            row.update(a_ms=float(match.group(2)), b_ms=float(match.group(3)), c_ms=float(match.group(4)), split_at=at)
            continue
        match = BUILT.search(line)
        if match:
            row = get(match.group(1))
            row.update(built=at, built_ms=float(match.group(2)), kind=match.group(5) or 'build')
            continue
        match = SEED_DEFERRED.search(line)
        if match:
            get(match.group(1))['seed_first'] = True
            continue
        match = SEED_EAGER.search(line)
        if match:
            reasons = facts['seed_eager']
            reasons[match.group(2)] = reasons.get(match.group(2), 0) + 1
            continue
        match = SEED_BUILT.search(line)
        if match:
            row = get(match.group(1))
            row.update(seed_built_ms=float(match.group(2)), after_seed_ms=float(match.group(3)))
    complete = dict((request, row) for request, row in admissions.items() if all(key in row for key in ('kv', 'step', 'route', 'split_at', 'built', 'phases')))
    return complete, facts


def derived(row):
    """One admission's segments in ms (None where the line is not in the log)."""
    segments = dict(hold=(row['dram'] if 'dram' in row else row['step']) - row['kv'], prefill=row['route_ms'], drain=(row['reused'] - row['route']) if 'reused' in row else None,
                    a=row['a_ms'], b=row['b_ms'], c=row['c_ms'], built=row['built_ms'])
    phases = row['phases']
    segments.update(fence=phases[0], rezero=phases[1], project=phases[2], copy=phases[3], history=phases[4], engine=phases[5])
    segments['bracket'] = max(0.0, row['c_ms'] - sum(phases))
    segments['readback'] = row['adopt_ms'] if row.get('readback') else None
    # the chain after the prefill's own clock: the drain, then the bind; plus the next arrival's hold is counted with that arrival
    segments['after_prefill'] = (segments['drain'] or 0.0) + row['built_ms']
    segments['serial'] = segments['hold'] + segments['after_prefill']
    segments['seed_gain'] = row.get('seed_built_ms')
    segments['seed_out'] = row.get('after_seed_ms')
    return segments


def quantile(values, fraction):
    values = sorted(values)
    return values[min(len(values) - 1, int(round(fraction * (len(values) - 1))))]


def summarise(values):
    values = [value for value in values if value is not None]
    if not values:
        return None
    return len(values), quantile(values, 0.5), quantile(values, 0.9), statistics.mean(values)


ORDER = (('hold', 'the scheduler\'s hold: kv fit to dram fit (the allocator reading)'), ('prefill', 'the prefill (the route\'s own clock)'),
         ('drain', 'the route\'s clock to the sampler (what was left to wait for)'), ('a', 'A: host checks, backstop, slot adoption'), ('b', 'B: placement, refusal check, window join'),
         ('c', 'C: the rebind (and, at a slot\'s first, the ledger bracket)'), ('rezero', '  C: re-zero'), ('project', '  C: projection'), ('history', '  C: K/V history'),
         ('engine', '  C: engine.rebind (page tables, carry)'), ('bracket', '  C: the ledger bracket (not a phase)'), ('readback', '  A: the conv-slice readback (first adoption of a slot)'),
         ('built', 'the bind (A + B + C)'), ('after_prefill', 'drain + bind'), ('serial', 'hold + drain + bind: the serial chain of one arrival'),
         ('seed_gain', 'seed first: the bind the first token did not wait for (the gain)'), ('seed_out', 'seed first: how long the token had been out when the bind began'))


def table(rows, label):
    lines = ['%-62s %5s %9s %9s %9s' % (label, 'n', 'p50', 'p90', 'mean')]
    segments = [derived(row) for row in rows]
    for key, description in ORDER:
        figures = summarise([segment[key] for segment in segments])
        lines.append('%-62s %5s %9s %9s %9s' % (description, *(('-', '-', '-', '-') if figures is None else (figures[0], '%.1f' % figures[1], '%.1f' % figures[2], '%.1f' % figures[3]))))
    return lines


def facts_lines(facts, rows):
    lines = []
    if facts['installed'] is not None:
        lines.append('pipeline installed: seed_first=%d hold_reuse=%d ledger_trim=%d' % facts['installed'])
    deferred = sum(1 for row in rows if row.get('seed_first'))
    eager = sum(facts['seed_eager'].values())
    if deferred or eager:
        lines.append('seed first: %d deferred, %d bound in the sampler%s' % (deferred, eager, ''.join('; %d x %s' % (count, why) for why, count in sorted(facts['seed_eager'].items()))))
    reads = sum(1 for row in rows if 'dram' in row) - facts['hold_reused']
    if facts['hold_reused']:
        ages = facts['hold_reuse_ages']
        lines.append('hold reuse: %d holds reused a reading (age p50 %.0f ms, max %.0f ms), %d took one' % (facts['hold_reused'], quantile(ages, 0.5), max(ages), max(reads, 0)))
    if facts['ledger_trimmed']:
        lines.append('ledger trim: on, %d rebind brackets in the log' % facts.get('bracket_lines', 0))
    return lines


def burst(rows):
    """One line an arrival for the first run of fresh prompts that arrive within a few seconds of each other."""
    fresh = sorted((row for row in rows if row.get('start', 0) == 0), key=lambda row: row['kv'])
    out = ['%3s %8s %7s %8s %6s %6s %6s %7s %7s %7s %9s' % ('k', 'at ms', 'hold', 'prefill', 'drain', 'A', 'B', 'C', 'bracket', 'readbk', 'gap after')]
    first = fresh[0]['kv'] if fresh else 0.0
    for index, row in enumerate(fresh[:16], 1):
        segment = derived(row)
        following = fresh[index]['kv'] - row['built'] if index < len(fresh) else None
        out.append('%3d %8.0f %7.0f %8.0f %6s %6.0f %6.0f %7.0f %7.0f %7s %9s' % (
            index, row['kv'] - first, segment['hold'], segment['prefill'], '-' if segment['drain'] is None else '%.0f' % segment['drain'], segment['a'], segment['b'], segment['c'], segment['bracket'],
            '-' if segment['readback'] is None else '%.0f' % segment['readback'], '-' if following is None else '%.0f' % following))
    return out


def report(path, second=None, bursts=False):
    out = []
    for index, name in enumerate([path] + ([second] if second else [])):
        with open(name, encoding='utf-8', errors='replace') as handle:
            rows, facts = parse(handle)
        rows = list(rows.values())
        out.append('== %s%s: %d admissions' % ('second ' if index else '', 'log', len(rows)))
        if bursts:
            out += burst(rows)
        else:
            returning = [row for row in rows if row.get('start', 0) > 0]
            fresh = [row for row in rows if row.get('start', 0) == 0]
            out += table(rows, 'all admissions') + [''] + table(returning, 'returning turns (start > 0)') + [''] + table(fresh, 'fresh prompts (start = 0)')
        out += [''] + facts_lines(facts, rows) + ['']
    return '\n'.join(out)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('log')
    parser.add_argument('--second', help='a second log, printed after the first (the arm after the control)')
    parser.add_argument('--burst', action='store_true', help='one line an arrival for the first run of fresh prompts')
    arguments = parser.parse_args(argv)
    print(report(arguments.log, arguments.second, arguments.burst))
    return 0


if __name__ == '__main__':
    sys.exit(main())
