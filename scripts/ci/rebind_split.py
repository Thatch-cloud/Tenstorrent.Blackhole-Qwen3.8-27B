"""Read a c2 serving container log for the parked-engine rebind's split (stdlib only; the RB pack's READ, references/fusion-jobs/RB).

    python3 scripts/ci/rebind_split.py <container or server log> [...]

Per request rebound onto a parked engine ('[PINDIAG] sticky engine built ... kind=rebind'), the bridge build in five segments:
  A  bridge start -> slot adoption line      host checks, the DRAM backstop's reading, the GDN slot adoption
  B  adoption -> ledger 'before op=rebind'   placement, the host refusal check with the FIRST capture.outputs() (the window join), the ledger reading
  C  -> '[PINDIAG] parked rebind'            ParkedEngineSet.rebind_slot (the drafter projection, the K/V seed, the engine's page tables and carry)
  D  -> ledger 'after op=rebind'             the ledger reading
  E  -> the sticky line
and the programs the bridge compiled ('four-card engine programs=A->B'). A rebind is SLOW when B >= 400 ms (the window join compiled kernels);
the lines group by programs compiled. Then the prefill side: each final prefill step's 'window=W' snapshot programs, and for every pair of
steps with the same prompt length and start (a cold twin and its hit, or a repeated turn) the route time of the one that compiled snapshot
programs minus the one that compiled none. And the window lever's own lines (tap_window_aligned): engaged, fell back, audit.
"""

import datetime
import re
import statistics
import sys

TS = re.compile(r'^(\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d\.\d+)Z|(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d\.\d+) \|')
ADOPT = re.compile(r'nothing to adopt for request (\S+)|adopted GDN slot \d+ into slot 0: .* for request (\S+)')
BEFORE = re.compile(r'\[MEMLEDGER\] before op=rebind point=req=(\S+) slot=\d+ chip0 ')
AFTER = re.compile(r'\[MEMLEDGER\] after op=rebind point=req=(\S+) slot=\d+ chip0 ')
REBIND = re.compile(r'\[PINDIAG\] parked rebind req=(\S+) slot=\d+ gen=\d+ .*?ms=([\d.]+)')
ENGINE = re.compile(r'four-card engine programs=(\d+)->(\d+) req=(\S+)')
STICKY = re.compile(r'\[PINDIAG\] sticky engine built req=(\S+) ms=([\d.]+) frontier=\d+ prompt=(\d+) kind=rebind')
ROUTE = re.compile(r'lever N route req=(\S+) start=(\d+) end=\d+ prompt=(\d+) final=1 wrote_slot=\d ms=([\d.]+) programs=\d+->\d+ window=(\d+)')
LEVER = re.compile(r'\[PINDIAG\] tap window aligned (engaged|fell back|audit) (.*)')
SLOW_MS = 400.0
SHORT_ID = 12


def stamp(line):
    match = TS.search(line)
    if not match:
        return None
    text = (match.group(1) or match.group(2)).replace(' ', 'T')
    whole, fraction = text.split('.')
    return datetime.datetime.strptime(whole, '%Y-%m-%dT%H:%M:%S').timestamp() + float('0.' + fraction)


def read(lines):
    """dict(rebinds=[dict(req, ms, prompt, programs, segments=[A..E] ms, rebind_ms, slow)], routes=[(start, prompt, ms, window)],
    lever=dict(engaged=n, fell_back=n, audit_exact=n, audit_mismatch=n))."""
    adopt, before, after, rebind, engine, sticky, routes = {}, {}, {}, {}, {}, [], []
    lever = dict(engaged=0, fell_back=0, audit_exact=0, audit_mismatch=0)
    for line in lines:
        if 'PINDIAG' not in line and 'MEMLEDGER' not in line:
            continue
        match = ADOPT.search(line)
        if match:
            adopt[match.group(1) or match.group(2)] = stamp(line)
            continue
        for pattern, table in ((BEFORE, before), (AFTER, after)):
            match = pattern.search(line)
            if match:
                table.setdefault(match.group(1), stamp(line))
                break
        else:
            match = REBIND.search(line)
            if match:
                rebind[match.group(1)] = (stamp(line), float(match.group(2)))
                continue
            match = ENGINE.search(line)
            if match:
                engine[match.group(3)] = int(match.group(2)) - int(match.group(1))
                continue
            match = STICKY.search(line)
            if match:
                sticky.append((match.group(1), stamp(line), float(match.group(2)), int(match.group(3))))
                continue
            match = ROUTE.search(line)
            if match:
                routes.append((int(match.group(2)), int(match.group(3)), float(match.group(4)), int(match.group(5))))
                continue
            match = LEVER.search(line)
            if match:
                kind, rest = match.groups()
                if kind == 'engaged':
                    lever['engaged'] += 1
                elif kind == 'fell back':
                    lever['fell_back'] += 1
                else:
                    lever['audit_exact' if 'exact=True' in rest else 'audit_mismatch'] += 1
    rebinds = []
    for req, end, ms, prompt in sticky:
        marks = (adopt.get(req), before.get(req[-SHORT_ID:]), rebind.get(req, (None, None))[0], after.get(req[-SHORT_ID:]))
        if None in marks or end is None:
            continue
        began = end - ms / 1000.0
        points = (began,) + marks + (end,)
        segments = [(points[index + 1] - points[index]) * 1000.0 for index in range(5)]
        rebinds.append(dict(req=req, ms=ms, prompt=prompt, programs=engine.get(req), segments=segments,
                            rebind_ms=rebind[req][1], slow=segments[1] >= SLOW_MS))
    return dict(rebinds=rebinds, routes=routes, lever=lever)


def route_pairs(routes):
    """(delta ms, window programs) for each step that compiled snapshot programs and a later one with the same start and prompt that compiled none."""
    pending, pairs = {}, []
    for start, prompt, ms, window in routes:
        key = (start, prompt)
        if window > 0:
            pending[key] = (ms, window)
        elif key in pending:
            first = pending.pop(key)
            pairs.append((first[0] - ms, first[1]))
    return pairs


def group(rebind):
    programs = rebind['programs']
    if programs is None:
        return 'programs ?'
    if programs > 10:
        return 'first of its slot (+%s)' % ('>10')
    return 'programs +%d %s' % (programs, 'SLOW' if rebind['slow'] else 'fast')


def report(result, out=sys.stdout):
    rebinds = result['rebinds']
    out.write('rebinds=%d slow=%d (B >= %.0f ms)\n' % (len(rebinds), sum(1 for r in rebinds if r['slow']), SLOW_MS))
    groups = {}
    for rebind in rebinds:
        groups.setdefault(group(rebind), []).append(rebind)
    for name in sorted(groups):
        members = groups[name]
        columns = list(zip(*[member['segments'] for member in members]))
        medians = [statistics.median(column) for column in columns]
        out.write('  %-26s n=%3d total=%6.0f | A=%4.0f B=%4.0f C=%4.0f D=%3.0f E=%2.0f | B %4.0f-%4.0f | rebind_slot %3.0f\n' % (
            name, len(members), statistics.median(member['ms'] for member in members), *medians,
            min(columns[1]), max(columns[1]), statistics.median(member['rebind_ms'] for member in members)))
    windows = [route[3] for route in result['routes']]
    pairs = route_pairs(result['routes'])
    out.write('final prefill steps=%d with snapshot programs=%d; twin pairs=%d, median delta %s ms, >200 ms in %d\n' % (
        len(windows), sum(1 for window in windows if window), len(pairs),
        '%.0f' % statistics.median(delta for delta, _ in pairs) if pairs else '-', sum(1 for delta, _ in pairs if delta > 200)))
    lever = result['lever']
    out.write('tap window aligned: engaged=%d fell_back=%d audit exact=%d mismatch=%d\n' % (
        lever['engaged'], lever['fell_back'], lever['audit_exact'], lever['audit_mismatch']))


def main(paths):
    for path in paths:
        with open(path, errors='replace') as stream:
            result = read(stream)
        sys.stdout.write('== %s\n' % path)
        report(result)
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
