"""round_probe_report: where the time of a decode round goes, per live count, draft shape and context, from ONE server log (stdlib only, Python 3.7 syntax).

    python3 scripts/ci/round_probe_report.py <container.log> [<container.log> ...] [--json out.json] [--stall-ms 400] [--min-rounds 8]

WHAT IT READS (every line is its producer's own; nothing here is guessed around, and a missing producer is a named GAP in the output, never a zero):
  [PACKED-ROUND-HOST]   round_host.Ledger, one a step: entry, the verify's split (v0 / v0_in / v0_tr / v0_sy / v0_rb, v1* for a second block), the commit enqueue (c0, c1), the early draft
                        (ed and its parts ed_pre, launch, window, fence, collect, flush, select, tail), the step, the gap, live, pos (the live users' mean frontier). Needs
                        QWEN_FAST_TP4_ROUND_HOST_LOG or _LEAN. The wall time of a round is exactly entry + step + ed + gap (closure checked below).
  [OCTO] round=         serving_packed_step, one a step when an octo profile is on: the shape the round RAN (octo | m3 | seq), rows a user, tokens committed, step_ms, gap_ms.
  [PACKED] request=     one a live user a round: the tokens it emitted (min of the accepted prefix and its row cap) and its frontier. These give the acceptance curve of every
                        shape; the sum over a round is checked against the committed count of its [OCTO] line.
  [ROUND-PROBE]         round_probe, only under QWEN_FAST_TP4_ROUND_PROBE=1: every EVERY-th round did two timed synchronizes (drain_ms: the commits still queued when the draft began;
                        launch_ms; sync_ms: the draft still running after its launches). The device time the host logs cannot show.
A record is one step: its [PACKED] lines, then its [OCTO] line, then (its draft) a probe line, then its ledger line, in that order in the file.

WHAT IT DROPS FROM EVERY TIMING TABLE: probe rounds (they are longer by design, and are tabled apart), and LONG rounds (wall above --stall-ms, default 400: a trace build at a shape
switch, a rebind, a prefill chunk in the gap; counted, their time shown, never averaged in). A group is (live, shape the round ran, context of the live users' mean frontier, number of
verify passes: 2 when the live users sat in both M3 blocks). Medians are the headline (a long round or a slow window does not move them); means are printed beside them.

THE PHASES (ms, medians of the group):
  entry    the step's entry (admit, runner update, reservations)
  verify   the device trace the host blocked on (v0_tr; + v1_tr when a second block ran): the verify pass as the host sees it. Its split into matmuls, attention, GDN recurrence,
           norms, collectives and the sampler is NOT in any log: it needs the device profiler (tp4_profile_report); this report says so.
  v.glue   the verify's host work around the trace: input stage, sync, read back (v0 - v0_tr)
  commit   the rest of the step: commit enqueue (c0, c1) and what the step does besides (step - verify blocks)
  draft    the early draft as the host saw it (ed): ed_pre + launch + window + fence + collect + flush + select + tail (parts shown). The window is host work under the device draft.
  gap      the host between the draft's end and the next step's entry
  other    wall - the sum of the above (closure; non-zero only when a field is missing)
Per user: tokens a user commits a round (committed / live), the rate a user (that over the median wall), the slowest user's rate (its mean emitted over the same wall), the aggregate.

EXIT CODE 1 when the log has a problem a reader must not average over (no ledger line, a join that does not close, a probe with no ledger line after it). 0 otherwise.
"""
import argparse
import json
import re
import statistics
import sys

LEDGER = re.compile(r'\[PACKED-ROUND-HOST\] round=(\d+) live=(\S+) pos=(\S+) ((?:\w+=\S+ )+)sel=(\d+) read=(\d+) keyed=(\d+) guard=(\d+)')
OCTO = re.compile(r'\[OCTO\] round=(\d+) shape=(\w+) planned=(\w+) live=(\d+) rows=(\d+) eligible=([01]) counted=([01]) octo_rounds=(\d+) m3_rounds=(\d+) '
                  r'committed=(\d+) step_ms=([0-9.]+) gap_ms=([0-9.]+)')
REQUEST = re.compile(r'\[PACKED\] request=(\S+) segment=(\d+) position=(\d+) prefix=(\d+) emitted=(\d+) predictions=\[[^\]]*\] cap=(\d+)')
PROBE = re.compile(r'\[ROUND-PROBE\] round=(\d+) live=(\d+) shape=(\w+) pairs=(\d+) singles=(\d+) every=(\d+) drain_ms=(\S+) launch_ms=([0-9.]+) sync_ms=(\S+) total_ms=([0-9.]+)')
STAMP = re.compile(r'(\d{4}-\d\d-\d\d) (\d\d):(\d\d):(\d\d)\.(\d{3})')

CTX_EDGES = (8000, 48000, 96000, 160000)
CTX_LABELS = ('<8k', '8-48k', '48-96k', '96-160k', '>=160k')
DRAFT_PARTS = ('ed_pre', 'launch', 'window', 'fence', 'collect', 'flush', 'select', 'tail')


def stamp(line):
    """Seconds since the start of the first day seen (a log never spans more than a few days), or None."""
    match = STAMP.search(line)
    if not match:
        return None
    day, hour, minute, second, milli = match.groups()
    return int(day[8:10]) * 86400 + int(hour) * 3600 + int(minute) * 60 + int(second) + int(milli) / 1000.0


def number(text):
    try:
        return float(text)
    except ValueError:
        return None


def context_label(position):
    if position is None:
        return '?'
    for edge, label in zip(CTX_EDGES, CTX_LABELS):
        if position < edge:
            return label
    return CTX_LABELS[-1]


def parse(lines):
    """-> dict(rounds=[record], probes_unpaired=n, has_octo=bool, ledger_lines=n). A record: dict(t, live, pos, shape, rows, committed, requests, octo, ledger, probe)."""
    text = lines if isinstance(lines, list) else list(lines)
    has_octo = any('[OCTO] round=' in line for line in text)
    rounds, pending, probe, current = [], [], None, None
    ledger_lines, unpaired = 0, 0
    for line in text:
        if '[PACKED] request=' in line:
            match = REQUEST.search(line)
            if match:
                pending.append(dict(request=match.group(1), segment=int(match.group(2)), position=int(match.group(3)), emitted=int(match.group(5)), cap=int(match.group(6))))
        elif '[OCTO] round=' in line:
            match = OCTO.search(line)
            if match:
                group = match.groups()
                current = dict(t=stamp(line), octo=dict(round=int(group[0]), shape=group[1], planned=group[2], live=int(group[3]), rows=int(group[4]), eligible=int(group[5]),
                                                       counted=int(group[6]), committed=int(group[9]), step_ms=float(group[10]), gap_ms=float(group[11])),
                               requests=pending, ledger=None, probe=None)
                rounds.append(current)
                pending = []
        elif '[ROUND-PROBE] round=' in line:
            match = PROBE.search(line)
            if match:
                group = match.groups()
                probe = dict(round=int(group[0]), live=int(group[1]), shape=group[2], pairs=int(group[3]), singles=int(group[4]), every=int(group[5]),
                             drain_ms=number(group[6]), launch_ms=float(group[7]), sync_ms=number(group[8]), total_ms=float(group[9]))
        elif '[PACKED-ROUND-HOST] round=' in line:
            match = LEDGER.search(line)
            if not match:
                continue
            ledger_lines += 1
            fields = {}
            for item in match.group(4).split():
                name, _, value = item.partition('=')
                fields[name] = number(value)
            ledger = dict(round=int(match.group(1)), live=None if match.group(2) == '-' else int(match.group(2)),
                          pos=None if match.group(3) == '-' else float(match.group(3)), fields=fields)
            if has_octo:
                if current is not None and current['ledger'] is None:
                    target = current
                else:
                    target = dict(t=stamp(line), octo=None, requests=[], ledger=None, probe=None)
                    rounds.append(target)
            else:
                target = dict(t=stamp(line), octo=None, requests=pending, ledger=None, probe=None)
                pending = []
                rounds.append(target)
            target['ledger'] = ledger
            target['probe'] = probe
            if probe is not None:
                probe = None
            current = None
    if probe is not None:
        unpaired += 1
    for record in rounds:
        octo, ledger = record['octo'], record['ledger']
        record['live'] = (octo or {}).get('live', (ledger or {}).get('live'))
        record['pos'] = (ledger or {}).get('pos')
        record['shape'] = (octo or {}).get('shape', '?')
        record['rows'] = (octo or {}).get('rows')
        record['committed'] = (octo or {}).get('committed')
    return dict(rounds=rounds, probes_unpaired=unpaired, has_octo=has_octo, ledger_lines=ledger_lines)


def wall_ms(ledger):
    """entry + step + ed + gap of one ledger line (a missing field counts 0); None when the line has no step."""
    fields = ledger['fields']
    if fields.get('step') is None:
        return None
    return sum(fields.get(name) or 0.0 for name in ('entry', 'step', 'ed', 'gap'))


def problems_of(parsed, tolerance=0.01):
    problems = []
    rounds = parsed['rounds']
    if not parsed['ledger_lines']:
        problems.append('no [PACKED-ROUND-HOST] line: the ledger is off (QWEN_FAST_TP4_ROUND_HOST_LOG or _LEAN), so no phase can be read')
    if parsed['probes_unpaired']:
        problems.append('%d [ROUND-PROBE] line(s) with no ledger line after them' % parsed['probes_unpaired'])
    both = [r for r in rounds if r['octo'] is not None and r['ledger'] is not None]
    if parsed['has_octo'] and parsed['ledger_lines']:
        if len(both) < 0.99 * len(rounds):
            problems.append('%d of %d steps lack their [OCTO] or ledger line: the join by order does not close' % (len(rounds) - len(both), len(rounds)))
        wrong = [r for r in both if r['octo']['live'] != r['ledger']['live']]
        if len(wrong) > tolerance * max(1, len(both)):
            problems.append('%d of %d joined steps disagree on the live count' % (len(wrong), len(both)))
    summed = [r for r in rounds if r['octo'] is not None and r['requests']]
    off = [r for r in summed if sum(q['emitted'] for q in r['requests']) != r['octo']['committed']]
    if summed and len(off) > tolerance * len(summed):
        problems.append('%d of %d steps: the [PACKED] emitted counts do not sum to the [OCTO] committed count' % (len(off), len(summed)))
    return problems


def percentile(values, share):
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(share * len(ordered)))]


def median_of(records, getter):
    values = [v for v in (getter(r) for r in records) if v is not None]
    return statistics.median(values) if values else None


def group_table(parsed, stall_ms=400.0, min_rounds=8):
    """{(live, shape, ctx, blocks): stats} over the steady rounds (no probe round, no long round, a ledger line with a step), plus the census of the long rounds.
    `blocks` is the number of verify passes the step ran (2 when the live users sat in both M3 blocks: the ledger has a second verify)."""
    steady, stalls, probes = {}, {'rounds': 0, 'ms': 0.0, 'total_ms': 0.0}, 0
    for record in parsed['rounds']:
        ledger = record['ledger']
        if ledger is None:
            continue
        wall = wall_ms(ledger)
        if wall is None:
            continue
        stalls['total_ms'] += wall
        if record['probe'] is not None:
            probes += 1
            continue
        if wall > stall_ms:
            stalls['rounds'] += 1
            stalls['ms'] += wall
            continue
        blocks = 2 if ledger['fields'].get('v1') is not None else 1
        key = (record['live'], record['shape'], context_label(record['pos']), blocks)
        steady.setdefault(key, []).append(record)
    table = {}
    for key, records in steady.items():
        if len(records) < min_rounds:
            continue
        walls = [wall_ms(r['ledger']) for r in records]
        wall = statistics.median(walls)

        def field(name, records=records):
            return median_of(records, lambda r: r['ledger']['fields'].get(name))

        verify = median_of(records, lambda r: (r['ledger']['fields'].get('v0_tr') or 0.0) + (r['ledger']['fields'].get('v1_tr') or 0.0)
                           if r['ledger']['fields'].get('v0_tr') is not None else None)
        v_total = median_of(records, lambda r: (r['ledger']['fields'].get('v0') or 0.0) + (r['ledger']['fields'].get('v1') or 0.0)
                            if r['ledger']['fields'].get('v0') is not None else None)
        step = field('step')
        entry, ed, gap = field('entry'), field('ed'), field('gap')
        glue = None if verify is None or v_total is None else v_total - verify
        commit = None if step is None or v_total is None else step - v_total
        parts = {name: field(name) for name in DRAFT_PARTS}
        accounted = sum(v or 0.0 for v in (entry, verify, glue, commit, ed, gap))
        committed = [r['committed'] for r in records if r['committed'] is not None]
        live = key[0] or 1
        per_request = {}
        for record in records:
            for q in record['requests']:
                per_request.setdefault(q['request'], []).append(q['emitted'])
        slowest = min((statistics.mean(v) for v in per_request.values() if len(v) >= 20), default=None)
        mean_committed = statistics.mean(committed) if committed else None
        table[key] = dict(
            n=len(records), wall=wall, wall_mean=statistics.mean(walls), wall_p90=percentile(walls, 0.9), entry=entry, verify=verify, verify_glue=glue, commit=commit, draft=ed,
            draft_parts=parts, gap=gap, other=wall - accounted,
            tokens_user_round=None if mean_committed is None else mean_committed / live, rows=records[0]['rows'], blocks=key[3],
            user_tok_s=None if mean_committed is None else mean_committed / live / wall * 1000.0,
            aggregate_tok_s=None if mean_committed is None else mean_committed / wall * 1000.0,
            slowest_user_tok_s=None if slowest is None else slowest / wall * 1000.0, positions=median_of(records, lambda r: r['pos']))
    return table, stalls, probes


def probe_table(parsed, min_rounds=3):
    """{(live, shape): medians of drain / launch / sync / total, and the probe rounds' own [ledger] verify trace} over the probe lines."""
    groups = {}
    for record in parsed['rounds']:
        probe = record['probe']
        if probe is None or record['ledger'] is None:
            continue
        groups.setdefault((probe['live'], probe['shape'], context_label(record['pos'])), []).append(record)
    table = {}
    for key, records in groups.items():
        if len(records) < min_rounds:
            continue
        table[key] = dict(
            n=len(records), drain=median_of(records, lambda r: r['probe']['drain_ms']), launch=median_of(records, lambda r: r['probe']['launch_ms']),
            sync=median_of(records, lambda r: r['probe']['sync_ms']), total=median_of(records, lambda r: r['probe']['total_ms']),
            verify=median_of(records, lambda r: (r['ledger']['fields'].get('v0_tr') or 0.0) + (r['ledger']['fields'].get('v1_tr') or 0.0)))
    return table


def survival(parsed, stall_ms=400.0, min_requests=40):
    """{(live, shape): (mean emitted, [P(e >= k) for k = 1..rows])} over the requests that ran at the shape's full row cap (a budget-narrowed cap is not a sample of the shape)."""
    groups = {}
    for record in parsed['rounds']:
        if record['probe'] is not None or record['octo'] is None or not record['requests']:
            continue
        wall = wall_ms(record['ledger']) if record['ledger'] else None
        if wall is not None and wall > stall_ms:
            continue
        rows = record['rows']
        for q in record['requests']:
            if q['cap'] == rows:
                groups.setdefault((record['live'], record['shape'], rows), []).append(q['emitted'])
    table = {}
    for key, values in groups.items():
        if len(values) < min_requests:
            continue
        rows = key[2]
        table[key] = (statistics.mean(values), [sum(1 for v in values if v >= k) / float(len(values)) for k in range(1, rows + 1)], len(values))
    return table


def fmt(value, width=7, places=1):
    return ('%*s' % (width, '-')) if value is None else ('%*.*f' % (width, places, value))


def render(parsed, stall_ms, min_rounds):
    out = []
    table, stalls, probes = group_table(parsed, stall_ms, min_rounds)
    out.append('ROUNDS %d  ledger lines %d  probe rounds %d  long rounds (> %g ms: a prefill, a trace build or a rebind in the gap) %d = %.1f s of %.1f s ledger wall'
               % (len(parsed['rounds']), parsed['ledger_lines'], probes, stall_ms, stalls['rounds'], stalls['ms'] / 1000.0, stalls['total_ms'] / 1000.0))
    out.append('')
    out.append('PER ROUND (ms, medians of steady rounds; wall = entry + verify + v.glue + commit + draft + gap)')
    head = ' live shape  ctx      blk rows     n   wall  (mean)   p90  entry verify v.glue commit  draft    gap  other | tok/user user t/s  slowest  aggreg'
    out.append(head)
    for key in sorted(table, key=lambda k: (k[0] if k[0] is not None else -1, k[1], CTX_LABELS.index(k[2]) if k[2] in CTX_LABELS else 9, k[3])):
        row = table[key]
        out.append('%5s %-5s %-8s %3d %4s %5d %s %s %s %s %s %s %s %s %s %s | %s %s %s %s' % (
            key[0], key[1], key[2], key[3], row['rows'], row['n'], fmt(row['wall'], 6), fmt(row['wall_mean'], 7), fmt(row['wall_p90'], 5), fmt(row['entry'], 5), fmt(row['verify'], 6),
            fmt(row['verify_glue'], 6), fmt(row['commit'], 6), fmt(row['draft'], 6), fmt(row['gap'], 6), fmt(row['other'], 6),
            fmt(row['tokens_user_round'], 8, 2), fmt(row['user_tok_s'], 8), fmt(row['slowest_user_tok_s'], 8), fmt(row['aggregate_tok_s'], 7, 0)))
    out.append('')
    out.append('THE DRAFT, BY PART (ms, medians): ed_pre launch window fence collect flush select tail')
    for key in sorted(table, key=lambda k: (k[0] if k[0] is not None else -1, k[1], k[2], k[3])):
        parts = table[key]['draft_parts']
        out.append('%5s %-5s %-8s %3d %s' % (key[0], key[1], key[2], key[3], ' '.join(fmt(parts[name], 7) for name in DRAFT_PARTS)))
    probe_rows = probe_table(parsed)
    out.append('')
    if probe_rows:
        out.append('PROBE ROUNDS (ms, medians; QWEN_FAST_TP4_ROUND_PROBE): drain = commits still queued at the draft start; launch = host launch time; sync = draft still running after the launches')
        out.append(' live shape  ctx          n  drain launch   sync  verify trace')
        for key in sorted(probe_rows, key=lambda k: (k[0], k[1], k[2])):
            row = probe_rows[key]
            out.append('%5s %-6s %-8s %4d %s %s %s %s' % (key[0], key[1], key[2], row['n'], fmt(row['drain'], 6), fmt(row['launch'], 6), fmt(row['sync'], 6), fmt(row['verify'], 7)))
    else:
        out.append('PROBE ROUNDS: none in this log. GAP: the device time of the commits and of the draft is not measured (the ledger shows only what the host waited for).')
    out.append('GAP: the verify pass is not split into matmul / attention / GDN recurrence / collectives / sampler by any log; that is the device profiler\'s (tp4_profile_report).')
    curve = survival(parsed, stall_ms)
    out.append('')
    out.append('ACCEPTANCE (P(a user commits >= k tokens in a round), requests at the shape\'s full row cap)')
    for key in sorted(curve, key=lambda k: (k[0] if k[0] is not None else -1, k[1])):
        mean, shares, count = curve[key]
        out.append('%5s live %-5s rows %2d  n=%5d  mean %.2f  %s' % (key[0], key[1], key[2], count, mean, ' '.join('%.2f' % share for share in shares)))
    return '\n'.join(out), table, stalls, probe_rows, curve


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('logs', nargs='+')
    parser.add_argument('--json')
    parser.add_argument('--stall-ms', type=float, default=400.0)
    parser.add_argument('--min-rounds', type=int, default=8)
    arguments = parser.parse_args(argv)
    lines = []
    for path in arguments.logs:
        with open(path, encoding='utf-8', errors='replace') as handle:
            lines.extend(handle.read().splitlines())
    parsed = parse(lines)
    text, table, stalls, probe_rows, curve = render(parsed, arguments.stall_ms, arguments.min_rounds)
    problems = problems_of(parsed)
    print(text)
    for problem in problems:
        print('PROBLEM: ' + problem)
    if arguments.json:
        def keyed(mapping):
            return {'|'.join(str(part) for part in key): value for key, value in mapping.items()}

        with open(arguments.json, 'w', encoding='utf-8', newline='\n') as handle:
            json.dump(dict(groups=keyed(table), stalls=stalls, probes=keyed(probe_rows), acceptance=keyed({k: dict(mean=v[0], survival=v[1], n=v[2]) for k, v in curve.items()}),
                           problems=problems), handle, indent=1, sort_keys=True)
            handle.write('\n')
    return 1 if problems else 0


if __name__ == '__main__':
    sys.exit(main())
