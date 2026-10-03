"""lookup_tau_report: the committed tokens per round (tau) of a timed window arm, and of a prompt-lookup arm by proposal source.

  python lookup_tau_report.py A_RESULTS [B_RESULTS] [--live 4] [--json OUT]

Each argument is a results tree (any depth) holding container.log or server.log files, or one log file. Stdlib only; python 3.7.

Two sources, one number each:
  [PACKED] request=<id> ... emitted=N        every packed round of every user (QWEN_FAST_PACKED_AUDIT lines; the live count
                                             comes from the [PACKED-PHASE] line before them). tau = the mean of N over the
                                             rounds with live == --live that are not their request's last (the last is cut by
                                             the end of the answer, not by a mismatch), as tau_lab_report counts it.
  [LOOKUP-ROUND] request=<id> position=.. source=lookup|dflash2 match=.. offered=.. proposed=.. committed=N
                                             one line per user per round of a QWEN_FAST_LOOKUP_DRAFT arm, logged when the
                                             round's commit is known. Split by source: how often the lookup was used, and what
                                             a lookup round and a DFlash2 round committed. It cannot say what the other source
                                             would have committed on the same round; the paired arms do (A without the flag,
                                             B with it, same admission order, same prompts).

Read the pair per round and per user, never by unpaired medians: the arms see different round boundaries once a proposal differs,
so the comparison is tau and the paired round time together (speed_window_compare for the texts, c2_smoke_check for the log).
tau x 1000 / round_ms is a user's tok/s only with BOTH from the same arm.
"""
import argparse
import json
import os
import re
import sys

PACKED = re.compile(r'\[PACKED\] request=(\S+) segment=(\d+) position=(\d+) prefix=(\d+) emitted=(\d+)')
PHASE = re.compile(r'\[PACKED-PHASE\] round=(\d+) users=(\d+)[^\n]*?\blive=(\d+)')
LOOKUP = re.compile(r'\[LOOKUP-ROUND\] request=(\S+) position=(\d+) source=(lookup|dflash2) match=(\d+) offered=(\d+) proposed=(\d+) committed=(\d+)')


def logs_under(path):
    if os.path.isfile(path):
        return [path]
    found = []
    for root, _, names in os.walk(path):
        for name in names:
            if name in ('container.log', 'server.log'):
                found.append(os.path.join(root, name))
    return sorted(found)


def parse(paths, live_wanted):
    """-> {packed: {request: [(position, emitted)]} for live rounds, lookup: [(request, position, source, match, offered, proposed, committed)]}"""
    packed, lookup = {}, []
    for path in paths:
        live = None
        with open(path, encoding='utf-8', errors='replace') as handle:
            for line in handle:
                if '[PACKED-PHASE]' in line:
                    match = PHASE.search(line)
                    if match:
                        live = int(match.group(3))
                    continue
                if '[PACKED]' in line:
                    match = PACKED.search(line)
                    if match and live == live_wanted:
                        packed.setdefault(match.group(1), []).append((int(match.group(3)), int(match.group(5))))
                elif '[LOOKUP-ROUND]' in line:
                    match = LOOKUP.search(line)
                    if match:
                        request, position, source, m, offered, proposed, committed = match.groups()
                        lookup.append((request, int(position), source, int(m), int(offered), int(proposed), int(committed)))
    return packed, lookup


def mean(values):
    values = list(values)
    return sum(values) / float(len(values)) if values else float('nan')


def summarize(paths, live_wanted):
    packed, lookup = parse(paths, live_wanted)
    emitted = [e for rounds in packed.values() for _, e in rounds[:-1] if e > 0]
    result = dict(logs=len(paths), live=live_wanted, packed_rounds=len(emitted), tau=mean(emitted),
                  requests=len(packed))
    if lookup:
        by = {}
        for request, position, source, match, offered, proposed, committed in lookup:
            by.setdefault(source, []).append((match, offered, proposed, committed))
        rounds = len(lookup)
        result['lookup_rounds_logged'] = rounds
        result['lookup_tau_all_logged'] = mean(row[6] for row in lookup)
        for source in ('lookup', 'dflash2'):
            rows = by.get(source, [])
            result[source] = dict(rounds=len(rows), share=len(rows) / float(rounds), tau=mean(r[3] for r in rows),
                                  mean_match=mean(r[0] for r in rows), mean_proposed=mean(r[2] for r in rows))
    return result


def render(label, result):
    lines = ['%s: %d log(s), %d request(s)' % (label, result['logs'], result['requests']),
             '  packed rounds at live=%d (last round of each request excluded): %d, tau %.3f' % (
                 result['live'], result['packed_rounds'], result['tau'])]
    if 'lookup' in result:
        lines.append('  [LOOKUP-ROUND] lines %d, tau over all of them %.3f' % (result['lookup_rounds_logged'], result['lookup_tau_all_logged']))
        for source in ('lookup', 'dflash2'):
            row = result[source]
            lines.append('    source %-8s rounds %6d (%5.1f%%)  tau %.3f  mean match %.1f  mean proposed %.1f' % (
                source, row['rounds'], 100.0 * row['share'], row['tau'], row['mean_match'], row['mean_proposed']))
    return lines


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('first')
    parser.add_argument('second', nargs='?')
    parser.add_argument('--live', type=int, default=4)
    parser.add_argument('--json')
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)
    results = []
    for label, path in (('A', args.first), ('B', args.second)):
        if path is None:
            continue
        paths = logs_under(path)
        if not paths:
            print('%s: no container.log or server.log under %s' % (label, path), file=sys.stderr)
            return 2
        results.append((label, summarize(paths, args.live)))
    lines = []
    for label, result in results:
        lines += render(label, result)
    if len(results) == 2:
        a, b = results[0][1], results[1][1]
        lines.append('B - A: tau %+.3f (%+.1f%%)' % (b['tau'] - a['tau'], 100.0 * (b['tau'] / a['tau'] - 1.0)) if a['tau'] == a['tau'] and b['tau'] == b['tau']
                     else 'B - A: no comparable rounds')
    print('\n'.join(lines))
    if args.json:
        with open(args.json, 'w', encoding='utf-8', newline='\n') as handle:
            json.dump(dict((label, result) for label, result in results), handle, indent=2, sort_keys=True)
            handle.write('\n')
    return 0


if __name__ == '__main__':
    sys.exit(main())
