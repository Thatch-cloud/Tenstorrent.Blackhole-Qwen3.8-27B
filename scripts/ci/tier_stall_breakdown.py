"""Where a returning session's stall went, from the serving container's log (stdlib only, Python 3.7 syntax; read-only).

    python3 scripts/ci/tier_stall_breakdown.py <server.log> [--request <tag or part of the request id>] [--gap 2.0]

The tier-concurrent plans (c2_prefix_gate --plan tier-concurrent*) show the seats' longest wait while a session returns from host RAM; this tool says what the engine did in that
time. For the returning request (default: the first 'host tier restore' line's) it prints the timeline of the engine's own log lines, in order, with the seconds between them:

  last packed round before   the seats' last round before the request reached the scheduler (PACKED-PHASE round=N)
  admission                  the first line that names the request after that round (the admission checks of serving_prefill_admission)
  prelims + digests          from there to the restore's start: its record checks, which with QWEN_PREFIX_HOST_TIER_VERIFY=all are a sha256 of every block
  restore work               the 'host tier restore' line's ms (sliced: the sum of its slices): the allocation, the spill of what it evicted, the raw device write, the read-back audit
  tail prefill               'lever N route ... ms=' (or the [PREFIX] row's ms): the request's own prefill of what the tier did not hold
  engine build               'sticky engine built ... ms=' and what lies between the prefill and it (slot adoption, the drafter's tap window, the parked rebind)
  first packed round after   the first round with the request live, and every later gap between rounds above --gap seconds (the first round of a new live set runs the drafter
                             audits of an audited arm, for instance)

and the totals: the freeze (last round before to first round after), the TTFT the record says, and the longest gap between two rounds in the whole window. Everything it prints is read
from the log; nothing is inferred except the restore's start (its line's time less its ms).
"""

import argparse
import re
import sys
from datetime import datetime

STAMP = re.compile(r'^(\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d)(\.\d+)?Z\s?')
ROUND = re.compile(r'\[PACKED-PHASE\] round=(\d+) users=(\d+)')
RESTORE = re.compile(r'\[PINDIAG\] prefix: host tier restore req=(\S+) blocks=(\d+) bytes=(\d+) ms=([\d.]+) from=(\d+) to=(\d+)(?: slices=(\d+) max_slice_ms=([\d.]+) held_steps=(\d+))?')
SPILL = re.compile(r'\[PINDIAG\] prefix: host tier spill blocks=(\d+) bytes=(\d+) ms=([\d.]+)')
TAIL = re.compile(r'\[PINDIAG\] lever N route req=(\S+) start=(\d+) end=(\d+).* ms=([\d.]+)')
ENGINE = re.compile(r'\[PINDIAG\] sticky engine built req=(\S+) ms=([\d.]+) frontier=(\d+) prompt=(\d+) kind=(\S+)')


def seconds(line):
    """The docker timestamp of a log line as seconds (float), or None."""
    found = STAMP.match(line)
    if not found:
        return None
    base = datetime.strptime(found.group(1), '%Y-%m-%dT%H:%M:%S')
    return base.replace(tzinfo=None).timestamp() + (float(found.group(2)) if found.group(2) else 0.0)


def timeline(lines, request=None, gap=2.0):
    """-> dict(request, events=[(seconds, label)], rounds=[(seconds, round)], phases=[(name, seconds)], freeze, gaps=[(seconds, from round, to round)]) or None when the log holds no restore."""
    stamped = [(seconds(line), line) for line in lines]
    restore = None
    for index, (when, line) in enumerate(stamped):
        found = RESTORE.search(line)
        if found and when is not None and (request is None or request in found.group(1)):
            restore = (index, when, found)
            break
    if restore is None:
        return None
    index, when, found = restore
    req = found.group(1)
    work_ms = float(found.group(4))
    began = when - work_ms / 1000.0
    rounds = [(w, int(ROUND.search(line).group(1))) for w, line in stamped if w is not None and ROUND.search(line)]
    before = [item for item in rounds if item[0] < began]
    last_before = before[-1] if before else None
    admission = None
    for w, line in stamped[:index]:
        if w is not None and req in line and (last_before is None or w >= last_before[0]):
            admission = w
            break
    tail = engine = None
    for w, line in stamped[index:]:
        if w is None:
            continue
        match = TAIL.search(line)
        if match and match.group(1) == req and tail is None:
            tail = (w, float(match.group(4)))
        match = ENGINE.search(line)
        if match and match.group(1) == req:
            engine = (w, float(match.group(2)), match.group(5))
            break
    spills = [(w, float(SPILL.search(line).group(3)), int(SPILL.search(line).group(1))) for w, line in stamped[:index + 1]
              if w is not None and SPILL.search(line) and began <= w <= when]
    events = []
    if last_before:
        events.append((last_before[0], 'last packed round before (round %d)' % last_before[1]))
    if admission is not None:
        events.append((admission, 'admission (the first line naming the request)'))
    events.append((began, 'restore begins (its line less its %.0f ms)' % work_ms))
    for w, ms, blocks in spills:
        events.append((w, 'host tier spill of %d displaced blocks (%.0f ms, inside the restore)' % (blocks, ms)))
    events.append((when, 'restore ends: %s blocks, %.2f GB%s' % (found.group(2), int(found.group(3)) / 1e9,
                                                                 ', %s slices (max %s ms), held %s steps' % found.group(7, 8, 9) if found.group(7) else '')))
    if tail:
        events.append((tail[0], 'tail prefill done (%.0f ms)' % tail[1]))
    if engine:
        events.append((engine[0], 'sticky engine built (%.0f ms, kind=%s)' % (engine[1], engine[2])))
    after_all = [item for item in rounds if item[0] > (engine[0] if engine else when)]
    if after_all:
        events.append((after_all[0][0], 'first packed round after (round %d)' % after_all[0][1]))
    events.sort()
    phases = []
    if last_before and admission is not None:
        phases.append(('last round before -> admission', admission - last_before[0]))
    if admission is not None:
        phases.append(('admission -> restore begins (prelims + digests)', began - admission))
    phases.append(('restore work (allocate, spill, write, audit)', work_ms / 1000.0))
    if tail:
        phases.append(('restore ends -> tail prefill done', tail[0] - when))
    if engine and tail:
        phases.append(('tail prefill done -> engine built (slot adoption, tap window, rebind)', engine[0] - tail[0]))
    if engine and after_all:
        phases.append(('engine built -> first packed round', after_all[0][0] - engine[0]))
    window = [item for item in rounds if (last_before is None or item[0] >= last_before[0])]
    gaps = []
    for (w1, r1), (w2, r2) in zip(window, window[1:]):
        if w2 - w1 >= gap:
            gaps.append((w2 - w1, r1, r2))
        if w2 - when > 60:
            break
    freeze = (after_all[0][0] - last_before[0]) if (after_all and last_before) else None
    return dict(request=req, events=events, phases=phases, freeze=freeze, gaps=gaps, rounds=len(rounds))


def render(result):
    out = ['returning request %s' % result['request']]
    start = result['events'][0][0]
    previous = start
    for when, label in result['events']:
        out.append('  +%8.3f s  (+%7.3f)  %s' % (when - start, when - previous, label))
        previous = when
    out.append('phases:')
    for name, value in result['phases']:
        out.append('  %8.3f s  %s' % (value, name))
    if result['freeze'] is not None:
        out.append('the seats\' freeze (last packed round before to first after): %.3f s' % result['freeze'])
    for length, first, second in result['gaps']:
        out.append('a gap of %.3f s between packed rounds %d and %d' % (length, first, second))
    return out


def main(argv=None, out=print):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('log')
    parser.add_argument('--request', default=None, help='a tag, or part of the request id (default: the first restore in the log)')
    parser.add_argument('--gap', type=float, default=2.0, help='report gaps between packed rounds of at least this many seconds')
    options = parser.parse_args(argv)
    try:
        with open(options.log, encoding='utf-8', errors='replace') as handle:
            lines = handle.read().splitlines()
    except OSError as error:
        out('refused: %s' % error)
        return 2
    result = timeline(lines, options.request, options.gap)
    if result is None:
        out('no "host tier restore" line%s in %s' % (' for %s' % options.request if options.request else '', options.log))
        return 1
    for line in render(result):
        out(line)
    return 0


if __name__ == '__main__':
    sys.exit(main())
