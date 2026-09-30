"""What a parked-engines server logs (Stage E, QWEN_FAST_PARKED_ENGINES), parsed: the evidence the parked gates judge.

Every line format below is its producer's own (each constant names its source; test_parked_markers renders the
producers' real lines - the parked set on the census world, the coordinator, the prewarm - and parses them here),
and every field is read by name, so a field a producer adds never hides a line. A missing marker is reported,
never guessed around. Stdlib only, Python 3.7 syntax: the gate reads this on the rig host.

  serving_parked_engines.ParkedEngineSet (attach):
    [PINDIAG] parked engines built k=<n> of <slots> attach_ms=<f> trace_used=<n|unavailable> free=<bytes>
        largest_free=<bytes>          (or 'dram unavailable (<reason>)' in place of the three readings)
    [PINDIAG] parked engines stopped at k=<n> of <slots>: short of <terms> (free=<b> largest_free=<b> need=<b>)
    [PINDIAG] parked drafter warm P=<rows> ms=<f> window=<rows> prewarm_pairs=<n>
    [PINDIAG] gate dram ballast bytes=<n> buffers=<n> (gate only; unread) <the built line's readings>
    [PINDIAG] parked negative control mode=<carry|drafter> (gate only: ...)  |  fault=park (gate only: ...)
  per request:
    [PINDIAG] parked rebind req=<id> slot=<k> gen=<n> P=<n> budget=<n> ms=<f> single_rebuilt=<0|1> window=<rows>
    [PINDIAG] parked rebind peak slot=<k> P=<n> free_before=<b> free_at_peak=<b> held=<b> (QWEN_FAST_PARKED_AUDIT)
    [PINDIAG] parked slot <k> unparked: <reason>
    [PINDIAG] parked slot <k> re-parked ms=<f>
    [PINDIAG] parked slot <k> single rebuilt at <park|idle> ms=<f> trace_delta=<bytes|n/a> dram_delta=<bytes|n/a>
    [PINDIAG] parked slot <k> single kept released at <park|idle>: short of <terms> (free=.. largest_free=.. need=..)
    [PACKED-PROPOSE] released parked slot=<k> quad=<0|1> pairs=[...]    (dflash_packed_proposal_coordinator)
  the publish prewarm (publish_prewarm), required beside the parked warm:
    [PINDIAG] publish prewarm pairs=<...> count=<n> ms=<f> program_cache=<a>-><b>   |   ... skipped history_rows=<n>
  the memory ledger (memory_ledger; QWEN_FAST_MEMORY_LEDGER=1, on in the image):
    [MEMLEDGER] phase=P7p point=parked k=<n> ... chip<c> allocated=<f>GB free=<f>GB largest_free=<f>MB ...
    [MEMLEDGER] before op=rebind ... / after op=rebind ...        (the rebind's bracket)

scan() returns the facts; problems() turns them into what a gate expects of an arm that ran the parked profile (or
of one that did not: every parked marker is then a leak).
"""
import re

BUILT = '[PINDIAG] parked engines built '
STOPPED = '[PINDIAG] parked engines stopped at '
WARM = '[PINDIAG] parked drafter warm '
BALLAST = '[PINDIAG] gate dram ballast '
NEGATIVE = '[PINDIAG] parked negative control '
REBIND = '[PINDIAG] parked rebind req='
PEAK = '[PINDIAG] parked rebind peak '
SLOT = '[PINDIAG] parked slot '
RELEASED = '[PACKED-PROPOSE] released parked '
PREWARM = '[PINDIAG] publish prewarm'
PROPOSAL_BUCKETS = '[PINDIAG] proposal buckets built request='

# Every marker a run with the flag off must not print (problems() with parked=False).
LEAK_MARKERS = (BUILT, STOPPED, WARM, BALLAST, NEGATIVE, REBIND, PEAK, SLOT, RELEASED)

BUILT_LINE = re.compile(r'\[PINDIAG\] parked engines built k=([0-9]+) of ([0-9]+) attach_ms=([0-9.]+) (.*)$')
READINGS = re.compile(r'trace_used=(\S+) free=([0-9]+) largest_free=([0-9]+)')
STOPPED_LINE = re.compile(r'\[PINDIAG\] parked engines stopped at k=([0-9]+) of ([0-9]+): short of (\S+) '
                          r'\(free=([0-9]+) largest_free=([0-9]+) need=([0-9]+)\)')
WARM_LINE = re.compile(r'\[PINDIAG\] parked drafter warm P=([0-9]+) ms=([0-9.]+) window=([0-9]+) '
                       r'prewarm_pairs=([0-9]+)')
BALLAST_LINE = re.compile(r'\[PINDIAG\] gate dram ballast bytes=([0-9]+) buffers=([0-9]+)')
NEGATIVE_LINE = re.compile(r'\[PINDIAG\] parked negative control (mode|fault)=(\S+)')
REBIND_LINE = re.compile(r'\[PINDIAG\] parked rebind req=(\S+) slot=([0-9]+) gen=([0-9]+) P=([0-9]+) budget=([0-9]+) '
                         r'ms=([0-9.]+) single_rebuilt=([01]) window=([0-9]+)')
PEAK_LINE = re.compile(r'\[PINDIAG\] parked rebind peak slot=([0-9]+) P=([0-9]+) free_before=([0-9]+) '
                       r'free_at_peak=([0-9]+) held=([0-9]+)')
UNPARKED_LINE = re.compile(r'\[PINDIAG\] parked slot ([0-9]+) unparked: (.*)$')
REPARKED_LINE = re.compile(r'\[PINDIAG\] parked slot ([0-9]+) re-parked ms=([0-9.]+)')
REBUILT_LINE = re.compile(r'\[PINDIAG\] parked slot ([0-9]+) single rebuilt at (\S+) ms=([0-9.]+)'
                          r'(?: trace_delta=(\S+) dram_delta=(\S+))?')
KEPT_LINE = re.compile(r'\[PINDIAG\] parked slot ([0-9]+) single kept released at (\S+): short of (\S+) '
                       r'\(free=([0-9]+) largest_free=([0-9]+) need=([0-9]+)\)')
RELEASED_LINE = re.compile(r'\[PACKED-PROPOSE\] released parked slot=(\S+) quad=([01]) pairs=(\[[^\]\n]*\])')
PREWARM_LINE = re.compile(r'\[PINDIAG\] publish prewarm pairs=(\S*) count=([0-9]+)')
PREWARM_SKIPPED = re.compile(r'\[PINDIAG\] publish prewarm skipped history_rows=([0-9]+)')
# The ledger's P7p phase: 'phase=P7p point=parked k=4' then, per chip, its readings (memory_ledger's own format,
# read by field name; a message past the ledger's line budget continues on '[MEMLEDGER] ...' lines).
P7P_LINE = re.compile(r'\[MEMLEDGER\] phase=P7p point=([^\n]*?) chip([0-9]+) allocated=([0-9.]+)GB free=([0-9.]+)GB '
                      r'largest_free=([0-9.]+)MB')
REBIND_OP = re.compile(r'\[MEMLEDGER\] (before|after) op=rebind\b([^\n]*)')


def scan(lines):
    """The parked facts of a log (an iterable of lines), in log order: built, stopped, warm, ballast, negative,
    rebinds, peaks, unparked, reparked, single_rebuilt, single_kept, released, prewarm (counts) and prewarm_skipped,
    p7p (per chip), rebind_ops (before/after, the ledger's bracket) and the count of every marker seen."""
    facts = dict(built=[], stopped=[], warm=[], ballast=[], negative=[], rebinds=[], peaks=[], unparked=[],
                 reparked=[], single_rebuilt=[], single_kept=[], released=[], prewarm=[], prewarm_skipped=[],
                 p7p=[], rebind_ops=[], markers=0)
    for line in lines:
        if 'parked' in line or 'ballast' in line or 'publish prewarm' in line or 'P7p' in line or 'op=rebind' in line:
            found = False
            match = BUILT_LINE.search(line)
            if match:
                entry = dict(k=int(match.group(1)), slots=int(match.group(2)), attach_ms=float(match.group(3)))
                readings = READINGS.search(match.group(4))
                if readings:
                    entry.update(trace_used=readings.group(1), free=int(readings.group(2)),
                                 largest_free=int(readings.group(3)))
                else:
                    entry.update(unavailable=match.group(4).strip())
                facts['built'].append(entry)
                found = True
            match = STOPPED_LINE.search(line)
            if match:
                facts['stopped'].append(dict(k=int(match.group(1)), slots=int(match.group(2)), short=match.group(3),
                                             free=int(match.group(4)), largest_free=int(match.group(5)),
                                             need=int(match.group(6))))
                found = True
            match = WARM_LINE.search(line)
            if match:
                facts['warm'].append(dict(rows=int(match.group(1)), ms=float(match.group(2)),
                                          window=int(match.group(3)), prewarm_pairs=int(match.group(4))))
                found = True
            match = BALLAST_LINE.search(line)
            if match:
                facts['ballast'].append(dict(bytes=int(match.group(1)), buffers=int(match.group(2))))
                found = True
            match = NEGATIVE_LINE.search(line)
            if match:
                facts['negative'].append(dict(kind=match.group(1), value=match.group(2)))
                found = True
            match = REBIND_LINE.search(line)
            if match:
                facts['rebinds'].append(dict(
                    request=match.group(1), slot=int(match.group(2)), generation=int(match.group(3)),
                    prompt=int(match.group(4)), budget=int(match.group(5)), ms=float(match.group(6)),
                    single_rebuilt=match.group(7) == '1', window=int(match.group(8))))
                found = True
            match = PEAK_LINE.search(line)
            if match:
                facts['peaks'].append(dict(slot=int(match.group(1)), prompt=int(match.group(2)),
                                           free_before=int(match.group(3)), free_at_peak=int(match.group(4)),
                                           held=int(match.group(5))))
                found = True
            match = UNPARKED_LINE.search(line)
            if match:
                facts['unparked'].append(dict(slot=int(match.group(1)), reason=match.group(2).strip()))
                found = True
            match = REPARKED_LINE.search(line)
            if match:
                facts['reparked'].append(dict(slot=int(match.group(1)), ms=float(match.group(2))))
                found = True
            match = REBUILT_LINE.search(line)
            if match:
                facts['single_rebuilt'].append(dict(
                    slot=int(match.group(1)), moment=match.group(2), ms=float(match.group(3)),
                    trace_delta=None if match.group(4) in (None, 'n/a') else int(match.group(4)),
                    dram_delta=None if match.group(5) in (None, 'n/a') else int(match.group(5))))
                found = True
            match = KEPT_LINE.search(line)
            if match:
                facts['single_kept'].append(dict(slot=int(match.group(1)), moment=match.group(2),
                                                 short=match.group(3), need=int(match.group(6))))
                found = True
            match = RELEASED_LINE.search(line)
            if match:
                facts['released'].append(dict(slot=match.group(1), quad=match.group(2) == '1', pairs=match.group(3)))
                found = True
            match = PREWARM_LINE.search(line)
            if match:
                facts['prewarm'].append(int(match.group(2)))
                found = True
            match = PREWARM_SKIPPED.search(line)
            if match:
                facts['prewarm_skipped'].append(int(match.group(1)))
                found = True
            match = P7P_LINE.search(line)
            if match:
                facts['p7p'].append(dict(point=match.group(1).strip(), chip=int(match.group(2)),
                                         allocated_gb=float(match.group(3)), free_gb=float(match.group(4)),
                                         largest_free_mb=float(match.group(5))))
                found = True
            match = REBIND_OP.search(line)
            if match:
                facts['rebind_ops'].append(dict(when=match.group(1), text=match.group(2).strip()))
                found = True
            if found:
                facts['markers'] += 1
    return facts


def problems(facts, parked, slots=4, warm_required=True, allow_unparks=False, prewarm_required=True):
    """(problems, not exercised) of one arm's log facts. With `parked` False (the flag-off twin) every parked
    marker is a problem: the profile leaves the flag off, so none may appear. With it on: the set built 4 of 4
    (the gate's design row G-E0: a stop at k < 4 is a failed attach at 131,072), its drafter warmed (P=2048) with a
    publish prewarm that counted a pair (its absence was silent before), a rebind line per request that carries a
    known generation, and - unless the arm injects a fault - no unpark."""
    out, missing = [], []
    if not parked:
        seen = sorted(set(key for key in ('built', 'stopped', 'warm', 'ballast', 'negative', 'rebinds', 'peaks',
                                          'unparked', 'reparked', 'single_rebuilt', 'single_kept', 'released')
                          if facts.get(key)))
        if seen:
            out.append('the profile leaves QWEN_FAST_PARKED_ENGINES off, but the server log carries parked markers '
                       '(%s): this is not the profile\'s path' % ', '.join(seen))
        return out, missing
    if not facts['built']:
        out.append('no "%s" line: the parked set was never built (the flag did not reach the attach)' % BUILT.strip())
    else:
        for entry in facts['built']:
            if entry['k'] != slots or entry['slots'] != slots:
                out.append('the parked set built %d of %d engines, not %d of %d' % (entry['k'], entry['slots'], slots,
                                                                                     slots))
    for entry in facts['stopped']:
        out.append('the parked build stopped at k=%d of %d, short of %s (free=%d largest_free=%d need=%d)' % (
            entry['k'], entry['slots'], entry['short'], entry['free'], entry['largest_free'], entry['need']))
    if warm_required:
        if not facts['warm']:
            out.append('no "%s" line: slot 0\'s drafter was never warmed at 2048' % WARM.strip())
        for entry in facts['warm']:
            if entry['rows'] != 2048:
                out.append('the drafter warm ran at P=%d, not 2048: the publish prewarm skips below 2048' % entry['rows'])
        if prewarm_required and not any(count > 0 for count in facts['prewarm']):
            out.append('no publish prewarm with count > 0 (%s): the first request after a restart would compile the '
                       'publication programs' % ('skipped at history_rows=%s' % facts['prewarm_skipped']
                                                 if facts['prewarm_skipped'] else 'no prewarm line'))
    if facts['unparked'] and not allow_unparks:
        out.append('%d slots unparked (%s): the parked set is expected to serve every request' % (
            len(facts['unparked']), '; '.join('slot %d: %s' % (e['slot'], e['reason']) for e in facts['unparked'][:3])))
    for entry in facts['rebinds']:
        if entry['generation'] < 1:
            out.append('a rebind at generation %d (request %s): the generation moves on every rebind' % (
                entry['generation'], entry['request']))
    return out, missing
