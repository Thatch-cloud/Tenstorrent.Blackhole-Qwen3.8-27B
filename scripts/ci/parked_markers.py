"""What an engine-reuse server logs (QWEN_FAST_PARKED_ENGINES), parsed: the evidence the parked gates judge. Stdlib only, Python 3.7 syntax: the gate
reads this on the rig host.

Every line format below is its producer's own (each constant names its source; test_parked_tp4_markers renders the producers' real lines - the set on
the four-chip census world, the coordinator, the prewarm - and parses them here), and every field is read by name, so a field a producer adds never
hides a line. A missing marker is reported, never guessed around.

  serving_parked_engines.ParkedEngineSet (attach):
    [PINDIAG] parked engines built k=<n> of <slots> attach_ms=<f> trace_used=<n|unavailable> free=<bytes> largest_free=<bytes>
        (or 'dram unavailable (<reason>)' in place of the three readings)
    [PINDIAG] parked engines stopped at k=<n> of <slots>: short of <terms> (free=<b> largest_free=<b> need=<b>)
    [PINDIAG] parked drafter warm P=<rows> ms=<f> window=<rows> prewarm_pairs=<n>
    [PINDIAG] gate dram ballast bytes=<n> buffers=<n> (gate only; unread) <the built line's readings>
    [PINDIAG] parked negative control mode=<carry|drafter|pages|widths> (gate only: ...)  |  fault=<park|rebind> (gate only: ...)
  per request:
    [PINDIAG] parked rebind req=<id> slot=<k> gen=<n> P=<n> budget=<n> ms=<f> single_rebuilt=<0|1> window=<rows> widths=<a,b,..> capacity=<tokens>
    [PINDIAG] parked rebind refused req=<id> slot=<k> reason=<text> fallback=build        (a host refusal, before any device write)
    [PINDIAG] parked rebind failed req=<id> slot=<k> reason=<text> fallback=build         (after the first device write; the slot is unparked)
    [PINDIAG] parked rebind peak slot=<k> P=<n> free_before=<b> free_at_peak=<b> held=<b>  (QWEN_FAST_PARKED_AUDIT)
    [PINDIAG] parked rebind digest slot=<k> snapshots=<n> tables=<n> slot0=<0|1> zeroed=<0|1> equal=1        (QWEN_FAST_PARKED_AUDIT)
    [PINDIAG] parked programs rebind slot=<k> programs=<a>-><b>      (the S8 tripwire: a rebind compiles nothing after the block's first replay)
    [PINDIAG] parked slot <k> unparked: <reason>
    [PINDIAG] parked slot <k> re-parked ms=<f>
    [PINDIAG] parked slot <k> single rebuilt at <park|idle> ms=<f> trace_delta=<bytes|n/a> dram_delta=<bytes|n/a>
    [PINDIAG] parked slot <k> single kept released at <park|idle>: short of <terms> (free=.. largest_free=.. need=..)
    [PINDIAG] parked release rung=<1|2|3> freed=<bytes> slot=<k>                                       (the release ladder)
    [PINDIAG] parked instance state slot=<k> problem=<text>                                             (an override left on a parked object)
    [PINDIAG] parked engines off (kill switch): unparked=<n>
    [PINDIAG] parked draft book quad slots=<..> captured_at_round=<n>                                  (2c: once per block per process)
    [PACKED-PROPOSE] released parked slot=<k> quad=<0|1> pairs=[...]                                    (E1 only)
    [PINDIAG] sticky engine built req=<id> ms=<f> frontier=<R> prompt=<P> kind=<build|rebind>
    [PINDIAG] dram hold age request=<id> ms=<f> decodes=<d>                                             (parked terms in force)
    [LEVERN] admission cost kind=<build|rebind> ms=<f> ewma=<f>                                         (QWEN_FAST_LEVERN_BUILD_MS=learned)
  the publish prewarm (publish_prewarm), required beside the parked warm:
    [PINDIAG] publish prewarm pairs=<...> count=<n> ms=<f> program_cache=<a>-><b>   |   ... skipped history_rows=<n>
  the memory ledger (memory_ledger; QWEN_FAST_MEMORY_LEDGER=1, on in the image):
    [MEMLEDGER] phase=P7p point=parked k=<n> ... chip<c> allocated=<f>GB free=<f>GB largest_free=<f>MB ...
    [MEMLEDGER] before op=rebind ... / after op=rebind ...        (the rebind's bracket)

scan() returns the facts; parked_judge turns them into what a gate expects of an arm that ran the parked profile (or of one that did not: every
parked marker is then a leak).
"""
import re

BUILT = '[PINDIAG] parked engines built '
STOPPED = '[PINDIAG] parked engines stopped at '
WARM = '[PINDIAG] parked drafter warm '
BALLAST = '[PINDIAG] gate dram ballast '
NEGATIVE = '[PINDIAG] parked negative control '
REBIND = '[PINDIAG] parked rebind req='
REFUSED = '[PINDIAG] parked rebind refused '
FAILED = '[PINDIAG] parked rebind failed '
PEAK = '[PINDIAG] parked rebind peak '
DIGEST = '[PINDIAG] parked rebind digest '
PROGRAMS = '[PINDIAG] parked programs '
SLOT = '[PINDIAG] parked slot '
LADDER = '[PINDIAG] parked release rung='
INSTANCE = '[PINDIAG] parked instance state '
OFF = '[PINDIAG] parked engines off '
BOOK_QUAD = '[PINDIAG] parked draft book quad '
RELEASED = '[PACKED-PROPOSE] released parked '
PREWARM = '[PINDIAG] publish prewarm'
PROPOSAL_BUCKETS = '[PINDIAG] proposal buckets built request='
STICKY = '[PINDIAG] sticky engine built req='
HOLD_AGE = '[PINDIAG] dram hold age '
COST = '[LEVERN] admission cost '

# Every marker a run with the flag off must not print (judge with parked=False).
LEAK_MARKERS = (BUILT, STOPPED, WARM, BALLAST, NEGATIVE, REBIND, REFUSED, FAILED, PEAK, DIGEST, PROGRAMS, SLOT, LADDER, INSTANCE, OFF, BOOK_QUAD, RELEASED,
                HOLD_AGE, COST)

BUILT_LINE = re.compile(r'\[PINDIAG\] parked engines built k=([0-9]+) of ([0-9]+) attach_ms=([0-9.]+) (.*)$')
READINGS = re.compile(r'trace_used=(\S+) free=([0-9]+) largest_free=([0-9]+)')
STOPPED_LINE = re.compile(r'\[PINDIAG\] parked engines stopped at k=([0-9]+) of ([0-9]+): short of (\S+) '
                          r'\(free=([0-9]+) largest_free=([0-9]+) need=([0-9]+)\)')
WARM_LINE = re.compile(r'\[PINDIAG\] parked drafter warm P=([0-9]+) ms=([0-9.]+) window=([0-9]+) prewarm_pairs=([0-9]+)')
BALLAST_LINE = re.compile(r'\[PINDIAG\] gate dram ballast bytes=([0-9]+) buffers=([0-9]+)')
NEGATIVE_LINE = re.compile(r'\[PINDIAG\] parked negative control (mode|fault)=(\S+)')
REBIND_LINE = re.compile(r'\[PINDIAG\] parked rebind req=(\S+) slot=([0-9]+) gen=([0-9]+) P=([0-9]+) budget=([0-9]+) ms=([0-9.]+) '
                         r'single_rebuilt=([01]) window=([0-9]+) widths=([0-9,]+) capacity=([0-9]+)')
REFUSED_LINE = re.compile(r'\[PINDIAG\] parked rebind refused req=(\S+) slot=([0-9]+) reason=(.*) fallback=build')
FAILED_LINE = re.compile(r'\[PINDIAG\] parked rebind failed req=(\S+) slot=([0-9]+) reason=(.*) fallback=build')
PEAK_LINE = re.compile(r'\[PINDIAG\] parked rebind peak slot=([0-9]+) P=([0-9]+) free_before=([0-9]+) free_at_peak=([0-9]+) held=([0-9]+)')
DIGEST_LINE = re.compile(r'\[PINDIAG\] parked rebind digest slot=([0-9]+) snapshots=([0-9]+) tables=([0-9]+) slot0=([01]) zeroed=([01]) equal=([01])')
PROGRAMS_LINE = re.compile(r'\[PINDIAG\] parked programs rebind slot=([0-9]+) programs=(\d+|None)->(\d+|None)')
UNPARKED_LINE = re.compile(r'\[PINDIAG\] parked slot ([0-9]+) unparked: (.*)$')
REPARKED_LINE = re.compile(r'\[PINDIAG\] parked slot ([0-9]+) re-parked ms=([0-9.]+)')
REBUILT_LINE = re.compile(r'\[PINDIAG\] parked slot ([0-9]+) single rebuilt at (\S+) ms=([0-9.]+)(?: trace_delta=(\S+) dram_delta=(\S+))?')
KEPT_LINE = re.compile(r'\[PINDIAG\] parked slot ([0-9]+) single kept released at (\S+): short of (\S+) '
                       r'\(free=([0-9]+) largest_free=([0-9]+) need=([0-9]+)\)')
LADDER_LINE = re.compile(r'\[PINDIAG\] parked release rung=([123]) freed=(-?[0-9]+) slot=(-?[0-9]+)')
INSTANCE_LINE = re.compile(r'\[PINDIAG\] parked instance state slot=([0-9]+) problem=(.*)$')
OFF_LINE = re.compile(r'\[PINDIAG\] parked engines off \(kill switch\): unparked=([0-9]+)')
BOOK_QUAD_LINE = re.compile(r'\[PINDIAG\] parked draft book quad slots=(\S+) captured_at_round=([0-9]+)')
RELEASED_LINE = re.compile(r'\[PACKED-PROPOSE\] released parked slot=(\S+) quad=([01]) pairs=(\[[^\n]*\])')
PREWARM_LINE = re.compile(r'\[PINDIAG\] publish prewarm pairs=(\S*) count=([0-9]+)')
PREWARM_SKIPPED = re.compile(r'\[PINDIAG\] publish prewarm skipped history_rows=([0-9]+)')
STICKY_LINE = re.compile(r'\[PINDIAG\] sticky engine built req=(\S+) ms=([0-9.]+) frontier=([0-9]+) prompt=([0-9]+)(?: kind=(build|rebind))?')
HOLD_AGE_LINE = re.compile(r'\[PINDIAG\] dram hold age request=(\S+) ms=([0-9]+) decodes=([0-9]+)')
COST_LINE = re.compile(r'\[LEVERN\] admission cost kind=(build|rebind) ms=([0-9.]+) ewma=([0-9.]+)')
# The ledger's P7 and P7p phases: 'phase=P7p point=parked k=8' then, per chip, its readings (memory_ledger's own format, read by field name).
LEDGER_LINE = re.compile(r'\[MEMLEDGER\] phase=(P7p?) point=([^\n]*?) chip([0-9]+) allocated=([0-9.]+)GB free=([0-9.]+)GB largest_free=([0-9.]+)MB')
REBIND_OP = re.compile(r'\[MEMLEDGER\] (before|after) op=rebind\b([^\n]*)')

KEYS = ('built', 'stopped', 'warm', 'ballast', 'negative', 'rebinds', 'refused', 'failed', 'peaks', 'digests', 'programs', 'unparked', 'reparked',
        'single_rebuilt', 'single_kept', 'ladder', 'instance', 'off', 'book_quads', 'released', 'prewarm', 'prewarm_skipped', 'sticky', 'hold_ages',
        'costs', 'ledger', 'rebind_ops')


def scan(lines):
    """The parked facts of a log (an iterable of lines), in log order, one list per key in KEYS, and `markers`: how many lines carried one."""
    facts = dict((key, []) for key in KEYS)
    facts['markers'] = 0
    for line in lines:
        if not ('parked' in line or 'ballast' in line or 'publish prewarm' in line or 'P7' in line or 'op=rebind' in line
                or 'sticky engine built' in line or 'dram hold age' in line or 'admission cost' in line):
            continue
        found = False
        match = BUILT_LINE.search(line)
        if match:
            entry = dict(k=int(match.group(1)), slots=int(match.group(2)), attach_ms=float(match.group(3)))
            readings = READINGS.search(match.group(4))
            if readings:
                entry.update(trace_used=readings.group(1), free=int(readings.group(2)), largest_free=int(readings.group(3)))
            else:
                entry.update(unavailable=match.group(4).strip())
            facts['built'].append(entry)
            found = True
        match = STOPPED_LINE.search(line)
        if match:
            facts['stopped'].append(dict(k=int(match.group(1)), slots=int(match.group(2)), short=match.group(3), free=int(match.group(4)),
                                         largest_free=int(match.group(5)), need=int(match.group(6))))
            found = True
        match = WARM_LINE.search(line)
        if match:
            facts['warm'].append(dict(rows=int(match.group(1)), ms=float(match.group(2)), window=int(match.group(3)), prewarm_pairs=int(match.group(4))))
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
            facts['rebinds'].append(dict(request=match.group(1), slot=int(match.group(2)), generation=int(match.group(3)), prompt=int(match.group(4)),
                                         budget=int(match.group(5)), ms=float(match.group(6)), single_rebuilt=match.group(7) == '1',
                                         window=int(match.group(8)), widths=tuple(int(part) for part in match.group(9).split(',') if part),
                                         capacity=int(match.group(10))))
            found = True
        match = REFUSED_LINE.search(line)
        if match:
            facts['refused'].append(dict(request=match.group(1), slot=int(match.group(2)), reason=match.group(3).strip()))
            found = True
        match = FAILED_LINE.search(line)
        if match:
            facts['failed'].append(dict(request=match.group(1), slot=int(match.group(2)), reason=match.group(3).strip()))
            found = True
        match = PEAK_LINE.search(line)
        if match:
            facts['peaks'].append(dict(slot=int(match.group(1)), prompt=int(match.group(2)), free_before=int(match.group(3)),
                                       free_at_peak=int(match.group(4)), held=int(match.group(5))))
            found = True
        match = DIGEST_LINE.search(line)
        if match:
            facts['digests'].append(dict(slot=int(match.group(1)), snapshots=int(match.group(2)), tables=int(match.group(3)),
                                         slot0=match.group(4) == '1', zeroed=match.group(5) == '1', equal=match.group(6) == '1'))
            found = True
        match = PROGRAMS_LINE.search(line)
        if match:
            before, after = [None if part == 'None' else int(part) for part in (match.group(2), match.group(3))]
            facts['programs'].append(dict(slot=int(match.group(1)), before=before, after=after))
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
            facts['single_kept'].append(dict(slot=int(match.group(1)), moment=match.group(2), short=match.group(3), need=int(match.group(6))))
            found = True
        match = LADDER_LINE.search(line)
        if match:
            facts['ladder'].append(dict(rung=int(match.group(1)), freed=int(match.group(2)), slot=int(match.group(3))))
            found = True
        match = INSTANCE_LINE.search(line)
        if match:
            facts['instance'].append(dict(slot=int(match.group(1)), problem=match.group(2).strip()))
            found = True
        match = OFF_LINE.search(line)
        if match:
            facts['off'].append(dict(unparked=int(match.group(1))))
            found = True
        match = BOOK_QUAD_LINE.search(line)
        if match:
            facts['book_quads'].append(dict(slots=match.group(1), round=int(match.group(2))))
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
        match = STICKY_LINE.search(line)
        if match:
            facts['sticky'].append(dict(request=match.group(1), ms=float(match.group(2)), frontier=int(match.group(3)), prompt=int(match.group(4)),
                                        kind=match.group(5)))
            found = True
        match = HOLD_AGE_LINE.search(line)
        if match:
            facts['hold_ages'].append(dict(request=match.group(1), ms=int(match.group(2)), decodes=int(match.group(3))))
            found = True
        match = COST_LINE.search(line)
        if match:
            facts['costs'].append(dict(kind=match.group(1), ms=float(match.group(2)), ewma=float(match.group(3))))
            found = True
        match = LEDGER_LINE.search(line)
        if match:
            facts['ledger'].append(dict(phase=match.group(1), point=match.group(2).strip(), chip=int(match.group(3)),
                                        allocated_gb=float(match.group(4)), free_gb=float(match.group(5)), largest_free_mb=float(match.group(6))))
            found = True
        match = REBIND_OP.search(line)
        if match:
            facts['rebind_ops'].append(dict(when=match.group(1), text=match.group(2).strip()))
            found = True
        if found:
            facts['markers'] += 1
    return facts
