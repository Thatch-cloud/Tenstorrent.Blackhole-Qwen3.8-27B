"""The engine-reuse gates' judgements (QWEN_FAST_PARKED_ENGINES): what an arm's container log must show, and how two arms compare, as functions of logs and
smoke results, where a CPU test can hold them. Stdlib and the markers module only, Python 3.7 syntax: the gate reads this on the rig host.

THE RULES OF A PARKED ARM (judge). Read from the arm's profile env and its container log:
  attach     one `parked engines built` line, k equal to the seats (8; 4 with one block), never a `stopped` line; slot 0's drafter warmed at P=2048 with a
             publish prewarm that counted a pair; no parked engine ever resident before the first request (the audit arm raises at attach if a build replayed
             a block's trace). ER0's numeric rules (ledger_problems): each parked build costs 0.46-0.51 GB per chip (0.468-0.495 measured), and the P7p free
             reading stays above LEDGER_FREE_FLOOR_GB (audited arms hold less, 3.6 GB, than audits-off ones, 4.7 GB).
  rebinds    one `parked rebind` line per request that reached a parked slot; its generation is at least 1; its widths are EXACTLY a cold engine's for the
             request (R4): capture_widths(P, capacity, 16, budget - 1, 4), the seed having been emitted; a rebind compiles no program (`parked programs`
             lines: programs before equal programs after, from the second rebind on - the S8 tripwire).
  fallbacks  none, except the injected faults: QWEN_FAST_PARKED_FAULT=park shows exactly one unpark with the injected reason and a later re-park;
             =rebind shows exactly one `rebind refused ... fallback=build` and the slot re-parked; a `rebind failed` line is never expected.
  audit      with QWEN_FAST_PARKED_AUDIT=1 one digest line per rebind, equal, with native slot 0 and the re-zeroed sample read.
  instance   no `parked instance state` line (an installer's override left on a parked object).
  book (2c)  the draft book's quad is captured at most once per block per process, and no `released parked` line is ever logged (the traces are the
             slots'); with the engines alone (E1) the coordinator's release lines are expected.
  holds      every `dram hold age` is within HOLD_AGE_LIMIT_MS (a hold under the parked terms clears without the server draining).
  markers    a request that built no engine logs its sticky line with kind=rebind and every other with kind=build.
  flag off   the arm's profile leaves QWEN_FAST_PARKED_ENGINES off: not one parked marker may appear in its log.

TWO ARMS (compare). parked_compare.py compares the arms' answers (levern_compare's rows and users) and then these: the drafter-equivalence judge (the
per-round accepted-prefix sequences of the solo requests are identical between the arms; concurrent, the mean acceptance is within ACCEPTANCE_TOLERANCE),
because greedy verification emits the target's argmax whatever was drafted, so no token gate can see a wrong drafter rebind. The negative controls
(negative_verdict) must fail their own judge: `carry` and `pages` must CHANGE tokens, `drafter` must keep every token and FAIL the equivalence judge,
`widths` must show a rebind whose widths are not a cold engine's (it may keep every token: the ticket-width line is its judge).
"""
import re

import parked_markers as markers

VERIFY_WIDTHS = (1, 2, 4, 8, 16, 32, 64)
CAPTURE_ROWS = 4
VERIFIER_ROWS = 16
PROPOSALS = 15
ACCEPTANCE_TOLERANCE = 0.02
# ER3: a hold under the parked terms clears within this long (a client deadline's worth), without the server draining.
HOLD_AGE_LIMIT_MS = 180000
# ER0: what one parked engine takes of a chip's DRAM (measured 0.468-0.495 GB), with a margin for the first build's own programs.
ENGINE_GB = (0.46, 0.51)
# ER0: the P7p free reading per chip at eight parked engines (design 3.2: 4.95-5.11 GB audits off, 3.69-3.85 audited, minus a margin).
LEDGER_FREE_FLOOR_GB = dict(plain=4.7, audited=3.6)
SEATS_ENV = 'QWEN_FAST_M3_BLOCKS'
PATH_LINE = re.compile(r'\[(PACKED|SEQUENTIAL|SEQ-PUBLISH)\] request=(\S+) ([^\n]*)')
PATH_FIELD = re.compile(r'(?<![A-Za-z0-9_])(prefix|emitted|rows)=([0-9]+|n/a)(?![0-9A-Za-z_])')
FAULT_PARK_REASON = 'injected park fault (gate only)'
FAULT_REBIND_REASON = 'injected rebind fault (gate only)'


def capture_widths(position, capacity, verifier_rows, remaining, cap=CAPTURE_ROWS):
    """verifier_engine.capture_widths, restated (the rig host has no torch): the widths a cold engine captures for a request."""
    return tuple(rows for rows in VERIFY_WIDTHS if rows <= min(verifier_rows, remaining, cap))


def seats(env):
    """How many pool slots (and parked engines) the profile has: eight beside QWEN_FAST_M3_BLOCKS=2, else four."""
    return 8 if str((env or {}).get(SEATS_ENV, '')) == '2' else 4


def parked_on(env):
    return str((env or {}).get('QWEN_FAST_PARKED_ENGINES', '')) == '1'


def lines_of(text):
    return (text or '').splitlines()


def ledger_phase(facts, phase):
    """{chip: (allocated GB, free GB, largest free MB)} of the LAST reading of a ledger phase ('P7' or 'P7p')."""
    found = {}
    for row in facts['ledger']:
        if row['phase'] == phase:
            found[row['chip']] = (row['allocated_gb'], row['free_gb'], row['largest_free_mb'])
    return found


def ledger_problems(facts, engines, audited):
    """ER0's numeric rules from the memory ledger: per chip, the P7 -> P7p allocation growth is `engines` engines' (0.46 to 0.51 GB each) and the free
    reading after them is above the floor. [] when the ledger was not on (nothing to read)."""
    before, after = ledger_phase(facts, 'P7'), ledger_phase(facts, 'P7p')
    if not before or not after:
        return []
    problems = []
    floor = LEDGER_FREE_FLOOR_GB['audited' if audited else 'plain']
    for chip in sorted(set(before) & set(after)):
        grown = after[chip][0] - before[chip][0]
        low, high = ENGINE_GB[0] * engines, ENGINE_GB[1] * engines
        if not low <= grown <= high:
            problems.append('chip %d: the %d parked engines took %.3f GB of DRAM, outside %.2f-%.2f GB (%.2f-%.2f each)'
                            % (chip, engines, grown, low, high, ENGINE_GB[0], ENGINE_GB[1]))
        if engines == 8 and after[chip][1] < floor:
            problems.append('chip %d: %.3f GB free after the parked engines, under the %.1f GB floor of the %s arm' % (
                chip, after[chip][1], floor, 'audited' if audited else 'audits-off'))
    return problems


def judge(env, text, smoke=None):
    """(problems, facts) of one arm's container log against its profile env. `facts` is small (counts) so a smoke check can print it."""
    env = env or {}
    facts = markers.scan(lines_of(text))
    summary = dict(markers=facts['markers'], rebinds=len(facts['rebinds']), refused=len(facts['refused']), failed=len(facts['failed']),
                   unparked=len(facts['unparked']), reparked=len(facts['reparked']), fallbacks=len(facts['refused']) + len(facts['failed']))
    problems = []
    if not parked_on(env):
        seen = sorted(set(key for key in ('built', 'stopped', 'warm', 'ballast', 'negative', 'rebinds', 'refused', 'failed', 'peaks', 'digests', 'programs',
                                          'unparked', 'reparked', 'single_rebuilt', 'single_kept', 'ladder', 'instance', 'off', 'book_quads', 'released',
                                          'hold_ages', 'costs') if facts.get(key)))
        if seen:
            problems.append('the profile leaves QWEN_FAST_PARKED_ENGINES off, but the server log carries parked markers (%s): this is not the '
                            'profile\'s path' % ', '.join(seen))
        if any(row.get('kind') for row in facts['sticky']):
            problems.append('the profile leaves QWEN_FAST_PARKED_ENGINES off, but its sticky lines carry kind=: this is not the profile\'s path')
        return problems, summary
    count = seats(env)
    audit = str(env.get('QWEN_FAST_PARKED_AUDIT', '0')) == '1'
    drafts = str(env.get('QWEN_FAST_PARKED_DRAFTS', '0')) == '1'
    negative = env.get('QWEN_FAST_PARKED_NEGATIVE') or None
    fault = env.get('QWEN_FAST_PARKED_FAULT') or None
    off_expected = bool(facts['off'])
    summary.update(seats=count, audit=audit, drafts=drafts, negative=negative, fault=fault)

    # attach
    if not facts['built']:
        problems.append('no "%s" line: the parked set was never built (the flag did not reach the attach)' % markers.BUILT.strip())
    for entry in facts['built']:
        if entry['k'] != count or entry['slots'] != count:
            problems.append('the parked set built %d of %d engines, not %d of %d' % (entry['k'], entry['slots'], count, count))
    for entry in facts['stopped']:
        problems.append('the parked build stopped at k=%d of %d, short of %s (free=%d largest_free=%d need=%d)' % (
            entry['k'], entry['slots'], entry['short'], entry['free'], entry['largest_free'], entry['need']))
    if not facts['warm']:
        problems.append('no "%s" line: slot 0\'s drafter was never warmed at 2048' % markers.WARM.strip())
    for entry in facts['warm']:
        if entry['rows'] != 2048:
            problems.append('the drafter warm ran at P=%d, not 2048: the publish prewarm skips below 2048' % entry['rows'])
    if facts['warm'] and not any(value > 0 for value in facts['prewarm']):
        problems.append('no publish prewarm with count > 0 (%s): the first request after a restart would compile the publication programs' % (
            'skipped at history_rows=%s' % facts['prewarm_skipped'] if facts['prewarm_skipped'] else 'no prewarm line'))
    problems += ledger_problems(facts, count, audit)

    # rebinds
    for entry in facts['rebinds']:
        if entry['generation'] < 1:
            problems.append('a rebind at generation %d (request %s): the generation moves on every rebind' % (entry['generation'], entry['request']))
        cold = capture_widths(entry['prompt'], entry['capacity'], VERIFIER_ROWS, entry['budget'] - 1)
        if entry['widths'] != cold and negative != 'widths':
            problems.append('request %s was rebound with widths %s, not the cold engine\'s %s for P=%d budget=%d (R4)' % (
                entry['request'], ','.join(map(str, entry['widths'])), ','.join(map(str, cold)), entry['prompt'], entry['budget']))
    summary['width_mismatches'] = sum(1 for entry in facts['rebinds'] if entry['widths'] != capture_widths(
        entry['prompt'], entry['capacity'], VERIFIER_ROWS, entry['budget'] - 1))
    seen_programs = [row for row in facts['programs'] if row['before'] is not None and row['after'] is not None]
    for row in seen_programs[1:]:
        if row['after'] != row['before']:
            problems.append('a rebind compiled %d program(s) (slot %d, %d to %d): the S8 hang class (a program first compiled after a block\'s first '
                            'replay)' % (row['after'] - row['before'], row['slot'], row['before'], row['after']))

    # fallbacks and faults
    expected_refused = 1 if fault == 'rebind' else 0
    if len(facts['refused']) != expected_refused:
        problems.append('%d rebind refusals (%s), expected %d' % (len(facts['refused']), fault or 'no fault injected', expected_refused))
    if facts['failed']:
        problems.append('%d rebinds failed after their first device write (%s): never expected' % (
            len(facts['failed']), '; '.join('slot %d: %s' % (row['slot'], row['reason'][:80]) for row in facts['failed'][:3])))
    for row in facts['refused']:
        if fault == 'rebind' and FAULT_REBIND_REASON not in row['reason']:
            problems.append('the refusal of %s names %r, not the injected fault' % (row['request'], row['reason'][:80]))
    unexpected = [row for row in facts['unparked'] if not (
        (fault == 'park' and row['reason'] == FAULT_PARK_REASON) or (fault == 'rebind' and FAULT_REBIND_REASON in row['reason'])
        or (off_expected and 'kill switch' in row['reason'].lower()) or (off_expected and 'latched' in row['reason'].lower()))]
    if unexpected:
        problems.append('%d slots unparked (%s): the parked set is expected to serve every request' % (
            len(unexpected), '; '.join('slot %d: %s' % (row['slot'], row['reason'][:80]) for row in unexpected[:3])))
    if fault in ('park', 'rebind'):
        injected = [row for row in facts['unparked'] if row['reason'] == FAULT_PARK_REASON or FAULT_REBIND_REASON in row['reason']]
        if len(injected) != 1:
            problems.append('the %s fault unparked %d slots, expected exactly one' % (fault, len(injected)))
        elif not any(row['slot'] == injected[0]['slot'] for row in facts['reparked']):
            problems.append('the faulted slot %d was never re-parked: it serves cold builds for good' % injected[0]['slot'])

    # audit
    if audit and negative not in ('carry', 'pages') and facts['rebinds']:
        if len(facts['digests']) < len(facts['rebinds']):
            problems.append('%d rebinds and %d digest lines under QWEN_FAST_PARKED_AUDIT=1: a rebind went unaudited' % (
                len(facts['rebinds']), len(facts['digests'])))
        for row in facts['digests']:
            if not row['equal'] or not row['slot0']:
                problems.append('a rebind digest of slot %d is not clean (equal=%d slot0=%d)' % (row['slot'], row['equal'], row['slot0']))
        if negative != 'drafter' and any(not row['zeroed'] for row in facts['digests']):
            problems.append('a rebind digest read no re-zeroed sample')
    if not audit and facts['digests']:
        problems.append('digest lines without QWEN_FAST_PARKED_AUDIT=1: this is not the profile\'s path')

    # instance state, the book, holds, markers
    for row in facts['instance']:
        problems.append('slot %d kept per-request state on a parked object: %s' % (row['slot'], row['problem'][:120]))
    if drafts:
        blocks = {}
        for row in facts['book_quads']:
            blocks[row['slots']] = blocks.get(row['slots'], 0) + 1
        for slots, count_seen in sorted(blocks.items()):
            if count_seen > 1 and not facts['unparked'] and not off_expected:
                problems.append('the book\'s quad over slots %s was captured %d times: it is captured once per block per process' % (slots, count_seen))
        if facts['released']:
            problems.append('%d "released parked" lines under QWEN_FAST_PARKED_DRAFTS=1: the book\'s traces are the slots\' for the process' % len(facts['released']))
    else:
        if facts['book_quads']:
            problems.append('draft book lines without QWEN_FAST_PARKED_DRAFTS=1')
    for row in facts['hold_ages']:
        if row['ms'] > HOLD_AGE_LIMIT_MS:
            problems.append('a DRAM hold of %s lasted %d ms (limit %d): the hold did not clear without a drain' % (row['request'], row['ms'], HOLD_AGE_LIMIT_MS))
    rebound_ids = {row['request'] for row in facts['rebinds']}
    for row in facts['sticky']:
        if row['kind'] is None:
            problems.append('the sticky line of %s carries no kind= under the parked profile' % row['request'])
        elif (row['kind'] == 'rebind') != (row['request'] in rebound_ids or any(row['request'] in request for request in rebound_ids)):
            problems.append('the sticky line of %s says kind=%s but its rebind line %s' % (
                row['request'], row['kind'], 'is present' if row['request'] in rebound_ids else 'is missing'))
    summary.update(width_rows=len(facts['rebinds']), built=[entry['k'] for entry in facts['built']], book_quads=len(facts['book_quads']),
                   holds=len(facts['hold_ages']), ladder=len(facts['ladder']), programs=len(facts['programs']))
    return problems, summary


# -- two arms --------------------------------------------------------------------------------------------------------------------

def accepted_sequences(text):
    """[(request id, [(path, prefix, emitted), ...])] in order of first appearance: every [PACKED], [SEQUENTIAL] and [SEQ-PUBLISH] round of every request."""
    order, rounds = [], {}
    for match in PATH_LINE.finditer(text or ''):
        kind, request, rest = match.groups()
        if kind == 'SEQ-PUBLISH' and not rest.startswith('rows='):
            continue
        fields = {}
        for name, value in PATH_FIELD.findall(rest):
            fields.setdefault(name, None if value == 'n/a' else int(value))
        if request not in rounds:
            order.append(request)
            rounds[request] = []
        rounds[request].append(('P' if kind == 'PACKED' else 'S', fields.get('prefix'), fields.get('emitted')))
    return [(request, rounds[request]) for request in order]


def acceptance(text):
    """(acceptance, rounds): (mean emitted per round - 1) / 15 over every round of every request, or (None, 0)."""
    emitted = [row[2] for _, sequence in accepted_sequences(text) for row in sequence if row[2] is not None]
    if not emitted:
        return None, 0
    return (sum(emitted) / float(len(emitted)) - 1.0) / PROPOSALS, len(emitted)


def equivalence(control_text, other_text, solo=True):
    """(problems, shortfalls, facts): solo, the two arms' requests (matched by order of first appearance) draft identical accepted-prefix sequences;
    concurrent (solo False), the mean acceptance is within ACCEPTANCE_TOLERANCE, absolute."""
    problems, shortfalls = [], []
    left, right = accepted_sequences(control_text), accepted_sequences(other_text)
    facts = dict(requests=(len(left), len(right)))
    if not left or not right:
        shortfalls.append('an arm carries no round lines ([SEQ-PUBLISH] or [PACKED]): its drafting is unseen')
        return problems, shortfalls, facts
    if solo:
        if len(left) != len(right):
            shortfalls.append('the arms ran %d and %d requests with round lines: they cannot be matched' % (len(left), len(right)))
        for index, ((_, a), (_, b)) in enumerate(zip(left, right)):
            if a != b:
                first = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
                problems.append('request %d drafts differently: accepted-prefix sequences differ at round %d (%s | %s)' % (
                    index, first, a[first] if first < len(a) else None, b[first] if first < len(b) else None))
        facts['identical'] = not problems
        return problems, shortfalls, facts
    one, rounds_one = acceptance(control_text)
    two, rounds_two = acceptance(other_text)
    facts.update(control=None if one is None else round(one, 4), other=None if two is None else round(two, 4), rounds=(rounds_one, rounds_two))
    if one is None or two is None:
        shortfalls.append('an arm has no full-draft acceptance: the concurrent drafting is unjudged')
    elif abs(two - one) > ACCEPTANCE_TOLERANCE:
        problems.append('concurrent acceptance %.4f against %.4f on the same script (%+.4f, limit %.2f absolute)' % (two, one, two - one, ACCEPTANCE_TOLERANCE))
    return problems, shortfalls, facts


def negative_verdict(kind, token_mismatches, control_text, negative_text, negative_facts=None):
    """(problems, facts) of a negative-control arm against the flag-off arm of its script. `token_mismatches` is what parked_compare found comparing
    their answers (a list; empty: every token equal). A control that does not do what it exists to do says the gate could not have seen the breakage it
    is there for."""
    facts = dict(kind=kind, token_mismatches=len(token_mismatches))
    if kind in ('carry', 'pages'):
        if not token_mismatches:
            return ['the %s control kept every token: an engine that %s would have passed the exactness gates' % (
                kind, 'never saved its carry' if kind == 'carry' else 'kept the previous request\'s page tables')], facts
        return [], facts
    if kind == 'drafter':
        problems = []
        if token_mismatches:
            problems.append('the drafter control changed tokens (%s): greedy verification should keep them, so the control is not the drafter-only '
                            'break it is meant to be' % token_mismatches[0])
        judged, shortfalls, detail = equivalence(control_text, negative_text)
        facts['equivalence'] = detail
        if shortfalls:
            problems.append('the drafter control could not be judged for equivalence: %s' % '; '.join(shortfalls))
        elif not judged:
            problems.append('the drafter control drafted exactly like the flag-off arm: skipping the bank rezero and reseed is invisible to the '
                            'equivalence judge, so a broken drafter rebind would pass')
        return problems, facts
    if kind == 'widths':
        scanned = markers.scan(lines_of(negative_text))
        wrong = sum(1 for entry in scanned['rebinds'] if entry['widths'] != capture_widths(
            entry['prompt'], entry['capacity'], VERIFIER_ROWS, entry['budget'] - 1))
        facts['width_mismatches'] = wrong
        if not wrong:
            return ['the widths control rebound every request with a cold engine\'s widths: R4\'s restriction was never off, or no budget under four ran'], facts
        return [], facts
    raise ValueError('unknown negative control %r' % (kind,))


def paired_turns(control_users, parked_users):
    """The ER5 report: [(index, control turn, parked turn)] and its summary - per turn the wall time and the longest gap of each arm, paired by index.
    Never a verdict: the adoption rule is the owner's (docs/tp4-engine-reuse.md)."""
    rows, ratios, gaps = [], [], []
    for index, (a, b) in enumerate(zip(control_users, parked_users)):
        if not isinstance(a, dict) or not isinstance(b, dict) or 'error' in a or 'error' in b:
            continue
        wall_a, wall_b = (a.get('ended_at') or 0) - (a.get('started_at') or 0), (b.get('ended_at') or 0) - (b.get('started_at') or 0)
        rows.append(dict(turn=index, control_wall_s=round(wall_a, 2), parked_wall_s=round(wall_b, 2), control_gap_s=a.get('longest_gap_s'),
                         parked_gap_s=b.get('longest_gap_s'), control_ttft_s=a.get('ttft'), parked_ttft_s=b.get('ttft')))
        if wall_a > 0 and wall_b > 0:
            ratios.append(wall_b / wall_a)
        if a.get('longest_gap_s') and b.get('longest_gap_s'):
            gaps.append(b['longest_gap_s'] / a['longest_gap_s'])
    ratios.sort()
    gaps.sort()
    summary = dict(turns=len(rows), wall_ratio_median=round(ratios[len(ratios) // 2], 3) if ratios else None,
                   gap_ratio_median=round(gaps[len(gaps) // 2], 3) if gaps else None,
                   parked_faster=sum(1 for ratio in ratios if ratio < 1.0), total=len(ratios))
    return rows, summary
