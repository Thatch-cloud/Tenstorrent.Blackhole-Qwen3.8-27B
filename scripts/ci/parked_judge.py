"""Stage E's gate judgements (QWEN_FAST_PARKED_ENGINES): what c2_serving_gate's parked plans decide, as functions of
arm reports and logs, where a CPU test can hold them. Stdlib and real_text_compare only, Python 3.7 syntax: the gate
runs on the rig host.

THE RUNGS (design section 9, G-E1 (a)). One solo chain on slot 0 over RUNGS prompt lengths with the budgets
CHAIN_BUDGETS (every one of 1, 2, 3, 4, 5, 16, 256 and 4096 appears) and ignore_eos on the even rungs, sent in three
orders - descending (123,136 first), ascending and a fixed shuffle - each held bitwise against the same chain on the
flag-off twin. The design's rungs 1, 2, 15, 16 and 17 cannot be sent through this harness: a real-text prompt below
real_text_prompts.COMPACT_MIN_TARGET (45 tokens) has no room for its own template, so the chain starts at 63 and
those five stay covered by the CPU tests' host checks only.

DRAFTER EQUIVALENCE (G-E1 (g)). Greedy verification emits only the target's argmax, so no token gate can see a wrong
drafter rebind (stale banks, a wrong history, a stale pair or quad trace): the emitted text is the same and only the
acceptance moves. Two arms of one script must therefore draft alike: solo, every user's per-round accepted-prefix
sequence - the path, the committed prefix and the emitted count of each [SEQ-PUBLISH] and [PACKED] line, as
acceptance_report.path_records reads them - is identical between the parked and the flag-off arm; concurrent, the
mean acceptance over the same script stays within ACCEPTANCE_TOLERANCE (acceptance = the share of the 15 proposals a
round accepts, (mean emitted - 1) / 15, absolute). The negative control that skips the drafter reseed keeps every
token and MUST fail this judge, and the one that skips the carry MUST change tokens: negative_verdict.
"""
import re

import real_text_compare

# G-E1 (a): the prompt lengths, descending (123,136 first: the longest replay span past capture), and their budgets.
RUNGS = (123136, 65536, 32768, 4096, 2049, 2048, 2047, 129, 128, 127, 65, 64, 63)
CHAIN_BUDGETS = (256, 16, 4096, 3, 5, 1, 2, 4, 256, 16, 4096, 3, 5)
BUDGET_SET = (1, 2, 3, 4, 5, 16, 256, 4096)
# The shuffled order: a fixed permutation of 0..len(RUNGS) - 1 (a literal, so 3.7 and 3.11 send the same order).
SHUFFLE = (7, 2, 11, 0, 9, 4, 12, 1, 6, 10, 3, 8, 5)
# G-E1 (g)'s mixed script: eight users on the four seats, every one of them decoding past 2048 of history.
DRAFTER_LENGTHS = (4096, 2049, 20000, 2048, 8192, 60000, 129, 32768)
DRAFTER_BUDGET = 256
PROPOSALS = 15
ACCEPTANCE_TOLERANCE = 0.02
# G-E3 (i): the idle allocation after a drain returns to the attach's within 16 MB per chip.
IDLE_RETURN_GB = 0.016
# The trace region's use may move by the pair, quad and single traces the region's headroom carries (design 2.1: 62 MB).
TRACE_SPREAD_GB = 0.062
TOKEN_POLICY_OK = 'IDENTICAL'


def chain(order='descending'):
    """(prompt lengths, budgets, ignore_eos users, the --start-order permutation or None) of the G-E1 solo chain in
    an order. The lengths list, and so every user's prompt (a window of the corpus by index), is the same in every
    order: only the order the requests are sent in changes."""
    count = len(RUNGS)
    if order == 'descending':
        permutation = None
    elif order == 'ascending':
        permutation = list(reversed(range(count)))
    elif order == 'shuffled':
        permutation = list(SHUFFLE)
    else:
        raise ValueError('unknown chain order %r' % (order,))
    return list(RUNGS), list(CHAIN_BUDGETS), [index for index in range(count) if index % 2 == 0], permutation


def churn_script(arm, users, longest, seed=20260930):
    """A deterministic random script for one G-E3 (i) churn arm: `users` prompt lengths and budgets drawn from a fixed
    linear congruential generator (no random module: the same draws on every Python), every user ignore_eos. Lengths
    are weighted to the short and the middle with a tail up to `longest`."""
    state = (seed + 7919 * (arm + 1)) & 0x7fffffff

    def draw(limit):
        nonlocal state
        state = (state * 1103515245 + 12345) & 0x7fffffff
        return (state >> 8) % limit

    pool = [60, 255, 1536, 2047, 2048, 2049, 4096, 4096, 8192, 8192, 16384, 30000, 60000, 90000, longest]
    lengths, budgets = [], []
    for _ in range(users):
        lengths.append(min(pool[draw(len(pool))], longest))
        budgets.append((16, 64, 128, 256, 512)[draw(5)])
    return lengths, budgets


def paths_of(report, user):
    """A report's path records for one user (report['s2']['paths']), or None."""
    return real_text_compare.report_paths(report, user)


def accepted_sequence(records):
    """One user's rounds as (path, prefix, emitted) - the accepted-prefix sequence (no position: a chained one is
    the same evidence and a logged one is noise)."""
    return [(record['path'], record['prefix'], record['emitted']) for record in records or ()]


def compare_sequences(reference, other, users, skip=()):
    """Per-round accepted-prefix sequences of two arms of one script, user by user: dict(identical, users=[...]),
    each differing user's first differing round (its index and both records, None past an end), `missing` the users
    either arm carries no path records for. Identical only with every user present and equal."""
    entries, missing = [], []
    for user in range(users):
        if user in skip:
            continue  # a request that ends at its first token has no round to record
        left, right = paths_of(reference, user), paths_of(other, user)
        if left is None or right is None:
            missing.append(user)
            continue
        a, b = accepted_sequence(left), accepted_sequence(right)
        entry = dict(user=user, rounds_reference=len(a), rounds_other=len(b), identical=a == b)
        if a != b:
            index = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
            entry['first_difference'] = dict(round=index, reference=a[index] if index < len(a) else None,
                                             other=b[index] if index < len(b) else None)
        entries.append(entry)
    return dict(identical=bool(entries) and not missing and all(entry['identical'] for entry in entries),
                users=entries, missing=missing)


def acceptance(report):
    """(acceptance, rounds) of an arm: the share of the 15 proposals a full-draft round accepts, (mean emitted - 1) /
    15, over every user's full-draft rounds weighted by their count; None without any."""
    rounds = total = 0
    for entry in ((report or {}).get('acceptance') or {}).get('users') or ():
        full = entry.get('full_draft') or {}
        count, mean = full.get('rounds') or 0, full.get('mean_emitted')
        if count and mean is not None:
            rounds += count
            total += count * mean
    if not rounds:
        return None, 0
    return (total / rounds - 1.0) / PROPOSALS, rounds


def acceptance_delta(reference, other):
    """dict(reference, other, delta, rounds) of two arms' acceptance, or None when either has none."""
    a, rounds_a = acceptance(reference)
    b, rounds_b = acceptance(other)
    if a is None or b is None:
        return None
    return dict(reference=round(a, 4), other=round(b, 4), delta=round(b - a, 4), rounds=(rounds_a, rounds_b))


def equivalence(solo_reference, solo_other, live_reference=None, live_other=None, users=None, skip=()):
    """The drafter-equivalence judge: (problems, shortfalls, facts). Solo: identical accepted-prefix sequences for
    every user. Concurrent (when both live arms are given): acceptance within ACCEPTANCE_TOLERANCE, absolute."""
    problems, shortfalls = [], []
    users = users if users is not None else len(((solo_reference or {}).get('streams')) or ())
    solo = compare_sequences(solo_reference, solo_other, users, skip)
    if solo['missing']:
        shortfalls.append('no path records for users %s: their drafting is unseen (the arms did not log [SEQ-PUBLISH] '
                          'or [PACKED] lines)' % solo['missing'])
    for entry in solo['users']:
        if not entry['identical']:
            problems.append('user %d drafts differently: accepted-prefix sequences differ at round %s (%s | %s)' % (
                entry['user'], entry['first_difference']['round'], entry['first_difference']['reference'],
                entry['first_difference']['other']))
    facts = dict(solo=solo)
    if live_reference is not None and live_other is not None:
        delta = acceptance_delta(live_reference, live_other)
        facts['live'] = delta
        if delta is None:
            shortfalls.append('a concurrent arm has no full-draft acceptance: the concurrent drafting is unjudged')
        elif abs(delta['delta']) > ACCEPTANCE_TOLERANCE:
            problems.append('concurrent acceptance %.4f against %.4f on the same script (%+.4f, limit %.2f absolute)' % (
                delta['other'], delta['reference'], delta['delta'], ACCEPTANCE_TOLERANCE))
    return problems, shortfalls, facts


def token_verdicts(reference, other, relaxation=None):
    """(users' verdicts, problems) of two arms' texts under the strict policy: every user IDENTICAL, else a problem
    naming the user and the first divergence."""
    compared = real_text_compare.policy_pass(other, reference)
    problems = []
    for user in compared['users']:
        if user['verdict'] != TOKEN_POLICY_OK:
            problems.append('user %s is %s at character %s (%s)' % (
                user['user'], user['verdict'], user.get('first_divergence'), user.get('reason') or user.get('detail')))
    return [user['verdict'] for user in compared['users']], problems


def negative_verdict(kind, reference, negative, solo_reference=None):
    """(problems, facts) of a negative-control arm against the flag-off arm of its script. 'carry' (no save_carry at
    the rebind) must change tokens for at least one user; 'drafter' (no bank rezero or reseed) must keep every token
    and must FAIL the drafter-equivalence judge. A control that does not do what it exists to do says the gate could
    not have seen the breakage it is there for."""
    verdicts, differences = token_verdicts(reference, negative)
    facts = dict(kind=kind, users=verdicts)
    if kind == 'carry':
        if not differences:
            return ['the carry control kept every token: an engine that never saved its carry would have passed the '
                    'exactness gates'], facts
        return [], facts
    if kind == 'drafter':
        problems = ['the drafter control changed tokens (%s): greedy verification should keep them, so the control is '
                    'not the drafter-only break it is meant to be' % differences[0]] if differences else []
        judged, shortfalls, detail = equivalence(reference, negative)
        facts['equivalence'] = detail
        if shortfalls:
            problems.append('the drafter control could not be judged for equivalence: %s' % '; '.join(shortfalls))
        elif not judged:
            problems.append('the drafter control drafted exactly like the flag-off arm: skipping the bank rezero and '
                            'reseed is invisible to the equivalence judge, so a broken drafter rebind would pass')
        return problems, facts
    raise ValueError('unknown negative control %r' % (kind,))


LOG_COUNTS = (('pair_fallbacks', re.compile(r'\[PACKED-PROPOSE\] pair=\[[^\]\n]*\] fallback=')),
              ('history_exceeded', re.compile(r'Committed history exceeds prepared request contexts')),
              ('quad_disabled', re.compile(r'\[PINDIAG\] quad draft disabled')),
              ('kv_shared', re.compile(r'KV_SHARED')))


def log_counts(log_text):
    """{name: count} of the lines G-E1 wants zero of: pair fallbacks, 'Committed history exceeds', quad disables and
    KV_SHARED."""
    return dict((name, len(pattern.findall(log_text or ''))) for name, pattern in LOG_COUNTS)


def count_problems(counts, label='arm'):
    names = dict(pair_fallbacks='pair fallbacks', history_exceeded='"Committed history exceeds" errors',
                 quad_disabled='quad disables', kv_shared='KV_SHARED lines')
    return ['%s: %d %s (G-E1 asks for none)' % (label, counts.get(name) or 0, text)
            for name, text in sorted(names.items()) if counts.get(name)]


def idle_return(drift, limit=IDLE_RETURN_GB):
    """(problems, shortfalls) of the idle allocation after the streams against the attach's, {chip: GB}."""
    if drift is None:
        return [], ['the ledger\'s idle readings are missing: the return to the P7p baseline is unjudged']
    grown = dict((chip, value) for chip, value in sorted(drift.items()) if abs(value) > limit)
    if grown:
        return ['idle allocation after the drain differs from the attach\'s by %s GB per chip (limit %s)' % (
            grown, limit)], []
    return [], []


def trace_spread(region, limit=TRACE_SPREAD_GB):
    """(problems, shortfalls) of the trace region's use across the run: it may move by the traces its headroom
    carries and no more (min_used_gb and max_used_gb over W6d's points)."""
    region = region or {}
    if not region.get('readings') or region.get('min_used_gb') is None or region.get('max_used_gb') is None:
        return [], ['no trace-region readings: the constant trace use is unjudged']
    spread = region['max_used_gb'] - region['min_used_gb']
    if spread > limit:
        return ['trace-region use moved by %.4f GB (%.4f to %.4f), past the %.3f GB its headroom carries' % (
            spread, region['min_used_gb'], region['max_used_gb'], limit)], []
    return [], []


def ballast_advice(p7p_free_bytes, environ=None):
    """The ballast (bytes per chip) that leaves the parked arrival at the boundary of its need less the reserve: the
    P7p free reading less (the prefill's transient + the rebind's peak + the stranded bytes), so what the arrival
    really uses is all that is left and the reserve the admission asks on top is gone. The single is held at idle
    (S = 0). None without a reading. serving_prefill_admission's terms, read at call time."""
    if p7p_free_bytes is None:
        return None
    import serving_prefill_admission as admission

    transient = admission.PREFILL_TRANSIENT_BYTES
    rebind = admission.PARKED_REBIND_BYTES
    stranded = admission.STRANDED_BYTES
    return max(int(p7p_free_bytes) - (transient + rebind + stranded), 0)
