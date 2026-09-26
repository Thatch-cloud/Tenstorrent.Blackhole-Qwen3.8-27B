"""The prefix-reuse gates' judgement (design 2.0.4 and 2.2), pure: no network, no docker, no clock.

1. The ORACLE: what Q every request should get, from the token ids the server returned and the
   design's rules alone - an independent model of the scheduler graft, not a copy of it. vLLM's hit
   h is the longest run of 64-token blocks that some earlier request of the SAME cache_salt
   published, capped at num_tokens - 1 (kv_cache_manager.py:206-246); under the cap a salted request
   publishes only its blocks below floor2048(prompt) (decode and tail blocks never); Q is the largest
   2048-multiple <= h with a checkpoint whose tokens are the request's (the trim); a request captures
   at floor2048(prompt) when that is past Q, and at the gap boundary floor2048(h) when that is at
   least one chunk past Q; checkpoints live in a byte-bounded LRU touched by every grant. An unsalted
   request gets nothing and publishes nothing (fail closed); the kill switch clears the checkpoints
   and turns grants and publishing off until the engine restarts; reset_prefix_cache clears both.
   Sequential phases must match it exactly (a Q above it is a grant the design forbids, a Q below
   it a lost hit, an observed h above it a block published past the cap, a capture plan other than
   its own a planning bug); where requests overlap or the pool evicts, it is an upper bound. It
   judges a request's FIRST admission: a preempted request's re-admission re-prefills its own
   output, which no oracle of prompts predicts.
2. COMPARISON: a hit against its cold twin - the same messages, a fresh cache_salt (a miss on other
   physical pages, design 2.0.4 "How the gate measures it") - in full: prompt token ids, every
   output token id, the finish reason (and the text, reasoning and tool calls when the server
   returned no token ids); and, byte for byte, the digests the model prints for each prefill row
   (the end-of-prefill GDN state and the last position's logits).
3. THE RE-RUN POLICY (PLAN 2.2 item 5's flip policy, applied to a pair): a first divergence sends a
   second cold run (a fresh salt) and a second hit at the SAME Q (the conversation's salt history
   replayed under a fresh salt, prefix_replay.Driver.pair). The two cold runs disagreeing is
   UNSTABLE (the engine is not run-to-run deterministic for that prompt, itself a finding); when
   they agree, the hit's divergence is DIVERGED - a FAIL - whatever the second hit does (whether it
   diverged again, and where, is reported). Hits that ran CONCURRENTLY are compared with cold twins
   run alone; a divergence there also runs a batch control (the same messages again, all at once,
   fresh salts): a concurrent cold run that differs from its solo run too means the engine's bytes
   depend on the batch, and the pair is NOT_COMPARABLE rather than DIVERGED (concurrent_verdict).
   settle() then excuses a DIVERGED pair whose hit or cold run vLLM preempted and resumed (more
   than one admission): the resumed prefill re-reads its own output, not a cold equivalent.
4. MARKERS: every request's [PREFIX] rows, grants and audit digests, matched by the harness's
   X-Request-Id tag (prefix_markers.request_tag), else by the request's log window and prompt length.

Stdlib only, Python 3.7 syntax: it runs on the rig host.
"""

import hashlib
from array import array
from collections import Counter, OrderedDict

CHUNK = 2048
BLOCK = 64
# One checkpoint, both chips: 48 layers x (fp32 rec [1,24,128,128] + bf16 carry [1,3,5120]) (design 2.0.3).
CHECKPOINT_NBYTES = 2 * 48 * (24 * 128 * 128 * 4 + 3 * 5120 * 2)
DEFAULT_STORE_GIB = 8.0
VERDICT_ORDER = ('FAIL', 'INFRA', 'RERUN', 'NOT_COMPARABLE', 'UNSTABLE', 'NOT_EXERCISED')


def floor_chunk(tokens):
    return (int(tokens) // CHUNK) * CHUNK


def worst(verdicts):
    verdicts = [verdict for verdict in verdicts if verdict]
    for verdict in VERDICT_ORDER:
        if verdict in verdicts:
            return verdict
    return 'PASS' if verdicts else 'FAIL'


def store_entries(store_gib=DEFAULT_STORE_GIB):
    return int(float(store_gib) * (1 << 30)) // CHECKPOINT_NBYTES


def prefix_digests(tokens, step=BLOCK):
    """digests[k-1] identifies tokens[0:k*step] (a running sha1 over the token ids)."""
    digest, out = hashlib.sha1(), []
    whole = len(tokens) - len(tokens) % step
    for start in range(0, whole, step):
        digest.update(array('q', tokens[start:start + step]).tobytes())
        out.append(digest.hexdigest())
    return out


def token_sha(tokens):
    return hashlib.sha256(array('q', list(tokens or ())).tobytes()).hexdigest()


class Oracle(object):
    """The expected hit, trim and captures of each admitted request, in admission order."""

    def __init__(self, capacity=None):
        self.capacity = store_entries() if capacity is None else int(capacity)
        self.published = {}
        self.checkpoints = OrderedDict()
        self.killed = False

    def admit(self, salt, tokens):
        tokens = list(tokens)
        if not salt or self.killed:
            return dict(h=0, q=0, plan=[], published=0)
        digests = prefix_digests(tokens)
        published = self.published.setdefault(salt, set())
        usable = max(0, len(tokens) - 1) // BLOCK
        blocks = 0
        while blocks < min(usable, len(digests)) and digests[blocks] in published:
            blocks += 1
        h = blocks * BLOCK
        q = 0
        for k in range(h // CHUNK, 0, -1):
            key = (salt, digests[k * CHUNK // BLOCK - 1])
            if key in self.checkpoints:
                q = k * CHUNK
                self.checkpoints.move_to_end(key)
                break
        plan = set()
        boundary = floor_chunk(len(tokens))
        if boundary > q:
            plan.add(boundary)
        if floor_chunk(h) - q >= CHUNK:
            plan.add(floor_chunk(h))
        for position in sorted(plan):
            self._put((salt, digests[position // BLOCK - 1]))
        cap = floor_chunk(len(tokens)) // BLOCK
        published.update(digests[0:cap])
        return dict(h=h, q=q, plan=sorted(plan), published=cap * BLOCK)

    def published_tokens(self, salt, tokens):
        """How much of `tokens` this salt has published (vLLM's raw hit before the num_tokens - 1
        cap), with no admission: nothing changes."""
        digests = prefix_digests(list(tokens))
        published = self.published.get(salt) or set()
        blocks = 0
        while blocks < len(digests) and digests[blocks] in published:
            blocks += 1
        return blocks * BLOCK

    def _put(self, key):
        if self.capacity <= 0:
            return
        if key in self.checkpoints:
            self.checkpoints.move_to_end(key)
        else:
            self.checkpoints[key] = True
        while len(self.checkpoints) > self.capacity:
            self.checkpoints.popitem(last=False)

    def kill(self):
        self.killed = True
        self.checkpoints.clear()

    def reset_prefix_cache(self):
        self.published.clear()
        self.checkpoints.clear()


# -- comparison ----------------------------------------------------------------------------------

def first_divergence(a, b):
    a, b = list(a or ()), list(b or ())
    for index in range(min(len(a), len(b))):
        if a[index] != b[index]:
            return index
    return None if len(a) == len(b) else min(len(a), len(b))


def _calls(record):
    return [((call.get('function') or {}).get('name'), (call.get('function') or {}).get('arguments'))
            for call in record.get('tool_calls') or ()]


def compare(cold, hit):
    """IDENTICAL, DIVERGED (with the first differing output token, or character), NOT_COMPARABLE
    (different prompts) or ERROR (either request failed or was cut short)."""
    for label, record in (('cold', cold), ('hit', hit)):
        if record is None or record.get('error') or record.get('aborted') or not record.get('ok'):
            return dict(verdict='ERROR', detail='%s: %s' % (label, (record or {}).get('error')
                                                              or (record or {}).get('aborted') or 'no answer'))
    if cold.get('prompt_sha') != hit.get('prompt_sha') or cold.get('prompt_tokens') != hit.get('prompt_tokens'):
        return dict(verdict='NOT_COMPARABLE', detail='the two requests rendered different prompts')
    if cold.get('token_ids') is not None and hit.get('token_ids') is not None:
        index = first_divergence(cold['token_ids'], hit['token_ids'])
        if index is None and cold.get('finish') == hit.get('finish'):
            return dict(verdict='IDENTICAL', tokens=len(cold['token_ids']))
        return dict(verdict='DIVERGED', token=index, detail='first differing output token %s (cold %d tokens %s, '
                    'hit %d tokens %s)' % (index, len(cold['token_ids']), cold.get('finish'), len(hit['token_ids']),
                                            hit.get('finish')))
    same = (cold.get('content') == hit.get('content') and cold.get('reasoning') == hit.get('reasoning')
            and _calls(cold) == _calls(hit) and cold.get('finish') == hit.get('finish'))
    if same:
        return dict(verdict='IDENTICAL', tokens=cold.get('completion_tokens'), detail='text only: no token ids')
    index = first_divergence((cold.get('reasoning') or '') + (cold.get('content') or ''),
                             (hit.get('reasoning') or '') + (hit.get('content') or ''))
    return dict(verdict='DIVERGED', character=index, detail='text differs at character %s (no token ids)' % index)


def _where(result):
    if result.get('token') is not None:
        return 'token %s' % result['token']
    return 'character %s' % result.get('character')


def pair_verdict(cold, hit, cold_again=None, hit_again=None):
    """The re-run policy over one cold/hit pair and, after a first divergence, its re-run: a second
    cold run, and (sequential pairs) a second hit at the same Q. hit_again may be None."""
    first = compare(cold, hit)
    if first['verdict'] != 'DIVERGED':
        return dict(verdict=first['verdict'], first=first)
    if cold_again is None:
        return dict(verdict='RERUN', first=first)
    colds = compare(cold, cold_again)
    result = dict(first=first, cold_repeat=colds)
    if colds['verdict'] == 'ERROR':
        result.update(verdict='ERROR', reason='the second cold run failed (%s): the divergence at %s is '
                      'unconfirmed' % (colds.get('detail'), _where(first)))
        return result
    if colds['verdict'] != 'IDENTICAL':
        result.update(verdict='UNSTABLE', reason='the two cold runs of the same prompt differ (%s): the engine is '
                      'not deterministic run to run' % colds.get('detail', colds['verdict']))
        return result
    reason = 'the two cold runs agree and the hit differs from them at %s' % _where(first)
    if hit_again is not None:
        second = compare(cold_again, hit_again)
        result['second'] = second
        if second['verdict'] == 'IDENTICAL':
            reason += '; the same-Q re-run hit matched (the divergence did not reproduce)'
        elif second['verdict'] == 'DIVERGED':
            reason += '; the same-Q re-run hit diverged again at %s%s' % (
                _where(second), ' (the same place)' if _where(second) == _where(first) else '')
        else:
            reason += '; the same-Q re-run hit: %s (%s)' % (second['verdict'], second.get('detail'))
    result.update(verdict='DIVERGED', reason=reason)
    return result


def concurrent_verdict(cold, hit, cold_again=None, batch_cold=None):
    """A hit that ran concurrently, against its cold twin run alone: pair_verdict's cold check, then
    the batch control - the same messages run again at the burst's concurrency (fresh salt). A
    control that differs from the solo cold run means the bytes depend on the batch (or the control
    was itself preempted): NOT_COMPARABLE, never DIVERGED."""
    result = pair_verdict(cold, hit, cold_again)
    if result['verdict'] != 'DIVERGED':
        return result
    if batch_cold is None:
        result.update(verdict='RERUN', reason='no batch control ran')
        return result
    control = compare(cold, batch_cold)
    result['batch'] = control
    if control['verdict'] != 'IDENTICAL':
        result.update(verdict='NOT_COMPARABLE', reason='the concurrent cold control differs from the solo cold run '
                      'too (%s): this batch shape changes the bytes, so a concurrent hit cannot be compared with a '
                      'solo cold run' % control.get('detail', control['verdict']))
        return result
    result['reason'] += '; the concurrent cold control matched the solo cold run'
    return result


def admissions(record):
    return ((record or {}).get('markers') or {}).get('admissions') or 0


def settle(pair, index):
    """A pair's final verdict once markers are resolved (index: tag -> record): a DIVERGED pair whose
    hit, cold run, second cold run or batch control vLLM preempted and resumed (admissions > 1) is
    NOT_COMPARABLE, and any pair that is not IDENTICAL names its preempted requests.
    -> the pair (updated in place)."""
    if pair.get('verdict') in ('IDENTICAL', None):
        return pair
    resumed = []
    for tag in (pair.get('hit'), pair.get('cold'), pair.get('cold2'), pair.get('batch')):
        count = admissions(index.get(tag)) if tag else 0
        if count > 1:
            resumed.append('%s (%d admissions)' % (tag, count))
    if resumed:
        if pair['verdict'] == 'DIVERGED':
            pair['verdict'] = 'NOT_COMPARABLE'
        pair['detail'] = ('%s; preempted and resumed: %s - a resumed prefill re-reads its own output, not a cold '
                          'equivalent' % (pair.get('detail'), ', '.join(resumed)))
    return pair


# -- markers onto records ------------------------------------------------------------------------

def resolve(records, scanned):
    """Attach each request's grants, [PREFIX] rows and audit row (prefix_markers.scan) by tag; a
    row without a request id falls back to the one row inside the request's log window whose L is
    its prompt length. The FIRST row is the admission judged against the oracle (q, l); the
    admissions count is the number of rows (a preempted request is re-admitted and prints again);
    `grant` is the first admission's (the grants logged before the second row). Sets
    record['markers'] and returns the records."""
    grants, rows, audits, skipped = {}, {}, {}, {}
    for entry in scanned.get('grants') or ():
        grants.setdefault(entry['tag'], []).append(entry)
    untagged = []
    for entry in scanned.get('rows') or ():
        if entry.get('tag'):
            rows.setdefault(entry['tag'], []).append(entry)
        else:
            untagged.append(entry)
    for entry in scanned.get('audits') or ():
        if entry.get('tag'):
            audits.setdefault(entry['tag'], []).append(entry)
    for entry in scanned.get('capture_skipped') or ():
        if entry.get('tag'):
            skipped.setdefault(entry['tag'], []).append(entry)
    for record in records:
        tag = record.get('tag')
        found = dict(grants=list(grants.get(tag) or ()), rows=list(rows.get(tag) or ()),
                     audit=(audits.get(tag) or [None])[0], skipped=list(skipped.get(tag) or ()), matched='tag')
        if not found['rows'] and untagged and record.get('log_window'):
            start, end = record['log_window']
            window = [entry for entry in untagged if start <= entry['index'] <= end
                      and entry.get('l') == record.get('prompt_tokens')]
            if len(window) == 1:
                found['rows'], found['matched'] = window, 'window'
            elif len(window) > 1:
                found['matched'] = 'ambiguous (%d rows in the window)' % len(window)
        found['rows'].sort(key=lambda entry: entry.get('index', 0))
        row = found['rows'][0] if found['rows'] else None
        found['row'] = row
        found['q'] = row.get('q') if row else None
        found['l'] = row.get('l') if row else None
        found['admissions'] = len(found['rows'])
        found['grant'] = first_grant(found['grants'], found['rows'])
        record['markers'] = found
    return records


def first_grant(grants, rows):
    """The first admission's grant line: the last one logged before the first row; failing that
    (the scheduler's and the model's lines reach the log by different streams) the first one
    logged before the second row."""
    if not grants:
        return None
    if not rows:
        return grants[-1]
    first = rows[0].get('index', 0)
    before = [entry for entry in grants if entry.get('index', 0) < first]
    if before:
        return before[-1]
    if len(rows) > 1:
        second = rows[1].get('index', 0)
        early = [entry for entry in grants if entry.get('index', 0) < second]
        return early[0] if early else None
    return grants[0]


def observed_q(record):
    return ((record.get('markers') or {}).get('q'))


FRESH_ROLES = ('cold', 'capture', 'cold-batch')


def reuse_problems(record, sequential=True):
    """What one request's markers say against the oracle's expectation (record['expected']) and the
    request's own role. -> list of (severity, text): FAIL for a missing or inconsistent row, a grant
    to an unsalted or fresh-salt request, a restore without a grant, a capture nobody planned, or
    (sequential) a Q, h or capture plan other than the oracle's - a grant the design forbids, a
    block published past the cap; LOST for a hit the oracle expected and the engine did not give;
    NOTE for an off-oracle reading where requests overlapped (the oracle is then only an estimate
    of admission order) and for each re-admission of a preempted request."""
    markers = record.get('markers') or {}
    expected = record.get('expected') or {}
    role, tag = record.get('role'), record.get('tag')
    out = []
    if not record.get('ok'):
        return out
    rows = markers.get('rows') or []
    grant, q = markers.get('grant'), markers.get('q')
    prompt = record.get('prompt_tokens')
    if not rows:
        out.append(('FAIL', '%s: no [PREFIX] row for this request (matched %s): the model graft did not report '
                            'its prefill' % (tag, markers.get('matched'))))
    elif markers.get('l') is not None and prompt is not None and markers['l'] != prompt:
        out.append(('FAIL', '%s: [PREFIX] L=%s on its first admission but the prompt is %s tokens' % (
            tag, markers['l'], prompt)))
    for later in rows[1:]:
        # A preempted request resumes by re-prefilling its prompt and the output so far.
        output = record.get('completion_tokens')
        ceiling = (prompt or 0) + output if output is not None else float('inf')
        if prompt is not None and later.get('l') is not None and not prompt <= later['l'] <= ceiling:
            out.append(('FAIL', '%s: a re-admission row has L=%s, outside the prompt (%s) plus the output so far '
                                '(%s)' % (tag, later['l'], prompt, ceiling)))
        else:
            out.append(('NOTE', '%s: preempted and re-admitted at L=%s Q=%s' % (tag, later.get('l'), later.get('q'))))
    for row in rows:
        if row.get('q') is not None and row['q'] % CHUNK:
            out.append(('FAIL', '%s: Q=%d is not a %d-token boundary' % (tag, row['q'], CHUNK)))
        if row.get('q') and row.get('l') is not None and row['q'] >= row['l']:
            out.append(('FAIL', '%s: Q=%d is not below L=%s' % (tag, row['q'], row['l'])))
    granted = sorted(entry['q'] for entry in markers.get('grants') or () if entry.get('q'))
    restored = sorted(row['q'] for row in rows if row.get('q'))
    if Counter(granted) != Counter(restored):
        out.append(('FAIL', '%s: the scheduler granted Q=%s but the model restored Q=%s (a restore without a grant, '
                            'or a grant the model ignored: F2)' % (tag, granted, restored)))
    first = rows[0] if rows else None
    if first is not None:
        captured = sorted(first.get('captured') or ())
        planned = sorted((grant or {}).get('plan') or ())
        stray = sorted(set(captured) - set(planned))
        if stray:
            out.append(('FAIL', '%s: captured %s that the scheduler never planned (plan %s)' % (tag, stray, planned)))
        missed = sorted(set(planned) - set(captured))
        if missed:
            skipped = [entry for entry in markers.get('skipped') or ()]
            out.append(('NOTE' if skipped else 'FAIL', '%s: planned captures %s were not taken%s' % (
                tag, missed, ' (capture skipped: %s)' % '; '.join(str(e.get('reason')) for e in skipped)
                if skipped else ' and no "capture skipped" line says why')))
    if role == 'unsalted':
        if markers.get('grants'):
            out.append(('FAIL', '%s: an unsalted request got a grant (%s): fail-closed tenancy broken' % (
                tag, markers['grants'][0])))
        if restored:
            out.append(('FAIL', '%s: an unsalted request restored Q=%s' % (tag, restored)))
        return out
    if role in FRESH_ROLES and q:
        out.append(('FAIL', '%s: a fresh cache_salt restored Q=%s: salts are not isolated' % (tag, q)))
    if not expected or q is None:
        return out
    above = 'FAIL' if sequential else 'NOTE'
    if q > expected.get('q', 0):
        out.append((above, '%s: Q=%d but the oracle allows at most %d: a grant the design forbids' % (
            tag, q, expected['q'])))
    elif q < expected.get('q', 0):
        out.append(('LOST', '%s: Q=%d where the oracle expected %d: a lost hit' % (tag, q, expected['q'])))
    if grant is not None and grant['h'] > expected.get('h', 0):
        out.append((above, '%s: vLLM hit h=%d past the oracle\'s %d: a block was published past the cap '
                           '(decode or prompt-tail KV)' % (tag, grant['h'], expected['h'])))
    elif grant is not None and grant['h'] < expected.get('h', 0):
        out.append(('LOST', '%s: vLLM hit h=%d under the oracle\'s %d' % (tag, grant['h'], expected['h'])))
    same_hit = grant is None or grant['h'] == expected.get('h', 0)
    if first is not None and 'plan' in expected and q == expected.get('q', 0) and same_hit:
        planned = sorted((grant or {}).get('plan') or ())
        if planned != sorted(expected['plan']):
            out.append((above, '%s: the scheduler planned captures %s where the oracle plans %s' % (
                tag, planned, sorted(expected['plan']))))
    return out


def row_growth(row, previous):
    """How many programs a [PREFIX] row compiled: its own programs - programs_before when the model
    prints both, else the growth since `previous` (the row before it, or None). -> (count or None,
    what was measured)."""
    if row.get('programs') is None:
        return None, 'no programs= field'
    if row.get('programs_before') is not None:
        return row['programs'] - row['programs_before'], 'the row itself'
    if previous is None or previous.get('programs') is None:
        return None, 'no programs_before= field and no row right before it'
    return row['programs'] - previous['programs'], 'since %s' % previous.get('tag')


def program_cache_problems(rows, pairs, first_capture=True, require_hit=True):
    """F3, a restore or a capture must compile nothing. `rows` are every [PREFIX] row of the arm (the
    scan's, in log order), `pairs` the arm's cold/hit pairs. Every hit (Q > 0) whose cold twin (the
    same prompt, so every shape it needs already compiled) ran before it must compile nothing, and so
    must the first row that captured, after an uncaptured row of the same prompt. A row's compiles are
    programs - programs_before when the model prints both (decode steps between two rows may compile
    on their own); without programs_before, only a row right after its twin is measured, against it.
    -> (problems, detail, not measured)."""
    ordered = sorted(rows, key=lambda row: row.get('index', 0))
    position = dict((id(row), place) for place, row in enumerate(ordered))
    first_of = {}
    for row in ordered:
        if row.get('tag') and row['tag'] not in first_of:
            first_of[row['tag']] = row
    problems, missing, measured = [], [], []
    for pair in pairs:
        hit_row, cold_row = first_of.get(pair.get('hit')), first_of.get(pair.get('cold'))
        if hit_row is None or not hit_row.get('q') or cold_row is None:
            continue
        place = position[id(hit_row)]
        if position[id(cold_row)] > place:
            continue
        adjacent = ordered[place - 1] is cold_row
        if hit_row.get('programs_before') is None and not adjacent:
            continue
        grown, what = row_growth(hit_row, ordered[place - 1] if adjacent else None)
        if grown is None:
            problems.append('%s: %s - the program cache across a hit is not measured' % (pair['hit'], what))
            continue
        measured.append(pair['hit'])
        if grown:
            problems.append('the hit %s (Q > 0, after its cold twin %s) compiled %d programs (%s): a restore compiled '
                            'after the traces were parked (F3, the second-request hang)' % (
                                pair['hit'], pair['cold'], grown, what))
    if require_hit and not measured and not problems:
        problems.append('no hit (Q > 0) ran after its cold twin with a measurable program count: the program cache '
                        'across a hit is not measured')
    detail = dict(hits_measured=len(measured), first_hit=measured[0] if measured else None)
    if first_capture:
        capturing = [row for row in ordered if row.get('captured')]
        if not capturing:
            missing.append('no [PREFIX] row captured a checkpoint: the program cache across a capture is not measured')
        else:
            row = capturing[0]
            place = position[id(row)]
            before = ordered[place - 1] if place else None
            detail['first_capture'] = row.get('tag')
            if before is None or before.get('captured') or before.get('l') != row.get('l'):
                missing.append('the first capturing row (%s) does not follow an uncaptured row of the same prompt: '
                               'the program cache across a capture is not measured' % row.get('tag'))
            else:
                grown, what = row_growth(row, before)
                if grown is None:
                    problems.append('the first capturing row %s: %s' % (row.get('tag'), what))
                else:
                    detail.update(capture_compiled=grown, capture_measured=what)
                    if grown:
                        problems.append('the first capture (%s) compiled %d programs (%s): a capture compiled after the '
                                        'traces were parked (F3)' % (row.get('tag'), grown, what))
    return problems, detail, missing


def digest_problems(cold, hit):
    """The model's per-row digests of a hit's first admission against its cold twin's: the
    end-of-prefill GDN state (slot_sha) and the last prompt position's logits (logits_sha).
    -> list of (severity, text): FAIL for differing bytes, NOT_EXERCISED for a missing digest."""
    a = (cold.get('markers') or {}).get('row') or {}
    b = (hit.get('markers') or {}).get('row') or {}
    out = []
    for name, what in (('slot_sha', 'GDN state after prefill'), ('logits_sha', 'last-position logits')):
        if not a.get(name) or not b.get(name):
            out.append(('NOT_EXERCISED', 'no %s on the [PREFIX] row of %s: the %s is not compared' % (
                name, cold.get('tag') if not a.get(name) else hit.get('tag'), what)))
        elif a[name] != b[name]:
            out.append(('FAIL', 'the %s differs between %s and %s (%s %s vs %s)' % (
                what, cold.get('tag'), hit.get('tag'), name, a[name], b[name])))
    return out


def audit_problems(cold, hit):
    """The program-free audit's digests of a hit against its cold twin over the same KV range.
    -> list of (severity, text): FAIL for differing bytes, NOT_EXERCISED for a missing or
    mismatched-range audit."""
    a = (cold.get('markers') or {}).get('audit')
    b = (hit.get('markers') or {}).get('audit')
    if a is None or b is None:
        return [('NOT_EXERCISED', 'no [PREFIX-AUDIT] row for %s' % (cold.get('tag') if a is None else hit.get('tag')))]
    problems = []
    absent = [name for name in ('kv_range', 'kv_sha', 'slot_sha') if not a.get(name) or not b.get(name)]
    if absent:
        return [('NOT_EXERCISED', 'the audit rows of %s / %s carry no %s: nothing is compared' % (
            cold.get('tag'), hit.get('tag'), ', '.join(absent)))]
    if a.get('kv_range') != b.get('kv_range'):
        problems.append(('NOT_EXERCISED', 'the audits cover different KV ranges (%s, %s)' % (
            a.get('kv_range'), b.get('kv_range'))))
    elif a.get('kv_sha') != b.get('kv_sha'):
        problems.append(('FAIL', 'KV %s differs between %s and %s' % (a.get('kv_range'), cold.get('tag'), hit.get('tag'))))
    if a.get('slot_sha') != b.get('slot_sha'):
        problems.append(('FAIL', 'the GDN slot bytes differ between %s and %s' % (cold.get('tag'), hit.get('tag'))))
    return problems


def counter_delta(record, name):
    """A metric's change across one request (record['counters'] = {before, after}), or None."""
    counters = record.get('counters') or {}
    before, after = (counters.get('before') or {}).get(name), (counters.get('after') or {}).get(name)
    if before is None or after is None:
        return None
    return after - before


def raw_hit_per_attempt(record):
    """vLLM's raw hit h (tokens) for one request from vllm:prefix_cache_hits and _queries across it:
    both count once per admission attempt (kv_cache_manager.py:238-244), queries the request's
    num_tokens each time. -> (h per attempt, attempts) or (None, None)."""
    hits, queries = counter_delta(record, 'vllm:prefix_cache_hits'), counter_delta(record, 'vllm:prefix_cache_queries')
    length = record.get('prompt_tokens')
    if hits is None:
        return None, None
    attempts = 1
    if queries and length:
        attempts = max(1, int(round(queries / float(length))))
    return hits / float(attempts), attempts
