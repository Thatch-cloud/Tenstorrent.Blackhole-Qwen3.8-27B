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
   it a lost hit, an observed h above it a block published past the cap); where requests overlap or
   the pool evicts, it is an upper bound.
2. COMPARISON: a hit against its cold twin - the same messages, a fresh cache_salt (a miss on other
   physical pages, design 2.0.4 "How the gate measures it") - in full: prompt token ids, every
   output token id, the finish reason (and the text, reasoning and tool calls when the server
   returned no token ids).
3. THE RE-RUN POLICY (PLAN 2.2 item 5's flip policy, applied to a pair): a first divergence sends
   the same messages once more, cold and hit. The cold pair disagreeing is UNSTABLE (the engine is
   not run-to-run deterministic, itself a finding); the hit diverging again at the same token, or
   byte for byte as before, is DIVERGED (a FAIL); diverging once only, or elsewhere, is UNSTABLE.
4. MARKERS: every request's [PREFIX] row, grant and audit digests, matched by the harness's
   X-Request-Id tag (prefix_markers.request_tag), else by the request's log window and prompt length.

Stdlib only, Python 3.7 syntax: it runs on the rig host.
"""

import hashlib
from array import array
from collections import OrderedDict

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


def pair_verdict(cold, hit, cold_again=None, hit_again=None):
    """The re-run policy over one cold/hit pair and, after a first divergence, its re-run."""
    first = compare(cold, hit)
    if first['verdict'] != 'DIVERGED':
        return dict(verdict=first['verdict'], first=first)
    if cold_again is None or hit_again is None:
        return dict(verdict='RERUN', first=first)
    colds = compare(cold, cold_again)
    second = compare(cold_again, hit_again)
    result = dict(first=first, second=second, cold_repeat=colds)
    if colds['verdict'] != 'IDENTICAL':
        result.update(verdict='UNSTABLE', reason='the two cold runs of the same prompt differ (%s): the engine is '
                      'not deterministic run to run' % colds.get('detail', colds['verdict']))
    elif second['verdict'] == 'IDENTICAL':
        result.update(verdict='UNSTABLE', reason='the hit diverged once and matched on the re-run')
    elif second['verdict'] == 'DIVERGED':
        same_place = (first.get('token') is not None and first.get('token') == second.get('token')) or (
            first.get('character') is not None and first.get('character') == second.get('character'))
        same_bytes = hit.get('token_ids') is not None and hit.get('token_ids') == hit_again.get('token_ids')
        if same_place or same_bytes:
            result.update(verdict='DIVERGED', reason='the hit diverged from the cold run the same way twice')
        else:
            result.update(verdict='UNSTABLE', reason='the hit diverged in both runs, at different places')
    else:
        result.update(verdict=second['verdict'], reason=second.get('detail'))
    return result


# -- markers onto records ------------------------------------------------------------------------

def resolve(records, scanned):
    """Attach each request's grant, [PREFIX] row and audit row (prefix_markers.scan) by tag; a row
    without a request id falls back to the one row inside the request's log window whose L is its
    prompt length. Sets record['markers'] and returns the records."""
    grants, rows, audits = {}, {}, {}
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
    for record in records:
        tag = record.get('tag')
        found = dict(grant=(grants.get(tag) or [None])[-1], rows=list(rows.get(tag) or ()),
                     audit=(audits.get(tag) or [None])[-1], matched='tag')
        if not found['rows'] and untagged and record.get('log_window'):
            start, end = record['log_window']
            window = [entry for entry in untagged if start <= entry['index'] <= end
                      and entry.get('l') == record.get('prompt_tokens')]
            if len(window) == 1:
                found['rows'], found['matched'] = window, 'window'
            elif len(window) > 1:
                found['matched'] = 'ambiguous (%d rows in the window)' % len(window)
        row = found['rows'][-1] if found['rows'] else None
        found['q'] = row.get('q') if row else None
        found['l'] = row.get('l') if row else None
        found['admissions'] = len(found['rows'])
        record['markers'] = found
    return records


def observed_q(record):
    return ((record.get('markers') or {}).get('q'))


def reuse_problems(record, sequential=True):
    """What one request's markers say against the oracle's expectation (record['expected']) and the
    request's own role. -> list of (severity, text): FAIL for a missing or inconsistent row, a grant
    to an unsalted or fresh-salt request, or (sequential) a Q or h above the oracle - a grant the
    design forbids, a block published past the cap; LOST for a hit the oracle expected and the
    engine did not give; NOTE for an above-oracle reading where requests overlapped (the oracle is
    then only an estimate of admission order)."""
    markers = record.get('markers') or {}
    expected = record.get('expected') or {}
    role, tag = record.get('role'), record.get('tag')
    out = []
    if not record.get('ok'):
        return out
    grant, q = markers.get('grant'), markers.get('q')
    if not markers.get('rows'):
        out.append(('FAIL', '%s: no [PREFIX] row for this request (matched %s): the model graft did not report '
                            'its prefill' % (tag, markers.get('matched'))))
    elif markers.get('l') is not None and markers['l'] != record.get('prompt_tokens'):
        out.append(('FAIL', '%s: [PREFIX] L=%s but the prompt is %s tokens' % (tag, markers['l'], record.get('prompt_tokens'))))
    if grant is not None and q is not None and grant['q'] != q:
        out.append(('FAIL', '%s: the scheduler granted Q=%d but the model restored Q=%s' % (tag, grant['q'], q)))
    if role == 'unsalted':
        if grant is not None:
            out.append(('FAIL', '%s: an unsalted request got a grant (%s): fail-closed tenancy broken' % (tag, grant)))
        if q:
            out.append(('FAIL', '%s: an unsalted request restored Q=%s' % (tag, q)))
        return out
    if role == 'cold' and q:
        out.append(('FAIL', '%s: a fresh cache_salt restored Q=%s: salts are not isolated' % (tag, q)))
    if q is not None and q % CHUNK:
        out.append(('FAIL', '%s: Q=%d is not a %d-token boundary' % (tag, q, CHUNK)))
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
    return out


def program_cache_problems(records):
    """F3: the first hit must compile nothing. The [PREFIX] row's programs= after the first row with
    Q > 0 must equal the one before it (the previous row in the log). -> (problems, detail)."""
    rows = []
    for record in records:
        for row in (record.get('markers') or {}).get('rows') or ():
            rows.append(row)
    rows.sort(key=lambda row: row.get('index', 0))
    for position, row in enumerate(rows):
        if row.get('q'):
            if row.get('programs') is None or position == 0 or rows[position - 1].get('programs') is None:
                return (['the first hit\'s [PREFIX] row (or the row before it) has no programs= field: the program '
                         'cache across the first hit is not measured'], dict(first_hit=row.get('tag')))
            before, after = rows[position - 1]['programs'], row['programs']
            detail = dict(first_hit=row.get('tag'), before=before, after=after)
            if after != before:
                return (['the program cache grew from %s to %s entries across the first hit (%s): a restore compiled '
                         'after the traces were parked (F3, the second-request hang)' % (before, after, row.get('tag'))],
                        detail)
            return [], detail
    return ['no hit (Q > 0) in the arm: the program cache across a first hit is not measured'], {}


def audit_problems(cold, hit):
    """The program-free audit's digests of a hit against its cold twin over the same KV range.
    -> list of (severity, text): FAIL for differing bytes, NOT_EXERCISED for a missing or
    mismatched-range audit."""
    a = (cold.get('markers') or {}).get('audit')
    b = (hit.get('markers') or {}).get('audit')
    if a is None or b is None:
        return [('NOT_EXERCISED', 'no [PREFIX-AUDIT] row for %s' % (cold.get('tag') if a is None else hit.get('tag')))]
    problems = []
    if a.get('kv_range') != b.get('kv_range'):
        problems.append(('NOT_EXERCISED', 'the audits cover different KV ranges (%s, %s)' % (
            a.get('kv_range'), b.get('kv_range'))))
    elif a.get('kv_sha') != b.get('kv_sha'):
        problems.append(('FAIL', 'KV %s differs between %s and %s' % (a.get('kv_range'), cold.get('tag'), hit.get('tag'))))
    if a.get('slot_sha') != b.get('slot_sha'):
        problems.append(('FAIL', 'the GDN slot bytes differ between %s and %s' % (cold.get('tag'), hit.get('tag'))))
    return problems
