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
   diverged again, and where, is reported).
   THE BATCHING RULE (concurrent_verdict, pair_summary): prefix reuse must not add divergence beyond
   what batching alone adds. The general path (the stock TT plugin's batched decode) is not batch-
   invariant - G1 v48 (run 36251045616) saw concurrent COLD runs differ from their solo runs with
   reuse off - so a hit that ran concurrently is judged against a cold run of the same batch:
     - a SOLO pair (kind 'sequential': the hit and its cold twin each ran with nothing else running)
       must be IDENTICAL, under the re-run policy above;
     - a CONCURRENT hit equal to its solo cold twin is IDENTICAL; one that differs is compared with
       its batch-matched cold control (the same messages under a fresh salt, sent again at once with
       the same co-runners in the same order): equal to it, IDENTICAL (basis 'batch');
     - the control equal to the solo cold run while the hit differs from both is DIVERGED, a FAIL:
       this batch leaves the baseline's bytes alone, so prefix reuse moved the hit;
     - only when the same run proves the baseline itself diverged under batching (the control
       differs from the solo cold run, and from the hit) is the pair NOT_COMPARABLE (basis
       'batching'); two disagreeing solo cold runs are UNSTABLE;
     - NOT_COMPARABLE pairs are listed and counted and do not hold the arm back, provided every solo
       pair of the arm is IDENTICAL and each scenario family (a pair's case before any ':' suffix)
       with a NOT_COMPARABLE pair has at least one IDENTICAL solo pair; prefix_replay gives each
       concurrent family that needs one a solo anchor (its hit replayed alone at the same Q).
   settle() then excuses a DIVERGED pair whose hit or cold run vLLM preempted and resumed (more
   than one admission; basis 'preemption', a batching effect: the pool was shared): the resumed
   prefill re-reads its own output, not a cold equivalent.
4. MARKERS: every request's [PREFIX] rows, grants and audit digests, matched by the harness's
   X-Request-Id tag (prefix_markers.request_tag), else by the request's log window and prompt length;
   on the fast path's sticky sessions also its sticky admit line, its engine build line and its audit
   window digests.
5. STICKY SESSIONS (the C2 fast path's phase 1, QWEN_FAST_STICKY_SESSIONS=1; Oracle(sticky=True)). The
   fast path decodes with DFlash, which vLLM counts as EAGLE-like: its unitary coordinator drops the
   last matched block of every hit (vLLM 0.25.1 single_type_kv_cache_manager.py FullAttentionManager.
   find_longest_cache_hit, after the num_tokens - 1 cap), so h is one block short; the trim never
   resumes above floor2048(P - 2048) (the drafter window [P - 2048, P) is the prefill's own chunks);
   a request captures C0 = floor2048(P) - 2048 in place of floor2048(P) (the next turn's hit stops a
   block short of what this one publishes, so its trim lands one chunk lower), and the gap boundary as
   before. The fast path is batch invariant and has no preemption, so a concurrent hit is judged
   against its solo cold twin strictly (no batch-matched control) and a preempted request is a fault,
   never an excuse.

Stdlib only, Python 3.7 syntax: it runs on the rig host.
"""

import hashlib
from array import array
from collections import Counter, OrderedDict

CHUNK = 2048
BLOCK = 64
# How many rows a finding names before it counts the rest (c2_prefix_gate.MAX_LISTED).
MAX_LISTED = 16
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


def resume_ceiling(prompt_tokens):
    """The highest Q sticky sessions grant a prompt of `prompt_tokens`: floor2048(P - 2048), so the fast
    path's drafter window [P - 2048, P) is this prefill's own chunks."""
    return floor_chunk(max(0, int(prompt_tokens) - CHUNK))


def capture_boundary(prompt_tokens, sticky=False):
    """The prompt's own capture candidate: floor2048(P), or under sticky sessions C0 = floor2048(P) - 2048."""
    boundary = floor_chunk(prompt_tokens)
    return boundary - CHUNK if sticky else boundary


def fresh_plan(prompt_tokens, sticky=False):
    """What a request with no hit (a fresh salt) captures: its prompt's capture candidate when positive."""
    position = capture_boundary(prompt_tokens, sticky)
    return [position] if position > 0 else []


class Oracle(object):
    """The expected hit, trim and captures of each admitted request, in admission order. sticky: the
    fast path's sticky sessions (module docstring item 5): the DFlash drop, the drafter-window ceiling
    and the C0 capture."""

    def __init__(self, capacity=None, sticky=False):
        self.capacity = store_entries() if capacity is None else int(capacity)
        self.sticky = bool(sticky)
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
        if self.sticky and blocks:
            blocks -= 1     # vLLM's EAGLE drop: DFlash's hit loses its last matched block
        h = blocks * BLOCK
        q = 0
        top = h // CHUNK
        if self.sticky:
            top = min(top, resume_ceiling(len(tokens)) // CHUNK)
        for k in range(top, 0, -1):
            key = (salt, digests[k * CHUNK // BLOCK - 1])
            if key in self.checkpoints:
                q = k * CHUNK
                self.checkpoints.move_to_end(key)
                break
        plan = set()
        boundary = capture_boundary(len(tokens), self.sticky)
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

    def expected_raw_hit(self, salt, tokens):
        """vLLM's own hit for `tokens` (what vllm:prefix_cache_hits counts, before the graft's trim): what
        this salt published, capped at num_tokens - 1, less the dropped block under sticky sessions."""
        tokens = list(tokens)
        capped = (max(0, len(tokens) - 1) // BLOCK) * BLOCK
        raw = min(self.published_tokens(salt, tokens), capped)
        if self.sticky and raw:
            raw -= BLOCK
        return raw

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
    """A hit that ran CONCURRENTLY, under the batching rule (module docstring, item 3): prefix reuse
    must not add divergence beyond what batching alone adds. `cold` and `cold_again` are its cold
    twin run ALONE, twice; `batch_cold` its batch-matched cold control - the same messages under a
    fresh salt, sent again at once with the same co-runners in the same order.
      - the hit equals its solo cold run: IDENTICAL (basis 'solo');
      - it differs, and equals its batch-matched control: IDENTICAL (basis 'batch');
      - it differs from both while the control equals the solo cold run: DIVERGED (basis 'reuse'),
        a FAIL - this batch leaves the baseline alone, so prefix reuse moved the hit;
      - it differs from both and the control differs from the solo cold run too: NOT_COMPARABLE
        (basis 'batching') - the run proves the baseline itself diverges under this batch;
      - the two solo cold runs disagree: UNSTABLE; a failed second cold run or control: ERROR; a
        second cold run or control that never ran: RERUN.
    -> dict(verdict, basis, first, [batch_hit, cold_repeat, batch], reason)."""
    first = compare(cold, hit)
    result = dict(first=first)
    if first['verdict'] != 'DIVERGED':
        basis = dict(IDENTICAL='solo', NOT_COMPARABLE='prompts').get(first['verdict'])
        result.update(verdict=first['verdict'], basis=basis)
        return result
    if cold_again is None:
        result.update(verdict='RERUN', reason='no second solo cold run')
        return result
    if batch_cold is None:
        result.update(verdict='RERUN', reason='no batch control ran')
        return result
    matched = compare(batch_cold, hit)
    result['batch_hit'] = matched
    if matched['verdict'] == 'IDENTICAL':
        result.update(verdict='IDENTICAL', basis='batch', reason='the hit differs from its solo cold run at %s and '
                      'equals its batch-matched cold control: batching moved both alike, prefix reuse added nothing'
                      % _where(first))
        return result
    colds = compare(cold, cold_again)
    result['cold_repeat'] = colds
    if colds['verdict'] == 'ERROR':
        result.update(verdict='ERROR', reason='the second solo cold run failed (%s): the divergence at %s is '
                      'unconfirmed' % (colds.get('detail'), _where(first)))
        return result
    if colds['verdict'] != 'IDENTICAL':
        result.update(verdict='UNSTABLE', reason='the two solo cold runs of the same prompt differ (%s): the engine is '
                      'not deterministic run to run' % colds.get('detail', colds['verdict']))
        return result
    control = compare(cold, batch_cold)
    result['batch'] = control
    if control['verdict'] == 'ERROR':
        result.update(verdict='ERROR', reason='the batch-matched cold control failed (%s): the divergence at %s is '
                      'unconfirmed' % (control.get('detail'), _where(first)))
    elif control['verdict'] == 'IDENTICAL':
        result.update(verdict='DIVERGED', basis='reuse', reason='the batch-matched cold control equals the solo cold '
                      'run (this batch leaves the baseline alone) and the hit differs from both (at %s from the solo '
                      'run, at %s from the control): prefix reuse introduced the divergence' % (
                          _where(first), _where(matched)))
    elif control['verdict'] == 'NOT_COMPARABLE':
        result.update(verdict='NOT_COMPARABLE', basis='prompts', reason='the batch-matched cold control rendered '
                      'another prompt than the solo cold run: a harness fault')
    else:
        result.update(verdict='NOT_COMPARABLE', basis='batching', reason='the batch-matched cold control differs from '
                      'the solo cold run (%s) and from the hit (%s): this batch moves the baseline itself, so the hit\'s '
                      'divergence from its solo run (at %s) is not put on prefix reuse' % (
                          control.get('detail', control['verdict']), matched.get('detail', matched['verdict']),
                          _where(first)))
    return result


def admissions(record):
    return ((record or {}).get('markers') or {}).get('admissions') or 0


def settle(pair, index, strict=False):
    """A pair's final verdict once markers are resolved (index: tag -> record): a DIVERGED pair whose
    hit, cold run, second cold run or batch control vLLM preempted and resumed (admissions > 1) is
    NOT_COMPARABLE, and any pair that is not IDENTICAL names its preempted requests. strict (the fast
    path, which has no preemption): nothing is excused, the preempted requests are only named.
    -> the pair (updated in place)."""
    if pair.get('verdict') in ('IDENTICAL', None):
        return pair
    resumed = []
    for tag in (pair.get('hit'), pair.get('cold'), pair.get('cold2'), pair.get('batch')):
        count = admissions(index.get(tag)) if tag else 0
        if count > 1:
            resumed.append('%s (%d admissions)' % (tag, count))
    if resumed:
        if pair['verdict'] == 'DIVERGED' and not strict:
            pair['verdict'] = 'NOT_COMPARABLE'
            pair['basis'] = 'preemption'
        pair['detail'] = ('%s; preempted and resumed: %s - a resumed prefill re-reads its own output, not a cold '
                          'equivalent' % (pair.get('detail'), ', '.join(resumed)))
    return pair


SOLO_KIND = 'sequential'
EXCUSED_BASES = ('batching', 'preemption')


def family(case):
    """A pair's scenario family: its case without a ':rerun', ':solo' or other suffix."""
    return str(case or '').split(':')[0]


def pair_summary(pairs):
    """The batching rule over one arm's settled pairs (module docstring, item 3). NOT_COMPARABLE pairs
    (basis batching or preemption) are TOLERATED - listed and counted, not the arm's verdict - when
    every solo pair of the arm is IDENTICAL and each family with a NOT_COMPARABLE pair has at least
    one IDENTICAL solo pair; any other NOT_COMPARABLE (different prompts: a harness fault) never is.
    -> dict(counts, families, tolerated, untolerated: the reasons it is not, line: the arm's summary
    line, tolerance: the line saying which)."""
    counts = dict(pairs=len(pairs), identical=0, identical_by_batch=0, not_comparable_batching=0,
                  not_comparable_preempted=0, not_comparable_other=0, failed=0, unstable=0, rerun=0, solo=0,
                  solo_identical=0)
    families = OrderedDict()
    for pair in pairs:
        verdict, basis = pair.get('verdict'), pair.get('basis')
        solo = pair.get('kind') == SOLO_KIND
        entry = families.setdefault(family(pair.get('case')), dict(pairs=0, solo=0, solo_identical=0,
                                                                   not_comparable=0))
        entry['pairs'] += 1
        if solo:
            counts['solo'] += 1
            entry['solo'] += 1
        if verdict == 'IDENTICAL':
            counts['identical'] += 1
            counts['identical_by_batch'] += 1 if basis == 'batch' else 0
            if solo:
                counts['solo_identical'] += 1
                entry['solo_identical'] += 1
        elif verdict == 'NOT_COMPARABLE':
            entry['not_comparable'] += 1
            key = dict(batching='not_comparable_batching', preemption='not_comparable_preempted').get(
                basis, 'not_comparable_other')
            counts[key] += 1
        elif verdict in ('DIVERGED', 'ERROR'):
            counts['failed'] += 1
        elif verdict == 'UNSTABLE':
            counts['unstable'] += 1
        elif verdict == 'RERUN':
            counts['rerun'] += 1
    not_comparable = [pair for pair in pairs if pair.get('verdict') == 'NOT_COMPARABLE']
    untolerated = []
    if not_comparable:
        other = [pair for pair in not_comparable if pair.get('basis') not in EXCUSED_BASES]
        if other:
            untolerated.append('%d not comparable for a reason other than batching (%s): never excused' % (
                len(other), ', '.join(sorted(set(str(pair.get('basis')) for pair in other)))))
        solo_bad = counts['solo'] - counts['solo_identical']
        if solo_bad:
            untolerated.append('%d of %d solo pairs are not IDENTICAL' % (solo_bad, counts['solo']))
        for name, entry in families.items():
            if entry['not_comparable'] and not entry['solo_identical']:
                untolerated.append('family %s has %d not comparable and no IDENTICAL solo pair' % (
                    name, entry['not_comparable']))
    tolerated = bool(not_comparable) and not untolerated
    batching = counts['not_comparable_batching'] + counts['not_comparable_preempted']
    line = 'pairs: %d identical (%d by the batch-matched control), %d not-comparable-by-batching%s, %d failed; ' \
           'solo %d of %d identical; of %d' % (
               counts['identical'], counts['identical_by_batch'], batching,
               ' (%d preempted)' % counts['not_comparable_preempted'] if counts['not_comparable_preempted'] else '',
               counts['failed'], counts['solo_identical'], counts['solo'], counts['pairs'])
    for key, label in (('not_comparable_other', 'not comparable otherwise'), ('unstable', 'unstable'),
                       ('rerun', 'without their re-run')):
        if counts[key]:
            line += ', %d %s' % (counts[key], label)
    if not not_comparable:
        tolerance = None
    elif tolerated:
        tolerance = ('not comparable tolerated (%d): every solo pair is IDENTICAL and %s %s an IDENTICAL solo pair' % (
            len(not_comparable), ', '.join(name for name, entry in families.items() if entry['not_comparable']),
            'has' if sum(1 for entry in families.values() if entry['not_comparable']) == 1 else 'each have'))
    else:
        tolerance = 'not comparable NOT tolerated (%d): %s' % (len(not_comparable), '; '.join(untolerated))
    return dict(counts, families=families, tolerated=tolerated, untolerated=untolerated, line=line,
                tolerance=tolerance)


# -- markers onto records ------------------------------------------------------------------------

def resolve(records, scanned):
    """Attach each request's grants, [PREFIX] rows and audit row (prefix_markers.scan) by tag; a
    row without a request id falls back to the one row inside the request's log window whose L is
    its prompt length. The FIRST row is the admission judged against the oracle (q, l); the
    admissions count is the number of rows (a preempted request is re-admitted and prints again);
    `grant` is the first admission's (the grants logged before the second row). Sets
    record['markers'] and returns the records."""
    grants, rows, audits, skipped, admits, windows = {}, {}, {}, {}, {}, {}
    for entry in scanned.get('grants') or ():
        grants.setdefault(entry['tag'], []).append(entry)
    for entry in scanned.get('sticky_admits') or ():
        if entry.get('tag'):
            admits.setdefault(entry['tag'], []).append(entry)
    for entry in scanned.get('audit_windows') or ():
        if entry.get('tag'):
            windows.setdefault(entry['tag'], []).append(entry)
    builds = list(scanned.get('sticky_builds') or ())
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
                     audit=(audits.get(tag) or [None])[0], skipped=list(skipped.get(tag) or ()), matched='tag',
                     sticky_admits=list(admits.get(tag) or ()), audit_windows=list(windows.get(tag) or ()),
                     sticky_builds=[entry for entry in builds if engine_id_matches(entry.get('req'), tag)])
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


def engine_id_matches(engine_id, tag):
    """Whether an engine request id a line printed - whole, or cut to its first 48 characters (the fast
    path's engine lines print str(req_id)[:48]) - is the request the harness tagged `tag`: the API server's
    'chatcmpl-<tag>' plus the input processor's '-<8 characters>' (prefix_markers.request_tag). Tags carry a
    unique counter ahead of their label, so a cut id names one request."""
    if not engine_id or not tag:
        return False
    text = str(engine_id)
    full = 'chatcmpl-' + tag
    if text == full:
        return True
    if text.startswith(full + '-') and len(text) > len(full) + 1:
        rest = text[len(full) + 1:]
        return len(rest) <= 8 and rest.isalnum()
    # A cut id: a prefix of 'chatcmpl-<tag>-', the cut ending on that '-' included (a 38-character tag).
    return len(text) >= 40 and (full + '-').startswith(text)


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


def reuse_problems(record, sequential=True, sticky=False):
    """What one request's markers say against the oracle's expectation (record['expected']) and the
    request's own role. -> list of (severity, text): FAIL for a missing or inconsistent row, a grant
    to an unsalted or fresh-salt request, a restore without a grant, a capture nobody planned, or
    (sequential) a Q, h or capture plan other than the oracle's - a grant the design forbids, a
    block published past the cap; LOST for a hit the oracle expected and the engine did not give;
    NOTE for an off-oracle reading where requests overlapped (the oracle is then only an estimate
    of admission order) and for each re-admission of a preempted request. sticky (the fast path's
    sticky sessions, module docstring item 5), whatever the overlap: FAIL for a Q above the drafter
    window's ceiling floor2048(L - 2048), a re-admission (the fast path has no preemption), and a hit
    whose sticky admit line is missing or disagrees with its row (Q, P, tail = P - Q), or an admit
    line for a request that resumed nothing."""
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
        if sticky:
            out.append(('FAIL', '%s: a second [PREFIX] row (L=%s Q=%s): the fast path serves no preemption, so no '
                                'request may be re-admitted' % (tag, later.get('l'), later.get('q'))))
            continue
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
    if sticky:
        out += sticky_problems(record, rows)
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


def sticky_problems(record, rows):
    """reuse_problems' sticky-session checks of one request's first admission. -> [(severity, text)]."""
    markers = record.get('markers') or {}
    tag = record.get('tag')
    prompt = record.get('prompt_tokens')
    out = []
    first = rows[0] if rows else None
    q = (first or {}).get('q')
    if q and prompt is not None and q > resume_ceiling(prompt):
        out.append(('FAIL', '%s: Q=%d is above floor2048(L - 2048) = %d: a resume inside the fast path\'s drafter '
                            'window (a C1 grant the sticky trim forbids)' % (tag, q, resume_ceiling(prompt))))
    admits = markers.get('sticky_admits') or []
    if q:
        matching = [entry for entry in admits if entry.get('q') == q]
        if not matching:
            out.append(('FAIL', '%s: restored Q=%d but no "[PINDIAG] sticky admit" line with Q=%d (%s): the fast '
                                'path\'s lifecycle did not take the resume it prefilled' % (
                                    tag, q, q, [entry.get('q') for entry in admits] or 'none')))
        else:
            admit = matching[0]
            if prompt is not None and (admit.get('p') != prompt or admit.get('tail') != prompt - q):
                out.append(('FAIL', '%s: its sticky admit line says P=%s tail=%s, the request is P=%s Q=%d (tail %d)' % (
                    tag, admit.get('p'), admit.get('tail'), prompt, q, prompt - q)))
        if len(admits) > 1:
            out.append(('FAIL', '%s: %d sticky admit lines for one request' % (tag, len(admits))))
    elif admits:
        out.append(('FAIL', '%s: a sticky admit line (Q=%s) for a request whose row restored nothing' % (
            tag, admits[0].get('q'))))
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


def prefill_shape(row):
    """What decides which prefill programs a [PREFIX] row runs, once the model graft's warm has compiled the
    restore path: its tail t = L mod 2048 (the masked bucket and whether it is full, the paged-fill width
    ceil(t/64), and the fast path's draft-window split all follow from it; the model warms exactly those,
    warmup_prefill_masked_buckets) and whether it runs a full 2048-token chunk from its Q (the chunk
    program; which chunk it is does not matter - G1's warmed rows compiled nothing at any length). None
    without an L."""
    length, q = row.get('l'), row.get('q') or 0
    if type(length) is not int:
        return None
    return length % CHUNK, length // CHUNK * CHUNK > q


def fast_path_growth(rows):
    """The no-compile rule where the model does not warm its prefill: the C2 fast path (sticky sessions on
    c2-packed-prefix), whose worker warmup runs the graft's restore warm but never the plugin's prefill warmup.
    Its first prefill of each shape compiles at request time, as S2's prefill always has, so only a row that
    repeats a shape an earlier row of the same engine process already ran (prefill_shape) must compile
    nothing; a hit and its cold twin are measured against each other by program_cache_problems. A row whose
    programs_before is below the previous row's count starts a new process (a restart): what it had seen is
    forgotten. Rows without both counts are not judged here. -> (problems, notes)."""
    problems, compiled = [], []
    seen, last = set(), None
    for row in sorted(rows, key=lambda row: row.get('index', 0)):
        before, after = row.get('programs_before'), row.get('programs')
        if not isinstance(before, int) or not isinstance(after, int):
            continue
        if last is not None and before < last:
            seen = set()
        last = after
        shape = prefill_shape(row)
        grown = after - before
        what = '%s (Q=%s L=%s, tail %s%s)' % (row.get('tag') or row.get('req'), row.get('q'), row.get('l'),
                                              shape[0] if shape else '?',
                                              ', a full chunk' if shape and shape[1] else '')
        if grown > 0 and shape is not None and shape in seen:
            problems.append('%s compiled %d programs inside its prefill row, though an earlier row of this engine '
                            'ran the same prefill shape: a compile the first request of that shape should have '
                            'made (F3)' % (what, grown))
        elif grown > 0:
            compiled.append('%s: %d' % (what, grown))
        if shape is not None:
            seen.add(shape)
    if len(problems) > MAX_LISTED:
        problems = problems[:MAX_LISTED] + ['%d more rows compiled a shape already run (server.log)'
                                            % (len(problems) - MAX_LISTED)]
    notes = []
    if compiled:
        notes.append('first rows of a prefill shape compiled programs (the fast path compiles each shape at its '
                     'first request): %s%s' % ('; '.join(compiled[:MAX_LISTED]),
                                               ' and %d more' % (len(compiled) - MAX_LISTED)
                                               if len(compiled) > MAX_LISTED else ''))
    return problems, notes


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


def window_findings(records):
    """The audit's per-window KV digests (QWEN_PREFIX_AUDIT=1: one '[PREFIX-AUDIT] ... window=w' line per
    2048-token window of each prefill row) across every audited request of an arm: any two rows whose
    prompts share their tokens up to a window's end must hold the same KV bytes in that window - a hit's
    restored windows (new=0: the shared or retained blocks, read after other requests decoded beside
    them) against the cold run or earlier hit that wrote them. Only full windows are compared; a row's
    prompt is identified per window by record['chunk_digests'] (prefix_digests at CHUNK). -> (problems,
    lines: how many windows were compared, and how many of them a hit read without writing, and that
    count alone)."""
    groups = {}
    for record in records:
        if not record.get('ok') and not record.get('aborted'):
            continue
        prefixes = record.get('chunk_digests') or []
        for window in (record.get('markers') or {}).get('audit_windows') or ():
            index, start, end = window.get('window'), window.get('start'), window.get('end')
            if index is None or start is None or end is None or end - start != CHUNK or index >= len(prefixes):
                continue
            key = (index, prefixes[index])
            groups.setdefault(key, []).append((record.get('tag'), window.get('kv'), window.get('new')))
    problems, compared, restored = [], 0, 0
    for (index, _), entries in sorted(groups.items()):
        if len(entries) < 2:
            continue
        compared += 1
        restored += 1 if any(new == 0 for _, _, new in entries) else 0
        digests = set(kv for _, kv, _ in entries)
        if len(digests) > 1:
            problems.append('KV window %d [%d, %d) differs between requests with the same tokens there: %s' % (
                index, index * CHUNK, (index + 1) * CHUNK,
                ', '.join('%s %s%s' % (tag, kv, ' (restored)' if new == 0 else '') for tag, kv, new in entries[:6])))
    lines = ['audit windows: %d shared windows compared, %d of them read by a hit without writing them' % (
        compared, restored)]
    return problems[:16], lines, restored


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
