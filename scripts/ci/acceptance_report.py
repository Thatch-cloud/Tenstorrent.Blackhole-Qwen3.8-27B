"""Speculative acceptance and per-user decode rate of a finished M3native gate run.

Reads the FULL server.log after the server has stopped: the only complete per-round record is
printed when each request closes, and for the last requests that is after SIGTERM (v155: two of
the four records follow the shutdown line). Four independent sources, and the report says which
agreed:

  (c) PHASES - primary. '{"stage": "fast_serving_phases", "blocks": [...]}', one JSON line per
      request at close (serving_fast_request.FastRequest.close under QWEN_FAST_PHASE_TIMING=1,
      which the arm always sets). One block per round the request took part in, packed or
      sequential, with `rows` (verified rows = 1 + the drafts verified) and `committed` (the
      tokens that round emitted: the accepted drafts plus the correction/bonus token, or the
      accepted drafts alone when one of them is EOS - greedy_verify.select_prefix). Packed
      4-user blocks carry verifier.packed=true. No request id, so a record is attributed to a
      user (attribute): in a sequential run by order (one request at a time, each record printed
      at its close), else by its first block's position when that is one user's prompt length
      (never on real text, where every prompt is the same length), by the packed audit's request
      id, by the stream's per-chunk token counts, or last by a chunk count no other unattributed
      record or stream shares.
  (a) PACKED AUDIT. '[PACKED] request=<id> segment= position= prefix= emitted=N ...', one line
      per user per packed 4-user round (serving_packed_step.AUDIT_LINE, QWEN_FAST_PACKED_AUDIT=1).
      Packed rounds only; single-stream runs log none. Its per-request emitted sequence must equal
      the packed blocks of one phases record.
  (b) vLLM. 'SpecDecoding metrics: Mean acceptance length: M, ..., Accepted: A tokens, Drafted: D
      tokens, Per-position acceptance rate: r1, ..., r15, ...' (vllm/v1/spec_decode/metrics.py:120),
      every 10 s while the engine is busy, over the interval since the previous line. M = 1 +
      A / drafts (two decimals); r_i = drafts accepting past position i / ALL drafts (three
      decimals), so 3-draft and tail rounds dilute the later positions. The draft count is not
      printed: recover_drafts finds the one n every printed figure is consistent with. The last
      interval is never printed (the server is stopped first): it covers a prefix of the run, and
      is checked for being consistent with (c), not equal to it. Both the packed and the
      single-stream path log it.
  (d) STREAM. With the gate's detail streams (continuous usage stats), the completion-token count
      of every chunk: the first chunk is the prefill's seed token and each later one is one
      round's committed tokens - unless the API server coalesced two rounds into one chunk, which
      the check accepts ('coalesced') when the chunk boundaries are a subset of the round
      boundaries. Without detail, only text chunks - 1 against the record's block count.

Statistics. `full_draft` is the acceptance measurement: blocks that verified all 15 drafts
(rows == 16), excluding each request's terminal block (cut by EOS or by the budget, not by a
draft mismatch). Over those: mean emitted, histogram 1..16, P(emitted > 8), P(emitted > 11),
P(emitted == 16: all 15 drafts accepted) and per-position acceptance (P(accepted >= i)). `all`
is every decode block, which is what the user saw (tail rounds of 1-4 rows included). The
per-position rates are also given conditionally over every non-terminal block that drafted the
position, and in vLLM's convention (over all drafting blocks) for comparison with (b).

Rates (decode_rates). Per user: `steady` - its own packed rounds (concurrent) or 16-row rounds
(sequential) joined chunk to block, minus the first decode round, the terminal one and capture
stalls (steady_rate) - is the figure to set against the programme's ~28.3 / 47.5 tok/s; the
median-gap and all-active figures are kept for comparison and read low (tail rounds, stalls).

Active users per round: every '[PHASE] execute total= new=0 cached=C spec=S' line
(serving_worker_hook) is one engine decode step with C running requests. A round with fewer than
four leaves the packed 64-row step and runs the survivors sequentially at four rows each (at most
four tokens), so these counts say how much of the run was the packed round being measured.

Stdlib only (json, re, statistics): mounted at /bench beside the gate, imported by the offline
real_text_compare.py too. Python 3.10 compatible.
"""

from collections import Counter
import json
import re
import statistics

POSITIONS = 15                 # T16: 15 drafts per verified block
FULL_ROWS = POSITIONS + 1
PHASES_KEY = '{"stage": "fast_serving_phases"'
PACKED_AUDIT = re.compile(r'\[PACKED\] request=(\S+) segment=([0-9]+) position=([0-9]+) prefix=([0-9]+) '
                          r'emitted=([0-9]+)')
SPEC_LINE = re.compile(r'SpecDecoding metrics: Mean acceptance length: ([0-9.]+|nan), [^\n]*?Accepted: ([0-9]+) '
                       r'tokens, Drafted: ([0-9]+) tokens, Per-position acceptance rate: ([0-9., na]*?), '
                       r'Avg Draft acceptance rate: ([0-9.]+|nan)%')
EXECUTE_LINE = re.compile(r'\[PHASE\] execute total=([0-9]+) new=([0-9]+) cached=([0-9]+) spec=([0-9]+)')
PACKED_PHASE_LINE = re.compile(r'\[PACKED-PHASE\] round=([0-9]+) users=([0-9]+)')
SHUTDOWN_MARKER = 'trigger received signal=SIGTERM'
STALL_FACTOR = 3.0             # a round gap over 3 x the median is a capture/warm-up stall, not a round


def _int(value, default=0):
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def phase_records(log_text):
    """Every fast_serving_phases record in print (= request close) order, and how many failed to parse."""
    decoder = json.JSONDecoder()
    records, unparsed = [], 0
    for number, line in enumerate(log_text.split('\n'), 1):
        at = line.find(PHASES_KEY)
        if at < 0:
            continue
        try:
            payload, _ = decoder.raw_decode(line, at)
        except ValueError:
            unparsed += 1
            continue
        blocks = []
        for block in payload.get('blocks') or []:
            verifier = block.get('verifier') or {}
            blocks.append(dict(position=_int(block.get('position'), -1), rows=_int(block.get('rows')),
                               committed=_int(block.get('committed')), packed=bool(verifier.get('packed')),
                               users=verifier.get('users')))
        records.append(dict(line=number, blocks=blocks, finished=payload.get('finished'),
                            cancelled=payload.get('cancelled')))
    return records, unparsed


def packed_audit(log_text):
    """{request id: [{segment, position, prefix, emitted}, ...]} in log order."""
    requests = {}
    for match in PACKED_AUDIT.finditer(log_text):
        requests.setdefault(match.group(1), []).append(dict(
            segment=int(match.group(2)), position=int(match.group(3)), prefix=int(match.group(4)),
            emitted=int(match.group(5))))
    return requests


def recover_drafts(mean_text, accepted, drafted, rate_texts, positions=POSITIONS):
    """(drafts, how) for one vLLM interval, from the figures as printed.

    drafts = n, the number of verified drafts, is not printed. Every draft proposes 1..k tokens,
    so ceil(D / k) <= n <= D; the accepted count per position is r_i * n (an integer), their sum
    is A, and '%.3f' % (count_i / n) and '%.2f' % (1 + A / n) must reproduce the printed rates and
    mean. 'exact' when exactly one n fits. A = 0 fits every n (all rates print 0.000, the mean
    1.00): then n = ceil(D / k), every draft full width ('no-acceptance'), so the interval still
    weighs in rather than biasing the pooled mean up. Several fits: the one nearest A / (M - 1)
    ('ambiguous'); none (rounding past 1000 drafts): A / (M - 1) itself ('mean')."""
    width = len(rate_texts) or positions
    if drafted <= 0:
        return 0, 'no-drafts'
    matches = []
    for n in range(-(-drafted // width), drafted + 1):
        counts = [int(round(float(text) * n)) for text in rate_texts]
        if sum(counts) != accepted:
            continue
        if any('%.3f' % (count / n) != text for count, text in zip(counts, rate_texts)):
            continue
        if mean_text != 'nan' and '%.2f' % (1.0 + accepted / n) != mean_text:
            continue
        matches.append(n)
    if len(matches) == 1:
        return matches[0], 'exact'
    if accepted == 0:
        return -(-drafted // width), 'no-acceptance'
    mean = float(mean_text) if mean_text != 'nan' else float('nan')
    estimate = accepted / (mean - 1.0) if mean == mean and mean > 1.0 else None
    if matches:
        return min(matches, key=lambda n: (abs(n - estimate) if estimate else 0, n)), 'ambiguous'
    if estimate:
        return int(round(estimate)), 'mean'
    return None, 'unknown'


def spec_decoding(log_text):
    """vLLM's SpecDecoding intervals, each with its draft count recovered (recover_drafts)."""
    intervals = []
    for match in SPEC_LINE.finditer(log_text):
        mean_text = match.group(1)
        accepted, drafted = int(match.group(2)), int(match.group(3))
        texts = [value.strip() for value in match.group(4).split(',') if value.strip()]
        drafts, how = recover_drafts(mean_text, accepted, drafted, texts)
        intervals.append(dict(mean_acceptance_length=float(mean_text), accepted=accepted, drafted=drafted,
                              drafts=drafts, drafts_recovered=how, per_position=[float(text) for text in texts]))
    return intervals


def spec_totals(intervals):
    """The printed intervals pooled (drafts-weighted), or None when there are none. Every sum is
    over the same intervals, those whose draft count is known."""
    if not intervals:
        return None
    known = [i for i in intervals if i['drafts'] is not None]
    drafts = sum(i['drafts'] for i in known)
    accepted = sum(i['accepted'] for i in known)
    drafted = sum(i['drafted'] for i in known)
    width = max((len(i['per_position']) for i in known), default=0)
    counts = [sum(i['drafts'] * (i['per_position'][p] if p < len(i['per_position']) else 0.0) for i in known)
              for p in range(width)]
    return dict(intervals=len(intervals), intervals_without_drafts=len(intervals) - len(known), drafts=drafts,
                accepted=accepted, drafted=drafted,
                drafts_recovered=dict(Counter(i['drafts_recovered'] for i in intervals)),
                mean_acceptance_length=round(1.0 + accepted / drafts, 3) if drafts else None,
                per_position=[round(c / drafts, 3) for c in counts] if drafts else None,
                per_position_accepted=[int(round(c)) for c in counts])


def execute_steps(log_text):
    """Engine steps from '[PHASE] execute': decode steps by running requests, prefill steps, and the rest."""
    decode, prefill, mixed = Counter(), 0, 0
    for match in EXECUTE_LINE.finditer(log_text):
        _total, new, cached, _spec = (int(v) for v in match.groups())
        if new == 0 and cached > 0:
            decode[cached] += 1
        elif new > 0 and cached == 0:
            prefill += 1
        else:
            mixed += 1
    return dict(decode_steps_by_active_users={str(k): decode[k] for k in sorted(decode, reverse=True)},
                prefill_steps=prefill, mixed_steps=mixed)


def block_stats(blocks, positions=POSITIONS):
    """Emitted-per-round statistics over `blocks`, or None for none."""
    emitted = [b['committed'] for b in blocks]
    if not emitted:
        return None
    rounds = len(emitted)
    over = lambda limit: round(sum(1 for e in emitted if e > limit) / rounds, 4)
    return dict(rounds=rounds, tokens=sum(emitted), mean_emitted=round(statistics.fmean(emitted), 3),
                histogram={str(k): emitted.count(k) for k in range(1, positions + 2)},
                p_emitted_gt_8=over(8), p_emitted_gt_11=over(11),
                p_emitted_eq_16=round(sum(1 for e in emitted if e == positions + 1) / rounds, 4),
                max_emitted=max(emitted))


def per_position(blocks, positions=POSITIONS, convention='conditional'):
    """P(accepted >= i) for draft position i = 1..positions. 'conditional': over the blocks that
    drafted position i (rows - 1 >= i). 'vllm': over every block that drafted anything."""
    rates = []
    drafting = [b for b in blocks if b['rows'] > 1]
    for index in range(1, positions + 1):
        pool = drafting if convention == 'vllm' else [b for b in drafting if b['rows'] - 1 >= index]
        hits = sum(1 for b in pool if b['committed'] - 1 >= index and b['rows'] - 1 >= index)
        rates.append(round(hits / len(pool), 3) if pool else None)
    return rates


def acceptance_blocks(record):
    """A record's decode blocks, each marked terminal (its last) or not; cancelled zero blocks dropped."""
    blocks = [dict(b) for b in record['blocks'] if b['committed'] > 0]
    for index, block in enumerate(blocks):
        block['terminal'] = index == len(blocks) - 1
    return blocks


def section(blocks):
    """The `all` / `full_draft` / per-position view of a set of marked blocks."""
    body = [b for b in blocks if not b['terminal']]
    full = [b for b in body if b['rows'] == FULL_ROWS]
    return dict(all=block_stats(blocks), full_draft=block_stats(full),
                per_position_full_draft=per_position(full) if full else None,
                per_position_conditional=per_position(body) if body else None,
                per_position_vllm_convention=per_position(body, convention='vllm') if body else None)


def stream_deltas(stream):
    """A detail stream's completion tokens per chunk (first chunk: the prefill seed), or None."""
    tokens = (stream or {}).get('chunk_tokens')
    if not tokens or any(t is None for t in tokens):
        return None
    return list(tokens)


def stream_agreement(stream, blocks):
    """'exact' when every chunk after the first is one round, 'coalesced' when the chunk boundaries
    are a subset of the round boundaries, 'disagree' otherwise; None without the data."""
    deltas = stream_deltas(stream)
    committed = [b['committed'] for b in blocks]
    if deltas is None:
        chunks = (stream or {}).get('tokens')
        if chunks is None or not committed:
            return None
        return 'chunks=blocks+1' if chunks == len(committed) + 1 else 'chunks!=blocks+1'
    if deltas == [1] + committed:
        return 'exact'
    # Token counts after the seed at every chunk end must all be round ends, and the last the total.
    chunk_edges = [edge - 1 for edge in _cumulative(deltas)]
    round_edges = [0] + _cumulative(committed)
    if chunk_edges and chunk_edges[-1] == round_edges[-1] and set(chunk_edges) <= set(round_edges):
        return 'coalesced'
    return 'disagree'


def _cumulative(values):
    total, out = 0, []
    for value in values:
        total += value
        out.append(total)
    return out


def attribute(records, streams, prompt_lengths=None, sequential=False, audit_owners=None):
    """[user index or None per record] and how each was attributed, strongest evidence first:
    'order' - a sequential run (one request at a time; each record prints at its request's
    close, before the next request starts) with one record per stream: record k is user k,
    checked against user k's prompt length when the lengths are known;
    'position' - the first block's position is exactly one user's prompt length (never on real
    text, where every prompt is the arm's request context);
    'audit' - the packed audit's request id extends this user's stream id and its emitted
    sequence is this record's packed rounds (audit_owners: {record index: user});
    'stream' - the user's per-chunk token counts are this record's rounds (detail streams);
    'chunk-count' - without detail, the user's text chunks are this record's blocks + 1, and no
    other unattributed user or record matches either side (a coalesced chunk could otherwise
    hand a record to the wrong user). A record no pass singles out stays None."""
    streams = list(streams or [])
    owners, how = [None] * len(records), [None] * len(records)
    taken = set()
    lengths = list(prompt_lengths or [])
    nonzero = [[b for b in record['blocks'] if b['committed'] > 0] for record in records]

    def claim(index, user, label):
        owners[index], how[index] = user, label
        taken.add(user)

    if sequential and len(records) == len(streams):
        for index, blocks in enumerate(nonzero):
            first = blocks[0]['position'] if blocks else None
            if not lengths or (index < len(lengths) and first == lengths[index]):
                claim(index, index, 'order')
    for index, blocks in enumerate(nonzero):
        if owners[index] is not None:
            continue
        first = blocks[0]['position'] if blocks else None
        candidates = [u for u, length in enumerate(lengths) if length == first and u not in taken]
        if len(candidates) == 1 and lengths.count(first) == 1:
            claim(index, candidates[0], 'position')
    for index, user in sorted((audit_owners or {}).items()):
        if owners[index] is None and user is not None and user not in taken:
            claim(index, user, 'audit')
    for label, accepted in (('stream', ('exact', 'coalesced')), ('chunk-count', ('chunks=blocks+1',))):
        for index, blocks in enumerate(nonzero):
            if owners[index] is not None:
                continue
            candidates = [u for u, stream in enumerate(streams) if u not in taken
                          and stream_agreement(stream, blocks) in accepted]
            if len(candidates) != 1:
                continue
            rivals = [other for other, others in enumerate(nonzero) if other != index and owners[other] is None
                      and stream_agreement(streams[candidates[0]], others) in accepted]
            if not rivals:
                claim(index, candidates[0], label)
    return owners, how


def round_gaps(stream, blocks):
    """(seconds per block, source) - the time from the previous chunk to the chunk carrying each
    block's tokens - or (None, None) when the chunks cannot be tied to the blocks one to one.
    'chunk_s': a detail stream whose chunks are exactly [seed] + one per round. 'gaps_ms': a
    plain stream whose text chunks are the blocks + 1 (each chunk had text)."""
    stream = stream or {}
    committed = [b['committed'] for b in blocks]
    offsets = stream.get('chunk_s')
    if stream_deltas(stream) == [1] + committed and offsets and len(offsets) == len(committed) + 1:
        return [offsets[i + 1] - offsets[i] for i in range(len(committed))], 'chunk_s'
    gaps = stream.get('gaps_ms')
    if stream.get('chunk_tokens') is None and gaps is not None and stream.get('tokens') == len(committed) + 1 \
            and len(gaps) == len(committed):
        return [gap / 1000.0 for gap in gaps], 'gaps_ms'
    return None, None


def steady_rate(stream, blocks, concurrent):
    """One user's steady decode rate from its own rounds, or None when its chunks cannot be tied
    to its blocks (round_gaps).

    The rounds: packed blocks in a concurrent run, 16-row blocks in a sequential one; never the
    first decode block (its gap runs from the prefill's seed chunk, so it holds the wait for the
    other users' prefills or the first capture) nor the terminal one (cut by EOS or the budget);
    and no round whose gap exceeds STALL_FACTOR x the median (a capture or warm-up stall: v155
    carries two of ~1 s, v149's first user eight). steady_tok_s = their committed tokens over
    their summed gaps; mean_over_median_tok_s = their mean emitted over the median round, the
    programme's own convention (the ~28.3 tok/s at 4 x 131k). The 4-row non-packed tail rounds a
    concurrent run ends with (at most four tokens each, ~500 ms) are not packed rounds, so
    neither figure includes them."""
    gaps, source = round_gaps(stream, blocks)
    if gaps is None:
        return None
    rounds = [(gap, block) for gap, block in list(zip(gaps, blocks))[1:] if not block.get('terminal')
              and (block['packed'] if concurrent else block['rows'] == FULL_ROWS and not block['packed'])]
    if not rounds:
        return dict(source=source, rounds=0, stalls_excluded=0, steady_tok_s=None, mean_over_median_tok_s=None,
                    median_round_ms=None)
    median = statistics.median(gap for gap, _ in rounds)
    kept = [(gap, block) for gap, block in rounds if gap <= STALL_FACTOR * median]
    tokens, seconds = sum(block['committed'] for _, block in kept), sum(gap for gap, _ in kept)
    mean = tokens / len(kept)
    return dict(source=source, rounds=len(kept), stalls_excluded=len(rounds) - len(kept), tokens=tokens,
                seconds=round(seconds, 4), median_round_ms=round(median * 1000.0, 2), mean_emitted=round(mean, 3),
                steady_tok_s=round(tokens / seconds, 2) if seconds > 0 else None,
                mean_over_median_tok_s=round(mean / median, 2) if median > 0 else None)


def audit_owner(request_id, streams):
    """The user whose stream id the engine request id extends (cmpl-<hex> -> cmpl-<hex>-0-<sfx>)."""
    for user, stream in enumerate(streams or []):
        stream_id = (stream or {}).get('request_id')
        if stream_id and (request_id == stream_id or request_id.startswith(stream_id + '-')):
            return user
    return None


def report(log_text, streams=None, prompt_lengths=None, sequential=None):
    """The acceptance report dict (see the module docstring); report['summary_line'] is one line.
    `sequential` (a --sequential-users run): attribution by order and 16-row steady rounds;
    None infers it from the log (no packed block at all) for the rates only, never for order."""
    streams = list(streams or [])
    records, unparsed = phase_records(log_text)
    audit = packed_audit(log_text)
    intervals = spec_decoding(log_text)
    shutdown = next((number for number, line in enumerate(log_text.split('\n'), 1) if SHUTDOWN_MARKER in line), None)
    marked = [acceptance_blocks(record) for record in records]
    everything = [b for blocks in marked for b in blocks]
    packed = [b for b in everything if b['packed']]
    concurrent = not sequential if sequential is not None else bool(packed)

    audit_matches = {}
    for request_id, lines in audit.items():
        emitted = [line['emitted'] for line in lines]
        match = [index for index, blocks in enumerate(marked)
                 if [b['committed'] for b in blocks if b['packed']] == emitted]
        audit_matches[request_id] = match[0] if len(match) == 1 else (None if not match else 'ambiguous')
    audit_owners = {}
    for request_id, matched in audit_matches.items():
        user = audit_owner(request_id, streams)
        if isinstance(matched, int) and user is not None:
            audit_owners[matched] = user
    owners, how = attribute(records, streams, prompt_lengths, sequential=bool(sequential), audit_owners=audit_owners)

    users = []
    for user in range(len(streams)):
        indices = [index for index, owner in enumerate(owners) if owner == user]
        entry = dict(user=user, attributed_by=how[indices[0]] if indices else None, steady=None)
        stream = streams[user] or {}
        if indices:
            record, blocks = records[indices[0]], marked[indices[0]]
            entry.update(record_line=record['line'], first_position=blocks[0]['position'] if blocks else None,
                         **section(blocks))
            entry['stream_agreement'] = stream_agreement(stream, blocks)
            entry['steady'] = steady_rate(stream, blocks, concurrent)
            completion = stream.get('completion_tokens')
            entry['completion_tokens'] = completion
            entry['completion_is_seed_plus_rounds'] = (completion == sum(b['committed'] for b in blocks) + 1
                                                       if completion is not None else None)
        requests = [rid for rid in audit if audit_owner(rid, streams) == user]
        if requests:
            rid = requests[0]
            emitted = [line['emitted'] for line in audit[rid]]
            matched = audit_matches.get(rid)
            entry['packed_audit'] = dict(
                request_id=rid, rounds=len(emitted), mean_emitted=round(statistics.fmean(emitted), 3),
                max_emitted=max(emitted),
                agrees_with_phases=(matched == indices[0]) if indices and isinstance(matched, int) else None)
        entry['finish_reason'] = stream.get('finish_reason')
        users.append(entry)

    compact = [dict(line=record['line'], user=owners[index], attributed_by=how[index],
                    first_position=blocks[0]['position'] if blocks else None, blocks=len(blocks),
                    packed_blocks=sum(1 for b in blocks if b['packed']),
                    committed=[b['committed'] for b in blocks],
                    after_shutdown=bool(shutdown and record['line'] > shutdown))
               for index, (record, blocks) in enumerate(zip(records, marked))]

    totals = spec_totals(intervals)
    drafting = [b for b in everything if b['rows'] > 1]
    phases_totals = dict(drafts=len(drafting), drafted=sum(b['rows'] - 1 for b in drafting),
                         accepted=sum(b['committed'] - 1 for b in drafting))
    vllm_consistent = None
    if totals is not None and records:
        vllm_consistent = bool(totals['drafts'] <= phases_totals['drafts']
                               and totals['drafted'] <= phases_totals['drafted']
                               and totals['accepted'] <= phases_totals['accepted'])
    audit_agree = sum(1 for m in audit_matches.values() if isinstance(m, int))
    stream_checks = [u.get('stream_agreement') for u in users if u.get('stream_agreement') is not None]
    result = dict(
        sources=dict(phases_records=len(records), phases_unparsed=unparsed,
                     phases_after_shutdown=sum(1 for c in compact if c['after_shutdown']),
                     shutdown_line=shutdown, packed_audit_requests=len(audit),
                     packed_audit_lines=sum(len(v) for v in audit.values()),
                     spec_decoding_intervals=len(intervals)),
        overall=section(everything), packed=section(packed) if packed else None,
        rounds=dict(execute_steps(log_text), packed_rounds=len(PACKED_PHASE_LINE.findall(log_text))),
        users=users, records=compact,
        vllm=dict(totals, phases=phases_totals,
                  coverage_of_drafting_blocks=round(totals['drafts'] / phases_totals['drafts'], 3)
                  if totals and phases_totals['drafts'] else None) if totals else None,
        agreement=dict(
            packed_audit_vs_phases='%d/%d' % (audit_agree, len(audit)) if audit else None,
            vllm_consistent_with_phases=vllm_consistent,
            stream_vs_phases=dict(Counter(stream_checks)) if stream_checks else None,
            users_attributed='%d/%d' % (sum(1 for o in owners if o is not None), len(records))))
    result['summary_line'] = summary_line(result)
    return result


def _fmt(value, pattern='%.3f'):
    return '-' if value is None else pattern % value


def summary_line(result):
    """One '[ACCEPT] ...' line: the full-draft acceptance, the whole run, the active-user rounds, agreement."""
    full = (result.get('overall') or {}).get('full_draft') or {}
    every = (result.get('overall') or {}).get('all') or {}
    positions = (result.get('overall') or {}).get('per_position_full_draft') or []
    steps = (result.get('rounds') or {}).get('decode_steps_by_active_users') or {}
    agreement = result.get('agreement') or {}
    vllm = result.get('vllm') or {}
    per_user = ' '.join('u%d=%s' % (u['user'], _fmt(((u.get('full_draft') or {}).get('mean_emitted')), '%.2f'))
                        for u in result.get('users') or [])
    return ('[ACCEPT] full-draft rounds=%s mean=%s P>8=%s P>11=%s P16=%s max=%s | all rounds=%s mean=%s | '
            'pos1-15=%s | %s | active-user steps %s | vLLM mean=%s over %s drafts | agree audit=%s vllm=%s stream=%s '
            'attributed=%s' % (
                full.get('rounds'), _fmt(full.get('mean_emitted'), '%.2f'), _fmt(full.get('p_emitted_gt_8')),
                _fmt(full.get('p_emitted_gt_11')), _fmt(full.get('p_emitted_eq_16')), full.get('max_emitted'),
                every.get('rounds'), _fmt(every.get('mean_emitted'), '%.2f'),
                ','.join(_fmt(p, '%.2f') for p in positions) or '-', per_user or 'no users',
                ' '.join('%s:%s' % (k, v) for k, v in steps.items()) or '-',
                _fmt(vllm.get('mean_acceptance_length'), '%.2f'), vllm.get('drafts'),
                agreement.get('packed_audit_vs_phases'), agreement.get('vllm_consistent_with_phases'),
                agreement.get('stream_vs_phases'), agreement.get('users_attributed')))


def decode_rates(streams, concurrent=True, acceptance=None):
    """Per-user decode tok/s, three ways; `steady` is the one to set against the programme's
    ~28.3 tok/s (4 x 131k) and 47.5 tok/s (single stream) figures.

    steady - from the acceptance report (`acceptance`, report() above): each user's own packed
      rounds (concurrent) or 16-row rounds (sequential), joined chunk to block, minus the first
      decode round, the terminal one and capture stalls (steady_rate). steady_tok_s = tokens /
      seconds over those rounds; mean_over_median_tok_s = mean emitted / median round.
    median_gap - the programme's earlier convention (docs/200tps-verdict-2026-09-22.md, section 1):
      tokens per round = completion_tokens / text chunks, divided by the median inter-chunk gap
      (gaps_ms: the first token is dropped, so admission stalls sit in a gap the median ignores).
      It understates the steady rate: the short tail rounds dilute tokens per chunk.
    all_active - the wall-clock rate each user saw while every stream was decoding. Each
      timeline drops its first chunk (the prefill's seed token: its arrival is the end of that
      user's prefill, not a decode round). T0 = the moment the LAST stream got its first decode
      chunk; T1 = the moment the FIRST stream got its last chunk. For each user, the chunks
      inside [T0, T1]: the tokens of all but the first over the time from the first to the last
      (chunk-aligned, so no fencepost token). With concurrent=False (a sequential reference run)
      each stream's window is its own first decode chunk to its last. It includes whatever the
      window holds - the 4-row non-packed tail rounds a concurrent run needs before any user's
      budget drops under a packed round's 16 tokens, and capture stalls - so it reads below
      steady. Needs the absolute chunk times the gate's detail streams record (started_s +
      chunk_s, one perf_counter clock across the gate's threads); None without them.
    """
    streams = [s or {} for s in streams]
    steady_by_user = {entry.get('user'): entry.get('steady') or {}
                      for entry in ((acceptance or {}).get('users') or [])}
    users = []
    for index, stream in enumerate(streams):
        gaps = stream.get('gaps_ms') or []
        chunks, completion = stream.get('tokens'), stream.get('completion_tokens')
        per_round = completion / chunks if completion and chunks else None
        median_gap = statistics.median(gaps) if gaps else None
        steady = steady_by_user.get(index) or {}
        users.append(dict(user=index, completion_tokens=completion, text_chunks=chunks,
                          tokens_per_chunk=round(per_round, 3) if per_round else None,
                          median_gap_ms=round(median_gap, 2) if median_gap else None,
                          median_gap_tok_s=round(per_round / (median_gap / 1000.0), 2)
                          if per_round and median_gap else None,
                          steady_tok_s=steady.get('steady_tok_s'),
                          mean_over_median_tok_s=steady.get('mean_over_median_tok_s'),
                          steady_rounds=steady.get('rounds'), steady_stalls_excluded=steady.get('stalls_excluded'),
                          median_round_ms=steady.get('median_round_ms'),
                          ttft_s=stream.get('ttft_s'), finish_reason=stream.get('finish_reason')))
    timeline = []
    for stream in streams:
        started, offsets, tokens = stream.get('started_s'), stream.get('chunk_s'), stream.get('chunk_tokens')
        if started is None or not offsets or not tokens or len(offsets) != len(tokens) or any(t is None for t in tokens) \
                or len(offsets) < 2:
            timeline = None
            break
        # The first chunk is the prefill's seed token, not a decode round.
        timeline.append([(started + offset, count) for offset, count in zip(offsets[1:], tokens[1:])])
    window = None
    if timeline:
        if concurrent:
            start, end = max(t[0][0] for t in timeline), min(t[-1][0] for t in timeline)
            # Negative when a stream finished before the last one started: no all-active window.
            window = dict(seconds=round(end - start, 4), empty=end <= start)
            bounds = [(start, end)] * len(timeline)
        else:
            bounds = [(t[0][0], t[-1][0]) for t in timeline]
        for entry, chunks, (start, end) in zip(users, timeline, bounds):
            inside = [(at, count) for at, count in chunks if start - 1e-9 <= at <= end + 1e-9]
            if len(inside) >= 2 and inside[-1][0] > inside[0][0]:
                tokens = sum(count for _, count in inside[1:])
                seconds = inside[-1][0] - inside[0][0]
                entry.update(all_active_tokens=tokens, all_active_seconds=round(seconds, 4),
                             all_active_chunks=len(inside), all_active_tok_s=round(tokens / seconds, 2))
            else:
                entry.update(all_active_tokens=None, all_active_seconds=None, all_active_chunks=len(inside),
                             all_active_tok_s=None)

    def mean_of(key):
        values = [u.get(key) for u in users if u.get(key)]
        return round(statistics.fmean(values), 2) if values else None

    return dict(concurrent=concurrent, window=window, users=users,
                mean_steady_tok_s=mean_of('steady_tok_s'),
                mean_mean_over_median_tok_s=mean_of('mean_over_median_tok_s'),
                mean_all_active_tok_s=mean_of('all_active_tok_s'),
                mean_median_gap_tok_s=mean_of('median_gap_tok_s'),
                scope=decode_rates.__doc__.strip().split('\n')[0])


def rate_line(rates):
    """One '[RATE] ...' line."""
    per_user = lambda key: ' '.join(_fmt(u.get(key), '%.1f') for u in rates['users'])
    return ('[RATE] per-user tok/s steady %s (mean %s; mean/median %s) | median-gap %s (mean %s) | '
            'all-active %s (mean %s) window %s s' % (
                per_user('steady_tok_s'), _fmt(rates.get('mean_steady_tok_s'), '%.1f'),
                _fmt(rates.get('mean_mean_over_median_tok_s'), '%.1f'),
                per_user('median_gap_tok_s'), _fmt(rates.get('mean_median_gap_tok_s'), '%.1f'),
                per_user('all_active_tok_s'), _fmt(rates.get('mean_all_active_tok_s'), '%.1f'),
                _fmt((rates.get('window') or {}).get('seconds'), '%.2f')))


def main(argv=None):
    """Offline: the acceptance report and rates of any gate run from its artifacts.

        py -3.11 scripts/ci/acceptance_report.py gate/server.log [m3native-gate.json] [--sequential]

    The gate report (m3native-gate.json, or its stdout: the JSON between BEGIN and END) supplies
    the streams and, for a real-text run, the prompt lengths; --sequential defaults to the
    report's sequential_users."""
    import argparse
    from pathlib import Path
    parser = argparse.ArgumentParser(description=main.__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('server_log', type=Path)
    parser.add_argument('gate_report', type=Path, nargs='?')
    parser.add_argument('--sequential', action='store_true', default=None)
    parser.add_argument('--json', type=Path, help='also write {acceptance, decode_rate} here')
    options = parser.parse_args(argv)
    gate = {}
    if options.gate_report:
        text = options.gate_report.read_text(encoding='utf-8', errors='replace')
        begin, end = '<<<M3NATIVE_GATE_JSON_BEGIN>>>', '<<<M3NATIVE_GATE_JSON_END>>>'
        if begin in text and end in text:
            text = text[text.rindex(begin) + len(begin):text.index(end, text.rindex(begin))]
        gate = json.loads(text)
    sequential = options.sequential if options.sequential is not None else bool(gate.get('sequential_users'))
    streams = gate.get('streams') or []
    lengths = [u['prompt_tokens'] for u in (gate.get('real_text') or {}).get('users') or []] or None
    accepted = report(options.server_log.read_text(encoding='utf-8', errors='replace'), streams, lengths,
                      sequential=sequential)
    rates = decode_rates(streams, concurrent=not sequential, acceptance=accepted)
    print(accepted['summary_line'])
    print(rate_line(rates))
    if options.json:
        options.json.write_text(json.dumps(dict(acceptance=accepted, decode_rate=rates), indent=2), encoding='utf-8')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
