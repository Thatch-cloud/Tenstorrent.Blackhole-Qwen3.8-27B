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
stalls (steady_rate) - measures the packed round alone, so it cannot see how long the run spends
outside it (v185, 4 x 131k: a 21.1 tok/s mean where the users' completion rates were 6.4-30.3). The
completion-based figures are the user-facing ones and the ones to set against a goal:
`completion_tok_s` - each user's completion tokens over the time from the decode start (the
moment the LAST prefill ended: its seed chunk) to that user's last chunk - their mean, and
`aggregate_tok_s`, every user's completion tokens over the decode wall (decode start to the last
chunk of any user). The median-gap and all-active figures are kept for comparison.

Active users per round: every '[PHASE] execute total=T new=0 cached=C spec=S' line
(serving_worker_hook, logged as the step begins) is one engine decode step with C live requests.
The packed 64-row step serves a round only when all four users are live and each has at least a
block round (16 tokens) of budget left (serving_packed_step.proposal_rows); every other round -
fewer than four live, or the narrow transition where four are live but one is within 16 tokens of
its budget - runs each live user on its own engine at the sequential widths (four rows, at most
four tokens). So C=4 does not mean packed: `decode_steps_by_active_users` counts steps, and
`rounds_by_live` (round_split) splits them by live count AND by whether the step was packed (its
blocks carry verifier.packed), with each group's median round (the time to the next step's line)
and the tokens each live user took per round - the view that shows where the decode wall goes.
Under QWEN_FAST_PADDED_BLOCK (variable-user rounds M2) the 64-row step also serves two or three
live users as one padded pass, so those rounds split as '3p' / '2p' too; `rounds.packed_by_live`
counts the packed rounds by the live users the verifier itself reported ([PACKED-PHASE] ... live=k,
logged only under that flag), and `rounds.padded` the padded rounds against the eligible ones the
runtime logged (lever_n_m3native_gate.padded_block_summary's reading of the same lines). The
completion rate and aggregate below are the figures that show the gain; the steady rate, which
now includes the padded rounds, does not.

Stdlib only (json, re, statistics, datetime): mounted at /bench beside the gate, imported by the
offline real_text_compare.py too. Python 3.10 compatible.
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
# The same line with its loguru timestamp ('2026-09-24 03:18:37.311 | INFO | ...').
TIMED_EXECUTE_LINE = re.compile(r'([0-9]{4}-[0-9]{2}-[0-9]{2} [0-9]{2}:[0-9]{2}:[0-9]{2}(?:\.[0-9]+)?) \|[^\n]*?'
                                r'\[PHASE\] execute total=([0-9]+) new=([0-9]+) cached=([0-9]+) spec=([0-9]+)')
PACKED_PHASE_LINE = re.compile(r'\[PACKED-PHASE\] round=([0-9]+) users=([0-9]+)')
# QWEN_FAST_PADDED_BLOCK only: the same line closed by the round's live and idle segments, and the
# runtime's padded-round and skipped-round lines (packed_verifier.PADDED_ROUND_MARKER and
# PADDED_SKIPPED_MARKER).
PACKED_PHASE_LIVE = re.compile(r'\[PACKED-PHASE\] round=([0-9]+) users=([0-9]+)[^\n]*? live=([0-9]+) idle=(\S+)')
PADDED_ROUND_LINE = re.compile(r'\[PINDIAG\] packed padded round live=([0-9]+) ')
PADDED_SKIPPED_LINE = re.compile(r'\[PINDIAG\] packed padded skipped live=([0-9]+) eligible=([01]) ')
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


# The whole [PACKED] line (serving_packed_step.AUDIT_LINE), predictions included: the draft-sequence
# comparison between two arms (packed_sequences, compare_packed).
PACKED_LINE = re.compile(r'\[PACKED\] request=(\S+) segment=([0-9]+) position=([0-9]+) prefix=([0-9]+) '
                         r'emitted=([0-9]+) predictions=(\[[^\]\n]*\])')


def packed_sequences(log_text, streams):
    """Each user's packed-round sequence - (position, prefix, emitted, predictions) per [PACKED] line, in
    log order, with no request id and no segment - and the admission order (the user in each segment, by
    the segment of the user's first line). A user is its stream's index (audit_owner); a request no stream
    owns is listed under `unattributed`. Round-fence plan H2's draft check: two arms of one image with the
    admission order fixed (M3NATIVE_STAGGER) must give every user the same sequence."""
    users, segments, unattributed = {}, {}, []
    for match in PACKED_LINE.finditer(log_text):
        request_id, segment = match.group(1), int(match.group(2))
        user = audit_owner(request_id, streams)
        if user is None:
            if request_id not in unattributed:
                unattributed.append(request_id)
            continue
        users.setdefault(user, []).append((int(match.group(3)), int(match.group(4)), int(match.group(5)),
                                           match.group(6)))
        segments.setdefault(user, segment)
    order = [None] * (max(segments.values()) + 1 if segments else 0)
    for user, segment in segments.items():
        order[segment] = user
    return dict(users=users, admission_order=order, unattributed=unattributed)


def packed_fingerprints(log_text, streams):
    """packed_sequences, hashed: per user its round count and a sha256 of its sequence (12 hex), the
    admission order, and the [PACKED] line count - small enough for the gate report, so two arms'
    reports can be compared without their logs."""
    import hashlib

    found = packed_sequences(log_text, streams)
    users = {str(user): dict(rounds=len(sequence), sha256=hashlib.sha256(
        json.dumps(sequence).encode('utf-8')).hexdigest()[:12]) for user, sequence in sorted(found['users'].items())}
    return dict(lines=len(PACKED_LINE.findall(log_text)), admission_order=found['admission_order'],
                users=users, unattributed=len(found['unattributed']))


def compare_packed(log_a, streams_a, log_b, streams_b):
    """Two arms' packed-round sequences, user by user (packed_sequences): identical when the admission
    orders match and every user's whole sequence does; otherwise each differing user's first differing
    round (its index, and both arms' (position, prefix, emitted, predictions) there - None past a
    sequence's end). Sequences from different admission orders are not comparable line by line (the pair
    drafter drafts a user in row 1 differently from row 0): `comparable` says so."""
    a, b = packed_sequences(log_a, streams_a), packed_sequences(log_b, streams_b)
    users = []
    for user in sorted(set(a['users']) | set(b['users'])):
        left, right = a['users'].get(user, []), b['users'].get(user, [])
        entry = dict(user=user, rounds_a=len(left), rounds_b=len(right), identical=left == right)
        if left != right:
            index = next((i for i, (x, y) in enumerate(zip(left, right)) if x != y), min(len(left), len(right)))
            entry['first_difference'] = dict(round=index, a=left[index] if index < len(left) else None,
                                             b=right[index] if index < len(right) else None)
        users.append(entry)
    comparable = a['admission_order'] == b['admission_order']
    return dict(identical=bool(users) and comparable and all(entry['identical'] for entry in users),
                comparable=comparable, admission_order_a=a['admission_order'], admission_order_b=b['admission_order'],
                lines_a=sum(len(s) for s in a['users'].values()), lines_b=sum(len(s) for s in b['users'].values()),
                unattributed_a=len(a['unattributed']), unattributed_b=len(b['unattributed']), users=users)


def compare_line(result):
    """One '[PACKED-COMPARE] ...' line for compare_packed's result."""
    differing = [entry['user'] for entry in result['users'] if not entry['identical']]
    return ('[PACKED-COMPARE] identical=%s comparable=%s order_a=%s order_b=%s lines=%d/%d differing_users=%s'
            % (result['identical'], result['comparable'], result['admission_order_a'], result['admission_order_b'],
               result['lines_a'], result['lines_b'], differing or '-'))


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


def padded_rounds(log_text):
    """QWEN_FAST_PADDED_BLOCK: {'packed_by_live': packed rounds by the verifier's own live count,
    'padded': padded rounds by live count, 'eligible_skipped' / 'ineligible_skipped': rounds of that
    many live users served sequentially, and 'engagement': padded / (padded + eligible skips)}, or
    None when the log carries none of those lines (the flag off: nothing is added to the report)."""
    phases = [match.groups() for match in PACKED_PHASE_LIVE.finditer(log_text)]
    padded = [match.group(1) for match in PADDED_ROUND_LINE.finditer(log_text)]
    skipped = [match.groups() for match in PADDED_SKIPPED_LINE.finditer(log_text)]
    if not phases and not padded and not skipped:
        return None
    by_live = Counter(live for _round, _users, live, _idle in phases)
    by_padded = Counter(padded)
    eligible = sum(1 for _live, flag in skipped if flag == '1')
    total = len(padded) + eligible
    return dict(packed_by_live={live: by_live[live] for live in sorted(by_live, reverse=True)},
                padded={live: by_padded[live] for live in sorted(by_padded, reverse=True)},
                eligible_skipped=eligible, ineligible_skipped=len(skipped) - eligible,
                engagement=round(len(padded) / total, 4) if total else None)


def timed_steps(log_text):
    """[(seconds since the first line, total, new, cached)] for every timestamped '[PHASE] execute'
    line, in log order."""
    from datetime import datetime
    steps, origin = [], None
    for match in TIMED_EXECUTE_LINE.finditer(log_text):
        stamp = match.group(1)
        moment = datetime.strptime(stamp, '%Y-%m-%d %H:%M:%S.%f' if '.' in stamp else '%Y-%m-%d %H:%M:%S')
        origin = moment if origin is None else origin
        total, new, cached, _spec = (int(value) for value in match.groups()[1:])
        steps.append(((moment - origin).total_seconds(), total, new, cached))
    return steps


def round_split(log_text, records):
    """Decode rounds split by live count and by packed or not: the view of where the decode wall
    goes (module docstring, 'Active users per round'), or None without timestamped steps.

    Every '[PHASE] execute ... new=0 cached=C' line is one decode step with C live users, timed to
    the next line when that is a decode step too (each line is logged as its step begins). A step
    followed by anything else - a prefill, which in a sequential reference also holds the client's
    wait for the next request, or the end of the log - is counted but left out of the times.
    Tokens: in the benchmark every user is admitted
    before the first decode step and takes part in every step while it is live, so the k-th
    decode step is every live record's k-th block. That is checked step by step - C must be the
    number of records with a k-th block and T the sum of those blocks' rows - and on the first
    disagreement (a staggered or Lever N run, a sequential reference) the token split and the
    packed/sequential split are withheld (`aligned` False, `alignment` says where) while the
    per-live-count times stay. Per group: rounds, the median and mean round, its wall and share of
    the timed decode wall, the tokens each live user took per round (mean committed) and its rows,
    and per_user_tok_s = those tokens over the mean round."""
    lines = timed_steps(log_text)
    steps = []
    for index, (at, total, new, cached) in enumerate(lines):
        if new == 0 and cached > 0:
            following = lines[index + 1] if index + 1 < len(lines) else None
            decoding = following is not None and following[2] == 0 and following[3] > 0
            steps.append(dict(at=at, total=total, live=cached, seconds=following[0] - at if decoding else None,
                              blocks=None))
    if not steps:
        return None
    histories = [record['blocks'] for record in records]
    aligned, reason = bool(histories), None if histories else 'no phases records'
    if aligned:
        for index, step in enumerate(steps):
            present = [blocks[index] for blocks in histories if len(blocks) > index]
            rows = sum(block['rows'] for block in present)
            if len(present) != step['live'] or rows != step['total']:
                aligned = False
                reason = 'step %d: %d live over %d rows, the records hold %d blocks over %d rows' % (
                    index + 1, step['live'], step['total'], len(present), rows)
                break
            step['blocks'] = present
        if aligned and max(len(blocks) for blocks in histories) != len(steps):
            aligned = False
            reason = '%d decode steps, the longest record %d blocks' % (len(steps), max(len(b) for b in histories))
    groups = {}
    for step in steps:
        packed = all(block['packed'] for block in step['blocks']) if aligned else None
        group = groups.setdefault((step['live'], packed), dict(live=step['live'], packed=packed, rounds=0, times=[],
                                                               tokens=0, rows=0, user_rounds=0, timed_tokens=0,
                                                               timed_user_rounds=0))
        group['rounds'] += 1
        if step['seconds'] is not None:
            group['times'].append(step['seconds'])
        if aligned:
            committed = sum(block['committed'] for block in step['blocks'])
            group['tokens'] += committed
            group['rows'] += sum(block['rows'] for block in step['blocks'])
            group['user_rounds'] += step['live']
            if step['seconds'] is not None:
                group['timed_tokens'] += committed
                group['timed_user_rounds'] += step['live']
    timed_wall = sum(step['seconds'] for step in steps if step['seconds'] is not None)
    out = []
    for key in sorted(groups, key=lambda key: (-key[0], not key[1])):
        group = groups[key]
        times = group.pop('times')
        wall = sum(times)
        mean_round = statistics.fmean(times) if times else None
        per_round = group['tokens'] / group['user_rounds'] if aligned and group['user_rounds'] else None
        timed_per_round = (group['timed_tokens'] / group['timed_user_rounds']
                           if aligned and group['timed_user_rounds'] else None)
        out.append(dict(live=group['live'], packed=group['packed'], rounds=group['rounds'], timed_rounds=len(times),
                        median_round_ms=round(statistics.median(times) * 1000.0, 2) if times else None,
                        mean_round_ms=round(mean_round * 1000.0, 2) if mean_round else None,
                        wall_s=round(wall, 3), wall_share=round(wall / timed_wall, 4) if timed_wall else None,
                        tokens=group['tokens'] if aligned else None,
                        tokens_per_user_per_round=round(per_round, 3) if per_round is not None else None,
                        rows_per_user=round(group['rows'] / group['user_rounds'], 3)
                        if aligned and group['user_rounds'] else None,
                        per_user_tok_s=round(timed_per_round / mean_round, 2)
                        if timed_per_round is not None and mean_round else None))
    return dict(aligned=aligned, alignment=reason or 'every decode step is its live records\' next block',
                decode_steps=len(steps), timed_decode_s=round(timed_wall, 3), groups=out)


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
    padded = padded_rounds(log_text)
    result = dict(
        sources=dict(phases_records=len(records), phases_unparsed=unparsed,
                     phases_after_shutdown=sum(1 for c in compact if c['after_shutdown']),
                     shutdown_line=shutdown, packed_audit_requests=len(audit),
                     packed_audit_lines=sum(len(v) for v in audit.values()),
                     spec_decoding_intervals=len(intervals)),
        overall=section(everything), packed=section(packed) if packed else None,
        rounds=dict(execute_steps(log_text), packed_rounds=len(PACKED_PHASE_LINE.findall(log_text)),
                    **(padded or {})),
        rounds_by_live=round_split(log_text, records),
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
            'attributed=%s | by live %s' % (
                full.get('rounds'), _fmt(full.get('mean_emitted'), '%.2f'), _fmt(full.get('p_emitted_gt_8')),
                _fmt(full.get('p_emitted_gt_11')), _fmt(full.get('p_emitted_eq_16')), full.get('max_emitted'),
                every.get('rounds'), _fmt(every.get('mean_emitted'), '%.2f'),
                ','.join(_fmt(p, '%.2f') for p in positions) or '-', per_user or 'no users',
                ' '.join('%s:%s' % (k, v) for k, v in steps.items()) or '-',
                _fmt(vllm.get('mean_acceptance_length'), '%.2f'), vllm.get('drafts'),
                agreement.get('packed_audit_vs_phases'), agreement.get('vllm_consistent_with_phases'),
                agreement.get('stream_vs_phases'), agreement.get('users_attributed'),
                by_live_label(result.get('rounds_by_live')))) + padded_label(result.get('rounds'))


def padded_label(rounds):
    """' | padded 3:40 2:12 of 55 eligible (0.95)' under QWEN_FAST_PADDED_BLOCK; '' otherwise, so a
    run without the flag keeps exactly its summary line."""
    rounds = rounds or {}
    if 'padded' not in rounds:
        return ''
    padded = rounds['padded']
    eligible = sum(padded.values()) + rounds.get('eligible_skipped', 0)
    return ' | padded %s of %d eligible (%s)' % (' '.join('%s:%d' % item for item in padded.items()) or '0', eligible,
                                                 _fmt(rounds.get('engagement'), '%.2f'))


def by_live_label(split):
    """'4p:25x247ms/5.33 4s:4x520ms/3.19 3s:55x376ms/2.87 ...': per group, live count, p(acked) /
    s(equential) / ? (not aligned), rounds x median round / tokens per user per round."""
    if not split:
        return '-'
    kind = {True: 'p', False: 's', None: '?'}
    return ' '.join('%d%s:%dx%sms/%s' % (group['live'], kind[group['packed']], group['rounds'],
                                          _fmt(group['median_round_ms'], '%.0f'),
                                          _fmt(group['tokens_per_user_per_round'], '%.2f'))
                    for group in split['groups'])


def completion_rates(streams, concurrent=True):
    """The user-facing rates, from the detail streams' absolute chunk times (started_s + chunk_s),
    or None without them for every stream.

    Concurrent: decode_start is the moment the LAST stream got its seed chunk - the end of the last
    prefill; the benchmark holds every user's first decode round until then, so it is when decode
    begins for everyone. Each user's completion_tok_s is its completion_tokens (the seed included:
    256 for a full budget) over its finish_s, the time from decode_start to its last chunk. The
    decode wall runs from decode_start to the last chunk of any stream; aggregate_tok_s is every
    stream's completion tokens over it. Sequential (a reference run of lone streams): each user's
    own window, seed chunk to last chunk; no decode wall and no aggregate."""
    timelines = []
    for stream in streams:
        stream = stream or {}
        started, offsets, completion = stream.get('started_s'), stream.get('chunk_s'), stream.get('completion_tokens')
        if started is None or not offsets or len(offsets) < 2 or not completion:
            return None
        timelines.append((started + offsets[0], started + offsets[-1], completion))
    if not timelines:
        return None
    start = max(seed for seed, _, _ in timelines)
    users = []
    for index, (seed, last, completion) in enumerate(timelines):
        begin = start if concurrent else seed
        finish = last - begin
        users.append(dict(user=index, completion_tokens=completion, finish_s=round(finish, 4),
                          completion_tok_s=round(completion / finish, 2) if finish > 0 else None))
    rates = [user['completion_tok_s'] for user in users if user['completion_tok_s']]
    wall = max(last for _, last, _ in timelines) - start if concurrent else None
    tokens = sum(completion for _, _, completion in timelines)
    return dict(decode_start='the last seed chunk' if concurrent else "each stream's own seed chunk",
                decode_wall_s=round(wall, 4) if wall is not None else None, tokens=tokens,
                aggregate_tok_s=round(tokens / wall, 2) if wall else None,
                mean_completion_tok_s=round(statistics.fmean(rates), 2) if rates else None, users=users)


def decode_rates(streams, concurrent=True, acceptance=None):
    """Per-user decode tok/s, four ways. `completion` (completion_rates) is what the users got and
    the figure a goal is judged by; `steady` is the packed round alone, the one to set against the
    programme's earlier ~28.3 tok/s (4 x 131k) and 47.5 tok/s (single stream) figures.

    completion - each user's completion tokens over its finish time from the decode start, their
      mean, and the aggregate over the decode wall (completion_rates).
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

    completion = completion_rates(streams, concurrent)
    for entry in users:
        mine = (completion or {}).get('users', [])[entry['user']] if completion else {}
        entry.update(completion_tok_s=mine.get('completion_tok_s'), finish_s=mine.get('finish_s'))

    def mean_of(key):
        values = [u.get(key) for u in users if u.get(key)]
        return round(statistics.fmean(values), 2) if values else None

    return dict(concurrent=concurrent, window=window, users=users,
                mean_steady_tok_s=mean_of('steady_tok_s'),
                mean_mean_over_median_tok_s=mean_of('mean_over_median_tok_s'),
                mean_all_active_tok_s=mean_of('all_active_tok_s'),
                mean_median_gap_tok_s=mean_of('median_gap_tok_s'),
                mean_completion_tok_s=(completion or {}).get('mean_completion_tok_s'),
                aggregate_tok_s=(completion or {}).get('aggregate_tok_s'),
                decode_wall_s=(completion or {}).get('decode_wall_s'), completion=completion,
                scope=decode_rates.__doc__.strip().split('\n')[0])


def rate_line(rates):
    """One '[RATE] ...' line."""
    per_user = lambda key: ' '.join(_fmt(u.get(key), '%.1f') for u in rates['users'])
    return ('[RATE] per-user tok/s steady %s (mean %s; mean/median %s) | median-gap %s (mean %s) | '
            'all-active %s (mean %s) window %s s | completion %s (mean %s) aggregate %s over a %s s decode wall' % (
                per_user('steady_tok_s'), _fmt(rates.get('mean_steady_tok_s'), '%.1f'),
                _fmt(rates.get('mean_mean_over_median_tok_s'), '%.1f'),
                per_user('median_gap_tok_s'), _fmt(rates.get('mean_median_gap_tok_s'), '%.1f'),
                per_user('all_active_tok_s'), _fmt(rates.get('mean_all_active_tok_s'), '%.1f'),
                _fmt((rates.get('window') or {}).get('seconds'), '%.2f'),
                per_user('completion_tok_s'), _fmt(rates.get('mean_completion_tok_s'), '%.1f'),
                _fmt(rates.get('aggregate_tok_s'), '%.1f'), _fmt(rates.get('decode_wall_s'), '%.2f')))


# S2 PATH RECORDS (s2-design.md W11 and section 6.1). Concurrent and solo run two arithmetic paths in S2:
# a round is either served by the packed block (P: one '[PACKED] request=' line per user,
# serving_packed_step.AUDIT_LINE under QWEN_FAST_PACKED_AUDIT) or by the sequential step (S: a
# '[SEQUENTIAL] request= position= prefix=' line, or the '[SEQ-PUBLISH] request= rows= prefix=' step line
# serving_sequential_step.SEQ_PUBLISH_LINE writes under QWEN_FAST_SEQ_PUBLISH_LOG, which the C2 image
# bakes). Per user and round: the path, the ticket position, the committed prefix (state rows), the
# emitted tokens, the ticket rows and the boundary cap (W4's cap=), each where its line carries it. Fields
# are read by name, so a field added to a line never shifts another. A [SEQ-PUBLISH] line names no
# position: it is chained from the user's prompt length (its first decode ticket sits there) by every
# round's prefix, and the chain is checked against each position a [PACKED] or [SEQUENTIAL] line does carry
# (position_mismatches). round_split cannot give this for staggered arms (it needs every live user in every
# step); these records explain a divergence and count SAME_PATH users, they judge nothing themselves.
PATH_LINE = re.compile(r'\[(PACKED|SEQUENTIAL|SEQ-PUBLISH)\] request=(\S+) ([^\n]*)')
PATH_FIELD = re.compile(r'(?<![A-Za-z0-9_])(segment|position|prefix|emitted|rows|cap)=([0-9]+|n/a)(?![0-9A-Za-z_])')
PATH_KINDS = {'PACKED': 'P', 'SEQUENTIAL': 'S', 'SEQ-PUBLISH': 'S'}
# One record in a report: '<P|S><position>:<prefix>:<emitted>:<rows>:<cap>', '-' where unknown, records
# space-separated in round order (real_text_compare.decode_paths reads it back).
PATH_FORMAT_FIELDS = ('position', 'prefix', 'emitted', 'rows', 'cap')
EMITTED_FIELD = re.compile(r'(?<![A-Za-z0-9_])emitted=([0-9]+)')
AUDIT_MS_FIELD = re.compile(r'\[EXTENT-AUDIT\] round=[0-9]+ [^\n]*?(?<![A-Za-z0-9_])ms=([0-9.]+)')


def path_records(log_text, streams, prompt_lengths=None):
    """{users: {user: [record, ...]}, ...} - every [PACKED], [SEQUENTIAL] and [SEQ-PUBLISH] step line
    attributed to its user (audit_owner), in log order, each record {round, path, position, prefix, emitted,
    rows, cap, logged_position}. Positions a line does not carry are chained from the user's prompt length
    (prompt_lengths) by the prefixes; position_checks counts the logged positions the chain was checked
    against and position_mismatches lists where it disagreed (a missed line or a prefix read wrong)."""
    users, unattributed, malformed = {}, [], 0
    for match in PATH_LINE.finditer(log_text):
        kind, request_id, rest = match.groups()
        if kind == 'SEQ-PUBLISH' and not rest.startswith('rows='):
            continue   # the same step's stages / splits lines
        fields = {}
        for name, value in PATH_FIELD.findall(rest):
            fields.setdefault(name, None if value == 'n/a' else int(value))
        path = PATH_KINDS[kind]
        if path == 'P' and (fields.get('position') is None or fields.get('prefix') is None):
            malformed += 1
            continue
        user = audit_owner(request_id, streams)
        if user is None:
            if request_id not in unattributed:
                unattributed.append(request_id)
            continue
        users.setdefault(user, []).append(dict(path=path, position=fields.get('position'), prefix=fields.get('prefix'),
                                               emitted=fields.get('emitted'), rows=fields.get('rows'),
                                               cap=fields.get('cap'),
                                               logged_position=fields.get('position') is not None))
    checks, mismatches = 0, []
    lengths = list(prompt_lengths or [])
    for user, records in users.items():
        expected = lengths[user] if user < len(lengths) else None
        for index, record in enumerate(records):
            record['round'] = index
            if record['position'] is not None:
                if expected is not None:
                    checks += 1
                    if expected != record['position']:
                        mismatches.append(dict(user=user, round=index, chained=expected, logged=record['position']))
                expected = record['position']
            else:
                record['position'] = expected
            expected = expected + record['prefix'] if expected is not None and record['prefix'] is not None else None
    return dict(users=users, unattributed=unattributed, malformed=malformed, position_checks=checks,
                position_mismatches=mismatches[:16], position_mismatch_count=len(mismatches))


def encode_paths(records):
    """One user's records as the report carries them (PATH_FORMAT_FIELDS, space-separated)."""
    def text(value):
        return '-' if value is None else str(value)
    return ' '.join(record['path'] + ':'.join(text(record.get(name)) for name in PATH_FORMAT_FIELDS)
                    for record in records)


def decode_steps(log_text):
    """Every decode step - a timestamped '[PHASE] execute ... new=0 cached=C' line (serving_worker_hook, logged
    as the step begins) - with what its stretch of the log, up to the next execute line, says: the emitted
    count of each '[PACKED] request=' line (a packed round), the sequential step lines, and the extent audit's
    ms ([EXTENT-AUDIT], S2 gate arms). Timed to the next execute line when that is a decode step too, as
    round_split times them. Unlike round_split it needs no alignment of records to steps, so a staggered
    or mixed-length arm is read as well."""
    from datetime import datetime
    steps, current, origin = [], None, None
    for line in log_text.split('\n'):
        match = TIMED_EXECUTE_LINE.search(line)
        if match:
            stamp = match.group(1)
            moment = datetime.strptime(stamp, '%Y-%m-%d %H:%M:%S.%f' if '.' in stamp else '%Y-%m-%d %H:%M:%S')
            origin = moment if origin is None else origin
            at = (moment - origin).total_seconds()
            _total, new, cached, _spec = (int(value) for value in match.groups()[1:])
            if current is not None:
                current['next'] = (at, new, cached)
            current = dict(at=at, new=new, live=cached, packed=[], sequential=0, audit_ms=None, next=None)
            steps.append(current)
            continue
        if current is None:
            continue
        if '[PACKED] request=' in line:
            emitted = EMITTED_FIELD.search(line)
            current['packed'].append(int(emitted.group(1)) if emitted else 0)
        elif '[SEQUENTIAL] request=' in line or ('[SEQ-PUBLISH] request=' in line and ' rows=' in line):
            current['sequential'] += 1
        elif '[EXTENT-AUDIT] round=' in line:
            audit = AUDIT_MS_FIELD.search(line)
            if audit:
                current['audit_ms'] = float(audit.group(1))
    decode = []
    for step in steps:
        if step['new'] != 0 or step['live'] <= 0:
            continue
        following = step['next']
        seconds = following[0] - step['at'] if following is not None and following[1] == 0 and following[2] > 0 else None
        decode.append(dict(at=step['at'], live=step['live'], seconds=seconds, packed=step['packed'],
                           sequential=step['sequential'], audit_ms=step['audit_ms']))
    return decode


def live_rate(log_text, live=4):
    """The packed rounds with `live` live users (decode steps whose stretch carries `live` [PACKED] lines):
    how many, the median round, the median extent-audit ms inside them, the tokens each user took per round,
    and the per-user decode rate = those tokens over the median round - and the same net of the audit's ms
    (an S2 gate arm's audit reads back after the replay, so it lengthens the round it checks). None without
    a timed round."""
    rounds = [step for step in decode_steps(log_text) if step['live'] == live and len(step['packed']) == live]
    timed = [step for step in rounds if step['seconds'] is not None and step['seconds'] > 0]
    if not timed:
        return None
    median = statistics.median(step['seconds'] for step in timed)
    emitted = [value for step in rounds for value in step['packed']]
    per_round = sum(emitted) / len(emitted)   # not fmean: c2_serving_gate reads this on the rig host (3.7 syntax)
    audits = [step['audit_ms'] for step in timed if step['audit_ms'] is not None]
    audit_ms = statistics.median(audits) if audits else None
    net = median - (audit_ms or 0.0) / 1000.0
    return dict(live=live, rounds=len(rounds), timed_rounds=len(timed), median_round_ms=round(median * 1000.0, 2),
                median_audit_ms=round(audit_ms, 3) if audit_ms is not None else None,
                net_median_round_ms=round(net * 1000.0, 2), tokens_per_user_per_round=round(per_round, 3),
                per_user_tok_s=round(per_round / median, 2),
                net_per_user_tok_s=round(per_round / net, 2) if net > 0 else None)


def read_gate_report(path):
    """A gate report: m3native-gate.json, or the gate's stdout with the JSON between BEGIN and END."""
    text = path.read_text(encoding='utf-8', errors='replace')
    begin, end = '<<<M3NATIVE_GATE_JSON_BEGIN>>>', '<<<M3NATIVE_GATE_JSON_END>>>'
    if begin in text and end in text:
        text = text[text.rindex(begin) + len(begin):text.index(end, text.rindex(begin))]
    return json.loads(text)


def main(argv=None):
    """Offline: the acceptance report and rates of any gate run from its artifacts.

        py -3.11 scripts/ci/acceptance_report.py gate/server.log [m3native-gate.json] [--sequential]

    The gate report (m3native-gate.json, or its stdout: the JSON between BEGIN and END) supplies
    the streams and, for a real-text run, the prompt lengths; --sequential defaults to the
    report's sequential_users.

    Two arms' draft sequences (round-fence plan H2: a B arm against a same-image A arm, admission
    order fixed by M3NATIVE_STAGGER), user by user over their [PACKED] lines:

        py -3.11 scripts/ci/acceptance_report.py B/server.log B/m3native-gate.json \\
            --packed-reference A/server.log A/m3native-gate.json

    prints '[PACKED-COMPARE] identical=True ...' when every user's (position, prefix, emitted,
    predictions) sequence and the admission order match (compare_packed), and exits 3 when not."""
    import argparse
    from pathlib import Path
    parser = argparse.ArgumentParser(description=main.__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('server_log', type=Path)
    parser.add_argument('gate_report', type=Path, nargs='?')
    parser.add_argument('--sequential', action='store_true', default=None)
    parser.add_argument('--json', type=Path, help='also write {acceptance, decode_rate} here')
    parser.add_argument('--packed-reference', type=Path, nargs=2, metavar=('SERVER_LOG', 'GATE_REPORT'),
                        help="a reference arm's server.log and gate report: compare the two arms' [PACKED] "
                             "sequences user by user (compare_packed)")
    options = parser.parse_args(argv)
    gate = read_gate_report(options.gate_report) if options.gate_report else {}
    sequential = options.sequential if options.sequential is not None else bool(gate.get('sequential_users'))
    streams = gate.get('streams') or []
    lengths = [u['prompt_tokens'] for u in (gate.get('real_text') or {}).get('users') or []] or None
    log_text = options.server_log.read_text(encoding='utf-8', errors='replace')
    accepted = report(log_text, streams, lengths, sequential=sequential)
    rates = decode_rates(streams, concurrent=not sequential, acceptance=accepted)
    print(accepted['summary_line'])
    print(rate_line(rates))
    compared = None
    if options.packed_reference:
        reference_log, reference_report = options.packed_reference
        compared = compare_packed(reference_log.read_text(encoding='utf-8', errors='replace'),
                                  read_gate_report(reference_report).get('streams') or [], log_text, streams)
        print(compare_line(compared))
    if options.json:
        payload = dict(acceptance=accepted, decode_rate=rates)
        if compared is not None:
            payload['packed_compare'] = compared
        options.json.write_text(json.dumps(payload, indent=2), encoding='utf-8')
    return 3 if compared is not None and not compared['identical'] else 0


if __name__ == '__main__':
    raise SystemExit(main())
