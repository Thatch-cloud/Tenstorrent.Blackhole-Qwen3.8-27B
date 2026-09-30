"""The lanes gate's reading of one arm: per-lane steady rates, tokens per round, the frame's times and both bars (host only).

The target for coding is ONE fast lane at >= 150 tok/s beside standard lanes at >= 75 tok/s each. This module says, from what an
arm left behind, whether an arm got there and why not:

  the server log   [LANE-ADMIT] (who asked for what, who was granted it), [LANE-ROUND] (every executed round: its kind, block,
                   members, cycle in ms, tokens committed per member, the ratio in force) and [LANE-FRAME]. The rates come from
                   HERE, round by round: a user's rate over a stretch of rounds is its committed tokens over the summed cycle of
                   EVERY round in the stretch, the other lane's rounds included - which is the frame, the thing measured.
  the client       each stream's own chunk timeline (chunk_s / chunk_tokens, one chunk per round): the same rates over the window
                   in which every stream was live. A cross-check on the log that shares none of its parsing.

A STRETCH is a run of consecutive timed rounds with one ratio in force and one live count (QWEN_FAST_LANE_SCHEDULE sweeps the ratio
on one boot; users leaving sweep the live count). A round with no cycle (the sequential tail: ms=-) or a stall (more than STALL_MS
since the last round: a prefill, an idle server) ends a stretch and is in none.

`lanes_report(log_text, streams, ...)` is the arm's report['lanes']; `verdict_lines` renders it; `predict` is the design's closed
form, printed beside every measured stretch so the model is checked against the run (a controller calibration, not a judgement).
Nothing here reads a rate from a model, a reference or a neighbour: an unmeasured cell says so.
"""

import collections
import re

FAST, STANDARD = 'fast', 'standard'
FAST_BAR, STANDARD_BAR = 150.0, 75.0
STALL_MS = 2000.0             # serving_fast_lane.STALL_MS: a round this far from the last is not one decode cadence
MIN_STRETCH_ROUNDS = 24       # packed rounds a stretch needs before its rate is read
MIN_CLIENT_SECONDS = 3.0      # the window in which every stream was live, to read a client rate
CLIENT_GUARD_SECONDS = 1.0    # dropped from each end of that window (the last arrival's first rounds, the first departure's last)
SLACK = 0.99                  # a bar is met at 99% of it: a rate is a ratio of sums over a finite window

ADMIT_LINE = re.compile(r'\[LANE-ADMIT\] request=(\S+) alias=(\S+) asked=(\S+) granted=(\S+) slots=(\S*) reason=(\S+)')
ROUND_LINE = re.compile(r'\[LANE-ROUND\] round=(\d+) kind=(\S+) block=(\S+) members=(\S*) live=(\d+) ms=(\S+) '
                        r'committed=(\S*) switch=([01]) ratio=(\S+)')
FRAME_LINE = re.compile(r'\[LANE-FRAME\] frame=(\d+) k=(\S+) F_ms=(\S+) P_ms=(\S+) sigma_ms=(\S+) tau_fast=(\S+) '
                        r'tau_std_min=(\S+) fast_rate=(\S+) std_rate_min=(\S+) floor=(\S+) reason=(\S+) ratio=(\S+)')
ENGAGED_LINE = re.compile(r'\[LANE\] engaged\b[^\n]*')
DESYNC_LINE = re.compile(r'\[LANE-DESYNC\][^\n]*')
REFUSED_LINE = re.compile(r'\[LANE\] refused:[^\n]*')
SCHED_LINE = re.compile(r'\[LANE-SCHED\][^\n]*')


def number(text):
    """A float from a log field, None for '-' or anything unreadable."""
    try:
        return None if text in ('-', '', None) else float(text)
    except ValueError:
        return None


def parse(log_text):
    """The lane lines of a server log: admits, rounds (in order), frames, and the lines that are problems on sight."""
    log_text = log_text or ''
    admits = [dict(request=m.group(1), alias=m.group(2), asked=m.group(3), granted=m.group(4), reason=m.group(6))
              for m in ADMIT_LINE.finditer(log_text)]
    rounds = []
    for m in ROUND_LINE.finditer(log_text):
        members = [name for name in m.group(4).split(',') if name]
        counts = [int(value) if value.lstrip('-').isdigit() else 0 for value in m.group(7).split(',') if value != '']
        rounds.append(dict(round=int(m.group(1)), kind=m.group(2), block=m.group(3), members=members,
                           live=int(m.group(5)), ms=number(m.group(6)), committed=dict(zip(members, counts)),
                           switch=int(m.group(8)), ratio=m.group(9)))
    frames = [dict(frame=int(m.group(1)), k=number(m.group(2)), F_ms=number(m.group(3)), P_ms=number(m.group(4)),
                   sigma_ms=number(m.group(5)), tau_fast=number(m.group(6)), tau_std_min=number(m.group(7)),
                   fast_rate=number(m.group(8)), std_rate_min=number(m.group(9)), reason=m.group(11), ratio=m.group(12))
              for m in FRAME_LINE.finditer(log_text)]
    engaged = ENGAGED_LINE.search(log_text)
    return dict(admits=admits, rounds=rounds, frames=frames, engaged=bool(engaged),
                engaged_line=engaged.group(0)[:300] if engaged else None,
                desync=[m.group(0)[:300] for m in DESYNC_LINE.finditer(log_text)],
                refused=[m.group(0)[:300] for m in REFUSED_LINE.finditer(log_text)],
                sched=len(SCHED_LINE.findall(log_text)))


def predict(packed_ms, solo_ms, tau_fast, tau_standard, k, sigma_ms=0.0):
    """The design's frame arithmetic: one packed round, k solo rounds and the switch cost paid twice a frame (once when k < 1: a
    solo round in only some frames), the fast user in 1 + k rounds and a standard user in one. tok/s for each lane, or None when
    an input is missing."""
    if None in (packed_ms, solo_ms, tau_fast, tau_standard, k):
        return None
    frame = packed_ms + k * solo_ms + 2.0 * (sigma_ms or 0.0) * min(k, 1.0)
    if frame <= 0:
        return None
    return dict(frame_ms=frame, fast=1000.0 * tau_fast * (1.0 + k) / frame, standard=1000.0 * tau_standard / frame)


def lanes_of(parsed):
    """{alias: lane granted} for every request the book granted (the last grant of an alias)."""
    return {admit['alias']: admit['granted'] for admit in parsed['admits']}


def stretches(parsed, min_rounds=MIN_STRETCH_ROUNDS):
    """Consecutive timed rounds under one (ratio, live count), each read: the standard lanes' rate over the frame, the fast lane's,
    the round times, the switch cost and each lane's tokens per round. Stretches with fewer than `min_rounds` packed rounds are
    listed as unread (rate None), never dropped: an unmeasured cell must say so."""
    lane = lanes_of(parsed)
    runs, current = [], None
    for entry in parsed['rounds']:
        ms = entry['ms']
        timed = ms is not None and ms <= STALL_MS and entry['kind'] in ('packed', 'solo')
        key = (entry['ratio'], entry['live'])
        if not timed:
            current = None
            continue
        if current is None or current['key'] != key:
            current = dict(key=key, rounds=[])
            runs.append(current)
        current['rounds'].append(entry)
    out = []
    for run in runs:
        rounds = run['rounds']
        ratio, live = run['key']
        packed = [entry for entry in rounds if entry['kind'] == 'packed']
        solo = [entry for entry in rounds if entry['kind'] == 'solo']
        total_ms = sum(entry['ms'] for entry in rounds)
        tokens = collections.Counter()
        member_rounds = collections.Counter()
        for entry in rounds:
            for alias, count in entry['committed'].items():
                tokens[alias] += count
                member_rounds[alias] += 1
        # A user counts in the stretch when it rode most of its packed rounds (one that arrived or left inside it does not).
        users = [alias for alias in tokens if sum(1 for entry in packed if alias in entry['members'])
                 >= 0.8 * max(len(packed), 1) or (not packed and lane.get(alias) == FAST)]
        fast = [alias for alias in users if lane.get(alias) == FAST]
        standard = sorted(alias for alias in users if lane.get(alias) != FAST)
        cell = dict(ratio=ratio, live=live, standard_users=len(standard), rounds=len(rounds), packed_rounds=len(packed),
                    solo_rounds=len(solo), seconds=round(total_ms / 1000.0, 4),
                    k_measured=round(len(solo) / len(packed), 3) if packed else None,
                    first_round=rounds[0]['round'], last_round=rounds[-1]['round'])
        cell['P_ms'] = round(sum(e['ms'] for e in packed) / len(packed), 2) if packed else None
        cell['F_ms'] = round(sum(e['ms'] for e in solo) / len(solo), 2) if solo else None
        following = [e['ms'] for e in packed if e['switch']]
        steady = [e['ms'] for e in packed if not e['switch']]
        cell['P_after_switch_ms'] = round(sum(following) / len(following), 2) if following else None
        cell['P_no_switch_ms'] = round(sum(steady) / len(steady), 2) if steady else None
        cell['sigma_ms'] = (round(cell['P_after_switch_ms'] - cell['P_no_switch_ms'], 2)
                            if following and steady else None)
        readable = len(packed) >= min_rounds and total_ms > 0
        cell['read'] = readable
        if fast:
            cell['tau_fast'] = round(tokens[fast[0]] / member_rounds[fast[0]], 3)
        cell['tau_standard'] = ({alias: round(tokens[alias] / member_rounds[alias], 3) for alias in standard}) or None
        if readable:
            if fast:
                cell['fast'] = round(1000.0 * tokens[fast[0]] / total_ms, 2)
            cell['standard'] = {alias: round(1000.0 * tokens[alias] / total_ms, 2) for alias in standard}
            cell['standard_min'] = min(cell['standard'].values()) if cell['standard'] else None
            if cell['tau_standard'] and fast and cell['P_ms'] is not None:
                worst = min(cell['tau_standard'].values())
                cell['predicted'] = predict(cell['P_ms'], cell['F_ms'] if cell['F_ms'] is not None else cell['P_ms'],
                                            cell['tau_fast'], worst, cell['k_measured'] or 0.0, cell['sigma_ms'] or 0.0)
            cell['fast_bar'] = bool(fast) and cell.get('fast', 0.0) >= FAST_BAR * SLACK
            cell['standard_bar'] = bool(standard) and all(rate >= STANDARD_BAR * SLACK for rate in cell['standard'].values())
            cell['both'] = cell['fast_bar'] and cell['standard_bar']
        out.append(cell)
    return out


def client_windows(streams, fast_users=(), guard=CLIENT_GUARD_SECONDS, min_seconds=MIN_CLIENT_SECONDS):
    """Each stream's rate over the one window in which EVERY stream was live (from the last first chunk to the first last chunk,
    less `guard` at each end), from its own chunk timeline: no log line is read. {user: dict(rate, tau)} plus the window, or None
    when the streams were never all live long enough or a stream carries no chunk timeline."""
    timelines = []
    for stream in streams or []:
        stream = stream or {}
        seconds, counts = stream.get('chunk_s'), stream.get('chunk_tokens')
        if not seconds or not counts or len(seconds) != len(counts) or len(seconds) < 3:
            return None
        timelines.append((list(seconds), list(counts)))
    if not timelines:
        return None
    start = max(seconds[0] for seconds, _ in timelines) + guard
    end = min(seconds[-1] for seconds, _ in timelines) - guard
    if end - start < min_seconds:
        return None

    def at(seconds, counts, moment):
        value = 0
        for stamp, count in zip(seconds, counts):
            if stamp > moment:
                break
            value = count
        return value

    users = {}
    for index, (seconds, counts) in enumerate(timelines):
        tokens = at(seconds, counts, end) - at(seconds, counts, start)
        rounds = sum(1 for stamp in seconds if start < stamp <= end)
        users[index] = dict(rate=round(tokens / (end - start), 2), tau=round(tokens / rounds, 3) if rounds else None,
                            lane=FAST if index in set(fast_users) else STANDARD)
    return dict(window_s=round(end - start, 3), users=users)


def stream_rates(streams):
    """Each stream's own decode rate and tokens per round from its chunk timeline, first two chunks (the prefill's seed and the first
    round, whose gap holds the wait for the prefill) and the last (the budget or EOS cut) excluded: what a lone user does (the D0 and per-request-engine arms run one stream at a time)."""
    out = []
    for index, stream in enumerate(streams or []):
        stream = stream or {}
        seconds, counts = stream.get('chunk_s'), stream.get('chunk_tokens')
        if not seconds or not counts or len(seconds) != len(counts) or len(seconds) < 4:
            out.append(dict(user=index, rate=None, tau=None, rounds=0))
            continue
        # From the first round's chunk (its gap holds the prefill's wait) to the last but one (the budget or EOS cuts the last).
        span = seconds[-2] - seconds[1]
        tokens = counts[-2] - counts[1]
        rounds = len(seconds) - 3
        out.append(dict(user=index, rate=round(tokens / span, 2) if span > 0 else None,
                        tau=round(tokens / rounds, 3), rounds=rounds, tokens=tokens))
    return out


def lanes_report(log_text, streams=None, *, fast_users=(), expect_lanes=True, fast_bar=FAST_BAR, standard_bar=STANDARD_BAR,
                 min_rounds=MIN_STRETCH_ROUNDS):
    """The arm's report['lanes']: the log's stretches, the client's window, the admissions and every problem.

    `fast_users` are the stream indices the arm marked fast (--user-lane); `expect_lanes` False is an arm with no lane runtime (the
    per-request-engine baseline, D0 alone): only the client's own rates are read."""
    parsed = parse(log_text)
    problems = []
    report = dict(fast_bar=fast_bar, standard_bar=standard_bar, engaged=parsed['engaged'], engaged_line=parsed['engaged_line'],
                  admits=parsed['admits'],
                  rounds=len(parsed['rounds']), frames=len(parsed['frames']), sched_lines=parsed['sched'],
                  fast_users=list(fast_users))
    report['client'] = client_windows(streams, fast_users)
    report['streams'] = stream_rates(streams)
    if not expect_lanes:
        report['problems'] = problems
        return report
    if not parsed['engaged']:
        problems.append('no "[LANE] engaged" line: the lane runtime never came up on this arm')
    problems += ['%s' % line for line in parsed['refused']]
    problems += ['%s' % line for line in parsed['desync']]
    if not parsed['rounds']:
        problems.append('no [LANE-ROUND] line: no round ran under the lane runtime')
    asked_fast = [admit for admit in parsed['admits'] if admit['asked'] == FAST]
    granted_fast = [admit for admit in parsed['admits'] if admit['granted'] == FAST]
    report['downgrades'] = [admit['alias'] for admit in asked_fast if admit['granted'] != FAST]
    if fast_users and len(asked_fast) < len(fast_users):
        problems.append('%d stream(s) were marked fast but %d admission line(s) say asked=fast: the mark (vllm_xargs qwen_lane) never '
                        'reached the worker' % (len(fast_users), len(asked_fast)))
    if fast_users and not granted_fast:
        problems.append('no request was granted the fast lane')
    cells = stretches(parsed, min_rounds)
    report['stretches'] = cells
    report['both_bars'] = [cell['ratio'] + '/n%d' % cell['standard_users'] for cell in cells if cell.get('both')]
    report['unread'] = [dict(ratio=cell['ratio'], live=cell['live'], packed_rounds=cell['packed_rounds']) for cell in cells
                        if not cell['read']]
    solo_rounds = sum(1 for entry in parsed['rounds'] if entry['kind'] == 'solo')
    packed_rounds = sum(1 for entry in parsed['rounds'] if entry['kind'] == 'packed')
    report['round_kinds'] = dict(solo=solo_rounds, packed=packed_rounds,
                                 other=len(parsed['rounds']) - solo_rounds - packed_rounds)
    if parsed['rounds'] and not solo_rounds and fast_users:
        report['notes'] = ['no solo round ran: every stretch was k = 0 or the fast user never had a standard user beside it']
    report['frame_last'] = parsed['frames'][-1] if parsed['frames'] else None
    report['problems'] = problems
    return report


def best_cells(report):
    """The best stretch per standard-user count: the one where both bars held with the largest fast rate, else the largest fast rate
    with the standard floor held, else the largest fast rate. [(users, cell, 'both' | 'standard-only' | 'neither')]."""
    by_users = collections.defaultdict(list)
    for cell in (report or {}).get('stretches') or []:
        if cell.get('read') and cell.get('fast') is not None:
            by_users[cell['standard_users']].append(cell)
    out = []
    for users in sorted(by_users):
        cells = by_users[users]
        both = [cell for cell in cells if cell.get('both')]
        floor = [cell for cell in cells if cell.get('standard_bar')]
        pool, label = (both, 'both') if both else ((floor, 'standard-only') if floor else (cells, 'neither'))
        out.append((users, max(pool, key=lambda cell: cell['fast']), label))
    return out


def verdict_lines(report):
    """The arm's rows for the gate's summary: one per stretch (measured against predicted), then the best cell per standard-user
    count and the bars."""
    lines = []
    report = report or {}
    for cell in report.get('stretches') or []:
        if not cell.get('read'):
            lines.append('lanes ratio=%s n=%d: %d packed rounds, unread (fewer than %d)' % (
                cell['ratio'], cell['standard_users'], cell['packed_rounds'], MIN_STRETCH_ROUNDS))
            continue
        predicted = cell.get('predicted') or {}
        lines.append('lanes ratio=%s n=%d: fast %s std_min %s tok/s (k %s, P %s F %s sigma %s ms, tau %s / %s)%s' % (
            cell['ratio'], cell['standard_users'], cell.get('fast'), cell.get('standard_min'), cell.get('k_measured'),
            cell.get('P_ms'), cell.get('F_ms'), cell.get('sigma_ms'), cell.get('tau_fast'),
            min(cell['tau_standard'].values()) if cell.get('tau_standard') else None,
            (' | model %.1f / %.1f' % (predicted['fast'], predicted['standard'])) if predicted else ''))
    for users, cell, label in best_cells(report):
        lines.append('lanes BEST n=%d: ratio %s fast %s std_min %s (%s; bars %g / %g)' % (
            users, cell['ratio'], cell.get('fast'), cell.get('standard_min'), label, report.get('fast_bar', FAST_BAR),
            report.get('standard_bar', STANDARD_BAR)))
    client = report.get('client')
    if client:
        lines.append('lanes client window %.1f s: %s' % (client['window_s'], ' '.join(
            '%s%d=%s' % ('F' if entry['lane'] == FAST else 'S', user, entry['rate']) for user, entry in sorted(client['users'].items()))))
    return lines
