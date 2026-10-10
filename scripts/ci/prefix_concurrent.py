"""The tier-concurrent plan's measurements (c2_prefix_gate --plan tier-concurrent / tier-concurrent-timed, prefix_replay.scenario_tier_concurrent; stdlib only, Python 3.7 syntax).

Seven seats decode steadily (real coding text, a fixed token budget, greedy) while an eighth session that left the device pool returns from host RAM, and while a long
fresh prompt displaces blocks into the tier. What the seats see is read from their streamed chunks: every request records the offset of each chunk from its send
(prefix_replay.StreamState.chunk_times), a seat's chunk is one round of the packed decode, so the interval between two chunks of one seat is that seat's round time and the
longest interval inside a window is the longest the seat waited for a token. The windows are the scenario's, read back from the records themselves (windows_from_records: record['sent_s'] is a request's send time in the replay clock, and the
control window is the stretch the scenario slept before it sent the returning turn), so the offline comparator needs nothing but records.jsonl:

  control         the seats alone, nothing else happening (the baseline of the same arm)
  restore         from the returning session's send to its first token: the tier's restore (tier on) or a cold re-prefill of the whole session (control profile)
  restore_decode  from that first token until the returning session's last: it decodes beside the seats as an eighth seat
  spill           from a long fresh prompt's send to its first token: its blocks displace cached ones, which the tier reads out to host RAM (tier on)
  cold            from the returning session's COLD TWIN's send to its first token: the same re-prefill without the tier, beside the same seats (the exactness twin, timed)

and this module turns them into numbers (window_stats), judges a budget (the Lever N decode gap floor), states the round-time regression of a window against the control
window as a number (regression), and compares two arms' seats token by token (compare_seats: the same prompts, greedy, batch invariant, so the same text whatever the tier does).
"""

import json
import math
import os

SEAT_ROLE = 'seat'
SEAT_CASE = 'tier-conc-seat'
RETURN_CASE = 'tier-conc-return'
SPILL_CASE = 'tier-conc-spill'
COLD_CASE = 'tier-conc-cold'
WINDOWS = ('control', 'restore', 'restore_decode', 'spill', 'cold')
# The windows a verdict needs every seat to have decoded across; the cold twin's is read when the seats are still running, and a note when they are not.
JUDGED_WINDOWS = ('control', 'restore', 'restore_decode', 'spill')
DEFAULT_CONTROL_S = 60.0
# The Lever N decode gap floor (QWEN_FAST_LEVERN_MAX_DECODE_GAP_S in the profile; 8 s by default, 0 = off): no seat may wait longer than this between two tokens.
GAP_FLAG = 'QWEN_FAST_LEVERN_MAX_DECODE_GAP_S'
DEFAULT_GAP_BUDGET_S = 8.0
# Fewer common tokens than this prove nothing about a seat's text.
MIN_COMMON_TOKENS = 1000


def percentile(values, fraction):
    """The `fraction` (0..1) quantile of values by linear interpolation; None for no values."""
    ordered = sorted(values)
    if not ordered:
        return None
    if len(ordered) == 1:
        return ordered[0]
    position = fraction * (len(ordered) - 1)
    low, high = int(math.floor(position)), int(math.ceil(position))
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def rounded(value, digits=3):
    return None if value is None else round(value, digits)


def seat_records(records):
    """The seats' records that carry chunk times and a send time (an aborted stream keeps both), in conversation order."""
    seats = [r for r in records if r.get('role') == SEAT_ROLE and r.get('case') == SEAT_CASE and r.get('chunk_times') and r.get('sent_s') is not None]
    return sorted(seats, key=lambda r: (r.get('conv') or '', r.get('sent_s')))


def chunk_times(record):
    """A seat's chunk times in the replay clock."""
    return [record['sent_s'] + offset for offset in record['chunk_times']]


def window_stats(seats, begin, end):
    """{seats: how many seats decoded across the whole window (streaming at its start and still at its end), intervals: the chunk intervals wholly inside it (of those
    seats), p50_s, p99_s, mean_s over them, max_gap_s: the longest interval of ANY seat that overlaps it (a wait that began before the window and ended inside it is the
    window's; a seat that was waiting is never left out for it)}."""
    inside, overlapping, covering = [], [], 0
    for record in seats:
        times = chunk_times(record)
        if len(times) < 2 or times[0] >= end or times[-1] <= begin:
            continue
        whole = times[0] <= begin and times[-1] >= end
        covering += 1 if whole else 0
        for earlier, later in zip(times, times[1:]):
            if later > begin and earlier < end:
                overlapping.append(later - earlier)
                if whole and earlier >= begin and later <= end:
                    inside.append(later - earlier)
    return dict(seats=covering, intervals=len(inside), p50_s=rounded(percentile(inside, 0.5)), p99_s=rounded(percentile(inside, 0.99)),
                mean_s=rounded(sum(inside) / len(inside)) if inside else None, max_gap_s=rounded(max(overlapping)) if overlapping else None)


def first_record(records, case):
    """The first record of a case that was sent and streamed a first token (None otherwise)."""
    found = [r for r in records if r.get('case') == case and r.get('sent_s') is not None]
    return min(found, key=lambda r: r['sent_s']) if found else None


def windows_from_records(records, control_s=None):
    """{name: [begin, end]} in the replay clock, from the returning turn, the long fresh prompt and the cold twin; the control window is `control_s` (default
    DEFAULT_CONTROL_S) before the returning turn was sent. A window whose request streamed nothing is left out."""
    out = {}
    span = float(control_s or DEFAULT_CONTROL_S)
    back = first_record(records, RETURN_CASE)
    if back is not None:
        sent = back['sent_s']
        out['control'] = [max(0.0, sent - span), sent]
        ttft, wall = back.get('ttft_s'), back.get('wall_s')
        if ttft is not None:
            out['restore'] = [sent, sent + ttft]
            if wall is not None and wall > ttft:
                out['restore_decode'] = [sent + ttft, sent + wall]
    for name, case in (('spill', SPILL_CASE), ('cold', COLD_CASE)):
        record = first_record(records, case)
        if record is not None and record.get('ttft_s') is not None:
            out[name] = [record['sent_s'], record['sent_s'] + record['ttft_s']]
    return out


def stats_of(records, control_s=None):
    """{window: window_stats} for the arm's records."""
    seats = seat_records(records)
    return dict((name, window_stats(seats, span[0], span[1])) for name, span in windows_from_records(records, control_s).items() if span[1] > span[0])


def regression(stats, name, field='p99_s'):
    """The percentage by which window `name`'s `field` exceeds the control window's, or None when either is missing."""
    base, other = (stats.get('control') or {}).get(field), (stats.get(name) or {}).get(field)
    if not base or other is None:
        return None
    return round((other / base - 1.0) * 100.0, 1)


def gap_budget(env):
    """The stall budget: the decode gap floor the served profile asks of the Lever N governor, else 8 s (a profile that turns the floor off, 0, is still held to 8 s)."""
    try:
        value = float((env or {}).get(GAP_FLAG, DEFAULT_GAP_BUDGET_S))
    except (TypeError, ValueError):
        return DEFAULT_GAP_BUDGET_S
    return value if value > 0 else DEFAULT_GAP_BUDGET_S


def budget_problems(stats, budget=DEFAULT_GAP_BUDGET_S):
    """[text] for each window whose longest seat wait exceeds the budget (the control window included: a seat that waits that long alone is the stack's, not the tier's)."""
    out = []
    for name in WINDOWS:
        gap = (stats.get(name) or {}).get('max_gap_s')
        if gap is not None and gap > budget:
            out.append('window %s: a seat waited %.3f s between two tokens, over the %.1f s decode gap floor (QWEN_FAST_LEVERN_MAX_DECODE_GAP_S)' % (name, gap, budget))
    return out


def render(label, stats):
    """One line per window: seats, intervals, round time p50/p99/mean, the longest wait, and the p99 regression against the control window."""
    lines = []
    for name in WINDOWS:
        window = stats.get(name)
        if window is None:
            continue
        change = regression(stats, name)
        lines.append('%s window %s: %s seats streaming, %s intervals, round time p50/p99/mean %s/%s/%s s, longest wait %s s%s' % (
            label, name, window['seats'], window['intervals'], window['p50_s'], window['p99_s'], window['mean_s'], window['max_gap_s'],
            '' if name == 'control' else ', round p99 vs control %s' % ('n/a' if change is None else '%+.1f%%' % change)))
    return lines


def seat_texts(records):
    """{conversation name: the seat's output token ids} from the records (an aborted stream keeps the ids it streamed)."""
    return dict((r.get('conv'), r.get('token_ids')) for r in records if r.get('role') == SEAT_ROLE and r.get('case') == SEAT_CASE and r.get('token_ids'))


def compare_seats(first, second, minimum=MIN_COMMON_TOKENS):
    """[row] per seat present in both arms: IDENTICAL when the two token streams agree over their common length (which is at least `minimum` tokens), DIVERGED at the first
    position they differ, SHORT when they agree but are too short to say anything. The seats are aborted when the scenario ends, at different lengths in different arms, so the
    common prefix is what is comparable."""
    rows = []
    a, b = seat_texts(first), seat_texts(second)
    for conv in sorted(set(a) & set(b)):
        common = min(len(a[conv]), len(b[conv]))
        position = next((index for index in range(common) if a[conv][index] != b[conv][index]), None)
        if position is not None:
            rows.append(dict(conv=conv, verdict='DIVERGED', tokens=common, at=position, lengths=[len(a[conv]), len(b[conv])]))
        else:
            rows.append(dict(conv=conv, verdict='IDENTICAL' if common >= minimum else 'SHORT', tokens=common, at=None, lengths=[len(a[conv]), len(b[conv])]))
    return rows


def load_records(path):
    """The records of a results directory (<dir>/records.jsonl, or <dir>/<arm>/records.jsonl for the first arm directory that holds one: a tier-concurrent job's
    results hold exactly one arm) or of a records.jsonl file. IOError when there is none."""
    candidates = [path]
    if os.path.isdir(path):
        candidates = [os.path.join(path, 'records.jsonl')]
        candidates += [os.path.join(path, name, 'records.jsonl') for name in sorted(os.listdir(path)) if name.startswith('tier-concurrent')]
    for candidate in candidates:
        if os.path.isfile(candidate):
            with open(candidate, encoding='utf-8') as handle:
                return [json.loads(line) for line in handle if line.strip()]
    raise IOError('no records.jsonl at %s' % path)
