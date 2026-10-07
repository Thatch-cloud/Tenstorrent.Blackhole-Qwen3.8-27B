"""Round-time comparison of the combined window's paired timing jobs (docs/tp4-combined-window.md, the T read).

WHAT IT READS. The container log of a timed smoke job (the arms carry no audit): every decode step with `live` live users that ran as one fully
packed round, from its '[PHASE] execute' line to the next one (acceptance_report.decode_steps' timing), with the mean position of its users (the
'[PACKED] request= ... position=' lines). It is NOT speed_window_compare.py, which compares texts and accepted prefixes between two matrix-gate result trees
and reads no round time at all (a smoke job never writes those trees).

WHAT IT KEEPS. Only rounds that are (a) eight live and fully packed, and (b) clean of Lever N: no 'lever N step ... kind=prefill' line inside the round or
inside the step before it. A paired arm without Lever N has none; the exclusion is the attribution check (the round delta is W2's, never a prefill step's).

HOW IT PAIRS. By mean-context bucket (default 1,024 tokens), never by the exact position tuple: the two arms reach eight-live at different offsets (a
Lever N arm admits its seats one prefill at a time), so exact matching barely overlaps. Per bucket the medians of the two arms' round times are
compared; the pair's delta is the mean of the per-bucket deltas weighted by min(nA, nB). A pair with fewer than `min_matched` matched rounds is VOID (not
measured: never GO, never NO-GO). The A-to-A drift floor is the spread of the A runs' medians.

PER LENGTH. The timed smoke runs four shapes (steady about 4k, eight at 32k, eight at about 120k, the skewed eight with two 254k users) and a loss at one
length must never be averaged away by a gain at another: every read (pair, floor, minimum matched rounds, judge) is made inside a mean-position WINDOW
(WINDOWS below; the shapes' mean positions are about 4.5k, 33.5k, 121k and 67k and the windows do not overlap), and `judge` prints one verdict per length plus
the W2 overall read (NO-GO when 32k, 128k or the skew reads NO-GO; GO when 32k and 128k read GO and the skew does not read NO-GO).

LOAD. The rig hosts CI and builds and the serving container's own threads raise the host load average, so the maximum is calibrated from a SERVING baseline
(`load-limit`: the median of a serving job's [LOAD] samples plus a margin), applied to each job's MEDIAN (one spike is no void), and a pair also goes VOID when its
two jobs' medians differ by more than a set gap (they must have run under the same host conditions). Texts are judged separately, by comparing the smoke JSON
hashes (`texts_equal`).

Stdlib only.
"""

import argparse
import json
import re
import statistics
import sys
from datetime import datetime
from pathlib import Path

EXECUTE_LINE = re.compile(r'([0-9]{4}-[0-9]{2}-[0-9]{2} [0-9]{2}:[0-9]{2}:[0-9]{2}(?:\.[0-9]+)?) \|[^\n]*?'
                          r'\[PHASE\] execute total=([0-9]+) new=([0-9]+) cached=([0-9]+) spec=([0-9]+)')
POSITION = re.compile(r'(?<![A-Za-z0-9_])position=([0-9]+)')
LEVERN_PREFILL = re.compile(r'lever N step n=\S+ kind=prefill')
LOAD_LINE = re.compile(r'\[LOAD\] (\d+) (\d+(?:\.\d+)?)')
DEFAULT_BUCKET = 1024
# Matched rounds a pair needs INSIDE ONE WINDOW: a shape's 800-token answers take about 240 packed rounds at about 3.3 tokens a round, most of them eight-live, so
# 100 matched is attainable per shape (200 was the pooled figure over four shapes and could never be met by one).
DEFAULT_MIN_MATCHED = 100
DEFAULT_LOAD_MARGIN = 3.0   # the load maximum = the serving baseline's median + this
DEFAULT_LOAD_GAP = 2.0      # the two jobs of a pair: medians no further apart than this
# name -> (mean position at or above, below): the four timed shapes of the T block
WINDOWS = {'steady': (0, 8192), '32k': (28672, 40960), 'skew': (57344, 77824), '128k': (106496, 139264)}
LENGTH_ORDER = ('steady', '32k', '128k', 'skew')
DEFAULT_LIVE = 8
GO_PAIRS = 3          # the window's three A/B pairs: GO needs all three measured and negative
NO_GO_PAIRS = 2       # NO-GO: at least two of the three positive
GO_MIN_FRACTION = 0.01


def timed_rounds(log_text, live=DEFAULT_LIVE):
    """[dict(mean_position, seconds)] of the clean fully packed `live`-user rounds of a log, plus the count of rounds dropped for a Lever N prefill step."""
    steps, current, previous_prefill = [], None, False
    origin = None
    for line in log_text.split('\n'):
        match = EXECUTE_LINE.search(line)
        if match:
            stamp = match.group(1)
            moment = datetime.strptime(stamp, '%Y-%m-%d %H:%M:%S.%f' if '.' in stamp else '%Y-%m-%d %H:%M:%S')
            origin = moment if origin is None else origin
            at = (moment - origin).total_seconds()
            _total, new, cached, _spec = (int(value) for value in match.groups()[1:])
            if current is not None:
                current['next'] = (at, new, cached)
            previous_prefill = bool(current and current['prefill'])
            current = dict(at=at, new=new, live=cached, positions=[], prefill=False, after_prefill=previous_prefill, next=None)
            steps.append(current)
            continue
        if current is None:
            continue
        if '[PACKED] request=' in line:
            position = POSITION.search(line)
            if position:
                current['positions'].append(int(position.group(1)))
        elif LEVERN_PREFILL.search(line):
            current['prefill'] = True
    rounds, dropped = [], 0
    for step in steps:
        if step['new'] != 0 or step['live'] != live or len(step['positions']) != live:
            continue
        following = step['next']
        if following is None or following[1] != 0 or following[2] <= 0:
            continue
        seconds = following[0] - step['at']
        if seconds <= 0:
            continue
        if step['prefill'] or step['after_prefill']:
            dropped += 1
            continue
        rounds.append(dict(mean_position=sum(step['positions']) / float(live), seconds=seconds))
    return rounds, dropped


def resolve_window(window):
    """(low, high) of a window name or a (low, high) pair; None for no window."""
    if window is None:
        return None
    if isinstance(window, str):
        if window not in WINDOWS:
            raise ValueError('unknown window %r (known: %s)' % (window, ', '.join(LENGTH_ORDER)))
        return WINDOWS[window]
    return (int(window[0]), int(window[1]))


def in_window(rounds, window):
    bounds = resolve_window(window)
    if bounds is None:
        return list(rounds)
    return [item for item in rounds if bounds[0] <= item['mean_position'] < bounds[1]]


def buckets(rounds, bucket=DEFAULT_BUCKET):
    out = {}
    for item in rounds:
        out.setdefault(int(item['mean_position'] // bucket), []).append(item['seconds'])
    return out


def compare(log_a, log_b, live=DEFAULT_LIVE, bucket=DEFAULT_BUCKET, min_matched=DEFAULT_MIN_MATCHED, window=None):
    """The paired reading of arm A (the control) against arm B: dict(verdict 'MEASURED' | 'VOID', matched, delta_ms (B minus A: negative is faster),
    a_median_ms, b_median_ms, per-bucket rows). VOID carries its reason. `window` (a WINDOWS name or (low, high) mean positions) keeps only the rounds of
    one timed shape: its matched count, minimum and medians are that shape's alone."""
    rounds_a, dropped_a = timed_rounds(log_a, live)
    rounds_b, dropped_b = timed_rounds(log_b, live)
    rounds_a, rounds_b = in_window(rounds_a, window), in_window(rounds_b, window)
    a, b = buckets(rounds_a, bucket), buckets(rounds_b, bucket)
    rows, weight, total = [], 0, 0.0
    for key in sorted(set(a) & set(b)):
        matched = min(len(a[key]), len(b[key]))
        delta = statistics.median(b[key]) - statistics.median(a[key])
        rows.append(dict(bucket_start=key * bucket, matched=matched, a_median_ms=round(statistics.median(a[key]) * 1000, 2),
                         b_median_ms=round(statistics.median(b[key]) * 1000, 2), delta_ms=round(delta * 1000, 2)))
        weight += matched
        total += delta * matched
    result = dict(live=live, bucket=bucket, window=window if isinstance(window, str) else (list(window) if window else None), rounds_a=len(rounds_a), rounds_b=len(rounds_b), dropped_for_prefill=dict(a=dropped_a, b=dropped_b),
                  matched=weight, min_matched=min_matched, buckets=rows)
    if weight < min_matched:
        result.update(verdict='VOID', reason='%d matched rounds, under the pre-registered %d' % (weight, min_matched))
        return result
    matched_a = [item['seconds'] for item in rounds_a if int(item['mean_position'] // bucket) in set(a) & set(b)]
    matched_b = [item['seconds'] for item in rounds_b if int(item['mean_position'] // bucket) in set(a) & set(b)]
    result.update(verdict='MEASURED', delta_ms=round(total / weight * 1000, 2), a_median_ms=round(statistics.median(matched_a) * 1000, 2),
                  b_median_ms=round(statistics.median(matched_b) * 1000, 2))
    result['ratio_b_over_a'] = round(result['b_median_ms'] / result['a_median_ms'], 4)
    return result


def median_ms(log_text, live=DEFAULT_LIVE, window=None):
    rounds, _ = timed_rounds(log_text, live)
    rounds = in_window(rounds, window)
    return statistics.median([item['seconds'] for item in rounds]) * 1000 if rounds else None


def drift_floor_ms(a_logs, live=DEFAULT_LIVE, window=None):
    """The A-to-A drift: the spread (max minus min) of the A runs' medians (inside `window`), in ms. None with fewer than two A runs that read."""
    medians = [value for value in (median_ms(text, live, window) for text in a_logs) if value is not None]
    return round(max(medians) - min(medians), 2) if len(medians) >= 2 else None


def load_samples(log_text):
    return [float(match.group(2)) for match in LOAD_LINE.finditer(log_text or '')]


def load_median(log_text):
    samples = load_samples(log_text)
    return statistics.median(samples) if samples else None


def load_limit(baseline_text, margin=DEFAULT_LOAD_MARGIN):
    """The load maximum a window's pairs are held to: the MEDIAN [LOAD] of a serving baseline's log plus `margin` (None with no samples). The baseline is a job
    that served eight users on the window image (S0-CTL's load.log), never the idle host: the serving container's own engine, dispatch and API threads raise the
    1-minute load by themselves."""
    median = load_median(baseline_text)
    return None if median is None else round(median + margin, 2)


def load_void(log_text, max_load):
    """(True, reason) when the MEDIAN of a job's '[LOAD] <epoch> <load1>' samples exceeds `max_load` (one spike is no void: the median is the host's condition
    through the job); (False, None) when it does not; (None, reason) with no samples (the load was not logged: the pair cannot be called clean)."""
    samples = load_samples(log_text)
    if not samples:
        return None, 'no [LOAD] samples in the job log'
    median = statistics.median(samples)
    return (True, 'median load average %.2f above %.2f' % (median, max_load)) if median > max_load else (False, None)


def load_gap_void(text_a, text_b, max_gap=DEFAULT_LOAD_GAP):
    """(True, reason) when the two jobs of a pair ran under host loads whose medians differ by more than `max_gap`; (None, reason) when either has no samples."""
    a, b = load_median(text_a), load_median(text_b)
    if a is None or b is None:
        return None, 'no [LOAD] samples in one of the pair\'s logs'
    return (True, 'median load %.2f against %.2f, a gap over %.2f' % (a, b, max_gap)) if abs(a - b) > max_gap else (False, None)


def apply_load(result, a_text, b_text, max_load=None, max_gap=DEFAULT_LOAD_GAP):
    """Mark a compare() result VOID when the pair's loads (logs plus load.log text) break the pre-registered limits; returns the result."""
    if max_load is None:
        return result
    for label, text in (('A', a_text), ('B', b_text)):
        void, reason = load_void(text, max_load)
        if void or void is None:
            result.update(verdict='VOID', reason='%s: %s' % (label, reason))
            return result
    void, reason = load_gap_void(a_text, b_text, max_gap)
    if void or void is None:
        result.update(verdict='VOID', reason=reason)
    return result


def judge(pairs, floor_ms, a_median_ms, go_pairs=GO_PAIRS, no_go_pairs=NO_GO_PAIRS, loss_floor_ms=None):
    """GO / NO-GO / INCONCLUSIVE for one length from its paired readings (compare()'s dicts, VOID pairs included).
    GO: at least `go_pairs` measured pairs (all three of the window's) have a negative delta and the pooled (median) gain exceeds max(floor, 1% of A's median).
    NO-GO: at least `no_go_pairs` measured pairs have a positive delta.
    Fewer than `go_pairs` measured pairs: INCONCLUSIVE unless the positive ones already reach `no_go_pairs` (a VOID pair is never a vote).
    `loss_floor_ms` (the skew read: a loss beyond d_skew) makes a pair positive only when its delta exceeds that floor."""
    measured = [pair for pair in pairs if pair.get('verdict') == 'MEASURED']
    deltas = [pair['delta_ms'] for pair in measured]
    threshold = loss_floor_ms or 0.0
    if sum(1 for value in deltas if value > threshold) >= no_go_pairs:
        return dict(verdict='NO-GO', pooled_gain_ms=round(-statistics.median(deltas), 2), faster=sum(1 for value in deltas if value < 0),
                    slower=sum(1 for value in deltas if value > 0))
    if len(measured) < go_pairs:
        return dict(verdict='INCONCLUSIVE', reason='%d measured pairs of %d, %d needed' % (len(measured), len(pairs), go_pairs))
    gain = -statistics.median(deltas)
    need = max(floor_ms or 0.0, GO_MIN_FRACTION * a_median_ms)
    faster = sum(1 for value in deltas if value < 0)
    slower = sum(1 for value in deltas if value > threshold)
    if faster >= go_pairs and gain > need:
        return dict(verdict='GO', pooled_gain_ms=round(gain, 2), needed_ms=round(need, 2), faster=faster, slower=slower)
    if slower >= no_go_pairs:
        return dict(verdict='NO-GO', pooled_gain_ms=round(gain, 2), needed_ms=round(need, 2), faster=faster, slower=slower)
    return dict(verdict='INCONCLUSIVE', pooled_gain_ms=round(gain, 2), needed_ms=round(need, 2), faster=faster, slower=slower,
                reason='no majority beyond the floor')


def lever_n_cost(b_logs, c_logs, floor_ms, window, live=DEFAULT_LIVE, bucket=DEFAULT_BUCKET, min_matched=DEFAULT_MIN_MATCHED, loads=None, max_load=None,
                 max_gap=DEFAULT_LOAD_GAP):
    """Lever N's steady-state cost on top of W2 at one length: COMBINED (C) against W2 alone (B), pair by pair (T3 against T2, T6 against T5). INSIDE: every
    pair measured, the deltas agree in sign and each sits inside the A-to-A floor; OUTSIDE: measured and not inside; UNMEASURED: a pair is VOID (never
    'inside'): a Lever N arm admits long prompts one step at a time, so its early users can finish before the last is admitted and leave few eight-live rounds."""
    pairs = []
    for index, (b, c) in enumerate(zip(b_logs, c_logs)):
        result = compare(b, c, live, bucket, min_matched, window)
        if loads:
            apply_load(result, b + chr(10) + loads['B'][index], c + chr(10) + loads['C'][index], max_load, max_gap)
        pairs.append(result)
    measured = [pair for pair in pairs if pair['verdict'] == 'MEASURED']
    if not pairs or len(measured) < len(pairs):
        return dict(verdict='UNMEASURED', pairs=pairs, reason='%d of %d pairs measured' % (len(measured), len(pairs)))
    deltas = [pair['delta_ms'] for pair in measured]
    agree = all(value >= 0 for value in deltas) or all(value <= 0 for value in deltas)
    inside = all(abs(value) <= (floor_ms or 0.0) for value in deltas)
    return dict(verdict='INSIDE' if inside and agree else 'OUTSIDE', pairs=pairs, deltas_ms=deltas, sign_agrees=agree, floor_ms=floor_ms)


def read_lengths(a_logs, b_logs, c_logs=(), loads=None, max_load=None, max_gap=DEFAULT_LOAD_GAP, windows=LENGTH_ORDER, live=DEFAULT_LIVE, bucket=DEFAULT_BUCKET,
                 min_matched=DEFAULT_MIN_MATCHED):
    """The T read, per length L: {L: dict(w2=judge(...) of the pairs (A_i, B_i), floor_ms, a_median_ms, pairs, lever_n=lever_n_cost(...) when C logs are given,
    c_against_a=[compare(A_i, C_i)])} for each window, and the overall W2 read under 'overall' (NO-GO when 32k, 128k or the skew reads NO-GO, GO when 32k and 128k
    read GO and the skew does not read NO-GO, otherwise INCONCLUSIVE). `loads` = dict(A=[load.log text], B=[...], C=[...]) in the logs' order, with `max_load`."""
    out = {}
    for name in windows:
        pairs = []
        for index, (a, b) in enumerate(zip(a_logs, b_logs)):
            result = compare(a, b, live, bucket, min_matched, name)
            if loads:
                apply_load(result, a + chr(10) + loads['A'][index], b + chr(10) + loads['B'][index], max_load, max_gap)
            pairs.append(result)
        floor = drift_floor_ms(a_logs, live, name)
        medians = [item['seconds'] for text in a_logs for item in in_window(timed_rounds(text, live)[0], name)]
        a_median = statistics.median(medians) * 1000 if medians else 0.0
        entry = dict(w2=judge(pairs, floor, a_median, loss_floor_ms=floor if name == 'skew' else None), floor_ms=floor, a_median_ms=round(a_median, 2), pairs=pairs)
        if c_logs:
            entry['lever_n'] = lever_n_cost(b_logs, c_logs, floor, name, live, bucket, min_matched, loads, max_load, max_gap)
            entry['c_against_a'] = [compare(a, c, live, bucket, min_matched, name) for a, c in zip(a_logs, c_logs)]
        out[name] = entry
    verdicts = dict((name, out[name]['w2']['verdict']) for name in out)
    kills = [name for name in ('32k', '128k', 'skew') if verdicts.get(name) == 'NO-GO']
    if kills:
        overall = dict(verdict='NO-GO', because=kills)
    elif verdicts.get('32k') == 'GO' and verdicts.get('128k') == 'GO' and verdicts.get('skew') != 'NO-GO':
        overall = dict(verdict='GO')
    else:
        overall = dict(verdict='INCONCLUSIVE', verdicts=verdicts)
    out['overall'] = overall
    return out


def user_hashes(smoke_results):
    """{test: [(content_sha256, reasoning_sha256)]} of the eight-user tests of a smoke result dict (the text identity of an arm)."""
    out = {}
    for name, entry in (smoke_results or {}).items():
        users = entry.get('users') if isinstance(entry, dict) else None
        if isinstance(users, list):
            out[name] = [(user.get('content_sha256'), user.get('reasoning_sha256')) if isinstance(user, dict) else None for user in users]
    return out


def texts_equal(smoke_a, smoke_b):
    """[mismatch] between two smoke result dicts: a test both ran whose users' content or reasoning hashes differ (the texts a lossless lever must keep)."""
    a, b = user_hashes(smoke_a), user_hashes(smoke_b)
    problems = []
    for name in sorted(set(a) & set(b)):
        if len(a[name]) != len(b[name]):
            problems.append('%s: %d users against %d' % (name, len(a[name]), len(b[name])))
            continue
        for index, (left, right) in enumerate(zip(a[name], b[name])):
            if left != right:
                problems.append('%s user %d: the answer differs between the arms' % (name, index))
    return problems


def smoke_of(log_text):
    found = None
    for line in log_text.splitlines():
        marker = line.find('SMOKE_JSON ')
        if marker >= 0:
            try:
                found = json.loads(line[marker + len('SMOKE_JSON '):])
            except ValueError:
                pass
    return found


def read_text(path):
    return Path(path).read_text(encoding='utf-8', errors='replace') if path else ''


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split(chr(10))[0])
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('pair', help='one A/B pair: server logs of the two arms (and optionally their smoke logs for the texts)')
    p.add_argument('a_log')
    p.add_argument('b_log')
    p.add_argument('--a-smoke')
    p.add_argument('--b-smoke')
    p.add_argument('--live', type=int, default=DEFAULT_LIVE)
    p.add_argument('--bucket', type=int, default=DEFAULT_BUCKET)
    p.add_argument('--min-matched', type=int, default=DEFAULT_MIN_MATCHED)
    p.add_argument('--window', choices=sorted(WINDOWS), help='read one timed shape only (its mean-position window)')
    p.add_argument('--max-load', type=float, help="void the pair when either job's MEDIAN [LOAD] is above this (load-limit calibrates it)")
    p.add_argument('--load-gap', type=float, default=DEFAULT_LOAD_GAP, help="void the pair when the two jobs' median loads differ by more than this")
    p.add_argument('--a-load', help='load.log of the A job: the [LOAD] samples its smoke step wrote beside its results')
    p.add_argument('--b-load', help='load.log of the B job')
    f = sub.add_parser('floor', help="the A-to-A drift floor from the A runs' server logs")
    f.add_argument('logs', nargs='+')
    f.add_argument('--live', type=int, default=DEFAULT_LIVE)
    f.add_argument('--window', choices=sorted(WINDOWS))
    j = sub.add_parser('judge', help="the T read per length: the three A/B pairs (and the C runs): GO / NO-GO / INCONCLUSIVE at steady, 32k, 128k and the skew, "
                       "Lever N's steady cost, and the overall W2 read")
    j.add_argument('--a', nargs='+', required=True, help='server logs of the A runs (T1 T4 T7)')
    j.add_argument('--b', nargs='+', required=True, help='server logs of the B runs (T2 T5 T8), in the A order')
    j.add_argument('--c', nargs='*', default=[], help='server logs of the C runs (T3 T6), paired with the first B runs (T2 T5)')
    j.add_argument('--a-load', nargs='*', default=[])
    j.add_argument('--b-load', nargs='*', default=[])
    j.add_argument('--c-load', nargs='*', default=[])
    j.add_argument('--max-load', type=float)
    j.add_argument('--load-gap', type=float, default=DEFAULT_LOAD_GAP)
    j.add_argument('--min-matched', type=int, default=DEFAULT_MIN_MATCHED)
    j.add_argument('--window', action='append', choices=sorted(WINDOWS), help='only these lengths (default: all four)')
    j.add_argument('--live', type=int, default=DEFAULT_LIVE)
    ll = sub.add_parser('load-limit', help="the --max-load of a window: the median [LOAD] of a serving baseline's load.log plus a margin")
    ll.add_argument('load_log')
    ll.add_argument('--margin', type=float, default=DEFAULT_LOAD_MARGIN)
    args = parser.parse_args(argv)
    if args.command == 'floor':
        print(json.dumps(dict(floor_ms=drift_floor_ms([read_text(path) for path in args.logs], args.live, args.window), window=args.window)))
        return 0
    if args.command == 'load-limit':
        text = read_text(args.load_log)
        print(json.dumps(dict(median=load_median(text), samples=len(load_samples(text)), margin=args.margin, max_load=load_limit(text, args.margin))))
        return 0
    if args.command == 'judge':
        a_logs, b_logs, c_logs = [read_text(path) for path in args.a], [read_text(path) for path in args.b], [read_text(path) for path in args.c]
        if len(a_logs) != len(b_logs):
            parser.error('--a and --b must list the same number of logs (the pairs are positional)')
        loads = None
        if args.max_load is not None:
            loads = dict(A=[read_text(path) for path in args.a_load] or [''] * len(a_logs), B=[read_text(path) for path in args.b_load] or [''] * len(b_logs),
                         C=[read_text(path) for path in args.c_load] or [''] * len(c_logs))
            if len(loads['A']) != len(a_logs) or len(loads['B']) != len(b_logs) or len(loads['C']) != len(c_logs):
                parser.error("--a-load, --b-load and --c-load list one load.log a log, in the logs' order")
        print(json.dumps(read_lengths(a_logs, b_logs, c_logs, loads, args.max_load, args.load_gap, tuple(args.window or LENGTH_ORDER), args.live,
                                      DEFAULT_BUCKET, args.min_matched), sort_keys=True))
        return 0
    a_text, b_text = read_text(args.a_log), read_text(args.b_log)
    result = compare(a_text, b_text, args.live, args.bucket, args.min_matched, args.window)
    if args.max_load is not None:
        apply_load(result, a_text + chr(10) + read_text(args.a_load), b_text + chr(10) + read_text(args.b_load), args.max_load, args.load_gap)
    if args.a_smoke and args.b_smoke:
        result['text_mismatches'] = texts_equal(smoke_of(read_text(args.a_smoke)), smoke_of(read_text(args.b_smoke)))
    print(json.dumps(result, sort_keys=True))
    return 1 if result.get('text_mismatches') else 0


if __name__ == '__main__':
    sys.exit(main())
