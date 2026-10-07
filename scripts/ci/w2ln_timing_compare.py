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
measured: never GO, never NO-GO). The A-to-A drift floor is the spread of the A runs' medians; a pair also goes VOID when its job logged a load average above
the pre-registered maximum (the rig hosts CI and builds). Texts are judged separately, by comparing the smoke JSON hashes (`texts_equal`).

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
DEFAULT_MIN_MATCHED = 200
DEFAULT_LIVE = 8
GO_PAIRS = 3
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


def buckets(rounds, bucket=DEFAULT_BUCKET):
    out = {}
    for item in rounds:
        out.setdefault(int(item['mean_position'] // bucket), []).append(item['seconds'])
    return out


def compare(log_a, log_b, live=DEFAULT_LIVE, bucket=DEFAULT_BUCKET, min_matched=DEFAULT_MIN_MATCHED):
    """The paired reading of arm A (the control) against arm B: dict(verdict 'MEASURED' | 'VOID', matched, delta_ms (B minus A: negative is faster),
    a_median_ms, b_median_ms, per-bucket rows). VOID carries its reason."""
    rounds_a, dropped_a = timed_rounds(log_a, live)
    rounds_b, dropped_b = timed_rounds(log_b, live)
    a, b = buckets(rounds_a, bucket), buckets(rounds_b, bucket)
    rows, weight, total = [], 0, 0.0
    for key in sorted(set(a) & set(b)):
        matched = min(len(a[key]), len(b[key]))
        delta = statistics.median(b[key]) - statistics.median(a[key])
        rows.append(dict(bucket_start=key * bucket, matched=matched, a_median_ms=round(statistics.median(a[key]) * 1000, 2),
                         b_median_ms=round(statistics.median(b[key]) * 1000, 2), delta_ms=round(delta * 1000, 2)))
        weight += matched
        total += delta * matched
    result = dict(live=live, bucket=bucket, rounds_a=len(rounds_a), rounds_b=len(rounds_b), dropped_for_prefill=dict(a=dropped_a, b=dropped_b),
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


def median_ms(log_text, live=DEFAULT_LIVE):
    rounds, _ = timed_rounds(log_text, live)
    return statistics.median([item['seconds'] for item in rounds]) * 1000 if rounds else None


def drift_floor_ms(a_logs, live=DEFAULT_LIVE):
    """The A-to-A drift: the spread (max minus min) of the A runs' medians, in ms. None with fewer than two A runs that read."""
    medians = [value for value in (median_ms(text, live) for text in a_logs) if value is not None]
    return round(max(medians) - min(medians), 2) if len(medians) >= 2 else None


def load_void(log_text, max_load):
    """(True, reason) when a '[LOAD] <epoch> <load1>' sample of the job's own log exceeds `max_load`; (False, None) when none does; (None, reason) with no samples
    (the load was not logged: the pair cannot be called clean)."""
    samples = [float(match.group(2)) for match in LOAD_LINE.finditer(log_text)]
    if not samples:
        return None, 'no [LOAD] samples in the job log'
    worst = max(samples)
    return (True, 'load average %.2f above %.2f' % (worst, max_load)) if worst > max_load else (False, None)


def judge(pairs, floor_ms, a_median_ms, go_pairs=GO_PAIRS):
    """GO / NO-GO / INCONCLUSIVE for one length from its paired readings (compare()'s dicts, VOID pairs included).
    GO: at least `go_pairs` measured pairs have a negative delta and the pooled (median) gain exceeds max(floor, 1% of A's median).
    NO-GO: at least `go_pairs` measured pairs have a positive delta.
    Fewer than `go_pairs` measured pairs: INCONCLUSIVE (a VOID pair is never a vote)."""
    measured = [pair for pair in pairs if pair.get('verdict') == 'MEASURED']
    if len(measured) < go_pairs:
        return dict(verdict='INCONCLUSIVE', reason='%d measured pairs of %d, %d needed' % (len(measured), len(pairs), go_pairs))
    deltas = [pair['delta_ms'] for pair in measured]
    gain = -statistics.median(deltas)
    need = max(floor_ms or 0.0, GO_MIN_FRACTION * a_median_ms)
    faster = sum(1 for value in deltas if value < 0)
    slower = sum(1 for value in deltas if value > 0)
    if faster >= go_pairs and gain > need:
        return dict(verdict='GO', pooled_gain_ms=round(gain, 2), needed_ms=round(need, 2), faster=faster, slower=slower)
    if slower >= go_pairs:
        return dict(verdict='NO-GO', pooled_gain_ms=round(gain, 2), needed_ms=round(need, 2), faster=faster, slower=slower)
    return dict(verdict='INCONCLUSIVE', pooled_gain_ms=round(gain, 2), needed_ms=round(need, 2), faster=faster, slower=slower,
                reason='no majority beyond the floor')


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


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('pair', help='one A/B pair: server logs of the two arms (and optionally their smoke logs for the texts)')
    p.add_argument('a_log')
    p.add_argument('b_log')
    p.add_argument('--a-smoke')
    p.add_argument('--b-smoke')
    p.add_argument('--live', type=int, default=DEFAULT_LIVE)
    p.add_argument('--bucket', type=int, default=DEFAULT_BUCKET)
    p.add_argument('--min-matched', type=int, default=DEFAULT_MIN_MATCHED)
    p.add_argument('--max-load', type=float, help='void the pair when either job logged a [LOAD] sample above this')
    p.add_argument('--a-load', help='load.log of the A job: the [LOAD] samples its smoke step wrote beside its results')
    p.add_argument('--b-load', help='load.log of the B job')
    f = sub.add_parser('floor', help='the A-to-A drift floor from the A runs\' server logs')
    f.add_argument('logs', nargs='+')
    f.add_argument('--live', type=int, default=DEFAULT_LIVE)
    args = parser.parse_args(argv)
    if args.command == 'floor':
        print(json.dumps(dict(floor_ms=drift_floor_ms([Path(path).read_text(encoding='utf-8', errors='replace') for path in args.logs], args.live))))
        return 0
    a_text = Path(args.a_log).read_text(encoding='utf-8', errors='replace')
    b_text = Path(args.b_log).read_text(encoding='utf-8', errors='replace')
    result = compare(a_text, b_text, args.live, args.bucket, args.min_matched)
    if args.max_load is not None:
        loads = dict(A=Path(args.a_load).read_text(encoding='utf-8', errors='replace') if args.a_load else '',
                     B=Path(args.b_load).read_text(encoding='utf-8', errors='replace') if args.b_load else '')
        for label, text in (('A', a_text + chr(10) + loads['A']), ('B', b_text + chr(10) + loads['B'])):
            void, reason = load_void(text, args.max_load)
            if void or void is None:
                result.update(verdict='VOID', reason='%s: %s' % (label, reason))
    if args.a_smoke and args.b_smoke:
        result['text_mismatches'] = texts_equal(smoke_of(Path(args.a_smoke).read_text(encoding='utf-8', errors='replace')),
                                                smoke_of(Path(args.b_smoke).read_text(encoding='utf-8', errors='replace')))
    print(json.dumps(result, sort_keys=True))
    return 1 if result.get('text_mismatches') else 0


if __name__ == '__main__':
    sys.exit(main())
