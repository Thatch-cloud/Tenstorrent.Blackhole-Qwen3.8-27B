#!/usr/bin/env python3
"""The Tier 2 comparator: production round time and TTFT before and after a cutover, against the approved thresholds.

The rule (an approved rollback rule; acted on at 72 h, final at 7 days):
  * ROUND TIME, per (context window, live count 5 to 8): with at least MIN_MATCHED (100) load-matched rounds a side, roll back when the
    candidate is worse than the baseline by more than max(3 %, floor_W): floor_W is the window's cross-boot floor from the W1 window's repeated
    timing runs (design item C9). It is an input (--floor-w); the baseline's own day-to-day drift is read as well and widens the threshold when it is larger.
  * TTFT p90 for prompts of at most 32k tokens: with at least MIN_REQUESTS (500) requests a side, roll back when the candidate's p90 is worse by
    more than max(20 %, twice the day-to-day spread).

How each statistic is read (the rule names the thresholds, not the estimator, so these are stated here):
  * A round is a record written by `prod_telemetry.py capture` on the serving host: one decode step timed to the next decode step, with the mean
    position of its users and the host's 1-minute load average (`load1`). Only rounds that are fully packed (one audit line per live user) and not
    beside a prefill step are read, on both sides (w2ln_timing_compare.timed_rounds' rule). The context window is the mean position's window: '32k' is
    28,672 to 40,960 and '128k' is 106,496 to 139,264 (w2ln_timing_compare.WINDOWS), so a loss at 128k is never averaged away by a gain at 32k.
  * LOAD-MATCHED is the approved rule's meaning (w2ln_timing_compare's: the rig runs every CI job, and CI load skews round time). In each stratum a
    round counts only when its load1 is at most the BASELINE side's median load plus LOAD_MARGIN (3), and the stratum is VOID when the two sides' median
    loads (of the rounds that count) differ by more than LOAD_GAP (2). A round with no load1 does not count; a stratum with no load samples on a side is
    VOID: without the load it cannot be called matched, so it can neither roll back nor pass. Within that, rounds are paired by the stratum's live count
    AND the same mean-context bucket (1,024 tokens, w2ln_timing_compare.DEFAULT_BUCKET): per bucket the two sides' medians are compared; the stratum's
    delta is the mean of the bucket deltas weighted by min(nBaseline, nCandidate), and the matched count is the sum of those weights. A stratum under
    MIN_MATCHED is VOID (never a pass, never a rollback).
  * THE DRIFT FLOOR is the baseline's day-to-day movement in the same stratum: each baseline day's context-adjusted shift (its bucket medians minus
    the pooled baseline's, weighted by its rounds), and the floor is the largest shift minus the smallest, over days with at least MIN_DAY_ROUNDS
    rounds, as a fraction of the baseline median. One baseline day gives no drift floor (the 3 % margin and floor_W alone apply and the report says so).
    Threshold = max(3 %, floor_W of the window when given, the drift floor).
  * TTFT is read from per-request records (time, prompt tokens, ttft), not from the log: the log has no arrival time. The prompt length is the TOTAL
    prompt: the gateway's request_record keeps `cached_tokens` as a separate counter, not a subset of `prompt_tokens`, so the export gives a
    `total_prompt_tokens` column (prompt_tokens + cached_tokens), which is read in preference; with only `prompt_tokens` and `cached_tokens` columns
    they are summed. The day-to-day spread is the range of the baseline days' p90 over the pooled baseline p90, over days with at least
    MIN_DAY_REQUESTS requests. With no request file the TTFT half is VOID ("TTFT not read") and the overall verdict cannot be a full PASS.

Verdicts. Per stratum: WORSE (beyond the threshold: ROLLBACK), BETTER, INSIDE, VOID. Overall (two halves: round time, TTFT):
  ROLLBACK       any stratum WORSE (a half that measured a loss is evidence whatever the other half did)
  PASS           none WORSE and every wanted stratum of both halves measured
  PASS-PARTIAL   none WORSE, each half measured at least one stratum, but some stratum was VOID (listed)
  UNMEASURED     none WORSE and one half measured NOTHING (every stratum VOID, or no request file for TTFT): the watch is NOT closed, take no action on it
  INSUFFICIENT   nothing measured in either half
Exit code: 0 PASS or PASS-PARTIAL, 1 ROLLBACK, 2 UNMEASURED, 3 INSUFFICIENT, 64 no cutover time, 65 the capture's image records contradict the cutover.

    prod_telemetry_compare.py --dir <capture dir> --cutover '2026-10-09 12:00:00' --stage 72h --requests ttft.csv [--floor-w 32k=0.04,128k=0.06]

Timestamps are the container's log clock (UTC in the serving container); give --cutover in the same clock. --cutover is REQUIRED: the capture follows the
log with `tail -n 0`, so a capture started or restarted after the cutover boot never saw the boot line, and "the first levern_installed event" would be a
later restart (a hand-back) with post-cutover rounds in the baseline. The capture's `follow` events carry the image id each log came from, and the
comparator checks them against the cutover: an image id on both sides of it refuses the read (exit 65) and the report lists the image ids behind each
side. --cutover-from-first-event keeps the old default for a replay where the capture is known to span the cutover boot. Stdlib only.
"""

import argparse
import csv
import glob
import gzip
import json
import os
import re
import statistics
import sys
from datetime import datetime, timedelta

WINDOWS = {'32k': (28672, 40960), '128k': (106496, 139264)}   # w2ln_timing_compare.WINDOWS, the two windows the rule names
BUCKET = 1024
LIVE_COUNTS = (5, 6, 7, 8)
MIN_MATCHED = 100
MIN_DAY_ROUNDS = 20
MIN_BUCKET_ROUNDS = 5
ROUND_MARGIN = 0.03
LOAD_MARGIN = 3.0
LOAD_GAP = 2.0
MIN_REQUESTS = 500
MIN_DAY_REQUESTS = 50
TTFT_MARGIN = 0.20
SPREAD_MULT = 2.0
MAX_PROMPT = 32768
STAGE_HOURS = {'72h': 72, 'final': 168}
BASELINE_DAYS = 7

TIME = re.compile(r'(\d{4})-(\d{2})-(\d{2})[T ](\d{2}):(\d{2}):(\d{2})(?:\.(\d+))?\s*(Z|[+-]\d{2}(?::?\d{2})?)?$')


def parse_time(text):
    """A naive UTC datetime from 'YYYY-MM-DD HH:MM:SS[.fff][Z|+hh[:mm]]'. A time with no zone is taken as already UTC (the container clock)."""
    match = TIME.match(str(text).strip())
    if not match:
        raise ValueError('unreadable time %r' % (text,))
    year, month, day, hour, minute, second = (int(match.group(i)) for i in range(1, 7))
    micro = int((match.group(7) or '0')[:6].ljust(6, '0'))
    moment = datetime(year, month, day, hour, minute, second, micro)
    zone = match.group(8)
    if zone and zone != 'Z':
        sign = -1 if zone[0] == '-' else 1
        digits = zone[1:].replace(':', '')
        offset = timedelta(hours=int(digits[:2]), minutes=int(digits[2:4] or 0))
        moment -= sign * offset
    return moment


def open_any(path):
    return gzip.open(path, 'rt', encoding='utf-8') if path.endswith('.gz') else open(path, 'r', encoding='utf-8', newline='')


def load_rounds(directories):
    """[dict(at, live, ms, mean_pos)] of the usable rounds in the capture directories (fully packed, not beside a prefill)."""
    rounds, skipped = [], 0
    for directory in directories:
        for path in sorted(glob.glob(os.path.join(directory, 'rounds-*.jsonl*'))):
            with open_any(path) as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    record = json.loads(line)
                    if not record.get('packed') or record.get('after_prefill') or record.get('mean_pos') is None:
                        skipped += 1
                        continue
                    rounds.append(dict(at=parse_time('%s %s' % (record['day'], record['t'])), live=int(record['live']), ms=float(record['ms']),
                                       mean_pos=float(record['mean_pos']), load=float(record['load1']) if record.get('load1') is not None else None))
    return rounds, skipped


def percentile(values, fraction):
    """Linear-interpolation percentile (numpy's default) of a non-empty list."""
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * fraction
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def split_at(items, cutover, until=None, since=None):
    base = [item for item in items if item['at'] < cutover and (since is None or item['at'] >= since)]
    cand = [item for item in items if item['at'] >= cutover and (until is None or item['at'] < until)]
    return base, cand


def by_bucket(rounds):
    out = {}
    for item in rounds:
        out.setdefault(int(item['mean_pos'] // BUCKET), []).append(item['ms'])
    return out


def day_floor(base_rounds, pooled_medians, min_day_rounds=MIN_DAY_ROUNDS):
    """(floor_ms, days): the largest minus the smallest per-day context-adjusted shift of the baseline against its pooled bucket medians."""
    days = {}
    for item in base_rounds:
        days.setdefault(item['at'].date(), []).append(item)
    shifts = {}
    for day, items in days.items():
        if len(items) < min_day_rounds:
            continue
        weight, total = 0, 0.0
        for key, values in by_bucket(items).items():
            if len(values) >= MIN_BUCKET_ROUNDS and key in pooled_medians:
                weight += len(values)
                total += len(values) * (statistics.median(values) - pooled_medians[key])
        if weight:
            shifts[str(day)] = total / weight
    if len(shifts) < 2:
        return None, shifts
    return max(shifts.values()) - min(shifts.values()), shifts


def load_match(base, cand, load_margin=LOAD_MARGIN, load_gap=LOAD_GAP):
    """(base, cand, info, void_reason): the rounds that count under the load rule (module docstring) and the loads they were read at."""
    base_loads = [item['load'] for item in base if item.get('load') is not None]
    cand_loads = [item['load'] for item in cand if item.get('load') is not None]
    if not base_loads or not cand_loads:
        return [], [], dict(load_limit=None), 'no host load samples on the %s side: the stratum cannot be called load-matched' % (
            'baseline' if not base_loads else 'candidate')
    limit = statistics.median(base_loads) + load_margin
    base = [item for item in base if item.get('load') is not None and item['load'] <= limit]
    cand = [item for item in cand if item.get('load') is not None and item['load'] <= limit]
    info = dict(load_limit=round(limit, 2), rounds_over_load_limit=dict(baseline=len(base_loads) - len(base), candidate=len(cand_loads) - len(cand)))
    if not base or not cand:
        return base, cand, info, 'no round at or under the load limit %.2f on one side' % limit
    medians = (statistics.median([item['load'] for item in base]), statistics.median([item['load'] for item in cand]))
    info['load_median'] = dict(baseline=round(medians[0], 2), candidate=round(medians[1], 2))
    if abs(medians[0] - medians[1]) > load_gap:
        return base, cand, info, 'the two sides ran under host loads whose medians differ by %.2f, over %.2f' % (abs(medians[0] - medians[1]), load_gap)
    return base, cand, info, None


def read_stratum(base, cand, window, live, min_matched=MIN_MATCHED, margin=ROUND_MARGIN, floor_mult=1.0, min_day_rounds=MIN_DAY_ROUNDS,
                 load_margin=LOAD_MARGIN, load_gap=LOAD_GAP, floor_w=None):
    low, high = WINDOWS[window]
    base = [item for item in base if item['live'] == live and low <= item['mean_pos'] < high]
    cand = [item for item in cand if item['live'] == live and low <= item['mean_pos'] < high]
    seen_base, seen_cand = len(base), len(cand)
    base, cand, load_info, load_void = load_match(base, cand, load_margin, load_gap)
    a, b = by_bucket(base), by_bucket(cand)
    shared = sorted(set(a) & set(b))
    matched = sum(min(len(a[key]), len(b[key])) for key in shared)
    result = dict(window=window, live=live, baseline_rounds=seen_base, candidate_rounds=seen_cand, matched=matched, min_matched=min_matched, **load_info)
    if load_void:
        result.update(verdict='VOID', reason=load_void)
        return result
    if matched < min_matched:
        result.update(verdict='VOID', reason='%d matched rounds, under the pre-registered %d' % (matched, min_matched))
        return result
    delta = sum((statistics.median(b[key]) - statistics.median(a[key])) * min(len(a[key]), len(b[key])) for key in shared) / float(matched)
    matched_base = [value for key in shared for value in a[key]]
    matched_cand = [value for key in shared for value in b[key]]
    base_median = statistics.median(matched_base)
    pooled = {key: statistics.median(a[key]) for key in shared}
    floor_ms, shifts = day_floor([item for item in base if int(item['mean_pos'] // BUCKET) in pooled], pooled, min_day_rounds)
    floor_rel = (floor_ms / base_median) if floor_ms is not None and base_median else 0.0
    floor_w_rel = (floor_w or {}).get(window)
    threshold = max(margin, floor_mult * floor_rel, floor_w_rel or 0.0)
    relative = delta / base_median
    verdict = 'WORSE' if relative > threshold else ('BETTER' if relative < -threshold else 'INSIDE')
    result.update(verdict=verdict, baseline_median_ms=round(base_median, 2), candidate_median_ms=round(statistics.median(matched_cand), 2),
                  delta_ms=round(delta, 2), delta_pct=round(relative * 100, 2), threshold_pct=round(threshold * 100, 2),
                  floor_pct=round(floor_rel * 100, 2), floor_w_pct=round(floor_w_rel * 100, 2) if floor_w_rel is not None else None,
                  floor_w_basis='given' if floor_w_rel is not None else 'not given for this window: the 3 % margin and the drift floor apply', floor_basis=('%d baseline days' % len(shifts)) if floor_ms is not None else
                  'under two baseline days: the margin alone applies', buckets=len(shared))
    return result


def read_rounds(base, cand, windows, live_counts, **options):
    return [read_stratum(base, cand, window, live, **options) for window in windows for live in live_counts]


def load_requests(paths):
    """[dict(at, prompt, ttft_s)] from CSV or JSONL files (columns: a time, prompt_tokens, ttft_ms or ttft_s; an optional status column drops errors)."""
    out = []
    for path in paths:
        with open_any(path) as handle:
            text = handle.read()
        rows = []
        stripped = text.lstrip()
        if stripped.startswith('{'):
            rows = [json.loads(line) for line in text.splitlines() if line.strip()]
        else:
            rows = list(csv.DictReader(text.splitlines()))
        for row in rows:
            when = None
            for key in ('occurred_at', 'at', 'ts', 'time'):
                if row.get(key) not in (None, ''):
                    when = parse_time(row[key])
                    break
            if when is None and row.get('day') and row.get('t'):
                when = parse_time('%s %s' % (row['day'], row['t']))
            ttft = row.get('ttft_s')
            if ttft in (None, ''):
                ttft = row.get('ttft_ms')
                ttft = None if ttft in (None, '') else float(ttft) / 1000.0
            else:
                ttft = float(ttft)
            prompt = row.get('total_prompt_tokens')
            if prompt in (None, ''):
                prompt = row.get('prompt_tokens', row.get('prompt'))
                if prompt not in (None, '') and row.get('cached_tokens') not in (None, ''):
                    prompt = int(prompt) + int(row['cached_tokens'])
            status = row.get('status')
            if when is None or ttft is None or prompt in (None, '') or (status not in (None, '') and int(status) >= 400):
                continue
            out.append(dict(at=when, prompt=int(prompt), ttft_s=ttft))
    return out


def read_ttft(base, cand, max_prompt=MAX_PROMPT, min_requests=MIN_REQUESTS, margin=TTFT_MARGIN, spread_mult=SPREAD_MULT,
              min_day_requests=MIN_DAY_REQUESTS):
    base = [item for item in base if item['prompt'] <= max_prompt]
    cand = [item for item in cand if item['prompt'] <= max_prompt]
    result = dict(window='prompt<=%d' % max_prompt, baseline_requests=len(base), candidate_requests=len(cand), min_requests=min_requests)
    if len(base) < min_requests or len(cand) < min_requests:
        result.update(verdict='VOID', reason='%d baseline and %d candidate requests, under the pre-registered %d a side' % (len(base), len(cand), min_requests))
        return result
    base_p90 = percentile([item['ttft_s'] for item in base], 0.9)
    cand_p90 = percentile([item['ttft_s'] for item in cand], 0.9)
    days = {}
    for item in base:
        days.setdefault(item['at'].date(), []).append(item['ttft_s'])
    day_p90 = {str(day): percentile(values, 0.9) for day, values in days.items() if len(values) >= min_day_requests}
    spread = ((max(day_p90.values()) - min(day_p90.values())) / base_p90) if len(day_p90) >= 2 and base_p90 else 0.0
    threshold = max(margin, spread_mult * spread)
    relative = cand_p90 / base_p90 - 1.0 if base_p90 else 0.0
    verdict = 'WORSE' if relative > threshold else ('BETTER' if relative < -threshold else 'INSIDE')
    result.update(verdict=verdict, baseline_p90_s=round(base_p90, 3), candidate_p90_s=round(cand_p90, 3), delta_pct=round(relative * 100, 2),
                  threshold_pct=round(threshold * 100, 2), day_spread_pct=round(spread * 100, 2),
                  spread_basis=('%d baseline days' % len(day_p90)) if len(day_p90) >= 2 else 'under two baseline days: the margin alone applies')
    return result


def overall(strata):
    verdicts = [item['verdict'] for item in strata]
    if 'WORSE' in verdicts:
        return 'ROLLBACK'
    measured = [v for v in verdicts if v != 'VOID']
    if not measured:
        return 'INSUFFICIENT'
    return 'PASS' if len(measured) == len(verdicts) else 'PASS-PARTIAL'


IMAGE = re.compile(r'\bimage=(\S+)')


def image_timeline(directories):
    """[(time, image id)] of the capture's `follow` events, oldest first: the image each engine log came from (the wrapper prints the marker on every attach)."""
    found = []
    for directory in directories:
        for path in sorted(glob.glob(os.path.join(directory, 'events-*.jsonl*'))):
            with open_any(path) as handle:
                for line in handle:
                    if 'follow' not in line:
                        continue
                    record = json.loads(line)
                    match = IMAGE.search(record.get('detail') or '')
                    if record.get('code') == 'follow' and match and record.get('t') and match.group(1) != 'unknown':
                        found.append((parse_time('%s %s' % (record['day'], record['t'])), match.group(1)))
    return sorted(found)


def sides_images(timeline, base, cand):
    """The image ids behind each side's rounds (the last follow event at or before each round), and the ids on BOTH sides. A round before the first
    follow event has no known image and is counted as `unknown`."""
    import bisect
    times = [at for at, _ in timeline]
    out = dict(baseline=set(), candidate=set(), unknown=dict(baseline=0, candidate=0))
    for name, items in (('baseline', base), ('candidate', cand)):
        for item in items:
            index = bisect.bisect_right(times, item['at']) - 1
            if index < 0:
                out['unknown'][name] += 1
            else:
                out[name].add(timeline[index][1])
    return dict(baseline_images=sorted(out['baseline']), candidate_images=sorted(out['candidate']),
                images_on_both_sides=sorted(out['baseline'] & out['candidate']), rounds_with_unknown_image=out['unknown'])


def judge(rounds, requests, cutover, stage, windows=('32k', '128k'), live_counts=LIVE_COUNTS, baseline_days=BASELINE_DAYS, timeline=None, **options):
    """The full read. `requests` None = no request file: the TTFT half is VOID ('TTFT not read'), so the overall verdict is at best PASS-PARTIAL."""
    until = cutover + timedelta(hours=STAGE_HOURS[stage])
    since = cutover - timedelta(days=baseline_days)
    base, cand = split_at(rounds, cutover, until, since)
    strata = read_rounds(base, cand, windows, live_counts, **{key: value for key, value in options.items() if key in
                                                              ('min_matched', 'margin', 'floor_mult', 'min_day_rounds', 'load_margin', 'load_gap',
                                                               'floor_w')}) if rounds else []
    report = dict(stage=stage, cutover=str(cutover), candidate_until=str(until), baseline_since=str(since), round_strata=strata,
                  rounds_verdict=overall(strata) if strata else 'INSUFFICIENT')
    if timeline:
        report['images'] = sides_images(timeline, base, cand)
        if report['images']['images_on_both_sides']:
            report['verdict'] = 'CUTOVER-INCONSISTENT'
            report['reason'] = ('the image id(s) %s produced rounds on BOTH sides of the cutover %s: the cutover time is wrong or the capture mixed images; '
                                'the comparison is refused' % (', '.join(report['images']['images_on_both_sides']), cutover))
            return report
    else:
        report['images'] = dict(note='no follow events with an image id in the capture: the cutover time could not be checked against the images')
    ttft = None
    if requests is not None:
        base_r, cand_r = split_at(requests, cutover, until, since)
        ttft = read_ttft(base_r, cand_r)
        report['ttft'] = ttft
    else:
        ttft = report['ttft'] = dict(window='prompt<=%d' % MAX_PROMPT, verdict='VOID', reason='TTFT not read: no request file was given (--requests)')
    verdicts = [report['rounds_verdict']] + ([{'WORSE': 'ROLLBACK', 'VOID': 'INSUFFICIENT'}.get(ttft['verdict'], 'PASS')] if ttft else [])
    if 'ROLLBACK' in verdicts:
        report['verdict'] = 'ROLLBACK'
    elif all(v == 'INSUFFICIENT' for v in verdicts):
        report['verdict'] = 'INSUFFICIENT'
    elif 'INSUFFICIENT' in verdicts:
        report['verdict'] = 'UNMEASURED'      # one half measured nothing: it reads as a pass only if it is not named one
        report['unmeasured'] = [name for name, v in zip(('round time', 'ttft'), verdicts) if v == 'INSUFFICIENT']
    elif 'PASS-PARTIAL' in verdicts:
        report['verdict'] = 'PASS-PARTIAL'
    else:
        report['verdict'] = 'PASS'
    return report


def first_event(directories, code):
    """The earliest `code` event of the capture directories as a naive UTC datetime, or None."""
    found = []
    for directory in directories:
        for path in sorted(glob.glob(os.path.join(directory, 'events-*.jsonl*'))):
            with open_any(path) as handle:
                for line in handle:
                    if code in line:
                        record = json.loads(line)
                        if record.get('code') == code and record.get('t'):
                            found.append(parse_time('%s %s' % (record['day'], record['t'])))
    return min(found) if found else None


def parse_floor_w(text):
    """'32k=0.04,128k=0.06' -> {'32k': 0.04, '128k': 0.06} (fractions)."""
    out = {}
    for part in (text or '').split(','):
        if part.strip():
            name, _, value = part.partition('=')
            if name.strip() not in WINDOWS:
                raise ValueError('unknown window %r in --floor-w' % name)
            out[name.strip()] = float(value)
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--dir', action='append', default=[], help='a capture directory (rounds-DAY.jsonl); repeat for several')
    parser.add_argument('--requests', action='append', default=[], help='per-request CSV or JSONL: a time, prompt_tokens, ttft_ms or ttft_s')
    parser.add_argument('--cutover', help="the cutover time, 'YYYY-MM-DD HH:MM:SS' on the container clock (UTC); required")
    parser.add_argument('--cutover-from-first-event', action='store_true',
                        help='a replay only: the cutover is the first levern_installed event in --dir (wrong when the capture started after the cutover boot)')
    parser.add_argument('--floor-w', default='', help="the cross-boot floor per window as fractions, '32k=0.04,128k=0.06' (design item C9); default: none given")
    parser.add_argument('--load-margin', type=float, default=LOAD_MARGIN, help='a round counts when its load is at most the baseline median plus this')
    parser.add_argument('--load-gap', type=float, default=LOAD_GAP, help="a stratum is void when the two sides' median loads differ by more than this")
    parser.add_argument('--stage', choices=sorted(STAGE_HOURS), default='72h', help='72h or final (7 days) of candidate data')
    parser.add_argument('--baseline-days', type=int, default=BASELINE_DAYS)
    parser.add_argument('--windows', default='32k,128k')
    parser.add_argument('--live', default='5,6,7,8')
    parser.add_argument('--min-matched', type=int, default=MIN_MATCHED)
    parser.add_argument('--margin', type=float, default=ROUND_MARGIN)
    parser.add_argument('--floor-mult', type=float, default=1.0)
    args = parser.parse_args(argv)
    rounds, skipped = load_rounds(args.dir)
    requests = load_requests(args.requests) if args.requests else None
    if args.cutover:
        cutover = parse_time(args.cutover)
    elif args.cutover_from_first_event:
        cutover = first_event(args.dir, 'levern_installed')
    else:
        print('--cutover is required (the time the cutover image started serving, on the container clock): the capture may have started after the cutover boot, '
              'and a guessed time would put post-cutover rounds in the baseline', file=sys.stderr)
        return 64
    if cutover is None:
        print('no levern_installed event in the capture directory', file=sys.stderr)
        return 64
    report = judge(rounds, requests, cutover, args.stage, tuple(args.windows.split(',')), tuple(int(v) for v in args.live.split(',')),
                   args.baseline_days, min_matched=args.min_matched, margin=args.margin, floor_mult=args.floor_mult, load_margin=args.load_margin,
                   load_gap=args.load_gap, floor_w=parse_floor_w(args.floor_w), timeline=image_timeline(args.dir))
    report['rounds_read'] = len(rounds)
    report['rounds_skipped_unusable'] = skipped
    print(json.dumps(report, indent=1, sort_keys=True))
    return {'PASS': 0, 'PASS-PARTIAL': 0, 'ROLLBACK': 1, 'UNMEASURED': 2, 'INSUFFICIENT': 3, 'CUTOVER-INCONSISTENT': 65}[report['verdict']]


if __name__ == '__main__':
    sys.exit(main())
