"""The tp4/octo-2 paired read (stdlib only, Python 3.7 syntax: the gate reads this on the rig host): what an octo lever changed, per round shape, between a control boot and an arm boot.

    python3 scripts/ci/octo2_report.py --control <container log A> --arm <container log B> [--live 8] [--min-gain 0.05]

Both boots run an octo profile (QWEN_FAST_OCTO=live or alternate) over the same tests, the arm being the control plus ONE lever (QWEN_FAST_OCTO_DRAFT, _ATTN_BUNDLE, a glue lever).
For each boot, for the rounds at `--live` live seats:

  * the rounds of each shape (octo, m3), their committed tokens per live seat per round (the TAU proxy: a lever that changes the drafts changes it, one that only changes
    time does not), the step and the gap, and the per-seat committed rate over the cycle (step + gap), exactly as octo_judge.seat_rate reads it;
  * the DRAFT time per planned shape: each `[PACKED-EARLY-DRAFT] ... draft_ms=` line is the early draft that ran at the end of round N for round N + 1, so it is joined to the
    `planned` shape of the next `[OCTO] round=` line (the draft is the drafter's wall time inside the step: the launch, the fence window and the readback);
  * the octo draft's own lines (`[OCTO-DRAFT] round=`): how many rounds it served and the host time to enqueue it.

The verdict compares the arm's OCTO rounds with the control's OCTO rounds (medians of per-round rates, and the aggregate over cycles) and the arm's M3 rounds with the control's M3 rounds
(the CONTROL OF THE CONTROL: a lever that touches only the octo shape leaves the m3 rate where it was, and a drifting m3 rate says the boots are not comparable - host load, CI, a
different prompt mix). It also reads the arm's own octo-against-m3 pairs (octo_judge.pair_verdict), so a lever on an `alternate` boot has its own control inside the boot.
GO needs the octo rate to rise by `--min-gain` (default 5%) in the aggregate AND the median, the m3 rates to agree within MAX_M3_DRIFT, and at least MIN_ROUNDS octo rounds in each boot.
It reads speed, never text: the texts are octo_compare's (every answer equal, user by user).
"""
import argparse
import json
import re
import sys

import octo_markers as markers

MIN_ROUNDS = 16
MAX_M3_DRIFT = 0.05
MIN_GAIN = 0.05
EARLY_PATTERN = re.compile(r'\[PACKED-EARLY-DRAFT\] round=(\d+) path=(\w+) live=(\d+) draft_ms=([0-9.]+)')
GAP_OUTLIER_FACTOR = 4.0
GAP_OUTLIER_FLOOR_MS = 50.0


def median(values):
    ordered = sorted(values)
    if not ordered:
        return None
    middle = len(ordered) // 2
    return ordered[middle] if len(ordered) % 2 else (ordered[middle - 1] + ordered[middle]) / 2.0


def percentile(values, share):
    ordered = sorted(values)
    if not ordered:
        return None
    return ordered[min(len(ordered) - 1, int(round(share * (len(ordered) - 1))))]


def rate(item):
    """Committed tokens per second per live seat over the round's cycle (its step and the gap before it), or None."""
    cycle = (item['step_ms'] + item['gap_ms']) / 1000.0
    return None if cycle <= 0 or not item['live'] else item['committed'] / float(item['live']) / cycle


def timeline(text):
    """[(kind, payload)] in log order: ('round', dict) for every [OCTO] round line and ('draft', dict) for every early-draft line."""
    events = []
    for line in (text or '').splitlines():
        round_match = markers.ROUND_PATTERN.search(line)
        if round_match:
            (number, shape, planned, live, rows, eligible, counted, octo_rounds, m3_rounds, committed, step_ms, gap_ms) = round_match.groups()
            events.append(('round', dict(round=int(number), shape=shape, planned=planned, live=int(live), rows=int(rows), eligible=int(eligible), counted=int(counted),
                                         committed=int(committed), step_ms=float(step_ms), gap_ms=float(gap_ms))))
            continue
        draft_match = EARLY_PATTERN.search(line)
        if draft_match:
            number, path, live, ms = draft_match.groups()
            events.append(('draft', dict(round=int(number), path=path, live=int(live), ms=float(ms))))
    return events


def drafts_by_planned_shape(events, live=None):
    """{planned shape: [early draft ms]}: a draft that follows round N drafts for round N + 1, whose line says which shape the plan was. With `live`, only the drafts of that
    many live seats (a boot's ramp and drain drafts at other counts are another cost)."""
    found = {'octo': [], 'm3': []}
    waiting = []
    for kind, item in events:
        if kind == 'draft':
            waiting.append(item)
        else:
            if waiting and item['planned'] in found:
                found[item['planned']].extend(draft['ms'] for draft in waiting if draft['live'] == item['live'] and (live is None or draft['live'] == live))
            waiting = []
    return found


def clean(rounds, live):
    """The rounds at `live` seats that can be timed: eligible and counted, with a gap read (the first of a boot has none), outliers of the gap left out
    (a prefill, an admission or an idle stretch sat before the round)."""
    chosen = [item for item in rounds if item['live'] == live and item['counted'] and item['gap_ms'] > 0]
    ceiling = GAP_OUTLIER_FACTOR * (median([item['gap_ms'] for item in chosen]) or 0.0) + GAP_OUTLIER_FLOOR_MS
    return [item for item in chosen if item['gap_ms'] <= ceiling]


def shape_summary(rounds, live):
    """dict(rounds, rate_aggregate, rate_median, tokens_per_seat_round, step_ms, gap_ms, cycle_ms) of the clean rounds of one shape, or {'rounds': 0}."""
    chosen = clean(rounds, live)
    rates = [rate(item) for item in chosen]
    rates = [value for value in rates if value is not None]
    if not chosen or not rates:
        return dict(rounds=0)
    cycle = sum(item['step_ms'] + item['gap_ms'] for item in chosen)
    committed = sum(item['committed'] for item in chosen)
    return dict(rounds=len(chosen), rate_aggregate=round(committed / float(live) / (cycle / 1000.0), 3), rate_median=round(median(rates), 3),
                tokens_per_seat_round=round(committed / float(live) / len(chosen), 3), step_ms=round(median([item['step_ms'] for item in chosen]), 2),
                gap_ms=round(median([item['gap_ms'] for item in chosen]), 2), cycle_ms=round(cycle / len(chosen), 2))


def boot_summary(text, live=8):
    """The facts of one boot: per shape the shape_summary and the early-draft times by planned shape; the octo draft's lines; the in-boot pair verdict."""
    import octo_judge

    events = timeline(text)
    rounds = [item for kind, item in events if kind == 'round']
    out = dict(live=live, shapes={}, early_draft_ms={}, octo_draft=None)
    for shape in ('octo', 'm3'):
        out['shapes'][shape] = shape_summary([item for item in rounds if item['shape'] == shape], live)
    for shape, values in drafts_by_planned_shape(events, live).items():
        out['early_draft_ms'][shape] = (dict(n=len(values), median=round(median(values), 2), p75=round(percentile(values, 0.75), 2), mean=round(sum(values) / len(values), 2))
                                        if values else dict(n=0))
    scan = markers.draft_scan(text)
    if scan['rounds'] or scan['engaged_lines']:
        out['octo_draft'] = dict(rounds=len(scan['rounds']), builds=sum(item['built'] for item in scan['rounds']), fallbacks=len(scan['fallbacks']),
                                 enqueue_ms_median=round(median([item['ms'] for item in scan['rounds'] if not item['built']]) or 0.0, 2))
    out['pair_verdict'] = octo_judge.pair_verdict(text, live=live)
    return out


def relative(arm, control):
    return None if not control else round(arm / float(control) - 1.0, 4)


def compare(control_text, arm_text, live=8, min_gain=MIN_GAIN):
    """dict(control, arm, octo_gain_aggregate, octo_gain_median, m3_drift, tau_change, draft_ms_change, go, reason). go is None when a boot has too few octo rounds to judge."""
    control, arm = boot_summary(control_text, live), boot_summary(arm_text, live)
    a, b = control['shapes']['octo'], arm['shapes']['octo']
    result = dict(control=control, arm=arm)
    if a['rounds'] < MIN_ROUNDS or b['rounds'] < MIN_ROUNDS:
        result.update(go=None, reason='%d control and %d arm octo rounds at %d live seats (%d wanted in each)' % (a['rounds'], b['rounds'], live, MIN_ROUNDS))
        return result
    gain_aggregate, gain_median = relative(b['rate_aggregate'], a['rate_aggregate']), relative(b['rate_median'], a['rate_median'])
    drift = None
    ca, cb = control['shapes']['m3'], arm['shapes']['m3']
    if ca['rounds'] >= MIN_ROUNDS and cb['rounds'] >= MIN_ROUNDS:
        drift = relative(cb['rate_aggregate'], ca['rate_aggregate'])
    tau = relative(b['tokens_per_seat_round'], a['tokens_per_seat_round'])
    draft_a, draft_b = control['early_draft_ms'].get('octo', {}), arm['early_draft_ms'].get('octo', {})
    draft_change = (round(draft_b['median'] - draft_a['median'], 2) if draft_a.get('n') and draft_b.get('n') else None)
    comparable = drift is None or abs(drift) <= MAX_M3_DRIFT
    go = bool(gain_aggregate is not None and gain_median is not None and gain_aggregate >= min_gain and gain_median >= min_gain and comparable)
    reason = ('octo rate per seat: arm %.2f vs control %.2f committed tok/s (aggregate %+.1f%%, median round %+.1f%%, bar %+.0f%%); tokens per seat-round %s; octo early draft median %s ms; '
              'm3 rounds drift %s'
              % (b['rate_aggregate'], a['rate_aggregate'], 100 * gain_aggregate, 100 * gain_median, 100 * min_gain,
                 'n/a' if tau is None else '%+.1f%%' % (100 * tau), 'n/a' if draft_change is None else '%+.1f' % draft_change,
                 'n/a (too few m3 rounds)' if drift is None else '%+.1f%%' % (100 * drift)))
    if not comparable:
        reason += ' - the m3 rounds drifted by more than %d%%: the boots are not comparable (host load, a different mix), read neither' % int(100 * MAX_M3_DRIFT)
    result.update(octo_gain_aggregate=gain_aggregate, octo_gain_median=gain_median, m3_drift=drift, tau_change=tau, draft_ms_change=draft_change, go=go if comparable else None,
                  reason=reason)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--control', required=True)
    parser.add_argument('--arm', required=True)
    parser.add_argument('--live', type=int, default=8)
    parser.add_argument('--min-gain', type=float, default=MIN_GAIN)
    options = parser.parse_args(argv)
    try:
        texts = []
        for path in (options.control, options.arm):
            with open(path, encoding='utf-8', errors='replace') as handle:
                texts.append(handle.read())
    except OSError as error:
        print('OCTO2_REPORT unreadable: %s' % error, file=sys.stderr)
        return 2
    result = compare(texts[0], texts[1], live=options.live, min_gain=options.min_gain)
    print('OCTO2_REPORT %s' % json.dumps(result, sort_keys=True))
    if result['go'] is None:
        print('OCTO2_VERDICT UNREAD: %s' % result['reason'])
        return 2
    print('OCTO2_VERDICT %s: %s' % ('GO' if result['go'] else 'NO-GO', result['reason']))
    return 0 if result['go'] else 1


if __name__ == '__main__':
    sys.exit(main())
