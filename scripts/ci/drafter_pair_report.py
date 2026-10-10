"""The paired drafter report: one candidate arm of the tau lab against the control arm, over the SAME turns.

    python3 scripts/ci/drafter_pair_report.py --control <control results dir> --candidate <candidate results dir> \
        --arm b16-bf8 --out <public dir> [--control-summary <tau-lab-summary.json>]

Both directories are c2_tau_lab.py results (private, rig-local: tapes.jsonl and outputs.jsonl come from the lab's own report);
only aggregates leave this module (tau_lab_report.assert_public holds the summary to the report's words). A drafter changes the
proposals and never the verified text, so the two arms answer the same turn ids with the same greedy text, and the number that
matters is the paired ratio of tau (committed tokens per verify round) on those turns.

THE GO RULE (pre-registered; docs/drafter-arms.md). All of:
  pooled    the ratio candidate / control of the equal-weight-per-set pooled tau, thinking-ON arms A1, A2, A4, over the turns both
            arms completed, has a 95% lower bound ABOVE 1.00 (a cluster bootstrap over conversations, resampled within each set,
            both arms moving together, 10,000 resamples)
  long      no regression in the long-context bucket (80k+): the ratio's point estimate is at least 0.98 and its 95% upper bound
            at least 1.00 (a significant or a material drop fails)
  p10       no regression at the per-turn p10 (inverse-probability weighted, turns with at least 8 counted rounds in both arms):
            the same two conditions on the ratio of the arms' p10
  text      every paired turn has the same output token ids in both arms (the exactness contract: a drafter moves speed only; a
            difference is a NO-GO until explained, and the ratios above are computed over the same-text turns only)
  coverage  the paired turns are at least 90% of the control's turns with counted rounds
  arms      the control run's drafter arm is exactly 'control' and the candidate run's is the arm named here
  launch    neither run recorded a launched-container problem (the log did not show the arm's own markers) or an error
  matched   the two runs agree on profile, seed, max_tokens and per-arm data counts; the control and candidate image tags differ
            for b16-bf8 and b32-bf8 (a different drafter image each) and are equal for dedf-bf16 and lookup (the same image, a
            switch only: the bf16 weights, the prompt lookup)
Any FAIL is NO-GO; no FAIL with something not established is NOT_ESTABLISHED; otherwise GO. The control arm is the one arm that
calibrates A3 (c2_tau_lab: a candidate that moves tau would fail it by construction). The A3 verdict (against an old-stack
reference, on 8 turns) is reported beside the verdict as `control_calibration` and is an annotation, not a GO precondition: the
paired ratio is a within-run comparison and does not need the stack to match an older one.

Python 3.7 syntax, stdlib only: it runs on the rig host and in CI.
"""
import argparse
import json
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import tau_lab_report as report  # noqa: E402

CANDIDATE_ARMS = ('b16-bf8', 'b32-bf8', 'dedf-bf16', 'lookup')
# The arms that run on the control's own image (a flag only); every other arm is a different drafter image.
SAME_IMAGE_ARMS = ('dedf-bf16', 'lookup')
MIN_ROUNDS = report.MIN_ROUNDS
RESAMPLES = 10000
LEVEL = 0.95
LONG_POINT_FLOOR = 0.98
MIN_PAIRED_SHARE = 0.90
SUMMARY_NAME = 'drafter-pair-summary.json'


def read_jsonl(path):
    out = []
    if not os.path.isfile(path):
        return out
    with open(path, encoding='utf-8') as handle:
        for line in handle:
            try:
                out.append(json.loads(line))
            except ValueError:
                continue
    return out


def round_dicts(rounds):
    keys = ('kind', 'segment', 'emitted', 'rows', 'cap', 'live')
    return [dict(zip(keys, entry)) for entry in rounds or []]


def read_run(directory):
    """({(arm, id): stat plus its set, cluster and output ids} for the thinking-ON primary arms of one results directory, its
    run.json, the public summary its report kept in report.private.json or None)."""
    tapes = read_jsonl(os.path.join(directory, 'tapes.jsonl'))
    outputs = dict(((entry['arm'], entry['id']), entry.get('output_ids')) for entry in read_jsonl(
        os.path.join(directory, 'outputs.jsonl')))
    info = {}
    path = os.path.join(directory, 'run.json')
    if os.path.isfile(path):
        with open(path, encoding='utf-8') as handle:
            info = json.load(handle)
    public = None
    path = os.path.join(directory, 'report.private.json')
    if os.path.isfile(path):
        with open(path, encoding='utf-8') as handle:
            public = (json.load(handle) or {}).get('public')
    stats = {}
    for tape in tapes:
        if tape.get('arm') not in report.PRIMARY_ARMS:
            continue
        rounds = round_dicts(tape.get('rounds'))
        live = report.LIVE_SEATS if any(entry.get('live') is not None for entry in rounds) else None
        stat = report.analyze_turn(dict(prompt_tokens=tape.get('prompt_tokens'), weight=tape.get('weight')), rounds, live)
        stat.update(set=tape.get('set') or report.SET_OF_ARM[tape['arm']], cluster=tape.get('cluster') or tape.get('id'),
                    output=outputs.get((tape['arm'], tape['id'])))
        stats[(tape['arm'], tape['id'])] = stat
    return stats, info, public


def pair(control, candidate):
    """The turns both arms completed with counted rounds in each, as dicts of both sides' weighted emitted / rounds."""
    items = []
    for key in sorted(set(control) & set(candidate), key=lambda item: (str(item[0]), str(item[1]))):
        base, other = control[key], candidate[key]
        if not base['rounds'] or not other['rounds']:
            continue
        weight = base['weight']
        items.append(dict(key=key, set=base['set'], cluster=(base['set'], base['cluster']), bucket=base['bucket'], weight=weight,
                          c_e=weight * other['emitted'], c_r=weight * other['rounds'],
                          b_e=weight * base['emitted'], b_r=weight * base['rounds'],
                          c_tau=other['tau'] if other['rounds'] >= MIN_ROUNDS else None,
                          b_tau=base['tau'] if base['rounds'] >= MIN_ROUNDS else None,
                          same_text=(base['output'] is not None and base['output'] == other['output'])))
    return items


def pooled(items, side):
    """Equal weight per set of the weighted emitted / rounds, `side` 'c' (candidate) or 'b' (base)."""
    per_set = {}
    for item in items:
        pair_ = per_set.setdefault(item['set'], [0.0, 0.0])
        pair_[0] += item[side + '_e']
        pair_[1] += item[side + '_r']
    values = [pair_[0] / pair_[1] for name, pair_ in sorted(per_set.items()) if pair_[1]]
    return sum(values) / len(values) if values else None


def by_cluster(items):
    """{set: [[items of one cluster]]}."""
    sets = {}
    for item in items:
        sets.setdefault(item['set'], {}).setdefault(item['cluster'], []).append(item)
    return dict((name, [clusters[key] for key in sorted(clusters, key=str)]) for name, clusters in sets.items())


def draw(sets, rng):
    """One cluster-bootstrap resample, within each set: the flat list of its items."""
    out = []
    for name in sorted(sets):
        pool = sets[name]
        for _ in range(len(pool)):
            out.extend(pool[rng.randrange(len(pool))])
    return out


def ratio_of(statistic, items):
    base = statistic(items, 'b')
    other = statistic(items, 'c')
    return (other / base) if base and other is not None else None


def p10_of(items, side):
    usable = [item for item in items if item['c_tau'] is not None and item['b_tau'] is not None]
    taus = [item[side + '_tau'] for item in usable]
    return report.weighted_quantile(taus, [item['weight'] for item in usable], 0.10) if taus else None


def interval(items, statistic, seed, resamples=RESAMPLES):
    """(point, low, high, usable draws) of the paired ratio under a cluster bootstrap."""
    point = ratio_of(statistic, items)
    sets = by_cluster(items)
    rng = random.Random(seed)
    draws = []
    for _ in range(resamples):
        value = ratio_of(statistic, draw(sets, rng))
        if value is not None:
            draws.append(value)
    if not draws or point is None:
        return point, None, None, len(draws)
    low = (1.0 - LEVEL) / 2.0
    return point, report.quantile(draws, low), report.quantile(draws, 1.0 - low), len(draws)


def rounded(value, digits=4):
    return None if value is None else round(value, digits)


def gate_above(low):
    return 'NOT_ESTABLISHED' if low is None else ('PASS' if low > 1.0 else 'FAIL')


def gate_no_regression(point, high):
    if point is None or high is None:
        return 'NOT_ESTABLISHED'
    return 'PASS' if point >= LONG_POINT_FLOOR and high >= 1.0 else 'FAIL'


def build(control, candidate, arm, control_info=None, candidate_info=None, control_summary=None, seed=0, resamples=RESAMPLES):
    """The public summary of the pair: aggregates only."""
    if arm not in CANDIDATE_ARMS:
        raise ValueError('the candidate arm is one of %s' % ', '.join(CANDIDATE_ARMS))
    control_info, candidate_info = control_info or {}, candidate_info or {}
    items = pair(control, candidate)
    controlled = [key for key, stat in control.items() if stat['rounds']]
    paired_text = [item for item in items if item['same_text']]
    # The ratios are over the turns whose text is the same in both arms; a turn that diverged is counted by the text gate only.
    point, low, high, draws = interval(paired_text, lambda rows, side: pooled(rows, side), seed, resamples)
    long_items = [item for item in paired_text if item['bucket'] == report.LONG_BUCKET]
    long_point, long_low, long_high, long_draws = interval(long_items, lambda rows, side: pooled(rows, side), seed + 1, resamples)
    p10_point, p10_low, p10_high, p10_draws = interval(paired_text, p10_of, seed + 2, resamples)
    share = len(items) / float(len(controlled)) if controlled else None
    calibration = ((control_summary or {}).get('calibration') or {}).get('verdict') if control_summary else None
    arms_ok = control_info.get('drafter_arm') == 'control' and candidate_info.get('drafter_arm') == arm
    launch_ok = all(info.get('launched_problems') == [] and 'error' not in info for info in (control_info, candidate_info))
    same_image = control_info.get('image_tag') == candidate_info.get('image_tag')
    matched_ok = (all(control_info.get(key) is not None and control_info.get(key) == candidate_info.get(key)
                      for key in ('profile', 'seed', 'max_tokens', 'data_counts'))
                  and control_info.get('image_tag') is not None
                  and same_image == (arm in SAME_IMAGE_ARMS))
    gates = dict(
        pooled=gate_above(low),
        long=gate_no_regression(long_point, long_high) if len(long_items) >= 3 else 'NOT_ESTABLISHED',
        p10=gate_no_regression(p10_point, p10_high),
        text=('NOT_ESTABLISHED' if not items else 'PASS' if len(paired_text) == len(items) else 'FAIL'),
        coverage=('NOT_ESTABLISHED' if share is None else 'PASS' if share >= MIN_PAIRED_SHARE else 'FAIL'),
        arms=('PASS' if arms_ok else 'FAIL'), launch=('PASS' if launch_ok else 'FAIL'), matched=('PASS' if matched_ok else 'FAIL'))
    verdict = ('NO-GO' if 'FAIL' in gates.values() else 'GO' if all(value == 'PASS' for value in gates.values())
               else 'NOT_ESTABLISHED')
    buckets = {}
    for label in [label for _, label in report.BUCKETS]:
        rows = [item for item in paired_text if item['bucket'] == label]
        if rows:
            buckets[label] = dict(turns=len(rows), ratio=rounded(ratio_of(lambda r, s: pooled(r, s), rows)))
    sets = {}
    for name in report.SET_ORDER:
        rows = [item for item in paired_text if item['set'] == name]
        if rows:
            sets[name] = dict(turns=len(rows), ratio=rounded(ratio_of(lambda r, s: pooled(r, s), rows)),
                              control_tau=rounded(pooled(rows, 'b')), candidate_tau=rounded(pooled(rows, 'c')))
    return dict(
        arm=arm, verdict=verdict, gates=gates,
        pairs=dict(turns=len(items), control_turns=len(controlled), share=rounded(share), same_text=len(paired_text),
                   different_text=len(items) - len(paired_text)),
        scored_turns=len(paired_text),
        pooled=dict(control_tau=rounded(pooled(paired_text, 'b')), candidate_tau=rounded(pooled(paired_text, 'c')), ratio=rounded(point),
                    ci95_low=rounded(low), ci95_high=rounded(high), resamples=draws),
        long_bucket=dict(turns=len(long_items), ratio=rounded(long_point), ci95_low=rounded(long_low), ci95_high=rounded(long_high),
                         resamples=long_draws),
        p10=dict(control=rounded(p10_of(paired_text, 'b')), candidate=rounded(p10_of(paired_text, 'c')), ratio=rounded(p10_point),
                 ci95_low=rounded(p10_low), ci95_high=rounded(p10_high), resamples=p10_draws),
        buckets=buckets, sets=sets, control_calibration=calibration if calibration in ('PASS', 'FAIL', 'NOT_RUN',
                                                                                    'NOT_ESTABLISHED') else None,
        rule=dict(long_point_floor=LONG_POINT_FLOOR, min_paired_share=MIN_PAIRED_SHARE, level=LEVEL, min_rounds=MIN_ROUNDS))


def summary_lines(public):
    pooled_ = public['pooled']
    return ['[DRAFTER-PAIR] arm=%s verdict=%s pooled ratio=%s ci95=%s..%s long=%s p10=%s paired=%d same_text=%d gates=%s' % (
        public['arm'], public['verdict'], pooled_['ratio'], pooled_['ci95_low'], pooled_['ci95_high'],
        public['long_bucket']['ratio'], public['p10']['ratio'], public['pairs']['turns'], public['pairs']['same_text'],
        ','.join('%s:%s' % pair_ for pair_ in sorted(public['gates'].items())))]


def main(argv=None, say=print):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--control', required=True)
    parser.add_argument('--candidate', required=True)
    parser.add_argument('--arm', required=True, choices=CANDIDATE_ARMS)
    parser.add_argument('--out', default=None)
    parser.add_argument('--control-summary', default=None, help='a tau-lab-summary.json to take the control A3 verdict from '
                        'instead of the control directory\'s own report')
    parser.add_argument('--seed', type=int, default=0)
    options = parser.parse_args(argv)
    control, control_info, summary = read_run(options.control)
    candidate, candidate_info, _ = read_run(options.candidate)
    if options.control_summary:
        with open(options.control_summary, encoding='utf-8') as handle:
            summary = json.load(handle)
    public = build(control, candidate, options.arm, control_info, candidate_info, summary, seed=options.seed)
    report.assert_public(public)
    if options.out:
        report.write_json(os.path.join(options.out, SUMMARY_NAME), public)
    for line in summary_lines(public):
        say(line)
    return 0 if public['verdict'] == 'GO' else 1


if __name__ == '__main__':
    sys.exit(main())
