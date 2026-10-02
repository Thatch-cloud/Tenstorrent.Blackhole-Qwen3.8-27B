"""The A0 screen's paired report: DSpark v2 (arm S) against DFlash2 (arm D, the control) over the same logged turns.
Python 3.7, stdlib only; it reads counts, never text, and runs on the laptop, the rig host or CI.

    python3 scripts/ci/tf_pair_report.py --results <dir> --out <public dir> [--resamples N] [--seed S]

The results directory (written by a0_run.py, private to the lab store) holds
    meta.jsonl           one line per turn: k, set, cluster, bucket, weight, think_tokens, tool_at   (counts and labels)
    arm-<name>.jsonl     one line per turn: k, status, rounds (tf_pair_walk.ROW_COLUMNS rows)
    validity.json        {V0: PASS|FAIL|NOT_RUN, ...} written by the run's gates (V2 is report-only)
Arm names the screen knows: dflash2-t16 (D), dspark-t16 (S), dspark-t16-w2048 / -w8192 / -w32768, dflash2-forced,
dspark-t8, dflash2-t8.

THE STATISTIC (design A.5). tau per arm: within a set, sum of weight * committed over sum of weight * rounds (the lab's
inverse-probability weights), the sets pooled with equal weight. R = tau_S / tau_D at T16 under the SERVED capped rule; the
uncapped R is reported beside it. The 95% interval is a PAIRED cluster bootstrap: conversations are drawn with replacement
within each set and BOTH arms are recomputed on the same draw (an unpaired draw would be wider and is a tested mutation).

GUARDRAILS (non-inferiority bounds, review P2): G-long, G-p10, G-w8; R_own below 1.00 forces HOLD.
VERDICT: NOT_ESTABLISHED when a validity gate failed or did not run (fix and rerun: never a kill); KILL when R's upper bound is
below 1.05 (unless the censored-kill exception asks for arm E first); GO when R >= 1.10 and every guardrail passes; HOLD otherwise.
The public summary is aggregates only and passes tau_lab_report.assert_public with this module's words.
"""
import argparse
import bisect
import json
import math
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import tau_lab_report as rep  # noqa: E402
import tf_pair_walk as walk  # noqa: E402

ARM_S, ARM_D = 'dspark-t16', 'dflash2-t16'
WINDOW_ARMS = ('dspark-t16-w2048', 'dspark-t16-w8192', 'dspark-t16-w32768')
SETS = ('swe', 'own')
LONG_BUCKETS = ('80k', '80k+')
BANDS = (('0-511', 0, 511), ('512-2047', 512, 2047))
KILL_UPPER = 1.05
GO_POINT = 1.10
LONG_POINT, LONG_LOW = 1.00, 0.90
P10_LOW = 0.95
W8_LOW = 0.95
CENSORED_GAP = 0.05
WINDOW_RETENTION = 0.90
MISS_REMOVED_NEEDED = 0.46
MIN_ROUNDS_P10 = 3
PRODUCTION_WEIGHTS = {'swe': 0.25, 'own': 0.75}
RESAMPLES = 10000
W8_RESAMPLES, W8_DRAWS, W8_K = 2000, 5000, 8
VALIDITY_GATES = ('V0', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6')
REPORT_ONLY = ('V2',)
VERDICT_WORDS = ('KILL', 'GO', 'HOLD', 'NOT_ESTABLISHED', 'PASS', 'FAIL', 'NOT_RUN', 'bounded_window', 'full_window', 'none',
                 'r_upper_below_1_05', 'r_below_1_10', 'g_long_failed', 'g_p10_failed', 'g_w8_failed', 'r_own_below_1',
                 'validity_gate_failed', 'validity_gate_not_run', 'censored_band_gap', 'no_data', 'all_clear')


class ReportError(ValueError):
    pass


# -- loading -------------------------------------------------------------------------------------------------------------

def read_jsonl(path):
    out = []
    with open(path, encoding='utf-8') as handle:
        for line in handle:
            if line.strip():
                out.append(json.loads(line))
    return out


def load(results_dir):
    """(meta {k: dict}, arms {name: {k: [round dicts]}}, validity {gate: status}) from a results directory."""
    meta = dict((entry['k'], entry) for entry in read_jsonl(os.path.join(results_dir, 'meta.jsonl')))
    arms = {}
    for name in sorted(os.listdir(results_dir)):
        if name.startswith('arm-') and name.endswith('.jsonl'):
            turns = {}
            for entry in read_jsonl(os.path.join(results_dir, name)):
                if entry.get('status') == 'ok' and entry['k'] in meta:
                    turns[entry['k']] = [walk.from_row(row, meta[entry['k']]) for row in entry['rounds']]
            arms[name[4:-6]] = turns
    path = os.path.join(results_dir, 'validity.json')
    validity = {}
    if os.path.exists(path):
        with open(path, encoding='utf-8') as handle:
            validity = json.load(handle)
    return meta, arms, validity


def paired_keys(meta, first, second):
    """The turns both arms completed, sorted; a turn only one arm finished leaves both."""
    keys = sorted(set(first) & set(second))
    unknown = [k for k in keys if k not in meta]
    if unknown:
        raise ReportError('a result for a turn with no metadata')
    return keys


# -- weighted tallies -----------------------------------------------------------------------------------------------------

def turn_tally(rounds, key='committed', keep=None):
    chosen = [entry for entry in rounds if keep is None or keep(entry)]
    return (float(sum(entry[key] for entry in chosen)), float(len(chosen)))


def units_by_set(meta, keys, arm_a, arm_b, key='committed', turn_keep=None, round_keep=None):
    """{set: [unit]} with one unit per cluster: [ea, ra, eb, rb], each the weight-multiplied sums over the cluster's turns."""
    cells = {}
    for k in keys:
        info = meta[k]
        if turn_keep is not None and not turn_keep(info):
            continue
        weight = float(info.get('weight') or 1.0)
        ea, ra = turn_tally(arm_a[k], key, round_keep)
        eb, rb = turn_tally(arm_b[k], key, round_keep)
        cell = cells.setdefault((info['set'], info['cluster']), [0.0, 0.0, 0.0, 0.0])
        cell[0] += weight * ea
        cell[1] += weight * ra
        cell[2] += weight * eb
        cell[3] += weight * rb
    out = {}
    for (name, _), cell in sorted(cells.items()):
        out.setdefault(name, []).append(cell)
    return out


def pooled(units, side, set_weights=None):
    """The sets' tau (sum e / sum r over the units) combined with `set_weights` (default equal) over the sets that have rounds.
    side 0 reads the first arm, 1 the second."""
    total, mass = 0.0, 0.0
    for name, members in units.items():
        e = sum(unit[2 * side] for unit in members)
        r = sum(unit[2 * side + 1] for unit in members)
        if r <= 0:
            continue
        weight = (set_weights or {}).get(name, 1.0 if set_weights is None else 0.0)
        total += weight * e / r
        mass += weight
    return total / mass if mass else None


def ratio_statistic(set_weights=None):
    def statistic(sample, rng=None):
        top, bottom = pooled(sample, 0, set_weights), pooled(sample, 1, set_weights)
        if top is None or bottom is None or bottom <= 0:
            return None
        return top / bottom
    return statistic


# -- the paired cluster bootstrap -------------------------------------------------------------------------------------------

def percentile(values, q):
    return rep.quantile(values, q)


def cluster_bootstrap(units, statistic, resamples=RESAMPLES, seed=0, level=0.95, paired=True):
    """The point estimate and percentile interval of statistic(sample) under a cluster bootstrap within each set. `units` holds
    BOTH arms' numbers per cluster, so every resample recomputes both arms on the same draw (`paired=False` is the mutation
    that draws each arm separately: it exists so the test can show the pairing matters). Returns dict(point, low, high, n)."""
    names = [name for name in units if units[name]]
    if not names:
        return dict(point=None, low=None, high=None, n=0)
    point = statistic(units, random.Random(seed))
    rng = random.Random(seed + 1)
    draws = []
    for _ in range(resamples):
        sample = {}
        for name in names:
            pool = units[name]
            sample[name] = [pool[rng.randrange(len(pool))] for _ in range(len(pool))]
        if not paired:
            other = {}
            for name in names:
                pool = units[name]
                other[name] = [pool[rng.randrange(len(pool))] for _ in range(len(pool))]
            top = pooled(sample, 0)
            bottom = pooled(other, 1)
            value = top / bottom if top is not None and bottom else None
        else:
            value = statistic(sample, rng)
        if value is not None and not (isinstance(value, float) and math.isnan(value)):
            draws.append(value)
    if not draws:
        return dict(point=rep.rounded(point, 4), low=None, high=None, n=0)
    low = (1.0 - level) / 2.0
    return dict(point=rep.rounded(point, 4), low=rep.rounded(percentile(draws, low), 4),
                high=rep.rounded(percentile(draws, 1.0 - low), 4), n=len(draws))


# -- the guardrail statistics ------------------------------------------------------------------------------------------------

def turn_units(meta, keys, arm_a, arm_b, minimum=MIN_ROUNDS_P10):
    """{set: [unit]}; a unit is one cluster's list of (tau_a, tau_b, weight, rounds_b) over its turns with at least `minimum`
    counted rounds in BOTH arms (the min-8 rule is biased, W-T1b)."""
    cells = {}
    for k in keys:
        a, b = arm_a[k], arm_b[k]
        if len(a) < minimum or len(b) < minimum:
            continue
        info = meta[k]
        tau_a = sum(entry['committed'] for entry in a) / float(len(a))
        tau_b = sum(entry['committed'] for entry in b) / float(len(b))
        cells.setdefault((info['set'], info['cluster']), []).append((tau_a, tau_b, float(info.get('weight') or 1.0), len(b)))
    out = {}
    for (name, _), cell in sorted(cells.items()):
        out.setdefault(name, []).append(cell)
    return out


def _set_normalised(sample, side):
    """(values, weights): every set's weights scaled to a total of 1, so the sets count equally in the quantile."""
    values, weights = [], []
    for name in sorted(sample):
        entries = [item for cluster in sample[name] for item in cluster]
        total = float(sum(item[2] for item in entries))
        if not total:
            continue
        for item in entries:
            values.append(item[side])
            weights.append(item[2] / total)
    return values, weights


def p10_ratio_statistic(sample, rng=None):
    top = rep.weighted_quantile(*_set_normalised(sample, 0), q=0.10)
    bottom = rep.weighted_quantile(*_set_normalised(sample, 1), q=0.10)
    if top is None or not bottom:
        return None
    return top / bottom


def worst_of_k_ratio(draws=W8_DRAWS, k=W8_K, q=0.10):
    """The ratio of each arm's q-quantile of the minimum tau over `k` concurrent turns. A draw picks a set uniformly, then a
    turn there by duration (its control-arm rounds times its weight); BOTH arms read the SAME drawn turns (pairing)."""
    def statistic(sample, rng=None):
        rng = rng or random.Random(0)
        tables = []
        for name in sorted(sample):
            turns = [item for cluster in sample[name] for item in cluster]
            total = float(sum(item[3] * item[2] for item in turns))
            if not total:
                continue
            running, cumulative = 0.0, []
            for item in turns:
                running += item[3] * item[2] / total
                cumulative.append(running)
            tables.append((cumulative, turns))
        if not tables:
            return None
        lows_a, lows_b = [], []
        for _ in range(draws):
            low_a = low_b = None
            for _ in range(k):
                cumulative, turns = tables[rng.randrange(len(tables))]
                at = min(bisect.bisect_left(cumulative, rng.random()), len(turns) - 1)
                low_a = turns[at][0] if low_a is None or turns[at][0] < low_a else low_a
                low_b = turns[at][1] if low_b is None or turns[at][1] < low_b else low_b
            lows_a.append(low_a)
            lows_b.append(low_b)
        top, bottom = percentile(lows_a, q), percentile(lows_b, q)
        return top / bottom if top is not None and bottom else None
    return statistic


def judged(entry, point_bar, low_bar):
    """PASS when the point and the lower bound clear their bars, FAIL when either misses, NOT_ESTABLISHED without data."""
    if entry.get('point') is None or entry.get('low') is None:
        return 'NOT_ESTABLISHED'
    if entry['point'] >= point_bar and entry['low'] >= low_bar:
        return 'PASS'
    return 'FAIL'


# -- secondary outputs --------------------------------------------------------------------------------------------------------

def r_point(meta, keys, arm_a, arm_b, key='committed', turn_keep=None, round_keep=None, set_weights=None):
    units = units_by_set(meta, keys, arm_a, arm_b, key, turn_keep, round_keep)
    top, bottom = pooled(units, 0, set_weights), pooled(units, 1, set_weights)
    if top is None or not bottom:
        return None
    return rep.rounded(top / bottom, 4)


def per_turn_comparison(meta, keys, arm_a, arm_b):
    """(mean log ratio, share of turns the first arm wins, turns): the turns with a counted round in both arms."""
    logs, wins, count = [], 0, 0
    for k in keys:
        if not arm_a[k] or not arm_b[k]:
            continue
        tau_a = sum(entry['committed'] for entry in arm_a[k]) / float(len(arm_a[k]))
        tau_b = sum(entry['committed'] for entry in arm_b[k]) / float(len(arm_b[k]))
        logs.append(math.log(tau_a / tau_b))
        wins += 1 if tau_a > tau_b else 0
        count += 1
    if not count:
        return None, None, 0
    return rep.rounded(sum(logs) / len(logs), 4), rep.rounded(wins / float(count), 4), count


def acceptance_curve(arm, keys, proposals=15):
    rounds = [entry for k in keys for entry in arm[k]]
    curve = walk.position_acceptance(rounds, proposals)
    return [rep.rounded(a / float(b), 4) if b else None for a, b in curve]


def miss_removed(meta, keys, arm_s, arm_d, proposals=15):
    """1 - h_S / h_D, h the share of the proposal block the drafter did not deliver: 1 - (uncapped tau - 1) / proposals.
    The 5.95 worst-seat bar needs about 46% of drafter misses removed."""
    def shortfall(arm):
        units = units_by_set(meta, keys, arm, arm, 'uncapped')
        value = pooled(units, 0)
        return None if value is None else 1.0 - (value - 1.0) / float(proposals)
    h_s, h_d = shortfall(arm_s), shortfall(arm_d)
    if h_s is None or not h_d:
        return None
    return rep.rounded(1.0 - h_s / h_d, 4)


def first_rounds_r(meta, keys, arm_s, arm_d, width=7):
    """R computed over the first `width` proposal positions only (a T8-like read of a T16 walk): each round's accepted prefix
    is cut at `width` and the commit re-derived against the same served cap."""
    def clipped(arm):
        return dict((k, [dict(entry, committed=min(min(entry['accepted'], width) + 1, entry['cap'])) for entry in arm[k]])
                    for k in keys)
    return r_point(meta, keys, clipped(arm_s), clipped(arm_d))


def pair_name(tag):
    return tag


def build(meta, arms, validity=None, resamples=RESAMPLES, seed=0, w8_resamples=W8_RESAMPLES, w8_draws=W8_DRAWS):
    """The private report and the public summary as (private, public). `arms` {name: {k: [round dicts]}}."""
    validity = dict(validity or {})
    if ARM_S not in arms or ARM_D not in arms:
        raise ReportError('the two core arms are required')
    arm_s, arm_d = arms[ARM_S], arms[ARM_D]
    keys = paired_keys(meta, arm_s, arm_d)
    excluded = len(set(arm_s) ^ set(arm_d))
    if not keys:
        raise ReportError('no turn finished in both core arms')
    ratio = ratio_statistic()
    units = units_by_set(meta, keys, arm_s, arm_d)
    r_main = cluster_bootstrap(units, ratio, resamples, seed)
    r_uncapped = r_point(meta, keys, arm_s, arm_d, 'uncapped')
    # arms
    public_arms = {}
    for name in sorted(arms):
        taus = units_by_set(meta, sorted(arms[name]), arms[name], arms[name])
        public_arms[name] = dict(turns=len(arms[name]), rounds=sum(len(rounds) for rounds in arms[name].values()),
                                 tau=rep.rounded(pooled(taus, 0), 4))
    # guardrails
    long_units = units_by_set(meta, keys, arm_s, arm_d, turn_keep=lambda info: info.get('bucket') in LONG_BUCKETS)
    g_long = cluster_bootstrap(long_units, ratio, resamples, seed + 11)
    g_long['status'] = judged(g_long, LONG_POINT, LONG_LOW)
    pairs = turn_units(meta, keys, arm_s, arm_d)
    g_p10 = cluster_bootstrap(pairs, p10_ratio_statistic, resamples, seed + 12)
    g_p10['status'] = judged(g_p10, 0.0, P10_LOW)
    g_w8 = cluster_bootstrap(pairs, worst_of_k_ratio(w8_draws), w8_resamples, seed + 13)
    g_w8['status'] = judged(g_w8, 0.0, W8_LOW)
    own_units = units_by_set(meta, keys, arm_s, arm_d, turn_keep=lambda info: info['set'] == 'own')
    r_own = cluster_bootstrap(own_units, ratio, resamples, seed + 14)
    # secondary
    by_bucket = dict((label, r_point(meta, keys, arm_s, arm_d, turn_keep=lambda info, label=label: info.get('bucket') == label))
                     for label in sorted(set(meta[k].get('bucket') for k in keys) - set([None])))
    by_set = dict((name, r_point(meta, keys, arm_s, arm_d, turn_keep=lambda info, name=name: info['set'] == name))
                  for name in SETS if any(meta[k]['set'] == name for k in keys))
    by_region = dict((region, r_point(meta, keys, arm_s, arm_d, round_keep=lambda entry, region=region: entry['region'] == region))
                     for region in ('reasoning', 'content', 'tool'))
    by_band = dict((label, r_point(meta, keys, arm_s, arm_d,
                                   round_keep=lambda entry, low=low, high=high: low <= entry['offset'] <= high))
                   for label, low, high in BANDS)
    mean_log, win_share, compared = per_turn_comparison(meta, keys, arm_s, arm_d)
    secondary = dict(r_by_bucket=by_bucket, r_by_set=by_set, r_by_region=by_region, r_by_band=by_band,
                     mean_log_ratio=mean_log, win_share=win_share, turns_compared=compared,
                     production_leaning_r=r_point(meta, keys, arm_s, arm_d, set_weights=PRODUCTION_WEIGHTS),
                     r_first7=first_rounds_r(meta, keys, arm_s, arm_d), miss_removed=miss_removed(meta, keys, arm_s, arm_d),
                     miss_removed_needed=MISS_REMOVED_NEEDED,
                     acceptance_dflash2=acceptance_curve(arm_d, keys), acceptance_dspark=acceptance_curve(arm_s, keys))
    window = window_effect(meta, arms, r_main['point'])
    verdict = decide(r_main, g_long, g_p10, g_w8, r_own, by_band, validity)
    public = dict(verdict=verdict['verdict'], reasons=verdict['reasons'], run_arm_e=verdict['run_arm_e'],
                  window_steer=window['steer'], turns_paired=len(keys), turns_unpaired=excluded,
                  r=dict(r_main, uncapped_point=r_uncapped), g_long=g_long, g_p10=g_p10, g_w8=g_w8, r_own=r_own,
                  secondary=secondary, window=window['public'], arms=public_arms,
                  validity=dict((gate, validity.get(gate, 'NOT_RUN')) for gate in VALIDITY_GATES))
    private = dict(public, per_turn=per_turn_rows(meta, keys, arm_s, arm_d))
    return private, public


def window_effect(meta, arms, r_full):
    """The window arms on the turns they share with both core arms: R at each window and the share of (R - 1) it keeps. The
    retention is judged on the SAME subset's full-context R (not the headline R), so the subset's own mix cancels."""
    out, steer = {}, 'none'
    for name in WINDOW_ARMS:
        if name not in arms:
            continue
        keys = sorted(set(arms[name]) & set(arms[ARM_S]) & set(arms[ARM_D]))
        if not keys:
            continue
        r_window = r_point(meta, keys, arms[name], arms[ARM_D])
        r_full_subset = r_point(meta, keys, arms[ARM_S], arms[ARM_D])
        kept = None
        if r_window is not None and r_full_subset is not None and r_full_subset > 1.0:
            kept = rep.rounded((r_window - 1.0) / (r_full_subset - 1.0), 4)
        out[name] = dict(turns=len(keys), r=r_window, r_full_context=r_full_subset, retention=kept)
    full = out.get('dspark-t16-w8192')
    if full and full['retention'] is not None:
        steer = 'bounded_window' if full['retention'] >= WINDOW_RETENTION else 'full_window'
    return dict(public=out, steer=steer)


def decide(r_main, g_long, g_p10, g_w8, r_own, by_band, validity):
    """(verdict, reasons, run_arm_e): the registered go/kill rule."""
    reasons = []
    for gate in VALIDITY_GATES:
        if gate in REPORT_ONLY:
            continue
        status = validity.get(gate, 'NOT_RUN')
        if status == 'FAIL':
            reasons.append('validity_gate_failed')
        elif status != 'PASS':
            reasons.append('validity_gate_not_run')
    if reasons or r_main['point'] is None:
        return dict(verdict='NOT_ESTABLISHED', reasons=sorted(set(reasons or ['no_data'])), run_arm_e=False)
    band_low, band_high = by_band.get('0-511'), by_band.get('512-2047')
    censored = band_low is not None and band_high is not None and band_high - band_low >= CENSORED_GAP
    if r_main['high'] is not None and r_main['high'] < KILL_UPPER:
        if censored:
            return dict(verdict='HOLD', reasons=['r_upper_below_1_05', 'censored_band_gap'], run_arm_e=True)
        return dict(verdict='KILL', reasons=['r_upper_below_1_05'], run_arm_e=False)
    if r_main['point'] < GO_POINT:
        reasons.append('r_below_1_10')
    for label, entry in (('g_long_failed', g_long), ('g_p10_failed', g_p10), ('g_w8_failed', g_w8)):
        if entry['status'] != 'PASS':
            reasons.append(label)
    if r_own['point'] is not None and r_own['point'] < 1.0:
        reasons.append('r_own_below_1')
    if reasons:
        return dict(verdict='HOLD', reasons=sorted(set(reasons)), run_arm_e=False)
    return dict(verdict='GO', reasons=['all_clear'], run_arm_e=False)


def per_turn_rows(meta, keys, arm_s, arm_d):
    """Private only: per turn, its opaque index and the two taus."""
    rows = []
    for k in keys:
        if arm_s[k] and arm_d[k]:
            rows.append(dict(k=k, tau_s=rep.rounded(sum(e['committed'] for e in arm_s[k]) / float(len(arm_s[k])), 4),
                             tau_d=rep.rounded(sum(e['committed'] for e in arm_d[k]) / float(len(arm_d[k])), 4)))
    return rows


# -- the public face --------------------------------------------------------------------------------------------------------

def assert_public(summary):
    """tau_lab_report.assert_public with this report's verdict words."""
    rep.assert_public(summary, words=frozenset(VERDICT_WORDS))


def write_json(path, value):
    with open(path, 'w', encoding='utf-8', newline='\n') as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write('\n')


def summary_lines(public):
    r = public['r']
    return ['verdict %s' % public['verdict'],
            'R %s [%s, %s] (uncapped %s) over %d paired turns' % (r['point'], r['low'], r['high'], r['uncapped_point'],
                                                                   public['turns_paired']),
            'guardrails: long %s, p10 %s, w8 %s; R_own %s' % (public['g_long']['status'], public['g_p10']['status'],
                                                              public['g_w8']['status'], public['r_own']['point'])]


def main(argv=None, say=print):
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--results', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--resamples', type=int, default=RESAMPLES)
    parser.add_argument('--w8-resamples', type=int, default=W8_RESAMPLES)
    parser.add_argument('--w8-draws', type=int, default=W8_DRAWS)
    parser.add_argument('--seed', type=int, default=0)
    options = parser.parse_args(argv)
    meta, arms, validity = load(options.results)
    private, public = build(meta, arms, validity, options.resamples, options.seed, options.w8_resamples, options.w8_draws)
    assert_public(public)                      # refuse before anything is written beside the public directory
    os.makedirs(options.out, exist_ok=True)
    write_json(os.path.join(options.results, 'report.private.json'), private)
    write_json(os.path.join(options.out, 'a0-summary.json'), public)
    for line in summary_lines(public):
        say(line)
    return 0 if public['verdict'] != 'NOT_ESTABLISHED' else 2


if __name__ == '__main__':
    sys.exit(main())
