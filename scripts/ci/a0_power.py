"""The A0 screen's power simulation (Python 3.7, stdlib): how often would the registered rule say GO or KILL if the true R were x, on
THE screen's own clusters, weights and round counts? Aggregates only; deterministic for a seed. Run it and commit its output BEFORE
the window, with the statistics plan (docs/a0-dspark-screen.md).

    python3 scripts/ci/a0_power.py --meta <bundle>/meta.jsonl --out power.json [--trials 200] [--seed 0]

The model (every number a parameter, none a measurement of the screen):
  * a cluster has a difficulty: tau_D(cluster) = tau_D * exp(N(0, sd_difficulty)), shared by both arms (that sharing is what makes
    the paired interval narrow);
  * DSpark's true effect in a cluster is R * exp(N(0, sd_effect)) (the effect varies by conversation);
  * a turn has `answer_tokens / tau_D` rounds; a round commits with sd `round_sd` around its arm's tau, so a cluster's measured tau
    carries round noise that shrinks with its rounds (committed counts are rounded and floored at 1);
  * the statistic is tf_pair_report's: weighted equal-set pooled tau, paired cluster bootstrap, R = tau_S / tau_D.
Output per true R: the share of trials with R_point >= 1.10 (the GO bar), with R_upper < 1.05 (the KILL bar), with both the GO point
bar and the lower bound above 1.00, and the mean interval half-width.
"""
import argparse
import json
import math
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import tf_pair_report as report  # noqa: E402

DEFAULT_R = (1.00, 1.05, 1.10, 1.15, 1.25)
TAU_D = 4.5
SD_DIFFICULTY = 0.30
SD_EFFECT = 0.08
ROUND_SD = 3.0
MAX_ROUNDS = 400            # a turn's round count is capped for speed; the cap only trims noise that is already small


def clusters_of(meta_rows):
    """{(set, cluster): [(weight, rounds at TAU_D)]} from the bundle's metadata."""
    out = {}
    for row in meta_rows:
        rounds = max(1, min(MAX_ROUNDS, int(round(row['answer_tokens'] / TAU_D))))
        out.setdefault((row['set'], row['cluster']), []).append((float(row.get('weight') or 1.0), rounds))
    return out


def draw_trial(clusters, true_r, rng, tau_d=TAU_D, sd_difficulty=SD_DIFFICULTY, sd_effect=SD_EFFECT, round_sd=ROUND_SD):
    """{set: [unit]} in tf_pair_report's unit layout [e_S, r_S, e_D, r_D] (one per cluster), weight-multiplied."""
    units = {}
    for (name, _), turns in sorted(clusters.items()):
        level = tau_d * math.exp(rng.gauss(0.0, sd_difficulty))
        effect = true_r * math.exp(rng.gauss(0.0, sd_effect))
        cell = [0.0, 0.0, 0.0, 0.0]
        for weight, rounds in turns:
            noise_d = rng.gauss(0.0, round_sd / math.sqrt(rounds))
            noise_s = rng.gauss(0.0, round_sd / math.sqrt(rounds))
            tau_d_turn = max(1.0, level + noise_d)
            tau_s_turn = max(1.0, level * effect + noise_s)
            cell[0] += weight * rounds * tau_s_turn
            cell[1] += weight * rounds
            cell[2] += weight * rounds * tau_d_turn
            cell[3] += weight * rounds
        units.setdefault(name, []).append(cell)
    return units


def simulate(meta_rows, r_values=DEFAULT_R, trials=200, seed=0, resamples=300, **model):
    """[dict(r, go_point, kill, go_clear, halfwidth)] over the grid of true R values (shares in [0, 1])."""
    clusters = clusters_of(meta_rows)
    statistic = report.ratio_statistic()
    out = []
    for r in r_values:
        rng = random.Random('%s:%s' % (seed, r))
        go_point = kill = go_clear = 0
        widths = []
        for trial in range(trials):
            units = draw_trial(clusters, r, rng, **model)
            result = report.cluster_bootstrap(units, statistic, resamples, seed=rng.randrange(1 << 30))
            if result['point'] is None or result['low'] is None:
                continue
            go_point += 1 if result['point'] >= report.GO_POINT else 0
            kill += 1 if result['high'] < report.KILL_UPPER else 0
            go_clear += 1 if result['point'] >= report.GO_POINT and result['low'] > 1.0 else 0
            widths.append((result['high'] - result['low']) / 2.0)
        out.append(dict(r=r, go_point=round(go_point / float(trials), 4), kill=round(kill / float(trials), 4),
                        go_clear=round(go_clear / float(trials), 4),
                        halfwidth=round(sum(widths) / len(widths), 4) if widths else None))
    return out


def public_summary(table, trials, seed):
    """The numbers only: flat keys, no strings (it passes tau_lab_report.assert_public as it stands)."""
    summary = dict(trials=trials, seed=seed)
    for row in table:
        tag = ('%.2f' % row['r']).replace('.', '_')
        summary['r_%s' % tag] = dict(go_point=row['go_point'], kill=row['kill'], go_clear=row['go_clear'], halfwidth=row['halfwidth'])
    return summary


def main(argv=None, say=print):
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--meta', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--trials', type=int, default=200)
    parser.add_argument('--resamples', type=int, default=300)
    parser.add_argument('--seed', type=int, default=0)
    options = parser.parse_args(argv)
    with open(options.meta, encoding='utf-8') as handle:
        meta_rows = [json.loads(line) for line in handle if line.strip()]
    table = simulate(meta_rows, DEFAULT_R, options.trials, options.seed, options.resamples)
    summary = public_summary(table, options.trials, options.seed)
    report.assert_public(summary)
    with open(options.out, 'w', encoding='utf-8', newline='\n') as handle:
        json.dump(summary, handle, indent=1, sort_keys=True)
        handle.write('\n')
    for row in table:
        say('R %.2f: GO point %.2f, KILL %.2f, half-width %s' % (row['r'], row['go_point'], row['kill'], row['halfwidth']))
    return 0


if __name__ == '__main__':
    sys.exit(main())
