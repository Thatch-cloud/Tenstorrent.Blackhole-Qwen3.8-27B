"""QWEN_FAST_DRAFTER_BF16: what keeping the DFlash2 drafter's weights in bfloat16 costs, and the paired tau report.

The flag itself lives in draft_mlp_branch.draft_projection_dtype (the single decision point for all 36 drafter
projection uploads: five layers of q/k/v/o and gate/up/down, plus the fc feature projection). This module is offline
and stdlib only (nothing served imports it):

  cost(tp, ...)       the per-chip weight bytes at bf16 and bf8, the extra DRAM read per drafter pass, per round at
                      a seat count, and the committed-token gain that pays for it.
  compare(log_a, log_b, live)
                      the PAIRED reading of two server logs (A = bf8 control, B = bf16): rounds paired by (episode ordinal, round index) over the
                      common prefix; per arm tau, mean round and mean(committed) / mean(seconds) over the paired rounds only, the B/A ratios and
                      the per-round rate difference.

    py -3.11 -B scripts/ci/drafter_bf16.py cost [--tp 4] [--bandwidth-gb-s 420]
    py -3.11 -B scripts/ci/drafter_bf16.py compare A.log B.log [--live 4]
"""

import argparse
import json
import sys

# The drafter's shapes (draft_attention_fixture.py, draft_mlp_fixture.py, draft_projection_full_fixture.py; tp_shapes DRAFT_*).
HIDDEN = 5120
LAYERS = 5
Q_ROWS, KV_ROWS, O_COLUMNS = 4096, 1024, 4096     # 32 query heads and 8 K/V heads of 128
MLP = 17408
TAPS = 5                                          # fc: (5 * 5120) -> 5120
# DRAM bytes per element: bfloat16 is two; bfloat8_b is one mantissa byte plus a shared exponent byte per 16 elements.
BYTES_BF16 = 2.0
BYTES_BF8 = 1.0 + 1.0 / 16.0
# Blackhole p150a: 512 GB/s of GDDR6; the target's verify streams its weights at about 420 GB/s (measured, tp4 profile:
# 17.0 ms for the per-chip target weights), which is the achieved figure used for the time estimate.
PEAK_GB_S = 512.0
ACHIEVED_GB_S = 420.0


def projection_elements():
    """Total elements of the 36 projection tensors (all chips together)."""
    attention = Q_ROWS * HIDDEN + 2 * KV_ROWS * HIDDEN + HIDDEN * O_COLUMNS
    mlp = 3 * MLP * HIDDEN
    return LAYERS * (attention + mlp) + HIDDEN * TAPS * HIDDEN


def cost(tp=4, *, passes_per_round=1, round_ms=None, bandwidth_gb_s=ACHIEVED_GB_S):
    """Per-chip weight bytes at both dtypes and the extra DRAM read of bf16 over bf8, per drafter pass and per round
    (passes_per_round drafter passes, e.g. the number of 64-row quad blocks). The projections shard evenly over the chips."""
    per_chip = projection_elements() / float(tp)
    bf16, bf8 = per_chip * BYTES_BF16, per_chip * BYTES_BF8
    extra = bf16 - bf8
    pass_ms = extra / (bandwidth_gb_s * 1e9) * 1e3
    result = dict(tp=tp, projection_elements=projection_elements(), per_chip_bf16_mb=round(bf16 / 1e6, 1),
                  per_chip_bf8_mb=round(bf8 / 1e6, 1), extra_mb_per_chip=round(extra / 1e6, 1),
                  extra_ms_per_pass=round(pass_ms, 3),
                  extra_ms_per_pass_at_peak=round(extra / (PEAK_GB_S * 1e9) * 1e3, 3),
                  passes_per_round=passes_per_round, extra_ms_per_round=round(pass_ms * passes_per_round, 3))
    if round_ms:
        result['round_ms'] = round_ms
        result['break_even_tau_gain_pct'] = round(100.0 * pass_ms * passes_per_round / round_ms, 2)
    return result


def episodes(log_text, live):
    """The timed packed rounds with `live` live users, split into episodes: a maximal run of back-to-back decode steps (one test's decode phase, or
    one steady stretch). A new episode starts when a step is not the immediate successor of the previous one (a prefill step, an admission or a gap
    came between). Each round is (emitted per user, seconds)."""
    import acceptance_report

    result, current, previous_end = [], None, None
    for step in acceptance_report.decode_steps(log_text):
        if step['live'] != live or len(step['packed']) != live or not step['seconds'] or step['seconds'] <= 0:
            previous_end = None
            continue
        if current is None or previous_end is None or abs(step['at'] - previous_end) > 0.002:
            current = []
            result.append(current)
        current.append((list(step['packed']), step['seconds']))
        previous_end = step['at'] + step['seconds']
    return result


def paired_rounds(log_text, live):
    """Every timed packed round with `live` live users, in log order (episodes flattened)."""
    return [round_ for episode in episodes(log_text, live) for round_ in episode]


def _paired_summary(rounds):
    emitted = [value for counts, _ in rounds for value in counts]
    seconds = [second for _, second in rounds]
    tau, mean_round = sum(emitted) / len(emitted), sum(seconds) / len(seconds)
    return dict(rounds=len(rounds), tau=round(tau, 3), mean_round_ms=round(mean_round * 1000.0, 2),
                per_user_tok_s=round(tau / mean_round, 2))


def compare(log_a, log_b, live=4):
    """The PAIRED reading of A (control, bf8) against B (bf16). Both arms run the same tests in the same admission order, so the episodes (one test's
    back-to-back decode rounds) are paired by ordinal and, inside an episode, the rounds by index over the common prefix. Only paired rounds are
    summarised: per arm, tau (mean committed per user per round), the mean round time, and the rate mean(committed) / mean(seconds); then the B/A
    ratios, the mean of the per-round rate difference (committed per user / seconds, B minus A) and how many paired rounds B won. Episodes beyond
    the shorter arm's count are dropped. Greedy lossless: the tokens committed differ only through the drafter's proposals."""
    a_episodes, b_episodes = episodes(log_a, live), episodes(log_b, live)
    if not a_episodes or not b_episodes:
        raise ValueError('no timed packed round with %d live users in %s' % (live, 'A' if not a_episodes else 'B'))
    pairs = []
    for left, right in zip(a_episodes, b_episodes):
        pairs += list(zip(left, right))
    a = [left for left, _ in pairs]
    b = [right for _, right in pairs]
    deltas = [sum(rb) / len(rb) / sb - sum(ra) / len(ra) / sa for (ra, sa), (rb, sb) in pairs]
    result = dict(live=live, episodes=min(len(a_episodes), len(b_episodes)), episodes_unpaired=(len(a_episodes), len(b_episodes)),
                  paired_rounds=len(pairs), a=_paired_summary(a), b=_paired_summary(b),
                  paired_rate_delta_mean_tok_s=round(sum(deltas) / len(deltas), 3), rounds_b_faster=sum(1 for d in deltas if d > 0),
                  per_round_committed=dict(a=[counts for counts, _ in a], b=[counts for counts, _ in b]))
    result['round_time_ratio_b_over_a'] = round(result['b']['mean_round_ms'] / result['a']['mean_round_ms'], 4)
    result['tau_ratio_b_over_a'] = round(result['b']['tau'] / result['a']['tau'], 4)
    result['rate_ratio_b_over_a'] = round(result['b']['per_user_tok_s'] / result['a']['per_user_tok_s'], 4)
    result['bf16_wins'] = result['rate_ratio_b_over_a'] > 1.0
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    sub = parser.add_subparsers(dest='command', required=True)
    c = sub.add_parser('cost')
    c.add_argument('--tp', type=int, default=4)
    c.add_argument('--bandwidth-gb-s', type=float, default=ACHIEVED_GB_S)
    k = sub.add_parser('compare')
    k.add_argument('log_a')
    k.add_argument('log_b')
    k.add_argument('--live', type=int, default=4)
    args = parser.parse_args(argv)
    if args.command == 'cost':
        for seats, passes, round_ms in ((4, 1, 100.0), (8, 2, 299.0)):
            print('seats=%d %s' % (seats, json.dumps(cost(args.tp, passes_per_round=passes, round_ms=round_ms,
                                                           bandwidth_gb_s=args.bandwidth_gb_s))))
        return 0
    with open(args.log_a, encoding='utf-8', errors='replace') as left, open(args.log_b, encoding='utf-8', errors='replace') as right:
        result = compare(left.read(), right.read(), args.live)
    print(json.dumps(result))
    return 0


if __name__ == '__main__':
    sys.exit(main())
