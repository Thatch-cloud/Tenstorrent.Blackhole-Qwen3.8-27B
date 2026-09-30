"""Matched ABBA scheduling and paired evaluation for layer-level candidates.

Nine-block ABBA order per repo convention; timing pairs (control, candidate)
per block are evaluated into win/tie/loss counts. A candidate that loses any
block is not promoted regardless of its mean (house rule).
"""

import argparse
import json
import statistics


def abba_order(blocks=9, seed=123):
    """Deterministic alternating order, control first."""
    if blocks < 2 or blocks % 2 == 0:
        raise ValueError("ABBA needs an odd block count of at least 3")
    order = []
    for index in range(blocks):
        control_first = index % 2 == 0
        order.append("control" if control_first else "candidate")
    return order


def evaluate(pairs, tie_fraction=0.01):
    """pairs: list of (control_ms, candidate_ms). Returns decision summary."""
    if len(pairs) < 3 or len(pairs) % 2 == 0:
        raise ValueError("Expected an odd number of matched pairs, at least 3")
    wins = losses = ties = 0
    deltas = []
    for control, candidate in pairs:
        if control <= 0 or candidate <= 0:
            raise ValueError("Timings must be positive")
        delta = (candidate - control) / control
        deltas.append(delta)
        if abs(delta) <= tie_fraction:
            ties += 1
        elif delta < 0:
            wins += 1
        else:
            losses += 1
    mean_delta = statistics.fmean(deltas)
    return {
        "blocks": len(pairs),
        "wins": wins,
        "ties": ties,
        "losses": losses,
        "mean_delta": round(mean_delta, 5),
        "all_blocks_not_worse": losses == 0,
        "promotable": losses == 0 and mean_delta < -tie_fraction,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--blocks", type=int, default=9)
    parser.add_argument("--pairs", help="JSON list of [control_ms, candidate_ms] pairs")
    args = parser.parse_args(argv)
    if args.pairs:
        print(json.dumps(evaluate([tuple(pair) for pair in json.loads(args.pairs)]), indent=2))
    else:
        print(json.dumps(abba_order(args.blocks), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
