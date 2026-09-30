"""A scheduling simulation of the fast lane beside standard lanes, on the modelled TP4 round times.

The REAL serving_fast_lane.LaneController drives it (decide before each round, note_round after it), so what this shows is the
controller's, not a restatement of the closed form: with the modelled round times and acceptance, does one frame of one
packed round plus k solo rounds give the fast user >= 150 tok/s while every standard user holds >= 75 tok/s?

The round times are the lanes design's model (docs/tp4-lanes-window.md section 3; every figure derived from the measured TP4
cycle decompositions in the 200 tok/s plan, none from a reference number): F, a solo round on the one-user block, and P(n), a
packed round with n live users (the padded M3 block pays for all four segments at two and three live users). Three lever sets:
  straight  the TP4 port as it stands
  phase1    the exact levers of phase 1 (D1-D5 for solo rounds; L1a, L4, L5, L7 for packed rounds)
  all       phase 1 plus phase 2
at three contexts (4k, 32k, 131k). A frame is P(N + 1) + k F, plus the lane switch cost sigma on each block change.

Acceptance is the second axis: tau, the tokens a user commits per round. The coding fixture the plans use committed 12.1; real
coding text measured 4.4-9.5 (refactors 7.1-9.5). tau decides whether the target is reachable at all, and the simulation says so
plainly where it is not (lanes_model's minimum-tau table is the same arithmetic).

    python lanes_sim.py                     the tables the design quotes
    python lanes_sim.py --tau 10 --n 3      one cell
"""

import argparse
import collections
import random
import sys

from serving_fast_lane import LaneConfig, LaneController

# Modelled round times, ms (the design's mid band). {lever: {context: (F, {live users: P})}}
ROUND_MS = {
    'straight': {'4k': (66.8, {2: 92.9, 3: 100.5, 4: 108.1}), '32k': (70.6, {2: 94.8, 3: 103.4, 4: 112.0}),
                 '131k': (74.0, {2: 101.6, 3: 113.6, 4: 125.5})},
    'phase1': {'4k': (52.9, {2: 73.1, 3: 78.9, 4: 84.7}), '32k': (55.3, {2: 75.1, 3: 81.8, 4: 88.6}),
               '131k': (58.7, {2: 81.8, 3: 92.0, 4: 102.1})},
    'all': {'4k': (45.9, {2: 59.7, 3: 64.6, 4: 69.5}), '32k': (46.7, {2: 61.1, 3: 66.8, 4: 72.5}),
            '131k': (49.2, {2: 66.2, 3: 74.4, 4: 82.6})},
}
LEVERS, CONTEXTS = tuple(ROUND_MS), ('4k', '32k', '131k')
FAST_BAR, STANDARD_BAR = 150.0, 75.0
SIGMA_MS = 1.0          # the lane switch cost (the design's mid: 0.5 / 1.0 / 3.0)


class Result(collections.namedtuple('Result', 'fast standard k_mean frames max_solo_run max_gap_ms solo_share reason')):
    __slots__ = ()

    def both(self, fast_bar=FAST_BAR, standard_bar=STANDARD_BAR, slack=0.99):
        return self.fast >= fast_bar and min(self.standard) >= standard_bar * slack if self.standard else self.fast >= fast_bar


def committed_tokens(rng, tau, jitter):
    """Tokens one user commits in one round: exactly tau, or (jitter) a draw around it kept within one round's 1..16."""
    if not jitter:
        return tau
    return max(1.0, min(16.0, rng.gauss(tau, jitter * tau)))


def simulate(lever, context, n_standard, tau_fast, tau_standard, *, config=None, seconds=60.0, warmup_s=5.0, seed=1,
             jitter=0.0, noise=0.0, scale=1.0, sigma_ms=SIGMA_MS, standard_taus=None):
    """One run of `seconds` of decode: a fast user and n_standard standard users, every round through the real controller.

    tau_standard is every standard user's tokens per round, or `standard_taus` gives each its own. `scale` stretches every round
    time (a pessimism knob: 1.15 is 15% slower than modelled); `noise` a relative gaussian on each round; `jitter` a relative
    gaussian on the tokens a user commits per round. Returns a Result over the steady part (after warmup_s)."""
    rng = random.Random(seed)
    solo_ms, packed = ROUND_MS[lever][context]
    config = LaneConfig.from_environment({}) if config is None else config
    controller = LaneController(config)
    fast = 'fast'
    standard = ['std%d' % index for index in range(n_standard)]
    taus = dict(fast=tau_fast)
    for index, name in enumerate(standard):
        taus[name] = standard_taus[index] if standard_taus else tau_standard
    live = 1 + n_standard
    t, previous = 0.0, None
    commits = collections.defaultdict(list)       # user -> [(time, tokens)]
    solo_rounds = packed_rounds = 0
    solo_run = max_solo_run = 0
    last_packed, max_gap = None, 0.0
    ks = []
    while t < seconds:
        decision = controller.decide(t, fast_live=True, standard_live=bool(standard), solo_possible=True)
        kind = decision.kind
        if kind == 'solo':
            base = solo_ms
        else:
            base = packed[live] if live >= 2 else solo_ms
        switch = previous is not None and previous != kind
        duration = base * scale * (1.0 + (rng.gauss(0.0, noise) if noise else 0.0)) + (sigma_ms if switch else 0.0)
        t += duration / 1000.0
        if kind == 'solo':
            members, solo_rounds, solo_run = [fast], solo_rounds + 1, solo_run + 1
            max_solo_run = max(max_solo_run, solo_run)
        else:
            members, packed_rounds, solo_run = [fast] + standard, packed_rounds + 1, 0
            if last_packed is not None:
                max_gap = max(max_gap, (t - last_packed) * 1000.0)
            last_packed = t
        counts = {user: committed_tokens(rng, taus[user], jitter) for user in members}
        for user in members:
            commits[user].append((t, counts[user]))
        controller.note_round(t, kind, committed=counts, fast_ids=[fast], standard_ids=[u for u in members if u != fast],
                              switch=switch)
        if kind == 'packed':
            ks.append(controller.k)
        previous = kind
    def rate(user):
        events = [event for event in commits[user] if event[0] >= warmup_s]
        if len(events) < 2:
            return 0.0
        return sum(tokens for _, tokens in events[1:]) / (events[-1][0] - events[0][0])
    rounds = solo_rounds + packed_rounds
    tail = ks[len(ks) // 2:] or [0.0]
    return Result(fast=rate(fast), standard=tuple(rate(user) for user in standard), k_mean=sum(tail) / len(tail),
                  frames=packed_rounds, max_solo_run=max_solo_run, max_gap_ms=max_gap,
                  solo_share=solo_rounds / rounds if rounds else 0.0, reason=controller.reason)


def table(levers=LEVERS, contexts=CONTEXTS, taus=(8.0, 10.0, 12.1), ns=(1, 2, 3), **options):
    """The rows the design quotes: (lever, tau, context, N) -> Result, controller on auto at the default floor and margin."""
    rows = collections.OrderedDict()
    for lever in levers:
        for tau in taus:
            for context in contexts:
                for n in ns:
                    rows[(lever, tau, context, n)] = simulate(lever, context, n, tau, tau, **options)
    return rows


def print_table(rows, out=sys.stdout):
    out.write('%-8s %5s %-5s %2s | %6s %6s %6s | %s\n' % ('levers', 'tau', 'ctx', 'N', 'fast', 'std', 'k', 'both bars'))
    for (lever, tau, context, n), result in rows.items():
        out.write('%-8s %5.1f %-5s %2d | %6.0f %6.0f %6.2f | %s\n' % (
            lever, tau, context, n, result.fast, min(result.standard) if result.standard else 0.0, result.k_mean,
            'OK' if result.both() else '--'))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--tau', type=float, action='append', help='tokens per round (repeat for several)')
    parser.add_argument('--n', type=int, action='append', help='standard users (repeat)')
    parser.add_argument('--lever', action='append', choices=LEVERS)
    parser.add_argument('--context', action='append', choices=CONTEXTS)
    parser.add_argument('--scale', type=float, default=1.0)
    options = parser.parse_args(argv)
    rows = table(levers=options.lever or LEVERS, contexts=options.context or CONTEXTS, taus=options.tau or (8.0, 10.0, 12.1),
                 ns=options.n or (1, 2, 3), scale=options.scale)
    print_table(rows)


if __name__ == '__main__':
    main()
