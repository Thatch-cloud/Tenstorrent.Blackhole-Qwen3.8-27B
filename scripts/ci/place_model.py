"""place_model: what the placement of a new request can buy at 2 to 5 live users on the eight-seat stack, as a Markov model of the two M3 blocks. An ESTIMATE; nothing here has run on a card.

    python3 scripts/ci/place_model.py                 the tables (stdlib only, Python 3.7 syntax, a second or two)
    python3 scripts/ci/place_model.py --exhaustive    and the best of ALL stationary admission policies (4,096 chains of 25 states, a few seconds)
    python3 scripts/ci/place_model.py --simulate      and the sensitivity to the memoryless turn: log-normal turns and arrivals in pairs, by event simulation (seeded, a few seconds)

THE QUESTION (docs/tp4-place-1.md). A decode round runs one verify pass per M3 block that holds a live user (at most 5 live: the octo block serves 6 or more). Live users spread over BOTH blocks
cost 1.8x a round that sits in one (measured: 119.6 against 65.3 ms at 3.6k, 146 against 81 ms at depth). The seat of a user is chosen once, when its turn is admitted
(ServingBufferPool.placement_slot), and cannot change while it decodes. So the split of the live users over the blocks is made of two things: which block an ARRIVAL takes (a rule), and
which user LEAVES (not a rule: whoever finishes). This module separates them.

THE MODEL. State (a, b): the live users in block A and block B, 0 to 4 each. A turn arrives at rate rho (per mean turn length), takes the block the RULE names, and leaves at rate 1 (memoryless;
the event simulation in docs/tp4-place-1.md repeats the result with log-normal turns and bursty arrivals). The long-run share of time in each state is the stationary distribution of that chain
(solved exactly, Gaussian elimination on 25 states). The load mixes over the day, so the offered load rho is fitted as a mixture of a few values to the measured production decode-time share by
live count (round-breakdown, s5 rounds files of 2026-10-19/20). A round costs W1 in one block, W2 in two, WO as the octo pass (6 or more live).

THE RULES (mirrors of ServingBufferPool.placement_slot, held equal to the real function by test_place_compact):
  lone_first      today's: the block holding exactly ONE live user first, else the fuller open block (the first on a tie).
  fullest_first   QWEN_FAST_TP4_PLACE_COMPACT=1: the fuller open block (the first on a tie).
BOUNDS AND ALTERNATIVES (no placement rule can reach them; they are what is left after one):
  ideal_compaction   every round at ceil(n / 4) passes whatever the order of leaving: the bound of any migration that moves a live user between blocks at no cost.
  octo_on_split      a round whose live users sit in both blocks (at most 5 live) runs as ONE octo pass (8 rows a user, tau x OCTO_TAU) when that is cheaper than two M3 passes: the octo block with
                     idle segments (QWEN_FAST_OCTO_MIN_LIVE below 6), which today needs scratch pages the octo block is not wired to. Not built; a candidate.
"""
import heapq
import itertools
import math
import random
import sys

# round wall (ms) at a context: (one block, 1 to 4 live; two blocks, live users in both; the octo pass, 6 or more live, and the octo block's wall for octo_on_split). Measured s5 rounds
# (round-breakdown, docs/tp4-place-1.md): 3.6k from the C1/L1 container logs; 64k and 128k the shape model's rows (round_shape_model: M3 verify 48.0 and 58.9 ms + 4.0 of extras and entry +
# the 4-live draft 16.5; two blocks 142.0 and 165.9 measured; the octo pass 90.5 and 105.8). The brief's production pair (81 and 146 ms) sits between the 64k and 128k rows.
WALL_ORDER = ('3.6k', '64k', '128k')
WALLS = {'3.6k': (65.3, 119.6, 81.7), '64k': (68.5, 142.0, 90.5), '128k': (79.4, 165.9, 105.8)}
OCTO_TAU = 0.9                  # the octo pass gives each user 8 rows against 16 (round_shape_model: 5 live, K1 against K0: 1.31 aggregate over 1.46 in wall)
# decode-time share by live count 1..8 (production, s5, 2026-10-19/20; docs/tp4-place-1.md)
OBSERVED_SHARE = (0.343, 0.271, 0.184, 0.099, 0.056, 0.026, 0.016, 0.005)
# the measured share of ROUNDS that ran two verify passes, by live count (same source)
OBSERVED_TWO_PASS = {3: 0.087, 4: 0.438, 5: 0.999}
RHOS = (0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
BLOCK_USERS = 4

STATES = [(a, b) for a in range(BLOCK_USERS + 1) for b in range(BLOCK_USERS + 1)]
INDEX = {state: position for position, state in enumerate(STATES)}


def lone_first(a, b):
    """The block (0 or 1) today's placement_slot gives a new request at counts (a, b), or None when both are full."""
    counts = (a, b)
    open_blocks = [position for position in (0, 1) if counts[position] < BLOCK_USERS]
    if not open_blocks:
        return None
    lone = [position for position in open_blocks if counts[position] == 1]
    return lone[0] if lone else max(open_blocks, key=lambda position: (counts[position], -position))


def fullest_first(a, b):
    """The block QWEN_FAST_TP4_PLACE_COMPACT=1 gives a new request, or None when both are full."""
    counts = (a, b)
    open_blocks = [position for position in (0, 1) if counts[position] < BLOCK_USERS]
    if not open_blocks:
        return None
    return max(open_blocks, key=lambda position: (counts[position], -position))


def stationary(rule, rho):
    """{(a, b): long-run share of time} of the chain: arrivals at rate rho into the block `rule` names, each live user leaving at rate 1."""
    size = len(STATES)
    rates = [[0.0] * size for _ in range(size)]
    for a, b in STATES:
        row = INDEX[(a, b)]
        block = rule(a, b)
        if block is not None:
            rates[row][INDEX[(a + 1, b) if block == 0 else (a, b + 1)]] += rho
        if a:
            rates[row][INDEX[(a - 1, b)]] += a
        if b:
            rates[row][INDEX[(a, b - 1)]] += b
    for row in range(size):
        rates[row][row] = -sum(rates[row][column] for column in range(size) if column != row)
    # pi Q = 0 with sum pi = 1: transpose, replace the last equation by the normalisation, eliminate with partial pivoting
    matrix = [[rates[column][row] for column in range(size)] for row in range(size)]
    matrix[-1] = [1.0] * size
    right = [0.0] * (size - 1) + [1.0]
    for column in range(size):
        pivot = max(range(column, size), key=lambda row: abs(matrix[row][column]))
        matrix[column], matrix[pivot] = matrix[pivot], matrix[column]
        right[column], right[pivot] = right[pivot], right[column]
        for row in range(size):
            if row != column and matrix[row][column] != 0.0:
                factor = matrix[row][column] / matrix[column][column]
                for k in range(column, size):
                    matrix[row][k] -= factor * matrix[column][k]
                right[row] -= factor * right[column]
    return dict((state, right[INDEX[state]] / matrix[INDEX[state]][INDEX[state]]) for state in STATES)


def truncated_poisson(rho):
    """The share of time at 0..8 live of an eight-seat loss system at offered load rho (the placement does not move it)."""
    weights = [1.0]
    for count in range(1, 9):
        weights.append(weights[-1] * rho / count)
    total = sum(weights)
    return [weight / total for weight in weights]


def conditional_share(weights):
    """Decode-time share by live count 1..8 of a mixture {rho: time weight}: the mixture's share at each count over the time at one or more live."""
    totals = [sum(weights[rho] * truncated_poisson(rho)[count] for rho in weights) for count in range(1, 9)]
    whole = sum(totals)
    return [value / whole for value in totals]


def fit_mixture(observed=OBSERVED_SHARE, rhos=RHOS):
    """{rho: weight} (sums to 1) minimising the squared error of the mixture's decode-time share against `observed`; deterministic coordinate descent from the uniform mixture."""
    weights = dict((rho, 1.0 / len(rhos)) for rho in rhos)

    def error(candidate):
        return sum((mine - theirs) ** 2 for mine, theirs in zip(conditional_share(candidate), observed))

    step = 0.2
    while step > 1e-5:
        improved = False
        for rho in rhos:
            for delta in (step, -step):
                candidate = dict(weights)
                candidate[rho] = max(0.0, candidate[rho] + delta)
                if sum(candidate.values()) > 0.0 and error(candidate) < error(weights) - 1e-15:
                    weights, improved = candidate, True
        if not improved:
            step /= 2.0
    total = sum(weights.values())
    return dict((rho, weight / total) for rho, weight in weights.items())


def wall_of(a, b, walls, octo_on_split=False):
    """The round wall in state (a, b), or None when nobody is live. `octo_on_split`: a split round at 5 live or fewer takes the octo pass when that is cheaper (per live user, tau included)."""
    one, two, octo = walls
    live = a + b
    if live == 0:
        return None
    if live >= 6:
        return octo
    if a and b:
        return min(two, octo / OCTO_TAU) if octo_on_split else two
    return one


def mixed_states(rule, weights):
    """{(a, b): time weight} of the mixture of loads, the placement `rule` in force."""
    states = dict((state, 0.0) for state in STATES)
    for rho, weight in weights.items():
        if weight > 0.0:
            for state, share in stationary(rule, rho).items():
                states[state] += weight * share
    return states


def throughput(states, walls, octo_on_split=False):
    """Aggregate tokens a unit of time at one token a live user a round: the time-weighted sum of live / wall."""
    return sum(share * (a + b) / wall_of(a, b, walls, octo_on_split) for (a, b), share in states.items() if a + b)


def two_pass_shares(states, walls):
    """({live: share of TIME in two passes}, {live: share of ROUNDS in two passes}) for 2 to 5 live (a round's weight is its time over its wall)."""
    one, two, _ = walls
    in_time, in_rounds = {}, {}
    for live in range(2, 6):
        split = sum(share for (a, b), share in states.items() if a + b == live and a and b)
        whole = sum(share for (a, b), share in states.items() if a + b == live)
        single = whole - split
        in_time[live] = split / whole
        in_rounds[live] = (split / two) / (split / two + single / one)
    return in_time, in_rounds


def ideal_throughput(weights, walls):
    """The bound of any migration: every round of n live at ceil(n / 4) passes (one block to 4 live, two at 5), whatever the order of leaving."""
    one, two, octo = walls
    total = 0.0
    for rho, weight in weights.items():
        share = truncated_poisson(rho)
        for live in range(1, 9):
            wall = octo if live >= 6 else (two if live == 5 else one)
            total += weight * share[live] * live / wall
    return total


def best_stationary_policy(rho, walls):
    """(policy bits, throughput) of the best of ALL stationary admission policies at one load: a block choice in every state where both blocks are open and unequal. Exhaustive."""
    choice_states = [(a, b) for a in range(BLOCK_USERS) for b in range(BLOCK_USERS) if a != b]
    best = (None, -1.0)
    for bits in itertools.product((0, 1), repeat=len(choice_states)):
        table = dict(zip(choice_states, bits))

        def rule(a, b, table=table):
            if a == BLOCK_USERS and b == BLOCK_USERS:
                return None
            if a == BLOCK_USERS:
                return 1
            if b == BLOCK_USERS:
                return 0
            return 0 if a == b else table[(a, b)]

        value = throughput(stationary(rule, rho), walls)
        if value > best[1]:
            best = (bits, value)
    table = dict(zip(choice_states, best[0]))
    return table, best[1]


def simulate(rule, rho, sigma=None, burst=1, seed=0, horizon=30000.0):
    """{(a, b): long-run share of time} by event simulation: turns arrive in bursts of `burst` at rate rho / burst (a Poisson process of bursts), each takes the block `rule` names (dropped when
    both are full), and lasts Exp(1) (sigma None) or log-normal with mean 1 and the given sigma, whichever user's turn ends. The same seed gives both rules the same turns."""
    generator = random.Random(seed)
    users, ends, now, arrivals_at = {}, [], 0.0, generator.expovariate(rho / burst)
    time_in = dict((state, 0.0) for state in STATES)
    serial = itertools.count()
    while now < horizon:
        moment = min(arrivals_at, ends[0][0] if ends else float('inf'))
        a = sum(1 for block in users.values() if block == 0)
        time_in[(a, len(users) - a)] += moment - now
        now = moment
        if ends and ends[0][0] <= arrivals_at:
            _, who = heapq.heappop(ends)
            del users[who]
            continue
        arrivals_at = now + generator.expovariate(rho / burst)
        for _ in range(burst):
            a = sum(1 for block in users.values() if block == 0)
            block = rule(a, len(users) - a)
            length = generator.expovariate(1.0) if sigma is None else math.exp(generator.gauss(-sigma * sigma / 2.0, sigma))
            if block is not None:
                who = next(serial)
                users[who] = block
                heapq.heappush(ends, (now + length, who))
    total = sum(time_in.values())
    return dict((state, value / total) for state, value in time_in.items())


def sensitivity_lines():
    """The fullest_first gain at the 64k walls and rho 2 under turns that are not memoryless, by simulation."""
    walls = WALLS['64k']
    lines = ['sensitivity to the turn (rho=2.0, walls 64k; event simulation, seeded): fullest_first against today',
             '  turn                        arrivals     two-pass rounds at 4 live (today -> fullest_first)   throughput']
    for sigma, label in ((None, 'exponential'), (1.0, 'log-normal sigma 1.0'), (1.5, 'log-normal sigma 1.5')):
        for burst in (1, 2):
            today, fuller = simulate(lone_first, 2.0, sigma, burst), simulate(fullest_first, 2.0, sigma, burst)
            rounds_today, rounds_fuller = two_pass_shares(today, walls)[1][4], two_pass_shares(fuller, walls)[1][4]
            lines.append('  %-27s %-12s %.3f -> %.3f                                          %+6.2f%%'
                         % (label, 'single' if burst == 1 else 'in pairs', rounds_today, rounds_fuller, 100.0 * (throughput(fuller, walls) / throughput(today, walls) - 1.0)))
    return lines


def tables(exhaustive=False, simulated=False):
    """The report as text."""
    weights = fit_mixture()
    lines = ['calibration: offered load rho (turns per mean turn) mixed over the day, fitted to the production decode-time share by live count',
             '  mixture  ' + '  '.join('rho=%.1f:%.3f' % (rho, weight) for rho, weight in sorted(weights.items()) if weight > 1e-3),
             '  live         ' + ' '.join('%6d' % live for live in range(1, 9)),
             '  production   ' + ' '.join('%6.3f' % value for value in OBSERVED_SHARE),
             '  model        ' + ' '.join('%6.3f' % value for value in conditional_share(weights)), '']
    today, fuller = mixed_states(lone_first, weights), mixed_states(fullest_first, weights)
    for name in WALL_ORDER:
        walls = WALLS[name]
        lines.append('walls %s: one block %.1f ms, two blocks %.1f ms, octo %.1f ms' % ((name,) + walls))
        rounds_today, rounds_fuller = two_pass_shares(today, walls)[1], two_pass_shares(fuller, walls)[1]
        lines.append('  share of ROUNDS in two passes   live:  ' + ' '.join('%6d' % live for live in range(2, 6)))
        lines.append('    production (measured)                 ' + ' '.join('%6s' % ('%.3f' % OBSERVED_TWO_PASS[live] if live in OBSERVED_TWO_PASS else '-') for live in range(2, 6)))
        lines.append('    model, lone_first (today)             ' + ' '.join('%6.3f' % rounds_today[live] for live in range(2, 6)))
        lines.append('    model, fullest_first                  ' + ' '.join('%6.3f' % rounds_fuller[live] for live in range(2, 6)))
        base = throughput(today, walls)
        lines.append('  aggregate decode throughput against today (occupancy held; the placement does not change who is live)')
        lines.append('    fullest_first (QWEN_FAST_TP4_PLACE_COMPACT=1)        %+6.2f%%' % (100.0 * (throughput(fuller, walls) / base - 1.0)))
        lines.append('    ideal_compaction (bound of any migration)            %+6.2f%%' % (100.0 * (ideal_throughput(weights, walls) / base - 1.0)))
        lines.append('    octo_on_split, today\'s placement (not built)         %+6.2f%%' % (100.0 * (throughput(today, walls, True) / base - 1.0)))
        lines.append('    octo_on_split, fullest_first (not built)             %+6.2f%%' % (100.0 * (throughput(fuller, walls, True) / base - 1.0)))
        lines.append('')
    if exhaustive:
        for rho in (1.5, 2.0, 3.0):
            walls = WALLS['64k']
            table, value = best_stationary_policy(rho, walls)
            same = all(table[state] == fullest_first(*state) for state in table)
            lines.append('best of all stationary admission policies at rho=%.1f: %+.2f%% against today, equal to fullest_first in every state: %s'
                         % (rho, 100.0 * (value / throughput(stationary(lone_first, rho), walls) - 1.0), same))
    if simulated:
        lines.extend(sensitivity_lines())
    return '\n'.join(lines)


if __name__ == '__main__':
    print(tables('--exhaustive' in sys.argv[1:], '--simulate' in sys.argv[1:]))
