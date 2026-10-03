"""Offline estimate of prompt-lookup drafting against logged coding rounds (CPU only; no data and no paths in this file).

THE QUESTION. Served S2 commits `tau` tokens per round per user; the DFlash2 drafter's rounds are logged. What would a
host-side lookup (match the last n tokens of prompt + committed-so-far to their most recent earlier occurrence, propose
the up-to-15 tokens that followed) add, and under which selection policy?

THE INPUT (a "tape" file, jsonl or jsonl.gz, one line per turn; build_tapes.py writes it from a lab run):
  id, set, arm, bucket, weight, cluster   labels and the inverse-probability weight
  plen                                    prompt length in tokens
  prompt                                  prompt token ids (may be empty: the lookup then sees committed tokens only,
                                          a LOWER BOUND; the file's header line says which)
  out                                     the greedy output token ids (out[0] is the prefill's token)
  think_end, tool_at                      output indices (or null) where the answer / the tool call begins
  rounds                                  [[off, emitted, counted, cap], ...] in order: the round committed
                                          out[off : off + emitted]; its seed was out[off - 1]; `counted` is the lab's
                                          population (live 4, not the request's terminal round); `cap` the extent-edge
                                          cap the round ran under (null: none)

TWO WAYS TO COUNT.
  recorded  evaluate every policy at the rounds' recorded starts. DFlash2's count is what was logged. Simple, and it
            OVERSTATES a lookup that commits a long run: the rounds that follow a long lookup round in the log commit
            the same copied tokens again (the walk below does not).
  walk      token-conserving: from the start of each span of consecutive logged rounds, advance by the policy's own
            committed count. At a start the log has no round for, DFlash2's outcome comes from the restart model
            (walk to the next logged drafter miss; a miss persists with probability q, else it becomes a hit and the
            walk goes on; unknown positions are misses at the turn's own miss rate; cap 15), the model the earlier
            alt-drafting analysis fitted at q = 0.65 on TP4 logs. A round is counted when its start lies in a counted
            logged round.

THE LOOKUP, as served (scripts/ci/prompt_lookup.py is imported, not copied): key = the last n tokens; candidate = the most
recent earlier occurrence; match length = the key extended backwards while the tokens before agree (cap 64). A lookup
that is shorter than 15 tokens is completed with DFlash2's own rows (the served ticket keeps its width): 'spliced'
below; 'pure' is the lookup's tokens alone.

POLICIES. dflash (always DFlash2), lookup>=m (lookup when match >= m, else DFlash2), max (the better of the two per
round: an upper bound no server can reach, since it needs the answer).

Python 3.7, stdlib only: it runs on the rig host.
"""
import argparse
import gzip
import json
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', '..', 'scripts', 'ci'))
from prompt_lookup import TokenLookup, MAX_BACK  # noqa: E402

KMAX = 15                    # proposal rows after the seed (T16)
EXTENT = 256                 # served rounds never commit across a 256-position family edge
GATES = (2, 3, 4, 6, 8, 12, 16, 24, 32)
NS = (2, 3, 4, 5, 6)
BUCKETS = ((4096, '4k'), (8192, '8k'), (16384, '16k'), (32768, '32k'), (65536, '64k'), (81920, '80k'), (None, '80k+'))


def opened(path):
    return gzip.open(path, 'rt', encoding='utf-8') if path.endswith('.gz') else open(path, encoding='utf-8')


def load(path):
    turns, header = [], {}
    with opened(path) as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get('header'):
                header = record
            else:
                turns.append(record)
    return header, turns


def bucket_of(n):
    for edge, label in BUCKETS:
        if edge is None or n <= edge:
            return label


def region_of(turn, index):
    if turn.get('tool_at') is not None and index >= turn['tool_at']:
        return 'tool'
    if turn.get('thinking') and (turn.get('think_end') is None or index < turn['think_end']):
        return 'reasoning'
    return 'content'


def lcp(tokens, out, start):
    """Common prefix length of the proposal `tokens` against out[start:]."""
    count = 0
    for token, truth in zip(tokens, out[start:]):
        if token != truth:
            break
        count += 1
    return count


def edge_left(plen, g, remaining):
    """Tokens a round starting at output index g may commit: what is left, and not across the next 256 edge (the
    served commit cap: the next position = 1 mod 256 strictly after the round's absolute start)."""
    s = plen + g
    boundary = ((s - 1) // EXTENT + 1) * EXTENT + 1
    return min(remaining, boundary - s)


class Turn:
    """One turn's per-start lookup facts for every n, and the drafter's tape."""

    def __init__(self, record, ns=NS):
        self.record = record
        self.out = record['out']
        self.plen = record.get('plen') or len(record.get('prompt') or ())
        self.nout = len(self.out)
        self.rounds = [tuple(r) + (None,) * (4 - len(r)) for r in record['rounds']]
        self.look = {}
        prompt = list(record.get('prompt') or ())
        for n in ns:
            self.look[n] = self.starts(prompt, n)
        self.status, self.rec = self.tape()

    def starts(self, prompt, n):
        """For every output index g in 1..nout-1: (match, k, accepted) of the lookup over prompt + out[:g]."""
        out, nout = self.out, self.nout
        index = TokenLookup(prompt + out[:1], n)
        rows = [None] * nout
        for g in range(1, nout):
            tokens, match = index.propose(KMAX)
            rows[g] = (match, len(tokens), lcp(tokens, out, g)) if tokens else (0, 0, 0)
            index.extend(out[g:g + 1])
        return rows

    def tape(self):
        """status[g]: 0 hit (a drafted token the target accepted), 1 miss (the first rejected), 2 unknown; rec[g] the logged
        accepted count of a round that started at g (-1 none, -2 capped by the extent edge)."""
        status = [2] * self.nout
        rec = [-1] * self.nout
        for off, em, counted, cap in self.rounds:
            if not em or off < 1 or off >= self.nout:
                continue
            censored = cap is not None and cap < 16 and em >= cap
            for j in range(1, em):
                if off + j - 1 < self.nout and status[off + j - 1] == 2:
                    status[off + j - 1] = 0
            if em < 16 and not censored and off + em - 1 < self.nout:
                status[off + em - 1] = 1
            rec[off] = -2 if censored else min(em - 1, KMAX)
        return status, rec

    def spans(self):
        spans, cur = [], None
        for off, em, counted, cap in self.rounds:
            if not em or off < 1 or off >= self.nout:
                cur = None
                continue
            if cur is not None and cur[1] == off:
                cur[1] = off + em
            else:
                cur = [off, off + em]
                spans.append(cur)
        return [(a, min(b, self.nout)) for a, b in spans if b > a]

    def counted_set(self):
        marks = set()
        for off, em, counted, cap in self.rounds:
            if counted and em:
                marks.update(range(off, min(off + em, self.nout)))
        return marks


class Drafter:
    """DFlash2's accepted count from any start, under the restart model."""

    def __init__(self, turn, q, rng):
        self.turn, self.q, self.rng = turn, q, rng
        self.cache = {}
        status = turn.status
        hits, misses = status.count(0), status.count(1)
        hazard = misses / float(max(1, hits + misses))
        self.miss = [s == 1 or (s == 2 and rng.random() < hazard) for s in status]
        nout = turn.nout
        self.next_miss = [nout] * (nout + 2)
        for g in range(nout - 1, -1, -1):
            self.next_miss[g] = g if self.miss[g] else self.next_miss[g + 1]

    def accepted(self, g):
        """Cached per start: every policy of one seed sees the same draw."""
        if g not in self.cache:
            self.cache[g] = self.draw(g)
        return self.cache[g]

    def draw(self, g):
        turn = self.turn
        rec = turn.rec[g]
        if rec >= 0:
            return rec
        nout = turn.nout
        acc, cur = 0, g
        for _ in range(KMAX + 2):
            x = self.next_miss[cur]
            stop = min(x, nout)
            acc += stop - cur
            if acc >= KMAX or x >= nout:
                return min(acc, KMAX)
            if self.rng.random() < self.q:
                return acc
            acc += 1
            cur = x + 1
        return min(acc, KMAX)


def lookup_commit(turn, n, g, drafter_acc, spliced):
    """(committed, match) of the lookup at round start g, before the edge/remaining cap."""
    match, k, accepted = turn.look[n][g]
    if k == 0:
        return None, 0
    if spliced and k < KMAX and accepted == k:
        accepted = min(KMAX, max(k, drafter_acc))
    return accepted + 1, match


def policies(gates):
    names = ['dflash']
    names += ['lookup>=%d' % m for m in gates]
    names += ['max']
    return names


def round_commit(policy, dflash, lookup, match):
    """The committed count a policy gets from one round's two candidates."""
    if policy == 'dflash' or lookup is None:
        return dflash, False
    if policy == 'max':
        return (lookup, True) if lookup > dflash else (dflash, False)
    gate = int(policy.split('>=')[1])
    return (lookup, True) if match >= gate else (dflash, False)


def outcomes(turn, drafter, n, name, mode, spliced, counted):
    """[(start, committed, lookup_used, lookup_won)] over the turn's counted rounds under one policy."""
    result = []
    if mode == 'recorded':
        for off, em, flag, cap in turn.rounds:
            if not (flag and em and 1 <= off < turn.nout):
                continue
            left = edge_left(turn.plen, off, turn.nout - off)
            lookup, match = lookup_commit(turn, n, off, em - 1, spliced)
            lookup = None if lookup is None else min(lookup, left)
            committed, used = round_commit(name, em, lookup, match)
            result.append((off, committed, used, used and committed > em))
        return result
    for a, b in turn.spans():
        g = a
        while g < b:
            left = edge_left(turn.plen, g, turn.nout - g)
            dflash_acc = drafter.accepted(g)
            dflash = min(dflash_acc + 1, left)
            lookup, match = lookup_commit(turn, n, g, dflash_acc, spliced)
            lookup = None if lookup is None else min(lookup, left)
            committed, used = round_commit(name, dflash, lookup, match)
            if g in counted:
                result.append((g, committed, used, used and committed > dflash))
            g += max(1, committed)
    return result


def simulate(turns, n, mode, q, seed, spliced, gates):
    """({policy: {key: [weighted_tokens, weighted_rounds, rounds, lookup_used, lookup_wins]}}, {policy: per-turn rows}); keys
    ('all',), ('set', s), ('bucket', b), ('arm', a), ('region', r)."""
    names = policies(gates)
    table = dict((name, {}) for name in names)
    per_turn = dict((name, []) for name in names)
    rng = random.Random(seed)
    for turn in turns:
        counted = turn.counted_set()
        record = turn.record
        weight = float(record.get('weight') or 1.0)
        labels = [('all',), ('set', record.get('set')), ('bucket', bucket_of(turn.plen)), ('arm', record.get('arm'))]
        drafter = Drafter(turn, q, rng) if mode == 'walk' else None
        for name in names:
            rows = {}
            for g, committed, used, win in outcomes(turn, drafter, n, name, mode, spliced, counted):
                for key in labels + [('region', region_of(record, g))]:
                    cell = rows.setdefault(key, [0.0, 0.0, 0, 0, 0])
                    cell[0] += weight * committed
                    cell[1] += weight
                    cell[2] += 1
                    cell[3] += 1 if used else 0
                    cell[4] += 1 if win else 0
            for key, cell in rows.items():
                total = table[name].setdefault(key, [0.0, 0.0, 0, 0, 0])
                for i in range(5):
                    total[i] += cell[i]
            cell = rows.get(('all',))
            if cell and cell[2] >= 8:
                per_turn[name].append((record.get('set'), cell[0] / cell[1], weight, cell[2]))
    return table, per_turn


def tau(cell):
    return cell[0] / cell[1] if cell and cell[1] else float('nan')


def pooled(table, name, sets):
    values = [tau(table[name].get(('set', s))) for s in sets]
    values = [v for v in values if v == v]
    return sum(values) / len(values) if values else float('nan')


def quantile(values, weights, q):
    pairs = sorted(zip(values, weights))
    total = float(sum(w for _, w in pairs))
    running, points = 0.0, []
    for v, w in pairs:
        points.append(((running + w / 2.0) / total, v))
        running += w
    if q <= points[0][0]:
        return points[0][1]
    for (l, lo), (r, hi) in zip(points, points[1:]):
        if q <= r:
            return lo + (hi - lo) * (q - l) / (r - l)
    return points[-1][1]


def report(turns, header, args):
    sets = sorted(set(t.record.get('set') for t in turns if t.record.get('set') is not None)) or [None]
    lines = []
    emit = lines.append
    emit('# prompt-lookup offline estimate')
    emit('turns %d, prompt in tape: %s, mode %s, restart q %.2f, spliced %s, seeds %d' % (
        len(turns), header.get('prompt_included'), args.mode, args.q, not args.pure, args.seeds))
    for n in args.ns:
        emit('')
        emit('## key n = %d' % n)
        results = []
        for seed in range(args.seeds):
            table, per_turn = simulate(turns, n, args.mode, args.q, seed, not args.pure, args.gates)
            results.append((table, per_turn))
        names = policies(args.gates)
        emit('policy | pooled tau | rounds | lookup used % | lookup wins % | ' + ' | '.join('set %s' % s for s in sets))
        for name in names:
            taus = [pooled(t, name, sets) for t, _ in results]
            cell = results[0][0][name].get(('all',))
            rounds = cell[2] if cell else 0
            used = 100.0 * cell[3] / rounds if cell and rounds else 0.0
            wins = 100.0 * cell[4] / rounds if cell and rounds else 0.0
            sets_tau = ['%.3f' % (sum(tau(t[name].get(('set', s))) for t, _ in results) / len(results)) for s in sets]
            emit('%s | %.3f | %d | %.1f | %.1f | %s' % (name, sum(taus) / len(taus), rounds, used, wins, ' | '.join(sets_tau)))
        base = sum(pooled(t, 'dflash', sets) for t, _ in results) / len(results)
        emit('dflash baseline %.3f; deltas: ' % base + ', '.join(
            '%s %+.3f (%+.1f%%)' % (name, sum(pooled(t, name, sets) for t, _ in results) / len(results) - base,
                                     100.0 * (sum(pooled(t, name, sets) for t, _ in results) / len(results) / base - 1))
            for name in names[1:]))
        if args.detail:
            for dimension in ('bucket', 'region', 'arm'):
                emit('')
                emit('by %s (policy: dflash, best gate, max):' % dimension)
                table = results[0][0]
                keys = sorted(set(k[1] for k in table['dflash'] if k[0] == dimension), key=str)
                best = max(names[1:-1], key=lambda name: pooled(table, name, sets))
                for key in keys:
                    cell = table['dflash'].get((dimension, key))
                    emit('  %s: rounds %d, dflash %.3f, %s %.3f, max %.3f' % (
                        key, cell[2], tau(cell), best, tau(table[best].get((dimension, key))),
                        tau(table['max'].get((dimension, key)))))
            table, per_turn = results[0]
            best = max(names[1:-1], key=lambda name: pooled(table, name, sets))
            emit('')
            for name in ('dflash', best):
                rows = per_turn[name]
                if rows:
                    vals, wts = [r[1] for r in rows], [r[2] for r in rows]
                    emit('per-turn %s: p10 %.3f p50 %.3f (turns with >= 8 rounds: %d)' % (
                        name, quantile(vals, wts, 0.10), quantile(vals, wts, 0.50), len(rows)))
    return '\n'.join(lines) + '\n'


def parse_args(argv):
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('tapes', help='tape jsonl or jsonl.gz (build_tapes.py)')
    parser.add_argument('--mode', choices=('recorded', 'walk'), default='walk')
    parser.add_argument('--q', type=float, default=0.65, help='restart-model miss persistence (walk)')
    parser.add_argument('--seeds', type=int, default=3, help='restart-model draws averaged (walk)')
    parser.add_argument('--ns', type=int, nargs='+', default=list(NS))
    parser.add_argument('--gates', type=int, nargs='+', default=list(GATES))
    parser.add_argument('--pure', action='store_true', help='the lookup tokens alone: no completion with DFlash2 rows')
    parser.add_argument('--detail', action='store_true')
    parser.add_argument('--limit', type=int, default=0, help='first N turns only')
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(sys.argv[1:] if argv is None else argv)
    header, records = load(args.tapes)
    if args.limit:
        records = records[:args.limit]
    turns = [Turn(r, args.ns) for r in records]
    sys.stdout.write(report(turns, header, args))


if __name__ == '__main__':
    main()
