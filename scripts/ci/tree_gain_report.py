"""What would a two-candidate (tree) verify have committed on the same text? An offline read of the lookup's SHADOW lines.

    python3 scripts/ci/tree_gain_report.py server.log [more logs] [--json out.json] [--min-rounds N]

A request run with QWEN_FAST_LOOKUP_DRAFT=<policy> and QWEN_FAST_LOOKUP_SHADOW=1 (prompt_lookup.py) logs one `[LOOKUP-SHADOW]` line per round: DFlash2's proposal rows,
the lookup's raw rows, the match length, the source the policy chose and the tokens the round committed. The committed tokens of consecutive rounds ARE the target's
greedy continuation (the verify only ever commits it), so for any chain a round could have verified, how many rows the target would have accepted is known exactly: the
longest common prefix of the chain with that continuation. This script replays that for the two candidates of every round (DFlash2's chain, and the lookup's rows
completed with DFlash2's, which is what a lookup round verifies) and reports what each policy would have committed per round:

* run:       what the logged run committed (its policy's choice at each round);
* dflash:    DFlash2's chain always (the control's rate on this text, as far as the logged rounds go);
* lookup:    the lookup-completed chain always, where the lookup offered anything, else DFlash2's;
* tree2:     the better of the two at every round (ties to DFlash2): what a two-candidate verify commits, one pass, rows being nearly free.

Two ways to turn per-round commits into a rate, because the rounds a policy visits depend on what it committed before (the earlier lookup_sim's "recorded" and "walk"):

* recorded:  the mean committed per logged round start, every policy evaluated at the starts the RUN visited. It counts a long commit without removing the rounds it would
             have skipped, so it OVERSTATES a policy that commits more than the run did: an upper bound on the gain.
* walk:      a token-conserving replay that only uses the logged starts: after a round it lands at the position its commit reaches and takes the latest logged start at or
             before that (a stale chain: only its tokens beyond the landing count, and when none does, one token for one round). The run's own policy walks exactly its own
             rounds (checked: walk(run) = run), and a policy that commits more lands on starts it was not logged at, where the stale chain is worse than a fresh one: a lower
             bound on the gain.

The truth lies between the two; the timed pair of the built device path is what settles it. Every number is derived from the target's own committed tokens, so a rounds-level
exactness check is part of the report: for rounds the run took from DFlash2 the replayed DFlash2 commit equals the logged one (`dflash_replay_mismatches` must be 0).

PRIVACY. The SHADOW lines carry token ids of the user's text and stay in the private container log. This script prints and writes AGGREGATES only (counts, rates,
ratios, bucketed shares): no token, no request id, no position.

Stdlib only; Python 3.7 syntax.
"""

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import prompt_lookup  # noqa: E402

MATCH_BUCKETS = ((0, 3), (4, 7), (8, 11), (12, 23), (24, 63), (64, 64))     # the lookup's match length in tokens (inclusive); the gate of n3m12 is 12


def load(lines):
    """({request id: [round dicts in log order]}, the count of SHADOW lines that did not parse)."""
    rounds, bad = {}, 0
    for line in lines:
        if prompt_lookup.SHADOW_MARKER not in line:
            continue
        parsed = prompt_lookup.parse_shadow_line(line)
        if parsed is None:
            bad += 1
            continue
        rounds.setdefault(parsed['request'], []).append(parsed)
    return rounds, bad


def continuation(rounds):
    """({position: token}, conflicts): the target's greedy continuation as the committed tokens of a request's rounds give it (a round at position p commits the
    tokens of positions p + 1, p + 2, ...). A position two rounds disagree about is a conflict (the log is not one request's, or a round was logged twice)."""
    known, conflicts = {}, 0
    for item in rounds:
        for offset, token in enumerate(item['committed']):
            where = item['position'] + 1 + offset
            if known.setdefault(where, token) != token:
                conflicts += 1
    return known, conflicts


def accepted_length(chain, known, position):
    """(accepted rows, censored): the rows of `chain` the target accepts, i.e. the longest common prefix of the chain with the continuation after `position`.
    `censored` when the prefix ran off the known continuation before the chain ended (the real count is at least this one)."""
    accepted = 0
    for token in chain:
        want = known.get(position + 1 + accepted)
        if want is None:
            return accepted, True
        if want != token:
            return accepted, False
        accepted += 1
    return accepted, False


def candidates(item):
    """[(name, chain)] of a round: DFlash2's rows, and the lookup's rows completed with them when the lookup offered something that is not DFlash2's chain."""
    chains = [('dflash', item['dflash'])]
    if item['lookup'] and item['dflash']:
        chain = prompt_lookup.lookup_chain(item['lookup'], item['dflash'])
        if chain != item['dflash']:
            chains.append(('lookup', chain))
    return chains


def evaluate(rounds):
    """One request's rounds -> a list of per-round dicts: the commit (accepted + 1) of the run, of DFlash2's chain, of the lookup chain, and of the better of the
    two, with whether any figure was censored and whether the two chains agree on their first row."""
    known, conflicts = continuation(rounds)
    out = []
    for item in rounds:
        position = item['position']
        committed = len(item['committed'])
        row = dict(position=position, run=committed, source=item['source'], match=item['match'], censored=False, second=False, shared_first=False)
        accepted = {}
        for name, chain in candidates(item):
            count, censored = accepted_length(chain, known, position)
            accepted[name] = count
            row['censored'] = row['censored'] or censored
        row['dflash'] = accepted['dflash'] + 1
        row['second'] = 'lookup' in accepted
        row['lookup'] = accepted['lookup'] + 1 if row['second'] else row['dflash']
        row['tree2'] = max(row['dflash'], row['lookup'])
        row['b_wins'] = row['lookup'] > row['dflash']
        if row['second']:
            row['shared_first'] = item['lookup'][0] == item['dflash'][0]
        # The chain the run verified, as the replay reads it: a round the run took from DFlash2 must replay to exactly what it committed (a round the replay could not
        # see to the end, the last of a request or one cut by its budget or an end-of-text row, is censored and exempt). This is the check that the log is read right.
        row['replay_mismatch'] = item['source'] == 'dflash2' and not row['censored'] and committed != row['dflash']
        out.append(row)
    return out, conflicts


def walk(rows, key):
    """(tokens, rounds) of the token-conserving replay of strategy `key` over one request's evaluated rounds (see the module docstring). The replay covers the
    positions from the first logged round's up to where the run ended."""
    if not rows:
        return 0, 0
    starts = [row['position'] for row in rows]
    end = rows[-1]['position'] + rows[-1]['run']
    position, tokens, count, at = starts[0], 0, 0, 0
    while position < end:
        while at + 1 < len(starts) and starts[at + 1] <= position:
            at += 1
        row = rows[at]
        reach = row['position'] + row[key]
        gained = min(reach, end) - position
        if gained < 1:
            gained = 1                       # a stale chain with nothing left: one token, one round
        position += gained
        tokens += gained
        count += 1
    return tokens, count


def bucket_of(match):
    for low, high in MATCH_BUCKETS:
        if low <= match <= high:
            return '%d-%d' % (low, high) if low != high else '%d' % low
    return 'other'


def mean(total, count):
    return None if not count else total / float(count)


def gain(top, bottom):
    """top / bottom - 1, None when either is missing or the bottom is zero."""
    return None if not top or not bottom else top / float(bottom) - 1


def analyze(rounds_by_request, bad_lines=0):
    """The aggregates (no token, id or position) of all requests."""
    keys = ('run', 'dflash', 'lookup', 'tree2')
    total = dict((key, 0) for key in keys)
    n = second = shared = wins = loses = censored = replay_mismatches = conflicts_total = gain_when_wins = 0
    walked = dict((key, [0, 0]) for key in keys)
    by_bucket = {}
    for rounds in rounds_by_request.values():
        rows, conflicts = evaluate(rounds)
        conflicts_total += conflicts
        for key in keys:
            tokens, count = walk(rows, key)
            walked[key][0] += tokens
            walked[key][1] += count
        for row in rows:
            n += 1
            for key in keys:
                total[key] += row[key]
            second += row['second']
            shared += row['second'] and row['shared_first']
            wins += row['b_wins']
            gain_when_wins += (row['lookup'] - row['dflash']) if row['b_wins'] else 0
            loses += row['second'] and row['lookup'] < row['dflash']
            censored += row['censored']
            replay_mismatches += row['replay_mismatch']
            if row['second']:
                bucket = by_bucket.setdefault(bucket_of(row['match']), dict(rounds=0, dflash=0, lookup=0, tree2=0, run=0, wins=0))
                bucket['rounds'] += 1
                for key in keys:
                    bucket[key] += row[key]
                bucket['wins'] += row['b_wins']
    recorded = dict((key, mean(total[key], n)) for key in keys)
    walking = dict((key, mean(walked[key][0], walked[key][1])) for key in keys)
    return dict(
        requests=len(rounds_by_request), rounds=n, unparsed_shadow_lines=bad_lines, continuation_conflicts=conflicts_total,
        dflash_replay_mismatches=replay_mismatches, censored_rounds=censored,
        rounds_with_second_candidate=second, second_shares_first_row=shared, second_candidate_wins=wins, second_candidate_loses=loses,
        mean_gain_rows_when_second_wins=mean(gain_when_wins, wins),
        recorded_tokens_per_round=recorded, walk_tokens_per_round=walking,
        recorded_gain_over_run=gain(recorded['tree2'], recorded['run']), walk_gain_over_run=gain(walking['tree2'], walking['run']),
        recorded_gain_over_dflash=gain(recorded['tree2'], recorded['dflash']), walk_gain_over_dflash=gain(walking['tree2'], walking['dflash']),
        walk_run_equals_run=(walking['run'] is None or recorded['run'] is None or abs(walking['run'] - recorded['run']) < 1e-9),
        by_lookup_match=dict((name, dict(rounds=b['rounds'], wins=b['wins'], mean_dflash=mean(b['dflash'], b['rounds']),
                                         mean_lookup=mean(b['lookup'], b['rounds']), mean_tree2=mean(b['tree2'], b['rounds'])))
                             for name, b in sorted(by_bucket.items())))


def format_value(value):
    return 'n/a' if value is None else ('%.3f' % value if isinstance(value, float) else str(value))


def summary_lines(summary):
    lines = ['[TREEGAIN] requests=%s rounds=%s second_candidate_rounds=%s second_wins=%s second_shares_first_row=%s censored=%s' % (
        summary['requests'], summary['rounds'], summary['rounds_with_second_candidate'], summary['second_candidate_wins'],
        summary['second_shares_first_row'], summary['censored_rounds'])]
    for label in ('recorded', 'walk'):
        rate = summary['%s_tokens_per_round' % label]
        lines.append('[TREEGAIN] %s tokens/round: run=%s dflash=%s lookup=%s tree2=%s | tree2 over run %s, over dflash %s' % (
            label, format_value(rate['run']), format_value(rate['dflash']), format_value(rate['lookup']), format_value(rate['tree2']),
            format_value(summary['%s_gain_over_run' % label]), format_value(summary['%s_gain_over_dflash' % label])))
    lines.append('[TREEGAIN] checks: dflash_replay_mismatches=%s continuation_conflicts=%s unparsed=%s walk(run)=run %s' % (
        summary['dflash_replay_mismatches'], summary['continuation_conflicts'], summary['unparsed_shadow_lines'], summary['walk_run_equals_run']))
    for name, bucket in summary['by_lookup_match'].items():
        lines.append('[TREEGAIN] match %s: rounds=%d second_wins=%d dflash=%s lookup=%s tree2=%s' % (
            name, bucket['rounds'], bucket['wins'], format_value(bucket['mean_dflash']), format_value(bucket['mean_lookup']), format_value(bucket['mean_tree2'])))
    return lines


def problems(summary, min_rounds=1):
    """Why the numbers are not to be read: the checks that failed."""
    found = []
    if summary['rounds'] < min_rounds:
        found.append('%d rounds, fewer than the %d asked for' % (summary['rounds'], min_rounds))
    if summary['dflash_replay_mismatches']:
        found.append('%d DFlash2 rounds replay to another commit than the one logged (the reading of the log is wrong)' % summary['dflash_replay_mismatches'])
    if summary['continuation_conflicts']:
        found.append('%d positions two rounds disagree about (logs of different requests share an id, or a round was logged twice)' % summary['continuation_conflicts'])
    if summary['unparsed_shadow_lines']:
        found.append('%d SHADOW lines did not parse' % summary['unparsed_shadow_lines'])
    if not summary['walk_run_equals_run']:
        found.append('the walk of the run\'s own policy is not the run (the rounds are not consecutive in the log)')
    return found


def from_text(text):
    rounds, bad = load(text.splitlines())
    return analyze(rounds, bad)


def main(argv=None, say=print):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('logs', nargs='+')
    parser.add_argument('--json', default=None, help='write the aggregates here')
    parser.add_argument('--min-rounds', type=int, default=1)
    options = parser.parse_args(argv)
    lines = []
    for path in options.logs:
        with open(path, encoding='utf-8', errors='replace') as handle:
            lines.extend(handle.read().splitlines())
    rounds, bad = load(lines)
    summary = analyze(rounds, bad)
    for line in summary_lines(summary):
        say(line)
    found = problems(summary, options.min_rounds)
    for problem in found:
        say('[TREEGAIN] PROBLEM: %s' % problem)
    if options.json:
        with open(options.json, 'w', encoding='utf-8') as handle:
            json.dump(dict(summary, problems=found), handle, indent=1, sort_keys=True)
    return 1 if found else 0


if __name__ == '__main__':
    sys.exit(main())
