"""The octo-T8 gates' judgements (QWEN_FAST_OCTO, QWEN_FAST_SOLO_PACKED): what an arm's container log must show, as functions of the log, where a CPU test can hold
them. Stdlib and the markers module only, Python 3.7 syntax: the gate reads this on the rig host.

A mounted change is not an executed change. A profile can carry QWEN_FAST_OCTO=alternate and run every round on the M3 blocks (the block never built, the policy
never asked, a fall-back on every round), and its texts will equal the control's - which proves nothing about the octo shape. So every rule below is read from a
line the ROUND wrote AFTER it ran (octo_markers: the step counts a round `shape=octo` only when the octo block's own verify counter moved).

THE RULES OF AN OCTO ARM (judge). Read from the arm's profile env and its container log:
  flag off   not one [OCTO] line may appear (the profile leaves QWEN_FAST_OCTO off or unset): a leak means the shape ran where nobody asked for it.
  attach     exactly one `[OCTO] admitted` line whose mode and min_live are the profile's, at least one UNQUALIFIED line (the admission says what nothing has
             qualified), and no `[OCTO] refused` line.
  executed   (a) the positive control: at least MIN_ROUNDS rounds with shape=octo, each at rows=8 with live at least min_live; the counters on the lines
             (octo_rounds, m3_rounds) are the lines' own count, so a dropped line is seen.
  alternate  (b) under `alternate`: at least MIN_ROUNDS counted rounds of EACH shape, and the counted rounds strictly alternate octo, m3, octo, m3 (a counted
             round is an eligible one that ran as planned). Rounds counted per shape are reported as facts. Under `live` no counted round is an m3 one.
  fall-backs an eligible round that ran as something else than planned is a fall-back: more than MAX_FALLBACK_SHARE of the eligible rounds fails the arm (the
             shape the arm was built to measure did not run).
  programs   (c) no program compiled on a shape switch: every `[OCTO] programs` line (the first round of each shape after a switch, and of the process) must
             have after == before; at least one line must carry numbers (an unreadable program counter means the tripwire did not run). This is a step-scoped
             count: the drafts and the prefills are judged by the rules that already cover them.
  solo       QWEN_FAST_SOLO_PACKED=1: at least one `packed padded round live=1` line (the lone-user round ran on the padded block) and no `packed padded skipped
             live=1 eligible=1` line (an eligible lone user must not have gone to the per-request engines).

PAIRED TIMING (pair_verdict). Under `alternate` the counted rounds come in pairs (octo, then m3) on ONE boot, the same users at the same positions. A round's
cycle is its step plus the gap before it (gap_ms: the drafts, the host work and the scheduler between two steps), so the drafts both shapes pay are in it. The
per-seat rate of a round is committed / live / cycle. GO at MIN_GAIN (+10%) committed tokens per second per seat at 8 live, from the median of the per-pair ratios AND
from the aggregate, every pair at 8 live; every text exact is the exactness job's, not this module's. Both arms of an alternate boot pay the full pre-stage a
switch costs (every shape switch bumps the fixture epoch), so the boot is the fair pair; the `live` boot against the flag-off control is the check without it.
"""
import argparse
import json
import re
import sys

import octo_markers as markers

OCTO_FLAG = 'QWEN_FAST_OCTO'
MIN_LIVE_FLAG = 'QWEN_FAST_OCTO_MIN_LIVE'
SOLO_PACKED_FLAG = 'QWEN_FAST_SOLO_PACKED'
MIN_LIVE_DEFAULT = 6
MIN_ROUNDS = 16
MAX_FALLBACK_SHARE = 0.10
MIN_GAIN = 1.10
MIN_PAIRS = 16
PADDED_LIVE1 = re.compile(r'\[PINDIAG\] packed padded round live=1 ')
PADDED_SKIPPED_LIVE1 = re.compile(r'\[PINDIAG\] packed padded skipped live=1 eligible=1 ')


def mode(env):
    """The profile's QWEN_FAST_OCTO: 'off', 'live' or 'alternate' (anything else comes back as written, for the profile test to refuse)."""
    return str((env or {}).get(OCTO_FLAG, 'off'))


def min_live(env):
    text = str((env or {}).get(MIN_LIVE_FLAG, MIN_LIVE_DEFAULT))
    return int(text) if text.isdigit() else MIN_LIVE_DEFAULT


def solo_packed(env):
    return str((env or {}).get(SOLO_PACKED_FLAG, '0')) == '1'


def lines_of(text):
    return (text or '').splitlines()


def shape_counts(rounds):
    counts = dict(octo=0, m3=0, seq=0)
    for item in rounds:
        counts[item['shape']] += 1
    return counts


def alternation_problem(counted):
    """None when the counted rounds strictly alternate, starting with octo; else the first break, named."""
    expected = 'octo'
    for index, item in enumerate(counted):
        if item['shape'] != expected:
            return ('counted round %d (log round %d) ran %s where the alternation owed %s: %s'
                    % (index + 1, item['round'], item['shape'], expected, ' '.join(c['shape'] for c in counted[max(0, index - 3):index + 2])))
        expected = 'm3' if expected == 'octo' else 'octo'
    return None


def judge(env, container_text):
    """(problems, facts): see the module docstring. facts is {} for a profile that asks for neither flag and whose log holds no octo line."""
    env = env or {}
    asked = mode(env)
    solo = solo_packed(env)
    text = container_text or ''
    found = markers.scan(text)
    problems, facts = [], {}
    if asked == 'off':
        if found['lines']:
            first = markers.marker_lines(text)[0].strip()[:200]
            problems.append('%d [OCTO] line(s) on a profile without %s: the shape ran where nobody asked for it (first: %s)'
                            % (found['lines'], OCTO_FLAG, first))
    elif asked not in ('live', 'alternate'):
        problems.append('%s=%r is neither off, live nor alternate' % (OCTO_FLAG, asked))
    else:
        problems.extend(octo_problems(env, asked, found, facts))
    if solo:
        problems.extend(solo_problems(text, facts))
    return problems, facts


def octo_problems(env, asked, found, facts):
    problems = []
    minimum = min_live(env)
    admitted = found['admitted']
    if found['refused']:
        problems.append('%d [OCTO] refused line(s): the attach did not admit the shape' % found['refused'])
    if len(admitted) != 1:
        problems.append('%d [OCTO] admitted lines (one wanted): the attach admitted the shape %s' % (len(admitted), 'never' if not admitted else 'more than once'))
    else:
        if admitted[0]['mode'] != asked or admitted[0]['min_live'] != minimum:
            problems.append('the admission says mode=%s min_live=%d and the profile asks %s %d' % (admitted[0]['mode'], admitted[0]['min_live'], asked, minimum))
    if not found['unqualified']:
        problems.append('no [OCTO] UNQUALIFIED line: the admission did not say what is unqualified')
    rounds = found['rounds']
    counts = shape_counts(rounds)
    eligible = [item for item in rounds if item['eligible']]
    counted = [item for item in rounds if item['counted']]
    fallbacks = [item for item in eligible if not item['counted']]
    facts.update(octo_mode=asked, rounds=counts, eligible_rounds=len(eligible), counted_rounds=len(counted), fallbacks=len(fallbacks),
                 programs_lines=len(found['programs']))
    # (a) the positive control: the octo block executed
    octo_rounds = [item for item in rounds if item['shape'] == 'octo']
    if len(octo_rounds) < MIN_ROUNDS:
        problems.append('only %d round(s) ran on the octo block (%d wanted): the flag is set and the shape did not execute' % (len(octo_rounds), MIN_ROUNDS))
    for item in octo_rounds:
        if item['rows'] != 8 or item['live'] < minimum:
            problems.append('log round %d ran as octo at rows=%d live=%d (rows 8 and live at least %d wanted)' % (item['round'], item['rows'], item['live'], minimum))
            break
    if rounds:
        last = rounds[-1]
        if last['octo_rounds'] != counts['octo'] or last['m3_rounds'] != counts['m3']:
            problems.append('the last round line counts octo_rounds=%d m3_rounds=%d and the log holds %d and %d round lines: lines are missing'
                            % (last['octo_rounds'], last['m3_rounds'], counts['octo'], counts['m3']))
    # (b) the shapes counted, and the alternation
    per_shape = dict(octo=sum(1 for item in counted if item['shape'] == 'octo'), m3=sum(1 for item in counted if item['shape'] == 'm3'))
    facts['counted_per_shape'] = per_shape
    if asked == 'alternate':
        for shape in ('octo', 'm3'):
            if per_shape[shape] < MIN_ROUNDS:
                problems.append('only %d counted %s round(s) under alternate (%d wanted): the pairs were not timed' % (per_shape[shape], shape, MIN_ROUNDS))
        broken = alternation_problem(counted)
        if broken:
            problems.append('the counted rounds do not alternate: ' + broken)
    elif per_shape['m3']:
        problems.append('%d counted m3 round(s) under live: an eligible round planned octo and ran as m3 is a fall-back, never a counted round' % per_shape['m3'])
    if eligible and len(fallbacks) > MAX_FALLBACK_SHARE * len(eligible):
        problems.append('%d of %d eligible rounds ran as something else than planned (more than %d%%): the shape this arm measures did not run'
                        % (len(fallbacks), len(eligible), int(MAX_FALLBACK_SHARE * 100)))
    # (c) no program compiled on a shape switch
    programs = found['programs']
    numbered = [item for item in programs if item['before'] is not None and item['after'] is not None]
    for item in numbered:
        if item['after'] > item['before']:
            problems.append('%d program(s) compiled on the first %s round after %s (log round %d: program cache %d -> %d): a shape switch must compile nothing'
                            % (item['after'] - item['before'], item['shape'], 'a switch' if item['switch'] else 'the process start', item['round'],
                               item['before'], item['after']))
    if rounds and not numbered:
        problems.append('no [OCTO] programs line carries numbers: the program counter was unreadable, so nothing checked that a shape switch compiles nothing')
    elif numbered and not any(item['shape'] == 'octo' for item in numbered):
        problems.append('no [OCTO] programs line for the octo shape: its first round was not checked')
    facts['programs_compiled'] = sum(max(0, item['after'] - item['before']) for item in numbered)
    return problems


def solo_problems(text, facts):
    problems = []
    ran = len(PADDED_LIVE1.findall(text))
    skipped = len(PADDED_SKIPPED_LIVE1.findall(text))
    facts.update(solo_packed_rounds=ran, solo_packed_skipped_eligible=skipped)
    if not ran:
        problems.append('%s=1 and not one padded round of a lone live user ran (no "packed padded round live=1" line): the policy never executed' % SOLO_PACKED_FLAG)
    if skipped:
        problems.append('%d eligible lone-user round(s) went to the per-request engines ("packed padded skipped live=1 eligible=1")' % skipped)
    return problems


def median(values):
    ordered = sorted(values)
    if not ordered:
        return None
    middle = len(ordered) // 2
    return ordered[middle] if len(ordered) % 2 else (ordered[middle - 1] + ordered[middle]) / 2.0


GAP_OUTLIER_FACTOR = 4.0
GAP_OUTLIER_FLOOR_MS = 50.0


def pairs_of(rounds, live=8):
    """[(octo round, m3 round)]: an octo round and the m3 round that is the very NEXT round of the log (an uncounted round between them is no pair), both counted and at
    `live` live seats. A round with no gap read (the first of a boot has no step before it) cannot be timed: its pair is left out. A pair whose gap is an outlier (more than
    GAP_OUTLIER_FACTOR times the median gap of the candidates, plus GAP_OUTLIER_FLOOR_MS: a prefill, an admission or an idle stretch sat before the round, and that is not
    the shape's cost) is left out too."""
    found = []
    for first, second in zip(rounds, rounds[1:]):
        if (first['counted'] and second['counted'] and first['shape'] == 'octo' and second['shape'] == 'm3' and second['round'] == first['round'] + 1
                and first['live'] == live and second['live'] == live and first['gap_ms'] > 0 and second['gap_ms'] > 0):
            found.append((first, second))
    gaps = [item['gap_ms'] for pair in found for item in pair]
    ceiling = GAP_OUTLIER_FACTOR * (median(gaps) or 0.0) + GAP_OUTLIER_FLOOR_MS
    return [pair for pair in found if max(pair[0]['gap_ms'], pair[1]['gap_ms']) <= ceiling]


def seat_rate(item):
    """Committed tokens per second per live seat over the round's cycle (its step and the gap before it), or None."""
    cycle = (item['step_ms'] + item['gap_ms']) / 1000.0
    return None if cycle <= 0 or not item['live'] else item['committed'] / float(item['live']) / cycle


def pair_verdict(container_text, bar=MIN_GAIN, live=8):
    """The paired timing of an alternate boot: dict(pairs, octo_rate, m3_rate, ratio, median_pair_ratio, go, reason). `go` is True/False, or None when there are fewer than
    MIN_PAIRS pairs at `live` seats to judge."""
    pairs = [(a, b) for a, b in pairs_of(markers.rounds(container_text), live)]
    if len(pairs) < MIN_PAIRS:
        return dict(pairs=len(pairs), go=None, reason='%d pairs at %d live seats (%d wanted)' % (len(pairs), live, MIN_PAIRS))
    ratios, octo_cycle, m3_cycle, octo_tokens, m3_tokens = [], 0.0, 0.0, 0, 0
    for first, second in pairs:
        rate_octo, rate_m3 = seat_rate(first), seat_rate(second)
        if rate_octo is None or rate_m3 is None or not rate_m3:
            continue
        ratios.append(rate_octo / rate_m3)
        octo_cycle += first['step_ms'] + first['gap_ms']
        m3_cycle += second['step_ms'] + second['gap_ms']
        octo_tokens += first['committed']
        m3_tokens += second['committed']
    if not ratios:
        return dict(pairs=len(pairs), go=None, reason='no pair has a readable cycle')
    octo_rate = octo_tokens / float(live) / (octo_cycle / 1000.0)
    m3_rate = m3_tokens / float(live) / (m3_cycle / 1000.0)
    ratio, middle = octo_rate / m3_rate, median(ratios)
    return dict(pairs=len(ratios), octo_rate=round(octo_rate, 2), m3_rate=round(m3_rate, 2), ratio=round(ratio, 4), median_pair_ratio=round(middle, 4),
                bar=bar, go=bool(ratio >= bar and middle >= bar),
                reason='octo %.2f vs m3 %.2f committed tok/s per seat at %d live: aggregate x%.3f, median pair x%.3f, bar x%.2f'
                       % (octo_rate, m3_rate, live, ratio, middle, bar))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--container-log', required=True)
    parser.add_argument('--profile')
    parser.add_argument('--profiles', default=None)
    parser.add_argument('--env', action='append', default=[], help='NAME=VALUE (repeatable): the profile env, when no --profile is given')
    parser.add_argument('--verdict', action='store_true', help='print the paired timing verdict of an alternate boot (exit 0 GO, 1 NO-GO, 2 unread)')
    parser.add_argument('--live', type=int, default=8)
    options = parser.parse_args(argv)
    try:
        with open(options.container_log, encoding='utf-8', errors='replace') as handle:
            text = handle.read()
        env = dict(item.split('=', 1) for item in options.env)
        if options.profile:
            import os

            path = options.profiles or os.path.join(os.path.dirname(os.path.abspath(__file__)), 'qwen_c2_profiles.json')
            with open(path, encoding='utf-8') as handle:
                env = dict(json.load(handle)['profiles'][options.profile].get('env') or {}, **env)
    except (OSError, KeyError, ValueError) as error:
        print('OCTO_JUDGE unreadable: %s' % error, file=sys.stderr)
        return 2
    problems, facts = judge(env, text)
    print('OCTO_JUDGE %s' % json.dumps(facts, sort_keys=True))
    for problem in problems:
        print('OCTO_JUDGE FAILED: %s' % problem)
    if options.verdict:
        verdict = pair_verdict(text, live=options.live)
        print('OCTO_VERDICT %s' % json.dumps(verdict, sort_keys=True))
        if problems:
            return 1                 # an arm that failed its rules has no verdict worth reading (and an arm whose octo shape never ran has no pairs: that is a 1, not an "unread")
        if verdict['go'] is None:
            return 2
        print('OCTO_VERDICT %s: %s' % ('GO' if verdict['go'] else 'NO-GO', verdict['reason']))
        return 0 if verdict['go'] else 1
    return 1 if problems else 0


if __name__ == '__main__':
    sys.exit(main())
