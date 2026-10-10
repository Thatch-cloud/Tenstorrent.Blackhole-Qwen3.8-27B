"""The paired read of the draft-vocabulary ABAB at eight live seats (QWEN_FAST_DRAFT_VOCAB, docs/tp4-draft-vocab.md). Stdlib only, Python 3.7 syntax: the gate reads this on the rig host.

    python3 scripts/ci/draft_vocab_report.py --control <A1 container log> <A2 container log> --arm <B1 container log> <B2 container log> [--live 8]

The jobs are TV1 (control), TV2 (arm), TV3 (control), TV4 (arm): the control profile and its shortlist twin, one boot each, the same tests, nothing else different. Pair k is the k-th
control log against the k-th arm log. For each log, at `--live` live seats:

  * the EARLY DRAFT time: every `[PACKED-EARLY-DRAFT] round=N path=reuse|redo live=K draft_ms=X` line at K == live is the drafter's wall time inside the step (the launches, the fence
    window and the readback); its median and mean are THE NUMBER THIS LEVER IS FOR (the head matmuls and the top-16 reads it removes are inside it);
  * the packed rounds (acceptance_report.decode_steps via drafter_bf16.compare): paired by episode ordinal and round index over the common prefix, the tau (committed tokens per user per
    round), the mean round, and the per-user committed rate (tau over the mean round) of each arm over the PAIRED rounds only.

The verdict is pre-registered here, before any card has run the lever. For each pair the arm's rate must be at least MIN_GAIN above the control's over the paired rounds, its early draft median
must be lower, and its tau must not have fallen below MIN_TAU_RATIO of the control's (a shortlist can only lose accepted length; what it saves must pay for what it loses). GO needs every
pair to pass. NO-GO is any pair whose rate fell or whose tau fell below MAX_TAU_LOSS_RATIO. Anything in between is MAYBE (repeat, or read the tau lab's `dvocab` arm). Too few rounds
or draft lines in any log is UNREAD. It reads speed and accepted length, never text: the texts are levern_compare.py's (every answer equal, user by user).
Exit 0 GO, 1 NO-GO, 2 UNREAD, 3 MAYBE.
"""
import argparse
import json
import re
import sys

import drafter_bf16

EARLY_PATTERN = re.compile(r'\[PACKED-EARLY-DRAFT\] round=(\d+) path=(\w+) live=(\d+) draft_ms=([0-9.]+)')
DRAFT_PATHS = ('reuse', 'redo')
MIN_PAIRED_ROUNDS = 100
MIN_EARLY_DRAFTS = 30
MIN_GAIN = 0.02
MIN_TAU_RATIO = 0.97
MAX_TAU_LOSS_RATIO = 0.95


def median(values):
    ordered = sorted(values)
    if not ordered:
        return None
    middle = len(ordered) // 2
    return ordered[middle] if len(ordered) % 2 else (ordered[middle - 1] + ordered[middle]) / 2.0


def early_draft_ms(text, live=8):
    """[ms] of every early draft at `live` seats that produced drafts (reuse or redo), in log order."""
    found = []
    for line in (text or '').splitlines():
        match = EARLY_PATTERN.search(line)
        if match and int(match.group(3)) == live and match.group(2) in DRAFT_PATHS:
            found.append(float(match.group(4)))
    return found


def draft_summary(values):
    if not values:
        return dict(n=0)
    return dict(n=len(values), median=round(median(values), 2), mean=round(sum(values) / len(values), 2), p90=round(sorted(values)[int(0.9 * (len(values) - 1))], 2))


def pair(control_text, arm_text, live=8):
    """dict for one pair: the early draft summaries, the draft change in ms (arm minus control, medians), the paired round read, and this pair's verdict
    ('GO' | 'NO-GO' | 'MAYBE' | 'UNREAD') with its reason."""
    control_drafts, arm_drafts = early_draft_ms(control_text, live), early_draft_ms(arm_text, live)
    result = dict(live=live, control_draft=draft_summary(control_drafts), arm_draft=draft_summary(arm_drafts))
    try:
        paired = drafter_bf16.compare(control_text, arm_text, live)
    except ValueError as failure:
        result.update(verdict='UNREAD', reason=str(failure))
        return result
    result['paired'] = {key: paired[key] for key in ('episodes', 'paired_rounds', 'a', 'b', 'tau_ratio_b_over_a', 'round_time_ratio_b_over_a', 'rate_ratio_b_over_a',
                                                       'paired_rate_delta_mean_tok_s', 'rounds_b_faster')}
    if len(control_drafts) < MIN_EARLY_DRAFTS or len(arm_drafts) < MIN_EARLY_DRAFTS or paired['paired_rounds'] < MIN_PAIRED_ROUNDS:
        result.update(verdict='UNREAD', reason='%d control and %d arm early drafts at %d live seats (%d wanted in each), %d paired rounds (%d wanted)' % (
            len(control_drafts), len(arm_drafts), live, MIN_EARLY_DRAFTS, paired['paired_rounds'], MIN_PAIRED_ROUNDS))
        return result
    change = round(result['arm_draft']['median'] - result['control_draft']['median'], 2)
    result['draft_ms_change'] = change
    rate, tau = paired['rate_ratio_b_over_a'], paired['tau_ratio_b_over_a']
    text = 'early draft median %.2f -> %.2f ms (%+.2f), tau x%.4f, round time x%.4f, per-user rate x%.4f over %d paired rounds' % (
        result['control_draft']['median'], result['arm_draft']['median'], change, tau, paired['round_time_ratio_b_over_a'], rate, paired['paired_rounds'])
    if rate < 1.0 or tau < MAX_TAU_LOSS_RATIO:
        verdict = 'NO-GO'
    elif rate >= 1.0 + MIN_GAIN and change < 0 and tau >= MIN_TAU_RATIO:
        verdict = 'GO'
    else:
        verdict = 'MAYBE'
    result.update(verdict=verdict, reason=text)
    return result


def compare(control_texts, arm_texts, live=8):
    """dict(pairs, verdict, reason, draft_ms_saved): the pairs read in order, and the window's verdict (see the module docstring)."""
    if len(control_texts) != len(arm_texts) or not control_texts:
        raise ValueError('the same number of control and arm logs (at least one) is required: pair k is control k against arm k')
    pairs = [pair(control, arm, live) for control, arm in zip(control_texts, arm_texts)]
    verdicts = [item['verdict'] for item in pairs]
    if 'UNREAD' in verdicts:
        verdict = 'UNREAD'
    elif 'NO-GO' in verdicts:
        verdict = 'NO-GO'
    elif all(item == 'GO' for item in verdicts):
        verdict = 'GO'
    else:
        verdict = 'MAYBE'
    saved = [-item['draft_ms_change'] for item in pairs if 'draft_ms_change' in item]
    return dict(pairs=pairs, verdict=verdict, draft_ms_saved=[round(value, 2) for value in saved],
                reason='; '.join('pair %d %s: %s' % (index + 1, item['verdict'], item['reason']) for index, item in enumerate(pairs)))


EXIT = {'GO': 0, 'NO-GO': 1, 'UNREAD': 2, 'MAYBE': 3}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--control', nargs='+', required=True)
    parser.add_argument('--arm', nargs='+', required=True)
    parser.add_argument('--live', type=int, default=8)
    options = parser.parse_args(argv)
    try:
        texts = []
        for group in (options.control, options.arm):
            texts.append([])
            for path in group:
                with open(path, encoding='utf-8', errors='replace') as handle:
                    texts[-1].append(handle.read())
        result = compare(texts[0], texts[1], options.live)
    except (OSError, ValueError) as failure:
        print('DRAFT_VOCAB_REPORT unreadable: %s' % failure, file=sys.stderr)
        return 2
    print('DRAFT_VOCAB_REPORT %s' % json.dumps(result, sort_keys=True))
    print('DRAFT_VOCAB_VERDICT %s: %s' % (result['verdict'], result['reason']))
    return EXIT[result['verdict']]


if __name__ == '__main__':
    sys.exit(main())
