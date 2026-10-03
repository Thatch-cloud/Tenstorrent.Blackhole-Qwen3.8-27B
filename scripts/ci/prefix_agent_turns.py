"""The agent-turn replay's reading (tp4/packed-prefix, sticky sessions at TP4 on eight seats).

prefix_replay.scenario_agent_turns runs busy coding agents - eight conversations of several turns each, every turn the
previous context plus the model's own answer plus a tool result, so the contexts grow - on ONE engine, as the
`agent-turns-prefix` arm (a sticky-session profile) and, for the control, the `agent-turns-baseline` arm (the same
profile without reuse). The conversations are seeded by the phase and the agent, never by the arm, so the two arms send the
same messages as long as the model answers the same: the end-to-end identity this module checks (every turn's output token
ids, and so every later turn's prompt) is the strongest exactness reading there is - one divergence anywhere changes every
later prompt of its conversation, and is reported once, at its first turn.

It reads the replay's records (after prefix_judge.resolve for the markers; the offline comparator reads records.jsonl):

  turn_rows      one row per served turn: the agent, the turn, the prompt tokens, the tokens REUSED (the model's [PREFIX]
                 row Q; 0 on a control), the tokens NEW (prompt - reused), TTFT, the decode rate after the first token
                 (completion tokens - 1 over wall - TTFT), and the engine build the sticky line timed;
  render_row     that row as one log line per turn (the per-turn logging);
  reuse_findings the prefix arm's smoke rules: every agent that continued reused tokens on at least one continuation (an
                 engaged marker alone is not reuse), and the turns that restored nothing counted;
  compare        two arms' transcripts, paired by (case, conversation, turn): IDENTICAL, DIVERGED (the first one per
                 conversation), NOT_COMPARABLE (downstream of a divergence: a different prompt), ERROR, and the turns only one
                 arm ran; plus the paired continuation TTFT and decode rate.

    python3 scripts/ci/prefix_agent_turns.py <arm A results dir or records.jsonl> <arm B ...> [--label-a X --label-b Y]

compares two single-arm runs (the ABAB jobs run each arm in its own job): exit 0 only when every paired turn is IDENTICAL and
neither arm lacks a turn the other ran, 1 otherwise, 2 for a missing input.

Stdlib only, Python 3.7 syntax: it runs on the rig host.
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

ROLE = 'agent'


def agent_records(records):
    return [r for r in records if r.get('role') == ROLE]


def _round(value, digits=2):
    return None if value is None else round(value, digits)


def percentile(values, fraction):
    values = sorted(value for value in values if value is not None)
    if not values:
        return None
    rank = max(1, int(-(-fraction * len(values) // 1)))
    return values[min(len(values), rank) - 1]


def decode_tps(record):
    """Tokens per second after the first token, or None when the answer is one token or the times are absent."""
    out, ttft, wall = record.get('completion_tokens'), record.get('ttft_s'), record.get('wall_s')
    if not isinstance(out, int) or out < 2 or ttft is None or wall is None or wall <= ttft:
        return None
    return (out - 1) / float(wall - ttft)


def turn_rows(records):
    """One row per agent turn, in send order. `reused` is the row's Q from the model's own marker (None when the record
    carries no marker: a control, or a turn the log did not resolve), never vLLM's raw prefix counter."""
    rows = []
    for record in sorted(agent_records(records), key=lambda r: r.get('sent_s') or 0):
        markers = record.get('markers') or {}
        prompt = record.get('prompt_tokens')
        reused = markers.get('q')
        builds = markers.get('sticky_builds') or ()
        rows.append(dict(
            tag=record.get('tag'), case=record.get('case'), conv=record.get('conv'), turn=record.get('turn'),
            ok=bool(record.get('ok')), continuation=bool(record.get('continuation')), prompt_tokens=prompt,
            reused=reused, new=(prompt - reused) if isinstance(prompt, int) and isinstance(reused, int) else None,
            completion_tokens=record.get('completion_tokens'), ttft_s=record.get('ttft_s'),
            wall_s=record.get('wall_s'), decode_tps=_round(decode_tps(record), 2),
            build_ms=builds[0].get('ms') if builds else None, finish=record.get('finish')))
    return rows


def render_row(arm, row):
    return ('[TURN] %s %s turn=%s prompt=%s reused=%s new=%s out=%s ttft=%s s decode=%s tok/s build=%s ms%s' % (
        arm, row['conv'], row['turn'], row['prompt_tokens'], row['reused'], row['new'], row['completion_tokens'],
        row['ttft_s'], row['decode_tps'], row['build_ms'], '' if row['ok'] else ' FAILED'))


def reuse_findings(records):
    """The prefix arm's reuse smoke rule. -> (problems, not exercised, lines)."""
    problems, missing, lines = [], [], []
    served = [r for r in agent_records(records) if r.get('ok')]
    by_conv = {}
    for record in served:
        by_conv.setdefault(record.get('conv'), []).append(record)
    continued = [r for r in served if r.get('continuation')]
    reused = [r for r in continued if (r.get('markers') or {}).get('q')]
    lines.append('continuations that reused tokens: %d of %d (tokens reused %d of %d prompt tokens)' % (
        len(reused), len(continued), sum((r['markers'] or {}).get('q') or 0 for r in reused),
        sum(r.get('prompt_tokens') or 0 for r in continued)))
    if not served:
        missing.append('no agent turn was served')
        return problems, missing, lines
    if not continued:
        missing.append('no agent continued a conversation (every turn was a first turn): nothing could be reused')
        return problems, missing, lines
    cold = []
    for conv, turns in sorted(by_conv.items(), key=lambda item: str(item[0])):
        later = [r for r in turns if r.get('continuation')]
        if later and not any((r.get('markers') or {}).get('q') for r in later):
            cold.append(str(conv))
    if cold:
        missing.append('%d of %d conversations never reused a token on any continuation (%s): the turns measured no '
                       'reuse for them' % (len(cold), len(by_conv), ', '.join(cold[:8])))
    unmarked = [r for r in continued if (r.get('markers') or {}).get('q') is None]
    if unmarked:
        lines.append('%d continuations carry no [PREFIX] row (an unsalted or unresolved turn)' % len(unmarked))
    return problems, missing, lines


def _keyed(records):
    return dict(((r.get('case'), r.get('conv'), r.get('turn')), r) for r in agent_records(records))


def _compare_pair(a, b):
    """IDENTICAL / DIVERGED / NOT_COMPARABLE / ERROR for two records of one (case, conversation, turn)."""
    for label, record in (('A', a), ('B', b)):
        if not record.get('ok'):
            return 'ERROR', '%s: %s' % (label, record.get('error') or record.get('aborted') or 'no answer')
    if a.get('prompt_sha') != b.get('prompt_sha') or a.get('prompt_tokens') != b.get('prompt_tokens'):
        return 'NOT_COMPARABLE', 'different prompts (an earlier turn of this conversation diverged)'
    ids_a, ids_b = a.get('token_ids'), b.get('token_ids')
    if ids_a is not None and ids_b is not None:
        at = None
        for index in range(min(len(ids_a), len(ids_b))):
            if ids_a[index] != ids_b[index]:
                at = index
                break
        if at is None and len(ids_a) != len(ids_b):
            at = min(len(ids_a), len(ids_b))
        if at is None and a.get('finish') == b.get('finish'):
            return 'IDENTICAL', None
        return 'DIVERGED', 'first differing output token %s (A %d tokens %s, B %d tokens %s)' % (
            at, len(ids_a), a.get('finish'), len(ids_b), b.get('finish'))
    same = (a.get('content') == b.get('content') and a.get('reasoning') == b.get('reasoning')
            and a.get('finish') == b.get('finish'))
    return ('IDENTICAL', None) if same else ('DIVERGED', 'texts differ (no token ids)')


def compare(records_a, records_b, label_a='A', label_b='B'):
    """Two arms' transcripts, turn by turn. -> dict(verdict, counts, problems, lines, paired)."""
    keyed_a, keyed_b = _keyed(records_a), _keyed(records_b)
    both = sorted(set(keyed_a) & set(keyed_b), key=lambda key: (str(key[0]), str(key[1]), key[2] if key[2] is not None
                                                                else -1))
    counts = dict(IDENTICAL=0, DIVERGED=0, NOT_COMPARABLE=0, ERROR=0)
    problems, lines, paired = [], [], []
    diverged_convs = set()
    for key in both:
        verdict, detail = _compare_pair(keyed_a[key], keyed_b[key])
        counts[verdict] += 1
        if verdict == 'DIVERGED' and key[1] not in diverged_convs:
            diverged_convs.add(key[1])
            problems.append('%s turn %s diverged between %s and %s: %s' % (key[1], key[2], label_a, label_b, detail))
        elif verdict == 'ERROR':
            problems.append('%s turn %s: %s' % (key[1], key[2], detail))
        elif verdict == 'NOT_COMPARABLE' and key[1] not in diverged_convs:
            # A different prompt with no divergence before it: the two arms sized or fitted the turn differently.
            problems.append('%s turn %s is not comparable (%s) and no earlier turn of it diverged' % (
                key[1], key[2], detail))
        if verdict == 'IDENTICAL':
            paired.append(key)
    only_a = sorted(set(keyed_a) - set(keyed_b), key=str)
    only_b = sorted(set(keyed_b) - set(keyed_a), key=str)
    for label, only in ((label_a, only_a), (label_b, only_b)):
        if only:
            problems.append('%d turns only %s ran (%s ...): the other arm lacks them' % (
                len(only), label, ', '.join('%s#%s' % (key[1], key[2]) for key in only[:4])))
    lines.append('transcripts %s vs %s: %d turns paired, %d identical, %d diverged, %d not comparable, %d error' % (
        label_a, label_b, len(both), counts['IDENTICAL'], counts['DIVERGED'], counts['NOT_COMPARABLE'], counts['ERROR']))
    for label, keyed in ((label_a, keyed_a), (label_b, keyed_b)):
        served = [r for r in keyed.values() if r.get('ok')]
        cont = [r for r in served if r.get('continuation')]
        lines.append('%s: %d turns served; TTFT p50/p90 %s/%s s (continuations %s/%s); decode p50 %s tok/s; turn p50 %s s' % (
            label, len(served), _round(percentile([r.get('ttft_s') for r in served], 0.5)),
            _round(percentile([r.get('ttft_s') for r in served], 0.9)),
            _round(percentile([r.get('ttft_s') for r in cont], 0.5)),
            _round(percentile([r.get('ttft_s') for r in cont], 0.9)),
            _round(percentile([decode_tps(r) for r in served], 0.5)),
            _round(percentile([r.get('wall_s') for r in served], 0.5))))
    ratios = []
    for key in paired:
        a, b = keyed_a[key], keyed_b[key]
        if a.get('continuation') and a.get('ttft_s') and b.get('ttft_s'):
            ratios.append(a['ttft_s'] / float(b['ttft_s']))
    if ratios:
        lines.append('paired continuation TTFT %s/%s: median %s (min %s, max %s) over %d turns' % (
            label_a, label_b, _round(percentile(ratios, 0.5), 3), _round(min(ratios), 3), _round(max(ratios), 3),
            len(ratios)))
    verdict = 'PASS' if not problems and counts['IDENTICAL'] and not counts['DIVERGED'] else 'FAIL'
    if not both:
        verdict = 'FAIL'
        problems.append('no turn is present in both arms')
    return dict(verdict=verdict, counts=counts, problems=problems, lines=lines, paired=len(paired))


def load_records(path):
    """The records of a results directory (<dir>/records.jsonl, or <dir>/<arm>/records.jsonl when it holds one arm) or of a
    records.jsonl file."""
    candidates = [path]
    if os.path.isdir(path):
        candidates = [os.path.join(path, 'records.jsonl')]
        candidates += [os.path.join(path, name, 'records.jsonl') for name in sorted(os.listdir(path))]
    for candidate in candidates:
        if os.path.isfile(candidate):
            with open(candidate, encoding='utf-8') as handle:
                return [json.loads(line) for line in handle if line.strip()]
    raise IOError('no records.jsonl at %s' % path)


def main(argv=None, out=print):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('a')
    parser.add_argument('b')
    parser.add_argument('--label-a', default='A')
    parser.add_argument('--label-b', default='B')
    options = parser.parse_args(argv)
    try:
        records_a, records_b = load_records(options.a), load_records(options.b)
    except (IOError, OSError, ValueError) as error:
        out('refused: %s' % error)
        return 2
    result = compare(records_a, records_b, options.label_a, options.label_b)
    for line in result['lines']:
        out(line)
    for problem in result['problems']:
        out('problem: %s' % problem)
    out('agent turns %s vs %s: %s' % (options.label_a, options.label_b, result['verdict']))
    return 0 if result['verdict'] == 'PASS' else 1


if __name__ == '__main__':
    sys.exit(main())
