"""Two or four tier-upload-timed results side by side (c2_prefix_gate --plan tier-upload-timed-control / tier-upload-timed and the sweeps; docs/tp4-upload1.md): the restore of
a returning session at ~64k, ~128k and ~254k with the host tier's upload levers off (A, the control) and on (B), each run as a job of its own (the ABAB pairs).

    python3 scripts/ci/upload_restore_compare.py <A results dir> <B ...> [<C> <D>] [--labels control,upload,control-repeat,upload-repeat]

A directory is one arm's results (the gate writes <results>/<arm>/: events.json, records.jsonl, server-final.log). Per size, from the container log's "host tier restore" line of the
returning turn and the arm's events:

  RESTORE   blocks, GB, ms and GB/s of the restore (the sliced restore's ms is the sum of its slices' work), B against A (percent);
  TTFT      the returning hit's first-token time, B against A, and the cold prefill's and the device-resident hit's of each arm (the same prompts: they move with the host, not the lever);
  TEXT      the returning turn's output token ids, B against A: the prompts are seeded by name, not by arm, and decoding is greedy, so they must be IDENTICAL.

With C and D (the repeat pair of an ABAB) it also compares D with C and prints A against C, the FLOOR: how much two runs of the very same control differ (the restore's ms and
the TTFT move with the host's load; a gain smaller than the floor is not a gain).

THE RULE (registered here, before any card run): GO for the levers when, in every pair, (1) every returning answer is IDENTICAL to the control's, (2) the restore's ms at the 128k and
254k sizes is lower by at least GAIN_BAR (25%), and (3) the returning hit's TTFT at no size is higher than the control's by more than max(TTFT_TOLERANCE (5%), the floor's own
difference at that size). NO-GO when (1) fails or the restore is not faster; INCONCLUSIVE when a size or a figure is missing (a restore line absent: the session was not restored).

Exit 0 only on GO for every pair; 1 on NO-GO or INCONCLUSIVE; 2 for a missing input. Stdlib only, Python 3.7 syntax: it runs on the rig host."""
import argparse
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import prefix_markers as markers  # noqa: E402

GAIN_BAR = 0.25
TTFT_TOLERANCE = 0.05
GAIN_SIZES = (128000, 254000)       # the sizes the restore's gain is judged at (nominal tokens; 64k is read, not judged: its fixed costs dominate)
SIZE = re.compile(r'tier-timed-(\d+)$')


def load_arm(path):
    """dict(sizes={nominal: row}) from one arm's results directory; ValueError when it holds none of the files."""
    base = path
    if os.path.isfile(path):
        raise ValueError('%s: give the arm\'s results directory, not a file' % path)
    names = os.listdir(base) if os.path.isdir(base) else []
    if 'events.json' not in names:
        raise ValueError('%s holds no events.json (is it one arm\'s results directory?)' % path)
    with open(os.path.join(base, 'events.json'), encoding='utf-8') as handle:
        events = json.load(handle).get('events') or {}
    records = []
    if 'records.jsonl' in names:
        with open(os.path.join(base, 'records.jsonl'), encoding='utf-8') as handle:
            records = [json.loads(line) for line in handle if line.strip()]
    log = []
    for name in ('server-final.log', 'server.log'):
        if name in names:
            with open(os.path.join(base, name), encoding='utf-8', errors='replace') as handle:
                log = handle.read().splitlines()
            break
    restores = dict((entry.get('tag'), entry) for entry in markers.scan(log).get('tier_restores') or ())
    returns = dict((record.get('conv'), record) for record in records if record.get('case') == 'tier-timed-return')
    sizes = {}
    for name, event in events.items():
        match = SIZE.match(name)
        if not match or not isinstance(event, dict):
            continue
        nominal = int(match.group(1))
        restore = restores.get(event.get('return_tag'))
        record = returns.get(name) or {}
        sizes[nominal] = dict(
            nominal=nominal, size=event.get('size'), prompt_tokens=event.get('return_prompt_tokens'), ok=bool(event.get('return_ok')),
            cold_ttft_s=event.get('build_ttft_s'), resident_ttft_s=event.get('resident_ttft_s'), return_ttft_s=event.get('return_ttft_s'),
            blocks=(restore or {}).get('blocks'), bytes=(restore or {}).get('bytes'), restore_ms=(restore or {}).get('ms'),
            slices=(restore or {}).get('slices'), token_ids=record.get('token_ids'), finish=record.get('finish'))
    return dict(path=path, sizes=sizes)


def gbps(row):
    if row.get('bytes') and row.get('restore_ms'):
        return row['bytes'] / (row['restore_ms'] * 1e6)
    return None


def percent(new, old):
    if new is None or not old:
        return None
    return (new / old - 1.0) * 100.0


def show(value, spec='%.2f'):
    return '-' if value is None else spec % value


def compare_pair(a, b, label_a='control', label_b='upload', floor=None):
    """-> dict(verdict GO|NO-GO|INCONCLUSIVE, problems, lines, rows). `floor`: {nominal: relative TTFT difference of the control against its own repeat} or None."""
    problems, unknown, lines, rows = [], [], [], []
    sizes = sorted(set(a['sizes']) | set(b['sizes']))
    if not sizes:
        unknown.append('neither arm holds a tier-timed size')
    for nominal in sizes:
        left, right = a['sizes'].get(nominal), b['sizes'].get(nominal)
        if left is None or right is None:
            unknown.append('size %d: only %s ran it' % (nominal, label_b if left is None else label_a))
            continue
        row = dict(nominal=nominal)
        rows.append(row)
        for tag, side in ((label_a, left), (label_b, right)):
            if not side['ok']:
                unknown.append('size %d: the returning turn of %s failed' % (nominal, tag))
            if side['restore_ms'] is None:
                unknown.append('size %d: %s has no "host tier restore" line for the returning turn: it was not restored from host RAM' % (nominal, tag))
        gain = percent(right['restore_ms'], left['restore_ms'])
        ttft = percent(right['return_ttft_s'], left['return_ttft_s'])
        row.update(restore_change=gain, ttft_change=ttft)
        lines.append('size %d (prompt %s tokens): restore %s blocks %s GB: %s ms (%s GB/s) %s vs %s ms (%s GB/s) %s, %s; returning hit TTFT %s s vs %s s (%s); cold %s s vs %s s; resident hit %s s vs %s s' % (
            nominal, right['prompt_tokens'], right['blocks'], show((right['bytes'] or 0) / 1e9 if right['bytes'] else None), show(left['restore_ms'], '%.1f'), show(gbps(left)),
            label_a, show(right['restore_ms'], '%.1f'), show(gbps(right)), label_b, 'n/a' if gain is None else '%+.1f%% against %s' % (gain, label_a),
            show(left['return_ttft_s']), show(right['return_ttft_s']), 'n/a' if ttft is None else '%+.1f%%' % ttft,
            show(left['cold_ttft_s'], '%.1f'), show(right['cold_ttft_s'], '%.1f'), show(left['resident_ttft_s']), show(right['resident_ttft_s'])))
        ids_a, ids_b = left.get('token_ids'), right.get('token_ids')
        if ids_a is None or ids_b is None:
            unknown.append('size %d: no output token ids recorded for the returning turn in %s' % (nominal, label_a if ids_a is None else label_b))
        elif ids_a != ids_b or left.get('finish') != right.get('finish'):
            at = next((index for index in range(min(len(ids_a), len(ids_b))) if ids_a[index] != ids_b[index]), min(len(ids_a), len(ids_b)))
            problems.append('size %d: the returning answers differ (first differing output token %d; %s %d tokens %s, %s %d tokens %s)' % (
                nominal, at, label_a, len(ids_a), left.get('finish'), label_b, len(ids_b), right.get('finish')))
            row['text'] = 'DIVERGED'
        else:
            row['text'] = 'IDENTICAL'
            lines.append('size %d: the returning answer is IDENTICAL across the two arms (%d tokens)' % (nominal, len(ids_a)))
        if nominal in GAIN_SIZES and gain is not None and not -gain / 100.0 >= GAIN_BAR:
            problems.append('size %d: the restore is %s against %s, not at least %.0f%% faster (the bar)' % (
                nominal, '%+.1f%%' % gain, label_a, GAIN_BAR * 100))
        if ttft is not None:
            allowed = max(TTFT_TOLERANCE * 100.0, (floor or {}).get(nominal) or 0.0)
            if ttft > allowed:
                problems.append('size %d: the returning hit\'s TTFT is %+.1f%% against %s, over the %.1f%% allowed (the larger of %.0f%% and the control\'s own repeat difference)' % (
                    nominal, ttft, label_a, allowed, TTFT_TOLERANCE * 100))
    missing_gain = [size for size in GAIN_SIZES if size not in a['sizes'] or size not in b['sizes']]
    if missing_gain:
        unknown.append('the sizes the gain is judged at (%s) are not in both arms: %s' % (', '.join(str(size) for size in GAIN_SIZES), ', '.join(str(size) for size in missing_gain)))
    verdict = 'NO-GO' if problems else ('INCONCLUSIVE' if unknown else 'GO')
    return dict(verdict=verdict, problems=problems + unknown, lines=lines, rows=rows)


def floor_of(a, c):
    """{nominal: |relative TTFT difference| in percent} of the control against its own repeat."""
    out = {}
    for nominal in set(a['sizes']) & set(c['sizes']):
        change = percent(c['sizes'][nominal].get('return_ttft_s'), a['sizes'][nominal].get('return_ttft_s'))
        if change is not None:
            out[nominal] = abs(change)
    return out


def main(argv=None, out=print):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('paths', nargs='+', help='A B [C D]: one arm\'s results directory each')
    parser.add_argument('--labels', default='control,upload,control-repeat,upload-repeat', help='comma-separated names for A B C D')
    options = parser.parse_args(argv)
    if len(options.paths) not in (2, 4):
        out('refused: give A B (one pair) or A B C D (the ABAB: two pairs and the control-to-control floor)')
        return 2
    labels = (options.labels.split(',') + ['A', 'B', 'C', 'D'])[:len(options.paths)]
    try:
        arms = [load_arm(path) for path in options.paths]
    except (IOError, OSError, ValueError) as error:
        out('refused: %s' % error)
        return 2
    floor = floor_of(arms[0], arms[2]) if len(arms) == 4 else None
    verdicts = []
    for first, second in [(0, 1)] + ([(2, 3)] if len(arms) == 4 else []):
        result = compare_pair(arms[first], arms[second], labels[first], labels[second], floor)
        for line in result['lines']:
            out(line)
        for problem in result['problems']:
            out('problem: %s' % problem)
        out('upload restore %s vs %s: %s' % (labels[first], labels[second], result['verdict']))
        verdicts.append(result['verdict'])
    if floor is not None:
        out('FLOOR (the same control twice, %s vs %s): returning-hit TTFT differs by %s' % (
            labels[0], labels[2], ', '.join('%d: %.1f%%' % (size, value) for size, value in sorted(floor.items())) or 'n/a'))
    return 0 if all(verdict == 'GO' for verdict in verdicts) else 1


if __name__ == '__main__':
    sys.exit(main())
