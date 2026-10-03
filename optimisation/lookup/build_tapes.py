"""Build the tape file lookup_sim.py reads from a tau-lab run (private inputs: pass paths, nothing is stored here).

  --turns    the lab's turns.jsonl (one line per turn: status, arm, id, set, cluster, weight, request_id, ...)
  --outputs  the lab's outputs.jsonl (output token ids per turn)
  --log      the container log (or its [PACKED] / [PACKED-PHASE] lines), plain or .gz
  --ids      zero or more *.ids.jsonl[.gz] files (one {turn_id, ids} per prompt); without them the tape has no prompts
             and the estimate sees committed tokens only, a lower bound (the header line says so)
  --out      the tape file (jsonl, .gz by name)

A [PACKED] request=<id> segment=.. position=P prefix=.. emitted=E line is a round that committed out[off : off + E],
off = P - plen + 1; the owner of a request id is the lab turn whose request_id is its longest dash-prefix. A round is
counted (the lab's population) when its live count is 4, it is not the request's last, and it committed something.
Sequential rounds are kept as empty entries so they break the spans of consecutive packed rounds.

Python 3.7, stdlib only.
"""
import argparse
import gzip
import json
import re
import sys
from collections import Counter

PATH_LINE = re.compile(r'\[(PACKED|SEQUENTIAL|SEQ-PUBLISH)\] request=(\S+) ([^\n]*)')
PATH_FIELD = re.compile(r'(?<![A-Za-z0-9_])(segment|position|prefix|emitted|rows|cap)=([0-9]+|n/a)(?![0-9A-Za-z_])')
PHASE = re.compile(r'\[PACKED-PHASE\] round=([0-9]+) users=([0-9]+)[^\n]*?\blive=([0-9]+)')
PRED = re.compile(r'predictions=\[([^\]]*)\]')
THINK_END, TOOL_CALL = 248069, 248058


def opened(path):
    return gzip.open(path, 'rt', encoding='utf-8', errors='replace') if path.endswith('.gz') else open(path, encoding='utf-8', errors='replace')


def jl(path):
    with opened(path) as handle:
        return [json.loads(line) for line in handle if line.strip()]


def owner_of(request_id, by_stream):
    current = request_id
    while current:
        if current in by_stream:
            return current
        cut = current.rfind('-')
        if cut <= 0:
            return None
        current = current[:cut]
    return None


def parse_log(path, by_stream):
    rounds, live, unattributed = {}, None, Counter()
    with opened(path) as handle:
        for line in handle:
            match = PHASE.search(line)
            if match and line.lstrip().startswith('[PACKED-PHASE]'):
                live = int(match.group(3))
                continue
            match = PATH_LINE.search(line)
            if not match:
                continue
            kind, request_id, rest = match.groups()
            if kind == 'SEQ-PUBLISH' and not rest.startswith('rows='):
                continue
            fields = {}
            for name, value in PATH_FIELD.findall(rest):
                fields.setdefault(name, None if value == 'n/a' else int(value))
            preds = PRED.search(rest)
            entry = dict(kind='P' if kind == 'PACKED' else 'S', position=fields.get('position'), emitted=fields.get('emitted'),
                         cap=fields.get('cap'), live=live if kind == 'PACKED' else None,
                         preds=[int(x) for x in preds.group(1).split(',') if x.strip()] if preds else None)
            owner = owner_of(request_id, by_stream)
            if owner is None:
                unattributed[kind] += 1
                continue
            rounds.setdefault(by_stream[owner], []).append(entry)
    return rounds, unattributed


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--turns', required=True)
    parser.add_argument('--outputs', required=True)
    parser.add_argument('--log', required=True)
    parser.add_argument('--ids', nargs='*', default=[])
    parser.add_argument('--out', required=True)
    parser.add_argument('--arms', nargs='*', default=['A1', 'A2'], help='arms kept (thinking-on arms by default)')
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)

    chosen = {}
    for turn in jl(args.turns):
        if turn.get('status') == 'ok':
            chosen[(turn['arm'], turn['id'])] = turn
    outputs = dict(((o['arm'], o['id']), o['output_ids']) for o in jl(args.outputs))
    ids = {}
    for path in args.ids:
        for record in jl(path):
            ids[record['turn_id']] = record['ids']
    by_stream = dict((t['request_id'], key) for key, t in chosen.items() if t.get('request_id'))
    rounds, unattributed = parse_log(args.log, by_stream)

    tapes, matched, total = [], 0, 0
    for key, turn in sorted(chosen.items()):
        if key[0] not in args.arms:
            continue
        out = outputs.get(key) or []
        prompt = ids.get(key[1]) or []
        plen = turn.get('prompt_tokens') or len(prompt)
        if prompt and len(prompt) != plen:
            print('prompt length mismatch for a turn: %d vs %d' % (len(prompt), plen), file=sys.stderr)
        log = rounds.get(key) or []
        entries, last = [], len(log) - 1
        for index, entry in enumerate(log):
            if entry['kind'] != 'P' or not entry['emitted'] or entry['position'] is None:
                entries.append([None, 0, 0, None])
                continue
            off = entry['position'] - plen + 1
            counted = int(index != last and entry['live'] == 4)
            entries.append([off, entry['emitted'], counted, entry['cap']])
            if entry['preds']:
                n = min(entry['emitted'], len(entry['preds']))
                total += 1
                matched += int(0 <= off and off + n <= len(out) and out[off:off + n] == entry['preds'][:n])
        tapes.append(dict(id=key[1], arm=key[0], set=turn.get('set'), cluster=turn.get('cluster'), weight=turn.get('weight') or 1.0,
                          plen=plen, prompt=prompt, out=out, rounds=entries, thinking=turn.get('thinking'),
                          think_end=(out.index(THINK_END) + 1) if THINK_END in out else None,
                          tool_at=out.index(TOOL_CALL) if TOOL_CALL in out else None))
    header = dict(header=True, prompt_included=bool(ids), turns=len(tapes), alignment_checked=total, alignment_matched=matched,
                  unattributed=dict(unattributed))
    opener = gzip.open if args.out.endswith('.gz') else open
    with opener(args.out, 'wt', encoding='utf-8', newline='\n') as handle:
        handle.write(json.dumps(header) + '\n')
        for tape in tapes:
            handle.write(json.dumps(tape, separators=(',', ':')) + '\n')
    print('tapes %d, prompts %s, rounds aligned %d/%d, unattributed %s' % (len(tapes), bool(ids), matched, total, dict(unattributed)))


if __name__ == '__main__':
    main()
