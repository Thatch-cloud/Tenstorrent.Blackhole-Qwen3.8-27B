"""The A0 bundle: the tau lab's logged turns as token ids and counts, in the form the GPU host reads. Runs ON THE RIG HOST
(Python 3.7, stdlib): the prompt ids live only there.

    python3 scripts/ci/a0_bundle.py --data <tau lab data dir> --results <tau lab results dir> --out <bundle dir> \
        --map <private map file> [--expect swe=240,own=96] [--seed N]

What goes in (all read through the lab's own readers, so the lab's refusals apply: manifest sizes and sha256, the scrub report,
unique turn ids, ids length = prompt tokens):
    a1_swe_heldout / a2_own_sessions   cluster, turn index, bucket inputs, inverse-probability weight, the thinking-ON prompt ids
    outputs.jsonl                      the answer token ids of every ok turn
    turns.jsonl + server.log           finish, think / tool offsets, the logged round schedule (emitted and cap per packed round)

What comes out (out/): bundle.jsonl.gz (one line per turn with its ids), meta.jsonl (the same turns WITHOUT ids: the report's
metadata), MANIFEST.json (sha256 and size of both, the counts, the checks). The turn ids of the lab never enter: every turn,
and every cluster, gets an OPAQUE index from a seeded shuffle, and the map from index to lab turn id is written to --map, a
separate file that stays on the rig. Token ids are equivalent to text, so the bundle is lab-store data: directory 0700, files
0600; stdout carries counts only.

Checks (counts, never values): `prefix_groups` (a turn whose prompt is a prefix of its conversation's next prompt shares a
group with it: the GPU host prefills the longest prompt once and branches), `scheduled` (turns whose logged emitted counts sum
to the answer: the forced-schedule arm can follow them), `cap_mismatch` (logged cap= fields against
extent_attention_replay.accept_limit at the walk's start convention; `cap_mismatch_alt` at the start + 1 convention).
"""
import argparse
import gzip
import json
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import c2_tau_lab as lab  # noqa: E402
import tau_lab_report as rep  # noqa: E402
from a0_bundle_io import (BUNDLE_NAME, FORMAT, MANIFEST_NAME, META_NAME, BundleError, read_bundle, sha256_file,  # noqa: E402,F401
                          verify_bundle)
from extent_attention_replay import accept_limit  # noqa: E402

SETS = (('A1', 'swe', 'a1_swe_heldout.jsonl', 'a1_swe_heldout.ids.jsonl.gz'),
        ('A2', 'own', 'a2_own_sessions.jsonl', 'a2_own_sessions.ids.jsonl.gz'))
DEFAULT_EXPECT = 'swe=240,own=96'
ROWS = 16
TAIL_ROUNDS = 2            # a cap below the extent's may be a budget cut: only the last rounds are allowed to differ that way


def parse_expect(text):
    out = {}
    for part in (text or '').split(','):
        if part.strip():
            name, _, value = part.partition('=')
            out[name.strip()] = int(value)
    return out


def read_jsonl(path):
    out = []
    if os.path.isfile(path):
        with open(path, encoding='utf-8') as handle:
            for line in handle:
                if line.strip():
                    try:
                        out.append(json.loads(line))
                    except ValueError:
                        continue
    return out


def schedule_of(rounds, answer_len):
    """([emitted], [position], [cap]) when the logged rounds are all packed with known emitted counts that cover the answer
    after its prefill seed exactly, else None."""
    if not rounds or any(entry['kind'] != 'P' or entry.get('emitted') is None for entry in rounds):
        return None
    emitted = [entry['emitted'] for entry in rounds]
    if sum(emitted) != answer_len - 1 or any(value < 1 for value in emitted):
        return None
    return emitted, [entry.get('position') for entry in rounds], [entry.get('cap') for entry in rounds]


def cap_mismatches(prompt_len, emitted, positions, caps, shift=0):
    """How many logged cap= fields differ from the extent rule when the walk's anchor start is the logged position or the
    position plus `shift` (counts only). The last rounds may be cut by the budget, so only a LARGER cap is a mismatch there."""
    count, last = 0, len(caps) - TAIL_ROUNDS
    for at, (position, cap) in enumerate(zip(positions, caps)):
        if position is None or cap is None:
            continue
        expected = accept_limit(position + shift, ROWS)
        if cap > expected or (cap < expected and at < last):
            count += 1
    return count


def prefix_groups(records):
    """Assign each record a `group` and its place `order` in it: within a conversation (set, cluster) in turn order, a turn whose
    prompt is a prefix of the next turn's prompt shares the next turn's group. -> (group count, chained turns)."""
    by_conv = {}
    for record in records:
        by_conv.setdefault((record['set'], record['cluster']), []).append(record)
    groups, chained = 0, 0
    for key in sorted(by_conv):
        previous = None
        for record in sorted(by_conv[key], key=lambda item: item['turn']):
            prompt = record['prompt_ids']
            if previous is not None and len(previous['prompt_ids']) <= len(prompt) \
                    and list(prompt[:len(previous['prompt_ids'])]) == list(previous['prompt_ids']):
                record['group'] = previous['group']
                record['order'] = previous['order'] + 1
                chained += 1
            else:
                record['group'] = groups
                record['order'] = 0
                groups += 1
            previous = record
    return groups, chained


def collect(data, results, arms=('A1', 'A2'), seed=0, expect=None):
    """The turn records (private lab ids still attached as `lab_arm` / `lab_id`) and the checks."""
    wanted = [item for item in SETS if item[0] in arms]
    manifest = lab.read_manifest(data)
    lab.verify_files(data, manifest, [item[0] for item in wanted])
    scrub = lab.check_scrub_report(data, [item[0] for item in wanted])
    log_path = os.path.join(results, 'server.log')
    log_text = ''
    if os.path.isfile(log_path):
        with open(log_path, encoding='utf-8', errors='replace') as handle:
            log_text = handle.read()
    chosen = rep.last_ok_turns(read_jsonl(os.path.join(results, 'turns.jsonl')))
    outputs = dict(((entry['arm'], entry['id']), entry) for entry in read_jsonl(os.path.join(results, 'outputs.jsonl')))
    by_stream = dict((turn['request_id'], key) for key, turn in chosen.items() if turn.get('request_id') and turn.get('status') == 'ok')
    turn_rounds = {}
    for request_id, rounds in rep.parse_rounds(log_text).items():
        stream_id = rep.owner_of(request_id, by_stream)
        if stream_id is not None:
            turn_rounds.setdefault(by_stream[stream_id], []).extend(rounds)

    records, missing = [], 0
    checks = dict(scheduled=0, cap_mismatch=0, cap_mismatch_alt=0, rounds_checked=0)
    for arm, set_name, conv_file, ids_file in wanted:
        for entry in lab.conversation_records(os.path.join(data, conv_file), os.path.join(data, ids_file), set_name):
            key = (arm, entry['id'])
            turn, output = chosen.get(key), outputs.get(key)
            if not turn or turn.get('status') != 'ok' or not output or not output.get('output_ids'):
                missing += 1
                continue
            answer = list(output['output_ids'])
            prompt = list(entry['prompt_ids'])
            rounds = turn_rounds.get(key) or []
            scheduled = schedule_of(rounds, len(answer))
            if scheduled:
                emitted, positions, caps = scheduled
                checks['scheduled'] += 1
                checks['rounds_checked'] += len(emitted)
                checks['cap_mismatch'] += cap_mismatches(len(prompt), emitted, positions, caps)
                checks['cap_mismatch_alt'] += cap_mismatches(len(prompt), emitted, positions, caps, 1)
            stat = rep.analyze_turn(turn, rounds, rep.LIVE_SEATS if rep.has_live(dict(x=rounds)) else None)
            records.append(dict(set=set_name, cluster=entry['cluster'], turn=entry['turn'], bucket=rep.bucket_of(len(prompt)),
                                weight=entry['weight'], prompt_ids=prompt, output_ids=answer, finish=turn.get('finish'),
                                think_tokens=turn.get('think_tokens') or 0, tool_at=turn.get('tool_at'),
                                schedule=scheduled[0] if scheduled else None,
                                tau_served=rep.rounded(stat['tau']) if stat.get('tau') is not None else None,
                                lab_arm=arm, lab_id=entry['id']))
    if missing:
        raise BundleError('%d turns have no ok result and answer ids in the results directory' % missing)
    counts = dict((name, sum(1 for record in records if record['set'] == name)) for _, name, _, _ in wanted)
    for name, want in (expect or {}).items():
        if name in counts and counts[name] != want:
            raise BundleError('set %s holds %d turns, %d expected' % (name, counts[name], want))
    checks['scrub_kept'] = (scrub or {}).get('kept')
    return records, counts, checks


def anonymise(records, seed):
    """Opaque `k` per turn and opaque `cluster` per conversation, from a seeded shuffle; returns {k: [arm, lab id]}."""
    rng = random.Random(seed)
    order = sorted(range(len(records)), key=lambda at: (records[at]['lab_arm'], records[at]['lab_id']))
    rng.shuffle(order)
    mapping = {}
    for k, at in enumerate(order):
        records[at]['k'] = k
        mapping[k] = [records[at]['lab_arm'], records[at]['lab_id']]
    clusters = {}
    for name in sorted(set(record['set'] for record in records)):
        labels = sorted(set(record['cluster'] for record in records if record['set'] == name))
        shuffled = list(range(len(labels)))
        rng.shuffle(shuffled)
        clusters[name] = dict(zip(labels, shuffled))
    for record in records:
        record['lab_cluster'] = record['cluster']
        record['cluster'] = clusters[record['set']][record['cluster']]
    return mapping


META_FIELDS = ('k', 'set', 'cluster', 'turn', 'group', 'order', 'bucket', 'weight', 'think_tokens', 'tool_at', 'finish',
               'tau_served')


def meta_of(record):
    meta = dict((name, record.get(name)) for name in META_FIELDS)
    meta['prompt_tokens'] = len(record['prompt_ids'])
    meta['answer_tokens'] = len(record['output_ids'])
    meta['scheduled'] = record['schedule'] is not None
    return meta


def write_private(path, writer, mode='wt'):
    """A file created 0600 (the directory is 0700 too): lab-store data."""
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    handle = os.fdopen(descriptor, 'wb')
    if mode == 'wt':
        import io
        text = io.TextIOWrapper(handle, encoding='utf-8', newline='\n')
        try:
            writer(text)
        finally:
            text.close()
    else:
        try:
            writer(handle)
        finally:
            handle.close()


def write_bundle(out, records, counts, checks, groups, chained, mapping, map_path):
    if os.path.exists(out) and os.listdir(out):
        raise BundleError('the output directory is not empty')
    os.makedirs(out, mode=0o700, exist_ok=True)
    os.chmod(out, 0o700)
    if os.path.exists(map_path):
        raise BundleError('the map file exists already')

    def bundle_writer(handle):
        with gzip.GzipFile(fileobj=handle, mode='wb', mtime=0) as zipped:
            for record in sorted(records, key=lambda item: item['k']):
                line = dict((name, record[name]) for name in (
                    'k', 'set', 'cluster', 'turn', 'group', 'order', 'bucket', 'weight', 'prompt_ids', 'output_ids', 'finish',
                    'think_tokens', 'tool_at', 'schedule', 'tau_served'))
                zipped.write((json.dumps(line, sort_keys=True) + '\n').encode('utf-8'))

    def meta_writer(handle):
        for record in sorted(records, key=lambda item: item['k']):
            handle.write(json.dumps(meta_of(record), sort_keys=True) + '\n')

    write_private(os.path.join(out, BUNDLE_NAME), bundle_writer, 'wb')
    write_private(os.path.join(out, META_NAME), meta_writer)
    files = dict((name, dict(bytes=os.path.getsize(os.path.join(out, name)), sha256=sha256_file(os.path.join(out, name))))
                 for name in (BUNDLE_NAME, META_NAME))
    manifest = dict(format=FORMAT, files=files, counts=dict(counts, turns=len(records)), prefix_groups=groups,
                    prefix_chained=chained, checks=dict((name, value) for name, value in checks.items() if value is not None))
    write_private(os.path.join(out, MANIFEST_NAME), lambda handle: handle.write(json.dumps(manifest, sort_keys=True, indent=1) + '\n'))
    write_private(map_path, lambda handle: handle.write(json.dumps(
        dict((str(k), value) for k, value in mapping.items()), sort_keys=True) + '\n'))
    return manifest


def build(data, results, out, map_path, arms=('A1', 'A2'), seed=0, expect=None):
    records, counts, checks = collect(data, results, arms, seed, expect)
    groups, chained = prefix_groups(records)
    mapping = anonymise(records, seed)
    return write_bundle(out, records, counts, checks, groups, chained, mapping, map_path)


def main(argv=None, say=print):
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--data', required=True)
    parser.add_argument('--results', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--map', required=True, dest='map_path')
    parser.add_argument('--arms', default='A1,A2')
    parser.add_argument('--expect', default=DEFAULT_EXPECT)
    parser.add_argument('--seed', type=int, default=20261003)
    options = parser.parse_args(argv)
    try:
        manifest = build(options.data, options.results, options.out, options.map_path,
                         tuple(options.arms.split(',')), options.seed, parse_expect(options.expect))
    except (BundleError, lab.LabError) as error:
        say('refused: %s' % error)           # the message names a file or a count
        return 2
    say('bundle: %d turns, %d prefix groups, %d scheduled, cap mismatches %d (alt %d)' % (
        manifest['counts']['turns'], manifest['prefix_groups'], manifest['checks']['scheduled'],
        manifest['checks']['cap_mismatch'], manifest['checks']['cap_mismatch_alt']))
    return 0


if __name__ == '__main__':
    sys.exit(main())
