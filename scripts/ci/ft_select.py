"""WP-F1: select training turns and write them as a tau-lab DATA DIRECTORY (Python 3.7, stdlib), so c2_tau_lab can drive the target to
regenerate their answers as the answers-kept arm G (`C2_TAULAB_ARMS=G`; the lab keeps every arm's output ids in outputs.jsonl).

The directory is the lab's FORMAT, restricted to the one arm:
    g_train.jsonl          one conversation per line: id, group, source, turns[] (turn_id, bucket, prompt_tokens_think_on and the
                           inverse-probability inputs, which are 1 / 1 here: the fine-tune wants the whole selected set, not a weighted sample)
    g_train.ids.jsonl.gz   {turn_id, think: "on", n, ids} per turn: the thinking-ON prompt ids (the lab sends ids, never text)
    MANIFEST.json          format "FORMAT.md", every file's size and sha256, arms {G: {conversations, turns}}
A data directory of training turns only has no A1..A5 tables and no scrub report; `c2_tau_lab.load_data(dir, ['G'])` reads it.

THE GUARD RUNS BEFORE ANYTHING IS WRITTEN: the split rules (ft_split) are checked over the selection, the mix is checked, and a refusal
leaves no file behind. Prompt ids are lab-store data (private to the rig); this module never prints one.
"""
import gzip
import hashlib
import json
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import ft_split  # noqa: E402
import tau_lab_report as rep  # noqa: E402

ARM = 'G'
CONVERSATIONS = 'g_train.jsonl'
IDS = 'g_train.ids.jsonl.gz'
MANIFEST = 'MANIFEST.json'


class SelectError(ValueError):
    pass


def select_conversations(candidates, mix, total_turns, seed, d4_cleared=False):
    """Whole conversations, per source up to the mix's turn quota, in a seeded order. `candidates`: conversation dicts with `source`,
    `group`, `id`, `turns` (each with `turn_id` and `ids`) and the source's identifiers (`repo`, `conversation`, `project`)."""
    mix = ft_split.check_mix(mix, d4_cleared)
    quota = ft_split.mix_counts(total_turns, mix)
    rng = random.Random('%s:select' % seed)
    chosen = []
    for source, want in sorted(quota.items()):
        pool = sorted((c for c in candidates if c['source'] == source), key=lambda c: c['id'])
        rng.shuffle(pool)
        have = 0
        for conversation in pool:
            if have >= want:
                break
            take = min(len(conversation['turns']), want - have)
            chosen.append(dict(conversation, turns=conversation['turns'][:take]))
            have += take
        if have < want:
            raise SelectError('source %s has %d turns, the mix asks for %d' % (source, have, want))
    return chosen


def conversation_record(conversation):
    turns = []
    for turn in conversation['turns']:
        bucket = rep.bucket_of(len(turn['ids']))
        turns.append(dict(turn_id=turn['turn_id'], bucket=bucket, prompt_tokens_think_on=len(turn['ids']),
                          eligible_per_bucket={bucket: 1}, chosen_in_bucket={bucket: 1}))
    return dict(id=conversation['id'], arm=ARM, source=conversation['source'], group=conversation['group'], turns=turns)


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def write_training_directory(out, conversations, a1=(), a2=(), eval_tier=(), d4_cleared=False):
    """Check the guard, then write the lab data directory. -> the manifest (counts only). Refuses an existing non-empty directory."""
    records = [dict((key, value) for key, value in c.items() if key not in ('turns', 'id')) for c in conversations]
    ft_split.guard_training_set(records, a1=a1, a2=a2, eval_tier=eval_tier, d4_cleared=d4_cleared)
    ids_seen = set()
    for conversation in conversations:
        for turn in conversation['turns']:
            if turn['turn_id'] in ids_seen or not turn['ids'] or any(type(t) is not int or t < 0 for t in turn['ids']):
                raise SelectError('a turn id repeats, or a turn has no valid prompt ids')
            ids_seen.add(turn['turn_id'])
    if os.path.exists(out) and os.listdir(out):
        raise SelectError('the output directory is not empty')
    os.makedirs(out, mode=0o700, exist_ok=True)
    with open(os.path.join(out, CONVERSATIONS), 'w', encoding='utf-8', newline='\n') as handle:
        for conversation in conversations:
            handle.write(json.dumps(conversation_record(conversation), sort_keys=True) + '\n')
    with gzip.GzipFile(os.path.join(out, IDS), 'wb', mtime=0) as handle:
        for conversation in conversations:
            for turn in conversation['turns']:
                row = dict(turn_id=turn['turn_id'], think='on', n=len(turn['ids']), ids=list(turn['ids']))
                handle.write((json.dumps(row, sort_keys=True) + '\n').encode('utf-8'))
    files = dict((name, dict(bytes=os.path.getsize(os.path.join(out, name)), sha256=sha256_file(os.path.join(out, name))))
                 for name in (CONVERSATIONS, IDS))
    manifest = dict(format='FORMAT.md', files=files,
                    arms={ARM: dict(conversations=len(conversations), turns=sum(len(c['turns']) for c in conversations))})
    with open(os.path.join(out, MANIFEST), 'w', encoding='utf-8', newline='\n') as handle:
        json.dump(manifest, handle, sort_keys=True, indent=1)
        handle.write('\n')
    for name in os.listdir(out):
        os.chmod(os.path.join(out, name), 0o600)
    return manifest
